"""Numpy-native point-cloud processing nodes for the depth-sensor pipeline.

These sit between a depth source (``femto`` / ``femto_bolt``) and a renderer
(``mgl_point_cloud`` / ``gl_vertex_buffer``) and operate on camera-agnostic
``(N, 3)`` float32 clouds. They are deliberately numpy-native rather than torch:
for these ops well-vectorised numpy lowers to SIMD C loops and avoids torch's
per-frame tensor-conversion + kernel-dispatch overhead, which dominates at 30fps
with a few hundred-thousand points. See torch_voxel_nodes.py for the torch
equivalents of crop/voxelise.

The shared engine is a dense voxel grid: every point is mapped to an integer
voxel via a precomputed linear index, then all the work (occupancy, density,
background model, temporal persistence) is gather/scatter on flat arrays with
``np.bincount`` — no hashing, no kd-tree, no per-point python.

Nodes:
  pc_crop        axis-aligned box crop (+ invert)
  pc_voxel       voxel-grid downsample -> occupied voxel centres / centroids,
                 optionally grouped into a lattice of boxes
  pc_background  static background subtraction (learn N frames, then remove)
  pc_denoise     density + temporal-persistence speckle/flicker removal
  pc_info        report point count / bounds / centroid (bounds-tuning aid)
  pc_cluster_filter  gate / smooth / difference the per-cluster values

Cloud-frame convention: a frame on the wire is either a raw (N, 3) array or a
dict {'point_cloud': pts, 'crop': (min, max), ...}. pc_crop attaches its crop
spec; every grid-based node downstream uses the carried crop as its volume
bounds (its own min/max widgets are only the fallback for raw input), and all
nodes pass the metadata through. pc_voxel likewise attaches its voxel size
(metres; float when cubic, (x, y, z) otherwise), and grid nodes downstream
adopt it the same way, so a chain shares one grid geometry. Renderers unwrap
the 'point_cloud' key, so either form draws directly.

Cluster-frame convention: a node that groups voxels attaches a 'clusters' dict
to the frame — {'labels': (N,) int32 cluster id per output point, 'values':
(K,) float32 per cluster, 'shape': (kx, ky, kz) when the clusters form a
regular lattice else None, 'origin'/'cell': (3,) float32 lattice geometry}.
Boxes (pc_voxel) are the first producer; connected components, k-means and
hand-painted regions are the same shape of thing and would emit the same dict,
so one display / threshold / OSC stage downstream serves all of them. Boxes
are special only in that they constrain the voxel size, which is why they live
in pc_voxel rather than in a node of their own.

pc_voxel additionally accepts an mgl chain, drawing the volume it is working
in as a wireframe — the box lattice when subdivided, the bare outline when not
(see VolumeGridDrawMixin). That is the one place these nodes touch GL, and
only through a lazy import, so the module stays numpy-only for patches with no
3D view.
"""

import threading
import time

import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import any_to_array

CLOUD_KEY = 'point_cloud'
CROP_KEY = 'crop'
VOXEL_SIZE_KEY = 'voxel_size'
CLUSTER_KEY = 'clusters'


def unwrap_cloud(data):
    """Split a cloud frame into (points, meta). Accepts the dict convention or
    raw array/list data (meta is then {})."""
    if isinstance(data, dict):
        meta = dict(data)
        pts = meta.pop(CLOUD_KEY, None)
        return pts, meta
    return data, {}

# A dense voxel grid of this many cells or more is refused: the state arrays
# (occupancy / background / persistence) and the bincount minlength would each
# allocate one entry per cell, so a mis-set voxel size + wide bounds could ask
# for gigabytes. At/above the cap the node passes its input through untouched
# and warns once — the fix is a coarser voxel size or tighter bounds.
MAX_VOXEL_CELLS = 40_000_000

# Fixed normalisation for pc_voxel's output weights: weight =
# count * (d * sense)^k / this (k = distance compensation power), so 'sense'
# — the analogue of the C++ VOXEL SENSE slider — is the only user-facing gain.
# 100 makes sense=1 match the node's previous default ('weight scale' 100).
VOXEL_WEIGHT_NORM = 100.0

# Slack when testing a smoothed count against the integer 'min points'; see
# PointCloudVoxelNode.execute.
COUNT_EPSILON = 1e-3


def register_point_cloud_nodes():
    Node.app.register_node('pc_crop', PointCloudCropNode.factory)
    Node.app.register_node('pc_voxel', PointCloudVoxelNode.factory)
    Node.app.register_node('pc_background', PointCloudBackgroundNode.factory)
    Node.app.register_node('pc_denoise', PointCloudDenoiseNode.factory)
    Node.app.register_node('pc_info', PointCloudInfoNode.factory)
    Node.app.register_node('pc_cluster_filter', PointCloudClusterFilterNode.factory)


class _VoxelGrid:
    """Maps points to integer voxels over an axis-aligned volume.

    Rebuilt (via ``configure``) only when the bounds or voxel size actually
    change, so steady-state capture pays nothing. ``index`` is the hot path:
    a floor + compare + linear-combination, fully vectorised."""

    def __init__(self):
        self.lo = np.zeros(3, dtype=np.float32)
        self.dims = np.ones(3, dtype=np.int64)   # (nx, ny, nz)
        self.inv = np.ones(3, dtype=np.float32)
        self.voxel_size = np.ones(3, dtype=np.float32)
        self.ncells = 1
        self._key = None

    def configure(self, lo, hi, voxel_size, dims=None):
        """(Re)build the grid. Returns True if the geometry changed. Raises
        ValueError if the resulting grid would exceed MAX_VOXEL_CELLS.

        ``voxel_size`` is a scalar for cubic voxels or a length-3 (x, y, z)
        for anisotropic ones; it is stored as a (3,) float32 either way.

        ``dims`` pins the grid to an exact voxel count per axis and derives the
        voxel size from it instead, which is what box subdivision needs: asking
        for ceil((hi - lo) / size) back would let one float32 ulp turn 64
        voxels into 65 and leave the last box a single voxel deep."""
        lo = np.asarray(lo, dtype=np.float32)
        hi = np.asarray(hi, dtype=np.float32)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        if dims is None:
            voxel_size = np.asarray(voxel_size, dtype=np.float32).reshape(-1)
            if voxel_size.size == 1:
                voxel_size = np.repeat(voxel_size, 3)
            voxel_size = np.maximum(voxel_size[:3], 1e-6)
            dims = np.maximum(np.ceil((hi - lo) / voxel_size).astype(np.int64), 1)
        else:
            dims = np.maximum(np.asarray(dims, dtype=np.int64).reshape(-1)[:3], 1)
            voxel_size = np.maximum((hi - lo) / dims, 1e-6).astype(np.float32)
        key = (tuple(lo.tolist()), tuple(hi.tolist()),
               tuple(voxel_size.tolist()), tuple(dims.tolist()))
        if key == self._key:
            return False
        ncells = int(dims[0] * dims[1] * dims[2])
        if ncells > MAX_VOXEL_CELLS:
            raise ValueError(
                f'voxel grid too large: {dims.tolist()} = {ncells:,} cells '
                f'(> {MAX_VOXEL_CELLS:,}); use a coarser voxel size or tighter bounds')
        self.lo = lo
        self.dims = dims
        self.inv = (1.0 / voxel_size).astype(np.float32)
        self.voxel_size = voxel_size
        self.ncells = ncells
        self._key = key
        return True

    def voxel_size_meta(self):
        """Metadata form of the voxel size: a float when cubic, else (x, y, z)."""
        vs = self.voxel_size
        if vs[0] == vs[1] == vs[2]:
            return float(vs[0])
        return [float(v) for v in vs]

    def index(self, pts):
        """Return (lin, valid): lin is the (N,) int64 linear voxel index (only
        meaningful where valid), valid is the (N,) bool in-bounds mask."""
        vi = np.floor((pts - self.lo) * self.inv).astype(np.int64)   # (N, 3)
        valid = ((vi[:, 0] >= 0) & (vi[:, 0] < self.dims[0]) &
                 (vi[:, 1] >= 0) & (vi[:, 1] < self.dims[1]) &
                 (vi[:, 2] >= 0) & (vi[:, 2] < self.dims[2]))
        nx, ny = self.dims[0], self.dims[1]
        lin = vi[:, 0] + vi[:, 1] * nx + vi[:, 2] * (nx * ny)
        return lin, valid

    def coords(self, lin_indices):
        """Unpack linear voxel indices to (M, 3) int64 (ix, iy, iz)."""
        nx, ny = self.dims[0], self.dims[1]
        iz = lin_indices // (nx * ny)
        rem = lin_indices - iz * (nx * ny)
        iy = rem // nx
        ix = rem - iy * nx
        return np.stack((ix, iy, iz), axis=1)

    def centres(self, lin_indices):
        """Voxel centres (M, 3) float32 for an array of linear voxel indices."""
        ijk = self.coords(lin_indices).astype(np.float32)
        return (self.lo + (ijk + 0.5) * self.voxel_size).astype(np.float32)


class PointCloudNode(Node):
    """Shared plumbing: pull an (N, 3) float32 cloud (raw or cloud-frame dict)
    off the trigger input, keep its metadata, and re-wrap on send."""

    in_raw = None    # the frame exactly as received (for passthrough)
    in_meta = {}     # metadata of the current frame ({} for raw input)

    def _get_cloud(self):
        self.in_raw = self.input()
        pts, self.in_meta = unwrap_cloud(self.in_raw)
        if pts is None:
            return None
        data = any_to_array(pts)
        if data is None or not isinstance(data, np.ndarray) or data.size == 0:
            return None
        if data.ndim == 1 and data.size % 3 == 0:
            data = data.reshape(-1, 3)
        if data.ndim != 2 or data.shape[1] != 3:
            if self.app.verbose:
                print(f'{self.label}: expected (N, 3) point cloud, got shape {data.shape}')
            return None
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        return data

    def _send(self, out_pin, pts, **meta_updates):
        """Send pts, carrying incoming metadata (plus any updates) forward as a
        cloud-frame dict; plain input with no metadata stays a plain array."""
        meta = {**self.in_meta, **meta_updates}
        if meta:
            out = dict(meta)
            out[CLOUD_KEY] = pts
            out_pin.send(out)
        else:
            out_pin.send(pts)

    def _bounds(self, fallback_lo, fallback_hi):
        """Volume bounds for grid-based nodes: the crop spec carried in the
        frame wins; the node's own min/max options are the raw-input fallback."""
        crop = self.in_meta.get(CROP_KEY)
        if crop is not None:
            try:
                lo = np.asarray(crop[0], dtype=np.float32).reshape(-1)[:3]
                hi = np.asarray(crop[1], dtype=np.float32).reshape(-1)[:3]
                if lo.size == 3 and hi.size == 3:
                    return lo, hi
            except (IndexError, TypeError, ValueError):
                pass
        return (self._vec3(self.min_option, fallback_lo),
                self._vec3(self.max_option, fallback_hi))

    def _carried_voxel_size(self):
        """Voxel size riding in on the frame (attached by an upstream
        pc_voxel), in metres — float or (x, y, z) — or None. Grid nodes prefer
        it over their own widget, mirroring how _bounds() treats the crop."""
        vs = self.in_meta.get(VOXEL_SIZE_KEY)
        if vs is None:
            return None
        try:
            v = np.asarray(vs, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if v.size not in (1, 3) or not np.all(v > 0):
            return None
        return v

    def _add_bounds_options(self, lo_default, hi_default):
        """Fallback volume bounds, used only when no crop spec rides in on the
        frame — tucked into options to keep the node body clean."""
        # Kept so anything needing the volume later (the bounds draw) can ask
        # for it without repeating the node's defaults.
        self._bounds_defaults = (list(lo_default), list(hi_default))
        self.min_option = self.add_option('min (x,y,z)', widget_type='drag_float_n',
                                          default_value=list(lo_default), columns=3, widget_width=60)
        self.max_option = self.add_option('max (x,y,z)', widget_type='drag_float_n',
                                          default_value=list(hi_default), columns=3, widget_width=60)

    def _vec3(self, node_input, fallback):
        try:
            v = np.asarray(any_to_array(node_input()), dtype=np.float32).reshape(-1)
        except Exception:
            return np.asarray(fallback, dtype=np.float32)
        if v.size < 3:
            return np.asarray(fallback, dtype=np.float32)
        return v[:3]


def _lattice_cuts(lo, hi, divisions):
    """Cut-plane positions per axis: d[a] + 1 of them, the ends included."""
    d = np.maximum(np.asarray(divisions, dtype=np.int64).reshape(-1)[:3], 1)
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    return [np.linspace(lo[a], hi[a], int(d[a]) + 1, dtype=np.float32) for a in range(3)]


def _lattice_vertices(lo, hi, divisions):
    """The (M, 3) float32 lattice nodes — every cut plane intersection, which
    is every corner of every cell, each listed once."""
    cx, cy, cz = _lattice_cuts(lo, hi, divisions)
    gx, gy, gz = np.meshgrid(cx, cy, cz, indexing='ij')
    return np.stack((gx.ravel(), gy.ravel(), gz.ravel()), axis=1).astype(np.float32)


def _lattice_lines(lo, hi, divisions, skip_outer_edges=False):
    """Vertices (M, 2, 3) float32 for the wireframe of a box subdivided into
    ``divisions`` cells per axis.

    Drawn as three families of parallel lines rather than twelve edges per
    cell: an 8x8x8 subdivision is 243 lines this way and 6,144 edges the other,
    for the same picture, since every interior edge is shared. Divisions of
    (1, 1, 1) degenerate to exactly the twelve edges of the outer box, which is
    what an unsubdivided volume wants.

    ``skip_outer_edges`` drops the twelve lines that ARE those box edges — the
    ones whose two perpendicular coordinates are both at an extreme. Without
    it, drawing the bounds over the lattice puts two lines in the same place
    and the bounds colour is whatever the two blend to rather than its own."""
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    cuts = _lattice_cuts(lo, hi, divisions)
    segments = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        # One line spanning `axis` at every node of the (u, v) cut lattice.
        gu, gv = np.meshgrid(cuts[u], cuts[v], indexing='ij')
        if skip_outer_edges:
            edge_u = np.zeros(cuts[u].size, dtype=bool)
            edge_v = np.zeros(cuts[v].size, dtype=bool)
            edge_u[[0, -1]] = True
            edge_v[[0, -1]] = True
            keep = ~(edge_u[:, None] & edge_v[None, :])
            gu, gv = gu[keep], gv[keep]
        n = gu.size
        if n == 0:
            continue
        seg = np.empty((n, 2, 3), dtype=np.float32)
        seg[:, :, u] = gu.reshape(n, 1)
        seg[:, :, v] = gv.reshape(n, 1)
        seg[:, 0, axis] = lo[axis]
        seg[:, 1, axis] = hi[axis]
        segments.append(seg)
    if not segments:
        return np.empty((0, 2, 3), dtype=np.float32)
    return np.concatenate(segments, axis=0)


_GL_MODULES = None


def _gl_modules():
    """(moderngl, MGLContext), or (None, None) where moderngl is not installed.

    Imported on first draw rather than at module import: these nodes are
    numpy-only and load in patches that never open a 3D view, and
    moderngl_nodes is an independently switchable module in the app config."""
    global _GL_MODULES
    if _GL_MODULES is None:
        try:
            import moderngl
            from dpg_system.moderngl_base import MGLContext
            _GL_MODULES = (moderngl, MGLContext)
        except ImportError:
            _GL_MODULES = (None, None)
    return _GL_MODULES


class VolumeGridDrawMixin:
    """Lets a grid node sit in an mgl chain and draw its working volume.

    The node gets 'mgl chain in' / 'mgl chain out' pins and behaves like any
    mgl_ node on the chain: a 'draw' message draws the volume as a wireframe
    and is then passed along. The volume is the one the node is actually
    working in — the crop carried on the last cloud frame, or the node's own
    min/max options when the frame arrived raw — so it shows the real working
    volume rather than a second copy of the numbers. Where the node subdivides
    that volume (``_grid_divisions``), the wireframe is the lattice of cell
    boundaries instead of a bare outline.

    The lattice draws as lines, as points at its nodes, or both, each with its
    own colour; both off is how the overlay is turned off. The colours are used
    as set rather than multiplied into the chain colour, so lines and points
    can be told apart at a glance — this is a reference overlay, not scene
    geometry that should take the chain's material.

    Drawing is main-thread only, for the reason given in MGLNode.execute: a
    cloud arriving on a sensor thread can find the chain's 'draw' sitting in
    the input, and running GL there has no context current and segfaults. The
    message is left unconsumed instead, for the main-thread chain trigger that
    is about to process it."""

    _grid_vert_src = '''
        #version 330
        uniform mat4 M;
        uniform mat4 V;
        uniform mat4 P;
        uniform float point_size;
        in vec3 in_position;
        void main() {
            gl_Position = P * V * M * vec4(in_position, 1.0);
            gl_PointSize = point_size;
        }
    '''
    # Unlit: a reference wireframe wants one flat colour, not lines that
    # brighten and dim with the light rig they happen to be drawn under.
    # One program serves both passes; `round_points` is off for the line pass,
    # where gl_PointCoord means nothing.
    _grid_frag_src = '''
        #version 330
        uniform vec4 color;
        uniform bool round_points;
        out vec4 f_color;
        void main() {
            if (round_points) {
                vec2 c = 2.0 * gl_PointCoord - 1.0;
                if (dot(c, c) > 1.0) discard;
            }
            f_color = vec4(color.rgb * color.a, color.a);
        }
    '''

    def _add_volume_draw(self):
        """Add the chain pins and the overlay options. Call after the node's
        own inputs and outputs so the pins land at the end of each column."""
        self.mgl_input = self.add_input('mgl chain in', triggers_execution=True)
        self.mgl_output = self.add_output('mgl chain out')
        # Lines, points at the lattice nodes, or both — both off is the way to
        # turn the overlay off. Colours are taken as set rather than multiplied
        # into the chain colour, so the two can be told apart at a glance.
        #
        # The defaults are the settings arrived at in patches/femto_test: green
        # lines at 18% alpha under brighter green points at 47%, which sits
        # over a live cloud without burying it. Colour defaults are on dpg's
        # own 0..255 scale, since that is what add_color_picker takes and what
        # the widget reads back (_rgba rescales); a 0..1 default here would
        # draw the picker almost black.
        self.show_lines_option = self.add_option('show lines', widget_type='checkbox',
                                                 default_value=True)
        self.line_color_option = self.add_option('line color', widget_type='color_picker',
                                                 default_value=[9.247, 255.0, 0.0, 45.333])
        self.show_points_option = self.add_option('show points', widget_type='checkbox',
                                                  default_value=True)
        self.point_color_option = self.add_option('point color', widget_type='color_picker',
                                                  default_value=[0.0, 255.0, 1.086, 118.996])
        self.point_size_option = self.add_option('point size', widget_type='drag_float',
                                                 default_value=3.0, min=1.0)
        # The outer box on its own colour. With a fine subdivision the lattice
        # reads as a haze and the extent of the volume is the thing you lose;
        # this puts it back without brightening every interior line.
        self.show_bounds_option = self.add_option('show bounds', widget_type='checkbox',
                                                  default_value=True)
        self.bounds_color_option = self.add_option('bounds color', widget_type='color_picker',
                                                   default_value=[255.0, 150.0, 0.0, 120.0])
        self._grid_prog = None
        self._grid_line_vbo = None
        self._grid_line_vao = None
        self._grid_line_verts = 0
        self._grid_point_vbo = None
        self._grid_point_vao = None
        self._grid_point_verts = 0
        self._grid_bounds_vbo = None
        self._grid_bounds_vao = None
        self._grid_bounds_verts = 0
        self._grid_key = None

    def _grid_divisions(self):
        """Cells per axis to draw the volume subdivided into. The default draws
        the outline only; a node that subdivides overrides this."""
        return (1, 1, 1)

    def _mgl_pending(self):
        """True if a chain message is waiting and this is the thread that may
        act on it."""
        return (self.mgl_input.fresh_input
                and threading.current_thread() is threading.main_thread())

    def _handle_mgl(self):
        """Consume one chain message. Anything that is not a 'draw' is simply
        dropped, as it is for the mgl_ nodes that do not implement it."""
        message = self.mgl_input()
        if isinstance(message, list):
            message = message[0] if message and isinstance(message[0], str) else None
        if message != 'draw':
            return
        if (self.show_lines_option() or self.show_points_option()
                or self.show_bounds_option()):
            try:
                self._draw_volume()
            except Exception as e:
                if self.app.verbose:
                    print(f'{self.label}: volume draw failed: {e}')
        self.mgl_output.send('draw')

    @staticmethod
    def _rgba(option):
        """A colour widget's value as an rgba 4-tuple of floats 0..1. The
        picker hands back 0..255 in some themes, hence the rescale."""
        white = (1.0, 1.0, 1.0, 1.0)
        try:
            c = [float(v) for v in any_to_array(option()).reshape(-1)[:4]]
        except (TypeError, ValueError):
            return white
        # A widget that is missing or not yet drawn reads back as a scalar or
        # an empty list; anything short of rgb is not a colour.
        if len(c) < 3:
            return white
        if max(c) > 1.0:
            c = [v / 255.0 for v in c]
        if len(c) < 4:
            c.append(1.0)
        return tuple(min(1.0, max(0.0, v)) for v in c)

    def _draw_volume(self):
        moderngl, MGLContext = _gl_modules()
        if moderngl is None:
            return
        ctx = MGLContext.get_instance()
        inner_ctx = getattr(ctx, 'ctx', None)
        if inner_ctx is None:
            return

        lo, hi = self._bounds(*self._bounds_defaults)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        divisions = self._grid_divisions()

        if self._grid_prog is None:
            self._grid_prog = inner_ctx.program(vertex_shader=self._grid_vert_src,
                                                fragment_shader=self._grid_frag_src)

        # Whether the bounds are drawn changes the lattice itself, so it is
        # part of the key.
        bounds = bool(self.show_bounds_option())
        key = (tuple(lo.tolist()), tuple(hi.tolist()), tuple(divisions), bounds)
        if key != self._grid_key:
            self._build_lattice(ctx, inner_ctx, lo, hi, divisions, bounds)
            self._grid_key = key

        prog = self._grid_prog
        prog['M'].write(ctx.get_model_matrix().astype('f4').T.tobytes())
        prog['V'].write(ctx.view_matrix.astype('f4').tobytes())
        prog['P'].write(ctx.projection_matrix.astype('f4').tobytes())

        if self.show_lines_option() and self._grid_line_verts:
            prog['color'].value = self._rgba(self.line_color_option)
            prog['round_points'].value = False
            self._grid_line_vao.render(mode=moderngl.LINES)
        if self.show_points_option() and self._grid_point_verts:
            prog['color'].value = self._rgba(self.point_color_option)
            prog['round_points'].value = True
            prog['point_size'].value = max(1.0, float(self.point_size_option()))
            self._grid_point_vao.render(mode=moderngl.POINTS)
        if self.show_bounds_option() and self._grid_bounds_verts:
            # Last, so the outline sits over the lattice rather than under it.
            prog['color'].value = self._rgba(self.bounds_color_option)
            prog['round_points'].value = False
            self._grid_bounds_vao.render(mode=moderngl.LINES)

    def _build_lattice(self, ctx, inner_ctx, lo, hi, divisions, skip_outer_edges):
        """(Re)fill the line and point buffers for this volume. The vertex
        count changes with the subdivision, so a buffer is reallocated when its
        shape changes and only rewritten when the volume merely moves."""
        lattice = _lattice_lines(lo, hi, divisions, skip_outer_edges)
        for verts, attr in ((lattice.reshape(-1, 3), 'line'),
                            (_lattice_vertices(lo, hi, divisions), 'point'),
                            # (1, 1, 1) degenerates to the twelve outer edges.
                            (_lattice_lines(lo, hi, (1, 1, 1)).reshape(-1, 3), 'bounds')):
            verts = np.ascontiguousarray(verts, dtype=np.float32)
            count = verts.shape[0]
            if count != getattr(self, f'_grid_{attr}_verts'):
                ctx.defer_release(getattr(self, f'_grid_{attr}_vao'),
                                  getattr(self, f'_grid_{attr}_vbo'))
                vbo = inner_ctx.buffer(reserve=max(verts.nbytes, 12))
                setattr(self, f'_grid_{attr}_vbo', vbo)
                setattr(self, f'_grid_{attr}_vao', inner_ctx.vertex_array(
                    self._grid_prog, [(vbo, '3f', 'in_position')]))
                setattr(self, f'_grid_{attr}_verts', count)
            if count:
                getattr(self, f'_grid_{attr}_vbo').write(verts.tobytes())

    def custom_cleanup(self):
        # Node deletion runs from a DPG handler callback with no GL context
        # current, so release has to be handed back to the context.
        _, MGLContext = _gl_modules()
        ctx = MGLContext._instance if MGLContext is not None else None
        if ctx is not None:
            ctx.defer_release(self._grid_line_vao, self._grid_line_vbo,
                              self._grid_point_vao, self._grid_point_vbo,
                              self._grid_bounds_vao, self._grid_bounds_vbo,
                              self._grid_prog)
        self._grid_line_vao = self._grid_line_vbo = None
        self._grid_point_vao = self._grid_point_vbo = None
        self._grid_bounds_vao = self._grid_bounds_vbo = None
        self._grid_line_verts = self._grid_point_verts = 0
        self._grid_bounds_verts = 0
        self._grid_prog = None
        self._grid_key = None
        super().custom_cleanup()


class PointCloudCropNode(PointCloudNode):
    """Keep only points inside an axis-aligned box (or, inverted, outside it)."""

    @staticmethod
    def factory(name, data, args=None):
        return PointCloudCropNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('point cloud', triggers_execution=True)
        self.min_input = self.add_input('min (x,y,z)', widget_type='drag_float_n',
                                        default_value=[-3.0, -3.0, 0.0], columns=3, widget_width=60)
        self.max_input = self.add_input('max (x,y,z)', widget_type='drag_float_n',
                                        default_value=[3.0, 3.0, 6.0], columns=3, widget_width=60)
        self.invert_input = self.add_input('invert', widget_type='checkbox', default_value=False)
        self.output = self.add_output('cropped')

    def execute(self):
        pts = self._get_cloud()
        if pts is None:
            return
        lo = self._vec3(self.min_input, [-3.0, -3.0, 0.0])
        hi = self._vec3(self.max_input, [3.0, 3.0, 6.0])
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        # Column-wise with in-place &= : ~8x faster than
        # np.all((pts >= lo) & (pts <= hi), axis=1), which allocates (N,3)
        # temporaries and reduces over a 3-wide axis.
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        mask = x >= lo[0]
        mask &= x <= hi[0]
        mask &= y >= lo[1]
        mask &= y <= hi[1]
        mask &= z >= lo[2]
        mask &= z <= hi[2]
        if self.invert_input():
            # Inverted output is NOT bounded by the box, so don't advertise it.
            np.invert(mask, out=mask)
            self._send(self.output, np.compress(mask, pts, axis=0))
            return
        # Attach the crop spec so downstream grid nodes inherit these bounds.
        self._send(self.output, np.compress(mask, pts, axis=0),
                   **{CROP_KEY: (lo.tolist(), hi.tolist())})


class PointCloudVoxelNode(VolumeGridDrawMixin, PointCloudNode):
    """Voxel-grid downsample: collapse each occupied voxel to one point (its
    centre or the centroid of the points it holds). ``min points`` doubles as a
    density floor, dropping sparse speckle voxels.

    The output frame carries per-voxel ``weights`` — count * (d * sense)^k /
    VOXEL_WEIGHT_NORM, clamped to 0..1 — for count-reflecting rendering in
    ``mgl_point_cloud``. ``distance compensation`` picks k: the voxel's
    distance from the sensor (linear) or its square (squared, matching the
    physics: a voxel at 2x distance subtends 1/4 the depth pixels); ``sense``
    is the C++ VOXEL SENSE gain, applied inside the compensation exactly as
    there ((d*sense)^2 / d*sense / sense for squared / linear / none). Radial
    distance is used rather than the C++ code's z so it survives leveling/yaw
    rotations, which preserve |p| but not z.

    ``boxes (x,y,z)`` groups the voxels into a coarser lattice of boxes, the
    first of the cluster-frame producers (see CLUSTER_KEY). Boxes are not a
    layer on top of the voxels but a constraint on them: the crop divides into
    exactly the requested number of boxes, and the voxel size is then derived
    so that a whole number of voxels spans each one —

        box size  = (hi - lo) / boxes
        n         = round(box size / target voxel size)   # voxels per box
        voxel size = box size / n

    which is cPointCloudToVoxels::CalcOptimalVoxelSize from the C++ app. It is
    closed form, so nothing has to search or be fed back: 'voxel size (cm)' is
    a target, and the size actually used (on the frame, and reported by
    pc_info) is the nearest one that divides the boxes evenly. Every box then
    holds exactly n.x * n.y * n.z voxels with no remainder. The cost is that
    voxels cannot stay exactly cubic for an arbitrary crop — the C++ app makes
    the same trade, snapping each axis independently after copying the x
    target across.

    The node also sits on an mgl chain: a 'draw' arriving on ``mgl chain in``
    draws the working volume as a wireframe — the box lattice when subdivided,
    the bare outline when not — and is passed on, so the volume the voxels are
    really built over can be seen alongside the cloud. See
    VolumeGridDrawMixin."""

    @staticmethod
    def factory(name, data, args=None):
        return PointCloudVoxelNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.grid = _VoxelGrid()
        self._warned_large = False
        self.box_count = None        # (3,) int64 boxes per axis, or None
        self.voxels_per_box = None   # (3,) int64, exact, when boxes are on
        self._count_lin = None       # (M,) sorted linear indices carrying state
        self._count_val = None       # (M,) their filtered counts
        self._count_dx = None        # (M,) their filtered rate of change
        self._count_time = None
        self._hyst_lin = None        # (M,) sorted indices with hysteresis state
        self._hyst_conf = None       # (M,) 0..1 how established each one is
        self._hyst_on = None         # (M,) whether it is currently occupied
        self.input = self.add_input('point cloud', triggers_execution=True)
        self.voxel_input = self.add_input('voxel size (cm)', widget_type='drag_float',
                                          default_value=5.0, min=0.01)
        self.distcomp_property = self.add_property('distance compensation', widget_type='combo',
                                                   default_value='squared')
        self.distcomp_property.widget.combo_items = ['none', 'linear', 'squared']
        # VOXEL SENSE analogue from the C++ voxels app: weight gain applied
        # inside the distance compensation — doubling it brightens 4x in
        # 'squared' mode, 2x in 'linear'/'none'.
        self.sense_property = self.add_property('sense', widget_type='drag_float',
                                                default_value=1.0, min=0.0, max=4.0)
        self.sense_property.widget.speed = 0.01
        self.min_points_property = self.add_property('min points', widget_type='drag_int',
                                                     default_value=1, min=1)
        # One euro on the per-voxel count: 'count cutoff' is the resting
        # cutoff in Hz (lower smooths harder), 'count beta' how far a fast
        # change opens it up. 0 cutoff is off.
        self.count_cutoff_property = self.add_property('count cutoff (Hz)',
                                                       widget_type='drag_float',
                                                       default_value=0.0, min=0.0)
        self.count_cutoff_property.widget.speed = 0.01
        # Hysteresis: what a voxel must hold to APPEAR, as against 'min points'
        # which is only what it must hold to stay. 0 turns it off.
        self.appear_points_property = self.add_property('points to appear',
                                                        widget_type='drag_int',
                                                        default_value=0, min=0)
        # 0 on any axis leaves the cloud unsubdivided; the voxel size is then
        # taken as typed rather than snapped. A drag_float_n shown as whole
        # numbers — there is no integer row widget, and the bounds vectors
        # beside it are drag_float_n too, so the node stays of a piece.
        self.boxes_input = self.add_input('boxes (x,y,z)', widget_type='drag_float_n',
                                          default_value=[0.0, 0.0, 0.0], columns=3,
                                          widget_width=60, min=0.0)
        self.boxes_input.widget.speed = 1.0
        self.output = self.add_output('voxel cloud')
        self.count_output = self.add_output('counts')
        self.box_output = self.add_output('box values')
        self.reduce_option = self.add_option('reduce', widget_type='combo', default_value='center')
        self.reduce_option.widget.combo_items = ['center', 'centroid']
        self._add_bounds_options([-3.0, -3.0, 0.0], [3.0, 3.0, 6.0])
        # Voxels are cubes ('voxel size (cm)') unless 'cubic voxels' is off, in
        # which case width/height/depth come from 'voxel size x,y,z (cm)' —
        # matching the C++ voxels app. UI is in cm; the cloud itself is metres.
        self.cubic_option = self.add_option('cubic voxels', widget_type='checkbox',
                                            default_value=True)
        self.voxel_xyz_option = self.add_option('voxel size x,y,z (cm)', widget_type='drag_float_n',
                                                default_value=[5.0, 5.0, 5.0],
                                                columns=3, widget_width=60)
        self.count_beta_option = self.add_option('count beta', widget_type='drag_float',
                                                 default_value=0.2, min=0.0)
        self.count_beta_option.widget.speed = 0.01
        # How quickly a voxel that keeps showing up earns the lower threshold.
        self.settle_option = self.add_option('settle (s)', widget_type='drag_float',
                                             default_value=0.25, min=0.01)
        self.settle_option.widget.speed = 0.01
        # Last, so the chain pins land at the foot of each column and 'show
        # volume' at the foot of the options — and so patches saved before the
        # pins existed still reconnect: a link restores by its saved index
        # first, and appending leaves every existing index where it was.
        self._add_volume_draw()

    def custom_create(self, from_file):
        # Format has to be applied to drawn items, so not in __init__.
        self.boxes_input.widget.set_format('%.0f')

    def _target_voxel_size(self):
        """The requested voxel size in metres, (3,) float32. Cubic copies the
        single widget across all three axes — before any box snapping, exactly
        as the C++ app does."""
        if self.cubic_option():
            size = np.repeat(np.float32(self.voxel_input()), 3)
        else:
            size = self._vec3(self.voxel_xyz_option, [5.0, 5.0, 5.0])
        return np.maximum(size.astype(np.float32) * 0.01, 1e-4)   # cm -> m

    def _requested_boxes(self):
        """Boxes per axis as (3,) int64, or None when subdivision is off."""
        try:
            b = np.rint(np.asarray(any_to_array(self.boxes_input()),
                                   dtype=np.float64)).astype(np.int64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if b.size == 1:
            b = np.repeat(b, 3)
        if b.size < 3 or np.any(b[:3] < 1):
            return None
        return b[:3]

    def _grid_divisions(self):
        """What the mgl chain draws: the box lattice, or the bare outline."""
        if self.box_count is None:
            return (1, 1, 1)
        return tuple(int(v) for v in self.box_count)

    def _ensure_grid(self):
        lo, hi = self._bounds(*self._bounds_defaults)
        target = self._target_voxel_size()
        boxes = self._requested_boxes()
        dims = None
        if boxes is not None:
            # CalcOptimalVoxelSize: the crop divides into exactly `boxes`
            # boxes, and the voxel size bends to the nearest one that fits a
            # whole number of voxels into each. dims is handed to the grid
            # rather than recomputed from the size, so the count is exact.
            extent = np.maximum(np.asarray(hi, dtype=np.float32) -
                                np.asarray(lo, dtype=np.float32), 1e-6)
            box_size = extent / boxes
            per_box = np.maximum(np.rint(box_size / target), 1).astype(np.int64)
            dims = boxes * per_box
            size = box_size / per_box
        else:
            per_box = None
            size = target
        try:
            if self.grid.configure(lo, hi, size, dims=dims):
                # A different grid means voxel k is a different piece of the
                # room; the smoothing history no longer refers to anything.
                self._count_lin = None
                self._count_val = None
                self._count_dx = None
                self._count_time = None
                self._hyst_lin = None
                self._hyst_conf = None
                self._hyst_on = None
            self.box_count = boxes
            self.voxels_per_box = per_box
            self._warned_large = False
            return True
        except ValueError as e:
            if not self._warned_large:
                print(f'{self.label}: {e}')
                self._warned_large = True
            return False

    def _measure_dt(self):
        """Seconds since the last frame, measured rather than assumed, and
        clamped: a stalled sensor must not empty the filters' history in one
        step, and the first frame has no interval."""
        now = time.perf_counter()
        previous, self._count_time = self._count_time, now
        if previous is None:
            return 1.0 / 30.0
        return min(max(now - previous, 1.0 / 240.0), 0.5)

    def _filter_counts(self, counts):
        """One Euro on the per-voxel count, returning (indices, filtered).

        Thresholding a raw count makes a voxel blink whenever the count crosses
        'min points', and within a point or two of the line the count is mostly
        sensor noise. What is wanted is a filter that ignores a stray reading
        but does not hesitate over a real one, and the two are told apart by
        how fast the count is moving, not by how big it is: One Euro smooths at
        'count cutoff' while a voxel is quiet and opens the cutoff up by
        'count beta' times the (low-passed) rate of change when it is not.

        The two simpler things both fail, in opposite directions. A plain
        low-pass is systematically below the true count while it is rising, so
        it dims and deletes the moving content it lags behind. Clamping it to
        never fall below the raw count fixes that but amplifies noise instead:
        a stray reading is taken at face value and then held. Measured over a
        two-frame stray of 4 and 3 points against 'min points' 3, the clamped
        version lights the voxel for 2 frames and a real body sweeping past
        leaves 13 frames of trail where the truth was 6; One Euro at 0.2/0.2
        lights the stray for 0 and trails 7, with no lag at all on arrival.

        Run over the union of the voxels holding points this frame and those
        still carrying a value, never the whole grid: a 6 m crop at 5 cm is
        1.7M cells, of which a live cloud occupies a few thousand.
        """
        cutoff = max(0.0, float(self.count_cutoff_property()))
        if cutoff <= 0.0:
            return None
        dt = self._measure_dt()

        here = np.nonzero(counts)[0]
        if self._count_lin is None or self._count_lin.size == 0:
            union = here
            prev = np.zeros(union.size, dtype=np.float32)
            prev_dx = np.zeros(union.size, dtype=np.float32)
        else:
            union = np.union1d(here, self._count_lin)      # sorted
            prev = np.zeros(union.size, dtype=np.float32)
            prev_dx = np.zeros(union.size, dtype=np.float32)
            pos = np.searchsorted(self._count_lin, union)
            pos = np.minimum(pos, self._count_lin.size - 1)
            hit = self._count_lin[pos] == union
            prev[hit] = self._count_val[pos[hit]]
            prev_dx[hit] = self._count_dx[pos[hit]]
        if union.size == 0:
            self._count_lin = self._count_val = self._count_dx = None
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32), dt

        raw = counts[union].astype(np.float32)
        # The derivative is taken against the previous filtered value and is
        # itself low-passed at 1 Hz, so one noisy sample cannot unlock the
        # filter at the moment it is most needed.
        a_d = dt / (dt + 1.0 / (2.0 * np.pi))
        dx = (raw - prev) / dt
        speed = (prev_dx + a_d * (dx - prev_dx)).astype(np.float32)
        fc = cutoff + max(0.0, float(self.count_beta_option())) * np.abs(speed)
        alpha = dt / (dt + 1.0 / (2.0 * np.pi * fc))
        filtered = (prev + alpha * (raw - prev)).astype(np.float32)
        # Drop the tail so the state does not grow to the size of the grid as
        # noise wanders over it. Well under any usable 'min points'.
        alive = filtered > 0.05
        self._count_lin = union[alive]
        self._count_val = filtered[alive]
        self._count_dx = speed[alive]
        return union, filtered, dt

    def _threshold(self, union, values, min_points, dt):
        """Which of ``union`` are occupied, returning (indices, their values).

        Plain comparison against ``min_points`` unless 'points to appear' is
        set above it, which turns the threshold into a Schmitt trigger: a voxel
        that has been empty has to clear the high threshold to appear at all,
        while one already showing stays on any count at or above 'min points'.

        The high threshold is not fixed either. A voxel that keeps showing up
        earns its way down towards the low one over 'settle' seconds, so
        established geometry is cheap to hold and a fresh voxel in a part of
        the room that has been quiet is expensive to create. That is the shape
        of the noise: a single frame over the threshold and nothing the next.

        It is a better answer than simply raising 'min points', which cannot
        work at range — a real surface at 6 m only puts about 9 points in a
        5 cm voxel, so a threshold high enough to reject speckle rejects the
        surface too. Measured over 90 frames of a simulated room, raising min
        points from 3 to 8 cut true voxels kept from 99.3% to 72.2% and made
        one-frame blinking WORSE (80k to 118k, since the count then chatters
        across the higher line). Hysteresis at 8 -> 3 kept 98.4% and took the
        blinking to 179; with the count filter in front of it, to 4.
        """
        appear = int(self.appear_points_property())
        if appear <= min_points:
            picked = values >= min_points - COUNT_EPSILON
            return union[picked], values[picked]

        # State has to cover voxels that are quiet now but still established,
        # so it is carried on its own index rather than this frame's.
        if self._hyst_lin is None or self._hyst_lin.size == 0:
            all_lin = union
            conf = np.zeros(union.size, dtype=np.float32)
            was_on = np.zeros(union.size, dtype=bool)
            vals = values
        else:
            all_lin = np.union1d(union, self._hyst_lin)
            conf = np.zeros(all_lin.size, dtype=np.float32)
            was_on = np.zeros(all_lin.size, dtype=bool)
            vals = np.zeros(all_lin.size, dtype=np.float32)
            pos = np.minimum(np.searchsorted(self._hyst_lin, all_lin), self._hyst_lin.size - 1)
            hit = self._hyst_lin[pos] == all_lin
            conf[hit] = self._hyst_conf[pos[hit]]
            was_on[hit] = self._hyst_on[pos[hit]]
            if union.size:
                upos = np.minimum(np.searchsorted(all_lin, union), all_lin.size - 1)
                vals[upos] = values

        entry = appear - (appear - min_points) * conf
        on = np.where(was_on,
                      vals >= min_points - COUNT_EPSILON,
                      vals >= entry - COUNT_EPSILON)
        # Confidence tracks whether the voxel is HOLDING POINTS, not whether it
        # won the argument about being drawn. Otherwise a voxel that is
        # genuinely there but weak — a real surface at 6 m puts only about 9
        # points in a 5 cm voxel — could never earn its threshold down, since
        # it would have to be on already to start. On this reading a voxel that
        # keeps its count up settles in after 'settle' seconds whatever the
        # entry threshold was, while one that blinks at a 50% duty only ever
        # reaches half way down and stays out.
        candidate = (vals >= min_points - COUNT_EPSILON).astype(np.float32)
        alpha = dt / (dt + max(0.01, float(self.settle_option())))
        conf = (conf + alpha * (candidate - conf)).astype(np.float32)

        # Keep only what still matters: occupied, or still carrying enough
        # confidence to be worth a lowered threshold.
        alive = on | (conf > 0.02) | (candidate > 0)
        self._hyst_lin = all_lin[alive]
        self._hyst_conf = conf[alive]
        self._hyst_on = on[alive]
        return all_lin[on], vals[on]

    def _cluster_boxes(self, occupied, weights):
        """Sum voxel weights into boxes, send the dense (bx, by, bz) array, and
        return the frame's cluster entry (None when subdivision is off).

        ``occupied`` may be empty: the box array still goes out, all zeros, so
        a display downstream clears rather than holding the last frame."""
        if self.box_count is None:
            return None
        bx, by, bz = (int(v) for v in self.box_count)
        n_boxes = bx * by * bz
        if occupied.size == 0:
            labels = np.empty((0,), dtype=np.int32)
            values = np.zeros(n_boxes, dtype=np.float32)
        else:
            # dims is boxes * voxels_per_box exactly, so this divides cleanly:
            # every voxel lands in a box and no box is short.
            bijk = self.grid.coords(occupied) // self.voxels_per_box
            labels = bijk[:, 0] + bijk[:, 1] * bx + bijk[:, 2] * (bx * by)
            values = np.bincount(labels, weights=weights,
                                 minlength=n_boxes).astype(np.float32)
            labels = labels.astype(np.int32)
        # The linear index is x + bx*y + bx*by*z, so the C-order unpack is
        # [z][y][x]; transpose back to [x][y][z] for the output array.
        self.box_output.send(np.ascontiguousarray(
            values.reshape(bz, by, bx).transpose(2, 1, 0)))
        return {
            'labels': labels,
            'values': values,
            'shape': (bx, by, bz),
            'origin': self.grid.lo.copy(),
            'cell': (self.grid.voxel_size * self.voxels_per_box).astype(np.float32),
        }

    def execute(self):
        # Two trigger inputs: the cloud and the mgl chain. A 'draw' must not
        # re-run the voxeliser over the retained cloud (self.input() hands back
        # the last frame whether or not it is fresh), and a cloud arriving on a
        # sensor thread must not stall waiting for the chain.
        if self._mgl_pending():
            self._handle_mgl()
            if not self.input.fresh_input:
                return
        pts = self._get_cloud()
        if pts is None:
            return
        if not self._ensure_grid():
            self._send(self.output, np.ascontiguousarray(pts))   # pass through unfiltered
            return
        lin, valid = self.grid.index(pts)
        lin_v = lin[valid]
        empty = np.empty((0,), dtype=np.int64)

        def send_empty():
            # An empty result still describes the same grid, so it carries the
            # same metadata. Dropping 'weights' and 'voxel_size' here is not
            # harmless: a renderer reads those to decide how to size its points
            # at all, so an empty frame used to switch mgl_point_cloud from
            # weighted, voxel-sized sprites to unweighted ones at its fallback
            # widget size — which is what a room with nobody in it produced,
            # every frame that nothing reached 'min points'.
            self._send(self.output, np.empty((0, 3), dtype=np.float32),
                       voxel_size=self.grid.voxel_size_meta(),
                       weights=np.empty((0,), dtype=np.float32))
            self.count_output.send(empty)
            self._cluster_boxes(empty, None)

        if lin_v.size == 0:
            send_empty()
            return
        counts = np.bincount(lin_v, minlength=self.grid.ncells)
        min_points = max(1, int(self.min_points_property()))
        filtered = self._filter_counts(counts)
        if filtered is None:
            union = np.nonzero(counts)[0]
            values_all = counts[union].astype(np.float32)
            dt = self._measure_dt()
        else:
            union, values_all, dt = filtered
        # COUNT_EPSILON throughout: the count filter approaches its target
        # asymptotically, so a voxel resting at exactly 'min points' settles a
        # hair under it and would be excluded for ever.
        occupied, values = self._threshold(union, values_all, min_points, dt)
        if occupied.size == 0:
            send_empty()
            return

        if self.reduce_option() == 'centroid':
            pts_v = pts[valid]
            sx = np.bincount(lin_v, weights=pts_v[:, 0], minlength=self.grid.ncells)
            sy = np.bincount(lin_v, weights=pts_v[:, 1], minlength=self.grid.ncells)
            sz = np.bincount(lin_v, weights=pts_v[:, 2], minlength=self.grid.ncells)
            # The centroid divides by the points that actually landed, never by
            # the smoothed count. A voxel held open by the smoothing can have
            # none this frame, and its centroid is then undefined — it falls
            # back to the voxel centre, which is where it was heading anyway.
            raw = counts[occupied].astype(np.float32)
            out = np.stack((sx[occupied], sy[occupied], sz[occupied]), axis=1).astype(np.float32)
            empty_now = raw <= 0.0
            np.divide(out, np.maximum(raw, 1.0)[:, None], out=out)
            if empty_now.any():
                out[empty_now] = self.grid.centres(occupied[empty_now])
        else:
            out = self.grid.centres(occupied)

        self.count_output.send(np.rint(values).astype(np.int64))

        weights = values.astype(np.float32)
        distcomp = self.distcomp_property()
        sense = max(0.0, float(self.sense_property()))
        if distcomp != 'none':
            d = np.sqrt(out[:, 0] ** 2 + out[:, 1] ** 2 + out[:, 2] ** 2)
            d *= sense
            weights *= d if distcomp == 'linear' else d * d
        else:
            weights *= sense
        weights = np.clip(weights / VOXEL_WEIGHT_NORM, 0.0, 1.0)

        meta = {VOXEL_SIZE_KEY: self.grid.voxel_size_meta(), 'weights': weights}
        clusters = self._cluster_boxes(occupied, weights)
        if clusters is not None:
            meta[CLUSTER_KEY] = clusters
        self._send(self.output, np.ascontiguousarray(out), **meta)


class PointCloudBackgroundNode(PointCloudNode):
    """Static background subtraction. Press ``learn`` with the scene empty to
    accumulate an occupancy model over ``frames`` captures; a voxel occupied in
    at least ``min hits`` of them becomes background. Thereafter, points landing
    in a background voxel are removed; out-of-bounds points pass through."""

    @staticmethod
    def factory(name, data, args=None):
        return PointCloudBackgroundNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.grid = _VoxelGrid()
        self._warned_large = False
        self.hits = None            # (ncells,) int32 accumulator during learning
        self.bg_mask = None         # (ncells,) bool background occupancy
        self.learn_remaining = 0
        self.learn_total = 0

        self.input = self.add_input('point cloud', triggers_execution=True)
        self.voxel_input = self.add_input('voxel size (cm)', widget_type='drag_float',
                                          default_value=5.0, min=0.01)
        self.learn_input = self.add_input('learn', widget_type='button', callback=self.start_learning)
        self.frames_input = self.add_input('frames', widget_type='drag_int', default_value=60, min=1)
        self.min_hits_input = self.add_input('min hits', widget_type='drag_int', default_value=20, min=1)
        self.dilate_input = self.add_input('dilate (voxels)', widget_type='drag_int',
                                           default_value=0, min=0)
        self.clear_input = self.add_input('clear', widget_type='button', callback=self.clear_background)
        self.output = self.add_output('foreground')
        self._add_bounds_options([-3.0, -3.0, 0.0], [3.0, 3.0, 6.0])

    def _ensure_grid(self):
        lo, hi = self._bounds([-3.0, -3.0, 0.0], [3.0, 3.0, 6.0])
        size = self._carried_voxel_size()
        if size is None:
            size = float(self.voxel_input()) * 0.01   # cm -> m
        try:
            changed = self.grid.configure(lo, hi, size)
            self._warned_large = False
        except ValueError as e:
            if not self._warned_large:
                print(f'{self.label}: {e}')
                self._warned_large = True
            return False
        if changed:
            # Geometry moved: any learned model no longer maps to these cells.
            # Say so when a learning run is abandoned -- pressing 'learn' before
            # the first cloud arrives builds the grid from the fallback bounds,
            # and the first frame's crop then rebuilds it, which used to cancel
            # the run in silence right after announcing it had started.
            if self.learn_remaining > 0:
                print(f'{self.label}: volume changed mid-learn -- learning '
                      f'abandoned. Press learn again with the cloud running.')
            self.hits = None
            self.bg_mask = None
            self.learn_remaining = 0
        return True

    def start_learning(self):
        if not self._ensure_grid():
            return
        self.learn_total = max(1, int(self.frames_input()))
        self.learn_remaining = self.learn_total
        self.hits = np.zeros(self.grid.ncells, dtype=np.int32)
        self.bg_mask = None
        print(f'{self.label}: learning background over {self.learn_total} frames '
              f'(keep the volume empty)')

    def clear_background(self):
        self.hits = None
        self.bg_mask = None
        self.learn_remaining = 0
        print(f'{self.label}: background cleared')

    def _finalize_background(self):
        min_hits = min(int(self.min_hits_input()), self.learn_total)
        self.bg_mask = self.hits >= max(1, min_hits)
        dilate = int(self.dilate_input())
        if dilate > 0:
            self._dilate_background(dilate)
        self.hits = None
        print(f'{self.label}: background learned ({int(self.bg_mask.sum()):,} '
              f'of {self.grid.ncells:,} voxels occupied)')

    def _dilate_background(self, iterations):
        """Grow the background by N voxels so points skimming a learned surface
        are still suppressed. Uses scipy if present; otherwise a cheap 6-neighbour
        shift dilation on the reshaped grid."""
        nx, ny, nz = (int(self.grid.dims[0]), int(self.grid.dims[1]), int(self.grid.dims[2]))
        vol = self.bg_mask.reshape(nz, ny, nx)
        try:
            from scipy.ndimage import binary_dilation
            vol = binary_dilation(vol, iterations=iterations)
        except Exception:
            for _ in range(iterations):
                grown = vol.copy()
                grown[1:, :, :] |= vol[:-1, :, :]
                grown[:-1, :, :] |= vol[1:, :, :]
                grown[:, 1:, :] |= vol[:, :-1, :]
                grown[:, :-1, :] |= vol[:, 1:, :]
                grown[:, :, 1:] |= vol[:, :, :-1]
                grown[:, :, :-1] |= vol[:, :, 1:]
                vol = grown
        self.bg_mask = np.ascontiguousarray(vol).reshape(-1)

    def execute(self):
        pts = self._get_cloud()
        if pts is None:
            return
        if not self._ensure_grid():
            self._send(self.output, np.ascontiguousarray(pts))
            return
        lin, valid = self.grid.index(pts)
        lin_v = lin[valid]

        if self.learn_remaining > 0:
            if self.hits is not None and lin_v.size:
                # count each occupied voxel once per frame
                occ = np.unique(lin_v)
                self.hits[occ] += 1
            self.learn_remaining -= 1
            if self.learn_remaining == 0:
                self._finalize_background()
            self._send(self.output, np.ascontiguousarray(pts))   # show the scene while learning
            return

        if self.bg_mask is None or lin_v.size == 0:
            self._send(self.output, np.ascontiguousarray(pts))
            return

        is_bg = self.bg_mask[lin_v]
        keep = np.ones(pts.shape[0], dtype=bool)
        keep[np.nonzero(valid)[0][is_bg]] = False   # remove in-bounds background hits
        self._send(self.output, np.compress(keep, pts, axis=0))


class PointCloudDenoiseNode(PointCloudNode):
    """Voxel-based speckle/flicker removal, kd-tree-free.

    ``min points`` drops voxels holding fewer than k points this frame (spatial
    density / statistical-outlier surrogate). ``persistence`` (0 disables) keeps
    an exponential-moving-average occupancy per voxel and drops voxels whose EMA
    is below the threshold — i.e. voxels that only flicker on briefly. Both
    filters apply only inside the volume; out-of-bounds points pass through."""

    @staticmethod
    def factory(name, data, args=None):
        return PointCloudDenoiseNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.grid = _VoxelGrid()
        self._warned_large = False
        self.persist = None         # (ncells,) float32 EMA occupancy

        self.input = self.add_input('point cloud', triggers_execution=True)
        self.voxel_input = self.add_input('voxel size (cm)', widget_type='drag_float',
                                          default_value=4.0, min=0.01)
        self.min_points_input = self.add_input('min points', widget_type='drag_int',
                                               default_value=2, min=1)
        self.persistence_input = self.add_input('persistence', widget_type='drag_float',
                                                default_value=0.0, min=0.0, max=1.0)
        self.decay_input = self.add_input('decay', widget_type='drag_float',
                                          default_value=0.7, min=0.0, max=0.999)
        self.output = self.add_output('denoised')
        self._add_bounds_options([-5.0, -5.0, -1.0], [5.0, 5.0, 10.0])

    def _ensure_grid(self):
        lo, hi = self._bounds([-5.0, -5.0, -1.0], [5.0, 5.0, 10.0])
        size = self._carried_voxel_size()
        if size is None:
            size = float(self.voxel_input()) * 0.01   # cm -> m
        try:
            changed = self.grid.configure(lo, hi, size)
            self._warned_large = False
        except ValueError as e:
            if not self._warned_large:
                print(f'{self.label}: {e}')
                self._warned_large = True
            return False
        if changed:
            self.persist = None
        return True

    def execute(self):
        pts = self._get_cloud()
        if pts is None:
            return
        if not self._ensure_grid():
            self._send(self.output, np.ascontiguousarray(pts))
            return
        lin, valid = self.grid.index(pts)
        lin_v = lin[valid]
        if lin_v.size == 0:
            self._send(self.output, np.ascontiguousarray(pts))
            return

        counts = np.bincount(lin_v, minlength=self.grid.ncells)
        min_points = max(1, int(self.min_points_input()))
        voxel_ok = counts >= min_points

        persistence = float(self.persistence_input())
        if persistence > 0.0:
            if self.persist is None or self.persist.shape[0] != self.grid.ncells:
                self.persist = np.zeros(self.grid.ncells, dtype=np.float32)
            decay = float(self.decay_input())
            occ = (counts > 0).astype(np.float32)
            # EMA in [0, 1]: steady occupancy -> 1, brief flicker stays low.
            self.persist *= decay
            self.persist += (1.0 - decay) * occ
            voxel_ok &= self.persist >= persistence

        is_ok = voxel_ok[lin_v]
        keep = np.ones(pts.shape[0], dtype=bool)
        keep[np.nonzero(valid)[0][~is_ok]] = False   # drop in-bounds noise voxels
        self._send(self.output, np.compress(keep, pts, axis=0))


class PointCloudInfoNode(PointCloudNode):
    """Report count / axis-aligned bounds / centroid of the incoming cloud, to
    help set crop and voxel bounds. Passes the cloud through unchanged."""

    @staticmethod
    def factory(name, data, args=None):
        return PointCloudInfoNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('point cloud', triggers_execution=True)
        self.count_output = self.add_output('count')
        self.min_output = self.add_output('min')
        self.max_output = self.add_output('max')
        self.centroid_output = self.add_output('centroid')
        self.passthrough_output = self.add_output('cloud out')

    def execute(self):
        pts = self._get_cloud()
        if pts is None:
            self.count_output.send(0)
            return
        self.passthrough_output.send(self.in_raw)   # frame unchanged, dict or raw
        self.centroid_output.send(pts.mean(axis=0).astype(np.float32))
        self.max_output.send(pts.max(axis=0).astype(np.float32))
        self.min_output.send(pts.min(axis=0).astype(np.float32))
        self.count_output.send(int(pts.shape[0]))

class PointCloudClusterFilterNode(PointCloudNode):
    """Condition the per-cluster values on a cloud frame, in place.

    Everything after pc_voxel's bincount is a (K,) signal per frame, and none
    of it is specific to voxels — which is why it lives here rather than in
    pc_voxel. Put one of these between pc_voxel and a display or an OSC stage,
    or chain two, or none. It will serve blobs and painted regions unchanged
    once those exist, because it only touches clusters['values'].

    The chain is the C++ cBins one, in its order, since the order matters:

      gate / squeeze   ThresholdMotions and SqueezeMotions. 'gate' zeroes
                       anything at or below the threshold and passes the rest
                       untouched; 'squeeze' subtracts the threshold instead, so
                       a box just over it starts from zero rather than jumping.
      filter           see below. 'decay' bleeds a constant off every frame
                       either way, which stops a cluster resting on a residue.
      motion           The C++ dynamicBins: the frame-to-frame change rather
                       than the level, |v - v_last|. Differencing amplifies
                       noise, which is why it comes after the smoothing rather
                       than before it — the same order cBins uses.

    Two filters, both adaptive — smooth hard while a cluster is idle, get out
    of the way when it moves — differing in how they decide which it is.

    'one euro' (Casiez, Roussel, Lafon, CHI 2012) is the default. It low-passes
    the value at a cutoff that rises with the cluster's *filtered* speed:
    cutoff = 'min cutoff' + 'beta' * |speed|. Tune it as the paper says — set
    beta to 0 and lower min cutoff until an idle cluster stops shimmering, then
    raise beta until a real arrival stops lagging.

    'adaptive' is the C++ AdaptiveBinFilter, kept for fidelity with the app:
    the smoothing coefficient is driven by the raw residual |new - filtered|,
    floored at 'smooth' and fully released at 'knee'.

    One euro wins on both counts it was measured on. On a 30 s synthetic box
    signal with counting noise it took less lag than the adaptive filter at
    every jitter budget (about 10% less). More importantly its parameters are
    in Hz against a measured dt, so its time constant does not move when the
    frame rate does: over 10..90 fps the adaptive filter's step response ranged
    0.044 s to 0.400 s, a factor of 9, while one euro held 0.400 s throughout.
    That matters here, where the capture rate is not guaranteed and a frame
    task's output is capped by the display refresh.
    """

    @staticmethod
    def factory(name, data, args=None):
        return PointCloudClusterFilterNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('cloud', triggers_execution=True)
        self.threshold_input = self.add_input('threshold', widget_type='drag_float',
                                              default_value=0.0, min=0.0)
        self.threshold_input.widget.speed = 0.1
        # One euro's two controls: min cutoff sets how still an idle cluster
        # looks, beta how little a moving one lags.
        self.cutoff_input = self.add_input('min cutoff', widget_type='drag_float',
                                           default_value=1.0, min=0.01)
        self.cutoff_input.widget.speed = 0.01
        self.beta_input = self.add_input('beta', widget_type='drag_float',
                                         default_value=0.005, min=0.0)
        self.beta_input.widget.speed = 0.001
        self.motion_input = self.add_input('motion', widget_type='checkbox',
                                           default_value=False)
        self.output = self.add_output('cloud out')
        self.values_output = self.add_output('values')
        self.gate_option = self.add_option('gate', widget_type='combo', default_value='squeeze')
        self.gate_option.widget.combo_items = ['none', 'gate', 'squeeze']
        self.filter_option = self.add_option('filter', widget_type='combo',
                                             default_value='one euro')
        self.filter_option.widget.combo_items = ['none', 'one euro', 'adaptive']
        # Cutoff for one euro's own speed estimate. 1 Hz is the paper's value
        # and rarely wants changing: it is what stops a noise spike unlocking
        # the filter at the moment it is most needed.
        self.dcutoff_option = self.add_option('d cutoff', widget_type='drag_float',
                                              default_value=1.0, min=0.01)
        self.dcutoff_option.widget.speed = 0.01
        # The 'adaptive' path: cBins' baseFilterDegree and filterThreshold.
        self.smooth_option = self.add_option('smooth', widget_type='drag_float',
                                             default_value=0.8, min=0.0, max=0.999)
        self.smooth_option.widget.speed = 0.005
        self.knee_option = self.add_option('knee', widget_type='drag_float',
                                           default_value=0.0, min=0.0)
        self.knee_option.widget.speed = 0.1
        self.decay_option = self.add_option('decay', widget_type='drag_float',
                                            default_value=0.0, min=0.0)
        self.decay_option.widget.speed = 0.005
        self._filtered = None    # (K,) filtered value, both filters
        self._speed = None       # (K,) one euro's filtered derivative
        self._last = None        # (K,) previous output, for motion
        self._state_size = 0
        self._last_time = None   # for the measured dt

    def _reset_state(self, k):
        """Drop the per-cluster history when the lattice changes shape — a new
        box count means cluster 7 is a different piece of the room."""
        if self._state_size != k:
            self._filtered = np.zeros(k, dtype=np.float32)
            self._speed = np.zeros(k, dtype=np.float32)
            self._last = np.zeros(k, dtype=np.float32)
            self._state_size = k
            self._last_time = None

    def _dt(self):
        """Seconds since the last frame, measured rather than assumed.

        Clamped: the first frame has no interval, and a patch that was paused
        or a sensor that stalled would otherwise hand the filter a dt of
        seconds and wipe its state in one step."""
        now = time.perf_counter()
        previous, self._last_time = self._last_time, now
        if previous is None:
            return 1.0 / 30.0
        return min(max(now - previous, 1.0 / 240.0), 0.5)

    @staticmethod
    def _smoothing_factor(dt, cutoff):
        """One euro's alpha: r / (r + 1) for r = 2*pi*cutoff*dt. Equivalent to
        1 / (1 + tau/dt) with tau = 1/(2*pi*cutoff), and cheaper."""
        r = 2.0 * np.pi * cutoff * dt
        return r / (r + 1.0)

    def _one_euro(self, x, dt):
        """Vectorised One Euro over every cluster at once.

        Follows the reference implementation: the derivative is taken against
        the previous *filtered* value, and is itself low-passed at 'd cutoff'
        before it is allowed to open the main filter up. That second filter is
        the whole difference from the C++ one — driving the adaptation from a
        raw residual lets a single noisy sample unlock the smoothing at exactly
        the wrong moment."""
        a_d = self._smoothing_factor(dt, max(0.01, float(self.dcutoff_option())))
        speed = (x - self._filtered) / dt
        self._speed = (a_d * speed + (1.0 - a_d) * self._speed).astype(np.float32)
        cutoff = (max(0.01, float(self.cutoff_input()))
                  + max(0.0, float(self.beta_input())) * np.abs(self._speed))
        a = self._smoothing_factor(dt, cutoff)
        return (a * x + (1.0 - a) * self._filtered).astype(np.float32)

    def _adaptive(self, x, smooth, knee):
        """cBins' AdaptiveBinFilter: the smoothing coefficient floors at
        'smooth' and is released as the raw residual approaches 'knee'."""
        if knee > 0.0:
            change = np.clip(np.sqrt(np.abs(x - self._filtered) / knee), 0.0, 1.0)
            degree = np.maximum(1.0 - change, smooth)
        else:
            degree = smooth
        return (self._filtered * degree + x * (1.0 - degree)).astype(np.float32)

    def _process(self, values):
        self._reset_state(values.size)
        out = values

        threshold = max(0.0, float(self.threshold_input()))
        gate = self.gate_option()
        if threshold > 0.0 and gate != 'none':
            if gate == 'squeeze':
                out = np.maximum(out - threshold, 0.0)
            else:
                out = np.where(out > threshold, out, 0.0)

        which = self.filter_option()
        decay = max(0.0, float(self.decay_option()))
        knee = max(0.0, float(self.knee_option()))
        if which != 'none':
            dt = self._dt()
            if which == 'one euro':
                filtered = self._one_euro(out, dt)
            else:
                filtered = self._adaptive(
                    out, min(0.999, max(0.0, float(self.smooth_option()))), knee)
            if decay > 0.0:
                # Per second, so it bleeds at the same rate whatever the frame
                # rate — the filters are dt-aware now, and this should be too.
                filtered = filtered - decay * dt
            self._filtered = np.maximum(filtered, 0.0).astype(np.float32)
            out = self._filtered

        if self.motion_input():
            motion = np.abs(out - self._last).astype(np.float32)
            self._last = np.array(out, dtype=np.float32, copy=True)
            out = motion
        return np.ascontiguousarray(out, dtype=np.float32)

    def execute(self):
        raw = self.input()
        pts, meta = unwrap_cloud(raw)
        clusters = meta.get(CLUSTER_KEY)
        if not isinstance(clusters, dict) or clusters.get('values') is None:
            # Nothing to condition: pass the frame on untouched rather than
            # swallowing it, so the node can sit in a chain before the boxes
            # are switched on.
            self.output.send(raw)
            return
        values = np.asarray(clusters['values'], dtype=np.float32).reshape(-1)
        out_values = self._process(values)

        clusters = dict(clusters)
        clusters['values'] = out_values
        meta = dict(meta)
        meta[CLUSTER_KEY] = clusters
        out = dict(meta)
        out[CLOUD_KEY] = pts
        self.output.send(out)

        shape = clusters.get('shape')
        if shape is not None and out_values.size == int(np.prod(shape)):
            bx, by, bz = (int(v) for v in shape)
            # Same unpack as pc_voxel's: linear index is x + bx*y + bx*by*z.
            self.values_output.send(np.ascontiguousarray(
                out_values.reshape(bz, by, bx).transpose(2, 1, 0)))
        else:
            self.values_output.send(out_values)
