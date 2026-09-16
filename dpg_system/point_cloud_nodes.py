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


def register_point_cloud_nodes():
    Node.app.register_node('pc_crop', PointCloudCropNode.factory)
    Node.app.register_node('pc_voxel', PointCloudVoxelNode.factory)
    Node.app.register_node('pc_background', PointCloudBackgroundNode.factory)
    Node.app.register_node('pc_denoise', PointCloudDenoiseNode.factory)
    Node.app.register_node('pc_info', PointCloudInfoNode.factory)


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


def _lattice_lines(lo, hi, divisions):
    """Vertices (M, 2, 3) float32 for the wireframe of a box subdivided into
    ``divisions`` cells per axis.

    Drawn as three families of parallel lines rather than twelve edges per
    cell: an 8x8x8 subdivision is 243 lines this way and 6,144 edges the other,
    for the same picture, since every interior edge is shared. Divisions of
    (1, 1, 1) degenerate to exactly the twelve edges of the outer box, which is
    what an unsubdivided volume wants."""
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    cuts = _lattice_cuts(lo, hi, divisions)
    segments = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        # One line spanning `axis` at every node of the (u, v) cut lattice.
        gu, gv = np.meshgrid(cuts[u], cuts[v], indexing='ij')
        n = gu.size
        seg = np.empty((n, 2, 3), dtype=np.float32)
        seg[:, :, u] = gu.reshape(n, 1)
        seg[:, :, v] = gv.reshape(n, 1)
        seg[:, 0, axis] = lo[axis]
        seg[:, 1, axis] = hi[axis]
        segments.append(seg)
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
        self.show_lines_option = self.add_option('show lines', widget_type='checkbox',
                                                 default_value=True)
        self.line_color_option = self.add_option('line color', widget_type='color_picker',
                                                 default_value=[1.0, 1.0, 1.0, 1.0])
        self.show_points_option = self.add_option('show points', widget_type='checkbox',
                                                  default_value=False)
        self.point_color_option = self.add_option('point color', widget_type='color_picker',
                                                  default_value=[1.0, 1.0, 1.0, 1.0])
        self.point_size_option = self.add_option('point size', widget_type='drag_float',
                                                 default_value=4.0, min=1.0)
        self._grid_prog = None
        self._grid_line_vbo = None
        self._grid_line_vao = None
        self._grid_line_verts = 0
        self._grid_point_vbo = None
        self._grid_point_vao = None
        self._grid_point_verts = 0
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
        if self.show_lines_option() or self.show_points_option():
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

        key = (tuple(lo.tolist()), tuple(hi.tolist()), tuple(divisions))
        if key != self._grid_key:
            self._build_lattice(ctx, inner_ctx, lo, hi, divisions)
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

    def _build_lattice(self, ctx, inner_ctx, lo, hi, divisions):
        """(Re)fill the line and point buffers for this volume. The vertex
        count changes with the subdivision, so a buffer is reallocated when its
        shape changes and only rewritten when the volume merely moves."""
        for verts, attr in ((_lattice_lines(lo, hi, divisions).reshape(-1, 3), 'line'),
                            (_lattice_vertices(lo, hi, divisions), 'point')):
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
                              self._grid_prog)
        self._grid_line_vao = self._grid_line_vbo = None
        self._grid_point_vao = self._grid_point_vbo = None
        self._grid_line_verts = self._grid_point_verts = 0
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
            self.grid.configure(lo, hi, size, dims=dims)
            self.box_count = boxes
            self.voxels_per_box = per_box
            self._warned_large = False
            return True
        except ValueError as e:
            if not self._warned_large:
                print(f'{self.label}: {e}')
                self._warned_large = True
            return False

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
        if lin_v.size == 0:
            self._send(self.output, np.empty((0, 3), dtype=np.float32))
            self.count_output.send(empty)
            self._cluster_boxes(empty, None)
            return
        counts = np.bincount(lin_v, minlength=self.grid.ncells)
        min_points = max(1, int(self.min_points_property()))
        occupied = np.nonzero(counts >= min_points)[0]
        if occupied.size == 0:
            self._send(self.output, np.empty((0, 3), dtype=np.float32))
            self.count_output.send(empty)
            self._cluster_boxes(empty, None)
            return

        if self.reduce_option() == 'centroid':
            pts_v = pts[valid]
            sx = np.bincount(lin_v, weights=pts_v[:, 0], minlength=self.grid.ncells)
            sy = np.bincount(lin_v, weights=pts_v[:, 1], minlength=self.grid.ncells)
            sz = np.bincount(lin_v, weights=pts_v[:, 2], minlength=self.grid.ncells)
            denom = counts[occupied].astype(np.float32)
            out = np.stack((sx[occupied], sy[occupied], sz[occupied]), axis=1).astype(np.float32)
            out /= denom[:, None]
        else:
            out = self.grid.centres(occupied)

        self.count_output.send(counts[occupied].astype(np.int64))

        weights = counts[occupied].astype(np.float32)
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