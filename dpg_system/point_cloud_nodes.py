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
  pc_voxel       voxel-grid downsample -> occupied voxel centres / centroids
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
"""

import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import any_to_array

CLOUD_KEY = 'point_cloud'
CROP_KEY = 'crop'
VOXEL_SIZE_KEY = 'voxel_size'


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

    def configure(self, lo, hi, voxel_size):
        """(Re)build the grid. Returns True if the geometry changed. Raises
        ValueError if the resulting grid would exceed MAX_VOXEL_CELLS.

        ``voxel_size`` is a scalar for cubic voxels or a length-3 (x, y, z)
        for anisotropic ones; it is stored as a (3,) float32 either way."""
        lo = np.asarray(lo, dtype=np.float32)
        hi = np.asarray(hi, dtype=np.float32)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        voxel_size = np.asarray(voxel_size, dtype=np.float32).reshape(-1)
        if voxel_size.size == 1:
            voxel_size = np.repeat(voxel_size, 3)
        voxel_size = np.maximum(voxel_size[:3], 1e-6)
        key = (tuple(lo.tolist()), tuple(hi.tolist()), tuple(voxel_size.tolist()))
        if key == self._key:
            return False
        dims = np.ceil((hi - lo) / voxel_size).astype(np.int64)
        dims = np.maximum(dims, 1)
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

    def centres(self, lin_indices):
        """Voxel centres (M, 3) float32 for an array of linear voxel indices."""
        nx, ny = self.dims[0], self.dims[1]
        iz = lin_indices // (nx * ny)
        rem = lin_indices - iz * (nx * ny)
        iy = rem // nx
        ix = rem - iy * nx
        ijk = np.stack((ix, iy, iz), axis=1).astype(np.float32)
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


class PointCloudVoxelNode(PointCloudNode):
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
    rotations, which preserve |p| but not z."""

    @staticmethod
    def factory(name, data, args=None):
        return PointCloudVoxelNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.grid = _VoxelGrid()
        self._warned_large = False
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
        self.output = self.add_output('voxel cloud')
        self.count_output = self.add_output('counts')
        self.reduce_option = self.add_option('reduce', widget_type='combo', default_value='center')
        self.reduce_option.widget.combo_items = ['center', 'centroid']
        self.min_points_option = self.add_option('min points', widget_type='drag_int',
                                                 default_value=1, min=1)
        self._add_bounds_options([-3.0, -3.0, 0.0], [3.0, 3.0, 6.0])
        # Voxels are cubes ('voxel size (cm)') unless 'cubic voxels' is off, in
        # which case width/height/depth come from 'voxel size x,y,z (cm)' —
        # matching the C++ voxels app. UI is in cm; the cloud itself is metres.
        self.cubic_option = self.add_option('cubic voxels', widget_type='checkbox',
                                            default_value=True)
        self.voxel_xyz_option = self.add_option('voxel size x,y,z (cm)', widget_type='drag_float_n',
                                                default_value=[5.0, 5.0, 5.0],
                                                columns=3, widget_width=60)

    def _ensure_grid(self):
        lo, hi = self._bounds([-3.0, -3.0, 0.0], [3.0, 3.0, 6.0])
        if self.cubic_option():
            size = float(self.voxel_input()) * 0.01        # cm -> m
        else:
            size = self._vec3(self.voxel_xyz_option, [5.0, 5.0, 5.0]) * 0.01
        try:
            self.grid.configure(lo, hi, size)
            self._warned_large = False
            return True
        except ValueError as e:
            if not self._warned_large:
                print(f'{self.label}: {e}')
                self._warned_large = True
            return False

    def execute(self):
        pts = self._get_cloud()
        if pts is None:
            return
        if not self._ensure_grid():
            self._send(self.output, np.ascontiguousarray(pts))   # pass through unfiltered
            return
        lin, valid = self.grid.index(pts)
        lin_v = lin[valid]
        if lin_v.size == 0:
            self._send(self.output, np.empty((0, 3), dtype=np.float32))
            self.count_output.send(np.empty((0,), dtype=np.int64))
            return
        counts = np.bincount(lin_v, minlength=self.grid.ncells)
        min_points = max(1, int(self.min_points_option()))
        occupied = np.nonzero(counts >= min_points)[0]
        if occupied.size == 0:
            self._send(self.output, np.empty((0, 3), dtype=np.float32))
            self.count_output.send(np.empty((0,), dtype=np.int64))
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

        self._send(self.output, np.ascontiguousarray(out),
                   voxel_size=self.grid.voxel_size_meta(), weights=weights)


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