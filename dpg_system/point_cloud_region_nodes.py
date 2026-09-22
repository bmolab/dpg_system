"""Hand-authored regions over the voxel volume, and the capture tool for them.

Where pc_voxel's boxes are a regular lattice, a region is an arbitrary set of
cells with a name: 'the chair', 'stage left', 'where she stood'. pc_regions
holds a map of them, sums the incoming cloud into them every frame, and emits
the same cluster-frame dict the boxes do (see point_cloud_nodes), so
pc_cluster_filter and whatever turns values into OSC serve both.

Nodes:
  pc_regions     author + evaluate a region map
  pc_accumulate  record where the cloud has been over a stretch of time

The model, and what it deliberately leaves room for:

A region is stored on its own — a sorted array of cell indices plus an
optional array of per-cell membership weights — rather than as one label per
cell. Nothing in that says two regions cannot hold the same cell, or that a
cell is wholly in or out, so overlapping regions and soft membership are both
already expressible in the map and in the file; v1 simply never creates them
(assignment is 'steal' or 'yield', and weights stay None). The per-frame path
does not walk the regions: they are compiled, on edit only, into as few dense
label layers as it takes for no layer to hold two regions in one cell — one
layer while regions are disjoint — and a frame is then a gather and a bincount
per layer. This is cVoxelMap's arrangement (boxMap authored, mapFrame derived
and handed to the bins) with the authored side generalised.

The map owns its grid, in world metres. It is adopted from the first frame
(crop + voxel size) and then frozen for as long as any region holds a cell,
and points are looked up by position, never by pc_voxel's voxel index. So
changing the voxel size or the crop upstream does not wipe the map, which
changing the box count did in the C++ app.

Authoring is direct: every action lands in the current region at once, and
'undo' is the safety net. There are two kinds. CAPTURE takes the cells under a
cloud — whatever is on 'capture cloud in', or the live cloud when nothing is
connected there: the occupied voxels right now, pc_accumulate's record of a
minute's movement, later a wand brush or a painted slice — and adds them to,
removes them from, or makes them the whole of, the region. EDIT reshapes the
region itself (grow / shrink / fill / hull / box / snap to boxes / move). The
C++ app staged both through a selection first; that meant a capture that
looked like a region but was not one yet, and one set of buttons that acted
on the selection or the region depending on a step you had to remember. To
sculpt a piece before it joins a region that already has cells, capture it
into a spare region number, shape it there, and 'absorb' it.

Cluster-frame additions made here: 'labels' is -1 for a point in no region
(boxes never had one); 'names' and 'colors' ride along per cluster; 'shape' is
None. Once regions may overlap a point can belong to several, and 'labels'
will then hold only the first — 'values' is always complete, and a COO
'membership' (point, cluster, weight) is the reserved form for the rest.
'geometry' is a RegionGeometry: the cell sets themselves, by reference, for
mgl_regions to draw.
"""

import colorsys
import threading
import time

import numpy as np
from dpg_system.node import Node, SaveDialog, LoadDialog
from dpg_system.conversion_utils import any_to_array
from dpg_system.point_cloud_nodes import (
    CLOUD_KEY, CROP_KEY, VOXEL_SIZE_KEY, CLUSTER_KEY,
    PointCloudNode, _VoxelGrid, unwrap_cloud)

REGION_FILE_VERSION = 1
UNDO_DEPTH = 32
# Labels are compiled to int16: half the memory traffic of int32 on the
# per-frame gather, and nobody paints thirty thousand regions by hand.
MAX_REGION_ID = 32000


def register_point_cloud_region_nodes():
    Node.app.register_node('pc_regions', PointCloudRegionsNode.factory)
    Node.app.register_node('pc_accumulate', PointCloudAccumulateNode.factory)


def _region_color(region_id):
    """A colour per region id, far from its neighbours': golden-ratio steps
    round the hue circle, so consecutive ids never land side by side."""
    hue = (0.11 + region_id * 0.61803398875) % 1.0
    return np.asarray(colorsys.hsv_to_rgb(hue, 0.75, 1.0), dtype=np.float32)


# ---------------------------------------------------------------------------
# Cell-set shaping. A cell set is a sorted, unique (M,) int64 array of linear
# indices into a _VoxelGrid. Shaping works on a dense bool volume over the
# set's bounding box only, so its cost follows the selection, not the stage.

def _to_volume(grid, cells, pad):
    """Dense (z, y, x) bool over the bounding box of ``cells`` grown by
    ``pad``, and the (ix, iy, iz) of its corner. The box is NOT clipped to the
    grid — growth is, on the way back — so the corner may be negative."""
    ijk = grid.coords(cells)
    lo = ijk.min(axis=0) - pad
    shape = ijk.max(axis=0) + pad - lo + 1
    vol = np.zeros((shape[2], shape[1], shape[0]), dtype=bool)
    r = ijk - lo
    vol[r[:, 2], r[:, 1], r[:, 0]] = True
    return vol, lo


def _inside_grid(grid, shape_zyx, lo):
    """(z, y, x) bool: which cells of a volume at ``lo`` exist in the grid."""
    axes = []
    for axis in range(3):
        i = np.arange(shape_zyx[2 - axis]) + lo[axis]
        axes.append((i >= 0) & (i < grid.dims[axis]))
    return axes[2][:, None, None] & axes[1][None, :, None] & axes[0][None, None, :]


def _from_volume(grid, vol, lo):
    """Back to a cell set, dropping whatever grew past the grid's edge."""
    vol = vol & _inside_grid(grid, vol.shape, lo)
    z, y, x = np.nonzero(vol)
    nx, ny = grid.dims[0], grid.dims[1]
    # nonzero walks z-major, which is ascending linear order: already sorted.
    return (x + lo[0]) + (y + lo[1]) * nx + (z + lo[2]) * (nx * ny)


def _dilate(vol, steps):
    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(vol, iterations=steps)
    except ImportError:
        for _ in range(steps):
            grown = vol.copy()
            grown[1:, :, :] |= vol[:-1, :, :]
            grown[:-1, :, :] |= vol[1:, :, :]
            grown[:, 1:, :] |= vol[:, :-1, :]
            grown[:, :-1, :] |= vol[:, 1:, :]
            grown[:, :, 1:] |= vol[:, :, :-1]
            grown[:, :, :-1] |= vol[:, :, 1:]
            vol = grown
        return vol


def grow_cells(grid, cells, steps):
    if cells.size == 0 or steps < 1:
        return cells
    vol, lo = _to_volume(grid, cells, steps)
    return _from_volume(grid, _dilate(vol, steps), lo)


def shrink_cells(grid, cells, steps):
    """Erode. Space beyond the grid counts as filled, so a region standing
    against the wall of the stage is not eaten away from the wall side."""
    if cells.size == 0 or steps < 1:
        return cells
    vol, lo = _to_volume(grid, cells, 1)
    outside = ~_inside_grid(grid, vol.shape, lo)
    eroded = ~_dilate(~(vol | outside), steps)
    return _from_volume(grid, eroded, lo)


def fill_cells(grid, cells, steps):
    """Close gaps up to ``steps`` cells wide, then fill what is enclosed. A
    captured body is a one-sided shell with holes in it, not a solid, so the
    closing is what does most of the work; fill_holes alone finds nothing to
    fill in a shell that is open to the far side."""
    if cells.size == 0:
        return cells
    steps = max(0, steps)
    vol, lo = _to_volume(grid, cells, steps + 1)
    if steps:
        vol = ~_dilate(~_dilate(vol, steps), steps)
    try:
        from scipy.ndimage import binary_fill_holes
        vol = binary_fill_holes(vol)
    except ImportError:
        pass
    return _from_volume(grid, vol, lo)


def hull_cells(grid, cells):
    """Every cell whose centre lies in the convex hull of the set. Returns
    None if the set has no volume to hull (flat, or too few cells)."""
    if cells.size < 4:
        return None
    try:
        from scipy.spatial import ConvexHull, Delaunay, QhullError
    except ImportError:
        return None
    ijk = grid.coords(cells)
    try:
        corners = ijk[ConvexHull(ijk).vertices]
        tri = Delaunay(corners)
    except (QhullError, ValueError):
        return None
    lo, hi = ijk.min(axis=0), ijk.max(axis=0)
    shape = hi - lo + 1
    z, y, x = np.indices((shape[2], shape[1], shape[0]))
    candidates = np.stack((x.ravel() + lo[0], y.ravel() + lo[1], z.ravel() + lo[2]), axis=1)
    inside = tri.find_simplex(candidates) >= 0
    nx, ny = grid.dims[0], grid.dims[1]
    c = candidates[inside]
    # The hull's own corners can test as outside by a rounding error.
    return np.union1d(c[:, 0] + c[:, 1] * nx + c[:, 2] * (nx * ny), cells)


def box_cells(grid, cells):
    """The axis-aligned bounding box of the set, filled."""
    if cells.size == 0:
        return cells
    ijk = grid.coords(cells)
    lo, hi = ijk.min(axis=0), ijk.max(axis=0)
    vol = np.ones((hi[2] - lo[2] + 1, hi[1] - lo[1] + 1, hi[0] - lo[0] + 1), dtype=bool)
    return _from_volume(grid, vol, lo)


def move_cells(grid, cells, offset):
    """Shift by whole cells; what leaves the grid is lost."""
    if cells.size == 0:
        return cells
    ijk = grid.coords(cells) + np.asarray(offset, dtype=np.int64)
    ok = np.all((ijk >= 0) & (ijk < grid.dims), axis=1)
    ijk = ijk[ok]
    nx, ny = grid.dims[0], grid.dims[1]
    return np.sort(ijk[:, 0] + ijk[:, 1] * nx + ijk[:, 2] * (nx * ny))


def extend_cells(grid, cells, axis, direction='both'):
    """Flood the set along one axis: every column of cells (fixed on the
    other two axes) is filled to the grid's edge — both ways, or '+' / '-'
    from what the column already holds. A captured footprint becomes a pillar
    from floor to ceiling; a wall becomes the whole depth of the stage."""
    if cells.size == 0:
        return cells
    ijk = grid.coords(cells)
    others = [a for a in range(3) if a != axis]
    key = ijk[:, others[0]] * grid.dims[others[1]] + ijk[:, others[1]]
    columns, inverse = np.unique(key, return_inverse=True)
    lo = np.full(columns.size, grid.dims[axis], dtype=np.int64)
    hi = np.full(columns.size, -1, dtype=np.int64)
    np.minimum.at(lo, inverse, ijk[:, axis])
    np.maximum.at(hi, inverse, ijk[:, axis])
    if direction in ('both', '-'):
        lo[:] = 0
    if direction in ('both', '+'):
        hi[:] = grid.dims[axis] - 1
    counts = hi - lo + 1
    along = np.repeat(lo, counts) + (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts))
    out = np.empty((along.size, 3), dtype=np.int64)
    out[:, axis] = along
    out[:, others[0]] = np.repeat(columns // grid.dims[others[1]], counts)
    out[:, others[1]] = np.repeat(columns % grid.dims[others[1]], counts)
    nx, ny = grid.dims[0], grid.dims[1]
    return np.sort(out[:, 0] + out[:, 1] * nx + out[:, 2] * (nx * ny))


def surface_cells(grid, cells):
    """The cells with at least one of their six neighbours outside the set:
    what there is to see of a solid region, at a fraction of the count."""
    if cells.size == 0:
        return cells
    vol, lo = _to_volume(grid, cells, 1)
    interior = ~_dilate(~vol, 1)
    return _from_volume(grid, vol & ~interior, lo)


def snap_cells_to_lattice(grid, cells, origin, cell, shape, coverage):
    """Replace the set by whole lattice boxes: every box at least ``coverage``
    (0..1) full of selected cells, in its entirety. Boxes are the coarse
    granularity of the C++ app, kept as a way of shaping a selection rather
    than as what a region is made of."""
    if cells.size == 0:
        return cells
    origin = np.asarray(origin, dtype=np.float32)
    cell = np.asarray(cell, dtype=np.float32)
    shape = np.asarray(shape, dtype=np.int64)
    # Which box each grid line of cells falls in, per axis, by cell centre.
    axis_box = []
    for a in range(3):
        centre = grid.lo[a] + (np.arange(grid.dims[a]) + 0.5) * grid.voxel_size[a]
        b = np.floor((centre - origin[a]) / cell[a]).astype(np.int64)
        b[(b < 0) | (b >= shape[a])] = -1
        axis_box.append(b)
    ijk = grid.coords(cells)
    b = np.stack([axis_box[a][ijk[:, a]] for a in range(3)], axis=1)
    b = b[np.all(b >= 0, axis=1)]
    if b.shape[0] == 0:
        return cells[:0]
    n_boxes = int(shape.prod())
    lin_b = b[:, 0] + b[:, 1] * shape[0] + b[:, 2] * (shape[0] * shape[1])
    selected = np.bincount(lin_b, minlength=n_boxes)
    per_axis = [np.bincount(axis_box[a][axis_box[a] >= 0], minlength=shape[a]) for a in range(3)]
    capacity = (per_axis[0][None, None, :] * per_axis[1][None, :, None]
                * per_axis[2][:, None, None]).reshape(-1)
    chosen = (selected >= np.maximum(coverage * capacity, 1e-9)) & (selected > 0)
    chosen = chosen.reshape(shape[2], shape[1], shape[0])
    # Expand the chosen boxes to cells, within their bounding slab only.
    picks = [np.flatnonzero(np.isin(axis_box[a], np.nonzero(chosen.any(
        axis=tuple(i for i in range(3) if i != 2 - a)))[0])) for a in range(3)]
    if any(p.size == 0 for p in picks):
        return cells[:0]
    x, y, z = picks
    keep = chosen[axis_box[2][z][:, None, None], axis_box[1][y][None, :, None],
                  axis_box[0][x][None, None, :]]
    zi, yi, xi = np.nonzero(keep)
    nx, ny = grid.dims[0], grid.dims[1]
    return x[xi] + y[yi] * nx + z[zi] * (nx * ny)


# ---------------------------------------------------------------------------
# Geometry for drawing a cell set (mgl_regions). Numpy only, like the rest of
# this module: the renderer just uploads what these return.

def boundary_faces(grid, cells):
    """The skin of a cell set: one quad per cell face that has the set on one
    side only. Returns (corners (F*4, 3) float32 in world metres, indices
    (F*6,) int32), wound counter-clockwise seen from outside.

    Faces rather than a cube per surface cell, because the drawing is
    translucent and additive: with back faces culled a convex region then puts
    exactly one face over any pixel, however many cells deep it is, which is
    what makes a region's brightness mean its value and not its thickness
    along the line of sight. It is the same reasoning as the culling note on
    mgl_cluster_boxes' cube, extended to a shape that is not one cube."""
    if cells.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)
    vol, lo = _to_volume(grid, cells, 1)
    quads = []
    for a in range(3):                      # a: the xyz axis the face looks along
        b, c = (a + 1) % 3, (a + 2) % 3     # a = b x c, so (b, c) order is CCW from +a
        ax = 2 - a                          # the same axis in the (z, y, x) volume
        here = [slice(None)] * 3
        there = [slice(None)] * 3
        for positive in (True, False):
            here[ax], there[ax] = (slice(0, -1), slice(1, None)) if positive else \
                                  (slice(1, None), slice(0, -1))
            z, y, x = np.nonzero(vol[tuple(here)] & ~vol[tuple(there)])
            if z.size == 0:
                continue
            cell = np.stack((x, y, z), axis=1)
            if not positive:
                cell[:, a] += 1             # 'here' began one cell in
            ring = ((0, 0), (1, 0), (1, 1), (0, 1)) if positive else \
                   ((0, 0), (0, 1), (1, 1), (1, 0))
            quad = np.repeat(cell[:, None, :], 4, axis=1)
            quad[:, :, a] += 1 if positive else 0
            for k, (db, dc) in enumerate(ring):
                quad[:, k, b] += db
                quad[:, k, c] += dc
            quads.append(quad)
    quads = np.concatenate(quads).reshape(-1, 3)
    corners = (grid.lo + (quads + lo).astype(np.float32) * grid.voxel_size).astype(np.float32)
    first = np.arange(0, corners.shape[0], 4, dtype=np.int32)[:, None]
    indices = (first + np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)).reshape(-1)
    return corners, indices


def crease_edges(grid, cells):
    """The outline of a cell set: the cell edges along which its skin turns a
    corner, as (E*2, 3) float32 line-segment endpoints in world metres. For a
    box-shaped region that is the twelve edges of the box and nothing else;
    the edges between coplanar faces, which would draw the whole voxel mesh
    over every flat side, are left out.

    Of the four cells round an edge, the skin is flat there exactly when they
    split two against two along a plane; any other mix — one in, three in, or
    two diagonally — is a corner."""
    if cells.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    vol, lo = _to_volume(grid, cells, 1)
    segments = []
    for a in range(3):                      # edges running along xyz axis a
        v = np.moveaxis(vol, 2 - a, 2)      # -> (p, q, along), p < q in zyx order
        a00, a01, a10, a11 = v[:-1, :-1], v[:-1, 1:], v[1:, :-1], v[1:, 1:]
        count = a00.astype(np.int8) + a01 + a10 + a11
        flat = ((a00 == a01) & (a10 == a11) & (a00 != a10)) | \
               ((a00 == a10) & (a01 == a11) & (a00 != a01))
        p, q, i = np.nonzero((count > 0) & (count < 4) & ~flat)
        if p.size == 0:
            continue
        others = [axis for axis in (2, 1, 0) if axis != a]   # xyz axes of p, q
        start = np.empty((p.size, 3), dtype=np.int64)
        start[:, others[0]] = p + 1         # the corner between cells p and p+1
        start[:, others[1]] = q + 1
        start[:, a] = i
        end = start.copy()
        end[:, a] += 1
        segments.append(np.stack((start, end), axis=1).reshape(-1, 3))
    if not segments:
        return np.empty((0, 3), dtype=np.float32)
    points = np.concatenate(segments)
    return (grid.lo + (points + lo).astype(np.float32) * grid.voxel_size).astype(np.float32)


class RegionGeometry:
    """What there is to draw, published by pc_regions on every edit and carried
    by reference in the cluster frame ('geometry'). Immutable once built: cell
    arrays are replaced on edit, never written into, so the renderer can tell
    which regions changed by identity and rebuild only those."""
    __slots__ = ('version', 'grid', 'regions', 'current')

    def __init__(self, version, grid, regions, current):
        self.version = version
        self.grid = grid
        self.regions = regions        # {id: cells}
        self.current = current        # id of the region being edited


# ---------------------------------------------------------------------------

class _Region:
    """One region. ``cells`` is sorted and unique; ``weights`` is None (every
    cell wholly in) or a float32 array alongside it. Both are replaced, never
    written into, so an undo snapshot or a compile can hold a reference."""
    __slots__ = ('id', 'name', 'color', 'cells', 'weights')

    def __init__(self, region_id, name=None, color=None, cells=None, weights=None):
        self.id = int(region_id)
        self.name = name if name else f'region {region_id}'
        self.color = _region_color(region_id) if color is None else np.asarray(color, dtype=np.float32)
        self.cells = np.empty((0,), dtype=np.int64) if cells is None else cells
        self.weights = weights

    def copy(self):
        return _Region(self.id, self.name, self.color, self.cells, self.weights)

    def keep(self, mask):
        self.cells = self.cells[mask]
        if self.weights is not None:
            self.weights = self.weights[mask]

    def merge(self, cells):
        """Union ``cells`` in; new cells are wholly in."""
        if self.weights is None:
            self.cells = np.union1d(self.cells, cells)
            return
        fresh = np.setdiff1d(cells, self.cells, assume_unique=True)
        merged = np.concatenate((self.cells, fresh))
        weights = np.concatenate((self.weights, np.ones(fresh.size, dtype=np.float32)))
        order = np.argsort(merged, kind='stable')
        self.cells, self.weights = merged[order], weights[order]


class _Compiled:
    """The map as the per-frame path wants it: immutable, published as one
    reference so a frame on the sensor thread never sees half an edit."""
    __slots__ = ('grid', 'layers', 'count', 'names', 'colors', 'version')


class _RegionMap:
    """The authored side: regions by id on a grid of their own. Ids are slots
    and are never renumbered — 'values' downstream is indexed by id, and an
    OSC mapping built on region 3 must survive region 2 being emptied."""

    def __init__(self):
        self.grid = None
        self.regions = {}
        self.version = 0
        self._undo = []

    def is_empty(self):
        return not any(r.cells.size for r in self.regions.values())

    def set_grid(self, lo, hi, voxel_size):
        """Adopt a grid. Only meaningful while the map is empty — cells are
        indices into the grid, so the caller must not move it under them.
        Returns True if the geometry changed."""
        grid = _VoxelGrid()
        grid.configure(lo, hi, voxel_size)
        if self.grid is not None and grid._key == self.grid._key:
            return False
        self.grid = grid       # a new object, never reconfigured in place
        self.version += 1
        return True

    def cells_under(self, pts):
        lin, valid = self.grid.index(pts)
        return np.unique(lin[valid])

    def region(self, region_id, create=False):
        r = self.regions.get(region_id)
        if r is None and create:
            r = self.regions[region_id] = _Region(region_id)
        return r

    def _checkpoint(self):
        self._undo.append({rid: r.copy() for rid, r in self.regions.items()})
        del self._undo[:-UNDO_DEPTH]

    def undo(self):
        if not self._undo:
            return False
        # Cells go back; names stay. A region named since the checkpoint was
        # not in it, and keeps its slot, empty, rather than vanishing — which
        # would also shorten 'values' under whatever reads it.
        current, self.regions = self.regions, self._undo.pop()
        for rid, region in current.items():
            if rid in self.regions:
                self.regions[rid].name = region.name
            else:
                region.keep(np.zeros(region.cells.size, dtype=bool))
                self.regions[rid] = region
        self.version += 1
        return True

    def assign(self, region_id, cells, mode='steal', replace=False):
        """Put ``cells`` in a region. 'steal' takes them from whoever holds
        them; 'yield' leaves held cells where they are. (The third mode, in
        which both keep them, is the whole of what overlap needs here.)"""
        self._checkpoint()
        others = [r for rid, r in self.regions.items() if rid != region_id and r.cells.size]
        if mode == 'yield':
            for other in others:
                cells = np.setdiff1d(cells, other.cells, assume_unique=True)
        else:
            for other in others:
                taken = np.isin(other.cells, cells, assume_unique=True)
                if taken.any():     # untouched regions keep their array: the
                    other.keep(~taken)      # renderer rebuilds by identity
        region = self.region(region_id, create=True)
        if replace:
            region.keep(np.isin(region.cells, cells, assume_unique=True))
        region.merge(cells)
        self.version += 1

    def remove(self, region_id, cells):
        region = self.region(region_id)
        if region is None or region.cells.size == 0:
            return
        self._checkpoint()
        region.keep(~np.isin(region.cells, cells, assume_unique=True))
        self.version += 1

    def reshape(self, region_id, op, mode='steal'):
        """Replace a region's cells by ``op(grid, cells)``. Cells the shape
        grows into that another region holds are settled by ``mode`` as in
        assign. Returns False if the op had nothing to work with."""
        region = self.region(region_id)
        if region is None or region.cells.size == 0:
            return False
        result = op(self.grid, region.cells)
        if result is None:
            return False
        self.assign(region_id, result, mode=mode, replace=True)
        return True

    def absorb(self, target_id, source_id):
        """Move every cell of one region into another, leaving the source
        empty (its slot and name stay). Membership weights on the source are
        not carried: what is absorbed is wholly in."""
        source = self.region(source_id)
        if source is None or source.cells.size == 0 or source_id == target_id:
            return False
        self._checkpoint()
        self.region(target_id, create=True).merge(source.cells)
        source.keep(np.zeros(source.cells.size, dtype=bool))
        self.version += 1
        return True

    def clear(self, region_id):
        region = self.region(region_id)
        if region is None or region.cells.size == 0:
            return
        self._checkpoint()
        region.keep(np.zeros(region.cells.size, dtype=bool))
        self.version += 1

    def rename(self, region_id, name):
        self.region(region_id, create=True).name = name
        self.version += 1

    def compile(self):
        c = _Compiled()
        c.grid = self.grid
        c.version = self.version
        c.count = (max(self.regions) + 1) if self.regions else 0
        c.names = [self.regions[i].name if i in self.regions else '' for i in range(c.count)]
        c.colors = np.zeros((c.count, 3), dtype=np.float32)
        for rid, r in self.regions.items():
            c.colors[rid] = r.color
        layers = []     # [labels (ncells,) int16, weights (ncells,) float32 or None]
        for rid in sorted(self.regions):
            r = self.regions[rid]
            if r.cells.size == 0:
                continue
            # First layer in which none of this region's cells is taken.
            # Disjoint regions — all that v1 authors — share layer 0.
            for layer in layers:
                if not (layer[0][r.cells] >= 0).any():
                    break
            else:
                layer = [np.full(self.grid.ncells, -1, dtype=np.int16), None]
                layers.append(layer)
            layer[0][r.cells] = rid
            if r.weights is not None:
                if layer[1] is None:
                    layer[1] = np.ones(self.grid.ncells, dtype=np.float32)
                layer[1][r.cells] = r.weights
        c.layers = tuple((labels, weights) for labels, weights in layers)
        return c

    @staticmethod
    def evaluate(compiled, pts, point_weights):
        """(labels (N,) int32, values (K,) float32) for a cloud. A point's
        contribution is its own weight (1 without any) times its membership
        weight (1 without any), summed per region."""
        n = pts.shape[0]
        labels = np.full(n, -1, dtype=np.int32)
        values = np.zeros(compiled.count, dtype=np.float32)
        if not compiled.layers or n == 0:
            return labels, values
        lin, valid = compiled.grid.index(pts)
        where = np.flatnonzero(valid)
        lin = lin[where]
        for layer_labels, layer_weights in compiled.layers:
            found = layer_labels[lin]
            hit = found >= 0
            if not hit.any():
                continue
            found = found[hit]
            w = None if point_weights is None else point_weights[where[hit]]
            if layer_weights is not None:
                member = layer_weights[lin[hit]]
                w = member if w is None else w * member
            values += np.bincount(found, weights=w, minlength=compiled.count)[:compiled.count]
            idx = where[hit]
            unset = labels[idx] < 0
            labels[idx[unset]] = found[unset]
        return labels, values

    def save(self, path):
        ids = sorted(self.regions)
        data = {
            'format_version': np.int32(REGION_FILE_VERSION),
            'grid_lo': self.grid.lo,
            'grid_voxel_size': self.grid.voxel_size,
            'grid_dims': self.grid.dims,
            'region_ids': np.asarray(ids, dtype=np.int32),
            'region_names': np.asarray([self.regions[i].name for i in ids], dtype=str),
            'region_colors': np.asarray([self.regions[i].color for i in ids],
                                        dtype=np.float32).reshape(-1, 3),
        }
        for i in ids:
            data[f'cells_{i}'] = self.regions[i].cells
            if self.regions[i].weights is not None:
                data[f'weights_{i}'] = self.regions[i].weights
        np.savez_compressed(path, **data)

    def load(self, path):
        with np.load(path, allow_pickle=False) as f:
            version = int(f['format_version'])
            if version > REGION_FILE_VERSION:
                raise ValueError(f'region file is format {version}, newer than this '
                                 f'node understands ({REGION_FILE_VERSION})')
            lo = f['grid_lo'].astype(np.float32)
            size = f['grid_voxel_size'].astype(np.float32)
            dims = f['grid_dims'].astype(np.int64)
            grid = _VoxelGrid()
            grid.configure(lo, lo + size * dims, size, dims=dims)
            regions = {}
            for i, name, color in zip(f['region_ids'], f['region_names'], f['region_colors']):
                i = int(i)
                weights = f[f'weights_{i}'].astype(np.float32) if f'weights_{i}' in f else None
                regions[i] = _Region(i, str(name), color, f[f'cells_{i}'].astype(np.int64), weights)
        self.grid = grid
        self.regions = regions
        self._undo = []
        self.version += 1


# ---------------------------------------------------------------------------

class PointCloudRegionsNode(PointCloudNode):
    """Author a map of named regions and sum the cloud into them.

    Every frame on ``point cloud`` goes out on ``cloud`` carrying a 'clusters'
    entry — one value per region id: the summed weights of the points inside
    it (their count, for a cloud with no weights) — and the values alone go
    out on ``region values``. Put it after pc_voxel.

    Authoring reads down the node in the order the work goes, and every
    action acts on the current region straight away. Every button is also a
    message of the same name, accepted on any input (``command`` is there for
    the purpose), so it can be driven from wherever the author actually is,
    which is rarely the keyboard.

    map
      pick region        make current the region that holds most of the cloud
                         'capture' reads — stand in it. (mgl_regions'
                         'picked region' does the same from a mouse click.)

    region — the one you are working on
      region, name       its number and name. An unused number is a new one.
      clear region

    capture into region — the cells under the cloud on ``capture cloud in``,
    or under the live cloud when nothing is connected there: 'where I am
    standing now'
      add / remove / replace
                         into / out of / as the whole of the region. 'assign
                         mode' settles a cell another region already holds:
                         steal it, or yield to it.

    edit region — reshape the region itself
      grow / shrink      by ``steps`` cells.
      fill               close gaps up to ``steps`` wide, then fill whatever
                         that encloses: mends a speckled capture.
      hull / box         convex hull / bounding box. A captured body is a
                         shell open to the far side; hull makes a solid of it.
      snap               whole pc_voxel boxes at least 'snap coverage' full.
      undo

    By message only, to keep the node itself short — put them on a
    button_set wired to ``command`` when they are wanted as buttons:
      left / right / down / up / back / forward
                         shift the region by ``steps`` cells along -x / +x,
                         -y / +y, -z / +z (the grid's axes: with a levelled
                         camera y is up, z is away from it).
      move x y z         shift by whole cells.
      extend x|y|z [+|-] flood the region along an axis to the grid's edge —
                         both ways, or one way from what it holds: a footprint
                         becomes a floor-to-ceiling pillar with 'extend y'.

    absorb (options, or the message 'absorb <n>') moves every cell of region
    n into the current one. It is how a piece is sculpted on its own before
    it joins a region that already has cells — hull run on the region after a
    second capture would bridge the two — capture it into a spare number,
    shape it there, absorb it.

    To see all of it — regions lit by their values, the one being edited —
    wire ``cloud`` to mgl_regions. ``region cells`` also goes out as a plain
    cell-centre cloud on every change, for mgl_point_cloud or anything else
    that takes a cloud. The map is kept in an .npz beside the patch: 'save' /
    'load', and 'path' reloads it with the patch."""

    @staticmethod
    def factory(name, data, args=None):
        return PointCloudRegionsNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.map = _RegionMap()
        self.compiled = None
        self._capture_frame = None
        self._live_pts = None
        self._lattice = None          # (origin, cell, shape) of upstream boxes
        self._geometry = None         # RegionGeometry, for mgl_regions
        self._geometry_version = 0
        self._edit_lock = threading.RLock()
        self._warned_large = False
        self._warned_outside = False

        self.input = self.add_input('point cloud', triggers_execution=True)
        self.capture_input = self.add_input('capture cloud in', callback=self._capture_arrived)

        # The body reads top to bottom in the order the work goes: which
        # region, capture into it, reshape it. The actions are rows of plain
        # buttons rather than an input apiece — a pin per button made a node
        # too tall to take in, with labels too long for the buttons — and
        # every one is still a message (see _ACTIONS), which any input
        # accepts; ``command`` is there to make that a visible place to wire
        # a pedal or OSC.
        #
        # 'map' is about all the regions, so above the region section, not in
        # it: everything in that section is about the current region, while
        # the count is of all of them and 'pick region' changes which is
        # current.
        self.add_spacer()
        self.add_label('- map -')
        self.map_label = self.add_label('no regions yet')
        self._button_row(('pick region',))

        self.add_spacer()
        self.add_label('- region -')
        self.region_input = self.add_input('region', widget_type='drag_int', default_value=0,
                                           min=0, max=MAX_REGION_ID, callback=self._region_changed)
        self.name_input = self.add_input('name', widget_type='text_input', default_value='',
                                         callback=self._name_changed)
        self.region_label = self.add_label('')
        self._button_row(('clear region',))

        self.add_spacer()
        self.add_label('- capture into region -')
        self._button_row(('add', 'remove', 'replace'))

        self.add_spacer()
        self.add_label('- edit region -')
        self.steps_input = self.add_input('steps', widget_type='drag_int', default_value=1, min=0)
        self._button_row(('grow', 'shrink', 'fill'))
        self._button_row(('hull', 'box', 'snap'))
        self._button_row(('undo',))
        self.command_input = self.add_input('command')

        self.output = self.add_output('cloud')
        self.values_output = self.add_output('region values')
        self.cells_output = self.add_output('region cells')
        self.info_output = self.add_output('info')

        self.mode_option = self.add_option('assign mode', widget_type='combo', default_value='steal')
        self.mode_option.widget.combo_items = ['steal', 'yield']
        self.coverage_option = self.add_option('snap coverage', widget_type='drag_float',
                                               default_value=0.25, min=0.0, max=1.0)
        self.coverage_option.widget.speed = 0.01
        self.absorb_from_option = self.add_option('absorb from', widget_type='drag_int',
                                                  default_value=0, min=0, max=MAX_REGION_ID)
        self.absorb_option = self.add_option('absorb', widget_type='button', callback=self.absorb)
        self.show_option = self.add_option('show', widget_type='combo', default_value='current region',
                                           callback=self._show)
        self.show_option.widget.combo_items = ['current region', 'all regions']
        self.surface_option = self.add_option('surface only', widget_type='checkbox',
                                              default_value=True, callback=self._show)
        self.save_button = self.add_option('save', widget_type='button', callback=self.save_map)
        self.load_button = self.add_option('load', widget_type='button', callback=self.load_map)
        self.path_option = self.add_option('path', widget_type='text_input', default_value='',
                                           callback=self._load_from_path)
        self.voxel_option = self.add_option('voxel size (cm)', widget_type='drag_float',
                                            default_value=5.0, min=0.01)
        self._add_bounds_options([-3.0, -3.0, 0.0], [3.0, 3.0, 6.0])
        self.message_handlers['move'] = self._move_message
        self.message_handlers['extend'] = self._extend_message
        self.message_handlers['absorb'] = self._absorb_message
        for name in self._DIRECTIONS:
            self.message_handlers[name] = self._direction_message
        for name in self._ACTIONS:
            self.message_handlers[name] = self._action_message
        # The names these went by before, for anything already wired to them.
        self._old_names = {'pick': 'pick region', 'snap to boxes': 'snap'}
        for old in self._old_names:
            self.message_handlers[old] = self._action_message

    # name -> (method, what the tooltip says). The names are what the buttons
    # show, so they are short; the tooltip carries the rest.
    _ACTIONS = {
        'pick region': ('pick_region', 'make current the region that holds most of the cloud on '
                                       "'capture cloud in' (or of the live cloud) - stand in it"),
        'clear region': ('clear_region', 'empty the current region'),
        'add': ('add', "add the cells under the cloud on 'capture cloud in' (or the live "
                       'cloud) to the region'),
        'remove': ('remove', 'take the cells under that cloud out of the region'),
        'replace': ('replace', 'make the cells under that cloud the whole of the region'),
        'grow': ('grow', "grow the region by 'steps' cells"),
        'shrink': ('shrink', "shrink the region by 'steps' cells"),
        'fill': ('fill', "close gaps up to 'steps' cells wide, then fill what that encloses"),
        'hull': ('hull', 'convex hull: makes a solid of a captured body, which is only a shell'),
        'box': ('box', 'bounding box of the region'),
        'snap': ('snap', "whole pc_voxel boxes at least 'snap coverage' full of the region"),
        'undo': ('undo', 'undo the last change to the regions'),
    }
    BUTTON_WIDTH = 66
    # Message -> (axis, sign) for the directional moves.
    _DIRECTIONS = {'left': (0, -1), 'right': (0, 1), 'down': (1, -1), 'up': (1, 1),
                   'back': (2, -1), 'forward': (2, 1)}

    def _button_row(self, names):
        row = self.add_display('')
        row.submit_callback = lambda names=names: self._draw_buttons(names)
        return row

    def _draw_buttons(self, names):
        import dearpygui.dearpygui as dpg
        if not hasattr(self, '_buttons'):
            self._buttons = {}
        with dpg.group(horizontal=True):
            for name in names:
                # Wide enough for the label, always: a button whose name is
                # cut off does not say what it does.
                width = max(self.BUTTON_WIDTH, 7 * len(name) + 16)
                button = dpg.add_button(label=name, width=width,
                                        callback=lambda s, a, u: self._run_action(u), user_data=name)
                with dpg.tooltip(parent=button):
                    dpg.add_text(self._ACTIONS[name][1], wrap=260)
                self._buttons[name] = button

    def _run_action(self, name):
        name = self._old_names.get(name, name)
        try:
            getattr(self, self._ACTIONS[name][0])()
        except Exception as e:
            print(f'{self.label}: {name} failed: {type(e).__name__}: {e}')

    def _action_message(self, message='', args=None):
        self._run_action(message)

    def post_load_callback(self):
        if self.map.is_empty():
            self._load_from_path()

    # -- grid ---------------------------------------------------------------

    def _ensure_grid(self):
        """While the map is empty its grid follows the frame; the first cell
        assigned freezes it. Returns False if there is no usable grid."""
        if self.map.grid is not None and not self.map.is_empty():
            return True
        lo, hi = self._bounds(*self._bounds_defaults)
        size = self._carried_voxel_size()
        if size is None:
            size = float(self.voxel_option()) * 0.01   # cm -> m
        with self._edit_lock:
            try:
                changed = self.map.set_grid(lo, hi, size)
                self._warned_large = False
            except ValueError as e:
                if not self._warned_large:
                    print(f'{self.label}: {e}')
                    self._warned_large = True
                return self.map.grid is not None
            if changed:
                self.compiled = self.map.compile()
                self._publish_geometry()
                self._reflect_grid()
        return True

    def _reflect_grid(self):
        """Show the volume actually in use in the bounds / voxel size options.
        Elsewhere those are only fallbacks; here the grid belongs to the map
        and stays put once regions exist, so what it is matters to see."""
        grid = self.map.grid
        if grid is None:
            return
        try:
            # Rounded for the eye only: float32 bounds read as 1.0000000298.
            hi = grid.lo + grid.dims * grid.voxel_size
            # Through the widget: the property's own set() keeps only the first
            # element of a list for a numeric widget, which would write one
            # number across all three columns.
            self.min_option.widget.set(np.round(grid.lo.astype(np.float64), 4).tolist(), False)
            self.max_option.widget.set(np.round(hi.astype(np.float64), 4).tolist(), False)
            self.voxel_option.set(round(float(grid.voxel_size[0]) * 100.0, 3), propagate=False)
        except Exception as e:
            print(f'{self.label}: could not show the volume in the options: '
                  f'{type(e).__name__}: {e}')

    def _check_frozen_grid_covers_frame(self):
        crop = self.in_meta.get(CROP_KEY)
        if crop is None or self._warned_outside:
            return
        grid = self.map.grid
        lo = np.asarray(crop[0], dtype=np.float32).reshape(-1)[:3]
        hi = np.asarray(crop[1], dtype=np.float32).reshape(-1)[:3]
        slack = grid.voxel_size
        if np.any(lo < grid.lo - slack) or np.any(hi > grid.lo + grid.voxel_size * grid.dims + slack):
            print(f'{self.label}: the crop now reaches outside the volume the regions were '
                  f'drawn in; points out there belong to no region')
            self._warned_outside = True

    # -- per frame ----------------------------------------------------------

    def execute(self):
        pts = self._get_cloud()
        if pts is None:
            # An empty frame still has to zero the values downstream.
            raw = self.input()
            if isinstance(raw, dict) and self.compiled is not None:
                pts = np.empty((0, 3), dtype=np.float32)
                _, self.in_meta = unwrap_cloud(raw)
            else:
                return
        self._live_pts = pts
        upstream = self.in_meta.get(CLUSTER_KEY)
        if isinstance(upstream, dict) and upstream.get('shape') is not None:
            self._lattice = (upstream['origin'], upstream['cell'], upstream['shape'])
        if not self._ensure_grid():
            self._send(self.output, pts)
            return
        compiled = self.compiled
        if compiled is None:
            with self._edit_lock:
                compiled = self.compiled = self.map.compile()
        if compiled.layers:
            self._check_frozen_grid_covers_frame()

        weights = self.in_meta.get('weights')
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float32).reshape(-1)
            if weights.shape[0] != pts.shape[0]:
                weights = None
        labels, values = _RegionMap.evaluate(compiled, pts, weights)
        self.values_output.send(values)
        self._send(self.output, pts, **{CLUSTER_KEY: {
            'labels': labels,
            'values': values,
            'shape': None,
            'origin': compiled.grid.lo.copy(),
            'cell': compiled.grid.voxel_size.copy(),
            'names': compiled.names,
            'colors': compiled.colors,
            'geometry': self._geometry,
        }})

    # -- capture --------------------------------------------------------------

    def _capture_arrived(self):
        self._capture_frame = self.capture_input()   # converted on use

    def _source_points(self):
        """The cloud that capture and 'pick region' read: whatever is on
        ``capture cloud in``, or the live cloud when nothing is connected
        there. (N, 3) float32, or None when there is nothing."""
        if self.capture_input.get_parents() and self._capture_frame is not None:
            pts, _ = unwrap_cloud(self._capture_frame)
            pts = any_to_array(pts) if pts is not None else None
        else:
            pts = self._live_pts
        if pts is None or not isinstance(pts, np.ndarray) or pts.size == 0:
            return None
        return np.asarray(pts, dtype=np.float32).reshape(-1, 3)

    def pick_region(self):
        """Make current the region that holds most of the source cloud: stand
        in a region (or point a wand into it, once there is one) and press."""
        pts = self._source_points()
        compiled = self.compiled
        if pts is None or compiled is None or not compiled.layers:
            print(f'{self.label}: nothing to pick with, or no regions to pick from')
            return
        _, values = _RegionMap.evaluate(compiled, pts, None)
        if values.max() <= 0.0:
            print(f'{self.label}: the cloud is in no region')
            return
        self.region_input.set(int(values.argmax()), propagate=False)
        self._region_changed()

    def _captured_cells(self):
        if self.map.grid is None:
            print(f'{self.label}: no cloud has arrived yet, so there is no volume to capture in')
            return None
        pts = self._source_points()
        if pts is None:
            print(f'{self.label}: nothing to capture - the cloud is empty')
            return None
        return self.map.cells_under(pts)

    def add(self):
        cells = self._captured_cells()
        if cells is None:
            return
        with self._edit_lock:
            self.map.assign(self._region_id(), cells, mode=self.mode_option())
            self._edited()

    def replace(self):
        cells = self._captured_cells()
        if cells is None:
            return
        with self._edit_lock:
            self.map.assign(self._region_id(), cells, mode=self.mode_option(), replace=True)
            self._edited()

    def remove(self):
        cells = self._captured_cells()
        if cells is None:
            return
        with self._edit_lock:
            self.map.remove(self._region_id(), cells)
            self._edited()

    # -- edit -------------------------------------------------------------------

    def _reshape(self, op):
        with self._edit_lock:
            if self.map.grid is None:
                return
            region = self.map.region(self._region_id())
            if region is None or region.cells.size == 0:
                print(f'{self.label}: region {self._region_id()} is empty - nothing to reshape')
                return
            if not self.map.reshape(self._region_id(), op, mode=self.mode_option()):
                print(f'{self.label}: the region has no volume to work with')
                return
            self._edited()

    def _steps(self):
        return max(0, int(self.steps_input()))

    def grow(self):
        self._reshape(lambda g, c: grow_cells(g, c, self._steps()))

    def shrink(self):
        self._reshape(lambda g, c: shrink_cells(g, c, self._steps()))

    def fill(self):
        self._reshape(lambda g, c: fill_cells(g, c, self._steps()))

    def hull(self):
        self._reshape(hull_cells)

    def box(self):
        self._reshape(box_cells)

    def snap(self):
        if self._lattice is None:
            print(f'{self.label}: no box lattice on the incoming frames — set pc_voxel\'s boxes')
            return
        origin, cell, shape = self._lattice
        coverage = float(self.coverage_option())
        self._reshape(lambda g, c: snap_cells_to_lattice(g, c, origin, cell, shape, coverage))

    def _move_message(self, message='', args=None):
        try:
            offset = [int(round(float(v))) for v in (args or [])][:3]
        except (TypeError, ValueError):
            offset = []
        if len(offset) != 3:
            print(f'{self.label}: move takes x y z, in whole cells')
            return
        self._reshape(lambda g, c: move_cells(g, c, offset))

    def _direction_message(self, message='', args=None):
        axis, sign = self._DIRECTIONS[message]
        offset = [0, 0, 0]
        offset[axis] = sign * max(1, self._steps())
        self._reshape(lambda g, c: move_cells(g, c, offset))

    def _extend_message(self, message='', args=None):
        args = [str(a).strip().lower() for a in (args or [])]
        axis = {'x': 0, 'y': 1, 'z': 2}.get(args[0] if args else '')
        direction = args[1] if len(args) > 1 else 'both'
        if axis is None or direction not in ('+', '-', 'both'):
            print(f'{self.label}: extend takes an axis (x, y or z) and optionally + or -')
            return
        self._reshape(lambda g, c: extend_cells(g, c, axis, direction))

    def absorb(self, source_id=None):
        if source_id is None:
            source_id = int(self.absorb_from_option())
        with self._edit_lock:
            if not self.map.absorb(self._region_id(), int(source_id)):
                print(f'{self.label}: region {source_id} has nothing to absorb')
                return
            self._edited()

    def _absorb_message(self, message='', args=None):
        try:
            self.absorb(int(float(args[0])))
        except (TypeError, ValueError, IndexError):
            print(f'{self.label}: absorb takes a region number')

    # -- regions ------------------------------------------------------------

    def _region_id(self):
        return int(np.clip(int(self.region_input()), 0, MAX_REGION_ID))

    def _edited(self):
        """Publish an edit: recompile for the frame path, refresh the views."""
        self.compiled = self.map.compile()
        self._warned_outside = False
        self._show()

    def clear_region(self):
        with self._edit_lock:
            self.map.clear(self._region_id())
            self._edited()

    def undo(self):
        with self._edit_lock:
            if not self.map.undo():
                return
            self._edited()
        self._region_changed()

    def _region_changed(self):
        region = self.map.region(self._region_id())
        self.name_input.set(region.name if region is not None else '', propagate=False)
        self._show()

    def _name_changed(self):
        name = str(self.name_input()).strip()
        if not name:
            return
        with self._edit_lock:
            self.map.rename(self._region_id(), name)
            self.compiled = self.map.compile() if self.map.grid is not None else None
        self._info()

    # -- views --------------------------------------------------------------

    def _publish_geometry(self):
        """Hand the renderer the current state, by reference. It rides on the
        next frame rather than going out on a wire of its own: there is then
        one connection to make, and a renderer connected later, or a patch
        just loaded, is up to date from its first frame."""
        with self._edit_lock:
            self._geometry_version += 1
            self._geometry = RegionGeometry(
                self._geometry_version, self.map.grid,
                {rid: r.cells for rid, r in self.map.regions.items() if r.cells.size},
                self._region_id())

    def _cell_cloud(self, cells, weights=None):
        grid = self.map.grid
        return {
            CLOUD_KEY: grid.centres(cells) if cells.size else np.empty((0, 3), dtype=np.float32),
            VOXEL_SIZE_KEY: grid.voxel_size_meta(),
            'weights': (np.ones(cells.size, dtype=np.float32) if weights is None else weights),
        }

    def _show(self):
        if self.map.grid is None:
            return
        with self._edit_lock:
            grid = self.map.grid
            if self.show_option() == 'all regions':
                shown = [r for r in self.map.regions.values() if r.cells.size]
            else:
                region = self.map.region(self._region_id())
                shown = [region] if region is not None and region.cells.size else []
            surface = bool(self.surface_option())
            parts = [surface_cells(grid, r.cells) if surface else r.cells for r in shown]
            cells = np.concatenate(parts) if parts else np.empty((0,), dtype=np.int64)
            self._publish_geometry()
        self.cells_output.send(self._cell_cloud(cells))
        self._info()

    def _info(self):
        rid = self._region_id()
        region = self.map.region(rid)
        held = region.cells.size if region is not None else 0
        name = region.name if region is not None else 'unused'
        live = sum(1 for r in self.map.regions.values() if r.cells.size)
        try:
            self.map_label.set('no regions yet' if not live else
                               f"{live} region{'s' if live != 1 else ''}")
            self.region_label.set(f'{held:,} cells' if held else 'empty')
            # A first 'add' into an unused number creates the region with a
            # default name; show it, without touching a name being typed.
            if region is not None and not str(self.name_input()).strip():
                self.name_input.set(region.name, propagate=False)
        except Exception:
            pass    # not drawn yet
        self.info_output.send(f"region {rid} '{name}': {held:,} cells | {live} regions")

    # -- file ---------------------------------------------------------------

    def save_map(self):
        if self.map.grid is None:
            print(f'{self.label}: nothing to save yet')
            return
        SaveDialog(self, callback=self._save_to, extensions=['.npz'],
                   default_filename='regions.npz')

    def _save_to(self, path):
        if not path:
            return
        if not path.endswith('.npz'):
            path += '.npz'
        try:
            with self._edit_lock:
                self.map.save(path)
        except Exception as e:
            print(f'{self.label}: could not save {path}: {type(e).__name__}: {e}')
            return
        self.path_option.set(path, propagate=False)
        print(f'{self.label}: saved {path}')

    def load_map(self):
        LoadDialog(self, callback=self._load_from, extensions=['.npz'])

    def _load_from(self, path):
        if path:
            self.path_option.set(path, propagate=False)
            self._load_from_path()

    def _load_from_path(self):
        path = str(self.path_option()).strip()
        if not path:
            return
        try:
            with self._edit_lock:
                self.map.load(path)
                self._edited()
        except Exception as e:
            print(f'{self.label}: could not load {path}: {type(e).__name__}: {e}')
            return
        self._region_changed()
        self._reflect_grid()
        print(f'{self.label}: loaded {path}')


class PointCloudAccumulateNode(PointCloudNode):
    """Record where the cloud has been. While ``record`` is on, every voxel
    the cloud touches counts a hit per frame; the output is the cloud of
    voxels with at least ``min hits``, weighted by dwell (hits over the most
    any voxel got). It is the C++ app's ACCUMULATE VOXELS, as a cloud — so it
    goes straight into pc_regions' ``capture cloud in``: walk the space, stop,
    capture. Raise ``min hits`` to drop what was only passed through.

    'delay (s)' and 'duration (s)' are for working alone: tick ``record``,
    walk into the volume during the delay, and it stops itself. Duration 0
    records until unticked. The input passes through on ``cloud out``."""

    @staticmethod
    def factory(name, data, args=None):
        return PointCloudAccumulateNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.grid = _VoxelGrid()
        self.hits = None
        self.frames = 0
        self._armed_at = None
        self._recording = False
        self._warned_large = False

        self.input = self.add_input('point cloud', triggers_execution=True)
        self.record_input = self.add_input('record', widget_type='checkbox', default_value=False,
                                           callback=self._record_changed)
        self.clear_input = self.add_input('clear', widget_type='button', callback=self.clear)
        self.min_hits_input = self.add_input('min hits', widget_type='drag_int', default_value=1,
                                             min=1, callback=self._emit)
        self.output = self.add_output('accumulated')
        self.frames_output = self.add_output('frames')
        self.passthrough_output = self.add_output('cloud out')
        self.delay_option = self.add_option('delay (s)', widget_type='drag_float',
                                            default_value=0.0, min=0.0)
        self.duration_option = self.add_option('duration (s)', widget_type='drag_float',
                                               default_value=0.0, min=0.0)
        self.voxel_option = self.add_option('voxel size (cm)', widget_type='drag_float',
                                            default_value=5.0, min=0.01)
        self._add_bounds_options([-3.0, -3.0, 0.0], [3.0, 3.0, 6.0])

    def _ensure_grid(self):
        lo, hi = self._bounds(*self._bounds_defaults)
        size = self._carried_voxel_size()
        if size is None:
            size = float(self.voxel_option()) * 0.01   # cm -> m
        try:
            changed = self.grid.configure(lo, hi, size)
            self._warned_large = False
        except ValueError as e:
            if not self._warned_large:
                print(f'{self.label}: {e}')
                self._warned_large = True
            return False
        if changed and self.hits is not None:
            print(f'{self.label}: volume changed — the accumulation no longer maps to it, cleared')
            self.hits = None
            self.frames = 0
        return True

    def _record_changed(self):
        if self.record_input():
            self._armed_at = time.monotonic()
        else:
            self._armed_at = None
            self._recording = False

    def clear(self):
        self.hits = None
        self.frames = 0
        self._emit()

    def _emit(self):
        if self.hits is None:
            cells = np.empty((0,), dtype=np.int64)
            weights = np.empty((0,), dtype=np.float32)
        else:
            cells = np.flatnonzero(self.hits >= max(1, int(self.min_hits_input())))
            weights = self.hits[cells].astype(np.float32)
            if cells.size:
                weights /= float(self.hits.max())
        pts = self.grid.centres(cells) if cells.size else np.empty((0, 3), dtype=np.float32)
        self.frames_output.send(self.frames)
        self._send(self.output, pts, voxel_size=self.grid.voxel_size_meta(), weights=weights)

    def execute(self):
        raw = self.input()
        pts = self._get_cloud()
        self.passthrough_output.send(raw)
        if self._armed_at is None:
            return
        elapsed = time.monotonic() - self._armed_at - float(self.delay_option())
        if elapsed < 0.0:
            return
        duration = float(self.duration_option())
        if duration > 0.0 and elapsed >= duration:
            self.record_input.set(False)
            self._record_changed()
            self._emit()
            print(f'{self.label}: recorded {self.frames} frames')
            return
        if not self._recording:
            self._recording = True
            print(f'{self.label}: recording')
        if not self._ensure_grid():
            return
        if self.hits is None:
            self.hits = np.zeros(self.grid.ncells, dtype=np.int32)
        if pts is not None:
            lin, valid = self.grid.index(pts)
            self.hits[np.unique(lin[valid])] += 1
        self.frames += 1
        self._emit()
