"""Mode tables worked out from a shape, instead of looked up.

The tables in `MODAL_MATERIALS` say what a bell or a bar sounds like by
listing what its modes are. This says what a shape you have described
sounds like by SOLVING for them: give it an outline, a way of sweeping
that outline into a volume, and what the thing is made of, and it
returns the same three columns modal~ already eats -- frequency ratio,
how much a strike wakes each mode, and a decay scale.

It is worth being clear about what makes this cheap enough to bother
with. Two things:

The sweeps that matter are STRUCTURED. Spin an outline about an axis,
or push it along a line, and the mesh you get is a regular grid --
station by station, round by round. So none of the hard part of turning
a shape into elements arises: no general-purpose mesher, no new
dependency, just arithmetic on a grid. That is why this is a few
hundred lines and not a package.

And the absolute frequency does not matter. modal~'s table is RATIOS,
with `frequency` setting the pitch, so any error in the overall scale
costs nothing at all -- the elastic constants only have to be right in
their proportions. What has to be right is the SHAPE.

Checked against the free-free bar, whose ratios are known to everyone:
refining the mesh gives 1.000 / 2.711 / 5.195 / 8.349 against the book's
1.000 / 2.756 / 5.404 / 8.933. The gap at the top is not error -- the
book assumes a bar of no thickness, and a real one rings its upper modes
flat of that, which is shear and rotary inertia doing what they do.

WHAT THIS DOES NOT DO. It solves for frequencies and mode shapes, and
those follow from geometry and elasticity. It does not solve for DECAY,
which follows from material damping, from how the thing is held, and
from what it radiates -- all much harder to predict than a frequency,
and none of it in here. The decay column is an imposed law with a knob
on it, honestly labelled, not a computed answer. Neither does it know
what radiates: a mode shape gives mechanical amplitude, and small high
modes push very little air. modal~'s `brightness` and `tilt` are the
place to answer that by ear.

SOLIDS ONLY, for now. Thin shells -- a bell, a bowl, a can, a glass --
are the interesting case and the hard one: eight-node bricks lock in
bending unless several of them sit across the wall, and a wall a
millimetre thick in a vessel two hundred across makes that ruinous.
Doing shells properly wants a shell element, which is its own job.
"""

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


# Young's modulus in pascals, Poisson's ratio, density in kg/m3, and the
# LOSS FACTOR. The first three are real numbers rather than invented
# ones, so the frequencies that come out are believable if you ever want
# to read them -- though only their proportions reach the table, since a
# ratio does not care.
#
# The loss factor is the one that decides how long a thing RINGS, and it
# is the only place a material's own character survives into the sound.
# An amplitude falls as exp(-pi * eta * f * t), so sixty decibels takes
# 2.2 / (eta * f) seconds: a steel bar at 440 Hz rings for fifty seconds
# and a rubber one for a fortieth of one. Spread over three orders of
# magnitude, which is far more than anything else here varies.
MATERIALS = {
    'aluminium': (7.0e10, 0.33, 2700.0, 1.0e-4),
    'steel': (2.0e11, 0.30, 7850.0, 1.0e-4),
    'brass': (1.0e11, 0.34, 8500.0, 1.0e-3),
    'bronze': (1.05e11, 0.34, 8800.0, 1.0e-3),
    'copper': (1.2e11, 0.34, 8960.0, 2.0e-3),
    'glass': (7.0e10, 0.22, 2500.0, 6.0e-4),
    'stone': (5.0e10, 0.25, 2700.0, 3.0e-3),
    'wood': (1.1e10, 0.35, 600.0, 8.0e-3),
    'bone': (1.8e10, 0.30, 1900.0, 1.0e-2),
    'nylon': (3.0e9, 0.40, 1150.0, 3.0e-2),
    'rubber': (5.0e7, 0.48, 1100.0, 2.0e-1),
}


def ring_time(material, fundamental):
    """How long the first mode takes to fall sixty decibels, in seconds.

    From the material's loss factor and the pitch it actually came out
    at, which is the only honest way round: the same steel rings for
    fifty seconds as a long bar and a fraction of one as a small hard
    thing, because the loss is per CYCLE and a high mode spends its
    cycles faster.
    """
    eta = (MATERIALS[material][3] if isinstance(material, str)
           else material[3])
    if eta <= 0.0 or fundamental <= 0.0:
        return 60.0
    return min(60.0, max(0.01, 2.2 / (eta * fundamental)))

# How an outline becomes a volume. Mirroring is NOT one of these: it is
# something done to the outline before any of them, so it belongs beside
# them rather than among them -- as a third sweep it could not be had at
# the same time as a revolve, and a symmetric vase drawn as one half is
# an obvious thing to want.
SWEEPS = ('revolve', 'extrude')

# Eight-node brick, corners in the natural cube, and the two-by-two-by-
# two Gauss points that integrate it.
_CORNER = np.array([(x, y, z) for z in (-1.0, 1.0) for y in (-1.0, 1.0)
                    for x in (-1.0, 1.0)])
_G = 1.0 / math.sqrt(3.0)
_GAUSS = np.array([(a, b, c) for a in (-_G, _G) for b in (-_G, _G)
                   for c in (-_G, _G)])


def _quad_grid(nx, ny):
    """Nodes and quads of a regular grid, as flat index arrays."""
    idx = lambda i, j: j * (nx + 1) + i
    quads = [[idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1)]
             for j in range(ny) for i in range(nx)]
    return np.array(quads)


def rect_section(across=3, through=3):
    """A rectangular cross-section on the unit square, [-1, 1] both ways.

    Three by three and not fewer. Eight-node bricks LOCK in bending
    unless a few of them sit across the thickness -- they cannot curve,
    so they resist by shearing instead and come out far too stiff. Two
    through the thickness was enough to put a spurious mode within two
    percent of the fundamental.
    """
    xs = np.linspace(-1.0, 1.0, across + 1)
    ys = np.linspace(-1.0, 1.0, through + 1)
    pts = np.array([(x, y) for y in ys for x in xs])
    return pts, _quad_grid(across, through)


def disc_section(core=2, rings=2):
    """A circular cross-section on the unit disc, without a singular axis.

    A polar grid would put every angle on top of itself at the middle
    and hand the solver a pile of collapsed elements. So the middle is
    a square block and the rest is rings blending that square out to the
    circle -- the usual way round it, and structured throughout.
    """
    half = 0.5
    xs = np.linspace(-half, half, core + 1)
    pts = [(x, y) for y in xs for x in xs]
    quads = list(_quad_grid(core, core))
    # Walk the square's edge once, anticlockwise, without repeating a
    # corner: that ring of nodes is what the annulus grows from.
    edge = ([(i, 0) for i in range(core)]
            + [(core, j) for j in range(core)]
            + [(core - i, core) for i in range(core)]
            + [(0, core - j) for j in range(core)])
    ring_prev = [j * (core + 1) + i for i, j in edge]
    n_edge = len(edge)
    for k in range(1, rings + 1):
        t = k / rings
        base = len(pts)
        for i, j in edge:
            x, y = xs[i], xs[j]
            r = math.hypot(x, y)
            # Straight from the square's edge out to the circle, so the
            # outer ring lands exactly on radius one.
            sx, sy = (x / r, y / r) if r > 1e-12 else (0.0, 0.0)
            pts.append(((1.0 - t) * x + t * sx, (1.0 - t) * y + t * sy))
        ring = [base + i for i in range(n_edge)]
        for i in range(n_edge):
            n = (i + 1) % n_edge
            # Outward first, then round: walking the edge the other
            # way round pairs with outward left-handed, and every brick
            # in the annulus comes out inside out.
            quads.append([ring_prev[i], ring[i], ring_prev[n], ring[n]])
        ring_prev = ring
    return np.array(pts), np.array(quads)


def _loop_ring(loop, wall, through):
    """Quads filling between a closed loop and a shrunken copy of it.

    Which is all a hollow section is, whatever shape the outside is: a
    circle gives a tube, a square gives a box with a hole down it. The
    inner boundary is the outer one scaled about the middle, so the wall
    is an even thickness in proportion rather than an even distance --
    right for a turned or moulded thing, which is what these are.
    """
    loop = np.asarray(loop, dtype=float)
    inner = loop * max(0.0, 1.0 - wall)
    pts, rings = [], []
    for k in range(through + 1):
        t = k / through
        here = inner + (loop - inner) * t
        rings.append(list(range(len(pts), len(pts) + len(loop))))
        pts.extend(tuple(point) for point in here)
    quads = []
    for k in range(through):
        low, high = rings[k], rings[k + 1]
        for a in range(len(loop)):
            b = (a + 1) % len(loop)
            # Outward first, then round, as in disc_section: the other
            # way about is left-handed and every brick comes out inside
            # out.
            quads.append([low[a], high[a], low[b], high[b]])
    return np.array(pts), np.array(quads)


def tube_section(wall=0.4, around=16, through=2):
    """A round hollow section, on the unit disc."""
    loop = [(math.cos(2.0 * math.pi * a / around),
             math.sin(2.0 * math.pi * a / around)) for a in range(around)]
    return _loop_ring(loop, wall, through)


def box_tube_section(wall=0.4, per_side=4, through=2):
    """A square hollow section, on the unit square."""
    steps = np.linspace(-1.0, 1.0, per_side + 1)[:-1]
    loop = ([(x, -1.0) for x in steps] + [(1.0, y) for y in steps]
            + [(-x, 1.0) for x in steps] + [(-1.0, -y) for y in steps])
    return _loop_ring(loop, wall, through)


def sweep(profile, length, sweep_mode='revolve', depth=None,
          section=None, carve='depth', mirror=False, wall=1.0):
    """Sweep an outline into a solid, as a regular grid of bricks.

    `profile` is a list of half-widths, evenly spaced along the length.
    What that half-width MEANS is the sweep:

      'revolve'  it is a radius, spun about the long axis -- a club, an
                 egg, a cone, a turned bead.
      'extrude'  it is a half-width, and the section is that wide by
                 `depth` deep -- a bar with a shaped outline, which is
                 what an undercut marimba bar is.
    `wall` hollows it, as a fraction of the way in from the outside: 1
    is solid, 0.4 leaves a wall four tenths of the radius. Bricks hold up
    to about a fifth -- bending comes out within three percent of a
    bar's, and the ovalling modes a tube has and a bar does not turn up
    where they should. Below a fifth they start to go: a tenth already
    puts a spurious mode three percent above the fundamental. A bell or a
    wine glass is one or two percent and wants a shell element, which is
    its own job.

    `mirror` says the outline is only HALF of one, running from an end
    to the middle, and reflects it: n half-widths become 2n-1 stations
    and the last one given sits at the centre. It happens to the outline
    before the sweep, so it goes with either -- a bar undercut
    symmetrically, or a vase drawn as one half.

    `carve` says WHICH way an extrude's outline is taken, and it is not
    a detail: a marimba bar's arch is cut into its UNDERSIDE, not into
    its plan. Carving the depth moves the second mode 2.70 -> 3.84 for
    the same cut that moves it only to 3.23 across the width, because
    bending stiffness goes as the depth cubed and only as the width
    itself. 'depth' is the default for that reason.
    """
    prof = np.asarray(profile, dtype=float).ravel()
    if prof.size < 2:
        raise ValueError('a profile needs at least two half-widths')
    if np.any(prof <= 0.0):
        raise ValueError('half-widths must all be greater than zero')
    if mirror and prof.size >= 2:
        # The last half-width given is the MIDDLE of the finished thing,
        # so the reflection starts one back from it. Reflecting the whole
        # list instead put the centre value in twice and left a flat
        # station there that nobody asked for.
        prof = np.concatenate([prof, prof[-2::-1]])
    if section is None:
        if wall >= 0.999:
            section = (disc_section() if sweep_mode == 'revolve'
                       else rect_section())
        else:
            section = (tube_section(wall) if sweep_mode == 'revolve'
                       else box_tube_section(wall))
    pts2d, quads = section
    if depth is None:
        depth = 2.0 * float(prof.max())

    n_sec = len(pts2d)
    stations = np.linspace(0.0, length, prof.size)
    nodes = np.empty((prof.size * n_sec, 3))
    for s, (z, w) in enumerate(zip(stations, prof)):
        if sweep_mode == 'revolve':
            # Spun about the axis, so the outline is a radius and both
            # ways across are the same.
            block = pts2d * w
        else:
            # Pushed along a line, so the outline shapes the WIDTH and
            # the depth is the depth -- it does not follow the outline.
            # Scaling both by it made a waisted outline pinch in
            # thickness as well, which is a spindle and not a bar with a
            # shaped edge; and it left no way at all to describe the one
            # thing this is for, a bar undercut in one plane only.
            if carve == 'width':
                block = pts2d * np.array([w, 0.5 * depth])
            else:
                # The outline is the ARCH under it, and the width is the
                # width all the way along.
                block = pts2d * np.array([0.5 * depth, w])
        nodes[s * n_sec:(s + 1) * n_sec, 0] = z
        nodes[s * n_sec:(s + 1) * n_sec, 1:] = block

    # Each quad of the section, carried from one station to the next,
    # is one brick. Corner order has to match _CORNER: x fastest, then
    # y, then z, with x here being the station.
    hexes = []
    for s in range(prof.size - 1):
        a, b = s * n_sec, (s + 1) * n_sec
        for q in quads:
            hexes.append([a + q[0], b + q[0], a + q[1], b + q[1],
                          a + q[2], b + q[2], a + q[3], b + q[3]])
    return nodes, np.array(hexes)


# A brick's six faces, in the corner order `sweep` lays them down --
# x fastest, then y, then z -- each wound so its normal points OUT of
# the element.
_FACES = ((0, 2, 3, 1), (4, 5, 7, 6),
          (0, 1, 5, 4), (2, 6, 7, 3),
          (0, 4, 6, 2), (1, 3, 7, 5))


def surface(nodes, hexes, crease=60.0):
    """The skin of a solid mesh, as triangles with normals.

    An inside face is shared by the two bricks either side of it and an
    outside one is not, so counting how often each face appears finds
    the skin without any geometry: the ones seen once are the surface.

    Corners are SPLIT at a hard edge. Averaging every face that meets at
    a corner is right where the surface curves and quite wrong where it
    turns: a box shaded that way has one normal at each corner pointing
    off into the diagonal, so its flats bulge and its edges melt. Faces
    meeting within `crease` of each other are averaged together and
    everything else gets a corner of its own, which keeps an edge an
    edge and leaves a barrel smooth.

    Sixty degrees because of what sits either side of it. A revolved
    barrel is a polygon -- eight-sided at the coarsest section here, so
    its facets meet at forty-five -- and a box turns at ninety. Anything
    between those two smooths the one and keeps the other, and forty was
    on the wrong side of it: the barrels came out faceted.
    """
    seen = {}
    for elem in hexes:
        for face in _FACES:
            quad = tuple(int(elem[i]) for i in face)
            key = tuple(sorted(quad))
            if key in seen:
                seen[key] = None          # shared, so inside
            else:
                seen[key] = quad
    tris = []
    for quad in seen.values():
        if quad is None:
            continue
        a, b, c, d = quad
        tris.append((a, b, c))
        tris.append((a, c, d))
    verts = np.asarray(nodes, dtype=np.float64)
    if not tris:
        return verts, np.zeros((0, 3), dtype=np.int32), np.zeros_like(verts)
    tris = np.array(tris, dtype=np.int32)

    # Face normals, kept unnormalised as well: their length is twice the
    # triangle's area, which is the weight a corner's average wants.
    raw = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]],
                   verts[tris[:, 2]] - verts[tris[:, 0]])
    span = np.linalg.norm(raw, axis=1)
    unit = raw / np.where(span < 1e-30, 1.0, span)[:, None]

    incident = [[] for _ in range(len(verts))]
    for index, tri in enumerate(tris):
        for corner in tri:
            incident[corner].append(index)

    limit = math.cos(math.radians(crease))
    out_pos, out_nrm = [], []
    # (original corner, which group of faces) -> the corner we emitted
    where = {}
    groups = {}
    for corner, faces in enumerate(incident):
        reps = []
        for face in faces:
            for slot, rep in enumerate(reps):
                if float(np.dot(unit[face], rep)) >= limit:
                    groups[(corner, face)] = slot
                    break
            else:
                groups[(corner, face)] = len(reps)
                reps.append(unit[face])
        for slot in range(len(reps)):
            acc = np.zeros(3)
            for face in faces:
                if groups[(corner, face)] == slot:
                    acc += raw[face]
            length = float(np.linalg.norm(acc))
            where[(corner, slot)] = len(out_pos)
            out_pos.append(verts[corner])
            out_nrm.append(acc / length if length > 1e-30 else reps[slot])
    remapped = np.array(
        [[where[(corner, groups[(corner, index)])] for corner in tri]
         for index, tri in enumerate(tris)], dtype=np.int32)
    return (np.array(out_pos, dtype=np.float64), remapped,
            np.array(out_nrm, dtype=np.float64))


def displaced(nodes, shape, mode, amount=1.0):
    """The mesh pushed into the shape of one of its modes.

    Scaled against the size of the thing rather than against the raw
    eigenvector, whose length means nothing on its own -- so a swell of
    one moves the surface by a tenth of the object however it was
    normalised.
    """
    verts = np.asarray(nodes, dtype=np.float64)
    if shape is None or shape.size == 0 or mode < 0 \
            or mode >= shape.shape[1]:
        return verts
    move = shape[:, int(mode)].reshape(-1, 3)
    peak = float(np.abs(move).max())
    if peak <= 0.0:
        return verts
    span = float(np.abs(verts.max(axis=0) - verts.min(axis=0)).max())
    return verts + move * (amount * 0.1 * span / peak)


def _elastic(young, poisson):
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    d = np.zeros((6, 6))
    d[:3, :3] = lam
    d[0, 0] = d[1, 1] = d[2, 2] = lam + 2.0 * mu
    d[3, 3] = d[4, 4] = d[5, 5] = mu
    return d


def _strain_rows(grad):
    """The six strain rows a displacement gradient contributes.

    The same pattern the ordinary B matrix uses, written once so the
    incompatible modes below cannot drift from it.
    """
    block = np.zeros((6, 3))
    block[0, 0] = grad[0]
    block[1, 1] = grad[1]
    block[2, 2] = grad[2]
    block[3, 0] = grad[1]
    block[3, 1] = grad[0]
    block[4, 1] = grad[2]
    block[4, 2] = grad[1]
    block[5, 0] = grad[2]
    block[5, 2] = grad[0]
    return block


def assemble(nodes, hexes, young, poisson, density, incompatible=True):
    """Stiffness and mass of the mesh, free of any support.

    Free-free is right for something struck: a bell held by its crown
    or a bar on a stand is much closer to floating than to clamped, and
    the six modes that come out at zero -- the thing drifting and
    turning rather than ringing -- are dropped afterwards.
    """
    d_mat = _elastic(young, poisson)
    n_node = len(nodes)
    rows, cols, vals = [], [], []
    lumped = np.zeros(n_node)
    for elem in hexes:
        xyz = nodes[elem]
        k_e = np.zeros((24, 24))
        k_ua = np.zeros((24, 9))
        k_aa = np.zeros((9, 9))
        volume = 0.0
        if incompatible:
            centre = np.linalg.inv(
                (0.125 * np.column_stack([_CORNER[:, 0], _CORNER[:, 1],
                                          _CORNER[:, 2]])).T @ xyz)
        for xi in _GAUSS:
            d_nat = 0.125 * np.column_stack([
                _CORNER[:, 0] * (1.0 + xi[1] * _CORNER[:, 1])
                * (1.0 + xi[2] * _CORNER[:, 2]),
                _CORNER[:, 1] * (1.0 + xi[0] * _CORNER[:, 0])
                * (1.0 + xi[2] * _CORNER[:, 2]),
                _CORNER[:, 2] * (1.0 + xi[0] * _CORNER[:, 0])
                * (1.0 + xi[1] * _CORNER[:, 1])])
            jac = d_nat.T @ xyz
            det = float(np.linalg.det(jac))
            if det <= 0.0:
                # A brick turned inside out would hand back negative
                # stiffness and a garbage spectrum. Better to say so.
                raise ValueError('the mesh has an inverted element -- '
                                 'the profile probably doubles back')
            volume += det
            d_xyz = d_nat @ np.linalg.inv(jac).T
            b_mat = np.zeros((6, 24))
            b_mat[0, 0::3] = d_xyz[:, 0]
            b_mat[1, 1::3] = d_xyz[:, 1]
            b_mat[2, 2::3] = d_xyz[:, 2]
            b_mat[3, 0::3] = d_xyz[:, 1]
            b_mat[3, 1::3] = d_xyz[:, 0]
            b_mat[4, 1::3] = d_xyz[:, 2]
            b_mat[4, 2::3] = d_xyz[:, 1]
            b_mat[5, 0::3] = d_xyz[:, 2]
            b_mat[5, 2::3] = d_xyz[:, 0]
            k_e += b_mat.T @ d_mat @ b_mat * det
            if incompatible:
                # Three bubbles, one per natural direction, each
                # vanishing on the faces it is normal to.
                slopes = centre.T @ np.diag([-2.0 * xi[0], -2.0 * xi[1],
                                             -2.0 * xi[2]])
                g_mat = np.hstack([_strain_rows(slopes[:, m])
                                   for m in range(3)])
                k_ua += b_mat.T @ d_mat @ g_mat * det
                k_aa += g_mat.T @ d_mat @ g_mat * det
        if incompatible:
            # A brick cannot curve. Its faces stay flat however it is
            # loaded, so asked to BEND it resists by shearing instead --
            # which it should not have to -- and comes out far too
            # stiff. That is locking, and it is why a thin wall was out
            # of reach: the thinner the element, the worse the lie.
            #
            # The cure is three extra displacement shapes inside the
            # element, bubbles that vanish at every face, which let it
            # curve. They belong to no neighbour, so they are condensed
            # away here and the assembled problem never sees them: same
            # size, same solver, same mesh.
            #
            # Their gradients are taken at the element's CENTRE, not at
            # each Gauss point. That is not a shortcut -- it is what
            # makes the element still get a constant strain exactly
            # right, which is the one thing it must never lose.
            k_e -= k_ua @ np.linalg.solve(k_aa, k_ua.T)
        dofs = np.array([3 * v + c for v in elem for c in range(3)])
        rows.append(np.repeat(dofs, 24))
        cols.append(np.tile(dofs, 24))
        vals.append(k_e.ravel())
        lumped[elem] += density * volume / 8.0
    stiff = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * n_node, 3 * n_node)).tocsc()
    return stiff, sp.diags(np.repeat(lumped, 3)).tocsc()


def cavity_mesh(profile, length, sweep_mode='revolve', depth=None,
                carve='depth', mirror=False, wall=0.4):
    """The AIR inside a hollow shape, as a solid mesh of its own.

    Which needs no new machinery at all: the bore of a hollow form is
    the same form scaled down by the wall, so the cavity is that shape
    swept SOLID at the inner size. What is meshed is the space, not the
    thing around it.
    """
    inner = max(1.0e-4, 1.0 - wall)
    prof = [value * inner for value in np.asarray(profile,
                                                  dtype=float).ravel()]
    inner_depth = None if depth is None else depth * inner
    return sweep(prof, length, sweep_mode, inner_depth, None, carve,
                 mirror, 1.0)


def assemble_acoustic(nodes, hexes):
    """The two matrices an air cavity's modes come out of.

    Sound in a cavity obeys the Helmholtz equation, and as an
    eigenproblem that is far lighter than elasticity: ONE unknown at
    each corner, the pressure, where a solid has three. The matrices are
    the plain Laplacian and the plain mass, with no elastic constants in
    them anywhere -- air's stiffness and density enter only through the
    speed of sound, and that is one multiplication at the end.
    """
    n_node = len(nodes)
    rows, cols, kvals, mvals = [], [], [], []
    for elem in hexes:
        xyz = nodes[elem]
        k_e = np.zeros((8, 8))
        m_e = np.zeros((8, 8))
        for xi in _GAUSS:
            shape = 0.125 * ((1.0 + xi[0] * _CORNER[:, 0])
                             * (1.0 + xi[1] * _CORNER[:, 1])
                             * (1.0 + xi[2] * _CORNER[:, 2]))
            d_nat = 0.125 * np.column_stack([
                _CORNER[:, 0] * (1.0 + xi[1] * _CORNER[:, 1])
                * (1.0 + xi[2] * _CORNER[:, 2]),
                _CORNER[:, 1] * (1.0 + xi[0] * _CORNER[:, 0])
                * (1.0 + xi[2] * _CORNER[:, 2]),
                _CORNER[:, 2] * (1.0 + xi[0] * _CORNER[:, 0])
                * (1.0 + xi[1] * _CORNER[:, 1])])
            jac = d_nat.T @ xyz
            det = float(np.linalg.det(jac))
            if det <= 0.0:
                raise ValueError('the cavity mesh has an inverted element')
            d_xyz = d_nat @ np.linalg.inv(jac).T
            k_e += d_xyz @ d_xyz.T * det
            # Consistent rather than lumped: the mass matrix is only
            # eight by eight here, and for a wave problem it is worth
            # far more than it costs.
            m_e += np.outer(shape, shape) * det
        idx = np.asarray(elem)
        rows.append(np.repeat(idx, 8))
        cols.append(np.tile(idx, 8))
        kvals.append(k_e.ravel())
        mvals.append(m_e.ravel())
    shape_ = (n_node, n_node)
    stiff = sp.coo_matrix((np.concatenate(kvals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=shape_).tocsc()
    mass = sp.coo_matrix((np.concatenate(mvals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=shape_).tocsc()
    return stiff, mass


def cavity_modes(nodes, hexes, speed=343.0, open_start=False,
                 open_end=False, want=16):
    """What the air inside would sing, in hertz.

    An end left CLOSED is a rigid wall, which is what the equation does
    if nothing is said -- the natural condition. An end OPENED is a
    pressure release, which has to be said: those corners are struck out
    of the problem so the pressure there is held at nothing. That is the
    whole difference between a stopped pipe and an open one, and it is
    worth a factor of two in the note.
    """
    stiff, mass = assemble_acoustic(nodes, hexes)
    along = nodes[:, 0]
    loose = np.zeros(len(nodes), dtype=bool)
    if open_start:
        loose |= np.abs(along - along.min()) < 1e-9
    if open_end:
        loose |= np.abs(along - along.max()) < 1e-9
    keep = np.flatnonzero(~loose)
    if keep.size < 8:
        return np.zeros(0)
    stiff = stiff[keep][:, keep]
    mass = mass[keep][:, keep]
    count = min(want + 4, keep.size - 2)
    if count < 1:
        return np.zeros(0), np.zeros((len(nodes), 0))
    lam, vec = spl.eigsh(stiff, k=count, M=mass, sigma=-1.0, which='LM')
    order = np.argsort(np.real(lam))
    lam = np.maximum(np.real(lam)[order], 0.0)
    vec = np.real(vec)[:, order]
    freq = speed * np.sqrt(lam) / (2.0 * math.pi)
    # Back out to every corner, with nothing at the ones held open.
    full = np.zeros((len(nodes), vec.shape[1]))
    full[keep] = vec
    if not (open_start or open_end):
        # Sealed all round, a constant pressure everywhere costs nothing
        # and comes out at zero -- the air sitting still. Not a note.
        live = freq > 1e-6 * max(freq.max(), 1.0)
        freq, full = freq[live], full[:, live]
    return freq[:want], full[:, :want]


def cavity_table(nodes, freq, press, length=1.0, strike=1.0, damping=1.0,
                 floor=0.02):
    """The cavity's modes as a table, weighted where it is driven.

    By how fast the air MOVES there, not by how hard it pushes. An air
    column is played at an opening -- across a bottle's mouth, at a
    flute's embouchure, at the end of a pipe -- and an opening is exactly
    where the pressure is nothing and the movement is everything. Weighted
    by pressure instead, driving at the open end of an open pipe woke no
    mode at all, since the pressure is held at zero there by definition:
    a table of nothing, from the one place anybody actually blows.

    Movement goes as the pressure's slope along the pipe, so that is what
    is measured -- and it puts the antinode where a player's mouth goes.
    """
    if freq.size == 0:
        return []
    along = nodes[:, 0]
    stations = np.unique(np.round(along, 9))
    if stations.size < 2:
        return []
    target = stations[0] + strike * (stations[-1] - stations[0])
    here = int(np.argmin(np.abs(stations - target)))
    # The station next door, on whichever side there is one.
    other = here + 1 if here + 1 < stations.size else here - 1
    span = abs(stations[other] - stations[here])
    if span <= 0.0:
        return []
    # The movement at every station, so each mode can be judged against
    # the BEST place it could have been driven rather than against the
    # other modes. Without that, driving somewhere nothing moves -- a
    # sealed cavity at its own rigid end, where the air cannot go
    # anywhere at all -- still produced a table, built out of whatever
    # numerical noise happened to survive being divided by itself.
    means = np.array([press[np.abs(along - x) < 1e-7].mean(axis=0)
                      for x in stations])
    flow = np.abs(np.diff(means, axis=0)) / np.abs(
        np.diff(stations))[:, None]
    reach = flow.max(axis=0)
    at = min(here, other)
    weight = np.where(reach > 0.0, flow[at] / np.where(reach > 0.0, reach,
                                                       1.0), 0.0)
    if float(weight.max() if weight.size else 0.0) <= 0.0:
        return []
    keep = weight >= floor
    if not keep.any():
        return []
    freq, weight = freq[keep], weight[keep]
    ratio = freq / freq[0]
    decay = ratio ** (-damping)
    return [[float(r), float(w), float(d)]
            for r, w, d in zip(ratio, weight, decay)]


# How long a cavity rings, as a Q. Not solved: it comes from what
# radiates out of the open ends and what rubs against the walls, and
# neither is here. An open pipe throws its energy away and a sealed one
# keeps it, which is most of the difference.
CAVITY_Q = {2: 30.0, 1: 60.0, 0: 220.0}


def cavity_ring(open_count, fundamental):
    """Sixty decibels of fall for the air, in seconds."""
    q = CAVITY_Q.get(int(open_count), 60.0)
    if fundamental <= 0.0:
        return 60.0
    return min(60.0, max(0.01, 2.2 * q / fundamental))


def solve_modes(nodes, hexes, material='aluminium', want=24):
    """Frequencies in hertz and their shapes, ringing modes only."""
    if isinstance(material, str):
        young, poisson, density = MATERIALS[material][:3]
    else:
        young, poisson, density = material[:3]
    stiff, mass = assemble(nodes, hexes, young, poisson, density)
    # Shifted below zero because free-free leaves the stiffness
    # singular: six ways to move the thing without straining it.
    lam, vec = spl.eigsh(stiff, k=min(want + 8, 3 * len(nodes) - 2),
                         M=mass, sigma=-1.0, which='LM')
    order = np.argsort(lam)
    lam, vec = np.maximum(lam[order], 0.0), vec[:, order]
    freq = np.sqrt(lam) / (2.0 * math.pi)
    if freq.size <= 6:
        return freq, vec
    # The six drifting-and-turning modes sit orders below the first real
    # one, so a gap test finds them without assuming there are exactly
    # six -- a mesh in one piece has six, but a bad profile may not be.
    live = freq > 0.02 * freq[6:].min()
    return freq[live][:want], vec[:, live][:, :want]


def mode_table(profile, length=1.0, sweep_mode='revolve',
               material='aluminium', depth=None, strike=1.0,
               damping=1.0, count=16, section=None,
               direction=(0.0, 0.0, 1.0), floor=0.02, carve='depth',
               mirror=False, wall=1.0):
    """The three columns modal~ wants, worked out from a shape.

    `strike` is where along the length it is hit, nought at one end and
    one at the other: a mode with a node there is not woken, which is
    the whole reason a marimba bar is struck where it is and a bell is
    not rung at its crown.

    `direction` is which way it is hit, and it matters as much as
    where. A bar struck on its face wakes the modes that bend it that
    way, and hardly touches the ones that bend it sideways or twist it
    -- so a weight taken from how far a mode moves, without asking
    WHICH WAY it moves, fills the table with modes the mallet never
    reached. Anything under `floor` of the loudest is left out
    altogether, and the ratios are counted from the lowest mode that
    survives, since that is the one that will be heard as the pitch.

    `damping` is a power law, and 1 is not an arbitrary default: a
    material with a constant loss factor loses the same FRACTION of a
    mode's energy every cycle, and a mode an octave up gets through its
    cycles twice as fast, so its ring is exactly half as long. What is
    still imposed rather than solved is everything that is not a
    constant loss factor -- how the thing is held, what it radiates,
    losses that climb or fall with frequency of their own accord -- so
    the knob is there. 0 leaves every mode ringing as long as the last.
    """
    nodes, hexes = sweep(profile, length, sweep_mode, depth, section,
                         carve, mirror, wall)
    freq, shape = solve_modes(nodes, hexes, material, want=count)
    return table_from(nodes, freq, shape, length, strike, damping,
                      direction, floor)


def table_from(nodes, freq, shape, length=1.0, strike=1.0, damping=0.5,
               direction=(0.0, 0.0, 1.0), floor=0.02):
    """The table, from a solve already done.

    Kept apart from the solving because where it is struck, which way,
    and how the decay is shaped cost nothing to change -- they only
    reweigh modes that are already known. Only the SHAPE, the size and
    the material need the eigenproblem doing again, so a knob for the
    mallet can stay live while one for the outline cannot.
    """
    if freq.size == 0:
        return []
    # Which node is nearest where it was hit, and how much each mode
    # moves there. A strike drives a mode in proportion to how far that
    # mode's own shape moves at the point struck.
    along = nodes[:, 0]
    target = along.min() + strike * (along.max() - along.min())
    near = int(np.argmin(np.abs(along - target)))
    ring = np.arange(len(nodes))[np.abs(along - along[near])
                                 < 1e-9 * max(length, 1.0)]
    axis = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(axis))
    axis = axis / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
    # AVERAGED over the contact, not maximised over it. A mallet lands
    # on an area and drives the whole of it one way, so what it feeds a
    # mode is that mode's mean motion under the contact, projected on
    # the way the mallet is going. Taking the largest instead, any
    # single node moving picks the mode up: a mode that bends the bar
    # SIDEWAYS moves nothing at all on average in the direction of a
    # face strike, but somewhere in the section it moves a little, so
    # the table filled with modes the mallet could not reach and every
    # weight came out much the same. An average cancels them, which is
    # what actually happens.
    weight = np.array([
        abs(float(np.mean(shape[:, i].reshape(-1, 3)[ring] @ axis)))
        for i in range(shape.shape[1])])
    peak = float(weight.max())
    if peak <= 0.0:
        return []
    weight = weight / peak
    keep = weight >= floor
    if not keep.any():
        return []
    freq, weight = freq[keep], weight[keep]
    # Modes at the SAME frequency are one thing to the ear. A square
    # section bends the same in both planes, so those two modes are
    # degenerate -- and a solver hands back an arbitrary mixture of any
    # degenerate pair, since within that subspace one direction is as
    # good as another. So they cannot be told apart by which way they
    # move, and both carry part of the strike. Added in quadrature,
    # which is what two modes at one frequency driven together come to,
    # they collapse back into the single mode that is actually heard --
    # and the table stops wasting half its rows on twins.
    merged_f, merged_w = [], []
    for f_i, w_i in zip(freq, weight):
        if merged_f and abs(f_i - merged_f[-1]) < 0.01 * merged_f[-1]:
            merged_w[-1] = math.hypot(merged_w[-1], w_i)
        else:
            merged_f.append(float(f_i))
            merged_w.append(float(w_i))
    freq = np.array(merged_f)
    weight = np.array(merged_w)
    peak = float(weight.max())
    if peak > 0.0:
        weight = weight / peak
    ratio = freq / freq[0]
    decay = ratio ** (-damping)
    return [[float(r), float(w), float(d)]
            for r, w, d in zip(ratio, weight, decay)]
