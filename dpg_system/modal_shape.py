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


# Young's modulus in pascals, Poisson's ratio, density in kg/m3. Only
# their proportions matter here, since the table is ratios -- but they
# are real numbers rather than invented ones, so the frequencies that
# come out are believable if you ever want to read them.
MATERIALS = {
    'aluminium': (7.0e10, 0.33, 2700.0),
    'steel': (2.0e11, 0.30, 7850.0),
    'brass': (1.0e11, 0.34, 8500.0),
    'bronze': (1.05e11, 0.34, 8800.0),
    'copper': (1.2e11, 0.34, 8960.0),
    'glass': (7.0e10, 0.22, 2500.0),
    'stone': (5.0e10, 0.25, 2700.0),
    'wood': (1.1e10, 0.35, 600.0),
    'bone': (1.8e10, 0.30, 1900.0),
    'nylon': (3.0e9, 0.40, 1150.0),
    'rubber': (5.0e7, 0.48, 1100.0),
}

SWEEPS = ('revolve', 'extrude', 'mirror')

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


def rect_section(across=4, through=2):
    """A rectangular cross-section on the unit square, [-1, 1] both ways."""
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


def sweep(profile, length, sweep_mode='revolve', depth=None,
          section=None):
    """Sweep an outline into a solid, as a regular grid of bricks.

    `profile` is a list of half-widths, evenly spaced along the length.
    What that half-width MEANS is the sweep:

      'revolve'  it is a radius, spun about the long axis -- a club, an
                 egg, a cone, a turned bead.
      'extrude'  it is a half-width, and the section is that wide by
                 `depth` deep -- a bar with a shaped outline, which is
                 what an undercut marimba bar is.
      'mirror'   the same, but the outline you gave is HALF the length
                 and is reflected end to end, so you draw one end of a
                 symmetric thing and get the whole.
    """
    prof = np.asarray(profile, dtype=float).ravel()
    if prof.size < 2:
        raise ValueError('a profile needs at least two half-widths')
    if np.any(prof <= 0.0):
        raise ValueError('half-widths must all be greater than zero')
    if sweep_mode == 'mirror':
        prof = np.concatenate([prof, prof[::-1]])
    if section is None:
        section = (disc_section() if sweep_mode == 'revolve'
                   else rect_section())
    pts2d, quads = section
    if sweep_mode != 'revolve':
        if depth is None:
            depth = float(prof.max())
        # `depth` is the whole depth, not half of it: the section runs
        # from minus to plus one, so it wants halving.
        scale = np.array([1.0,
                          0.5 * depth / max(float(prof.max()), 1e-12)])
    else:
        scale = np.array([1.0, 1.0])

    n_sec = len(pts2d)
    stations = np.linspace(0.0, length, prof.size)
    nodes = np.empty((prof.size * n_sec, 3))
    for s, (z, w) in enumerate(zip(stations, prof)):
        block = pts2d * scale * w
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


def _elastic(young, poisson):
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    d = np.zeros((6, 6))
    d[:3, :3] = lam
    d[0, 0] = d[1, 1] = d[2, 2] = lam + 2.0 * mu
    d[3, 3] = d[4, 4] = d[5, 5] = mu
    return d


def assemble(nodes, hexes, young, poisson, density):
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
        volume = 0.0
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
        dofs = np.array([3 * v + c for v in elem for c in range(3)])
        rows.append(np.repeat(dofs, 24))
        cols.append(np.tile(dofs, 24))
        vals.append(k_e.ravel())
        lumped[elem] += density * volume / 8.0
    stiff = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * n_node, 3 * n_node)).tocsc()
    return stiff, sp.diags(np.repeat(lumped, 3)).tocsc()


def solve_modes(nodes, hexes, material='aluminium', want=24):
    """Frequencies in hertz and their shapes, ringing modes only."""
    if isinstance(material, str):
        young, poisson, density = MATERIALS[material]
    else:
        young, poisson, density = material
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
               damping=0.5, count=16, section=None,
               direction=(0.0, 0.0, 1.0), floor=0.02):
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

    `damping` is IMPOSED, not solved. Higher modes really do die away
    faster, but how much faster depends on the material's own losses, on
    how the thing is held and on what it radiates -- none of which is in
    here. So it is a power law with a knob on it: 0 leaves every mode
    ringing as long as the last, and 1 has a mode an octave up die twice
    as fast.
    """
    nodes, hexes = sweep(profile, length, sweep_mode, depth, section)
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
