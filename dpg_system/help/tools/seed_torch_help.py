"""Seed the help pages for the first batch of new torch nodes (2026-10-25): one page per node.

    python3 dpg_system/help/tools/seed_torch_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

t.matmul, t.einsum, t.linalg.norm, t.linalg.inv, t.linalg.det, t.linalg.pinv,
t.linalg.solve, t.linalg.lstsq, t.std, t.var, t.argmin, t.quantile, t.mode, t.sort,
t.topk, t.where, t.unique, t.isnan, t.isfinite, t.isinf, t.nan_to_num, t.pad, t.unfold,
t.interpolate, t.stft, t.istft, t.convolve, t.correlate.

The stft and istft demos build their signals in subpatchers (make_chirp_4096,
make_wave). Where a fan-out feeds both a cold and a hot inlet of one node, the cold
branch is linked FIRST (an outlet sends in link order). Not a make_* script: rebuild.py
never runs it, and it will not write over an existing page without --force. Treat the
pages as hand-made once written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from seed_numpy_more_help import LIST, MSG, HM, BARS, C, N, see_also
from seed_scipy_help import S


PAGES = {}


def page(stem, title, body, demo, links, also, also_y, text_height=820, subpatch=None):
    PAGES[stem] = dict(title=title, body=body.strip('\n'), demo=demo + see_also(also_y, also),
                       links=links, text_height=text_height, subpatch=subpatch)


METRO_HALF = {'on': True, 'period': 500.0, 'units': 'milliseconds'}
METRO_1S = {'on': True, 'period': 1000.0, 'units': 'milliseconds'}

DEVICE = """The answer stays on the input's device: a tensor on the GPU is worked on there and
comes out there."""


def metro(y=53, label='twice a second', props=METRO_HALF, init='metro 500'):
    return [C('cm', label, 28, y), N('mt', init, 28, y + 39, 129, 70, props)]


def LINE(count, miny, maxy, width=360, height=130):
    return dict(BARS(count, maxy, width, height, 'line'), **{'min y': miny})


def ROUND(key, x, y, places=2):
    return N(key, 'round', x, y, 100, 70, {'decimals': places})


def W(text):
    """A message or list wide enough for this text."""
    return max(60, int(len(text) * 13.5) + 30)


# ------------------------------------------------------------------- t.matmul
page('t.matmul', 't.matmul - multiply matrices (or a whole stack of them)',
"""
Matrix multiplication: each row of the first input combined with each column of the
second. Turning points with a rotation matrix, applying a transform, mixing channels
with a weight matrix, projecting data onto principal axes - all are one t.matmul.

A stack is handled in one step: give it 24 rotation matrices (shape 24 x 3 x 3) and 24
vectors (24 x 3 x 1) and every joint is turned by its own matrix at once. A single
matrix as the second input is applied to every matrix of a stack.

SYNTAX:
t.matmul

INPUTS and PARAMETERS:

in 1:
The left matrix (or stack). Every input multiplies and sends - the only inlet that
makes the node send.

in 2:
The right matrix (or stack), held. Its last-but-one size must equal in 1's last size.
It is converted to in 1's device and type.

OUTPUTS:

result:
The product. Whole numbers come out as floats.

""" + DEVICE + """

READING THE DEMO:
Four points, one per row: right, up, left, down. Times the matrix 0 1, -1 0 - a quarter
turn - each point moves a quarter of the way round, counter-clockwise: right becomes up,
up becomes left.

RELATED:
np.matmul is the numpy version. t.einsum writes any product of this kind as a formula.
t.linalg.inv undoes a matrix.
""",
metro() + [
    C('c1', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 214, 88, 46),
    C('c2', 'a quarter turn', 180, 175),
    N('mr', 'message', 180, 214, W('0 1 -1 0'), 42, MSG('0 1 -1 0')),
    N('rr', 't.reshape 2 2', 340, 214, 150, 70),
    C('c3', 'four points: x y in each row', 180, 53),
    N('mp', 'message', 180, 92, W('1 0 0 1 -1 0 0 -1'), 42, MSG('1 0 0 1 -1 0 0 -1')),
    N('rp', 't.reshape 4 2', 460, 92, 150, 70),
    C('c4', 'each point times the turn', 600, 175),
    N('mm', 't.matmul', 600, 214, 150, 90),
    C('c5', 'turned: right is now up, up is now left', 480, 330),
    N('l1', 'list', 480, 369, 360, 42, LIST),
],
[('lb', 'out', 'mr', ''), ('mr', 'message out', 'rr', 'tensor in'), ('rr', 'output', 'mm', 'in 2'),
 ('mt', '', 'mp', ''), ('mp', 'message out', 'rp', 'tensor in'), ('rp', 'output', 'mm', 'in 1'),
 ('mm', 'result', 'l1', '')],
['np.matmul', 't.einsum', 't.linalg.inv', 't.linalg.solve'], 470, 1050)


# ------------------------------------------------------------------- t.einsum
page('t.einsum', 't.einsum - products and sums written as a formula',
"""
Combines tensors according to a short formula naming their dimensions. Each input gets
a letter for each of its dimensions; letters shared between inputs are multiplied
together, and letters missing from the result (after ->) are summed away.

ij,jk->ik     matrix product (t.matmul)
ij,ij->i      the dot product of each row of a with the same row of b
ij->j         the sum down each column
ii->          the trace: the sum of the diagonal
bij,bjk->bik  a stack of matrix products
i,j->ij       the outer product: every a times every b

It says exactly which dimensions meet, so it is clear and fast where t.matmul would
need t.permute and t.unsqueeze around it.

SYNTAX:
t.einsum <formula>
Default: ij,jk->ik. The number of commas before -> sets the number of inlets; the
formula can be changed later in the subscripts option, with the same number of inputs.

INPUTS and PARAMETERS:

in:
The first tensor. Every input computes and sends - the only inlet that makes the node
send.

in 2, in 3, ...:
The other tensors, held, converted to in's device and type.

subscripts:
An option, under the node: the formula.

OUTPUTS:

result:
The tensor the formula describes.

""" + DEVICE + """

READING THE DEMO:
ij,ij->i: three rows of a, each multiplied by the same row of b and summed - three dot
products at once. 1 2 . 1 0 = 1; 3 4 . 0 1 = 4; 5 6 . 1 1 = 11.

RELATED:
np.einsum is the numpy version. t.matmul is the matrix product. t.sum sums along one
dimension.
""",
metro() + [
    C('c1', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 214, 88, 46),
    C('c2', 'b: three rows', 180, 175),
    N('mb', 'message', 180, 214, W('1 0 0 1 1 1'), 42, MSG('1 0 0 1 1 1')),
    N('rb', 't.reshape 3 2', 420, 214, 150, 70),
    C('c3', 'a: three rows', 180, 53),
    N('ma', 'message', 180, 92, W('1 2 3 4 5 6'), 42, MSG('1 2 3 4 5 6')),
    N('ra', 't.reshape 3 2', 420, 92, 150, 70),
    C('c4', 'each row of a dot the same row of b', 680, 53),
    N('es', 't.einsum ij,ij->i', 680, 92, 200, 90),
    N('l1', 'list', 680, 214, 160, 42, LIST),
],
[('lb', 'out', 'mb', ''), ('mb', 'message out', 'rb', 'tensor in'), ('rb', 'output', 'es', 'in 2'),
 ('mt', '', 'ma', ''), ('ma', 'message out', 'ra', 'tensor in'), ('ra', 'output', 'es', 'in'),
 ('es', 'result', 'l1', '')],
['np.einsum', 't.matmul', 't.sum', 't.permute'], 340, 1200)


# -------------------------------------------------------------- t.linalg.norm
page('t.linalg.norm', 't.linalg.norm - the length of vectors',
"""
Measures the size of a tensor: by default its length, the square root of the sum of
the squares. Given a dim, it measures along that dim only - so t.linalg.norm -1 on 24
velocity vectors (24 x 3) gives the 24 speeds, one per joint.

SYNTAX:
t.linalg.norm <dim>
Without a dim, the whole tensor gives one number. With one, a dim box and a keep_dims
checkbox appear.

INPUTS and PARAMETERS:

tensor in:
The tensor to measure. Every input sends.

dim (int box, also an inlet; only when a dim was given):
The dim measured along: -1, the last, measures each row.

keep_dims (checkbox, also an inlet; only when a dim was given):
Keep the measured dim, as size 1, so the answer lines up with the input (to divide by
it, for instance, making unit vectors).

ord:
An option, under the node. 2 (the default) is ordinary length; 1 is the sum of the
sizes (distance walking a grid); inf is the largest size; 0 counts the non-zero
values.

OUTPUTS:

norm:
The length(s).

""" + DEVICE + """

READING THE DEMO:
Top: the whole tensor 3 4 is one vector, length 5. Bottom: with dim -1, each row is its
own vector: 3 4 is 5, 6 8 is 10, 5 12 is 13.

RELATED:
np.linalg.norm is the numpy version. t.length gives the length of a single vector;
t.cdist the distances between points; t.normalize makes vectors length 1.
""",
metro() + [
    C('c1', 'one vector', 180, 53),
    N('m1', 'message', 180, 92, W('3 4'), 42, MSG('3 4')),
    N('n1', 't.linalg.norm', 340, 92, 160, 70),
    N('l1', 'list', 540, 92, 80, 42, LIST),
    C('c2', 'three vectors, one per row', 180, 190),
    N('m2', 'message', 180, 229, W('3 4 6 8 5 12'), 42, MSG('3 4 6 8 5 12')),
    N('r2', 't.reshape 3 2', 380, 229, 150, 70),
    C('c3', 'dim -1: the length of each row', 640, 190),
    N('n2', 't.linalg.norm -1', 640, 229, 180, 110),
    N('l2', 'list', 640, 370, 160, 42, LIST),
],
[('mt', '', 'm1', ''), ('mt', '', 'm2', ''), ('m1', 'message out', 'n1', 'tensor in'),
 ('n1', 'norm', 'l1', ''), ('m2', 'message out', 'r2', 'tensor in'), ('r2', 'output', 'n2', 'tensor in'),
 ('n2', 'norm', 'l2', '')],
['np.linalg.norm', 't.length', 't.cdist', 't.normalize'], 470, 1050)


# --------------------------------------------------------------- t.linalg.inv
page('t.linalg.inv', 't.linalg.inv - the matrix that undoes a matrix',
"""
Gives the inverse of a square matrix: the matrix that undoes it. If a matrix turns,
stretches or shears points, its inverse turns, stretches or shears them back; the two
multiplied together give the identity (1s down the diagonal, 0s elsewhere), which
changes nothing.

Use it to go back from a transformed space to the original one: a camera's view back to
the room, a body's frame back to the world, a calibration undone.

A matrix that flattens space (its determinant is 0: two of its rows point the same way)
cannot be undone, and the node sends nothing. t.linalg.pinv gives the closest thing.

SYNTAX:
t.linalg.inv

INPUTS and PARAMETERS:

tensor in:
A square matrix, or a stack of them (each is inverted). Every input sends.

OUTPUTS:

inverse:
The inverse, the same shape.

""" + DEVICE + """

READING THE DEMO:
The matrix 2 1, 1 1 and its inverse 1 -1, -1 2. Multiplied together (right) they give
the identity, 1 0, 0 1.

RELATED:
np.linalg.inv is the numpy version. t.linalg.solve solves A x = b directly, which is
quicker and more accurate than multiplying by the inverse. t.linalg.det says whether a
matrix can be inverted. t.linalg.pinv works for any matrix.
""",
metro() + [
    C('c1', 'a matrix', 180, 53),
    N('m1', 'message', 180, 92, W('2 1 1 1'), 42, MSG('2 1 1 1')),
    N('r1', 't.reshape 2 2', 340, 92, 150, 70),
    C('c2', 'its inverse', 600, 53),
    N('iv', 't.linalg.inv', 600, 92, 150, 70),
    N('l1', 'list', 600, 190, 180, 42, LIST),
    C('c3', 'the matrix times its inverse', 340, 280),
    N('mm', 't.matmul', 340, 319, 150, 90),
    C('c4', 'the identity', 600, 280),
    N('l2', 'list', 600, 319, 180, 42, LIST),
],
# the inverse reaches t.matmul's held inlet before the matrix triggers it
[('mt', '', 'm1', ''), ('m1', 'message out', 'r1', 'tensor in'), ('r1', 'output', 'iv', 'tensor in'),
 ('r1', 'output', 'mm', 'in 1'), ('iv', 'inverse', 'l1', ''), ('iv', 'inverse', 'mm', 'in 2'),
 ('mm', 'result', 'l2', '')],
['np.linalg.inv', 't.linalg.solve', 't.linalg.det', 't.linalg.pinv'], 460, 1050)


# --------------------------------------------------------------- t.linalg.det
page('t.linalg.det', 't.linalg.det - how much a matrix scales space',
"""
Gives the determinant of a square matrix: the factor by which it scales areas (in 2-D)
or volumes (in 3-D). 2 means everything comes out twice the size; 1 means size is kept
(a rotation); 0 means space is flattened onto a line or plane - the matrix cannot be
undone; a negative number means it is also mirrored.

Use it to test whether a matrix can be inverted, whether a transform mirrors (a
rotation estimated from noisy data with det -1 is a reflection), or how much a mapping
stretches.

SYNTAX:
t.linalg.det

INPUTS and PARAMETERS:

tensor in:
A square matrix, or a stack of them (one determinant each). Every input sends.

OUTPUTS:

determinant:
The determinant.

""" + DEVICE + """

READING THE DEMO:
2 0, 0 3 stretches by 2 one way and 3 the other: areas grow 6 times. 1 2, 2 4 has a
second row twice the first, so it flattens the plane onto a line: 0 (shown as -0.000: the
arithmetic lands on zero from just below). 0 1, 1 0 swaps x and y, a mirror: -1.

RELATED:
np.linalg.det is the numpy version. t.linalg.inv inverts a matrix whose determinant is
not 0. t.linalg.eig gives the separate stretch factors, whose product is the
determinant.
""",
metro() + [
    C('c1', 'stretch 2 and 3', 180, 53),
    N('m1', 'message', 180, 92, W('2 0 0 3'), 42, MSG('2 0 0 3')),
    N('r1', 't.reshape 2 2', 340, 92, 150, 70),
    N('d1', 't.linalg.det', 600, 92, 150, 70),
    N('l1', 'list', 780, 92, 80, 42, LIST),
    C('c2', 'flattens onto a line', 180, 190),
    N('m2', 'message', 180, 229, W('1 2 2 4'), 42, MSG('1 2 2 4')),
    N('r2', 't.reshape 2 2', 340, 229, 150, 70),
    N('d2', 't.linalg.det', 600, 229, 150, 70),
    N('l2', 'list', 780, 229, 80, 42, LIST),
    C('c3', 'a mirror', 180, 327),
    N('m3', 'message', 180, 366, W('0 1 1 0'), 42, MSG('0 1 1 0')),
    N('r3', 't.reshape 2 2', 340, 366, 150, 70),
    N('d3', 't.linalg.det', 600, 366, 150, 70),
    N('l3', 'list', 780, 366, 80, 42, LIST),
],
[('mt', '', 'm1', ''), ('mt', '', 'm2', ''), ('mt', '', 'm3', ''),
 ('m1', 'message out', 'r1', 'tensor in'), ('r1', 'output', 'd1', 'tensor in'), ('d1', 'determinant', 'l1', ''),
 ('m2', 'message out', 'r2', 'tensor in'), ('r2', 'output', 'd2', 'tensor in'), ('d2', 'determinant', 'l2', ''),
 ('m3', 'message out', 'r3', 'tensor in'), ('r3', 'output', 'd3', 'tensor in'), ('d3', 'determinant', 'l3', '')],
['np.linalg.det', 't.linalg.inv', 't.linalg.eig'], 480, 1000)


# -------------------------------------------------------------- t.linalg.pinv
page('t.linalg.pinv', 't.linalg.pinv - the best possible undoing of any matrix',
"""
Gives the pseudo-inverse: the closest thing to an inverse for a matrix that has none -
one that is not square, or that flattens space. Where an exact undoing is impossible,
it gives the one that misses by the least (in the least-squares sense), and where
several are possible it gives the smallest.

Use it to fit: with a matrix whose rows are measurements of known things, pinv times
the readings gives the best estimate of the unknowns (t.linalg.lstsq does the same in
one step). It also inverts matrices too close to flat for t.linalg.inv to trust.

SYNTAX:
t.linalg.pinv

INPUTS and PARAMETERS:

tensor in:
Any matrix, or a stack. A matrix of 3 rows and 2 columns gives one of 2 rows and 3
columns. Every input sends.

OUTPUTS:

pseudo-inverse:
The pseudo-inverse.

""" + DEVICE + """ On the Mac's GPU (mps) part of the work is done on the CPU, which
torch says in a warning the first time.

READING THE DEMO:
The 3 x 2 matrix 1 1, 1 -1, 1 1 has no inverse; its pseudo-inverse is 2 x 3: 0.25 0.5
0.25, 0.25 -0.5 0.25. pinv times the matrix is exactly the 2 x 2 identity (bottom): it
undoes the matrix for anything the matrix can produce.

RELATED:
np.linalg.pinv is the numpy version. t.linalg.inv is the exact inverse of a square
matrix. t.linalg.lstsq solves a fitting problem directly. t.linalg.svd is how pinv is
computed.
""",
metro() + [
    C('c1', 'three rows of two', 180, 53),
    N('m1', 'message', 180, 92, W('1 1 1 -1 1 1'), 42, MSG('1 1 1 -1 1 1')),
    N('r1', 't.reshape 3 2', 400, 92, 150, 70),
    C('c2', 'two rows of three', 640, 53),
    N('pv', 't.linalg.pinv', 640, 92, 160, 70),
    ROUND('rd', 640, 190),
    N('l1', 'list', 440, 300, 360, 42, LIST),
    C('c3', 'pinv times the matrix', 180, 250),
    N('mm', 't.matmul', 180, 289, 150, 90),
    ROUND('r2', 180, 400),
    C('c4', 'the identity', 380, 500),
    N('l2', 'list', 180, 490, 180, 42, LIST),
],
# the matrix reaches t.matmul's held inlet before pinv triggers it
[('mt', '', 'm1', ''), ('m1', 'message out', 'r1', 'tensor in'), ('r1', 'output', 'mm', 'in 2'),
 ('r1', 'output', 'pv', 'tensor in'), ('pv', 'pseudo-inverse', 'rd', 'in'), ('rd', 'result', 'l1', ''),
 ('pv', 'pseudo-inverse', 'mm', 'in 1'), ('mm', 'result', 'r2', 'in'), ('r2', 'result', 'l2', '')],
['np.linalg.pinv', 't.linalg.inv', 't.linalg.lstsq', 't.linalg.svd'], 600, 1100)


# ------------------------------------------------------------- t.linalg.solve
page('t.linalg.solve', 't.linalg.solve - find the x that makes A x = b',
"""
Solves a set of linear equations: given the matrix A of their coefficients and the
values b they must equal, finds the unknowns x. For example, 2x + y = 3.5 and x + 3y = 3
is A = 2 1, 1 3 and b = 3.5 3.

Use it to find the mix of known patterns that adds up to a reading, the weights that
reach given targets, or the point where lines or planes meet. It is quicker and more
accurate than multiplying b by the inverse of A.

SYNTAX:
t.linalg.solve

INPUTS and PARAMETERS:

A matrix:
The square matrix of coefficients, one row per equation. Every input solves and sends
- the only inlet that makes the node send.

b:
What each equation must equal, held: one value per row of A, or a matrix with one
column per problem to solve with the same A.

OUTPUTS:

x:
The unknowns. Nothing is sent if A has no inverse (its equations contradict or repeat
each other); t.linalg.lstsq gives the best answer there.

""" + DEVICE + """

READING THE DEMO:
2x + y = 3.5 and x + 3y = 3: x is 1.5 and y is 0.5 (2 x 1.5 + 0.5 = 3.5; 1.5 + 3 x 0.5 =
3).

RELATED:
np.linalg.solve is the numpy version. t.linalg.lstsq solves when there are more
equations than unknowns. t.linalg.inv gives the inverse itself.
""",
metro() + [
    C('c1', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 214, 88, 46),
    C('c2', 'b: what they equal', 180, 175),
    N('mb', 'message', 180, 214, W('3.5 3'), 42, MSG('3.5 3')),
    C('c3', 'A: 2x + y and x + 3y', 180, 53),
    N('ma', 'message', 180, 92, W('2 1 1 3'), 42, MSG('2 1 1 3')),
    N('ra', 't.reshape 2 2', 340, 92, 150, 70),
    C('c4', 'x and y', 600, 53),
    N('sv', 't.linalg.solve', 600, 92, 160, 90),
    ROUND('rd', 600, 205),
    N('l1', 'list', 730, 205, 120, 42, LIST),
],
[('lb', 'out', 'mb', ''), ('mb', 'message out', 'sv', 'b'),
 ('mt', '', 'ma', ''), ('ma', 'message out', 'ra', 'tensor in'), ('ra', 'output', 'sv', 'A matrix'),
 ('sv', 'x', 'rd', 'in'), ('rd', 'result', 'l1', '')],
['np.linalg.solve', 't.linalg.lstsq', 't.linalg.inv'], 340, 1000)


# ------------------------------------------------------------- t.linalg.lstsq
page('t.linalg.lstsq', 't.linalg.lstsq - the best fit when the equations disagree',
"""
Solves A x = b when there are more equations than unknowns and no x satisfies them all
exactly - the usual case with measured data. It finds the x that misses by the least:
the smallest sum of squared differences between A x and b.

Fitting a straight line through noisy points is the classic case: each point says
slope x its position + offset = its value; with many points the line that best agrees
with all of them is the answer. Fitting a calibration, a trend, or how much of each
known pattern a reading contains all take the same form.

SYNTAX:
t.linalg.lstsq

INPUTS and PARAMETERS:

A matrix:
One row per equation (per measurement), one column per unknown. Every input solves and
sends - the only inlet that makes the node send.

b:
The measured values, one per row of A, held. A matrix fits several sets at once.

OUTPUTS:

x:
The best-fitting unknowns.

residuals:
The sum of the squared misses that remain: 0 for a perfect fit. Sent first.

""" + DEVICE + """ On the Mac's GPU (mps), torch has no lstsq: it is done on the CPU
and the answer moved back.

READING THE DEMO:
Six points near the line 2 x position + 1, four of them 0.25 off. Each row of A is a
position and a 1 (for the offset). The misses balance out, so the fit finds slope 2
and offset 1 exactly; the residual, 0.25, is the four misses of 0.25, squared and added.

RELATED:
np.linalg.lstsq is the numpy version. scipy.linregress fits a line and reports how good
it is. scipy.curve_fit fits any formula. t.linalg.solve is the exact case.
""",
metro() + [
    C('c1', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 214, 88, 46),
    C('c2', 'b: the six values, each a little off', 180, 175),
    N('mb', 'message', 180, 214, W('1.25 2.75 5 7 8.75 11.25'), 42, MSG('1.25 2.75 5 7 8.75 11.25')),
    C('c3', 'A: each position, and a 1 for the offset', 180, 53),
    N('ma', 'message', 180, 92, W('0 1 1 1 2 1 3 1 4 1 5 1'), 42, MSG('0 1 1 1 2 1 3 1 4 1 5 1')),
    N('ra', 't.reshape 6 2', 530, 92, 150, 70),
    C('c4', 'slope, offset', 560, 175),
    N('ls', 't.linalg.lstsq', 560, 214, 160, 90),
    ROUND('r1', 560, 330),
    N('l1', 'list', 690, 330, 120, 42, LIST),
    C('c5', 'the misses that remain', 560, 420),
    ROUND('r2', 560, 459, 3),
    N('l2', 'list', 690, 459, 100, 42, LIST),
],
[('lb', 'out', 'mb', ''), ('mb', 'message out', 'ls', 'b'),
 ('mt', '', 'ma', ''), ('ma', 'message out', 'ra', 'tensor in'), ('ra', 'output', 'ls', 'A matrix'),
 ('ls', 'x', 'r1', 'in'), ('r1', 'result', 'l1', ''), ('ls', 'residuals', 'r2', 'in'), ('r2', 'result', 'l2', '')],
['np.linalg.lstsq', 'scipy.linregress', 'scipy.curve_fit', 't.linalg.solve'], 570, 1150)


# ------------------------------------------------------------- t.std / t.var
def spread_page(stem, word, other, value1, value0):
    noun = 'standard deviation' if stem == 't.std' else 'variance'
    meaning = ('how far the values typically are from their mean, in the same units as the values'
               if stem == 't.std' else
               'the mean of the squared distances from the mean: the standard deviation squared')
    page(stem, stem + ' - ' + word,
"""
Gives the """ + noun + """ - """ + meaning + """. It measures how
spread out values are: how jittery a sensor is when still, how much a joint moves over a
window, how varied the dancers' positions are.

SYNTAX:
""" + stem + """ <dim>
Without a dim, the whole tensor gives one number. With one, a dim box and a keep_dims
checkbox appear, and each slice along the dim gets its own (t.std 0 on frames x joints
gives one per joint).

INPUTS and PARAMETERS:

tensor in:
The values. Every input sends.

dim (int box, also an inlet; only when a dim was given):
The dim measured along.

keep_dims (checkbox, also an inlet; only when a dim was given):
Keep the measured dim, as size 1, so the answer lines up with the input.

correction:
An option, under the node. 1 (torch's default) divides by one less than the number of
values - the estimate of a whole population's spread from a sample of it. 0 divides by
the number of values, as np.std and np.var do; so the two disagree unless correction
is 0.

OUTPUTS:

output:
The """ + noun + """(s).

""" + DEVICE + """

READING THE DEMO:
2 4 4 4 5 5 7 9 has mean 5. With correction 1 the """ + noun + """ is """ + value1 + """;
with correction 0 it is """ + value0 + """.

RELATED:
""" + other + """ np.std and np.var are the numpy versions. t.mean gives the mean.
scipy.zscore measures each value in standard deviations from the mean.
""",
metro() + [
    C('c1', 'eight values, mean 5', 180, 53),
    N('m1', 'message', 180, 92, W('2 4 4 4 5 5 7 9'), 42, MSG('2 4 4 4 5 5 7 9')),
    C('c2', 'correction 1 (the default)', 450, 53),
    N('s1', stem, 450, 92, 150, 70),
    ROUND('r1', 640, 92, 3),
    N('l1', 'list', 760, 92, 90, 42, LIST),
    C('c3', 'correction 0, as numpy', 450, 190),
    N('s2', stem, 450, 229, 150, 70, {'correction': 0}),
    ROUND('r2', 640, 229, 3),
    N('l2', 'list', 760, 229, 90, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 's1', 'tensor in'), ('m1', 'message out', 's2', 'tensor in'),
 ('s1', 'output', 'r1', 'in'), ('r1', 'result', 'l1', ''), ('s2', 'output', 'r2', 'in'), ('r2', 'result', 'l2', '')],
[('t.var' if stem == 't.std' else 't.std'), 'np.std' if stem == 't.std' else 'np.var', 't.mean', 'scipy.zscore'],
     360, 1050)


spread_page('t.std', 'how spread out values are', 't.var gives the square of it.', '2.138', '2')
spread_page('t.var', 'the spread of values, squared', 't.std gives its square root, in the values\' units.',
            '4.571', '4')


# ------------------------------------------------------------------- t.argmin
page('t.argmin', 't.argmin - where the smallest value is',
"""
Gives the position of the smallest value: which joint is slowest, which sensor reads
lowest, which of several candidates is closest (the argmin of their distances).

SYNTAX:
t.argmin <dim>
Without a dim, the position counts through the whole tensor as if it were flattened.
With one, a dim box appears, and each slice gets its own position.

INPUTS and PARAMETERS:

tensor in:
The values. Every input sends.

dim (int box, also an inlet; only when a dim was given):
The dim searched along: 1 finds the smallest in each row.

OUTPUTS:

min index:
The position(s), counting from 0. If the smallest value appears more than once, the
first.

""" + DEVICE + """

READING THE DEMO:
Top: in 5 3 8 1 9 the smallest, 1, is at position 3. Bottom: with dim 1, each row's
smallest: 3 in the first row (position 1), 0 in the second (position 0).

RELATED:
t.argmax finds the largest. t.min gives the smallest value itself (and, with a dim, its
position too). t.topk with largest off finds the several smallest. np.argmin is the numpy
version.
""",
metro() + [
    C('c1', 'five values', 180, 53),
    N('m1', 'message', 180, 92, W('5 3 8 1 9'), 42, MSG('5 3 8 1 9')),
    N('a1', 't.argmin', 400, 92, 150, 70),
    N('l1', 'list', 580, 92, 80, 42, LIST),
    C('c2', 'two rows', 180, 190),
    N('m2', 'message', 180, 229, W('3 1 2 0 5 9'), 42, MSG('3 1 2 0 5 9')),
    N('r2', 't.reshape 2 3', 400, 229, 150, 70),
    C('c3', 'dim 1: in each row', 660, 190),
    N('a2', 't.argmin 1', 660, 229, 150, 110),
    N('l2', 'list', 660, 360, 80, 42, LIST),
],
[('mt', '', 'm1', ''), ('mt', '', 'm2', ''), ('m1', 'message out', 'a1', 'tensor in'),
 ('a1', 'min index', 'l1', ''), ('m2', 'message out', 'r2', 'tensor in'), ('r2', 'output', 'a2', 'tensor in'),
 ('a2', 'min index', 'l2', '')],
['t.argmax', 't.min', 't.topk', 'np.argmin'], 400, 900)


# ----------------------------------------------------------------- t.quantile
page('t.quantile', 't.quantile - the value below which a given share falls',
"""
Gives the value below which a given share of the values falls. q 0.5 is the median:
half the values are below it. q 0.9 is the value only a tenth exceed - a robust
"loud" or "fast" level that a few outliers cannot drag up the way they drag the
maximum. q 0.1 and 0.9 together give a range that ignores the extremes.

Use it to set thresholds from the data itself, to normalise a feature to its typical
range (from the 5% to the 95% quantile), or to summarise a distribution.

SYNTAX:
t.quantile <q> <dim>
Defaults: q 0.5, the whole tensor. q is a share from 0 to 1 (np.percentile uses 0 to
100). Giving a dim adds dim and keep_dims boxes.

INPUTS and PARAMETERS:

tensor in:
The values. Every input sends.

q (float box, also an inlet):
The share, 0 to 1. A list - 0.1 0.5 0.9 - asks for several at once.

dim, keep_dims (also inlets; only when a dim was given):
Each slice along dim gets its own quantile(s).

interpolation:
An option, under the node: how a q that falls between two values is answered - linear
(the default) blends them, lower and higher pick one, nearest the closer, midpoint
their mean.

ignore nan:
An option, under the node: skip nan values (otherwise any nan makes the answer nan).

OUTPUTS:

output:
The quantile(s).

""" + DEVICE + """

READING THE DEMO:
The values 1 to 9. A quarter of the way up is 3, half way 5, three quarters 7.

RELATED:
np.percentile is the numpy version (q in 0 to 100). t.median is q 0.5. t.sort puts
every value in order. t.histc counts how many fall in each range.
""",
metro() + [
    C('c1', 'the values 1 to 9', 180, 53),
    N('rn', 'message', 180, 92, W('1 2 3 4 5 6 7 8 9'), 42, MSG('1 2 3 4 5 6 7 8 9')),
    C('c2', 'once at load', 28, 220),
    N('lb', 'load_bang', 28, 259, 88, 46),
    C('c3', 'three shares', 180, 220),
    N('mq', 'message', 180, 259, W('0.25 0.5 0.75'), 42, MSG('0.25 0.5 0.75')),
    C('c4', 'the value each share falls below', 460, 53),
    N('qn', 't.quantile', 460, 92, 180, 130),
    N('l1', 'list', 680, 92, 160, 42, LIST),
],
[('lb', 'out', 'mq', ''), ('mq', 'message out', 'qn', 'q'),
 ('mt', '', 'rn', ''), ('rn', 'message out', 'qn', 'tensor in'), ('qn', 'output', 'l1', '')],
['np.percentile', 't.median', 't.sort', 't.histc'], 350, 1250)


# --------------------------------------------------------------------- t.mode
page('t.mode', 't.mode - the most common value',
"""
Gives the value that appears most often, and where it appears. Use it to smooth a
stream of labels: the most common gesture class over the last 15 frames is a steadier
answer than any one frame's guess. Also for the most common note, colour index or
state.

SYNTAX:
t.mode <dim>
Without a dim, along the last dim (each row of a matrix gets its own). With one, dim and
keep_dims boxes appear.

INPUTS and PARAMETERS:

tensor in:
The values. Every input sends.

dim, keep_dims (also inlets; only when a dim was given):
The dim counted along, and whether to keep it as size 1.

OUTPUTS:

mode:
The most common value. Of several equally common, the smallest.

indices:
A position where it appears. Sent first.

""" + DEVICE + """ On the Mac's GPU (mps), torch has no mode: it is done on the CPU and
the answer moved back.

READING THE DEMO:
3 1 3 2 3 1: 3 appears three times, more than any other.

RELATED:
t.unique counts every value. t.bincount counts whole numbers. t.median is the middle
value, not the most common one. t.rolling_buffer collects the last frames to vote over.
""",
metro() + [
    C('c1', 'six labels', 180, 53),
    N('m1', 'message', 180, 92, W('3 1 3 2 3 1'), 42, MSG('3 1 3 2 3 1')),
    C('c2', 'the most common, and where', 420, 53),
    N('md', 't.mode', 420, 92, 150, 70),
    N('l1', 'list', 420, 190, 80, 42, LIST),
    N('l2', 'list', 560, 190, 80, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'md', 'tensor in'), ('md', 'mode', 'l1', ''),
 ('md', 'indices', 'l2', '')],
['t.unique', 't.bincount', 't.median', 't.rolling_buffer'], 290, 900)


# --------------------------------------------------------------------- t.sort
page('t.sort', 't.sort - put values in order',
"""
Puts values in order, and says where each came from. The positions are as useful as the
values: sorting joint speeds gives the joints from slowest to fastest; sorting distances
gives the neighbours from nearest to farthest.

SYNTAX:
t.sort <dim>
Without a dim, along the last (each row of a matrix is sorted on its own). With one, a
dim box appears.

INPUTS and PARAMETERS:

tensor in:
The values. Every input sends.

dim (int box, also an inlet; only when a dim was given):
The dim sorted along.

descending (checkbox, also an inlet):
Largest first.

stable (checkbox, also an inlet):
Equal values keep their original order.

OUTPUTS:

sorted:
The values in order.

indices:
For each place in the order, the position the value came from. Sent first.

""" + DEVICE + """

READING THE DEMO:
5 3 8 1 9 sorted is 1 3 5 8 9; the 1 was at position 3, the 3 at position 1, and so on.

RELATED:
t.argsort gives only the positions. t.topk gives just the largest (or smallest) few,
quicker than sorting everything. np.sort is the numpy version. t.take_along_dim puts
other data in the same order.
""",
metro() + [
    C('c1', 'five values', 180, 53),
    N('m1', 'message', 180, 92, W('5 3 8 1 9'), 42, MSG('5 3 8 1 9')),
    C('c2', 'in order', 400, 53),
    N('st', 't.sort', 400, 92, 150, 120),
    C('c3', 'the values', 600, 53),
    N('l1', 'list', 600, 92, 150, 42, LIST),
    C('c4', 'where each came from', 600, 160),
    N('l2', 'list', 600, 199, 150, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'st', 'tensor in'), ('st', 'sorted', 'l1', ''),
 ('st', 'indices', 'l2', '')],
['t.argsort', 't.topk', 'np.sort', 't.take_along_dim'], 300, 950)


# --------------------------------------------------------------------- t.topk
page('t.topk', 't.topk - the k largest values, and where they are',
"""
Gives the k largest values (or smallest), largest first, and their positions - without
sorting everything. The two fastest joints, the five loudest frequency bins, the three
nearest neighbours (smallest distances), the most likely classes of a classifier.

SYNTAX:
t.topk <k> <dim>
Defaults: k 3, along the last dim. Giving a dim adds a dim box.

INPUTS and PARAMETERS:

tensor in:
The values. Every input sends.

k (int box, also an inlet):
How many. Asking for more than there are gives all of them.

dim (int box, also an inlet; only when a dim was given):
The dim searched along: each slice gets its own k.

largest (checkbox, also an inlet):
On (the default), the largest; off, the smallest.

OUTPUTS:

values:
The k values, best first.

indices:
Their positions. Sent first.

""" + DEVICE + """

READING THE DEMO:
Five joint speeds. The two fastest are 2.0 (joint 4) and 1.5 (joint 1).

RELATED:
t.sort orders all of them. t.argmax finds just the largest. np.top_k is the numpy
version. t.index_select picks the rows the positions name.
""",
metro() + [
    C('c1', 'five joint speeds', 180, 53),
    N('m1', 'message', 180, 92, W('0.2 1.5 0.1 0.9 2.0'), 42, MSG('0.2 1.5 0.1 0.9 2.0')),
    C('c2', 'the two fastest', 480, 53),
    N('tk', 't.topk 2', 480, 92, 150, 120),
    C('c3', 'their speeds', 680, 53),
    N('l1', 'list', 680, 92, 120, 42, LIST),
    C('c4', 'which joints', 680, 160),
    N('l2', 'list', 680, 199, 120, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'tk', 'tensor in'), ('tk', 'values', 'l1', ''),
 ('tk', 'indices', 'l2', '')],
['t.sort', 't.argmax', 'np.top_k', 't.index_select'], 300, 1000)


# -------------------------------------------------------------------- t.where
page('t.where', 't.where - choose, value by value, between two sources',
"""
For each element, takes the value from one source where a condition holds and from
another where it does not. Replace values below a threshold with 0, keep a sensor's
reading only where it is confident, fall back to a previous frame where tracking was
lost, switch between two signals element by element.

SYNTAX:
t.where <if true> <if false>
Numbers given here are used until a tensor or number arrives at that inlet: t.where 1 0
turns a condition into 1s and 0s.

INPUTS and PARAMETERS:

condition:
True and false (or non-zero and zero) for each element: usually from t.gt, t.lt,
t.isnan... Every condition computes and sends - the only inlet that makes the node
send.

if true:
The value used where the condition holds, held: a tensor the condition's shape (or
that broadcasts to it), or a single number for every place.

if false:
The value used elsewhere, held, likewise.

OUTPUTS:

selected:
The condition's shape, each element from if true or if false.

""" + DEVICE + """

READING THE DEMO:
-2 5 -1 3 compared with 0 gives the condition. Where it is true the value itself is
kept (if true is the values); elsewhere 0 (if false) - every negative value becomes 0.

RELATED:
np.where is the numpy version. t.gt and the other comparisons make conditions. t.clamp
limits values to a range. t.masked_select keeps only the values where a condition
holds.
""",
metro() + [
    C('c1', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 214, 88, 46),
    N('m0', 'message', 180, 214, W('0'), 42, MSG('0')),
    C('c2', 'four values', 180, 53),
    N('m1', 'message', 180, 92, W('-2 5 -1 3'), 42, MSG('-2 5 -1 3')),
    C('c3', 'above 0?', 400, 53),
    N('gt', 't.gt', 400, 92, 150, 90),
    C('c4', 'yes: the value; no: 0', 600, 53),
    N('wh', 't.where', 600, 92, 150, 110),
    N('l1', 'list', 600, 230, 160, 42, LIST),
],
# 0 reaches t.gt and t.where's 'if false' at load; each tick the values reach 'if true'
# (cold) before the comparison triggers t.where
[('lb', 'out', 'm0', ''), ('m0', 'message out', 'gt', 'tensor b in'), ('m0', 'message out', 'wh', 'if false'),
 ('mt', '', 'm1', ''), ('m1', 'message out', 'wh', 'if true'), ('m1', 'message out', 'gt', 'tensor a in'),
 ('gt', 'tensor result', 'wh', 'condition'), ('wh', 'selected', 'l1', '')],
['np.where', 't.gt', 't.clamp', 't.masked_select'], 330, 1100)


# ------------------------------------------------------------------- t.unique
page('t.unique', 't.unique - the different values, and how many of each',
"""
Lists the different values that occur, in order, with how many times each occurs and,
for every element, which of them it is. Which labels were seen, how often each gesture
class appeared, which performers (by id) are present.

SYNTAX:
t.unique

INPUTS and PARAMETERS:

tensor in:
The values. Every input sends.

OUTPUTS:

values:
The different values, smallest first.

counts:
How many times each occurs. Sent second.

inverse:
For every element of the input, the position in values of its value - the input
rewritten as numbers 0, 1, 2 ... Sent first.

""" + DEVICE + """

READING THE DEMO:
3 1 3 2 1 3 has three different values, 1 2 3, occurring 2, 1 and 3 times. The input
rewritten: 3 is value 2, 1 is value 0, and so on.

RELATED:
np.unique is the numpy version. t.mode gives the most common value. t.bincount counts
whole numbers 0, 1, 2 ...
""",
metro() + [
    C('c1', 'six values', 180, 53),
    N('m1', 'message', 180, 92, W('3 1 3 2 1 3'), 42, MSG('3 1 3 2 1 3')),
    N('un', 't.unique', 420, 92, 150, 120),
    C('c2', 'the different values', 620, 53),
    N('l1', 'list', 620, 92, 150, 42, LIST),
    C('c3', 'how many of each', 620, 160),
    N('l2', 'list', 620, 199, 150, 42, LIST),
    C('c4', 'which one each element is', 620, 267),
    N('l3', 'list', 620, 306, 200, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'un', 'tensor in'), ('un', 'values', 'l1', ''),
 ('un', 'counts', 'l2', ''), ('un', 'inverse', 'l3', '')],
['np.unique', 't.mode', 't.bincount'], 400, 950)


# ----------------------------------------------- t.isnan / t.isfinite / t.isinf
BAD = '1 nan 3 inf -inf 0'


def test_page(stem, title, what, reading, also):
    page(stem, stem + ' - ' + title,
"""
""" + what + """

nan ("not a number") comes from 0 / 0, from sqrt or log of a negative number, from a
sensor that lost track. Infinity comes from dividing by zero or from exp of something
large. Either spreads through everything computed from it - a mean, a filter's memory,
a plot's range - so it is worth finding and replacing early (t.nan_to_num, t.where).

SYNTAX:
""" + stem + """

INPUTS and PARAMETERS:

tensor in:
The values. Every input sends.

OUTPUTS:

mask:
True or false for each element, the input's shape.

count:
How many are true. Sent first.

""" + DEVICE + """

READING THE DEMO:
""" + reading + """

RELATED:
t.isnan, t.isfinite and t.isinf are the three tests. t.nan_to_num replaces nan and
infinity; t.where replaces them with anything. np.isnan and np.isfinite are the numpy
versions.
""",
metro() + [
    C('c1', 'six values', 180, 53),
    N('m1', 'message', 180, 92, W(BAD), 42, MSG(BAD)),
    N('ts', stem, 460, 92, 150, 90),
    C('c3', 'how many', 650, 53),
    N('l2', 'list', 650, 92, 80, 42, LIST),
    C('c2', 'which', 460, 210),
    N('l1', 'list', 460, 249, 380, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'ts', 'tensor in'), ('ts', 'mask', 'l1', ''),
 ('ts', 'count', 'l2', '')],
also, 350, 950)


test_page('t.isnan', 'where the values are nan',
          'Marks the elements that are nan, and counts them.',
          'Of 1 nan 3 inf -inf 0, one is nan. Infinity is not nan: it is a number, only too large.',
          ['t.isfinite', 't.isinf', 't.nan_to_num', 'np.isnan'])
test_page('t.isfinite', 'where the values are ordinary numbers',
          'Marks the elements that are ordinary, finite numbers - neither nan nor infinite - and counts\n'
          'them. Use the mask to keep only the good values (t.masked_select) or the count to notice\n'
          'that something has gone wrong.',
          'Of 1 nan 3 inf -inf 0, three are ordinary numbers: 1, 3 and 0.',
          ['t.isnan', 't.isinf', 't.nan_to_num', 'np.isfinite'])
test_page('t.isinf', 'where the values are infinite',
          'Marks the elements that are infinite - plus or minus - and counts them.',
          'Of 1 nan 3 inf -inf 0, two are infinite: inf and -inf. nan is not infinite.',
          ['t.isnan', 't.isfinite', 't.nan_to_num', 'np.isfinite'])


# --------------------------------------------------------------- t.nan_to_num
page('t.nan_to_num', 't.nan_to_num - replace nan and infinity with numbers',
"""
Replaces every nan with a number you choose (0 by default), and plus and minus
infinity with numbers you choose, leaving the rest alone - so a lost sensor reading or
a division by zero does not spread through everything downstream.

SYNTAX:
t.nan_to_num <nan value>
Default: 0.

INPUTS and PARAMETERS:

tensor in:
The values. Every input sends. Whole numbers cannot be nan or infinite, so they pass
through unchanged.

nan (float box, also an inlet):
What nan becomes.

posinf, neginf (float boxes, also inlets):
What plus and minus infinity become. Both 0 by default.

OUTPUTS:

out:
The values with the replacements made.

""" + DEVICE + """

READING THE DEMO:
1 nan 3 inf -inf 0 with nan set to -1: the nan becomes -1, and both infinities 0.

RELATED:
t.isnan and t.isfinite find the bad values. t.where replaces them with values from
another tensor (the last good frame). t.nanmean averages while skipping nan. np.nan_to_num
is the numpy version.
""",
metro() + [
    C('c1', 'six values', 180, 53),
    N('m1', 'message', 180, 92, W(BAD), 42, MSG(BAD)),
    C('c2', 'nan becomes -1', 460, 53),
    N('nn', 't.nan_to_num -1', 460, 92, 190, 130),
    N('l1', 'list', 460, 250, 300, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'nn', 'tensor in'), ('nn', 'out', 'l1', '')],
['t.isnan', 't.where', 't.nanmean', 'np.nan_to_num'], 350, 900)


# ---------------------------------------------------------------------- t.pad
page('t.pad', 't.pad - add values at the ends',
"""
Adds values before and after the existing ones along a dim: zeros (or any constant),
copies of the end values, a mirror image, or a wrap-around from the other end.

Use it before filtering or windowing so the result keeps the input's length and the
ends are not lost, to bring sequences of different lengths to the same length so they
can be stacked, or to give an image a border.

SYNTAX:
t.pad <before> <after> <mode>
Defaults: 1 and 1, constant. One number pads both ends by it. The mode can also be
given here: constant, replicate, reflect, circular (or numpy's names edge and wrap).

INPUTS and PARAMETERS:

tensor in:
The values. Every input sends.

before, after (int boxes, also inlets):
How many values to add at the start and at the end.

value (float box, also an inlet):
What constant padding adds.

dim (int box, also an inlet):
The dim padded along; -1, the last, by default.

mode:
An option, under the node. constant adds the value; replicate repeats the end values;
reflect mirrors the values next to the ends (1 2 3 -> 3 2 1 2 3); circular continues
from the other end, as if the values went round in a loop.

OUTPUTS:

padded:
The longer tensor.

""" + DEVICE + """

READING THE DEMO:
1 2 3 4 padded by 2 at each end: with zeros (top), and reflected (bottom), which
continues the shape smoothly and is the usual choice before a filter.

RELATED:
np.pad is the numpy version. t.cat joins tensors end to end. t.unfold and t.convolve
are often padded first.
""",
metro() + [
    C('c1', 'four values', 180, 53),
    N('m1', 'message', 180, 92, W('1 2 3 4'), 42, MSG('1 2 3 4')),
    C('c2', 'two zeros at each end', 400, 53),
    N('p1', 't.pad 2', 400, 92, 150, 170),
    N('l1', 'list', 400, 285, 360, 42, LIST),
    C('c3', 'two reflected at each end', 400, 360),
    N('p2', 't.pad 2 reflect', 400, 399, 170, 170),
    N('l2', 'list', 400, 592, 360, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'p1', 'tensor in'), ('m1', 'message out', 'p2', 'tensor in'),
 ('p1', 'padded', 'l1', ''), ('p2', 'padded', 'l2', '')],
['np.pad', 't.cat', 't.unfold', 't.convolve'], 680, 1150)


# ------------------------------------------------------------------- t.unfold
page('t.unfold', 't.unfold - every window of a sequence, at once',
"""
Cuts a sequence into windows of a given length - overlapping or not - and gives them
all at once, one window per row. Every 30-frame window of a movement, ready to be
averaged, measured or fed to a model in one step instead of one window at a time.

SYNTAX:
t.unfold <window> <step>
Defaults: 4, 1.

INPUTS and PARAMETERS:

tensor in:
The sequence. Every input sends.

window (int box, also an inlet):
How many values in each window.

step (int box, also an inlet):
How far each window starts after the last: 1 for every possible window; the window
length for windows side by side with no overlap.

dim (int box, also an inlet):
The dim the windows run along; -1, the last, by default. For frames x joints, use 0:
the result is windows x joints x window.

OUTPUTS:

windows:
One more dim than the input: the windows, then the values in each. A copy, so changing
it does not change the input.

""" + DEVICE + """

READING THE DEMO:
0 1 2 3 4 5 in windows of 3: every window, each starting one later (top); with step 3,
side by side (bottom).

RELATED:
np.sliding_window_view is the numpy version. t.rolling_buffer keeps the latest window of
a stream. t.mean with a dim then averages each window. t.stft analyses each window's
frequencies.
""",
metro() + [
    C('c1', 'six values', 180, 53),
    N('m1', 'message', 180, 92, W('0 1 2 3 4 5'), 42, MSG('0 1 2 3 4 5')),
    C('c2', 'windows of 3, step 1', 420, 53),
    N('u1', 't.unfold 3', 420, 92, 150, 150),
    N('l1', 'list', 180, 265, 560, 42, LIST),
    C('c3', 'windows of 3, step 3', 420, 330),
    N('u2', 't.unfold 3 3', 420, 369, 150, 150),
    N('l2', 'list', 180, 540, 300, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'u1', 'tensor in'), ('m1', 'message out', 'u2', 'tensor in'),
 ('u1', 'windows', 'l1', ''), ('u2', 'windows', 'l2', '')],
['np.sliding_window_view', 't.rolling_buffer', 't.mean', 't.stft'], 640, 1150)


# -------------------------------------------------------------- t.interpolate
page('t.interpolate', 't.interpolate - resample to a new length, or resize',
"""
Stretches or squeezes a tensor to a new size, filling in the values between the
original ones. With one size, the first dim is resampled: a movement of 47 frames
becomes 100, so takes of different lengths can be compared or averaged frame by frame;
a short curve drawn by hand becomes a smooth one. With two sizes, the first two dims are
resized: an image (height x width x colours) at a new resolution.

Any dims after the resized ones are carried along unchanged: 47 frames x 72 values
resampled to 100 gives 100 x 72.

SYNTAX:
t.interpolate <size> <size>
Default: 100.

INPUTS and PARAMETERS:

tensor in:
The tensor. Every input sends.

size (text box, also an inlet):
The new size: one number, or two (or three) for that many leading dims.

mode:
An option, under the node. linear (the default) draws straight lines between the
values; cubic draws smooth curves through them; nearest repeats the closest value
(steps, and keeps labels as labels); area averages (best for shrinking).

align corners:
An option, under the node. On (the default), the first and last values stay exactly
first and last; off, each value is treated as the centre of a cell, as images usually
are.

OUTPUTS:

resized:
The tensor at its new size.

""" + DEVICE + """

READING THE DEMO:
Eight random values, new every second, resampled to 64. Linear (left) joins them with
straight lines; cubic (right) passes smoothly through the same points.

RELATED:
np.interp interpolates at positions you choose. scipy.cubic_spline gives a curve through
points. scipy.resample_poly changes the sample rate of a signal without blurring its
frequencies.
""",
metro(label='every second', props=METRO_1S, init='metro 1000') + [
    C('c1', 'eight random values', 180, 53),
    N('rn', 'np.rand 8', 180, 92, 150, 110),
    C('c2', 'linear', 28, 240),
    N('i1', 't.interpolate 64', 28, 279, 200, 110),
    N('p1', 'plot', 28, 420, 380, 160, LINE(64, 0.0, 1.0)),
    C('c3', 'cubic', 440, 240),
    N('i2', 't.interpolate 64', 440, 279, 200, 110, {'mode': 'cubic'}),
    N('p2', 'plot', 440, 420, 380, 160, LINE(64, 0.0, 1.0)),
],
[('mt', '', 'rn', ''), ('rn', 'random array', 'i1', 'tensor in'), ('rn', 'random array', 'i2', 'tensor in'),
 ('i1', 'resized', 'p1', 'y'), ('i2', 'resized', 'p2', 'y')],
['np.interp', 'scipy.cubic_spline', 'scipy.resample_poly', 't.reshape'], 620, 1300)


# --------------------------------------------------------------------- t.stft
# Inside make_chirp_4096: 4096 samples of a wave whose angle grows with the square of
# time (sin works in degrees), so its frequency rises steadily, to just under the highest a
# signal can carry: 2 x 0.016 x 4096 = 131 degrees a sample at the end.
CHIRP_SUB = {
    'name': 'make_chirp_4096',
    'host': 'ch',
    'demo': [
        S('mt', 'metro 1000', 40, 40, 129, 70, METRO_1S),
        S('ct', 'comment every second', 190, 55, 200, 30),
        S('ar', 'np.arange 4096', 40, 130, 160, 110),
        S('ca', 'comment 4096 samples', 220, 150, 200, 30),
        S('sq', '^ 2', 40, 270, 129, 70),
        S('m1', '* 0.016', 40, 360, 129, 70),
        S('s1', 'sin', 40, 450, 129, 70),
        S('c1', 'comment the angle grows with the square of the time: ever faster', 190, 380, 520, 30),
        S('out', 'out chirp', 40, 550, 150, 80),
    ],
    'links': [('mt', '', 'ar', ''), ('ar', 'arange out', 'sq', 'in'), ('sq', 'result', 'm1', 'in'),
              ('m1', 'result', 's1', 'in'), ('s1', 'out', 'out', 'chirp')],
}

page('t.stft', 't.stft - which frequencies, at which moments',
"""
The short-time Fourier transform: cuts a signal into short overlapping windows and finds
the frequencies in each. The result is a picture with frequency one way and time the
other, bright where there is energy - a spectrogram - showing how a sound or a movement
changes: a pitch rising, a rhythm speeding up, a tremor that comes and goes.

It runs on the GPU when the signal is there, so long audio buffers are cheap.

SYNTAX:
t.stft <window> <hop>
Defaults: 256 samples, hop 0 (a quarter of the window).

INPUTS and PARAMETERS:

signal:
A line of samples, or several (channels x samples), each analysed separately. Every
input sends.

sample rate (float box, also an inlet):
Samples per second: only used to label the frequencies output.

window (int box, also an inlet):
Samples per window. Longer windows tell frequencies apart more finely but blur when
things happened; the result has window / 2 + 1 frequency bins.

hop (int box, also an inlet):
Samples from the start of one window to the next; 0 means a quarter of the window.

window shape:
An option, under the node: hann (the default), hamming, blackman, or rectangular (no
shaping, which smears frequencies).

output:
An option, under the node: magnitude (the default), power (magnitude squared), decibels
(10 log10 of the power), or complex - the full result, which t.istft can turn back into
a signal.

center:
An option, under the node. On (the default), each window is centred on its moment, with
the signal mirrored at the ends.

OUTPUTS:

spectrogram:
Frequency bins x windows (channels x bins x windows for several channels): low
frequencies in the first row, so a heat_map shows them at the top.

frequencies:
The frequency of each bin, in cycles per second at the sample rate. Sent first.

""" + DEVICE + """

READING THE DEMO:
A chirp - a wave whose pitch rises steadily (made in the make_chirp_4096 subpatcher;
double-click it to look inside) - as decibels. Time runs left to right and frequency top
to bottom: a bright line running down from the top left as the pitch rises.

RELATED:
t.istft turns a complex spectrogram back into a signal. scipy.spectrogram is the scipy
version. t.rfft gives the spectrum of the whole signal at once. t.unfold cuts the windows
without transforming them.
""",
[
    C('c1', 'a rising pitch', 28, 53),
    N('ch', 'p make_chirp_4096', 28, 92, 200, 70),
    C('c2', 'windows of 256, hop 64', 270, 53),
    N('st', 't.stft 256 64', 270, 92, 200, 210, {'output': 'decibels'}),
    C('c3', 'frequency down, time across', 510, 53),
    N('hm', 'heat_map', 510, 92, 268, 300, HM(250, 260, 1, -30.0, 40.0, '', 'plasma')),
],
[('ch', 'chirp', 'st', 'signal'), ('st', 'spectrogram', 'hm', 'y')],
['t.istft', 'scipy.spectrogram', 't.rfft', 't.unfold'], 440, 1650, subpatch=CHIRP_SUB)


# -------------------------------------------------------------------- t.istft
# Inside make_wave: 512 samples of a slow swell (3 cycles) with a wave 0.4 tall on it
# (20 cycles); sin works in degrees.
WAVE_SUB = {
    'name': 'make_wave',
    'host': 'wv',
    'demo': [
        S('mt', 'metro 1000', 40, 40, 129, 70, METRO_1S),
        S('ct', 'comment every second', 190, 55, 200, 30),
        S('ar', 'np.arange 512', 40, 130, 160, 110),
        S('m1', '* 2.1', 40, 270, 129, 70),
        S('s1', 'sin', 40, 360, 129, 70),
        S('m2', '* 14', 330, 270, 129, 70),
        S('s2', 'sin', 330, 360, 129, 70),
        S('m3', '* 0.4', 330, 450, 129, 70),
        S('c2', 'comment a slow swell, and a faster wave 0.4 tall', 480, 300, 400, 30),
        S('ad', '+', 40, 550, 129, 70),
        S('out', 'out wave', 40, 650, 150, 80),
    ],
    # the faster wave is linked first, so it reaches the + operand before the slow one
    # arrives to trigger the sum (an outlet sends in the order its links were made)
    'links': [('mt', '', 'ar', ''), ('ar', 'arange out', 'm2', 'in'), ('ar', 'arange out', 'm1', 'in'),
              ('m2', 'result', 's2', 'in'), ('s2', 'out', 'm3', 'in'), ('m3', 'result', 'ad', 'operand', 0, 1),
              ('m1', 'result', 's1', 'in'), ('s1', 'out', 'ad', 'in'), ('ad', 'result', 'out', 'wave')],
}

page('t.istft', 't.istft - turn a spectrogram back into a signal',
"""
The inverse of t.stft: rebuilds a signal from its complex spectrogram. On its own it
gives back the original; its use is to change the spectrogram first - silence some
frequencies, keep only others, scale them - and hear or see the result: filtering that
can change over time, noise removal, separating a tremor from a movement.

The spectrogram must be the complex output of t.stft (output set to complex), made
with the same hop, window shape and center setting given here. The window size is read
from the spectrogram itself.

SYNTAX:
t.istft <hop>
Default: hop 0 (a quarter of the window, as t.stft).

INPUTS and PARAMETERS:

spectrogram:
A complex spectrogram from t.stft: frequency bins x windows, or channels x bins x
windows. Every input sends.

length (int box, also an inlet):
How many samples to make; 0 (the default) gives as many as the windows cover. Set it to
the original length to get exactly that back.

hop (int box, also an inlet):
The hop t.stft used.

window shape, center:
Options, under the node: as set on t.stft.

OUTPUTS:

signal:
The rebuilt samples.

""" + DEVICE + """

READING THE DEMO:
A wave (made in the make_wave subpatcher; double-click it to look inside) taken apart
by t.stft and put back together by t.istft: the two plots are the same.

RELATED:
t.stft makes the spectrogram. t.irfft inverts the spectrum of a whole signal. scipy.filtfilt
and the Butterworth filters filter without a spectrogram.
""",
[
    C('c1', 'a wave', 28, 53),
    N('wv', 'p make_wave', 28, 92, 160, 70),
    C('c2', 'taken apart', 240, 53),
    N('st', 't.stft 64 16', 240, 92, 180, 210, {'output': 'complex'}),
    C('c3', 'put back together', 460, 53),
    N('is', 't.istft 16', 460, 92, 160, 150, {'length': 512}),
    C('c4', 'length 512, as the wave', 640, 53),
    C('c5', 'the wave', 28, 320),
    N('p1', 'plot', 28, 359, 380, 160, LINE(512, -1.5, 1.5)),
    C('c6', 'rebuilt', 440, 320),
    N('p2', 'plot', 440, 359, 380, 160, LINE(512, -1.5, 1.5)),
],
[('wv', 'wave', 'p1', 'y'), ('wv', 'wave', 'st', 'signal'), ('st', 'spectrogram', 'is', 'spectrogram'),
 ('is', 'signal', 'p2', 'y')],
['t.stft', 't.irfft', 'scipy.filtfilt'], 560, 1250, subpatch=WAVE_SUB)


# ------------------------------------------------------- t.convolve / correlate
page('t.convolve', 't.convolve - smooth, blur or shape with a kernel',
"""
Slides a kernel - a short list of weights - along the values, and at each position adds
up the values under it, each times its weight. A kernel of equal weights averages
(smoothing); a bell-shaped one blurs gently; 1 -1 gives the change from one value to
the next. Where the input has a single spike, a copy of the kernel appears: convolving
places the kernel's shape at every impulse.

This is true convolution, as np.convolve does it (the kernel is applied reversed). For
symmetric kernels that makes no difference; torch's own conv1d correlates, as
t.correlate does.

SYNTAX:
t.convolve <kernel values ...> <mode>
The kernel can be given here: t.convolve 1 2 1. The mode can also be given: same, full
or valid.

INPUTS and PARAMETERS:

tensor in:
The values. Every input sends.

kernel:
The weights, held.

dim (int box, also an inlet):
The dim the kernel slides along; -1, the last, by default. For frames x joints, use 0.

mode:
An option, under the node. same (the default) keeps the input's length; full includes
every position where the kernel overlaps the input at all (length + kernel - 1); valid
only the positions where it overlaps completely.

OUTPUTS:

convolved:
The result.

""" + DEVICE + """

READING THE DEMO:
Two spikes, convolved with the kernel 1 2 1: each spike becomes a copy of the kernel.

RELATED:
np.convolve is the numpy version. t.correlate finds where a pattern occurs. scipy.filtfilt
and the Butterworth filters smooth by frequency. scipy.gaussian_filter blurs.
""",
metro() + [
    C('c1', 'two spikes', 180, 53),
    N('m1', 'message', 180, 92, W('0 0 1 0 0 0 1 0 0'), 42, MSG('0 0 1 0 0 0 1 0 0')),
    C('c2', 'the kernel 1 2 1', 480, 53),
    N('cv', 't.convolve 1 2 1', 480, 92, 190, 110),
    C('c3', 'a copy of the kernel at each spike', 480, 230),
    N('l1', 'list', 480, 269, 300, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'cv', 'tensor in'), ('cv', 'convolved', 'l1', '')],
['np.convolve', 't.correlate', 'scipy.filtfilt', 'scipy.gaussian_filter'], 370, 1300)


page('t.correlate', 't.correlate - where a pattern occurs',
"""
Slides a pattern along the values and, at each position, measures how well they match:
the values under the pattern, each times the pattern's value there, added up. The
result is highest where the values look most like the pattern - a template matcher for
a gesture, a beat, a shape in a curve.

It is t.convolve without reversing the pattern, and it is what torch's conv1d does.

SYNTAX:
t.correlate <pattern values ...> <mode>
The pattern can be given here: t.correlate 1 2 1. The mode can also be given: same, full
or valid.

INPUTS and PARAMETERS:

tensor in:
The values to search. Every input sends.

pattern:
The pattern, held.

dim (int box, also an inlet):
The dim searched along; -1, the last, by default.

mode:
An option, under the node: same (the default, the input's length), full, or valid, as
for t.convolve.

OUTPUTS:

correlated:
How well the pattern matches at each position (in same mode, centred on it).

""" + DEVICE + """

READING THE DEMO:
The pattern 1 2 1 occurs twice in the values, centred at positions 2 and 7: the match
is highest, 6, exactly there. t.argmax would find the best one.

RELATED:
np.correlate is the numpy version. scipy.correlate finds the delay between two whole
signals. t.convolve shapes with a kernel. scipy.find_peaks finds every high point of
the match.
""",
metro() + [
    C('c1', 'the pattern 1 2 1 appears twice', 180, 53),
    N('m1', 'message', 180, 92, W('0 1 2 1 0 0 1 2 1 0'), 42, MSG('0 1 2 1 0 0 1 2 1 0')),
    C('c2', 'search for 1 2 1', 500, 53),
    N('cr', 't.correlate 1 2 1', 500, 92, 190, 110),
    C('c3', 'highest where it matches', 500, 230),
    N('l1', 'list', 420, 269, 360, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'cr', 'tensor in'), ('cr', 'correlated', 'l1', '')],
['np.correlate', 'scipy.correlate', 't.convolve', 'scipy.find_peaks'], 370, 1150)


def main():
    args = sys.argv[1:]
    force = '--force' in args
    wanted = [a for a in args if not a.startswith('--')] or list(PAGES)
    for stem in wanted:
        p = PAGES[stem]
        path = os.path.join(HELP_DIR, stem + '_help.json')
        if os.path.exists(path) and not force:
            print('exists, skipped (use --force):', path)
            continue
        print(build(stem, p['title'], p['body'], p['demo'], p['links'], demo_width=880,
                    text_width=760, text_height=p['text_height'], subpatch=p['subpatch']))


if __name__ == '__main__':
    main()
