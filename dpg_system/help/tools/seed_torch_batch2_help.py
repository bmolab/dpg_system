"""Seed the help pages for the second batch of new torch nodes (2026-10-25): one page per node.

    python3 dpg_system/help/tools/seed_torch_batch2_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

t.cross, t.dot, t.outer, t.cov, t.linalg.eigh, t.gradient, t.trapezoid,
t.cumulative_trapezoid, t.meshgrid, t.logical_and, t.logical_or, t.logical_xor,
t.logical_not.

The demos follow the numpy pages for the same operations. Values that float32 does not
hold exactly (covariances, eigenvalues) are shown in heat maps, which print with a
format, rather than in lists, which print float32 as the double it really is. Held
(cold) inputs come from a load_bang; the metro drives the hot inlet, so the held value
is in place by the second tick at the latest. Not a make_* script: rebuild.py never
runs it, and it will not write over an existing page without --force. Treat the pages
as hand-made once written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from seed_numpy_more_help import LIST, MSG, HM, C, N, see_also


PAGES = {}


def page(stem, title, body, demo, links, also, also_y, text_height=820):
    PAGES[stem] = dict(title=title, body=body.strip('\n'), demo=demo + see_also(also_y, also),
                       links=links, text_height=text_height)


METRO = {'on': True, 'period': 200.0, 'units': 'milliseconds'}
METRO_HALF = {'on': True, 'period': 500.0, 'units': 'milliseconds'}

DEVICE = """The answer stays on the input's device: a tensor on the GPU is worked on there and
comes out there."""


def metro(y=53, label='five times a second', props=METRO, init='metro 200'):
    return [C('cm', label, 28, y), N('mt', init, 28, y + 39, 129, 70, props)]


def held(y=175, key='lb'):
    return [C('c_lb', 'once at load', 28, y), N(key, 'load_bang', 28, y + 35, 88, 46)]


def W(text):
    """A message or list wide enough for this text."""
    return max(60, int(len(text) * 13.5) + 30)


# -------------------------------------------------------------------- t.cross
CROSS_A = '1 0 0 0 1 0 0 0 1'
page('t.cross', 't.cross - the vector at right angles to two others',
f"""
Gives the cross product of 3-D vectors: a third vector at right angles to both.

Its direction is perpendicular to the plane the two inputs lie in, and its length is
the area of the parallelogram they span - largest when they are at right angles, 0
when they are parallel. Order matters: in 1 x in 2 points the opposite way from in 2 x
in 1 (the right-hand rule: fingers from in 1 towards in 2, thumb along the result).

Reach for it to find a surface normal from two edges of a triangle, the axis to turn
about to get from one direction to another, which way is "up" given a forward and a
sideways direction, or a torque from a lever arm and a force.

It works on whole stacks at once: 24 bone directions crossed with 24 others (both 24 x
3), or 24 directions crossed with one vector, give 24 answers in one step.
{DEVICE}

SYNTAX:
t.cross <dim>
Default: dim -1 - the vectors run along the last dim.

INPUTS and PARAMETERS:

in 1:
The first vector, or a stack of them (a list is turned into a tensor). Every input
calculates and sends - the only inlet that makes the node send.

in 2:
The second vector or stack, held: send it first. Its shape must match in 1, or
broadcast against it (one vector goes with every row).

dim (int box, also an inlet):
The dim the 3 numbers of each vector run along: -1 (the default) for N x 3 stacks, 0
for 3 x N.

OUTPUTS:

cross product:
The perpendicular vectors, the shape of the inputs together. Vectors that are not 3
long are refused, with a note in verbose mode.

READING THE DEMO:
Three vectors - x, y and z, one per row - each crossed with z (held). x times z is
(0 -1 0): minus y. y times z is (1 0 0): x. z times z is 0 0 0: parallel vectors have
no perpendicular.

RELATED:
t.dot measures how much two vectors agree; the cross product is what they do not
share. t.linalg.norm gives its length (the area). np.cross is the numpy version.
""",
metro() + held() + [
    C('c1', 'x, y and z: one per row', 180, 53),
    N('m1', 'message', 180, 92, W(CROSS_A), 42, MSG(CROSS_A)),
    N('rs', 't.reshape 3 3', 340, 160, 150, 70),
    C('c2', 'z, held', 180, 175),
    N('m2', 'message', 180, 210, 120, 42, MSG('0 0 1')),
    C('c3', 'each row crossed with z', 540, 53),
    N('cr', 't.cross', 540, 92, 150, 90),
    C('c4', 'rows: -y, x, nothing', 540, 230),
    N('hm', 'heat_map', 540, 265, 168, 150, HM(150, 100, 1, -1.0, 1.0, '%.0f', 'red-blue')),
],
[('mt', '', 'm1', ''), ('lb', 'out', 'm2', ''), ('m2', 'message out', 'cr', 'in 2'),
 ('m1', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'cr', 'in 1'),
 ('cr', 'cross product', 'hm', 'y')],
['t.dot', 't.linalg.norm', 'np.cross', 't.outer'], 470, 960)


# ---------------------------------------------------------------------- t.dot
DOT_A = '1 0 0 0 1 0 0.5 0.5 0 0 -1 0'
page('t.dot', 't.dot - how much vectors point the same way',
f"""
Gives the dot product of vectors: each pair of entries multiplied, and the products
added up. For two directions of length 1 it is the cosine of the angle between them:
1 when they point the same way, 0 at right angles, -1 when opposite.

For one direction of length 1, the dot product of any vector with it is that vector's
component along it - how far the vector reaches in that direction. How high each joint
is (dotted with up), how much of a hand's velocity is towards the audience, how much a
limb is aligned with another.

It works along the last dim, on whole stacks at once: 24 positions (24 x 3) dotted with
one direction give 24 answers; two stacks of 24 give 24 pairwise answers.
{DEVICE}

SYNTAX:
t.dot <dim>
Default: dim -1.

INPUTS and PARAMETERS:

in 1:
A vector, or a stack of them. Every input calculates and sends - the only inlet that
makes the node send.

in 2:
The other vector or stack, held: send it first. Its shape must match in 1, or
broadcast against it.

dim (int box, also an inlet):
The dim the vectors run along: -1 (the default) for N x 3 stacks, 0 for 3 x N.

OUTPUTS:

dot product:
One number per vector (or per pair).

READING THE DEMO:
Four directions, one per row, each dotted with up (0 1 0, held). Pointing sideways
gives 0, straight up 1, halfway between 0.5, straight down -1: each direction's
height.

RELATED:
t.cosine_similarity divides by both lengths, so only the angle counts. t.cross gives
what two vectors do not share. t.matmul and t.einsum do the same sums for whole
matrices.
""",
metro() + held() + [
    C('c1', 'four directions, one per row', 180, 53),
    N('m1', 'message', 180, 92, W(DOT_A), 42, MSG(DOT_A)),
    N('rs', 't.reshape 4 3', 380, 160, 150, 70),
    C('c2', 'up, held', 180, 175),
    N('m2', 'message', 180, 210, 120, 42, MSG('0 1 0')),
    C('c3', 'each one dotted with up', 580, 53),
    N('dt', 't.dot', 580, 92, 150, 90),
    C('c4', 'how high each points', 580, 230),
    N('ls', 'list', 580, 265, 200, 42, LIST),
],
[('mt', '', 'm1', ''), ('lb', 'out', 'm2', ''), ('m2', 'message out', 'dt', 'in 2'),
 ('m1', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'dt', 'in 1'),
 ('dt', 'dot product', 'ls', '')],
['t.cosine_similarity', 't.cross', 't.matmul', 't.einsum'], 420, 960)


# -------------------------------------------------------------------- t.outer
page('t.outer', 't.outer - every element times every element',
f"""
Multiplies every element of in 1 by every element of in 2, laying the products out as
a table: row i, column j holds in 1's i-th value times in 2's j-th.

Two curves make a surface this way: a horizontal profile times a vertical one gives a
2-D shape - a spotlight, a window, a weighting over a grid. It is also how one pattern
is given a different amplitude per channel (gains times a signal), and the building
block of covariance and of many matrix methods.

A stack of vectors (N x n and N x m) gives N tables at once.
{DEVICE}

SYNTAX:
t.outer

INPUTS and PARAMETERS:

in 1:
The values for the rows. Every input calculates and sends - the only inlet that makes
the node send.

in 2:
The values for the columns, held: send them first.

OUTPUTS:

outer product:
An n x m table (or N of them).

READING THE DEMO:
1 2 3 times 1 10 (held): a 3 x 2 table, 1 2 3 down the first column and 10 20 30 down
the second.

RELATED:
t.dot adds the products of matching entries into one number; t.outer keeps every
product. t.matmul multiplies whole matrices. np.outer is the numpy version.
""",
metro() + held() + [
    C('c1', 'rows', 180, 53),
    N('m1', 'message', 180, 92, 120, 42, MSG('1 2 3')),
    C('c2', 'columns, held', 180, 175),
    N('m2', 'message', 180, 210, 120, 42, MSG('1 10')),
    C('c3', 'every pair multiplied', 360, 53),
    N('ou', 't.outer', 360, 92, 150, 90),
    C('c4', '3 x 2', 560, 53),
    N('hm', 'heat_map', 560, 92, 168, 150, HM(150, 100, 1, 0.0, 30.0, '%.0f')),
],
[('mt', '', 'm1', ''), ('lb', 'out', 'm2', ''), ('m2', 'message out', 'ou', 'in 2'),
 ('m1', 'message out', 'ou', 'in 1'), ('ou', 'outer product', 'hm', 'y')],
['t.dot', 't.matmul', 'np.outer', 't.einsum'], 340, 860)


# ---------------------------------------------------------------------- t.cov
DATA3 = '1 2 3 2 4 1 3 5 4 4 8 1 5 10 5'
page('t.cov', 't.cov - how pairs of variables vary together',
f"""
Takes many observations of several variables - one row per moment, one column per
sensor, joint or feature - and gives, for every pair of variables, how much they vary
together: the covariance.

A positive covariance means the two tend to rise and fall together; negative, that one
rises as the other falls; near zero, that they are unrelated. On the diagonal each
variable is paired with itself, which is just its variance. The size of the numbers
depends on the units and the spread of the variables; t.corrcoef scales them to -1 to
1 for comparing.

The covariance matrix is the starting point of principal component analysis:
t.linalg.eigh of it gives the directions along which the data varies most - the
dominant axis of a gesture, the main modes of a set of joints moving together.
{DEVICE}

SYNTAX:
t.cov <variables: rows or columns>
Default: columns.

INPUTS and PARAMETERS:

in:
A 2-D tensor of observations, with at least two of each variable. Every input is
computed and sent.

variables:
An option, under the node. columns (the default here) takes each column as a variable
and each row as one observation - the shape a rolling buffer of samples has. rows
takes each row as a variable, as torch.cov itself does (and as t.corrcoef does).

correction:
An option, under the node: 1 (the default) divides by one fewer than the number of
observations, the usual estimate from a sample, as np.cov does. 0 divides by the
number of observations.

OUTPUTS:

covariance:
A square matrix with one row and column per variable.

READING THE DEMO:
Five observations of a, b and c. b is roughly twice a, so they vary together strongly:
covariance 5. c wanders on its own: its covariances with a and b (1 and 1.7) are small
for their spread. The diagonal holds each variable's variance: a 2.5, b 10.2 (twice the
size, four times the variance), c 3.2.

RELATED:
t.corrcoef gives the relations scaled to -1 to 1 (taking rows as variables). t.var is
one variable's variance. t.linalg.eigh of a covariance matrix gives its principal
directions. np.cov is the numpy version.
""",
metro() + [
    C('c1', 'five rows (observations) of three columns (variables)', 180, 53),
    N('m1', 'message', 180, 92, 330, 42, MSG(DATA3)),
    N('rs', 't.reshape 5 3', 540, 92, 150, 70),
    C('c2', 'as a 5 x 3 tensor: a b c', 28, 230),
    N('hd', 'heat_map', 28, 265, 168, 210, HM(150, 150, 1, 0.0, 10.0, '%.0f')),
    C('c3', 'every pair of columns', 260, 230),
    N('cv', 't.cov', 260, 265, 150, 70),
    C('c4', 'covariance of each pair', 450, 230),
    N('hc', 'heat_map', 450, 265, 228, 150, HM(210, 120, 1, -10.0, 10.0, '%.1f', 'red-blue')),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'rs', 'tensor in'),
 ('rs', 'output', 'hd', 'y'), ('rs', 'output', 'cv', 'in'), ('cv', 'covariance', 'hc', 'y')],
['t.corrcoef', 't.var', 't.linalg.eigh', 'np.cov'], 540, 960)


# -------------------------------------------------------------- t.linalg.eigh
POINTS = '0 0.125 1 0.875 2 2.25 3 2.75 4 4.125 5 5'
page('t.linalg.eigh', 't.linalg.eigh - the principal axes of a symmetric matrix',
f"""
Finds the eigenvalues and eigenvectors of a symmetric matrix: the directions it only
stretches, without turning, and how much it stretches along each.

Its most common use is principal component analysis. Feed it the covariance matrix of
some data (t.cov) and the eigenvectors are the data's principal axes - the direction
it varies most along, then the next most at right angles to that, and so on - and the
eigenvalues say how much variance lies along each. The first axis of a cloud of hand
positions is the main direction of the gesture; the first few axes of many joints are
the main ways the body moves together.

Unlike t.linalg.eig, which takes any square matrix, it assumes symmetry, and so its
answers are always real numbers and its axes always at right angles. A stack of
matrices is analysed in one step. The Mac's GPU has no eigh, so there it runs on the
cpu and the answers are moved back.

SYNTAX:
t.linalg.eigh

INPUTS and PARAMETERS:

matrix:
A square, symmetric matrix (equal to its own transpose), such as a covariance or
correlation matrix, or a stack of them. Only its lower triangle is read, so a matrix
that is not symmetric gives the answer for a different matrix. Every input is analysed
and sent.

order:
An option, under the node: largest first (the default here, the principal axis first)
or smallest first (the order torch itself gives).

OUTPUTS:

eigenvalues:
How much the matrix stretches along each eigenvector.

eigenvectors:
One eigenvector per COLUMN, in the same order as the eigenvalues, each of length 1.
Sent first. The sign of each is arbitrary: (0.71, 0.71) and (-0.71, -0.71) are the
same axis.

READING THE DEMO:
Six points lying close to the diagonal line y = x. Their covariance, analysed: the
first eigenvalue (6.94) is hundreds of times the second (0.02), and the first
eigenvector, the first column, is about (-0.71, -0.71) - the diagonal, the direction
the points run along (its sign is arbitrary: (0.71, 0.71) is the same axis). The second axis, at right angles, holds almost nothing: the points
barely stray from the line.

RELATED:
t.cov makes the covariance matrix to feed it. t.linalg.svd does the same job for any
matrix, symmetric or not. t.linalg.eig takes any square matrix; t.linalg.pca_low_rank
does a whole PCA in one step. np.linalg.eigh is the numpy version.
""",
metro() + [
    C('c1', 'six points along y = x: x y x y ...', 180, 53),
    N('m1', 'message', 180, 92, W(POINTS), 42, MSG(POINTS)),
    N('rs', 't.reshape 6 2', 380, 160, 150, 70),
    C('c2', 'how x and y vary', 28, 260),
    N('cv', 't.cov', 28, 295, 150, 70),
    C('c3', 'its principal axes', 220, 260),
    N('eg', 't.linalg.eigh', 220, 295, 150, 90),
    C('c4', 'eigenvalues', 420, 260),
    N('hv', 'heat_map', 420, 295, 168, 80, HM(150, 40, 1, 0.0, 7.0, '%.2f')),
    C('c5', 'eigenvectors: columns', 620, 260),
    N('he', 'heat_map', 620, 295, 168, 130, HM(150, 80, 1, -1.0, 1.0, '%.2f', 'red-blue')),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'cv', 'in'),
 ('cv', 'covariance', 'eg', 'matrix'), ('eg', 'eigenvalues', 'hv', 'y'),
 ('eg', 'eigenvectors', 'he', 'y')],
['t.cov', 't.linalg.svd', 't.linalg.eig', 't.linalg.pca_low_rank', 'np.linalg.eigh'], 520, 1100)


# ----------------------------------------------------------------- t.gradient
page('t.gradient', 't.gradient - the rate of change at every point',
f"""
Gives the slope of a tensor at every point: how fast the values are changing there.
Inside it takes the centred difference - half the distance from the value before to
the value after - and at the two ends the one-sided difference.

Unlike t.diff, the result is the same length as the input and each value is centred on
its own point, not between two. That makes it the natural way to get velocities from a
recorded path (and accelerations from those), slopes from a curve, or edges from a row
of an image, while keeping everything lined up with the original. A buffer of frames x
joints gives every joint's velocity at once (dim 0).
{DEVICE}

SYNTAX:
t.gradient <spacing: float> <dim: int>
Defaults: spacing 1, dim -1 (along each row).

EXAMPLE:
t.gradient 0.01667 0

INPUTS and PARAMETERS:

in:
The tensor (a list is turned into one). Every input is differentiated and sent.

spacing (float box, also an inlet):
The distance between neighbouring values. The slopes are divided by it, so with the
time between samples (1/60 for 60 frames a second) the result is per second.

dim (int box, also an inlet):
Which way to measure: -1 along each row, 0 down each column.

edge order:
An option, under the node: 1 (the default) or 2. 2 uses three points at each end,
which follows curves more closely there.

OUTPUTS:

gradient:
The slope at each point, the same shape as the input.

READING THE DEMO:
The squares 0 1 4 9 16 25 are a path speeding up. t.gradient gives 1 2 4 6 8 9: at
each point, its speed. Inside, these are exactly twice the position's index, as they
should be for squares; the two ends are rougher. t.diff gives 1 3 5 7 9 - the steps
BETWEEN points, one fewer.

RELATED:
t.diff gives the steps between neighbours. t.cumulative_trapezoid goes the other way,
from a rate back to a running total. np.gradient is the numpy version.
""",
metro() + [
    C('c1', 'a path speeding up', 180, 53),
    N('m1', 'message', 180, 92, 200, 42, MSG('0 1 4 9 16 25')),
    C('c2', 'the speed at each point', 420, 53),
    N('g1', 't.gradient', 420, 92, 150, 110),
    N('l1', 'list', 600, 92, 220, 42, LIST),
    C('c3', 'the steps between points: one fewer', 420, 240),
    N('d1', 't.diff', 420, 275, 150, 90),
    N('l2', 'list', 600, 275, 220, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'g1', 'in'), ('m1', 'message out', 'd1', 'tensor in'),
 ('g1', 'gradient', 'l1', ''), ('d1', 'tensor out', 'l2', '')],
['t.diff', 't.cumulative_trapezoid', 'np.gradient', 't.cumsum'], 430, 960)


# ---------------------------------------------------------------- t.trapezoid
TRAP_SPEEDS = '0 1 2 3 4'
page('t.trapezoid', 't.trapezoid - the area under a curve',
f"""
Adds up the area under a curve given as samples: the integral. Between each pair of
neighbouring samples it takes the area of the trapezoid they make - their average
times the spacing - and sums them.

From speeds it gives the distance travelled; from a joint's power, the work done; from
a rate, the total; from a level over time, the overall dose. It is more accurate than
adding the samples and multiplying by the spacing (t.sum), which counts each end as if
it had lasted a whole step.
{DEVICE}

SYNTAX:
t.trapezoid <spacing: float>
Default: spacing 1.

EXAMPLE:
t.trapezoid 0.01667

INPUTS and PARAMETERS:

in:
The samples (a list or tensor). Every input is integrated and sent.

spacing (float box, also an inlet):
The distance between samples - the time between frames (1/60 for 60 frames a
second), so the answer comes out per second rather than per frame.

dim (int box, also an inlet):
Which way to integrate: -1 (the default) along each row, 0 down each column.

OUTPUTS:

area:
The total area: one number, or one per row or column.

READING THE DEMO:
A speed rising steadily from 0 to 4 metres a second, sampled every half second. The
distance covered is the area of a triangle, 2 seconds by 4 metres a second, halved: 4
metres, which t.trapezoid gets exactly. t.sum times the spacing gives 5, counting the
final speed of 4 as if it had lasted a whole half second.

RELATED:
t.cumulative_trapezoid gives the running total at every sample. t.cumsum is the
running sum without the averaging. t.gradient goes the other way, from a total to its
rate. np.trapezoid is the numpy version.
""",
metro(label='twice a second', props=METRO_HALF, init='metro 500') + [
    C('c1', 'speeds, every half second', 180, 53),
    N('m1', 'message', 180, 92, 160, 42, MSG(TRAP_SPEEDS)),
    C('c2', 'the distance covered', 380, 53),
    N('tz', 't.trapezoid 0.5', 380, 92, 170, 110),
    N('l1', 'list', 580, 92, 80, 42, LIST),
    C('c3', 't.sum times the spacing: too much', 380, 240),
    N('sm', 't.sum', 380, 275, 150, 90),
    N('ml', '* 0.5', 380, 390, 129, 70),
    N('l2', 'list', 580, 390, 80, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'tz', 'in'), ('m1', 'message out', 'sm', 'tensor in'),
 ('tz', 'area', 'l1', ''), ('sm', 'output', 'ml', 'in'), ('ml', 'result', 'l2', '')],
['t.cumulative_trapezoid', 't.cumsum', 't.gradient', 'np.trapezoid'], 520, 960)


page('t.cumulative_trapezoid', 't.cumulative_trapezoid - the area so far, at every sample',
f"""
Gives the running total of the area under a curve: at each sample, the integral from
the start up to that point. It starts at 0 and has one value per sample, so it lines
up with the input.

From speeds it gives the distance travelled so far - the position at each moment; from
accelerations, the speed so far; from a joint's power, the work done so far. Each step
adds the trapezoid between two neighbouring samples: their average times the spacing.

torch's own cumulative_trapezoid leaves out the starting 0 and so is one shorter; this
node puts it back, as np.cumulative_trapezoid does.
{DEVICE}

SYNTAX:
t.cumulative_trapezoid <spacing: float>
Default: spacing 1.

INPUTS and PARAMETERS:

in:
The samples (a list or tensor). Every input is integrated and sent.

spacing (float box, also an inlet):
The distance between samples - the time between frames (1/60 for 60 frames a second),
so the totals come out per second.

dim (int box, also an inlet):
Which way to integrate: -1 (the default) along each row, 0 down each column - one
running total per joint for a buffer of frames x joints.

OUTPUTS:

running total:
The area so far at each sample, starting from 0, the same shape as the input.

READING THE DEMO:
A speed rising steadily from 0 to 4 metres a second, sampled every half second. The
distance so far is 0, 0.25, 1, 2.25 and 4 metres: growing faster and faster as the
speed builds - the squares, as a steady acceleration gives. The last value is
t.trapezoid's total.

RELATED:
t.trapezoid gives only the final total. t.cumsum is the running sum without the
averaging. t.gradient goes the other way, from the running total back to the rate.
np.cumulative_trapezoid is the numpy version.
""",
metro(label='twice a second', props=METRO_HALF, init='metro 500') + [
    C('c1', 'speeds, every half second', 180, 53),
    N('m1', 'message', 180, 92, 160, 42, MSG(TRAP_SPEEDS)),
    C('c2', 'the distance so far', 380, 53),
    N('ct', 't.cumulative_trapezoid 0.5', 380, 92, 250, 110),
    N('l1', 'list', 380, 240, 260, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'ct', 'in'), ('ct', 'running total', 'l1', '')],
['t.trapezoid', 't.cumsum', 't.gradient', 'np.cumulative_trapezoid'], 380, 960)


# ----------------------------------------------------------------- t.meshgrid
page('t.meshgrid', 't.meshgrid - the x and y of every point on a grid',
"""
Takes the positions along x and the positions along y, and makes two matrices the
size of the whole grid: X holds every point's x, and Y every point's y.

With those, any formula of x and y can be worked out over the whole grid at once,
element by element: the distance from the centre (hypot of X and Y), a wave (sin of
X), a gradient, a mask of a region - or a trained network run over every point of a
plane (t.model). It is how a 2-D picture, field or terrain is computed from a formula
instead of drawn point by point. The grids are made on x's device.

SYNTAX:
t.meshgrid

INPUTS and PARAMETERS:

x:
The positions along x (a list or 1-D tensor). Every input makes the grid and sends.

y:
The positions along y, held: send them first. Without y the grid is square, using x
for both.

indexing:
An option, under the node. xy (the default) makes X and Y with one row per y and one
column per x, the way an image is laid out. ij makes one row per x instead.

OUTPUTS:

X:
Each point's x: every row the same.

Y:
Each point's y: every column the same. Sent first, then X.

READING THE DEMO:
Top: x 1 2 3 4 and y 10 20 30 make a 3 x 4 grid. X repeats 1 2 3 4 down every row; Y
repeats 10 20 30 across every column.

Bottom: 32 positions from -1 to 1 make a 32 x 32 grid, and hypot of X and Y gives each
point's distance from the centre: a round glow, 0 in the middle and 1.41 in the
corners.

RELATED:
t.linspace and t.arange make the positions. hypot, atan2 and the maths nodes then work
on the whole grid. np.meshgrid is the numpy version.
""",
metro() + held() + [
    C('c2', 'x', 180, 53),
    N('mx', 'message', 180, 92, 140, 42, MSG('1 2 3 4')),
    C('c3', 'y, held', 180, 175),
    N('my', 'message', 180, 210, 140, 42, MSG('10 20 30')),
    C('c4', 'a 3 x 4 grid', 360, 53),
    N('mg', 't.meshgrid', 360, 92, 150, 90),
    C('c5', 'X', 540, 53),
    N('hx', 'heat_map', 540, 92, 218, 130, HM(200, 80, 1, 0.0, 4.0, '%.0f')),
    C('c6', 'Y', 540, 250),
    N('hy', 'heat_map', 540, 285, 218, 130, HM(200, 80, 1, 0.0, 30.0, '%.0f')),
    C('c7', '32 positions, -1 to 1', 28, 470),
    N('ls', 't.linspace -1 1 32', 28, 505, 170, 130),
    C('c8', 'a square grid', 230, 470),
    N('m2', 't.meshgrid', 230, 505, 150, 90),
    C('c9', 'distance from the centre', 230, 620),
    N('hp', 'hypot', 230, 655, 140, 70),
    N('hd', 'heat_map', 420, 505, 228, 230, HM(210, 210, 1, 0.0, 1.42, '', 'plasma')),
],
[('mt', '', 'mx', ''), ('lb', 'out', 'my', ''), ('lb', 'out', 'ls', '###input'),
 ('mx', 'message out', 'mg', 'x'), ('my', 'message out', 'mg', 'y'),
 ('mg', 'X', 'hx', 'y'), ('mg', 'Y', 'hy', 'y'),
 ('ls', 'linspace out', 'm2', 'x'), ('m2', 'Y', 'hp', 'y'), ('m2', 'X', 'hp', 'x'),
 ('hp', 'length', 'hd', 'y')],
['t.linspace', 'hypot', 'np.meshgrid', 't.model'], 780, 960)


# ---------------------------------------------------------- t.logical_and/or/xor
LOGIC = {
    'and': ('both are true', '1 0 0 0', 'true only where A AND B are both true',
            """
Use it to combine conditions that must all hold: inside a range (above the low limit
and below the high one), a hand both raised and still, a joint both moving and
finite (t.isfinite). Chain two to require three conditions.""",
            't.logical_or', 't.logical_xor'),
    'or': ('either is true', '1 1 1 0', 'true where A OR B (or both) is true',
           """
Use it to combine conditions any one of which is enough: outside a range (below the
low limit or above the high one), either hand raised, a value that is nan or
infinite (t.isnan, t.isinf).""",
           't.logical_and', 't.logical_xor'),
    'xor': ('exactly one is true', '0 1 1 0', 'true where A and B DIFFER',
            """
Use it to find disagreement: where two masks differ, which cells changed between two
thresholded frames, which of two sensors is reporting alone. It is true where one is
true and the other is not - "one or the other, but not both".""",
            't.logical_and', 't.logical_or'),
}

for op, (gist, result, says, use, sib1, sib2) in LOGIC.items():
    stem = 't.logical_' + op
    page(stem, stem + ' - true where ' + gist,
f"""
Combines two tensors of true and false element by element, giving true {says[5:]}.

A     B     {op}
1     1     {'1' if op != 'xor' else '0'}
1     0     {'0' if op == 'and' else '1'}
0     1     {'0' if op == 'and' else '1'}
0     0     0
{use}

The inputs need not be true and false: any number other than 0 counts as true, and 0
as false, so 2.5 and -1 are both true. The masks from t.gt, t.lt, t.eq, t.isnan and the
other tests are ready to use as they are. B is moved to A's device.

SYNTAX:
{stem}

INPUTS and PARAMETERS:

in:
The first tensor, A (a list is turned into one). Every input combines and sends - the
only inlet that makes the node send.

in 2:
The second tensor, B. It is held, and used with every A that follows; it does not send
by itself, so send it first. It must have the same shape as A, or a shape that
broadcasts against it (a single value goes with every element).

output_type:
An option, under the node: bool (the default), int (1 and 0) or float (1.0 and 0.0).
Choose int or float to plot the result, or to multiply with it.

OUTPUTS:

result:
True {says[5:]}, the shape of A and B together.

READING THE DEMO:
A is 1 1 0 0 and B is 1 0 1 0 - every pairing of true and false once - so the result,
{result}, is the whole table above in one go (output_type is set to int here, so it
shows 1s and 0s). Type other 1s and 0s into either message and click it.

RELATED:
{sib1} and {sib2} combine the same way with other rules; t.logical_not turns true into
false. t.any and t.all ask whether ANY or ALL of one tensor is true. np.logical_{op}
is the numpy version.
""",
    metro() + held() + [
        C('c2', 'A', 180, 53),
        N('ma', 'message', 180, 92, 140, 42, MSG('1 1 0 0')),
        C('c4', 'B, held', 180, 175),
        N('mb', 'message', 180, 210, 140, 42, MSG('1 0 1 0')),
        C('c5', 'A ' + op + ' B', 370, 53),
        N('op', stem, 370, 92, 170, 90, {'output_type': 'int'}),
        C('c6', 'result', 590, 53),
        N('ls', 'list', 590, 92, 220, 42, LIST),
    ],
    [('mt', '', 'ma', ''), ('lb', 'out', 'mb', ''), ('mb', 'message out', 'op', 'in 2'),
     ('ma', 'message out', 'op', 'in'), ('op', 'result', 'ls', '')],
    [sib1, sib2, 't.logical_not', 't.any', 't.all', 'np.logical_' + op], 340, 860)


page('t.logical_not', 't.logical_not - true becomes false',
"""
Turns every true into false and every false into true, element by element.

Use it to flip a mask: the cells NOT above a threshold, the frames where a hand is NOT
moving, everything outside a selected region, the values that are NOT nan. With
t.where it selects the other branch; with t.logical_and it subtracts one condition from
another (A and not B: true in A but not in B).

Any number other than 0 counts as true and 0 as false, so not of 2.5 is false and not
of 0 is true.

SYNTAX:
t.logical_not

INPUTS and PARAMETERS:

in:
The tensor (a list is turned into one). Every input is flipped and sent.

output_type:
An option, under the node: bool (the default), int (1 and 0) or float (1.0 and 0.0).
Choose int or float to plot the result, or to multiply with it.

OUTPUTS:

result:
The opposite of each element, the same shape as the input.

READING THE DEMO:
Top: 1 0 0 1 becomes 0 1 1 0. Bottom: 0 2.5 -1 0 becomes 1 0 0 1 - only the zeros were
false (output_type is set to int here, so they show as 1s and 0s). Edit a message and
click it.

RELATED:
t.logical_and, t.logical_or and t.logical_xor combine two tensors. t.ne 0 makes a mask
from numbers; t.where picks between two tensors by a mask. np.logical_not is the numpy
version.
""",
metro() + [
    C('c1', 'true and false', 180, 53),
    N('m1', 'message', 180, 92, 140, 42, MSG('1 0 0 1')),
    C('c2', 'flip each', 370, 53),
    N('n1', 't.logical_not', 370, 92, 170, 70, {'output_type': 'int'}),
    C('c3', 'result', 590, 53),
    N('l1', 'list', 590, 92, 220, 42, LIST),
    C('c4', 'any numbers: only 0 is false', 180, 175),
    N('m2', 'message', 180, 210, 160, 42, MSG('0 2.5 -1 0')),
    N('n2', 't.logical_not', 370, 210, 170, 70, {'output_type': 'int'}),
    N('l2', 'list', 590, 210, 220, 42, LIST),
],
[('mt', '', 'm1', ''), ('mt', '', 'm2', ''), ('m1', 'message out', 'n1', 'in'),
 ('m2', 'message out', 'n2', 'in'), ('n1', 'result', 'l1', ''), ('n2', 'result', 'l2', '')],
['t.logical_and', 't.logical_or', 't.where', 'np.logical_not'], 340, 820)


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
                    text_width=760, text_height=p['text_height']))


if __name__ == '__main__':
    main()
