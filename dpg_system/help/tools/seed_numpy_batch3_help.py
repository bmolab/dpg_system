"""Seed the help pages for the third batch of numpy-side nodes (2026-10-24): one page per node.

    python3 dpg_system/help/tools/seed_numpy_batch3_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

hypot, np.full, np.meshgrid, np.trace, np.tril, np.triu, np.take_along_axis,
np.tensordot, np.linalg.qr, np.linalg.eig. (sign, log1p, degrees, radians, tanh,
sinh and cosh are in derive_math_single_help.py.)

Like the first two seeders it is not a make_* script, so rebuild.py never runs it,
and it will not write over an existing page without --force. Treat the pages as
hand-made once written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from help_common import SIG, PLOT
from seed_numpy_more_help import LIST, MSG, HM, C, N, see_also
from seed_numpy_batch2_help import METRO, metro_row


PAGES = {}


def page(stem, title, body, demo, links, also, also_y, text_height=820):
    PAGES[stem] = dict(title=title, body=body.strip('\n'), demo=demo + see_also(also_y, also),
                       links=links, text_height=text_height)


def matrix_source(text, shape, x=180, y=53, note=None):
    """A message of numbers and the np.reshape that makes it a matrix, with a comment."""
    return [C('cs', note or ('as ' + shape.replace(' ', ' x ')), x, y),
            N('ms', 'message', x, y + 39, 200, 42, MSG(text)),
            N('rs', 'np.reshape ' + shape, x + 230, y + 39, 150, 90)]


# ------------------------------------------------------------------ hypot
page('hypot', 'hypot - the distance of a point from the origin',
"""
Gives the length of the line from the origin to the point (x, y): the square root of
x x + y y, the long side of a right-angled triangle whose other sides are x and y.

It is the distance between two points once you subtract one from the other, the
speed of something moving with velocity (x, y), the size of a 2-D vector. Unlike
squaring, adding and taking the square root yourself, it does not overflow or lose
precision when the numbers are very large or very small.

With atan2, which gives the point's angle, it converts a point to polar form:
distance and direction.

SYNTAX:
hypot <y: float>
Default: y 0 (so hypot of x alone is the size of x).

EXAMPLE:
hypot 4

INPUTS and PARAMETERS:

x:
One side. A number, list, array or tensor - each element is paired with the
matching element of y. Every input sends: the only inlet that makes the node send.

y (float box, also an inlet):
The other side. A single number pairs with every x; a list, array or tensor must
have the same shape as x. It does not send by itself: send y first, then x.

OUTPUTS:

length:
The distance, never negative, the same shape as x.

READING THE DEMO:
Top: a point moves round an ellipse, 1 wide and 0.5 high. Its distance from the
centre swings between 1 (at the ends) and 0.5 (at the top and bottom), twice per
turn.

Bottom: the classic 3 4 5 triangle: hypot of 3 and 4 is 5.

RELATED:
atan2 gives the point's angle. np.linalg.norm gives the length of a vector of any
number of dimensions. sqrt, ^ and + do the sum by hand.
""",
[
    C('c1', 'a point going round an ellipse', 28, 53),
    N('sx', 'signal 4.0 cos', 28, 92, 129, 78, SIG('cos', 4.0)),
    N('sy', 'signal 4.0 sin', 180, 92, 129, 78, dict(SIG('sin', 4.0), range=0.5)),
    C('c2', 'x 1 wide, y 0.5 high', 340, 53),
    N('hy', 'hypot', 340, 92, 140, 70),
    C('c3', 'its distance from the centre', 520, 53),
    N('pl', 'plot', 520, 92, 208, 178, PLOT(0.0, 1.2, 300)),
    C('c4', 'once at load', 28, 335),
    N('lb', 'load_bang', 28, 370, 88, 46),
    C('c5', 'x 3 ...', 140, 335),
    N('m1', 'message', 140, 370, 60, 42, MSG('3')),
    C('c6', '... and y 4', 330, 335),
    N('h2', 'hypot 4', 330, 370, 140, 70),
    C('c7', 'length', 520, 335),
    N('l1', 'list', 520, 370, 120, 42, LIST),
],
[('sy', '', 'hy', 'y'), ('sx', '', 'hy', 'x'), ('hy', 'length', 'pl', 'y'),
 ('lb', 'out', 'm1', ''), ('m1', 'message out', 'h2', 'x'), ('h2', 'length', 'l1', '')],
['atan2', 'np.linalg.norm', 'sqrt', 'np.meshgrid'], 520, 820)


# ------------------------------------------------------------------- full
page('np.full', 'np.full - an array filled with one value',
"""
Makes an array of any shape with every element set to the same value.

np.zeros and np.ones do this for 0 and 1; np.full does it for anything else - a
starting level of 0.5 for every channel, -1 to mark "no reading yet", a background
value for an image, a default weight for every joint.

SYNTAX:
np.full <dims ...> <value>
Default: 3 values of 0. With one number it is the length; with two or more, the last
is the value and the rest are the shape.

EXAMPLES:
np.full 10 0.5
np.full 4 4 -1

INPUTS and PARAMETERS:

(button):
Makes the array and sends it.

shape (text box, also an inlet):
The size along each axis: 10 for a list of 10, 2 3 for 2 rows of 3.

value (float box, also an inlet):
The value every element gets.

OUTPUTS:

full out:
The array (32-bit floats).

READING THE DEMO:
Made at load: 2 rows of 3, all 7; and a list of 5, all 0.5. Change a shape or a value
and click its button.

RELATED:
np.zeros and np.ones fill with 0 and 1. t.full is the PyTorch version. np.tile repeats
a whole array rather than one value.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', '2 rows of 3, all 7', 180, 53),
    N('f1', 'np.full 2 3 7', 180, 92, 170, 110),
    N('h1', 'heat_map', 400, 92, 168, 110, HM(150, 60, 1, 0.0, 10.0, '%.0f')),
    C('c3', '5 values, all 0.5', 180, 250),
    N('f2', 'np.full 5 0.5', 180, 285, 170, 110),
    N('l1', 'list', 400, 285, 260, 42, LIST),
],
[('lb', 'out', 'f1', ''), ('lb', 'out', 'f2', ''), ('f1', 'full out', 'h1', 'y'),
 ('f2', 'full out', 'l1', '')],
['np.zeros', 'np.ones', 't.full', 'np.tile'], 440, 760)


# --------------------------------------------------------------- meshgrid
page('np.meshgrid', 'np.meshgrid - the x and y of every point on a grid',
"""
Takes the positions along x and the positions along y, and makes two matrices the
size of the whole grid: X holds every point's x, and Y every point's y.

With those, any formula of x and y can be worked out over the whole grid at once,
element by element: the distance from the centre (hypot of X and Y), a wave (sin of
X), a gradient, a mask of a region. It is how a 2-D picture, field or terrain is
computed from a formula instead of drawn point by point.

SYNTAX:
np.meshgrid

INPUTS and PARAMETERS:

x:
The positions along x (a list or 1-D array). Every input makes the grid and sends.

y:
The positions along y, held: send them first. Without y the grid is square, using
x for both.

indexing:
An option, under the node. xy (the default) makes X and Y with one row per y and one
column per x, the way an image is laid out. ij makes one row per x instead.

OUTPUTS:

X:
Each point's x: every row the same.

Y:
Each point's y: every column the same. Sent first, then X.

READING THE DEMO:
Top: x 1 2 3 4 and y 10 20 30 make a 3 x 4 grid. X repeats 1 2 3 4 down every row;
Y repeats 10 20 30 across every column.

Bottom: 32 positions from -1 to 1 make a 32 x 32 grid, and hypot of X and Y gives each
point's distance from the centre: a round glow, 0 in the middle and 1.41 in the
corners.

RELATED:
np.grid makes a list of xyz points on a grid instead. np.linspace and np.arange make
the positions. hypot, atan2 and the maths nodes then work on the whole grid.
""",
metro_row() + [
    C('c1', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 210, 88, 46),
    C('c2', 'x', 180, 53),
    N('mx', 'message', 180, 92, 140, 42, MSG('1 2 3 4')),
    C('c3', 'y, held', 180, 175),
    N('my', 'message', 180, 210, 140, 42, MSG('10 20 30')),
    C('c4', 'a 3 x 4 grid', 360, 53),
    N('mg', 'np.meshgrid', 360, 92, 150, 90),
    C('c5', 'X', 540, 53),
    N('hx', 'heat_map', 540, 92, 218, 130, HM(200, 80, 1, 0.0, 4.0, '%.0f')),
    C('c6', 'Y', 540, 250),
    N('hy', 'heat_map', 540, 285, 218, 130, HM(200, 80, 1, 0.0, 30.0, '%.0f')),
    C('c7', '32 positions, -1 to 1', 28, 470),
    N('ls', 'np.linspace -1 1 32', 28, 505, 170, 150),
    C('c8', 'a square grid', 230, 470),
    N('m2', 'np.meshgrid', 230, 505, 150, 90),
    C('c9', 'distance from the centre', 230, 620),
    N('hp', 'hypot', 230, 655, 140, 70),
    N('hd', 'heat_map', 420, 505, 228, 230, HM(210, 210, 1, 0.0, 1.42, '', 'plasma')),
],
[('mt', '', 'mx', ''), ('lb', 'out', 'my', ''), ('lb', 'out', 'ls', ''),
 ('mx', 'message out', 'mg', 'x'), ('my', 'message out', 'mg', 'y'),
 ('mg', 'X', 'hx', 'y'), ('mg', 'Y', 'hy', 'y'),
 ('ls', 'linspace out', 'm2', 'x'), ('m2', 'Y', 'hp', 'y'), ('m2', 'X', 'hp', 'x'),
 ('hp', 'length', 'hd', 'y')],
['np.grid', 'np.linspace', 'hypot', 'atan2'], 790, 1000)


# ------------------------------------------------------------------ trace
page('np.trace', 'np.trace - the sum of a diagonal',
"""
Adds up the values along a matrix's diagonal, from the top left corner down to the
right.

For a covariance matrix it is the total variance - how much the data varies
altogether, in every direction combined. For a rotation matrix it gives the angle of
rotation (the trace is 1 + 2 cos of the angle), so it measures how far one
orientation is from another without needing the axis. For a correlation matrix it is
simply the number of variables.

SYNTAX:
np.trace <diagonal: int>
Default: 0, the main diagonal.

INPUTS and PARAMETERS:

in:
A matrix (or a stack of them: the last two axes are used, one trace each). Every
input is summed and sent.

diagonal (int box, also an inlet):
Which diagonal: 0 the main one, 1 the one above it, -1 the one below.

OUTPUTS:

trace:
The sum.

READING THE DEMO:
1 to 9 as a 3 x 3 matrix. Its diagonal is 1 5 9, so the trace is 15; the diagonal
just above it is 2 6, which sums to 8. A metro re-sends the matrix five times a second.

RELATED:
np.diag reads the diagonal off as a list. np.sum adds everything. np.cov's trace is
its total variance.
""",
metro_row() + matrix_source('1 2 3 4 5 6 7 8 9', '3 3', note='1 to 9 as a 3 x 3 matrix') + [
    N('hm', 'heat_map', 600, 92, 168, 170, HM(150, 120, 1, 0.0, 9.0, '%.0f')),
    C('c1', 'the main diagonal: 1 + 5 + 9', 180, 250),
    N('t1', 'np.trace', 180, 285, 150, 90),
    N('l1', 'list', 360, 285, 80, 42, LIST),
    C('c2', 'the one above: 2 + 6', 180, 410),
    N('t2', 'np.trace 1', 180, 445, 150, 90),
    N('l2', 'list', 360, 445, 80, 42, LIST),
],
[('mt', '', 'ms', ''), ('ms', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'hm', 'y'),
 ('rs', 'output', 't1', 'in'), ('rs', 'output', 't2', 'in'),
 ('t1', 'trace', 'l1', ''), ('t2', 'trace', 'l2', '')],
['np.diag', 'np.sum', 'np.cov', 'np.eye'], 590, 760)


# -------------------------------------------------------------- tril/triu
TRI = {
    'np.tril': ('lower', 'below', 'above', '1 0 0 / 4 5 0 / 7 8 9', 'np.tril -1',
                'the diagonal too is cleared with -1', 'np.triu'),
    'np.triu': ('upper', 'above', 'below', '1 2 3 / 0 5 6 / 0 0 9', 'np.triu 1',
                'the diagonal too is cleared with 1', 'np.tril'),
}
for stem, (part, keep, clear, result, second, second_note, other) in TRI.items():
    page(stem, f'{stem} - keep the {part} triangle of a matrix',
f"""
Keeps the diagonal of a matrix and everything {keep} it, and sets everything {clear}
it to zero: the {part} triangle.

A triangular matrix is half of a symmetric one - a covariance, a correlation, a table
of distances between every pair of points - and the half is all the information
there is, so {stem} removes the duplicate copy - and, with diagonal
{'-1' if stem == 'np.tril' else '1'}, the diagonal of self-pairs as well. It is also how masks are made that
let each step of a sequence see only the steps {'before' if stem == 'np.tril' else 'after'} it.

SYNTAX:
{stem} <diagonal: int>
Default: 0, keep the main diagonal.

INPUTS and PARAMETERS:

in:
A matrix (or a stack of them). Every input is cut and sent.

diagonal (int box, also an inlet):
Where the triangle's edge is: 0 keeps the main diagonal, 1 moves the edge one
diagonal up, -1 one down.

OUTPUTS:

triangle:
The matrix with the other side set to zero.

READING THE DEMO:
1 to 9 as a 3 x 3 matrix, cut to {result}. Below, {second}: {second_note}.
A metro re-sends the matrix five times a second.

RELATED:
{other} keeps the other triangle. np.diag keeps only the diagonal. np.where can
blank any region given a mask.
""",
    metro_row() + matrix_source('1 2 3 4 5 6 7 8 9', '3 3', note='1 to 9 as a 3 x 3 matrix') + [
        N('hm', 'heat_map', 600, 92, 168, 170, HM(150, 120, 1, 0.0, 9.0, '%.0f')),
        C('c1', f'the {part} triangle', 180, 250),
        N('t1', stem, 180, 285, 150, 90),
        N('h1', 'heat_map', 360, 285, 168, 170, HM(150, 120, 1, 0.0, 9.0, '%.0f')),
        C('c2', second_note, 180, 490),
        N('t2', second, 180, 525, 150, 90),
        N('h2', 'heat_map', 360, 525, 168, 170, HM(150, 120, 1, 0.0, 9.0, '%.0f')),
    ],
    [('mt', '', 'ms', ''), ('ms', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'hm', 'y'),
     ('rs', 'output', 't1', 'in'), ('rs', 'output', 't2', 'in'),
     ('t1', 'triangle', 'h1', 'y'), ('t2', 'triangle', 'h2', 'y')],
    [other, 'np.diag', 'np.where', 't.tril' if stem == 'np.tril' else 't.triu'], 740, 760)


# -------------------------------------------------------- take_along_axis
page('np.take_along_axis', 'np.take_along_axis - pick from each row by its own positions',
"""
Picks elements from each row (or column) of an array using that row's own list of
positions. Where np.take uses one list of positions for every row, this takes a
whole array of positions, shaped like the data, so each row can be reordered or
sampled differently.

It is the partner of np.argsort, np.argmax and np.argmin used along an axis: they
give positions row by row, and np.take_along_axis uses them to fetch the values -
sorting every row of one array by the order of another, picking each joint's largest
reading together with its time, keeping the three strongest channels per frame.

SYNTAX:
np.take_along_axis <axis: int>
Default: axis -1 (along each row).

INPUTS and PARAMETERS:

in:
The array to pick from. Every input is picked from and sent.

indices:
The positions, held: send them first. They must have as many axes as the array; along
the other axes their shape must match it (or be 1). -1 counts from the end;
positions past the end are clipped to it.

axis (int box, also an inlet):
Which way the positions run: -1 along each row, 0 down each column.

OUTPUTS:

taken:
The picked elements, shaped like the indices.

READING THE DEMO:
Two rows, 3 1 2 and 9 7 8. np.argsort gives each row's sorting order (1 2 0, the same
for both), and np.take_along_axis uses it to fetch the values: 1 2 3 and 7 8 9 - each
row sorted. The positions come from one array and are used on another, which is the
point: sort the frames of a recording by the order of one joint's height, for
example. A metro re-sends the array five times a second.

RELATED:
np.take uses one list of positions for every row. np.argsort, np.argmax and
np.argmin make positions. t.take_along_dim is the PyTorch version.
""",
metro_row() + matrix_source('3 1 2 9 7 8', '2 3', note='two rows of three') + [
    C('c1', 'the order that sorts each row', 180, 230),
    N('as', 'np.argsort', 180, 265, 150, 90),
    N('la', 'list', 180, 380, 200, 42, LIST),
    C('c2', 'fetch the values in that order', 420, 230),
    N('ta', 'np.take_along_axis', 420, 265, 190, 110),
    N('lt', 'list', 420, 400, 220, 42, LIST),
],
[('mt', '', 'ms', ''), ('ms', 'message out', 'rs', 'tensor in'),
 ('rs', 'output', 'as', 'in'), ('as', 'argsort', 'ta', 'indices'), ('as', 'argsort', 'la', ''),
 ('rs', 'output', 'ta', 'in'), ('ta', 'taken', 'lt', '')],
['np.take', 'np.argsort', 'np.argmax', 't.take_along_dim'], 520, 960)


# --------------------------------------------------------------- tensordot
page('np.tensordot', 'np.tensordot - multiply and sum over matching axes',
"""
Multiplies two arrays and adds up over a number of matching axes: the last axes of
the first array against the first axes of the second.

With axes 1 it is the matrix product (np.matmul) - each row of the first against
each column of the second - extended to arrays of any number of dimensions: a 3-D
array of frames x joints x xyz times a 3 x 3 rotation transforms every point of
every frame at once. With axes 2 it multiplies the last two axes of the first by the
first two of the second, element by element, and adds everything up; for two
matrices that gives a single number measuring how alike they are. With axes 0 it is
the outer product.

SYNTAX:
np.tensordot <axes: int>
Default: axes 2 (NumPy's default).

EXAMPLE:
np.tensordot 1

INPUTS and PARAMETERS:

in:
The first array. Every input is multiplied and sent - the only inlet that makes the
node send.

in 2:
The second array, held: send it first.

axes (int box, also an inlet):
How many axes to sum over. The sizes of the last that many axes of in must equal the
first that many of in 2.

OUTPUTS:

result:
The product. Its axes are in's remaining axes followed by in 2's.

READING THE DEMO:
A is 1 2 / 3 4; B is 0 1 / 1 0, the matrix that swaps columns. With axes 1, A times B
is 2 1 / 4 3: A with its columns swapped. With axes 2, every element of A times the
matching one of B, all added: 0 + 2 + 3 + 0 = 5. A metro re-sends A five times a second.

RELATED:
np.matmul and np.dot are the axes 1 case for matrices. np.einsum can say exactly
which axes to sum. np.outer is the axes 0 case for lists.
""",
metro_row() + matrix_source('1 2 3 4', '2 2', note='A: 1 2 / 3 4') + [
    C('c1', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 210, 88, 46),
    C('c2', 'B: swaps columns, held', 180, 175),
    N('mb', 'message', 180, 210, 120, 42, MSG('0 1 1 0')),
    N('rb', 'np.reshape 2 2', 330, 210, 150, 90),
    C('c3', 'axes 1: the matrix product', 28, 360),
    N('d1', 'np.tensordot 1', 28, 395, 170, 110),
    N('h1', 'heat_map', 230, 395, 168, 130, HM(150, 80, 1, 0.0, 4.0, '%.0f')),
    C('c4', 'axes 2: one number', 450, 360),
    N('d2', 'np.tensordot', 450, 395, 170, 110),
    N('l2', 'list', 650, 395, 80, 42, LIST),
],
[('lb', 'out', 'mb', ''), ('mb', 'message out', 'rb', 'tensor in'),
 ('rb', 'output', 'd1', 'in 2'), ('rb', 'output', 'd2', 'in 2'),
 ('mt', '', 'ms', ''), ('ms', 'message out', 'rs', 'tensor in'),
 ('rs', 'output', 'd1', 'in'), ('rs', 'output', 'd2', 'in'),
 ('d1', 'result', 'h1', 'y'), ('d2', 'result', 'l2', '')],
['np.matmul', 'np.einsum', 'np.dot', 'np.outer'], 600, 1000)


# --------------------------------------------------------------------- qr
page('np.linalg.qr', 'np.linalg.qr - square up a set of directions',
"""
Breaks a matrix into Q, whose columns are directions all at right angles to one
another and each of length 1, and R, an upper triangular matrix, so that the matrix
equals Q times R.

Q's columns span the same directions as the matrix's columns, but squared up: the
first is the first column made length 1, the second is the second column with any
part along the first removed, and so on. That is how a rough frame of axes - three
vectors from a sensor or a body segment that are nearly but not quite at right
angles - is turned into a proper rotation. R records how the original columns were
built from the clean ones.

It is also a numerically safe way to solve least-squares problems.

SYNTAX:
np.linalg.qr

INPUTS and PARAMETERS:

matrix:
Any 2-D matrix (or a stack of them). Every input is decomposed and sent.

mode:
An option, under the node. reduced (the default) gives Q with as many columns as the
matrix; complete gives a square Q.

OUTPUTS:

Q:
The squared-up directions, one per column.

R:
The upper triangular matrix. Sent first, then Q.

READING THE DEMO:
The 3 x 2 matrix 1 2 / 3 4 / 5 6. Q's two columns are at right angles and length 1;
R is upper triangular (its lower left is 0). np.matmul of Q and R gives back the
original matrix. A metro re-sends it five times a second.

SIGNS:
NumPy may return a column of Q negated, with the matching row of R negated too; the
product is the same.

RELATED:
np.linalg.svd and np.linalg.eigh find a matrix's principal directions rather than
squaring up its own. np.linalg.lstsq solves least squares directly. t.linalg.qr is
the PyTorch version.
""",
metro_row() + matrix_source('1 2 3 4 5 6', '3 2', note='six numbers as 3 rows of 2') + [
    C('c1', 'decompose', 600, 53),
    N('qr', 'np.linalg.qr', 600, 92, 150, 90),
    C('c2', 'Q: columns at right angles', 28, 250),
    N('hq', 'heat_map', 28, 285, 168, 170, HM(150, 120, 1, -1.0, 1.0, '%.2f', 'red-blue')),
    C('c3', 'R: upper triangular', 250, 250),
    N('hr', 'heat_map', 250, 285, 168, 130, HM(150, 80, 1, -8.0, 8.0, '%.2f', 'red-blue')),
    C('c4', 'Q times R: the original', 470, 250),
    N('mm', 'np.matmul', 470, 285, 150, 90),
    N('ho', 'heat_map', 470, 400, 168, 170, HM(150, 120, 1, 0.0, 6.0, '%.1f')),
],
[('mt', '', 'ms', ''), ('ms', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'qr', 'matrix'),
 ('qr', 'Q', 'hq', 'y'), ('qr', 'R', 'hr', 'y'), ('qr', 'R', 'mm', 'in 2'), ('qr', 'Q', 'mm', 'in 1'),
 ('mm', 'mat mul result', 'ho', 'y')],
['np.linalg.svd', 'np.linalg.lstsq', 'np.linalg.eigh', 't.linalg.qr'], 640, 1000)


# -------------------------------------------------------------------- eig
page('np.linalg.eig', 'np.linalg.eig - the directions a matrix only stretches',
"""
Finds the eigenvectors of a square matrix - the directions it stretches or shrinks
without turning - and the eigenvalues, how much it stretches along each.

A transformation applied over and over (a feedback loop, a step of a simulation, a
Markov chain of states) ends up dominated by its largest eigenvalue: above 1 it
grows, below 1 it dies away, and the matching eigenvector is the shape it settles
into. Eigenvalues show whether a system is stable, and how fast it changes.

Some matrices turn every direction - a rotation is the plain example - and then the
eigenvalues come out as complex numbers, whose angle is the turn. For a symmetric
matrix (a covariance, a correlation, a distance table) use np.linalg.eigh instead:
it is faster, always real, and sorted.

SYNTAX:
np.linalg.eig

INPUTS and PARAMETERS:

matrix:
A square matrix. Every input is analysed and sent.

OUTPUTS:

eigenvalues:
How much the matrix stretches along each eigenvector, in no particular order. Real
numbers when they all are; complex numbers when the matrix turns some direction.

eigenvectors:
One eigenvector per COLUMN, in the same order, each of length 1. Sent first. Each
one's sign is arbitrary.

READING THE DEMO:
Top: 2 1 / 0 3 stretches by 2 along x - the first eigenvector, the first column, is
1 0 - and by 3 along the diagonal 0.71 0.71.

Bottom: 0 -1 / 1 0 is a quarter turn. No direction survives it unturned, so the
eigenvalues are complex: 1j and -1j, a turn of 90 degrees one way and the other. A
metro re-sends both five times a second.

RELATED:
np.linalg.eigh for symmetric matrices. np.linalg.svd works for any matrix, square
or not. t.linalg.eig is the PyTorch version.
""",
metro_row() + matrix_source('2 1 0 3', '2 2', note='2 1 / 0 3') + [
    C('c1', 'its stretches', 600, 53),
    N('e1', 'np.linalg.eig', 600, 92, 150, 90),
    C('c2', 'eigenvalues', 180, 230),
    N('l1', 'list', 180, 265, 140, 42, LIST),
    C('c3', 'eigenvectors: columns', 380, 230),
    N('hv', 'heat_map', 380, 265, 168, 130, HM(150, 80, 1, -1.0, 1.0, '%.2f', 'red-blue')),
    C('c4', 'a quarter turn: 0 -1 / 1 0', 180, 440),
    N('m2', 'message', 180, 475, 140, 42, MSG('0 -1 1 0')),
    N('r2', 'np.reshape 2 2', 350, 475, 150, 90),
    N('e2', 'np.linalg.eig', 530, 475, 150, 90),
    C('c5', 'complex: no direction escapes turning', 350, 600),
    N('l2', 'list', 350, 635, 200, 42, LIST),
],
[('mt', '', 'ms', ''), ('ms', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'e1', 'matrix'),
 ('e1', 'eigenvalues', 'l1', ''), ('e1', 'eigenvectors', 'hv', 'y'),
 ('mt', '', 'm2', ''), ('m2', 'message out', 'r2', 'tensor in'), ('r2', 'output', 'e2', 'matrix'),
 ('e2', 'eigenvalues', 'l2', '')],
['np.linalg.eigh', 'np.linalg.svd', 't.linalg.eig', 'np.linalg.det'], 740, 1000)


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
        print(build(stem, p['title'], p['body'], p['demo'], p['links'], demo_width=840,
                    text_width=760, text_height=p['text_height']))


if __name__ == '__main__':
    main()
