"""np linear algebra: one page per node.

np.linalg.det, np.linalg.inv, np.linalg.matrix_rank, np.linalg.solve,
np.linalg.norm (euclidean_distance shares it - same class, same settings, same
operation), np.dot, np.inner, np.outer, np.cross, np.matmul. These were one
family page (np_linalg); each now has its own page whose demo makes the maths
visible - matrices as heat_maps, vectors you can edit, and a check that the
answer is right - with a SEE ALSO row under it.

Live vectors and matrices are `vector` nodes: every cell is a drag box, and
the node sends the whole array whenever a cell changes. Where a node's second
inlet only stores (dot, cross, solve ...), the second vector goes through
`t bang array`: the array is stored first, then the bang makes the FIRST
vector re-send, so editing either one updates the answer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build


def HM(width, height, count, miny=0.0, maxy=1.0, fmt='%.2f', color='viridis'):
    """Property block for a heat_map that draws a whole array as one picture."""
    return {'color': color, 'width': width, 'height': height, 'sample count': count,
            'min y': miny, 'max y': maxy, 'update_mode': 'heat_map',
            'number format': fmt}


LIST = {'text in': '', 'font size': '24'}
FLT = {'format': '%.3f', 'width': 100, 'font size': '24'}
INT = {'format': '%d', 'width': 100, 'font size': '24'}
METRO = {'on': True, 'period': 100.0, 'units': 'milliseconds'}


def see_also(y, stems, x=30):
    """A SEE ALSO row: one help_link per related page, left to right."""
    row = []
    for s in stems:
        w = int(len(s) * 7.7) + 30
        row.append({'key': 'sa_' + s, 'init': 'help_link ' + s, 'pos': (x, y),
                    'w': w, 'h': 30})
        x += w + 14
    row.append({'key': 'sa_c', 'comment': True, 'text': 'SEE ALSO - click to open',
                'pos': (x, y)})
    SA_END[0] = x + 190
    return row


SA_END = [0]


def DW(width):
    """Demo width wide enough for the SEE ALSO row just built."""
    return max(width, SA_END[0])


def VEC(rows):
    """Properties for a `vector` node holding these rows (numbers or lists)."""
    p = {'output type': 'numpy', 'component count': len(rows),
         'column count': len(rows[0]) if isinstance(rows[0], list) else 1,
         'number format': '%.2f'}
    for i, r in enumerate(rows):
        p['[' + str(i) + ']'] = r
    return p


def th(body):
    """Text block height that fits the prose."""
    return body.count('\n') * 16 + 60


# ------------------------------------------------------------ shared subpatches

# The F from the rearrange pages: 7 rows by 6 columns, no symmetry, so any
# rearrangement of its columns is easy to read.
F_GLYPH = ('1 1 1 1 1 1  '
           '1 1 0 0 0 0  '
           '1 1 0 0 0 0  '
           '1 1 1 1 0 0  '
           '1 1 0 0 0 0  '
           '1 1 0 0 0 0  '
           '1 1 0 0 0 0')

F_SUB = {
    'name': 'make_F',
    'host': 'pF',
    'demo': [
        {'key': 'met', 'init': 'metro 100', 'pos': (40, 40), 'w': 129, 'h': 70,
         'props': METRO},
        {'key': 'cm', 'init': 'comment re-sends the F ten times a second',
         'pos': (190, 50), 'w': 300, 'h': 30},
        {'key': 'msg', 'init': 'message', 'pos': (40, 130), 'w': 290, 'h': 42,
         'props': {'text in': F_GLYPH, 'font size': '24', 'width': 260}},
        {'key': 'cg', 'init': 'comment the letter F as 42 numbers, 1 = letter, 0 = background',
         'pos': (40, 185), 'w': 480, 'h': 30},
        {'key': 'rs', 'init': 'np.reshape 7 6', 'pos': (40, 230), 'w': 230, 'h': 70},
        {'key': 'cr', 'init': 'comment folded into 7 rows of 6',
         'pos': (290, 245), 'w': 260, 'h': 30},
        {'key': 'out', 'init': 'out F', 'pos': (40, 330), 'w': 150, 'h': 80},
    ],
    'links': [('met', '', 'msg', ''),
              ('msg', 'message out', 'rs', 'tensor in'),
              ('rs', 'output', 'out', 'F')],
}

IDENT6 = '  '.join(' '.join('1' if c == r else '0' for c in range(6)) for r in range(6))

# The 6 x 6 identity matrix, re-sent ten times a second.
IDENT_SUB = {
    'name': 'make_identity',
    'host': 'pI',
    'demo': [
        {'key': 'met', 'init': 'metro 100', 'pos': (40, 40), 'w': 129, 'h': 70,
         'props': METRO},
        {'key': 'cm', 'init': 'comment re-sends it ten times a second',
         'pos': (190, 50), 'w': 300, 'h': 30},
        {'key': 'msg', 'init': 'message', 'pos': (40, 130), 'w': 290, 'h': 42,
         'props': {'text in': IDENT6, 'font size': '24', 'width': 260}},
        {'key': 'cg', 'init': 'comment 36 numbers: 1 on the diagonal, 0 elsewhere',
         'pos': (40, 185), 'w': 420, 'h': 30},
        {'key': 'rs', 'init': 'np.reshape 6 6', 'pos': (40, 230), 'w': 230, 'h': 70},
        {'key': 'cr', 'init': 'comment the 6 x 6 identity matrix',
         'pos': (290, 245), 'w': 280, 'h': 30},
        {'key': 'out', 'init': 'out identity', 'pos': (40, 330), 'w': 150, 'h': 80},
    ],
    'links': [('met', '', 'msg', ''),
              ('msg', 'message out', 'rs', 'tensor in'),
              ('rs', 'output', 'out', 'identity')],
}


def direction_sub(host):
    """An angle in degrees -> the unit vector [cos, sin] pointing that way."""
    return {
        'name': 'make_direction',
        'host': host,
        'demo': [
            {'key': 'in', 'init': 'in angle', 'pos': (40, 40), 'w': 150, 'h': 80},
            {'key': 'tr', 'init': 't float float', 'pos': (40, 150), 'w': 140, 'h': 46},
            {'key': 'ct', 'init': 'comment right first: sin, then cos',
             'pos': (200, 155), 'w': 300, 'h': 30},
            {'key': 'co', 'init': 'cos', 'pos': (40, 230), 'w': 120, 'h': 70,
             'props': {'degrees': True}},
            {'key': 'si', 'init': 'sin', 'pos': (200, 230), 'w': 120, 'h': 70,
             'props': {'degrees': True}},
            {'key': 'pk', 'init': 'pack 2', 'pos': (40, 330), 'w': 140, 'h': 70},
            {'key': 'cp', 'init': 'comment [cos, sin]: length 1, pointing at the angle',
             'pos': (200, 345), 'w': 420, 'h': 30},
            {'key': 'out', 'init': 'out direction', 'pos': (40, 430), 'w': 150, 'h': 80},
        ],
        'links': [('in', 'angle', 'tr', ''),
                  ('tr', 'float', 'si', 'in', 1), ('tr', 'float', 'co', 'in', 0),
                  ('si', 'out', 'pk', 'in 2'), ('co', 'out', 'pk', 'in 1'),
                  ('pk', 'out', 'out', 'direction')],
    }


# Twelve directions round a circle, 30 degrees apart, as a 12 x 2 array: each
# row is [cos, sin] of 0, 30, 60 ... 330 degrees.
DIRS_SUB = {
    'name': 'make_directions',
    'host': 'pD',
    'demo': [
        {'key': 'met', 'init': 'metro 100', 'pos': (40, 40), 'w': 129, 'h': 70,
         'props': METRO},
        {'key': 'cm', 'init': 'comment re-sends the directions ten times a second',
         'pos': (190, 50), 'w': 380, 'h': 30},
        {'key': 'ls', 'init': 'np.linspace 0 330 12', 'pos': (40, 130), 'w': 190, 'h': 150,
         'props': {'start': 0.0, 'stop': 330.0, 'steps': 12, 'dtype': 'float32'}},
        {'key': 'cs', 'init': 'comment 12 angles, 30 degrees apart',
         'pos': (250, 145), 'w': 300, 'h': 30},
        {'key': 'tr', 'init': 't array array', 'pos': (40, 300), 'w': 140, 'h': 46},
        {'key': 'co', 'init': 'cos', 'pos': (40, 370), 'w': 120, 'h': 70,
         'props': {'degrees': True}},
        {'key': 'si', 'init': 'sin', 'pos': (200, 370), 'w': 120, 'h': 70,
         'props': {'degrees': True}},
        {'key': 'st', 'init': 'np.stack', 'pos': (40, 470), 'w': 150, 'h': 100,
         'props': {'axis': '0'}},
        {'key': 'cst', 'init': 'comment 2 rows of 12: the cosines, then the sines',
         'pos': (210, 485), 'w': 400, 'h': 30},
        {'key': 'tp', 'init': 'np.transpose', 'pos': (40, 600), 'w': 150, 'h': 70},
        {'key': 'ctp', 'init': 'comment 12 rows of 2: one direction per row',
         'pos': (210, 615), 'w': 380, 'h': 30},
        {'key': 'out', 'init': 'out directions', 'pos': (40, 700), 'w': 150, 'h': 80},
    ],
    'links': [('met', '', 'ls', ''), ('ls', 'linspace out', 'tr', ''),
              ('tr', 'array', 'si', 'in', 1), ('tr', 'array', 'co', 'in', 0),
              ('si', 'out', 'st', 'in 2'), ('co', 'out', 'st', 'in'),
              ('st', 'stack', 'tp', 'input'),
              ('tp', 'transposed array', 'out', 'directions')],
}


# ------------------------------------------------------------- np.linalg.det
body = """Gives the determinant of a square matrix: one number that says how much
the matrix scales area (or volume), and whether it can be undone.

Think of a matrix as a transformation. The determinant is the factor by
which it stretches area: 2 doubles every area, 0.5 halves it. 0 means the
matrix squashes space flat - two different inputs land on the same output,
so nothing can undo it (np.linalg.inv refuses, np.linalg.solve has no single
answer). A negative determinant means the matrix also mirrors, turning
shapes over.

Reach for it to test whether a matrix can be inverted before you try, or to
tell whether two directions are really different.

FOR A 2 x 2 MATRIX:
rows [a b] and [c d] give a x d - b x c. The rows, drawn as arrows, span a
parallelogram; the determinant is its area, signed. When the rows point
the same way (one is a multiple of the other) the parallelogram has no
area and the determinant is 0.

SYNTAX:
np.linalg.det
(no arguments)

INPUTS and PARAMETERS:

input:
A square matrix - a list of lists is turned into an array. Each matrix
that arrives is measured and sent. A matrix that is not square, or a
1-D list, sends nothing (the reason is printed only when the app is in
verbose mode). A stack of square matrices, shape (count, n, n), gives
one determinant each.

OUTPUTS:

determinant:
The determinant, a float (an array of them for a stack).

READING THE DEMO:
The vector node is the matrix: drag any of its four numbers. It starts
at rows [2 1] and [1 1]: 2 x 1 - 1 x 1 = 1. Drag the bottom-left number
up to 2 and the two rows are the same - det reads 0. Keep going and it
turns negative. Any row 2 that is a multiple of row 1 (1 0.5, 4 2) also
gives 0.

RELATED:
np.linalg.matrix_rank says how many independent directions a matrix
has; it drops below full exactly when det is 0. np.linalg.inv and
np.linalg.solve need a non-zero determinant. np.cross of two 3-D vectors
has the area of their parallelogram as its length. t.linalg.svd breaks a
matrix into its stretches."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'A', 'init': 'vector 2 2', 'pos': (30, 130), 'w': 170, 'h': 150,
     'props': VEC([[2.0, 1.0], [1.0, 1.0]])},
    {'key': 'c0', 'comment': True, 'text': 'the matrix: drag any number',
     'pos': (220, 135)},
    {'key': 'hA', 'init': 'heat_map', 'pos': (30, 300), 'w': 220, 'h': 210,
     'props': HM(200, 160, 2, -3.0, 3.0, color='red-blue')},
    {'key': 'c1', 'comment': True, 'text': 'red below 0, blue above', 'pos': (270, 305)},
    {'key': 'det', 'init': 'np.linalg.det', 'pos': (30, 530), 'w': 170, 'h': 70},
    {'key': 'f1', 'init': 'float', 'pos': (30, 620), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': 'the determinant: 0 when the rows are parallel',
     'pos': (180, 625)},
] + see_also(700, ['np.linalg.matrix_rank', 'np.linalg.inv', 'np.linalg.solve',
                   'np.cross', 't.linalg.svd'])
links = [('lb', 'out', 'A', 'in'), ('A', 'out', 'hA', 'y'), ('A', 'out', 'det', 'input'),
         ('det', 'determinant', 'f1', '')]
print(build('np.linalg.det', 'np.linalg.det - how much a matrix scales area',
            body, demo, links, demo_width=DW(640), text_width=760, text_height=th(body)))

# ----------------------------------------------------- np.linalg.matrix_rank
body = """Counts how many genuinely independent directions a matrix has - its
rank.

Each row of a matrix is a direction. If one row can be made by adding up
multiples of the others, it adds nothing new, and the rank is less than
the number of rows. A 3 x 3 matrix of rank 3 reaches every point in 3-D
space; rank 2 flattens everything onto a plane, rank 1 onto a line, rank
0 (all zeros) onto a single point.

Reach for it to check whether data is really as many-dimensional as it
looks - three sensors that always move together are rank 1, not 3 - or
whether a square matrix can be inverted (only at full rank).

LOOKS CAN DECEIVE:
A smooth picture can have rank 1: every row of an outer product is the
same row scaled, however many rows there are. Random noise of the same
size has full rank.

SYNTAX:
np.linalg.matrix_rank
(no arguments)

INPUTS and PARAMETERS:

input:
Any 2-D array - it need not be square. Each one that arrives is measured
and sent. A 1-D list gives 1 (or 0 if it is all zeros). A stack of
matrices gives one rank each.

NEARLY DEPENDENT ROWS:
The count uses a small tolerance scaled to the matrix (numpy's singular
value test), so rows that are almost but not exactly parallel still
count as independent. Rows that are exact multiples count as one.

OUTPUTS:

matrix rank:
The rank, a whole number.

READING THE DEMO:
Top: a 3 x 3 matrix to edit, starting as the identity - rank 3. Type
1 0 0 into its second row so it copies the first: rank 2. Make all three
rows the same: rank 1. Press zero on the vector: rank 0.
Bottom left: an 8 x 8 outer product of two ramps (made in make_outer) -
a full picture, but rank 1. Bottom right: 8 x 8 random noise - rank 8.
Click np.rand for new noise; the rank stays 8.

RELATED:
np.linalg.det is 0 exactly when a square matrix is below full rank.
np.outer always makes rank-1 matrices. np.linalg.solve needs full rank.
t.linalg.svd gives the singular values the rank is counted from."""

OUTER_SUB = {
    'name': 'make_outer',
    'host': 'pO',
    'demo': [
        {'key': 'lb', 'init': 'load_bang', 'pos': (40, 40), 'w': 88, 'h': 46},
        {'key': 'cl', 'init': 'comment once, when the patch opens',
         'pos': (150, 50), 'w': 280, 'h': 30},
        {'key': 'ls', 'init': 'np.linspace 0.2 1 8', 'pos': (40, 110), 'w': 190, 'h': 150,
         'props': {'start': 0.2, 'stop': 1.0, 'steps': 8, 'dtype': 'float32'}},
        {'key': 'tr', 'init': 't array array', 'pos': (40, 280), 'w': 140, 'h': 46},
        {'key': 'ou', 'init': 'np.outer', 'pos': (40, 350), 'w': 150, 'h': 70},
        {'key': 'co', 'init': 'comment a ramp outer itself: every row a scaled copy of one row',
         'pos': (210, 365), 'w': 500, 'h': 30},
        {'key': 'out', 'init': 'out outer', 'pos': (40, 450), 'w': 150, 'h': 80},
    ],
    'links': [('lb', 'out', 'ls', ''), ('ls', 'linspace out', 'tr', ''),
              ('tr', 'array', 'ou', 'in 2', 1), ('tr', 'array', 'ou', 'in 1', 0),
              ('ou', 'outer_product', 'out', 'outer')],
}

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'A', 'init': 'vector 3 3', 'pos': (30, 130), 'w': 230, 'h': 175,
     'props': VEC([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])},
    {'key': 'c0', 'comment': True, 'text': 'a 3 x 3 matrix: edit its rows',
     'pos': (280, 135)},
    {'key': 'hA', 'init': 'heat_map', 'pos': (280, 175), 'w': 200, 'h': 180,
     'props': HM(180, 130, 3, 0.0, 1.0, fmt='none')},
    {'key': 'rk', 'init': 'np.linalg.matrix_rank', 'pos': (30, 380), 'w': 230, 'h': 70},
    {'key': 'i1', 'init': 'int', 'pos': (30, 470), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 'its rank: 3, 2, 1 or 0', 'pos': (180, 475)},
    {'key': 'pO', 'init': 'p make_outer', 'pos': (30, 560), 'w': 180, 'h': 70},
    {'key': 'rnd', 'init': 'np.rand 8 8', 'pos': (330, 540), 'w': 160, 'h': 160,
     'props': {'min': 0.0, 'max': 1.0, 'dim 0': 8, 'dim 1': 8, 'dtype': 'float32'}},
    {'key': 'hO', 'init': 'heat_map', 'pos': (30, 720), 'w': 240, 'h': 250,
     'props': HM(220, 200, 8, 0.0, 1.0, fmt='none')},
    {'key': 'hR', 'init': 'heat_map', 'pos': (330, 720), 'w': 240, 'h': 250,
     'props': HM(220, 200, 8, 0.0, 1.0, fmt='none')},
    {'key': 'rk2', 'init': 'np.linalg.matrix_rank', 'pos': (30, 990), 'w': 230, 'h': 70},
    {'key': 'rk3', 'init': 'np.linalg.matrix_rank', 'pos': (330, 990), 'w': 230, 'h': 70},
    {'key': 'i2', 'init': 'int', 'pos': (30, 1080), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'i3', 'init': 'int', 'pos': (330, 1080), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'left: a smooth picture, rank 1. right: noise, rank 8',
     'pos': (30, 1135)},
] + see_also(1180, ['np.linalg.det', 'np.outer', 'np.linalg.solve', 't.linalg.svd'])
links = [('lb', 'out', 'A', 'in'), ('A', 'out', 'hA', 'y'), ('A', 'out', 'rk', 'input'),
         ('rk', 'matrix rank', 'i1', ''),
         ('pO', 'outer', 'hO', 'y'), ('pO', 'outer', 'rk2', 'input'),
         ('rk2', 'matrix rank', 'i2', ''),
         ('lb', 'out', 'rnd', ''), ('rnd', 'random array', 'hR', 'y'),
         ('rnd', 'random array', 'rk3', 'input'), ('rk3', 'matrix rank', 'i3', '')]
print(build('np.linalg.matrix_rank', 'np.linalg.matrix_rank - how many independent directions',
            body, demo, links, demo_width=DW(640), text_width=760, text_height=th(body),
            subpatch=OUTER_SUB))

# ------------------------------------------------------------- np.linalg.inv
body = """Gives the inverse of a square matrix: the matrix that undoes it.

If a matrix turns, stretches or shears things, its inverse turns,
shrinks or un-shears them back. Multiplying a matrix by its inverse gives
the identity matrix - 1s down the diagonal, 0s everywhere else - which
changes nothing. That is the definition, and the demo checks it live.

Reach for it when you need the undoing matrix itself: to map screen
positions back into a camera's frame, to reverse a calibration, to keep
both directions of a transformation. To answer one question of the form
"which input gives this output?", np.linalg.solve is faster and more
accurate than inverting and multiplying.

NOT EVERY MATRIX HAS ONE:
A matrix whose determinant is 0 squashes space flat, and that cannot be
undone. Such a matrix is called singular; np.linalg.inv sends nothing for
it and the displays keep the last good answer. A matrix that is NEARLY
singular does invert, but into huge numbers - small errors in the input
become large errors in the output.

SYNTAX:
np.linalg.inv
(no arguments)

INPUTS and PARAMETERS:

input:
A square matrix (a list of lists is turned into an array). Each one that
arrives is inverted and sent. A matrix that is not square, a 1-D list,
or a singular matrix sends nothing (the reason is printed only in verbose
mode). A stack of square matrices is inverted one by one.

OUTPUTS:

inverse matrix:
The inverse, the same shape as the input, as floats.

READING THE DEMO:
Drag the numbers of the matrix. Its rows start [2 1] and [1 1], whose
inverse is [1 -1] and [-1 2]. The third picture is np.matmul of the
matrix and its inverse: the identity, blue 1s on the diagonal, pale 0s
off it - and it stays the identity however you change the matrix. Make
the rows parallel (bottom row 2 1) and the matrix is singular: no
inverse is sent, the middle picture keeps the last good one, and the
product of the new matrix with that old inverse is no longer the
identity - the sign that something went wrong.
The t array array sends to np.linalg.inv first (it fires right to left),
so the inverse is waiting in np.matmul's second inlet when the matrix
arrives at its first.

RELATED:
np.linalg.solve answers A x = b without forming the inverse.
np.linalg.det says whether an inverse exists (it must not be 0).
np.matmul applies a matrix. np.transpose is the inverse of a pure
rotation matrix, and much cheaper."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'A', 'init': 'vector 2 2', 'pos': (30, 130), 'w': 170, 'h': 150,
     'props': VEC([[2.0, 1.0], [1.0, 1.0]])},
    {'key': 'c0', 'comment': True, 'text': 'the matrix: drag any number',
     'pos': (220, 135)},
    {'key': 'tr', 'init': 't array array', 'pos': (30, 300), 'w': 140, 'h': 46},
    {'key': 'inv', 'init': 'np.linalg.inv', 'pos': (240, 370), 'w': 170, 'h': 70},
    {'key': 'mm', 'init': 'np.matmul', 'pos': (440, 460), 'w': 150, 'h': 70},
    {'key': 'hA', 'init': 'heat_map', 'pos': (30, 560), 'w': 190, 'h': 200,
     'props': HM(170, 150, 2, -3.0, 3.0, color='red-blue')},
    {'key': 'hI', 'init': 'heat_map', 'pos': (240, 560), 'w': 190, 'h': 200,
     'props': HM(170, 150, 2, -3.0, 3.0, color='red-blue')},
    {'key': 'hP', 'init': 'heat_map', 'pos': (450, 560), 'w': 190, 'h': 200,
     'props': HM(170, 150, 2, -3.0, 3.0, color='red-blue')},
    {'key': 'c1', 'comment': True,
     'text': 'the matrix          its inverse         matrix x inverse = identity',
     'pos': (30, 775)},
] + see_also(830, ['np.linalg.solve', 'np.linalg.det', 'np.matmul', 'np.transpose'])
links = [('lb', 'out', 'A', 'in'), ('A', 'out', 'tr', ''), ('A', 'out', 'hA', 'y'),
         ('tr', 'array', 'inv', 'input', 1), ('inv', 'inverse matrix', 'hI', 'y'),
         ('inv', 'inverse matrix', 'mm', 'in 2'),
         ('tr', 'array', 'mm', 'in 1', 0), ('mm', 'mat mul result', 'hP', 'y')]
print(build('np.linalg.inv', 'np.linalg.inv - the matrix that undoes a matrix',
            body, demo, links, demo_width=DW(680), text_width=760, text_height=th(body)))

# ----------------------------------------------------------- np.linalg.solve
body = """Solves A x = b: given a matrix A and a result b, finds the x that A
turns into b.

It answers "which input gives this output?" for any linear relation, and
it answers a set of simultaneous equations in one step. If 2 apples and 1
pear cost 5, and 1 apple and 3 pears cost 10, then A is the table of
quantities [2 1] [1 3], b is the prices [5 10], and x - the price of one
apple and one pear - comes out as [1 3].

It gives the same answer as np.linalg.inv of A times b, but faster and
more accurately, because it never forms the inverse.

SYNTAX:
np.linalg.solve
(no arguments)

INPUTS and PARAMETERS:

A matrix:
The square matrix A - one row per equation, one column per unknown.
Receiving it solves and sends. It must be square; a matrix that is not
sends nothing.

b:
The right-hand side: a list with one number per row of A. It is only
stored, so it must arrive BEFORE A, and nothing is sent until it has.
b may also be a matrix with one column per separate problem, all solved
at once against the same A.

If A is singular (its determinant is 0 - the equations contradict or
repeat each other) there is no single answer and nothing is sent; nor
is anything sent if b does not match A's size. The reason is printed
only in verbose mode.

OUTPUTS:

x:
The solution, the same shape as b.

READING THE DEMO:
The left vector node is A, the right one b; drag any number in either.
x is the answer, and the check underneath multiplies A by x
(np.matmul) - it always gives b back. Make A's rows parallel (bottom row
4 2) and no x is sent: those two equations say the same thing twice and
cannot pin down two unknowns. x keeps the last answer, and the check,
multiplying the new A by that old x, no longer matches b.
Wiring: b's 't bang array' stores b first, then bangs A so it re-sends.
A's 't array array' reaches np.linalg.solve first, so x is waiting in
np.matmul's second inlet when A arrives at its first.

RELATED:
np.linalg.inv gives the undoing matrix itself. np.matmul applies a
matrix - here, to check the answer. np.linalg.det tells you in advance
whether there is a unique answer. np.dot is matrix multiplication for
2-D arrays too."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'A', 'init': 'vector 2 2', 'pos': (30, 130), 'w': 170, 'h': 150,
     'props': VEC([[2.0, 1.0], [1.0, 3.0]])},
    {'key': 'c0', 'comment': True, 'text': 'A: 2 apples + 1 pear, 1 apple + 3 pears',
     'pos': (30, 290)},
    {'key': 'b', 'init': 'vector 2', 'pos': (380, 130), 'w': 120, 'h': 150,
     'props': VEC([5.0, 10.0])},
    {'key': 'c1', 'comment': True, 'text': 'b: the prices, 5 and 10', 'pos': (380, 290)},
    {'key': 'tb', 'init': 't bang array', 'pos': (380, 330), 'w': 140, 'h': 46},
    {'key': 'tA', 'init': 't array array', 'pos': (30, 400), 'w': 140, 'h': 46},
    {'key': 'sv', 'init': 'np.linalg.solve', 'pos': (240, 470), 'w': 180, 'h': 70},
    {'key': 'lx', 'init': 'list', 'pos': (240, 560), 'w': 200, 'h': 42, 'props': LIST},
    {'key': 'c2', 'comment': True, 'text': 'x: one apple costs 1, one pear 3',
     'pos': (240, 610)},
    {'key': 'mm', 'init': 'np.matmul', 'pos': (30, 660), 'w': 150, 'h': 70},
    {'key': 'lc', 'init': 'list', 'pos': (30, 750), 'w': 200, 'h': 42, 'props': LIST},
    {'key': 'c3', 'comment': True, 'text': 'check: A times x gives b back', 'pos': (30, 800)},
] + see_also(850, ['np.linalg.inv', 'np.matmul', 'np.linalg.det', 'np.dot'])
links = [('lb', 'out', 'b', 'in'),
         ('b', 'out', 'tb', ''), ('tb', 'array', 'sv', 'b', 1), ('tb', 'bang', 'A', 'in', 0),
         ('A', 'out', 'tA', ''), ('tA', 'array', 'sv', 'A matrix', 1),
         ('sv', 'x', 'lx', ''), ('sv', 'x', 'mm', 'in 2'),
         ('tA', 'array', 'mm', 'in 1', 0), ('mm', 'mat mul result', 'lc', '')]
print(build('np.linalg.solve', 'np.linalg.solve - find the x that gives b',
            body, demo, links, demo_width=DW(640), text_width=760, text_height=th(body)))

# ------------------------------------------------------------ np.linalg.norm
body = """Gives the length of a vector: the square root of the sum of its
squared values.

np.linalg.norm and euclidean_distance are two names for the same node,
with the same settings. euclidean_distance is named for its commonest
use - but it still takes ONE input: feed it the DIFFERENCE between two
positions (with a - node) and the length of that difference is the
distance between them.

[3 4] has length 5; [1 1 1] has length 1.73. Reach for it to measure how
far something has moved, how fast (the length of a velocity), how strong
a force is regardless of its direction, or how far every point of a set
is from a target.

ONE LENGTH, OR ONE PER VECTOR - THE axis:
With axis None the whole array counts as one long vector, and you get a
single number. With an axis set, the array is read as a set of vectors
lying along that axis, and you get one length for each. For 100 points
stored as 100 rows of 3 (x, y, z), axis 1 gives 100 distances. For a
grid of points (rows, columns, 3), axis 2 gives a picture of lengths.

SYNTAX:
np.linalg.norm
euclidean_distance
(no arguments - set the axis with the menu)

INPUTS and PARAMETERS:

input:
The array (a list is turned into one). Each one that arrives is measured
and sent.

axis (menu, also an inlet):
None (the default) or the axis the vectors lie along. The menu offers
as many axes as the last array had. An axis past the last one is taken
as the last. Changing it does NOT send by itself - the next array uses
it; in the demo a metro re-sends the grid ten times a second.

OUTPUTS:

norm:
The length - a float with axis None, otherwise an array one dimension
smaller than the input.

READING THE DEMO:
Top: drag the three numbers of the vector; [3 4 0] has length 5.
Bottom: make_points sends a 16 x 16 grid of points, x and y running
from -1 to 1 (each point is x, y, 0). The two sliders set a centre; -
subtracts it from every point; euclidean_distance with axis 2 measures
each difference. The picture is the distance from the centre: dark at
the centre, brighter further out. Move the sliders and the dark spot
follows.

RELATED:
np.distance_from_target does the subtract-and-measure for you, against
one or several targets. norm (no np.) rescales a vector to length 1.
np.dot of a vector with itself is its length squared. t.length is the
PyTorch version."""

POINTS_SUB = {
    'name': 'make_points',
    'host': 'pP',
    'demo': [
        {'key': 'met', 'init': 'metro 100', 'pos': (40, 40), 'w': 129, 'h': 70,
         'props': METRO},
        {'key': 'cm', 'init': 'comment re-sends the grid ten times a second',
         'pos': (190, 50), 'w': 340, 'h': 30},
        {'key': 'gr', 'init': 'np.grid -1 1 -1 1 16 16', 'pos': (40, 130), 'w': 220, 'h': 230,
         'props': {'min x': -1.0, 'max x': 1.0, 'min y': -1.0, 'max y': 1.0,
                   'divisions x': 16, 'divisions y': 16, 'z': 0.0}},
        {'key': 'cg', 'init': 'comment 16 x 16 points, each x, y, 0: shape (16, 16, 3)',
         'pos': (280, 145), 'w': 440, 'h': 30},
        {'key': 'out', 'init': 'out points', 'pos': (40, 390), 'w': 150, 'h': 80},
    ],
    'links': [('met', '', 'gr', ''), ('gr', 'grid points', 'out', 'points')],
}

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'v', 'init': 'vector 3', 'pos': (30, 130), 'w': 120, 'h': 175,
     'props': VEC([3.0, 4.0, 0.0])},
    {'key': 'c0', 'comment': True, 'text': 'a vector: drag its numbers', 'pos': (170, 135)},
    {'key': 'nm', 'init': 'np.linalg.norm', 'pos': (230, 180), 'w': 170, 'h': 100,
     'props': {'axis': 'None'}},
    {'key': 'f1', 'init': 'float', 'pos': (420, 190), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'its length: 5', 'pos': (420, 240)},
    {'key': 'pP', 'init': 'p make_points', 'pos': (30, 360), 'w': 180, 'h': 70},
    {'key': 'sx', 'init': 'slider', 'pos': (250, 360), 'w': 200, 'h': 60,
     'props': {'min': -1.0, 'max': 1.0, 'format': '%.2f', 'width': 180, '': 0.3}},
    {'key': 'sy', 'init': 'slider', 'pos': (250, 430), 'w': 200, 'h': 60,
     'props': {'min': -1.0, 'max': 1.0, 'format': '%.2f', 'width': 180, '': -0.2}},
    {'key': 'c2', 'comment': True, 'text': 'the centre: x and y', 'pos': (470, 375)},
    {'key': 'pk', 'init': 'pak f f 0', 'pos': (250, 510), 'w': 150, 'h': 100},
    {'key': 'sub', 'init': '-', 'pos': (30, 640), 'w': 130, 'h': 70},
    {'key': 'c3', 'comment': True, 'text': 'every point minus the centre', 'pos': (180, 645)},
    {'key': 'ed', 'init': 'euclidean_distance', 'pos': (30, 730), 'w': 200, 'h': 100,
     'props': {'axis': '2'}},
    {'key': 'c4', 'comment': True, 'text': 'axis 2: one length per point', 'pos': (250, 735)},
    {'key': 'hD', 'init': 'heat_map', 'pos': (30, 850), 'w': 300, 'h': 300,
     'props': HM(280, 250, 16, 0.0, 2.0, fmt='none')},
    {'key': 'c5', 'comment': True, 'text': 'distance from the centre, 0 (dark) to 2',
     'pos': (350, 855)},
] + see_also(1180, ['np.distance_from_target', 'norm', 'np.dot', 't.length'])
links = [('lb', 'out', 'v', 'in'), ('v', 'out', 'nm', 'input'), ('nm', 'norm', 'f1', ''),
         ('lb', 'out', 'sx', ''), ('lb', 'out', 'sy', ''),
         ('sy', 'float out', 'pk', 'float', 0, 1), ('sx', 'float out', 'pk', 'float', 0, 0),
         ('pk', 'out', 'sub', '###operand'),
         ('pP', 'points', 'sub', 'in'), ('sub', 'result', 'ed', 'input'),
         ('ed', 'norm', 'hD', 'y')]
print(build('np.linalg.norm', 'np.linalg.norm / euclidean_distance - the length of a vector',
            body, demo, links, demo_width=DW(640), text_width=760, text_height=th(body),
            subpatch=POINTS_SUB))

# -------------------------------------------------------------------- np.dot
body = """Gives the dot product of two vectors: multiply them element by
element and add up - one number that says how much they point the same
way.

For two vectors of length 1 the dot product is the cosine of the angle
between them: 1 when they point the same way, 0 at right angles, -1 when
opposite. For longer vectors it is scaled by both lengths. With one of
them a unit direction, it is the PROJECTION: how far the other reaches
along that direction - the component of a movement along an axis, of a
force along a surface, of a pose along a learned direction.

Reach for it to compare directions (similarity), to project onto an axis,
or to test for right angles (dot 0).

FOR ARRAYS OF MORE THAN ONE DIMENSION:
2-D with 2-D is matrix multiplication, the same as np.matmul. A set of
vectors (N rows of 3) with one vector of 3 gives N dot products - every
row against the same direction at once.

SYNTAX:
np.dot
(no arguments)

INPUTS and PARAMETERS:

in 1:
The first vector (a list is turned into an array). Receiving it
calculates and sends.

in 2:
The second vector. It is only stored, so it must arrive BEFORE in 1;
nothing is sent until it has. The lengths must match (for 2-D, the
columns of in 1 must match the rows of in 2), or an error is printed and
nothing is sent.

OUTPUTS:

dot product:
A float for two vectors; an array for the larger cases.

READING THE DEMO:
in 1 is the vector a, which starts as [1 0], pointing along x; drag its
numbers. in 2 is b, a vector of length 1 at the angle on the slider
(make_direction turns the angle into [cos, sin]). The dot product reads
1 at 0 degrees, 0 at 90, -1 at 180 - the cosine of the angle between
them. Make a longer, [2 0], and every reading doubles. 'b' goes through
't bang array': the array is stored in in 2 first, then the bang makes
a re-send, so turning the slider updates the answer.

RELATED:
np.inner is the same for vectors but pairs rows with rows for 2-D
arrays. np.matmul is the matrix product. np.linalg.norm gives a length
(the square root of a vector dotted with itself). np.cross gives the
vector at right angles to both. t.cosine_similarity gives the cosine
directly, whatever the lengths."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'a', 'init': 'vector 2', 'pos': (30, 130), 'w': 120, 'h': 150,
     'props': VEC([1.0, 0.0])},
    {'key': 'c0', 'comment': True, 'text': 'a: drag its numbers', 'pos': (30, 290)},
    {'key': 'sl', 'init': 'slider 360.0', 'pos': (330, 62), 'w': 220, 'h': 60,
     'props': {'min': 0.0, 'max': 360.0, 'format': '%.0f', 'width': 200, '': 60.0}},
    {'key': 'c1', 'comment': True, 'text': 'the angle of b, in degrees', 'pos': (330, 130)},
    {'key': 'pB', 'init': 'p make_direction', 'pos': (330, 170), 'w': 200, 'h': 70},
    {'key': 'lb2', 'init': 'list', 'pos': (330, 260), 'w': 220, 'h': 42, 'props': LIST},
    {'key': 'c2', 'comment': True, 'text': 'b: length 1, at that angle', 'pos': (330, 310)},
    {'key': 'tb', 'init': 't bang array', 'pos': (330, 350), 'w': 140, 'h': 46},
    {'key': 'dt', 'init': 'np.dot', 'pos': (30, 420), 'w': 150, 'h': 70},
    {'key': 'f1', 'init': 'float', 'pos': (30, 510), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'a . b: 1 together, 0 at right angles, -1 opposite',
     'pos': (180, 515)},
] + see_also(580, ['np.inner', 'np.matmul', 'np.linalg.norm', 'np.cross',
                   't.cosine_similarity'])
links = [('lb', 'out', 'sl', ''), ('sl', 'float out', 'pB', 'angle'),
         ('pB', 'direction', 'lb2', ''), ('pB', 'direction', 'tb', ''),
         ('tb', 'array', 'dt', 'in 2', 1), ('tb', 'bang', 'a', 'in', 0),
         ('a', 'out', 'dt', 'in 1'), ('dt', 'dot product', 'f1', '')]
print(build('np.dot', 'np.dot - how much two vectors point the same way',
            body, demo, links, demo_width=DW(640), text_width=760, text_height=th(body),
            subpatch=direction_sub('pB')))

# ------------------------------------------------------------------ np.inner
body = """Gives the inner product: for two vectors, the dot product; for two sets
of vectors, every vector of the first dotted with every vector of the
second.

For plain vectors np.inner and np.dot give the same number - multiply
element by element and add up. They differ for 2-D arrays. np.inner
treats each ROW as a vector and pairs rows with rows: a set of 12
directions (12 rows of 2) against one direction gives 12 numbers; a set
against itself gives a 12 x 12 table of how alike every pair is. np.dot
of two 2-D arrays is matrix multiplication instead (rows with columns).

Reach for it to score many vectors against one at once - which of these
poses is closest to this one, which sensor points most nearly up - or
to build a similarity table of a set of vectors.

SYNTAX:
np.inner
(no arguments)

INPUTS and PARAMETERS:

in 1:
The first vector or set of vectors (a list is turned into an array).
Receiving it calculates and sends.

in 2:
The second. It is only stored, so it must arrive BEFORE in 1; nothing is
sent until it has. Both must have the same LAST dimension (the vector
length), or an error is printed and nothing is sent.

OUTPUTS:

inner_product:
A float for two vectors. Otherwise the shape is in 1's leading shape
followed by in 2's: (12, 2) with (2,) gives 12 values; (12, 2) with
(12, 2) gives 12 x 12.

READING THE DEMO:
make_directions sends 12 directions of length 1, 30 degrees apart (the
picture top left: each row is cos and sin of one angle). The slider
sets one more direction. Middle: np.inner of the 12 with that one - a
strip of 12 cosines, blue where a direction agrees with the slider, red
where it points away. Turn the slider and the blue band moves round.
Bottom: np.inner of the 12 with themselves - every pair compared. The
diagonal is 1 (each agrees with itself), and the stripes show that
directions 6 apart (180 degrees) are opposite.

RELATED:
np.dot is the same for vectors and is matrix multiplication for 2-D.
np.outer multiplies every element by every element, without adding up.
np.matmul is the matrix product. t.cosine_similarity compares
directions whatever their lengths."""

demo = [
    {'key': 'pD', 'init': 'p make_directions', 'pos': (30, 62), 'w': 200, 'h': 70},
    {'key': 'hD', 'init': 'heat_map', 'pos': (30, 150), 'w': 150, 'h': 290,
     'props': HM(130, 240, 12, -1.0, 1.0, color='red-blue')},
    {'key': 'c0', 'comment': True, 'text': '12 directions: cos, sin', 'pos': (30, 450)},
    {'key': 'sl', 'init': 'slider 360.0', 'pos': (330, 62), 'w': 220, 'h': 60,
     'props': {'min': 0.0, 'max': 360.0, 'format': '%.0f', 'width': 200, '': 90.0}},
    {'key': 'c1', 'comment': True, 'text': 'one more direction, in degrees', 'pos': (330, 130)},
    {'key': 'pB', 'init': 'p make_direction', 'pos': (330, 170), 'w': 200, 'h': 70},
    {'key': 'tr', 'init': 't array array', 'pos': (240, 490), 'w': 140, 'h': 46},
    {'key': 'in1', 'init': 'np.inner', 'pos': (30, 560), 'w': 150, 'h': 70},
    {'key': 'h1', 'init': 'heat_map', 'pos': (30, 650), 'w': 560, 'h': 110,
     'props': HM(540, 60, 1, -1.0, 1.0, color='red-blue')},
    {'key': 'c2', 'comment': True, 'text': 'the 12 against the slider: 12 cosines',
     'pos': (30, 770)},
    {'key': 'in2', 'init': 'np.inner', 'pos': (30, 810), 'w': 150, 'h': 70},
    {'key': 'h2', 'init': 'heat_map', 'pos': (30, 900), 'w': 320, 'h': 330,
     'props': HM(300, 280, 12, -1.0, 1.0, fmt='none', color='red-blue')},
    {'key': 'c3', 'comment': True, 'text': 'the 12 against themselves: 12 x 12',
     'pos': (370, 905)},
] + see_also(1260, ['np.dot', 'np.outer', 'np.matmul', 't.cosine_similarity'])
links = [('pD', 'directions', 'hD', 'y'), ('pD', 'directions', 'tr', ''),
         ('lb', 'out', 'sl', ''),
         ('sl', 'float out', 'pB', 'angle'), ('pB', 'direction', 'in1', 'in 2'),
         ('tr', 'array', 'in1', 'in 1', 1),
         ('tr', 'array', 'in2', 'in 2', 1), ('tr', 'array', 'in2', 'in 1', 0),
         ('in1', 'inner_product', 'h1', 'y'), ('in2', 'inner_product', 'h2', 'y')]
demo.insert(0, {'key': 'lb', 'init': 'load_bang', 'pos': (580, 62), 'w': 88, 'h': 46})
print(build('np.inner', 'np.inner - every vector against every vector',
            body, demo, links, demo_width=DW(700), text_width=760, text_height=th(body),
            subpatch=[DIRS_SUB, direction_sub('pB')]))

# ------------------------------------------------------------------ np.outer
body = """Multiplies every element of one array by every element of another,
giving a table: row i, column j holds in 1's value i times in 2's value
j.

Two lists of 12 and 16 values give a 12 x 16 array. Nothing is added up -
that is the difference from the dot or inner product, which collapse the
same multiplications into one number. Every row of the result is in 2
scaled by one value of in 1, and every column is in 1 scaled by one value
of in 2.

Reach for it to build a 2-D picture from two 1-D profiles - a vignette
from a horizontal and a vertical fall-off, a gradient, a separable
filter kernel - or a table of every pairing of two sets of weights.
Any outer product has rank 1 (see np.linalg.matrix_rank): however
detailed it looks, it is made of one row pattern.

SYNTAX:
np.outer
(no arguments)

INPUTS and PARAMETERS:

in 1:
The values for the rows (a list is turned into an array). Receiving it
calculates and sends.

in 2:
The values for the columns. It is only stored, so it must arrive BEFORE
in 1; nothing is sent until it has.
Arrays of more than one dimension are flattened first, so the result is
always 2-D: (length of in 1) x (length of in 2). The two need not be the
same length.

OUTPUTS:

outer_product:
The 2-D table of products.

READING THE DEMO:
in 1 is a ramp of 12 values from -1 to 1 (drag start and stop on its
np.linspace - a metro re-sends it); in 2 is a hump of 16 values, rising
from 0 to 1 and back (sin of 0 to 180 degrees, made once in make_hump).
The picture is the outer product, 12 rows by 16 columns: every row is
the hump, scaled by that row's ramp value - red (negative) at the top,
pale through the middle row, blue at the bottom. Drag stop to -1 and
the picture turns over; set start and stop both to 1 and every row is
the same hump.

RELATED:
np.inner and np.dot multiply AND add up, giving a number. np.matmul of a
column (12 x 1) and a row (1 x 16) gives the same table. np.linalg.matrix_rank
of any outer product is 1."""

HUMP_SUB = {
    'name': 'make_hump',
    'host': 'pH',
    'demo': [
        {'key': 'lb', 'init': 'load_bang', 'pos': (40, 40), 'w': 88, 'h': 46},
        {'key': 'cl', 'init': 'comment once, when the patch opens',
         'pos': (150, 50), 'w': 280, 'h': 30},
        {'key': 'ls', 'init': 'np.linspace 0 180 16', 'pos': (40, 110), 'w': 190, 'h': 150,
         'props': {'start': 0.0, 'stop': 180.0, 'steps': 16, 'dtype': 'float32'}},
        {'key': 'si', 'init': 'sin', 'pos': (40, 280), 'w': 120, 'h': 70,
         'props': {'degrees': True}},
        {'key': 'cs', 'init': 'comment sin of 0 to 180 degrees: 0 up to 1 and back to 0',
         'pos': (180, 295), 'w': 460, 'h': 30},
        {'key': 'out', 'init': 'out hump', 'pos': (40, 380), 'w': 150, 'h': 80},
    ],
    'links': [('lb', 'out', 'ls', ''), ('ls', 'linspace out', 'si', 'in'),
              ('si', 'out', 'out', 'hump')],
}

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70, 'props': METRO},
    {'key': 'ls', 'init': 'np.linspace -1 1 12', 'pos': (30, 150), 'w': 190, 'h': 150,
     'props': {'start': -1.0, 'stop': 1.0, 'steps': 12, 'dtype': 'float32'}},
    {'key': 'c0', 'comment': True, 'text': 'in 1: drag start and stop', 'pos': (240, 155)},
    {'key': 'h1', 'init': 'heat_map', 'pos': (30, 320), 'w': 560, 'h': 110,
     'props': HM(540, 60, 1, -1.0, 1.0, color='red-blue')},
    {'key': 'c1', 'comment': True, 'text': 'in 1: 12 values, -1 to 1', 'pos': (30, 440)},
    {'key': 'pH', 'init': 'p make_hump', 'pos': (30, 480), 'w': 180, 'h': 70},
    {'key': 'h2', 'init': 'heat_map', 'pos': (30, 570), 'w': 560, 'h': 110,
     'props': HM(540, 60, 1, -1.0, 1.0, color='red-blue')},
    {'key': 'c2', 'comment': True, 'text': 'in 2: 16 values, a hump from 0 to 1 and back',
     'pos': (30, 690)},
    {'key': 'ou', 'init': 'np.outer', 'pos': (30, 730), 'w': 150, 'h': 70},
    {'key': 'hO', 'init': 'heat_map', 'pos': (30, 820), 'w': 420, 'h': 340,
     'props': HM(400, 290, 12, -1.0, 1.0, fmt='none', color='red-blue')},
    {'key': 'c3', 'comment': True, 'text': 'in 1 x in 2: 12 rows by 16 columns',
     'pos': (470, 825)},
    {'key': 'sh', 'init': 'np.shape', 'pos': (470, 880), 'w': 110, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (470, 970), 'w': 120, 'h': 42, 'props': LIST},
] + see_also(1190, ['np.inner', 'np.dot', 'np.matmul', 'np.linalg.matrix_rank'])
links = [('met', '', 'ls', ''), ('ls', 'linspace out', 'h1', 'y'),
         ('pH', 'hump', 'h2', 'y'), ('pH', 'hump', 'ou', 'in 2'),
         ('ls', 'linspace out', 'ou', 'in 1'), ('ou', 'outer_product', 'hO', 'y'),
         ('ou', 'outer_product', 'sh', 'np in'), ('sh', 'shape', 'l1', '')]
print(build('np.outer', 'np.outer - every element times every element',
            body, demo, links, demo_width=DW(640), text_width=760, text_height=th(body),
            subpatch=HUMP_SUB))

# ------------------------------------------------------------------ np.cross
body = """Gives the cross product of two 3-D vectors: a third vector at right
angles to both.

Its direction is perpendicular to the plane the two inputs lie in, and
its length is the area of the parallelogram they span - largest when
they are at right angles, 0 when they are parallel (then there is no
single plane, and no perpendicular). Order matters: in 1 x in 2 points
the opposite way from in 2 x in 1 (the right-hand rule: fingers from in
1 towards in 2, thumb along the result).

Reach for it to find a surface normal from two edges of a triangle, the
axis to turn about to get from one direction to another, or which way
is "up" given a forward and a sideways direction.

SYNTAX:
np.cross
(no arguments)

INPUTS and PARAMETERS:

in 1:
The first vector, 3 numbers (a list is turned into an array). Receiving
it calculates and sends.

in 2:
The second vector. It is only stored, so it must arrive BEFORE in 1;
nothing is sent until it has.
Sets of vectors work row by row: N rows of 3 with N rows of 3 (or with
one vector of 3) give N cross products. Two 2-D vectors give a single
number (the z of the result) and numpy warns that this form is going
away. Other lengths are an error: it is printed and nothing is sent.

OUTPUTS:

cross product:
The perpendicular vector, 3 numbers (N rows of 3 for sets).

READING THE DEMO:
a and b start as the x and y directions, [1 0 0] and [0 1 0]; their
cross product is [0 0 1], the z direction. Drag any of their numbers.
The two np.dot nodes check the result against each input, and both stay
at 0 - the cross product is always at right angles to a and to b.
np.linalg.norm gives its length: 1 for x and y, smaller as you swing b
towards a, 0 when they are parallel.
Wiring: b goes through 't bang array' - stored first, then the bang
makes a re-send - and a's 't array array' stores a in the first check
before the cross product is worked out.

RELATED:
np.dot measures how much two vectors agree; the cross product is what
they do not share. np.linalg.norm gives the cross product's length (the
area). np.linalg.det of the 3 x 3 matrix with rows a, b, c is the
volume they span - the dot of c with a x b. np.outer is unrelated
despite the name."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'a', 'init': 'vector 3', 'pos': (30, 130), 'w': 120, 'h': 175,
     'props': VEC([1.0, 0.0, 0.0])},
    {'key': 'c0', 'comment': True, 'text': 'a', 'pos': (160, 135)},
    {'key': 'b', 'init': 'vector 3', 'pos': (330, 130), 'w': 120, 'h': 175,
     'props': VEC([0.0, 1.0, 0.0])},
    {'key': 'c1', 'comment': True, 'text': 'b: drag any number', 'pos': (460, 135)},
    {'key': 'tb', 'init': 't bang array', 'pos': (330, 330), 'w': 140, 'h': 46},
    {'key': 'ta', 'init': 't array array', 'pos': (30, 330), 'w': 140, 'h': 46},
    {'key': 'cr', 'init': 'np.cross', 'pos': (30, 410), 'w': 150, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (30, 500), 'w': 220, 'h': 42, 'props': LIST},
    {'key': 'c2', 'comment': True, 'text': 'a x b: at right angles to both', 'pos': (270, 505)},
    {'key': 'da', 'init': 'np.dot', 'pos': (30, 580), 'w': 150, 'h': 70},
    {'key': 'db', 'init': 'np.dot', 'pos': (220, 580), 'w': 150, 'h': 70},
    {'key': 'nm', 'init': 'np.linalg.norm', 'pos': (410, 580), 'w': 170, 'h': 100,
     'props': {'axis': 'None'}},
    {'key': 'fa', 'init': 'float', 'pos': (30, 700), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'fb', 'init': 'float', 'pos': (220, 700), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'fn', 'init': 'float', 'pos': (410, 700), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True,
     'text': '(a x b) . a = 0      (a x b) . b = 0      length = area',
     'pos': (30, 755)},
] + see_also(800, ['np.dot', 'np.linalg.norm', 'np.linalg.det', 'np.outer'])
links = [('lb', 'out', 'b', 'in'),
         ('b', 'out', 'tb', ''), ('tb', 'array', 'cr', 'in 2', 1),
         ('tb', 'array', 'db', 'in 2', 1), ('tb', 'bang', 'a', 'in', 0),
         ('a', 'out', 'ta', ''), ('ta', 'array', 'da', 'in 2', 1),
         ('ta', 'array', 'cr', 'in 1', 0),
         ('cr', 'cross product', 'l1', ''), ('cr', 'cross product', 'da', 'in 1'),
         ('cr', 'cross product', 'db', 'in 1'), ('cr', 'cross product', 'nm', 'input'),
         ('da', 'dot product', 'fa', ''), ('db', 'dot product', 'fb', ''),
         ('nm', 'norm', 'fn', '')]
print(build('np.cross', 'np.cross - the vector at right angles to two others',
            body, demo, links, demo_width=DW(640), text_width=760, text_height=th(body)))

# ----------------------------------------------------------------- np.matmul
body = """Multiplies two matrices: each value of the result is a row of in 1
dotted with a column of in 2.

This is how a matrix is APPLIED. A matrix times a set of points moves
the points (turns, stretches, shears them); two transformation matrices
multiplied together make one matrix that does both. In the result, row
i column j is row i of in 1 times column j of in 2, added up - so each
column of the result is a MIX of in 1's columns, in the proportions
given by that column of in 2.

THE SHAPES MUST FIT:
(rows x n) times (n x columns) gives (rows x columns): the columns of in
1 must equal the rows of in 2, or an error is printed and nothing is
sent. Order matters - in 1 x in 2 is generally not in 2 x in 1, and may
not even fit. A 1-D list is treated as a single vector: on the left it
is a row, on the right a column, and that dimension is dropped from the
result. Stacks of matrices (3-D arrays) are multiplied pair by pair.
Plain numbers are refused - use * to scale.

SYNTAX:
np.matmul
(no arguments)

INPUTS and PARAMETERS:

in 1:
The left matrix (a list of lists is turned into an array). Receiving it
calculates and sends.

in 2:
The right matrix. It is only stored, so it must arrive BEFORE in 1;
nothing is sent until it has.

OUTPUTS:

mat mul result:
The product.

READING THE DEMO:
in 1 is the F, 7 by 6 (made in make_F). in 2 is a 6 x 6 matrix with a
single 1 in each column: the identity matrix (from make_identity),
rolled sideways by np.roll. Column j of the result takes the column of
the F that column j of the matrix has its 1 in. At shifts 0 the matrix
is the identity and the F comes out unchanged; step shifts and the
diagonal of 1s slides over, and the F slides with it, wrapping round.
The (7 x 6) times (6 x 6) gives 7 x 6 again. Any 6 x 6 matrix would do:
a column holding 0.5 and 0.5 would blend two columns of the F.

RELATED:
np.dot does the same for 2-D arrays. np.linalg.inv gives the matrix that
undoes another (their product is the identity). np.transpose swaps rows
and columns, often needed to make shapes fit. np.outer is the product of
a column and a row. np.roll, used here to build the matrix, does the
same shift directly."""

demo = [
    {'key': 'pF', 'init': 'p make_F', 'pos': (30, 62), 'w': 160, 'h': 70},
    {'key': 'pI', 'init': 'p make_identity', 'pos': (330, 62), 'w': 200, 'h': 70},
    {'key': 'ro', 'init': 'np.roll 0 1', 'pos': (330, 160), 'w': 150, 'h': 100,
     'props': {'shifts': 0, 'axis': 1}},
    {'key': 'c0', 'comment': True, 'text': 'step shifts', 'pos': (500, 165)},
    {'key': 'hF', 'init': 'heat_map', 'pos': (30, 290), 'w': 200, 'h': 260,
     'props': HM(180, 210, 7, fmt='none')},
    {'key': 'hM', 'init': 'heat_map', 'pos': (330, 290), 'w': 200, 'h': 230,
     'props': HM(180, 180, 6, fmt='none', color='greys')},
    {'key': 'c1', 'comment': True, 'text': 'in 1: the F, 7 x 6       in 2: 6 x 6, one 1 per column',
     'pos': (30, 560)},
    {'key': 'mm', 'init': 'np.matmul', 'pos': (30, 600), 'w': 150, 'h': 70},
    {'key': 'hO', 'init': 'heat_map', 'pos': (30, 690), 'w': 200, 'h': 260,
     'props': HM(180, 210, 7, fmt='none')},
    {'key': 'sh', 'init': 'np.shape', 'pos': (330, 690), 'w': 110, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (330, 780), 'w': 120, 'h': 42, 'props': LIST},
    {'key': 'c2', 'comment': True, 'text': 'F x matrix: the F with its columns moved',
     'pos': (30, 960)},
] + see_also(1010, ['np.dot', 'np.linalg.inv', 'np.transpose', 'np.outer', 'np.roll'])
links = [('pI', 'identity', 'ro', 'input'), ('ro', 'rolled array', 'hM', 'y'),
         ('ro', 'rolled array', 'mm', 'in 2'),
         ('pF', 'F', 'hF', 'y'), ('pF', 'F', 'mm', 'in 1'),
         ('mm', 'mat mul result', 'hO', 'y'), ('mm', 'mat mul result', 'sh', 'np in'),
         ('sh', 'shape', 'l1', '')]
print(build('np.matmul', 'np.matmul - apply one matrix to another',
            body, demo, links, demo_width=DW(640), text_width=760, text_height=th(body),
            subpatch=[F_SUB, IDENT_SUB]))
