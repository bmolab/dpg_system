"""torch lengths, distances, decompositions and losses: one page per node.

t.length, t.dist, t.cdist, t.cosine_similarity, t.normalize, t.linalg.svd,
t.linalg.qr, t.linalg.eig, t.linalg.pca_low_rank, t.mse_loss, t.l1_loss,
t.cross_entropy_loss. These were the family pages t.distance, t.linalg.svd
(all four decompositions) and t.mse_loss (all three losses); each node now
has its own page whose demo shows the input and the result side by side, with
a SEE ALSO row under it.

t.length and t.dist share a class but are NOT aliases: t.length measures one
tensor from zero, t.dist measures between two tensors (a second inlet, a
different outlet), so each has its own page.

Points you move are pairs of sliders packed into a list. Where a node's second
inlet only stores, a `t bang list` (or a metro) makes the first input re-send
so that moving either point updates the answer. Fixed demo matrices are made
inside `p make_<thing>` subpatchers: a metro re-sends a message of numbers
folded by np.reshape, so a property changed on the page shows at once.
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


def SLIDER(lo, hi, value, width=180, fmt='%.2f'):
    return {'min': float(lo), 'max': float(hi), 'format': fmt, 'width': width, '': float(value)}


def VEC(rows):
    """Properties for a `vector` node holding these rows (numbers or lists)."""
    p = {'output type': 'numpy', 'component count': len(rows),
         'column count': len(rows[0]) if isinstance(rows[0], list) else 1,
         'number format': '%.2f'}
    for i, r in enumerate(rows):
        p['[' + str(i) + ']'] = r
    return p


SA_END = [0]


def see_also(y, stems, x=30, right=640):
    """A SEE ALSO row: one help_link per related page, wrapping onto a second
    row rather than running under the prose."""
    row = []
    x0 = x
    far = 0
    for s in stems:
        w = int(len(s) * 7.7) + 30
        if x > x0 and x + w > right:
            x, y = x0, y + 44
        row.append({'key': 'sa_' + s, 'init': 'help_link ' + s, 'pos': (x, y),
                    'w': w, 'h': 30})
        x += w + 14
        far = max(far, x)
    if x + 200 > right:
        x, y = x0, y + 44
    row.append({'key': 'sa_c', 'comment': True, 'text': 'SEE ALSO - click to open',
                'pos': (x, y)})
    SA_END[0] = max(far, x + 190)
    return row


def DW(width):
    """Demo width wide enough for the SEE ALSO row just built."""
    return max(width, SA_END[0])


def th(body):
    """Text block height that fits the prose."""
    return body.count('\n') * 16 + 60


def fixed_sub(name, host, numbers, rows, cols, outname, what):
    """A subpatcher that re-sends a fixed matrix ten times a second."""
    text = '  '.join(' '.join(str(v) for v in r) for r in numbers)
    return {
        'name': name,
        'host': host,
        'demo': [
            {'key': 'met', 'init': 'metro 100', 'pos': (40, 40), 'w': 129, 'h': 70,
             'props': METRO},
            {'key': 'cm', 'init': 'comment re-sends it ten times a second',
             'pos': (190, 50), 'w': 300, 'h': 30},
            {'key': 'msg', 'init': 'message', 'pos': (40, 130), 'w': 420, 'h': 42,
             'props': {'text in': text, 'font size': '24', 'width': 400}},
            {'key': 'cg', 'init': 'comment ' + what, 'pos': (40, 185),
             'w': 12 * len(what), 'h': 30},
            {'key': 'rs', 'init': 'np.reshape %d %d' % (rows, cols), 'pos': (40, 230),
             'w': 230, 'h': 70},
            {'key': 'cr', 'init': 'comment folded into %d rows of %d' % (rows, cols),
             'pos': (290, 245), 'w': 280, 'h': 30},
            {'key': 'out', 'init': 'out ' + outname, 'pos': (40, 330), 'w': 150, 'h': 80},
        ],
        'links': [('met', '', 'msg', ''),
                  ('msg', 'message out', 'rs', 'tensor in'),
                  ('rs', 'output', 'out', outname)],
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


def point(key, x, y, vx, vy, lo=-5.0, hi=5.0, label='the point'):
    """Two sliders (x, y) packed into a list [x, y], with a list display."""
    return [
        {'key': key + 'x', 'init': 'slider', 'pos': (x, y), 'w': 200, 'h': 60,
         'props': SLIDER(lo, hi, vx)},
        {'key': key + 'y', 'init': 'slider', 'pos': (x, y + 70), 'w': 200, 'h': 60,
         'props': SLIDER(lo, hi, vy)},
        {'key': key + 'c', 'comment': True, 'text': label + ': x and y', 'pos': (x + 220, y + 15)},
        {'key': key + 'p', 'init': 'pak f f', 'pos': (x, y + 150), 'w': 150, 'h': 100},
        {'key': key + 'l', 'init': 'list', 'pos': (x + 180, y + 180), 'w': 150, 'h': 42,
         'props': LIST},
    ]


def point_links(key, lb='lb'):
    return [(lb, 'out', key + 'x', ''), (lb, 'out', key + 'y', ''),
            (key + 'y', 'float out', key + 'p', 'float', 0, 1),
            (key + 'x', 'float out', key + 'p', 'float', 0, 0),
            (key + 'p', 'out', key + 'l', '')]


# ------------------------------------------------------------------ t.length
body = """Gives the length of a tensor: the square root of the sum of all its
squared values - its straight-line distance from zero.

[3 4] has length 5; [1 1 1] has length 1.73. Reach for it to measure how
far something has moved, how fast (the length of a velocity), or how
strong a force is regardless of its direction. It is the number that
t.normalize divides by.

THE WHOLE TENSOR IS ONE VECTOR:
t.length does not work along an axis. Every value counts as one
coordinate of a single long vector, and one number comes back: a 4 x 4
tensor gives the length of its 16 values together. For one length per
row use np.linalg.norm with an axis.

t.length and t.dist are the same node class, but not the same
operation: t.length has one inlet and measures from zero; t.dist has a
second inlet and measures between two tensors.

SYNTAX:
t.length
(no arguments)

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or a NumPy array is turned into a float32 tensor).
Each one that arrives is measured and sent. An empty tensor sends
nothing.

OUTPUTS:

length:
The length, as a plain float.

READING THE DEMO:
Top: the two sliders are the x and y of a point; pak makes them the
list [x y]. It starts at [3 4], whose length is 5. Move either slider
and the length follows; put both at 0 and it reads 0. The length never
goes negative - [-3 -4] is 5 too.
Bottom: t.rand makes a 4 x 4 tensor of random values from 0 to 1 (click
the t.rand node for a new one). t.length gives ONE number for all 16.

RELATED:
t.dist measures between two tensors. t.normalize divides a tensor by
this length. t.cdist gives the distances between many points at once.
np.linalg.norm is the NumPy version, and can measure along an axis."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
] + point('a', 30, 130, 3.0, 4.0) + [
    {'key': 'ln', 'init': 't.length', 'pos': (30, 410), 'w': 150, 'h': 80},
    {'key': 'f1', 'init': 'float', 'pos': (30, 510), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'its length: 5 for [3 4]', 'pos': (180, 515)},
    {'key': 'rnd', 'init': 't.rand 4 4', 'pos': (30, 590), 'w': 180, 'h': 180},
    {'key': 'c2', 'comment': True, 'text': 'click for a new 4 x 4 tensor', 'pos': (230, 595)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (230, 640), 'w': 220, 'h': 210,
     'props': HM(200, 160, 4)},
    {'key': 'ln2', 'init': 't.length', 'pos': (30, 870), 'w': 150, 'h': 80},
    {'key': 'f2', 'init': 'float', 'pos': (30, 970), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'one length for all 16 values', 'pos': (180, 975)},
] + see_also(1050, ['t.dist', 't.normalize', 't.cdist', 'np.linalg.norm',
                    'np.distance_from_target'])
links = point_links('a') + [
    ('ap', 'out', 'ln', 'tensor in'), ('ln', 'length', 'f1', ''),
    ('lb', 'out', 'rnd', '###input'),
    ('rnd', 'random tensor', 'hm', 'y'), ('rnd', 'random tensor', 'ln2', 'tensor in'),
    ('ln2', 'length', 'f2', '')]
print(build('t.length', 't.length - how far a tensor is from zero',
            body, demo, links, demo_width=DW(640), text_width=760, text_height=th(body)))

# -------------------------------------------------------------------- t.dist
body = """Gives the distance between two tensors: the length of their
difference - the square root of the sum of the squared differences.

Two points [0 0] and [3 4] are 5 apart. Reach for it to know how far a
tracked hand is from a spot, how far a pose has moved from a reference
pose, or how close two feature vectors are.

THE WHOLE TENSOR IS ONE VECTOR:
As with t.length, the tensors are not taken row by row: every value of
the difference counts, and one number comes back. Two poses of 24
joints x 3 give one distance between the two whole poses.

SYNTAX:
t.dist
(no arguments)

INPUTS and PARAMETERS:

tensor in:
The first tensor (a list or a NumPy array is turned into a float32
tensor). Receiving it measures and sends.

tensor 2 in:
The second tensor. It is only stored, so it must arrive BEFORE tensor
in; nothing is sent until it has. It is converted to the type of the
first. The two shapes must match, or be broadcastable - a single number
here measures every value against that number. Shapes that do not fit
print an error and send nothing.

OUTPUTS:

distance:
The distance, as a plain float. 0 when the two are the same; never
negative.

READING THE DEMO:
Point A and point B are each two sliders packed into a list. A starts
at [0 0] and B at [3 4]: 5 apart. Move any slider and the distance
follows. B goes through 't bang list': the list is stored in tensor 2
in first, then the bang makes A re-send, so moving B updates the
answer too. Put B where A is and the distance reads 0.

RELATED:
t.length is the distance from zero (one input). t.cosine_similarity
compares directions rather than positions. t.cdist gives the
distances between many points at once. np.distance_from_target is the NumPy version,
measuring against a captured spot; np.linalg.norm of a difference is
the same number."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
] + point('a', 30, 130, 0.0, 0.0, label='point A') \
  + point('b', 360, 130, 3.0, 4.0, label='point B') + [
    {'key': 'tb', 'init': 't bang list', 'pos': (360, 410), 'w': 140, 'h': 46},
    {'key': 'ct', 'comment': True, 'text': 'B stored first, then A re-sent', 'pos': (520, 415)},
    {'key': 'ds', 'init': 't.dist', 'pos': (30, 490), 'w': 150, 'h': 90},
    {'key': 'f1', 'init': 'float', 'pos': (30, 600), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'how far apart A and B are', 'pos': (180, 605)},
] + see_also(680, ['t.length', 't.cosine_similarity', 't.cdist', 'np.distance_from_target',
                   'np.linalg.norm'])
links = point_links('b') + point_links('a') + [
    ('bp', 'out', 'tb', ''),
    ('tb', 'list', 'ds', 'tensor 2 in', 1), ('tb', 'bang', 'ax', '', 0),
    ('ap', 'out', 'ds', 'tensor in'), ('ds', 'distance', 'f1', '')]
print(build('t.dist', 't.dist - how far apart two tensors are',
            body, demo, links, demo_width=DW(700), text_width=760, text_height=th(body)))

# ------------------------------------------------------------------- t.cdist
CD_PTS = [[0.0, 0.0], [0.5, 0.2], [0.2, 0.6], [0.6, 0.5],
          [3.0, 3.0], [3.4, 2.8], [2.8, 3.5], [3.3, 3.4]]

body = """Measures how far apart points are: every row of the input is one
point, and the answer is a table of the distance from each point to
each other point - or to each of a second set of points you hold.

A table of distances is the starting point for anything that asks
"which are near which": finding the nearest of a set of targets to a
tracked hand, grouping poses that are alike, spotting the one point
that sits apart from the rest. The distance is the ordinary straight
line distance (square root of the sum of the squared differences).

THREE CASES:
A 2-D input (N rows of points) with nothing in points 2: an N x N
table, row i column j the distance from point i to point j. It is
symmetric and its diagonal is 0 (each point is 0 from itself).
Anything with points 2 held (M points): an N x M table, row i column
j the distance from input point i to held point j. A 1-D input counts
as one point here, giving 1 x M.
A 1-D input (a list, one vector) with nothing in points 2: its length
- the distance from zero - as a plain float, the same number t.length
gives.

SYNTAX:
t.cdist
(no arguments)

INPUTS and PARAMETERS:

input:
The points, one per row (a list, a list of lists or a NumPy array is
turned into a float32 tensor). Receiving it measures and sends.

points 2:
A second set of points, one per row, with the same number of columns
as the input. It is only stored: send it before the input, and it is
then used for every input that follows. A 1-D list is one point. Once
something has been sent here there is no way to empty it again - for
the all-pairs table use a separate t.cdist. Points with a different
number of columns print an error and send nothing.

OUTPUTS:

output:
The table of distances as a tensor (rows = input points, columns =
the points they are measured to), or one float for the length of a
single vector.

READING THE DEMO:
make_points sends 8 points in 2-D ten times a second: four close
together near [0.3 0.3] and four near [3.1 3.2]. The top picture shows
them column by column (x on top, y below).
Middle: t.cdist with nothing in points 2 gives the 8 x 8 table. The
two dark squares on the diagonal are the two groups - their points
are close to each other - and the bright squares off it are the
distances between the groups, 3.5 to 4.7. The diagonal is 0.
Bottom: the same 8 points, measured against one point held in points
2 (the sliders, starting at [3 3]). The answer is 8 x 1, one distance
per point; t.t lays it out as one row, so each cell sits under its
point's column in the top picture. Drag the sliders towards [0 0] and
the dark end moves from the right half to the left.

RELATED:
t.dist gives the one distance between two whole tensors. t.length is
the distance from zero. np.distance_from_target measures one or
several points against stored targets. np.linalg.norm with an axis
gives one length per row. t.cosine_similarity compares directions
instead of positions."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (560, 62), 'w': 88, 'h': 46},
    {'key': 'pP', 'init': 'p make_points', 'pos': (30, 62), 'w': 180, 'h': 70},
    {'key': 'cpP', 'comment': True, 'text': '8 points in 2-D: two groups of four',
     'pos': (230, 77)},
    {'key': 'tt', 'init': 't.t', 'pos': (30, 150), 'w': 130, 'h': 70},
    {'key': 'hP', 'init': 'heat_map', 'pos': (30, 240), 'w': 500, 'h': 110,
     'props': HM(480, 60, 2, 0.0, 4.0, fmt='%.1f')},
    {'key': 'cP', 'comment': True, 'text': 'the 8 points, one per column: x on top, y below',
     'pos': (30, 360)},
    {'key': 'cd1', 'init': 't.cdist', 'pos': (30, 410), 'w': 150, 'h': 90},
    {'key': 'ccd1', 'comment': True, 'text': 'nothing in points 2: every point to every point',
     'pos': (200, 430)},
    {'key': 'hD', 'init': 'heat_map', 'pos': (30, 520), 'w': 400, 'h': 400,
     'props': HM(380, 380, 8, 0.0, 5.0, fmt='%.1f')},
    {'key': 'cD', 'comment': True, 'text': 'row i, column j: how far point i is from point j',
     'pos': (30, 930)},
] + point('a', 30, 990, 3.0, 3.0, lo=0.0, hi=4.0, label='a point held in points 2') + [
    {'key': 'cd2', 'init': 't.cdist', 'pos': (30, 1260), 'w': 150, 'h': 90},
    {'key': 'ccd2', 'comment': True, 'text': 'the 8 points against the held point: 8 x 1',
     'pos': (200, 1280)},
    {'key': 'tt2', 'init': 't.t', 'pos': (30, 1370), 'w': 130, 'h': 70},
    {'key': 'hS', 'init': 'heat_map', 'pos': (30, 1460), 'w': 500, 'h': 80,
     'props': HM(480, 40, 1, 0.0, 5.0, fmt='%.1f')},
    {'key': 'cS', 'comment': True, 'text': 'how far each point (same columns as the top) is from the held point',
     'pos': (30, 1550)},
] + see_also(1600, ['t.dist', 't.length', 'np.distance_from_target', 'np.linalg.norm',
                    't.cosine_similarity'], right=700)
links = point_links('a') + [
    ('pP', 'points', 'tt', 'tensor in'), ('tt', 'tensor out', 'hP', 'y'),
    ('pP', 'points', 'cd1', 'input'), ('cd1', 'output', 'hD', 'y'),
    ('ap', 'out', 'cd2', 'points 2'),
    ('pP', 'points', 'cd2', 'input'), ('cd2', 'output', 'tt2', 'tensor in'),
    ('tt2', 'tensor out', 'hS', 'y')]
print(build('t.cdist', 't.cdist - the distances between points',
            body, demo, links, demo_width=DW(700), text_width=760, text_height=th(body),
            subpatch=fixed_sub('make_points', 'pP', CD_PTS, 8, 2, 'points',
                               '16 numbers: 8 points, two groups of four')))

# ------------------------------------------------------- t.cosine_similarity
body = """Gives how much two vectors point the same way, with their sizes taken
out: the cosine of the angle between them.

1 means the same direction, 0 at right angles, -1 opposite. Size does
not matter: [3 4] and [6 8] point the same way and score 1. So two poses
that are the same shape at different scales, or two movements that are
the same gesture at different strengths, still read as alike. Reach for
it to compare directions, poses or feature vectors when magnitude would
get in the way.

It is the dot product of the two vectors divided by both their lengths
(np.dot of two unit vectors gives the same number).

SYNTAX:
t.cosine_similarity
(no arguments)

INPUTS and PARAMETERS:

input 1:
The first vector (a list or a NumPy array is turned into a float32
tensor). Receiving it calculates and sends.

input 2:
The second vector, the same length as the first. It is only stored:
send it before input 1, and it is then used for every input 1 that
follows until a new one arrives. Nothing is sent until it has
arrived. In the demo B goes through 't bang list', so that moving B
also makes A re-send.

Two 1-D vectors give one number. Two tables of the same shape are
compared column by column (each column is one vector), giving one
cosine per column. A vector of all zeros has no direction and gives 0.
Lengths that do not match print an error and send nothing.

OUTPUTS:

output:
The cosine, a plain float from -1 to 1 - or, for two tables, a 1-D
tensor with one cosine per column.

READING THE DEMO:
Vector A is two sliders packed into [x y], starting at [3 4]. Vector B
is the angle on the slider turned into a direction (make_direction
gives [cos, sin]). B is stored in input 2 first, then the bang from
't bang list' makes A re-send, so moving either one updates the answer.
With B at 53 degrees it points almost exactly along [3 4] and the
answer is near 1. Turn B to 143 degrees (right angles): 0. To 233:
-1. Now make A shorter in the same direction - [1.5 2] - and nothing
changes: only the direction counts.

RELATED:
np.dot is the same calculation without dividing by the lengths.
t.normalize makes vectors length 1, after which the dot product is the
cosine. t.dist and t.length compare positions and sizes instead."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
] + point('a', 30, 130, 3.0, 4.0, label='vector A') + [
    {'key': 'sl', 'init': 'slider', 'pos': (360, 270), 'w': 220, 'h': 60,
     'props': SLIDER(0.0, 360.0, 53.0, width=200, fmt='%.0f')},
    {'key': 'csl', 'comment': True, 'text': 'the angle of B, in degrees', 'pos': (600, 285)},
    {'key': 'pB', 'init': 'p make_direction', 'pos': (360, 350), 'w': 200, 'h': 70},
    {'key': 'lB', 'init': 'list', 'pos': (360, 440), 'w': 220, 'h': 42, 'props': LIST},
    {'key': 'cB', 'comment': True, 'text': 'vector B: length 1, at that angle',
     'pos': (600, 445)},
    {'key': 'tb', 'init': 't bang list', 'pos': (360, 500), 'w': 140, 'h': 46},
    {'key': 'ctb', 'comment': True, 'text': 'B stored first, then A re-sent', 'pos': (520, 505)},
    {'key': 'cs', 'init': 't.cosine_similarity', 'pos': (30, 580), 'w': 220, 'h': 90},
    {'key': 'f1', 'init': 'float', 'pos': (30, 690), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': '1 same way, 0 at right angles, -1 opposite',
     'pos': (180, 695)},
] + see_also(770, ['np.dot', 't.normalize', 't.dist', 't.length', 'np.inner'])
links = point_links('a') + [
    ('lb', 'out', 'sl', ''),
    ('sl', 'float out', 'pB', 'angle'), ('pB', 'direction', 'lB', ''),
    ('pB', 'direction', 'tb', ''),
    ('tb', 'list', 'cs', 'input 2', 1), ('tb', 'bang', 'ax', '', 0),
    ('ap', 'out', 'cs', 'input 1'), ('cs', 'output', 'f1', '')]
print(build('t.cosine_similarity', 't.cosine_similarity - alike in direction, whatever the size',
            body, demo, links, demo_width=DW(700), text_width=760, text_height=th(body),
            subpatch=direction_sub('pB')))

# --------------------------------------------------------------- t.normalize
ROWS = [[1, 0, 0], [2, 0, 0], [1, 1, 0], [2, 2, 1],
        [0, 3, 0], [0.5, 0.5, 0.5], [1, 2, 2], [0, 0.4, 0.3]]

body = """Divides a tensor by its length along one axis, so what is left is only
direction: every slice comes out with length 1.

[3 4] becomes [0.6 0.8]: the same direction, length 1. Two inputs that
point the same way but differ in size come out identical - which is how
you compare shapes or gestures without their strength getting in the
way, or turn a movement into a pure direction.

IT WORKS ALONG AN AXIS - 'dim':
dim says which axis the length is measured along, and every slice along
it is scaled on its own. The default, -1, is the last axis: a single
vector is made length 1 as a whole, and a table of rows (8 rows of
x y z) has each ROW made length 1. For that table dim 0 makes each
COLUMN length 1 instead. An axis the tensor does not have prints an
error and sends nothing.

SYNTAX:
t.normalize
(no arguments)

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or a NumPy array is turned into a float32 tensor).
Receiving it normalizes and sends.

dim (number box, also an inlet):
The axis to measure length along. Default -1, the last axis; negative
counts from the end. Changing it does not send
by itself - the next tensor uses it (in the bottom demo the rows are
re-sent ten times a second, so a change shows at once).

OUTPUTS:

normalized tensor out:
The same shape, every slice along dim scaled to length 1.

NEAR ZERO:
A slice whose length is nearly zero has no meaningful direction, yet
dividing by that tiny length still gives a full-size direction -
pointing wherever the noise points. If normalized data jumps about
while things are still, that is why: check the length first (t.length)
and gate on it. Only an exactly zero slice comes back as zeros.

READING THE DEMO:
Top: the point [x y], starting at [3 4], normalized with the default
dim gives [0.6 0.8]. Move the sliders: the result always has length 1, and it
only changes when the DIRECTION changes - drag both towards zero
together and it stays put.
Bottom: make_rows sends 8 rows of 3 values of very different sizes
(lengths 0.5 to 3; picture on a 0 to 3 scale). Normalized with the
same default dim -1, each row keeps its pattern but every row now has length 1 (picture on
a 0 to 1 scale). The strips under each picture are the length of each
row (np.linalg.norm, axis 1): uneven before, all 1 after. Set dim to 0
and the COLUMNS become length 1 instead, so the row lengths are no
longer 1.

RELATED:
t.length gives the length that is divided out. t.cosine_similarity
compares two directions directly. norm (no prefix) does the same for a
single vector of numbers. np.linalg.norm measures lengths along an
axis."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
] + point('a', 30, 130, 3.0, 4.0) + [
    {'key': 'nv', 'init': 't.normalize', 'pos': (30, 410), 'w': 200, 'h': 90},
    {'key': 'cv', 'comment': True, 'text': 'default dim -1: the whole vector', 'pos': (250, 430)},
    {'key': 'lv', 'init': 'list', 'pos': (30, 520), 'w': 220, 'h': 42, 'props': LIST},
    {'key': 'c1', 'comment': True, 'text': 'the direction alone: [0.6 0.8] for [3 4]',
     'pos': (270, 525)},
    {'key': 'pR', 'init': 'p make_rows', 'pos': (30, 600), 'w': 170, 'h': 70},
    {'key': 'hI', 'init': 'heat_map', 'pos': (30, 690), 'w': 200, 'h': 290,
     'props': HM(180, 240, 8, 0.0, 3.0)},
    {'key': 'nm', 'init': 't.normalize', 'pos': (400, 600), 'w': 200, 'h': 90},
    {'key': 'hO', 'init': 'heat_map', 'pos': (400, 690), 'w': 200, 'h': 290,
     'props': HM(180, 240, 8, 0.0, 1.0)},
    {'key': 'c2', 'comment': True, 'text': 'in: 8 rows of 3 (0 to 3)     out: dim -1, each row (0 to 1)',
     'pos': (30, 990)},
    {'key': 'nI', 'init': 'np.linalg.norm', 'pos': (30, 1030), 'w': 170, 'h': 100,
     'props': {'axis': '1'}},
    {'key': 'nO', 'init': 'np.linalg.norm', 'pos': (400, 1030), 'w': 170, 'h': 100,
     'props': {'axis': '1'}},
    {'key': 'sI', 'init': 'heat_map', 'pos': (30, 1150), 'w': 360, 'h': 100,
     'props': HM(340, 50, 8, 0.0, 3.0)},
    {'key': 'sO', 'init': 'heat_map', 'pos': (400, 1150), 'w': 360, 'h': 100,
     'props': HM(340, 50, 8, 0.0, 3.0)},
    {'key': 'c3', 'comment': True, 'text': 'the length of each row: uneven before, all 1 after',
     'pos': (30, 1260)},
] + see_also(1310, ['t.length', 't.cosine_similarity', 'norm', 'np.linalg.norm', 't.dist'])
links = point_links('a') + [
    ('ap', 'out', 'nv', 'tensor in'), ('nv', 'normalized tensor out', 'lv', ''),
    ('pR', 'rows', 'hI', 'y'), ('pR', 'rows', 'nI', 'input'), ('pR', 'rows', 'nm', 'tensor in'),
    ('nm', 'normalized tensor out', 'hO', 'y'), ('nm', 'normalized tensor out', 'nO', 'input'),
    ('nI', 'norm', 'sI', 'y'), ('nO', 'norm', 'sO', 'y')]
print(build('t.normalize', 't.normalize - direction without size',
            body, demo, links, demo_width=DW(780), text_width=760, text_height=th(body),
            subpatch=fixed_sub('make_rows', 'pR', ROWS, 8, 3, 'rows',
                               '24 numbers: rows of very different sizes')))

# --------------------------------------------------------------- t.linalg.svd
SVD_BASE = [[0.0, 0.5, 1.5, 2.0], [0.4, 0.7, 1.3, 1.6], [0.8, 0.9, 1.1, 1.2],
            [1.2, 1.1, 0.9, 0.8], [1.6, 1.3, 0.7, 0.4], [2.0, 1.5, 0.5, 0.0]]
SVD_EXTRA = [[0.5, -0.5, -0.5, 0.5], [-0.5, 0.5, 0.5, -0.5]] * 3


def _flat(m):
    return ' '.join(str(v) for r in m for v in r)


SVD_SUB = {
    'name': 'make_matrix',
    'host': 'pM',
    'demo': [
        {'key': 'in', 'init': 'in amount', 'pos': (500, 40), 'w': 150, 'h': 80},
        {'key': 'met', 'init': 'metro 100', 'pos': (40, 40), 'w': 129, 'h': 70,
         'props': METRO},
        {'key': 'tbb', 'init': 't bang bang', 'pos': (40, 130), 'w': 140, 'h': 46},
        {'key': 'cm', 'init': 'comment ten times a second: the extra pattern first',
         'pos': (200, 135), 'w': 420, 'h': 30},
        {'key': 'mE', 'init': 'message', 'pos': (300, 200), 'w': 420, 'h': 42,
         'props': {'text in': _flat(SVD_EXTRA), 'font size': '24', 'width': 400}},
        {'key': 'rE', 'init': 'np.reshape 6 4', 'pos': (300, 260), 'w': 230, 'h': 70},
        {'key': 'cE', 'init': 'comment a checker: a third, independent pattern',
         'pos': (300, 340), 'w': 400, 'h': 30},
        {'key': 'mul', 'init': '*', 'pos': (300, 380), 'w': 130, 'h': 70},
        {'key': 'cmu', 'init': 'comment times the amount', 'pos': (450, 395), 'w': 220, 'h': 30},
        {'key': 'mB', 'init': 'message', 'pos': (40, 470), 'w': 420, 'h': 42,
         'props': {'text in': _flat(SVD_BASE), 'font size': '24', 'width': 400}},
        {'key': 'rB', 'init': 'np.reshape 6 4', 'pos': (40, 530), 'w': 230, 'h': 70},
        {'key': 'cB', 'init': 'comment 6 rows, each a blend of two patterns: rank 2',
         'pos': (40, 610), 'w': 440, 'h': 30},
        {'key': 'add', 'init': '+', 'pos': (40, 650), 'w': 130, 'h': 70},
        {'key': 'out', 'init': 'out matrix', 'pos': (40, 750), 'w': 150, 'h': 80},
    ],
    'links': [('in', 'amount', 'mul', '###operand'),
              ('met', '', 'tbb', ''),
              ('tbb', 'bang', 'mE', '', 1), ('tbb', 'bang', 'mB', '', 0),
              ('mE', 'message out', 'rE', 'tensor in'), ('rE', 'output', 'mul', 'in'),
              ('mul', 'result', 'add', '###operand'),
              ('mB', 'message out', 'rB', 'tensor in'), ('rB', 'output', 'add', 'in'),
              ('add', 'result', 'out', 'matrix')],
}

body = """Splits a matrix into three pieces - turn, stretch, turn - with the
singular value decomposition: the input equals U x diag(S) x Vh.

S, the singular values, is the part to look at first. They come out
largest first, and how fast they fall away says how many independent
patterns the matrix really holds. If the first two are large and the
rest are near zero, the data is really two patterns mixed in different
amounts, with noise on top. For a set of poses, or joint angles over
time, the answer is usually far fewer than the number of channels.
U says how much of each pattern each ROW uses; the rows of Vh are the
patterns themselves.

SYNTAX:
t.linalg.svd
(no arguments)

INPUTS and PARAMETERS:

tensor in:
The matrix (a list of lists or a NumPy array is turned into a float32
tensor). Any shape, rows x columns; a stack of matrices is split one by
one. Receiving it decomposes and sends.

full (checkbox):
On (the default): U is square, rows x rows, and Vh columns x columns -
for a 6 x 4 input U is 6 x 6, two of its columns multiplying nothing.
Off: the reduced form, U 6 x 4 - smaller, and what fits the rebuild
in this demo (with full on, the rebuild prints a shape error). Changing
it does not send by itself; the next matrix uses it.

OUTPUTS (sent right to left - Vh first, U last):

U out:
One column per pattern, one row per input row: how much of each
pattern each row uses.

S out:
The singular values, largest first (1-D, never negative): how strong
each pattern is.

Vh out:
One row per pattern, one column per input column: the patterns
themselves, each of length 1.

READING THE DEMO:
make_matrix sends a 6 x 4 matrix ten times a second: every row a blend
of two patterns, plus a checker pattern scaled by the slider. With the
slider at 0 the singular values read about 4.9, 2.6, 0, 0 - two
patterns, so two values. Raise the slider and a third singular value
grows from 0. The fourth stays 0: there are still only three patterns.
U and Vh are shown as pictures (blue above 0, red below). The bottom
picture multiplies the three pieces back together - t.diag makes S
into a square matrix, np.matmul does the products - and it is the input
again.

RELATED:
t.linalg.pca_low_rank finds the leading patterns of a set of points,
centred first. t.linalg.eig finds the directions a square matrix only
stretches. t.linalg.qr splits into perpendicular directions and a
triangle. np.linalg.matrix_rank counts the singular values that are not
zero. np.matmul rebuilds."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'sl', 'init': 'slider', 'pos': (150, 62), 'w': 200, 'h': 60,
     'props': SLIDER(0.0, 1.0, 0.0)},
    {'key': 'csl', 'comment': True, 'text': 'how much of a third pattern', 'pos': (370, 75)},
    {'key': 'pM', 'init': 'p make_matrix', 'pos': (30, 150), 'w': 180, 'h': 70},
    {'key': 'hA', 'init': 'heat_map', 'pos': (30, 240), 'w': 200, 'h': 250,
     'props': HM(180, 200, 6, -2.0, 2.0, color='red-blue')},
    {'key': 'cA', 'comment': True, 'text': 'the input: 6 x 4', 'pos': (30, 500)},
    {'key': 'sv', 'init': 't.linalg.svd', 'pos': (330, 150), 'w': 200, 'h': 130,
     'props': {'full': False}},
    {'key': 'cf', 'comment': True, 'text': 'full off: the reduced form', 'pos': (550, 160)},
    {'key': 'lS', 'init': 'list', 'pos': (330, 300), 'w': 300, 'h': 42, 'props': LIST},
    {'key': 'hS', 'init': 'heat_map', 'pos': (330, 360), 'w': 280, 'h': 100,
     'props': HM(260, 50, 4, 0.0, 5.0)},
    {'key': 'cS', 'comment': True, 'text': 'S: the singular values, largest first',
     'pos': (330, 470)},
    {'key': 'hU', 'init': 'heat_map', 'pos': (30, 540), 'w': 200, 'h': 250,
     'props': HM(180, 200, 6, -1.0, 1.0, color='red-blue')},
    {'key': 'cU', 'comment': True, 'text': 'U: 6 x 4, one column per pattern', 'pos': (30, 800)},
    {'key': 'hV', 'init': 'heat_map', 'pos': (330, 540), 'w': 200, 'h': 250,
     'props': HM(180, 200, 4, -1.0, 1.0, color='red-blue')},
    {'key': 'cV', 'comment': True, 'text': 'Vh: 4 x 4, one pattern per row', 'pos': (330, 800)},
    {'key': 'dg', 'init': 't.diag', 'pos': (330, 850), 'w': 150, 'h': 90},
    {'key': 'm2', 'init': 'np.matmul', 'pos': (330, 960), 'w': 150, 'h': 70},
    {'key': 'c2', 'comment': True, 'text': 'diag(S) x Vh', 'pos': (500, 975)},
    {'key': 'm1', 'init': 'np.matmul', 'pos': (30, 960), 'w': 150, 'h': 70},
    {'key': 'c1', 'comment': True, 'text': 'U x that', 'pos': (200, 975)},
    {'key': 'hR', 'init': 'heat_map', 'pos': (30, 1050), 'w': 200, 'h': 250,
     'props': HM(180, 200, 6, -2.0, 2.0, color='red-blue')},
    {'key': 'cR', 'comment': True, 'text': 'rebuilt: the input again', 'pos': (30, 1310)},
] + see_also(1360, ['t.linalg.pca_low_rank', 't.linalg.eig', 't.linalg.qr',
                    'np.linalg.matrix_rank', 'np.matmul'])
links = [('lb', 'out', 'sl', ''), ('sl', 'float out', 'pM', 'amount'),
         ('pM', 'matrix', 'hA', 'y'), ('pM', 'matrix', 'sv', 'tensor in'),
         ('sv', 'Vh out', 'hV', 'y'), ('sv', 'Vh out', 'm2', 'in 2'),
         ('sv', 'S out', 'lS', ''), ('sv', 'S out', 'hS', 'y'),
         ('sv', 'S out', 'dg', 'tensor in'),
         ('dg', 'output', 'm2', 'in 1'), ('m2', 'mat mul result', 'm1', 'in 2'),
         ('sv', 'U out', 'hU', 'y'), ('sv', 'U out', 'm1', 'in 1'),
         ('m1', 'mat mul result', 'hR', 'y')]
print(build('t.linalg.svd', 't.linalg.svd - the patterns a matrix is made of',
            body, demo, links, demo_width=DW(660), text_width=760, text_height=th(body),
            subpatch=SVD_SUB))

# ---------------------------------------------------------------- t.linalg.qr
QR_A = [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

body = """Splits a matrix into Q, whose columns are perpendicular directions of
length 1, and R, a triangle: the input equals Q x R.

Q is the useful half on its own: it turns a set of arbitrary directions
(the columns of the input) into a set of mutually perpendicular ones
spanning the same space - the first column of Q points along the first
column of the input, the second is the part of the second column at
right angles to the first, and so on. R records how to mix Q's columns
back into the input, and is zero below its diagonal because each input
column only needs the Q columns up to its own. It is also the stable
way least-squares problems are solved.

SYNTAX:
t.linalg.qr
(no arguments)

INPUTS and PARAMETERS:

tensor in:
The matrix (a list of lists or a NumPy array is turned into a float32
tensor), rows x columns. Receiving it decomposes and sends.

mode (menu):
reduced (the default): for a 4 x 3 input, Q is 4 x 3 and R 3 x 3.
complete: Q is square, 4 x 4 (one more perpendicular direction), and R
4 x 3 with a row of zeros.
r: R only - Q is sent as an EMPTY tensor, which the pictures here
cannot draw.
Changing it does not send by itself; the next matrix uses it (here the
matrix is re-sent ten times a second, so a change shows at once).

OUTPUTS:

Q tensor out:
Q, columns of length 1 at right angles to each other. Sent second.

R tensor out:
R, upper triangular. Sent first.

READING THE DEMO:
make_matrix sends a 4 x 3 matrix. Q (blue above 0, red below) has the
same shape; R's zeros below the diagonal show as the neutral colour.
The bottom picture is Q times R from np.matmul (R arrives first and is
stored, then Q triggers): the input again. Choose 'complete' and Q
gains a fourth column and R a zero row; the product is unchanged.
Signs are free: a column of Q and the matching row of R can both flip
and the product stays the same.

RELATED:
t.linalg.svd breaks a matrix into patterns and their strengths.
t.linalg.eig finds the directions a square matrix only stretches.
np.linalg.solve solves the equations that qr is often used for.
np.matmul multiplies the pieces back together."""

demo = [
    {'key': 'pM', 'init': 'p make_matrix', 'pos': (30, 62), 'w': 180, 'h': 70},
    {'key': 'hA', 'init': 'heat_map', 'pos': (30, 150), 'w': 220, 'h': 210,
     'props': HM(200, 160, 4, -2.0, 2.0, color='red-blue')},
    {'key': 'cA', 'comment': True, 'text': 'the input: 4 x 3', 'pos': (30, 370)},
    {'key': 'qr', 'init': 't.linalg.qr', 'pos': (330, 62), 'w': 200, 'h': 110,
     'props': {'mode': 'reduced'}},
    {'key': 'cq', 'comment': True, 'text': 'choose the mode', 'pos': (550, 75)},
    {'key': 'hQ', 'init': 'heat_map', 'pos': (30, 420), 'w': 220, 'h': 210,
     'props': HM(200, 160, 4, -2.0, 2.0, color='red-blue')},
    {'key': 'cQ', 'comment': True, 'text': 'Q: perpendicular columns, length 1', 'pos': (30, 640)},
    {'key': 'hR', 'init': 'heat_map', 'pos': (330, 420), 'w': 220, 'h': 210,
     'props': HM(200, 160, 3, -2.0, 2.0, color='red-blue')},
    {'key': 'cR', 'comment': True, 'text': 'R: zero below the diagonal', 'pos': (330, 640)},
    {'key': 'mm', 'init': 'np.matmul', 'pos': (30, 690), 'w': 150, 'h': 70},
    {'key': 'hP', 'init': 'heat_map', 'pos': (30, 780), 'w': 220, 'h': 210,
     'props': HM(200, 160, 4, -2.0, 2.0, color='red-blue')},
    {'key': 'cP', 'comment': True, 'text': 'Q x R: the input again', 'pos': (30, 1000)},
] + see_also(1050, ['t.linalg.svd', 't.linalg.eig', 'np.linalg.solve', 'np.matmul'])
links = [('pM', 'matrix', 'hA', 'y'), ('pM', 'matrix', 'qr', 'tensor in'),
         ('qr', 'R tensor out', 'hR', 'y'), ('qr', 'R tensor out', 'mm', 'in 2'),
         ('qr', 'Q tensor out', 'hQ', 'y'), ('qr', 'Q tensor out', 'mm', 'in 1'),
         ('mm', 'mat mul result', 'hP', 'y')]
print(build('t.linalg.qr', 't.linalg.qr - perpendicular directions and a triangle',
            body, demo, links, demo_width=DW(660), text_width=760, text_height=th(body),
            subpatch=fixed_sub('make_matrix', 'pM', QR_A, 4, 3, 'matrix',
                               '12 numbers: four rows of 3')))

# --------------------------------------------------------------- t.linalg.eig
body = """Finds the eigenvectors of a square matrix - the directions it only
stretches, without turning - and the eigenvalues, how much it stretches
each one.

Apply the matrix to one of its eigenvectors and you get the same vector
back, scaled by its eigenvalue. For a covariance matrix those directions
are the axes along which a set of points varies, and the eigenvalues
are how much it varies along each. For a transformation they say what
it does at heart: stretches along these axes by these amounts.

THE ANSWERS ARE COMPLEX NUMBERS:
A matrix that turns things (a rotation) has no direction it leaves
unturned, and its eigenvalues are complex. So the node ALWAYS sends
complex tensors, even when every answer is real. Pass them through
t.real (and t.imag) before plotting or doing arithmetic - the demo
does. A symmetric matrix (the same across its diagonal, like a
covariance matrix) always has real eigenvalues.

SYNTAX:
t.linalg.eig
(no arguments)

INPUTS and PARAMETERS:

tensor in:
A square matrix (a list of lists or a NumPy array is turned into a
float32 tensor). A matrix that is not square, or a 1-D list, sends
nothing (the reason is printed only when the app is in verbose mode).
Receiving it decomposes and sends.

OUTPUTS:

L tensor out:
The eigenvalues, a 1-D complex tensor, one per row. Not sorted. Sent
second.

V tensor out:
The eigenvectors as COLUMNS of a complex matrix: column i goes with
eigenvalue i, and has length 1. Sent first.

READING THE DEMO:
The vector node is the matrix: drag any of its four numbers. It starts
at rows [2 1] and [1 2]: the eigenvalues read 3 and 1 (in the order
the solver finds them, not sorted). The first eigenvector column,
[0.71 0.71], goes with 3: the matrix stretches the diagonal [1 1] by 3.
The second, [-0.71 0.71], goes with 1: that direction is left alone. Make it [0 -1] and [1 0] - a quarter turn - and the real parts
read 0 while the imaginary parts read 1 and -1: no direction survives a
turn.

RELATED:
t.linalg.svd works for any shape and always gives real answers.
t.linalg.pca_low_rank gives the axes of variation of a set of points
directly. t.real and t.imag split the complex results. np.linalg.det
is the product of the eigenvalues."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'A', 'init': 'vector 2 2', 'pos': (30, 130), 'w': 170, 'h': 150,
     'props': VEC([[2.0, 1.0], [1.0, 2.0]])},
    {'key': 'c0', 'comment': True, 'text': 'the matrix: drag any number', 'pos': (220, 135)},
    {'key': 'hA', 'init': 'heat_map', 'pos': (30, 300), 'w': 220, 'h': 210,
     'props': HM(200, 160, 2, -3.0, 3.0, color='red-blue')},
    {'key': 'eg', 'init': 't.linalg.eig', 'pos': (30, 530), 'w': 180, 'h': 90},
    {'key': 'rL', 'init': 't.real', 'pos': (330, 530), 'w': 130, 'h': 70},
    {'key': 'iL', 'init': 't.imag', 'pos': (500, 530), 'w': 130, 'h': 70},
    {'key': 'lR', 'init': 'list', 'pos': (330, 620), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'lI', 'init': 'list', 'pos': (500, 620), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'cL', 'comment': True, 'text': 'eigenvalues: real part, imaginary part',
     'pos': (330, 675)},
    {'key': 'rV', 'init': 't.real', 'pos': (30, 640), 'w': 130, 'h': 70},
    {'key': 'hV', 'init': 'heat_map', 'pos': (30, 730), 'w': 220, 'h': 210,
     'props': HM(200, 160, 2, -1.0, 1.0, color='red-blue')},
    {'key': 'cV', 'comment': True, 'text': 'eigenvectors (real part), one per column',
     'pos': (30, 950)},
] + see_also(1000, ['t.linalg.svd', 't.linalg.pca_low_rank', 't.real', 't.imag',
                    'np.linalg.det'])
links = [('lb', 'out', 'A', 'in'), ('A', 'out', 'hA', 'y'), ('A', 'out', 'eg', 'tensor in'),
         ('eg', 'V tensor out', 'rV', 'tensor in'), ('rV', 'real tensor', 'hV', 'y'),
         ('eg', 'L tensor out', 'iL', 'tensor in'), ('eg', 'L tensor out', 'rL', 'tensor in'),
         ('rL', 'real tensor', 'lR', ''), ('iL', 'imaginary tensor', 'lI', '')]
print(build('t.linalg.eig', 't.linalg.eig - the directions a matrix only stretches',
            body, demo, links, demo_width=DW(680), text_width=760, text_height=th(body)))

# ----------------------------------------------------- t.linalg.pca_low_rank
PCA_PTS = [[-1.88, 2.32], [-1.69, 2.58], [-1.29, 2.26], [-1.18, 2.73], [-0.9, 2.73],
           [-0.55, 2.57], [-0.46, 3.08], [-0.09, 2.86], [0.16, 2.94], [0.29, 3.36],
           [0.7, 3.03], [0.86, 3.36], [1.07, 3.56], [1.46, 3.27], [1.57, 3.76],
           [1.87, 3.7]]

body = """Finds the principal components of a set of points: the directions along
which they spread out most, and how much they spread along each.

Give it N points (one per row, any number of columns) and it answers
"along which few directions does this data really vary?". For 16 points
lying roughly along a line, the first direction is the line and the
second carries almost nothing. For poses or joint angles, a handful of
directions usually carries most of the movement - those directions are
the coordinated patterns, and the rest is noise. It finds only the
leading few, quickly and approximately, which matters on big data.

CENTRE FIRST - THE 'center' CHECKBOX:
Principal components describe spread AROUND THE MIDDLE of the points,
so the mean is taken off first: 'center' is on by default. Turn it off
only for data that is already centred - otherwise the first direction
points from zero towards where the cloud sits, not along the way it
spreads.

SYNTAX:
t.linalg.pca_low_rank
(no arguments)

INPUTS and PARAMETERS:

tensor in:
The points, one per row (a list of lists or a NumPy array is turned
into a float32 tensor). Receiving it calculates and sends. It keeps up
to 6 components, or fewer if there are fewer rows or columns.

center (checkbox):
Take off the mean of each column first. Default on.

iterations (number box):
The number of refinement passes of the approximate method, default 2,
0 or more. More passes, more accurate on big data; small data like the
demo's is exact anyway.

Changing either does not send by itself; the next points use it (here
they are re-sent ten times a second, so a change shows at once).

OUTPUTS:

U tensor out:
For each point, where it lies along each direction (divided by the
spread). One row per point. Sent last.

S tensor out:
How much the points spread along each direction, largest first.

V tensor out:
The directions, one per COLUMN, each of length 1. Sent first.
A direction and its opposite are the same axis, so signs may come out
either way.

READING THE DEMO:
make_points sends 16 points in 2-D, lying along a line at 20 degrees
whose middle sits at [0 3]. The top picture shows them as two rows
(x on top, y below) for the 16 points. With center ON the first column
of V is about [-0.94 -0.34] - the 20 degree line (sign flipped) - and S
reads about 4.9 and 0.7: nearly all the spread is along the line. The
bottom picture is U x S, each point's position along the two
directions: the first row sweeps across, the second barely moves.
Turn center OFF and the first direction swings round to about
[-0.06 -1], pointing at the middle of the cloud, and S jumps to 12.2 -
it is measuring distance from zero, not spread.

RELATED:
t.linalg.svd is the full decomposition this approximates (pca with
center on is the svd of the centred points). t.linalg.eig of the
points' covariance gives the same directions. np.mean takes the mean
off by hand."""

demo = [
    {'key': 'pP', 'init': 'p make_points', 'pos': (30, 62), 'w': 180, 'h': 70},
    {'key': 'tt', 'init': 't.t', 'pos': (30, 150), 'w': 130, 'h': 70},
    {'key': 'hP', 'init': 'heat_map', 'pos': (30, 240), 'w': 500, 'h': 110,
     'props': HM(480, 60, 2, -4.0, 4.0, fmt='none', color='red-blue')},
    {'key': 'cP', 'comment': True, 'text': 'the 16 points: x on top, y below (blue above 0)',
     'pos': (30, 360)},
    {'key': 'pc', 'init': 't.linalg.pca_low_rank', 'pos': (330, 62), 'w': 240, 'h': 130,
     'props': {'center': True, 'iterations': 2}},
    {'key': 'cc', 'comment': True, 'text': 'try center off', 'pos': (590, 75)},
    {'key': 'hV', 'init': 'heat_map', 'pos': (30, 410), 'w': 220, 'h': 210,
     'props': HM(200, 160, 2, -1.0, 1.0, color='red-blue')},
    {'key': 'cV', 'comment': True, 'text': 'V: the directions, one per column', 'pos': (30, 630)},
    {'key': 'lS', 'init': 'list', 'pos': (330, 410), 'w': 220, 'h': 42, 'props': LIST},
    {'key': 'cS', 'comment': True, 'text': 'S: the spread along each', 'pos': (330, 465)},
    {'key': 'mul', 'init': '*', 'pos': (30, 680), 'w': 130, 'h': 70},
    {'key': 'cm', 'comment': True, 'text': 'U x S: each point along each direction',
     'pos': (180, 695)},
    {'key': 'tt2', 'init': 't.t', 'pos': (30, 770), 'w': 130, 'h': 70},
    {'key': 'hU', 'init': 'heat_map', 'pos': (30, 860), 'w': 500, 'h': 110,
     'props': HM(480, 60, 2, -4.0, 4.0, fmt='none', color='red-blue')},
    {'key': 'cU', 'comment': True, 'text': 'first direction on top: it carries the spread',
     'pos': (30, 980)},
] + see_also(1030, ['t.linalg.svd', 't.linalg.eig', 'np.mean', 't.normalize'])
links = [('pP', 'points', 'tt', 'tensor in'), ('tt', 'tensor out', 'hP', 'y'),
         ('pP', 'points', 'pc', 'tensor in'),
         ('pc', 'V tensor out', 'hV', 'y'),
         ('pc', 'S tensor out', 'lS', ''), ('pc', 'S tensor out', 'mul', '###operand'),
         ('pc', 'U tensor out', 'mul', 'in'), ('mul', 'result', 'tt2', 'tensor in'),
         ('tt2', 'tensor out', 'hU', 'y')]
print(build('t.linalg.pca_low_rank', 't.linalg.pca_low_rank - the directions a cloud spreads along',
            body, demo, links, demo_width=DW(700), text_width=760, text_height=th(body),
            subpatch=fixed_sub('make_points', 'pP', PCA_PTS, 16, 2, 'points',
                               '32 numbers: 16 points along a line at 20 degrees')))

# ----------------------------------------------------------------- losses
TARGET = [0.2, 0.5, 0.8, 0.6, 0.3, 0.1, 0.4, 0.7]


def loss_demo(node, square):
    word = 'squared' if square else 'size of'
    demo = [
        {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
        {'key': 'tbb', 'init': 't bang bang', 'pos': (150, 62), 'w': 140, 'h': 46},
        {'key': 'ct', 'comment': True, 'text': 'the target first, then the input',
         'pos': (310, 67)},
        {'key': 'mT', 'init': 'message', 'pos': (330, 130), 'w': 300, 'h': 42,
         'props': {'text in': ' '.join(str(v) for v in TARGET), 'font size': '24',
                   'width': 280}},
        {'key': 'cT', 'comment': True, 'text': 'the target: 8 values', 'pos': (330, 185)},
        {'key': 'v', 'init': 'vector 8', 'pos': (30, 130), 'w': 120, 'h': 330,
         'props': VEC([0.6] + TARGET[1:])},
        {'key': 'cv', 'comment': True, 'text': 'the input: drag any value', 'pos': (30, 470)},
        {'key': 'hT', 'init': 'heat_map', 'pos': (330, 230), 'w': 360, 'h': 100,
         'props': HM(340, 50, 8, 0.0, 1.0)},
        {'key': 'hI', 'init': 'heat_map', 'pos': (330, 350), 'w': 360, 'h': 100,
         'props': HM(340, 50, 8, 0.0, 1.0)},
        {'key': 'cI', 'comment': True, 'text': 'target (top) and input (below)',
         'pos': (330, 460)},
        {'key': 'sub', 'init': '-', 'pos': (330, 510), 'w': 130, 'h': 70},
        {'key': 'hD', 'init': 'heat_map', 'pos': (330, 600), 'w': 360, 'h': 100,
         'props': HM(340, 50, 8, -1.0, 1.0, color='red-blue')},
        {'key': 'cD', 'comment': True, 'text': 'input minus target: blue too high, red too low',
         'pos': (330, 710)},
        {'key': 'ls', 'init': node, 'pos': (30, 520), 'w': 200, 'h': 110},
        {'key': 'f1', 'init': 'float', 'pos': (30, 650), 'w': 127, 'h': 42, 'props': FLT},
        {'key': 'c1', 'comment': True, 'text': 'the loss: the ' + word + ' each difference, added up',
         'pos': (30, 700)},
    ]
    links = [('lb', 'out', 'tbb', ''),
             ('tbb', 'bang', 'mT', '', 1), ('tbb', 'bang', 'v', 'in', 0),
             ('mT', 'message out', 'hT', 'y'), ('mT', 'message out', 'ls', 'target'),
             ('mT', 'message out', 'sub', '###operand'),
             ('v', 'out', 'hI', 'y'), ('v', 'out', 'sub', 'in'), ('v', 'out', 'ls', 'tensor in'),
             ('sub', 'result', 'hD', 'y'), ('ls', 'loss', 'f1', '')]
    return demo, links


LOSS_COMMON = """SYNTAX:
{node}
(no arguments)

INPUTS and PARAMETERS:

tensor in:
What you got (a list or a NumPy array is turned into a float32
tensor). Receiving it computes the loss and sends it.

target:
What you wanted, converted to the type of the input. It is a COLD
inlet: it is only stored, so set it BEFORE sending the input - send
them the other way round and the answer is computed against the
previous target, which looks like the loss lagging one step behind.
Nothing is sent until a target has arrived. Same shape as the input,
or one that broadcasts to it: a single number measures every value
against that number (torch prints a warning about the size mismatch,
and the answer is still right).

OUTPUTS:

loss:
One plain float. 0 when input and target are equal; never negative.
"""

body = """Measures how far a tensor is from a target as one number: the sum of
the SQUARED differences, element by element (mean squared error, but
added up rather than averaged).

0 means identical. Squaring makes big differences count far more than
small ones: one value off by 4 costs 16, while four values off by 1
cost only 4. So it is dominated by the worst offender - the measure to
use when one large discrepancy is what matters and should shout. And
below a difference of 1 squaring makes things SMALLER: off by 0.4 costs
0.16, where t.l1_loss says 0.40.

Reach for it as a running measure of how far a signal or a pose has
drifted from a reference, or as the thing a training loop minimises.

A SUM, NOT AN AVERAGE:
It adds up rather than averaging, so the number grows with the size of
the tensor: 16 across 4 values and 16 across 400 are not the same at
all. Divide by the element count yourself if you need numbers that
compare across sizes.

""" + LOSS_COMMON.format(node='t.mse_loss') + """
READING THE DEMO:
The vector node is the input: eight values, equal to the target
except the first, which is 0.4 too high (0.6 against 0.2) - so the
loss reads 0.16, the difference squared. The red-blue strip is the
difference. Drag the first value back to 0.2 and the loss reads 0.
Drag it 0.5 off: 0.25. Move two values 0.3 each: 0.18. Each difference
is squared, then they are all added.

RELATED:
t.l1_loss adds the plain sizes of the differences - the robust one.
t.cross_entropy_loss is for choosing between categories. t.dist is the
square root of this loss (the straight-line distance). np.linalg.norm
of the difference, squared, is the same number."""

demo, links = loss_demo('t.mse_loss', True)
demo += see_also(780, ['t.l1_loss', 't.cross_entropy_loss', 't.dist', 'np.linalg.norm'])
print(build('t.mse_loss', 't.mse_loss - the squared error, added up',
            body, demo, links, demo_width=DW(720), text_width=760, text_height=th(body)))

body = """Measures how far a tensor is from a target as one number: the sum of
the plain sizes of the differences, element by element (L1, or absolute
error, added up rather than averaged).

0 means identical. Every difference counts its own size: one value off
by 4 costs 4, and four values off by 1 also cost 4. So one wild value
does not swamp the reading the way it does with t.mse_loss - L1 is the
robust one, for when the occasional outlier is noise rather than the
point. Below a difference of 1 it is the LARGER of the two: off by 0.4
costs 0.40 here and 0.16 with t.mse_loss.

Reach for it to measure drift from a reference when outliers should
not dominate, or as the thing a training loop minimises when you want
that robustness.

A SUM, NOT AN AVERAGE:
It adds up rather than averaging, so the number grows with the size of
the tensor. Divide by the element count yourself if you need numbers
that compare across sizes.

""" + LOSS_COMMON.format(node='t.l1_loss') + """
READING THE DEMO:
The vector node is the input: eight values, equal to the target
except the first, which is 0.4 too high (0.6 against 0.2) - so the
loss reads 0.40. The red-blue strip is the difference. Drag the first
value back to 0.2 and the loss reads 0. Drag another value down by
0.3 as well: 0.70 - up or down, every difference adds its size.

RELATED:
t.mse_loss squares the differences first, so big ones dominate.
t.cross_entropy_loss is for choosing between categories. t.dist gives
the straight-line distance. np.abs and np.sum of the difference give
the same number by hand."""

demo, links = loss_demo('t.l1_loss', False)
demo += see_also(780, ['t.mse_loss', 't.cross_entropy_loss', 't.dist', 'np.linalg.norm'])
print(build('t.l1_loss', 't.l1_loss - the size of the error, added up',
            body, demo, links, demo_width=DW(720), text_width=760, text_height=th(body)))

# ------------------------------------------------------- t.cross_entropy_loss
body = """Measures how wrong a choice between categories is: given one score per
category and the right answer, it is low when the right category has by
far the highest score and high when another one does.

The scores are raw numbers, any size, one per category (what a
classifier puts out before softmax). Internally they are turned into
probabilities with softmax, and the loss is minus the log of the
probability given to the right answer. With scores [2.0 0.5 0.1] the
first category gets 0.73 of the probability:

    right answer 0  ->  0.32     confident and right
    right answer 1  ->  1.82
    right answer 2  ->  2.22     confident and wrong

Being confidently wrong costs far more than being unsure, so it rewards
well-placed confidence, not just right guesses. 0 would need all of the
probability on the right answer. e to the minus the loss is the
probability the right answer got.

SYNTAX:
t.cross_entropy_loss
(no arguments)

INPUTS and PARAMETERS:

tensor in:
The scores (a list or a NumPy array is turned into a float32 tensor):
one row of one score per category, or several rows, one per item, as
an items x categories tensor. Receiving it computes the loss and sends.

target:
The right answer. A COLD inlet: it is only stored, so set it BEFORE
the scores. Either one whole number per row - the index of the right
category, counting from 0 - or, for an answer that is a blend, a full
row of probabilities the same shape as the scores. The node tells them
apart by shape. Nothing is sent until a target has arrived; an index
past the last category prints an error.

OUTPUTS:

loss:
One plain float, never negative. For several rows it is the AVERAGE
over the rows (unlike t.mse_loss and t.l1_loss, which add up).

READING THE DEMO:
The vector node holds the scores for three categories, [2.0 0.5 0.1]:
drag any of them. The strip shows them as softmax probabilities (the
node does that conversion itself; t.nn.softmax is only here to show
it). The int box is the right answer; it goes through 't bang int' so
the index is stored first, then the scores are re-sent. Answer 0: 0.32.
Answer 2: 2.22. Raise the third score above the others and answer 2
becomes cheap.

RELATED:
t.nn.softmax turns scores into the probabilities used here.
t.nn.log_softmax gives their logs: the loss is minus the log
probability of the right answer. t.mse_loss and t.l1_loss compare
numbers rather than choices. np.argmax picks the highest score."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'ix', 'init': 'int', 'pos': (330, 62), 'w': 127, 'h': 42,
     'props': dict(INT, **{'min': 0, 'max': 2, '': 0})},
    {'key': 'ci', 'comment': True, 'text': 'the right answer: 0, 1 or 2', 'pos': (480, 67)},
    {'key': 'tb', 'init': 't bang int', 'pos': (330, 130), 'w': 140, 'h': 46},
    {'key': 'ct', 'comment': True, 'text': 'stored first, then the scores re-sent',
     'pos': (490, 135)},
    {'key': 'v', 'init': 'vector 3', 'pos': (30, 130), 'w': 120, 'h': 175,
     'props': VEC([2.0, 0.5, 0.1])},
    {'key': 'cv', 'comment': True, 'text': 'the scores: drag any', 'pos': (30, 315)},
    {'key': 'sm', 'init': 't.nn.softmax', 'pos': (330, 210), 'w': 180, 'h': 90,
     'props': {'dim': 0}},
    {'key': 'hP', 'init': 'heat_map', 'pos': (330, 320), 'w': 280, 'h': 100,
     'props': HM(260, 50, 3, 0.0, 1.0)},
    {'key': 'cP', 'comment': True, 'text': 'as probabilities: they add up to 1',
     'pos': (330, 430)},
    {'key': 'ce', 'init': 't.cross_entropy_loss', 'pos': (30, 370), 'w': 260, 'h': 110},
    {'key': 'f1', 'init': 'float', 'pos': (30, 500), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'the loss: low when the right answer scores highest',
     'pos': (30, 550)},
] + see_also(610, ['t.nn.softmax', 't.nn.log_softmax', 't.mse_loss', 't.l1_loss',
                   'np.argmax'])
links = [('lb', 'out', 'ix', ''), ('ix', 'int out', 'tb', ''),
         ('tb', 'int', 'ce', 'target', 1), ('tb', 'bang', 'v', 'in', 0),
         ('v', 'out', 'sm', 'tensor in'), ('sm', 'output', 'hP', 'y'),
         ('v', 'out', 'ce', 'tensor in'), ('ce', 'loss', 'f1', '')]
print(build('t.cross_entropy_loss', 't.cross_entropy_loss - how wrong a choice is',
            body, demo, links, demo_width=DW(680), text_width=760, text_height=th(body)))
