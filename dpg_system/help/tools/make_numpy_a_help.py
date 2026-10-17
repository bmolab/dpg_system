"""np statistics, linear algebra, and shape."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

AXIS = """
THE axis INLET:
Most of these take an axis, and it decides what "across" means. 
Leave it alone and the whole array is reduced to a single number. 
Set it to 0 and each COLUMN is reduced, giving you one result per column; 
set it to 1 and each ROW is. 

For a 4 by 4 array of joint readings that is the difference between "the 
average of everything", "the average of each joint over time" and "the average 
of all joints at each instant" - three different questions from one node.
"""

# ------------------------------------------------------------------ np_stats
body = """These nodes reduce an array to a summary of what is in it.

An array of a thousand numbers is not something you can look at. 
These answer specific questions about it - what is the largest, where is the 
largest, what is typical, how spread out is it - and give you back something 
small enough to act on.

THE NODES:

np.sum        add everything up
np.mean       the average
np.median     the middle value when sorted - unlike the mean, one wild 
              value cannot drag it far
np.std        standard deviation: how far from the mean values typically sit
np.var        variance, which is the standard deviation squared
np.amax       the largest value
np.amin       the smallest value
np.argmax     WHERE the largest value is - its index, not its value
np.argmin     where the smallest is
np.any        true if any value is non-zero
np.all        true only if every value is non-zero

IGNORING MISSING DATA:

np.nansum     add up, skipping any not-a-number entries
np.nanmean    average, skipping them
np.nanmedian  median, skipping them

Real data has holes in it - a dropped frame, a sensor that lost tracking - and 
those arrive as NaN. An ordinary mean over an array containing one NaN returns 
NaN for the whole thing, which is technically correct and useless. 
The nan versions skip the holes and average what is actually there.

REJECTING OUTLIERS:

np.robust_mean accumulates samples over time and returns a mean with the 
outliers thrown out - values more than a few median-absolute-deviations from 
the middle are dropped. Use it to settle on one good estimate from a noisy 
stream, where a plain running mean would be pulled off by the bad frames.
""" + AXIS + """
SYNTAX:
np.mean
np.robust_mean <threshold: float>

EXAMPLE:
np.mean

INPUTS and PARAMETERS:

in:
The array. Receiving it triggers the calculation. 
Lists and single numbers are accepted and converted.

axis:
Which direction to reduce along - see above.

threshold (np.robust_mean):
How many median-absolute-deviations from the middle a sample may be and still 
count. Default 3.0. Lower is stricter.

reset / samples (np.robust_mean):
Clear the accumulated samples, and how many are being kept.

OUTPUTS: 

out:
The result. A single number when no axis is set, an array of results when one is.

inlier count (np.robust_mean):
How many samples survived the rejection - worth watching, because if it is 
falling towards zero your threshold is too strict."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 'np.rand 4 4', 'pos': (30, 120), 'w': 140, 'h': 130,
     'props': {'min': 0.0, 'max': 1.0, 'dim 0': 4, 'dim 1': 4, 'dtype': 'float32'}},
    {'key': 'c0', 'comment': True, 'text': 'click for a new 4 by 4 of random numbers',
     'pos': (30, 262)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (230, 120), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 4,
               'min y': 0.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'mn', 'init': 'np.mean', 'pos': (30, 305), 'w': 140, 'h': 70},
    {'key': 'f1', 'init': 'float', 'pos': (30, 390), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'no axis: one number for the whole array',
     'pos': (30, 440)},
    {'key': 'mx', 'init': 'np.amax', 'pos': (230, 305), 'w': 140, 'h': 70},
    {'key': 'f2', 'init': 'float', 'pos': (230, 390), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'am', 'init': 'np.argmax', 'pos': (230, 480), 'w': 140, 'h': 70},
    {'key': 'i1', 'init': 'int', 'pos': (230, 565), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'argmax gives WHERE, not what',
     'pos': (230, 615)},
]
links = [('btn', '', 'rnd', ''),
         ('rnd', '', 'hm', 'y', 0),
         ('rnd', '', 'mn', 'in', 0), ('mn', '', 'f1', '', 0),
         ('rnd', '', 'mx', 'in', 0), ('mx', '', 'f2', '', 0),
         ('rnd', '', 'am', 'in', 0), ('am', '', 'i1', '', 0)]
print(build('np_stats', 'np statistics - reduce an array to an answer', body,
            demo, links, demo_width=460, text_width=820, text_height=760))

# ----------------------------------------------------------------- np_linalg
body = """These nodes do linear algebra: the arithmetic of vectors and matrices.

A vector is a direction and a length. A matrix is a transformation - a rotation, 
a scaling, a projection. Most of what these nodes do is either measuring 
vectors or applying and undoing transformations, and the reason they matter in 
a patch is that positions, orientations and motions are all naturally these 
things rather than lists of separate numbers.

MEASURING:

np.linalg.norm       the length of a vector
euclidean_distance   the same node, named for the commoner use - 
                     feed it the difference between two positions and you get 
                     the distance between them
np.dot               the dot product: how much two vectors point the same way. 
                     Zero means at right angles; for unit vectors it is the 
                     cosine of the angle between them
np.cross             the cross product: a vector at right angles to both 
                     inputs, whose length reflects how much they differ in 
                     direction. This is how you get a surface normal, or an 
                     axis of rotation

MULTIPLYING:

np.matmul   matrix multiplication - apply a transformation, or compose two 
            into one
np.inner    the inner product
np.outer    the outer product: every element of one against every element of 
            the other, giving a matrix from two vectors

SOLVING AND INVERTING:

np.linalg.solve         solve A x = b for x - the right way to answer 
                        "what input produces this output"
np.linalg.inv           the inverse of a matrix: the transformation that undoes it
np.linalg.det           the determinant. Zero means the matrix flattens space 
                        and cannot be inverted
np.linalg.matrix_rank   how many independent directions the matrix actually 
                        spans - a check on whether your data is as 
                        multi-dimensional as you think

SYNTAX:
np.linalg.norm
np.matmul

EXAMPLE:
euclidean_distance

INPUTS and PARAMETERS:

input / in 1 / in 2:
The vectors or matrices. The last one to arrive triggers the calculation.

axis:
For norm and distance: which direction to measure along. 
Without it, the whole array is treated as one long vector; with axis set you 
get one length per row or column - which is how you measure a whole set of 
vectors in one go.

A matrix / b (np.linalg.solve):
The transformation and the result you want, giving you the input that produces it.

OUTPUTS: 

norm / dot product / cross product / mat mul result / x:
The result, named for the operation.

SOLVE, DO NOT INVERT:
When you want to undo a transformation, np.linalg.solve is both faster and 
numerically better behaved than inverting with np.linalg.inv and then 
multiplying. Reach for inv only when you genuinely need the inverse matrix 
itself, rather than an answer that happens to involve it."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 160, 'h': 42,
     'props': {'text in': '3.0 4.0 0.0', 'font size': '24'}},
    {'key': 'm2', 'init': 'message', 'pos': (210, 118), 'w': 160, 'h': 42,
     'props': {'text in': '0.0 1.0 0.0', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'two vectors', 'pos': (30, 168)},
    {'key': 'nm', 'init': 'np.linalg.norm', 'pos': (30, 205), 'w': 170, 'h': 70},
    {'key': 'f1', 'init': 'float', 'pos': (30, 290), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'length of the first: 5', 'pos': (30, 340)},
    {'key': 'dt', 'init': 'np.dot', 'pos': (230, 205), 'w': 150, 'h': 70},
    {'key': 'f2', 'init': 'float', 'pos': (230, 290), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'cr', 'init': 'np.cross', 'pos': (230, 380), 'w': 150, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (230, 465), 'w': 180, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'at right angles to both', 'pos': (230, 515)},
]
links = [('btn', '', 'm1', ''), ('btn', '', 'm2', ''),
         ('m1', 'message out', 'nm', 'input'), ('nm', 'norm', 'f1', ''),
         ('m2', 'message out', 'dt', 'in 2'), ('m1', 'message out', 'dt', 'in 1'),
         ('dt', 'dot product', 'f2', ''),
         ('m2', 'message out', 'cr', 'in 2'), ('m1', 'message out', 'cr', 'in 1'),
         ('cr', 'cross product', 'l1', '')]
print(build('np_linalg', 'np linear algebra - vectors and matrices', body,
            demo, links, demo_width=440, text_width=830, text_height=780))


# ------------------------------------------------- np reshape nodes: one page each
# np.reshape, np.shape, flatten (+ np.ravel), np.expand_dims (+ np.unsqueeze),
# np.squeeze, np.transpose. These were one family page (np_reshape); each now
# has its own page whose demo shows the array before and after, with the shape
# as text, and a SEE ALSO row of help_links under it. flatten / np.ravel and
# np.expand_dims / np.unsqueeze are registered to the same class and do the
# same thing, so each pair shares a page.
#
# Most demos start from the same readable pattern: np.linspace 0 1 12, a ramp
# of twelve values 0, 0.09 ... 1, each heat_map cell printing its value, so you
# can follow where every number goes. A metro re-sends the ramp 10 times a
# second, so a parameter changed on a node shows at once.


def HM(width, height, count, miny=0.0, maxy=1.0, fmt='%.2f', color='viridis'):
    """Property block for a heat_map that draws a whole array as one picture."""
    return {'color': color, 'width': width, 'height': height, 'sample count': count,
            'min y': miny, 'max y': maxy, 'update_mode': 'heat_map',
            'number format': fmt}


LIST = {'text in': '', 'font size': '24'}


def MSG(text):
    return {'text in': text, 'font size': '24'}


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
    return row


def ramp(y_strip=None):
    """metro 100 -> np.linspace 0 1 12, optionally drawn as a one-row strip."""
    nodes = [
        {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
         'props': {'on': True, 'period': 100.0, 'units': 'milliseconds'}},
        {'key': 'ls', 'init': 'np.linspace 0 1 12', 'pos': (30, 150), 'w': 190, 'h': 150,
         'props': {'start': 0.0, 'stop': 1.0, 'steps': 12, 'dtype': 'float32'}},
        {'key': 'cr', 'comment': True, 'text': 'twelve values 0 .. 1, sent 10 times a second',
         'pos': (30, 155)},
    ]
    if y_strip is not None:
        nodes.append({'key': 'hin', 'init': 'heat_map', 'pos': (30, y_strip), 'w': 512,
                      'h': 90, 'props': HM(492, 40, 1)})
    return nodes


RAMP_LINKS = [('met', '', 'ls', '')]

# ------------------------------------------------------------------- np.reshape
body = """Rearranges an array into a new shape, keeping every number and its order.

Twelve numbers can be one row of twelve, three rows of four, four rows of
three or a column of twelve. np.reshape changes which of these you have;
the numbers themselves are untouched. Reach for it when a node wants its
data in a particular layout - a list of 3-D points as rows of three, an
image as rows and columns, a 2-D result as one long row.

HOW THE NUMBERS ARE PLACED:
The array is read row by row, as it would print, and the new shape is
filled row by row in the same order. So a ramp 0 .. 1 reshaped to 3 4
puts the first four values in row 0, the next four in row 1 and so on -
the ramp still reads left to right, top to bottom.

SYNTAX:
np.reshape <dim> <dim> ...
With no arguments the shape is -1: everything in one row, like flatten.

EXAMPLE:
np.reshape 3 4

INPUTS and PARAMETERS:

tensor in:
The array (a list or a single number is turned into an array first).
Receiving it is what makes the node send.

shape (the text box, unnamed inlet):
The new shape. Any numbers in the text are taken as the dimensions, so
3 4, [3, 4] and 3,4 all mean three rows of four. To set it from a
cord, send it as a STRING (from a string box, as in the demo): a list -
which is what a message box sends - is not accepted and raises an error
on the console. The dimensions must multiply to the number of values you
actually have: 3 4 works for twelve and fails for ten. One dimension may be -1, meaning "whatever fits": 6 -1 gives six
rows of however many columns are needed.

Changing the shape does NOT send by itself - the next array uses it. In
the demo the metro re-sends the ramp, so a click on one of the string
boxes (or an edit in the text box) shows at once.

If the shape does not fit, an error is printed to the console and
nothing is sent; the heat_map keeps its last picture.

OUTPUTS:

output:
The same numbers in the new shape.

READING THE DEMO:
The strip at the top is the ramp as it arrives: shape [12], one row.
Click the string 3 4 and the heat_map below shows it folded into three rows of four,
the values still in order reading across each row. 4 3 and 2 6 fold it
differently; 6 -1 lets np.reshape work out the 2; 12 1 makes a column -
twelve rows of one value each, the ramp now running down the picture.
np.shape prints each result's shape.

RELATED:
flatten (np.ravel) goes back to one row. np.expand_dims and np.squeeze
add or remove a single axis of length 1 without spelling out the whole
shape. np.transpose is NOT a reshape: it turns rows into columns, which
moves the numbers. np.shape reports the shape. t.reshape is the PyTorch
version."""

demo = ramp(320) + [
    {'key': 'c0', 'comment': True, 'text': 'in: shape [12], one row', 'pos': (30, 325)},
    {'key': 'm1', 'init': 'string', 'pos': (30, 430), 'w': 130, 'h': 42, 'props': MSG('3 4')},
    {'key': 'm2', 'init': 'string', 'pos': (175, 430), 'w': 130, 'h': 42, 'props': MSG('4 3')},
    {'key': 'm3', 'init': 'string', 'pos': (320, 430), 'w': 130, 'h': 42, 'props': MSG('2 6')},
    {'key': 'm4', 'init': 'string', 'pos': (465, 430), 'w': 130, 'h': 42, 'props': MSG('6 -1')},
    {'key': 'm5', 'init': 'string', 'pos': (610, 430), 'w': 130, 'h': 42, 'props': MSG('12 1')},
    {'key': 'c1', 'comment': True, 'text': 'click a shape, or type one in the box',
     'pos': (30, 435)},
    {'key': 'rs', 'init': 'np.reshape 3 4', 'pos': (30, 500), 'w': 260, 'h': 70},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 590), 'w': 320, 'h': 230,
     'props': HM(300, 180, 3)},
    {'key': 'c2', 'comment': True, 'text': 'out: the same twelve values, refolded',
     'pos': (30, 595)},
    {'key': 'shp', 'init': 'np.shape', 'pos': (380, 590), 'w': 110, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (380, 680), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c3', 'comment': True, 'text': 'its shape: [rows, columns]', 'pos': (380, 685)},
] + see_also(850, ['flatten', 'np.expand_dims', 'np.squeeze', 'np.transpose',
                   'np.shape', 't.reshape'])
links = RAMP_LINKS + [
    ('ls', 'linspace out', 'hin', 'y'), ('ls', 'linspace out', 'rs', 'tensor in'),
    ('m1', 'string out', 'rs', '', 0, 1), ('m2', 'string out', 'rs', '', 0, 1),
    ('m3', 'string out', 'rs', '', 0, 1), ('m4', 'string out', 'rs', '', 0, 1),
    ('m5', 'string out', 'rs', '', 0, 1),
    ('rs', 'output', 'hout', 'y'), ('rs', 'output', 'shp', 'np in'),
    ('shp', 'shape', 'l1', '')]
print(build('np.reshape', 'np.reshape - the same numbers in a new shape', body,
            demo, links, demo_width=760, text_width=760, text_height=800))

# --------------------------------------------------------------------- np.shape
body = """Reports the shape of an array: how many entries along each axis, as a list.

It is the first thing to check when a node refuses data or does something
odd with it - most "why won't this work" problems in array work are a
shape mismatch rather than a data problem, and the shape is often not
what the picture suggests.

WHAT THE LIST MEANS:
One number per axis. [3, 5] is three rows of five. [12] is a plain row of
twelve, a 1-D array. [1, 12] holds the same twelve numbers as a 2-D array
with a single row - many nodes treat these two differently even though
they draw the same. A colour image is [rows, columns, 3]. A single number
is measured as an array of one value: [1].

SYNTAX:
np.shape
It takes no arguments.

INPUTS:

np in:
Any array. A list is measured as the array it would become, and a torch
tensor is measured too. Receiving it is what makes the node send.

OUTPUTS:

shape:
A list of whole numbers, one per axis, outermost first.

READING THE DEMO:
Top: change dim 0 or dim 1 on np.rand, then click its button. The
heat_map shows the new array and np.shape follows - dim 0 is the number
of rows, dim 1 the number of columns.

Bottom: two arrays that LOOK identical. The first strip is np.linspace's
output, shape [8]. The second is the same strip after np.reshape 1 8:
shape [1, 8]. Only np.shape can tell them apart.

RELATED:
t.info reports the shape of a torch tensor, along with its type and
device. np.reshape changes a shape; np.expand_dims and np.squeeze add or
remove axes of length 1 - the difference between [8] and [1, 8]."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'starts both demos when the page opens',
     'pos': (30, 70)},
    {'key': 'rnd', 'init': 'np.rand 3 5', 'pos': (30, 130), 'w': 160, 'h': 160,
     'props': {'min': 0.0, 'max': 1.0, 'dim 0': 3, 'dim 1': 5, 'dtype': 'float32'}},
    {'key': 'c1', 'comment': True, 'text': 'change dim 0 / dim 1, then click the button',
     'pos': (30, 135)},
    {'key': 'hr', 'init': 'heat_map', 'pos': (220, 130), 'w': 270, 'h': 200,
     'props': HM(250, 150, 3, fmt='none')},
    {'key': 'shr', 'init': 'np.shape', 'pos': (30, 350), 'w': 110, 'h': 70},
    {'key': 'lr', 'init': 'list', 'pos': (160, 350), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c2', 'comment': True, 'text': '[rows, columns]', 'pos': (160, 355)},
    {'key': 'ls', 'init': 'np.linspace 0 1 8', 'pos': (30, 470), 'w': 190, 'h': 150,
     'props': {'start': 0.0, 'stop': 1.0, 'steps': 8, 'dtype': 'float32'}},
    {'key': 'ha', 'init': 'heat_map', 'pos': (30, 640), 'w': 340, 'h': 90,
     'props': HM(320, 40, 1)},
    {'key': 'sha', 'init': 'np.shape', 'pos': (400, 640), 'w': 110, 'h': 70},
    {'key': 'la', 'init': 'list', 'pos': (530, 640), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c3', 'comment': True, 'text': 'a 1-D row of 8', 'pos': (530, 645)},
    {'key': 'rs', 'init': 'np.reshape 1 8', 'pos': (30, 750), 'w': 260, 'h': 70},
    {'key': 'hb', 'init': 'heat_map', 'pos': (30, 840), 'w': 340, 'h': 90,
     'props': HM(320, 40, 1)},
    {'key': 'shb', 'init': 'np.shape', 'pos': (400, 840), 'w': 110, 'h': 70},
    {'key': 'lb2', 'init': 'list', 'pos': (530, 840), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c4', 'comment': True, 'text': 'the same picture: a 2-D array of 1 row',
     'pos': (530, 845)},
] + see_also(960, ['t.info', 'np.reshape', 'np.expand_dims', 'np.squeeze'])
links = [('lb', 'out', 'rnd', ''), ('lb', 'out', 'ls', ''),
         ('rnd', 'random array', 'hr', 'y'), ('rnd', 'random array', 'shr', 'np in'),
         ('shr', 'shape', 'lr', ''),
         ('ls', 'linspace out', 'ha', 'y'), ('ls', 'linspace out', 'sha', 'np in'),
         ('sha', 'shape', 'la', ''),
         ('ls', 'linspace out', 'rs', 'tensor in'),
         ('rs', 'output', 'hb', 'y'), ('rs', 'output', 'shb', 'np in'),
         ('shb', 'shape', 'lb2', '')]
print(build('np.shape', 'np.shape - how many along each axis', body,
            demo, links, demo_width=700, text_width=760, text_height=740))

# ---------------------------------------------------------- flatten / np.ravel
body = """Collapses an array of any shape into one long row of all its values.

flatten and np.ravel are two names for the same node. Use it when a node
wants a plain list of numbers - a plot, a list display, a send to another
program - or to undo a reshape. The values are not changed, only laid
end to end.

THE ORDER:
order C (the default) reads row by row, as the array prints: all of row
0, then all of row 1, and so on. A ramp that np.reshape folded into rows
comes back out exactly as it went in.
order F reads column by column instead: down column 0, then column 1.
order A and K read the numbers in the order they sit in memory. For
almost every array in a patch that is the same as C; they differ only
for an array that came straight out of np.transpose, where they read it
as F would.

SYNTAX:
flatten <order>
np.ravel <order>
order is C, F, A or K; default C.

EXAMPLE:
flatten F

INPUTS and PARAMETERS:

input:
The array (a list or number is turned into an array first). Receiving it
is what makes the node send.

order (menu):
As above. Changing it does not send by itself - the next array uses it.
In the demo the metro re-sends the ramp, so the strip changes at once.

OUTPUTS:

flattened array:
A 1-D array of every value, shape [rows x columns x ...].

READING THE DEMO:
The ramp is folded into 3 rows of 4 (the middle heat_map), then
flattened. With order C the bottom strip is the original ramp again,
0 rising to 1. Switch order to F and it reads down the columns instead:
0, 0.36, 0.73, then 0.09, 0.45, 0.82 ... - three short climbs, one per
column. The shape is [12] either way.

RELATED:
np.reshape is the reverse, and np.reshape -1 does the same as flatten C.
np.squeeze removes just one length-1 axis rather than all structure.
np.transpose swaps rows and columns first, if you want to flatten the
other way without changing order. t.reshape covers the PyTorch versions
(t.flatten, t.ravel)."""

demo = ramp() + [
    {'key': 'rs', 'init': 'np.reshape 3 4', 'pos': (30, 320), 'w': 260, 'h': 70},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 410), 'w': 260, 'h': 200,
     'props': HM(240, 150, 3)},
    {'key': 'c0', 'comment': True, 'text': 'in: 3 rows of 4, reading across',
     'pos': (30, 415)},
    {'key': 'fl', 'init': 'flatten', 'pos': (30, 630), 'w': 160, 'h': 90,
     'props': {'order': 'C'}},
    {'key': 'c1', 'comment': True, 'text': 'switch order between C and F',
     'pos': (30, 635)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 740), 'w': 512, 'h': 90,
     'props': HM(492, 40, 1)},
    {'key': 'c2', 'comment': True, 'text': 'out: one row of twelve', 'pos': (30, 745)},
    {'key': 'shp', 'init': 'np.shape', 'pos': (30, 850), 'w': 110, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (160, 850), 'w': 150, 'h': 42, 'props': LIST},
] + see_also(950, ['np.reshape', 'np.squeeze', 'np.transpose', 'np.shape', 't.reshape'])
links = RAMP_LINKS + [
    ('ls', 'linspace out', 'rs', 'tensor in'),
    ('rs', 'output', 'hin', 'y'), ('rs', 'output', 'fl', 'input'),
    ('fl', 'flattened array', 'hout', 'y'), ('fl', 'flattened array', 'shp', 'np in'),
    ('shp', 'shape', 'l1', '')]
print(build('flatten', 'flatten / np.ravel - every value in one row', body,
            demo, links, demo_width=560, text_width=760, text_height=800))

# ------------------------------------------------- np.expand_dims / np.unsqueeze
body = """Adds a new axis of length 1 to an array, without changing any values.

np.expand_dims and np.unsqueeze (the PyTorch name) are the same node. Use
it when a node insists on a 2-D array and you have a 1-D one: a row of
twelve, shape [12], becomes a 2-D array of one row, [1, 12], or of one
column, [12, 1]. It is also how you give a single frame a "batch" or
"time" axis so it can be stacked with others.

WHERE THE AXIS GOES:
axis says where in the shape list the new 1 is inserted. For [12]:
axis 0 gives [1, 12], one row. axis 1 gives [12, 1], one column.
For a [3, 4] array: axis 0 gives [1, 3, 4], 1 gives [3, 1, 4] and 2
gives [3, 4, 1].

SYNTAX:
np.expand_dims <axis: int>
np.unsqueeze <axis: int>
Default axis 0.

EXAMPLE:
np.expand_dims 1

INPUTS and PARAMETERS:

input:
The array (a list or number is turned into an array first). Receiving it
is what makes the node send.

axis (on the node):
Where to insert the new axis, from 0 up to the number of axes the input
has. Values past the end are treated as the end; negative values are
treated as 0 (in numpy itself -1 would mean the end - not here). Changing
it does not send by itself; in the demo the metro re-sends the ramp.

OUTPUTS:

array out:
The same values with one more axis.

READING THE DEMO:
The ramp arrives as a row, shape [12]. With axis 1 it comes out as a
column, [12, 1]: the heat_map draws twelve rows of one cell, the ramp
running down. Set axis to 0 and it is [1, 12], one row - drawn just like
the input, though it is now 2-D. A heat_map always draws a 1-D array as
a row, so the change only shows when the new axis makes columns out of
the values; np.shape shows it either way.

RELATED:
np.squeeze is the reverse: it takes a length-1 axis away. np.reshape can
do the same job when you know the whole shape (np.reshape 12 1). np.shape
shows the result. t.reshape covers the PyTorch version, t.unsqueeze."""

demo = ramp(320) + [
    {'key': 'c0', 'comment': True, 'text': 'in: shape [12], one row', 'pos': (30, 325)},
    {'key': 'shi', 'init': 'np.shape', 'pos': (30, 430), 'w': 110, 'h': 70},
    {'key': 'li', 'init': 'list', 'pos': (160, 430), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'ex', 'init': 'np.expand_dims 1', 'pos': (30, 530), 'w': 180, 'h': 90,
     'props': {'axis': 1}},
    {'key': 'c1', 'comment': True, 'text': 'set axis to 0 or 1', 'pos': (30, 535)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 640), 'w': 260, 'h': 290,
     'props': HM(240, 240, 12)},
    {'key': 'c2', 'comment': True, 'text': 'out: axis 1 makes a column of twelve',
     'pos': (30, 645)},
    {'key': 'sho', 'init': 'np.shape', 'pos': (320, 640), 'w': 110, 'h': 70},
    {'key': 'lo', 'init': 'list', 'pos': (320, 730), 'w': 150, 'h': 42, 'props': LIST},
] + see_also(960, ['np.squeeze', 'np.reshape', 'np.shape', 't.reshape'])
links = RAMP_LINKS + [
    ('ls', 'linspace out', 'hin', 'y'), ('ls', 'linspace out', 'shi', 'np in'),
    ('shi', 'shape', 'li', ''),
    ('ls', 'linspace out', 'ex', 'input'),
    ('ex', 'array out', 'hout', 'y'), ('ex', 'array out', 'sho', 'np in'),
    ('sho', 'shape', 'lo', '')]
print(build('np.expand_dims', 'np.expand_dims / np.unsqueeze - add an axis of length 1',
            body, demo, links, demo_width=560, text_width=760, text_height=800))

# -------------------------------------------------------------------- np.squeeze
body = """Removes one axis of length 1 from an array, without changing any values.

A result often comes out with a stray extra axis - [12, 1] where you
wanted a plain row of twelve, or [1, 3, 4] where you wanted [3, 4].
np.squeeze takes that axis away. It is the reverse of np.expand_dims.

ONE AXIS AT A TIME:
Unlike numpy's own squeeze, this node never removes every length-1 axis
at once: it removes the single axis you name, and only if that axis
really has length 1. If it does not, nothing is sent (the reason is
printed to the console when the app is verbose).

SYNTAX:
np.squeeze <axis: int>
Default axis 0.

EXAMPLE:
np.squeeze 1

INPUTS and PARAMETERS:

input:
The array (a list or number is turned into an array first). Receiving it
is what makes the node send - if the axis can be removed.

axis (on the node):
Which axis to remove, counting from 0. It must have length 1. Negative
values (numpy's "count from the end") are not accepted here: nothing is
sent. Changing it does not send by itself; in the demo the metro
re-sends the ramp.

OUTPUTS:

squeezed array:
The same values with that axis gone.

READING THE DEMO:
np.reshape 12 1 turns the ramp into a column, shape [12, 1]. np.squeeze 1
removes the length-1 axis: shape [12], drawn as a row. Now click 1 12 to
make the input a single row, [1, 12]: axis 1 has length 12, so nothing
comes out and the bottom displays freeze. Set axis to 0 and it flows
again - [1, 12] squeezed on axis 0 is [12].

RELATED:
np.expand_dims (np.unsqueeze) adds a length-1 axis. flatten removes all
structure at once, whatever the shape. np.reshape can drop the axis too
if you spell out the shape. t.reshape covers the PyTorch version,
t.squeeze."""

demo = ramp() + [
    {'key': 'm1', 'init': 'string', 'pos': (30, 320), 'w': 130, 'h': 42, 'props': MSG('12 1')},
    {'key': 'm2', 'init': 'string', 'pos': (175, 320), 'w': 130, 'h': 42, 'props': MSG('1 12')},
    {'key': 'c0', 'comment': True, 'text': 'a column, or a row', 'pos': (30, 325)},
    {'key': 'rs', 'init': 'np.reshape 12 1', 'pos': (30, 390), 'w': 260, 'h': 70},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 480), 'w': 260, 'h': 290,
     'props': HM(240, 240, 12)},
    {'key': 'c1', 'comment': True, 'text': 'in: [12, 1], a column of twelve',
     'pos': (30, 485)},
    {'key': 'shi', 'init': 'np.shape', 'pos': (320, 480), 'w': 110, 'h': 70},
    {'key': 'li', 'init': 'list', 'pos': (320, 570), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'sq', 'init': 'np.squeeze 1', 'pos': (30, 790), 'w': 180, 'h': 90,
     'props': {'axis': 1}},
    {'key': 'c2', 'comment': True, 'text': 'axis must have length 1, or nothing is sent',
     'pos': (30, 795)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 900), 'w': 512, 'h': 90,
     'props': HM(492, 40, 1)},
    {'key': 'c3', 'comment': True, 'text': 'out: [12], drawn as a row', 'pos': (30, 905)},
    {'key': 'sho', 'init': 'np.shape', 'pos': (30, 1010), 'w': 110, 'h': 70},
    {'key': 'lo', 'init': 'list', 'pos': (160, 1010), 'w': 150, 'h': 42, 'props': LIST},
] + see_also(1100, ['np.expand_dims', 'flatten', 'np.reshape', 'np.shape', 't.reshape'])
links = RAMP_LINKS + [
    ('ls', 'linspace out', 'rs', 'tensor in'),
    ('m1', 'string out', 'rs', '', 0, 1), ('m2', 'string out', 'rs', '', 0, 1),
    ('rs', 'output', 'hin', 'y'), ('rs', 'output', 'shi', 'np in'),
    ('shi', 'shape', 'li', ''),
    ('rs', 'output', 'sq', 'input'),
    ('sq', 'squeezed array', 'hout', 'y'), ('sq', 'squeezed array', 'sho', 'np in'),
    ('sho', 'shape', 'lo', '')]
print(build('np.squeeze', 'np.squeeze - remove an axis of length 1', body,
            demo, links, demo_width=560, text_width=760, text_height=800))

# ------------------------------------------------------------------ np.transpose
body = """Turns an array's rows into columns: the value at row r, column c moves to row c, column r.

A 3 by 4 array becomes 4 by 3, flipped across its top-left to
bottom-right diagonal. Use it when data arrives the wrong way round -
joints as columns when a node wants them as rows, a matrix that needs its
transpose for a calculation, an image whose x and y are swapped.

TRANSPOSE IS NOT RESHAPE:
np.reshape 4 3 and np.transpose both turn [3, 4] into [4, 3], but they
put the numbers in different places. reshape keeps the reading order and
refolds; transpose keeps each number's row and column and swaps them. In
the demo a ramp that reads ACROSS each row comes out reading DOWN each
column.

SYNTAX:
np.transpose
An argument is accepted but has no effect.

INPUTS:

input:
The array (a list or number is turned into an array first). Receiving it
is what makes the node send.

There is no axis choice: the order of ALL the axes is reversed. A 2-D
array swaps rows and columns; a 1-D row is returned unchanged; a 3-D
[a, b, c] array becomes [c, b, a].

OUTPUTS:

transposed array:
The transposed array.

READING THE DEMO:
The upper heat_map is the ramp folded into 3 rows of 4, rising across
each row. The lower one is its transpose: 4 rows of 3, rising down each
column - the first row of the input is now the first column. Click 2 6 or
1 12 to transpose other shapes; 1 12 becomes a column of twelve.

RELATED:
np.reshape changes the shape without moving any number out of reading
order. flatten reads an array out into one row, by rows (C) or columns
(F). np.shape shows the shapes. t.reshape covers the PyTorch versions,
t.transpose and t.permute (which, unlike this node, choose which axes to
swap)."""

demo = ramp() + [
    {'key': 'm1', 'init': 'string', 'pos': (30, 320), 'w': 130, 'h': 42, 'props': MSG('3 4')},
    {'key': 'm2', 'init': 'string', 'pos': (175, 320), 'w': 130, 'h': 42, 'props': MSG('2 6')},
    {'key': 'm3', 'init': 'string', 'pos': (320, 320), 'w': 130, 'h': 42, 'props': MSG('1 12')},
    {'key': 'c0', 'comment': True, 'text': 'pick the input shape', 'pos': (30, 325)},
    {'key': 'rs', 'init': 'np.reshape 3 4', 'pos': (30, 390), 'w': 260, 'h': 70},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 480), 'w': 320, 'h': 230,
     'props': HM(300, 180, 3)},
    {'key': 'c1', 'comment': True, 'text': 'in: 3 rows of 4, rising across',
     'pos': (30, 485)},
    {'key': 'shi', 'init': 'np.shape', 'pos': (380, 480), 'w': 110, 'h': 70},
    {'key': 'li', 'init': 'list', 'pos': (380, 570), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'tr', 'init': 'np.transpose', 'pos': (30, 730), 'w': 160, 'h': 60},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 810), 'w': 320, 'h': 230,
     'props': HM(300, 180, 4)},
    {'key': 'c2', 'comment': True, 'text': 'out: 4 rows of 3, rising down',
     'pos': (30, 815)},
    {'key': 'sho', 'init': 'np.shape', 'pos': (380, 810), 'w': 110, 'h': 70},
    {'key': 'lo', 'init': 'list', 'pos': (380, 900), 'w': 150, 'h': 42, 'props': LIST},
] + see_also(1070, ['np.reshape', 'flatten', 'np.shape', 't.reshape'])
links = RAMP_LINKS + [
    ('ls', 'linspace out', 'rs', 'tensor in'),
    ('m1', 'string out', 'rs', '', 0, 1), ('m2', 'string out', 'rs', '', 0, 1),
    ('m3', 'string out', 'rs', '', 0, 1),
    ('rs', 'output', 'hin', 'y'), ('rs', 'output', 'shi', 'np in'),
    ('shi', 'shape', 'li', ''),
    ('rs', 'output', 'tr', 'input'),
    ('tr', 'transposed array', 'hout', 'y'), ('tr', 'transposed array', 'sho', 'np in'),
    ('sho', 'shape', 'lo', '')]
print(build('np.transpose', 'np.transpose - rows become columns', body,
            demo, links, demo_width=560, text_width=760, text_height=800))
