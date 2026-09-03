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

# ---------------------------------------------------------------- np_reshape
body = """These nodes change an array's SHAPE without changing the numbers in it.

An array of twelve numbers can be twelve in a row, or three rows of four, or 
four rows of three, or a single 1 by 12 slab. The numbers are identical; 
only the arrangement differs. Nodes downstream care a great deal about the 
arrangement, and most "why won't this connect" problems in array work are a 
shape mismatch rather than a data problem.

THE NODES:

np.shape         report the current shape - the first thing to check when 
                 something is refusing data
np.reshape       rearrange into a shape you specify
flatten          collapse everything into one long row
np.ravel         the same as flatten
np.transpose     swap the axes over - rows become columns
np.expand_dims   add a new axis of length 1
np.unsqueeze     the same as expand_dims, under the torch name
np.squeeze       remove axes of length 1

WHY LENGTH-1 AXES MATTER:
An array of shape (3,) and one of shape (1, 3) hold the same three numbers, 
and many operations treat them differently - one is a vector, the other a 
matrix with a single row. expand_dims and squeeze exist to convert between the 
two without touching the data. When a node insists it wants a 2D array and you 
have a 1D one, expand_dims is usually the answer; when a result comes out with 
a stray extra dimension, squeeze removes it.

SYNTAX:
np.reshape <dim> <dim> ...
np.expand_dims

EXAMPLE:
np.reshape 3 4

INPUTS and PARAMETERS:

input:
The array. Receiving it triggers the node.

shape (np.reshape):
The new shape, as a list of dimensions. They must multiply to the number of 
elements you actually have - 3 by 4 works for twelve numbers and fails for 
ten. A -1 means "work this one out for me", so "6 -1" gives six rows of 
whatever fits.

axis (expand_dims, squeeze):
Where to add or remove the length-1 axis.

order (flatten, np.ravel):
Whether to read the array row by row or column by column when flattening.

OUTPUTS: 

output / flattened array / transposed array / array out:
The same numbers, differently arranged.

shape:
A list of the dimensions.

TRANSPOSE DOES NOT MOVE ANYTHING:
Transposing swaps how the array is READ rather than reorganising memory, 
so it is effectively free however large the array. Reshaping is usually free 
too. Neither is a copy unless it has to be."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 'np.rand 3 4', 'pos': (30, 120), 'w': 140, 'h': 130,
     'props': {'min': 0.0, 'max': 1.0, 'dim 0': 3, 'dim 1': 4, 'dtype': 'float32'}},
    {'key': 'sh', 'init': 'np.shape', 'pos': (230, 120), 'w': 140, 'h': 60},
    {'key': 'l1', 'init': 'list', 'pos': (230, 195), 'w': 160, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'three rows of four', 'pos': (230, 245)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 270), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 4,
               'min y': 0.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'tr', 'init': 'np.transpose', 'pos': (30, 440), 'w': 160, 'h': 60},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (30, 515), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 3,
               'min y': 0.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c1', 'comment': True, 'text': 'the same numbers, read the other way',
     'pos': (30, 675)},
    {'key': 'fl', 'init': 'flatten', 'pos': (280, 440), 'w': 150, 'h': 60},
    {'key': 'sh2', 'init': 'np.shape', 'pos': (280, 515), 'w': 140, 'h': 60},
    {'key': 'l2', 'init': 'list', 'pos': (280, 590), 'w': 160, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'flattened: twelve in a row', 'pos': (280, 640)},
]
links = [('btn', '', 'rnd', ''),
         ('rnd', '', 'sh', 'np in', 0), ('sh', 'shape', 'l1', ''),
         ('rnd', '', 'hm', 'y', 0),
         ('rnd', '', 'tr', 'input', 0), ('tr', 'transposed array', 'hm2', 'y'),
         ('rnd', '', 'fl', 'input', 0), ('fl', 'flattened array', 'sh2', 'np in'),
         ('sh2', 'shape', 'l2', '')]
print(build('np_reshape', 'np shape - the same numbers, arranged differently', body,
            demo, links, demo_width=460, text_width=800, text_height=740))
