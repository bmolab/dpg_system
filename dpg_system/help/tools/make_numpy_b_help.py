"""np rearranging, generators, selecting, clipping."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# -------------------------------------------------------------- np_rearrange
body = """These nodes move the contents of an array around, or join arrays together.

The numbers are the same ones; what changes is their order or their company. 
Reordering matters more than it sounds - a great deal of array work is getting 
data into the arrangement some other node insists on.

REORDERING:

np.sort       put the values in order
np.argsort    give the INDICES that would sort it, rather than the sorted 
              values. Use this when you want to reorder something ELSE by 
              the same ranking - sort joints by speed, then apply that order 
              to their names
np.flip       reverse along an axis
np.roll       shift everything along, with what falls off one end reappearing 
              at the other. This is how you make a circular buffer, or a delay
np.rot90      rotate a 2D array by quarter turns
np.rotate     the same node
np.repeat     repeat each element a number of times

JOINING AND SPLITTING:

np.concatenate   join arrays end to end along an existing axis
np.stack         join them along a NEW axis
np.split         cut one array into several

The difference between concatenate and stack catches everyone once. 
Joining two arrays of 3 numbers with concatenate gives you 6 numbers in a row. 
With stack it gives you a 2 by 3 array - the originals kept separate, 
side by side. Concatenate extends, stack layers.

SYNTAX:
np.sort
np.roll <shift: int>
np.concatenate

EXAMPLE:
np.roll 1

INPUTS and PARAMETERS:

in / input:
The array. Receiving it triggers the node.

in 2 (concatenate, stack):
The second array. It must match the first in every dimension except the one 
being joined along.

axis:
Which direction to work along. As elsewhere, leaving it alone works on the 
flattened array; setting it works row-wise or column-wise.

shifts (np.roll):
How far to shift, and which way - negative goes the other way.

k / axis 1 / axis 2 (np.rot90):
How many quarter turns, and which plane to rotate in.

repeats (np.repeat):
How many copies of each element.

descending (np.sort, np.argsort):
Largest first rather than smallest.

OUTPUTS: 

The rearranged array, or - for np.split - one outlet per piece.

A NOTE ON np.roll:
Nothing is lost. Everything that falls off the end comes back at the start, 
which is what makes it a rotation rather than a shift. If you want values to 
fall off and zeros to arrive instead, that is not this node."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 250, 'h': 42,
     'props': {'text in': '5 3 9 1 7', 'font size': '24'}},
    {'key': 'so', 'init': 'np.sort', 'pos': (30, 180), 'w': 150, 'h': 70,
     'props': {'descending': False}},
    {'key': 'l1', 'init': 'list', 'pos': (30, 265), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'sorted: the values in order', 'pos': (30, 315)},
    {'key': 'ar', 'init': 'np.argsort', 'pos': (280, 180), 'w': 150, 'h': 70,
     'props': {'descending': False}},
    {'key': 'l2', 'init': 'list', 'pos': (280, 265), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'argsort: where each one came from',
     'pos': (280, 315)},
    {'key': 'ro', 'init': 'np.roll 1', 'pos': (30, 360), 'w': 150, 'h': 90,
     'props': {'shifts': 1}},
    {'key': 'l3', 'init': 'list', 'pos': (30, 465), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'rolled: nothing is lost, it wraps',
     'pos': (30, 515)},
    {'key': 'fp', 'init': 'np.flip', 'pos': (280, 360), 'w': 150, 'h': 70},
    {'key': 'l4', 'init': 'list', 'pos': (280, 465), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'flipped: reversed', 'pos': (280, 515)},
]
links = [('btn', '', 'm1', ''),
         ('m1', 'message out', 'so', 'in'), ('so', '', 'l1', '', 0),
         ('m1', 'message out', 'ar', 'in'), ('ar', '', 'l2', '', 0),
         ('m1', 'message out', 'ro', 'input'), ('ro', 'rolled array', 'l3', ''),
         ('m1', 'message out', 'fp', 'input'), ('fp', 'flipped array', 'l4', '')]
print(build('np_rearrange', 'np rearranging - reorder, join, split', body, demo, links,
            demo_width=500, text_width=820, text_height=740))

# -------------------------------------------------------------- np_generator
body = """These nodes make arrays out of nothing - the starting material for array work.

You need an array before you can do anything to one, and often the array you 
need is regular rather than measured: a block of zeros to accumulate into, 
a ramp to use as an x axis, a grid of positions to draw at, noise to test with.

THE NODES:

np.zeros      an array of zeros
np.ones       an array of ones
np.rand       an array of random numbers between limits you set
np.linspace   evenly spaced values from one number to another
np.grid       a 2D grid of coordinates
np.sequence   record a stream of arrays over time and play it back

np.linspace is the one to remember. Give it a start, a stop and a count and it 
divides the interval evenly - which is how you make an x axis to evaluate 
something across, a set of evenly spaced positions, or a smooth interpolation 
between two states.

np.grid gives you every (x, y) position in a rectangle at a spacing you choose, 
which is what you want for laying things out, sampling a field, or generating 
positions to draw.

np.sequence is different in kind: it records what passes through it, frame by 
frame, and can save that to disk and play it back. A way of capturing a 
performance of data and re-running it.

SYNTAX:
np.zeros <dim> <dim> ...
np.rand <dim> <dim> ...
np.linspace <start> <stop> <steps>
np.grid <min x> <max x> <min y> <max y> <divisions x> <divisions y>

EXAMPLE:
np.rand 4 4

INPUTS and PARAMETERS:

in:
A button on most of these - click it, or send anything, to produce a new array. 
np.rand gives you fresh numbers each time; the others give the same thing again.

dim 0, dim 1, ...:
The shape to produce.

min / max (np.rand):
The range the random numbers fall in.

dtype:
The number type - float32 is the usual choice, float64 for precision, 
int for whole numbers.

start / stop / steps (np.linspace):
The two ends and how many values between them. Both ends are included.

min x / max x / min y / max y / divisions x / divisions y (np.grid):
The rectangle and how finely to divide it.

data to record / current frame / save / load / path (np.sequence):
The stream to capture, where to play from, and where to keep it.

OUTPUTS: 

The generated array.

RELATED:
t.rand and the t.dist nodes do the same for PyTorch tensors, and the 
distribution nodes let you choose the SHAPE of the randomness rather than 
taking it flat."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 'np.rand 4 4', 'pos': (30, 120), 'w': 140, 'h': 130,
     'props': {'min': 0.0, 'max': 1.0, 'dim 0': 4, 'dim 1': 4, 'dtype': 'float32'}},
    {'key': 'hm', 'init': 'heat_map', 'pos': (230, 120), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 4,
               'min y': 0.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c0', 'comment': True, 'text': 'click for fresh random numbers',
     'pos': (30, 262)},
    {'key': 'ls', 'init': 'np.linspace 0.0 1.0 8', 'pos': (30, 305), 'w': 190, 'h': 130,
     'props': {'start': 0.0, 'stop': 1.0, 'steps': 8, 'dtype': 'float32'}},
    {'key': 'l1', 'init': 'list', 'pos': (30, 450), 'w': 300, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'evenly spaced, both ends included',
     'pos': (30, 500)},
    {'key': 'zr', 'init': 'np.zeros 8', 'pos': (30, 545), 'w': 140, 'h': 110,
     'props': {'dim 0': 8, 'dtype': 'float32'}},
    {'key': 'l2', 'init': 'list', 'pos': (30, 670), 'w': 300, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
]
links = [('btn', '', 'rnd', ''), ('rnd', '', 'hm', 'y', 0),
         ('btn', '', 'ls', ''), ('ls', 'linspace out', 'l1', ''),
         ('btn', '', 'zr', ''), ('zr', '', 'l2', '', 0)]
print(build('np_generator', 'np generators - arrays out of nothing', body, demo, links,
            demo_width=470, text_width=800, text_height=720))

# ----------------------------------------------------------------- np_select
body = """These nodes pick parts of an array out, by position or by condition.

Most work with arrays is not about the whole thing. You want the third row, 
or the middle of an image, or every value above a threshold. 
These are the four ways of saying which part.

THE NODES:

np.[]        take elements by index, the way you would slice in code
np.crop      cut a rectangle out of an image or 2D array
np.where     choose between two arrays element by element, according to a 
             condition - wherever the condition holds take one, elsewhere 
             take the other
np.argwhere  the same node, whose second outlet gives the INDICES where the 
             condition held rather than the values

np.where is the interesting one, and it is worth thinking of as a masked blend 
rather than a lookup. Feed it a comparison as the condition and two arrays for 
the true and false cases, and you have "use this where the reading is good and 
that where it is not" as a single operation across a whole array, with no loop 
and no branching.

The indices outlet answers a different question: not what the values are, 
but WHERE the interesting ones were. Feed the comparison in and you get back 
the positions of everything above the threshold - which joint, which frame, 
which pixel.

SYNTAX:
np.[] <indices>
np.crop
np.where

EXAMPLE:
np.[] 0

INPUTS and PARAMETERS:

tensor in / image in:
The array. Receiving it triggers the node.

Indices (np.[]):
Which elements to take.

left / top / right / bottom (np.crop):
The edges of the rectangle to keep.

uncrop (np.crop):
Puts the crop back into a full-sized frame of zeros instead of returning just 
the piece - so the result stays the same shape as the original, with 
everything outside the rectangle blanked.

condition (np.where):
An array of true and false values, the same shape as the data. 
This normally comes from a comparison node - feed an array into ">" and its 
result here.

if true / if false:
The two arrays to choose between. A single number works for either, and is 
used everywhere it is needed. 
BOTH must have been supplied before anything comes out - the node stays silent 
until it has each of them, so a np.where with only one branch wired produces 
nothing at all and gives no indication why.

OUTPUTS: 

selected:
The chosen values.

indices:
The positions where the condition held.

out array:
The cropped region."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 'np.rand 4 4', 'pos': (30, 120), 'w': 140, 'h': 130,
     'props': {'min': 0.0, 'max': 1.0, 'dim 0': 4, 'dim 1': 4, 'dtype': 'float32'}},
    {'key': 'hm', 'init': 'heat_map', 'pos': (230, 120), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 4,
               'min y': 0.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'gt', 'init': '> 0.5', 'pos': (30, 275), 'w': 130, 'h': 70,
     'props': {'output_type': 'bool'}},
    {'key': 'c0', 'comment': True, 'text': 'a mask: true where the value is high',
     'pos': (30, 355)},
    {'key': 'tz', 'init': 't 0.0', 'pos': (210, 275), 'w': 45, 'h': 46},
    {'key': 'wh', 'init': 'np.where', 'pos': (30, 395), 'w': 160, 'h': 100},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (230, 395), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 4,
               'min y': 0.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c1', 'comment': True, 'text': 'high values kept, the rest zeroed',
     'pos': (30, 510)},
    {'key': 'c2', 'comment': True, 'text': 'one operation across the whole array',
     'pos': (30, 540)},
]
# 'if false' must be fed as well -- np.where sends nothing until it has both
links = [('btn', '', 'rnd', ''), ('rnd', '', 'hm', 'y', 0),
         ('btn', '', 'tz', ''), ('tz', '0.0', 'wh', 'if false'),
         ('rnd', '', 'wh', 'if true', 0),
         ('rnd', '', 'gt', 'in', 0), ('gt', 'result', 'wh', 'condition'),
         ('wh', 'selected', 'hm2', 'y')]
print(build('np_select', 'np selecting - which part of the array', body, demo, links,
            demo_width=470, text_width=800, text_height=700))

# ------------------------------------------------------------------- np_clip
body = """These nodes hold every value in an array inside limits.

Anything below the minimum comes out as the minimum, anything above the maximum 
as the maximum, and everything in between passes untouched. 
It is the array version of the clamp node.

THE NODES:

np.clip   limit at both ends
np.max    limit at the bottom only - the result is never less than the value 
          you set
np.min    limit at the top only - never more

The names read backwards until you see where they come from: np.max means 
"take the maximum of the data and this number", which is exactly a lower 
bound. np.min is the upper bound for the same reason.

Use these to keep a signal in range before it drives something that would 
misbehave outside it, to keep colour components inside 0 to 1, or to floor a 
value at zero so nothing downstream sees a negative.

SYNTAX:
np.clip <min> <max>

EXAMPLE:
np.clip 0.0 1.0

INPUTS and PARAMETERS:

input:
The array. Receiving it triggers the node. Single numbers work too.

min / max:
The limits.

OUTPUTS: 

out array:
The limited array, in the same shape as the input.

RELATED:
clamp does the same for single values and small lists, with the limits on 
inlets so they can be driven from the patch. 
np.where lets you do something other than flattening at the limit - replace 
out-of-range values with zero, or with a different array altogether."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 3.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 3.0)},
    {'key': 'cl', 'init': 'np.clip -0.4 0.4', 'pos': (30, 232), 'w': 160, 'h': 100,
     'props': {'min': -0.4, 'max': 0.4}},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 350), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c0', 'comment': True, 'text': 'the peaks are flattened off',
     'pos': (30, 535)},
    {'key': 'c1', 'comment': True, 'text': 'drag min and max to see it change',
     'pos': (30, 565)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'cl', 'input'), ('cl', 'out array', 'p1', 'y')]
print(build('np_clip', 'np.clip - hold values inside limits', body, demo, links,
            demo_width=420, text_width=780, text_height=580))
