"""np generators (one page each) and clipping."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ------------------------------------------------- np generators: one page each
# np.zeros, np.ones, np.rand, np.linspace, np.grid, np.sequence. These were one
# family page (np_generator); each now has its own page whose demo SHOWS the
# array, with a SEE ALSO row of help_links under it.


def HM(width, height, count, miny=0.0, maxy=1.0, fmt='%.2f', color='viridis'):
    """Property block for a heat_map that draws a whole array as one picture."""
    return {'color': color, 'width': width, 'height': height, 'sample count': count,
            'min y': miny, 'max y': maxy, 'update_mode': 'heat_map',
            'number format': fmt}


LIST = {'text in': '', 'font size': '24'}


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


# ------------------------------------------------------------------ np.linspace
body = """Makes a 1-D array of evenly spaced values running from start to stop.

Reach for it whenever you need a regular series rather than a measured one:
an x axis to evaluate something across, evenly spaced positions, the steps
of a fade, or a smooth path from one state to another.

BOTH ENDS ARE INCLUDED:
steps is the number of VALUES, so there is one gap fewer than there are
values. With start 0, stop 1 and 9 steps the gap is (1 - 0) / 8 = 0.125,
and you get 0, 0.125, 0.25 ... 0.875, 1. In general the gap is
(stop - start) / (steps - 1). Start can be larger than stop: the values
then count down.

SYNTAX:
np.linspace <start: float> <stop: float> <steps: int>
Defaults: 0.0, 1.0 and 50.

EXAMPLE:
np.linspace 0 1 9

INPUTS and PARAMETERS:

in (button):
Click it, or send it anything, to produce the array. This is the only thing
that makes the node send. Changing start, stop or steps does NOT send by
itself - in the demo a metro clicks the node 20 times a second, so the
strip follows your drags.

start / stop (drag floats):
The first and last values.

steps (drag int):
How many values. 1 gives just start; 0 sends an empty array (the
heat_map keeps its last picture); a negative count is an error and
nothing is sent.

dtype (combo):
float32 (the default), float or double (both 64-bit), int64 or uint8.
The integer types cut each value down to a whole number, so
np.linspace 0 1 9 as int64 is eight 0s and a single 1.

OUTPUTS:

linspace out:
The 1-D array, steps values long.

READING THE DEMO:
The heat_map shows the ramp as a strip of cells, start on the left and
stop on the right, each coloured from the heat_map's min y (dark purple)
to its max y (yellow) - set here to 0 and 1, the default range. Lower stop
to 0.5 and the ramp only gets half way to yellow; swap start and stop and
it runs the other way; drag steps and the strip divides more finely.

A heat_map draws a 1-D array as one row like this. A 2-D array it
draws as it would print: row 0 at the top, columns left to right.

The heat_map's min y and max y are the scale, not the data: values outside
them saturate. If you change the range of the ramp a lot, change them too.

RELATED:
np.grid crosses two linspaces into a 2-D grid of points. t.linspace is the
PyTorch version. np.rand is the irregular counterpart - values anywhere in
a range rather than evenly spaced. np.reshape folds a long ramp into a 2-D
array, which is a quick way to make a gradient image."""

demo = [
    {'key': 'met', 'init': 'metro 50', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 50.0, 'units': 'milliseconds'}},
    {'key': 'c0', 'comment': True, 'text': 'clicks np.linspace 20 times a second',
     'pos': (30, 70)},
    {'key': 'ls', 'init': 'np.linspace 0 1 9', 'pos': (30, 150), 'w': 190, 'h': 150,
     'props': {'start': 0.0, 'stop': 1.0, 'steps': 9, 'dtype': 'float32'}},
    {'key': 'c1', 'comment': True, 'text': 'drag start, stop or steps on the node',
     'pos': (30, 155)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 320), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 'one cell per value, start on the left:',
     'pos': (30, 325)},
    {'key': 'c3', 'comment': True, 'text': '0 is dark purple, 1 is yellow',
     'pos': (30, 347)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 450), 'w': 468, 'h': 42, 'props': LIST},
    {'key': 'c4', 'comment': True, 'text': 'the same values as numbers',
     'pos': (30, 455)},
] + see_also(530, ['np.grid', 't.linspace', 'np.rand', 'np.reshape'])
links = [('met', '', 'ls', ''), ('ls', 'linspace out', 'hm', 'y'),
         ('ls', 'linspace out', 'l1', '')]
print(build('np.linspace', 'np.linspace - evenly spaced values from start to stop',
            body, demo, links, demo_width=520, text_width=760, text_height=760))

# ---------------------------------------------------------------------- np.rand
body = """Makes an array of random numbers, spread evenly between min and max.

Each click gives fresh numbers. Use it to test a chain with data that
changes, as noise to mix into a signal, for random positions or colours,
or wherever a starting point should be different every time.

The numbers are UNIFORM: every value between min and max is equally
likely, so the picture has no structure at all - no value turns up more
often than any other. For randomness with a shape (a bell curve, say)
see the t.dist nodes.

SYNTAX:
np.rand <dim 0: int> <dim 1: int> ...
One number per dimension. np.rand 12 12 is a 12 by 12 array; np.rand 100
is a row of 100. With no arguments you get a 1-element array. The NUMBER
of dimensions is fixed when the node is made - to go from a row to a
grid, retype the node.

EXAMPLE:
np.rand 12 12

INPUTS and PARAMETERS:

in (button):
The small button at the top left of the node. Click it, or send it
anything, for a new array - there is no need for a separate button node.
Only this inlet makes the node send.

min / max (drag floats, also inlets):
The range. Setting them does NOT make a new array by itself - the next
click uses them. In the demo each float goes through 't bang float', which
passes the number to the inlet first and THEN clicks the node, so dragging
it gives you a new array at once.

dim 0, dim 1, ... (properties):
The size of each dimension, used on the next click.

dtype (option):
float32 (the default), float or double: any value from min up to (but
never exactly) max.
int64: whole numbers from min to max, both included. Choosing it while max
is 1 changes max to 32767.
uint8: whole numbers 0 to 255. Choosing it raises a max below 255 to 255
and a negative min to 0.
bool: true or false at random; min and max are ignored.

OUTPUTS:

random array:
The new array, in the shape the dims give.

READING THE DEMO:
load_bang makes the first array when the page opens; click the button on
np.rand itself for another. The heat_map colours 0 dark purple and 1
yellow. Drag min and max towards each other and the noise flattens to a
single colour; push max past 1 and the cells saturate yellow. np.amin and
np.amax report the smallest and largest values in each new array - always
inside min and max.

RELATED:
t.rand is the PyTorch version. np.zeros and np.ones make arrays of one
fixed value, np.linspace makes evenly spaced values instead of random
ones. random gives single random numbers, with a choice of distributions."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'fills the heat_map once when the page opens',
     'pos': (30, 70)},
    {'key': 'fmin', 'init': 'float', 'pos': (250, 62), 'w': 110, 'h': 42,
     'props': dict(FLT, **{'': 0.0, 'width': 80})},
    {'key': 'fmax', 'init': 'float', 'pos': (380, 62), 'w': 110, 'h': 42,
     'props': dict(FLT, **{'': 1.0, 'width': 80})},
    {'key': 'tmin', 'init': 't bang float', 'pos': (250, 120), 'w': 110, 'h': 46},
    {'key': 'tmax', 'init': 't bang float', 'pos': (380, 120), 'w': 110, 'h': 46},
    {'key': 'c1', 'comment': True, 'text': 'drag min and max: new noise as you drag',
     'pos': (250, 125)},
    {'key': 'rnd', 'init': 'np.rand 12 12', 'pos': (30, 190), 'w': 160, 'h': 160,
     'props': {'min': 0.0, 'max': 1.0, 'dim 0': 12, 'dim 1': 12, 'dtype': 'float32'}},
    {'key': 'c0b', 'comment': True,
     'text': "click np.rand's own button (top left of the node)\nfor a fresh array",
     'pos': (30, 195)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (220, 190), 'w': 260, 'h': 280,
     'props': HM(240, 240, 12, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 'a 12 x 12 array: 0 purple, 1 yellow',
     'pos': (220, 200)},
    {'key': 'amin', 'init': 'np.amin', 'pos': (30, 490), 'w': 110, 'h': 70},
    {'key': 'amax', 'init': 'np.amax', 'pos': (220, 490), 'w': 110, 'h': 70},
    {'key': 'f1', 'init': 'float', 'pos': (30, 580), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'f2', 'init': 'float', 'pos': (220, 580), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'smallest and largest: inside min..max',
     'pos': (30, 585)},
] + see_also(660, ['t.rand', 'np.linspace', 'np.zeros', 'np.ones', 'random'])
links = [('lb', 'out', 'rnd', ''),
         ('fmin', 'float out', 'tmin', ''), ('fmax', 'float out', 'tmax', ''),
         ('tmin', 'float', 'rnd', 'min'), ('tmin', 'bang', 'rnd', ''),
         ('tmax', 'float', 'rnd', 'max'), ('tmax', 'bang', 'rnd', ''),
         ('rnd', 'random array', 'hm', 'y'),
         ('rnd', 'random array', 'amin', 'in'), ('rnd', 'random array', 'amax', 'in'),
         ('amin', 'amin', 'f1', ''), ('amax', 'amax', 'f2', '')]
print(build('np.rand', 'np.rand - an array of random numbers', body, demo, links,
            demo_width=520, text_width=760, text_height=800))

# --------------------------------------------------------------------- np.zeros
body = """Makes an array filled with zeros, in the shape you give it.

On its own it is the dullest array there is, and that is its use: a blank
canvas. Start from zeros and write into it with np.edit, add frames into
it to accumulate, or hold a correctly shaped place in a chain until real
data arrives.

SYNTAX:
np.zeros <dim 0: int> <dim 1: int> ...
One number per dimension: np.zeros 8 8 is an 8 by 8 array, np.zeros 16 a
row of 16. With no arguments you get a 1-element array. The NUMBER of
dimensions is fixed when the node is made - retype it to change that.

EXAMPLE:
np.zeros 8 8

INPUTS and PARAMETERS:

in (button):
Click it, or send it anything, to send the array. The node sends nothing
until it is clicked; each click sends a fresh copy, so whatever is done to
one copy downstream never reaches the next.

dim 0, dim 1, ... (properties):
The size of each dimension, used on the next click.

dtype (option):
float32 (the default), float, double, int64, uint8 or bool (all false).

OUTPUTS:

array of zeros:
The array.

READING THE DEMO:
The load_bang clicks np.zeros once when the page opens. The left heat_map
is the blank array - every cell 0, one flat colour. np.edit then writes 1
into rows 2 to 5 and columns 1 to 6 (indices 2:6, 1:7 - the end of a
range is not included), and the right heat_map shows the result. Type
other indices or another value (0 to 1) into np.edit to draw something
else on the same blank.

RELATED:
np.ones is the same with every value 1. t.zeros is the PyTorch version.
np.edit writes values into part of an array; np.rand fills one with noise
instead."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'zr', 'init': 'np.zeros 8 8', 'pos': (30, 125), 'w': 140, 'h': 120,
     'props': {'dim 0': 8, 'dim 1': 8, 'dtype': 'float32'}},
    {'key': 'hm0', 'init': 'heat_map', 'pos': (30, 265), 'w': 180, 'h': 200,
     'props': HM(160, 160, 8, fmt='none')},
    {'key': 'ed', 'init': 'np.edit 2:6, 1:7', 'pos': (240, 125), 'w': 200, 'h': 100,
     'props': {'indices': '2:6, 1:7', 'values': '1'}},
    {'key': 'hm1', 'init': 'heat_map', 'pos': (240, 265), 'w': 180, 'h': 200,
     'props': HM(160, 160, 8, fmt='none')},
    {'key': 'c0', 'comment': True, 'text': 'left: the blank. right: 1 written into part of it',
     'pos': (30, 280)},
    {'key': 'c1', 'comment': True, 'text': 'type new indices or a value into np.edit',
     'pos': (240, 130)},
] + see_also(490, ['np.ones', 't.zeros', 'np.edit', 'np.rand'])
links = [('lb', 'out', 'zr', ''), ('zr', 'array of zeros', 'hm0', 'y'),
         ('zr', 'array of zeros', 'ed', 'array in'), ('ed', 'output', 'hm1', 'y')]
print(build('np.zeros', 'np.zeros - an array of zeros, a blank to write into',
            body, demo, links, demo_width=470, text_width=760, text_height=640))

# ---------------------------------------------------------------------- np.ones
body = """Makes an array filled with ones, in the shape you give it.

Multiplying by one changes nothing, which makes ones the neutral start for
many things: times a number they become an array full of that number (a
constant of any shape), and as weights or a mask they let everything
through until you change some of them.

SYNTAX:
np.ones <dim 0: int> <dim 1: int> ...
One number per dimension: np.ones 8 is a row of 8, np.ones 4 4 a 4 by 4
array. With no arguments you get a 1-element array. The NUMBER of
dimensions is fixed when the node is made - retype it to change that.

EXAMPLE:
np.ones 8

INPUTS and PARAMETERS:

in (button):
Click it, or send it anything, to send the array. Nothing is sent until
then; each click sends a fresh copy.

dim 0, dim 1, ... (properties):
The size of each dimension, used on the next click.

dtype (option):
float32 (the default), float, double, int64, uint8 or bool (all true).

OUTPUTS:

array of ones:
The array.

READING THE DEMO:
The top list is the array itself. Below, '*' scales it by the float: drag
the float and 't bang float' sets the multiplier first, then clicks
np.ones, so you get a row of whatever number you drag to.

RELATED:
np.zeros is the same with every value 0. t.ones is the PyTorch version
(and t.full makes a constant array in one step). np.edit changes part of
an array, for a mask that is mostly ones."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'fv', 'init': 'float', 'pos': (140, 62), 'w': 110, 'h': 42,
     'props': dict(FLT, **{'': 0.5, 'width': 80})},
    {'key': 'c0', 'comment': True, 'text': 'drag the float', 'pos': (140, 70)},
    {'key': 'tv', 'init': 't bang float', 'pos': (140, 120), 'w': 110, 'h': 46},
    {'key': 'on', 'init': 'np.ones 8', 'pos': (30, 185), 'w': 140, 'h': 100,
     'props': {'dim 0': 8, 'dtype': 'float32'}},
    {'key': 'l1', 'init': 'list', 'pos': (30, 300), 'w': 300, 'h': 42, 'props': LIST},
    {'key': 'c1', 'comment': True, 'text': 'the array of ones', 'pos': (30, 305)},
    {'key': 'mul', 'init': '*', 'pos': (30, 360), 'w': 110, 'h': 70},
    {'key': 'l2', 'init': 'list', 'pos': (30, 445), 'w': 300, 'h': 42, 'props': LIST},
    {'key': 'c2', 'comment': True, 'text': 'ones times the float: a constant array',
     'pos': (30, 450)},
] + see_also(520, ['np.zeros', 't.ones', 't.full', 'np.edit'])
links = [('lb', 'out', 'fv', ''), ('fv', 'float out', 'tv', ''),
         ('tv', 'float', 'mul', 'operand'), ('tv', 'bang', 'on', ''),
         ('on', 'array of ones', 'l1', ''), ('on', 'array of ones', 'mul', 'in'),
         ('mul', 'result', 'l2', '')]
print(build('np.ones', 'np.ones - an array of ones', body, demo, links,
            demo_width=420, text_width=760, text_height=560))

# ---------------------------------------------------------------------- np.grid
body = """Makes a grid of points covering a rectangle: the x, y and z of every point.

Use it for anything laid out on a regular grid - positions to draw or
place things at, points to sample a field or an image at, or the x and y
inputs to a calculation over a plane.

THE OUTPUT IS A 3-D ARRAY:
shaped (divisions y, divisions x, 3) - rows going down, columns going
across, and three numbers in every cell: that point's x, its y and its z.
Slice the last axis to get one coordinate as a plane: np.[] :,:,0 gives
every point's x, np.[] :,:,1 every point's y.

x runs from min x to max x along each row; y runs from min y to max y
down the rows, so row 0 holds min y. Both ends are included, as in
np.linspace - the grid is two linspaces crossed. z is the same at every
point.

SYNTAX:
np.grid <min x> <max x> <min y> <max y> <divisions x> <divisions y>
Defaults: 0 1 0 1 10 10.

EXAMPLE:
np.grid 0 1 0 1 5 4

INPUTS and PARAMETERS:

in (button):
Click it, or send it anything, to send the grid. Changing the properties
does not send by itself - in the demo a metro clicks the node 10 times a
second, so the planes follow your drags.

min x / max x / min y / max y (drag floats):
The rectangle.

divisions x / divisions y (drag ints):
How many POINTS along each direction (not gaps). Anything below 2 is
treated as 2.

z (drag float):
The z of every point. Default 0; it has no argument.

OUTPUTS:

grid points:
float32, shaped (divisions y, divisions x, 3).

READING THE DEMO:
np.shape shows the layout: 4 rows, 5 columns, 3 numbers each. The left
heat_map is the x plane - the same across every row, rising to the right.
The right one is the y plane - the same along every row, rising row by
row. Each cell prints its value while there is room.

The heat_map draws row 0 at the TOP, so y grows DOWN the picture - the
opposite of a graph. Both heat_maps are scaled 0 to 1; drag max x or
max y past 1 and the far cells saturate.

RELATED:
np.linspace makes one of these axes on its own. np.[] slices out a plane.
np.reshape turns the grid into a plain list of points: np.reshape -1 3."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 100.0, 'units': 'milliseconds'}},
    {'key': 'gr', 'init': 'np.grid 0 1 0 1 5 4', 'pos': (30, 150), 'w': 200, 'h': 220,
     'props': {'min x': 0.0, 'max x': 1.0, 'min y': 0.0, 'max y': 1.0,
               'divisions x': 5, 'divisions y': 4, 'z': 0.0}},
    {'key': 'c0', 'comment': True, 'text': 'drag any of its values', 'pos': (30, 160)},
    {'key': 'shp', 'init': 'np.shape', 'pos': (290, 150), 'w': 110, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (290, 240), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c1', 'comment': True, 'text': '(rows, columns, x y z)', 'pos': (290, 245)},
    {'key': 'sx', 'init': 'np.[] :,:,0', 'pos': (30, 400), 'w': 150, 'h': 70},
    {'key': 'sy', 'init': 'np.[] :,:,1', 'pos': (310, 400), 'w': 150, 'h': 70},
    {'key': 'hx', 'init': 'heat_map', 'pos': (30, 490), 'w': 270, 'h': 210,
     'props': HM(250, 160, 4)},
    {'key': 'hy', 'init': 'heat_map', 'pos': (310, 490), 'w': 270, 'h': 210,
     'props': HM(250, 160, 4)},
    {'key': 'c2', 'comment': True, 'text': 'left: x of every point. right: y',
     'pos': (30, 500)},
] + see_also(730, ['np.linspace', 'np.[]', 'np.reshape', 'np.shape'])
links = [('met', '', 'gr', ''), ('gr', 'grid points', 'shp', 'np in'),
         ('shp', 'shape', 'l1', ''),
         ('gr', 'grid points', 'sx', 'tensor in'), ('gr', 'grid points', 'sy', 'tensor in'),
         ('sx', 'output', 'hx', 'y'), ('sy', 'output', 'hy', 'y')]
print(build('np.grid', 'np.grid - the coordinates of every point in a grid',
            body, demo, links, demo_width=600, text_width=760, text_height=760))

# ------------------------------------------------------------------ np.sequence
# np.sequence_help.json is laid out by hand (David rebuilt it around a
# make_roving_circle subpatcher) and is no longer generated here -- a rebuild
# would overwrite it.
