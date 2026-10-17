"""np rearranging: one page per node.

np.flip, np.roll, np.rot90 (np.rotate shares it - same class, same code),
np.repeat, np.sort, np.argsort, np.split, np.concatenate, np.stack. These were
one family page (np_rearrange); each now has its own page whose demo shows the
array before and after as heat_maps, with a SEE ALSO row under it.
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


METRO = {'on': True, 'period': 100.0, 'units': 'milliseconds'}

# A capital F, 7 rows by 6 columns: 1 is the letter, 0 the background. It has
# no symmetry at all, so every flip, roll and quarter turn gives a different
# picture - you can always tell which way it went.
F_GLYPH = ('1 1 1 1 1 1  '
           '1 1 0 0 0 0  '
           '1 1 0 0 0 0  '
           '1 1 1 1 0 0  '
           '1 1 0 0 0 0  '
           '1 1 0 0 0 0  '
           '1 1 0 0 0 0')


def f_source(y=62, x=30, key=''):
    """The F, made inside a make_F subpatcher so the page is about the node.

    Keyed 'rs' (as the np.reshape it replaces was) so the pages' links still
    read ('rs', 'F', ...)."""
    return [
        {'key': 'rs' + key, 'init': 'p make_F', 'pos': (x, y), 'w': 160, 'h': 70},
    ]


def f_links(key=''):
    return []


# Inside make_F: metro -> message (the F as 42 numbers) -> np.reshape 7 6.
F_SUB = {
    'name': 'make_F',
    'host': 'rs',
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


# ---------------------------------------------------------------------- np.flip
body = """Reverses an array along one axis: the first row becomes the last,
or the first column the last.

Use it to mirror an image or a grid (upside down, or left to right), to
reverse a sequence, or to turn an ascending order into a descending one.

WHICH AXIS:
axis 0 reverses the order of the ROWS - the picture turns upside down.
axis 1 reverses the order of the COLUMNS - each row is read backwards, a
mirror image. For a 1-D array there is only axis 0, so the whole row is
reversed. The node flips one axis at a time; to flip both, chain two
(np.flip 0 into np.flip 1), which is the same as np.rot90 2.

SYNTAX:
np.flip <axis: int>
Default axis 0.

EXAMPLE:
np.flip 1

INPUTS and PARAMETERS:

input:
The array (a list is turned into one). Every array that arrives is flipped
and sent - this is the only inlet that makes the node send.

axis (int box, also an inlet):
Which axis to reverse. Changing it does NOT send by itself; the next array
uses it. In the demo a metro inside make_F re-sends the F ten times a
second, so the picture follows your edits.
A negative axis is treated as 0 - NOT as counting from the end, as
in numpy - and an axis past the last one is treated as the last.

OUTPUTS:

flipped array:
Same shape and values as the input, in reversed order along the axis.

READING THE DEMO:
The F is a 7 by 6 array of 0s (purple) and 1s (yellow), drawn row 0 at
the top. With axis 0 the F stands on its head, its bar at the bottom;
type 1 into axis and it faces left instead.

RELATED:
np.rot90 turns by quarter turns instead of mirroring. np.roll shifts
along an axis with wrap-around. np.transpose swaps rows and columns
(a flip across the diagonal). np.sort has a descending option that is
a flip of its result. t.flip is the PyTorch version, on the t.roll page."""

demo = f_source() + [
    {'key': 'c0', 'comment': True, 'text': 'the F, 7 x 6 - open make_F to see how it is made',
     'pos': (30, 160)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 300), 'w': 200, 'h': 260,
     'props': HM(180, 210, 7, fmt='none')},
    {'key': 'fl', 'init': 'np.flip 0', 'pos': (280, 230), 'w': 150, 'h': 70,
     'props': {'axis': 0}},
    {'key': 'c1', 'comment': True, 'text': 'type 0 or 1 into axis', 'pos': (280, 235)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (280, 320), 'w': 200, 'h': 260,
     'props': HM(180, 210, 7, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 'left: in. right: flipped',
     'pos': (30, 590)},
] + see_also(640, ['np.rot90', 'np.roll', 'np.transpose', 'np.sort', 't.roll'])
links = f_links() + [('rs', 'F', 'hin', 'y'), ('rs', 'F', 'fl', 'input'),
                     ('fl', 'flipped array', 'hout', 'y')]
print(build('np.flip', 'np.flip - reverse the order along an axis',
            body, demo, links, demo_width=600, text_width=760, text_height=760,
            subpatch=F_SUB))

# ---------------------------------------------------------------------- np.roll
body = """Shifts everything in an array along an axis, and what falls off one
end comes back at the other.

Nothing is lost: it is a rotation of the order, not a shift into empty
space. Use it to scroll an image or a pattern round and round, to offset
one copy of a sequence against another, or to step through a circular
list. If you want old values to drop off and new ones to arrive, that is a
buffer, not a roll - see np.rolling_buffer.

SYNTAX:
np.roll <shifts: int> <axis: int>
Defaults: shifts 2, axis 0.

EXAMPLE:
np.roll 2 1

INPUTS and PARAMETERS:

input:
The array (a list is turned into one). Each array that arrives is rolled
and sent. Only this inlet makes the node send.

shifts (int box, also an inlet):
How many places to move. Positive moves towards the end (down for
axis 0, right for axis 1); negative the other way. A shift larger than
the axis just goes round again: on 6 columns, 7 is the same as 1.
Changing it does NOT send by itself - in the demo a metro inside make_F
re-sends the F ten times a second, so the picture follows your edits.
A negative shift typed into the box works, but a patch saved with one
reopens with 0 (a bug: the box's lower limit is applied on load). To
keep a backwards roll in a saved patch, use the forward equivalent: on 6
columns, 5 instead of -1.

axis (int box, also an inlet):
Which axis to roll along: 0 moves rows, 1 moves columns. For a 1-D
array only 0 means anything. A negative axis is treated as 0 - not as
counting from the end - and an axis past the last one as the last.

OUTPUTS:

rolled array:
Same shape and values as the input, moved round along the axis.

READING THE DEMO:
np.roll 2 1 moves every column 2 places to the right; the two columns
pushed off the right edge reappear on the left, so the F's stem now
sits in the middle. Step shifts up and the F scrolls right, wrapping;
set axis to 0 and it scrolls down instead, its top bar wrapping to the
bottom.

RELATED:
np.flip reverses instead of shifting. np.rot90 turns by quarter turns.
np.rolling_buffer keeps a moving window of recent values, dropping the
oldest. t.roll is the PyTorch version."""

demo = f_source() + [
    {'key': 'c0', 'comment': True, 'text': 'the F, 7 x 6 - open make_F to see how it is made',
     'pos': (30, 160)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 300), 'w': 200, 'h': 260,
     'props': HM(180, 210, 7, fmt='none')},
    {'key': 'ro', 'init': 'np.roll 2 1', 'pos': (280, 200), 'w': 150, 'h': 100,
     'props': {'shifts': 2, 'axis': 1}},
    {'key': 'c1', 'comment': True, 'text': 'step shifts; axis 0 rows, 1 columns',
     'pos': (280, 205)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (280, 320), 'w': 200, 'h': 260,
     'props': HM(180, 210, 7, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 'left: in. right: rolled, wrapping round',
     'pos': (30, 590)},
] + see_also(640, ['np.flip', 'np.rot90', 'np.rolling_buffer', 't.roll'])
links = f_links() + [('rs', 'F', 'hin', 'y'), ('rs', 'F', 'ro', 'input'),
                     ('ro', 'rolled array', 'hout', 'y')]
print(build('np.roll', 'np.roll - shift along an axis, wrapping round',
            body, demo, links, demo_width=600, text_width=760, text_height=780,
            subpatch=F_SUB))

# --------------------------------------------------------------------- np.rot90
body = """Turns an array by quarter turns, like rotating a picture.

np.rot90 and np.rotate are two names for the same node.

Use it to turn an image or a grid on its side, or to change which way a
pattern faces. One quarter turn swaps the number of rows and columns: a 7
by 6 array becomes 6 by 7.

WHICH WAY:
With the default axes, a positive k turns ANTICLOCKWISE as a heat_map
draws the array (row 0 at the top): the top row becomes the left column.
A negative k turns clockwise. k 2 is a half turn (the same as flipping
both axes), k 4 is back where you started, and k 5 is the same as k 1.

SYNTAX:
np.rot90 <k: int> <axis 1: int> <axis 2: int>
np.rotate <k: int> <axis 1: int> <axis 2: int>
Defaults: k 1, axis 1 0, axis 2 1.

EXAMPLE:
np.rot90 1

INPUTS and PARAMETERS:

input:
The array (a list is turned into one). Each array that arrives is turned
and sent; only this inlet makes the node send. A 1-D array is treated as
a single row, so one quarter turn makes it a column (N rows of 1).

k (int box, also an inlet):
How many quarter turns. Changing it does NOT send by itself - in the
demo a metro inside make_F re-sends the F ten times a second.

axis 1 / axis 2 (int boxes, also inlets):
The plane to turn in: the turn goes FROM axis 1 TOWARDS axis 2. Swapping
them reverses the direction. For a 2-D array leave them at 0 and 1; they
matter for 3-D arrays, where they choose which pair of axes turns. A
negative axis is treated as 0, one past the last as the last; if the two
end up the same, nothing is sent.

OUTPUTS:

rotated array:
The turned array. With an odd k its first two dimensions have swapped.

READING THE DEMO:
The F (7 by 6) turned once anticlockwise lies on its back, 6 by 7, its
top bar now down the left side. np.shape shows the shapes swap. Step k
and it keeps turning; set k to 2 and it stands on its head; type 1 into
axis 1 and 0 into axis 2 and the same k turns it clockwise.

RELATED:
np.flip mirrors along one axis. np.transpose swaps rows and columns,
which is a flip across the diagonal rather than a turn. np.roll shifts
with wrap-around. np.reshape changes the shape without moving anything."""

demo = f_source() + [
    {'key': 'c0', 'comment': True, 'text': 'the F, 7 x 6 - open make_F to see how it is made',
     'pos': (30, 160)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 300), 'w': 230, 'h': 260,
     'props': HM(210, 210, 7, fmt='none')},
    {'key': 'rot', 'init': 'np.rot90 1', 'pos': (340, 170), 'w': 150, 'h': 130,
     'props': {'k': 1, 'axis 1': 0, 'axis 2': 1}},
    {'key': 'c1', 'comment': True, 'text': 'step k: one quarter turn each',
     'pos': (340, 175)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (340, 320), 'w': 230, 'h': 260,
     'props': HM(210, 210, 6, fmt='none')},
    {'key': 'shi', 'init': 'np.shape', 'pos': (30, 600), 'w': 110, 'h': 70},
    {'key': 'sho', 'init': 'np.shape', 'pos': (340, 600), 'w': 110, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (30, 690), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'l2', 'init': 'list', 'pos': (340, 690), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c2', 'comment': True, 'text': 'the shapes: rows and columns swap',
     'pos': (30, 695)},
] + see_also(770, ['np.flip', 'np.transpose', 'np.roll', 'np.reshape'])
links = f_links() + [('rs', 'F', 'hin', 'y'), ('rs', 'F', 'rot', 'input'),
                     ('rot', 'rotated array', 'hout', 'y'),
                     ('rs', 'F', 'shi', 'np in'), ('rot', 'rotated array', 'sho', 'np in'),
                     ('shi', 'shape', 'l1', ''), ('sho', 'shape', 'l2', '')]
print(build('np.rot90', 'np.rot90 / np.rotate - turn an array by quarter turns',
            body, demo, links, demo_width=600, text_width=760, text_height=860,
            subpatch=F_SUB))

# -------------------------------------------------------------------- np.repeat
body = """Repeats each element of an array a number of times, in place.

Each value is copied next to itself - 0, 1, 2 repeated twice is
0, 0, 1, 1, 2, 2 - not the whole array tiled end to end (that would be
0, 1, 2, 0, 1, 2; join copies with np.concatenate for that). Use it to
stretch an array by a whole-number factor, to blow a small grid up into
blocks (pixel doubling), or to hold each value for several steps.

SYNTAX:
np.repeat <repeats: int> <axis: int>
Defaults: repeats 2, axis 0.

EXAMPLE:
np.repeat 3

INPUTS and PARAMETERS:

input:
The array (a list is turned into one). Each array that arrives is
repeated and sent; only this inlet makes the node send.

repeats (int box, also an inlet):
How many copies of each element. 1 passes the array through; 0 gives an
empty array; a negative number is an error and nothing is sent.
Changing it does NOT send by itself - in the demo a metro re-sends the
row ten times a second.

axis (int box, also an inlet):
Which axis to repeat along. On a 2-D array, 0 repeats each row and 1
each column. Set it to the number of dimensions the array has (1 for a
row) and the node first ADDS a new axis there, so each value is copied
into a new dimension: a row of 4 becomes 4 rows of copies (4 by
repeats). A negative axis is treated as 0; anything past that is
treated as the last axis.

OUTPUTS:

array out:
The repeated array: the chosen axis is repeats times longer.

READING THE DEMO:
A row of 4 values, 0 to 1, repeated 3 times along axis 0 gives 12
cells: each colour three times in a row. Raise repeats and the blocks
widen (the numbers hide once cells get too narrow to hold them). Set
axis to 1 and the same 4 values become a 4 by 3 grid - each row one
value, copied across. np.shape shows the shape either way.

RELATED:
np.concatenate joins whole arrays end to end - feed it the same array
twice to tile it. np.stack piles arrays up along a new axis. np.reshape
refolds an array without copying anything. t.repeat (on the t.roll page)
is NOT the PyTorch version of this: it tiles the whole tensor."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70, 'props': METRO},
    {'key': 'ls', 'init': 'np.linspace 0 1 4', 'pos': (30, 150), 'w': 190, 'h': 150,
     'props': {'start': 0.0, 'stop': 1.0, 'steps': 4, 'dtype': 'float32'}},
    {'key': 'c0', 'comment': True, 'text': 'four values: 0, 0.33, 0.67, 1',
     'pos': (30, 155)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 320), 'w': 540, 'h': 110,
     'props': HM(520, 60, 1)},
    {'key': 'rp', 'init': 'np.repeat 3', 'pos': (30, 450), 'w': 150, 'h': 100,
     'props': {'repeats': 3, 'axis': 0}},
    {'key': 'c1', 'comment': True, 'text': 'step repeats; axis 1 adds a new axis',
     'pos': (30, 455)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 570), 'w': 540, 'h': 130,
     'props': HM(520, 80, 1)},
    {'key': 'c2', 'comment': True, 'text': 'each value copied in place',
     'pos': (30, 575)},
    {'key': 'sh', 'init': 'np.shape', 'pos': (30, 720), 'w': 110, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (160, 720), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c3', 'comment': True, 'text': 'shape of the result', 'pos': (160, 725)},
] + see_also(820, ['np.concatenate', 'np.stack', 'np.reshape', 't.roll'])
links = [('met', '', 'ls', ''), ('ls', 'linspace out', 'hin', 'y'),
         ('ls', 'linspace out', 'rp', 'input'), ('rp', 'array out', 'hout', 'y'),
         ('rp', 'array out', 'sh', 'np in'), ('sh', 'shape', 'l1', '')]
print(build('np.repeat', 'np.repeat - repeat each element in place',
            body, demo, links, demo_width=600, text_width=760, text_height=860))

# ---------------------------------------------------------------------- np.sort
body = """Puts the values of an array in order, smallest first.

Use it to rank values, to find the middle or the extremes by position, to
tidy a list before display, or to turn scattered values into a smooth
ramp. A 2-D array is sorted one row (or one column) at a time - each
line on its own, not the array as a whole.

SYNTAX:
np.sort <axis: int>
Default axis -1, the last axis: each row is sorted.

EXAMPLE:
np.sort 0

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every array that arrives is sorted
and sent; only this inlet makes the node send.

axis (int box, also an inlet):
Which way to sort. On a 2-D array, -1 or 1 sorts along each ROW (each
row runs dark to bright, left to right); 0 or -2 sorts down each COLUMN.
Negative numbers count from the end, as in numpy. An axis the array does
not have sends nothing. There is no setting that sorts the whole array
as one list - flatten it first (np.ravel), then sort.
Changing axis does NOT send by itself; in the demo a metro re-sends the
same array ten times a second (through var), so a new axis shows at once.

descending (checkbox, under show options):
Largest first. The sorted result is simply reversed along the axis.

OUTPUTS:

sorted:
The same values, in order, in the same shape as the input.

READING THE DEMO:
np.rand makes a 10 by 10 array of noise; click its button for fresh
numbers. var holds the latest array and the metro re-sends it, so both
heat_maps show the same numbers. With axis -1 every row of the right
picture runs purple to yellow; type 0 and every column does instead.
Turn on descending (show options) and the ramps run the other way.

RELATED:
np.argsort gives WHERE each sorted value came from instead of the values.
np.flip reverses an order. t.argsort is the PyTorch counterpart of
np.argsort, with its own descending switch."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 'np.rand 10 10', 'pos': (30, 130), 'w': 160, 'h': 160,
     'props': {'min': 0.0, 'max': 1.0, 'dim 0': 10, 'dim 1': 10, 'dtype': 'float32'}},
    {'key': 'c0', 'comment': True, 'text': 'click for fresh noise', 'pos': (30, 135)},
    {'key': 'met', 'init': 'metro 100', 'pos': (280, 62), 'w': 129, 'h': 70,
     'props': METRO},
    {'key': 'var', 'init': 'var np_sort_help_noise', 'pos': (280, 220), 'w': 230, 'h': 70,
     'props': {'name': 'np_sort_help_noise'}},
    {'key': 'c1', 'comment': True, 'text': 'holds the array; the metro re-sends it',
     'pos': (280, 225)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 320), 'w': 240, 'h': 270,
     'props': HM(220, 220, 10, fmt='none')},
    {'key': 'so', 'init': 'np.sort', 'pos': (300, 320), 'w': 150, 'h': 70,
     'props': {'axis': -1, 'descending': False}},
    {'key': 'c2', 'comment': True, 'text': 'axis -1: each row. 0: each column',
     'pos': (300, 325)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (300, 410), 'w': 240, 'h': 270,
     'props': HM(220, 220, 10, fmt='none')},
    {'key': 'c3', 'comment': True, 'text': 'left: noise. right: sorted',
     'pos': (30, 700)},
] + see_also(750, ['np.argsort', 'np.flip', 't.argsort'])
links = [('lb', 'out', 'rnd', ''), ('rnd', 'random array', 'var', 'in'),
         ('met', '', 'var', 'in'),
         ('var', 'out', 'hin', 'y'), ('var', 'out', 'so', 'in'),
         ('so', 'sorted', 'hout', 'y')]
print(build('np.sort', 'np.sort - put values in order',
            body, demo, links, demo_width=600, text_width=760, text_height=820))

# ------------------------------------------------------------------- np.argsort
body = """Gives the positions that would put an array in order: the index of
the smallest value first, then the next smallest, and so on.

It does not move any values - it tells you where they are. Use it when
you want to reorder something ELSE by the same ranking: sort joints by
speed, then use the same order for their names; find which three
sensors read highest; rank a set of scores and keep track of whose
they were.

READING THE INDICES:
Positions count from 0. If the first index is 5, the smallest value is
in cell 5 of the input. Reading the input at each index in turn gives the
sorted values, exactly what np.sort sends. When two values are equal
their order is not guaranteed.

SYNTAX:
np.argsort <axis: int>
Default axis -1, the last axis: each row is ranked on its own.

EXAMPLE:
np.argsort

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every array that arrives is ranked
and sent; only this inlet makes the node send.

axis (int box, also an inlet):
For a 2-D array, -1 or 1 ranks each row separately (indices are column
numbers), 0 or -2 ranks each column (indices are row numbers). Negative
counts from the end. An axis the array does not have sends nothing.

descending (checkbox, under show options):
Largest first: the index list is reversed.

OUTPUTS:

indices:
Whole numbers (int64), in the same shape as the input.

READING THE DEMO:
np.rand makes 8 values; click its button for new ones. The top strip
shows them with their values. The middle strip and the list are the
indices from np.argsort: the first names the cell of the top strip
holding the smallest value, the next the cell holding the next smallest,
and so on along the row.

The bottom strip USES those indices: np.[] reads the values at the
positions it is given, in the order given, so reading them in argsort's
order puts the values in order - the same result np.sort gives, made
from the ranking. The 't array array' fires right to left, so the
indices reach np.[] before the values do; the values arriving is what
makes np.[] send. An integer array sent to np.[]'s Indices inlet picks
those positions along the first axis.

To reorder something ELSE by the same ranking, send the other array
into np.[] instead - the indices still come from the first.

RELATED:
np.sort gives the sorted values directly. np.[] picks positions. t.argsort is the PyTorch
version. np.flip reverses an index list (smallest-first to largest-first)."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 'np.rand 8', 'pos': (30, 130), 'w': 160, 'h': 140,
     'props': {'min': 0.0, 'max': 1.0, 'dim 0': 8, 'dtype': 'float32'}},
    {'key': 'c0', 'comment': True, 'text': 'click for 8 new values', 'pos': (30, 135)},
    {'key': 'tr', 'init': 't array array', 'pos': (220, 200), 'w': 140, 'h': 46},
    {'key': 'ct', 'comment': True, 'text': 'right first: indices, then values',
     'pos': (370, 205)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 290), 'w': 540, 'h': 110,
     'props': HM(520, 60, 1)},
    {'key': 'c1', 'comment': True, 'text': 'the values, cells 0 to 7', 'pos': (30, 295)},
    {'key': 'ar', 'init': 'np.argsort', 'pos': (30, 420), 'w': 150, 'h': 70,
     'props': {'axis': -1, 'descending': False}},
    {'key': 'hidx', 'init': 'heat_map', 'pos': (30, 510), 'w': 540, 'h': 110,
     'props': HM(520, 60, 1, miny=0.0, maxy=7.0, fmt='%.0f', color='greys')},
    {'key': 'c2', 'comment': True, 'text': 'indices that would order the input smallest to largest',
     'pos': (30, 515)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 640), 'w': 300, 'h': 42, 'props': LIST},
    {'key': 'so', 'init': 'np.[]', 'pos': (30, 700), 'w': 150, 'h': 70},
    {'key': 'hso', 'init': 'heat_map', 'pos': (30, 790), 'w': 540, 'h': 110,
     'props': HM(520, 60, 1)},
    {'key': 'c3', 'comment': True, 'text': 'the values read at those indices: sorted',
     'pos': (30, 795)},
] + see_also(930, ['np.sort', 'np.[]', 't.argsort', 'np.flip'])
links = [('lb', 'out', 'rnd', ''), ('rnd', 'random array', 'tr', ''),
         ('tr', 'array', 'hin', 'y', 0), ('tr', 'array', 'so', 'tensor in', 0),
         ('tr', 'array', 'ar', 'in', 1), ('ar', 'indices', 'hidx', 'y'),
         ('ar', 'indices', 'l1', ''), ('ar', 'indices', 'so', 'Indices'),
         ('so', 'output', 'hso', 'y')]
print(build('np.argsort', 'np.argsort - the positions that would sort an array',
            body, demo, links, demo_width=600, text_width=760, text_height=860))

# --------------------------------------------------------------------- np.split
body = """Cuts an array into a number of pieces along one axis, and sends
each piece from its own outlet.

Use it to separate an array that holds several things side by side -
the left and right halves of an image, the x, y and z columns of a list
of points, the first and second half of a recording.

THE NUMBER OF PIECES IS FIXED:
It is the first argument, and it decides how many outlets the node has
(array 0, array 1, ...), so it cannot change on a live node - retype the
node to change it. The smallest is 2. If the axis does not divide evenly,
the first pieces get one extra: 7 rows in 3 pieces are 3, 2 and 2.

SYNTAX:
np.split <pieces: int> <axis: int>
Defaults: 2 pieces, axis 0.

EXAMPLE:
np.split 3

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every array that arrives is cut
and the pieces sent, array 0 first. Only this inlet makes the node send.

axis (int box, also an inlet):
Which axis to cut across: 0 cuts the rows into groups (horizontal cuts),
1 cuts the columns into groups (vertical cuts). Negative counts from the
end. An axis the array does not have - or one shorter than the number of
pieces - sends nothing. Changing it does NOT send by itself; in the demo
a metro inside make_F re-sends the F ten times a second.

OUTPUTS:

array 0, array 1, ...:
The pieces, in order. Each has the same number of dimensions as the
input; only the length along the axis is smaller.

READING THE DEMO:
np.split 3 cuts the 7 by 6 F into three bands of rows: the top 3 rows
(the bar), then 2 and 2. The lists show each piece's shape. Type 1 into
axis and it cuts the columns instead, three slices 2 columns wide: the
stem, then the two halves of the arms.

RELATED:
np.concatenate puts pieces back together. np.stack joins along a new
axis. np.reshape refolds without cutting. t.chunk and t.tensor_split, on
the t.cat page, are the PyTorch versions."""

demo = f_source() + [
    {'key': 'c0', 'comment': True, 'text': 'the F, 7 x 6 - open make_F to see how it is made',
     'pos': (30, 160)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (340, 62), 'w': 200, 'h': 260,
     'props': HM(180, 210, 7, fmt='none')},
    {'key': 'sp', 'init': 'np.split 3', 'pos': (30, 340), 'w': 150, 'h': 120,
     'props': {'axis': 0}},
    {'key': 'c1', 'comment': True, 'text': 'type 0 or 1 into axis', 'pos': (30, 345)},
    {'key': 'h0', 'init': 'heat_map', 'pos': (30, 480), 'w': 170, 'h': 190,
     'props': HM(150, 140, 3, fmt='%.0f')},
    {'key': 'h1', 'init': 'heat_map', 'pos': (215, 480), 'w': 170, 'h': 190,
     'props': HM(150, 140, 2, fmt='%.0f')},
    {'key': 'h2', 'init': 'heat_map', 'pos': (400, 480), 'w': 170, 'h': 190,
     'props': HM(150, 140, 2, fmt='%.0f')},
    {'key': 's0', 'init': 'np.shape', 'pos': (30, 690), 'w': 110, 'h': 70},
    {'key': 's1', 'init': 'np.shape', 'pos': (215, 690), 'w': 110, 'h': 70},
    {'key': 's2', 'init': 'np.shape', 'pos': (400, 690), 'w': 110, 'h': 70},
    {'key': 'l0', 'init': 'list', 'pos': (30, 780), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'l1', 'init': 'list', 'pos': (215, 780), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'l2', 'init': 'list', 'pos': (400, 780), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c2', 'comment': True, 'text': 'array 0, array 1, array 2 and their shapes',
     'pos': (30, 840)},
] + see_also(890, ['np.concatenate', 'np.stack', 'np.reshape', 't.cat'])
links = f_links() + [('rs', 'F', 'hin', 'y'), ('rs', 'F', 'sp', 'in'),
                     ('sp', 'array 0', 'h0', 'y'), ('sp', 'array 1', 'h1', 'y'),
                     ('sp', 'array 2', 'h2', 'y'),
                     ('sp', 'array 0', 's0', 'np in'), ('sp', 'array 1', 's1', 'np in'),
                     ('sp', 'array 2', 's2', 'np in'),
                     ('s0', 'shape', 'l0', ''), ('s1', 'shape', 'l1', ''),
                     ('s2', 'shape', 'l2', '')]
print(build('np.split', 'np.split - cut an array into pieces',
            body, demo, links, demo_width=600, text_width=760, text_height=860,
            subpatch=F_SUB))

# --------------------------------------------------------------- np.concatenate
# Inside make_ramp: load_bang -> np.linspace 0 1 42 -> np.reshape 7 6.
RAMP_SUB = {
    'name': 'make_ramp',
    'host': 'rs2',
    'demo': [
        {'key': 'lb', 'init': 'load_bang', 'pos': (40, 40), 'w': 88, 'h': 46},
        {'key': 'cl', 'init': 'comment once, when the patch opens',
         'pos': (150, 50), 'w': 280, 'h': 30},
        {'key': 'ls', 'init': 'np.linspace 0 1 42', 'pos': (40, 110), 'w': 190, 'h': 150,
         'props': {'start': 0.0, 'stop': 1.0, 'steps': 42, 'dtype': 'float32'}},
        {'key': 'cs', 'init': 'comment 42 values from 0 to 1',
         'pos': (250, 125), 'w': 240, 'h': 30},
        {'key': 'rs', 'init': 'np.reshape 7 6', 'pos': (40, 280), 'w': 230, 'h': 70},
        {'key': 'cr', 'init': 'comment folded into 7 rows of 6, the same shape as the F',
         'pos': (290, 295), 'w': 440, 'h': 30},
        {'key': 'out', 'init': 'out ramp', 'pos': (40, 380), 'w': 150, 'h': 80},
    ],
    'links': [('lb', 'out', 'ls', ''),
              ('ls', 'linspace out', 'rs', 'tensor in'),
              ('rs', 'output', 'out', 'ramp')],
}


body = """Joins two arrays end to end along an axis they already have.

Use it to put two images side by side or one above the other, to append
new rows to a table, or to join two lists into one longer list. The
result has the same number of dimensions as the inputs - it is just
longer along the axis. (To keep the two arrays apart as layers of a new
axis, use np.stack.)

THE SHAPES MUST FIT:
The two arrays must match in every dimension except the one being joined
along: two 7 by 6 arrays join along axis 0 into 14 by 6, or along axis 1
into 7 by 12. If they do not fit, numpy refuses, an error is printed and
nothing is sent.

SYNTAX:
np.concatenate <axis: int>
Default axis 0.

EXAMPLE:
np.concatenate 1

INPUTS and PARAMETERS:

in:
The first array - it goes first (on top, or on the left). Receiving it
makes the node join and send.

in 2:
The second array. It is only stored, so it must arrive BEFORE in. Until
something has reached in 2, nothing is sent.

axis (menu, also an inlet):
0 joins below (more rows), 1 joins to the right (more columns). The menu
offers as many axes as the array at in has. None flattens both arrays to
rows and joins them into one long 1-D array.
Changing it does NOT send by itself; in the demo a metro inside make_F
re-sends the F ten times a second.

OUTPUTS:

concatenate:
The joined array.

READING THE DEMO:
in is the F (7 by 6, purple and yellow); in 2 is a ramp of the same
shape, made once when the patch opens inside make_ramp. With axis 0 the ramp hangs below
the F (14 by 6); choose 1 and it sits to its right (7 by 12); choose None
and both become one strip of 84 values.

RELATED:
np.stack joins along a NEW axis instead. np.split cuts an array into
pieces - the reverse. np.repeat copies each element in place. t.cat is
the PyTorch version."""

demo = f_source() + [
    {'key': 'c0', 'comment': True, 'text': 'in: the F, 7 x 6', 'pos': (30, 160)},
    {'key': 'rs2', 'init': 'p make_ramp', 'pos': (330, 62), 'w': 180, 'h': 70},
    {'key': 'c1', 'comment': True, 'text': 'in 2: a ramp, 7 x 6, made once',
     'pos': (330, 160)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 390), 'w': 200, 'h': 260,
     'props': HM(180, 210, 7, fmt='none')},
    {'key': 'hin2', 'init': 'heat_map', 'pos': (330, 390), 'w': 200, 'h': 260,
     'props': HM(180, 210, 7, fmt='none')},
    {'key': 'cc', 'init': 'np.concatenate 0', 'pos': (30, 670), 'w': 180, 'h': 100,
     'props': {'axis': '0'}},
    {'key': 'c2', 'comment': True, 'text': 'choose axis 0, 1 or None', 'pos': (30, 675)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 790), 'w': 330, 'h': 350,
     'props': HM(310, 300, 14, fmt='%.1f')},
    {'key': 'sh', 'init': 'np.shape', 'pos': (400, 790), 'w': 110, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (400, 880), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c3', 'comment': True, 'text': 'the joined array and its shape',
     'pos': (400, 940)},
] + see_also(1170, ['np.stack', 'np.split', 'np.repeat', 't.cat'])
links = f_links() + [('rs2', 'ramp', 'hin2', 'y'), ('rs2', 'ramp', 'cc', 'in 2'),
                     ('rs', 'F', 'hin', 'y'), ('rs', 'F', 'cc', 'in'),
                     ('cc', 'concatenate', 'hout', 'y'), ('cc', 'concatenate', 'sh', 'np in'),
                     ('sh', 'shape', 'l1', '')]
print(build('np.concatenate', 'np.concatenate - join arrays end to end',
            body, demo, links, demo_width=600, text_width=760, text_height=900,
            subpatch=[F_SUB, RAMP_SUB]))

# --------------------------------------------------------------------- np.stack
body = """Piles two arrays of the same shape on top of each other, along a NEW
axis.

The result has one more dimension than the inputs, and the two arrays
stay separate inside it: stacking two rows of 8 gives a 2 by 8 array -
row 0 is the first array, row 1 the second. Use it to turn two signals
into one two-row array, to pair up x and y lists, or to build a batch out
of single frames.

STACK OR CONCATENATE?
Joining two rows of 8 with np.concatenate gives ONE row of 16 - the
second carries straight on from the first. With np.stack it gives 2 rows
of 8, the originals kept side by side. Concatenate extends; stack layers.
The demo does both to the same two rows.

SYNTAX:
np.stack <axis: int>
Default axis 0.

EXAMPLE:
np.stack

INPUTS and PARAMETERS:

in:
The first array - row 0 of the result. Receiving it makes the node
stack and send.

in 2:
The second array. It is only stored, so it must arrive BEFORE in. It
must have exactly the same shape as the first, or numpy refuses, an error
is printed and nothing is sent.

axis (menu, also an inlet):
Where the new axis goes. 0 (the default) puts it first: the arrays become
the rows (for 1-D inputs) or the layers (for 2-D inputs). The menu only
lists the axes the inputs ALREADY have, so the new axis cannot be put
last - for two 1-D rows only 0 is offered, which is the useful one. Do
not choose None: np.stack has no flattened form, and the node prints an
error and sends nothing.

OUTPUTS:

stack:
The stacked array, with one more dimension than the inputs.

READING THE DEMO:
in is a ramp rising across 8 cells (drag stop on its np.linspace); in 2
is the same ramp falling, made once when the patch opens. np.stack gives
a 2 by 8 picture - rising on top, falling below. np.concatenate of the
same two gives a single strip of 16. Keep steps at 8: the two must have
the same shape.

RELATED:
np.concatenate joins along an existing axis. np.split cuts an array
apart. np.reshape can add an axis of length 1 to a single array. t.stack,
on the t.cat page, is the PyTorch version."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70, 'props': METRO},
    {'key': 'lsa', 'init': 'np.linspace 0 1 8', 'pos': (30, 150), 'w': 190, 'h': 150,
     'props': {'start': 0.0, 'stop': 1.0, 'steps': 8, 'dtype': 'float32'}},
    {'key': 'c0', 'comment': True, 'text': 'in: rising. drag stop', 'pos': (30, 155)},
    {'key': 'lb', 'init': 'load_bang', 'pos': (300, 62), 'w': 88, 'h': 46},
    {'key': 'lsb', 'init': 'np.linspace 1 0 8', 'pos': (300, 150), 'w': 190, 'h': 150,
     'props': {'start': 1.0, 'stop': 0.0, 'steps': 8, 'dtype': 'float32'}},
    {'key': 'c1', 'comment': True, 'text': 'in 2: falling, made once', 'pos': (300, 155)},
    {'key': 'st', 'init': 'np.stack', 'pos': (30, 330), 'w': 150, 'h': 100,
     'props': {'axis': '0'}},
    {'key': 'hst', 'init': 'heat_map', 'pos': (30, 450), 'w': 540, 'h': 170,
     'props': HM(520, 120, 2)},
    {'key': 'c2', 'comment': True, 'text': 'stack: 2 rows of 8, kept apart',
     'pos': (30, 455)},
    {'key': 'shs', 'init': 'np.shape', 'pos': (30, 640), 'w': 110, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (160, 640), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'cc', 'init': 'np.concatenate', 'pos': (30, 740), 'w': 180, 'h': 100,
     'props': {'axis': '0'}},
    {'key': 'c3', 'comment': True, 'text': 'for contrast: the same two, concatenated',
     'pos': (30, 745)},
    {'key': 'hcc', 'init': 'heat_map', 'pos': (30, 860), 'w': 540, 'h': 110,
     'props': HM(520, 60, 1, fmt='none')},
    {'key': 'shc', 'init': 'np.shape', 'pos': (30, 990), 'w': 110, 'h': 70},
    {'key': 'l2', 'init': 'list', 'pos': (160, 990), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c4', 'comment': True, 'text': 'concatenate: one row of 16', 'pos': (160, 995)},
] + see_also(1090, ['np.concatenate', 'np.split', 'np.reshape', 't.cat'])
links = [('lb', 'out', 'lsb', ''),
         ('lsb', 'linspace out', 'st', 'in 2'), ('lsb', 'linspace out', 'cc', 'in 2'),
         ('met', '', 'lsa', ''),
         ('lsa', 'linspace out', 'st', 'in'), ('lsa', 'linspace out', 'cc', 'in'),
         ('st', 'stack', 'hst', 'y'), ('st', 'stack', 'shs', 'np in'),
         ('shs', 'shape', 'l1', ''),
         ('cc', 'concatenate', 'hcc', 'y'), ('cc', 'concatenate', 'shc', 'np in'),
         ('shc', 'shape', 'l2', '')]
print(build('np.stack', 'np.stack - join arrays along a new axis',
            body, demo, links, demo_width=600, text_width=760, text_height=900))
