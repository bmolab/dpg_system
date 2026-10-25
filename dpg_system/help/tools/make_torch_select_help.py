"""t selecting, writing into, and rearranging: one page per node.

selecting:     t[] t.select t.index_select t.masked_select t.take
               t.take_along_dim t.narrow
writing into:  t.scatter t.scatter_hold
rearranging:   t.diag t.tril t.triu t.roll t.flip t.repeat t.tile

These were four family pages (t.index_select, t.scatter, t.diag, t.roll in
make_torch_manip_b_help.py). Each node now has its own page whose demo shows
the tensor before and after as heat_maps, with a SEE ALSO row under it.
The demo data (a numbered 10 x 10 tensor, the letter F, a disc) is made in
subpatchers so each page shows the node itself. The t.take_along_dim page is
hand-made now (see tools/README.md), so this script no longer builds it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build


def HM(width, height, count, miny=0.0, maxy=1.0, fmt='%.2f', color='viridis'):
    """Property block for a heat_map that draws a whole tensor as one picture."""
    return {'color': color, 'width': width, 'height': height, 'sample count': count,
            'min y': miny, 'max y': maxy, 'update_mode': 'heat_map',
            'number format': fmt}


LIST = {'text in': '', 'font size': '24'}
INT = {'format': '%d', 'width': 100, 'font size': '24'}
METRO = {'on': True, 'period': 100.0, 'units': 'milliseconds'}


def MSG(text, width=None):
    p = {'text in': text, 'font size': '24'}
    if width:
        p['width'] = width
    return p


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


# ------------------------------------------------------------ demo sources
# The numbered 10 x 10 tensor: t.arange 0 100 folded into 10 rows, so every
# cell's number IS its position - 37 is row 3, column 7, and also flat
# position 37. Whatever comes out of a selector names the cells it took.
NUM_SUB = {
    'name': 'make_numbered',
    'host': 'src',
    'demo': [
        {'key': 'met', 'init': 'metro 100', 'pos': (40, 40), 'w': 129, 'h': 70,
         'props': METRO},
        {'key': 'cm', 'init': 'comment re-sends the tensor ten times a second',
         'pos': (190, 50), 'w': 330, 'h': 30},
        {'key': 'ar', 'init': 't.arange 0 100 1', 'pos': (40, 130), 'w': 190, 'h': 150,
         'props': {'start': 0.0, 'stop': 100.0, 'step': 1.0}},
        {'key': 'ca', 'init': 'comment 0, 1, 2 ... 99', 'pos': (250, 140), 'w': 200, 'h': 30},
        {'key': 'rs', 'init': 't.reshape 10 10', 'pos': (40, 300), 'w': 230, 'h': 70},
        {'key': 'cr', 'init': 'comment folded into 10 rows of 10',
         'pos': (290, 315), 'w': 280, 'h': 30},
        {'key': 'out', 'init': 'out numbered', 'pos': (40, 400), 'w': 150, 'h': 80},
    ],
    'links': [('met', '', 'ar', '###input'),
              ('ar', 'arange out', 'rs', 'tensor in'),
              ('rs', 'output', 'out', 'numbered')],
}


def numbered(x=30, y=62):
    return [
        {'key': 'src', 'init': 'p make_numbered', 'pos': (x, y), 'w': 190, 'h': 70},
        {'key': 'csrc0', 'comment': True,
         'text': 'open make_numbered to see how it is made', 'pos': (x, y + 5)},
        {'key': 'hsrc', 'init': 'heat_map', 'pos': (x, y + 100), 'w': 428, 'h': 310,
         'props': HM(400, 260, 10, 0.0, 99.0, fmt='%.0f')},
        {'key': 'csrc', 'comment': True,
         'text': 'the source: 10 x 10, each cell numbered\nrow then column - 37 is row 3, column 7',
         'pos': (x, y + 110)},
    ]


# A capital F, 7 rows by 6 columns: 1 is the letter, 0 the background. It has
# no symmetry at all, so every flip, roll and tiling reads unambiguously.
F_GLYPH = ('1 1 1 1 1 1  '
           '1 1 0 0 0 0  '
           '1 1 0 0 0 0  '
           '1 1 1 1 0 0  '
           '1 1 0 0 0 0  '
           '1 1 0 0 0 0  '
           '1 1 0 0 0 0')

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
        {'key': 'rs', 'init': 't.reshape 7 6', 'pos': (40, 230), 'w': 230, 'h': 70},
        {'key': 'cr', 'init': 'comment folded into 7 rows of 6',
         'pos': (290, 245), 'w': 260, 'h': 30},
        {'key': 'out', 'init': 'out F', 'pos': (40, 330), 'w': 150, 'h': 80},
    ],
    'links': [('met', '', 'msg', ''),
              ('msg', 'message out', 'rs', 'tensor in'),
              ('rs', 'output', 'out', 'F')],
}


def f_source(x=30, y=62):
    return [
        {'key': 'rs', 'init': 'p make_F', 'pos': (x, y), 'w': 160, 'h': 70},
        {'key': 'cF', 'comment': True, 'text': 'the F, 7 x 6 - open make_F to see how it is made',
         'pos': (x, y + 5)},
        {'key': 'hin', 'init': 'heat_map', 'pos': (x, y + 238), 'w': 200, 'h': 260,
         'props': HM(180, 210, 7, fmt='none')},
    ]


# The numbered tensor plus each cell's distance from the centre (for a disc
# mask). One metro, distance first, so the mask is in place before the
# numbers arrive at t.masked_select's triggering inlet.
DISC_SUB = {
    'name': 'make_disc',
    'host': 'src',
    'demo': [
        {'key': 'met', 'init': 'metro 100', 'pos': (40, 40), 'w': 129, 'h': 70,
         'props': METRO},
        {'key': 'cm', 'init': 'comment ten times a second: the distances, then the numbers',
         'pos': (190, 50), 'w': 460, 'h': 30},
        {'key': 'tb', 'init': 't bang bang', 'pos': (40, 130), 'w': 100, 'h': 46},
        {'key': 'gr', 'init': 'np.grid -1 1 -1 1 10 10', 'pos': (300, 200), 'w': 200, 'h': 220,
         'props': {'min x': -1.0, 'max x': 1.0, 'min y': -1.0, 'max y': 1.0,
                   'divisions x': 10, 'divisions y': 10, 'z': 0.0}},
        {'key': 'nm', 'init': 'np.linalg.norm', 'pos': (300, 440), 'w': 160, 'h': 70,
         'props': {'axis': '2'}},
        {'key': 'cg', 'init': 'comment a 10 x 10 grid from -1 to 1: each point\'s distance from the centre',
         'pos': (300, 520), 'w': 520, 'h': 30},
        {'key': 'od', 'init': 'out distance', 'pos': (300, 570), 'w': 150, 'h': 80},
        {'key': 'ar', 'init': 't.arange 0 100 1', 'pos': (40, 200), 'w': 190, 'h': 150,
         'props': {'start': 0.0, 'stop': 100.0, 'step': 1.0}},
        {'key': 'rs', 'init': 't.reshape 10 10', 'pos': (40, 370), 'w': 230, 'h': 70},
        {'key': 'on', 'init': 'out numbered', 'pos': (40, 570), 'w': 150, 'h': 80},
    ],
    'links': [('met', '', 'tb', ''),
              ('tb', 'bang', 'gr', '', 1), ('gr', 'grid points', 'nm', 'input'),
              ('nm', 'norm', 'od', 'distance'),
              ('tb', 'bang', 'ar', '###input', 0), ('ar', 'arange out', 'rs', 'tensor in'),
              ('rs', 'output', 'on', 'numbered')],
}


# ====================================================================== t[]
body = """Takes part of a tensor by position, written the way you would slice it in Python.

Use it for a row, a column, a block, every other element, or the last
entry - one joint out of a tensor of joints, the x column out of a list of
points, the middle of a frame.

THE INDEX STRING:
The indices field holds the part to take, one entry per dimension,
separated by commas - or, when there is no comma anywhere in it, by
spaces: 2:5, 4:8 and 2:5 4:8 mean the same, and so do 0, 3 and 0 3. The
first entry picks rows, the second columns, and so on. Each entry is one
of:

  3        one position. Counting starts at 0; -1 is the last. That
           dimension is REMOVED - 3 on a 10 x 10 tensor gives a 1-D row of
           10, and 3,7 gives a single value (a tensor with no dimensions).
  2:5      positions 2, 3 and 4 - the stop is NOT included. Leave out the
           start or the stop to run from the beginning or to the end;
           : alone is the whole dimension. The dimension is kept.
  ::2      a step: every second position. 1::3 is every third, from 1.
           The step must be positive - ::-1 (reverse) is refused by
           PyTorch: the console prints an error and nothing is sent. Use
           t.flip to reverse.
  ...      every remaining dimension: ...,0 takes the first entry of the
           LAST dimension, whatever the tensor's shape.

Fewer entries than dimensions leave the rest whole, so 2 on a 10 x 10
tensor is the same as 2,:. Square brackets around the whole string are
optional: [2:5, 4:8] and 2:5, 4:8 mean the same.

Unlike np.[], t[] does NOT take a list of positions: [0 3 7] is read
as three entries, row 0, column 3, position 7 of a third dimension - on
a 10 x 10 tensor that is one entry too many, and nothing is sent. An
entry that is not a number or a slice is read as : (the whole
dimension). To pick chosen rows or columns use t.index_select.

When you separate entries with spaces, do not put spaces inside an
entry: 2 : 5 would be read as three entries. With commas, spaces are
ignored.

SYNTAX:
t[] <index string>
Default ':', the whole tensor. The arguments become the index string,
read the same way as the field: t[] 0 3 is row 0, column 3 (one value),
and t[] 2:5 4:8 or t[] 2:5, 4:8 a 3 by 4 block.

EXAMPLE:
t[] 2:5 4:8

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or numpy array is converted to a float32 tensor).
Sends the selection on arrival.

indices (text field, also an inlet):
The index string. Changing it - typing, or a string or radio choice
arriving at the inlet - re-slices the LAST tensor received and sends at
once.

OUTPUTS:

output:
The selected part, on the same device and of the same dtype as the
input. Each single-number entry removes one dimension. It is a view of
the input, not a copy.

When the string has more entries than the tensor has dimensions, nothing
is sent and nothing is said. A position past the end (12 on a 10-row
tensor) prints an error in the console and sends nothing. A RANGE past
the end is not an error: 8:20 gives rows 8 and 9.

READING THE DEMO:
The left heat_map is the source, every cell printing its own number - the
tens digit is the row, the units digit the column. Pick an index string
in the radio and the right heat_map shows exactly the cells taken, still
numbered; t.info gives the shape of the result (3,7 is the single value
37, with no shape at all, so t.info's list goes blank). Or type your own
into the field on t[] - with commas or with spaces (3 7 is the same as
3,7).

Watch :,3 against :,3:4. Both take column 3, but :,3 removes the column
dimension, leaving a 1-D tensor of 10 - which a heat_map draws lying
down, as one row. :,3:4 keeps it as 10 rows of 1, standing up.

RELATED:
t.select takes one position along one dimension, chosen by numbers
rather than typed. t.narrow takes a run of positions along one
dimension. t.index_select picks a list of positions. t.masked_select
selects by a condition. np.[] is the numpy version, and does take lists
of positions."""

demo = numbered() + [
    {'key': 'rad', 'init': 'radio : 2 -1 :,3 :,3:4 2:5,4:8 ::2,::2 1::3 ...,0 3,7',
     'pos': (490, 62), 'w': 150, 'h': 380},
    {'key': 'c1', 'comment': True, 'text': 'pick an index string', 'pos': (490, 70)},
    {'key': 'sl', 'init': 't[] 2:5 4:8', 'pos': (490, 470), 'w': 250, 'h': 70},
    {'key': 'c2', 'comment': True, 'text': 'or type one into the field here',
     'pos': (490, 480)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (490, 560), 'w': 428, 'h': 310,
     'props': HM(400, 260, 3, 0.0, 99.0, fmt='%.0f')},
    {'key': 'c3', 'comment': True, 'text': 'the cells taken, still numbered',
     'pos': (490, 570)},
    {'key': 'inf', 'init': 't.info', 'pos': (30, 560), 'w': 150, 'h': 130},
    {'key': 'l1', 'init': 'list', 'pos': (30, 710), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c4', 'comment': True, 'text': 'shape of the selection', 'pos': (30, 715)},
] + see_also(920, ['t.select', 't.narrow', 't.index_select', 't.masked_select', 'np.[]'])
links = [('src', 'numbered', 'hsrc', 'y'),
         ('src', 'numbered', 'sl', 'tensor in'), ('rad', '', 'sl', 'indices'),
         ('sl', 'output', 'hout', 'y'), ('sl', 'output', 'inf', 'in'),
         ('inf', 'shape', 'l1', '')]
print(build('t[]', 't[] - take part of a tensor by index', body, demo, links,
            demo_width=940, text_width=760, text_height=1180, subpatch=NUM_SUB))

# ================================================================= t.select
body = """Takes one position along one dimension - one row, one column, one frame - and that dimension disappears.

t.select 0 3 on a 10 x 10 tensor is row 3: a 1-D tensor of 10. t.select
1 3 is column 3, also a 1-D tensor of 10. Use it to pull one joint, one
channel or one frame out of a larger tensor when you want the thing
itself rather than a tensor with a single row in it.

select VERSUS index_select:
Taking row 3 with t.select gives shape (10) - the row dimension is gone,
because a single row has no row dimension left. Taking rows [3] with
t.index_select gives shape (1, 10): still a matrix, with one row in it.
If something downstream expects a matrix, use t.index_select or t[] 3:4;
if you want the vector itself, t.select gets it with no squeeze.

SYNTAX:
t.select <dim: int> <index: int>
Defaults: dim 0, index 0.

EXAMPLE:
t.select 1 3

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or numpy array is converted to a float32 tensor).
Sends the selection on arrival. This is the only inlet that sends.

dim (int box, also an inlet):
Which dimension to take from: 0 rows, 1 columns, and so on. Negative
counts from the end (-1 is the last dimension).

index (int box, also an inlet):
Which position along it. Negative counts from the end (-1 is the last).

Changing dim or index does NOT send by itself - the next tensor uses
them. In the demo a metro inside make_numbered re-sends the source ten
times a second, so the picture follows your edits.

If dim or index is out of range for the tensor, nothing is sent and
nothing is said.

OUTPUTS:

output:
The slice, with one dimension fewer than the input. A 1-D input gives a
single value (a tensor with no dimensions). It is a view of the input.

READING THE DEMO:
The left heat_map is the numbered source - tens digit the row, units
digit the column. t.select 0 3 takes row 3: the strip shows 30 to 39.
Set dim to 1 and it is column 3 instead - 3, 13, 23 ... 93 - still laid
out as one row, because it is now a plain 1-D tensor with no memory of
having been a column. t.info shows the shape: [10] either way.

RELATED:
t[] does the same with a typed index (3 or :,3), and can take ranges
too. t.index_select takes several positions and keeps the dimension.
t.narrow takes a run of positions and keeps the dimension. t.unsqueeze
puts a removed dimension back. np.[] is the numpy version."""

demo = numbered() + [
    {'key': 'sel', 'init': 't.select 0 3', 'pos': (490, 162), 'w': 170, 'h': 110,
     'props': {'dim': 0, 'index': 3}},
    {'key': 'c1', 'comment': True, 'text': 'dim: 0 rows, 1 columns. index: which one',
     'pos': (490, 170)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (490, 300), 'w': 428, 'h': 100,
     'props': HM(400, 50, 1, 0.0, 99.0, fmt='%.0f')},
    {'key': 'c2', 'comment': True, 'text': 'the slice: a plain row of 10', 'pos': (490, 310)},
    {'key': 'inf', 'init': 't.info', 'pos': (490, 420), 'w': 150, 'h': 130},
    {'key': 'l1', 'init': 'list', 'pos': (490, 570), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c3', 'comment': True, 'text': 'its shape: one dimension fewer',
     'pos': (490, 575)},
] + see_also(660, ['t[]', 't.index_select', 't.narrow', 't.unsqueeze', 'np.[]'])
links = [('src', 'numbered', 'hsrc', 'y'), ('src', 'numbered', 'sel', 'tensor in'),
         ('sel', 'output', 'hout', 'y'), ('sel', 'output', 'inf', 'in'),
         ('inf', 'shape', 'l1', '')]
print(build('t.select', 't.select - one row or column, dimension removed', body, demo, links,
            demo_width=940, text_width=760, text_height=1000, subpatch=NUM_SUB))

# ============================================================ t.index_select
body = """Picks a list of positions along one dimension - rows 0, 3 and 7, say - in the order given, and keeps the dimension.

Use it to pull chosen joints out of a joint tensor, to reorder channels,
to repeat a row, or to reverse an axis with a descending list. The
indices can come from anywhere in the patch: a message, t.argsort, a
t.argwhere result.

SYNTAX:
t.index_select <dim: int>
With no argument the node works along dimension 0 (rows) and shows no
dim inlet. Give a dim (t.index_select 1) to get a dim box you can change.

EXAMPLE:
t.index_select 1

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or numpy array is converted to a float32 tensor).
Sends the selection on arrival. This is the only inlet that sends.

indices:
The positions to take, as whole numbers: a list (0 3 7), an array or a
tensor. They are converted to integers. Repeats are allowed (2 2 5) and
the order is kept. A 2-D list of indices is flattened to 1-D first. A
position past the end prints an error in the console and nothing is
sent; negative positions are refused the same way (unlike t[]). The
node sends nothing until indices have arrived. This inlet is cold:
changing the list does not send - the next tensor uses it. In the demo
a metro re-sends the source ten times a second.

dim (int box, also an inlet - only when an argument was given):
Which dimension the indices refer to: 0 rows, 1 columns. A dim the
tensor does not have sends nothing.

OUTPUTS:

output:
The selected rows (or columns), in the order of the indices. Same number
of dimensions as the input; along dim it is as long as the index list.
It is a new tensor, not a view.

READING THE DEMO:
The left heat_map is the numbered source - tens digit the row, units
digit the column. t.index_select 1 takes columns. Click a list: 0 3 7
gives three columns, 3, 13 ... 93 down the middle one; 9 8 7 ... 0
reverses the columns (the units digits now count down); 2 2 5 5 repeats
two of them. Set dim to 0 and the same lists pick rows instead.

RELATED:
t.select takes ONE position and removes the dimension. t[] slices by a
typed range. t.take treats the tensor as flat. t.take_along_dim takes a
different position in each row. t.argsort makes index lists that sort.
np.[] takes lists of positions in numpy."""

demo = numbered() + [
    {'key': 'lb', 'init': 'load_bang', 'pos': (490, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (490, 120), 'w': 100, 'h': 42, 'props': MSG('0 3 7')},
    {'key': 'm2', 'init': 'message', 'pos': (640, 120), 'w': 270, 'h': 42,
     'props': MSG('9 8 7 6 5 4 3 2 1 0', 240)},
    {'key': 'm3', 'init': 'message', 'pos': (490, 175), 'w': 120, 'h': 42, 'props': MSG('2 2 5 5')},
    {'key': 'c1', 'comment': True, 'text': 'click a list of positions', 'pos': (640, 180)},
    {'key': 'isel', 'init': 't.index_select 1', 'pos': (490, 250), 'w': 200, 'h': 110,
     'props': {'dim': 1}},
    {'key': 'c2', 'comment': True, 'text': 'dim 1: the list picks columns; 0 picks rows',
     'pos': (490, 255)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (490, 390), 'w': 428, 'h': 310,
     'props': HM(400, 260, 10, 0.0, 99.0, fmt='%.0f')},
    {'key': 'c3', 'comment': True, 'text': 'the columns taken, in the order given',
     'pos': (490, 400)},
    {'key': 'inf', 'init': 't.info', 'pos': (30, 500), 'w': 150, 'h': 130},
    {'key': 'l1', 'init': 'list', 'pos': (30, 650), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c4', 'comment': True, 'text': 'shape of the result', 'pos': (30, 655)},
] + see_also(760, ['t.select', 't[]', 't.take', 't.take_along_dim', 't.argsort', 'np.[]'])
links = [('lb', 'out', 'm1', ''),
         ('src', 'numbered', 'hsrc', 'y'), ('src', 'numbered', 'isel', 'tensor in'),
         ('m1', 'message out', 'isel', 'indices'), ('m2', 'message out', 'isel', 'indices'),
         ('m3', 'message out', 'isel', 'indices'),
         ('isel', 'output', 'hout', 'y'), ('isel', 'output', 'inf', 'in'),
         ('inf', 'shape', 'l1', '')]
print(build('t.index_select', 't.index_select - chosen rows or columns, in order', body,
            demo, links, demo_width=940, text_width=760, text_height=1040, subpatch=NUM_SUB))

# =========================================================== t.masked_select
body = """Takes out every value where a mask is true, and returns them as one flat list.

The mask is a tensor of true and false the same shape as the source -
usually the result of a comparison (<, >, ==). Use it to gather every
reading above a threshold, every joint inside a region, every pixel of a
colour, for counting, averaging or sorting.

THE RESULT IS FLAT:
However the source is shaped, what comes back is 1-D: the chosen values
in reading order (along the first row, then the next). The number that
pass is not known in advance and cannot in general be arranged into a
rectangle, so the shape is given up. To KEEP the shape - zeros where the
mask is false - multiply the source by the mask instead, or use
np.where.

SYNTAX:
t.masked_select
It takes no arguments.

INPUTS:

source tensor:
The values (a list or numpy array is converted to a float32 tensor).
Sends the selection on arrival. This is the only inlet that sends.

mask:
True where a value should be taken. Anything non-zero counts as true;
it is converted to a bool tensor on the source's device. This inlet is
cold: the mask is stored and used when the next source arrives, so send
it FIRST. Nothing is sent until a mask has arrived. The mask must be the
source's shape, or one that broadcasts to it (a single row of 10 applies
to every row); otherwise the console prints an error and nothing is sent.

OUTPUTS:

selection tensor:
A 1-D tensor of the chosen values, as many as the mask has true cells.
When none are true it is empty (shape [0]).

READING THE DEMO:
make_disc sends two pictures ten times a second: each cell's distance
from the centre, then the numbered 10 x 10 source (tens digit the row,
units digit the column). < 0.6 turns the distances into the mask - a
white disc - which reaches t.masked_select before the numbers do. The
list shows the numbers inside the disc, in reading order, and the strip
below draws them as one row. Drag 0.6 on the < node: the disc grows or
shrinks, and t.info counts how many cells pass. (Below about 0.16 no
cell passes: the list empties, but t.info keeps its last count, because
it ignores empty tensors.)

RELATED:
np.where keeps the shape and fills the rejected cells from a second
source. t.argwhere gives the POSITIONS of the true cells rather than
their values. The comparison nodes (greater, less) make masks.
t.index_select picks by position."""

demo = [
    {'key': 'src', 'init': 'p make_disc', 'pos': (30, 62), 'w': 190, 'h': 70},
    {'key': 'c0', 'comment': True, 'text': 'open make_disc to see how it is made',
     'pos': (30, 67)},
    {'key': 'lt', 'init': '< 0.6', 'pos': (480, 160), 'w': 130, 'h': 70,
     'props': {'': 0.6, 'output_type': 'bool'}},
    {'key': 'c1', 'comment': True, 'text': 'drag 0.6: the radius of the disc',
     'pos': (480, 165)},
    {'key': 'hsrc', 'init': 'heat_map', 'pos': (30, 250), 'w': 428, 'h': 310,
     'props': HM(400, 260, 10, 0.0, 99.0, fmt='%.0f')},
    {'key': 'csrc', 'comment': True, 'text': 'the source, each cell numbered',
     'pos': (30, 260)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (480, 250), 'w': 268, 'h': 300,
     'props': HM(240, 240, 10, 0.0, 1.0, fmt='none', color='hot')},
    {'key': 'c2', 'comment': True, 'text': 'the mask: white cells are taken',
     'pos': (480, 260)},
    {'key': 'ms', 'init': 't.masked_select', 'pos': (30, 590), 'w': 220, 'h': 90},
    {'key': 'l1', 'init': 'list', 'pos': (30, 700), 'w': 720, 'h': 42, 'props': LIST},
    {'key': 'c3', 'comment': True, 'text': 'the numbers inside the disc, in reading order',
     'pos': (30, 750)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 790), 'w': 728, 'h': 100,
     'props': HM(700, 50, 1, 0.0, 99.0, fmt='none')},
    {'key': 'c4', 'comment': True, 'text': 'the same values as one flat row',
     'pos': (30, 800)},
    {'key': 'inf', 'init': 't.info', 'pos': (480, 590), 'w': 150, 'h': 130},
    {'key': 'l2', 'init': 'list', 'pos': (650, 590), 'w': 100, 'h': 42, 'props': LIST},
    {'key': 'c5', 'comment': True, 'text': 'how many passed', 'pos': (650, 640)},
] + see_also(920, ['np.where', 't.argwhere', 'less', 't.index_select', 't[]'])
links = [('src', 'distance', 'lt', 'in'),
         ('lt', 'result', 'hm', 'y'), ('lt', 'result', 'ms', 'mask'),
         ('src', 'numbered', 'hsrc', 'y'), ('src', 'numbered', 'ms', 'source tensor'),
         ('ms', 'selection tensor', 'l1', ''), ('ms', 'selection tensor', 'hout', 'y'),
         ('ms', 'selection tensor', 'inf', 'in'), ('inf', 'shape', 'l2', '')]
print(build('t.masked_select', 't.masked_select - the values where a mask is true', body,
            demo, links, demo_width=780, text_width=760, text_height=1000, subpatch=DISC_SUB))

# =================================================================== t.take
body = """Takes values by their position in the tensor read as one flat list, whatever its shape.

Position 0 is the first value, and the count runs along the first row,
then the next: on a 10 x 10 tensor position 37 is row 3, column 7. Use it
when your positions come from something that counts cells rather than
rows and columns - t.argmax with no dim, a flattened index, a list of
pixel numbers.

SYNTAX:
t.take
It takes no arguments.

INPUTS:

tensor in:
The tensor (a list or numpy array is converted to a float32 tensor).
Sends on arrival. This is the only inlet that sends.

indices in:
The flat positions, as whole numbers: a list, an array or a tensor.
Negative positions count back from the end (-1 is the last value). A
position past the end prints an error in the console and nothing is
sent. The node sends nothing until indices have arrived. This inlet is
cold - the next tensor uses the new list; in the demo a metro re-sends
the source ten times a second.

OUTPUTS:

output:
The values taken, shaped like the INDICES, not like the input: a list of
4 positions gives 4 values; a 2 x 2 tensor of positions gives a 2 x 2
result.

READING THE DEMO:
In the numbered source every cell's number is also its flat position, so
what comes out names the cells taken. 0 11 22 ... 99 walks down the
diagonal; 9 19 29 39 goes down the last column; 99 0 50 jumps about.
The strip and the list show the values, in the order asked for.

RELATED:
t.index_select takes whole rows or columns by position. t[] slices by
row and column. t.take_along_dim takes one position per row. t.argmax
(with no dim) gives a flat position that t.take can use. np.[] is the
nearest numpy node."""

demo = numbered() + [
    {'key': 'lb', 'init': 'load_bang', 'pos': (490, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (490, 120), 'w': 330, 'h': 42,
     'props': MSG('0 11 22 33 44 55 66 77 88 99', 310)},
    {'key': 'm2', 'init': 'message', 'pos': (490, 175), 'w': 140, 'h': 42,
     'props': MSG('9 19 29 39')},
    {'key': 'm3', 'init': 'message', 'pos': (650, 175), 'w': 110, 'h': 42,
     'props': MSG('99 0 50')},
    {'key': 'c1', 'comment': True, 'text': 'click a list of flat positions', 'pos': (490, 225)},
    {'key': 'tk', 'init': 't.take', 'pos': (490, 270), 'w': 160, 'h': 90},
    {'key': 'hout', 'init': 'heat_map', 'pos': (490, 380), 'w': 428, 'h': 100,
     'props': HM(400, 50, 1, 0.0, 99.0, fmt='%.0f')},
    {'key': 'c2', 'comment': True, 'text': 'the values taken, in order', 'pos': (490, 390)},
    {'key': 'l1', 'init': 'list', 'pos': (490, 500), 'w': 400, 'h': 42, 'props': LIST},
] + see_also(620, ['t.index_select', 't[]', 't.take_along_dim', 't.argmax', 'np.[]'])
links = [('lb', 'out', 'm1', ''),
         ('src', 'numbered', 'hsrc', 'y'), ('src', 'numbered', 'tk', 'tensor in'),
         ('m1', 'message out', 'tk', 'indices in'), ('m2', 'message out', 'tk', 'indices in'),
         ('m3', 'message out', 'tk', 'indices in'),
         ('tk', 'output', 'hout', 'y'), ('tk', 'output', 'l1', '')]
print(build('t.take', 't.take - values by flat position', body, demo, links,
            demo_width=940, text_width=760, text_height=900, subpatch=NUM_SUB))

# t.take_along_dim: its page is hand-made now (see tools/README.md).

# ================================================================= t.narrow
body = """Takes a run of consecutive positions along one dimension: start here, take this many.

t.narrow with dim 0, start 2, length 3 gives rows 2, 3 and 4 - the same as
t[] 2:5 - but the three numbers sit in boxes you can drag or drive from
the patch, which a typed slice cannot. Use it for a moving window over
frames, a band of rows, a block of channels.

SYNTAX:
t.narrow <dim: int> <start: int> <length: int>
Defaults: dim 0, start 0, length 1. t.narrow 0 2 3 takes rows 2, 3 and
4; t.narrow 1 0 4 the first four columns.

EXAMPLE:
t.narrow 0 2 3

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or numpy array is converted to a float32 tensor).
Sends the run on arrival. This is the only inlet that sends.

dim (int box, also an inlet):
Which dimension to cut along: 0 rows, 1 columns. Negative counts from
the end.

start (drag int, also an inlet):
The first position kept. Negative counts from the end (-3 is three from
the end). Default 0, or the second argument.

length (drag int, also an inlet):
How many positions to keep. Default 1, or the third argument.

Changing dim, start or length does NOT send by itself - the next tensor
uses them; in the demo a metro inside make_numbered re-sends the source
ten times a second. The run must fit: a start past the end, a length of
0 or less, or a run that would go past the end sends nothing and prints
an error in the console - once per tensor, so with a streaming source
the console fills until you bring the numbers back in range.

OUTPUTS:

tensor out:
The run. The dimension is KEPT (length 1 gives one row, still a matrix).
It is a view of the input, not a copy.

READING THE DEMO:
The left heat_map is the numbered source - tens digit the row, units
digit the column. t.narrow 0 2 3 starts on rows 2, 3 and 4 (20 to 49).
Drag start and the band of rows slides; drag length
and it widens. Set dim to 1 and it cuts columns instead.

RELATED:
t[] does the same with a typed slice (2:5), and more. t.select takes one
position and removes the dimension. t.chunk cuts a tensor into equal
pieces. np.crop cuts a rectangle in numpy."""

demo = numbered() + [
    {'key': 'nr', 'init': 't.narrow 0 2 3', 'pos': (490, 162), 'w': 170, 'h': 150,
     'props': {'dim': 0, 'start': 2, 'length': 3}},
    {'key': 'c1', 'comment': True, 'text': 'drag start and length; dim 0 rows, 1 columns',
     'pos': (490, 170)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (490, 340), 'w': 428, 'h': 310,
     'props': HM(400, 260, 3, 0.0, 99.0, fmt='%.0f')},
    {'key': 'c2', 'comment': True, 'text': 'the run kept, still numbered', 'pos': (490, 350)},
    {'key': 'inf', 'init': 't.info', 'pos': (30, 500), 'w': 150, 'h': 130},
    {'key': 'l1', 'init': 'list', 'pos': (30, 650), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c3', 'comment': True, 'text': 'shape of the run', 'pos': (30, 655)},
] + see_also(720, ['t[]', 't.select', 't.chunk', 't.index_select', 'np.crop'])
links = [('src', 'numbered', 'hsrc', 'y'), ('src', 'numbered', 'nr', 'tensor in'),
         ('nr', 'tensor out', 'hout', 'y'), ('nr', 'tensor out', 'inf', 'in'),
         ('inf', 'shape', 'l1', '')]
print(build('t.narrow', 't.narrow - a run of positions along one dimension', body, demo,
            links, demo_width=940, text_width=760, text_height=1000, subpatch=NUM_SUB))

# ================================================================ t.scatter
body = """Writes the values of a source tensor INTO a copy of a target tensor, at positions you give.

It is the reverse of selecting: t.index_select reads values out at
positions, t.scatter puts values in at positions. Everything the
positions do not name keeps the target's value. Use it to place readings
into a larger frame, to build a one-hot vector, or to undo a
t.argsort.

HOW THE POSITIONS ARE READ:
For a 1-D target, source value number i goes to position indices[i].
Only as many source values are used as there are indices, so the source
may be longer than the index list but not shorter. For 2-D and more,
indices must have as many dimensions as the target, and along dim each
index says where in that row (dim 1) or column (dim 0) its source value
goes - the same layout t.take_along_dim reads, written the other way.
If two indices name the same position, only one of the values ends up
there, and PyTorch does not promise which.

SYNTAX:
t.scatter <dim: int>
Default dim 0. The dim box is always shown.

EXAMPLE:
t.scatter 0

INPUTS and PARAMETERS:

tensor to scatter into:
The target (a list or numpy array is converted to a float32 tensor). It
is not changed: the result is a new tensor. Sends on arrival - this is
the only inlet that sends.

indices in:
Where to write, as whole numbers (a list, an array or a tensor). A
position past the end prints an error in the console and nothing is
sent. Cold: stored for the next target.

source in:
The values to write, converted to a float32 tensor. Cold. Its dtype must
match the target's: a float32 target is the normal case; an integer
target (from t.arange with int64, say) is refused with an error.

dim (int box, also an inlet):
Which dimension the indices run along. Negative counts from the end (-1
is the last dimension). A dimension the target does not have sends
nothing.

Nothing is sent until both indices and source have arrived.

OUTPUTS:

output:
The target with the source values written in.

READING THE DEMO:
The source is four values, 0.25 0.5 0.75 1 (top strip). A metro sends
t.zeros 10 - ten zeros - as the target ten times a second. Click a list
of positions: 0 1 2 3 puts the four values at the start; 9 6 3 0 spreads
them backwards from the end; 2 2 5 5 writes two values to each of two
places, and only one of each pair survives.

RELATED:
t.scatter_hold keeps its tensor between messages and takes index, value
pairs - the easier node for building a tensor up piece by piece.
t.index_select and t.take_along_dim read values out the same way.
np.edit writes into a region of a numpy array."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'ls', 'init': 't.linspace 0.25 1 4', 'pos': (30, 130), 'w': 200, 'h': 150,
     'props': {'start': 0.25, 'stop': 1.0, 'steps': 4}},
    {'key': 'hs', 'init': 'heat_map', 'pos': (260, 130), 'w': 228, 'h': 100,
     'props': HM(200, 50, 1, 0.0, 1.0, fmt='%.2f')},
    {'key': 'c0', 'comment': True, 'text': 'the source: four values', 'pos': (260, 240)},
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 300), 'w': 129, 'h': 70, 'props': METRO},
    {'key': 'zr', 'init': 't.zeros 10', 'pos': (30, 390), 'w': 180, 'h': 150,
     'props': {'shape': '[10]'}},
    {'key': 'c1', 'comment': True, 'text': 'the target: ten zeros, ten times a second',
     'pos': (30, 395)},
    {'key': 'm1', 'init': 'message', 'pos': (330, 300), 'w': 110, 'h': 42, 'props': MSG('0 1 2 3')},
    {'key': 'm2', 'init': 'message', 'pos': (460, 300), 'w': 110, 'h': 42, 'props': MSG('9 6 3 0')},
    {'key': 'm3', 'init': 'message', 'pos': (590, 300), 'w': 110, 'h': 42, 'props': MSG('2 2 5 5')},
    {'key': 'c2', 'comment': True, 'text': 'click a list of positions', 'pos': (330, 350)},
    {'key': 'sc', 'init': 't.scatter 0', 'pos': (30, 570), 'w': 230, 'h': 130,
     'props': {'dim': 0}},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 720), 'w': 528, 'h': 100,
     'props': HM(500, 50, 1, 0.0, 1.0, fmt='%.2f')},
    {'key': 'c3', 'comment': True, 'text': 'the ten values, with the source written in',
     'pos': (30, 830)},
] + see_also(880, ['t.scatter_hold', 't.index_select', 't.take_along_dim', 'np.edit'])
links = [('lb', 'out', 'ls', '###input'), ('lb', 'out', 'm1', ''),
         ('ls', 'linspace out', 'hs', 'y'), ('ls', 'linspace out', 'sc', 'source in'),
         ('met', '', 'zr', '###input'), ('zr', 'tensor of zeros', 'sc', 'tensor to scatter into'),
         ('m1', 'message out', 'sc', 'indices in'), ('m2', 'message out', 'sc', 'indices in'),
         ('m3', 'message out', 'sc', 'indices in'),
         ('sc', 'output', 'hout', 'y')]
print(build('t.scatter', 't.scatter - write values in at chosen positions', body, demo,
            links, demo_width=720, text_width=760, text_height=1080))

# =========================================================== t.scatter_hold
body = """Holds a 1-D tensor and writes values into it from index, value pairs, keeping everything else as it was.

Send it 3 0.8 and slot 3 becomes 0.8; send 3 0.8 7 0.2 and two slots
change at once. It remembers between messages, so values arriving one
at a time, from different nodes and at different rates, build up into a
single tensor that always holds the latest of each - per-joint readings
that arrive separately and something downstream that wants them all
together.

SYNTAX:
t.scatter_hold <length: int>
Default length 16.

EXAMPLE:
t.scatter_hold 8

INPUTS and PARAMETERS:

list of index value pairs:
Alternating positions and values: index, value, index, value... (a list,
an array or a tensor). Writes them and sends the whole tensor. An odd
number of entries is ignored - it cannot be read as pairs. Indices are
rounded down to whole numbers and pulled into range: anything below 0
writes slot 0 and anything past the end writes the LAST slot, so a stray
index quietly overwrites an end slot rather than failing.

length of target tensor (drag int, also an inlet):
How many slots. Changing it makes a new tensor of zeros - what was held
is lost. Minimum 1.

clear (button, also an inlet):
Sets every slot back to zero and sends the cleared tensor, so a display
downstream clears at once.

OUTPUTS:

output:
The whole held tensor (float32) after each write - not just the slots
that changed - so anything downstream always sees the complete current
state. Each send is a copy: a tensor already sent is not changed by
later writes, so a node further on that keeps one (var, a buffer) keeps
the state at that moment.

READING THE DEMO:
The first message is sent once on load. Click either message: each
changes only its own slots and the rest keep
what they had. Drag the slider to write its value into the slot named by
the int box (3 to start); pak sends the pair on every move, so that one
slot follows the slider. Type another number in the int box to move the
slider to another slot - the old one keeps its last value. Click clear
on the node and the strip goes back to zeros at once; move the slider
or click a message to start building it up again.

RELATED:
t.scatter writes a whole source tensor into a target at given positions,
without holding anything. np.edit writes into a region of a numpy array.
t.index_select reads values out by position."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 20), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 80), 'w': 160, 'h': 42,
     'props': MSG('2 0.9 5 0.4')},
    {'key': 'm2', 'init': 'message', 'pos': (210, 80), 'w': 160, 'h': 42,
     'props': MSG('0 0.7 7 0.2')},
    {'key': 'c0', 'comment': True, 'text': 'click: index, value, index, value',
     'pos': (390, 85)},
    {'key': 'ix', 'init': 'int', 'pos': (30, 140), 'w': 110, 'h': 42,
     'props': dict(INT, **{'': 3})},
    {'key': 'sl', 'init': 'slider', 'pos': (160, 140), 'w': 220, 'h': 60,
     'props': {'': 0.5, 'min': 0.0, 'max': 1.0, 'format': '%.2f', 'width': 200}},
    {'key': 'c1', 'comment': True, 'text': 'left: which slot. right: its value',
     'pos': (400, 145)},
    {'key': 'pk', 'init': 'pak i f', 'pos': (30, 220), 'w': 130, 'h': 80},
    {'key': 'sh', 'init': 't.scatter_hold 8', 'pos': (30, 310), 'w': 260, 'h': 110,
     'props': {'length of target tensor': 8}},
    {'key': 'c2', 'comment': True, 'text': 'click clear to zero it', 'pos': (310, 315)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 440), 'w': 448, 'h': 100,
     'props': HM(420, 50, 1, 0.0, 1.0, fmt='%.2f')},
    {'key': 'c3', 'comment': True, 'text': 'the eight slots: each write changes only its own',
     'pos': (30, 550)},
] + see_also(600, ['t.scatter', 'np.edit', 't.index_select', 'pak'])
links = [('lb', 'out', 'm1', ''),
         ('m1', 'message out', 'sh', 'list of index value pairs'),
         ('m2', 'message out', 'sh', 'list of index value pairs'),
         ('ix', '', 'pk', 'int'), ('sl', '', 'pk', 'float'),
         ('pk', 'out', 'sh', 'list of index value pairs'),
         ('sh', 'output', 'hm', 'y')]
print(build('t.scatter_hold', 't.scatter_hold - assemble a tensor from index, value pairs',
            body, demo, links, demo_width=640, text_width=760, text_height=920))

# =================================================================== t.diag
body = """Goes both ways between a matrix and its diagonal: a matrix in gives its diagonal as a vector; a vector in gives a square matrix with that vector on the diagonal.

Which you get depends only on what you send: 2-D in, 1-D out; 1-D in,
2-D out. The diagonal of a covariance matrix is each channel's variance
on its own, with the cross-terms removed - so t.diag after t.cov or
t.corrcoef separates "how much does each vary" from "what does it vary
with". Going the other way, t.diag of a list of gains makes a matrix
that scales each channel by its own gain.

SYNTAX:
t.diag
It takes no arguments; set which diag in its box.

INPUTS and PARAMETERS:

tensor in:
A matrix (2-D) or a vector (1-D). A list or numpy array is converted to
a float32 tensor. Sends on arrival. Any other number of dimensions is
refused by PyTorch with an error in the console.

which diag (int box):
Which diagonal: 0 the main one (top left to bottom right), 1 the one
just above it, -1 just below, and so on. For a matrix this picks the
diagonal read out (one shorter for each step away from the middle). For
a vector it places the vector on that diagonal, and the matrix grows to
fit: 5 values on diagonal 1 give a 6 x 6 matrix. Changing it does not
send; the next tensor uses it.

OUTPUTS:

output:
The diagonal (from a matrix) or the matrix (from a vector).

READING THE DEMO:
Left: the numbered 10 x 10 source (tens digit the row, units digit the
column). Its diagonal is 0 11 22 ... 99. Set which diag to 1 and it reads
1 12 23 ... 89, nine values; -1 reads 10 21 ... 98.
Right: the vector 1 2 3 4 5 becomes a 5 x 5 matrix, zero except the
diagonal. Set its which diag to 1 and the values move up a step and the
matrix becomes 6 x 6.

RELATED:
t.tril and t.triu keep a triangle of a matrix, on either side of the
same diagonals. t.eye makes the identity matrix (ones on the diagonal).
t.corrcoef gives a matrix whose diagonal is all ones."""

demo = numbered() + [
    {'key': 'dg', 'init': 't.diag', 'pos': (30, 500), 'w': 170, 'h': 90,
     'props': {'which diag': 0}},
    {'key': 'c1', 'comment': True, 'text': 'which diag: 0 the main one, 1 above, -1 below',
     'pos': (30, 505)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 610), 'w': 428, 'h': 100,
     'props': HM(400, 50, 1, 0.0, 99.0, fmt='%.0f')},
    {'key': 'c2', 'comment': True, 'text': 'the diagonal, as a vector', 'pos': (30, 720)},
    {'key': 'met', 'init': 'metro 100', 'pos': (500, 62), 'w': 129, 'h': 70, 'props': METRO},
    {'key': 'ls', 'init': 't.linspace 1 5 5', 'pos': (500, 150), 'w': 190, 'h': 150,
     'props': {'start': 1.0, 'stop': 5.0, 'steps': 5}},
    {'key': 'hv', 'init': 'heat_map', 'pos': (500, 320), 'w': 268, 'h': 100,
     'props': HM(240, 50, 1, 0.0, 5.0, fmt='%.0f')},
    {'key': 'c3', 'comment': True, 'text': 'a vector: 1 2 3 4 5', 'pos': (500, 430)},
    {'key': 'dg2', 'init': 't.diag', 'pos': (500, 470), 'w': 170, 'h': 90,
     'props': {'which diag': 0}},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (500, 580), 'w': 268, 'h': 260,
     'props': HM(240, 210, 5, 0.0, 5.0, fmt='%.0f')},
    {'key': 'c4', 'comment': True, 'text': 'the matrix built from it', 'pos': (500, 590)},
] + see_also(880, ['t.tril', 't.triu', 't.eye', 't.corrcoef'])
links = [('src', 'numbered', 'hsrc', 'y'), ('src', 'numbered', 'dg', 'tensor in'),
         ('dg', 'output', 'hout', 'y'),
         ('met', '', 'ls', '###input'), ('ls', 'linspace out', 'hv', 'y'),
         ('ls', 'linspace out', 'dg2', 'tensor in'), ('dg2', 'output', 'hm2', 'y')]
print(build('t.diag', 't.diag - a matrix\'s diagonal, or a matrix from a vector', body,
            demo, links, demo_width=800, text_width=760, text_height=980, subpatch=NUM_SUB))


# ======================================================== t.tril and t.triu
def triangle_page(name, which, kept, zeroed, other, example_text):
    body = f"""Keeps the {which} triangle of a matrix - the diagonal and everything {kept} it - and sets the rest to zero.

The shape stays the same; only the values {zeroed} the diagonal become
0. A matrix of relationships between things - a covariance, a distance
matrix, a similarity - is symmetric: the entry for A against B equals
the one for B against A. Half of it is repeated, and keeping one
triangle is how you count each pair once. With which diag set so the
diagonal itself goes too, the entries comparing each thing with itself
are removed as well - almost always what you want when asking about
pairs.

SYNTAX:
{name}
It takes no arguments; set which diag in its box.

INPUTS and PARAMETERS:

tensor in:
The matrix (a list or numpy array is converted to a float32 tensor).
Sends on arrival. A tensor of more than two dimensions is treated as a
stack of matrices, each cut the same way; a 1-D tensor is refused by
PyTorch with an error in the console.

which diag (int box):
Where the cut is. 0 cuts along the main diagonal (top left to bottom
right) and keeps it. 1 moves the cut one step up and to the right, -1
one step down and to the left. {example_text} Changing it does not send;
the next matrix uses it - in the demo a metro inside make_numbered
re-sends the source ten times a second.

OUTPUTS:

output:
The matrix, same shape, with the values {zeroed} the cut set to 0.

READING THE DEMO:
The left heat_map is the numbered source (tens digit the row, units
digit the column; the diagonal is 0 11 22 ... 99). The right one is
{name}'s result: the {which} triangle keeps its numbers and the rest is
dark zero. Step which diag and watch the edge of the triangle slide.

RELATED:
{other} keeps the opposite triangle. t.diag reads one diagonal out as a
vector. t.eye makes the identity matrix. In numpy, np.where with a
comparison of row and column numbers does the same job."""
    demo = numbered() + [
        {'key': 'tr', 'init': name, 'pos': (490, 162), 'w': 170, 'h': 90,
         'props': {'which diag': 0}},
        {'key': 'c1', 'comment': True, 'text': 'which diag: 0 main, 1 up a step, -1 down',
         'pos': (490, 170)},
        {'key': 'hout', 'init': 'heat_map', 'pos': (490, 280), 'w': 428, 'h': 310,
         'props': HM(400, 260, 10, 0.0, 99.0, fmt='%.0f')},
        {'key': 'c2', 'comment': True, 'text': f'the {which} triangle kept, the rest zero',
         'pos': (490, 290)},
    ] + see_also(640, [other, 't.diag', 't.eye', 'np.where'])
    links = [('src', 'numbered', 'hsrc', 'y'), ('src', 'numbered', 'tr', 'tensor in'),
             ('tr', 'output', 'hout', 'y')]
    print(build(name, f'{name} - keep the {which} triangle of a matrix', body, demo,
                links, demo_width=940, text_width=760, text_height=920, subpatch=NUM_SUB))


triangle_page('t.tril', 'lower', 'below', 'above', 't.triu',
              'So t.tril with which diag -1 keeps only what is strictly below\n'
              'the diagonal, and which diag 1 keeps one extra diagonal above it.')
triangle_page('t.triu', 'upper', 'above', 'below', 't.tril',
              'So t.triu with which diag 1 keeps only what is strictly above\n'
              'the diagonal - each pair once, nothing compared with itself.')

# =================================================================== t.roll
body = """Shifts everything in a tensor along a dimension, and what falls off one end comes back at the other.

Nothing is lost: it is a rotation of the order, not a shift into empty
space. Use it to scroll a pattern round, to offset a sequence against
itself by a fixed number of steps (a circular delay), or to step through
a circular list. If you want old values to fall off and zeros to arrive,
that is not this node.

SYNTAX:
t.roll <shifts: int> <dims: int>
Defaults: shifts 1, dims 0.

EXAMPLE:
t.roll 2 1

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or numpy array is converted to a float32 tensor).
Each tensor that arrives is rolled and sent. Only this inlet sends.

shifts (text field, also an inlet):
How many places to move. Positive moves towards the end (down for dim 0,
right for dim 1), negative the other way. A shift larger than the
dimension goes round again: on 6 columns, 7 is the same as 1. Give a
list (2, 1) to roll several dimensions at once.

dims (text field, also an inlet):
Which dimension each shift applies to: 0 rows, 1 columns; negative
counts from the end. With a list of shifts give a list of dims of the
same length (0, 1). If the two lists differ in length the tensor is
passed through unrolled (with verbose on, the console says why).

Changing shifts or dims does NOT send by itself - in the demo a metro
inside make_F re-sends the F ten times a second, so the picture follows
your edits.

OUTPUTS:

rolled tensor:
Same shape and values as the input, moved round. A new tensor.

READING THE DEMO:
t.roll 2 1 moves every column 2 places to the right; the two columns
pushed off the right edge come back on the left, so the F's stem now
sits in the middle. Type 0 into dims and it rolls down instead, its top
bar wrapping to the bottom. Type 2, 3 into shifts and 1, 0 into dims to
roll both ways at once.

RELATED:
t.flip reverses instead of shifting. t.repeat and t.tile make copies.
t.cat joins tensors. np.roll is the numpy version (one axis at a time)."""

demo = f_source() + [
    {'key': 'ro', 'init': 't.roll 2 1', 'pos': (280, 180), 'w': 200, 'h': 110,
     'props': {'shifts': '2', 'dims': '1'}},
    {'key': 'c1', 'comment': True, 'text': 'type shifts; dims 0 rows, 1 columns',
     'pos': (280, 185)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (280, 300), 'w': 200, 'h': 260,
     'props': HM(180, 210, 7, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 'left: in. right: rolled, wrapping round',
     'pos': (30, 590)},
] + see_also(640, ['t.flip', 't.repeat', 't.cat', 'np.roll'])
links = [('rs', 'F', 'hin', 'y'), ('rs', 'F', 'ro', 'tensor in'),
         ('ro', 'rolled tensor', 'hout', 'y')]
print(build('t.roll', 't.roll - shift along a dimension, wrapping round', body, demo,
            links, demo_width=600, text_width=760, text_height=920, subpatch=F_SUB))

# =================================================================== t.flip
body = """Reverses a tensor along one or more dimensions: the first row becomes the last, or the first column the last.

Use it to mirror an image or a grid (upside down, or left to right), to
reverse a sequence in time, or to turn an ascending order into a
descending one. It is also the way to reverse in PyTorch: t[] does not
accept a negative step (::-1).

SYNTAX:
t.flip <dim: int> <dim: int> ...
The dimensions to reverse. Default 0. t.flip 1 mirrors left to right;
t.flip 0 1 reverses both. They appear in the flip dims field, where they
can be changed.

EXAMPLE:
t.flip 1

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or numpy array is converted to a float32 tensor).
Each tensor that arrives is flipped and sent. Only this inlet sends.

flip dims (text field):
Which dimensions to reverse - the arguments, or 0 if there were none. 0 reverses the order of the
ROWS - the picture turns upside down. 1 reverses the COLUMNS - a mirror
image. Several at once, separated by commas or spaces: 0, 1 reverses
both, the same as a half turn. Negative counts from the end (-1 is the
last dimension). The next tensor uses a change.

When there are more entries than the tensor has dimensions, or the
field is empty, the tensor is passed through unchanged. A dimension
the tensor does not have (2 on a matrix), or the same one twice (0, 0),
raises an error in the console and nothing is sent.

OUTPUTS:

output:
Same shape and values as the input, in reversed order along the chosen
dimensions. A new tensor.

READING THE DEMO:
The F is a 7 by 6 tensor of 0s (purple) and 1s (yellow), drawn row 0
at the top. With flip dims 0 the F stands on its head, its bar at the
bottom; type 1 and it faces left instead; type 0 1 and it is turned
half way round. (The node is t.flip 0; typing t.flip 1 or t.flip 0 1
would start it on those.)

RELATED:
t.roll shifts with wrap-around instead of reversing. t.transpose swaps
rows and columns (a flip across the diagonal). t.argsort with
descending sorts the other way. np.flip and np.rot90 are the numpy
versions."""

demo = f_source() + [
    {'key': 'fl', 'init': 't.flip 0', 'pos': (280, 200), 'w': 170, 'h': 70,
     'props': {'flip dims': '0'}},
    {'key': 'c1', 'comment': True, 'text': 'type 0, 1, or 0 1 into flip dims',
     'pos': (280, 205)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (280, 300), 'w': 200, 'h': 260,
     'props': HM(180, 210, 7, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 'left: in. right: flipped', 'pos': (30, 590)},
] + see_also(640, ['t.roll', 't.transpose', 't.argsort', 'np.flip', 'np.rot90'])
links = [('rs', 'F', 'hin', 'y'), ('rs', 'F', 'fl', 'tensor in'),
         ('fl', 'output', 'hout', 'y')]
print(build('t.flip', 't.flip - reverse the order along a dimension', body, demo, links,
            demo_width=600, text_width=760, text_height=920, subpatch=F_SUB))


# ======================================================= t.repeat and t.tile
def repeat_page(name, inlet, outlet, counts, first, extra, see, fewer, try_more):
    rows, cols = 7 * counts[0], 6 * counts[1]
    count_text = f'{counts[0]}, {counts[1]}'
    body = first + f"""

Each count says how many copies to make along one dimension, so {name}
{counts[0]} {counts[1]} on a 7 x 6 tensor stacks {counts[0]} copies down and
{counts[1]} across: {rows} x {cols}. This is NOT numpy's repeat, which copies each
element next to itself (0 0 1 1 2 2); here the WHOLE tensor is laid
end to end (0 1 2 0 1 2).

SYNTAX:
{name} <count: int> <count: int> ...
Usually one count per dimension (see the counts field below). With no
arguments the field is empty and the tensor is passed through
unchanged.

EXAMPLE:
{name} {counts[0]} {counts[1]}

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or numpy array is converted to a float32 tensor).
Sends the copies on arrival. Only this inlet sends.

{inlet} (text field, also an inlet):
The counts, separated by commas or spaces. {fewer} More counts than dimensions add new
dimensions at the front: 2, 1, 1 on a 7 x 6 tensor gives 2 x 7 x 6, two
copies stacked along a new first dimension. A count of 1 leaves that
dimension alone; 0 empties it. The next tensor uses a change - in the
demo a metro inside make_F re-sends the F ten times a second.
{extra}
OUTPUTS:

{outlet}:
The copies, as one new tensor.

READING THE DEMO:
The F (7 x 6, left) tiled {count_text} gives {rows} x {cols} on the right - {counts[0]} rows of
{counts[1]} F's. Type other counts: 1, 4 is a row of four, 3, 1 a column of
three{try_more}. t.info shows the shape of the result."""
    demo = f_source() + [
        {'key': 'rp', 'init': f'{name} {counts[0]} {counts[1]}', 'pos': (280, 62), 'w': 220,
         'h': 70, 'props': {inlet: count_text}},
        {'key': 'c1', 'comment': True, 'text': 'type counts: down, across', 'pos': (280, 67)},
        {'key': 'hout', 'init': 'heat_map', 'pos': (280, 160), 'w': 400, 'h': 360,
         'props': HM(380, 320, rows, fmt='none')},
        {'key': 'c2', 'comment': True, 'text': 'the copies, laid end to end',
         'pos': (280, 170)},
        {'key': 'inf', 'init': 't.info', 'pos': (280, 560), 'w': 150, 'h': 130},
        {'key': 'l1', 'init': 'list', 'pos': (450, 560), 'w': 150, 'h': 42, 'props': LIST},
        {'key': 'c3', 'comment': True, 'text': 'shape of the result', 'pos': (450, 610)},
    ] + see_also(720, see)
    links = [('rs', 'F', 'hin', 'y'), ('rs', 'F', 'rp', 'tensor in'),
             ('rp', outlet, 'hout', 'y'), ('rp', outlet, 'inf', 'in'),
             ('inf', 'shape', 'l1', '')]
    print(build(name, f'{name} - copies of the whole tensor, end to end', body, demo,
                links, demo_width=720, text_width=760, text_height=1000, subpatch=F_SUB))


repeat_page('t.repeat', 'repeats', 'repeated tensor out', (2, 3),
            """Makes copies of the whole tensor and lays them end to end, along each dimension - tiling.

Use it to fill a larger grid with a pattern, to make a batch of
identical copies, or to stretch a single row into a matrix of identical
rows (t.repeat 4 1 on a 1 x 10 row gives 4 x 10).""",
            '',
            ['t.tile', 't.cat', 't.roll', 'np.repeat', 'np.concatenate'],
            """There must be AT LEAST as
many counts as the tensor has dimensions - fewer prints an error in the
console and nothing is sent (t.tile accepts fewer).""",
            '')

repeat_page('t.tile', 'tiling', 'tiled tensor out', (3, 2),
            """Makes copies of the whole tensor and lays them end to end, along each dimension - tiling.

Use it to fill a larger grid with a pattern or to make a batch of
identical copies. It is t.repeat with one difference: t.tile accepts
fewer counts than the tensor has dimensions.""",
            '',
            ['t.repeat', 't.cat', 't.roll', 'np.repeat', 'np.concatenate'],
            """Fewer counts than the tensor has
dimensions are filled out with 1s at the FRONT, so the counts you give
apply to the last dimensions: 2 on a 7 x 6 tensor means 1, 2 and doubles
its width (7 x 12). (t.repeat refuses fewer counts.)""",
            ', and a single 2 is two side by side')
