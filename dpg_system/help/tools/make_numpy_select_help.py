"""np selecting: np.[], np.crop, np.where, np.argwhere -- one page each.

These were one family page (np_select). Each now has its own page whose demo
shows the source array and the selected part side by side, with a SEE ALSO row
of help_links under it.
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


# The numbered 10 x 10 array used by np.[] and np.crop: np.linspace 0 99 100
# folded into 10 rows, so every cell's number IS its position - 37 is row 3,
# column 7. Whatever comes out of the selector names the cells it took.
def numbered_source(trigger_key, trigger_init, trigger_props=None):
    head = {'key': trigger_key, 'init': trigger_init, 'pos': (30, 62),
            'w': 129 if trigger_init.startswith('metro') else 88,
            'h': 70 if trigger_init.startswith('metro') else 46}
    if trigger_props:
        head['props'] = trigger_props
    return [
        head,
        {'key': 'ls', 'init': 'np.linspace 0 99 100', 'pos': (30, 150), 'w': 190, 'h': 150,
         'props': {'start': 0.0, 'stop': 99.0, 'steps': 100, 'dtype': 'float32'}},
        {'key': 'rs', 'init': 'np.reshape 10 10', 'pos': (30, 320), 'w': 220, 'h': 46},
        {'key': 'hsrc', 'init': 'heat_map', 'pos': (30, 390), 'w': 428, 'h': 310,
         'props': HM(400, 260, 10, 0.0, 99.0, fmt='%.0f')},
        {'key': 'csrc', 'comment': True,
         'text': 'the source: 10 x 10, each cell numbered\nrow then column - 37 is row 3, column 7',
         'pos': (30, 400)},
    ]


SOURCE_LINKS = [('ls', 'linspace out', 'rs', 'tensor in'), ('rs', 'output', 'hsrc', 'y')]


# ------------------------------------------------------------------------ np.[]
body = """Takes part of an array by position, written the way you would slice it in Python.

Use it whenever you want a row, a column, a block, every other element, or
a few chosen entries - one joint out of a list of joints, the x column out
of a list of points, the middle of a frame, the last value of a buffer.

THE INDEX STRING:
The Indices field holds the part to take, one entry per dimension,
separated by commas. The first entry picks rows, the second columns, and
so on. Each entry is one of:

  3        one position. Counting starts at 0; -1 is the last. That
           dimension is REMOVED - 3 on a 10 x 10 array gives a 1-D row of
           10, and 3,7 gives a single number.
  2:5      positions 2, 3 and 4 - the stop is NOT included. Leave out the
           start or the stop to run from the beginning or to the end;
           : alone is the whole dimension. The dimension is kept.
  ::2      a step: every second position. 1::2 the odd ones, ::-1 the
           whole dimension reversed.
  [0 3 7]  a list of positions, taken in the order given. Inside a
           longer string any separator works (:,[0,3,7] is columns 0, 3
           and 7). On its own, use spaces or semicolons - [0 3 7] or
           [0;3;7] - because [0,3,7] with commas reads as brackets around
           three single positions (0,3,7), which a 2-D array cannot take.
  ...      every remaining dimension: ...,0 takes the first entry of the
           LAST dimension, whatever the array's shape.

Fewer entries than dimensions leave the rest whole, so 2 on a 10 x 10
array is the same as 2,:. Square brackets around the whole string are
optional: [2:5, 4:8] and 2:5, 4:8 mean the same.

SYNTAX:
np.[] <index string>
Default ':', the whole array. The arguments are joined back together
with spaces, so np.[] 2:5, 4:8 and np.[] [0 3 7] both work as typed.

EXAMPLE:
np.[] 2:5,4:8

INPUTS and PARAMETERS:

tensor in:
The array (a list is converted). Sends the selection on arrival.

Indices (text field, also an inlet):
The index string. Changing it - typing, or a string or radio choice
arriving at the inlet - re-slices the LAST array received and sends at
once, so the demo needs no metro. An array of whole numbers sent here
(np.argsort's indices, say) picks those positions along the first axis,
in that order - see the np.argsort page, which sorts with it.

indices field width (option):
The field resizes to fit what you type; this records the width.

OUTPUTS:

output:
The selected part. Its number of dimensions follows the rules above:
each single-number entry removes one. Selecting every dimension with a
single number gives a plain number rather than an array.

When the string asks for more dimensions than the array has, nothing is
sent and nothing is said. A position past the end (12 on a 10-row array)
prints an error in the console and sends nothing. A RANGE past the end is
not an error: 8:20 gives rows 8 and 9.

READING THE DEMO:
The left heat_map is the source, every cell printing its own number - the
tens digit is the row, the units digit the column. Pick an index in the
radio and the right heat_map shows exactly the cells taken, still
numbered; np.shape gives the shape of the result. Or type your own into
the field on np.[].

Watch :,3 against :,3:4. Both take column 3, but :,3 removes the column
dimension, leaving a 1-D array of 10 - which a heat_map draws lying down,
as one row. :,3:4 keeps it as 10 rows of 1, standing up.

RELATED:
np.crop cuts a rectangle out by its edges, which is easier to drive from
sliders. np.where and np.argwhere select by a condition rather than by
position. np.edit WRITES into a part of an array named the same way. t[]
is the PyTorch version (on the t.index_select page)."""

demo = numbered_source('lb', 'load_bang') + [
    {'key': 'c0', 'comment': True, 'text': 'sends the numbered array once, on load',
     'pos': (30, 70)},
    {'key': 'rad', 'init': 'radio : 2 -1 :,3 :,3:4 2:5,4:8 ::2,::2 ::-1 [0;3;7] [3,7]',
     'pos': (490, 62), 'w': 150, 'h': 380},
    {'key': 'c1', 'comment': True, 'text': 'pick an index string', 'pos': (490, 70)},
    {'key': 'sl', 'init': 'np.[] 2:5,4:8', 'pos': (490, 470), 'w': 190, 'h': 70},
    {'key': 'c2', 'comment': True, 'text': 'or type one into the field here',
     'pos': (490, 480)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (490, 560), 'w': 428, 'h': 310,
     'props': HM(400, 260, 4, 0.0, 99.0, fmt='%.0f')},
    {'key': 'c3', 'comment': True, 'text': 'the cells taken, still numbered',
     'pos': (490, 570)},
    {'key': 'shp', 'init': 'np.shape', 'pos': (30, 730), 'w': 110, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (30, 820), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c4', 'comment': True, 'text': 'shape of the selection', 'pos': (30, 825)},
] + see_also(920, ['np.crop', 'np.where', 'np.edit', 'np.argsort', 't.index_select'])
links = [('lb', 'out', 'ls', '')] + SOURCE_LINKS + [
    ('rs', 'output', 'sl', 'tensor in'), ('rad', '', 'sl', 'Indices'),
    ('sl', 'output', 'hout', 'y'), ('sl', 'output', 'shp', 'np in'),
    ('shp', 'shape', 'l1', '')]
print(build('np.[]', 'np.[] - take part of an array by index', body, demo, links,
            demo_width=940, text_width=760, text_height=1100))

# ---------------------------------------------------------------------- np.crop
body = """Cuts a rectangle out of a 2-D array or an image, given its left, top, right and bottom edges.

It is the picture-shaped way of selecting: four numbers you can put on
sliders, rather than a slice to type. Use it to keep a region of a camera
frame, a window of a depth map, or the corner of a matrix.

THE EDGES:
left and top are the first column and row KEPT. right and bottom are the
first column and row LEFT OUT - so the result is (bottom - top) rows by
(right - left) columns. np.crop 2 3 7 8 keeps rows 3 to 7 and columns 2 to
6: 5 x 5. This is the same as np.[] 3:8,2:7 - note the order differs:
np.crop is x first (left, top...), the slice is rows first.

Only the first two dimensions are cut, so an image shaped (height, width,
3) keeps all three colours. A 1-D array is ignored: nothing is sent.

SYNTAX:
np.crop <left: int> <top: int> <right: int> <bottom: int>
Defaults: 0 0 256 256.

EXAMPLE:
np.crop 2 3 7 8

INPUTS and PARAMETERS:

image in:
The array (a list is converted). Sends the cropped piece on arrival.
This is the ONLY inlet that makes the node send.

left / top / right / bottom (drag ints, also inlets):
The edges. Changing them does NOT re-crop the last array - the next one
to arrive uses them. In the demo a metro resends the source 10 times a
second, so the crop follows your drags.
Out-of-range edges are pulled inside the array before cutting: a right or
bottom beyond the array means its far edge, and the rectangle is never
less than one cell. The boxes keep showing what you typed - right 256 on a
10-column array crops at column 10.

uncrop (button):
Sets the edges to the whole array (0, 0, width, height) of the last array
received. It does not send by itself; the next array comes out whole.

OUTPUTS:

out array:
The rectangle, as a new contiguous array.

READING THE DEMO:
The left heat_map is the source, each cell printing its own number - tens
digit the row, units digit the column. Drag the edges on np.crop and the
right heat_map shows the rectangle kept, still numbered, so you can read
which rows and columns survived. np.shape gives its size.

RELATED:
np.[] selects by a typed slice and can also step, reverse and pick lists
of rows. np.clip is a different thing despite the similar name: it limits
VALUES, keeping the shape. np.edit writes into a region. pc_crop crops a
point cloud by a 3-D box."""

demo = numbered_source('met', 'metro 100',
                       {'on': True, 'period': 100.0, 'units': 'milliseconds'}) + [
    {'key': 'c0', 'comment': True, 'text': 'resends the source 10 times a second',
     'pos': (30, 70)},
    {'key': 'cr', 'init': 'np.crop 2 3 7 8', 'pos': (490, 150), 'w': 170, 'h': 190,
     'props': {'left': 2, 'top': 3, 'right': 7, 'bottom': 8}},
    {'key': 'c1', 'comment': True, 'text': 'drag left, top, right, bottom',
     'pos': (490, 160)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (490, 390), 'w': 428, 'h': 310,
     'props': HM(400, 260, 5, 0.0, 99.0, fmt='%.0f')},
    {'key': 'c2', 'comment': True, 'text': 'the rectangle kept, still numbered',
     'pos': (490, 400)},
    {'key': 'shp', 'init': 'np.shape', 'pos': (490, 730), 'w': 110, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (490, 820), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c3', 'comment': True, 'text': '(rows, columns) kept', 'pos': (490, 825)},
] + see_also(910, ['np.[]', 'np.clip', 'np.edit', 'pc_crop'])
links = [('met', '', 'ls', '')] + SOURCE_LINKS + [
    ('rs', 'output', 'cr', 'image in'),
    ('cr', 'out array', 'hout', 'y'), ('cr', 'out array', 'shp', 'np in'),
    ('shp', 'shape', 'l1', '')]
print(build('np.crop', 'np.crop - cut a rectangle out of an array', body, demo, links,
            demo_width=940, text_width=760, text_height=980))

# --------------------------------------------------------------------- np.where
body = """Builds a new array cell by cell: where the condition is true it takes the value from 'if true', elsewhere from 'if false'.

Think of it as a stencil or a masked blend rather than a lookup. Feed it a
comparison as the condition and you have "use this where the reading is
good and that where it is not" across a whole array in one step - no
loop, no branching. Replace bad readings with a fallback, paste one image
into another through a mask, zero everything below a threshold.

SYNTAX:
np.where
It takes no arguments.

INPUTS and PARAMETERS:

condition:
The mask. Usually the result of a comparison node (>, <, ==) on an array,
but any array works: a non-zero value counts as true. This is the only
inlet that makes the node send.

if true / if false:
The values to choose from. Each can be an array the same shape as the
condition, or a single number used everywhere - 'if false' 0 blanks every
cell the condition rejects. These inlets are cold: they are stored, and
used when the next condition arrives. Send them FIRST.

The node sends NOTHING until both 'if true' and 'if false' have received
something. A np.where with one branch unwired stays silent and gives no
sign why. If the shapes cannot be matched up (a 10 x 10 condition with a
6 x 6 'if true'), nothing is sent either; with verbose on the console
names the shapes.

OUTPUTS:

selected:
An array the shape of the condition, each cell from one source or the
other.

READING THE DEMO:
np.grid makes a 10 x 10 square of points from -1 to 1. Two pictures come
out of it: 'if true' is the x of each point (dark on the left, bright on
the right), 'if false' is the y (dark at the top, bright at the bottom).
np.linalg.norm gives each point's distance from the centre, and < 0.75
turns that into the condition: a white disc, true inside, black outside.
The result shows the left-to-right ramp inside the disc and the top-to-
bottom ramp around it. Drag the number on the < node and the disc grows
or shrinks.

The 't array array' fires right to left, so both pictures reach np.where
before the condition does - the order matters, because only the condition
makes it send.

RELATED:
np.argwhere answers a different question about the same mask: not what
values to use, but WHERE the true cells are. The comparison nodes (>, <,
...) make the condition. np.clip limits values without a mask. In PyTorch,
t.masked_select (on the t.index_select page) takes out just the values
where a mask is true. np.[] selects by position."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 100.0, 'units': 'milliseconds'}},
    {'key': 'c0', 'comment': True, 'text': 'remakes the grid 10 times a second',
     'pos': (30, 70)},
    {'key': 'gr', 'init': 'np.grid -1 1 -1 1 10 10', 'pos': (30, 150), 'w': 200, 'h': 220,
     'props': {'min x': -1.0, 'max x': 1.0, 'min y': -1.0, 'max y': 1.0,
               'divisions x': 10, 'divisions y': 10, 'z': 0.0}},
    {'key': 'tr', 'init': 't array array', 'pos': (30, 390), 'w': 140, 'h': 46},
    {'key': 'sx', 'init': 'np.[] :,:,0', 'pos': (250, 470), 'w': 150, 'h': 70},
    {'key': 'sy', 'init': 'np.[] :,:,1', 'pos': (470, 470), 'w': 150, 'h': 70},
    {'key': 'hx', 'init': 'heat_map', 'pos': (250, 560), 'w': 208, 'h': 250,
     'props': HM(200, 200, 10, -1.0, 1.0, fmt='none')},
    {'key': 'hy', 'init': 'heat_map', 'pos': (470, 560), 'w': 208, 'h': 250,
     'props': HM(200, 200, 10, -1.0, 1.0, fmt='none')},
    {'key': 'c1', 'comment': True, 'text': "'if true': x of each point. 'if false': y",
     'pos': (250, 570)},
    {'key': 'nm', 'init': 'np.linalg.norm', 'pos': (30, 470), 'w': 160, 'h': 70,
     'props': {'axis': '2'}},
    {'key': 'lt', 'init': '< 0.75', 'pos': (30, 560), 'w': 130, 'h': 70,
     'props': {'': 0.75, 'output_type': 'bool'}},
    {'key': 'c2', 'comment': True, 'text': 'drag 0.75: the radius of the disc',
     'pos': (30, 570)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 840), 'w': 208, 'h': 250,
     'props': HM(200, 200, 10, 0.0, 1.0, fmt='none', color='hot')},
    {'key': 'c3', 'comment': True, 'text': 'the condition: true inside the disc',
     'pos': (30, 850)},
    {'key': 'wh', 'init': 'np.where', 'pos': (470, 840), 'w': 160, 'h': 100},
    {'key': 'hr', 'init': 'heat_map', 'pos': (470, 960), 'w': 208, 'h': 250,
     'props': HM(200, 200, 10, -1.0, 1.0, fmt='none')},
    {'key': 'c4', 'comment': True, 'text': 'x inside the disc, y outside it',
     'pos': (470, 970)},
] + see_also(1240, ['np.argwhere', 'greater', 'np.clip', 't.index_select', 'np.[]'])
links = [('met', '', 'gr', ''), ('gr', 'grid points', 'tr', ''),
         ('tr', 'array', 'sx', 'tensor in', 1), ('tr', 'array', 'sy', 'tensor in', 1),
         ('sx', 'output', 'hx', 'y'), ('sy', 'output', 'hy', 'y'),
         ('sx', 'output', 'wh', 'if true'), ('sy', 'output', 'wh', 'if false'),
         ('tr', 'array', 'nm', 'input', 0), ('nm', 'norm', 'lt', 'in'),
         ('lt', 'result', 'hm', 'y'), ('lt', 'result', 'wh', 'condition'),
         ('wh', 'selected', 'hr', 'y')]
print(build('np.where', 'np.where - choose between two arrays by a mask', body, demo, links,
            demo_width=720, text_width=760, text_height=1000))

# ------------------------------------------------------------------ np.argwhere
body = """Lists the positions of every true (non-zero) cell in an array.

Where np.where uses a mask to build a new array, np.argwhere tells you
WHERE the mask is set: which joints are over the speed limit, which
frames are glitched, which pixels are above threshold. Each found cell
comes out as its coordinates.

SYNTAX:
np.argwhere
It takes no arguments.

INPUTS:

condition:
Any array. A cell counts as found when it is true, or non-zero for
numbers - so a comparison's result is the usual input, but a plain
numeric array works too. Sends on arrival.

(np.argwhere is built by the same code as np.where but has only this one
inlet: it does not take 'if true' or 'if false'.)

OUTPUTS:

indices:
An integer array with one ROW per found cell and one column per dimension
of the input: for a 2-D mask each row is (row, column). The rows come in
reading order - along the first row, then the next. Its shape is (number
found, number of dimensions); when nothing is found it is empty, shape
(0, 2) for a 2-D mask.

Feed one of those rows to np.[] (or split the columns with np.[] :,0 and
:,1) to go from where to what.

READING THE DEMO:
np.rand makes an 8 x 8 array of noise; > 0.85 keeps roughly one cell in
seven. The heat_map shows the mask in two tones - white for found, black
for not. np.argwhere lists the found cells as [row, column] pairs, and
np.shape counts them. Click np.rand for a fresh scatter, and check a pair
against the picture: row 0 is the top, column 0 the left. Drag 0.85 on
the > node to change how many are found; it applies from the next click.

RELATED:
np.where uses the same mask to choose values. The comparison nodes (>,
<, ...) make the mask. t.argwhere is the PyTorch version (on the t.any
page). np.[] takes values out once you know where they are."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 'np.rand 8 8', 'pos': (30, 130), 'w': 160, 'h': 160,
     'props': {'min': 0.0, 'max': 1.0, 'dim 0': 8, 'dim 1': 8, 'dtype': 'float32'}},
    {'key': 'c0', 'comment': True, 'text': 'click np.rand for a fresh scatter',
     'pos': (30, 140)},
    {'key': 'gt', 'init': '> 0.85', 'pos': (30, 310), 'w': 130, 'h': 70,
     'props': {'': 0.85, 'output_type': 'bool'}},
    {'key': 'c1', 'comment': True, 'text': 'true where the noise is above 0.85',
     'pos': (30, 320)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 400), 'w': 268, 'h': 300,
     'props': HM(240, 240, 8, 0.0, 1.0, fmt='none', color='hot')},
    {'key': 'c2', 'comment': True, 'text': 'the mask: white cells are found',
     'pos': (30, 410)},
    {'key': 'aw', 'init': 'np.argwhere', 'pos': (330, 400), 'w': 130, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (330, 490), 'w': 300, 'h': 42, 'props': LIST},
    {'key': 'c3', 'comment': True, 'text': '[row, column] of each white cell',
     'pos': (330, 495)},
    {'key': 'shp', 'init': 'np.shape', 'pos': (330, 560), 'w': 110, 'h': 70},
    {'key': 'l2', 'init': 'list', 'pos': (330, 650), 'w': 150, 'h': 42, 'props': LIST},
    {'key': 'c4', 'comment': True, 'text': '(how many found, 2)', 'pos': (330, 655)},
] + see_also(740, ['np.where', 'greater', 't.any', 'np.[]'])
links = [('lb', 'out', 'rnd', ''), ('rnd', 'random array', 'gt', 'in'),
         ('gt', 'result', 'hm', 'y'), ('gt', 'result', 'aw', 'condition'),
         ('aw', 'indices', 'l1', ''), ('aw', 'indices', 'shp', 'np in'),
         ('shp', 'shape', 'l2', '')]
print(build('np.argwhere', 'np.argwhere - where the true cells are', body, demo, links,
            demo_width=660, text_width=760, text_height=860))
