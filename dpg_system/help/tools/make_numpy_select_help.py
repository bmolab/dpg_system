"""np selecting: np.where, np.argwhere -- one page each. (The np.[] and
np.crop pages are hand-made.)

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


# np.[]: its page is hand-made now (see tools/README.md).

# np.crop: its page is hand-made now (see tools/README.md).

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
