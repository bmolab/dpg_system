"""np selecting: np.argwhere. (The np.[], np.crop and np.where pages are
hand-made.)

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

# np.where: its page is hand-made now (see tools/README.md).

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
