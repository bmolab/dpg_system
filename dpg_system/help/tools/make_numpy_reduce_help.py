"""np reductions: np.argmax, np.argmin -- one page each. (The np.sum,
np.nansum, np.any, np.all, np.amax and np.amin pages are hand-made.)

These were part of one family page (np_stats). Each now has its own page whose
demo shows the array as a heat_map and the reduction beside it: the whole
array reduced to one number, and the same array reduced along each axis, laid
out so every result sits next to the row or column it came from.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import INT, FLT


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


def sub_comments(sub):
    """Inside a subpatch build() takes comments as 'comment <text>' nodes."""
    for d in sub['demo']:
        if d.pop('comment', False):
            text = d.pop('text')
            d['init'] = 'comment ' + text
            d['w'] = int(len(text) * 7.7) + 20
            d['h'] = 30
    return sub


# ------------------------------------------------------------ shared prose
AXIS_TEXT = """THE axis MENU:
None reduces the WHOLE array to one answer. 0 reduces down each COLUMN
(across the rows), giving one answer per column. 1 reduces along each
ROW (across the columns), giving one answer per row. In general the
axis you name is the one that disappears: an 8 by 10 array reduced
along axis 0 gives 10 answers, along axis 1 gives 8. The answers come
out as a flat 1-D array - the reduced axis is removed, not kept as a
length-1 axis (there is no keepdims option)."""

AXIS_PARAM = """axis (menu, also an inlet):
None, or one of the input's axes. A number typed after the node name
sets it (np.{op} 1); with no number it starts at None. The menu lists
None, 0 and 1 at first, and once an array arrives it lists None and
exactly that array's axes (0, 1, 2 for a 3-D array). If the chosen axis
is past the end of a smaller array, that array is reduced along its last
axis, but the choice is kept for the arrays that follow. Changing the
axis does NOT send by itself; the next array uses it. Pick None in the
menu, or send the message None into the axis inlet, to reduce the whole
array again."""


def th(body):
    """Prose box height: about 14 px a line at text size 30, plus a margin."""
    return (body.count('\n') + 1) * 14 + 40


def wrong_input(op):
    return """A list is turned into an array. A single number is sent back as
itself while axis is None (with an axis set, it sends nothing). An
empty array is an error and nothing is sent."""


# ------------------------------------------------------- the moving spot
def spot_sub(dark):
    """make_spot / make_dark_spot: 8 x 10 of dim (or bright) noise with one
    cell forced to 1 (or 0), at the row and column fed into it.

    Two one-hot rows (np.linspace == position) are crossed with np.outer
    into an 8 x 10 array that is 1 at the spot and 0 elsewhere; adding it to
    noise in 0..0.6 and clipping to 1 makes the spot exactly 1 and the
    largest value. The dark version subtracts it from noise in 0.4..1, so
    the spot is exactly 0 and the smallest. A metro re-sends ten times a
    second so the picture follows the sliders."""
    lo, hi = (0.4, 1.0) if dark else (0.0, 0.6)
    name = 'make_dark_spot' if dark else 'make_spot'
    combine = '!-' if dark else '+'
    return sub_comments({
        'name': name,
        'host': 'spot',
        'demo': [
            {'key': 'lb', 'init': 'load_bang', 'pos': (40, 40), 'w': 88, 'h': 46},
            {'key': 'rnd', 'init': 'np.rand 8 10', 'pos': (40, 110), 'w': 170, 'h': 160,
             'props': {'min': lo, 'max': hi, 'dim 0': 8, 'dim 1': 10, 'dtype': 'float32'}},
            {'key': 'cn', 'comment': True,
             'text': 'noise %.1f to %.1f' % (lo, hi), 'pos': (40, 290)},
            {'key': 'met', 'init': 'metro 100', 'pos': (260, 40), 'w': 129, 'h': 70,
             'props': METRO},
            {'key': 'inr', 'init': 'in row', 'pos': (480, 40), 'w': 150, 'h': 80},
            {'key': 'inc', 'init': 'in column', 'pos': (660, 40), 'w': 150, 'h': 80},
            {'key': 'lr', 'init': 'np.linspace 0 7 8', 'pos': (260, 140), 'w': 190, 'h': 150,
             'props': {'start': 0.0, 'stop': 7.0, 'steps': 8, 'dtype': 'float32'}},
            {'key': 'lc', 'init': 'np.linspace 0 9 10', 'pos': (480, 140), 'w': 190, 'h': 150,
             'props': {'start': 0.0, 'stop': 9.0, 'steps': 10, 'dtype': 'float32'}},
            {'key': 'er', 'init': '== 5', 'pos': (260, 320), 'w': 150, 'h': 70,
             'props': {'output_type': 'float'}},
            {'key': 'ec', 'init': '== 6', 'pos': (480, 320), 'w': 150, 'h': 70,
             'props': {'output_type': 'float'}},
            {'key': 'ce', 'comment': True,
             'text': 'a 1 at the chosen row, and at the chosen column', 'pos': (480, 400)},
            {'key': 'op', 'init': 'np.outer', 'pos': (260, 430), 'w': 150, 'h': 70},
            {'key': 'co', 'comment': True,
             'text': '8 x 10, 1 at the spot, 0 everywhere else', 'pos': (430, 440)},
            {'key': 'ad', 'init': combine, 'pos': (260, 520), 'w': 150, 'h': 70},
            {'key': 'cl', 'init': 'np.clip 0 1', 'pos': (260, 610), 'w': 150, 'h': 100,
             'props': {'min': 0.0, 'max': 1.0}},
            {'key': 'cc', 'comment': True,
             'text': ('noise minus the spot, clipped: the spot is exactly 0' if dark else
                      'noise plus the spot, clipped: the spot is exactly 1'),
             'pos': (430, 620)},
            {'key': 'out', 'init': 'out spot', 'pos': (260, 730), 'w': 150, 'h': 80},
        ],
        'links': [('lb', 'out', 'rnd', ''), ('rnd', 'random array', 'ad', 'operand'),
                  ('met', '', 'lc', ''), ('met', '', 'lr', ''),
                  ('inr', 'row', 'er', '', 0, 1), ('inc', 'column', 'ec', '', 0, 1),
                  ('lr', 'linspace out', 'er', 'in'), ('lc', 'linspace out', 'ec', 'in'),
                  ('ec', 'result', 'op', 'in 2'), ('er', 'result', 'op', 'in 1'),
                  ('op', 'outer_product', 'ad', 'in'), ('ad', 'result', 'cl', 'input'),
                  ('cl', 'out array', 'out', 'spot')],
    })


def spot_source():
    """The sliders that place the spot, the load_bang that starts it at
    row 5, column 6, and the make_spot host; the picture under them."""
    return [
        {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
        {'key': 'tt', 'init': 't 5 6', 'pos': (150, 62), 'w': 70, 'h': 46},
        {'key': 'ct', 'comment': True, 'text': 'starts the spot at row 5, column 6',
         'pos': (240, 67)},
        {'key': 'slr', 'init': 'slider 7', 'pos': (30, 130), 'w': 200, 'h': 42,
         'props': {'min': 0, 'max': 7, 'format': '%d', 'width': 180}},
        {'key': 'slc', 'init': 'slider 9', 'pos': (260, 130), 'w': 200, 'h': 42,
         'props': {'min': 0, 'max': 9, 'format': '%d', 'width': 180}},
        {'key': 'cs', 'comment': True, 'text': 'row (0-7) and column (0-9) of the spot',
         'pos': (30, 180)},
        {'key': 'spot', 'init': 'p make_spot', 'pos': (30, 220), 'w': 220, 'h': 70},
        {'key': 'hm', 'init': 'heat_map', 'pos': (30, 320), 'w': 460, 'h': 290,
         'props': HM(440, 240, 8)},
        {'key': 'chm', 'comment': True,
         'text': 'the array: 8 rows by 10 columns, row 0 at the top', 'pos': (30, 615)},
    ]


SPOT_LINKS = [('lb', 'out', 'tt', ''), ('tt', '', 'slr', '', 0), ('tt', '', 'slc', '', 1),
              ('slr', 'int out', 'spot', 'row'), ('slc', 'int out', 'spot', 'column'),
              ('spot', 'spot', 'hm', 'y')]


def index_page(op):
    """Demo for np.argmax / np.argmin: whole array -> flat index, split into
    row and column with // and %; axis 1 and axis 0 -> lists of indices."""
    demo = spot_source() + [
        {'key': 'r1', 'init': 'np.%s 1' % op, 'pos': (530, 130), 'w': 150, 'h': 70,
         'props': {'axis': '1'}},
        {'key': 'c1', 'comment': True, 'text': 'axis 1: for each row, which column',
         'pos': (530, 135)},
        {'key': 'l1', 'init': 'list', 'pos': (530, 220), 'w': 260, 'h': 42, 'props': LIST},
        {'key': 'r', 'init': 'np.%s' % op, 'pos': (530, 330), 'w': 150, 'h': 70,
         'props': {'axis': 'None'}},
        {'key': 'cr', 'comment': True, 'text': 'axis None: one index into the whole array',
         'pos': (530, 335)},
        {'key': 'i0', 'init': 'int', 'pos': (530, 420), 'w': 127, 'h': 42, 'props': INT},
        {'key': 'ci', 'comment': True, 'text': 'counting cells row by row from 0',
         'pos': (530, 425)},
        {'key': 'fd', 'init': '// 10', 'pos': (530, 490), 'w': 110, 'h': 60},
        {'key': 'md', 'init': '% 10', 'pos': (680, 490), 'w': 110, 'h': 60},
        {'key': 'ir', 'init': 'int', 'pos': (530, 570), 'w': 127, 'h': 42, 'props': INT},
        {'key': 'ic', 'init': 'int', 'pos': (680, 570), 'w': 127, 'h': 42, 'props': INT},
        {'key': 'crc', 'comment': True, 'text': 'row = index // 10, column = index % 10',
         'pos': (530, 620)},
        {'key': 'r0', 'init': 'np.%s 0' % op, 'pos': (30, 650), 'w': 150, 'h': 70,
         'props': {'axis': '0'}},
        {'key': 'c0', 'comment': True, 'text': 'axis 0: for each column, which row',
         'pos': (30, 655)},
        {'key': 'l0', 'init': 'list', 'pos': (30, 740), 'w': 320, 'h': 42, 'props': LIST},
    ]
    links = SPOT_LINKS + [('spot', 'spot', 'r1', 'in'), ('r1', op, 'l1', ''),
                          ('spot', 'spot', 'r', 'in'), ('r', op, 'i0', ''),
                          ('r', op, 'fd', 'in'), ('r', op, 'md', 'in'),
                          ('fd', 'result', 'ir', ''), ('md', 'result', 'ic', ''),
                          ('spot', 'spot', 'r0', 'in'), ('r0', op, 'l0', '')]
    return demo, links


# np.amax: its page is hand-made now (see tools/README.md).

# np.amin: its page is hand-made now (see tools/README.md).

# -------------------------------------------------------------------- np.argmax
ARG_INDEX = """THE INDEX WITH NO AXIS:
With axis None the answer is ONE whole number counting the cells of the
whole array in reading order - along row 0, then row 1, and so on - not
a row and a column. In an array 10 columns wide, index 56 is row 5,
column 6: the row is the index divided by the width (whole-number
division, // 10) and the column is the remainder (% 10).
With an axis, each answer is a position along THAT axis: axis 0 gives,
for each column, the row it was found in; axis 1 gives, for each row,
the column."""

body = """Gives the POSITION of the largest value in an array - its index, not
the value itself.

Use it to pick a winner: which sensor reads highest, which frequency
band is loudest, which class a set of scores votes for, where the
brightest pixel is. To get the value itself, use np.amax. If several
cells share the largest value, the first one (in reading order, or the
lowest along the axis) is the one given.

""" + ARG_INDEX + """

""" + AXIS_TEXT + """

SYNTAX:
np.argmax <axis: int>
Default: no axis - one index into the whole array.

EXAMPLE:
np.argmax 1

INPUTS and PARAMETERS:

in:
The array. Every array that arrives is searched and the answer sent;
this is the only inlet that makes the node send.
""" + wrong_input('argmax') + """
A NaN counts as larger than everything: the index of the first NaN is
given.

""" + AXIS_PARAM.replace('{op}', 'argmax') + """

OUTPUTS:

argmax:
Whole numbers (int64). With axis None, one index; with an axis, an
array of indices, one per row or column.

READING THE DEMO:
make_spot draws 8 by 10 of dim noise with one bright cell (exactly 1.0)
where the sliders put it - it starts at row 5, column 6. np.argmax with
axis None answers 56; the two int boxes under it split that into row
5 and column 6. Move the sliders and the index follows the spot.
np.argmax 1 asks each ROW where its largest value is: 8 column numbers,
and the spot's row (the sixth entry) says 6. The others point at the
brightest noise in their row. np.argmax 0 asks each COLUMN: 10 row
numbers, and the spot's column (the seventh entry) says 5.

RELATED:
np.amax gives the largest value itself. np.argmin finds the smallest.
np.argsort ranks every value, not just the top one. np.argwhere lists
every position where a condition holds. t.argmax is the PyTorch
version."""

demo, links = index_page('argmax')
demo += see_also(820, ['np.amax', 'np.argmin', 'np.argsort', 'np.argwhere', 't.argmax'])
print(build('np.argmax', 'np.argmax - where the largest value is',
            body, demo, links, demo_width=830, text_width=760, text_height=th(body),
            subpatch=spot_sub(dark=False)))

# -------------------------------------------------------------------- np.argmin
body = """Gives the POSITION of the smallest value in an array - its index, not
the value itself.

Use it to find the nearest of a set of points (the smallest distance),
the quietest channel, the lowest point of a curve, or the best match
when smaller scores are better. To get the value itself, use np.amin.
If several cells share the smallest value, the first one (in reading
order, or the lowest along the axis) is the one given.

""" + ARG_INDEX + """

""" + AXIS_TEXT + """

SYNTAX:
np.argmin <axis: int>
Default: no axis - one index into the whole array.

EXAMPLE:
np.argmin 0

INPUTS and PARAMETERS:

in:
The array. Every array that arrives is searched and the answer sent;
this is the only inlet that makes the node send.
""" + wrong_input('argmin') + """
A NaN counts as smaller than everything: the index of the first NaN is
given.

""" + AXIS_PARAM.replace('{op}', 'argmin') + """

OUTPUTS:

argmin:
Whole numbers (int64). With axis None, one index; with an axis, an
array of indices, one per row or column.

READING THE DEMO:
make_dark_spot draws 8 by 10 of light noise with one dark cell
(exactly 0) where the sliders put it - it starts at row 5, column 6.
np.argmin with axis None answers 56; the int boxes under it split that
into row 5 and column 6. Move the sliders and the index follows the
spot. np.argmin 1 asks each ROW where its smallest value is: 8 column
numbers, and the spot's row (the sixth entry) says 6. np.argmin 0 asks
each COLUMN: 10 row numbers, and the spot's column (the seventh entry)
says 5.

RELATED:
np.amin gives the smallest value itself. np.argmax finds the largest.
np.argsort ranks every value. np.argwhere lists every position where a
condition holds. There is no t.argmin; t.min (on the t.max page) gives
the PyTorch minimum."""

demo, links = index_page('argmin')
demo[[d['key'] for d in demo].index('spot')]['init'] = 'p make_dark_spot'
demo += see_also(820, ['np.amin', 'np.argmax', 'np.argsort', 'np.argwhere'])
print(build('np.argmin', 'np.argmin - where the smallest value is',
            body, demo, links, demo_width=830, text_width=760, text_height=th(body),
            subpatch=spot_sub(dark=True)))

# np.sum and np.nansum: their pages are hand-made now (see tools/README.md).


# np.any and np.all: their pages are hand-made now (see tools/README.md).
