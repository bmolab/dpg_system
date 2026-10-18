"""np reductions: np.amax, np.amin, np.argmax, np.argmin, np.sum, np.nansum,
np.any, np.all -- one page each.

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


def value_page(op, dark):
    """Demo for np.amax / np.amin: whole array -> float; axis 1 -> one per
    row, turned into a column beside the picture; axis 0 -> one per column,
    a strip under the picture."""
    demo = spot_source() + [
        {'key': 'r1', 'init': 'np.%s 1' % op, 'pos': (530, 130), 'w': 150, 'h': 70,
         'props': {'axis': '1'}},
        {'key': 'c1', 'comment': True, 'text': 'axis 1: one answer per row',
         'pos': (530, 135)},
        {'key': 'ex', 'init': 'np.expand_dims 1', 'pos': (530, 220), 'w': 170, 'h': 70,
         'props': {'axis': 1}},
        {'key': 'cx', 'comment': True, 'text': 'only stands the 8 answers up as a column',
         'pos': (530, 225)},
        {'key': 'hcol', 'init': 'heat_map', 'pos': (530, 320), 'w': 90, 'h': 290,
         'props': HM(60, 240, 8)},
        {'key': 'r', 'init': 'np.%s' % op, 'pos': (660, 320), 'w': 150, 'h': 70,
         'props': {'axis': 'None'}},
        {'key': 'cr', 'comment': True, 'text': 'axis None: the whole array',
         'pos': (660, 325)},
        {'key': 'f', 'init': 'float', 'pos': (660, 410), 'w': 127, 'h': 42, 'props': FLT},
        {'key': 'r0', 'init': 'np.%s 0' % op, 'pos': (30, 650), 'w': 150, 'h': 70,
         'props': {'axis': '0'}},
        {'key': 'c0', 'comment': True, 'text': 'axis 0: one answer per column',
         'pos': (30, 655)},
        {'key': 'hrow', 'init': 'heat_map', 'pos': (30, 740), 'w': 460, 'h': 110,
         'props': HM(440, 60, 1)},
    ]
    links = SPOT_LINKS + [('spot', 'spot', 'r1', 'in'), ('r1', op, 'ex', 'input'),
                          ('ex', 'array out', 'hcol', 'y'),
                          ('spot', 'spot', 'r', 'in'), ('r', op, 'f', ''),
                          ('spot', 'spot', 'r0', 'in'), ('r0', op, 'hrow', 'y')]
    return demo, links


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


# ---------------------------------------------------------------------- np.amax
body = """Finds the largest value in an array - in the whole array, or in each
row or each column.

Use it for the peak of a signal, the loudest band of a spectrum, the
highest reading among a set of sensors, or the brightest pixel of an
image. To know WHERE the largest value is rather than what it is, use
np.argmax.

NOT THE SAME AS np.max:
np.amax REDUCES - an array goes in and one value (or one per row or
column) comes out. np.max, a different node, works element by element:
it compares every value with a fixed number and keeps the larger, so
an array goes in and an array of the same shape comes out (a floor).
np.max is the clamp; np.amax is the question "what is the largest?".

""" + AXIS_TEXT + """

SYNTAX:
np.amax <axis: int>
Default: no axis - the whole array.

EXAMPLE:
np.amax 1

INPUTS and PARAMETERS:

in:
The array. Every array that arrives is reduced and the answer sent;
this is the only inlet that makes the node send.
""" + wrong_input('amax') + """
If the array holds a NaN (not-a-number) the answer is NaN - there is
no nan-skipping version of this node.

""" + AXIS_PARAM.replace('{op}', 'amax') + """

OUTPUTS:

amax:
With axis None, one number (the largest value). With an axis, an array
holding the largest value of each row or column, as described above.
It has the type of the input (float32 in, float32 out).

READING THE DEMO:
make_spot draws 8 by 10 of dim noise (0 to 0.6) with one bright cell,
exactly 1.0, where the two sliders put it. np.amax with axis None
always answers 1.0: the spot is the largest value wherever it is.
np.amax 1 asks each ROW: its 8 answers stand beside the picture (the
expand_dims node only turns the flat row of 8 into a column so each
answer sits next to its row). The spot's row reads 1.0, the others
their own brightest noise. np.amax 0 asks each COLUMN: the strip under
the picture has 10 answers, one under each column, and the spot's
column reads 1.0. Move a slider and the 1.0 moves with the spot.

RELATED:
np.amin finds the smallest. np.argmax gives the position of the
largest. np.max is the element-wise floor described above, not a
reduction. np.mean, np.median and np.sum are the other common
reductions. t.max is the PyTorch version."""

demo, links = value_page('amax', dark=False)
demo += see_also(890, ['np.amin', 'np.argmax', 'np.max', 'np.mean', 't.max'])
print(build('np.amax', 'np.amax - the largest value',
            body, demo, links, demo_width=830, text_width=760, text_height=th(body),
            subpatch=spot_sub(dark=False)))

# ---------------------------------------------------------------------- np.amin
body = """Finds the smallest value in an array - in the whole array, or in
each row or each column.

Use it for the trough of a signal, the quietest channel, the closest of
a set of distances, or the darkest pixel of an image. To know WHERE the
smallest value is rather than what it is, use np.argmin.

NOT THE SAME AS np.min:
np.amin REDUCES - an array goes in and one value (or one per row or
column) comes out. np.min, a different node, works element by element:
it compares every value with a fixed number and keeps the smaller, so
an array goes in and an array of the same shape comes out (a ceiling).
np.min is the clamp; np.amin is the question "what is the smallest?".

""" + AXIS_TEXT + """

SYNTAX:
np.amin <axis: int>
Default: no axis - the whole array.

EXAMPLE:
np.amin 0

INPUTS and PARAMETERS:

in:
The array. Every array that arrives is reduced and the answer sent;
this is the only inlet that makes the node send.
""" + wrong_input('amin') + """
If the array holds a NaN (not-a-number) the answer is NaN - there is
no nan-skipping version of this node.

""" + AXIS_PARAM.replace('{op}', 'amin') + """

OUTPUTS:

amin:
With axis None, one number (the smallest value). With an axis, an
array holding the smallest value of each row or column. It has the
type of the input.

READING THE DEMO:
make_dark_spot draws 8 by 10 of light noise (0.4 to 1) with one dark
cell, exactly 0, where the two sliders put it. np.amin with axis None
always answers 0: the spot is the smallest value wherever it is.
np.amin 1 asks each ROW: its 8 answers stand beside the picture (the
expand_dims node only turns them into a column so each sits next to
its row), and the spot's row reads 0. np.amin 0 asks each COLUMN: the
strip under the picture has 10 answers, one under each column, and the
spot's column reads 0. Move a slider and the 0 moves with the spot.

RELATED:
np.amax finds the largest. np.argmin gives the position of the
smallest. np.min is the element-wise ceiling described above, not a
reduction. np.mean and np.median summarise the middle instead of the
extremes. t.min is the PyTorch version."""

demo, links = value_page('amin', dark=True)
demo[[d['key'] for d in demo].index('spot')]['init'] = 'p make_dark_spot'
demo += see_also(890, ['np.amax', 'np.argmin', 'np.min', 'np.median', 't.min'])
print(build('np.amin', 'np.amin - the smallest value',
            body, demo, links, demo_width=830, text_width=760, text_height=th(body),
            subpatch=spot_sub(dark=True)))

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

# ----------------------------------------------------------------------- np.sum
def counts_source():
    """6 x 8 whole numbers 0 to 3 from np.rand (its own button makes more),
    shown with the numbers in the cells so the sums can be checked."""
    return [
        {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
        {'key': 'rnd', 'init': 'np.rand 6 8', 'pos': (30, 130), 'w': 170, 'h': 160,
         'props': {'min': 0.0, 'max': 3.0, 'dim 0': 6, 'dim 1': 8, 'dtype': 'int64'}},
        {'key': 'crnd', 'comment': True,
         'text': 'whole numbers 0 to 3 - click its button for new ones', 'pos': (30, 135)},
    ]


body = """Adds up the values in an array - all of them, or each row or each
column separately.

Use it for a total (the energy in a frame, the number of active cells in
a mask - True counts as 1), a row or column total (the activity of each
joint over a window, the profile of an image projected onto one axis),
or as the first step of an average you want to weight yourself.

""" + AXIS_TEXT + """

SYNTAX:
np.sum <axis: int>
Default: no axis - the whole array.

EXAMPLE:
np.sum 1

INPUTS and PARAMETERS:

in:
The array. Every array that arrives is added up and the total sent;
this is the only inlet that makes the node send. A list is turned into
an array; a single number is sent back as itself while axis is None.
An empty array sums to 0. If the array holds a NaN (not-a-number), the total it is part of
is NaN - use np.nansum to skip them.

""" + AXIS_PARAM.replace('{op}', 'sum') + """

OUTPUTS:

sum:
With axis None, one number; with an axis, an array of totals, one per
row or column. Whole numbers sum to whole numbers (int64), floats to
floats of the same type; True and False sum to a count.

READING THE DEMO:
np.rand makes 6 rows by 8 columns of whole numbers 0 to 3, written in
the cells; click its button for new ones and every total follows.
np.sum 1 adds along each ROW: the 6 totals stand beside the picture
(the expand_dims node only turns them into a column so each sits next
to its row). np.sum 0 adds down each COLUMN: 8 totals in the strip
underneath, one under each column. np.sum with axis None adds all 48
cells - the same number as adding up either set of totals.

RELATED:
np.nansum skips NaN values. np.mean is the sum divided by the count.
np.cumsum keeps the running total at every step instead of only the
end. np.any tells you whether anything is non-zero at all. t.sum (on
the t.mean page) is the PyTorch version."""

demo = counts_source() + [
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 320), 'w': 380, 'h': 230,
     'props': HM(352, 180, 6, 0.0, 3.0, fmt='%.0f')},
    {'key': 'chm', 'comment': True, 'text': 'the array: 6 rows by 8 columns',
     'pos': (30, 555)},
    {'key': 's1', 'init': 'np.sum 1', 'pos': (440, 130), 'w': 150, 'h': 70,
     'props': {'axis': '1'}},
    {'key': 'c1', 'comment': True, 'text': 'axis 1: one total per row', 'pos': (440, 135)},
    {'key': 'ex', 'init': 'np.expand_dims 1', 'pos': (440, 220), 'w': 170, 'h': 70,
     'props': {'axis': 1}},
    {'key': 'cx', 'comment': True, 'text': 'only stands the 6 totals up as a column',
     'pos': (440, 225)},
    {'key': 'hcol', 'init': 'heat_map', 'pos': (440, 320), 'w': 90, 'h': 230,
     'props': HM(60, 180, 6, 0.0, 24.0, fmt='%.0f')},
    {'key': 's', 'init': 'np.sum', 'pos': (620, 320), 'w': 150, 'h': 70,
     'props': {'axis': 'None'}},
    {'key': 'cs', 'comment': True, 'text': 'axis None: all 48 cells', 'pos': (620, 325)},
    {'key': 'i', 'init': 'int', 'pos': (620, 410), 'w': 127, 'h': 42, 'props': INT},
    {'key': 's0', 'init': 'np.sum 0', 'pos': (30, 590), 'w': 150, 'h': 70,
     'props': {'axis': '0'}},
    {'key': 'c0', 'comment': True, 'text': 'axis 0: one total per column', 'pos': (30, 595)},
    {'key': 'hrow', 'init': 'heat_map', 'pos': (30, 680), 'w': 380, 'h': 100,
     'props': HM(352, 50, 1, 0.0, 18.0, fmt='%.0f')},
] + see_also(820, ['np.nansum', 'np.mean', 'np.cumsum', 'np.any', 't.mean t.sum'])
links = [('lb', 'out', 'rnd', ''), ('rnd', 'random array', 'hm', 'y'),
         ('rnd', 'random array', 's1', 'in'), ('s1', 'sum', 'ex', 'input'),
         ('ex', 'array out', 'hcol', 'y'),
         ('rnd', 'random array', 's', 'in'), ('s', 'sum', 'i', ''),
         ('rnd', 'random array', 's0', 'in'), ('s0', 'sum', 'hrow', 'y')]
print(build('np.sum', 'np.sum - add up an array',
            body, demo, links, demo_width=800, text_width=760, text_height=th(body)))

# -------------------------------------------------------------------- np.nansum
HOLE_SUB = sub_comments({
    'name': 'make_hole',
    'host': 'hole',
    'demo': [
        {'key': 'ina', 'init': 'in array', 'pos': (40, 40), 'w': 150, 'h': 80},
        {'key': 'inh', 'init': 'in hole', 'pos': (240, 40), 'w': 150, 'h': 80},
        {'key': 'lb', 'init': 'load_bang', 'pos': (440, 40), 'w': 88, 'h': 46},
        {'key': 'nan', 'init': 'message', 'pos': (440, 110), 'w': 100, 'h': 42,
         'props': {'text in': 'nan', 'font size': '24', 'width': 80}},
        {'key': 'cnan', 'comment': True, 'text': 'the NaN that fills the hole',
         'pos': (560, 115)},
        {'key': 'met', 'init': 'metro 100', 'pos': (40, 150), 'w': 129, 'h': 70,
         'props': METRO},
        {'key': 'zr', 'init': 'np.zeros 6 8', 'pos': (40, 240), 'w': 170, 'h': 110,
         'props': {'dim 0': 6, 'dim 1': 8, 'dtype': 'float32'}},
        {'key': 'ed', 'init': 'np.edit 2,3', 'pos': (40, 370), 'w': 170, 'h': 90,
         'props': {'indices': '2,3', 'values': '1'}},
        {'key': 'ced', 'comment': True, 'text': 'a 1 at row 2, column 3: where the hole goes',
         'pos': (230, 375)},
        {'key': 'ml', 'init': '*', 'pos': (40, 480), 'w': 130, 'h': 70},
        {'key': 'cml', 'comment': True, 'text': 'times the toggle: no hole when it is off',
         'pos': (230, 485)},
        {'key': 'wh', 'init': 'np.where', 'pos': (40, 570), 'w': 150, 'h': 100},
        {'key': 'cwh', 'comment': True, 'text': 'NaN at the hole, the array everywhere else',
         'pos': (230, 575)},
        {'key': 'out', 'init': 'out holed', 'pos': (40, 690), 'w': 160, 'h': 80},
    ],
    'links': [('lb', 'out', 'nan', ''), ('nan', 'message out', 'wh', 'if true'),
              ('ina', 'array', 'wh', 'if false'), ('inh', 'hole', 'ml', 'operand'),
              ('met', '', 'zr', ''), ('zr', 'array of zeros', 'ed', 'array in'),
              ('ed', 'output', 'ml', 'in'), ('ml', 'result', 'wh', 'condition'),
              ('wh', 'selected', 'out', 'holed')],
})

body = """Adds up the values in an array as np.sum does, but skips any NaN
(not-a-number) values instead of letting them spoil the total.

Real data has holes in it - a dropped frame, a sensor that lost
tracking, a division by zero upstream - and those often arrive as NaN.
NaN plus anything is NaN, so ONE missing value turns an np.sum total
into NaN. np.nansum treats each NaN as 0 and adds up what is actually
there. Reach for it whenever an array might contain gaps you would
rather ignore than propagate.

WHAT IT GIVES FOR A ROW OF NOTHING BUT NaN:
0 - an empty total, not NaN. If you need to know that a row had no data
at all, count the NaNs separately.

""" + AXIS_TEXT + """

SYNTAX:
np.nansum <axis: int>
Default: no axis - the whole array.

EXAMPLE:
np.nansum 1

INPUTS and PARAMETERS:

in:
The array. Every array that arrives is added up and the total sent;
this is the only inlet that makes the node send. A list is turned into
an array. An empty array sums to 0. With no NaN in it, the answer is
exactly what np.sum gives.

""" + AXIS_PARAM.replace('{op}', 'nansum') + """

OUTPUTS:

nansum:
With axis None, one number; with an axis, an array of totals, one per
row or column, never NaN.

READING THE DEMO:
np.rand makes 6 by 8 whole numbers 0 to 3; click its button for new
ones. make_hole passes them through untouched until you click the
toggle: then the cell at row 2, column 3 becomes NaN (the heat_map
shows its number as nan). With the toggle off, np.sum and np.nansum
agree everywhere. With it on, np.sum's whole total turns to nan, and so
does its total for row 2 in the left column - every other row is
unaffected. np.nansum's totals stay numbers: row 2's total simply
leaves the hole out.

RELATED:
np.sum is the plain version. np.nanmean and np.nanmedian skip NaNs the
same way for the average and the middle value. np.mean and np.median
do not. np.where can replace NaNs (or anything else) with a chosen
value before an ordinary reduction."""

demo = counts_source() + [
    {'key': 'tg', 'init': 'toggle', 'pos': (240, 200), 'w': 60, 'h': 46},
    {'key': 'ctg', 'comment': True, 'text': 'click: put a NaN at row 2, column 3',
     'pos': (310, 205)},
    {'key': 'hole', 'init': 'p make_hole', 'pos': (30, 320), 'w': 220, 'h': 90},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 440), 'w': 380, 'h': 230,
     'props': HM(352, 180, 6, 0.0, 3.0, fmt='%.0f')},
    {'key': 'chm', 'comment': True, 'text': 'the array, 6 x 8, with or without the hole',
     'pos': (30, 675)},
    {'key': 'su1', 'init': 'np.sum 1', 'pos': (440, 260), 'w': 150, 'h': 70,
     'props': {'axis': '1'}},
    {'key': 'exs', 'init': 'np.expand_dims 1', 'pos': (440, 350), 'w': 170, 'h': 70,
     'props': {'axis': 1}},
    {'key': 'hcs', 'init': 'heat_map', 'pos': (440, 440), 'w': 90, 'h': 230,
     'props': HM(60, 180, 6, 0.0, 24.0, fmt='%.0f')},
    {'key': 'cs', 'comment': True, 'text': 'np.sum per row', 'pos': (440, 675)},
    {'key': 'nn1', 'init': 'np.nansum 1', 'pos': (630, 260), 'w': 150, 'h': 70,
     'props': {'axis': '1'}},
    {'key': 'exn', 'init': 'np.expand_dims 1', 'pos': (630, 350), 'w': 170, 'h': 70,
     'props': {'axis': 1}},
    {'key': 'hcn', 'init': 'heat_map', 'pos': (630, 440), 'w': 90, 'h': 230,
     'props': HM(60, 180, 6, 0.0, 24.0, fmt='%.0f')},
    {'key': 'cn', 'comment': True, 'text': 'np.nansum per row', 'pos': (630, 675)},
    {'key': 'su', 'init': 'np.sum', 'pos': (30, 720), 'w': 150, 'h': 70,
     'props': {'axis': 'None'}},
    {'key': 'fs', 'init': 'float', 'pos': (30, 810), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'cfs', 'comment': True, 'text': 'np.sum of everything', 'pos': (30, 860)},
    {'key': 'nn', 'init': 'np.nansum', 'pos': (260, 720), 'w': 150, 'h': 70,
     'props': {'axis': 'None'}},
    {'key': 'fn', 'init': 'float', 'pos': (260, 810), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'cfn', 'comment': True, 'text': 'np.nansum of everything', 'pos': (260, 860)},
] + see_also(910, ['np.sum', 'np.nanmean', 'np.nanmedian', 'np.where'])
links = [('lb', 'out', 'rnd', ''), ('rnd', 'random array', 'hole', 'array'),
         ('tg', '', 'hole', 'hole'), ('hole', 'holed', 'hm', 'y'),
         ('hole', 'holed', 'su1', 'in'), ('su1', 'sum', 'exs', 'input'),
         ('exs', 'array out', 'hcs', 'y'),
         ('hole', 'holed', 'nn1', 'in'), ('nn1', 'nansum', 'exn', 'input'),
         ('exn', 'array out', 'hcn', 'y'),
         ('hole', 'holed', 'su', 'in'), ('su', 'sum', 'fs', ''),
         ('hole', 'holed', 'nn', 'in'), ('nn', 'nansum', 'fn', '')]
print(build('np.nansum', 'np.nansum - add up, skipping NaN',
            body, demo, links, demo_width=820, text_width=760, text_height=th(body),
            subpatch=HOLE_SUB))


# -------------------------------------------------------------- np.any / np.all
def mask_demo(op):
    """np.rand 6 x 6 (0.2 to 0.8) held in a var that a metro re-sends; a
    slider sets the threshold of a '>' whose mask is drawn in two tones and
    fed to np.any / np.all."""
    demo = [
        {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
        {'key': 'rnd', 'init': 'np.rand 6 6', 'pos': (30, 130), 'w': 170, 'h': 160,
         'props': {'min': 0.2, 'max': 0.8, 'dim 0': 6, 'dim 1': 6, 'dtype': 'float32'}},
        {'key': 'crnd', 'comment': True,
         'text': 'values 0.2 to 0.8 - click its button for new ones', 'pos': (30, 135)},
        {'key': 'met', 'init': 'metro 100', 'pos': (300, 62), 'w': 129, 'h': 70,
         'props': METRO},
        {'key': 'var', 'init': 'var np_%s_help_array' % op, 'pos': (300, 170), 'w': 230,
         'h': 70, 'props': {'name': 'np_%s_help_array' % op}},
        {'key': 'cvar', 'comment': True, 'text': 'holds the array; the metro re-sends it',
         'pos': (300, 175)},
        {'key': 'hin', 'init': 'heat_map', 'pos': (30, 320), 'w': 280, 'h': 260,
         'props': HM(240, 210, 6)},
        {'key': 'chin', 'comment': True, 'text': 'the array', 'pos': (30, 585)},
        {'key': 'sl', 'init': 'slider', 'pos': (360, 270), 'w': 200, 'h': 42,
         'props': {'min': 0.0, 'max': 1.0, 'format': '%.3f', 'width': 180}},
        {'key': 'csl', 'comment': True, 'text': 'threshold: drag it', 'pos': (360, 275)},
        {'key': 'gt', 'init': '>', 'pos': (360, 330), 'w': 130, 'h': 70,
         'props': {'output_type': 'float'}},
        {'key': 'hmk', 'init': 'heat_map', 'pos': (360, 420), 'w': 280, 'h': 260,
         'props': HM(240, 210, 6, fmt='none', color='hot')},
        {'key': 'chmk', 'comment': True,
         'text': 'the mask: white where the value is above the threshold',
         'pos': (360, 685)},
        {'key': 'r', 'init': 'np.%s' % op, 'pos': (30, 620), 'w': 130, 'h': 70},
        {'key': 'out', 'init': 'toggle', 'pos': (30, 710), 'w': 60, 'h': 46},
    ]
    links = [('lb', 'out', 'rnd', ''), ('rnd', 'random array', 'var', 'in'),
             ('met', '', 'var', 'in'), ('var', 'out', 'hin', 'y'),
             ('var', 'out', 'gt', 'in'), ('sl', 'float out', 'gt', 'operand'),
             ('gt', 'result', 'hmk', 'y'), ('gt', 'result', 'r', 'in'),
             ('r', 'result', 'out', '')]
    return demo, links


ANYALL_IN = """in:
The array (a list is turned into one; a single number counts as a
one-element array). Every array that arrives is tested and the answer
sent - this is the only inlet. Any shape is accepted, and the WHOLE
array is always tested: there is no axis, so you get one True or False,
never one per row. Values are read as true or false the numpy way: 0,
0.0 and False are false, anything else - including negative numbers
and NaN - is true."""

body = """Answers True if ANY value in an array is true (non-zero), False only
if every one is zero.

Its usual job is the end of a test: compare an array with something,
then ask "did it happen anywhere?" - is any joint moving faster than a
limit, has any sensor dropped out, is any pixel above the threshold. It
turns a whole mask into one yes or no you can gate on.

""" + """SYNTAX:
np.any

EXAMPLE:
np.any

INPUTS and PARAMETERS:

""" + ANYALL_IN + """
An empty array gives False (there is nothing true in it).

OUTPUTS:

result:
True or False.

READING THE DEMO:
np.rand makes 6 by 6 values between 0.2 and 0.8. The > node compares
each with the slider and draws the answer as a mask: white where the
value is above the threshold, dark red where it is not. np.any looks at the
mask and lights the toggle if even ONE cell is white. Drag the slider
up: cells go dark one by one, and np.any stays True until the very last
one - the largest value - goes dark, just under 0.8. Click np.rand's
button for new values.

RELATED:
np.all is the opposite question: true only if EVERY value is true.
np.sum of a mask counts how many are true. np.amax > threshold is the
same test as np.any of (array > threshold). np.argwhere lists the
positions of the true cells. t.any is the PyTorch version."""

demo, links = mask_demo('any')
demo += [{'key': 'cr', 'comment': True, 'text': 'True if any cell of the mask is white',
          'pos': (30, 625)}]
demo += see_also(800, ['np.all', 'np.sum', 'np.amax', 'np.argwhere', 't.any'])
print(build('np.any', 'np.any - is anything true?',
            body, demo, links, demo_width=700, text_width=760, text_height=th(body)))

body = """Answers True only if EVERY value in an array is true (non-zero), and
False as soon as a single one is zero.

Use it to check that a condition holds everywhere: are all joints
inside their range, did every sensor report, is every value finite,
has every step finished. It turns a whole mask into one yes or no.

SYNTAX:
np.all

EXAMPLE:
np.all

INPUTS and PARAMETERS:

""" + ANYALL_IN + """
An empty array gives True (there is nothing false in it).

OUTPUTS:

result:
True or False.

READING THE DEMO:
np.rand makes 6 by 6 values between 0.2 and 0.8. The > node compares
each with the slider and draws the answer as a mask: white where the
value is above the threshold, dark red where it is not. With the slider at
0 every cell is white and np.all lights the toggle. Drag the slider
up: as soon as the FIRST cell goes dark - the smallest value, just
over 0.2 - np.all turns False, even though nearly all the mask is
still white. Click np.rand's button for new values.

RELATED:
np.any is the opposite question: true if ANY value is true. np.sum of a
mask counts how many are true; np.all is the same as that count
equalling the number of cells. np.amin > threshold is the same test as
np.all of (array > threshold). t.all is the PyTorch version."""

demo, links = mask_demo('all')
demo += [{'key': 'cr', 'comment': True, 'text': 'True only if every cell of the mask is white',
          'pos': (30, 625)}]
demo += see_also(800, ['np.any', 'np.sum', 'np.amin', 'np.argwhere', 't.all'])
print(build('np.all', 'np.all - is everything true?',
            body, demo, links, demo_width=700, text_width=760, text_height=th(body)))
