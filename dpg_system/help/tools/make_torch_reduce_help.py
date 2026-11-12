"""t reductions: t.mean, t.median, t.nanmean, t.nanmedian, t.sum, t.nansum,
t.prod, t.max, t.min, t.argmax, t.any, t.all, t.count_nonzero, t.argwhere
(t.non_zero is the same node as t.argwhere and is documented on its page).

These were three family pages (t.mean, t.max, t.any). Each node now has its
own page whose demo shows the tensor as a heat_map, the whole-tensor answer
beside it, and a second copy of the node with a dim argument whose dim is
driven live by a slider, its answers drawn as a heat_map strip.

The slider goes through 't bang int': the number reaches the dim inlet first,
then the bang makes a var re-send the held tensor, so the reduction runs again
with the new dim straight away (a dim change alone does not send).
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


def sub_comments(sub):
    """Inside a subpatch build() takes comments as 'comment <text>' nodes."""
    for d in sub['demo']:
        if d.pop('comment', False):
            text = d.pop('text')
            d['init'] = 'comment ' + text
            d['w'] = int(len(text) * 7.7) + 20
            d['h'] = 30
    return sub


def th(body):
    """Prose box height: about 14 px a line at text size 30, plus a margin."""
    return (body.count('\n') + 1) * 14 + 40


def rand_props(lo, hi, dtype='float32', shape='[6, 8]'):
    return {'shape': shape, 'min': float(lo), 'max': float(hi), 'dtype': dtype}


# ------------------------------------------------------------ shared prose
DIM_TEXT = """THE dim ARGUMENT:
With no argument the WHOLE tensor is reduced to one value. A number
typed after the name ({op} 0) chooses a dim to reduce along and adds a
dim inlet{kd} to the node - it is the argument that creates them,
so a node made without one cannot be given a dim later (retype it).
For a table of rows and columns, dim 0 reduces DOWN each column (one
answer per column) and dim 1 ALONG each row (one answer per row). The
dim named is the one that disappears: a 6 by 8 tensor gives 8 answers
on dim 0 and 6 on dim 1. Negative numbers count from the end: -1 is
the last dim."""

DIM_INLET = """dim (inlet, only when an argument was given):
A whole number. Changing it does NOT send by itself; the next tensor
uses it. A dim the tensor does not have (2 or more for a table, or 1
for a 1-D tensor) is an error: the console prints "Dimension out of
range" and nothing is sent."""

KEEP_DIMS = """keep_dims (checkbox, only when an argument was given; default off):
Off, the reduced dim is removed: 6 by 8 reduced on dim 0 has shape [8].
On, it is kept with length 1: shape [1, 8] on dim 0, [6, 1] on dim 1.
Keep it when the answer must line up against the original again - to
subtract each row's {what} from that row, for instance."""

TENSOR_IN = """tensor in:
The tensor. Every one that arrives is {verb} and the answer sent - this
is the only inlet that makes the node send. A list or a NumPy array is
first turned into a float32 tensor on the cpu; a tensor is used as it
is, on its own device, and the answer stays there. An empty tensor is
ignored: nothing is sent."""

DIM_DEMO = """The slider sets the dim of the second node: it goes through
't bang int', so the number reaches the dim inlet first and the bang
then makes the var re-send the held tensor - a dim change alone would
not send. t.info shows the shape of the answer: [8] for dim 0, [6]
for dim 1."""


def fill(text, **kw):
    for k, v in kw.items():
        text = text.replace('{' + k + '}', v)
    return text


# ------------------------------------------------------- the shared source
def source(op, props, hm_props, rows=6, cols=8, note='click its button for new values'):
    """load_bang -> t.rand -> var (holds it, so a dim change can re-send it),
    and the tensor drawn as a heat_map with its numbers."""
    var = 't_%s_help_tensor' % op[2:]
    return [
        {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
        {'key': 'rnd', 'init': 't.rand %d %d' % (rows, cols), 'pos': (30, 130),
         'w': 200, 'h': 230, 'props': props},
        {'key': 'crnd', 'comment': True, 'text': note, 'pos': (250, 135)},
        {'key': 'var', 'init': 'var ' + var, 'pos': (250, 200), 'w': 250, 'h': 70,
         'props': {'name': var}},
        {'key': 'cvar', 'comment': True,
         'text': 'holds the tensor so it can be sent again', 'pos': (250, 280)},
        {'key': 'hm', 'init': 'heat_map', 'pos': (30, 400), 'w': 460, 'h': 290,
         'props': hm_props},
        {'key': 'chm', 'comment': True,
         'text': 'the tensor: %d rows by %d columns, row 0 at the top' % (rows, cols),
         'pos': (30, 695)},
    ]


SOURCE_LINKS = [('lb', 'out', 'rnd', '###input'), ('rnd', 'random tensor', 'var', 'in'),
                ('var', 'out', 'hm', 'y')]


def dim_controls(y, maxdim=1, label='dim: drag it between 0 and 1'):
    return [
        {'key': 'sld', 'init': 'slider %d' % maxdim, 'pos': (30, y), 'w': 160, 'h': 42,
         'props': {'min': 0, 'max': maxdim, 'format': '%d', 'width': 140}},
        {'key': 'csld', 'comment': True, 'text': label, 'pos': (210, y + 5)},
        {'key': 'tbi', 'init': 't bang int', 'pos': (30, y + 60), 'w': 110, 'h': 46},
    ]


DIM_LINKS = [('sld', 'int out', 'tbi', ''), ('tbi', 'int', 'rd', 'dim'),
             ('tbi', 'bang', 'var', 'in'), ('var', 'out', 'rd', 'tensor in')]


def value_demo(op, out, props, hm_props, strip_props, whole='float', index_out=None,
               keep_dims=True, note='click its button for new values', whole_note=None):
    """Whole-tensor answer beside the picture; a dim copy under it with its
    answers as a strip, its shape from t.info, and (median / max / min) its
    indices in a list."""
    demo = source(op, props, hm_props, note=note) + [
        {'key': 'r', 'init': op, 'pos': (530, 400), 'w': 150, 'h': 70},
        {'key': 'cr', 'comment': True,
         'text': whole_note or 'no argument: the whole tensor', 'pos': (530, 405)},
        {'key': 'w', 'init': whole, 'pos': (530, 490), 'w': 127, 'h': 42,
         'props': FLT if whole == 'float' else INT},
    ] + dim_controls(740) + [
        {'key': 'rd', 'init': op + ' 0', 'pos': (30, 870), 'w': 200,
         'h': 130 if keep_dims else 100,
         'props': ({'dim': 0, 'keep_dims': False} if keep_dims else {'dim': 0})},
        {'key': 'crd', 'comment': True, 'text': 'with a dim: one answer per column (0) or row (1)',
         'pos': (250, 875)},
        {'key': 'st', 'init': 'heat_map', 'pos': (30, 1030), 'w': 460, 'h': 110,
         'props': strip_props},
        {'key': 'cst', 'comment': True, 'text': 'the answers, first on the left',
         'pos': (30, 1145)},
        {'key': 'inf', 'init': 't.info', 'pos': (530, 1030), 'w': 150, 'h': 120},
        {'key': 'shp', 'init': 'list', 'pos': (530, 1170), 'w': 120, 'h': 42, 'props': LIST},
        {'key': 'cshp', 'comment': True, 'text': 'shape of the answer', 'pos': (530, 1220)},
    ]
    links = SOURCE_LINKS + DIM_LINKS + [
        ('var', 'out', 'r', 'tensor in'), ('r', out, 'w', ''),
        ('rd', out, 'st', 'y'), ('rd', out, 'inf', 'in'), ('inf', 'shape', 'shp', '')]
    if index_out:
        demo += [
            {'key': 'ix', 'init': 'list', 'pos': (530, 870), 'w': 260, 'h': 42, 'props': LIST},
            {'key': 'cix', 'comment': True, 'text': 'and where each was found',
             'pos': (530, 920)},
        ]
        links.append(('rd', index_out, 'ix', ''))
    return demo, links


SEE_Y = 1270


# ======================================================================= t.mean
body = """Averages the values of a tensor: adds them up and divides by how
many there are - the whole tensor, or each row or each column.

The mean is the everyday "typical value", the balance point of the
data: one number for a frame of sensor readings, the centre of a cloud
of points, or each joint averaged over a window of frames (dim 0 of a
frames by joints tensor). Every value counts equally, so one wild
value pulls it a long way - t.median is the outlier-proof choice.

""" + fill(DIM_TEXT, op='t.mean', kd=' and a keep_dims checkbox') + """

SYNTAX:
t.mean <dim: int>
Default: no dim - the whole tensor.

EXAMPLE:
t.mean 0

INPUTS and PARAMETERS:

""" + fill(TENSOR_IN, verb='averaged') + """
A tensor of WHOLE NUMBERS (int64, or the bool mask a comparison
gives) is refused: torch prints "mean(): could not infer output
dtype" and nothing is sent. Put a t.to set to float32 in front.
(A list or NumPy array of whole numbers is fine - it arrives as
float32.) A NaN anywhere makes the answer NaN; t.nanmean skips them.

""" + DIM_INLET + """

""" + fill(KEEP_DIMS, what='mean') + """

OUTPUTS:

output:
With no dim, one number (a 0-dimensional tensor). With a dim, a tensor
with that dim removed (or kept at length 1 with keep_dims). Same dtype
and device as the input.

READING THE DEMO:
t.rand makes 6 rows by 8 columns of values between 0 and 1, written in
the cells; click its button for new ones. The plain t.mean beside it
averages all 48 - always near 0.5. """ + DIM_DEMO + """ At dim 0
each cell of the strip is the mean of one column; at dim 1, of one
row (row 0 on the left).

RELATED:
t.median is the middle value, which ignores outliers. t.nanmean skips
NaN values. t.sum is the total before dividing. np.mean is the NumPy
version, with an axis menu instead of a dim argument."""

demo, links = value_demo('t.mean', 'output', rand_props(0, 1), HM(440, 240, 6),
                         HM(440, 60, 1))
demo += see_also(SEE_Y, ['np.mean', 't.median', 't.nanmean', 't.sum', 't.max'])
print(build('t.mean', 't.mean - the average of a tensor',
            body, demo, links, demo_width=820, text_width=760, text_height=th(body)))

# ===================================================================== t.median
body = """Gives the middle value of a tensor: sort the values and take the one
in the middle - of the whole tensor, or of each row or each column.

Reach for it instead of t.mean when the data has outliers: a glitch,
a dropped reading, one joint flung out of place. Half the values lie
above the median and half below, so one wild value barely moves it.

THE LOWER MIDDLE, NOT AN AVERAGE:
When there is an even number of values there are two in the middle.
t.median gives the LOWER of the two - it does not average them, as
np.median does. So the answer is always one of the values actually
in the tensor: the median of 1 2 3 8 is 2 here, 2.5 in np.median.

""" + fill(DIM_TEXT, op='t.median', kd=', a keep_dims checkbox and an index outlet') + """

SYNTAX:
t.median <dim: int>
Default: no dim - the whole tensor.

EXAMPLE:
t.median 1

INPUTS and PARAMETERS:

""" + fill(TENSOR_IN, verb='searched for its middle value') + """
Whole-number tensors are fine (unlike t.mean) and the answer is a whole
number. If any value is NaN the answer is NaN; t.nanmedian skips them.

""" + DIM_INLET + """

""" + fill(KEEP_DIMS, what='median') + """

OUTPUTS:

output:
With no dim, one value (a 0-dimensional tensor). With a dim, a tensor
of middle values with that dim removed (or kept at length 1).

index output (only with a dim):
For each answer, WHERE along the dim it was found: at dim 0 the row
each column's median sits in, at dim 1 the column. It is sent just
before the values.

READING THE DEMO:
t.rand makes 6 rows by 8 columns of whole numbers 0 to 9 (dtype
int64), written in the cells; click its button for new ones. The plain
t.median gives the 24th of the 48 values once sorted. """ + DIM_DEMO + """
At dim 0 each column has 6 values, an even count, so each answer is
the 3rd smallest of its column - check one against the picture. The
list shows the row each answer was taken from.

RELATED:
t.mean is the average, pulled about by outliers. t.nanmedian skips
NaN values. t.max and t.min give the extremes. np.median is the NumPy
version (it averages the two middle values)."""

demo, links = value_demo('t.median', 'output', rand_props(0, 10, 'int64'),
                         HM(440, 240, 6, 0.0, 9.0, fmt='%.0f'),
                         HM(440, 60, 1, 0.0, 9.0, fmt='%.0f'),
                         whole='int', index_out='index output')
demo += see_also(SEE_Y, ['np.median', 't.mean', 't.nanmedian', 't.max', 't.min'])
print(build('t.median', 't.median - the middle value of a tensor',
            body, demo, links, demo_width=820, text_width=760, text_height=th(body)))

# ======================================================================== t.sum
body = """Adds up the values of a tensor - all of them, or each row or each
column separately.

Use it for a total (the energy in a frame, the number of true cells in
a mask - True counts as 1), a row or column total (the activity of each
joint over a window, an image projected onto one axis), or as the first
step of an average you want to weight yourself.

""" + fill(DIM_TEXT, op='t.sum', kd=' and a keep_dims checkbox') + """

SYNTAX:
t.sum <dim: int>
Default: no dim - the whole tensor.

EXAMPLE:
t.sum 1

INPUTS and PARAMETERS:

""" + fill(TENSOR_IN, verb='added up') + """
Any dtype is accepted. A NaN makes every total it is part of NaN;
t.nansum skips them.

""" + DIM_INLET + """

""" + fill(KEEP_DIMS, what='total') + """

OUTPUTS:

output:
With no dim, one number (a 0-dimensional tensor); with a dim, a tensor
of totals with that dim removed (or kept at length 1). Whole numbers
and bools sum to int64; floats keep their dtype.

READING THE DEMO:
t.rand makes 6 rows by 8 columns of whole numbers 0 to 3 (int64),
written in the cells; click its button for new ones. The plain t.sum
adds all 48. """ + DIM_DEMO + """ At dim 0 each cell of the strip
is a column total (up to 18), at dim 1 a row total (up to 24) - either
set adds up to the whole-tensor total.

RELATED:
t.nansum skips NaN values. t.mean is the sum divided by the count.
t.prod multiplies instead of adding. t.cumsum keeps the running total
at every step. t.count_nonzero counts the non-zero values. np.sum is
the NumPy version."""

demo, links = value_demo('t.sum', 'output', rand_props(0, 4, 'int64'),
                         HM(440, 240, 6, 0.0, 3.0, fmt='%.0f'),
                         HM(440, 60, 1, 0.0, 24.0, fmt='%.0f'), whole='int')
demo += see_also(SEE_Y, ['np.sum', 't.nansum', 't.mean', 't.prod', 't.cumsum', 't.count_nonzero'])
print(build('t.sum', 't.sum - add up a tensor',
            body, demo, links, demo_width=820, text_width=760, text_height=th(body)))

# ======================================================================= t.prod
body = """Multiplies the values of a tensor together - all of them, or each row
or each column separately.

Use it to combine factors: gains applied one after another, survival
probabilities, the number of cells in a shape (the product of its
sizes). A single 0 anywhere makes the product 0, and many values below
1 shrink it fast, so it is most useful on a few numbers near 1 or on
small whole numbers.

""" + fill(DIM_TEXT, op='t.prod', kd=' and a keep_dims checkbox') + """

SYNTAX:
t.prod <dim: int>
Default: no dim - the whole tensor.

EXAMPLE:
t.prod 0

INPUTS and PARAMETERS:

""" + fill(TENSOR_IN, verb='multiplied out') + """
Any dtype is accepted. A NaN makes every product it is part of NaN.
Whole numbers multiply in int64, which wraps around silently past about
9 million million million; floats overflow to inf.

""" + DIM_INLET + """

""" + fill(KEEP_DIMS, what='product') + """

OUTPUTS:

output:
With no dim, one number (a 0-dimensional tensor); with a dim, a tensor
of products with that dim removed (or kept at length 1). Whole numbers
and bools give int64; floats keep their dtype.

READING THE DEMO:
t.rand makes 3 rows by 4 columns of whole numbers 1 to 3 (int64),
written in the cells; click its button for new ones. The plain t.prod
multiplies all 12 (at most 3 to the 12th, 531441). The slider sets the
dim of the second node through 't bang int' (number first, then a bang
that re-sends the held tensor). At dim 0 each cell of the strip is a
column's product of 3 values (up to 27), at dim 1 a row's product of 4
(up to 81); t.info shows the shape, [4] or [3].

RELATED:
t.sum adds instead of multiplying. t.cumprod keeps the running product
at every step. t.mean, t.max and t.min are the other reductions. There
is no NumPy np.prod node; np.cumprod's last value is the same product."""

demo, links = value_demo('t.prod', 'output',
                         rand_props(1, 4, 'int64', shape='[3, 4]'),
                         HM(440, 240, 3, 1.0, 3.0, fmt='%.0f'),
                         HM(440, 60, 1, 0.0, 81.0, fmt='%.0f'), whole='int')
demo[[d['key'] for d in demo].index('rnd')]['init'] = 't.rand 3 4'
demo[[d['key'] for d in demo].index('chm')]['text'] = \
    'the tensor: 3 rows by 4 columns, row 0 at the top'
demo += see_also(SEE_Y, ['t.sum', 't.cumprod', 't.mean', 'np.cumprod', 't.max'])
print(build('t.prod', 't.prod - multiply a tensor out',
            body, demo, links, demo_width=820, text_width=760, text_height=th(body)))


# ============================================================ t.max / t.min
def minmax_body(op):
    big = op == 't.max'
    word = 'largest' if big else 'smallest'
    other = 't.min' if big else 't.max'
    return """Finds the """ + word + """ value in a tensor - in the whole tensor, or in
each row or each column, together with WHERE it was found.

""" + ("""Use it for the peak of a signal, the loudest band of a spectrum, the
fastest joint in a frame, the brightest pixel. With a dim it also says
which row or column the peak was in, so "how big" and "which one" come
from one node and cannot disagree."""
       if big else
       """Use it for the trough of a signal, the quietest channel, the closest
of a set of distances, the darkest pixel. With a dim it also says which
row or column the smallest was in, so "how small" and "which one" come
from one node and cannot disagree.""") + """

""" + fill(DIM_TEXT, op=op, kd=' and a second outlet for the indices') + """

NOT THE SAME AS t.""" + ('maximum' if big else 'minimum') + """:
""" + op + """ REDUCES - a tensor goes in and one value (or one per row or
column) comes out. t.""" + ('maximum' if big else 'minimum') + """ compares two tensors element by
element and keeps the """ + ('larger' if big else 'smaller') + """ at every position.

SYNTAX:
""" + op + """ <dim: int>
Default: no dim - the whole tensor, and only one outlet.

EXAMPLE:
""" + op + """ 1

INPUTS and PARAMETERS:

""" + fill(TENSOR_IN, verb='searched') + """
If the tensor holds a NaN the answer is NaN (a NaN counts as """ + word + """).

""" + DIM_INLET + """
There is no keep_dims option on this node.

OUTPUTS:

""" + ('max tensor' if big else 'min tensor') + """:
With no dim, one value (a 0-dimensional tensor). With a dim, a tensor of
""" + word + """ values with that dim removed. Same dtype as the input.

""" + ('max indices' if big else 'min indices') + """ (only with a dim):
For each answer, where along the dim it was found (int64): at dim 0
the row of each column's """ + word + """, at dim 1 the column. When several
share the """ + word + """ value the first is given. It is sent just before the
values. For the position in the WHOLE tensor use t.argmax""" + ('' if big else ' on the\nnegated tensor (there is no t.argmin)') + """.

READING THE DEMO:
t.rand makes 6 rows by 8 columns of values between 0 and 1, written in
the cells; click its button for new ones. The plain """ + op + """ gives the
""" + word + """ of all 48. """ + DIM_DEMO + """ At dim 0 the strip holds
each column's """ + word + """ value and the list the row it was found in;
at dim 1, each row's and the column.

RELATED:
""" + other + """ finds the """ + ('smallest' if big else 'largest') + """. t.argmax gives only the position of the
largest. t.median and t.mean summarise the middle. t.""" + ('maximum' if big else 'minimum') + """ is the
element-wise version. np.""" + ('amax' if big else 'amin') + """ is the NumPy reduction."""


for op in ('t.max', 't.min'):
    big = op == 't.max'
    body = minmax_body(op)
    demo, links = value_demo(op, 'max tensor' if big else 'min tensor', rand_props(0, 1),
                             HM(440, 240, 6), HM(440, 60, 1),
                             index_out='max indices' if big else 'min indices',
                             keep_dims=False)
    demo += see_also(SEE_Y, (['np.amax', 't.min', 't.argmax', 't.maximum', 't.median'] if big
                             else ['np.amin', 't.max', 't.argmax', 't.minimum', 't.median']))
    print(build(op, op + (' - the largest value, and where' if big
                          else ' - the smallest value, and where'),
                body, demo, links, demo_width=820, text_width=760, text_height=th(body)))

# ===================================================================== t.argmax
body = """Gives the POSITION of the largest value in a tensor - its index, not
the value itself.

Use it to pick a winner: which sensor reads highest, which frequency
band is loudest, which class a set of scores votes for, where the
brightest pixel is. If several cells share the largest value, the
first one is given. There is no t.argmin: use t.argmax on the tensor
times -1, or t.min with a dim for the indices.

THE INDEX WITH NO dim:
With no argument the answer is ONE whole number counting the cells of
the whole tensor in reading order - along row 0, then row 1, and so on
- not a row and a column. In a tensor 8 columns wide, index 21 is row
2, column 5: the row is the index divided by the width (whole-number
division, // 8) and the column is the remainder (% 8).

""" + fill(DIM_TEXT, op='t.argmax', kd='') + """ With a dim each
answer is a position along THAT dim: dim 0 gives, for each column, the
row it was found in; dim 1 gives, for each row, the column.

SYNTAX:
t.argmax <dim: int>
Default: no dim - one index into the whole tensor.

EXAMPLE:
t.argmax 1

INPUTS and PARAMETERS:

""" + fill(TENSOR_IN, verb='searched') + """
A NaN counts as the largest: the index of the first NaN is given.

""" + DIM_INLET + """
There is no keep_dims option on this node.

OUTPUTS:

max index:
Whole numbers (int64). With no dim, one index (a 0-dimensional tensor);
with a dim, a tensor of indices, one per column (dim 0) or row (dim 1).

READING THE DEMO:
t.rand makes 6 rows by 8 columns of values between 0 and 1, written in
the cells; click its button for new ones. The plain t.argmax gives the
index of the largest of all 48; // 8 and % 8 split it into the row and
column - find that cell in the picture. The slider sets the dim of the
second node through 't bang int' (number first, then a bang that
re-sends the held tensor). Its list holds, at dim 0, the row of each
column's largest value (8 numbers); at dim 1, the column of each row's
(6 numbers).

RELATED:
t.max gives the largest value, and with a dim the same indices too.
t.argsort ranks every value, not just the top one. t.argwhere lists
every position where a condition holds. np.argmax is the NumPy
version."""

demo = source('t.argmax', rand_props(0, 1), HM(440, 240, 6)) + [
    {'key': 'r', 'init': 't.argmax', 'pos': (530, 400), 'w': 150, 'h': 70},
    {'key': 'cr', 'comment': True, 'text': 'no argument: one index into the whole tensor',
     'pos': (530, 405)},
    {'key': 'w', 'init': 'int', 'pos': (530, 490), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'cw', 'comment': True, 'text': 'counting cells row by row from 0',
     'pos': (530, 540)},
    {'key': 'fd', 'init': '// 8', 'pos': (530, 580), 'w': 110, 'h': 60},
    {'key': 'md', 'init': '% 8', 'pos': (680, 580), 'w': 110, 'h': 60},
    {'key': 'ir', 'init': 'int', 'pos': (530, 660), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'ic', 'init': 'int', 'pos': (680, 660), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'crc', 'comment': True, 'text': 'row = index // 8, column = index % 8',
     'pos': (530, 710)},
] + dim_controls(740) + [
    {'key': 'rd', 'init': 't.argmax 0', 'pos': (30, 870), 'w': 200, 'h': 100,
     'props': {'dim': 0}},
    {'key': 'crd', 'comment': True, 'text': 'with a dim: one index per column (0) or row (1)',
     'pos': (250, 875)},
    {'key': 'ix', 'init': 'list', 'pos': (30, 1000), 'w': 300, 'h': 42, 'props': LIST},
    {'key': 'cix', 'comment': True, 'text': 'dim 0: the row of each column\'s largest',
     'pos': (30, 1050)},
] + see_also(1110, ['np.argmax', 't.max', 't.argsort', 't.argwhere', 't.min'])
links = SOURCE_LINKS + DIM_LINKS + [
    ('var', 'out', 'r', 'tensor in'), ('r', 'max index', 'w', ''),
    ('r', 'max index', 'fd', 'in'), ('r', 'max index', 'md', 'in'),
    ('fd', 'result', 'ir', ''), ('md', 'result', 'ic', ''),
    ('rd', 'max index', 'ix', '')]
print(build('t.argmax', 't.argmax - where the largest value is',
            body, demo, links, demo_width=820, text_width=760, text_height=th(body)))


# ================================================================== NaN pages
HOLE_SUB = sub_comments({
    'name': 'make_hole',
    'host': 'hole',
    'demo': [
        {'key': 'ina', 'init': 'in tensor', 'pos': (40, 40), 'w': 150, 'h': 80},
        {'key': 'inh', 'init': 'in hole', 'pos': (240, 40), 'w': 150, 'h': 80},
        {'key': 'lb', 'init': 'load_bang', 'pos': (440, 40), 'w': 88, 'h': 46},
        {'key': 'nan', 'init': 'message', 'pos': (440, 110), 'w': 100, 'h': 42,
         'props': {'text in': 'nan', 'font size': '24', 'width': 80}},
        {'key': 'cnan', 'comment': True, 'text': 'the NaN that fills the hole',
         'pos': (560, 115)},
        {'key': 'zr', 'init': 'np.zeros 6 8', 'pos': (40, 240), 'w': 170, 'h': 110,
         'props': {'dim 0': 6, 'dim 1': 8, 'dtype': 'float32'}},
        {'key': 'czr', 'comment': True, 'text': 'made fresh for every tensor that arrives',
         'pos': (230, 245)},
        {'key': 'ed', 'init': 'np.edit 2,3', 'pos': (40, 370), 'w': 170, 'h': 90,
         'props': {'indices': '2,3', 'values': '1'}},
        {'key': 'ced', 'comment': True, 'text': 'a 1 at row 2, column 3: where the hole goes',
         'pos': (230, 375)},
        {'key': 'ml', 'init': '*', 'pos': (40, 480), 'w': 130, 'h': 70},
        {'key': 'cml', 'comment': True, 'text': 'times the toggle: no hole when it is off',
         'pos': (230, 485)},
        {'key': 'wh', 'init': 'np.where', 'pos': (40, 570), 'w': 150, 'h': 100},
        {'key': 'cwh', 'comment': True, 'text': 'NaN at the hole, the tensor everywhere else',
         'pos': (230, 575)},
        {'key': 'out', 'init': 'out holed', 'pos': (40, 690), 'w': 160, 'h': 80},
        {'key': 'tt', 'init': 't bang array', 'pos': (40, 150), 'w': 130, 'h': 46},
        {'key': 'ctt', 'comment': True, 'text': 'the tensor (as a NumPy array) first, then a fresh mask',
         'pos': (230, 155)},
    ],
    'links': [('lb', 'out', 'nan', ''), ('nan', 'message out', 'wh', 'if true'),
              ('ina', 'tensor', 'tt', ''), ('tt', 'array', 'wh', 'if false'),
              ('tt', 'bang', 'zr', ''),
              ('inh', 'hole', 'ml', 'operand'),
              ('zr', 'array of zeros', 'ed', 'array in'),
              ('ed', 'output', 'ml', 'in'), ('ml', 'result', 'wh', 'condition'),
              ('wh', 'selected', 'out', 'holed')],
})


def nan_demo(plain, nan, out, props, hm_props, strip_props, whole='float',
             index_out=None):
    """t.rand -> var -> make_hole (toggle puts a NaN at row 2, column 3);
    plain and nan versions side by side, whole and along the live dim."""
    demo = source(nan, props, hm_props) + [
        {'key': 'tg', 'init': 'toggle', 'pos': (530, 130), 'w': 60, 'h': 46},
        {'key': 'tbt', 'init': 't bang int', 'pos': (620, 130), 'w': 110, 'h': 46},
        {'key': 'ctg', 'comment': True, 'text': 'click: put a NaN at row 2, column 3',
         'pos': (750, 135)},
        {'key': 'hole', 'init': 'p make_hole', 'pos': (530, 200), 'w': 220, 'h': 90},
        {'key': 'cho', 'comment': True, 'text': 'open it to see how the hole is made',
         'pos': (530, 300)},
        {'key': 'rp', 'init': plain, 'pos': (530, 400), 'w': 150, 'h': 70},
        {'key': 'wp', 'init': whole, 'pos': (530, 490), 'w': 127, 'h': 42,
         'props': FLT if whole == 'float' else INT},
        {'key': 'cwp', 'comment': True, 'text': plain + ' of everything', 'pos': (530, 540)},
        {'key': 'rn', 'init': nan, 'pos': (530, 590), 'w': 150, 'h': 70},
        {'key': 'wn', 'init': whole, 'pos': (530, 680), 'w': 127, 'h': 42,
         'props': FLT if whole == 'float' else INT},
        {'key': 'cwn', 'comment': True, 'text': nan + ' of everything', 'pos': (530, 730)},
    ] + dim_controls(780) + [
        {'key': 'rdp', 'init': plain + ' 1', 'pos': (30, 910), 'w': 200, 'h': 130,
         'props': {'dim': 1, 'keep_dims': False}},
        {'key': 'stp', 'init': 'heat_map', 'pos': (30, 1060), 'w': 460, 'h': 110,
         'props': strip_props},
        {'key': 'cstp', 'comment': True, 'text': plain + ' along the dim', 'pos': (30, 1175)},
        {'key': 'rd', 'init': nan + ' 1', 'pos': (30, 1220), 'w': 200, 'h': 130,
         'props': {'dim': 1, 'keep_dims': False}},
        {'key': 'st', 'init': 'heat_map', 'pos': (30, 1370), 'w': 460, 'h': 110,
         'props': strip_props},
        {'key': 'cst', 'comment': True, 'text': nan + ' along the dim', 'pos': (30, 1485)},
    ]
    # the slider starts at 1 here: one answer per row, so the hole's row shows
    demo[[d['key'] for d in demo].index('sld')]['props'][''] = 1
    links = [('lb', 'out', 'rnd', '###input'), ('rnd', 'random tensor', 'var', 'in'),
             ('var', 'out', 'hole', 'tensor'), ('tg', '', 'tbt', ''),
             ('tbt', 'int', 'hole', 'hole'), ('tbt', 'bang', 'var', 'in'),
             ('hole', 'holed', 'hm', 'y'),
             ('hole', 'holed', 'rp', 'tensor in'), ('rp', out, 'wp', ''),
             ('hole', 'holed', 'rn', 'tensor in'), ('rn', out, 'wn', ''),
             ('sld', 'int out', 'tbi', ''), ('tbi', 'int', 'rdp', 'dim'),
             ('tbi', 'int', 'rd', 'dim'), ('tbi', 'bang', 'var', 'in'),
             ('hole', 'holed', 'rdp', 'tensor in'), ('hole', 'holed', 'rd', 'tensor in'),
             ('rdp', out, 'stp', 'y'), ('rd', out, 'st', 'y')]
    return demo, links


NAN_DEMO = """make_hole passes the tensor through untouched until you click the
toggle: then the cell at row 2, column 3 becomes NaN (the heat_map
writes it as nan). The slider starts at dim 1 - one answer per row -
and goes through 't bang int' to both dim nodes, re-sending the held
tensor."""

NAN_ARG = """THE dim ARGUMENT:
With no argument the WHOLE tensor is reduced to one value. A number
typed after the name ({op} 0) chooses a dim to reduce along and adds a
dim inlet and a keep_dims checkbox{more} - the argument creates them, so
a node made without one cannot be given a dim later. For a table,
dim 0 gives one answer per column, dim 1 one per row; -1 is the last
dim."""

# ---------------------------------------------------------------- t.nanmean
body = """Averages the values of a tensor as t.mean does, but skips any NaN
(not-a-number) values instead of letting them spoil the answer.

Real data has holes in it - a dropped frame, a sensor that lost
tracking, a division by zero upstream - and those often arrive as NaN.
NaN plus anything is NaN, so ONE missing value turns a t.mean into NaN.
t.nanmean averages only the values that are there: the sum of the
real values divided by how many real values there are.

A ROW OF NOTHING BUT NaN:
gives NaN - there is nothing to average. (t.nansum gives 0 instead.)

""" + fill(NAN_ARG, op='t.nanmean', more='') + """

SYNTAX:
t.nanmean <dim: int>
Default: no dim - the whole tensor.

EXAMPLE:
t.nanmean 0

INPUTS and PARAMETERS:

""" + fill(TENSOR_IN, verb='averaged') + """
Like t.mean it needs floats: an int64 or bool tensor is refused
("expected input to have floating point or complex dtype") - whole
numbers cannot hold a NaN anyway.
With no NaN in it the answer is exactly what t.mean gives.

""" + DIM_INLET + """

""" + fill(KEEP_DIMS, what='mean') + """

OUTPUTS:

output:
With no dim, one number; with a dim, a tensor of means with that dim
removed (or kept at length 1). NaN only where a whole row or column
was NaN.

READING THE DEMO:
t.rand makes 6 rows by 8 columns of values between 0 and 1; click its
button for new ones. """ + NAN_DEMO + """ With the toggle off, t.mean and
t.nanmean agree everywhere. With it on, t.mean of everything turns to
nan, and so does its answer for row 2 (the third cell of its strip);
every other row is unaffected. t.nanmean's answers stay numbers: row
2 is averaged over its 7 remaining values.

RELATED:
t.mean is the plain version. t.nansum and t.nanmedian skip NaN the same
way for the total and the middle value. np.nanmean is the NumPy
version."""

demo, links = nan_demo('t.mean', 't.nanmean', 'output', rand_props(0, 1),
                       HM(440, 240, 6), HM(440, 60, 1))
demo += see_also(1530, ['np.nanmean', 't.mean', 't.nansum', 't.nanmedian'])
print(build('t.nanmean', 't.nanmean - the average, skipping NaN',
            body, demo, links, demo_width=820, text_width=760, text_height=th(body),
            subpatch=HOLE_SUB))

# -------------------------------------------------------------- t.nanmedian
body = """Gives the middle value of a tensor as t.median does, but skips any
NaN (not-a-number) values instead of letting them spoil the answer.

A lost reading arrives as NaN, and t.median of anything containing a
NaN is NaN. t.nanmedian sorts only the values that are there and takes
the middle of those - an answer that is both outlier-proof and
gap-proof, the most forgiving summary of messy sensor data.

THE LOWER MIDDLE:
Like t.median, with an even number of real values it gives the LOWER
of the two middle ones, not their average - so the answer is always a
value that is really in the tensor. A row of nothing but NaN gives
NaN.

""" + fill(NAN_ARG, op='t.nanmedian', more=' and an index outlet') + """

SYNTAX:
t.nanmedian <dim: int>
Default: no dim - the whole tensor.

EXAMPLE:
t.nanmedian 1

INPUTS and PARAMETERS:

""" + fill(TENSOR_IN, verb='searched for its middle value') + """
With no NaN in it the answer is exactly what t.median gives.

""" + DIM_INLET + """

""" + fill(KEEP_DIMS, what='median') + """

OUTPUTS:

output:
With no dim, one value; with a dim, a tensor of middle values with that
dim removed (or kept at length 1).

index output (only with a dim):
For each answer, where along the dim it was found. Sent just before the
values.

READING THE DEMO:
t.rand makes 6 rows by 8 columns of whole numbers 0 to 9; click its
button for new ones. """ + NAN_DEMO + """ With the toggle off, the two
agree everywhere. With it on, t.median of everything turns to nan, and
so does its answer for row 2 (the third cell of its strip).
t.nanmedian's answers stay numbers: row 2's middle is taken from its 7
remaining values - with an odd count, the true middle one.

RELATED:
t.median is the plain version. t.nanmean and t.nansum skip NaN the same
way for the average and the total. np.nanmedian is the NumPy version
(it averages the two middle values)."""

demo, links = nan_demo('t.median', 't.nanmedian', 'output', rand_props(0, 10, 'int64'),
                       HM(440, 240, 6, 0.0, 9.0, fmt='%.0f'),
                       HM(440, 60, 1, 0.0, 9.0, fmt='%.0f'))
demo += see_also(1530, ['np.nanmedian', 't.median', 't.nanmean', 't.nansum'])
print(build('t.nanmedian', 't.nanmedian - the middle value, skipping NaN',
            body, demo, links, demo_width=820, text_width=760, text_height=th(body),
            subpatch=HOLE_SUB))

# ----------------------------------------------------------------- t.nansum
body = """Adds up the values of a tensor as t.sum does, but skips any NaN
(not-a-number) values instead of letting them spoil the total.

NaN plus anything is NaN, so ONE missing value - a dropped frame, a
lost sensor - turns a t.sum total into NaN. t.nansum treats each NaN
as 0 and adds up what is actually there.

A ROW OF NOTHING BUT NaN:
gives 0 - an empty total, not NaN. If you need to know that a row had
no data at all, count its NaNs separately.

""" + fill(NAN_ARG, op='t.nansum', more='') + """

SYNTAX:
t.nansum <dim: int>
Default: no dim - the whole tensor.

EXAMPLE:
t.nansum 1

INPUTS and PARAMETERS:

""" + fill(TENSOR_IN, verb='added up') + """
With no NaN in it the answer is exactly what t.sum gives.

""" + DIM_INLET + """

""" + fill(KEEP_DIMS, what='total') + """

OUTPUTS:

output:
With no dim, one number; with a dim, a tensor of totals with that dim
removed (or kept at length 1), never NaN.

READING THE DEMO:
t.rand makes 6 rows by 8 columns of whole numbers 0 to 3; click its
button for new ones. """ + NAN_DEMO + """ With the toggle off, t.sum and
t.nansum agree everywhere. With it on, t.sum of everything turns to
nan, and so does its total for row 2 (the third cell of its strip);
every other row is unaffected. t.nansum's totals stay numbers: row 2's
simply leaves the hole out.

RELATED:
t.sum is the plain version. t.nanmean and t.nanmedian skip NaN the same
way for the average and the middle value. np.nansum is the NumPy
version."""

demo, links = nan_demo('t.sum', 't.nansum', 'output', rand_props(0, 4, 'int64'),
                       HM(440, 240, 6, 0.0, 3.0, fmt='%.0f'),
                       HM(440, 60, 1, 0.0, 24.0, fmt='%.0f'))
demo += see_also(1530, ['np.nansum', 't.sum', 't.nanmean', 't.nanmedian'])
print(build('t.nansum', 't.nansum - add up, skipping NaN',
            body, demo, links, demo_width=820, text_width=760, text_height=th(body),
            subpatch=HOLE_SUB))


# ============================================================== mask pages
def mask_source(op, thresh=0.5, note='values 0 to 1 - click its button for new ones'):
    """t.rand 6 x 8 held in a var; a threshold slider (through t bang float)
    feeds t.gt and re-sends; the mask drawn in two tones."""
    var = 't_%s_help_tensor' % op[2:]
    demo = [
        {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
        {'key': 'rnd', 'init': 't.rand 6 8', 'pos': (30, 130), 'w': 200, 'h': 230,
         'props': rand_props(0, 1)},
        {'key': 'crnd', 'comment': True, 'text': note, 'pos': (250, 135)},
        {'key': 'var', 'init': 'var ' + var, 'pos': (250, 200), 'w': 250, 'h': 70,
         'props': {'name': var}},
        {'key': 'hin', 'init': 'heat_map', 'pos': (30, 400), 'w': 380, 'h': 260,
         'props': HM(352, 210, 6)},
        {'key': 'chin', 'comment': True, 'text': 'the tensor', 'pos': (30, 665)},
        {'key': 'th', 'init': 'message', 'pos': (530, 62), 'w': 80, 'h': 42,
         'props': {'text in': str(thresh), 'font size': '24'}},
        {'key': 'cth', 'comment': True, 'text': 'starting threshold', 'pos': (630, 67)},
        {'key': 'sl', 'init': 'slider', 'pos': (530, 130), 'w': 200, 'h': 42,
         'props': {'min': 0.0, 'max': 1.0, 'format': '%.3f', 'width': 180}},
        {'key': 'csl', 'comment': True, 'text': 'threshold: drag it', 'pos': (750, 135)},
        {'key': 'tbf', 'init': 't bang float', 'pos': (530, 200), 'w': 120, 'h': 46},
        {'key': 'gt', 'init': 't.gt', 'pos': (530, 290), 'w': 150, 'h': 80},
        {'key': 'cgt', 'comment': True, 'text': 'true where the value is above the threshold',
         'pos': (700, 295)},
        {'key': 'hmk', 'init': 'heat_map', 'pos': (440, 400), 'w': 380, 'h': 260,
         'props': HM(352, 210, 6, fmt='none', color='hot')},
        {'key': 'chmk', 'comment': True, 'text': 'the mask: white where true',
         'pos': (440, 665)},
    ]
    links = [('lb', 'out', 'rnd', '###input'), ('lb', 'out', 'th', ''),
             ('th', 'message out', 'sl', ''), ('rnd', 'random tensor', 'var', 'in'),
             ('var', 'out', 'hin', 'y'), ('var', 'out', 'gt', 'tensor a in'),
             ('sl', 'float out', 'tbf', ''), ('tbf', 'float', 'gt', 'tensor b in'),
             ('tbf', 'bang', 'var', 'in'), ('gt', 'tensor result', 'hmk', 'y')]
    return demo, links


ANYALL_IN = """tensor in:
The tensor - usually a mask from a comparison (t.gt, t.lt, ...). Every
one that arrives is tested and the answer sent; it is the only inlet.
Any shape is accepted and the WHOLE tensor is always tested: there is no
dim, so you get one True or False, never one per row. A value counts as
true when it is not zero - so negative numbers and NaN are true. A list
or NumPy array is turned into a float32 tensor first; an empty tensor
is ignored (nothing is sent)."""

for op in ('t.any', 't.all'):
    is_any = op == 't.any'
    if is_any:
        body = """Answers True if ANY value in a tensor is true (non-zero), False only
if every one is zero.

Its usual job is the end of a test: compare a tensor with something,
then ask "did it happen anywhere?" - is any joint moving faster than a
limit, has any sensor dropped out, is any pixel above the threshold. It
turns a whole mask into one yes or no you can gate on.

SYNTAX:
t.any
It takes no arguments.

INPUTS and PARAMETERS:

""" + ANYALL_IN + """

OUTPUTS:

any:
True or False, as a 0-dimensional bool tensor (a toggle shows it).

READING THE DEMO:
t.rand makes 6 rows by 8 columns of values between 0 and 1. t.gt
compares each with the slider (which starts at 0.5) and draws the
answer as a mask: white where the value is above the threshold, black
where it is not. t.any looks at the mask and lights the toggle if even
ONE cell is white. Drag the slider up: cells go dark one by one, and
t.any stays True until the very last one - the largest value - goes
dark. Click t.rand's button for new values.

RELATED:
t.all is the opposite question: true only if EVERY value is true.
t.count_nonzero counts how many are true; t.argwhere says which.
Asking whether t.max of the tensor is above the threshold is the same
test. np.any is the NumPy version."""
        see = ['np.any', 't.all', 't.count_nonzero', 't.argwhere', 't.gt']
        note = 'True if any cell of the mask is white'
    else:
        body = """Answers True only if EVERY value in a tensor is true (non-zero), and
False as soon as a single one is zero.

Use it to check that a condition holds everywhere: are all joints
inside their range, did every sensor report, is every value finite,
has every step finished. It turns a whole mask into one yes or no.

SYNTAX:
t.all
It takes no arguments.

INPUTS and PARAMETERS:

""" + ANYALL_IN + """

OUTPUTS:

all:
True or False, as a 0-dimensional bool tensor (a toggle shows it).

READING THE DEMO:
t.rand makes 6 rows by 8 columns of values between 0 and 1. t.gt
compares each with the slider (which starts at 0.5) and draws the
answer as a mask: white where the value is above the threshold, black
where it is not. At 0.5 roughly half are black and t.all is False.
Drag the slider down to 0: every cell turns white and the toggle
lights - but only once the very last one, the smallest value, has
turned. Click t.rand's button for new values.

RELATED:
t.any is the opposite question: true if ANY value is true.
t.count_nonzero counts how many are true; t.all is that count equalling
the number of cells, and asking whether t.min of the tensor is above
the threshold is the same test.
np.all is the NumPy version."""
        see = ['np.all', 't.any', 't.count_nonzero', 't.min', 't.gt']
        note = 'True only if every cell of the mask is white'
    demo, links = mask_source(op)
    demo += [
        {'key': 'r', 'init': op, 'pos': (30, 710), 'w': 130, 'h': 70},
        {'key': 'out', 'init': 'toggle', 'pos': (30, 800), 'w': 60, 'h': 46},
        {'key': 'cr', 'comment': True, 'text': note, 'pos': (110, 805)},
    ] + see_also(880, see)
    links += [('gt', 'tensor result', 'r', 'tensor in'),
              ('r', op[2:], 'out', '')]
    print(build(op, op + (' - is anything true?' if is_any else ' - is everything true?'),
                body, demo, links, demo_width=840, text_width=760, text_height=th(body)))

# ------------------------------------------------------------ t.count_nonzero
body = """Counts the values in a tensor that are not zero - in the whole
tensor, or in each row or each column.

On a mask from a comparison it counts how many cells passed the test:
how many joints are over a speed limit, how many frames a joint spent
above a threshold, how many pixels are lit. For a frames by joints
mask, counting along dim 0 gives each joint's time over the threshold;
along dim 1, how many joints were over it at each frame - two quite
different summaries of the same mask.

""" + fill(DIM_TEXT, op='t.count_nonzero', kd='') + """

SYNTAX:
t.count_nonzero <dim: int>
Default: no dim - one count for the whole tensor.

EXAMPLE:
t.count_nonzero 0

INPUTS and PARAMETERS:

""" + fill(TENSOR_IN, verb='counted') + """
Any dtype. A value counts when it is not exactly zero - NaN counts.
A tensor of measurements passed straight in reports nearly everything,
since almost nothing is exactly zero: compare first, then count.

""" + DIM_INLET + """
There is no keep_dims option on this node.

OUTPUTS:

tensor out:
Whole numbers (int64). With no dim, one count (a 0-dimensional
tensor); with a dim, one count per column (dim 0) or row (dim 1).

READING THE DEMO:
t.rand makes 6 rows by 8 columns of values between 0 and 1; t.gt marks
those above the threshold slider (starting at 0.5), drawn as a mask,
white for true. The plain t.count_nonzero counts the white cells of
the whole mask. The dim slider sets the dim of the second node through
't bang int' (number first, then a bang that re-sends the held
tensor): at dim 0 the strip counts the white cells in each column (0
to 6), at dim 1 in each row (0 to 8). Drag the threshold and every
count follows.

RELATED:
t.sum of a mask gives the same count (True adds as 1). t.any and t.all
ask whether the count is above 0 or equal to everything. t.argwhere
lists WHICH cells are non-zero. t.bincount counts how often each whole
number occurs."""

demo, links = mask_source('t.count_nonzero')
demo += [
    {'key': 'r', 'init': 't.count_nonzero', 'pos': (440, 710), 'w': 200, 'h': 70},
    {'key': 'w', 'init': 'int', 'pos': (440, 800), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'cw', 'comment': True, 'text': 'no argument: white cells in the whole mask',
     'pos': (440, 850)},
] + dim_controls(710) + [
    {'key': 'rd', 'init': 't.count_nonzero 0', 'pos': (30, 840), 'w': 220, 'h': 100,
     'props': {'dim': 0}},
    {'key': 'st', 'init': 'heat_map', 'pos': (30, 960), 'w': 460, 'h': 110,
     'props': HM(440, 60, 1, 0.0, 8.0, fmt='%.0f')},
    {'key': 'cst', 'comment': True, 'text': 'counts per column (dim 0) or row (dim 1)',
     'pos': (30, 1075)},
    {'key': 'inf', 'init': 't.info', 'pos': (530, 960), 'w': 150, 'h': 120},
    {'key': 'shp', 'init': 'list', 'pos': (530, 1100), 'w': 120, 'h': 42, 'props': LIST},
] + see_also(1170, ['t.sum', 't.any', 't.all', 't.argwhere', 'np.sum'])
links += [('gt', 'tensor result', 'r', 'tensor in'), ('r', 'tensor out', 'w', ''),
          ('sld', 'int out', 'tbi', ''), ('tbi', 'int', 'rd', 'dim'),
          ('tbi', 'bang', 'var', 'in'), ('gt', 'tensor result', 'rd', 'tensor in'),
          ('rd', 'tensor out', 'st', 'y'), ('rd', 'tensor out', 'inf', 'in'),
          ('inf', 'shape', 'shp', '')]
print(build('t.count_nonzero', 't.count_nonzero - how many are not zero',
            body, demo, links, demo_width=840, text_width=760, text_height=th(body)))

# ----------------------------------------------------------------- t.argwhere
body = """Lists the positions of every non-zero (true) cell in a tensor.

Where t.count_nonzero says HOW MANY cells of a mask are set,
t.argwhere says WHICH: which joints are over the speed limit, which
frames are glitched, which pixels are above threshold. Each found cell
comes out as its coordinates.

t.non_zero IS THE SAME NODE:
t.non_zero is another name for exactly this node - same code, same
answer (it matches torch.nonzero, which gives the same coordinates).

SYNTAX:
t.argwhere
t.non_zero
Neither takes arguments.

INPUTS and PARAMETERS:

tensor in:
Any tensor, usually a mask from a comparison (t.gt, t.lt, ...). A cell
counts as found when it is not zero - so a plain tensor of measurements
finds nearly everything. Every tensor that arrives is searched and the
answer sent. A list or NumPy array is turned into a float32 tensor
first; an empty tensor is ignored (nothing is sent).

OUTPUTS:

index tensor where non-zero:
A whole-number (int64) tensor with one ROW per found cell and one
column per dim of the input: for a table each row is (row, column),
for a 1-D tensor a single position. The rows come in reading order -
along the first row, then the next. Its shape is (number found, number
of dims), so its length changes from one tensor to the next; when
nothing is found it is empty, shape [0, 2] for a table.

READING THE DEMO:
t.rand makes 6 rows by 8 columns of values between 0 and 1; t.gt marks
those above the threshold slider (starting at 0.85, so only a few),
drawn as a mask, white for true. t.argwhere lists the white cells as
[row, column] pairs, and t.info shows its shape - the first number is
how many were found. Check a pair against the picture: row 0 is the
top, column 0 the left. Drag the threshold down and the list grows.

RELATED:
t.count_nonzero counts the same cells. t.any asks whether there are
any. t.masked_select gives the VALUES at the true cells rather than
their positions. t.argmax gives the position of the single largest
value. np.argwhere is the NumPy version."""

demo, links = mask_source('t.argwhere', thresh=0.85)
demo += [
    {'key': 'aw', 'init': 't.argwhere', 'pos': (30, 710), 'w': 250, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (30, 800), 'w': 500, 'h': 42, 'props': LIST},
    {'key': 'c1', 'comment': True, 'text': '[row, column] of each white cell',
     'pos': (30, 850)},
    {'key': 'inf', 'init': 't.info', 'pos': (560, 710), 'w': 150, 'h': 120},
    {'key': 'shp', 'init': 'list', 'pos': (560, 850), 'w': 120, 'h': 42, 'props': LIST},
    {'key': 'c2', 'comment': True, 'text': '[how many found, 2]', 'pos': (560, 900)},
] + see_also(950, ['np.argwhere', 't.count_nonzero', 't.masked_select', 't.argmax', 't.any'])
links += [('gt', 'tensor result', 'aw', 'tensor in'),
          ('aw', 'index tensor where non-zero', 'l1', ''),
          ('aw', 'index tensor where non-zero', 'inf', 'in'), ('inf', 'shape', 'shp', '')]
print(build('t.argwhere', 't.argwhere - where the non-zero cells are',
            body, demo, links, demo_width=840, text_width=760, text_height=th(body)))
