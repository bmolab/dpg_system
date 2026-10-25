"""t ordering, counting and running totals: one page per node.

t.argsort, t.histc, t.bincount, t.bucketize, t.cumsum, t.cumprod, t.cummax,
t.cummin, t.logcumsumexp, t.diff, t.corrcoef, t.energy. t.histc and t.cumsum
were family pages (histc / bincount / bucketize, and the cumsum family with
diff); their stems are node names, so they are now those nodes' own pages.
Each demo shows the input and the output side by side, with a SEE ALSO row.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import PLOT, INT, FLT


def HM(width, height, count, miny=0.0, maxy=1.0, fmt='%.2f', color='viridis'):
    """Property block for a heat_map that draws a whole array as one picture."""
    return {'color': color, 'width': width, 'height': height, 'sample count': count,
            'min y': miny, 'max y': maxy, 'update_mode': 'heat_map',
            'number format': fmt}


LIST = {'text in': '', 'font size': '24'}


def see_also(y, stems, x=30, right=580):
    """A SEE ALSO row: one help_link per related page, left to right, wrapping
    onto a second row rather than running under the prose."""
    row = []
    x0 = x
    for s in stems:
        w = int(len(s) * 7.7) + 30
        if x > x0 and x + w > right:
            x, y = x0, y + 44
        row.append({'key': 'sa_' + s, 'init': 'help_link ' + s, 'pos': (x, y),
                    'w': w, 'h': 30})
        x += w + 14
    if x + 200 > right:
        x, y = x0, y + 44
    row.append({'key': 'sa_c', 'comment': True, 'text': 'SEE ALSO - click to open',
                'pos': (x, y)})
    return row


def MULTI_PLOT(miny=-1.0, maxy=1.0, count=200, width=300):
    """A plot that draws each row of a 2-D tensor as its own line."""
    return dict(PLOT(miny, maxy, count),
                **{'width': width, 'update style': 'input is multi-channel sample'})


def ARRAY_PLOT(miny, maxy, count, width=300, style='line'):
    """A plot that draws one whole 1-D tensor as a curve, first value on the left."""
    return dict(PLOT(miny, maxy, count, style=style),
                **{'width': width, 'update style': 'buffer holds one sample of input'})


METRO = lambda ms: {'on': True, 'period': float(ms), 'units': 'milliseconds'}
MSG = lambda text, w=None: dict({'text in': text, 'font size': '24'},
                                **({'width': w} if w else {}))


def text_h(body):
    """Height of the prose block: 15px per line at text size 30, plus a margin."""
    return (body.count('\n') + 1) * 15 + 30


def page(name, title, body, demo, links, sub=None):
    print(build(name, title, body, demo, links, demo_width=600, text_width=760,
                text_height=text_h(body), subpatch=sub))


# --------------------------------------------------------- demo subpatchers
# Inside make_walk: metro 500 -> t.rand 200 (-1 to 1) -> t.cumsum: a random
# walk of 200 steps, a new one twice a second.
WALK_SUB = {
    'name': 'make_walk',
    'host': 'wk',
    'demo': [
        {'key': 'met', 'init': 'metro 500', 'pos': (40, 40), 'w': 129, 'h': 70,
         'props': METRO(500)},
        {'key': 'cm', 'init': 'comment a new walk twice a second',
         'pos': (190, 50), 'w': 300, 'h': 30},
        {'key': 'rnd', 'init': 't.rand 200', 'pos': (40, 130), 'w': 190, 'h': 160,
         'props': {'shape': '[200]', 'min': -1.0, 'max': 1.0}},
        {'key': 'cr', 'init': 'comment 200 random steps between -1 and 1',
         'pos': (250, 145), 'w': 340, 'h': 30},
        {'key': 'cs', 'init': 't.cumsum', 'pos': (40, 310), 'w': 120, 'h': 70},
        {'key': 'cc', 'init': 'comment added up one after another: a random walk',
         'pos': (180, 325), 'w': 400, 'h': 30},
        {'key': 'out', 'init': 'out walk', 'pos': (40, 410), 'w': 150, 'h': 80},
    ],
    'links': [('met', '', 'rnd', '###input'),
              ('rnd', 'random tensor', 'cs', 'tensor in'),
              ('cs', 'output', 'out', 'walk')],
}

# Inside make_channels: four channels of 64 samples stacked into a 4 x 64
# tensor. Row 0 is random, row 1 is row 0 plus some of row 3 (the slider sets
# how much), row 2 is row 0 upside down, row 3 is separate random values.
# The metro fires the noise first, then x; x reaches the stack's first
# (triggering) inlet last, so every row comes from the same tick.
CHANNELS_SUB = {
    'name': 'make_channels',
    'host': 'ch',
    'demo': [
        {'key': 'met', 'init': 'metro 200', 'pos': (40, 40), 'w': 129, 'h': 70,
         'props': METRO(200)},
        {'key': 'cm', 'init': 'comment new channels five times a second',
         'pos': (190, 50), 'w': 320, 'h': 30},
        {'key': 'nz', 'init': 't.rand 64', 'pos': (380, 130), 'w': 190, 'h': 160,
         'props': {'shape': '[64]', 'min': -1.0, 'max': 1.0}},
        {'key': 'cn', 'init': 'comment channel 3: its own random values',
         'pos': (380, 300), 'w': 320, 'h': 30},
        {'key': 'x', 'init': 't.rand 64', 'pos': (40, 130), 'w': 190, 'h': 160,
         'props': {'shape': '[64]', 'min': -1.0, 'max': 1.0}},
        {'key': 'cx', 'init': 'comment channel 0: random values',
         'pos': (40, 300), 'w': 280, 'h': 30},
        {'key': 'inn', 'init': 'in noise', 'pos': (600, 130), 'w': 150, 'h': 80},
        {'key': 'mul', 'init': '* 0.5', 'pos': (380, 360), 'w': 120, 'h': 46},
        {'key': 'cmu', 'init': 'comment times the noise amount',
         'pos': (520, 365), 'w': 260, 'h': 30},
        {'key': 'add', 'init': '+ 0.0', 'pos': (200, 440), 'w': 120, 'h': 46},
        {'key': 'cad', 'init': 'comment channel 1: channel 0 plus some of channel 3',
         'pos': (340, 445), 'w': 420, 'h': 30},
        {'key': 'neg', 'init': '* -1', 'pos': (40, 440), 'w': 120, 'h': 46},
        {'key': 'cng', 'init': 'comment channel 2: channel 0 upside down',
         'pos': (40, 495), 'w': 340, 'h': 30},
        {'key': 'st', 'init': 't.stack 4', 'pos': (40, 560), 'w': 160, 'h': 150},
        {'key': 'cst', 'init': 'comment the four rows as one 4 x 64 tensor',
         'pos': (220, 600), 'w': 360, 'h': 30},
        {'key': 'out', 'init': 'out channels', 'pos': (40, 740), 'w': 160, 'h': 80},
    ],
    'links': [('met', '', 'nz', '###input'), ('met', '', 'x', '###input'),
              ('inn', 'noise', 'mul', 'operand'),
              ('nz', 'random tensor', 'mul', 'in'),
              ('nz', 'random tensor', 'st', 'tensor 4'),
              ('mul', 'result', 'add', 'operand'),
              ('x', 'random tensor', 'neg', 'in'),
              ('x', 'random tensor', 'add', 'in'),
              ('x', 'random tensor', 'st', 'tensor 1'),
              ('neg', 'result', 'st', 'tensor 3'),
              ('add', 'result', 'st', 'tensor 2'),
              ('st', 'output', 'out', 'channels')],
}


# ------------------------------------------------------------------ t.argsort
body = """Gives the positions that would put a tensor in order: the index of
the smallest value first, then the next smallest, and so on.

It does not move any values - it tells you where they are. Use it when
you want to reorder something ELSE by the same ranking: sort joints by
speed, then use the same order for their names or colours; find which
three sensors read highest; rank a set of scores and keep track of whose
they were.

READING THE INDICES:
Positions count from 0. If the first index is 5, the smallest value is
in cell 5 of the input. Reading the input at each index in turn gives the
values sorted.

SYNTAX:
t.argsort <dim: int>
With no argument it ranks along the LAST axis (each row of a table on its
own) and there is no dim inlet. Give an axis - t.argsort 0 - and a dim
box appears so it can be changed.

EXAMPLE:
t.argsort

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or NumPy array is turned into a float32 tensor). Every
input is ranked and sent - the only inlet that makes the node send.

dim (int box, only with an argument):
Which axis to rank along. For a 2-D tensor, -1 or 1 ranks each row
(indices are column numbers), 0 ranks each column (indices are row
numbers).

descending (checkbox):
Largest first. Changing it does not send by itself; the next tensor
uses it.

stable (checkbox):
Equal values keep their original order. Without it, the order of ties is
not guaranteed - which matters when the ranking drives something that
would flicker between two joints that are momentarily equal.

OUTPUTS:

output:
The indices, whole numbers (int64), the same shape as the input.

READING THE DEMO:
t.rand makes 8 values when the patch opens; click its small button for
new ones. The top strip shows them with their values. The middle strip
and the list are the indices from t.argsort: the first names the cell of
the top strip holding the smallest value, the next the cell holding the
next smallest, and so on.

The bottom strip USES those indices: t.take_along_dim reads the values at
the positions it is given, in the order given, so it sends the values
sorted. Its indices inlet does not trigger it - the values arriving at
tensor in do - so the cord into t.argsort was made first, and the
indices are already waiting. Tick descending, click t.rand, and every
strip below the first turns round.

To reorder something ELSE by the same ranking, send the other tensor into
t.take_along_dim instead - the indices still come from the first.

RELATED:
np.argsort is the NumPy version (descending is an option there). np.sort
gives the sorted values directly. t.take_along_dim reads values at the
indices. t.argmax is the first entry of a descending argsort, on its own.
t.flip reverses an index list."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 8', 'pos': (30, 130), 'w': 180, 'h': 160,
     'props': {'shape': '[8]', 'min': 0.0, 'max': 1.0}},
    {'key': 'c0', 'comment': True, 'text': "8 values - click t.rand's button for new ones",
     'pos': (30, 135)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 310), 'w': 540, 'h': 110,
     'props': HM(520, 60, 1)},
    {'key': 'c1', 'comment': True, 'text': 'the values, cells 0 to 7', 'pos': (30, 315)},
    {'key': 'ar', 'init': 't.argsort', 'pos': (30, 440), 'w': 150, 'h': 110,
     'props': {'descending': False, 'stable': False}},
    {'key': 'car', 'comment': True, 'text': 'tick descending, then click t.rand',
     'pos': (30, 445)},
    {'key': 'hidx', 'init': 'heat_map', 'pos': (30, 570), 'w': 540, 'h': 110,
     'props': HM(520, 60, 1, miny=0.0, maxy=7.0, fmt='%.0f', color='greys')},
    {'key': 'c2', 'comment': True,
     'text': 'indices that would order the input smallest to largest',
     'pos': (30, 575)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 700), 'w': 300, 'h': 42, 'props': LIST},
    {'key': 'td', 'init': 't.take_along_dim', 'pos': (30, 760), 'w': 180, 'h': 110},
    {'key': 'ctd', 'comment': True, 'text': 'reads the input at those indices',
     'pos': (30, 765)},
    {'key': 'hso', 'init': 'heat_map', 'pos': (30, 890), 'w': 540, 'h': 110,
     'props': HM(520, 60, 1)},
    {'key': 'c3', 'comment': True, 'text': 'the values read in that order: sorted',
     'pos': (30, 895)},
] + see_also(1030, ['np.argsort', 'np.sort', 't.take_along_dim', 't.argmax', 't.flip'])
links = [('lb', 'out', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hin', 'y'),
         ('rnd', 'random tensor', 'ar', 'tensor in'),
         ('ar', 'output', 'hidx', 'y'), ('ar', 'output', 'l1', ''),
         ('ar', 'output', 'td', 'indices in'),
         ('rnd', 'random tensor', 'td', 'tensor in'),
         ('td', 'output', 'hso', 'y')]
page('t.argsort', 't.argsort - the positions that would sort a tensor', body, demo, links)

# -------------------------------------------------------------------- t.histc
body = """Counts how many values of a tensor fall into each of a number of
equal bins between min and max: a histogram.

A mean says where the middle is; a histogram shows the whole spread -
whether the values cluster, whether there are two groups, how long the
tails are. For anything measured rather than computed, that shape is
usually the interesting thing. Feed it a window of recent values (from
np.rolling_buffer, say) and it becomes a picture of what a movement has
actually been doing: a mean stays put while a single peak splits in two.

SYNTAX:
t.histc
No arguments: set the bins and the range on the node.

EXAMPLE:
t.histc

INPUTS and PARAMETERS:

(first inlet):
The values, any shape - they are all counted together. A list or NumPy
array is turned into a float32 tensor. Every input is counted and sent -
the only inlet that makes the node send. The values must be floating
point: an integer tensor (t.rand with dtype int64, say) is refused with
an error. Send it through t.to float32 first.

bin count (drag int):
How many bins. Default 100, at least 1.

min / max (drag floats):
The range the bins span, cut into equal widths. Values below min or above
max are NOT counted at all - a histogram that seems to be missing data
usually has its range set too narrow. A value exactly at max goes in the
last bin. Defaults 0 and 0: when min and max are equal the node uses the
lowest and highest values of each input instead, so the range follows
the data and two histograms are no longer comparable bin for bin.

Changing any of these does not send by itself; the next input uses it.
In the demo a metro sends fresh values four times a second, so the bars
follow your drags.

OUTPUTS:

histogram tensor out:
The counts, one per bin, lowest bin first, as a 1-D float32 tensor of
length bin count. They add up to the number of values inside the range.

READING THE DEMO:
t.dist.normal draws 1000 values from a bell curve centred on loc (0) with
spread scale (1). The upper plot shows them one after another as dots:
noise, with no visible shape. t.histc puts them in 32 bins from -4 to 4,
each a quarter wide, and the lower plot shows the count in each bin as a
bar: the bell appears, about 100 values high in the middle bins.

Drag scale on t.dist.normal up to 2 and the bell spreads out and gets
lower (the same 1000 values over more bins); past about 2, the tails run
off the edges and are no longer counted. Drag loc to 2 and the bell
slides right. Drag bin count down to 8: coarser bars, each counting more.
Set min and max both to 0 and the bins stretch to fit whatever came in.

RELATED:
t.bincount counts whole numbers exactly, one bin per value, with no
range to set. t.bucketize says WHICH bin each value falls in, with
boundaries you choose. np.rolling_buffer collects a window of a stream
to count. t.cumsum of a histogram gives the cumulative distribution."""

demo = [
    {'key': 'met', 'init': 'metro 250', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': METRO(250)},
    {'key': 'c0', 'comment': True, 'text': 'fresh values four times a second', 'pos': (30, 70)},
    {'key': 'rnd', 'init': 't.dist.normal 1000', 'pos': (30, 150), 'w': 240, 'h': 160,
     'props': {'shape': '[1000]', 'loc': 0.0, 'scale': 1.0}},
    {'key': 'c1', 'comment': True, 'text': '1000 values from a bell curve - drag scale and loc',
     'pos': (30, 155)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 330), 'w': 308, 'h': 178,
     'props': ARRAY_PLOT(-4.0, 4.0, 1000, style='scatter')},
    {'key': 'c2', 'comment': True, 'text': 'the values one by one: no shape to see (-4 to 4)',
     'pos': (30, 335)},
    {'key': 'hc', 'init': 't.histc', 'pos': (30, 530), 'w': 200, 'h': 130,
     'props': {'bin count': 32, 'min': -4.0, 'max': 4.0}},
    {'key': 'c3', 'comment': True, 'text': '32 bins from -4 to 4 - drag bin count, min, max',
     'pos': (30, 535)},
    {'key': 'p2', 'init': 'plot', 'pos': (30, 680), 'w': 308, 'h': 178,
     'props': ARRAY_PLOT(0.0, 150.0, 32, style='bar')},
    {'key': 'c4', 'comment': True, 'text': 'the count in each bin: the bell (0 to 150)',
     'pos': (30, 685)},
] + see_also(890, ['t.bincount', 't.bucketize', 'np.rolling_buffer', 't.dist.normal',
                   't.cumsum'])
links = [('met', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'p1', 'y'),
         ('rnd', 'random tensor', 'hc', ''),
         ('hc', 'histogram tensor out', 'p2', 'y')]
page('t.histc', 't.histc - a histogram: how many values in each bin', body, demo, links)

# ----------------------------------------------------------------- t.bincount
body = """Counts how many times each whole number occurs: the first value of
the output is how many 0s there were, the next how many 1s, and so on up
to the largest value in the input.

Use it when the values ARE categories rather than measurements - labels,
joint ids, note numbers, states, the band numbers t.bucketize gives -
and you want exact counts with no bins or range to choose. It is the
histogram for whole numbers.

SYNTAX:
t.bincount
No arguments.

EXAMPLE:
t.bincount

INPUTS and PARAMETERS:

int tensor in:
The whole numbers, any shape - a 2-D tensor is flattened and all its
values counted together. Every input is counted and sent - the only
inlet that makes the node send. Values that are not whole numbers are
cut toward zero first (2.7 counts as a 2). Lists arrive as floats and
are cut the same way.

NEGATIVE VALUES ARE COUNTED AS 0:
torch.bincount refuses negative numbers, so the node raises them to 0
before counting - every negative value adds one to the count of 0s,
silently. Shift the values up first (+ 5, say) if the input can go below
0 and you want it counted honestly.

OUTPUTS:

bin count tensor out:
The counts, as int64, one per whole number from 0 to the largest value
in the input - so the output is as long as the largest value plus one,
and its length changes when the largest value does. Numbers that never
occur get 0.

READING THE DEMO:
Top: 0 1 1 3 3 3 5 holds one 0, two 1s, no 2s, three 3s, no 4s and one
5, and t.bincount sends exactly that: 1 2 0 3 0 1. Edit the message and
click it.

Bottom: t.rand (dtype int64) makes 2 rows of 2000 whole numbers from 0
to 9 - two throws of a ten-sided die, 2000 times over, a new set four
times a second. (For whole numbers t.rand's max is never reached.)
t.sum 0 adds the two rows: 2000 totals from 0 to 18. The plot shows how
often each total came up, as a bar per number: a triangle peaking at 9,
about 200 times, falling to about 20 at 0 and 18 - there are ten ways to
throw 9 and only one to throw 0. Drag max on t.rand down to 4 and the
output gets SHORTER (totals 0 to 6 only): its length follows the largest
value that occurs.

RELATED:
t.histc counts measurements in equal bins over a range you set.
t.bucketize turns measurements into band numbers you can then count
here. t.argmax of the counts is the most common value. t.rand makes random
whole numbers when its dtype is int64."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 130), 'w': 220, 'h': 42,
     'props': MSG('0 1 1 3 3 3 5')},
    {'key': 'c0', 'comment': True, 'text': 'whole numbers - edit and click', 'pos': (30, 135)},
    {'key': 'bc', 'init': 't.bincount', 'pos': (30, 200), 'w': 160, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (30, 290), 'w': 300, 'h': 42, 'props': LIST},
    {'key': 'c1', 'comment': True, 'text': 'how many 0s, 1s, 2s ... 5s', 'pos': (30, 295)},
    {'key': 'met', 'init': 'metro 250', 'pos': (30, 380), 'w': 129, 'h': 70,
     'props': METRO(250)},
    {'key': 'c2', 'comment': True, 'text': 'new throws four times a second', 'pos': (30, 385)},
    {'key': 'rnd', 'init': 't.rand 2 2000 int64', 'pos': (30, 470), 'w': 240, 'h': 160,
     'props': {'shape': '[2, 2000]', 'min': 0.0, 'max': 10.0, 'dtype': 'int64'}},
    {'key': 'c3', 'comment': True, 'text': 'two dice, 0 to 9, thrown 2000 times - drag max',
     'pos': (30, 475)},
    {'key': 'sm', 'init': 't.sum 0', 'pos': (30, 650), 'w': 150, 'h': 110},
    {'key': 'csm', 'comment': True, 'text': 'the two throws added: totals 0 to 18',
     'pos': (30, 655)},
    {'key': 'bc2', 'init': 't.bincount', 'pos': (30, 780), 'w': 160, 'h': 70},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 870), 'w': 308, 'h': 178,
     'props': ARRAY_PLOT(0.0, 300.0, 19, style='bar')},
    {'key': 'c4', 'comment': True, 'text': 'how often each total came up (0 to 300)',
     'pos': (30, 875)},
] + see_also(1080, ['t.histc', 't.bucketize', 't.argmax', 't.rand'])
links = [('lb', 'out', 'm1', ''), ('m1', 'message out', 'bc', 'int tensor in'),
         ('bc', 'bin count tensor out', 'l1', ''),
         ('met', '', 'rnd', '###input'), ('rnd', 'random tensor', 'sm', 'tensor in'),
         ('sm', 'output', 'bc2', 'int tensor in'),
         ('bc2', 'bin count tensor out', 'pl', 'y')]
page('t.bincount', 't.bincount - how many of each whole number', body, demo, links)

# ---------------------------------------------------------------- t.bucketize
body = """Says which band each value falls in, given the boundaries between the
bands: 0 for below the first boundary, 1 for between the first and the
second, and so on up to the number of boundaries.

It is quantising with boundaries you choose. Use it when the bands mean
something - 'still', 'moving', 'fast' at speeds that matter - rather than
being equal slices of a range. Its output can go on into t.bincount to
count how much time was spent in each band: a histogram of unequal,
meaningful bins.

SYNTAX:
t.bucketize
No arguments.

EXAMPLE:
t.bucketize

INPUTS and PARAMETERS:

tensor in:
The values, any shape. A single number works too. Every input is
assigned and sent - the only inlet that makes the node send - but only
once boundaries have arrived; until then nothing is sent.

boundaries tensor in:
The boundaries, a 1-D list in INCREASING order (out of order gives
wrong answers, with no error). They are kept until replaced, and
converted to the type of the input - so with an integer input, boundaries
like 2.5 are cut to 2.

right (checkbox):
Which band a value exactly ON a boundary goes in. Off (the default): the
band below - band i holds values above boundary i-1 up to and including
boundary i. On: the band above.

int32 indices (checkbox):
Send the band numbers as int32 instead of int64.

Changing either checkbox does not send by itself; the next input uses
it. The metro in the demo re-sends, so the effect shows at once.

OUTPUTS:

bin count tensor out:
The band number of each value, whole numbers from 0 to the number of
boundaries, the same shape as the input. (Despite the name, it is not a
count.)

READING THE DEMO:
The boundaries are 2 3 8, which make four bands: up to 2, above 2 up to
3, above 3 up to 8, above 8. The top strip is 0 to 10; the strip under
it is each value's band. 0 1 2 are band 0, 3 is band 1, 4 to 8 band 2, 9
and 10 band 3 - the values sitting exactly on 2, 3 and 8 go to the band
below. Tick right and they move up: 2 joins band 1, 3 and 8 band 2 and 3.
Edit the boundaries message (they must increase) and click it.

Below, the slider sends single values from 0 to 10 into a second
t.bucketize with the same boundaries, and the number shows the band it
is in - the everyday use: which state is this reading in now?

RELATED:
t.bincount counts the band numbers. t.histc counts values in equal bins.
t.round and t.clamp are simpler ways to quantise or limit values."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'mb', 'init': 'message', 'pos': (30, 130), 'w': 140, 'h': 42,
     'props': MSG('2 3 8')},
    {'key': 'c0', 'comment': True, 'text': 'the boundaries - edit and click', 'pos': (30, 135)},
    {'key': 'met', 'init': 'metro 200', 'pos': (30, 200), 'w': 129, 'h': 70,
     'props': METRO(200)},
    {'key': 'c1', 'comment': True, 'text': 're-sends the values five times a second',
     'pos': (30, 205)},
    {'key': 'ls', 'init': 't.linspace 0 10 11', 'pos': (30, 290), 'w': 220, 'h': 150,
     'props': {'start': 0.0, 'stop': 10.0, 'steps': 11}},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 460), 'w': 540, 'h': 110,
     'props': HM(520, 60, 1, miny=0.0, maxy=10.0, fmt='%.0f')},
    {'key': 'c2', 'comment': True, 'text': 'in: 0 to 10', 'pos': (30, 465)},
    {'key': 'bz', 'init': 't.bucketize', 'pos': (30, 590), 'w': 220, 'h': 110,
     'props': {'int32 indices': False, 'right': False}},
    {'key': 'c3', 'comment': True, 'text': 'tick right: values ON a boundary move up',
     'pos': (30, 595)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 720), 'w': 540, 'h': 110,
     'props': HM(520, 60, 1, miny=0.0, maxy=3.0, fmt='%.0f')},
    {'key': 'c4', 'comment': True, 'text': 'out: the band of each value, 0 to 3',
     'pos': (30, 725)},
    {'key': 'sl', 'init': 'slider', 'pos': (30, 860), 'w': 260, 'h': 60,
     'props': {'': 5.0, 'min': 0.0, 'max': 10.0, 'format': '%.2f', 'width': 240}},
    {'key': 'c5', 'comment': True, 'text': 'one value at a time', 'pos': (30, 865)},
    {'key': 'bz2', 'init': 't.bucketize', 'pos': (30, 940), 'w': 220, 'h': 110,
     'props': {'int32 indices': False, 'right': False}},
    {'key': 'iv', 'init': 'int', 'pos': (30, 1070), 'w': 110, 'h': 42, 'props': INT},
    {'key': 'c6', 'comment': True, 'text': 'the band it is in', 'pos': (30, 1075)},
] + see_also(1150, ['t.bincount', 't.histc', 't.round', 't.clamp'])
links = [('lb', 'out', 'mb', ''),
         ('mb', 'message out', 'bz', 'boundaries tensor in'),
         ('mb', 'message out', 'bz2', 'boundaries tensor in'),
         ('met', '', 'ls', '###input'),
         ('ls', 'linspace out', 'hin', 'y'), ('ls', 'linspace out', 'bz', 'tensor in'),
         ('bz', 'bin count tensor out', 'hout', 'y'),
         ('sl', '', 'bz2', 'tensor in'), ('bz2', 'bin count tensor out', 'iv', '')]
page('t.bucketize', 't.bucketize - which band each value falls in', body, demo, links)

# ------------------------------------------------------------------- t.cumsum
body = """Replaces each value of a tensor with the running total up to and
including it: the first value, then the first two added, then the first
three, and so on.

A sum gives you the total; a cumulative sum gives you the total SO FAR at
every point. Use it to turn steps into a path (movements into positions,
speeds into distance travelled), to turn a histogram into a cumulative
distribution, or to turn a list of durations into start times.

SYNTAX:
t.cumsum <dim: int>
Default dim 0, the FIRST axis. For a 1-D tensor that is the only axis;
for a 2-D tensor it runs DOWN each column - unlike np.cumsum, which runs
along each row by default. Give an axis - t.cumsum 1, or -1 for the last
- and a dim box appears so it can be changed.

EXAMPLE:
t.cumsum

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or NumPy array is turned into a float32 tensor). Every
input is summed and sent - the only inlet that makes the node send. An
axis the tensor does not have sends nothing.

dim (int box, only with an argument):
Which axis to run along. Negative counts from the end. Changing it does
not send by itself; the next tensor uses it.

OUTPUTS:

output:
The running totals, the same shape as the input. The last value is the
total of the whole run (what t.sum would give for a 1-D tensor).

READING THE DEMO:
Top: t.rand makes 200 random steps between -1 and 1, a new set every
half second. The upper plot shows them - noise, with no shape. The lower
plot is their cumsum: a random walk, the path you get by taking those
steps one after another. Drag t.rand's max up to 1.2 (or min down) and
the steps are no longer balanced: the walk drifts steadily up (or down).

Bottom: ten 1s summed give the ramp 1 2 3 ... 10 - the cumsum of a
constant is a straight line whose slope is the constant. Edit the
message (2 2 2, or 1 -1 1 -1) and click it.

RELATED:
np.cumsum is the NumPy version. t.diff is the reverse: it turns a path
back into its steps. t.cumprod multiplies instead of adding. t.sum gives
only the final total. accumulate keeps a running total of a STREAM, one
number at a time."""

demo = [
    {'key': 'met', 'init': 'metro 500', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': METRO(500)},
    {'key': 'c0', 'comment': True, 'text': 'new steps twice a second', 'pos': (30, 70)},
    {'key': 'rnd', 'init': 't.rand 200', 'pos': (30, 150), 'w': 190, 'h': 160,
     'props': {'shape': '[200]', 'min': -1.0, 'max': 1.0}},
    {'key': 'c1', 'comment': True, 'text': '200 steps between min and max - drag them',
     'pos': (30, 155)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 330), 'w': 308, 'h': 178,
     'props': ARRAY_PLOT(-1.5, 1.5, 200)},
    {'key': 'c2', 'comment': True, 'text': 'the steps: noise', 'pos': (30, 335)},
    {'key': 'cs', 'init': 't.cumsum', 'pos': (30, 530), 'w': 150, 'h': 70},
    {'key': 'p2', 'init': 'plot', 'pos': (30, 620), 'w': 308, 'h': 178,
     'props': ARRAY_PLOT(-30.0, 30.0, 200)},
    {'key': 'c3', 'comment': True, 'text': 'their running total: a random walk (-30 to 30)',
     'pos': (30, 625)},
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 840), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 900), 'w': 300, 'h': 42,
     'props': MSG('1 1 1 1 1 1 1 1 1 1')},
    {'key': 'c4', 'comment': True, 'text': 'ten 1s - edit and click', 'pos': (30, 905)},
    {'key': 'h1', 'init': 'heat_map', 'pos': (30, 960), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, miny=0.0, maxy=10.0, fmt='%.0f')},
    {'key': 'c5', 'comment': True, 'text': 'in: ten 1s', 'pos': (30, 965)},
    {'key': 'cs2', 'init': 't.cumsum', 'pos': (30, 1090), 'w': 150, 'h': 70},
    {'key': 'h2', 'init': 'heat_map', 'pos': (30, 1180), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, miny=0.0, maxy=10.0, fmt='%.0f')},
    {'key': 'c6', 'comment': True, 'text': 'out: 1 to 10, a ramp', 'pos': (30, 1185)},
] + see_also(1320, ['np.cumsum', 't.diff', 't.cumprod', 't.sum', 'accumulate'])
links = [('met', '', 'rnd', '###input'), ('rnd', 'random tensor', 'p1', 'y'),
         ('rnd', 'random tensor', 'cs', 'tensor in'), ('cs', 'output', 'p2', 'y'),
         ('lb', 'out', 'm1', ''), ('m1', 'message out', 'h1', 'y'),
         ('m1', 'message out', 'cs2', 'tensor in'), ('cs2', 'output', 'h2', 'y')]
page('t.cumsum', 't.cumsum - the running total', body, demo, links)

# ------------------------------------------------------------------ t.cumprod
body = """Replaces each value of a tensor with the running product up to and
including it: the first value, then the first two multiplied, then the
first three, and so on.

It is cumsum with multiplying instead of adding. Use it for anything that
compounds: growth by a rate per step (interest, a population, a pitch
rising by a fixed ratio per note), decay by a factor per step (an echo
fading), or a chain of gains or scale factors applied one after another.

SYNTAX:
t.cumprod <dim: int>
Default dim 0, the FIRST axis - for a 2-D tensor, down each column
(np.cumprod runs along rows). Give an axis as the argument and a dim box
appears so it can be changed.

EXAMPLE:
t.cumprod

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or NumPy array is turned into a float32 tensor). Every
input is multiplied out and sent - the only inlet that makes the node
send. An axis the tensor does not have sends nothing.

dim (int box, only with an argument):
Which axis to run along. Negative counts from the end.

OUTPUTS:

output:
The running products, the same shape as the input. A single 0 anywhere
makes every value after it 0. A float32 tensor keeps about seven
significant digits, so long products round; an int64 tensor (from t.rand
with dtype int64) stays whole but silently overflows past about 9 x 10
to the 18th - beyond 20 factorial.

READING THE DEMO:
Top: t.full makes 40 copies of its value, 1.05 - a growth rate of 5 per
cent a step. t.stack puts that flat input line and its cumprod into one
2-row tensor, and the plot draws both rows: the flat line at 1.05, and
1.05, then 1.05 times 1.05, ... - compound growth, a curve that bends
upward ever faster and reaches about 7 after 40 steps. Drag value on
t.full: 1.02 grows slowly, 0.95 decays towards 0, 1 stays flat. Above
about 1.055 the curve runs off the top of the plot (0 to 8).

Bottom: 1 2 3 4 5 6 multiplied out gives 1 2 6 24 120 720 - the
factorials.

RELATED:
np.cumprod is the NumPy version. t.cumsum adds instead of multiplying.
t.prod gives only the final product. t.diff is the reverse of cumsum,
not of cumprod (for cumprod, divide each value by the one before)."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': METRO(100)},
    {'key': 'c0', 'comment': True, 'text': 'ten times a second', 'pos': (30, 70)},
    {'key': 'fl', 'init': 't.full 40', 'pos': (30, 150), 'w': 190, 'h': 160,
     'props': {'value': 1.05, 'dim 0': 40}},
    {'key': 'c1', 'comment': True, 'text': '40 copies of the growth rate - drag value',
     'pos': (30, 155)},
    {'key': 'cp', 'init': 't.cumprod', 'pos': (30, 330), 'w': 150, 'h': 70},
    {'key': 'st', 'init': 't.stack', 'pos': (30, 420), 'w': 150, 'h': 110},
    {'key': 'cst', 'comment': True, 'text': 'row 0: the cumprod, row 1: the input',
     'pos': (30, 425)},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 550), 'w': 308, 'h': 178,
     'props': MULTI_PLOT(0.0, 8.0, 40)},
    {'key': 'c3', 'comment': True,
     'text': 'the flat line is the input (1.05 forty times);\nthe curve is its cumprod: compound growth',
     'pos': (30, 740)},
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 810), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 870), 'w': 200, 'h': 42,
     'props': MSG('1 2 3 4 5 6')},
    {'key': 'cp2', 'init': 't.cumprod', 'pos': (30, 930), 'w': 150, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (30, 1020), 'w': 300, 'h': 42, 'props': LIST},
    {'key': 'c4', 'comment': True, 'text': '1 x 2 x 3 ...: the factorials', 'pos': (30, 1025)},
] + see_also(1100, ['np.cumprod', 't.cumsum', 't.prod', 't.full'])
links = [('met', '', 'fl', '###input'),
         ('fl', 'filled tensor', 'st', 'tensor 2'),
         ('fl', 'filled tensor', 'cp', 'tensor in'), ('cp', 'output', 'st', 'tensor 1'),
         ('st', 'output', 'pl', 'y'),
         ('lb', 'out', 'm1', ''), ('m1', 'message out', 'cp2', 'tensor in'),
         ('cp2', 'output', 'l1', '')]
page('t.cumprod', 't.cumprod - the running product', body, demo, links)


# ------------------------------------------------------------- t.cummax/cummin
def cum_extreme(name, word, other, stat, see):
    up = word == 'largest'
    record = 'high' if up else 'low'
    body = f"""Replaces each value of a tensor with the {word} value seen so far:
a running {'maximum' if up else 'minimum'}, the {record}-water mark.

Running it along time gives, at every instant, the {word} value that
has occurred up to then - {'a peak hold' if up else 'a floor'} that never lets go. It
answers 'has this ever {'exceeded' if up else 'dropped below'} X, and when did it first
do so?', and its second outlet says WHERE each record was set.

SYNTAX:
{name} <dim: int>
Default dim 0, the FIRST axis - for a 2-D tensor, down each column.
Give an axis as the argument - {name} 1, or -1 for the last - and a dim
box appears so it can be changed.

EXAMPLE:
{name}

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or NumPy array is turned into a float32 tensor). Every
input is scanned and sent - the only inlet that makes the node send. An
axis the tensor does not have sends nothing.

dim (int box, only with an argument):
Which axis to run along. Negative counts from the end.

OUTPUTS:

output:
The running {'maximum' if up else 'minimum'}, the same shape as the input. It can only
{'rise or stay level' if up else 'fall or stay level'}, never {'fall' if up else 'rise'}.

indices:
For each position, WHERE the {word} value so far was found (an int64
position along the axis), the same shape as the input. It stays put
while the record stands and jumps forward each time a new one is set.
With equal values it points at the later one. The indices are sent just
before the output.

READING THE DEMO:
make_walk sends a random walk of 200 steps twice a second (open it to
see how). t.stack puts {name}'s output and the walk itself into one
2-row tensor, and the upper plot draws both: the walk wanders, and the
{name} line {'climbs' if up else 'steps down'} in a staircase - level while the walk is
{'below its best' if up else 'above its lowest'}, stepping {'up' if up else 'down'} each time the walk sets a new {record}.
At the start the two lines are one.

The lower plot is the indices outlet: the position, 0 to 199, of the
current record. Each flat stretch is a record that stood; each jump is
a new {record} being set.

RELATED:
{other} is the running {'minimum' if up else 'maximum'}. {stat} gives only the final
{'largest' if up else 'smallest'} value (and, along an axis, where it is). t.cumsum is the running total.
t.logcumsumexp is a smooth version of the running maximum."""
    demo = [
        {'key': 'wk', 'init': 'p make_walk', 'pos': (30, 62), 'w': 170, 'h': 70},
        {'key': 'c0', 'comment': True, 'text': 'a random walk twice a second - open it to see how',
         'pos': (30, 70)},
        {'key': 'cm', 'init': name, 'pos': (30, 160), 'w': 150, 'h': 90},
        {'key': 'st', 'init': 't.stack', 'pos': (30, 270), 'w': 150, 'h': 110},
        {'key': 'cst', 'comment': True, 'text': f'row 0: the running {record}, row 1: the walk',
         'pos': (30, 275)},
        {'key': 'p1', 'init': 'plot', 'pos': (30, 400), 'w': 308, 'h': 178,
         'props': MULTI_PLOT(-20.0, 20.0, 200)},
        {'key': 'c1', 'comment': True,
         'text': f'the walk, and the staircase of its {record}s so far\n(-20 to 20)',
         'pos': (30, 405)},
        {'key': 'p2', 'init': 'plot', 'pos': (30, 600), 'w': 308, 'h': 178,
         'props': ARRAY_PLOT(0.0, 200.0, 200, style='stair')},
        {'key': 'c2', 'comment': True,
         'text': 'indices: where the current record was set\n(0 to 200)',
         'pos': (30, 605)},
    ] + see_also(810, see)
    links = [('wk', 'walk', 'st', 'tensor 2'),
             ('wk', 'walk', 'cm', 'tensor in'),
             ('cm', 'output', 'st', 'tensor 1', 0),
             ('cm', 'indices', 'p2', 'y', 1),
             ('st', 'output', 'p1', 'y')]
    title = f"{name} - the running {'maximum' if up else 'minimum'}"
    page(name, title, body, demo, links, sub=WALK_SUB)


cum_extreme('t.cummax', 'largest', 't.cummin', 't.max',
            ['t.cummin', 't.max', 't.cumsum', 't.logcumsumexp'])
cum_extreme('t.cummin', 'smallest', 't.cummax', 't.min',
            ['t.cummax', 't.min', 't.cumsum', 't.logcumsumexp'])

# ------------------------------------------------------------- t.logcumsumexp
body = """A running total done in the log domain: each output value is the log
of (e to the power of each value so far, all added up).

Two uses. First, when the values are LOG probabilities (or log
likelihoods, log energies), it gives the log of the running total of the
underlying amounts, without ever turning them back into ordinary numbers
- which would underflow to 0 for very small probabilities, or overflow
to infinity for large values. Second, it behaves like a SMOOTH running
maximum: once one value is well above the rest, the result follows it,
lifted a little by its near rivals.

HOW IT RELATES TO THE MAXIMUM:
The result is never below the largest value so far, and never more than
the log of the count above it. Values far below the leader add almost
nothing; values tied with it add log 2, log 3 ... So it sits on, or just
above, t.cummax's staircase, with the corners rounded off.

SYNTAX:
t.logcumsumexp <dim: int>
Default dim 0, the FIRST axis - for a 2-D tensor, down each column. Give
an axis as the argument and a dim box appears so it can be changed.

EXAMPLE:
t.logcumsumexp

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or NumPy array is turned into a float32 tensor). Every
input is processed and sent - the only inlet that makes the node send.
It must be floating point: an integer tensor is refused with an error.
An axis the tensor does not have sends nothing.

dim (int box, only with an argument):
Which axis to run along. Negative counts from the end.

OUTPUTS:

output:
The running log-sum-exp, the same shape as the input. It only ever
rises: each new value adds something positive inside the log.

READING THE DEMO:
Top: make_walk sends a random walk of 200 steps twice a second. t.stack
puts three rows together and the plot draws them: the walk, t.cummax of
the walk (its running maximum, a staircase), and t.logcumsumexp of the
walk. The logcumsumexp line rides on top of the staircase - right on it
where the walk climbs to a fresh high, a little above it where the walk
hangs around near its best, since each near-equal value adds to the
total.

Bottom: eight 0s give 0 0.69 1.10 1.39 1.61 1.79 1.95 2.08 - the logs
of 1, 2, 3 ... 8, since e to the 0 is 1 and the running total of 1s is
the count. Then 1000 1000 1000: e to the 1000 is far too large for any
float, yet the answer comes out exactly as 1000, 1000.69, 1001.10 -
1000 plus the logs of 1, 2, 3. Click either message.

RELATED:
t.cumsum is the plain running total. t.cummax is the hard-edged running
maximum. t.nn.log_softmax uses the same log-sum-exp across a whole axis
at once."""

demo = [
    {'key': 'wk', 'init': 'p make_walk', 'pos': (30, 62), 'w': 170, 'h': 70},
    {'key': 'c0', 'comment': True, 'text': 'a random walk twice a second - open it to see how',
     'pos': (30, 70)},
    {'key': 'cx', 'init': 't.cummax', 'pos': (250, 160), 'w': 150, 'h': 90},
    {'key': 'ccx', 'comment': True, 'text': 'for comparison', 'pos': (420, 175)},
    {'key': 'lc', 'init': 't.logcumsumexp', 'pos': (30, 160), 'w': 180, 'h': 70},
    {'key': 'st', 'init': 't.stack 3', 'pos': (30, 280), 'w': 150, 'h': 130},
    {'key': 'cst', 'comment': True,
     'text': 'row 0: logcumsumexp, 1: running max, 2: the walk',
     'pos': (30, 285)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 430), 'w': 308, 'h': 178,
     'props': MULTI_PLOT(-20.0, 20.0, 200)},
    {'key': 'c1', 'comment': True,
     'text': 'the walk, its running maximum, and logcumsumexp\nriding just above it (-20 to 20)',
     'pos': (30, 435)},
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 650), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 710), 'w': 220, 'h': 42,
     'props': MSG('0 0 0 0 0 0 0 0')},
    {'key': 'c2', 'comment': True, 'text': 'eight 0s - click', 'pos': (30, 715)},
    {'key': 'm2', 'init': 'message', 'pos': (270, 710), 'w': 200, 'h': 42,
     'props': MSG('1000 1000 1000')},
    {'key': 'c3', 'comment': True, 'text': 'or these', 'pos': (490, 715)},
    {'key': 'lc2', 'init': 't.logcumsumexp', 'pos': (30, 780), 'w': 180, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (30, 870), 'w': 420, 'h': 42, 'props': LIST},
    {'key': 'c4', 'comment': True, 'text': 'the logs of 1, 2, 3 ... (plus 1000)',
     'pos': (30, 875)},
] + see_also(950, ['t.cumsum', 't.cummax', 't.nn.log_softmax'])
links = [('wk', 'walk', 'st', 'tensor 3'),
         ('wk', 'walk', 'cx', 'tensor in'),
         ('wk', 'walk', 'lc', 'tensor in'),
         ('cx', 'output', 'st', 'tensor 2'),
         ('lc', 'output', 'st', 'tensor 1'),
         ('st', 'output', 'p1', 'y'),
         ('lb', 'out', 'm1', ''), ('m1', 'message out', 'lc2', 'tensor in'),
         ('m2', 'message out', 'lc2', 'tensor in'),
         ('lc2', 'output', 'l1', '')]
page('t.logcumsumexp', 't.logcumsumexp - a running total in the log domain', body, demo,
     links, sub=WALK_SUB)

# --------------------------------------------------------------------- t.diff
body = """Replaces a tensor with the differences between neighbouring values:
each value minus the one before it.

It turns positions into steps - where something WAS into how far it
MOVED between samples. Applied twice it gives the change in the change:
from positions, n 1 is speed and n 2 is acceleration. Use it to find
where a sequence jumps, to get velocities from a recorded path, or to
undo a cumsum.

THE OUTPUT IS SHORTER:
Nine values have only eight gaps between them, so the result is one
value shorter than the input along the axis - and each further n loses
one more. Anything expecting the original length will notice.

SYNTAX:
t.diff <dim: int>
With no argument it differences along the LAST axis (along each row of
a table) and there is no dim inlet. Give an axis as the argument - t.diff
0 - and a dim box appears so it can be changed. Note the argument is the
AXIS, not n (np.diff's first argument is the order).

EXAMPLE:
t.diff

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or NumPy array is turned into a float32 tensor). Every
input is differenced and sent - the only inlet that makes the node send.

n (int box):
How many times to difference. Default 1. 0 sends the input unchanged. If
the tensor is not longer than n along the axis, an EMPTY tensor is sent.
Negative n is refused with an error, and nothing is sent. Changing it
does not send by itself - in the demo a metro re-sends the input five
times a second.

dim (int box, only with an argument):
Which axis to difference along. An axis the tensor does not have sends
nothing.

OUTPUTS:

tensor out:
The differences, shorter by n along the axis.

READING THE DEMO:
Top: 0 1 4 9 16 25 36 49 64 are the positions of something speeding up
(the squares). n 1 gives 1 3 5 7 9 11 13 15 - the distance covered in
each step, growing steadily: the speed. Type 2 into n and you get
2 2 2 2 2 2 2 - the speed grows by the same amount every step, a
constant acceleration. n 3 gives zeros: the acceleration never changes.

Bottom, the inverse: 3 1 4 1 5 9 summed by t.cumsum is the path
3 4 8 9 14 23, and t.diff of that gives back 1 4 1 5 9 - the steps, all
but the first, which had nothing before it to differ from. To rebuild a
path from its steps exactly you also need where it started.

RELATED:
np.diff is the NumPy version. t.cumsum is the reverse: steps back into a
path. t.energy adds up the sizes of the differences into one number.
diff does the same for a STREAM of numbers, one sample at a time."""

demo = [
    {'key': 'met', 'init': 'metro 200', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': METRO(200)},
    {'key': 'c0', 'comment': True, 'text': 're-sends the positions five times a second',
     'pos': (30, 70)},
    {'key': 'm1', 'init': 'message', 'pos': (30, 150), 'w': 300, 'h': 42,
     'props': MSG('0 1 4 9 16 25 36 49 64')},
    {'key': 'c1', 'comment': True, 'text': 'positions of something speeding up',
     'pos': (30, 155)},
    {'key': 'h1', 'init': 'heat_map', 'pos': (30, 210), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, miny=0.0, maxy=64.0, fmt='%.0f')},
    {'key': 'c2', 'comment': True, 'text': 'in: 9 positions', 'pos': (30, 215)},
    {'key': 'df', 'init': 't.diff', 'pos': (30, 340), 'w': 150, 'h': 90,
     'props': {'n': 1}},
    {'key': 'c3', 'comment': True, 'text': 'type 1, 2 or 3 into n', 'pos': (30, 345)},
    {'key': 'h2', 'init': 'heat_map', 'pos': (30, 450), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, miny=0.0, maxy=16.0, fmt='%.0f')},
    {'key': 'c4', 'comment': True, 'text': 'out: one value fewer for each n',
     'pos': (30, 455)},
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 600), 'w': 88, 'h': 46},
    {'key': 'm2', 'init': 'message', 'pos': (30, 660), 'w': 200, 'h': 42,
     'props': MSG('3 1 4 1 5 9')},
    {'key': 'c5', 'comment': True, 'text': 'six steps - click to re-send', 'pos': (30, 665)},
    {'key': 'cs', 'init': 't.cumsum', 'pos': (30, 720), 'w': 150, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (30, 810), 'w': 300, 'h': 42, 'props': LIST},
    {'key': 'c6', 'comment': True, 'text': 'the path they make', 'pos': (30, 815)},
    {'key': 'df2', 'init': 't.diff', 'pos': (30, 870), 'w': 150, 'h': 90,
     'props': {'n': 1}},
    {'key': 'l2', 'init': 'list', 'pos': (30, 980), 'w': 300, 'h': 42, 'props': LIST},
    {'key': 'c7', 'comment': True, 'text': 'the steps again, minus the first',
     'pos': (30, 985)},
] + see_also(1060, ['np.diff', 't.cumsum', 't.energy', 'diff'])
links = [('met', '', 'm1', ''), ('m1', 'message out', 'h1', 'y'),
         ('m1', 'message out', 'df', 'tensor in'), ('df', 'tensor out', 'h2', 'y'),
         ('lb', 'out', 'm2', ''), ('m2', 'message out', 'cs', 'tensor in'),
         ('cs', 'output', 'l1', ''), ('cs', 'output', 'df2', 'tensor in'),
         ('df2', 'tensor out', 'l2', '')]
page('t.diff', 't.diff - the differences between neighbours', body, demo, links)

# ------------------------------------------------------------------- t.energy
body = """Adds up how much a tensor CHANGES from one value to the next: the
differences between neighbours, made positive, summed into one number.

It measures the total distance travelled, not how far the values ended
up from where they started. A signal that swings a long way and comes
back has high energy and no overall change. Use it to ask how much
something moved over a stretch of samples - a window of a joint's
positions, a recorded gesture - or, with n 2, how jerky the movement
was: the total change in the change.

(Despite the name, this is not the physicist's energy or the sum of
squares: it is the total variation, the sum of the sizes of the steps.)

SYNTAX:
t.energy <dim: int>
With no argument the differences run along the LAST axis and there is no
dim inlet. Give an axis as the argument - t.energy 0 - and a dim box
appears so it can be changed. Either way the answer is ONE number,
summed over everything.

EXAMPLE:
t.energy

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or NumPy array is turned into a float32 tensor). Every
input is measured and sent - the only inlet that makes the node send.

n (int box):
How many times to difference before summing. 1 (the default) is total
change; 2 is total change in the change. 0 sums the sizes of the values
themselves. A tensor not longer than n along the axis gives 0. Negative n
is refused with an error. Changing it does not send by itself - the
metro in the demo re-sends.

dim (int box, only with an argument):
The axis to take the differences along.

OUTPUTS:

tensor out:
A single number, as a one-value tensor.

READING THE DEMO:
Both messages hold values from 0 to 8, sent five times a second. The
steady climb 0 1 2 ... 8 moves 1 at each of its 8 steps: energy 8, the
same as the distance from start to finish. The zigzag 0 8 0 8 ... moves
8 at each step: energy 64, although it ends where it began. Same range,
same start and end - eight times the travel.

Type 2 into n on both: the climb gives 0 (its speed never changes),
the zigzag 112 (it reverses with all its speed every step).

RELATED:
t.diff gives the differences themselves, before summing. np.diff is
the NumPy difference. t.length is the size of a tensor, not how much it
changes. t.sum adds values without differencing."""

demo = [
    {'key': 'met', 'init': 'metro 200', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': METRO(200)},
    {'key': 'c0', 'comment': True, 'text': 're-sends both five times a second', 'pos': (30, 70)},
    {'key': 'ma', 'init': 'message', 'pos': (30, 150), 'w': 260, 'h': 42,
     'props': MSG('0 1 2 3 4 5 6 7 8')},
    {'key': 'ca', 'comment': True, 'text': 'a steady climb', 'pos': (30, 155)},
    {'key': 'ha', 'init': 'heat_map', 'pos': (30, 210), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, miny=0.0, maxy=8.0, fmt='%.0f')},
    {'key': 'ea', 'init': 't.energy', 'pos': (30, 340), 'w': 150, 'h': 90,
     'props': {'n': 1}},
    {'key': 'fa', 'init': 'float', 'pos': (30, 450), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'cfa', 'comment': True, 'text': 'energy 8: 1 per step', 'pos': (30, 455)},
    {'key': 'mb', 'init': 'message', 'pos': (30, 540), 'w': 260, 'h': 42,
     'props': MSG('0 8 0 8 0 8 0 8 0')},
    {'key': 'cb', 'comment': True, 'text': 'a zigzag over the same range', 'pos': (30, 545)},
    {'key': 'hb', 'init': 'heat_map', 'pos': (30, 600), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, miny=0.0, maxy=8.0, fmt='%.0f')},
    {'key': 'eb', 'init': 't.energy', 'pos': (30, 730), 'w': 150, 'h': 90,
     'props': {'n': 1}},
    {'key': 'fb', 'init': 'float', 'pos': (30, 840), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'cfb', 'comment': True, 'text': 'energy 64: 8 per step', 'pos': (30, 845)},
] + see_also(920, ['t.diff', 'np.diff', 't.length', 't.sum'])
links = [('met', '', 'ma', ''), ('met', '', 'mb', ''),
         ('ma', 'message out', 'ha', 'y'), ('ma', 'message out', 'ea', 'tensor in'),
         ('ea', 'tensor out', 'fa', ''),
         ('mb', 'message out', 'hb', 'y'), ('mb', 'message out', 'eb', 'tensor in'),
         ('eb', 'tensor out', 'fb', '')]
page('t.energy', 't.energy - how much a tensor changes', body, demo, links)

# ----------------------------------------------------------------- t.corrcoef
body = """Measures how much the rows of a tensor move TOGETHER: feed it rows of
samples (one row per channel) and it returns a square table with the
correlation of every pair of rows.

   1   the two rise and fall together
   0   no straight-line relation
  -1   one rises exactly as the other falls

The diagonal is always 1 - every row moves exactly with itself - and the
table is symmetric, since A with B is B with A. Correlation removes each
row's average and its scale before comparing, so two channels that
wobble in step correlate fully even if one is far bigger than the other
or sits far from zero (unlike t.cosine_similarity). Over a long
recording of several joints or sensors, the table is a measured picture
of which things habitually happen at once.

SYNTAX:
t.corrcoef
No arguments.

EXAMPLE:
t.corrcoef

INPUTS and PARAMETERS:

tensor in:
A 2-D tensor: rows are channels, columns are samples (a list or NumPy
array is turned into a float32 tensor). If your data has one channel per
COLUMN, as a rolling buffer of vectors does, turn it round with t.t
first. Every input is measured and sent - the only inlet.

A 1-D tensor counts as a single row and gives 1. A row that never
changes has no correlation with anything: its row and column come out
as nan (not a number).

FEW SAMPLES, LOUD CORRELATIONS:
With only a handful of columns, chance alone produces sizeable
correlations between unrelated rows. Give it many samples before
trusting a number.

OUTPUTS:

tensor out:
The table of correlations, one row and one column per input row, each
between -1 and 1.

READING THE DEMO:
make_channels sends four channels of 64 samples five times a second, as
a 4 x 64 tensor (upper picture, one row per channel):
  row 0: random values
  row 1: row 0 plus some of row 3 - the slider sets how much
  row 2: row 0 upside down
  row 3: other random values
The lower picture is the 4 x 4 table (dark blue is +1, dark red -1,
pale is 0). Row 0 with row 2 is -1: one exactly mirrors the other. Row 0 with
row 1 is high but below 1 (about 0.9 with the slider at 0.5), and row 1
also correlates with row 3, whose noise it carries (about 0.45). Row 3 with rows
0 and 2 is near 0 - but it flickers around 0 by 0.1 or more, which is
chance with only 64 samples.

Drag the slider to 0 and row 1 becomes an exact copy: 1 with row 0, -1
with row 2. Drag it to 2 and row 1 is mostly noise: its correlation with
row 0 falls to about 0.45, with row 3 rises to about 0.9.

RELATED:
t.cosine_similarity compares direction without removing the average.
t.mean and t.sum summarise one channel. t.linalg.pca_low_rank finds the
directions the channels vary along together. t.stack builds rows into
one tensor."""

demo = [
    {'key': 'sl', 'init': 'slider', 'pos': (30, 62), 'w': 260, 'h': 60,
     'props': {'': 0.5, 'min': 0.0, 'max': 2.0, 'format': '%.2f', 'width': 240}},
    {'key': 'csl', 'comment': True, 'text': 'how much noise is mixed into row 1',
     'pos': (30, 70)},
    {'key': 'ch', 'init': 'p make_channels', 'pos': (30, 150), 'w': 200, 'h': 70},
    {'key': 'cch', 'comment': True, 'text': 'four channels, 64 samples each - open it to see how',
     'pos': (30, 155)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 250), 'w': 540, 'h': 170,
     'props': HM(520, 120, 4, -1.0, 1.0, fmt='none')},
    {'key': 'c1', 'comment': True, 'text': 'in: rows 0 to 3, top to bottom', 'pos': (30, 255)},
    {'key': 'cc', 'init': 't.corrcoef', 'pos': (30, 440), 'w': 160, 'h': 70},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 530), 'w': 340, 'h': 210,
     'props': HM(320, 160, 4, -1.0, 1.0, fmt='%.2f', color='red-blue')},
    {'key': 'c2', 'comment': True,
     'text': 'out: row i, column j is how rows i and j move\ntogether - blue +1, pale 0, red -1',
     'pos': (30, 535)},
] + see_also(780, ['t.cosine_similarity', 't.mean', 't.linalg.pca_low_rank', 't.stack'])
links = [('sl', '', 'ch', 'noise'),
         ('ch', 'channels', 'hin', 'y'), ('ch', 'channels', 'cc', 'tensor in'),
         ('cc', 'tensor out', 'hout', 'y')]
page('t.corrcoef', 't.corrcoef - how the rows move together', body, demo, links,
     sub=CHANNELS_SUB)
