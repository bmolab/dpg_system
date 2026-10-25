"""Seed the help pages for the fourth batch of numpy nodes (2026-10-24): one page per node.

    python3 dpg_system/help/tools/seed_numpy_batch4_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

np.real, np.imag, np.angle, np.unwrap, np.nanmin, np.nanmax, np.nanstd,
np.nanvar, np.average, np.histogram2d, np.top_k, np.isclose, np.allclose,
np.trapezoid, np.cumulative_trapezoid, np.fft.fft, np.fft.ifft.

Like the earlier seeders it is not a make_* script, so rebuild.py never runs it,
and it will not write over an existing page without --force. Treat the pages as
hand-made once written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from seed_numpy_more_help import LIST, MSG, HM, BARS, C, N, see_also
from seed_numpy_batch2_help import METRO_SLOW, metro_row


PAGES = {}


def page(stem, title, body, demo, links, also, also_y, text_height=820):
    PAGES[stem] = dict(title=title, body=body.strip('\n'), demo=demo + see_also(also_y, also),
                       links=links, text_height=text_height)


# ------------------------------------------------------- real / imag / angle
PARTS = {
    'np.real': ('the real part of complex numbers', 'real', '1 0 -1 0',
                """
Gives the real part of each complex number: the a in a + b j, the horizontal
coordinate when the number is drawn as a point.

Complex numbers come out of np.fft.fft (each frequency's amount and phase together),
np.fft.rfft's spectrum, and np.linalg.eig for a matrix that turns. np.real gets an
ordinary array back out: the in-phase (cosine) part of a spectrum, or a signal from
np.fft.ifft whose imaginary part is just rounding left over. Real numbers pass
through unchanged.""",
                'the real parts'),
    'np.imag': ('the imaginary part of complex numbers', 'imag', '0 -1 0 1',
                """
Gives the imaginary part of each complex number: the b in a + b j, the vertical
coordinate when the number is drawn as a point.

Complex numbers come out of np.fft.fft (each frequency's amount and phase together),
np.fft.rfft's spectrum, and np.linalg.eig for a matrix that turns. np.imag gets the
quadrature (sine) part of a spectrum, or checks how far from real a result is. For
real numbers it is all zeros.""",
                'the imaginary parts'),
    'np.angle': ('the angle of complex numbers', 'angle', '0 -90 180 90',
                 """
Gives the angle of each complex number: the direction from the origin to the point a
+ b j, measured counter-clockwise from the positive real axis. It is the phase of a
frequency in a spectrum - where in its cycle that wave starts.

Together with abs (the length, the magnitude) it is the polar form of a complex
number, as atan2 and hypot are for a point (x, y). Real numbers have angle 0, or 180
degrees (pi) when negative.""",
                 'angles, in degrees'),
}

for stem, (gist, part, result, intro, note) in PARTS.items():
    angle = stem == 'np.angle'
    summary = 'Their angles, in degrees, are' if angle else f'Their {part} parts are'
    page(stem, f'{stem} - {gist}',
f"""{intro}

SYNTAX:
{stem}

INPUTS and PARAMETERS:

in:
A number, list or array, complex or not. Every input is converted and sent.
""" + ("""
degrees:
An option, under the node. Off (the default) gives radians, -pi to pi, as NumPy does;
on gives degrees, -180 to 180. The demo has it on.
""" if angle else '') + f"""
OUTPUTS:

{part}:
The {part}{' of each number' if angle else ' part of each number'}, the same shape as the input.

READING THE DEMO:
The spectrum of 0 1 0 0 - a single pulse one step in - is 1, -1j, -1 and 1j: four
complex numbers of length 1, each a quarter turn further round. {summary} {result}.
A metro re-sends the pulse twice a second.

RELATED:
np.real, np.imag and np.angle split a complex array; abs gives each number's length
(its magnitude). np.fft.fft and np.linalg.eig make complex arrays. atan2 and hypot are
the same pair of ideas for a point given as x and y.
""",
    metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
        C('c1', 'a pulse, one step in', 180, 53),
        N('m1', 'message', 180, 92, 140, 42, MSG('0 1 0 0')),
        C('c2', 'its spectrum: 1 -1j -1 1j', 360, 53),
        N('ff', 'np.fft.fft', 360, 92, 180, 150),
        N('ls', 'list', 360, 270, 260, 42, LIST),
        C('c3', note, 580, 53),
        N('pt', stem, 580, 92, 130, 70, {'degrees': True} if angle else None),
        N('lp', 'list', 580, 180, 200, 42, LIST),
    ],
    [('mt', '', 'm1', ''), ('m1', 'message out', 'ff', 'in'), ('ff', 'spectrum', 'ls', ''),
     ('ff', 'spectrum', 'pt', 'in'), ('pt', part, 'lp', '')],
    [s for s in ['np.real', 'np.imag', 'np.angle'] if s != stem] + ['abs', 'np.fft.fft', 'atan2'],
    420, 760)


# ------------------------------------------------------------------ unwrap
page('np.unwrap', 'np.unwrap - remove the jumps from a run of angles',
"""
Takes a run of angles that jumps back by a whole turn each time it passes 180 (or
pi), and adds or subtracts whole turns so the run continues smoothly instead.

An angle from atan2, a heading, a joint rotation or the phase of a spectrum is
usually given between -180 and 180. Something turning steadily then reads 170, 175,
-180, -175 ... - a jump of almost 360 that is not real movement, and that wrecks any
speed, average or smoothing worked out from it. np.unwrap turns that into 170, 175,
180, 185 ... and keeps counting: two whole turns end at 720, not back at 0.

SYNTAX:
np.unwrap <units: degrees or radians>
Default: degrees.

INPUTS and PARAMETERS:

in:
The run of angles (a list or array). Every input is unwrapped and sent.

axis (int box, also an inlet):
Which way the run goes: -1 (the default) along each row, 0 down each column - one
run per joint for a buffer of frames x joints.

units:
An option, under the node: degrees (the default here, a turn of 360) or radians (a
turn of 2 pi, NumPy's own default).

OUTPUTS:

unwrapped:
The same angles with the jumps taken out. The first value is unchanged.

HOW IT DECIDES:
Any step between neighbours of more than half a turn is taken to be a wrap and is
corrected. So it only works if the angle really changes by less than half a turn
between samples; a genuinely fast spin, sampled too slowly, cannot be told apart.

READING THE DEMO:
Something turning steadily through two whole turns, 15 degrees a step, written as an
angle from -180 to 180: a sawtooth (left) that climbs to 165 and drops back to -180 as
each turn completes. Unwrapped, it is a straight ramp from -180 to 525 (right). Below, the run 150 170 -170 -150 becomes
150 170 190 210.

RELATED:
continuous_rotation does the same for a stream of angles, one at a time. atan2 makes
angles from points. np.diff of the unwrapped angles gives the true turning speed.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', '0 to 705, 15 at a time', 140, 53),
    N('ar', 'np.arange 0 720 15', 140, 92, 180, 130),
    C('c3', 'wrapped: -180 to 180', 350, 53),
    N('md', '% 360', 350, 92, 129, 70),
    N('sb', '- 180', 350, 180, 129, 70),
    C('c4', 'unwrap', 560, 53),
    N('uw', 'np.unwrap', 560, 92, 150, 90),
    C('c5', 'in: a sawtooth', 28, 290),
    N('p1', 'plot', 28, 325, 368, 180, dict(BARS(48, 600.0, 360, 130, 'line'), **{'min y': -200.0})),
    C('c6', 'out: a steady ramp', 430, 290),
    N('p2', 'plot', 430, 325, 368, 180, dict(BARS(48, 600.0, 360, 130, 'line'), **{'min y': -200.0})),
    C('c7', 'four angles across the jump', 140, 560),
    N('m1', 'message', 140, 595, 200, 42, MSG('150 170 -170 -150')),
    N('u2', 'np.unwrap', 380, 595, 150, 90),
    N('l1', 'list', 560, 595, 220, 42, LIST),
],
[('lb', 'out', 'ar', ''), ('lb', 'out', 'm1', ''), ('ar', 'arange out', 'md', 'in'),
 ('md', 'result', 'sb', 'in'), ('sb', 'result', 'uw', 'in'), ('sb', 'result', 'p1', 'y'),
 ('uw', 'unwrapped', 'p2', 'y'), ('m1', 'message out', 'u2', 'in'), ('u2', 'unwrapped', 'l1', '')],
['continuous_rotation', 'atan2', 'np.diff', 'np.angle'], 740, 1000)


# ------------------------------------------------- nanmin/nanmax/nanstd/nanvar
NANS = {
    'np.nanmin': ('the smallest value, skipping gaps', 'smallest', 'np.amin', 'amin', '1', None,
                  'np.nanmax'),
    'np.nanmax': ('the largest value, skipping gaps', 'largest', 'np.amax', 'amax', '7', None,
                  'np.nanmin'),
    'np.nanstd': ('the spread of values, skipping gaps', 'standard deviation', 'np.std', 'std',
                  '2.49', 2, 'np.nanvar'),
    'np.nanvar': ('the variance of values, skipping gaps', 'variance', 'np.var', 'var', '6.22', 2,
                  'np.nanstd'),
}

for stem, (gist, what, plain, plain_out, result, decimals, sibling) in NANS.items():
    out = stem.split('.')[-1]
    rounded = decimals is not None
    page(stem, f'{stem} - {gist}',
f"""
Gives the {what} of an array, ignoring any nan ("not a number") in it.

{plain} gives nan as soon as a single value is nan - a dropped sensor reading, an
empty slot, a division by zero upstream - so one gap spoils the whole answer.
{stem} leaves the gaps out and works with the values that are there. Only a row or
column made entirely of nan gives nan.

SYNTAX:
{stem} <axis: int>
Default: axis None (the whole array).

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input is measured and sent.

axis (menu, also an inlet):
None measures over every element. A number measures along that axis: for a 2-D
array, 0 gives one answer per column, 1 one per row.

OUTPUTS:

{out}:
The {what}: a single value, or one per column or row.

READING THE DEMO:
3 nan 1 7 has a gap. {plain} gives nan; {stem} gives {result}, the {what} of 3 1 7.{' The round node only tidies the number.' if rounded else ''} A metro
re-sends the list twice a second.

RELATED:
{plain} is the version that does not skip gaps. {sibling}, np.nanmean,
np.nanmedian and np.nansum skip them the same way. np.isnan finds the gaps;
np.nan_to_num fills them.
""",
    metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
        C('c1', 'a list with a gap', 180, 53),
        N('m1', 'message', 180, 92, 160, 42, MSG('3 nan 1 7')),
        C('c2', 'skipping the gap', 380, 53),
        N('n1', stem, 380, 92, 150, 90),
    ] + ([N('r1', 'round', 560, 92, 100, 70, {'decimals': 2}),
          N('l1', 'list', 680, 92, 100, 42, LIST)] if rounded else
         [N('l1', 'list', 560, 92, 100, 42, LIST)]) + [
        C('c3', 'without skipping: nan', 380, 230),
        N('n2', plain, 380, 265, 150, 90),
        N('l2', 'list', 560, 265, 100, 42, LIST),
    ],
    [('mt', '', 'm1', ''), ('m1', 'message out', 'n1', 'in'), ('m1', 'message out', 'n2', 'in'),
     ('n2', plain_out, 'l2', '')] +
    ([('n1', out, 'r1', 'in'), ('r1', 'result', 'l1', '')] if rounded else [('n1', out, 'l1', '')]),
    [plain, sibling, 'np.nanmean', 'np.isnan', 'np.nan_to_num'], 420, 760)


# ----------------------------------------------------------------- average
page('np.average', 'np.average - a mean in which some values count more',
"""
Gives the weighted average of an array: each value counts in proportion to its
weight. Without weights it is the same as np.mean.

Weights let more trustworthy, more recent or more important values pull harder: a
pose averaged over several sensors weighted by each one's confidence, a position
blended between targets by how near each is, the centre of mass of joints weighted by
the mass of each limb, a smoothing window that favours the latest frame.

SYNTAX:
np.average <axis: int>
Default: axis None (the whole array).

INPUTS and PARAMETERS:

in:
The values (a list is turned into one). Every input is averaged and sent - the only
inlet that makes the node send.

weights:
How much each value counts, held: send them first. Either one weight per element
along the axis (one per row, for axis 0), or the same shape as the whole array. They
need not add up to 1 - np.average divides by their total - but they must not all
be 0. Without weights every value counts the same.

axis (menu, also an inlet):
None averages everything to one value. A number averages along that axis: for a 2-D
array, 0 gives one average per column (each row weighted), 1 one per row.

OUTPUTS:

average:
The weighted average.

READING THE DEMO:
0 and 10, weighted 3 and 1: the 0 counts three times as much, so the average is 2.5,
a quarter of the way along - where np.mean gives the halfway 5. Change the weights
and click them to move the average. A metro re-sends the values twice a second.

RELATED:
np.mean is the unweighted average. lerp and crossfade blend two values by one
fraction. np.robust_mean down-weights outliers automatically.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', 'two values', 180, 53),
    N('m1', 'message', 180, 92, 100, 42, MSG('0 10')),
    C('c2', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 210, 88, 46),
    C('c3', 'their weights, held', 180, 175),
    N('mw', 'message', 180, 210, 100, 42, MSG('3 1')),
    C('c4', 'weighted: nearer the 0', 380, 53),
    N('av', 'np.average', 380, 92, 150, 110),
    N('l1', 'list', 560, 92, 100, 42, LIST),
    C('c5', 'plain mean: halfway', 380, 250),
    N('mn', 'np.mean', 380, 285, 150, 90),
    N('l2', 'list', 560, 285, 100, 42, LIST),
],
[('mt', '', 'm1', ''), ('lb', 'out', 'mw', ''), ('mw', 'message out', 'av', 'weights'),
 ('m1', 'message out', 'av', 'in'), ('m1', 'message out', 'mn', 'in'),
 ('av', 'average', 'l1', ''), ('mn', 'mean', 'l2', '')],
['np.mean', 'lerp', 'crossfade', 'np.robust_mean'], 440, 800)


# ------------------------------------------------------------ histogram2d
page('np.histogram2d', 'np.histogram2d - how many points fall in each cell of a grid',
"""
Divides a rectangle into a grid of cells and counts how many of a set of points fall
in each. The counts make a picture of where the points gather: an occupancy map.

Feed it the floor positions of performers over a scene and it shows where they spent
their time; a hand's positions over a phrase, the reach of a gesture; any two
features of a stream against each other, how they go together.

SYNTAX:
np.histogram2d <bins: int> <x min> <x max> <y min> <y max>
Defaults: 10 x 10 bins, auto range. Giving the four limits turns auto range off.

EXAMPLE:
np.histogram2d 20 -2 2 -2 2

INPUTS and PARAMETERS:

points:
The points, as rows of x y (an array with 2 columns; any further columns are
ignored). Points with nan or infinity are left out. Every input is counted and sent.

bins (int box, also an inlet):
How many cells along each side.

x min, x max, y min, y max (float boxes, also inlets):
The rectangle the grid covers, when auto range is off. Points outside it are not
counted.

auto range:
An option, under the node. On, the grid spans the points' own extent, so it moves
with them; off, it stays fixed, so one picture can be compared with the next.

density:
An option, under the node. On, the counts are scaled so they add up to 1 over the
area: comparable between sets of different sizes.

OUTPUTS:

counts:
The grid of counts, as a picture: one row per y cell and one column per x cell. Row
0 is the lowest y, which a heat_map draws at the top.

x edges, y edges:
Where the cells start and end along each side. Sent before the counts.

READING THE DEMO:
Top: a thousand points, each coordinate the sum of four random numbers, so they pile
up around the middle (2, 2) and thin out towards the edges. Counted in a 16 x 16 grid,
they make a soft blob, redrawn twice a second.

Bottom: four points in a 2 x 2 grid over 0 to 2. Three have y below 1 (the top row):
one with x below 1 and two above it, so 1 2. One has y above 1 and x below 1: 1 0.

RELATED:
np.histogram counts along one line. np.digitize gives each value's bin. heat_map
draws the counts.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', '4 x 1000 points of x y', 180, 53),
    N('rn', 'np.rand 4 1000 2', 180, 92, 150, 150),
    C('c2', 'add each four', 360, 53),
    N('sm', 'np.sum 0', 360, 92, 150, 70, {'axis': '0'}),
    C('c3', 'count them in a 16 x 16 grid', 540, 53),
    N('h1', 'np.histogram2d 16 0 4 0 4', 540, 92, 240, 230),
    C('c4', 'they gather in the middle', 28, 300),
    N('hm', 'heat_map', 28, 335, 268, 270, HM(250, 250, 1, 0.0, 30.0, '', 'plasma')),
    C('c5', 'once at load', 340, 360),
    N('lb', 'load_bang', 340, 395, 88, 46),
    C('c6', 'four points: x y x y ...', 340, 460),
    N('mp', 'message', 340, 495, 260, 42, MSG('0.5 0.5 1.5 0.5 1.5 0.6 0.5 1.5')),
    N('rs', 'np.reshape 4 2', 340, 570, 150, 90),
    N('h2', 'np.histogram2d 2 0 2 0 2', 520, 570, 240, 230),
    N('hc', 'heat_map', 340, 690, 168, 130, HM(150, 80, 1, 0.0, 2.0, '%.0f')),
],
[('mt', '', 'rn', ''), ('rn', 'random array', 'sm', 'in'), ('sm', 'sum', 'h1', 'points'),
 ('h1', 'counts', 'hm', 'y'), ('lb', 'out', 'mp', ''), ('mp', 'message out', 'rs', 'tensor in'),
 ('rs', 'output', 'h2', 'points'), ('h2', 'counts', 'hc', 'y')],
['np.histogram', 'np.digitize', 'heat_map', 'np.rand'], 880, 1150)


# ------------------------------------------------------------------- top_k
page('np.top_k', 'np.top_k - the k largest (or smallest) values and where they are',
"""
Finds the k largest values of an array - or the k smallest - and gives their
positions and the values themselves, best first.

Use it to pick the three most active joints in a frame, the five loudest frequency
bands, the nearest few points to a target (k smallest distances), the strongest
matches from np.correlate. It works along an axis, so each row of a 2-D array gets its
own top k.

This is not a NumPy function by this name: it is built on np.argpartition, which
finds the k without sorting everything else, and then sorts only those k.

SYNTAX:
np.top_k <k: int> <largest or smallest>
Defaults: 3, largest.

EXAMPLES:
np.top_k 5
np.top_k 1 smallest

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input is searched and sent.

k (int box, also an inlet):
How many to find. More than there are gives all of them, sorted.

axis (int box, also an inlet):
Which way to look: -1 (the default) within each row, 0 down each column.

which:
An option, under the node: largest (the default) or smallest.

OUTPUTS:

indices:
The positions of the k values, best first.

values:
The k values themselves, best first. Sent first, then the indices.

READING THE DEMO:
Top: of 5 9 1 7 3 the two largest are 9 (at position 1) and 7 (at position 3).

Bottom: the same list, the two smallest: 1 (at 2) and 3 (at 4). A metro re-sends the
list twice a second; edit it and click.

RELATED:
np.argmax and np.amax find just the one largest. np.argsort and np.sort order
everything. np.take and np.take_along_axis fetch values by position.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', 'five values', 180, 53),
    N('m1', 'message', 180, 92, 160, 42, MSG('5 9 1 7 3')),
    C('c2', 'the two largest', 380, 53),
    N('t1', 'np.top_k 2', 380, 92, 150, 110),
    C('c3', 'where', 560, 53),
    N('i1', 'list', 560, 92, 100, 42, LIST),
    C('c4', 'what', 680, 53),
    N('v1', 'list', 680, 92, 100, 42, LIST),
    C('c5', 'the two smallest', 380, 240),
    N('t2', 'np.top_k 2 smallest', 380, 275, 190, 110),
    N('i2', 'list', 600, 275, 80, 42, LIST),
    N('v2', 'list', 700, 275, 80, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 't1', 'in'), ('m1', 'message out', 't2', 'in'),
 ('t1', 'indices', 'i1', ''), ('t1', 'values', 'v1', ''),
 ('t2', 'indices', 'i2', ''), ('t2', 'values', 'v2', '')],
['np.argmax', 'np.argsort', 'np.take', 'np.amax'], 440, 900)


# ---------------------------------------------------------- isclose/allclose
CLOSE_PROSE = """
Two decimals that should be equal rarely are exactly: 0.1 + 0.2 is
0.30000000000000004, and a sensor at rest still jitters in its last digits. So ==
almost never matches decimals. {stem} compares with a tolerance instead: two values
count as equal if they differ by no more than the tolerance.

{use}

SYNTAX:
{stem} <tolerance: float>
Default: tolerance 0.001.

INPUTS and PARAMETERS:

in:
The values to test (a list is turned into one). Every input is compared and sent -
the only inlet that makes the node send.

in 2:
What to compare them with, held: send it first. The same shape, or a single value to
compare every element with.

tolerance (float box, also an inlet):
The largest difference that still counts as equal.

relative (float box, also an inlet):
An extra allowance in proportion to the size of in 2: 0.01 allows a further 1% of it.
0 (the default) uses the tolerance alone. NumPy's own defaults are a relative 0.00001
and an absolute 0.00000001, too fine for most sensor data, so here the tolerance
leads.

OUTPUTS:

{out}:
{out_text}

READING THE DEMO:
{demo}

RELATED:
{other} {other_text} == compares exactly. np.all of a mask asks whether every element
is true.
"""

page('np.isclose', 'np.isclose - which values are equal, within a tolerance',
"""
Compares two arrays element by element and marks where they are equal to within a
tolerance.
""" + CLOSE_PROSE.format(
    stem='np.isclose',
    use="""Use it to find which joints have returned to a rest pose, which values match a
target, which readings have stopped changing.""",
    out='close',
    out_text='True where the two are within the tolerance, the shape of the inputs together.',
    demo="""1.0 1.0005 1.01 against 1, within 0.001: the first two count as equal and 1.01 does
not - true true false. A metro re-sends the values twice a second; change the
tolerance and see the last one join them.""",
    other='np.allclose', other_text='asks whether ALL of them are close, as one true or false.'),
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', 'three values', 180, 53),
    N('m1', 'message', 180, 92, 180, 42, MSG('1.0 1.0005 1.01')),
    C('c2', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 210, 88, 46),
    C('c3', 'compared with 1, held', 180, 175),
    N('m2', 'message', 180, 210, 60, 42, MSG('1')),
    C('c4', 'within 0.001?', 400, 53),
    N('ic', 'np.isclose', 400, 92, 170, 150),
    N('l1', 'list', 600, 92, 200, 42, LIST),
],
[('mt', '', 'm1', ''), ('lb', 'out', 'm2', ''), ('m2', 'message out', 'ic', 'in 2'),
 ('m1', 'message out', 'ic', 'in'), ('ic', 'close', 'l1', '')],
['np.allclose', 'equal', 'np.all', 'np.where'], 420, 900)


page('np.allclose', 'np.allclose - are all values equal, within a tolerance',
"""
Compares two arrays and says, with a single true or false, whether every element is
equal to its partner to within a tolerance.
""" + CLOSE_PROSE.format(
    stem='np.allclose',
    use="""Use it to ask whether a whole pose has returned to rest, whether a body has stopped
moving (this frame against the last), whether a fit has converged, whether two
matrices are the same.""",
    out='all close',
    out_text='True if every pair is within the tolerance, otherwise false.',
    demo="""1.0 1.05 against 1 1. Within 0.001 the 1.05 is too far, so false; the second node
allows 0.1, and then they all count as equal - true. A metro re-sends the values twice a
second.""",
    other='np.isclose', other_text='marks which elements are close, one by one.'),
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', 'two values', 180, 53),
    N('m1', 'message', 180, 92, 120, 42, MSG('1.0 1.05')),
    C('c2', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 210, 88, 46),
    C('c3', 'compared with 1 1, held', 180, 175),
    N('m2', 'message', 180, 210, 80, 42, MSG('1 1')),
    C('c4', 'all within 0.001?', 400, 53),
    N('a1', 'np.allclose', 400, 92, 170, 150),
    N('l1', 'list', 600, 92, 80, 42, LIST),
    C('c5', 'all within 0.1?', 400, 280),
    N('a2', 'np.allclose 0.1', 400, 315, 170, 150),
    N('l2', 'list', 600, 315, 80, 42, LIST),
],
[('mt', '', 'm1', ''), ('lb', 'out', 'm2', ''), ('m2', 'message out', 'a1', 'in 2'),
 ('m2', 'message out', 'a2', 'in 2'), ('m1', 'message out', 'a1', 'in'),
 ('m1', 'message out', 'a2', 'in'), ('a1', 'all close', 'l1', ''), ('a2', 'all close', 'l2', '')],
['np.isclose', 'equal', 'np.all', 'np.any'], 520, 900)


# ---------------------------------------------------------------- trapezoid
TRAP_SPEEDS = '0 1 2 3 4'

page('np.trapezoid', 'np.trapezoid - the area under a curve',
"""
Adds up the area under a curve given as samples: the integral. Between each pair of
neighbouring samples it takes the area of the trapezoid they make - their average
times the spacing - and sums them.

From speeds it gives the distance travelled; from a joint's power, the work done;
from a rate, the total; from a level over time, the overall dose. It is more accurate
than adding the samples and multiplying by the spacing (np.sum), which counts each end
as if it had lasted a whole step.

SYNTAX:
np.trapezoid <spacing: float>
Default: spacing 1.

EXAMPLE:
np.trapezoid 0.01667

INPUTS and PARAMETERS:

in:
The samples (a list or array). Every input is integrated and sent.

spacing (float box, also an inlet):
The distance between samples - the time between frames (1/60 for 60 frames a
second), so the answer comes out per second rather than per frame.

axis (int box, also an inlet):
Which way to integrate: -1 (the default) along each row, 0 down each column.

OUTPUTS:

area:
The total area: one number, or one per row or column.

READING THE DEMO:
A speed rising steadily from 0 to 4 metres a second, sampled every half second. The
distance covered is the area of a triangle, 2 seconds by 4 metres a second, halved:
4 metres, which np.trapezoid gets exactly. np.sum times the spacing gives 5, counting
the final speed of 4 as if it had lasted a whole half second. A metro re-sends the
speeds twice a second.

RELATED:
np.cumulative_trapezoid gives the running total at every sample. np.cumsum is the
running sum without the averaging. np.gradient goes the other way, from a total to
its rate.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', 'speeds, every half second', 180, 53),
    N('m1', 'message', 180, 92, 160, 42, MSG(TRAP_SPEEDS)),
    C('c2', 'the distance covered', 380, 53),
    N('tz', 'np.trapezoid 0.5', 380, 92, 170, 110),
    N('l1', 'list', 580, 92, 80, 42, LIST),
    C('c3', 'np.sum times the spacing: too much', 380, 240),
    N('sm', 'np.sum', 380, 275, 150, 90),
    N('ml', '* 0.5', 380, 390, 129, 70),
    N('l2', 'list', 580, 390, 80, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'tz', 'in'), ('m1', 'message out', 'sm', 'in'),
 ('tz', 'area', 'l1', ''), ('sm', 'sum', 'ml', 'in'), ('ml', 'result', 'l2', '')],
['np.cumulative_trapezoid', 'np.cumsum', 'np.gradient', 'np.sum'], 520, 900)


page('np.cumulative_trapezoid', 'np.cumulative_trapezoid - the area so far, at every sample',
"""
Gives the running total of the area under a curve: at each sample, the integral from
the start up to that point. It starts at 0 and has one value per sample, so it lines
up with the input.

From speeds it gives the distance travelled so far - the position at each moment;
from accelerations, the speed so far; from a joint's power, the work done so far.
Each step adds the trapezoid between two neighbouring samples: their average times
the spacing.

NumPy itself has only np.trapezoid, the total; this running version is SciPy's
cumulative_trapezoid with the starting 0 included.

SYNTAX:
np.cumulative_trapezoid <spacing: float>
Default: spacing 1.

INPUTS and PARAMETERS:

in:
The samples (a list or array). Every input is integrated and sent.

spacing (float box, also an inlet):
The distance between samples - the time between frames (1/60 for 60 frames a
second), so the totals come out per second.

axis (int box, also an inlet):
Which way to integrate: -1 (the default) along each row, 0 down each column - one
running total per joint for a buffer of frames x joints.

OUTPUTS:

running total:
The area so far at each sample, starting from 0, the same shape as the input.

READING THE DEMO:
A speed rising steadily from 0 to 4 metres a second, sampled every half second. The
distance so far is 0, 0.25, 1, 2.25 and 4 metres: growing faster and faster as the speed
builds - the squares, as a steady acceleration gives. The last value is
np.trapezoid's total. A metro re-sends the speeds twice a second.

RELATED:
np.trapezoid gives only the final total. np.cumsum is the running sum without the
averaging. np.gradient goes the other way, from the running total back to the rate.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', 'speeds, every half second', 180, 53),
    N('m1', 'message', 180, 92, 160, 42, MSG(TRAP_SPEEDS)),
    C('c2', 'the distance so far', 380, 53),
    N('ct', 'np.cumulative_trapezoid 0.5', 380, 92, 250, 110),
    N('l1', 'list', 380, 240, 260, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'ct', 'in'), ('ct', 'running total', 'l1', '')],
['np.trapezoid', 'np.cumsum', 'np.gradient', 'np.diff'], 380, 900)


# ---------------------------------------------------------------- fft / ifft
WAVE = [
    C('cw', 'a wave: 64 samples, 8 cycles', 180, 53),
    N('ar', 'np.arange 64', 180, 92, 150, 110),
    N('ml', '* 45', 360, 92, 129, 70),
    N('sn', 'sin', 360, 180, 129, 70),
]
WAVE_LINKS = [('mt', '', 'ar', ''), ('ar', 'arange out', 'ml', 'in'), ('ml', 'result', 'sn', 'in')]

page('np.fft.fft', 'np.fft.fft - the frequencies in a signal, complex or real',
"""
The full Fourier transform: breaks a signal into sine waves of every frequency, and
gives how much of each there is (magnitude) and where it starts (phase).

For an ordinary real signal - sensor data, sound - np.fft.rfft is simpler and gives
the same information in half the space. np.fft.fft is for signals that are already
complex (a point moving in a plane written as x + y j, whose spectrum shows clockwise
and counter-clockwise motion separately), and for when you need the whole spectrum
to change and turn back with np.fft.ifft.

The spectrum has one entry per sample. The first half are the frequencies 0, 1, 2 ...
cycles per window; the second half are the NEGATIVE frequencies, counting back up to
-1. For a real signal the second half mirrors the first.

SYNTAX:
np.fft.fft <sample rate: float>
Default: sample rate 1.

INPUTS and PARAMETERS:

in:
The signal: real or complex values. Every input is transformed and sent.

sample rate (float box, also an inlet):
Samples per second. Only the frequencies outlet uses it.

axis (int box, also an inlet):
For a 2-D array, which way the signal runs: -1 along each row, 0 down each column.

norm:
An option, under the node: how the result is scaled, as for np.fft.rfft.

OUTPUTS:

spectrum:
The complex spectrum, one entry per sample.

magnitude:
How much of each frequency there is.

phase:
Where each wave starts in its cycle, in radians.

frequencies:
The frequency of each entry: 0 and the positive ones first, then the negative ones.
Sent first.

READING THE DEMO:
The same wave as on the np.fft.rfft page, 8 cycles in 64 samples. Its full spectrum
has two spikes: at entry 8, and at entry 56 - frequency -8, the mirror image a real
signal always has.

RELATED:
np.fft.rfft does the same for real signals without the mirror half. np.fft.ifft turns
a spectrum back into a signal. np.real, np.imag and np.angle take a spectrum apart.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + WAVE + [
    C('c2', 'to frequencies', 540, 53),
    N('ff', 'np.fft.fft 64', 540, 92, 180, 150),
    C('c3', 'magnitude: spikes at 8 and its mirror, 56', 28, 290),
    N('p2', 'plot', 28, 325, 528, 200, BARS(64, 40.0, 520, 150)),
],
WAVE_LINKS + [('sn', 'out', 'ff', 'in'), ('ff', 'magnitude', 'p2', 'y')],
['np.fft.rfft', 'np.fft.ifft', 'np.angle', 'np.real'], 590, 1250)


page('np.fft.ifft', 'np.fft.ifft - a signal from its full spectrum',
"""
Turns a full complex spectrum, from np.fft.fft, back into a signal: the inverse
Fourier transform. Each frequency's wave is rebuilt at its magnitude and phase and
all of them are added.

Round trip it gives back the original. The point is to change the spectrum first:
remove or boost frequencies, shift phases, or keep only the positive frequencies of a
real signal to make its "analytic" version, whose np.abs is a smooth envelope.

SYNTAX:
np.fft.ifft

INPUTS and PARAMETERS:

spectrum:
A complex spectrum, one entry per sample, as np.fft.fft makes it. Every input is
transformed and sent.

axis (int box, also an inlet):
For a 2-D spectrum, which way the frequencies run: -1 along each row, 0 down each
column.

norm:
An option, under the node: must match the norm used for np.fft.fft.

OUTPUTS:

signal:
The rebuilt signal. When its imaginary part is only rounding (as it is for any
spectrum that came from a real signal, unchanged or changed symmetrically) it is sent
as plain real numbers; otherwise as complex numbers.

READING THE DEMO:
The wave goes to its full spectrum and straight back: the right-hand plot is the
same wave, as plain numbers. Put any change to the spectrum between the two nodes to
hear or see what it does.

RELATED:
np.fft.fft makes the spectrum. np.fft.irfft is the inverse for np.fft.rfft's half
spectrum. np.real takes the real part if a changed spectrum leaves a complex signal.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + WAVE + [
    C('c1', 'in: the wave', 28, 290),
    N('p1', 'plot', 28, 325, 368, 160, dict(BARS(64, 1.0, 360, 110, 'line'), **{'min y': -1.0})),
    C('c2', 'to frequencies', 540, 53),
    N('ff', 'np.fft.fft', 540, 92, 180, 150),
    C('c3', 'and back again', 540, 270),
    N('if', 'np.fft.ifft', 540, 305, 150, 90),
    C('c4', 'out: the same wave', 430, 430),
    N('p2', 'plot', 430, 465, 368, 160, dict(BARS(64, 1.0, 360, 110, 'line'), **{'min y': -1.0})),
],
WAVE_LINKS + [('sn', 'out', 'p1', 'y'), ('sn', 'out', 'ff', 'in'), ('ff', 'spectrum', 'if', 'spectrum'),
              ('if', 'signal', 'p2', 'y')],
['np.fft.fft', 'np.fft.irfft', 'np.real', 'np.fft.rfft'], 700, 1150)


def main():
    args = sys.argv[1:]
    force = '--force' in args
    wanted = [a for a in args if not a.startswith('--')] or list(PAGES)
    for stem in wanted:
        p = PAGES[stem]
        path = os.path.join(HELP_DIR, stem + '_help.json')
        if os.path.exists(path) and not force:
            print('exists, skipped (use --force):', path)
            continue
        print(build(stem, p['title'], p['body'], p['demo'], p['links'], demo_width=840,
                    text_width=760, text_height=p['text_height']))


if __name__ == '__main__':
    main()
