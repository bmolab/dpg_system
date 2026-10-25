"""Seed the help pages for the fourth batch of scipy nodes (2026-10-24): one page per node.

    python3 dpg_system/help/tools/seed_scipy4_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

scipy.linear_sum_assignment, scipy.griddata, scipy.correlate, scipy.coherence,
scipy.local_maxima, scipy.zscore, scipy.linregress, scipy.spearmanr, scipy.entropy.

The correlate and coherence demos build their signals in subpatchers (make_pair,
make_two_rhythms). Every fan-out that feeds both a cold and a hot inlet links the cold
branch FIRST (an outlet sends in link order), and in make_two_rhythms the b signal's
chain is linked before a's, so b is held at the node before a triggers it. Not a
make_* script: rebuild.py never runs it, and it will not write over an existing page
without --force. Treat the pages as hand-made once written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from seed_numpy_more_help import LIST, MSG, HM, BARS, C, N, see_also
from seed_scipy_help import S


PAGES = {}


def page(stem, title, body, demo, links, also, also_y, text_height=820, subpatch=None):
    PAGES[stem] = dict(title=title, body=body.strip('\n'), demo=demo + see_also(also_y, also),
                       links=links, text_height=text_height, subpatch=subpatch)


METRO_HALF = {'on': True, 'period': 500.0, 'units': 'milliseconds'}
METRO_1S = {'on': True, 'period': 1000.0, 'units': 'milliseconds'}
METRO_2S = {'on': True, 'period': 2000.0, 'units': 'milliseconds'}


def metro(y=53, label='twice a second', props=METRO_HALF, init='metro 500'):
    return [C('cm', label, 28, y), N('mt', init, 28, y + 39, 129, 70, props)]


def LINE(count, miny, maxy, width=360, style='line'):
    return dict(BARS(count, maxy, width, 130, style), **{'min y': miny})


def ROUND(key, x, y, places=2):
    return N(key, 'round', x, y, 100, 70, {'decimals': places})


# ------------------------------------------------------ linear_sum_assignment
page('scipy.linear_sum_assignment', 'scipy.linear_sum_assignment - keep track of who is who',
"""
Pairs up two sets of points - this frame's and the last frame's, or this frame's and a
reference - so that the total distance between partners is as small as possible. That
is how identities are kept: the blob found near where performer 2 was is performer 2,
even though a detector (scipy.label, scipy.kmeans, a depth camera) lists the people in
a different order every frame.

With track on, the node remembers: each frame is matched against the last, and the
points come out in a stable order, one row per identity. Someone who disappears keeps
their row (as nan) and their last position, so they are picked up again when they
return; someone new is added as a new row.

It also solves the general problem: give it a cost matrix (input: cost matrix) and it
finds the one-to-one pairing of rows with columns with the smallest total cost -
assigning performers to roles, lights to targets, sounds to dancers.

SYNTAX:
scipy.linear_sum_assignment

INPUTS and PARAMETERS:

points:
This frame's points, rows of coordinates (or, as a cost matrix, the costs). Every input
is matched and sent - the only inlet that makes the node send.

reference:
The points to match against, held. With track on, the first frame becomes the reference
and after that each frame's result does.

max distance (float box, also an inlet):
Partners further apart than this are not paired - the reference point counts as lost,
the new point as new. 0 (the default) pairs everything it can.

track:
An option, under the node: keep identities from frame to frame, as described above.

input:
An option, under the node: points (the default), or cost matrix - rows to be assigned
to columns, the matrix holding the cost of each pairing.

OUTPUTS:

matched:
For each reference point (each row of the cost matrix), the position of its partner
in this frame, or -1 for none.

tracked:
This frame's points in the reference's order: one row per identity, nan for any that
were lost, then any new points.

cost:
The total distance (or cost) of the pairing. Sent first, then tracked, then matched.

READING THE DEMO:
Three points, sent in a different random order twice a second (left: the list keeps
changing). With track on, the tracked output (right) is always in the same order - the
same point in the same row - because each frame is matched to the last.

RELATED:
scipy.kdtree finds the nearest point to each point, without making the pairs one to
one. scipy.label and scipy.kmeans find the points to track. np.random.permutation
shuffles.
""",
metro() + [
    C('c1', 'three points: x y, x y, x y', 180, 53),
    N('mp', 'message', 180, 92, 260, 42, MSG('0 0 10 0 0 10')),
    N('rp', 'np.reshape 3 2', 470, 92, 150, 90),
    C('c2', 'shuffled each time', 650, 53),
    N('sh', 'np.random.permutation', 650, 92, 200, 70),
    C('c3', 'in: a new order every time', 28, 240),
    N('l1', 'list', 28, 275, 330, 42, LIST),
    C('c4', 'track: frame to frame', 400, 240),
    N('la', 'scipy.linear_sum_assignment', 400, 275, 260, 130, {'track': True}),
    C('c5', 'out: always the same order', 400, 440),
    N('l2', 'list', 400, 475, 330, 42, LIST),
],
[('mt', '', 'mp', ''), ('mp', 'message out', 'rp', 'tensor in'), ('rp', 'output', 'sh', 'in'),
 ('sh', 'shuffled', 'l1', ''), ('sh', 'shuffled', 'la', 'points'), ('la', 'tracked', 'l2', '')],
['scipy.kdtree', 'scipy.label', 'scipy.kmeans', 'np.random.permutation'], 570, 1300)


# -------------------------------------------------------------------- griddata
page('scipy.griddata', 'scipy.griddata - fill a grid from scattered readings',
"""
Takes values measured at scattered points - not on a grid - and fills in a whole grid
from them, estimating every cell from the readings around it.

Use it to make a map of a room from sensor readings taken wherever you happened to
stand (a magnetic field survey, a light level, a sound level), a heat map of a quantity
known only at the performers' positions, or a smooth surface through a few control
points to steer a parameter by position.

SYNTAX:
scipy.griddata <width> <height>
Defaults: 32 x 32.

INPUTS and PARAMETERS:

values:
One value per point. Every input is gridded and sent - the only inlet that makes the
node send - so the points can stay put while their readings change.

points:
Where the readings were taken: rows of x y. Held.

width, height (int boxes, also inlets):
How many cells across and down.

x min, x max, y min, y max (float boxes, also inlets):
The area the grid covers, when auto range is off.

method:
An option, under the node. linear (the default) joins the points with flat triangles;
nearest gives each cell the value of the closest reading (patches); cubic is smoother
within the points; smooth fits a thin-plate spline through every reading - smooth
everywhere, and fills the whole grid.

auto range:
An option, under the node. On (the default), the grid spans the points' own extent.

outside:
An option, under the node: what the cells beyond the outermost readings get (linear
and cubic cannot reach them) - the nearest reading (the default), nan, or 0.

OUTPUTS:

grid:
The filled grid: one row per y, one column per x.

READING THE DEMO:
Six readings at fixed places, with new random values every second. Linear (left) is
made of flat triangles with visible creases; smooth (right) flows through the same
six readings with no creases.

RELATED:
np.interp and scipy.cubic_spline fill in along a line. np.histogram2d counts points
into a grid. heat_map draws the grid.
""",
metro(label='every second', props=METRO_1S, init='metro 1000') + [
    C('c1', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 210, 88, 46),
    C('c2', 'six places: x y ...', 180, 175),
    N('mp', 'message', 180, 210, 330, 42, MSG('0 0 1 0 0 1 1 1 0.3 0.6 0.7 0.3')),
    N('rp', 'np.reshape 6 2', 540, 210, 150, 90),
    C('c3', 'six readings', 580, 53),
    N('rn', 'np.rand 6', 580, 92, 150, 110),
    C('c4', 'linear', 28, 330),
    N('g1', 'scipy.griddata', 28, 365, 190, 230),
    N('h1', 'heat_map', 28, 610, 228, 250, HM(210, 210, 1, 0.0, 1.0, '', 'viridis')),
    C('c5', 'smooth', 400, 330),
    N('g2', 'scipy.griddata', 400, 365, 190, 230, {'method': 'smooth'}),
    N('h2', 'heat_map', 400, 610, 228, 250, HM(210, 210, 1, 0.0, 1.0, '', 'viridis')),
],
[('lb', 'out', 'mp', ''), ('mp', 'message out', 'rp', 'tensor in'), ('rp', 'output', 'g1', 'points'),
 ('rp', 'output', 'g2', 'points'), ('mt', '', 'rn', ''), ('rn', 'random array', 'g1', 'values'),
 ('rn', 'random array', 'g2', 'values'), ('g1', 'grid', 'h1', 'y'), ('g2', 'grid', 'h2', 'y')],
['np.interp', 'scipy.cubic_spline', 'np.histogram2d', 'heat_map'], 920, 1150)


# ------------------------------------------------------------------ correlate
# Inside make_pair: 5 seconds of a wave (b) and the same wave 12 samples (0.2 s)
# later (a). b is linked out first, so it is held at the correlate node before a.
PAIR_SUB = {
    'name': 'make_pair',
    'host': 'pr',
    'demo': [
        S('mt', 'metro 1000', 40, 40, 129, 70, METRO_1S),
        S('ct', 'comment every second', 190, 55, 200, 30),
        S('ar', 'np.arange 300', 40, 130, 150, 110),
        S('m1', '* 6', 40, 270, 129, 70),
        S('s1', 'sin', 40, 360, 129, 70),
        S('c1', 'comment a wave, 1 cycle a second', 190, 375, 280, 30),
        S('ob', 'out b', 40, 470, 150, 80),
        S('rl', 'np.roll 12', 330, 470, 150, 110),
        S('c2', 'comment the same wave 12 samples later', 330, 590, 320, 30),
        S('oa', 'out a', 330, 640, 150, 80),
    ],
    'links': [('mt', '', 'ar', ''), ('ar', 'arange out', 'm1', 'in'), ('m1', 'result', 's1', 'in'),
              ('s1', 'out', 'ob', 'b'), ('s1', 'out', 'rl', 'input'), ('rl', 'rolled array', 'oa', 'a')],
}

page('scipy.correlate', 'scipy.correlate - how far one signal trails another',
"""
Slides one signal past another and finds where they match best: how far one is behind
the other - the lag, in seconds - and how strongly they match there.

Use it to measure synchrony: whether a dancer leads or follows the music's pulse and by
how much; how far a performer's movement trails another's (or a mirror's); the delay
through a sensor, a network or a processing chain. A strength near 1 means the two
move alike once the lag is allowed for; near 0, that they are unrelated.

SYNTAX:
scipy.correlate <sample rate>
Default: sample rate 60.

INPUTS and PARAMETERS:

a:
The first signal. Every input is compared and sent - the only inlet that makes the node
send.

b:
The second signal, held: send it first.

sample rate (float box, also an inlet):
Samples per second, so the lag comes out in seconds.

max lag (float box, also an inlet):
Only look for lags up to this many seconds either way. 0 (the default) looks at all of
them. For a repeating signal several lags match; this keeps the search near zero.

OUTPUTS:

lag:
How far a trails b, in seconds: positive when a comes later, negative when it comes
earlier.

strength:
How well they match at that lag, from -1 to 1 (both are standardized first, so the size
of the signals does not matter).

correlation:
The match at every lag.

lags:
The lag of each value of correlation, in seconds. Sent first, then correlation,
strength and lag.

READING THE DEMO:
The make_pair subpatcher makes a wave (b) and the same wave 12 samples later (a). At 60
samples a second the lag comes out as 0.2 seconds, with a strength near 1.

RELATED:
np.correlate finds a short pattern within a signal. scipy.coherence measures how two
signals share each rhythm. np.corrcoef measures how alike two signals are with no
lag allowed.
""",
[
    C('c1', 'a wave, and the same 0.2 s later', 28, 53),
    N('pr', 'p make_pair', 28, 92, 180, 90),
    C('c2', 'how far a trails b', 300, 53),
    N('co', 'scipy.correlate', 300, 92, 190, 150),
    C('c3', 'lag, seconds', 540, 53),
    ROUND('r1', 540, 92, 3),
    N('l1', 'list', 660, 92, 100, 42, LIST),
    C('c4', 'strength', 540, 190),
    ROUND('r2', 540, 225, 2),
    N('l2', 'list', 660, 225, 100, 42, LIST),
],
[('pr', 'b', 'co', 'b'), ('pr', 'a', 'co', 'a'), ('co', 'lag', 'r1', 'in'), ('r1', 'result', 'l1', ''),
 ('co', 'strength', 'r2', 'in'), ('r2', 'result', 'l2', '')],
['np.correlate', 'scipy.coherence', 'np.corrcoef', 'scipy.hilbert'], 430, 1050, subpatch=PAIR_SUB)


# ------------------------------------------------------------------- coherence
# Inside make_two_rhythms: a = 3 + 7 cycles a second; b = 3 (shifted) + 11. They share
# only the 3. b's chains are linked first so b leaves before a.
RHYTHMS_SUB = {
    'name': 'make_two_rhythms',
    'host': 'tr',
    'demo': [
        S('mt', 'metro 2000', 40, 40, 129, 70, METRO_2S),
        S('ct', 'comment every two seconds; 600 samples, 10 s at 60 a second', 190, 55, 480, 30),
        S('ar', 'np.arange 600', 40, 130, 150, 110),
        S('mb2', '* 66', 40, 280, 129, 70),
        S('sb2', 'sin', 40, 370, 129, 70),
        S('mb1', '* 18', 200, 280, 129, 70),
        S('pb', '+ 60', 200, 370, 129, 70),
        S('sb1', 'sin', 200, 460, 129, 70),
        S('ab', '+', 200, 550, 129, 70),
        S('cb', 'comment b: 3 (shifted) and 11 cycles a second', 40, 640, 380, 30),
        S('ob', 'out b', 200, 690, 150, 80),
        S('ma2', '* 42', 450, 280, 129, 70),
        S('sa2', 'sin', 450, 370, 129, 70),
        S('ma1', '* 18', 610, 280, 129, 70),
        S('sa1', 'sin', 610, 370, 129, 70),
        S('aa', '+', 610, 460, 129, 70),
        S('ca', 'comment a: 3 and 7 cycles a second', 450, 550, 300, 30),
        S('oa', 'out a', 610, 600, 150, 80),
    ],
    'links': [('mt', '', 'ar', ''),
              ('ar', 'arange out', 'mb2', 'in'), ('ar', 'arange out', 'mb1', 'in'),
              ('ar', 'arange out', 'ma2', 'in'), ('ar', 'arange out', 'ma1', 'in'),
              ('mb2', 'result', 'sb2', 'in'), ('sb2', 'out', 'ab', 'operand', 0, 1),
              ('mb1', 'result', 'pb', 'in'), ('pb', 'result', 'sb1', 'in'), ('sb1', 'out', 'ab', 'in'),
              ('ab', 'result', 'ob', 'b'),
              ('ma2', 'result', 'sa2', 'in'), ('sa2', 'out', 'aa', 'operand', 0, 1),
              ('ma1', 'result', 'sa1', 'in'), ('sa1', 'out', 'aa', 'in'),
              ('aa', 'result', 'oa', 'a')],
}

page('scipy.coherence', 'scipy.coherence - which rhythms two signals share',
"""
Measures, for each frequency, how consistently two signals move together at that
rhythm: 1 where they share it in a steady relationship, 0 where they do not. It does
not care whether one leads the other - only whether the relationship holds.

Where a single correlation mixes all rhythms together, coherence separates them: two
dancers may share the slow sway of the music while their fast gestures are independent;
a breathing rhythm may lock to a movement while everything else drifts.

SYNTAX:
scipy.coherence <window: int> <sample rate>
Defaults: window 64, sample rate 60.

INPUTS and PARAMETERS:

a:
The first signal. Every input is compared and sent - the only inlet that makes the node
send.

b:
The second signal, held: send it first. Only as much of the two as they both cover is
used.

sample rate (float box, also an inlet):
Samples per second, so frequencies come out in cycles per second.

window (int box, also an inlet):
How many samples per window; the signals are cut into overlapping windows and averaged.
Longer gives finer frequency steps (sample rate / window apart) but fewer windows to
average, so a noisier estimate. Coherence needs several windows to mean anything.

OUTPUTS:

coherence:
From 0 to 1 at each frequency.

frequencies:
The frequency of each value. Sent first.

READING THE DEMO:
The make_two_rhythms subpatcher makes a (3 and 7 cycles a second) and b (3, shifted in
time, and 11). Their coherence is near 1 at 3 cycles a second - they share it, even
though shifted - and low elsewhere, including at 7 and 11, which each has alone.

RELATED:
scipy.correlate finds the lag between two signals. scipy.welch gives each signal's own
spectrum. np.corrcoef compares two signals as a whole.
""",
[
    C('c1', 'two signals sharing one rhythm', 28, 53),
    N('tr', 'p make_two_rhythms', 28, 92, 200, 90),
    C('c2', 'shared rhythms', 300, 53),
    N('ch', 'scipy.coherence', 300, 92, 190, 130),
    C('c3', 'near 1 at 3 cycles a second', 28, 260),
    N('pl', 'plot', 28, 295, 528, 230, BARS(33, 1.0, 520, 180)),
],
[('tr', 'b', 'ch', 'b'), ('tr', 'a', 'ch', 'a'), ('ch', 'coherence', 'pl', 'y')],
['scipy.correlate', 'scipy.welch', 'np.corrcoef', 'scipy.hilbert'], 590, 1050, subpatch=RHYTHMS_SUB)


# ------------------------------------------------------------- local_maxima
page('scipy.local_maxima', 'scipy.local_maxima - the peaks of a grid',
"""
Finds the peaks of a grid - cells higher than every other cell in their neighbourhood -
and gives their positions and heights, highest first, and how many there are. It is
the 2-D (or 3-D) counterpart of scipy.find_peaks.

scipy.label finds regions above a threshold, so two hills that run into each other
become one region. local_maxima finds the summits, so two people standing close in an
occupancy map, or two bright spots in a camera image, are still two. Use it for hot
spots in a heat map, the centres of people in a depth or occupancy grid, the brightest
points of an image, the strongest cells of a spectrogram.

It is built on scipy.ndimage.maximum_filter, which is how peaks in a grid are usually
found; SciPy has no function by this name.

SYNTAX:
scipy.local_maxima <size: int> <threshold: float>
Defaults: size 3, threshold 0.5.

INPUTS and PARAMETERS:

in:
The grid (any number of axes). Every input is searched and sent.

size (int box, also an inlet):
How wide a neighbourhood a peak must top, in cells: 3 compares each cell with the 8
around it; larger keeps only peaks that stand out over a wider area, and merges close
ones.

threshold (float box, also an inlet):
Peaks below this are ignored.

OUTPUTS:

positions:
Where each peak is, row then column (in cells), highest peak first. A flat top gives
its middle.

values:
The height of each.

count:
How many peaks. Sent first, then values, then positions.

READING THE DEMO:
Random values smoothed into hills (left), new every two seconds. With a neighbourhood
of 5 and a threshold of 0.55, the summits are found: the count and their positions are
listed, highest first.

RELATED:
scipy.label finds whole regions above a threshold. scipy.find_peaks finds peaks along a
line. scipy.gaussian_filter smooths a grid first, so its peaks are hills and not noise.
""",
metro(label='every two seconds', props=METRO_2S, init='metro 2000') + [
    C('c1', '24 x 24 random values', 180, 53),
    N('rn', 'np.rand 24 24', 180, 92, 150, 130),
    C('c2', 'smoothed into hills', 360, 53),
    N('gf', 'scipy.gaussian_filter 2', 360, 92, 210, 70),
    C('c3', 'the summits', 600, 53),
    N('lm', 'scipy.local_maxima 5 0.55', 600, 92, 220, 130),
    C('c4', 'the hills', 28, 290),
    N('h1', 'heat_map', 28, 325, 228, 250, HM(210, 210, 1, 0.35, 0.65, '', 'viridis')),
    C('c5', 'how many', 300, 290),
    N('lc', 'list', 300, 325, 80, 42, LIST),
    C('c6', 'where: row, column, highest first', 300, 400),
    ROUND('rp', 300, 435, 1),
    N('lp', 'list', 300, 520, 500, 42, LIST),
],
[('mt', '', 'rn', ''), ('rn', 'random array', 'gf', 'in'), ('gf', 'filtered', 'h1', 'y'),
 ('gf', 'filtered', 'lm', 'in'), ('lm', 'count', 'lc', ''), ('lm', 'positions', 'rp', 'in'),
 ('rp', 'result', 'lp', '')],
['scipy.label', 'scipy.find_peaks', 'scipy.gaussian_filter', 'np.argmax'], 640, 1150)


# -------------------------------------------------------------------- zscore
page('scipy.zscore', 'scipy.zscore - put values on a common scale',
"""
Rescales values so that their mean is 0 and their spread (standard deviation) is 1:
each value becomes how many standard deviations it is above or below the average.

Features measured in different units - a height in metres, a speed in km/h, an
energy in joules - cannot be compared or combined directly; their z-scores can. Use it
before combining features, clustering (scipy.kmeans) or comparing distances, and to
spot outliers - a z-score beyond about 3 is rare.

SYNTAX:
scipy.zscore <axis>
Default: axis None (every value together).

EXAMPLE:
scipy.zscore 0

INPUTS and PARAMETERS:

in:
The values (a list or array). Every input is rescaled and sent. nan values are skipped
in working out the mean and spread, and stay nan.

axis (text box, also an inlet):
None rescales everything together. A number rescales along that axis: 0 rescales each
column separately - each feature on its own scale.

OUTPUTS:

z:
The rescaled values, the same shape. A column that never changes has no spread; its
values come out as 0.

READING THE DEMO:
Four people's heights (metres) and weights (kg) as two columns, rescaled down each
column (axis 0). The two columns, a hundred times different in size, come out on the
same scale: about -1.3 to 1.4 each.

RELATED:
np.mean and np.std give the mean and spread. ranger maps one range onto another.
np.percentile gives a scale that ignores outliers.
""",
metro() + [
    C('c1', 'heights and weights: four rows of two', 180, 53),
    N('mh', 'message', 180, 92, 300, 42, MSG('1.6 60 1.7 72 1.8 80 1.9 95')),
    N('rh', 'np.reshape 4 2', 510, 92, 150, 90),
    C('c2', 'each column on its own scale', 180, 230),
    N('zs', 'scipy.zscore 0', 180, 265, 160, 90),
    N('hz', 'heat_map', 380, 265, 168, 210, HM(150, 160, 1, -1.5, 1.5, '%.2f', 'red-blue')),
],
[('mt', '', 'mh', ''), ('mh', 'message out', 'rh', 'tensor in'), ('rh', 'output', 'zs', 'in'),
 ('zs', 'z', 'hz', 'y')],
['np.mean', 'np.std', 'ranger', 'scipy.kmeans'], 540, 900)


# ------------------------------------------------------------------ linregress
page('scipy.linregress', 'scipy.linregress - the best straight line, and how good it is',
"""
Fits a straight line through points - y = slope x + intercept - and also says how good
the fit is: r, how closely the points follow a line (1 or -1 exactly, 0 not at all), and
p, how likely a slope that strong would be by chance if there were really no trend.

Use it to measure a trend and judge it at the same time: is a performer really speeding
up across a phrase or is it noise; does a joint's angle really follow a fader; is a
sensor drifting. A p below about 0.05 is conventionally taken as a real trend.

SYNTAX:
scipy.linregress

INPUTS and PARAMETERS:

y:
The values. Every input is fitted and sent - the only inlet that makes the node send.

x:
Where the values are, held. Without x they are at 0, 1, 2 ... Pairs with nan are left
out; at least 3 are needed.

OUTPUTS:

slope:
How much y changes per step of x.

intercept:
The value of the line at x = 0.

r:
How closely the points follow the line: from -1 (falling exactly) through 0 to 1
(rising exactly).

p:
The chance of a slope this strong if there were no real trend. Sent first, then r,
intercept and slope.

READING THE DEMO:
Five values rising by about 2 each step, from about 1: slope 1.99, intercept 1.04, r
0.999 - an almost perfect line - and p 0.00006, so the trend is real. A metro re-sends
the values twice a second.

RELATED:
np.polyfit fits a line (or curve) without judging it. scipy.spearmanr measures a trend
that need not be straight. scipy.curve_fit fits any formula.
""",
metro() + [
    C('c1', 'five values, rising', 180, 53),
    N('my', 'message', 180, 92, 240, 42, MSG('1.1 2.9 5.2 6.8 9.1')),
    C('c2', 'the line, and how good', 450, 53),
    N('lr', 'scipy.linregress', 450, 92, 180, 130),
    C('c3', 'slope', 28, 260),
    ROUND('r1', 28, 295, 3),
    N('l1', 'list', 28, 380, 100, 42, LIST),
    C('c4', 'intercept', 200, 260),
    ROUND('r2', 200, 295, 3),
    N('l2', 'list', 200, 380, 100, 42, LIST),
    C('c5', 'r', 370, 260),
    ROUND('r3', 370, 295, 3),
    N('l3', 'list', 370, 380, 100, 42, LIST),
    C('c6', 'p', 540, 260),
    ROUND('r4', 540, 295, 5),
    N('l4', 'list', 540, 380, 120, 42, LIST),
],
[('mt', '', 'my', ''), ('my', 'message out', 'lr', 'y'),
 ('lr', 'slope', 'r1', 'in'), ('r1', 'result', 'l1', ''), ('lr', 'intercept', 'r2', 'in'),
 ('r2', 'result', 'l2', ''), ('lr', 'r', 'r3', 'in'), ('r3', 'result', 'l3', ''),
 ('lr', 'p', 'r4', 'in'), ('r4', 'result', 'l4', '')],
['np.polyfit', 'scipy.spearmanr', 'scipy.curve_fit', 'np.corrcoef'], 480, 1000)


# ------------------------------------------------------------------- spearmanr
page('scipy.spearmanr', 'scipy.spearmanr - do two things rise and fall together',
"""
Measures whether two lists rise and fall together, using only their ORDER: rho is 1 if
the larger a is, the larger b always is, -1 if the larger a, the smaller b, and near 0 if
there is no consistent pattern. p is the chance of so strong a pattern if the two were
unrelated.

Unlike the ordinary correlation (np.corrcoef), the relationship need not be a straight
line - anything that only ever goes up counts as perfect - and a single extreme value
cannot dominate. Use it for relationships that are consistent but curved (effort and
speed, loudness and size), for ranks and ratings, and for data with outliers.

SYNTAX:
scipy.spearmanr

INPUTS and PARAMETERS:

a:
The first list. Every input is compared and sent - the only inlet that makes the node
send.

b:
The second list, the same length, held. At least 3 values.

OUTPUTS:

rho:
From -1 to 1: how consistently the two go up and down together.

p:
The chance of a pattern this strong if they were unrelated. Sent first.

READING THE DEMO:
1 to 5 against 1 4 9 16 100: always rising together, though not in a line and with a
jump at the end - rho is 1, a perfect match by order. 1 to 5 against 5 3 4 1 2: mostly
falling - rho -0.8.

RELATED:
np.corrcoef measures straight-line correlation (0.80 for the first pair). scipy.
linregress fits the line and judges it.
""",
metro() + [
    C('c1', 'a: 1 to 5', 180, 53),
    N('ma', 'message', 180, 92, 160, 42, MSG('1 2 3 4 5')),
    C('c2', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 210, 88, 46),
    C('c3', 'b: always rising, not in a line', 180, 175),
    N('mb', 'message', 180, 210, 180, 42, MSG('1 4 9 16 100')),
    C('c4', 'by order: perfect', 420, 175),
    N('s1', 'scipy.spearmanr', 420, 210, 170, 90),
    ROUND('r1', 620, 210, 2),
    N('l1', 'list', 740, 210, 80, 42, LIST),
    C('c5', 'b: mostly falling', 180, 330),
    N('mc', 'message', 180, 365, 160, 42, MSG('5 3 4 1 2')),
    N('s2', 'scipy.spearmanr', 420, 365, 170, 90),
    ROUND('r2', 620, 365, 2),
    N('l2', 'list', 740, 365, 80, 42, LIST),
],
[('lb', 'out', 'mb', ''), ('lb', 'out', 'mc', ''), ('mb', 'message out', 's1', 'b'),
 ('mc', 'message out', 's2', 'b'), ('mt', '', 'ma', ''), ('ma', 'message out', 's1', 'a'),
 ('ma', 'message out', 's2', 'a'), ('s1', 'rho', 'r1', 'in'), ('r1', 'result', 'l1', ''),
 ('s2', 'rho', 'r2', 'in'), ('r2', 'result', 'l2', '')],
['np.corrcoef', 'scipy.linregress', 'np.argsort', 'scipy.correlate'], 500, 1000)


# --------------------------------------------------------------------- entropy
page('scipy.entropy', 'scipy.entropy - how unpredictable a set of outcomes is',
"""
Measures how unpredictable something is from how often each of its possible outcomes
occurs: 0 when one outcome is certain, largest when every outcome is equally likely. In
bits, it is how many yes/no questions it takes on average to guess the outcome: a fair
coin is 1 bit, a fair four-way choice 2.

Use it to measure variety: how evenly a performer uses the space (counts per zone from
np.histogram2d), how varied the gestures in a phrase are (counts per cluster from
scipy.kmeans and np.bincount), how predictable a sequence of states is, how spread a
distribution of directions is.

SYNTAX:
scipy.entropy

INPUTS and PARAMETERS:

counts:
How often each outcome occurs (or each one's probability): a list of numbers 0 or more.
They need not add up to anything; they are scaled to total 1. Every input is measured
and sent.

units:
An option, under the node: bits (base 2, the default), natural (base e) or digits (base
10).

OUTPUTS:

entropy:
The unpredictability.

evenness:
The entropy as a fraction of the most possible for that many outcomes: 1 when all are
equally likely, 0 when one is certain - comparable between lists of different lengths.
Sent first.

READING THE DEMO:
A fair coin (5 5): 1 bit. A coin that lands heads 9 times in 10 (9 1): 0.47 bits - much
more predictable. Four equally likely outcomes (1 1 1 1): 2 bits. A metro re-sends the
counts twice a second.

RELATED:
np.histogram, np.bincount and np.unique count outcomes. scipy.circstd measures the
spread of angles.
""",
metro() + [
    C('c1', 'a fair coin', 180, 53),
    N('m1', 'message', 180, 92, 100, 42, MSG('5 5')),
    N('e1', 'scipy.entropy', 380, 92, 160, 90),
    ROUND('r1', 580, 92, 2),
    N('l1', 'list', 700, 92, 80, 42, LIST),
    C('c2', 'a coin that lands heads 9 times in 10', 180, 210),
    N('m2', 'message', 180, 245, 100, 42, MSG('9 1')),
    N('e2', 'scipy.entropy', 380, 245, 160, 90),
    ROUND('r2', 580, 245, 2),
    N('l2', 'list', 700, 245, 80, 42, LIST),
    C('c3', 'four equally likely', 180, 360),
    N('m3', 'message', 180, 395, 120, 42, MSG('1 1 1 1')),
    N('e3', 'scipy.entropy', 380, 395, 160, 90),
    ROUND('r3', 580, 395, 2),
    N('l3', 'list', 700, 395, 80, 42, LIST),
],
[('mt', '', 'm1', ''), ('mt', '', 'm2', ''), ('mt', '', 'm3', ''),
 ('m1', 'message out', 'e1', 'counts'), ('m2', 'message out', 'e2', 'counts'), ('m3', 'message out', 'e3', 'counts'),
 ('e1', 'entropy', 'r1', 'in'), ('r1', 'result', 'l1', ''), ('e2', 'entropy', 'r2', 'in'),
 ('r2', 'result', 'l2', ''), ('e3', 'entropy', 'r3', 'in'), ('r3', 'result', 'l3', '')],
['np.histogram', 'np.bincount', 'np.unique', 'scipy.circstd'], 530, 1000)


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
                    text_width=760, text_height=p['text_height'], subpatch=p['subpatch']))


if __name__ == '__main__':
    main()
