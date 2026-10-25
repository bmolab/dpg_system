"""Seed the help pages for the second batch of numpy nodes (2026-10-24): one page per node.

    python3 dpg_system/help/tools/seed_numpy_batch2_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

np.ptp, np.unique, np.cov, np.corrcoef, np.gradient, np.fft.rfft, np.fft.irfft,
np.take, np.pad, np.tile, np.digitize, np.arange, np.eye, np.diag, np.polyfit,
np.polyval, np.linalg.svd, np.linalg.eigh.

Like seed_numpy_more_help.py (whose building blocks it uses) it is not a make_*
script, so rebuild.py never runs it, and it will not write over an existing page
without --force. Treat the pages as hand-made once written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from help_common import SIG, PLOT
from seed_numpy_more_help import LIST, MSG, HM, BARS, C, N, see_also


METRO = {'on': True, 'period': 200.0, 'units': 'milliseconds'}
METRO_SLOW = {'on': True, 'period': 500.0, 'units': 'milliseconds'}

PAGES = {}


def page(stem, title, body, demo, links, also, also_y, text_height=820):
    PAGES[stem] = dict(title=title, body=body.strip('\n'), demo=demo + see_also(also_y, also),
                       links=links, text_height=text_height)


def metro_row(y=53, label='five times a second', props=METRO, init='metro 200'):
    """The metro that re-sends a demo's input, with its comment."""
    return [C('cm', label, 28, y), N('mt', init, 28, y + 39, 129, 70, props)]


# ------------------------------------------------------------------- ptp
page('np.ptp', 'np.ptp - the range from lowest to highest',
"""
Gives the distance from the smallest value to the largest: the maximum minus the
minimum. ptp stands for "peak to peak".

It is the simplest measure of how much something varies. Over a window of recent
samples it says how far a sensor has moved, how loud a signal swings, how much a
joint has travelled - and near zero means stillness. Because it rests entirely on
the two most extreme values, a single glitch sets it; np.std or np.percentile give a
steadier measure of spread when the data has outliers.

SYNTAX:
np.ptp <axis: int>
Default: axis None (the whole array).

EXAMPLE:
np.ptp 0

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input is measured and sent.

axis (menu, also an inlet):
None measures over every element. A number measures along that axis: for a 2-D
array, 0 gives one range per column, 1 one per row.

OUTPUTS:

ptp:
The range: a single value, or one per column or row.

READING THE DEMO:
Top: 3 9 1 7 runs from 1 to 9, a range of 8.

Bottom: the same idea column by column. The 3 x 2 array has columns 1 4 2 (range 3)
and 5 6 5 (range 1). A metro re-sends both five times a second; edit a message and
click it.

RELATED:
np.amax and np.amin give the two ends separately. np.std measures spread from every
value, not just the two extremes. np.percentile 5 and 95 give a range that ignores
the most extreme tenth.
""",
metro_row() + [
    C('c1', 'four values', 180, 53),
    N('m1', 'message', 180, 92, 140, 42, MSG('3 9 1 7')),
    C('c2', 'highest minus lowest', 360, 53),
    N('n1', 'np.ptp', 360, 92, 150, 90),
    N('l1', 'list', 560, 92, 80, 42, LIST),
    C('c3', 'six values as 3 rows of 2', 180, 230),
    N('m2', 'message', 180, 265, 160, 42, MSG('1 5 4 6 2 5')),
    N('rs', 'np.reshape 3 2', 180, 340, 150, 90),
    C('c4', 'the range of each column', 400, 230),
    N('n2', 'np.ptp 0', 400, 265, 150, 90, {'axis': '0'}),
    N('l2', 'list', 580, 265, 120, 42, LIST),
],
[('mt', '', 'm1', ''), ('mt', '', 'm2', ''),
 ('m1', 'message out', 'n1', 'in'), ('n1', 'ptp', 'l1', ''),
 ('m2', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'n2', 'in'), ('n2', 'ptp', 'l2', '')],
['np.amax', 'np.amin', 'np.std', 'np.percentile'], 490, 760)


# ----------------------------------------------------------------- unique
page('np.unique', 'np.unique - the distinct values, and how often each occurs',
"""
Lists each different value in an array once, in increasing order, with how many
times it occurs - and, for every element of the input, which of those values it is.

Use it to find which labels, notes, states or zones turned up in a stream of them
and how often; to count votes or classifications; or, with the inverse, to turn
arbitrary values (MIDI notes 60 64 67) into consecutive numbers (0 1 2) that can
index a list, a colour table or a set of outlets.

SYNTAX:
np.unique

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every element is counted, whatever the
shape. Every input is analysed and sent.

OUTPUTS:

values:
The distinct values, smallest first.

counts:
How many times each of them occurs.

inverse:
For each element of the input, the position of its value in values - the input
re-labelled 0, 1, 2 ... It has the input's shape. Sent first, then counts, then
values.

READING THE DEMO:
3 1 3 2 3 holds three different values, 1 2 3. 1 and 2 occur once each and 3 three
times. The inverse, 2 0 2 1 2, is the input rewritten as positions in 1 2 3: the 3s
became 2, the 1 became 0. A metro re-sends the input five times a second; edit it
and click.

RELATED:
np.histogram counts values falling into ranges rather than exact values.
np.sort puts every value in order, repeats and all. np.take turns positions back
into values.
""",
metro_row() + [
    C('c1', 'five values, three different', 180, 53),
    N('m1', 'message', 180, 92, 160, 42, MSG('3 1 3 2 3')),
    C('c2', 'find the distinct ones', 400, 53),
    N('n1', 'np.unique', 400, 92, 150, 90),
    C('c3', 'values', 28, 230),
    N('l1', 'list', 28, 265, 140, 42, LIST),
    C('c4', 'counts', 200, 230),
    N('l2', 'list', 200, 265, 140, 42, LIST),
    C('c5', 'inverse: each input as a position', 380, 230),
    N('l3', 'list', 380, 265, 200, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'n1', 'in'),
 ('n1', 'values', 'l1', ''), ('n1', 'counts', 'l2', ''), ('n1', 'inverse', 'l3', '')],
['np.histogram', 'np.sort', 'np.take', 'np.digitize'], 380, 760)


# ------------------------------------------------------------ cov / corrcoef
DATA3 = '1 2 3 2 4 1 3 5 4 4 8 1 5 10 5'

COV_LAYOUT = lambda node, out, props, note: metro_row() + [
    C('c1', 'five rows (observations) of three columns (variables)', 180, 53),
    N('m1', 'message', 180, 92, 330, 42, MSG(DATA3)),
    N('rs', 'np.reshape 5 3', 540, 92, 150, 90),
    C('c2', 'as a 5 x 3 array: a b c', 28, 230),
    N('hd', 'heat_map', 28, 265, 168, 210, HM(150, 150, 1, 0.0, 10.0, '%.0f')),
    C('c3', 'every pair of columns', 260, 230),
    N('cv', node, 260, 265, 150, 70),
    C('c4', note, 450, 230),
    N('hc', 'heat_map', 450, 265, 228, 150, props),
]

COV_LINKS = lambda out: [('mt', '', 'm1', ''), ('m1', 'message out', 'rs', 'tensor in'),
                         ('rs', 'output', 'hd', 'y'), ('rs', 'output', 'cv', 'in'),
                         ('cv', out, 'hc', 'y')]

page('np.cov', 'np.cov - how pairs of variables vary together',
"""
Takes many observations of several variables - one row per moment, one column per
sensor, joint or feature - and gives, for every pair of variables, how much they vary
together: the covariance.

A positive covariance means the two tend to rise and fall together; negative, that
one rises as the other falls; near zero, that they are unrelated. On the diagonal
each variable is paired with itself, which is just its variance. The size of the
numbers depends on the units and the spread of the variables; np.corrcoef scales
them to -1 to 1 for comparing.

The covariance matrix is the starting point of principal component analysis:
np.linalg.eigh of it gives the directions along which the data varies most - the
dominant axis of a gesture, the main modes of a set of joints moving together.

SYNTAX:
np.cov <variables: rows or columns>
Default: columns.

INPUTS and PARAMETERS:

in:
A 2-D array of observations, with at least two of each variable. Every input is
computed and sent.

variables:
An option, under the node. columns (the default here) takes each column as a
variable and each row as one observation - the shape a rolling buffer of samples
has. rows takes each row as a variable, as NumPy itself does by default.

OUTPUTS:

covariance:
A square matrix with one row and column per variable.

READING THE DEMO:
Five observations of a, b and c. b is roughly twice a, so they vary together
strongly: covariance 5. c wanders on its own: its covariances with a and b (1 and
1.7) are small for their spread. The diagonal holds each variable's variance: a
2.5, b 10.2 (twice the size, four times the variance), c 3.2.

RELATED:
np.corrcoef gives the same relations scaled to -1 to 1. np.var is one variable's
variance. np.linalg.eigh of a covariance matrix gives its principal directions.
""",
COV_LAYOUT('np.cov', 'covariance', HM(210, 120, 1, -10.0, 10.0, '%.1f', 'red-blue'),
           'covariance of each pair'),
COV_LINKS('covariance'),
['np.corrcoef', 'np.var', 'np.linalg.eigh', 'np.rolling_buffer'], 540, 900)


page('np.corrcoef', 'np.corrcoef - how closely pairs of variables move together',
"""
Takes many observations of several variables - one row per moment, one column per
sensor, joint or feature - and gives, for every pair, how closely they move
together: the correlation coefficient, from -1 to 1.

1 means the two rise and fall in perfect step (one is a scaled, shifted copy of the
other); -1 that one falls exactly as the other rises; 0 that knowing one tells you
nothing about the other. Unlike np.cov the answer does not depend on units or size:
a hand's height in metres and a fader in 0 to 127 can be compared directly.

Use it to find which sensors are redundant, which joints move as a unit, whether two
performers are in sync over a window of time, or how well a gesture follows a
reference.

SYNTAX:
np.corrcoef <variables: rows or columns>
Default: columns.

INPUTS and PARAMETERS:

in:
A 2-D array of observations, with at least two of each variable. Every input is
computed and sent.

variables:
An option, under the node. columns (the default here) takes each column as a
variable and each row as one observation - the shape a rolling buffer of samples
has. rows takes each row as a variable, as NumPy itself does by default.

OUTPUTS:

correlation:
A square matrix with one row and column per variable, 1 on the diagonal. A
variable that never changes has no correlation with anything: its row and column
are nan.

READING THE DEMO:
Five observations of a, b and c. b is roughly twice a: correlation 0.99, nearly
perfect, though b is twice the size. c wanders on its own: about 0.3 with each of the
others - a weak link, probably chance in only five observations.

RELATED:
np.cov gives the unscaled covariances. t.corrcoef is the PyTorch version;
np.correlate finds a shape within a signal rather than comparing whole variables.
""",
COV_LAYOUT('np.corrcoef', 'correlation', HM(210, 120, 1, -1.0, 1.0, '%.2f', 'red-blue'),
           'correlation of each pair'),
COV_LINKS('correlation'),
['np.cov', 't.corrcoef', 'np.correlate', 'np.rolling_buffer'], 540, 860)


# --------------------------------------------------------------- gradient
page('np.gradient', 'np.gradient - the rate of change at every point',
"""
Gives the slope of an array at every point: how fast the values are changing there.
Inside the array it takes the centred difference - half the distance from the value
before to the value after - and at the two ends the one-sided difference.

Unlike np.diff, the result is the same length as the input and each value is centred
on its own point, not between two. That makes it the natural way to get velocities
from a recorded path (and accelerations from those), slopes from a curve, or edges
from a row of an image, while keeping everything lined up with the original.

SYNTAX:
np.gradient <spacing: float> <axis: int>
Defaults: spacing 1, axis -1 (along each row).

EXAMPLE:
np.gradient 0.01667

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input is differentiated and sent.

spacing (float box, also an inlet):
The distance between neighbouring values. The slopes are divided by it, so with the
time between samples (1/60 for 60 frames a second) the result is per second.

axis (int box, also an inlet):
Which way to measure: -1 along each row, 0 down each column.

edge order:
An option, under the node: 1 (the default) or 2. 2 uses three points at each end,
which follows curves more closely there.

OUTPUTS:

gradient:
The slope at each point, the same shape as the input.

READING THE DEMO:
The squares 0 1 4 9 16 25 are a path speeding up. np.gradient gives 1 2 4 6 8 9: at
each point, its speed. Inside, these are exactly twice the position's index, as they
should be for squares; the two ends are rougher. np.diff gives 1 3 5 7 9 - the
steps BETWEEN points, one fewer. A metro re-sends the path five times a second.

RELATED:
np.diff gives the steps between neighbours. np.cumsum goes the other way, from
steps back to a path. diff measures change in a stream, one sample at a time.
""",
metro_row() + [
    C('c1', 'a path speeding up', 180, 53),
    N('m1', 'message', 180, 92, 200, 42, MSG('0 1 4 9 16 25')),
    C('c2', 'the speed at each point', 420, 53),
    N('g1', 'np.gradient', 420, 92, 150, 110),
    N('l1', 'list', 600, 92, 220, 42, LIST),
    C('c3', 'the steps between points: one fewer', 420, 240),
    N('d1', 'np.diff', 420, 275, 150, 90),
    N('l2', 'list', 600, 275, 220, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'g1', 'in'), ('m1', 'message out', 'd1', 'in'),
 ('g1', 'gradient', 'l1', ''), ('d1', 'differences', 'l2', '')],
['np.diff', 'np.cumsum', 'diff', 't.diff'], 430, 840)


# -------------------------------------------------------------- fft.rfft
WAVE = [
    C('cw', 'a wave: 64 samples, 8 cycles', 180, 53),
    N('ar', 'np.arange 64', 180, 92, 150, 110),
    N('ml', '* 45', 360, 92, 129, 70),
    N('sn', 'sin', 360, 180, 129, 70),
]
WAVE_LINKS = [('mt', '', 'ar', ''), ('ar', 'arange out', 'ml', 'in'), ('ml', 'result', 'sn', 'in')]

page('np.fft.rfft', 'np.fft.rfft - the frequencies in a signal',
"""
Breaks a signal into the sine waves that make it up: the Fourier transform. For each
frequency from 0 (the steady average) up to half the sampling rate, it says how much
of that wave the signal contains (the magnitude) and where in its cycle it starts
(the phase).

It turns a window of samples - sound, a joint's motion over the last two seconds, a
row of pixels - into a spectrum: which rhythms, vibrations or textures are in it and
how strong each is. A shake shows as a peak at the shake's frequency; a slow sway as
energy near 0; noise as energy spread across all of them.

rfft is the version for real signals (all ordinary sensor data). A window of N
samples gives N/2 + 1 frequencies.

SYNTAX:
np.fft.rfft <sample rate: float>
Default: sample rate 1.

EXAMPLE:
np.fft.rfft 60

INPUTS and PARAMETERS:

in:
The signal: a list or array of real values. Every input is transformed and sent.

sample rate (float box, also an inlet):
Samples per second. Only the frequencies outlet uses it: with 60 for a 60 frames per
second stream, the frequencies come out in cycles per second (Hz).

axis (int box, also an inlet):
For a 2-D array, which way the signal runs: -1 (the default) along each row, 0 down
each column - one spectrum per joint for a buffer of frames x joints.

norm:
An option, under the node: how the result is scaled. backward (the default) leaves
the forward transform unscaled; ortho scales both ways equally; forward divides by N.

OUTPUTS:

spectrum:
The complex result, for np.fft.irfft or further calculation.

magnitude:
How much of each frequency there is.

phase:
Where each wave starts in its cycle, in radians (-pi to pi).

frequencies:
The frequency of each entry, from 0 to half the sample rate. Sent first.

READING THE DEMO:
np.arange 64, times 45 degrees, through sin: a sine wave that goes round 8 times in 64
samples. Its spectrum is a single spike at entry 8, and nothing anywhere else.
Change 45 to 90 and the spike moves to 16; to 22.5, and it moves to 4. A frequency
that does not fit a whole number of times in the window (try 40) smears across its
neighbours.

RELATED:
np.fft.irfft turns a spectrum back into a signal. t.rfft is the PyTorch version, and
spectrum analyses a stream as it arrives. The t.window nodes taper a window's ends,
which reduces the smearing.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + WAVE + [
    C('c1', 'in: the wave', 28, 290),
    N('p1', 'plot', 28, 325, 368, 160, dict(BARS(64, 1.0, 360, 110, 'line'), **{'min y': -1.0})),
    C('c2', 'to frequencies', 540, 53),
    N('ff', 'np.fft.rfft 64', 540, 92, 180, 150),
    C('c3', 'out: magnitude - one spike, at 8', 430, 290),
    N('p2', 'plot', 430, 325, 368, 160, BARS(33, 40.0, 360, 110)),
],
WAVE_LINKS + [('sn', 'out', 'p1', 'y'), ('sn', 'out', 'ff', 'in'), ('ff', 'magnitude', 'p2', 'y')],
['np.fft.irfft', 't.rfft', 'spectrum', 't.window.hann'], 540, 1250)


page('np.fft.irfft', 'np.fft.irfft - a signal from its frequencies',
"""
Turns a spectrum back into a signal: the inverse of np.fft.rfft. Each frequency's
sine wave is rebuilt at its magnitude and phase, and all of them are added up.

Round trip, it gives back the original signal exactly. The point is to change the
spectrum in between: remove frequencies (a filter as sharp as you like), keep only
the strongest ones (compression, denoising), smear or shift them, or put one
signal's magnitudes with another's phases.

SYNTAX:
np.fft.irfft <length: int>
Default: length 0 (2 x (frequencies - 1): an even-length signal).

EXAMPLE:
np.fft.irfft 64

INPUTS and PARAMETERS:

spectrum:
The complex spectrum from np.fft.rfft's spectrum outlet. Every input is transformed
and sent. Real values are taken as magnitudes instead, combined with the phases in the
phase inlet.

phase:
Phases in radians, held, to go with magnitudes arriving at spectrum (from rfft's
magnitude and phase outlets). Without them, or if their shape does not match, every
phase is 0. Not used when the spectrum is complex.

length (int box, also an inlet):
How many samples to make. Set it to the original window length: an odd length cannot
be told from the spectrum alone, and 0 always gives an even one.

axis (int box, also an inlet):
For a 2-D spectrum, which way the frequencies run: -1 along each row, 0 down each
column.

norm:
An option, under the node: must match the norm used for np.fft.rfft (backward,
the default, for both).

OUTPUTS:

signal:
The rebuilt signal.

READING THE DEMO:
64 random values (left) go to frequencies. np.take keeps only the first 6 of the 33
- the steady average and the 5 slowest waves - and np.pad puts 27 zeros back in place
of the rest. Rebuilt, the jagged noise becomes a smooth curve that follows its slow
drift (right): a low-pass filter with a perfectly sharp cut. Change the 27 and the 6
of np.take together to keep more or fewer.

RELATED:
np.fft.rfft makes the spectrum. t.irfft is the PyTorch version. np.convolve filters
without going through frequencies.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', '64 random values', 180, 53),
    N('rn', 'np.rand 64', 180, 92, 150, 110),
    C('c2', 'to frequencies', 360, 53),
    N('ff', 'np.fft.rfft', 360, 92, 180, 150),
    C('c3', 'keep the 6 slowest', 580, 53),
    N('tk', 'np.take 0 1 2 3 4 5', 580, 92, 200, 110),
    C('c4', 'zeros for the other 27', 580, 230),
    N('pd', 'np.pad 0 27', 580, 265, 150, 170),
    C('c5', 'back to a signal', 580, 470),
    N('if', 'np.fft.irfft 64', 580, 505, 180, 130),
    C('c6', 'in: noise', 28, 290),
    N('p1', 'plot', 28, 325, 248, 160, BARS(64, 1.0, 240, 110, 'line')),
    C('c7', 'out: its slow drift', 300, 470),
    N('p2', 'plot', 300, 505, 248, 160, BARS(64, 1.0, 240, 110, 'line')),
],
[('mt', '', 'rn', ''), ('rn', 'random array', 'ff', 'in'), ('rn', 'random array', 'p1', 'y'),
 ('ff', 'spectrum', 'tk', 'in'), ('tk', 'taken', 'pd', 'in'), ('pd', 'padded', 'if', 'spectrum'),
 ('if', 'signal', 'p2', 'y')],
['np.fft.rfft', 't.irfft', 'np.convolve', 'np.take'], 720, 1250)


# ------------------------------------------------------------------ take
page('np.take', 'np.take - pick elements by their positions',
"""
Picks elements out of an array by a list of positions, in any order and as often as
you like. np.take 4 0 2 of 10 20 30 40 50 gives 50 10 30.

Use it to reorder (a joint list into another skeleton's order), to select a subset
(the arm joints from a full body), to duplicate, or to look values up: a list of
indices from np.argsort, np.digitize or np.unique's inverse, taken from a table of
colours, notes or names.

Along axis 0 of a 2-D array it picks whole rows; along axis 1, whole columns.

SYNTAX:
np.take <indices: ints ...>
Default: no indices (send them to the indices inlet).

EXAMPLE:
np.take 0 2 4

INPUTS and PARAMETERS:

in:
The array to pick from (a list is turned into one). Every input is picked from and
sent.

indices:
The positions, replacing any given as arguments. Held: send them first. A negative
position counts from the end: -1 is the last.

axis (int box, also an inlet):
Which way positions count: 0 (the default) along the first axis - single elements of
a list, whole rows of a 2-D array; 1 columns.

mode:
An option, under the node: what happens to positions past the end. clip (the
default) gives the last element (or the first, for positions before the start);
wrap goes round again from the start, so position 5 of 3 elements is position 2.

OUTPUTS:

taken:
The picked elements, in the order of the indices.

READING THE DEMO:
Top: positions 4 0 2 of 10 20 30 40 50 are 50 10 30.

Bottom: the positions come from a message instead: -1 is the last element, and 0 1
twice shows the same element can be taken more than once. A metro re-sends the array
five times a second.

RELATED:
np.[] takes a slice - a run of neighbouring elements. np.argsort, np.argmax and
np.digitize make lists of positions; t.take is the PyTorch version.
""",
metro_row() + [
    C('c1', 'five values', 180, 53),
    N('m1', 'message', 180, 92, 180, 42, MSG('10 20 30 40 50')),
    C('c2', 'positions 4 0 2', 400, 53),
    N('t1', 'np.take 4 0 2', 400, 92, 160, 110),
    N('l1', 'list', 600, 92, 160, 42, LIST),
    C('c3', 'once at load', 28, 250),
    N('lb', 'load_bang', 28, 285, 88, 46),
    C('c4', 'positions from a message', 180, 250),
    N('mi', 'message', 180, 285, 160, 42, MSG('-1 0 1 0 1')),
    N('t2', 'np.take', 400, 285, 160, 110),
    N('l2', 'list', 600, 285, 200, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 't1', 'in'), ('m1', 'message out', 't2', 'in'),
 ('t1', 'taken', 'l1', ''), ('lb', 'out', 'mi', ''), ('mi', 'message out', 't2', 'indices'),
 ('t2', 'taken', 'l2', '')],
['np.[]', 'np.argsort', 'np.digitize', 't.take'], 460, 900)


# ------------------------------------------------------------------- pad
page('np.pad', 'np.pad - add values around the edges of an array',
"""
Makes an array longer by adding values before and after it along one axis. What the
new values are depends on the mode: a constant, copies of the edge value, a mirror
image of what is inside, or a wrap-around from the other end.

Use it to bring arrays to a common length before stacking or comparing them, to give
np.convolve or a smoothing filter something sensible to work on at the ends, to
leave a margin round an image, or to zero-pad a window before np.fft.rfft for a
finer-grained spectrum.

SYNTAX:
np.pad <before: int> <after: int> <mode>
Defaults: 1 before, 1 after, constant. A single number pads both sides.

EXAMPLES:
np.pad 2
np.pad 0 10
np.pad 3 edge

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input is padded and sent.

before (int box, also an inlet):
How many values to add before the start.

after (int box, also an inlet):
How many values to add after the end.

value (float box, also an inlet):
What the new values are in constant mode.

axis (int box, also an inlet):
Which way to pad: -1 (the default) along each row, 0 above and below the columns.

mode:
An option, under the node:
constant   the value, 0 unless set
edge       copies of the first or last value
reflect    a mirror image, not repeating the edge value (3 2 | 1 2 3 | 2 1)
symmetric  a mirror image that does repeat it (2 1 | 1 2 3 | 3 2)
wrap       continue from the other end, as if the array repeated

OUTPUTS:

padded:
The longer array.

READING THE DEMO:
1 2 3 padded by 2 on each side three ways: with zeros, 0 0 1 2 3 0 0; with the edge
values, 1 1 1 2 3 3 3; and wrapped round, 2 3 1 2 3 1 2. A metro re-sends the array
five times a second.

RELATED:
np.concatenate joins two arrays end to end. np.tile repeats a whole array.
np.crop and np.[] go the other way, cutting the edges off.
""",
metro_row() + [
    C('c1', 'three values', 180, 53),
    N('m1', 'message', 180, 92, 120, 42, MSG('1 2 3')),
    C('c2', 'zeros', 340, 53),
    N('p1', 'np.pad 2', 340, 92, 150, 190),
    N('l1', 'list', 520, 92, 200, 42, LIST),
    C('c3', 'the edge values', 340, 310),
    N('p2', 'np.pad 2 edge', 340, 345, 150, 190),
    N('l2', 'list', 520, 345, 200, 42, LIST),
    C('c4', 'wrapped round', 340, 565),
    N('p3', 'np.pad 2 wrap', 340, 600, 150, 190),
    N('l3', 'list', 520, 600, 200, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'p1', 'in'), ('m1', 'message out', 'p2', 'in'),
 ('m1', 'message out', 'p3', 'in'), ('p1', 'padded', 'l1', ''), ('p2', 'padded', 'l2', ''),
 ('p3', 'padded', 'l3', '')],
['np.concatenate', 'np.tile', 'np.crop', 'np.convolve'], 830, 1050)


# ------------------------------------------------------------------ tile
page('np.tile', 'np.tile - repeat a whole array',
"""
Repeats an array end to end - and, with more than one count, in rows and columns as
well, like tiles on a floor.

np.tile 3 of 1 2 3 gives 1 2 3 1 2 3 1 2 3: the whole array three times over.
np.repeat is different: it repeats each ELEMENT in place (1 1 1 2 2 2 3 3 3).

Use it to loop a pattern to fill a length, to copy one row into every row of a
matrix (one offset for every frame, one colour for every point), or to build a
grid from a single row.

SYNTAX:
np.tile <counts: ints ...>
Default: 2.

EXAMPLES:
np.tile 4
np.tile 3 1

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input is repeated and sent.

reps (text box, also an inlet):
How many times to repeat along each axis, the last count for the last axis. 3 repeats
along the row; 2 1 makes 2 rows of the original row; 2 2 a 2 x 2 block of copies.

OUTPUTS:

tiled:
The repeated array.

READING THE DEMO:
1 2 3 tiled 3 times along its length, and 2 1 times: two rows, each a single copy.
A metro re-sends the array five times a second; edit the counts in reps.

RELATED:
np.repeat repeats each element in place. np.stack and np.concatenate join different
arrays. t.tile is the PyTorch version.
""",
metro_row() + [
    C('c1', 'three values', 180, 53),
    N('m1', 'message', 180, 92, 120, 42, MSG('1 2 3')),
    C('c2', 'three times along', 340, 53),
    N('t1', 'np.tile 3', 340, 92, 160, 70),
    N('l1', 'list', 540, 92, 240, 42, LIST),
    C('c3', 'as two rows', 340, 200),
    N('t2', 'np.tile 2 1', 340, 235, 160, 70),
    N('l2', 'list', 540, 235, 240, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 't1', 'in'), ('m1', 'message out', 't2', 'in'),
 ('t1', 'tiled', 'l1', ''), ('t2', 'tiled', 'l2', '')],
['np.repeat', 'np.stack', 'np.concatenate', 't.tile'], 380, 760)


# -------------------------------------------------------------- digitize
page('np.digitize', 'np.digitize - which bin each value falls in',
"""
Sorts values into bins marked by a list of edges, and gives each value the number of
its bin. With edges -0.5 0 0.5, anything below -0.5 is bin 0, -0.5 up to 0 is bin 1, 0
up to 0.5 is bin 2, and 0.5 or more is bin 3.

Use it to turn a continuous control into steps or zones - a fader into one of five
scenes, a hand's height into low, middle and high, a speed into still, slow, fast -
with the boundaries wherever you want them, not evenly spaced. The bin numbers can
then pick from a list with np.take, or route to outlets.

SYNTAX:
np.digitize <edges: floats ...>
Default: no edges (send them to the edges inlet).

EXAMPLE:
np.digitize 0.2 0.4 0.8

INPUTS and PARAMETERS:

in:
The values: a number, list or array of any shape. Every input is sorted and sent.

edges:
The bin boundaries, in increasing (or decreasing) order, replacing any given as
arguments. Held: send them first. N edges make N + 1 bins, numbered 0 to N.

right:
An option, under the node. Off (the default), a value exactly on an edge goes into
the bin above it; on, into the bin below.

OUTPUTS:

bin indices:
The bin number of each value, the shape of the input.

READING THE DEMO:
Top: a sine wave from -1 to 1 through edges -0.5 0 0.5 becomes a staircase of 0 1 2
3 - flat while the wave stays inside a zone, stepping as it crosses an edge. The
outer steps are wider than the inner ones because the wave lingers near its peaks.

Bottom: -0.9 -0.2 0.3 0.8 fall in bins 0 1 2 3.

RELATED:
np.histogram counts how many values fall in each bin. np.take turns bin numbers
into values. round and floor make evenly spaced steps.
""",
[
    C('c1', 'a sine wave, -1 to 1', 28, 53),
    N('sg', 'signal 3.0 sin', 28, 92, 129, 78, SIG('sin', 3.0)),
    C('c2', 'edges at -0.5, 0 and 0.5', 200, 53),
    N('dg', 'np.digitize -0.5 0 0.5', 200, 92, 220, 70),
    C('c3', 'its bin: 0 1 2 or 3', 460, 53),
    N('pl', 'plot', 460, 92, 208, 178, PLOT(-0.5, 3.5, 300)),
    C('c4', 'five times a second', 28, 320),
    N('mt', 'metro 200', 28, 355, 129, 70, METRO),
    C('c5', 'four values', 180, 320),
    N('m1', 'message', 180, 355, 180, 42, MSG('-0.9 -0.2 0.3 0.8')),
    N('d2', 'np.digitize -0.5 0 0.5', 400, 355, 220, 70),
    N('l1', 'list', 640, 355, 140, 42, LIST),
],
[('sg', '', 'dg', 'in'), ('dg', 'bin indices', 'pl', 'y'),
 ('mt', '', 'm1', ''), ('m1', 'message out', 'd2', 'in'), ('d2', 'bin indices', 'l1', '')],
['np.histogram', 'np.take', 'floor', 'ranger'], 500, 900)


# ---------------------------------------------------------------- arange
page('np.arange', 'np.arange - evenly stepped values',
"""
Makes an array of values from start up to, but not including, stop, going up by
step: np.arange 5 is 0 1 2 3 4, np.arange 0 1 0.25 is 0 0.25 0.5 0.75.

It fixes the STEP and lets the count follow. np.linspace does the opposite: it fixes
the count and both ends, and works out the step. Use np.arange for indices (0 to N-1),
sample times at a known rate, or anything where the spacing matters more than the
exact end.

SYNTAX:
np.arange <stop>
np.arange <start> <stop> <step>
Defaults: start 0, stop 10, step 1. Whole-number arguments make whole-number
values; a decimal anywhere makes decimals.

EXAMPLES:
np.arange 64
np.arange 0 2 0.1

INPUTS and PARAMETERS:

(button):
Makes the array and sends it.

start (number box, also an inlet):
The first value.

stop (number box, also an inlet):
Where to stop: values reach up to it but never include it.

step (number box, also an inlet):
The step between values. A negative step counts down (start must then be above
stop). 0 makes nothing.

OUTPUTS:

arange out:
The array.

THE END IS LEFT OUT:
np.arange 0 1 0.25 stops at 0.75, not 1. With decimal steps the number of values can
also come out one more or fewer than expected because of rounding; when the exact
count matters, use np.linspace.

READING THE DEMO:
Clicked at load: np.arange 5 gives 0 1 2 3 4, and np.arange 0 1 0.25 gives 0 0.25 0.5
0.75 - no 1. Click a button to send again.

RELATED:
np.linspace fixes the count instead of the step. t.arange is the PyTorch version.
range_counter counts in a stream, one number at a time.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', 'up to 5, not including it', 180, 53),
    N('a1', 'np.arange 5', 180, 92, 150, 130),
    N('l1', 'list', 380, 92, 200, 42, LIST),
    C('c3', '0 to 1 in quarters, without the 1', 180, 260),
    N('a2', 'np.arange 0 1 0.25', 180, 295, 170, 130),
    N('l2', 'list', 380, 295, 260, 42, LIST),
],
[('lb', 'out', 'a1', ''), ('lb', 'out', 'a2', ''), ('a1', 'arange out', 'l1', ''),
 ('a2', 'arange out', 'l2', '')],
['np.linspace', 't.arange', 'range_counter', 'np.zeros'], 480, 960)


# ------------------------------------------------------------------- eye
page('np.eye', 'np.eye - a matrix of ones on a diagonal',
"""
Makes a matrix that is 1 along a diagonal and 0 everywhere else. Square, with the
main diagonal, it is the identity matrix: the matrix that changes nothing when you
multiply by it, as 1 does for ordinary numbers.

Use it as the starting point for a transformation you then modify, as the "no
change" reference to compare a rotation or fit against, to add a little to a
diagonal to steady an inversion, or (shifted off the main diagonal) as a matrix
that moves each element one place along.

SYNTAX:
np.eye <rows: int> <columns: int> <diagonal: int>
Defaults: 3 rows, square, the main diagonal.

EXAMPLES:
np.eye 4
np.eye 3 4 1

INPUTS and PARAMETERS:

(button):
Makes the matrix and sends it.

rows (int box, also an inlet):
How many rows.

columns (int box, also an inlet):
How many columns; 0 (the default) makes it square.

diagonal (int box, also an inlet):
Which diagonal holds the ones: 0 the main one, 1 the one just above it, -1 the one
just below.

OUTPUTS:

eye out:
The matrix (32-bit floats).

READING THE DEMO:
Clicked at load: a 3 x 3 identity, and a 3 x 4 matrix with its ones one place to the
right of the main diagonal. Change the numbers and click a button to see others.

RELATED:
np.diag makes a matrix with any values on the diagonal. t.eye is the PyTorch version.
np.zeros and np.ones fill a whole matrix.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', '3 x 3 identity', 180, 53),
    N('e1', 'np.eye 3', 180, 92, 150, 130),
    N('h1', 'heat_map', 380, 92, 168, 170, HM(150, 120, 1, 0.0, 1.0, '%.0f')),
    C('c3', '3 x 4, the diagonal above the main one', 180, 310),
    N('e2', 'np.eye 3 4 1', 180, 345, 150, 130),
    N('h2', 'heat_map', 380, 345, 218, 170, HM(200, 120, 1, 0.0, 1.0, '%.0f')),
],
[('lb', 'out', 'e1', ''), ('lb', 'out', 'e2', ''), ('e1', 'eye out', 'h1', 'y'),
 ('e2', 'eye out', 'h2', 'y')],
['np.diag', 't.eye', 'np.zeros', 'np.matmul'], 580, 900)


# ------------------------------------------------------------------ diag
page('np.diag', 'np.diag - put values on a diagonal, or read them off',
"""
Works both ways. Give it a list and it makes a square matrix with the list down its
diagonal and zeros elsewhere. Give it a matrix and it reads the diagonal off as a
list.

A diagonal matrix scales each axis separately - diag of 2 1 0.5 stretches x, leaves y
and squashes z - so it is the way to build a scaling for np.matmul. Reading the
diagonal gets the variances out of a covariance matrix, the ones from a correlation
check, or the strength of each axis.

SYNTAX:
np.diag <diagonal: int>
Default: 0, the main diagonal.

INPUTS and PARAMETERS:

in:
A list (to make a matrix) or a 2-D matrix (to read a diagonal). Every input is
converted and sent.

diagonal (int box, also an inlet):
Which diagonal: 0 the main one, 1 the one above it, -1 the one below. Making a
matrix with diagonal 1 makes it one larger, to leave room.

OUTPUTS:

diag out:
The matrix, or the list.

READING THE DEMO:
1 2 3 becomes a 3 x 3 matrix with 1 2 3 down the diagonal; a second np.diag reads them
back off. A metro re-sends the list five times a second; edit it and click.

RELATED:
np.eye makes a diagonal of ones. np.cov puts variances on its diagonal. t.diag is the
PyTorch version.
""",
metro_row() + [
    C('c1', 'three values', 180, 53),
    N('m1', 'message', 180, 92, 120, 42, MSG('1 2 3')),
    C('c2', 'onto a diagonal', 340, 53),
    N('d1', 'np.diag', 340, 92, 150, 90),
    N('h1', 'heat_map', 530, 92, 168, 170, HM(150, 120, 1, 0.0, 3.0, '%.0f')),
    C('c3', 'and read back off', 340, 300),
    N('d2', 'np.diag', 340, 335, 150, 90),
    N('l1', 'list', 530, 335, 140, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'd1', 'in'), ('d1', 'diag out', 'h1', 'y'),
 ('d1', 'diag out', 'd2', 'in'), ('d2', 'diag out', 'l1', '')],
['np.eye', 'np.cov', 't.diag', 'np.matmul'], 480, 760)


# --------------------------------------------------------------- polyfit
page('np.polyfit', 'np.polyfit - the curve that best fits some points',
"""
Finds the polynomial curve - a straight line, a parabola, a cubic... - that passes
as close as possible to a set of points, and gives its coefficients.

Degree 1 is a straight line: the two coefficients are its slope and its offset.
Degree 2 is a parabola, a + b x + c x x, and fits anything that curves once (a thrown
object's path, an arc of a gesture). Higher degrees follow more wiggles - and, past
a point, start following the noise instead of the shape.

Use it to find a trend in a window of data (and subtract it), to measure how fast
something is rising, to smooth a curve to its overall shape, or to calibrate a
sensor whose response bends. np.polyval draws the curve from the coefficients.

SYNTAX:
np.polyfit <degree: int>
Default: degree 1.

EXAMPLE:
np.polyfit 2

INPUTS and PARAMETERS:

y:
The values of the points. Every input is fitted and sent. A 2-D array fits each
column separately, one set of coefficients per column.

x:
Where the points are, held. Without it they are at 0, 1, 2 ...

degree (int box, also an inlet):
The degree of the curve. It needs at least one more point than its degree.

OUTPUTS:

coefficients:
Highest power first: for degree 2, c b a in c x x + b x + a. Ready for np.polyval.

READING THE DEMO:
Top: five noisy points rising by about 2 each step, from about 1. The line through
them has slope 1.99 and offset 1.04.

Bottom: five points near 0 1 4 9 16 - the squares. Degree 2 gives about 0.96 0.11
-0.01: close to 1 x x + 0 x + 0. A metro re-sends both five times a second; the
round nodes only tidy the numbers for reading.

RELATED:
np.polyval evaluates the curve. np.linalg.lstsq fits any combination of functions,
of which a polynomial is one. np.interp passes exactly through every point instead.
""",
metro_row() + [
    C('c1', 'noisy points along a line', 180, 53),
    N('m1', 'message', 180, 92, 240, 42, MSG('1.1 2.9 5.2 6.8 9.1')),
    C('c2', 'fit a line', 450, 53),
    N('f1', 'np.polyfit', 450, 92, 150, 110),
    N('r1', 'round', 630, 92, 100, 70, {'decimals': 2}),
    N('l1', 'list', 630, 180, 160, 42, LIST),
    C('c3', 'points near the squares', 180, 270),
    N('m2', 'message', 180, 305, 240, 42, MSG('0 1.1 3.9 9.2 15.8')),
    C('c4', 'fit a parabola', 450, 270),
    N('f2', 'np.polyfit 2', 450, 305, 150, 110),
    N('r2', 'round', 630, 305, 100, 70, {'decimals': 2}),
    N('l2', 'list', 630, 395, 190, 42, LIST),
],
[('mt', '', 'm1', ''), ('mt', '', 'm2', ''),
 ('m1', 'message out', 'f1', 'y'), ('f1', 'coefficients', 'r1', 'in'), ('r1', 'result', 'l1', ''),
 ('m2', 'message out', 'f2', 'y'), ('f2', 'coefficients', 'r2', 'in'), ('r2', 'result', 'l2', '')],
['np.polyval', 'np.linalg.lstsq', 'np.interp', 'np.gradient'], 520, 1000)


page('np.polyval', 'np.polyval - evaluate a polynomial curve',
"""
Works out a polynomial at each x you send: given coefficients 1 0 -1 0, the curve is
1 x x x + 0 x x - 1 x + 0, and np.polyval gives its value at each x.

Use it to draw the curve np.polyfit found, to extend a fitted trend into the future,
to subtract a trend from data, or simply as a shaping function: a control through a
cubic can rise, dip and rise again, and with the right coefficients it can bend any
way you like.

SYNTAX:
np.polyval <coefficients: floats ...>
Default: no coefficients (send them to the coefficients inlet).

EXAMPLE:
np.polyval 1 0 0

INPUTS and PARAMETERS:

x:
Where to evaluate: a number, list or array. Every input is evaluated and sent.

coefficients:
Highest power first, as np.polyfit gives them, replacing any given as arguments.
Held: send them first. A 2-D array (several curves from np.polyfit, one per column)
gives one value per curve at each x.

OUTPUTS:

values:
The curve's value at each x.

READING THE DEMO:
Top: a ramp from -1 to 1 through x x x - x. The curve starts at 0, rises, dips below
zero through the middle and comes back to 0 at 1.

Bottom: 2 1 is the straight line 2 x + 1: at 0 1 2 3 it gives 1 3 5 7. A metro
re-sends the x values five times a second.

RELATED:
np.polyfit finds the coefficients from points. np.interp follows a curve drawn
through points. ^ raises to a single power.
""",
[
    C('c1', 'a ramp, -1 to 1', 28, 53),
    N('sg', 'signal 2.0 saw', 28, 92, 129, 78, SIG('saw', 2.0, 1.0, True)),
    C('c2', 'x x x - x', 200, 53),
    N('pv', 'np.polyval 1 0 -1 0', 200, 92, 200, 90),
    C('c3', 'up, down through the middle, up', 450, 53),
    N('pl', 'plot', 450, 92, 208, 178, PLOT(-0.5, 0.5, 240)),
    C('c4', 'five times a second', 28, 320),
    N('mt', 'metro 200', 28, 355, 129, 70, METRO),
    C('c5', 'x', 180, 320),
    N('m1', 'message', 180, 355, 140, 42, MSG('0 1 2 3')),
    C('c6', '2 x + 1', 360, 320),
    N('p2', 'np.polyval 2 1', 360, 355, 180, 90),
    N('l1', 'list', 580, 355, 180, 42, LIST),
],
[('sg', '', 'pv', 'x'), ('pv', 'values', 'pl', 'y'),
 ('mt', '', 'm1', ''), ('m1', 'message out', 'p2', 'x'), ('p2', 'values', 'l1', '')],
['np.polyfit', 'np.interp', 'power', 'ranger'], 520, 860)


# ------------------------------------------------------------------- svd
page('np.linalg.svd', 'np.linalg.svd - break a matrix into rotations and stretches',
"""
The singular value decomposition: writes any matrix as three simpler ones, U, S and
Vh, so that the matrix equals U times diag(S) times Vh. Vh turns the input into a new
set of axes, S stretches along each of them, and U turns the result into place.

The singular values S, largest first, say how much the matrix acts along each of its
independent directions. A matrix with one large value and the rest tiny is nearly
one-dimensional - everything in it lies along one line - and keeping only the large
values gives the best possible simpler version of it. That is the idea behind
principal components, noise reduction, finding the best-fitting rotation between two
sets of points, and compressing a set of poses into a few basic movements.

SYNTAX:
np.linalg.svd

INPUTS and PARAMETERS:

matrix:
Any 2-D matrix (or a stack of them). Every input is decomposed and sent.

full matrices:
An option, under the node. Off (the default) gives the compact form, with no more
columns in U (rows in Vh) than there are singular values. On gives square U and Vh.

OUTPUTS:

U:
The output directions, one column per singular value.

singular values:
How strongly the matrix acts along each direction, largest first, never negative.

Vh:
The input directions, one row per singular value. Sent first, then the singular
values, then U.

READING THE DEMO:
The 3 x 2 matrix 1 2 / 3 4 / 5 6 has singular values about 9.53 and 0.51. The first is
nineteen times the second: its rows nearly lie on one line, and it is almost a
one-dimensional matrix. U and Vh show the directions; the round node only tidies the
values for reading.

RELATED:
np.linalg.eigh does the same for a symmetric matrix such as a covariance.
np.linalg.pinv and np.linalg.lstsq are built on the SVD. np.linalg.matrix_rank
counts the singular values that are not essentially zero. t.linalg.svd is the
PyTorch version.
""",
metro_row() + [
    C('c1', 'six numbers ...', 180, 53),
    N('m1', 'message', 180, 92, 160, 42, MSG('1 2 3 4 5 6')),
    C('c2', '... as 3 rows of 2', 370, 53),
    N('rs', 'np.reshape 3 2', 370, 92, 150, 90),
    C('c3', 'decompose', 650, 53),
    N('sv', 'np.linalg.svd', 650, 92, 150, 110),
    C('c4', 'U', 28, 260),
    N('hu', 'heat_map', 28, 295, 168, 170, HM(150, 120, 1, -1.0, 1.0, '%.2f', 'red-blue')),
    C('c5', 'singular values', 240, 260),
    N('rd', 'round', 240, 295, 100, 70, {'decimals': 2}),
    N('ls', 'list', 240, 385, 140, 42, LIST),
    C('c6', 'Vh', 420, 260),
    N('hv', 'heat_map', 420, 295, 168, 130, HM(150, 80, 1, -1.0, 1.0, '%.2f', 'red-blue')),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'sv', 'matrix'),
 ('sv', 'U', 'hu', 'y'), ('sv', 'singular values', 'rd', 'in'), ('rd', 'result', 'ls', ''),
 ('sv', 'Vh', 'hv', 'y')],
['np.linalg.eigh', 'np.linalg.pinv', 'np.linalg.matrix_rank', 't.linalg.svd'], 540, 1050)


# ------------------------------------------------------------------ eigh
page('np.linalg.eigh', 'np.linalg.eigh - the principal axes of a symmetric matrix',
"""
Finds the eigenvalues and eigenvectors of a symmetric matrix: the directions it only
stretches, without turning, and how much it stretches along each.

Its most common use is principal component analysis. Feed it the covariance matrix
of some data (np.cov) and the eigenvectors are the data's principal axes - the
direction it varies most along, then the next most at right angles to that, and so
on - and the eigenvalues say how much variance lies along each. The first axis of a
cloud of hand positions is the main direction of the gesture; the first few axes
of many joints are the main ways the body moves together.

SYNTAX:
np.linalg.eigh

INPUTS and PARAMETERS:

matrix:
A square, symmetric matrix (equal to its own transpose), such as a covariance or
correlation matrix. Only its lower triangle is read, so a matrix that is not
symmetric gives the answer for a different matrix. Every input is analysed and sent.

order:
An option, under the node: largest first (the default here, the principal axis
first) or smallest first (the order NumPy itself gives).

OUTPUTS:

eigenvalues:
How much the matrix stretches along each eigenvector.

eigenvectors:
One eigenvector per COLUMN, in the same order as the eigenvalues, each of length 1.
Sent first. The sign of each is arbitrary: (0.71, 0.71) and (-0.71, -0.71) are the
same axis.

READING THE DEMO:
Six points lying close to the diagonal line y = x. Their covariance, analysed: the
first eigenvalue (6.95) is hundreds of times the second (0.01), and the first
eigenvector, the first column, is about (0.71, 0.71) - the diagonal, the direction
the points run along. The second axis, at right angles, holds almost nothing: the
points barely stray from the line. The round node only tidies the numbers.

RELATED:
np.cov makes the covariance matrix to feed it. np.linalg.svd does the same job for
any matrix, symmetric or not. t.linalg.eig is the PyTorch version;
t.linalg.pca_low_rank does a whole PCA in one step.
""",
metro_row() + [
    C('c1', 'six points along y = x: x y x y ...', 180, 53),
    N('m1', 'message', 180, 92, 330, 42, MSG('0 0.1 1 0.9 2 2.2 3 2.8 4 4.1 5 5.0')),
    N('rs', 'np.reshape 6 2', 540, 92, 150, 90),
    C('c2', 'how x and y vary', 28, 240),
    N('cv', 'np.cov', 28, 275, 150, 70),
    C('c3', 'its principal axes', 220, 240),
    N('eg', 'np.linalg.eigh', 220, 275, 150, 90),
    C('c4', 'eigenvalues', 420, 240),
    N('rd', 'round', 420, 275, 100, 70, {'decimals': 2}),
    N('lv', 'list', 420, 365, 140, 42, LIST),
    C('c5', 'eigenvectors: columns', 600, 240),
    N('hv', 'heat_map', 600, 275, 168, 130, HM(150, 80, 1, -1.0, 1.0, '%.2f', 'red-blue')),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'cv', 'in'),
 ('cv', 'covariance', 'eg', 'matrix'), ('eg', 'eigenvalues', 'rd', 'in'), ('rd', 'result', 'lv', ''),
 ('eg', 'eigenvectors', 'hv', 'y')],
['np.cov', 'np.linalg.svd', 't.linalg.eig', 't.linalg.pca_low_rank'], 520, 1050)


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
