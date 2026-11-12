"""Seed the help pages for the second batch of scipy nodes (2026-10-24): one page per node.

    python3 dpg_system/help/tools/seed_scipy2_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

scipy.resample_poly, scipy.spectrogram, scipy.welch, scipy.label,
scipy.gaussian_filter, scipy.median_filter.

The spectrum pages build their test signals in subpatchers (make_chirp,
make_two_tones) so the page itself shows the node. Like the other seeders it is not
a make_* script, so rebuild.py never runs it, and it will not write over an
existing page without --force. Treat the pages as hand-made once written.
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


METRO_1S = {'on': True, 'period': 1000.0, 'units': 'milliseconds'}
METRO_2S = {'on': True, 'period': 2000.0, 'units': 'milliseconds'}


def LINE(count, miny, maxy, width=360, style='line'):
    return dict(BARS(count, maxy, width, 130, style), **{'min y': miny})


# ------------------------------------------------------------ resample_poly
page('scipy.resample_poly', 'scipy.resample_poly - change the sample rate of a recording',
"""
Converts a recording from one sample rate to another - 60 frames a second to 100, 120
to 30 - so that it describes the same movement over the same time with a different
number of samples.

It does this properly: it filters out anything too fast for the new rate before
thinning a recording out (so nothing turns into false slow wobbles), and draws a smooth
curve between samples when adding more. Taking every other sample, or np.interp's
straight lines, can do neither. Use it to bring recordings from different systems to a
common rate, to match motion to an audio or video frame rate, or to thin a long take
before analysing it.

SYNTAX:
scipy.resample_poly <from rate> <to rate>
Defaults: 60 to 100.

EXAMPLE:
scipy.resample_poly 120 60

INPUTS and PARAMETERS:

in:
The recording (a list or array). Every input is resampled and sent.

from rate (float box, also an inlet):
The recording's sample rate.

to rate (float box, also an inlet):
The rate wanted. Only the ratio of the two matters: 60 to 100 is the same as 3 to 5.

axis (int box, also an inlet):
Which way time runs: -1 along each row, 0 down each column - for a buffer of frames x
joints, use 0.

OUTPUTS:

resampled:
The recording at the new rate: n samples become n x to / from.

THE ENDS:
The first and last few samples are slightly less exact, as with any filter.

READING THE DEMO:
One cycle of a wave in 12 samples (left), resampled from 60 to 100 a second: the same
cycle in 20 samples (right), the curve unchanged. A new copy is sent every second.

RELATED:
np.interp resamples with straight lines; scipy.cubic_spline with a smooth curve but no
filtering. subsample thins a live stream.
""",
[
    C('c1', 'every second', 28, 53),
    N('mt', 'metro 1000', 28, 92, 129, 70, METRO_1S),
    C('c2', '12 samples', 180, 53),
    N('ar', 'np.arange 12', 180, 92, 150, 110),
    C('c3', 'one cycle', 360, 53),
    N('ml', '* 30', 360, 92, 129, 70),
    N('sn', 'sin', 360, 180, 129, 70),
    C('c4', 'from 60 to 100 a second', 540, 53),
    N('rs', 'scipy.resample_poly', 540, 92, 190, 110),
    C('c5', 'in: 12 samples', 28, 290),
    N('p1', 'plot', 28, 325, 368, 180, LINE(12, -1.2, 1.2, 360, 'stem')),
    C('c6', 'out: the same cycle in 20', 440, 290),
    N('p2', 'plot', 440, 325, 368, 180, LINE(20, -1.2, 1.2, 360, 'stem')),
],
[('mt', '', 'ar', ''), ('ar', 'arange out', 'ml', 'in'), ('ml', 'result', 'sn', 'in'),
 ('sn', 'out', 'p1', 'y'), ('sn', 'out', 'rs', 'in'), ('rs', 'resampled', 'p2', 'y')],
['np.interp', 'scipy.cubic_spline', 'subsample', 'scipy.filtfilt'], 560, 1000)


# ------------------------------------------------------------- spectrogram
# Inside make_chirp: 10 seconds at 60 a second of a wave whose speed rises steadily
# from 0 to 20 cycles a second (its angle grows with the square of the time).
CHIRP_SUB = {
    'name': 'make_chirp',
    'host': 'ch',
    'demo': [
        S('mt', 'metro 2000', 40, 40, 129, 70, METRO_2S),
        S('ct', 'comment every two seconds', 190, 55, 220, 30),
        S('ar', 'np.arange 600', 40, 130, 150, 110),
        S('ca', 'comment 600 samples: 10 seconds at 60 a second', 210, 150, 380, 30),
        S('sq', '^ 2', 40, 270, 129, 70),
        S('m1', '* 0.1', 40, 360, 129, 70),
        S('s1', 'sin', 40, 450, 129, 70),
        S('c1', 'comment the angle grows with the square of the time: ever faster', 190, 380, 520, 30),
        S('out', 'out chirp', 40, 550, 150, 80),
    ],
    'links': [('mt', '', 'ar', ''), ('ar', 'arange out', 'sq', 'in'), ('sq', 'result', 'm1', 'in'),
              ('m1', 'result', 's1', 'in'), ('s1', 'out', 'out', 'chirp')],
}

page('scipy.spectrogram', 'scipy.spectrogram - which frequencies, at which moments',
"""
Cuts a signal into short overlapping windows and finds the frequencies in each: a
picture with frequency one way and time the other, bright where there is energy. It
shows how the content of a signal changes - a rhythm speeding up, a tremor that comes
and goes, a vibration that starts at one moment and stops at another.

A single spectrum (np.fft.rfft, scipy.welch) mixes the whole signal together; the
spectrogram keeps the order of events.

SYNTAX:
scipy.spectrogram <window: int> <sample rate>
Defaults: window 64, sample rate 60.

EXAMPLE:
scipy.spectrogram 128 100

INPUTS and PARAMETERS:

in:
The signal: a list or 1-D array. Every input is analysed and sent.

sample rate (float box, also an inlet):
Samples per second, so that frequencies come out in cycles per second and times in
seconds.

window (int box, also an inlet):
How many samples in each window. A long window tells close frequencies apart but blurs
when things happen; a short one is sharp in time but coarse in frequency. The
frequencies are spaced sample rate / window apart.

overlap (float box, also an inlet):
How much each window overlaps the last, as a fraction: 0.5 (the default) starts each
halfway into the previous one.

decibels:
An option, under the node. On (the default here), the power is given in decibels (10
log10 of the power), which shows faint and strong content together; off, as plain
power.

OUTPUTS:

power:
One row per frequency, one column per window: the picture.

frequencies:
The frequency of each row, from 0 up to half the sample rate.

times:
The time at the centre of each window. Sent first, then the frequencies, then the
power.

READING THE DEMO:
The make_chirp subpatcher makes 10 seconds of a wave that gets steadily faster, from
0 to about 20 cycles a second. In the spectrogram time runs left to right and frequency
top to bottom (0 at the top): a bright line running down from the top left, from about 1
cycle a second in the first window to about 18 in the last.

RELATED:
scipy.welch gives one steady spectrum for the whole signal. np.fft.rfft gives the
spectrum of one window. np.sliding_window_view and np.fft.rfft can build the same
picture step by step. spectrum analyses a live stream.
""",
[
    C('c1', 'a wave getting faster', 28, 53),
    N('ch', 'p make_chirp', 28, 92, 180, 70),
    C('c2', 'windows of 64 at 60 a second', 260, 53),
    N('sp', 'scipy.spectrogram', 260, 92, 200, 150),
    C('c3', 'frequency down, time across', 500, 53),
    N('hm', 'heat_map', 500, 92, 248, 290, HM(230, 250, 1, -60.0, 0.0, '', 'plasma')),
],
[('ch', 'chirp', 'sp', 'in'), ('sp', 'power', 'hm', 'y')],
['scipy.welch', 'np.fft.rfft', 'np.sliding_window_view', 'spectrum'], 440, 1150, subpatch=CHIRP_SUB)


# ------------------------------------------------------------------- welch
# Inside make_two_tones: 10 seconds at 60 a second of 3 cycles a second plus half as
# much of 8 cycles a second.
TONES_SUB = {
    'name': 'make_two_tones',
    'host': 'tt',
    'demo': [
        S('mt', 'metro 2000', 40, 40, 129, 70, METRO_2S),
        S('ct', 'comment every two seconds', 190, 55, 220, 30),
        S('ar', 'np.arange 600', 40, 130, 150, 110),
        S('ca', 'comment 600 samples: 10 seconds at 60 a second', 210, 150, 380, 30),
        S('m1', '* 18', 40, 270, 129, 70),
        S('s1', 'sin', 40, 360, 129, 70),
        S('c1', 'comment 3 cycles a second', 40, 440, 220, 30),
        S('m2', '* 48', 330, 270, 129, 70),
        S('s2', 'sin', 330, 360, 129, 70),
        S('m3', '* 0.5', 330, 450, 129, 70),
        S('c2', 'comment 8 cycles a second, half as tall', 330, 530, 320, 30),
        S('ad', '+', 40, 580, 129, 70),
        S('out', 'out tones', 40, 680, 150, 80),
    ],
    # the ripple branch is linked first, so it reaches the + operand before the slow
    # wave arrives to trigger the sum (an outlet sends in the order its links were made)
    'links': [('mt', '', 'ar', ''), ('ar', 'arange out', 'm2', 'in'), ('ar', 'arange out', 'm1', 'in'),
              ('m1', 'result', 's1', 'in'), ('m2', 'result', 's2', 'in'), ('s2', 'out', 'm3', 'in'),
              ('m3', 'result', 'ad', 'operand', 0, 1), ('s1', 'out', 'ad', 'in'),
              ('ad', 'result', 'out', 'tones')],
}

page('scipy.welch', 'scipy.welch - a steady spectrum of a whole signal',
"""
Gives the spectrum of a whole signal - how much power there is at each frequency - by
cutting it into overlapping windows, taking the spectrum of each and averaging them.
Welch's method.

A single spectrum of a long, noisy recording is ragged: every bin jumps about.
Averaging many windows gives a steady, readable answer, at the price of coarser
frequency steps. Use it to find the main rhythms in a take - a gait's step rate, a
breathing rate, a tremor's frequency - or to compare how much fast and slow movement
two performances have.

SYNTAX:
scipy.welch <window: int> <sample rate>
Defaults: window 64, sample rate 60.

EXAMPLE:
scipy.welch 256 100

INPUTS and PARAMETERS:

in:
The signal: a list or 1-D array. Every input is analysed and sent.

sample rate (float box, also an inlet):
Samples per second, so the frequencies come out in cycles per second.

window (int box, also an inlet):
How many samples in each window. Longer gives finer frequency steps (sample rate /
window apart) but fewer windows to average.

overlap (float box, also an inlet):
How much each window overlaps the last, as a fraction (0.5 by default).

decibels:
An option, under the node. Off (the default here), plain power; on, in decibels.

OUTPUTS:

power:
The power at each frequency.

frequencies:
The frequency of each value, from 0 up to half the sample rate. Sent first.

READING THE DEMO:
The make_two_tones subpatcher makes 10 seconds of a wave at 3 cycles a second plus
one half as tall at 8. Welch's spectrum, in windows of 128 (bins 0.47 cycles a second
apart), has two peaks: near 3, and near 8 - a quarter as tall, since power goes with
the square of the size.

RELATED:
scipy.spectrogram keeps the windows apart, to show change over time. np.fft.rfft gives
the spectrum of a single window. spectrum analyses a live stream.
""",
[
    C('c1', 'two rhythms, mixed', 28, 53),
    N('tt', 'p make_two_tones', 28, 92, 200, 70),
    C('c2', 'windows of 128 at 60 a second', 280, 53),
    N('we', 'scipy.welch 128', 280, 92, 200, 150),
    C('c3', 'peaks near 3 and 8', 28, 290),
    N('pl', 'plot', 28, 325, 528, 230, BARS(65, 0.7, 520, 180)),
],
[('tt', 'tones', 'we', 'in'), ('we', 'power', 'pl', 'y')],
['scipy.spectrogram', 'np.fft.rfft', 'spectrum', 'scipy.find_peaks'], 610, 1100, subpatch=TONES_SUB)


# ------------------------------------------------------------------- label
page('scipy.label', 'scipy.label - find the separate blobs in a grid',
"""
Finds the separate regions - blobs - in a grid: groups of neighbouring cells above a
threshold, each group cut off from the others. It numbers them 1, 2, 3 ..., and gives
how many there are, the size of each and the centre of each.

The centres make it a simple tracker: the blobs in a depth or occupancy grid are the
people in a room; in a thresholded camera image, the bright objects; in a heat map of
activity, the hot spots. The sizes let you ignore specks.

SYNTAX:
scipy.label <threshold: float>
Default: threshold 0.5.

INPUTS and PARAMETERS:

in:
The grid (usually 2-D; any number of axes works). Every input is searched and sent.
A true/false mask is used as it is; numbers count where they are above the threshold.

threshold (float box, also an inlet):
The level a cell must be above to count as part of a blob.

diagonals connect:
An option, under the node. Off (the default), cells join a blob only side by side; on,
corner to corner too - two blobs touching only at a corner become one.

OUTPUTS:

labels:
The grid with each cell numbered by its blob: 0 for background, 1 to count for the
blobs.

count:
How many blobs.

sizes:
How many cells in each blob, blob 1 first.

centres:
The centre of each blob, one row each: row then column, in cells. Sent first, then
sizes, count and labels.

READING THE DEMO:
Random values smoothed with scipy.gaussian_filter make soft hills (left). Cut at 0.55,
the hilltops become separate blobs, each a colour (right); the count and the centres
change with each new picture, every two seconds.

RELATED:
scipy.gaussian_filter smooths a grid first, so its blobs are shapes and not specks.
np.histogram2d makes an occupancy grid from points. np.argwhere lists every cell above
a threshold.
""",
[
    C('c1', 'every two seconds', 28, 53),
    N('mt', 'metro 2000', 28, 92, 129, 70, METRO_2S),
    C('c2', '24 x 24 random values', 180, 53),
    N('rn', 'np.rand 24 24', 180, 92, 150, 130),
    C('c3', 'smoothed into hills', 360, 53),
    N('gf', 'scipy.gaussian_filter 2', 360, 92, 210, 70),
    C('c4', 'the hilltops above 0.55', 600, 53),
    N('lb', 'scipy.label 0.55', 600, 92, 190, 150),
    C('c5', 'the hills', 28, 290),
    N('h1', 'heat_map', 28, 325, 228, 250, HM(210, 210, 1, 0.35, 0.65, '', 'viridis')),
    C('c6', 'the blobs, numbered', 290, 290),
    N('h2', 'heat_map', 290, 325, 228, 250, HM(210, 210, 1, 0.0, 6.0, '', 'paired')),
    C('c7', 'how many', 560, 290),
    N('lc', 'list', 560, 325, 80, 42, LIST),
    C('c8', 'their centres: row, column', 28, 610),
    N('rz', 'round', 28, 645, 100, 70, {'decimals': 1}),
    N('lz', 'list', 150, 645, 640, 42, LIST),
],
[('mt', '', 'rn', ''), ('rn', 'random array', 'gf', 'in'), ('gf', 'filtered', 'h1', 'y'),
 ('gf', 'filtered', 'lb', 'in'), ('lb', 'labels', 'h2', 'y'), ('lb', 'count', 'lc', ''),
 ('lb', 'centres', 'rz', 'in'), ('rz', 'result', 'lz', '')],
['scipy.gaussian_filter', 'np.histogram2d', 'np.argwhere', 'scipy.kdtree'], 750, 1000)


# --------------------------------------------------------- gaussian_filter
page('scipy.gaussian_filter', 'scipy.gaussian_filter - a smooth blur',
"""
Blurs an array: replaces every value with a weighted average of its neighbourhood,
the weights falling off in a bell curve (a Gaussian) with distance. It works on lists,
images, grids and volumes alike, along every axis at once.

Use it to soften an occupancy or heat map so its shapes stand out from the specks, to
smooth a recorded curve without lag, to spread single points into soft glows, or to
blur an image. sigma sets how far the blur reaches.

SYNTAX:
scipy.gaussian_filter <sigma ...>
Default: sigma 1.

EXAMPLES:
scipy.gaussian_filter 3
scipy.gaussian_filter 0 2

INPUTS and PARAMETERS:

in:
The array (a list, image or grid). Every input is blurred and sent.

sigma (text box, also an inlet):
How far the blur reaches, in samples or cells: the width of the bell curve. One number
for every axis, or one per axis - 0 leaves that axis alone, so 0 2 blurs a buffer of
frames x joints along the joints only, and 2 0 along time only.

edges:
An option, under the node: what lies beyond the edges. reflect (the default) mirrors
the array; nearest repeats the edge values; wrap continues from the other side;
constant uses 0.

OUTPUTS:

filtered:
The blurred array, the same shape.

READING THE DEMO:
Random values (left), blurred with sigma 2 (right): the speckle merges into soft
clouds. Blurring also averages away contrast, so the right-hand picture is shown over
a narrower range. A new picture comes every two seconds.

RELATED:
scipy.median_filter removes spikes while keeping edges sharp. np.convolve smooths with
weights you choose. k.gaussian_blur and tv.gaussian_blur blur images as tensors.
scipy.filtfilt smooths a recording by frequency.
""",
[
    C('c1', 'every two seconds', 28, 53),
    N('mt', 'metro 2000', 28, 92, 129, 70, METRO_2S),
    C('c2', '32 x 32 random values', 180, 53),
    N('rn', 'np.rand 32 32', 180, 92, 150, 130),
    C('c3', 'blur, sigma 2', 380, 53),
    N('gf', 'scipy.gaussian_filter 2', 380, 92, 210, 70),
    C('c4', 'in', 28, 290),
    N('h1', 'heat_map', 28, 325, 228, 250, HM(210, 210, 1, 0.0, 1.0, '', 'viridis')),
    C('c5', 'out: soft clouds', 290, 290),
    N('h2', 'heat_map', 290, 325, 228, 250, HM(210, 210, 1, 0.38, 0.62, '', 'viridis')),
],
[('mt', '', 'rn', ''), ('rn', 'random array', 'gf', 'in'), ('rn', 'random array', 'h1', 'y'),
 ('gf', 'filtered', 'h2', 'y')],
['scipy.median_filter', 'np.convolve', 'k.gaussian_blur', 'scipy.label'], 640, 1000)


# ----------------------------------------------------------- median_filter
page('scipy.median_filter', 'scipy.median_filter - remove spikes, keep edges',
"""
Replaces every value with the median of its neighbourhood: the middle value when the
neighbours are put in order. A single wild value - a glitch, a dropped frame, a
sensor's spike - is never in the middle, so it simply disappears, while the values
around it are left almost untouched.

That is what makes it the standard despiking filter. An averaging filter (scipy.
gaussian_filter, np.convolve) smears a spike into its neighbours and rounds off every
sharp step; the median removes the spike entirely and keeps a step a step.

SYNTAX:
scipy.median_filter <size ...>
Default: size 3.

EXAMPLES:
scipy.median_filter 5
scipy.median_filter 5 1

INPUTS and PARAMETERS:

in:
The array (a list, image or grid). Every input is filtered and sent.

size (text box, also an inlet):
How many values in each neighbourhood, centred on each value: 3 looks one each side.
A spike up to (size - 1) / 2 samples long is removed. One number for every axis, or one
per axis - 1 leaves that axis alone, so 5 1 despikes a buffer of frames x joints along
time only.

edges:
An option, under the node: what lies beyond the edges, as for scipy.gaussian_filter.

OUTPUTS:

filtered:
The despiked array, the same shape.

READING THE DEMO:
A wave with two glitches in it (left): 9 where it should be about 1, and -8 where it
should be about -0.5. With size 3 both are gone and the rest of the wave is as it was
(right). A metro re-sends the list twice a second.

RELATED:
scipy.gaussian_filter blurs instead. np.nanmedian and np.median take the median of a
whole array. np.isnan and np.nan_to_num deal with gaps rather than spikes.
""",
[
    C('c1', 'twice a second', 28, 53),
    N('mt', 'metro 500', 28, 92, 129, 70, {'on': True, 'period': 500.0, 'units': 'milliseconds'}),
    C('c2', 'a wave with two glitches', 180, 53),
    N('m1', 'message', 180, 92, 560, 42, MSG('0 0.3 0.6 0.8 1 9 0.9 0.7 0.4 0.1 -0.2 -8 -0.6 -0.8 -0.9 -1')),
    C('c3', 'neighbourhoods of 3', 180, 170),
    N('mf', 'scipy.median_filter', 180, 205, 190, 110),
    C('c4', 'in: two spikes', 28, 360),
    N('p1', 'plot', 28, 395, 388, 180, LINE(16, -9.0, 9.0, 380)),
    C('c5', 'out: spikes gone, the wave kept', 440, 360),
    N('p2', 'plot', 440, 395, 388, 180, LINE(16, -9.0, 9.0, 380)),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'mf', 'in'), ('m1', 'message out', 'p1', 'y'),
 ('mf', 'filtered', 'p2', 'y')],
['scipy.gaussian_filter', 'np.median', 'np.nanmedian', 'scipy.filtfilt'], 630, 1000)


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
