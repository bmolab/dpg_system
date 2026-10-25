"""Seed the help pages for the scipy nodes (2026-10-24): one page per node.

    python3 dpg_system/help/tools/seed_scipy_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

scipy.find_peaks, scipy.filtfilt, scipy.kdtree, scipy.cubic_spline.

The peak and filter pages share a make_wobble subpatcher (a slow wave with a faster
ripple on it), so the page itself shows the node. Like the numpy seeders it is not a
make_* script, so rebuild.py never runs it, and it will not write over an existing
page without --force. Treat the pages as hand-made once written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from help_common import SIG, PLOT
from seed_numpy_more_help import LIST, MSG, BARS, C, N, see_also


PAGES = {}


def page(stem, title, body, demo, links, also, also_y, text_height=820, subpatch=None):
    PAGES[stem] = dict(title=title, body=body.strip('\n'), demo=demo + see_also(also_y, also),
                       links=links, text_height=text_height, subpatch=subpatch)


def S(key, init, x, y, w=150, h=70, props=None):
    """A subpatcher node (build_help gives them their own defaults)."""
    d = {'key': key, 'init': init, 'pos': (x, y), 'w': w, 'h': h}
    if props:
        d['props'] = props
    return d


# Inside make_wobble: twice a second, 120 samples (2 seconds at 60 a second) of a slow
# wave, 3 cycles, with a ripple 5 times as fast and 0.3 as tall added to it.
WOBBLE_SUB = {
    'name': 'make_wobble',
    'host': 'wb',
    'demo': [
        S('mt', 'metro 500', 40, 40, 129, 70, {'on': True, 'period': 500.0, 'units': 'milliseconds'}),
        S('ct', 'comment twice a second', 190, 55, 200, 30),
        S('ar', 'np.arange 120', 40, 130, 150, 110),
        S('ca', 'comment 120 samples: 2 seconds at 60 a second', 210, 150, 380, 30),
        S('m1', '* 9', 40, 270, 129, 70),
        S('s1', 'sin', 40, 360, 129, 70),
        S('c1', 'comment a slow wave: 3 cycles', 40, 440, 260, 30),
        S('m2', '* 45', 330, 270, 129, 70),
        S('s2', 'sin', 330, 360, 129, 70),
        S('m3', '* 0.3', 330, 450, 129, 70),
        S('c2', 'comment a ripple: 15 cycles, 0.3 tall', 330, 530, 300, 30),
        S('ad', '+', 40, 580, 129, 70),
        S('c3', 'comment added together', 190, 595, 200, 30),
        S('out', 'out wobble', 40, 680, 150, 80),
    ],
    # the ripple branch is linked first, so it reaches the + operand before the slow
    # wave arrives to trigger the sum (an outlet sends in the order its links were made)
    'links': [('mt', '', 'ar', ''), ('ar', 'arange out', 'm2', 'in'), ('ar', 'arange out', 'm1', 'in'),
              ('m1', 'result', 's1', 'in'), ('m2', 'result', 's2', 'in'), ('s2', 'out', 'm3', 'in'),
              ('m3', 'result', 'ad', 'operand', 0, 1), ('s1', 'out', 'ad', 'in'),
              ('ad', 'result', 'out', 'wobble')],
}

WOBBLE_NOTE = """The wobble comes from the make_wobble subpatcher (double-click it to look inside):
120 samples - two seconds at 60 a second - of a slow wave going round 3 times, with a
ripple five times as fast and a third as tall added to it."""


# ------------------------------------------------------------- find_peaks
page('scipy.find_peaks', 'scipy.find_peaks - find the peaks of a curve',
"""
Finds the peaks of a curve - the points higher than their neighbours - and gives their
positions, their heights and how many there are. With valleys chosen it finds the
dips instead.

Every little wiggle is a peak, so the useful work is in choosing which peaks count.
Prominence is the best tool: how far a peak stands above the surrounding ground - a
small ripple on a big wave has little prominence however high it is. Spacing keeps
only the highest of peaks closer together than a minimum; a minimum height ignores
everything below it.

Use it on a recorded curve to find footfalls in a vertical position, beats in an
envelope, the accents of a gesture in a speed, the turning points of a joint angle,
the onsets of a movement.

SYNTAX:
scipy.find_peaks <prominence: float> <spacing: int>
Defaults: prominence 0 (off), spacing 1.

EXAMPLE:
scipy.find_peaks 0.5 10

INPUTS and PARAMETERS:

in:
The curve: a list or 1-D array. Every input is searched and sent. A nan is treated as
a deep gap, so it does not hide the peaks beside it.

prominence (float box, also an inlet):
How far a peak must stand above the ground around it. 0 counts every peak.

spacing (int box, also an inlet):
The smallest distance between peaks, in samples. Where two are closer, only the higher
is kept.

min height (float box, also an inlet):
Ignore peaks below this - for valleys, dips above it. Used only when use min height is
on.

use min height:
An option, under the node: turns the minimum height on.

find:
An option, under the node: peaks (the default) or valleys.

OUTPUTS:

indices:
The position of each peak, in samples from the start.

heights:
The value of the curve at each peak.

count:
How many peaks. Sent first, then the heights, then the indices.

READING THE DEMO:
""" + WOBBLE_NOTE + """

With prominence 0, every crest of the ripple counts: 15 peaks. With prominence 1, only
the three crests of the slow wave stand out far enough: at samples 10, 50 and 90, each
1.3 high (the wave's 1 plus the ripple's 0.3 riding on top).

RELATED:
scipy.filtfilt smooths a curve first, so that its peaks are the shape's and not the
noise's. np.argmax finds the single highest point. np.diff and np.sign show where a
curve turns.
""",
[
    C('c1', 'a slow wave with a ripple on it', 28, 53),
    N('wb', 'p make_wobble', 28, 92, 180, 70),
    C('c2', 'the curve', 28, 200),
    N('pl', 'plot', 28, 235, 368, 180, BARS(120, 1.5, 360, 130, 'line') | {'min y': -1.5}),
    C('c3', 'every crest', 440, 53),
    N('f1', 'scipy.find_peaks', 440, 92, 190, 150),
    C('c4', 'how many', 660, 53),
    N('n1', 'list', 660, 92, 80, 42, LIST),
    C('c5', 'prominence 1: the big ones', 440, 290),
    N('f2', 'scipy.find_peaks 1', 440, 325, 190, 150),
    C('c6', 'where', 660, 290),
    N('i2', 'list', 660, 325, 160, 42, LIST),
    C('c7', 'how high', 660, 400),
    N('h2', 'list', 660, 435, 160, 42, LIST),
],
[('wb', 'wobble', 'pl', 'y'), ('wb', 'wobble', 'f1', 'in'), ('wb', 'wobble', 'f2', 'in'),
 ('f1', 'count', 'n1', ''), ('f2', 'indices', 'i2', ''), ('f2', 'heights', 'h2', '')],
['scipy.filtfilt', 'np.argmax', 'np.top_k', 'np.diff'], 560, 1250, subpatch=WOBBLE_SUB)


# --------------------------------------------------------------- filtfilt
page('scipy.filtfilt', 'scipy.filtfilt - filter a recording without any delay',
"""
Filters a whole recording with a Butterworth filter - low-pass, high-pass, band-pass
or band-stop - running it once forwards and once backwards, so that the delay of the
forward pass is cancelled exactly. Every peak, step and turn stays at the moment it
happened.

A filter that runs as data arrives (filter, band_pass, one_euro_filter) can only look
at the past, so it always lags: the smoother it is, the later. When the whole recording
is in hand - a take, a buffer of the last few seconds - filtfilt can look both ways, and
the lag disappears. That matters whenever timing matters: lining movement up with sound
or with another performer, measuring when something happened, finding peaks.

low-pass keeps what is slower than the cutoff: smoothing, removing jitter. high-pass
keeps what is faster: removing drift and slow sway. band-pass keeps a range; band-stop
removes one - a mains hum, a known vibration.

SYNTAX:
scipy.filtfilt <kind> <cutoff> <cutoff 2>
Defaults: lowpass, cutoff 5, sample rate 60.

EXAMPLES:
scipy.filtfilt lowpass 6
scipy.filtfilt bandpass 1 3

INPUTS and PARAMETERS:

in:
The recording (a list or array). Every input is filtered and sent.

cutoff (float box, also an inlet):
Where the filter starts to act, in cycles per second (Hz). It must be below half the
sample rate.

cutoff 2 (float box, also an inlet):
The other edge, for band-pass and band-stop.

sample rate (float box, also an inlet):
Samples per second of the recording: 60 for 60 frames a second.

order (int box, also an inlet):
How sharply the filter cuts, 1 to 10. Higher is sharper; 4 (the default) suits most
uses. Running twice doubles the effect.

axis (int box, also an inlet):
Which way time runs: -1 along each row, 0 down each column - for a buffer of frames x
joints, use 0 to filter each joint.

kind:
An option, under the node: lowpass, highpass, bandpass or bandstop.

OUTPUTS:

filtered:
The filtered recording, the same shape.

THE ENDS:
Near the very beginning and end there is less to look at in one direction, so the
first and last few samples are the least reliable.

READING THE DEMO:
""" + WOBBLE_NOTE + """

At 60 samples a second the slow wave is 1.5 cycles a second and the ripple 7.5.
Low-passed at 4, the ripple is gone and the slow wave is left (right) - its crests
exactly where they were on the left, not a moment later.

RELATED:
filter, band_pass and one_euro_filter filter a live stream as it arrives.
np.convolve smooths with any weights you choose. scipy.find_peaks works better on a
filtered curve.
""",
[
    C('c1', 'a slow wave with a ripple on it', 28, 53),
    N('wb', 'p make_wobble', 28, 92, 180, 70),
    C('c2', 'low-pass at 4 cycles a second', 260, 53),
    N('ff', 'scipy.filtfilt lowpass 4', 260, 92, 230, 230),
    C('c3', 'in', 28, 360),
    N('p1', 'plot', 28, 395, 388, 180, BARS(120, 1.5, 380, 130, 'line') | {'min y': -1.5}),
    C('c4', 'out: the ripple gone, nothing late', 440, 360),
    N('p2', 'plot', 440, 395, 388, 180, BARS(120, 1.5, 380, 130, 'line') | {'min y': -1.5}),
],
[('wb', 'wobble', 'ff', 'in'), ('wb', 'wobble', 'p1', 'y'), ('ff', 'filtered', 'p2', 'y')],
['filter', 'band_pass', 'one_euro_filter', 'np.convolve', 'scipy.find_peaks'], 630, 1350,
    subpatch=WOBBLE_SUB)


# ----------------------------------------------------------------- kdtree
page('scipy.kdtree', 'scipy.kdtree - the nearest of a set of points',
"""
Holds a set of points and finds, for any point you ask about, the nearest of them -
or the nearest few - and how far away they are.

It sorts the points into a tree once, when they arrive, so that each question after
that is answered without measuring the distance to every point: thousands of points
cost little more than ten. Use it to find which performer, marker, zone centre or light
is closest to a hand; the nearest pose in a library of poses (each pose a point with
many coordinates); the nearest neighbours in a point cloud.

SYNTAX:
scipy.kdtree <k: int>
Default: k 1 (the single nearest).

EXAMPLE:
scipy.kdtree 3

INPUTS and PARAMETERS:

query:
The point to look near - a list of coordinates - or several, as rows. Every input is
looked up and sent - the only inlet that makes the node send.

points:
The set of points, as rows of coordinates (x y, or x y z, or as many as you like). Held;
the tree is rebuilt only when a new set arrives.

k (int box, also an inlet):
How many nearest points to give, nearest first.

max distance (float box, also an inlet):
Ignore points further away than this. 0 (the default) means no limit. Where nothing is
within reach, the index is -1 and the distance inf.

OUTPUTS:

indices:
Which of the points is nearest - its row number - or, for k above 1, the nearest k.

distances:
How far away each is. Sent first, then the indices.

READING THE DEMO:
Four points at the corners of a square, numbered 0 to 3: -1 -1, 1 -1, 1 1, -1 1. A
point circles round the middle (left), and the plot shows which corner it is nearest
to: 0, 1, 2, 3 in turn as it goes round, and the distance to it.

RELATED:
np.distance_from_target and proximity_trigger measure the distance to a few targets.
t.cdist gives every distance between two sets. hypot and np.linalg.norm measure one.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', 'four corners: x y, x y ...', 140, 53),
    N('mp', 'message', 140, 92, 220, 42, MSG('-1 -1 1 -1 1 1 -1 1')),
    N('rs', 'np.reshape 4 2', 390, 92, 150, 90),
    C('c3', 'a point going round', 28, 220),
    N('sx', 'signal 4.0 cos', 28, 255, 129, 78, SIG('cos', 4.0)),
    N('sy', 'signal 4.0 sin', 180, 255, 129, 78, SIG('sin', 4.0)),
    N('pk', 'pack 2', 28, 360, 130, 100),
    C('c4', 'which corner is nearest?', 390, 220),
    N('kd', 'scipy.kdtree', 390, 255, 180, 130),
    N('pi', 'plot', 600, 255, 208, 178, PLOT(-0.5, 3.5, 300, 'stair')),
    C('c5', 'how far, to 2 places', 390, 430),
    N('rd', 'round', 390, 465, 100, 70, {'decimals': 2}),
    N('ld', 'list', 510, 465, 100, 42, LIST),
],
[('lb', 'out', 'mp', ''), ('mp', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'kd', 'points'),
 ('sx', '', 'pk', 'in 1'), ('sy', '', 'pk', 'in 2'), ('pk', 'out', 'kd', 'query'),
 ('kd', 'indices', 'pi', 'y'), ('kd', 'distances', 'rd', 'in'), ('rd', 'result', 'ld', '')],
['np.distance_from_target', 'proximity_trigger', 't.cdist', 'hypot'], 580, 1000)


# ------------------------------------------------------------ cubic_spline
page('scipy.cubic_spline', 'scipy.cubic_spline - a smooth curve through points',
"""
Draws a smooth curve through a set of points and reads values off it. Where np.interp
joins the points with straight lines, meeting at corners, the spline bends smoothly
through every point, with no corners and no sudden changes of speed.

Use it for animation and camera paths through key positions, for resampling a
recorded movement without the kinks straight lines would put into its speed, or for a
control curve drawn with a few points that should feel continuous. A 2-D table of
values - one row per point, xyz in the columns - gives a smooth path in space.

SYNTAX:
scipy.cubic_spline <boundary>
Default: not-a-knot.

INPUTS and PARAMETERS:

x:
Where to read the curve: a number, list or array. Every input is read and sent - the
only inlet that makes the node send.

xp:
The positions of the points, held. Sorted for you if they are not in order. Without
xp the positions are 0, 1, 2 ... - the index of each value.

fp:
The values at those positions, held. A 2-D fp, one row per position, gives a curve
for each column.

boundary:
An option, under the node: how the curve behaves at its two ends. not-a-knot (the
default) lets the ends follow the nearest points naturally; natural lets the curve
straighten out at its ends; clamped makes it flat at both ends; periodic makes it a
loop that joins its end smoothly to its start (the last value is set equal to the
first), and repeats beyond the ends.

extrapolate:
An option, under the node. Off (the default), the curve is held flat beyond the first
and last points, as np.interp does. On, the curve carries on past them - which can
quickly run far off.

OUTPUTS:

values:
The curve's value at each x.

READING THE DEMO:
Five points: 0 at 0, 0.8 at 0.25, 0.3 at 0.5, 0.9 at 0.75 and 1 at 1. A ramp from 0 to 1
reads them two ways. np.interp (left) goes from point to point in straight lines, with
sharp corners at each; the spline (right) passes through the same five points as one
smooth swing.

RELATED:
np.interp joins the points with straight lines. np.polyfit fits one curve near the
points rather than through them. lerp blends between two values.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', 'positions', 140, 53),
    N('mx', 'message', 140, 92, 200, 42, MSG('0 0.25 0.5 0.75 1')),
    C('c3', 'values at them', 140, 150),
    N('mf', 'message', 140, 185, 200, 42, MSG('0 0.8 0.3 0.9 1')),
    C('c4', 'a ramp, 0 to 1', 380, 53),
    N('sg', 'signal 2.0 saw', 380, 92, 129, 78, SIG('saw', 2.0, 1.0, False)),
    C('c5', 'straight lines', 28, 280),
    N('ip', 'np.interp', 28, 315, 150, 110),
    N('p1', 'plot', 28, 450, 368, 180, PLOT(0.0, 1.2, 240) | {'width': 360}),
    C('c6', 'one smooth curve', 440, 280),
    N('cs', 'scipy.cubic_spline', 440, 315, 190, 110),
    N('p2', 'plot', 440, 450, 368, 180, PLOT(0.0, 1.2, 240) | {'width': 360}),
],
[('lb', 'out', 'mx', ''), ('lb', 'out', 'mf', ''),
 ('mx', 'message out', 'ip', 'xp'), ('mf', 'message out', 'ip', 'fp'),
 ('mx', 'message out', 'cs', 'xp'), ('mf', 'message out', 'cs', 'fp'),
 ('sg', '', 'ip', 'x'), ('sg', '', 'cs', 'x'), ('ip', 'values', 'p1', 'y'), ('cs', 'values', 'p2', 'y')],
['np.interp', 'np.polyfit', 'lerp', 'crossfade'], 690, 1150)


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
