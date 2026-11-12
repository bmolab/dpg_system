"""Seed the help pages for the third batch of scipy nodes (2026-10-24): one page per node.

    python3 dpg_system/help/tools/seed_scipy3_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

scipy.slerp, scipy.rotation_mean, scipy.align_vectors, scipy.rotation_apply,
scipy.circmean, scipy.circstd, scipy.hilbert, scipy.detrend, scipy.convex_hull,
scipy.kmeans, scipy.procrustes, scipy.curve_fit.

Quaternions are scalar first (w x y z), as in the quaternion nodes. The hilbert and
detrend demos build their signals in subpatchers (make_swell, make_drift); as in
make_wobble, the branch feeding the cold operand of the * or + is linked FIRST, so it
arrives before the hot branch triggers (an outlet sends in link order). Not a make_*
script: rebuild.py never runs it, and it will not write over an existing page
without --force. Treat the pages as hand-made once written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from help_common import SIG
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


def ROUND(key, x, y, places=3):
    return N(key, 'round', x, y, 100, 70, {'decimals': places})


MULTI = {'color': 'none', 'width': 380, 'height': 160, 'style': 'line',
         'update style': 'input is multi-channel sample', 'sample count': 240,
         'min x': 0.0, 'max x': 240.0, 'min y': -0.1, 'max y': 1.1}

ROT_NOTE = """QUATERNIONS:
Four numbers w x y z, scalar first - the same order as the quaternion nodes. A turn of
an angle about an axis is w = cos(angle / 2) and x y z = the axis times sin(angle / 2):
no turn is 1 0 0 0; a quarter turn about z is 0.7071 0 0 0.7071. q and -q are the same
turn."""


# ------------------------------------------------------------------- slerp
page('scipy.slerp', 'scipy.slerp - turn smoothly between orientations',
"""
Interpolates between orientations: given key rotations (quaternions) and where they
fall in time, it gives the orientation at any moment in between, turning at a steady
rate along the shortest path. Slerp is spherical linear interpolation.

Blending orientations number by number - np.interp, lerp or a filter on the four
quaternion values - gives quaternions that are no longer proper rotations, and a turn
that speeds up and slows down. slerp does it correctly. Use it to resample a recording
of joint or sensor orientations to another rate, to fill a dropped frame, to blend
between two poses, or to animate an object from one orientation to another.

SYNTAX:
scipy.slerp

INPUTS and PARAMETERS:

t:
When to read the orientation: a number, or a list of times. Every input is read and
sent - the only inlet that makes the node send. Before the first key or after the
last, the first or last orientation is given.

quaternions:
The key orientations, one row of w x y z each, held. At least two.

times:
When each key happens, held. Without times the keys are at 0, 1, 2 ... - so with two
keys, t from 0 to 1 goes from one to the other.

OUTPUTS:

quaternions:
The orientation at each t (one quaternion for a single t).

""" + ROT_NOTE + """

READING THE DEMO:
Two keys: no turn (1 0 0 0) and a quarter turn about z (0.7071 0 0 0.7071). A ramp from 0
to 1 slides between them; the plot shows the four values - w falling from 1 to 0.71 and
z rising from 0 to 0.71, along smooth arcs (x and y stay 0).

RELATED:
scipy.rotation_mean averages orientations. quaternion_diff and quaternion_distance
compare two. np.interp and scipy.cubic_spline interpolate ordinary values.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', 'two keys: no turn, then a quarter turn about z', 140, 53),
    N('mk', 'message', 140, 92, 300, 42, MSG('1 0 0 0 0.7071 0 0 0.7071')),
    N('rk', 'np.reshape 2 4', 470, 92, 150, 90),
    C('c3', 'a ramp, 0 to 1', 28, 220),
    N('sg', 'signal 2.0 saw', 28, 255, 129, 78, SIG('saw', 2.0, 1.0, False)),
    C('c4', 'the turn at each moment', 200, 220),
    N('sl', 'scipy.slerp', 200, 255, 160, 110),
    C('c5', 'w x y z: w falls as z rises', 400, 220),
    N('pl', 'plot', 400, 255, 388, 210, MULTI),
],
[('lb', 'out', 'mk', ''), ('mk', 'message out', 'rk', 'tensor in'), ('rk', 'output', 'sl', 'quaternions'),
 ('sg', '', 'sl', 't'), ('sl', 'quaternions', 'pl', 'y')],
['scipy.rotation_mean', 'quaternion_diff', 'quaternion_distance', 'np.interp'], 520, 1100)


# ----------------------------------------------------------- rotation_mean
page('scipy.rotation_mean', 'scipy.rotation_mean - the average of several orientations',
"""
Averages a set of orientations into the one that is, overall, closest to all of them.

Averaging the four quaternion numbers directly goes wrong: q and -q are the same
turn, so a set that happens to mix signs averages towards nonsense - towards zero,
which is no rotation at all. scipy.rotation_mean is unaffected by signs. Use it to
find a rest pose from a stretch of frames, a calibration orientation from repeated
readings, a consensus of several sensors on one segment, or a typical orientation of a
joint over a phrase.

SYNTAX:
scipy.rotation_mean

INPUTS and PARAMETERS:

quaternions:
The orientations, one row of w x y z each. Every input is averaged and sent - the only
inlet that makes the node send.

weights:
How much each counts, held: one non-negative number per row. Without weights all
count the same.

OUTPUTS:

mean:
The average orientation, as w x y z.

""" + ROT_NOTE + """

READING THE DEMO:
Three turns about z, of 10, 20 and 30 degrees - the middle one written with the
opposite sign (-0.98 ... -0.17 instead of 0.98 ... 0.17), the same turn. Their mean is
20 degrees about z: 0.985 0 0 0.174. np.mean of the same numbers gives 0.326 0 0 0.057
- not even a proper rotation. A metro re-sends the set twice a second.

RELATED:
scipy.slerp turns smoothly between orientations. np.mean averages ordinary values;
scipy.circmean averages single angles.
""",
metro() + [
    C('c1', 'z turns of 10, 20 (sign flipped) and 30 degrees', 180, 53),
    N('mq', 'message', 180, 92, 560, 42, MSG('0.9962 0 0 0.0872 -0.9848 0 0 -0.1736 0.9659 0 0 0.2588')),
    N('rq', 'np.reshape 3 4', 180, 165, 150, 90),
    C('c2', 'the proper average: 20 degrees', 360, 230),
    N('rm', 'scipy.rotation_mean', 360, 265, 190, 90),
    ROUND('r1', 580, 265),
    N('l1', 'list', 580, 350, 220, 42, LIST),
    C('c3', 'np.mean: not even a rotation', 360, 430),
    N('nm', 'np.mean 0', 360, 465, 150, 90, {'axis': '0'}),
    ROUND('r2', 580, 465),
    N('l2', 'list', 580, 550, 220, 42, LIST),
],
[('mt', '', 'mq', ''), ('mq', 'message out', 'rq', 'tensor in'), ('rq', 'output', 'rm', 'quaternions'),
 ('rq', 'output', 'nm', 'in'), ('rm', 'mean', 'r1', 'in'), ('r1', 'result', 'l1', ''),
 ('nm', 'mean', 'r2', 'in'), ('r2', 'result', 'l2', '')],
['scipy.slerp', 'scipy.circmean', 'np.mean', 'quaternion_norm'], 650, 1050)


# ----------------------------------------------------------- align_vectors
page('scipy.align_vectors', 'scipy.align_vectors - the best rotation from one set of points onto another',
"""
Finds the rotation that turns one set of points (or directions) onto another as
closely as possible - the Kabsch method - and gives that rotation, the turned points
and how well they fit.

Use it to calibrate: line a sensor's or tracker's frame up with the room from a few
matching points. To align a skeleton or a captured pose with a reference whatever way
it faces. To compare two poses after taking their facing out, or to measure how much a
rigid group of markers has turned between two moments.

SYNTAX:
scipy.align_vectors

INPUTS and PARAMETERS:

from:
The points to be turned, one row of x y z each. Every input is fitted and sent - the
only inlet that makes the node send.

onto:
Where they should end up, the same number of rows, held. Row n of from is matched
with row n of onto.

treat as:
An option, under the node. points (the default): each set is first centred on its
own middle, so the fit finds the turn of the shape whatever its position - the
aligned output is moved onto the target too. directions: the rows are directions from
the origin (unit vectors, axes) and are not centred.

OUTPUTS:

rotation:
The best rotation, as w x y z.

aligned:
The from points after the turn (and, for points, the move).

error:
How far off the fit is: the square root of the summed squared misses. 0 is a perfect
fit. Sent first, then aligned, then rotation.

""" + ROT_NOTE + """

READING THE DEMO:
Four points, and the same four turned 30 degrees about y and moved 5 along x. The
rotation found is 0.9659 0 0.2588 0 - exactly 30 degrees about y - and the error is 0.
A metro re-sends the points twice a second.

RELATED:
tracker_align lines up a tracker with a body. scipy.procrustes compares shapes while
also ignoring size. scipy.rotation_apply turns points by a rotation.
""",
metro() + [
    C('c1', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 210, 88, 46),
    C('c2', 'four points, x y z each', 180, 53),
    N('mf', 'message', 180, 92, 300, 42, MSG('1 0 0 0 2 0 0 0 3 1 1 1')),
    N('rf', 'np.reshape 4 3', 510, 92, 150, 90),
    C('c3', 'the same, turned 30 about y and moved 5 along x', 180, 175),
    N('mo', 'message', 180, 210, 440, 42, MSG('5.866 0 -0.5 5 2 0 6.5 0 2.598 6.366 1 0.366')),
    N('ro', 'np.reshape 4 3', 650, 210, 150, 90),
    C('c4', 'the turn that fits', 180, 330),
    N('av', 'scipy.align_vectors', 180, 365, 200, 110),
    ROUND('r1', 420, 365),
    N('l1', 'list', 540, 365, 240, 42, LIST),
    C('c5', 'error', 420, 470),
    ROUND('r2', 420, 505),
    N('l2', 'list', 540, 505, 100, 42, LIST),
],
[('lb', 'out', 'mo', ''), ('mo', 'message out', 'ro', 'tensor in'), ('ro', 'output', 'av', 'onto'),
 ('mt', '', 'mf', ''), ('mf', 'message out', 'rf', 'tensor in'), ('rf', 'output', 'av', 'from'),
 ('av', 'rotation', 'r1', 'in'), ('r1', 'result', 'l1', ''), ('av', 'error', 'r2', 'in'), ('r2', 'result', 'l2', '')],
['tracker_align', 'scipy.procrustes', 'scipy.rotation_apply', 'quaternion_to_euler'], 650, 1150)


# ---------------------------------------------------------- rotation_apply
page('scipy.rotation_apply', 'scipy.rotation_apply - turn points by a rotation',
"""
Turns points or vectors by a rotation given as a quaternion: a direction in a sensor's
own frame into the room's frame, a set of marker positions by a body segment's
orientation, a model's points by the way it is facing.

One quaternion turns every point; one quaternion per point (as many rows as there are
points) turns each by its own - every joint's offset by every joint's orientation.

SYNTAX:
scipy.rotation_apply

INPUTS and PARAMETERS:

vectors:
The points or vectors, one row of x y z each (or a single x y z). Every input is turned
and sent - the only inlet that makes the node send.

quaternion:
The rotation as w x y z, or one row per vector. Held.

inverse:
An option, under the node. On, the rotation is undone instead - from the room's frame
back into the sensor's.

OUTPUTS:

rotated:
The turned points, the same shape.

""" + ROT_NOTE + """

READING THE DEMO:
1 0 0 - pointing along x - turned a quarter turn about z points along y: 0 1 0. The
second node has inverse on and turns it the other way: 0 -1 0. A metro re-sends the
vector twice a second.

RELATED:
rotate_position turns about one axis by an angle. scipy.align_vectors finds the
rotation between two sets of points. quaternion_to_matrix gives the rotation as a
matrix for np.matmul.
""",
metro() + [
    C('c1', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 210, 88, 46),
    C('c2', 'a vector along x', 180, 53),
    N('mv', 'message', 180, 92, 100, 42, MSG('1 0 0')),
    C('c3', 'a quarter turn about z, held', 180, 175),
    N('mq', 'message', 180, 210, 200, 42, MSG('0.7071 0 0 0.7071')),
    C('c4', 'turned: along y', 420, 53),
    N('ap', 'scipy.rotation_apply', 420, 92, 200, 90),
    ROUND('r1', 650, 92),
    N('l1', 'list', 650, 180, 160, 42, LIST),
    C('c5', 'inverse: the other way', 420, 280),
    N('ai', 'scipy.rotation_apply', 420, 315, 200, 90, {'inverse': True}),
    ROUND('r2', 650, 315),
    N('l2', 'list', 650, 400, 160, 42, LIST),
],
[('lb', 'out', 'mq', ''), ('mq', 'message out', 'ap', 'quaternion'), ('mq', 'message out', 'ai', 'quaternion'),
 ('mt', '', 'mv', ''), ('mv', 'message out', 'ap', 'vectors'), ('mv', 'message out', 'ai', 'vectors'),
 ('ap', 'rotated', 'r1', 'in'), ('r1', 'result', 'l1', ''), ('ai', 'rotated', 'r2', 'in'), ('r2', 'result', 'l2', '')],
['rotate_position', 'scipy.align_vectors', 'quaternion_to_matrix', 'np.matmul'], 500, 1000)


# ---------------------------------------------------------- circmean/circstd
CIRC_SYNTAX = """
SYNTAX:
{stem} <axis: int> <units>
Defaults: axis None (every value together), degrees.

INPUTS and PARAMETERS:

angles:
The angles (a list or array). Every input is measured and sent. nan values are
skipped.

axis (text box, also an inlet):
None measures every angle together. A number measures along that axis: for a 2-D
array, 0 gives one answer per column.

units:
An option, under the node: degrees (the default) or radians.
"""

page('scipy.circmean', 'scipy.circmean - the average of angles',
"""
Averages angles - headings, joint angles, phases, times of day - the way they need to
be averaged: around the circle.

The ordinary mean treats -170 and 170 as far apart and gives 0, pointing exactly the
wrong way; they are 20 degrees apart, either side of 180, and their average is 180.
Whenever angles can cross the ±180 line, np.mean gives answers that are not just a
little off but turned right round. scipy.circmean adds the angles as directions -
arrows of length 1 - and gives the direction of the sum.
""" + CIRC_SYNTAX.format(stem='scipy.circmean') + """
OUTPUTS:

mean:
The average angle, from -180 to 180 (or -pi to pi).

READING THE DEMO:
Four headings just either side of due south: 170 -170 175 -175. scipy.circmean gives
180 (shown as -180 - the same direction); np.mean gives 0, due north. A metro re-sends
the angles twice a second.

RELATED:
scipy.circstd measures how spread the angles are. np.unwrap removes the jumps from a run
of angles. scipy.rotation_mean averages full 3-D orientations.
""",
metro() + [
    C('c1', 'headings either side of 180', 180, 53),
    N('ma', 'message', 180, 92, 200, 42, MSG('170 -170 175 -175')),
    C('c2', 'around the circle: south', 420, 53),
    N('cm', 'scipy.circmean', 420, 92, 170, 110),
    N('l1', 'list', 620, 92, 100, 42, LIST),
    C('c3', 'np.mean: north, the wrong way', 420, 240),
    N('nm', 'np.mean', 420, 275, 150, 90),
    N('l2', 'list', 620, 275, 100, 42, LIST),
],
[('mt', '', 'ma', ''), ('ma', 'message out', 'cm', 'angles'), ('ma', 'message out', 'nm', 'in'),
 ('cm', 'mean', 'l1', ''), ('nm', 'mean', 'l2', '')],
['scipy.circstd', 'np.unwrap', 'scipy.rotation_mean', 'np.mean'], 430, 900)


page('scipy.circstd', 'scipy.circstd - how spread out angles are',
"""
Measures how spread out a set of angles is, around the circle - the circular standard
deviation. Near 0 when the angles agree; larger the more they scatter.

np.std treats angles as ordinary numbers, so a tight bunch either side of 180 (170 and
-170, only 20 degrees apart) looks hugely spread. scipy.circstd measures the real
scatter of the directions. Use it to tell whether a performer keeps a steady heading or
wanders, how consistently a joint returns to the same angle, how locked two rhythms'
phases are.
""" + CIRC_SYNTAX.format(stem='scipy.circstd') + """
OUTPUTS:

spread:
The circular standard deviation, in the units chosen.

READING THE DEMO:
170 and -170, two headings 20 degrees apart: scipy.circstd gives about 10 degrees;
np.std gives 170. A metro re-sends the angles twice a second.

RELATED:
scipy.circmean gives the average direction. np.std measures the spread of ordinary
values.
""",
metro() + [
    C('c1', 'two headings 20 degrees apart', 180, 53),
    N('ma', 'message', 180, 92, 120, 42, MSG('170 -170')),
    C('c2', 'around the circle: small', 420, 53),
    N('cs', 'scipy.circstd', 420, 92, 170, 110),
    ROUND('r1', 620, 92, 1),
    N('l1', 'list', 740, 92, 80, 42, LIST),
    C('c3', 'np.std: huge', 420, 240),
    N('ns', 'np.std', 420, 275, 150, 90),
    N('l2', 'list', 620, 275, 100, 42, LIST),
],
[('mt', '', 'ma', ''), ('ma', 'message out', 'cs', 'angles'), ('ma', 'message out', 'ns', 'in'),
 ('cs', 'spread', 'r1', 'in'), ('r1', 'result', 'l1', ''), ('ns', 'std', 'l2', '')],
['scipy.circmean', 'np.std', 'np.unwrap', 'scipy.hilbert'], 430, 860)


# ------------------------------------------------------------------ hilbert
# Inside make_swell: 10 seconds at 60 a second of a wave at 2 cycles a second whose
# size swells from 0.2 to 1. The swell (np.linspace) is linked first, so it reaches
# the * operand before the wave.
SWELL_SUB = {
    'name': 'make_swell',
    'host': 'sw',
    'demo': [
        S('mt', 'metro 2000', 40, 40, 129, 70, METRO_2S),
        S('ct', 'comment every two seconds', 190, 55, 220, 30),
        S('ls', 'np.linspace 0.2 1 600', 430, 130, 190, 150),
        S('cl', 'comment the swell: 0.2 to 1', 430, 290, 240, 30),
        S('ar', 'np.arange 600', 40, 130, 150, 110),
        S('ca', 'comment 600 samples: 10 s at 60 a second', 40, 250, 340, 30),
        S('m1', '* 12', 40, 300, 129, 70),
        S('s1', 'sin', 40, 390, 129, 70),
        S('c1', 'comment 2 cycles a second', 190, 405, 220, 30),
        S('sc', '*', 40, 480, 129, 70),
        S('out', 'out swell', 40, 580, 150, 80),
    ],
    'links': [('mt', '', 'ls', ''), ('mt', '', 'ar', ''), ('ls', 'linspace out', 'sc', 'operand', 0, 1),
              ('ar', 'arange out', 'm1', 'in'), ('m1', 'result', 's1', 'in'), ('s1', 'out', 'sc', 'in'),
              ('sc', 'result', 'out', 'swell')],
}

page('scipy.hilbert', 'scipy.hilbert - the envelope and phase of an oscillation',
"""
Takes a signal that swings back and forth and separates it into how big the swing is
at each moment - the envelope - and where it is in its cycle - the phase - plus how
fast that phase is turning: the instantaneous frequency. It uses the Hilbert
transform to build the signal's "analytic" partner, a wave a quarter cycle behind it.

Phase is the part that is hard to get otherwise: it says at every sample whether a
repetitive movement is at the top of its swing, on the way down, at the bottom - so
two performers' phases can be compared for synchrony, a gait can be followed through
its cycle, or sound can be locked to the cycle of a movement. The envelope follows the
size of the swing without the ripples of the swing itself.

It works best on a signal with one main rhythm; filter first (scipy.filtfilt
band-pass) if there are several.

SYNTAX:
scipy.hilbert <sample rate>
Default: sample rate 60.

INPUTS and PARAMETERS:

in:
The signal (a list or array). Every input is analysed and sent.

sample rate (float box, also an inlet):
Samples per second, for the frequency outlet.

axis (int box, also an inlet):
Which way time runs: -1 along each row, 0 down each column.

remove mean:
An option, under the node. On (the default), the average level is taken away first, so
the phase measures the swing about the middle.

OUTPUTS:

envelope:
The size of the swing at each sample.

phase:
Where in its cycle each sample is, in degrees from -180 to 180. 0 is the top of the
swing ... -90 rising through the middle, 90 falling.

frequency:
How fast the phase turns, in cycles per second. Sent first, then phase, then envelope.

READING THE DEMO:
The make_swell subpatcher makes a wave at 2 cycles a second that grows from 0.2 to 1
across 10 seconds (left). Its envelope (middle) is a smooth ramp from 0.2 to 1, without
the wiggles; its phase (right) is a sawtooth climbing from -180 to 180 twice a second -
20 teeth in the 10 seconds - the position in each cycle.

RELATED:
envelope and speech_envelope follow loudness in a live stream. np.unwrap turns the phase
into a steadily counting one. scipy.circmean and scipy.circstd average and compare
phases. scipy.filtfilt isolates one rhythm first.
""",
[
    C('c1', 'a wave that swells', 28, 53),
    N('sw', 'p make_swell', 28, 92, 180, 70),
    C('c2', 'envelope and phase', 260, 53),
    N('hb', 'scipy.hilbert', 260, 92, 180, 130),
    C('c3', 'the wave', 28, 270),
    N('p1', 'plot', 28, 305, 268, 180, LINE(600, -1.1, 1.1, 260)),
    C('c4', 'its envelope', 310, 270),
    N('p2', 'plot', 310, 305, 268, 180, LINE(600, 0.0, 1.1, 260)),
    C('c5', 'its phase: a sawtooth', 592, 270),
    N('p3', 'plot', 592, 305, 268, 180, LINE(600, -180.0, 180.0, 260)),
],
[('sw', 'swell', 'hb', 'in'), ('sw', 'swell', 'p1', 'y'), ('hb', 'envelope', 'p2', 'y'),
 ('hb', 'phase', 'p3', 'y')],
['envelope', 'np.unwrap', 'scipy.circmean', 'scipy.filtfilt'], 540, 1300, subpatch=SWELL_SUB)


# ------------------------------------------------------------------ detrend
# Inside make_drift: 2 seconds of a wave, 3 cycles, riding on a steady drift from 0
# to 3. The drift (np.linspace) is linked first, so it reaches the + operand before
# the wave.
DRIFT_SUB = {
    'name': 'make_drift',
    'host': 'dr',
    'demo': [
        S('mt', 'metro 1000', 40, 40, 129, 70, METRO_1S),
        S('ct', 'comment every second', 190, 55, 200, 30),
        S('ls', 'np.linspace 0 3 120', 330, 130, 190, 150),
        S('cl', 'comment the drift: 0 to 3', 330, 290, 220, 30),
        S('ar', 'np.arange 120', 40, 130, 150, 110),
        S('m1', '* 9', 40, 270, 129, 70),
        S('s1', 'sin', 40, 360, 129, 70),
        S('c1', 'comment 3 cycles', 190, 375, 160, 30),
        S('ad', '+', 40, 450, 129, 70),
        S('out', 'out drifting', 40, 550, 150, 80),
    ],
    'links': [('mt', '', 'ls', ''), ('mt', '', 'ar', ''), ('ls', 'linspace out', 'ad', 'operand', 0, 1),
              ('ar', 'arange out', 'm1', 'in'), ('m1', 'result', 's1', 'in'), ('s1', 'out', 'ad', 'in'),
              ('ad', 'result', 'out', 'drifting')],
}

page('scipy.detrend', 'scipy.detrend - take away a drift',
"""
Removes a steady drift from a recording - the best-fitting straight line through it -
leaving only the movement around that line. Or, with constant, just removes the
average level.

Sensors drift; performers walk slowly across the room while they gesture; a breathing
signal rises as the sensor warms. A drift swamps everything else: it dominates a
spectrum, misleads a mean, defeats a threshold. Taking it away first leaves the
movement itself.

SYNTAX:
scipy.detrend <linear or constant>
Default: linear.

INPUTS and PARAMETERS:

in:
The recording (a list or array). Every input is detrended and sent.

axis (int box, also an inlet):
Which way time runs: -1 along each row, 0 down each column - for a buffer of frames x
joints, use 0.

remove:
An option, under the node. linear (the default) takes away the best straight line;
constant takes away only the average level.

OUTPUTS:

detrended:
The recording with the drift taken away, the same shape.

READING THE DEMO:
The make_drift subpatcher makes a wave of 3 cycles riding on a steady rise from 0 to 3
(left). Detrended (right), the rise is gone and the wave swings about 0.

RELATED:
scipy.filtfilt high-pass removes slow movement of any shape, not only straight lines.
np.polyfit fits a curved trend. np.mean and - remove just the average.
""",
[
    C('c1', 'a wave on a drift', 28, 53),
    N('dr', 'p make_drift', 28, 92, 180, 70),
    C('c2', 'take the drift away', 260, 53),
    N('dt', 'scipy.detrend', 260, 92, 160, 90),
    C('c3', 'in', 28, 250),
    N('p1', 'plot', 28, 285, 388, 180, LINE(120, -1.5, 4.5, 380)),
    C('c4', 'out: the wave about 0', 440, 250),
    N('p2', 'plot', 440, 285, 388, 180, LINE(120, -1.5, 4.5, 380)),
],
[('dr', 'drifting', 'dt', 'in'), ('dr', 'drifting', 'p1', 'y'), ('dt', 'detrended', 'p2', 'y')],
['scipy.filtfilt', 'np.polyfit', 'np.mean', 'np.diff'], 520, 900, subpatch=DRIFT_SUB)


# --------------------------------------------------------------- convex_hull
page('scipy.convex_hull', 'scipy.convex_hull - the space a set of points encloses',
"""
Finds the convex hull of a set of points - the smallest shape with no dents that
contains them all, like a rubber band stretched round them (or, in 3-D, shrink-wrap) -
and gives its size, the length of its boundary and which points lie on it.

The size of the hull is a measure of expansiveness: how much space a body's hands,
feet and head span (the reach of a pose - its kinesphere), how much floor a group of
performers covers, how wide a gesture's path is. Watching it over time shows a body
opening and closing.

SYNTAX:
scipy.convex_hull

INPUTS and PARAMETERS:

points:
The points: rows of x y (in 2-D) or x y z (in 3-D). Every input is measured and sent.
Rows with nan are left out. Points all in a line (or, in 3-D, all in a plane) enclose
nothing, and give 0.

OUTPUTS:

size:
The area enclosed (2-D) or the volume (3-D).

boundary:
The length of the boundary (2-D) or the area of the surface (3-D).

vertices:
Which points lie on the hull, by row number. Sent first, then boundary, then size.

READING THE DEMO:
Top: a 2 x 2 square's corners and its centre. The hull is the square: area 4, perimeter
8; the centre, inside, is not one of the vertices (0 1 2 3).

Bottom: 12 random points in a 1 x 1 square, new every second - the area they span
changes as they scatter.

RELATED:
scipy.kdtree finds nearest points. np.ptp gives the extent along each axis - a box
rather than a hull. scipy.label finds regions in a grid.
""",
metro(label='every second', props=METRO_1S, init='metro 1000') + [
    C('c1', 'a square and its centre: x y ...', 180, 53),
    N('ms', 'message', 180, 92, 220, 42, MSG('0 0 2 0 2 2 0 2 1 1')),
    N('rs', 'np.reshape 5 2', 430, 92, 150, 90),
    C('c2', 'the hull', 620, 53),
    N('h1', 'scipy.convex_hull', 620, 92, 190, 90),
    C('c3', 'area', 28, 230),
    N('a1', 'list', 28, 265, 80, 42, LIST),
    C('c4', 'perimeter', 140, 230),
    N('b1', 'list', 140, 265, 80, 42, LIST),
    C('c5', 'on the hull', 250, 230),
    N('v1', 'list', 250, 265, 160, 42, LIST),
    C('c6', '12 random points', 180, 360),
    N('rn', 'np.rand 12 2', 180, 395, 150, 130),
    N('h2', 'scipy.convex_hull', 400, 395, 190, 90),
    C('c7', 'the area they span', 620, 360),
    ROUND('r2', 620, 395, 2),
    N('a2', 'list', 620, 480, 80, 42, LIST),
],
[('mt', '', 'ms', ''), ('ms', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'h1', 'points'),
 ('h1', 'size', 'a1', ''), ('h1', 'boundary', 'b1', ''), ('h1', 'vertices', 'v1', ''),
 ('mt', '', 'rn', ''), ('rn', 'random array', 'h2', 'points'), ('h2', 'size', 'r2', 'in'),
 ('r2', 'result', 'a2', '')],
['scipy.kdtree', 'np.ptp', 'scipy.label', 'np.histogram2d'], 600, 1000)


# -------------------------------------------------------------------- kmeans
page('scipy.kmeans', 'scipy.kmeans - sort points into k groups',
"""
Sorts a set of points - positions, poses, feature vectors of any length - into k groups
of points that lie close together, and gives which group each point is in and the
centre of each group. k-means clustering.

Use it to find where people tend to gather on a floor, to build a small vocabulary of
recurring poses or gestures from a recording (each centre a typical pose), to divide a
point cloud into objects, or to quantize a continuous feature into a few states.

SYNTAX:
scipy.kmeans <k: int>
Default: k 3.

INPUTS and PARAMETERS:

points:
The points, as rows of coordinates (a plain list is points on a line). Every input is
grouped and sent. Rows with nan are left out.

k (int box, also an inlet):
How many groups.

follow:
An option, under the node. On (the default), each new set starts from the last set's
centres, so group 0 stays the same group from one frame to the next as the points
move. Off, every set is grouped afresh, and the numbers can swap.

seed:
An option, under the node. -1 (the default) picks the starting centres differently each
run; any other number repeats the same grouping.

OUTPUTS:

labels:
Which group each point is in: 0 to k - 1.

centres:
The centre of each group, one row each. Sent first, then the labels.

READING THE DEMO:
Six points: three near 0 0 and three near 5 5. With k 2 the labels are three of one
number and three of the other, and the centres are about 0.1 0.13 and 5 5. A metro
re-sends the points twice a second; the labels stay put because follow is on.

RELATED:
scipy.kdtree finds which of a fixed set of points is nearest. pc_cluster_filter groups
the points of a point cloud. scipy.label finds regions in a grid.
""",
metro() + [
    C('c1', 'three points near 0 0, three near 5 5', 180, 53),
    N('mp', 'message', 180, 92, 380, 42, MSG('0 0 0.2 0.1 0.1 0.3 5 5 5.2 4.9 4.8 5.1')),
    N('rp', 'np.reshape 6 2', 590, 92, 150, 90),
    C('c2', 'two groups', 180, 220),
    N('km', 'scipy.kmeans 2', 180, 255, 170, 130),
    C('c3', 'which group', 400, 220),
    N('l1', 'list', 400, 255, 200, 42, LIST),
    C('c4', 'their centres', 400, 330),
    ROUND('r2', 400, 365, 2),
    N('l2', 'list', 520, 365, 280, 42, LIST),
],
[('mt', '', 'mp', ''), ('mp', 'message out', 'rp', 'tensor in'), ('rp', 'output', 'km', 'points'),
 ('km', 'labels', 'l1', ''), ('km', 'centres', 'r2', 'in'), ('r2', 'result', 'l2', '')],
['scipy.kdtree', 'pc_cluster_filter', 'scipy.label', 'np.unique'], 500, 1000)


# ---------------------------------------------------------------- procrustes
page('scipy.procrustes', 'scipy.procrustes - how alike two shapes are',
"""
Compares two shapes - two sets of matching points - after taking away everything that
is not shape: where they are, how big they are and which way they are turned. The
difference that is left is 0 for the same shape and grows the more they differ.

Use it to compare poses or gestures regardless of where a performer stands, how tall
they are or which way they face; to match a pose against a library of reference poses;
to measure how a group's formation keeps or loses its shape as it moves.

SYNTAX:
scipy.procrustes

INPUTS and PARAMETERS:

shape:
The shape to compare: rows of coordinates (x y, or x y z, ...). Every input is compared
and sent - the only inlet that makes the node send.

reference:
The shape to compare it with, the same number of rows, held. Row n of each must be
the same point (the same joint, the same marker).

OUTPUTS:

difference:
How different the shapes are once position, size and turn are taken away: 0 for the
same shape, up to 1.

aligned:
The shape, moved, scaled and turned to fit the reference as well as it can (in the
reference's standardized form).

reference standardized:
The reference centred and scaled to size 1, for comparison with aligned. Sent first,
then aligned, then difference.

READING THE DEMO:
Top: a right-angled triangle, and the same triangle turned 40 degrees, three times as
big and moved: difference 0. Bottom: a different triangle against the same reference:
difference about 0.42. A metro re-sends the shapes twice a second.

RELATED:
scipy.align_vectors finds the turn between two sets of points, keeping their size.
np.linalg.norm of the difference of two poses compares them without any alignment.
""",
metro() + [
    C('c1', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 210, 88, 46),
    C('c2', 'reference: a right-angled triangle', 180, 175),
    N('mr', 'message', 180, 210, 160, 42, MSG('0 0 1 0 0 1')),
    N('rr', 'np.reshape 3 2', 370, 210, 150, 90),
    C('c3', 'the same, turned, scaled and moved', 180, 53),
    N('ms', 'message', 180, 92, 330, 42, MSG('7 7 9.298 8.928 5.072 9.298')),
    N('rs', 'np.reshape 3 2', 540, 92, 150, 90),
    C('c4', 'same shape: 0', 560, 260),
    N('p1', 'scipy.procrustes', 560, 295, 200, 110),
    ROUND('r1', 560, 430, 3),
    N('l1', 'list', 680, 430, 100, 42, LIST),
    C('c5', 'a different triangle', 180, 340),
    N('md', 'message', 180, 375, 160, 42, MSG('0 0 1 0 3 1')),
    N('rd', 'np.reshape 3 2', 370, 375, 150, 90),
    C('c6', 'different: more than 0', 560, 510),
    N('p2', 'scipy.procrustes', 560, 545, 200, 110),
    ROUND('r2', 560, 680, 3),
    N('l2', 'list', 680, 680, 100, 42, LIST),
],
[('lb', 'out', 'mr', ''), ('mr', 'message out', 'rr', 'tensor in'),
 ('rr', 'output', 'p1', 'reference'), ('rr', 'output', 'p2', 'reference'),
 ('mt', '', 'ms', ''), ('ms', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'p1', 'shape'),
 ('mt', '', 'md', ''), ('md', 'message out', 'rd', 'tensor in'), ('rd', 'output', 'p2', 'shape'),
 ('p1', 'difference', 'r1', 'in'), ('r1', 'result', 'l1', ''),
 ('p2', 'difference', 'r2', 'in'), ('r2', 'result', 'l2', '')],
['scipy.align_vectors', 'np.linalg.norm', 'scipy.kdtree', 'scipy.kmeans'], 780, 1100)


# ---------------------------------------------------------------- curve_fit
page('scipy.curve_fit', 'scipy.curve_fit - fit any formula to data',
"""
Fits a formula you write to a set of points: it finds the values of the formula's
parameters that make it pass as close as possible to the data, and gives them, the
fitted curve and how far off it is.

np.polyfit fits only polynomials. curve_fit fits whatever describes the process: a
decay that settles (a exp(-b x) + c), a rise that levels off, a damped oscillation
(a exp(-b x) sin(c x + d)), a power law (a x ^ b). The parameters then mean something -
how fast a movement settles, the level it settles to, how springy it is.

SYNTAX:
scipy.curve_fit <formula>
Default: a * exp(-b * x) + c.

EXAMPLES:
scipy.curve_fit m * x + k
scipy.curve_fit a * x ^ b

THE FORMULA:
Written in x, with any other names as the parameters to find (in the order they first
appear). It can use + - * / and ^ (or **) for powers, and exp, log, log10, log2, sqrt,
sin, cos, tan, arctan, arcsin, arccos, tanh, sinh, cosh, abs, sign, pi and e.

INPUTS and PARAMETERS:

y:
The data's values. Every input is fitted and sent - the only inlet that makes the node
send.

x:
Where the values are, held. Without x they are at 0, 1, 2 ...

guess:
Starting values for the parameters, held, in the same order. Without a guess every
parameter starts at 1. A formula with a sharp bend may need a sensible guess to find
the right fit; if none is found, nothing is sent.

formula (text box, also an inlet):
The formula, as above.

OUTPUTS:

parameters:
The fitted values, in the order of names.

names:
The parameters' names, in that order.

fitted:
The formula's values at each x, with the fitted parameters.

error:
How far off, on average: the root mean square of the misses. Sent first, then fitted,
names, parameters.

READING THE DEMO:
Nine measurements of something settling - 3.48 down to about 0.5, at x 0 to 4 in steps of
0.5 - with a little noise. The fit gives a about 3, b about 1.43 and c about 0.47: it
starts 3 above its resting level of 0.47 and settles at a rate of 1.43. The fitted curve
(right) passes smoothly through the noisy points (left). A metro re-sends the data
twice a second.

RELATED:
np.polyfit fits polynomials quickly and always. np.linalg.lstsq fits any sum of
functions. scipy.cubic_spline passes exactly through every point.
""",
metro() + [
    C('c1', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 210, 88, 46),
    C('c2', 'nine values, settling', 180, 53),
    N('my', 'message', 180, 92, 380, 42, MSG('3.48 1.91 1.22 0.84 0.60 0.57 0.51 0.52 0.46')),
    C('c3', 'at x, held', 180, 175),
    N('mx', 'message', 180, 210, 300, 42, MSG('0 0.5 1 1.5 2 2.5 3 3.5 4')),
    C('c4', 'fit a decay', 600, 53),
    N('cf', 'scipy.curve_fit', 600, 92, 220, 170),
    C('c5', 'a b c', 600, 300),
    ROUND('r1', 600, 335, 2),
    N('l1', 'list', 600, 420, 200, 42, LIST),
    C('c6', 'the data', 28, 330),
    N('p1', 'plot', 28, 365, 268, 180, LINE(9, 0.0, 4.0, 260, 'stem')),
    C('c7', 'the fitted curve', 310, 330),
    N('p2', 'plot', 310, 365, 268, 180, LINE(9, 0.0, 4.0, 260)),
],
[('lb', 'out', 'mx', ''), ('mx', 'message out', 'cf', 'x'), ('mt', '', 'my', ''),
 ('my', 'message out', 'cf', 'y'), ('my', 'message out', 'p1', 'y'),
 ('cf', 'parameters', 'r1', 'in'), ('r1', 'result', 'l1', ''), ('cf', 'fitted', 'p2', 'y')],
['np.polyfit', 'np.linalg.lstsq', 'scipy.cubic_spline', 'np.polyval'], 600, 1300)


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
