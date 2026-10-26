"""Seed the help pages for the fourth batch of new torch nodes (2026-10-25): one page per node.

    python3 dpg_system/help/tools/seed_torch_batch4_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

t.interp, t.hilbert, t.scatter_reduce. (The same batch gave t.fft and t.rfft a
frequencies outlet; their pages were edited in place, as the live pages differ from
what make_torch_signal_help.py builds.)

t.interp follows the np.interp page; t.hilbert uses scipy.hilbert's make_swell
subpatcher. Held inputs come from a load_bang and the metro drives the hot inlet. Not
a make_* script: rebuild.py never runs it, and it will not write over an existing page
without --force. Treat the pages as hand-made once written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from seed_numpy_more_help import LIST, MSG, BARS, C, N, see_also
from seed_scipy3_help import SWELL_SUB, LINE


PAGES = {}


def page(stem, title, body, demo, links, also, also_y, text_height=820, subpatch=None):
    PAGES[stem] = dict(title=title, body=body.strip('\n'), demo=demo + see_also(also_y, also),
                       links=links, text_height=text_height, subpatch=subpatch)


METRO = {'on': True, 'period': 200.0, 'units': 'milliseconds'}


def metro(y=53, label='five times a second', props=METRO, init='metro 200'):
    return [C('cm', label, 28, y), N('mt', init, 28, y + 39, 129, 70, props)]


def held(y=175, key='lb'):
    return [C('c_lb', 'once at load', 28, y), N(key, 'load_bang', 28, y + 35, 88, 46)]


def W(text):
    """A message or list wide enough for this text."""
    return max(60, int(len(text) * 13.5) + 30)


# ------------------------------------------------------------------- t.interp
page('t.interp', 't.interp - look up values between the points of a table',
"""
Draws straight lines between the points of a table and reads values off them. The
table is a set of positions (xp) and the values at them (fp); for each x you send,
t.interp finds where it falls among the positions and blends the two nearest values in
proportion.

Use it as a breakpoint curve - a control that rises fast, levels off and rises again,
drawn with a few points - or as a lookup table, a transfer function, a calibration from
raw sensor readings to real units. Send it many x at once to resample at any positions,
not only evenly spaced ones: a movement recorded with uneven timestamps read off at
steady frames, or played back along a tempo curve (time-warping) - x is then the
moment in the recording to read at each output frame. A table of poses (one row per
position) gives back whole poses.

t.interpolate resizes by an even factor; t.interp reads wherever you ask. The work is
done on x's device.

SYNTAX:
t.interp

INPUTS and PARAMETERS:

x:
Where to read the table: a number, list or tensor of any shape. Every input is looked
up and sent - the only inlet that makes the node send.

xp:
The positions of the table's points. They are held; send them first. If they are not
in increasing order they are sorted (with their values) for you. If xp is never sent,
the positions are 0, 1, 2 ... - the index of each value - so x 1.5 reads halfway
between the second and third values.

fp:
The values at those positions, held. One per position. A 2-D fp - one row per
position, with several columns - is read column by column, so a table of xyz points
gives back xyz points.

OUTPUTS:

values:
The value read at each x, the shape of x (plus fp's columns). Below the first position
the first value is given, and above the last the last: the curve is held flat beyond
its ends.

READING THE DEMO:
Top: a curve with three points: at 0 it is 0, at 0.25 it has already reached 0.875,
and at 1 it is 1. 64 positions from 0 to 1 read through it rise steeply then level off
- the shape of the table, not a straight line.

Bottom: with no xp, the table 0 10 30 is indexed by position. 0.5 is halfway between 0
and 10, so 5; 1.5 halfway between 10 and 30, so 20; 2 and beyond are 30.

RELATED:
t.interpolate resamples by an even factor. np.interp is the numpy version. lerp blends
between two values by a fraction; t.linspace makes evenly spaced positions.
""",
metro() + held() + [
    C('c2', 'positions', 180, 175),
    N('mx', 'message', 180, 210, 160, 42, MSG('0 0.25 1')),
    C('c3', 'values at them', 180, 270),
    N('mf', 'message', 180, 305, 160, 42, MSG('0 0.875 1')),
    C('c4', '64 positions, 0 to 1', 380, 53),
    N('ls', 't.linspace 0 1 64', 380, 92, 170, 130),
    C('c5', 'read off the curve', 380, 230),
    N('ip', 't.interp', 380, 265, 150, 110),
    C('c6', 'steep, then level', 600, 230),
    N('pl', 'plot', 600, 265, 228, 178, dict(BARS(64, 1.0, 210, 130, 'line'), **{'min y': 0.0})),
    C('c7', 'with no positions: by index', 180, 480),
    N('m3', 'message', 180, 515, 140, 42, MSG('0 10 30')),
    C('c9', 'x', 180, 580),
    N('m4', 'message', 180, 615, 180, 42, MSG('0 0.5 1.5 2 3')),
    N('i2', 't.interp', 400, 515, 150, 110),
    C('c10', 'values', 600, 480),
    N('l1', 'list', 600, 515, 220, 42, LIST),
],
[('lb', 'out', 'mx', ''), ('lb', 'out', 'mf', ''), ('lb', 'out', 'm3', ''),
 ('mx', 'message out', 'ip', 'xp'), ('mf', 'message out', 'ip', 'fp'),
 ('mt', '', 'ls', '###input'), ('ls', 'linspace out', 'ip', 'x'), ('ip', 'values', 'pl', 'y'),
 ('m3', 'message out', 'i2', 'fp'), ('mt', '', 'm4', ''), ('m4', 'message out', 'i2', 'x'),
 ('i2', 'values', 'l1', '')],
['t.interpolate', 'np.interp', 'lerp', 't.linspace'], 720, 1000)


# ------------------------------------------------------------------ t.hilbert
page('t.hilbert', 't.hilbert - the envelope and phase of an oscillation',
"""
Takes a signal that swings back and forth and separates it into how big the swing is
at each moment - the envelope - and where it is in its cycle - the phase - plus how fast
that phase is turning: the instantaneous frequency. It uses the Hilbert transform to
build the signal's "analytic" partner, a wave a quarter cycle behind it.

Phase is the part that is hard to get otherwise: it says at every sample whether a
repetitive movement is at the top of its swing, on the way down, at the bottom - so two
performers' phases can be compared for synchrony, a gait can be followed through its
cycle, or sound can be locked to the cycle of a movement. The envelope follows the size
of the swing without the ripples of the swing itself.

It works best on a signal with one main rhythm; band-pass it first if there are
several. Every row of a stack (every joint of a buffer, with dim 0 for frames x joints)
is analysed at once, on the input's device. It is scipy.hilbert, for tensors.

SYNTAX:
t.hilbert <sample rate>
Default: sample rate 60.

INPUTS and PARAMETERS:

in:
The signal (a list or tensor). Every input is analysed and sent.

sample rate (float box, also an inlet):
Samples per second, for the frequency outlet.

dim (int box, also an inlet):
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
scipy.hilbert is the numpy version. envelope and speech_envelope follow loudness in a
live stream. t.stft shows every frequency at every moment rather than one rhythm's
phase.
""",
[
    C('c1', 'a wave that swells', 28, 53),
    N('sw', 'p make_swell', 28, 92, 180, 70),
    C('c2', 'envelope and phase', 260, 53),
    N('hb', 't.hilbert', 260, 92, 180, 130),
    C('c3', 'the wave', 28, 270),
    N('p1', 'plot', 28, 305, 268, 180, LINE(600, -1.1, 1.1, 260)),
    C('c4', 'its envelope', 310, 270),
    N('p2', 'plot', 310, 305, 268, 180, LINE(600, 0.0, 1.1, 260)),
    C('c5', 'its phase: a sawtooth', 592, 270),
    N('p3', 'plot', 592, 305, 268, 180, LINE(600, -180.0, 180.0, 260)),
],
[('sw', 'swell', 'hb', 'in'), ('sw', 'swell', 'p1', 'y'), ('hb', 'envelope', 'p2', 'y'),
 ('hb', 'phase', 'p3', 'y')],
['scipy.hilbert', 'envelope', 't.stft', 't.rfft'], 540, 1100, subpatch=SWELL_SUB)


# ------------------------------------------------------------ t.scatter_reduce
SPEEDS = '1 2 3 4 5 5'
LIMBS = '0 1 0 2 1 0'
page('t.scatter_reduce', 't.scatter_reduce - one answer per group',
"""
Gathers values by a group label and reduces each group to one answer: the sum, the
mean, the largest, the smallest or the product of the values that carry that label.

Label 24 joints by limb and it gives a total, or an average, or a peak per limb in one
step - how hard the left arm is working, how fast the head is moving, each gang of
joints' combined torque. Label points by the cluster they fell into (scipy.kmeans) and
it gives each cluster's centre (mean). Label frames by the section of a piece and it
gives each section's total. Labels need not be in order or grouped together.

It works along a dim - 0 by default, so for a stack of joints (24 x 3) the labels name
the rows and whole vectors are reduced. The answer stays on the input's device; on the
Mac's GPU the reductions it lacks run on the cpu and the answer is moved back.

SYNTAX:
t.scatter_reduce <reduce> <group count>
Defaults: sum; as many groups as the largest label needs.

EXAMPLE:
t.scatter_reduce mean 5

INPUTS and PARAMETERS:

values:
The values to reduce (a list or tensor). Every input is reduced and sent - the only
inlet that makes the node send.

groups:
One label per value along the dim: whole numbers from 0. Held: send them first.

group count (int box, also an inlet):
How many groups there are, so the answer is the same length whatever labels arrive.
-1 takes the largest label plus one. A group with no values gets 0.

dim (int box, also an inlet):
The dim the labels run along: 0 (the default) for rows.

reduce:
An option, under the node: sum (the default), mean, max, min or prod.

OUTPUTS:

per group:
One answer per group, in label order: 0 first.

READING THE DEMO:
Six joints' speeds, 1 2 3 4 5 5, labelled by limb: 0 the left arm (joints 1, 3 and 6),
1 the right arm (joints 2 and 5), 2 the head (joint 4). The sums per limb are 9, 7 and
4; the means 3, 3.5 and 4.

RELATED:
t.sum and t.mean reduce everything, or a whole dim, to one answer. t.bincount counts
how many carry each label. scipy.kmeans makes the labels for clusters.
""",
metro() + held() + [
    C('c1', 'six joint speeds', 180, 53),
    N('m1', 'message', 180, 92, W(SPEEDS), 42, MSG(SPEEDS)),
    C('c2', 'their limbs, held', 180, 175),
    N('m2', 'message', 180, 210, W(LIMBS), 42, MSG(LIMBS)),
    C('c3', 'sum per limb', 440, 53),
    N('s1', 't.scatter_reduce', 440, 92, 190, 130),
    N('l1', 'list', 660, 92, 160, 42, LIST),
    C('c4', 'mean per limb', 440, 260),
    N('s2', 't.scatter_reduce mean', 440, 295, 190, 130),
    N('l2', 'list', 660, 295, 160, 42, LIST),
],
[('mt', '', 'm1', ''), ('lb', 'out', 'm2', ''), ('m2', 'message out', 's1', 'groups'),
 ('m2', 'message out', 's2', 'groups'), ('m1', 'message out', 's1', 'values'),
 ('m1', 'message out', 's2', 'values'), ('s1', 'per group', 'l1', ''), ('s2', 'per group', 'l2', '')],
['t.sum', 't.mean', 't.bincount', 'scipy.kmeans'], 480, 1000)


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
        print(build(stem, p['title'], p['body'], p['demo'], p['links'], demo_width=880,
                    text_width=760, text_height=p['text_height'], subpatch=p['subpatch']))


if __name__ == '__main__':
    main()
