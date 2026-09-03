"""the parallel Kalman filters, the ES-EKFs, band splitting, Savitzky-Golay."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

PARALLEL = """
THESE FILTER EVERY JOINT AT ONCE:
All of these are vectorised over parallel streams. A tensor of shape
[joints, components] is one time-step for the whole body, and the filter keeps
independent state for every stream. That is why they are torch nodes rather
than one filter per joint: twenty-two filters running as one operation.

Send a whole pose, get a whole filtered pose back.
"""

QUAT = """
WHY QUATERNIONS NEED THEIR OWN FILTERS:
You cannot filter a quaternion by smoothing its four numbers separately. They
are not independent - a valid rotation has unit length - so the average of two
quaternions is not a rotation at all, and the result has to be renormalised
into something that is not the rotation you wanted. The double-cover problem
makes it worse: q and -q are the same rotation, so a sign flip looks like an
enormous jump to anything treating the components as numbers.

The quaternion versions of these filters work in the rotation's own terms
instead, so the result is always a valid rotation and a sign flip is not an
event.
"""

# ------------------------------------------------------------ t.smart_clamp_kf
body = """These are Kalman filters that try to tell a glitch from a fast movement.

An ordinary smoothing filter cannot. Both look like a large sudden change, and
any setting that removes the glitch also removes the movement. These four
attack that problem from different directions, and which is right depends on
what your bad frames actually look like.

THE NODES:

t.smart_clamp_kf        limits how far one frame may correct the estimate
t.smart_clamp_quat_kf   the same, for quaternions
t.persistence_quat_kf   holds its estimate through brief disagreement
t.hybrid_quat_kf        blends between a damping and a responsive mode
t.jerk_aware_quat_kf    classifies motion by the coherence of its jerk

SMART CLAMPING LIMITS THE CORRECTION, NOT THE SIGNAL:
A Kalman filter works by correcting its prediction toward the measurement. 
Smart clamping caps the MAGNITUDE of that correction per frame - so a 
one-frame glitch cannot teleport the filter's state however wrong the reading 
is, while a genuinely fast movement, which arrives as many consecutive large 
corrections, still tracks because each one is individually plausible.

That is the useful asymmetry: a glitch is one big step, real motion is a run of 
them.

JERK-AWARE USES COHERENCE, WHICH IS THE SHARPEST TEST:
It looks at angular jerk over time and asks whether the change is COHERENT. 
A real movement accelerates, sustains and decelerates - its jerk has structure. 
A sensor spike does not; it is one frame unrelated to its neighbours.

When it decides a frame is a spike, it does not simply hold - it extrapolates 
from the last good pose using SMOOTHED velocity. Using the single-frame velocity 
would carry the noise into the extrapolation, which is the mistake this avoids.

HYBRID BLENDS RATHER THAN SWITCHES:
It runs a damping mode and a responsive mode and blends between them 
continuously. Its 'alphas' outlet reports the blend, which is worth watching: 
it tells you what the filter thinks is happening, and a blend that is pinned at 
one end means the transition ranges are set wrong.

Blending rather than switching matters. A filter that flips between two modes 
produces a discontinuity at every flip, which is a new artefact in place of the 
old one.
""" + PARALLEL + QUAT + """
SYNTAX:
t.smart_clamp_kf
t.jerk_aware_quat_kf

EXAMPLE:
t.smart_clamp_quat_kf

INPUTS and PARAMETERS:

input:
The stream. A tensor of parallel streams, or a pose.

dt:
The time between frames, in seconds. Get this right - every rate the filter
reasons about is derived from it.

responsiveness / smoothness:
The basic trade. As always, more of one is less of the other.

jerk_threshold / accel_limit (t.jerk_aware_quat_kf):
What counts as an implausible change.

Damp Smoothness / Damp Responsiveness / Resp Smoothness (t.hybrid_quat_kf):
The two modes it blends between.

reset:
Clear the filter's state and start again.

OUTPUTS: 

filtered:
The result, in the same shape as the input.

alphas:
What the filter is doing - the blend between modes. Watch this while tuning.

TUNE AGAINST REAL BAD FRAMES:
None of these can be set sensibly against clean data, because their whole
purpose only engages when something goes wrong. Record a take containing the
failure you are trying to remove, and tune against that - with the alphas
outlet plotted, so you can see the filter engaging rather than guessing."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 3.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 3.0, 0.7)},
    {'key': 'tog', 'init': 'toggle', 'pos': (200, 62), 'w': 45, 'h': 42,
     'props': {'': True}},
    {'key': 'met', 'init': 'metro 16', 'pos': (200, 110), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 16.0, 'units': 'milliseconds'}},
    {'key': 'rnd', 'init': 'random.gauss 0.0 0.3', 'pos': (200, 190), 'w': 175, 'h': 100},
    {'key': 'add', 'init': '+ 0.0', 'pos': (30, 232), 'w': 120, 'h': 70,
     'props': {'operand': 0.0}},
    {'key': 'c0', 'comment': True, 'text': 'a clean wave with occasional spikes',
     'pos': (30, 310)},
    {'key': 'p0', 'init': 'plot', 'pos': (30, 350), 'w': 208, 'h': 176,
     'props': PLOT(-2.0, 2.0)},
    {'key': 'sc', 'init': 't.smart_clamp_kf', 'pos': (30, 545), 'w': 280, 'h': 220},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 785), 'w': 208, 'h': 176,
     'props': PLOT(-2.0, 2.0)},
    {'key': 'c1', 'comment': True, 'text': 'the spike cannot move it far in one frame',
     'pos': (30, 970)},
    {'key': 'c2', 'comment': True, 'text': 'but a run of large steps still tracks',
     'pos': (30, 1000)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('tog', '', 'met', 'on'), ('met', '', 'rnd', 'trigger'),
         ('sig', '', 'add', 'in'), ('rnd', 'out', 'add', 'operand'),
         ('add', 'result', 'p0', 'y'),
         ('add', 'result', 'sc', 'input'), ('sc', 'filtered', 'p1', 'y')]
print(build('t.smart_clamp_kf', 't Kalman filters - glitch or fast movement?', body,
            demo, links, demo_width=560, text_width=820, text_height=820))

# --------------------------------------------------------------------- t.ESEKF
body = """The error-state extended Kalman filters: the most careful option here.

THE NODES:

t.ESEKF       for ordinary numeric streams
t.quat_ESEKF  for quaternions

WHAT 'ERROR-STATE' MEANS AND WHY IT HELPS:
An ordinary Kalman filter tracks the quantity itself. An error-state filter
tracks the SMALL DIFFERENCE between its prediction and reality, and applies that
correction to the prediction afterwards.

For rotations this is the difference between working and not. A rotation lives
on a curved space where addition is meaningless, but a small error near the
current orientation behaves almost like an ordinary vector - so the filter's
arithmetic is valid, and it stays valid at any orientation. The result is
rotationally invariant: the filter behaves the same whichever way the body
happens to be facing.

That last property is what you want when a filter's behaviour must not depend
on which direction the performer is standing.

THE TRISTATE BLEND:
Both of these run damping and responsive modes and blend continuously between
them, with a third path for rejecting acceleration that is not physically
plausible. The 'alphas' outlet reports all three, so you can see which regime
the filter is in at any moment.

Continuous blending rather than switching is deliberate and matters: a filter
that flips modes introduces a step at every flip, replacing the artefact you
were removing with one of its own.
""" + PARALLEL + QUAT + """
SYNTAX:
t.ESEKF
t.quat_ESEKF

EXAMPLE:
t.quat_ESEKF

INPUTS and PARAMETERS:

input / quaternions:
The streams.

dt (sec):
Time between frames. Everything the filter reasons about scales with this.

Blending Speed:
How quickly it moves between damping and responsive behaviour.

Acceleration Rejection:
The threshold beyond which a change is treated as implausible rather than fast.

Damping Mode / Responsive Mode:
The two regimes' settings.

reset:
Clear the state.

OUTPUTS: 

filtered:
The result.

alphas (damp, resp, err):
The blend between the three regimes. This is the diagnostic outlet - plot it
while tuning, because the numbers on the node tell you what you asked for and
this tells you what the filter is actually doing.

CHOOSING BETWEEN THESE AND THE SIMPLER ONES:
These are the most principled and the most to set up. When a smart-clamped
filter removes your glitches without hurting the movement, use that. Reach for
these when the simpler filters are forcing a choice you do not want to make, or
when rotational invariance genuinely matters - which it does as soon as a
filter's behaviour differs depending on which way the performer is facing."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'aj', 'init': 'active_joints', 'pos': (30, 400), 'w': 220, 'h': 90},
    {'key': 'ek', 'init': 't.quat_ESEKF', 'pos': (30, 505), 'w': 300, 'h': 260},
    {'key': 'c0', 'comment': True, 'text': 'a whole pose in, a whole pose out',
     'pos': (30, 780)},
    {'key': 'qd', 'init': 'quaternion_diff_and_axis', 'pos': (30, 820), 'w': 280, 'h': 180},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 1015), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 0.2)},
    {'key': 'c1', 'comment': True, 'text': 'per-joint speed, before and after',
     'pos': (30, 1200)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (360, 505), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 3,
               'min y': 0.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c2', 'comment': True, 'text': 'the alphas: damp, responsive, error',
     'pos': (360, 665)},
    {'key': 'c3', 'comment': True, 'text': 'this says what the filter is doing -',
     'pos': (360, 695)},
    {'key': 'c4', 'comment': True, 'text': 'the settings only say what you asked',
     'pos': (360, 725)},
]
links = [('sh', 'body 1 quaternions', 'aj', 'full pose quats in'),
         ('aj', 'active joint quats out', 'ek', 'quaternions'),
         ('ek', 'filtered', 'qd', 'quaternions in'),
         ('qd', 'magnitudes', 'p1', 'y'),
         ('ek', 'alphas (damp, resp, err)', 'hm', 'y')]
print(build('t.ESEKF', 't.ESEKF - error-state, rotationally invariant', body, demo,
            links, demo_width=600, text_width=810, text_height=800))

# ---------------------------------------------------------------- t.filter_bank
body = """These split motion into frequency bands - what is happening slowly, and what quickly.

THE NODES:

t.filter_bank         a bank of filters across a frequency range
t.rotation_band_diff  the same idea applied to rotations, per joint

WHY BANDS RATHER THAN ONE FILTER:
A body does several things at once at different rates: a slow shift of weight, a
gesture over a second, a tremor, and sensor noise on top. A single filter forces
you to choose one boundary and throw away everything on the wrong side of it.

A bank keeps them all, separated. Then you can ask which band the movement is
in, rather than only how much movement there is - and "this person is trembling"
and "this person is swaying" become different numbers instead of the same one.

t.rotation_band_diff IS THE ONE BUILT FOR BODIES:
It takes per-joint rotations and returns the rotation DIFFERENCE in each band,
shaped [joints, bands, 3]. So for every joint you get its movement broken down
by timescale, all in one tensor.

That shape is the point. A single node turns a pose stream into a
joint-by-timescale picture that numpy and torch operations can then work on
directly - sum over bands for total effort, take one band for a tremor measure,
compare bands to ask whether a joint is doing something slow or something fast.

SAMPLE FREQUENCY MUST BE RIGHT:
Both take a sample frequency, and the band edges mean nothing without it. Data
arriving at 100 Hz through a node configured for 60 puts every band in the wrong
place while producing entirely plausible-looking numbers.

If your source has been upsampled or delivered at a different rate than it was
captured, that is worth checking before anything else - see the cadence_filter
help patch, which is about exactly that problem in Shadow data.
""" + PARALLEL + """
SYNTAX:
t.filter_bank
t.rotation_band_diff

EXAMPLE:
t.rotation_band_diff

INPUTS and PARAMETERS:

signal / rotations:
The streams.

input format (t.rotation_band_diff):
Whether the rotations arrive as axis-angle or quaternions.

sample freq:
How many frames a second. Get this right first.

number of bands:
How finely to divide the range.

filter type / filter design / order (t.filter_bank):
Bandpass, lowpass and so on; butter or chebyshev; and how steeply each band cuts.

reset:
Clear the filter state.

OUTPUTS: 

filtered:
One filtered signal per band.

banded rotation diffs:
Shaped [joints, bands, 3] - every joint's movement, broken down by timescale.

RELATED:
diff_filter does a cheaper version of this with one-pole filters at several
degrees, separating by timescale rather than by frequency proper.
band_pass and spectrum do it for a single signal."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'aj', 'init': 'active_joints', 'pos': (30, 400), 'w': 220, 'h': 90},
    {'key': 'rb', 'init': 't.rotation_band_diff', 'pos': (30, 505), 'w': 300, 'h': 240},
    {'key': 'c0', 'comment': True, 'text': 'set sample freq to match the source',
     'pos': (30, 760)},
    {'key': 'c1', 'comment': True, 'text': 'out: [joints, bands, 3]', 'pos': (30, 790)},
    {'key': 'nm', 'init': 'np.linalg.norm', 'pos': (30, 830), 'w': 220, 'h': 90},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 935), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 8,
               'min y': 0.0, 'max y': 0.2, 'update_mode': 'heat_map',
               'number format': '%.3f'}},
    {'key': 'c2', 'comment': True, 'text': 'how much movement in each band -',
     'pos': (30, 1095)},
    {'key': 'c3', 'comment': True, 'text': 'swaying and trembling become',
     'pos': (30, 1125)},
    {'key': 'c4', 'comment': True, 'text': 'different numbers, not the same one',
     'pos': (30, 1155)},
]
links = [('sh', 'body 1 quaternions', 'aj', 'full pose quats in'),
         ('aj', 'active joint quats out', 'rb', 'rotations'),
         ('rb', 'banded rotation diffs', 'nm', 'input'),
         ('nm', 'norm', 'hm', 'y')]
print(build('t.filter_bank', 't.filter_bank - movement split by timescale', body,
            demo, links, demo_width=600, text_width=800, text_height=760))

# ------------------------------------------------------------- t.sav_gol_filter
body = """The Savitzky-Golay filter smooths by fitting a curve, not by averaging.

Over a sliding window it fits a low-order polynomial to the data and takes the
value of that polynomial as the smoothed result. That sounds like a detail and
changes the character completely.

WHAT AVERAGING COSTS AND THIS DOES NOT:
A moving average flattens peaks. It has to: the average of a peak and its
lower neighbours is lower than the peak. Smooth a movement enough to remove the
noise and every sharp gesture is rounded off with it - the timing survives, the
shape does not.

A polynomial fit follows the shape. A peak fits a curve with a peak in it, so
the height and position of the extreme are largely preserved while the noise
around them is not. For anything where the SHAPE of a gesture matters - its
sharpness, its peak value, when exactly it turned around - this is the
difference between a usable smoothed signal and a smeared one.

The trade is that it assumes the data is locally polynomial. Where it is not -
a genuine step, a discontinuity - the fit overshoots and rings on either side.

THE TWO SETTINGS INTERACT:
'window length' is how many samples the fit spans; 'polynomial order' is the
degree of curve fitted. A higher order over the same window follows the data
more closely and smooths less; a longer window at the same order smooths more.

Order 2 or 3 over a modest window is the usual starting point. An order close to
the window length fits the noise as well as the signal and does nothing at all.
""" + PARALLEL + """
SYNTAX:
t.sav_gol_filter

EXAMPLE:
t.sav_gol_filter

INPUTS and PARAMETERS:

signal:
The streams, as a 2D tensor of shape [streams, components] - ONE time-step for
however many parallel streams you are filtering. A single value must be shaped
[1, 1] rather than sent as a bare number.

This node checks and refuses anything else, printing to the console rather than
raising - so a wrongly shaped input produces no output at all and no visible
error in the patch. If it appears to do nothing, check the shape first.

window length:
How many samples the fit spans. Longer is smoother and laggier.

polynomial order:
The degree of the fitted curve. Must be less than the window length.

normalize output:
Whether to rescale the result.

reset:
Clear the buffer.

OUTPUTS: 

filtered:
The smoothed streams.

WHEN TO USE THIS RATHER THAN filter OR one_euro_filter:
Use this when peak shape matters and the signal is reasonably smooth between
peaks - a velocity trace you are going to find maxima in, a force curve whose
sharpness is the measurement.

Use one_euro_filter when the trade between lag and jitter is the problem and you
want it resolved automatically. Use the plain filter node when you just want
something quieter and do not mind the rounding."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 3.0 triangle', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('triangle', 3.0, 0.8)},
    {'key': 'tog', 'init': 'toggle', 'pos': (200, 62), 'w': 45, 'h': 42,
     'props': {'': True}},
    {'key': 'met', 'init': 'metro 16', 'pos': (200, 110), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 16.0, 'units': 'milliseconds'}},
    {'key': 'rnd', 'init': 'random.gauss 0.0 0.08', 'pos': (200, 190), 'w': 175, 'h': 100},
    {'key': 'add', 'init': '+ 0.0', 'pos': (30, 232), 'w': 120, 'h': 70,
     'props': {'operand': 0.0}},
    {'key': 'c0', 'comment': True, 'text': 'sharp corners, buried in noise',
     'pos': (30, 310)},
    # sav_gol wants [streams, components]; a bare float is refused
    {'key': 'rs', 'init': 't.reshape 1 1', 'pos': (30, 350), 'w': 180, 'h': 90},
    {'key': 'sg', 'init': 't.sav_gol_filter', 'pos': (30, 450), 'w': 260, 'h': 200},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 670), 'w': 208, 'h': 176,
     'props': PLOT(-1.0, 1.0)},
    {'key': 'c1', 'comment': True, 'text': 'the corners survive', 'pos': (30, 855)},
    {'key': 'flt', 'init': 'filter 0.9', 'pos': (330, 350), 'w': 160, 'h': 70,
     'props': {'degree': 0.9}},
    {'key': 'p2', 'init': 'plot', 'pos': (330, 570), 'w': 208, 'h': 176,
     'props': PLOT(-1.0, 1.0)},
    {'key': 'c2', 'comment': True, 'text': 'a moving average, as quiet -',
     'pos': (330, 755)},
    {'key': 'c3', 'comment': True, 'text': 'but the corners are rounded off',
     'pos': (330, 785)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('tog', '', 'met', 'on'), ('met', '', 'rnd', 'trigger'),
         ('sig', '', 'add', 'in'), ('rnd', 'out', 'add', 'operand'),
         ('add', 'result', 'rs', 'tensor in'),
         ('rs', 'output', 'sg', 'signal'), ('sg', 'filtered', 'p1', 'y'),
         ('add', 'result', 'flt', 'in'), ('flt', 'out', 'p2', 'y')]
print(build('t.sav_gol_filter', 't.sav_gol_filter - smoothing that keeps the shape',
            body, demo, links, demo_width=580, text_width=800, text_height=740))
