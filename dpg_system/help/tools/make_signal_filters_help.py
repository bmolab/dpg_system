"""multi_filter, band_pass, adaptive_filter, physics_filter, kalman_filter."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# --------------------------------------------------------------- multi_filter
body = """These nodes run several smoothing filters on the same signal at once, 
each at a different degree, and hand you all the results together.

Every one of them is the same one-pole filter as the filter node - move part of 
the way towards the new value each time - just several of them side by side. 
You give the degrees as arguments, and you get back a NumPy array with one 
element per filter.

The interesting one is diff_filter. Instead of the smoothed values it sends the 
DIFFERENCES between neighbouring filters: the fast one minus the slower one 
next to it, and so on. Each difference isolates the movement that happens 
between the two timescales - motion too quick for the slow filter to follow, 
but slow enough that the fast one keeps up. 

That is a band, in time rather than in frequency, and it is a cheap way to ask 
"is this signal moving on a scale of a tenth of a second, or a second, 
or ten seconds?" without doing any frequency analysis at all.

THE NODES:

multi_filter        the smoothed values, one per degree
diff_filter         the differences between neighbouring smoothed values
diff_filter_bank    the same as diff_filter

Note that with N degrees, multi_filter sends N values but diff_filter sends 
N minus 1 - there is one fewer gap than there are filters.

SYNTAX:
multi_filter <degree> <degree> ...
diff_filter <degree> <degree> ...

EXAMPLE:
diff_filter 0.5 0.9 0.99

INPUTS and PARAMETERS:

in:
The value to filter. Receiving data here triggers the node. 
A single number: these nodes spread one signal across many filters, 
they do not filter an array element by element.

filter 0, filter 1, ...:
One inlet per filter, holding that filter's degree, from 0.0 to 1.0 - 
the same meaning as on the filter node. Higher is smoother and laggier. 
Give them in increasing order so the differences come out in a sensible 
fast-to-slow sequence.

MESSAGES:

set <value> <value> ...
Forces the filters' internal values, one per filter.

clear
Sets every filter back to zero.

OUTPUTS: 

out:
A NumPy array - the smoothed values for multi_filter, 
the differences between neighbours for diff_filter."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 2.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 2.0)},
    {'key': 'c0', 'comment': True, 'text': 'one signal in', 'pos': (30, 215)},
    {'key': 'mf', 'init': 'multi_filter 0.5 0.9 0.99', 'pos': (30, 250), 'w': 190, 'h': 120,
     'props': {'filter 0': 0.5, 'filter 1': 0.9, 'filter 2': 0.99}},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 390), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 3,
               'min y': -1.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.3f'}},
    {'key': 'c1', 'comment': True, 'text': 'three timescales at once', 'pos': (30, 550)},
    {'key': 'df', 'init': 'diff_filter 0.5 0.9 0.99', 'pos': (30, 590), 'w': 190, 'h': 120,
     'props': {'filter 0': 0.5, 'filter 1': 0.9, 'filter 2': 0.99}},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (30, 730), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 2,
               'min y': -0.5, 'max y': 0.5, 'update_mode': 'heat_map',
               'number format': '%.3f'}},
    {'key': 'c2', 'comment': True, 'text': 'two gaps between three filters',
     'pos': (30, 890)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'mf', 'in'), ('mf', 'out', 'hm', 'y'),
         ('sig', '', 'df', 'in'), ('df', 'out', 'hm2', 'y')]
print(build('multi_filter', 'multi_filter - many timescales at once', body,
            demo, links, demo_width=430, text_width=810, text_height=700))

# ------------------------------------------------------------------ band_pass
body = """These nodes select parts of a signal by how FAST they wiggle, rather than how big they are.

A slow drift and a fast tremor can sit on top of each other in the same stream, 
at the same amplitude. No threshold separates them, and no amount of smoothing 
separates them cleanly either. What tells them apart is frequency, and these 
three nodes are the frequency tools.

They are true digital filters - Butterworth or Chebyshev designs, running at an 
order you choose. Because such a filter has to know how fast samples arrive, 
you must tell it the sample frequency, and getting that wrong makes every other 
setting wrong with it.

THE NODES:

band_pass    one filter, whose type you choose: bandpass, lowpass, 
             highpass or bandstop
filter_bank  a row of bandpass filters spread across a range, sending the 
             filtered signal from each band
spectrum     the same row of bands, but reporting how much ENERGY is in each - 
             a running picture of where the movement is

Use band_pass to isolate or remove one range. Use filter_bank when you want to 
keep working with the separated signals. Use spectrum when you only want to 
know where the activity is - it is the one to reach for when asking whether a 
movement is a slow sway or a fast shake.

SYNTAX:
band_pass
filter_bank
spectrum

INPUTS and PARAMETERS:

signal:
The value to filter. Receiving data here triggers the node. 
A single number per frame - these build their picture over time, from a stream.

sample freq:
How many values arrive per second. Default 60. 
This is the setting to get right first: the filter's idea of what "10 Hz" means 
comes entirely from this number. If your data arrives at 100 frames a second and 
this says 60, every band is in the wrong place.

low / high:
The edges of the range, in Hz. On band_pass these are the two edges of the one 
filter; on filter_bank and spectrum they are the ends of the whole spread of 
bands, which are spaced logarithmically between them.

band count (filter_bank and spectrum):
How many bands to divide the range into. Default 8.

filter type (band_pass only):
bandpass keeps the range, bandstop removes it, 
lowpass keeps everything below high, highpass keeps everything above.

filter design:
butter is smooth and well behaved and is the sensible default. 
cheby1 and cheby2 cut off more sharply at the cost of ripple.

order:
How steeply the filter cuts off, 1 to 8. Higher is sharper, and also rings more 
and reacts more slowly. Raise it only if a lower order is genuinely not 
separating what you need.

OUTPUTS: 

filtered:
band_pass sends the filtered signal. filter_bank sends a NumPy array with one 
filtered signal per band.

spectrum:
A NumPy array with one energy value per band.

A LIMIT WORTH KNOWING:
No filter can work above half the sample frequency - the Nyquist limit. 
At 60 frames a second that is 30 Hz, and the nodes clamp a high setting to stay 
below it. Anything faster than that was never really in your data to begin with."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 4.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 4.0, 1.0)},
    {'key': 'sig2', 'init': 'signal 0.15 sin', 'pos': (200, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 0.15, 0.3)},
    {'key': 'ca', 'comment': True, 'text': 'slow sway', 'pos': (30, 215)},
    {'key': 'cb', 'comment': True, 'text': 'fast tremor on top', 'pos': (200, 215)},
    {'key': 'add', 'init': '+ 0.0', 'pos': (30, 255), 'w': 120, 'h': 70,
     'props': {'operand': 0.0}},
    {'key': 'p0', 'init': 'plot', 'pos': (30, 340), 'w': 208, 'h': 176,
     'props': PLOT(-1.5, 1.5)},
    {'key': 'c0', 'comment': True, 'text': 'the two mixed together', 'pos': (30, 525)},
    {'key': 'bp', 'init': 'band_pass', 'pos': (30, 565), 'w': 190, 'h': 180,
     'props': {'filter type': 'lowpass', 'filter design': 'butter', 'order': 5,
               'low': 0.2, 'high': 1.0, 'sample freq': 60.0}},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 765), 'w': 208, 'h': 176,
     'props': PLOT(-1.5, 1.5)},
    {'key': 'c1', 'comment': True, 'text': 'lowpass: the sway survives, the tremor goes',
     'pos': (30, 950)},
    {'key': 'c2', 'comment': True, 'text': 'switch filter type to highpass to swap them',
     'pos': (30, 980)},
    {'key': 'spec', 'init': 'spectrum', 'pos': (280, 565), 'w': 190, 'h': 180,
     'props': {'band count': 8, 'filter design': 'butter', 'order': 5,
               'low': 1.0, 'high': 20.0, 'sample freq': 60.0}},
    {'key': 'hm', 'init': 'heat_map', 'pos': (280, 765), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 8,
               'min y': 0.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c3', 'comment': True, 'text': 'where the energy sits, band by band',
     'pos': (280, 925)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'), ('tt', '1', 'sig2', 'on'),
         ('sig', '', 'add', 'in'), ('sig2', '', 'add', 'operand'),
         ('add', 'result', 'p0', 'y'),
         ('add', 'result', 'bp', 'signal'), ('bp', 'filtered', 'p1', 'y'),
         ('add', 'result', 'spec', 'signal'), ('spec', 'spectrum', 'hm', 'y')]
print(build('band_pass', 'band_pass - select by frequency, not by size', body,
            demo, links, demo_width=520, text_width=830, text_height=760))

# ------------------------------------------------------------- adaptive_filter
body = """These filters vary their own smoothing according to how fast the signal is moving.

A plain filter forces one choice on you: smooth enough to kill the noise, or 
responsive enough to keep up with real movement. You cannot have both, because 
the filter cannot tell the difference between the two.

These can, using one simple assumption: when the signal is barely moving, 
whatever movement there is is probably noise, so smooth hard; when it is moving 
quickly, that is probably real, so get out of the way. The result follows fast 
gestures closely and sits still when nothing is happening.

THE NODES:

adaptive_filter             for ordinary numbers and arrays
adaptive_quaternion_filter  the same idea for quaternions, where the 
                            "distance" between two values is an angle
one_euro_filter             the published One Euro filter, the standard 
                            version of this idea

adaptive_filter gives you the most control and reports the smoothing it chose. 
one_euro_filter has fewer knobs and a well-documented tuning procedure. 
If you are new to these, start with one_euro_filter.

SYNTAX:
adaptive_filter <power: float>
one_euro_filter <min_cutoff> <beta> <d_cutoff>

EXAMPLE:
one_euro_filter 1.0 0.5 1.0

INPUTS and PARAMETERS - adaptive_filter:

in:
The value to filter. Receiving data here triggers the node.

power:
How sharply the filter reacts to speed. 
Low values behave almost like a plain filter; high values snap open on the 
slightest movement. Default 2.0.

responsiveness:
The base smoothing used when the signal is still, from 0 to 1. 
This is the "how calm is calm" end of the trade.

signal range:
How large a movement counts as fast. Set it to roughly the size of the 
excursions you care about; the node judges speed relative to this.

smooth response / offset response:
How quickly the filter's own smoothing setting is allowed to change. 
Raise them if you can hear or see the filter switching character abruptly.

reset response:
A button that clears the adaption and starts again.

INPUTS and PARAMETERS - one_euro_filter:

input:
The value to filter. Receiving data here triggers the node.

min_cutoff:
The smoothing used when the signal is still. LOWER means smoother. 
This is the first thing to tune: set beta to 0, then lower min_cutoff until 
the signal is still when you are.

beta:
How much speed opens the filter up. Tune this second: raise it until fast 
movements stop lagging. Higher means more responsive when moving.

d_cutoff:
Smoothing applied to the filter's own speed estimate. 
The default of 1.0 is almost always right; leave it alone unless the filter is 
reacting erratically to jittery input.

dt:
The time between samples, in seconds. Default 1/60. 
It must match your real frame rate or the speed estimate is wrong and beta 
means something other than what you set.

OUTPUTS: 

out:
The filtered value.

degree out (adaptive_filter only):
The smoothing degree the filter chose on this frame. 
Worth plotting while tuning - it shows you directly when the filter is opening 
up and when it is clamping down."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 5.0 square', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('square', 5.0, 0.7)},
    {'key': 'tog', 'init': 'toggle', 'pos': (200, 62), 'w': 45, 'h': 42, 'props': {'': True}},
    {'key': 'met', 'init': 'metro 16', 'pos': (200, 112), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 16.0, 'units': 'milliseconds'}},
    {'key': 'rnd', 'init': 'random.gauss 0.0 0.1', 'pos': (200, 192), 'w': 175, 'h': 100},
    {'key': 'add', 'init': '+ 0.0', 'pos': (30, 232), 'w': 120, 'h': 70,
     'props': {'operand': 0.0}},
    {'key': 'c0', 'comment': True, 'text': 'sharp steps buried in noise', 'pos': (30, 312)},
    {'key': 'p0', 'init': 'plot', 'pos': (30, 350), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'oe', 'init': 'one_euro_filter 1.0 0.5 1.0', 'pos': (30, 550), 'w': 190, 'h': 140,
     'props': {'min_cutoff': 1.0, 'beta': 0.5, 'd_cutoff': 1.0, 'dt': 0.0166}},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 710), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c1', 'comment': True, 'text': 'still between steps, quick at the edges',
     'pos': (30, 895)},
    {'key': 'flt', 'init': 'filter 0.9', 'pos': (280, 550), 'w': 130, 'h': 70,
     'props': {'degree': 0.9}},
    {'key': 'p2', 'init': 'plot', 'pos': (280, 710), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c2', 'comment': True, 'text': 'a plain filter for comparison:', 'pos': (280, 895)},
    {'key': 'c3', 'comment': True, 'text': 'as quiet, but the steps are smeared',
     'pos': (280, 925)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('tog', '', 'met', 'on'), ('met', '', 'rnd', 'trigger'),
         ('sig', '', 'add', 'in'), ('rnd', 'out', 'add', 'operand'),
         ('add', 'result', 'p0', 'y'),
         ('add', 'result', 'oe', 'input'), ('oe', 'out', 'p1', 'y'),
         ('add', 'result', 'flt', 'in'), ('flt', 'out', 'p2', 'y')]
print(build('adaptive_filter', 'adaptive_filter - smooth when still, quick when moving',
            body, demo, links, demo_width=520, text_width=820, text_height=780))

# -------------------------------------------------------------- physics_filter
body = """These filters smooth a signal by refusing to let it move in ways a physical object could not.

Instead of averaging, they carry a position, a velocity and an acceleration, and 
move that towards the incoming value under limits you set. A spike in the input 
does not get averaged away - it simply cannot be followed, because reaching it 
would need an impossible acceleration.

That gives a very different character from a plain filter. Smooth output with no 
constant lag: the result tracks the input exactly while the input behaves 
plausibly, and falls behind only when the input does something abrupt. 
For anything driving a physical or apparently-physical thing - a motor, a camera 
move, a rendered object - it usually looks right where an averaging filter looks 
soggy.

THE NODES:

physics_filter   a spring-damper chase with velocity, acceleration and jerk limits
kinetic_filter   a simpler limiter, reporting position, velocity and acceleration 
                 separately

physics_filter is the one to reach for. kinetic_filter is useful when you want 
the derivatives as well as the smoothed value - it hands you all three.

SYNTAX:
physics_filter
kinetic_filter

INPUTS and PARAMETERS - physics_filter:

input:
The target to chase. Receiving data here triggers the node. 
Accepts single numbers, lists, NumPy arrays and PyTorch tensors.

max_vel / max_accel / max_jerk:
The limits, in units per second, per second squared, and per second cubed. 
Velocity caps how fast the output can travel, acceleration caps how fast it can 
change speed, and jerk caps how abruptly it can change acceleration - 
that last one is what removes the visible corners from the motion.

freq:
How stiff the spring is, in Hz. Higher chases harder and arrives sooner.

zeta:
The damping. Below 1 the output overshoots and springs back; at 1 it arrives 
without overshoot; above 1 it eases in slowly. Default 2.5 - firmly damped, 
no bounce.

dt:
Time between samples, in seconds. Default 1/60. The limits above are all 
per-second, so this has to be right for them to mean what they say.

INPUTS and PARAMETERS - kinetic_filter:

in:
The target. Receiving data here triggers the node.

max delta accel:
The largest change in acceleration allowed per step - the jerk limit, 
and the main smoothing control.

max accel / max velocity:
Ceilings on acceleration and speed.

reset:
A button that returns position, velocity and acceleration to zero.

OUTPUTS: 

out (physics_filter):
The smoothed value.

position out / velocity out / accel out (kinetic_filter):
The smoothed value and its first two derivatives. 
The velocity outlet is a much cleaner speed estimate than putting diff after a 
filter, because it is part of the model rather than a difference of noisy 
samples.

TUNING:
Start loose - limits high enough that nothing is constrained - and bring them 
down until the jitter goes. The first limit that changes anything is the one 
doing the work. If the output lags badly, that limit is too tight; if it still 
jitters, it is not tight enough."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 5.0 square', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('square', 5.0, 0.7)},
    {'key': 'c0', 'comment': True, 'text': 'an abrupt step', 'pos': (30, 215)},
    {'key': 'p0', 'init': 'plot', 'pos': (30, 250), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'pf', 'init': 'physics_filter', 'pos': (30, 450), 'w': 190, 'h': 180,
     'props': {'max_vel': 500.0, 'max_accel': 50.0, 'max_jerk': 500.0,
               'freq': 5.0, 'zeta': 2.5, 'dt': 0.0166}},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 650), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c1', 'comment': True, 'text': 'it accelerates, travels, and settles',
     'pos': (30, 835)},
    {'key': 'c2', 'comment': True, 'text': 'drop zeta below 1 to make it overshoot',
     'pos': (30, 865)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'p0', 'y'),
         ('sig', '', 'pf', 'input'), ('pf', 'out', 'p1', 'y')]
print(build('physics_filter', 'physics_filter - smooth by obeying physics', body,
            demo, links, demo_width=430, text_width=820, text_height=780))

# --------------------------------------------------------------- kalman_filter
body = """The kalman_filter node estimates where a signal really is, given that your 
measurements of it are noisy.

It is not an averaging filter. It carries a model of the thing being measured - 
here a value with a velocity and an acceleration - and on every frame it does 
two things: it PREDICTS where that model says the value should be now, then 
CORRECTS that prediction using the measurement that just arrived.

How much it trusts the measurement against its own prediction is the whole 
question, and it works that out for itself from two numbers you supply: how 
unpredictable you think the underlying thing is, and how noisy you think your 
measurements are. Say the measurements are bad and it leans on its model, 
producing a smooth, confident, slightly stubborn estimate. Say they are good 
and it follows them closely.

Because it predicts before it corrects, it does not lag the way an averaging 
filter does. On a signal that really does move smoothly, it can track with 
almost no delay while still rejecting a lot of noise - which no amount of 
tuning will get you from the filter node.

SYNTAX:
kalman_filter

EXAMPLE:
kalman_filter

INPUTS and PARAMETERS:

in:
The measurement. Receiving data here triggers the node. A single number.

process noise:
How much the underlying value is expected to wander on its own, as a 3 by 3 
matrix. Larger values say "this thing genuinely moves unpredictably", 
and the filter becomes more willing to believe the measurements.

measurement noise:
How noisy each reading is, as a single number. 
Larger values say "do not trust any one reading", and the filter leans harder 
on its own prediction - smoother, but slower to accept a real change.

These two are a ratio, not two independent settings. What matters is how big one 
is relative to the other; scaling both changes nothing.

OUTPUTS: 

out:
The estimated value.

kalman gain:
How much weight the filter is currently giving the measurement over its own 
prediction. Near zero it is ignoring your data and running on the model; 
larger and it is following the measurements. 
Watch this while tuning - it tells you which of the two numbers above is 
actually in charge.

WHEN THIS IS THE WRONG NODE:
The model assumes the value moves smoothly, with a velocity and acceleration 
that change gradually. For a signal that genuinely jumps - a switch, a step, a 
category - the model is wrong and the filter will fight the data. 
Reach for physics_filter when you want plausible motion, one_euro_filter when 
you want responsive smoothing without a model, and this when your signal really 
is a smoothly moving quantity that you are measuring badly."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 3.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 3.0, 0.7)},
    {'key': 'tog', 'init': 'toggle', 'pos': (200, 62), 'w': 45, 'h': 42, 'props': {'': True}},
    {'key': 'met', 'init': 'metro 16', 'pos': (200, 112), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 16.0, 'units': 'milliseconds'}},
    {'key': 'rnd', 'init': 'random.gauss 0.0 0.2', 'pos': (200, 192), 'w': 175, 'h': 100},
    {'key': 'add', 'init': '+ 0.0', 'pos': (30, 232), 'w': 120, 'h': 70,
     'props': {'operand': 0.0}},
    {'key': 'c0', 'comment': True, 'text': 'a smooth thing, measured badly', 'pos': (30, 312)},
    {'key': 'p0', 'init': 'plot', 'pos': (30, 350), 'w': 208, 'h': 176,
     'props': PLOT(-1.5, 1.5)},
    {'key': 'kf', 'init': 'kalman_filter', 'pos': (30, 550), 'w': 190, 'h': 100},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 670), 'w': 208, 'h': 176,
     'props': PLOT(-1.5, 1.5)},
    {'key': 'c1', 'comment': True, 'text': 'the estimate, with very little lag',
     'pos': (30, 855)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('tog', '', 'met', 'on'), ('met', '', 'rnd', 'trigger'),
         ('sig', '', 'add', 'in'), ('rnd', 'out', 'add', 'operand'),
         ('add', 'result', 'p0', 'y'),
         ('add', 'result', 'kf', 'in'), ('kf', 'out', 'p1', 'y')]
print(build('kalman_filter', 'kalman_filter - predict, then correct', body,
            demo, links, demo_width=440, text_width=820, text_height=740))
