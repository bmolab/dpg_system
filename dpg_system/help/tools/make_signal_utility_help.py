"""trigger/hysteresis, noise_gate, ranger, register, stream, subsample."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# -------------------------------------------------------------------- trigger
body = """The trigger node fires when a signal rises past a threshold, and again when it falls back.

It has two thresholds, not one, and that is the whole point. 
The signal has to climb above the trigger threshold to turn the node on, 
and then fall below the RELEASE threshold - which you normally set lower - 
before it will turn off again. 
The gap between the two is the dead zone where nothing happens.

Without that gap, a noisy signal sitting right on the threshold would 
rattle on and off many times a second. With it, the signal has to make a real 
excursion before the node changes its mind. This is called hysteresis, 
and the hysteresis node is the same node with the trigger threshold preset 
to 0.2 to make the gap obvious.

Use it to turn a continuous reading - a loudness, a speed, a distance - into a 
clean on/off decision that you can act on.

SYNTAX:
trigger <threshold: float> <release threshold: float>
hysteresis <threshold: float> <release threshold: float>

EXAMPLE:
trigger 0.6 0.3

INPUTS and PARAMETERS:

input:
The value to watch. Receiving data here triggers the node. 
Numbers only - integers and floats, not arrays.

threshold:
The level the input must rise ABOVE to fire. Default 0.1, or 0.2 for hysteresis.

release threshold:
The level the input must fall BELOW to release. Default 0.1. 
Set this lower than the trigger threshold; the distance between them is how 
much noise the node will ignore. Setting them equal removes the protection 
and you are back to a plain comparison.

trigger mode:
"output toggle" sends 1 when it fires and 0 when it releases - a state you can 
use to gate something. 
"output bang" sends a bang from each outlet instead - an event you can use to 
start something.

retrig delay:
A minimum time, in seconds, before the node is allowed to fire again after 
firing. Use it when a single physical event produces several bursts and you 
only want the first.

OUTPUTS: 

out:
In toggle mode, 1 on firing and 0 on release. In bang mode, a bang on firing.

release:
In toggle mode, 1 on release. In bang mode, a bang on release. 
Having the release on its own outlet lets you start one thing when the signal 
arrives and a different thing when it goes away.

RELATED:
togedge reports crossings of zero with no threshold and no dead zone. 
noise_gate silences small values rather than reporting them."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 3.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 3.0, 1.0, False)},
    {'key': 'c0', 'comment': True, 'text': 'a slow rise and fall, 0 to 1', 'pos': (30, 215)},
    {'key': 'trg', 'init': 'trigger 0.6 0.3', 'pos': (30, 250), 'w': 150, 'h': 120,
     'props': {'threshold': 0.6, 'release threshold': 0.3,
               'trigger mode': 'output toggle', 'retrig delay': 0.0}},
    {'key': 'c1', 'comment': True, 'text': 'fires above 0.6, releases below 0.3\nthe gap between them is the dead zone',
     'pos': (30, 380)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 450), 'w': 208, 'h': 176,
     'props': PLOT(-0.2, 1.2)},
    {'key': 'cnt', 'init': 'counter', 'pos': (280, 450), 'w': 123, 'h': 84,
     'props': {'step': 1}},
    {'key': 'i1', 'init': 'int', 'pos': (280, 545), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c3', 'comment': True, 'text': 'counts releases', 'pos': (280, 600)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'trg', 'input'), ('sig', '', 'p1', 'y'),
         ('trg', 'release', 'cnt', 'input'), ('cnt', 'count out', 'i1', '')]
print(build('trigger', 'trigger - fire on the way up, release on the way down',
            body, demo, links, demo_width=430, text_width=800, text_height=700))

# ----------------------------------------------------------------- noise_gate
body = """The noise_gate node silences small values, letting only the ones that matter through.

Anything closer to zero than the threshold comes out as exactly zero. 
Anything beyond it passes. This is how you stop a sensor's idle jitter from 
driving the rest of your patch - the resting wobble becomes a clean, still zero.

It differs from a smoothing filter in an important way. A filter reduces noise 
everywhere, at the cost of lag. A gate leaves the signal completely untouched 
above the threshold and removes it entirely below, with no lag at all. 
When your problem is "it is never quite still", this is the node, not filter.

SYNTAX:
noise_gate <threshold: float>

EXAMPLE:
noise_gate 0.1

INPUTS and PARAMETERS:

input:
The value to gate. Receiving data here triggers the node.

threshold:
How far from zero a value has to be to survive. Default 0.1.

bipolar:
When unchecked, the node only looks at values below the threshold on the 
positive side. When checked, it gates a band on BOTH sides of zero, from minus 
threshold to plus threshold - which is what you want for a signal that swings 
either way, like a velocity.

squeeze:
Changes what happens to the values that DO pass. 
Unchecked, they pass at full size, so the output jumps abruptly from 0 to the 
threshold value the moment the gate opens. 
Checked, the threshold is subtracted from them, so the output starts from zero 
and grows smoothly. Squeeze costs you a little amplitude but removes the step, 
and is usually the better choice when the result drives something continuous.

OUTPUTS: 

out:
The gated value: zero if it was inside the dead band, otherwise the input, 
reduced by the threshold if squeeze is on."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 3.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 3.0, 0.3)},
    {'key': 'c0', 'comment': True, 'text': 'a small signal, mostly near zero',
     'pos': (30, 215)},
    {'key': 'ng', 'init': 'noise_gate 0.15', 'pos': (30, 255), 'w': 150, 'h': 100,
     'props': {'threshold': 0.15, 'bipolar': True, 'squeeze': False}},
    {'key': 'c1', 'comment': True, 'text': 'bipolar on: gates both sides of zero\ntry squeeze to remove the step',
     'pos': (30, 365)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 435), 'w': 208, 'h': 176,
     'props': PLOT(-0.4, 0.4)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'ng', 'input'), ('ng', 'out', 'p1', 'y')]
print(build('noise_gate', 'noise_gate - silence the small stuff', body, demo, links,
            demo_width=400, text_width=790, text_height=620))

# --------------------------------------------------------------------- ranger
body = """The ranger node rescales a signal from the range it arrives in to the range you need.

You tell it what the input's low and high look like, and what you want them to 
become, and it maps proportionally between the two. An input sitting halfway 
between its limits comes out halfway between yours.

It also inverts: set the output minimum above the output maximum and the signal 
comes out backwards, so a rising input produces a falling result.

The useful part is the calibrate switch. Rather than working out a sensor's real 
range by hand, tick calibrate, move through the full range of the thing you are 
measuring, and untick it. The node watches the highest and lowest values it saw 
while calibrating and writes them into the input limits for you.

SYNTAX:
ranger <input_min> <input_max> <output_min> <output_max>

EXAMPLE:
ranger 0.0 1.0 0.0 360.0

INPUTS and PARAMETERS:

in:
The value to rescale. Receiving data here triggers the node.

input_min / input_max:
The range the signal actually arrives in. These are what calibrate fills in.

output_min / output_max:
The range you want it in. Setting output_min above output_max flips the signal.

clamp:
When checked, results are held inside the output range, so an input beyond its 
stated limits cannot push the output past yours. On by default. 
Uncheck it when you want to allow overshoot.

calibrate:
While checked, the node records the smallest and largest values it sees and 
does not change the input limits. When you uncheck it, the recorded low and 
high are written into input_min and input_max. 
Move through the full range you care about before switching it off - the node 
can only learn from what it has seen.

OUTPUTS: 

rescaled:
The mapped value.

A NOTE ON CALIBRATION:
Calibration restarts each time you switch the checkbox on, so a bad pass 
costs nothing - just do it again. Because it records extremes, a single spurious 
spike will widen the range and flatten everything afterwards; if the result 
looks compressed, that is usually why."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 3.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 3.0, 1.0, True)},
    {'key': 'c0', 'comment': True, 'text': 'a wave that swings -1 to 1', 'pos': (30, 215)},
    {'key': 'rng', 'init': 'ranger -1.0 1.0 0.0 360.0', 'pos': (30, 255), 'w': 175, 'h': 180,
     'props': {'input_min': -1.0, 'input_max': 1.0,
               'output_min': 0.0, 'output_max': 360.0, 'clamp': True,
               'calibrate': False}},
    {'key': 'c1', 'comment': True, 'text': 'rescaled to degrees, 0 to 360\nswap output_min and output_max to invert', 'pos': (30, 445)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 515), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 360.0)},
    {'key': 'f1', 'init': 'float', 'pos': (280, 515), 'w': 127, 'h': 42, 'props': FLT},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'rng', 'in'), ('rng', 'rescaled', 'p1', 'y'),
         ('rng', 'rescaled', 'f1', '')]
print(build('ranger', 'ranger - map a signal into the range you need', body,
            demo, links, demo_width=430, text_width=790, text_height=690))

# ------------------------------------------------------------------- register
body = """The register node holds a value and sends it on when you ask.

It has three inlets and they do three different jobs. 
Values arriving at "input" are just watched - nothing is stored and nothing 
comes out. Send anything to "sample" and whatever is currently at the input is 
copied into the register, quietly. Send anything to "trigger" and the current 
input is copied AND sent out.

The separation is the point. It lets you decide when to look and, separately, 
when to speak. A stream can run through the input all day; the register only 
notices when you tell it to.

SYNTAX:
register

EXAMPLE:
register

INPUTS and PARAMETERS:

input:
The value to be captured. This inlet is passive - data arriving here is held 
ready but produces no output, and does not update the stored value.

sample:
Copies the current input into the register without sending anything. 
Use this to capture a value now and send it later.

trigger:
Copies the current input into the register AND sends it out. 
Anything sent here works; only the arrival matters.

OUTPUTS: 

out:
The stored value, sent when triggered. Before anything has been captured, 
the register holds 0.

RELATED:
sample_hold does the opposite division of labour - data arriving at its input 
drives the output continuously, and a switch decides whether the stored value 
follows or freezes. Use sample_hold when you want a continuous stream out, 
and register when you want one value at a moment of your choosing."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42, 'props': {'': True}},
    {'key': 'met', 'init': 'metro 50', 'pos': (30, 112), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 50.0, 'units': 'milliseconds'}},
    {'key': 'rnd', 'init': 'random 100.0', 'pos': (30, 192), 'w': 140, 'h': 80,
     'props': {'range': 100.0, 'bipolar': False}},
    {'key': 'c0', 'comment': True, 'text': 'a stream of changing numbers', 'pos': (30, 280)},
    {'key': 'i0', 'init': 'int', 'pos': (30, 315), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'btn', 'init': 'button', 'pos': (30, 372), 'w': 88, 'h': 46},
    {'key': 'c1', 'comment': True, 'text': 'click to grab one value', 'pos': (30, 425)},
    {'key': 'reg', 'init': 'register', 'pos': (30, 460), 'w': 130, 'h': 100},
    {'key': 'i1', 'init': 'int', 'pos': (30, 575), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'holds still until you click again',
     'pos': (30, 625)},
]
links = [('tog', '', 'met', 'on'), ('met', '', 'rnd', 'trigger'),
         ('rnd', 'out', 'i0', ''), ('rnd', 'out', 'reg', 'input'),
         ('btn', '', 'reg', 'trigger'), ('reg', 'out', 'i1', '')]
print(build('register', 'register - hold a value, send it when asked', body,
            demo, links, demo_width=400, text_width=780, text_height=640))

# --------------------------------------------------------------------- stream
body = """stream resends the most recent value it received, on every frame.

It turns an event into a continuous supply. Something upstream that only
speaks when it changes - a slider, a message, a value that arrives now and
then - becomes a steady stream at the frame rate, about 60 times a second.
You use it when something downstream needs to be fed continuously.

It does not change the value. Arriving at 'input' does not send anything by
itself; the value is stored and goes out on the next frame, and every frame
after, until a new value replaces it.

IT RUNS FROM THE START:
It fires on every frame from the moment it exists, before any value has
arrived. Nodes after it see a trigger with no new value; most ignore that,
but one that reacts to any trigger - a counter, say - starts counting at once.

SYNTAX:
stream

INPUTS and PARAMETERS:

input:
The value to repeat. It does not trigger anything.

stream:
When checked the node repeats on every frame. Unchecked, it stops entirely -
it does not pass values through either. On by default.

OUTPUTS:

out:
The most recent value, every frame.

RELATED:
subsample does the opposite job: it thins a fast stream down.
tick fires every frame without carrying a value."""

demo = [
    {'key': 'f0', 'init': 'float', 'pos': (30, 62), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c0', 'comment': True, 'text': 'drag this now and then', 'pos': (30, 112)},
    {'key': 'st', 'init': 'stream', 'pos': (30, 150), 'w': 140, 'h': 70},
    {'key': 'f1', 'init': 'float', 'pos': (30, 235), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'cnt', 'init': 'counter 1000000', 'pos': (200, 235), 'w': 123, 'h': 84,
     'props': {'count': 1000000, 'step': 1}},
    {'key': 'i1', 'init': 'int', 'pos': (200, 335), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 'counts sends: about 60 a second,\nhowever seldom the value changes\nuntick stream and it all stops',
     'pos': (30, 390)},
]
links = [('f0', 'float out', 'st', 'input'), ('st', 'out', 'f1', ''),
         ('st', 'out', 'cnt', 'input'), ('cnt', 'count out', 'i1', '')]
print(build('stream', 'stream - resend the latest value every frame', body,
            demo, links, demo_width=400, text_width=790, text_height=600))

# ------------------------------------------------------------------ subsample
body = """subsample passes on only every Nth value it receives.

It thins a fast stream down to something slower. You use it when a fast sensor
or a per-frame source is driving something expensive - a network send, a
model, a redraw - more often than it needs.

It does not average or smooth: the values in between are simply dropped. If
you want fewer values without losing the shape, filter first, then subsample.

A NOTE ON ITS TIMING:
subsample also watches how fast values arrive. If the input goes quiet for
longer than the usual gap times the rate, it sends the latest value once more
and starts counting afresh. So a stream that stops does not leave the node
holding a value it never passed on - though if the last value WAS passed on,
it goes out a second time.

At a rate of 1 that check is tight enough that ordinary unevenness in arrival
can trip it, so some values are sent twice; at 0 it trips after every value.
Use 2 or more.

SYNTAX:
subsample <rate: int>

EXAMPLE:
subsample 4

INPUTS and PARAMETERS:

input:
The data. Receiving it triggers the node.

rate:
How many values arrive for each one that leaves. 2 halves the rate, 10 sends
one in ten. 2 by default.

OUTPUTS:

out:
The values that get through.

RELATED:
stream does the opposite job: it repeats a value on every frame.
rate_limiter thins by time rather than by count."""

demo = [
    {'key': 'tk', 'init': 'tick', 'pos': (30, 62), 'w': 33, 'h': 42, 'props': {'': True}},
    {'key': 'cnt', 'init': 'counter 1000000', 'pos': (30, 115), 'w': 123, 'h': 84,
     'props': {'count': 1000000, 'step': 1}},
    {'key': 'c0', 'comment': True, 'text': 'counts every frame', 'pos': (30, 210)},
    {'key': 'i0', 'init': 'int', 'pos': (30, 245), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'sub', 'init': 'subsample 10', 'pos': (30, 305), 'w': 140, 'h': 70,
     'props': {'rate': 10}},
    {'key': 'i1', 'init': 'int', 'pos': (30, 390), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 'one value in ten gets through\ndrag rate and watch it thin out',
     'pos': (30, 440)},
]
links = [('tk', '', 'cnt', 'input'),
         ('cnt', 'count out', 'i0', ''), ('cnt', 'count out', 'sub', 'input'),
         ('sub', 'out', 'i1', '')]
print(build('subsample', 'subsample - pass on every Nth value', body,
            demo, links, demo_width=400, text_width=790, text_height=600))
