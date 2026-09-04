"""adsr~ and timing, delay~, the nonlinear nodes, the mapping nodes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ------------------------------------------------------- adsr~ / ramp~ / clock~
body = """These make things happen over time, at audio rate rather than frame rate.

THE NODES:

adsr~   an envelope generator, with both a gate and a one-shot trigger
ramp~   a linear ramp to a target over a set time
line~   the same node
clock~  a master clock: a pulse train for the audio graph, bangs for the patch
metro~  the same node

GATE VERSUS TRIGGER:
adsr~ has both, and they are different gestures. The 'gate' inlet SUSTAINS: 
hold it up and the envelope goes attack, decay, then sits at sustain until it 
is let go, at which point it releases. The 'trigger' inlet is a one-shot - it 
fires the whole shape and lets go by itself.

Tick the gate by hand, send it 0 and 1 from the patch, or drive it from a sig~ 
carrying thresholded effort - so that moving past a threshold holds the note 
and dropping back releases it.

ramp~ NEVER STEPS:
Send a value to 'target' and the output leaves where it currently is and 
arrives at the new value exactly 'time' seconds later. Re-aim it mid-move and 
it starts a fresh line from wherever it had got to. That is what makes it safe 
to feed a stream of targets - it can never jump, however often the target 
changes.

clock~ IS EXACT:
Its 'trigger' outlet is a signal whose rising edge is accurate to the sample, 
so patching it into an adsr~ trigger fires the envelope with no block 
quantization - which is audible as a loose feel when timing comes through the 
ordinary node world. The 'bang' outlet is the same clock as ordinary messages, 
for sequencers and counters that do not need that precision.

SYNTAX:
adsr~
ramp~ <time>
clock~ <rate>

EXAMPLE:
adsr~

INPUTS and PARAMETERS:

gate (adsr~):
Hold it up to sustain.

trigger (adsr~):
Fire the whole shape once.

attack / decay / sustain / release:
The four stages. Attack and decay are times, sustain is a level, release is a 
time.

target / time (ramp~):
Where to go and how long to take.

run / rate / pulse width / reset (clock~):
Whether it is running, how fast, how long each pulse stays up, and a restart.

OUTPUTS: 

signal:
The envelope or ramp, as an audio signal.

done:
Fires when the envelope or ramp finishes - use it to chain one into the next.

trigger / bang / count (clock~):
The sample-accurate pulse, the ordinary bang, and how many have passed."""

demo = [
    {'key': 'ck', 'init': 'clock~ 2', 'pos': (30, 62), 'w': 220, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'tick run: two beats a second', 'pos': (30, 272)},
    {'key': 'ad', 'init': 'adsr~', 'pos': (30, 315), 'w': 220, 'h': 220},
    {'key': 'c1', 'comment': True, 'text': 'the clock trigger is exact to the sample',
     'pos': (30, 545)},
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (300, 315), 'w': 220, 'h': 200},
    {'key': 'vca', 'init': 'vca~', 'pos': (30, 585), 'w': 220, 'h': 160},
    {'key': 'sc', 'init': 'scope~', 'pos': (300, 545), 'w': 260, 'h': 220},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 765), 'w': 220, 'h': 220},
    {'key': 'c2', 'comment': True, 'text': 'the envelope shapes each note',
     'pos': (30, 1000)},
]
links = [('ck', 'trigger', 'ad', 'trigger'),
         ('vco', 'left out', 'vca', 'left in'),
         ('ad', 'signal', 'vca', 'gain'),
         ('vca', 'left out', 'sc', 'in'),
         ('vca', 'left out', 'fo', 'left')]
print(build('adsr~', 'adsr~ - shapes in time, at audio rate', body, demo, links,
            demo_width=590, text_width=810, text_height=760))

# --------------------------------------------------------------------- delay~
body = """delay~ is a delay line with damped feedback and an audio-rate delay time.

The feedback belongs to the node rather than to the patch, and it has to. 
A cord from the outlet back to the inlet is a cycle, and the compiler runs a 
cycle one block late - so the shortest delay a patched feedback loop can make 
is around twelve milliseconds. Everything shorter than that is only reachable 
from inside the node, and everything shorter than that is where the interesting 
sounds are: flanging, comb filtering, the resonance of a short tube.

Because the delay time is an audio-rate inlet rather than a setting, you can 
modulate it with an LFO for chorus and flanging, or with an envelope for a 
pitch-bending sweep as the line lengthens.

'damping' rolls the high end off each time round the loop, which is what makes 
repeats decay the way a real space does rather than ringing forever with the 
same brightness.

SYNTAX:
delay~ <time>
echo~ <time>

EXAMPLE:
delay~ 0.25

INPUTS and PARAMETERS:

left in / right in:
The signal.

time:
The delay, at audio rate. Modulate it.

feedback:
How much of the output goes round again. High values ring for a long time; 
at 1 it never dies.

damping:
How much high end is lost each time round.

freeze:
Holds what is in the line and stops taking new input, so the current contents 
loop indefinitely.

mode:
How the line behaves - the character of the repeats.

OUTPUTS: 

left out / right out:
The delayed signal.

A NOTE ON MODULATING TIME:
Changing the delay time changes the pitch of what is already in the line, 
because the material is being read faster or slower. That is not a defect - 
it is what a flanger is, and what a tape delay does when you push it."""

demo = [
    {'key': 'ck', 'init': 'clock~ 1', 'pos': (30, 62), 'w': 220, 'h': 200},
    {'key': 'ad', 'init': 'adsr~', 'pos': (30, 280), 'w': 220, 'h': 220},
    {'key': 'vco', 'init': 'vco~ 330', 'pos': (300, 62), 'w': 220, 'h': 200},
    {'key': 'vca', 'init': 'vca~', 'pos': (30, 520), 'w': 220, 'h': 160},
    {'key': 'lfo', 'init': 'lfo~ 0.2', 'pos': (300, 280), 'w': 200, 'h': 160},
    {'key': 'dl', 'init': 'delay~ 0.25', 'pos': (30, 700), 'w': 240, 'h': 220},
    {'key': 'c0', 'comment': True, 'text': 'the lfo modulates the delay time\nwhich bends the pitch of the repeats',
     'pos': (30, 930)},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 1000), 'w': 220, 'h': 220},
]
links = [('ck', 'trigger', 'ad', 'trigger'),
         ('vco', 'left out', 'vca', 'left in'), ('ad', 'signal', 'vca', 'gain'),
         ('vca', 'left out', 'dl', 'left in'),
         ('lfo', 'signal', 'dl', 'time'),
         ('dl', 'left out', 'fo', 'left')]
print(build('delay~', 'delay~ - repeats, and the short ones you cannot patch', body,
            demo, links, demo_width=570, text_width=800, text_height=700))

# ------------------------------------------------------------ fold~ and crush~
body = """These add harmonics that were not there, by breaking the signal in various ways.

THE NODES:

fold~      saturation and wavefolding, with the aliasing dealt with
distort~   the same node
crush~     bit depth and sample rate reduction
decimate~  the same node
mult~      multiply two signals together
*~         the same node
ring~      the same node

ALIASING, AND WHY fold~ IS ITS OWN NODE:
Any nonlinearity produces harmonics above the ones it was handed, and the ones 
that land past half the sample rate fold back down as tones unrelated to the 
pitch - and, unlike real harmonics, they do not move when the pitch moves. 
That is the fizz around bright distorted sound. fold~ deals with it. 
shaper~ will apply any curve you can draw, but cannot.

crush~ IS SEPARATE BECAUSE IT IS NOT A CURVE:
Bit reduction is a staircase whose steps are fixed in AMPLITUDE; sample rate 
reduction is a staircase in TIME. Neither is a transfer function, and they 
sound nothing like each other - one grits, the other aliases.

mult~ IS MULTIPLICATION, NOT AMPLIFICATION:
Use it rather than vca~ whenever either signal is bipolar. Two oscillators into 
mult~ is ring modulation: sum and difference frequencies, no original pitches, 
the classic metallic clang. An LFO into mult~ is tremolo that goes through zero 
and out the other side. vca~ would clamp that negative half away.

SYNTAX:
fold~
crush~
mult~

EXAMPLE:
mult~

INPUTS and PARAMETERS:

left in / right in:
The signal.

drive / bias / shape (fold~):
How hard into the nonlinearity, how far off-centre, and which curve. 
Bias matters more than it looks: an asymmetric curve makes even harmonics, 
which is a warmer and more valve-like sound than the odd ones symmetry gives.

bits / rate (crush~):
How many bits to keep, and how far to drop the sample rate.

in 1 / in 2 (mult~):
The two signals to multiply.

OUTPUTS: 

left out / right out / signal:
The result."""

demo = [
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (30, 62), 'w': 220, 'h': 200},
    {'key': 'vco2', 'init': 'vco~ 317', 'pos': (300, 62), 'w': 220, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'two unrelated pitches', 'pos': (30, 272)},
    {'key': 'ml', 'init': 'mult~', 'pos': (30, 315), 'w': 200, 'h': 140},
    {'key': 'c1', 'comment': True, 'text': 'ring modulation: neither pitch survives',
     'pos': (30, 465)},
    {'key': 'fd', 'init': 'fold~', 'pos': (30, 505), 'w': 240, 'h': 200},
    {'key': 'c2', 'comment': True, 'text': 'drive it hard; try bias off centre',
     'pos': (30, 715)},
    {'key': 'sc', 'init': 'scope~', 'pos': (300, 505), 'w': 260, 'h': 220},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 755), 'w': 220, 'h': 220},
]
links = [('vco', 'left out', 'ml', 'in 1'), ('vco2', 'left out', 'ml', 'in 2'),
         ('ml', 'signal', 'fd', 'left in'),
         ('fd', 'left out', 'sc', 'in'), ('fd', 'left out', 'fo', 'left')]
print(build('fold~', 'fold~ and friends - harmonics that were not there', body,
            demo, links, demo_width=590, text_width=810, text_height=720))

# ------------------------------------------------------- shaper~ and scaler~
body = """These map a signal through a curve - one number in, a different number out.

THE NODES:

shaper~     a drawn breakpoint curve, applied to every sample
lookup~     the same node
envelope~   the same node
scaler~     map a range into another range, with a response curve
scale~      the same node

shaper~ IS THE ENVELOPE NODE AT AUDIO RATE:
The ordinary envelope node maps one x to one y per message. shaper~ maps every 
sample of every block through the same kind of drawn curve. Drag a point to 
move it, right-click to add or remove one, shift and left-drag a segment to 
bend it - the same gestures - and the table behind it is rebuilt.

What that means depends on what you feed it. Given an audio signal it is a 
waveshaper, and the curve is a distortion characteristic. Given a slow control 
signal it is a response curve - the way to say "this control should be gentle 
at the bottom and steep at the top" by drawing it rather than calculating it.

Note that shaper~ will apply any curve you draw but does nothing about the 
aliasing that a steep one produces. When you want distortion rather than 
mapping, fold~ is the node that deals with it.

scaler~ IS THE ARITHMETIC CASE:
Take a signal in a known range - an envelope at 0 to 1, an LFO at -1 to 1 - and 
put it into the range something else wants, with a curve on the way. 

Worth knowing before you reach for it: a plain linear range change is already 
available without this node. Every modulation inlet in the system computes 
base plus depth times the incoming signal, so the knob is the low end and the 
inlet's own depth is the span. scaler~ is for when you want a CURVE, or when 
the range has to be set from the patch.

SYNTAX:
shaper~
scaler~

EXAMPLE:
shaper~

INPUTS and PARAMETERS:

in:
The signal to map.

in low / in high:
The range the input is expected to arrive in.

out low / out high (scaler~):
The range to produce.

curve / mode (scaler~):
The response between the two ends.

points / range (shaper~):
The curve's control points, so a shape can be stored or sent, and the vertical 
span it covers.

OUTPUTS: 

signal:
The mapped signal.

points out (shaper~):
The curve's points, for saving or copying to another shaper~."""

demo = [
    {'key': 'lfo', 'init': 'lfo~ 0.5', 'pos': (30, 62), 'w': 200, 'h': 160},
    {'key': 'c0', 'comment': True, 'text': 'a slow triangle, -1 to 1', 'pos': (30, 232)},
    {'key': 'sh', 'init': 'shaper~', 'pos': (30, 275), 'w': 320, 'h': 280},
    {'key': 'c1', 'comment': True, 'text': 'drag the points to change the response\nright-click to add one, shift-drag to bend',
     'pos': (30, 565)},
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (400, 275), 'w': 220, 'h': 200},
    {'key': 'sc', 'init': 'scope~', 'pos': (400, 500), 'w': 260, 'h': 220},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 640), 'w': 220, 'h': 220},
    {'key': 'c3', 'comment': True, 'text': 'the curve now drives the pitch',
     'pos': (30, 875)},
]
links = [('lfo', 'signal', 'sh', 'in'),
         ('sh', 'signal', 'vco', 'pitch'),
         ('vco', 'left out', 'sc', 'in'),
         ('vco', 'left out', 'fo', 'left')]
print(build('shaper~', 'shaper~ - a drawn curve, applied every sample', body,
            demo, links, demo_width=690, text_width=810, text_height=740))
