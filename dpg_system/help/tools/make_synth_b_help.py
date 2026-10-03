"""adsr~ and ramp~, clock~, delay~, the nonlinear nodes, the mapping nodes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ---------------------------------------------------------- adsr~ / ramp~
body = """These make shapes in time, at audio rate rather than frame rate.

THE NODES:

adsr~   an envelope generator, with both a gate and a one-shot trigger
ramp~   a straight line to a target over a set time
line~   the same node

GATE VERSUS TRIGGER:
adsr~ has both, and they are different gestures. The 'gate' inlet SUSTAINS:
hold it up and the envelope goes attack, decay, then sits at sustain until it
is let go, at which point it releases. The 'trigger' inlet is a one-shot -
attack, decay, then straight on into release, with nothing held. On a
trigger, sustain is just the level the two falling stages meet at: set it to
0 for a plain percussive shape.

Tick the gate by hand, send it 0 and 1 from the patch, or drive it from a sig~
carrying thresholded effort - so that moving past a threshold holds the note
and dropping back releases it. Click the trigger, send it a bang, or patch a
signal into it - a clock~ 'trigger' fires it exactly on the sample. Gate and
trigger edges are both acted on at the sample they happen, not at the next
block.

ramp~ NEVER STEPS:
Send a value to 'target' and the output leaves where it currently is and
arrives at the new value exactly 'time' seconds later. Re-aim it mid-move and
it starts a fresh line from wherever it had got to. That is what makes it safe
to feed a stream of targets - it can never jump, however often the target
changes. 'time' is read when a move begins, so changing it affects the next
move, not the one in flight.

Set the 'time source' option to 'measured' and each move takes as long as the
gap between values arriving: a stream of effort data at 60 frames a second
becomes a continuous signal that reaches each value just as the next one
lands. 'stretch' scales that measured time; a little over 1 is best, since
arriving early means sitting still until the next value - the steps you were
trying to be rid of.

SYNTAX:
adsr~ <attack> <decay> <sustain> <release>
ramp~ <time> <starting value>

EXAMPLE:
ramp~ 0.5

All the arguments are optional. Times are in seconds; sustain is a level from
0 to 1. adsr~ defaults to 0.01, 0.1, 0.7, 0.3, and ramp~ to 0.1 seconds
starting from 0.

INPUTS and PARAMETERS (adsr~):

gate:
Hold it up to sustain. Up means at or over the 'gate threshold' option
(default 0.5).

trigger:
Fire the whole shape once.

attack / decay / sustain / release:
The four stages. Attack, decay and release are times in seconds; sustain is a
level.

enable:
Untick to fade out and stop.

options:
retrigger (on: a new gate restarts the attack from wherever the level is;
off: a new gate is ignored until the envelope has finished), legato (a new
gate during attack, decay or sustain is ignored), gate threshold.

INPUTS and PARAMETERS (ramp~):

target:
Where to go.

time:
How long to take, in seconds.

trigger:
Run the move again from where the output now is, towards the target. 'done'
fires 'time' seconds later.

bypass:
Pass the target straight through, unsmoothed.

options:
jump to target (arrive at once), time source (manual or measured), stretch.

stream:
The measured rate of the incoming values, in 'measured' mode.

OUTPUTS:

signal:
The envelope or ramp, as an audio signal.

done:
A bang when the envelope's release reaches silence, or when the ramp arrives -
use it to chain one into the next.

RELATED:
clock~, sig~, vca~, lfo~, shaper~"""

demo = [
    {'key': 'ad', 'init': 'adsr~', 'pos': (30, 62), 'w': 220, 'h': 220},
    {'key': 'c0', 'comment': True,
     'text': 'click trigger for one note\ntick gate to hold it, untick to release',
     'pos': (30, 295)},
    {'key': 'rp', 'init': 'ramp~ 0.5', 'pos': (30, 350), 'w': 220, 'h': 180},
    {'key': 'c1', 'comment': True,
     'text': 'set target to 1: up an octave in half a second',
     'pos': (30, 545)},
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (30, 585), 'w': 220, 'h': 200},
    {'key': 'vca', 'init': 'vca~', 'pos': (30, 790), 'w': 220, 'h': 160},
    {'key': 'sc', 'init': 'scope~', 'pos': (30, 945), 'w': 260, 'h': 220},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 1220), 'w': 220, 'h': 220},
    {'key': 'c2', 'comment': True, 'text': 'the envelope shapes each note',
     'pos': (30, 1545)},
]
links = [('rp', 'signal', 'vco', 'pitch'),
         ('vco', 'left out', 'vca', 'left in'),
         ('ad', 'signal', 'vca', 'gain'),
         ('vca', 'left out', 'sc', 'in'),
         ('vca', 'left out', 'fo', 'left')]
print(build('adsr~', 'adsr~ and ramp~ - shapes in time, at audio rate', body,
            demo, links, demo_width=420, text_width=810, text_height=760))

# ------------------------------------------------------------- clock~ / metro~
body = """clock~ is a master clock: a pulse train for the audio graph, and bangs for
the patch, from one phase so the two never drift apart.

THE NODES:

clock~  a clock, as a sample-accurate signal and as ordinary bangs
metro~  the same node

clock~ IS EXACT:
Its 'trigger' outlet is a signal whose rising edge is accurate to the sample,
so patching it into an adsr~ trigger (or a modal~ strike) fires it with no
block quantization - which is audible as a loose feel when timing comes
through the ordinary node world. The 'bang' outlet is the same clock as
ordinary messages, for sequencers and counters that do not need that
precision. A 20 Hz clock still delivers 20 bangs a second; they arrive one or
two per frame.

STARTING AND STOPPING:
Nothing happens until 'run' is ticked. Starting puts the clock on a downbeat
and ticks at once rather than waiting out a period. Stopping holds the phase
where it was, so stopping and starting without a reset resumes mid-bar.

RATE:
The 'units' option reads the rate knob as hz, bpm, a period in ms, or a period
in seconds. A signal patched into 'rate' is always in hz and adds to the knob,
scaled by the 'rate depth' option - patch an lfo~ or an envelope there for
accelerando and rubato. The phase runs per sample, so a sweep is smooth.

AFTER A STALL:
If the patch stalls - a load, a heavy node - only the most recent 32 bangs of
the backlog are sent; the rest are not heard. 'count' still advances by the
whole backlog, so a sequencer stays in the right bar.

SYNTAX:
clock~ <rate> <units>

EXAMPLE:
clock~ 120 bpm

Both arguments are optional; the default is 2 hz.

INPUTS and PARAMETERS:

run:
Whether it is running. Off at first.

rate:
How fast, in the units the 'units' option names.

pulse width:
How long each pulse stays up, as a fraction of the period (default 0.5).

reset:
Back to the downbeat, ticking immediately. Click it or patch a signal.

enable:
Untick to stop it rendering altogether.

options:
units (hz, bpm, ms, seconds), rate depth.

OUTPUTS:

trigger:
The sample-accurate pulse, as an audio signal.

bang:
A bang on every tick.

count:
Which tick this is, counting from when the node was made.

RELATED:
adsr~, lfo~, phasor~, modal~"""

demo = [
    {'key': 'ck', 'init': 'clock~ 2', 'pos': (30, 62), 'w': 220, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'tick run: two beats a second', 'pos': (30, 272)},
    {'key': 'i1', 'init': 'int', 'pos': (30, 315), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 'count: which beat this is', 'pos': (30, 370)},
    {'key': 'ad', 'init': 'adsr~ 0.005 0.15 0 0.2', 'pos': (30, 410), 'w': 220, 'h': 220},
    {'key': 'c2', 'comment': True, 'text': 'the clock trigger is exact to the sample',
     'pos': (30, 640)},
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (30, 680), 'w': 220, 'h': 200},
    {'key': 'vca', 'init': 'vca~', 'pos': (30, 895), 'w': 220, 'h': 160},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 1070), 'w': 220, 'h': 220},
    {'key': 'c3', 'comment': True, 'text': 'raise the fader to hear it', 'pos': (30, 1305)},
]
links = [('ck', 'count', 'i1', ''),
         ('ck', 'trigger', 'ad', 'trigger'),
         ('vco', 'left out', 'vca', 'left in'),
         ('ad', 'signal', 'vca', 'gain'),
         ('vca', 'left out', 'fo', 'left')]
print(build('clock~', 'clock~ - a beat exact to the sample', body, demo, links,
            demo_width=420, text_width=810, text_height=760))

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
