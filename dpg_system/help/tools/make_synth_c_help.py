"""clean~, the data bridges, additive~, vst~, shape_modes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# --------------------------------------------------------- clean~ / one_euro~
body = """These clean up a signal before it goes anywhere else.

THE NODES:

clean~      subsonics off the bottom, fizz off the top
condition~  the same node
one_euro~   smoothing that gets out of the way when you move
smooth~     the same node

clean~ IS THE CHANNEL STRIP'S HYGIENE STAGE:
The intended place is source -> fader~ -> clean~ -> place~ -> audio_out~. 
It exists for when a bowed 5 Hz, or a low mode blooming out of a resonator, 
is eating headroom without being music. 24 dB per octave each way, flat and 
resonance-free between, and bypassed it passes the signal untouched.

Physical models are the reason. They produce energy well below what anything 
can reproduce, and that energy is real - it moves the meters and steals the 
loudness - but nobody hears it.

one_euro~ CHOOSES PER SAMPLE:
Any fixed smoothing has to choose between passing jitter and lagging behind a 
gesture, because those are the same setting. This one does not choose once. 
At rest the cutoff drops to 'min cutoff' and the signal settles hard; as the 
signal moves the cutoff opens in proportion to how fast it is moving, so a fast 
gesture arrives on time.

This is the audio-rate version of the one_euro_filter node, and it is what you 
put between effort data and anything that will be heard - because control data 
that steps is audible as zippering, and control data that lags feels dead.

SYNTAX:
clean~
one_euro~

EXAMPLE:
clean~

INPUTS and PARAMETERS:

left in / right in:
The signal.

low cut / high cut (clean~):
Where the two filters act.

min cutoff (one_euro~):
The smoothing when nothing is moving. Lower is calmer. Set this first, with 
beta at zero, until the resting signal is still.

beta (one_euro~):
How much movement opens the filter up. Raise it until fast gestures stop 
lagging.

OUTPUTS: 

left out / right out:
The conditioned signal."""

demo = [
    {'key': 'sl', 'init': 'slider 0.0', 'pos': (30, 62), 'w': 220, 'h': 60,
     'props': {'min': 0.0, 'max': 1.0, 'format': '%.2f', 'width': 200}},
    {'key': 'sg', 'init': 'sig~', 'pos': (30, 135), 'w': 200, 'h': 140},
    {'key': 'oe', 'init': 'one_euro~', 'pos': (30, 290), 'w': 240, 'h': 180},
    {'key': 'c0', 'comment': True, 'text': 'still at rest, quick when you move',
     'pos': (30, 480)},
    {'key': 'sc', 'init': 'scope~', 'pos': (300, 290), 'w': 260, 'h': 220},
    {'key': 'md', 'init': 'modal~', 'pos': (30, 520), 'w': 260, 'h': 300},
    {'key': 'cl', 'init': 'clean~', 'pos': (30, 840), 'w': 240, 'h': 200},
    {'key': 'c1', 'comment': True, 'text': 'takes the inaudible rumble out',
     'pos': (30, 1050)},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 1090), 'w': 220, 'h': 220},
]
links = [('sl', 'float out', 'sg', 'value'),
         ('sg', 'signal', 'oe', 'left in'),
         ('oe', 'left out', 'sc', 'in'),
         ('oe', 'left out', 'md', 'excite in'),
         ('md', 'out', 'cl', 'left in'),
         ('cl', 'left out', 'fo', 'left')]
print(build('clean~', 'clean~ and one_euro~ - conditioning before it is heard',
            body, demo, links, demo_width=590, text_width=800, text_height=700))

# ------------------------------------------------------------- data bridges
body = """These carry audio back into the ordinary node world, at three different rates.

The audio graph runs every sample; the patch runs once a frame. Anything that 
has to cross between them loses something, and which of these you want depends 
on what you can afford to lose.

THE NODES:

snapshot~  one value per frame, as an ordinary float
capture~   every sample, as a numpy array
array~     the same node
stream~    the other way: an array from the patch, played as a signal
audio_in~  the same node
scope~     every sample, drawn
place~     not a bridge - a spatializer, included here because it is the last 
           stage before the output

snapshot~ IS FOR CONTROL SIGNALS:
Patch any ~ signal in and the current value appears on the node face and goes 
out at frame rate, ready for number boxes, math nodes, OSC, anything. 
An adsr~ or an lfo~ becomes an ordinary float stream. For a slow-moving control 
signal that is exactly right.

For an audio signal it is not. Sixty samples a second of something oscillating 
at 440 Hz is noise - the waveform is gone, aliased beyond recognition. 
This is the commonest mistake with these nodes: a plot fed through snapshot~ 
showing something that looks like a signal and is not.

capture~ AND scope~ KEEP EVERYTHING:
Both use a ring buffer holding every sample. capture~ hands you the array, so 
plot, spectrum, numpy and torch nodes can work on the actual waveform. 
scope~ draws it directly, with a trigger, which is what you want when the 
question is "what does this look like" rather than "what is this value".

stream~ GOES THE OTHER WAY:
An array or tensor arriving on 'audio in' - from t.audio_source, 
t.audio.file_stream, a capture~ elsewhere, any numpy or torch chain - comes out 
as a signal, so a microphone can drive vocoder~, a recording excite string~, or 
live input reach a vst~. Set 'rate' to the rate the chunks were made at; 
file_stream's sample_rate outlet can drive it. 'latency' is how much to hold 
before starting: too little and a bursty source runs dry, counted on 
'underruns'; a backlog past a quarter second is skipped, counted on 'dropped'.

place~ PUTS IT SOMEWHERE:
One outlet per speaker, patched onward to audio_out~'s inputs. Several place~ 
into one output sum at its inlets, which is how each source gets its own 
position in the room. Stereo is a fact rather than a switch: patch 'right in' 
and the pair is held apart by 'width'.

SYNTAX:
snapshot~
capture~
stream~ <rate> <latency ms>
scope~

EXAMPLE:
scope~

INPUTS and PARAMETERS:

in:
The signal.

bang (snapshot~, capture~):
Ask for a value or an array now.

sync / level / time (scope~):
The trigger, where it triggers, and how much of the buffer to show.

left in / right in / pan / width / front-rear / top-bottom (place~):
The source and where to put it.

OUTPUTS: 

value / peak / rms (snapshot~):
The current value, and its peak and average over the frame - the last two 
being the honest way to follow an audio signal's LEVEL at frame rate, 
where following its value is meaningless.

array (capture~, scope~):
The samples.

dropped (capture~):
How many blocks were missed, so you know whether the patch is keeping up.

left out / right out, underruns / dropped (stream~):
The signal, and how often it ran dry or had to skip ahead."""

demo = [
    {'key': 'lfo', 'init': 'lfo~ 0.5', 'pos': (30, 62), 'w': 200, 'h': 160},
    {'key': 'sn', 'init': 'snapshot~', 'pos': (30, 240), 'w': 220, 'h': 180},
    {'key': 'f1', 'init': 'float', 'pos': (30, 435), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c0', 'comment': True, 'text': 'right for a slow control signal',
     'pos': (30, 485)},
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (300, 62), 'w': 220, 'h': 200},
    {'key': 'sc', 'init': 'scope~', 'pos': (300, 280), 'w': 260, 'h': 220},
    {'key': 'c1', 'comment': True, 'text': 'audio needs every sample, not one a frame',
     'pos': (300, 510)},
    {'key': 'cp', 'init': 'capture~', 'pos': (30, 525), 'w': 220, 'h': 180},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 720), 'w': 208, 'h': 176,
     'props': {'color': 'none', 'width': 200, 'height': 128, 'style': 'line',
               'update style': 'input is multi-channel sample', 'sample count': 512,
               'min x': 0.0, 'max x': 512.0, 'min y': -1.0, 'max y': 1.0}},
    {'key': 'c2', 'comment': True, 'text': 'the real waveform, as an array',
     'pos': (30, 905)},
]
links = [('lfo', 'signal', 'sn', 'in'), ('sn', 'value', 'f1', ''),
         ('vco', 'left out', 'sc', 'in'),
         ('vco', 'left out', 'cp', 'in'), ('cp', 'array', 'p1', 'y')]
print(build('snapshot~', 'snapshot~ - carrying audio back to the patch', body,
            demo, links, demo_width=590, text_width=810, text_height=740))

# ------------------------------------------------- additive~, vst~, shape_modes
body = """Three nodes that each build something from a description rather than a preset.

additive~ - AN OSCILLATOR FROM A DRAWN SPECTRUM:
Draw the amplitude of each partial against its index - partial 1 is the 
fundamental, 2 the octave above, 3 the twelfth - and the node sounds their sum. 
The gestures are shaper~'s: drag a point, right-click to add or remove one, 
shift and left-drag to bend a segment.

'stretch' pulls the partials away from whole-number ratios, which is the 
difference between a harmonic tone and a bell. 'odd/even' fades between the two 
families - all odd partials is a clarinet, both is a saw. spectrum~ is the same 
node.

shape_modes - A MODE TABLE FROM A SHAPE:
Patch its 'modes' outlet into modal~, rub~ or blow~ and their table stops being 
a preset and becomes whatever you described. Give it an outline - a list of 
half-widths along the length - say how that outline is swept into a volume, 
and say what it is made of, and it solves for the modes.

That is what makes an object you invented playable. A bar, a bowl, a bell, a 
tube of a shape nobody makes: describe the geometry and the material, and the 
three model nodes will strike it, bow it and blow it.

vst~ - SOMEBODY ELSE'S EFFECT:
A VST3 or AudioUnit plugin, patched like any other unit. The argument is part 
of a plugin's filename - 'valhalla', 'waveshell'. A file holding several 
plugins offers them in the 'plugin' option; pick one and it reloads. 
plugin~ is the same node.

SYNTAX:
additive~ <frequency>
shape_modes
vst~ <part of a filename>

EXAMPLE:
vst~ valhalla

INPUTS and PARAMETERS:

frequency / pitch (additive~):
Where the fundamental sits, and exponential transposition.

partials / tilt / odd-even / stretch / spread (additive~):
How many partials, how the level falls across them, which family, how far from 
harmonic, and how much they are detuned from each other.

spectrum (additive~):
The drawn shape, as data - so a spectrum can be stored or sent.

profile / sweep / mirror / carve (shape_modes):
The outline, and how it becomes a volume.

material / length / width / depth / wall (shape_modes):
What it is made of and how big it is.

solve / compute / count (shape_modes):
Work the modes out, and how many to find.

left in / right in / mix (vst~):
The signal and the wet-dry balance.

OUTPUTS: 

signal / out:
The sound.

modes (shape_modes):
The mode table, for modal~, rub~ or blow~.

report / frequency / decay / mesh (shape_modes):
What it found, and the geometry it solved.

spectrum out (additive~):
The drawn spectrum."""

demo = [
    {'key': 'sm', 'init': 'shape_modes', 'pos': (30, 62), 'w': 320, 'h': 340},
    {'key': 'c0', 'comment': True, 'text': 'describe a shape, then click solve',
     'pos': (30, 415)},
    {'key': 'md', 'init': 'modal~', 'pos': (30, 455), 'w': 260, 'h': 300},
    {'key': 'c1', 'comment': True, 'text': 'its table is now your shape, not a preset',
     'pos': (30, 765)},
    {'key': 'ck', 'init': 'clock~ 0.7', 'pos': (400, 62), 'w': 220, 'h': 200},
    {'key': 'st', 'init': 'strike~', 'pos': (400, 280), 'w': 220, 'h': 200},
    {'key': 'ad', 'init': 'additive~ 220', 'pos': (400, 500), 'w': 320, 'h': 300},
    {'key': 'c2', 'comment': True, 'text': 'draw the partials; stretch makes it a bell',
     'pos': (400, 810)},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 805), 'w': 220, 'h': 220},
]
links = [('sm', 'modes', 'md', 'modes'),
         ('ck', 'trigger', 'st', 'hit'),
         ('st', 'out', 'md', 'excite in'),
         ('md', 'out', 'fo', 'left')]
print(build('additive~', 'additive~, shape_modes, vst~ - built from a description',
            body, demo, links, demo_width=750, text_width=810, text_height=760))
