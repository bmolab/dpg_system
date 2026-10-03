"""clean~, the data bridges, additive~ and shape_modes, vst~."""
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
    {'key': 'cl', 'init': 'clean~', 'pos': (30, 926), 'w': 240, 'h': 200},
    {'key': 'c1', 'comment': True, 'text': 'takes the inaudible rumble out',
     'pos': (30, 1136)},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 1176), 'w': 220, 'h': 220},
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

capture~ IS ALSO THE WAY INTO TORCH:
Its 'format' option makes the chunks numpy arrays, or torch tensors on the CPU 
('torch cpu' - no copy, the tensor shares the chunk's memory) or on the Mac's 
GPU ('torch mps'). That is the bridge from any live signal to t.rfft, t.cwt or 
a model, so nothing else in the patch has to carry tensors.

stream~ GOES THE OTHER WAY:
An array arriving on 'audio in' - a capture~ elsewhere, record~'s take, any 
numpy or torch chain - comes out as a signal, so data can drive vocoder~, 
excite string~, or reach a vst~ or a speech node. It is the one way arrays 
enter the ~ world, as capture~ is the one way out. Set 'rate' to the rate the 
chunks were made at. 'latency' is how much to hold 
before starting: too little and a bursty source runs dry, counted on 
'underruns'; a backlog past the 'max backlog' option is skipped, counted on 
'dropped' - set it to 0 for speech or anything else that arrives faster than 
it plays.

For a microphone or interface, adc~ is the better route: its device writes 
straight into the audio engine instead of waiting on GUI frames, so a busy 
patch cannot drop or delay it. See the record~ help patch.

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
The samples - numpy, or a torch tensor if capture~'s 'format' says so.

dropped (capture~):
How many blocks were missed, so you know whether the patch is keeping up.

rate (capture~):
The engine's sample rate, sent once, for whatever the array is patched into.

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

# ------------------------------------------------------ additive~, shape_modes
body = """Two nodes that each build something from a description rather than a preset.

THE NODES:

additive~    an oscillator whose spectrum you draw
spectrum~    the same node
shape_modes  a mode table for modal~, rub~ or blow~, solved from a shape

additive~ - AN OSCILLATOR FROM A DRAWN SPECTRUM:
Draw the amplitude of each partial against its index - partial 1 is the
fundamental, 2 the octave above, 3 the twelfth - and the node sounds their sum.
The gestures are shaper~'s: drag a point, right-click to add or remove one,
shift and left-drag to bend a segment. Set the 'edit' option to 'bars' to set
the partials one at a time instead - bar 9 is partial 9 exactly, for a tone
that is a particular handful of harmonics and nothing else.

The drawing is the character; the inlets are the ways it moves, all at audio
rate. Sweeping 'tilt' with an envelope is the workhorse gesture: a filter
sweep that cannot ring or lose the fundamental, because nothing is filtered.
'stretch' pulls the partials off whole-number ratios, which is the difference
between a harmonic tone and a bell. At exactly 0 the partials are baked into
a wavetable and five hundred cost what one does; off 0 it becomes a real bank
of oscillators, capped at 64 partials. The small button beside tilt, odd/even
and stretch puts that control exactly back on its default.

shape_modes - A MODE TABLE FROM A SHAPE:
Patch its 'modes' outlet into modal~, rub~ or blow~ and their table stops
being a preset and becomes whatever you described. Give it an outline - a list
of half-widths along the length - say how that outline is swept into a volume,
and say what it is made of, and it solves for the modes. A bar, a bead, a
tube of a shape nobody makes: describe it and the model nodes will strike it,
bow it and blow it.

Anything you CLICK re-solves at once, and so does a profile arriving on the
cord. Anything you DRAG - length, width, depth, detail - waits for 'compute',
because a solve takes tens to hundreds of milliseconds. The mallet controls
(strike, direction, damping, count) are free: they only reweigh modes
already solved.

Material and size barely change the table - every mode moves together, so
they cancel out of a ratio. What they change is the PITCH and the ring, so
patch the 'frequency' and 'decay' outlets into the inlets of the same name.

SYNTAX:
additive~ <frequency> <partials> <tilt>
additive~ <preset>
shape_modes

EXAMPLE:
additive~ 110 bell

The presets are saw, square, triangle, pulse, sine, organ, vocal, bell and
gong, and can be mixed with the numbers: additive~ 220 organ.

INPUTS and PARAMETERS (additive~):

frequency / pitch / linear fm:
Where the fundamental sits (default 110 Hz), transposition in octaves, and
frequency modulation in Hz.

tilt:
A slope across the spectrum in dB per octave - its brightness. Default -6,
which makes a row of equal partials a saw.

partials:
How many sound, 1 to 512 (default 32). Fractional: the top one fades in.

odd/even:
0 is the odd partials alone (hollow - a square, a clarinet), 0.5 all of them,
1 the even alone.

stretch:
0 is harmonic. A few thousandths is a stiff string, more is a bell, negative
squeezes the partials together towards a gong.

spread / phase:
How the partials' starting phases are spread across the cycle. At 0 they all
start together: one narrow spike a cycle, which reads as a buzz and costs
level. Opened up, the same spectrum smears into a smoother tone. 'phase'
(aligned, random, schroeder) chooses what it spreads towards. Changes glide
over the 'phase glide' option rather than clicking.

phase mod / sync:
Phase modulation in cycles, and a rising edge that restarts the cycle.

spectrum:
A list of amplitudes, to load a spectrum from elsewhere - an analysis,
another node's 'spectrum out'.

preset:
Load one of the presets.

options:
normalize (none, rms, peak), edit (curve, bars), span (whether the curve
stretches to the partial count or covers all 512), phase glide, start phase,
reset phase, flatten, clear bars, and the drawing's width and height.

INPUTS and PARAMETERS (shape_modes):

profile:
The outline, as a list of at least two numbers, taken as proportions and
scaled to 'width'.

solve / open:
Solve the 'body' (the thing itself) or the 'cavity' (the air inside it), and
for a cavity, which ends are open.

sweep / mirror / carve / wall:
How the outline becomes a volume: revolved as a radius or extruded as a
half-width; mirrored from a half outline; for an extrude, whether it is cut
into the depth or the width; and how far it is hollowed (1 is solid).

material / length / width / depth:
What it is made of, and its size in metres.

detail / compute:
How finely it is meshed, and the button that solves it.

strike / direction / damping / count:
Where along it the mallet lands (0 to 1), which way, how much faster the upper
modes die, and how many modes to send.

show / swell:
Which mode the 'mesh' outlet shows (0 is the shape itself), and how far.

OUTPUTS (additive~):

signal:
The sound.

spectrum out:
The drawn spectrum, whenever it is edited.

OUTPUTS (shape_modes):

report:
What it found, or why it could not.

frequency / decay / modes:
The pitch and ring time the material implies, and the mode table - three
cords into the inlets of the same names on modal~, rub~ or blow~.

mesh:
The solved geometry, for looking at.

RELATED:
modal~, rub~, blow~, shaper~, vco~, vst~"""

demo = [
    {'key': 'sm', 'init': 'shape_modes', 'pos': (30, 62), 'w': 320, 'h': 340},
    {'key': 'c0', 'comment': True,
     'text': 'describe a shape, then click compute\nto send the table',
     'pos': (30, 420)},
    {'key': 'md', 'init': 'modal~', 'pos': (30, 470), 'w': 260, 'h': 300},
    {'key': 'c1', 'comment': True,
     'text': 'its table is now your shape, not a preset\nclick strike to hear it',
     'pos': (30, 865)},
    {'key': 'f1', 'init': 'fader_out~ 1 2', 'pos': (30, 910), 'w': 220, 'h': 220},
    {'key': 'ad', 'init': 'additive~ 220', 'pos': (30, 1250), 'w': 320, 'h': 300},
    {'key': 'c2', 'comment': True, 'text': 'draw the partials; stretch makes it a bell',
     'pos': (30, 1615)},
    {'key': 'f2', 'init': 'fader_out~ 1 2', 'pos': (30, 1660), 'w': 220, 'h': 220},
    {'key': 'c3', 'comment': True, 'text': 'raise a fader to hear it', 'pos': (30, 1985)},
]
links = [('sm', 'modes', 'md', 'modes'),
         ('sm', 'frequency', 'md', 'frequency'),
         ('sm', 'decay', 'md', 'decay'),
         ('md', 'out', 'f1', 'left'),
         ('ad', 'signal', 'f2', 'left')]
print(build('additive~', 'additive~ and shape_modes - built from a description',
            body, demo, links, demo_width=420, text_width=810, text_height=760))

# ------------------------------------------------------------------- vst~
body = """vst~ hosts somebody else's effect - a VST3 or AudioUnit plugin - patched like
any other unit.

THE NODES:

vst~     a plugin effect, in the audio graph
plugin~  the same node

FINDING THE PLUGIN:
The argument is any distinctive part of a plugin's filename - 'supermassive',
'valhalla', 'waveshell' - or a full path. The first installed file containing
it is loaded, so be specific when several match. If nothing matches, 'status'
says so and the console lists every plugin file installed. A file holding
several plugins (a Waves shell) offers them in the 'plugin' option; pick one
and it reloads. Effects only: an instrument is refused, and so are macOS's
own built-in AudioUnits, which cannot be hosted. The node exists only when
the pedalboard package is installed.

KNOBS AND MENUS:
A plugin's parameters come in two kinds. The ones with a range are KNOBS:
choose one by name in a 'param n source' option and the 'param n' inlet drives
it, 0 to 1 whatever the plugin calls its own range - so a knob, an lfo~ or a
joint's effort all reach it the same way. Each slot then shows the name of
what it drives. The rest are MENUS - a reverb mode, a sync division - set by
picking the parameter in 'choice n' and its setting in 'choice n value'.
Menus cannot be modulated; they are set by name, which is exact.

Press 'print parameters' to list what a loaded plugin offers, marked [knob]
or [menu]. 'open editor' brings up the plugin's own window. The patch UI
waits while it is open - the audio carries on - and when it closes, every
knob and menu the node shows is read back from the plugin, so it is saved
with the patch. Edits to parameters the node does not show stay in the
plugin only and are not saved.

THREE THINGS WORTH KNOWING:
Parameters move once per block, about 86 times a second. That is what plugin
automation is. An audio-rate signal patched to a param inlet is read at the
last sample of each block - fine for effort data or an lfo~, wrong for
anything you want to hear as modulation.

Latency is not compensated. 'status' reports the plugin's delay. The 'mix'
control is a parallel dry path, so on a plugin with real latency it combs
rather than blends - leave mix at 1 there and blend outside with mix~.

A plugin that throws, returns the wrong number of samples, or keeps missing
the audio deadline is dropped, and the node passes its input through
instead. The reason lands in 'status' and on the console.

SYNTAX:
vst~ <part of a filename> params=<n> choices=<n>

EXAMPLE:
vst~ supermassive params=12

'params' is how many knob slots (default 8, up to 24) and 'choices' how many
menu slots (default 3, up to 12). Both are optional; everything that is not
one of them is part of the name, so 'vst~ WaveShell1-VST3 17.1' reads
correctly.

INPUTS and PARAMETERS:

left in / right in:
The signal. With nothing patched to 'right in', the left feeds both sides.

bypass:
Passes the input through untouched.

mix:
Wet against dry, 0 to 1. Default 1, all plugin.

param 1, param 2, ...:
The knob slots, 0 to 1.

options:
file (the name to look for), plugin, param n source, choice n,
choice n value, print parameters, open editor.

status:
The plugin's name, mono or stereo, its cost per block and its latency - or
why nothing is loaded.

OUTPUTS:

left out / right out:
The processed sound. A mono plugin sends the same signal to both.

RELATED:
mix~, fader~, additive~, delay~"""

demo = [
    {'key': 'ck', 'init': 'clock~ 1', 'pos': (30, 62), 'w': 220, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'tick run: one strike a second',
     'pos': (30, 272)},
    {'key': 'md', 'init': 'modal~', 'pos': (30, 315), 'w': 260, 'h': 300},
    {'key': 'vs', 'init': 'vst~ supermassive', 'pos': (30, 730), 'w': 300, 'h': 420},
    {'key': 'c1', 'comment': True,
     'text': 'a plugin, patched like any unit\npick knobs in the param n source options',
     'pos': (30, 980)},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 1030), 'w': 220, 'h': 220},
    {'key': 'c2', 'comment': True, 'text': 'raise the fader to hear it', 'pos': (30, 1350)},
]
links = [('ck', 'trigger', 'md', 'strike'),
         ('md', 'out', 'vs', 'left in'),
         ('vs', 'left out', 'fo', 'left'),
         ('vs', 'right out', 'fo', 'right')]
print(build('vst~', "vst~ - somebody else's effect, patched in",
            body, demo, links, demo_width=420, text_width=810, text_height=760))
