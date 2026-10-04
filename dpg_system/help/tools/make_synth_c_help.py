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

# ------------------------------------------------------------------ snapshot~
body = """snapshot~ carries a signal's value back into the ordinary node world, once a frame.

The audio graph runs every sample; the patch runs once a frame. snapshot~
crosses between them by taking one reading per frame, which is right for a
control signal and wrong for audio.

THE NODES:

snapshot~  a signal's value, peak and level, as ordinary floats

snapshot~ IS FOR CONTROL SIGNALS:
Patch any ~ signal in and the current value appears on the node face and goes
out at frame rate, ready for number boxes, math nodes, OSC, anything.
An adsr~ or an lfo~ becomes an ordinary float stream. For a slow-moving control
signal that is exactly right.

For an audio signal it is not. Sixty samples a second of something oscillating
at 440 Hz is noise - the waveform is gone, aliased beyond recognition.
This is the commonest mistake with this node: a plot fed through snapshot~
showing something that looks like a signal and is not. Use capture~ for the
samples, or scope~ to look at them.

PEAK AND RMS ARE THE HONEST LEVEL:
'peak' (the largest size the signal reached) and 'rms' (its average power)
cover every sample since the previous reading, not one block, so a transient
shorter than a frame still counts. They are the right way to follow an audio
signal's LEVEL at frame rate, where following its value is meaningless.

ON CHANGE OR CONTINUOUS:
By default ('on change') the outlets send only when the value has moved by
more than 'deadband', so a resting signal does not push sixty identical floats
a second into the patch. 'continuous' sends every frame. A bang sends a
reading now, whatever the mode.

SYNTAX:
snapshot~

EXAMPLE:
snapshot~

INPUTS and PARAMETERS:

enable:
Off stops it reading.

in:
The signal.

bang:
Send a reading now.

value (on the face):
The newest value, shown to 'precision' decimal places.

mode / deadband / precision (options):
'on change' (default) or 'continuous'; how far the value must move before
'on change' sends again (0 by default); and the number of decimals shown
(3 by default).

OUTPUTS:

value:
The newest sample of the signal.

peak:
The largest absolute value since the previous reading.

rms:
The root-mean-square level since the previous reading.

The three go out right to left, so whatever 'value' drives already has the
matching peak and rms.

RELATED:
capture~ (and array~) hands over every sample as an array.
scope~ draws the waveform. vu~ shows a level as bars.
stream~ goes the other way: arrays from the patch into the ~ graph."""

demo = [
    {'key': 'lfo', 'init': 'lfo~ 0.5', 'pos': (30, 62), 'w': 180, 'h': 145},
    {'key': 'c0', 'comment': True, 'text': 'a slow control signal', 'pos': (290, 62)},
    {'key': 'sn', 'init': 'snapshot~', 'pos': (30, 260), 'w': 220, 'h': 140},
    {'key': 'f1', 'init': 'float', 'pos': (30, 450), 'w': 130, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'one value a frame - right for this',
     'pos': (290, 450)},
]
links = [('lfo', 'signal', 'sn', 'in'), ('sn', 'value', 'f1', '')]
print(build('snapshot~', "snapshot~ - a signal's value, once a frame", body,
            demo, links, demo_width=560, text_width=810, text_height=740))

# ------------------------------------------------------------ capture~ / array~
body = """capture~ hands a live signal to the patch as arrays - every sample, not one a frame.

THE NODES:

capture~  every sample of a signal, as numpy arrays or torch tensors
array~    the same node

capture~ KEEPS EVERYTHING:
The audio engine writes every sample into a ring buffer, and once a frame
capture~ sends some of it out as an array, so plot, spectrum, numpy and torch
nodes can work on the actual waveform. snapshot~, which samples one value a
frame, cannot do that for an audio signal.

TWO MODES:
'latest' (the default) sends the newest 'size' samples every frame. Frames and
audio blocks do not divide evenly, so successive arrays overlap or skip a
little. Right for a display or a spectrum, where you want the current window
and do not care about continuity.

'continuous' sends gapless chunks of exactly 'size' samples, in order, every
sample delivered once. A frame that has gathered two chunks sends two; a
partial remainder waits for the next frame, so the length never varies. Right
for analysis, recording or anything cumulative. If the patch falls far enough
behind that samples are lost, 'dropped' says how many.

'size' is 512 by default, the engine's audio block; any value from 16 to 32768
works.

capture~ IS ALSO THE WAY INTO TORCH:
Its 'format' option makes the chunks numpy arrays, or torch tensors on the CPU
('torch cpu' - no copy, the tensor shares the chunk's memory) or on the Mac's
GPU ('torch mps'). That is the bridge from any live signal to t.rfft, t.cwt or
a model, so nothing else in the patch has to carry tensors.

ON BANG:
Set 'send' to 'on bang' and nothing goes out until a bang arrives. In 'latest'
mode each bang sends the newest window. In 'continuous' mode each bang sends
everything that has gathered since the last one, as whole chunks in order; a
partial remainder waits for the next bang. The buffer holds about 0.74 s, so
bang at least that often or 'dropped' reports what was lost.

SYNTAX:
capture~ [<size>] [latest | continuous]

EXAMPLE:
capture~ 1024 continuous

INPUTS and PARAMETERS:

enable:
Off stops it capturing.

in:
The signal.

bang:
Send an array now.

size / mode / send / format (options):
Samples per array; 'latest' or 'continuous'; 'every frame' or 'on bang'; and
'numpy', 'torch cpu' or 'torch mps'.

OUTPUTS:

array:
The samples - 1-D numpy, or a torch tensor if 'format' says so.

dropped:
In continuous mode, how many samples were lost because the patch fell behind -
so you know whether it is keeping up.

rate:
The engine's sample rate, sent once at the start, for whatever the array is
patched into (stream~'s 'rate', a speech node, a spectrum).

RELATED:
snapshot~ gives one value a frame, with peak and level.
scope~ draws the same kind of ring buffer, with a trigger.
stream~ (and audio_in~) goes the other way: arrays into the ~ graph.
record~ keeps every sample as a WAV take."""

demo = [
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (30, 62), 'w': 200, 'h': 185},
    {'key': 'c0', 'comment': True, 'text': 'an audio signal', 'pos': (290, 62)},
    {'key': 'cp', 'init': 'capture~', 'pos': (30, 300), 'w': 220, 'h': 160},
    {'key': 'c1', 'comment': True, 'text': 'the newest 512 samples, every frame',
     'pos': (290, 300)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 510), 'w': 208, 'h': 180,
     'props': {'color': 'none', 'width': 200, 'height': 128, 'style': 'line',
               'update style': 'input is multi-channel sample', 'sample count': 512,
               'min x': 0.0, 'max x': 512.0, 'min y': -1.0, 'max y': 1.0}},
    {'key': 'c2', 'comment': True, 'text': 'the real waveform, as an array',
     'pos': (290, 510)},
]
links = [('vco', 'left out', 'cp', 'in'), ('cp', 'array', 'p1', 'y')]
print(build('capture~', 'capture~ - every sample, as an array', body,
            demo, links, demo_width=590, text_width=810, text_height=740))

# ------------------------------------------------------------ stream~ / audio_in~
body = """stream~ plays arrays from the patch as a signal - the one way data enters the ~ graph.

THE NODES:

stream~    arrays or tensors from the patch, played as a signal
audio_in~  the same node

stream~ IS capture~ BACKWARDS:
An array arriving on 'audio in' - a capture~ elsewhere, record~'s take, a
speech node, any numpy or torch chain - comes out as a signal, so data can
drive vocoder~, excite string~, or reach a vst~. It is the one way arrays
enter the ~ world, as capture~ is the one way out.

Chunks may be 1-D (mono), (channels, frames) or (frames, channels) - the
longer side is taken as time. A stereo chunk fills both outlets; a mono one
sends the same signal from both.

RATE AND LATENCY:
Set 'rate' to the sample rate the chunks were made at - there is no way to
read it off the numbers. It is an inlet, so a sample_rate or rate outlet can
set it; stream~ converts to the engine's rate as it plays, so a 16 kHz source
plays at the right speed.

Chunks arrive in bursts, once a frame or whenever a device delivers, and the
audio engine plays evenly, so stream~ holds 'latency' ms in hand before it
starts. Too little and a bursty source runs dry, counted on 'underruns' - it
then waits until 'latency' has built up again. Too much and the sound lags.

MAX BACKLOG:
A live source that falls behind should skip rather than play late, so audio
queued beyond 'max backlog' (250 ms by default) is skipped and counted on
'dropped'. That is wrong for anything that arrives faster than it plays -
speech synthesis delivers a whole phrase in a fraction of its length - so set
'max backlog' to 0 to keep everything and play it out in order (the queue
holds about 20 seconds).

For a microphone or interface, adc~ is the better route: its device writes
straight into the audio engine instead of waiting on GUI frames, so a busy
patch cannot drop or delay it.

SYNTAX:
stream~ [<rate> [<latency ms>]]

EXAMPLE:
stream~ 16000 80

With no arguments the rate is the engine's and the latency 50 ms.

INPUTS and PARAMETERS:

enable:
Off fades out and stops; whatever was queued is discarded.

audio in:
The chunks to play.

rate:
The sample rate of the chunks, in Hz.

latency:
How much to hold before starting, in ms.

level:
Output gain, 0 to 2. An inlet, so a signal can ride it; the 'level depth'
option scales whatever is patched there.

max backlog (option):
In ms: queued audio past this is skipped. 0 keeps everything.

OUTPUTS:

left out / right out:
The signal.

underruns:
How many times it has run dry, as a running count, sent when it changes.

dropped:
How many samples have been skipped or refused because the queue was full, as
a running count, sent when it changes.

RELATED:
capture~ (and array~) is the way out: signal to arrays.
adc~ (and mic~) brings a live input device in directly.
record~ saves a signal as WAV takes."""

demo = [
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (30, 62), 'w': 200, 'h': 185},
    {'key': 'cp', 'init': 'capture~ 512 continuous', 'pos': (30, 300), 'w': 220, 'h': 160},
    {'key': 'c0', 'comment': True, 'text': 'a signal turned into arrays...',
     'pos': (290, 300)},
    {'key': 'st', 'init': 'stream~', 'pos': (30, 510), 'w': 240, 'h': 240},
    {'key': 'c1', 'comment': True, 'text': '...and played as a signal again',
     'pos': (290, 510)},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 810), 'w': 70, 'h': 310,
     'props': {'fader': 0.0}},
    {'key': 'c2', 'comment': True, 'text': 'raise the fader to hear it',
     'pos': (290, 810)},
]
links = [('vco', 'left out', 'cp', 'in'),
         ('cp', 'array', 'st', 'audio in'),
         ('cp', 'rate', 'st', 'rate'),
         ('st', 'left out', 'fo', 'left'),
         ('st', 'right out', 'fo', 'right')]
print(build('stream~', 'stream~ - arrays played as a signal', body,
            demo, links, demo_width=590, text_width=810, text_height=760))

# ----------------------------------------------------------------------- scope~
body = """scope~ draws a signal's waveform: an oscilloscope with a trigger.

THE NODES:

scope~  an oscilloscope: every sample, drawn

WHY NOT plot?
plot takes one value a frame, so an audio signal reaching it through snapshot~
is aliased beyond recognition: sixty samples a second of something oscillating
at 440 Hz is noise. scope~ keeps every sample in a ring buffer and draws a
window of it, so what you see is the waveform.

THE TRIGGER HOLDS IT STILL:
Untriggered ('free'), each frame starts at an unrelated point in the cycle and
a steady tone scrolls and tears. 'rising' (the default) and 'falling' start the
window where the signal crosses 'level' - zero by default, so a zero crossing -
going up or down, and a periodic signal then stands still. What moves on the
screen is what is actually changing in the sound. The level shows as a faint
line.

'noise reject' is the trigger's tolerance for wobble: a crossing only counts
once the signal has been clear of the level by that much on the other side.
Raise it until a noisy trace sits still; too high and quiet material stops
triggering.

With nothing to trigger on - silence, or a cycle longer than the window - the
last trace is held for half a second, then the display runs free, so it goes
back to showing the truth rather than freezing on a stale waveform.

THE READOUT:
'window' shows how long the window is and, when triggered, the frequency
implied by the spacing of the crossings.

SYNTAX:
scope~ [<samples>] [free | rising | falling]

EXAMPLE:
scope~ 1024 falling

With no arguments the window is 512 samples, triggered rising.

INPUTS and PARAMETERS:

enable:
Off stops the trace.

in:
The signal.

sync:
free, rising or falling.

level:
The value the trace starts on.

time:
How much time is on screen, in ms. It and the 'samples' option are two
handles on the same thing: move one and the other follows.

samples / noise reject / min y / max y / width / height (options):
The window in samples (16 to 16384), the trigger's tolerance (0 by default),
the vertical range (-1 to 1 by default), and the size of the display (300 by
128 by default; the strip under the display drags it bigger).

OUTPUTS:

array:
The window on screen, every frame, lined up on the trigger - for a patch that
wants to measure what it is looking at. Where the trigger is beside the point,
capture~ is the better source.

RELATED:
capture~ (and array~) for the samples as arrays.
snapshot~ for a control signal's value. vu~ for a level."""

demo = [
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (30, 62), 'w': 200, 'h': 185},
    {'key': 'sc', 'init': 'scope~', 'pos': (30, 300), 'w': 310, 'h': 250},
    {'key': 'c0', 'comment': True, 'text': 'triggered on the rising zero crossing\nset sync to free to see it tear',
     'pos': (380, 300)},
]
links = [('vco', 'left out', 'sc', 'in')]
print(build('scope~', 'scope~ - the waveform, drawn', body,
            demo, links, demo_width=680, text_width=810, text_height=740))

# ----------------------------------------------------------------------- place~
body = """place~ puts a source somewhere among the speakers.

THE NODES:

place~  a spatializer: one signal in, one outlet per speaker

ONE OUTLET PER SPEAKER:
Patch the outlets onward to audio_out~'s inputs. Several place~ into one
output sum at its inlets, which is how each source gets its own position in
the room. The speaker count (2 to 16, 4 by default) is fixed when the node is
made. The intended place in a channel strip is
source -> fader~ -> clean~ -> place~ -> audio_out~.

STEREO IS A FACT RATHER THAN A SWITCH:
Patch only 'left in' and the source is a single point. Patch 'right in' too and
the pair is two points, held apart by 'width' around the pan position. Width
starts at 2 divided by the speaker count, which on a ring puts the pair on
neighbouring speakers.

RING:
The outlets are speakers equally spaced around a circle, in order. 'pan' is the
direction: 0 front centre, plus or minus 0.5 the sides, plus or minus 1 the
rear, where the two ends meet - so a ramp from -1 to 1 goes once round. Front
centre falls midway between out 1 and out 2, and increasing pan moves toward
out 2, out 3 and on; for four speakers, out 1 front-left, out 2 front-right,
out 3 rear-right, out 4 rear-left. A sound is in at most two neighbouring
speakers at a time, which keeps it sharp as it moves.

CORNERS:
With 4 or 8 speakers, the 'space' option 'corners' reads the outlets as the
corners of the room instead: front-left, front-right, rear-left, rear-right,
then (with 8) the same four on the top layer - the first four are the bottom.
The position is three equal-power faders: 'pan' (now labelled left/right),
'front/rear' and 'top/bottom', each -1 to +1. Those two extra controls only
show in corners mode. Other speaker counts fall back to the ring.

TWO SPEAKERS:
place~ 2 is a stereo pair whatever 'space' says: pan runs from hard left at -1
to hard right at +1, equal-power.

All the position controls are inlets, so an lfo~ orbits a sound and effort
data pushes it around the room. Moves are smoothed across each audio block,
so a sweep does not click.

SYNTAX:
place~ [<speakers>] [ring | corners]

EXAMPLE:
place~ 8 corners

INPUTS and PARAMETERS:

bypass:
Stands aside: left and right pass straight to out 1 and out 2.

left in / right in:
The source. Patch 'right in' for a stereo source.

pan:
-1 to +1: around the ring, or left to right in corners and on a pair.

width:
0 to 2: how far apart a stereo pair is held. Ignored for a mono source.

front/rear / top/bottom:
-1 to +1, corners mode only: -1 is front and top, +1 rear and bottom.
top/bottom does something only with 8 speakers, where there is a top layer.

space (option):
ring or corners.

OUTPUTS:

out 1, out 2, ...:
One per speaker.

RELATED:
audio_out~ is the socket the outlets go to.
pan~ is a plain stereo panner; fader~ and fader_out~ (on the vca~ help page)
pan as part of a channel strip. clean~ is the stage before."""

demo = [
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (30, 62), 'w': 200, 'h': 185},
    {'key': 'fd', 'init': 'fader~', 'pos': (30, 300), 'w': 70, 'h': 310,
     'props': {'fader': 0.0}},
    {'key': 'c0', 'comment': True, 'text': 'raise the fader to hear it',
     'pos': (290, 300)},
    {'key': 'lfo', 'init': 'lfo~ 0.1 ramp', 'pos': (30, 670), 'w': 180, 'h': 145},
    {'key': 'c1', 'comment': True, 'text': 'a slow ramp: once round the ring\nevery ten seconds',
     'pos': (290, 670)},
    {'key': 'pl', 'init': 'place~ 4', 'pos': (30, 870), 'w': 200, 'h': 200},
    {'key': 'c2', 'comment': True, 'text': 'four speakers in a ring',
     'pos': (290, 870)},
    {'key': 'ao', 'init': 'audio_out~ 1 2 3 4', 'pos': (30, 1120), 'w': 200, 'h': 180},
    {'key': 'c3', 'comment': True, 'text': 'with only two speakers you hear\nit pass across the front',
     'pos': (290, 1120)},
]
links = [('vco', 'left out', 'fd', 'left'),
         ('fd', 'left', 'pl', 'left in'),
         ('lfo', 'signal', 'pl', 'pan'),
         ('pl', 'out 1', 'ao', 'left in'),
         ('pl', 'out 2', 'ao', 'right in'),
         ('pl', 'out 3', 'ao', 'in 3'),
         ('pl', 'out 4', 'ao', 'in 4')]
print(build('place~', 'place~ - a source among the speakers', body,
            demo, links, demo_width=590, text_width=810, text_height=760))

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
    {'key': 'f1', 'init': 'fader_out~ 1 2', 'pos': (30, 910), 'w': 220, 'h': 220,
     'props': {'fader': 0.0}},
    {'key': 'ad', 'init': 'additive~ 220', 'pos': (30, 1250), 'w': 320, 'h': 300},
    {'key': 'c2', 'comment': True, 'text': 'draw the partials; stretch makes it a bell',
     'pos': (30, 1615)},
    {'key': 'f2', 'init': 'fader_out~ 1 2', 'pos': (30, 1660), 'w': 220, 'h': 220,
     'props': {'fader': 0.0}},
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
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 1030), 'w': 220, 'h': 220,
     'props': {'fader': 0.0}},
    {'key': 'c2', 'comment': True, 'text': 'raise the fader to hear it', 'pos': (30, 1350)},
]
links = [('ck', 'trigger', 'md', 'strike'),
         ('md', 'out', 'vs', 'left in'),
         ('vs', 'left out', 'fo', 'left'),
         ('vs', 'right out', 'fo', 'right')]
print(build('vst~', "vst~ - somebody else's effect, patched in",
            body, demo, links, demo_width=420, text_width=810, text_height=760))
