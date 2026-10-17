"""audio_out~ (the framework), oscillators, sig~, noise~, sampler_osc~, filters, levels."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ----------------------------------------------------------------- audio_out~
body = """audio_out~ is where sound leaves the patch. Nothing is heard without one.

HOW THE ~ NODES WORK:

Every node whose name ends in ~ is part of the audio graph, and that graph is 
not the same thing as the patch. Ordinary nodes pass messages when something 
happens; ~ nodes run continuously in the audio callback, every sample, whether 
or not anything is arriving.

The cords between ~ nodes declare the TOPOLOGY - what feeds what - and from 
that a DSP program is compiled and run in the audio thread. Repatching 
recompiles it, but the units themselves are created once and live as long as 
the node does, so reordering the graph does not reset an oscillator's phase, 
a filter's state, or an envelope's stage. You can rewire while it plays.

CROSSING BETWEEN THE TWO WORLDS:

sig~        a control value INTO the audio graph, with glide so a 60 Hz 
            stream does not zipper
snapshot~   an audio signal OUT to the patch, at frame rate, as an ordinary float
capture~    audio out as a whole numpy array, keeping every sample
scope~      the signal drawn, at audio rate

sig~ is the way effort data drives sound. snapshot~ is the way sound drives 
anything else.

THE CHANNEL STRIP:

The intended order is source -> fader~ -> clean~ -> place~ -> audio_out~. 
Level is fader~'s job, hygiene is clean~'s, position is place~'s, and this 
node is only the socket. fader_out~ is a fader and a socket in one, which is 
how most patches start.

SYNTAX:
audio_out~ <channel> <channel> ...

EXAMPLE:
audio_out~ 1 2

INPUTS and PARAMETERS:

left in / right in / in 3 ...:
One inlet per channel you listed. Several place~ or fader~ can sum into the 
same inlet - that is how a mix happens.

mute:
Silence everything without unpatching.

device:
Which audio interface to use. This is engine-wide: changing it here changes it 
for every audio node in the patch.

OUTPUTS: 

peak:
The level reaching the output, for metering.

status:
Whether the device is running.

WHEN THERE IS NO SOUND:
Check in this order. Is the device right? Is mute off? Does the channel exist 
on that device - a channel number beyond what the interface has is silent 
rather than an error. Is anything actually patched to the inlet? 
And is there a fader~ or vca~ in the chain sitting at zero?"""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'lfo', 'init': 'lfo~', 'pos': (30, 120), 'w': 200, 'h': 160},
    {'key': 'c0', 'comment': True, 'text': 'a slow wobble to hear something moving',
     'pos': (30, 290)},
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (30, 330), 'w': 220, 'h': 200},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 545), 'w': 220, 'h': 220},
    {'key': 'c1', 'comment': True, 'text': 'fader_out~ is a fader and the socket\nraise its fader to hear it',
     'pos': (30, 780)},
    {'key': 'sc', 'init': 'scope~', 'pos': (290, 545), 'w': 260, 'h': 220},
    {'key': 'c3', 'comment': True, 'text': 'the actual waveform, at audio rate',
     'pos': (290, 780)},
]
links = [('lfo', 'signal', 'vco', 'pitch'),
         ('vco', 'left out', 'fo', 'left'),
         ('vco', 'left out', 'sc', 'in')]
print(build('audio_out~', 'audio_out~ - where sound leaves the patch', body, demo,
            links, demo_width=580, text_width=820, text_height=800))

# ----------------------------------------------------------------- oscillators
body = """These are the oscillators: nodes that make a repeating signal rather 
than change one.

THE NODES:

vco~     a band-limited oscillator, with detuned unison. The one to listen to
lfo~     a low frequency oscillator for modulating things - though nothing 
         stops it reaching audio rate, where it becomes an unbandlimited 
         modulator for FM and AM
phasor~  a plain 0 to 1 ramp, for driving positions rather than for hearing

vco~ PITCH IS TWO INLETS:
'frequency' is the base pitch in Hz, and 'pitch' is added to it in OCTAVES: 
1 is an octave up, -1 an octave down. Patch an envelope into pitch for a 
sweep, an LFO for vibrato. Because it is exponential, the same modulation 
gives the same musical interval at any base frequency. The demo's lfo~, at 
depth 0.05, wobbles the pitch by a twentieth of an octave either way.

There is a third, 'linear fm', in Hz rather than octaves. Linear FM detunes 
rather than transposes, which is what makes it inharmonic and metallic.

vco~ UNISON:
The 'voices' option stacks up to eight copies of the oscillator, spread 
'detune' cents apart around the note; 'spread' pans them across left and 
right out, and 'drift' gives each its own slow wander in cents. With one 
voice, or spread at 0, right out carries the same signal as left out.

lfo~ IS FOR MODULATION:
It is not band limited, so its edges are clean steps - which is right for 
driving other things and harsh to listen to. Besides the usual waves it has 
'sample_hold' (a new random value each cycle), 'smooth_random' (a glide from 
one random value to the next each cycle) and 'noise'. 'depth' scales the 
output and 'offset' is added after, so depth 0.2 offset 0.5 swings between 
0.3 and 0.7. Untick the 'bipolar' option and the wave runs 0 to 1 before 
depth and offset, instead of -1 to 1.

phasor~ IS A POSITION:
Its output climbs from 0 to 1 and jumps back, once per cycle; 'start' and 
'end' rescale that range. 'period' in seconds, when above 0, overrides 
'frequency' - patch sampler_osc~'s length outlet into it and one cycle 
scans the whole file. A negative frequency runs it backwards. The 'wrap' 
outlet gives a one-sample pulse each time the ramp turns over.

Leave phasor~'s 'phase' inlet at 0: a non-zero value is meant as an offset, 
but it is added again on every audio block, so the ramp lurches forward 
instead of shifting.

SYNTAX:
vco~ <frequency> <voices> <detune> <shape>
lfo~ <rate> <shape>
phasor~ <frequency>

EXAMPLE:
vco~ 110 3 12 saw

INPUTS and PARAMETERS:

frequency (vco~, phasor~) / rate (lfo~):
The base frequency in Hz. vco~ defaults to 110, lfo~ and phasor~ to 1.

pitch (vco~):
Added in octaves, exponentially.

linear fm (vco~):
Added in Hz.

width (vco~, lfo~):
Pulse width of the square wave, 0.01 to 0.99. Only the square uses it.

phase mod (vco~):
Shifts the wave's phase, in cycles - patch an audio signal here for phase 
modulation.

sync (vco~):
Hard sync: each rising edge past 0.5 restarts the cycle.

detune (vco~):
Cents between unison voices. Default 10. Does nothing with one voice.

shape:
vco~: saw, square, triangle, sine, noise, pink (the last two ignore pitch).
lfo~: sine, triangle, saw (falling), ramp (rising), square, sample_hold, 
smooth_random, noise.

depth / offset (lfo~):
Scales the output, then adds to it.

reset (lfo~, phasor~):
A rising edge past 0.5 restarts the cycle at the 'start phase' option.

period / start / end / phase (phasor~):
As described above.

enable:
Off fades out over a few milliseconds, then stops computing altogether, so 
a switched-off oscillator costs almost nothing.

Each inlet with a knob takes a number or a patched signal; where there is 
a narrow box beside the knob, that is a depth scaling whatever is patched in.

OUTPUTS: 

left out / right out (vco~), signal (lfo~):
The signal.

phase / wrap (phasor~):
The ramp, and a pulse at each turn-over.

RELATED:
sig~ brings a control value into the audio graph.
noise~ is the noise source; sampler_osc~ treats a recording as an oscillator.
additive~ builds an oscillator from a drawn spectrum instead of a chosen wave."""

demo = [
    {'key': 'lfo', 'init': 'lfo~ 5', 'pos': (30, 62), 'w': 200, 'h': 160,
     'props': {'depth': 0.05}},
    {'key': 'c0', 'comment': True, 'text': '5 Hz, depth 0.05 of an octave',
     'pos': (260, 62)},
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (30, 260), 'w': 220, 'h': 200},
    {'key': 'c1', 'comment': True, 'text': 'into pitch: vibrato', 'pos': (30, 470)},
    {'key': 'sc', 'init': 'scope~', 'pos': (300, 260), 'w': 260, 'h': 220},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 510), 'w': 220, 'h': 220,
     'props': {'fader': 0.0}},
    {'key': 'c2', 'comment': True, 'text': 'raise the fader to hear it', 'pos': (30, 745)},
    {'key': 'ph', 'init': 'phasor~ 0.5', 'pos': (300, 510), 'w': 220, 'h': 180},
    {'key': 'sn', 'init': 'snapshot~', 'pos': (300, 710), 'w': 220, 'h': 140},
    {'key': 'f1', 'init': 'float', 'pos': (300, 870), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'phasor~: 0 to 1 every two seconds',
     'pos': (300, 920)},
]
links = [('lfo', 'signal', 'vco', 'pitch'),
         ('vco', 'left out', 'sc', 'in'),
         ('vco', 'left out', 'fo', 'left'),
         ('ph', 'phase', 'sn', 'in'), ('sn', 'value', 'f1', '')]
print(build('vco~', 'vco~, lfo~, phasor~ - oscillators', body, demo, links,
            demo_width=590, text_width=820, text_height=1150))

# ------------------------------------------------------------------------ sig~
body = """sig~ brings a control value into the audio graph, gliding smoothly to 
each new value.

It is the entry point for effort data and for anything else that arrives as 
messages - a slider, a 60 Hz sensor stream, a computed number. Patched 
straight into a gain or a pitch, a stream like that steps, and you hear each 
step as a zipper or a click. sig~ turns it into a continuous signal: each 
new value is approached smoothly rather than jumped to.

GLIDE:
'glide' is in seconds: how long it takes to cover about two thirds of the 
way to a new value. It closes the rest of the gap ever more slowly, so it is 
effectively there after about three glides. The default, 0.02, is just long 
enough to stop a zipper; longer glides become audible portamento, as in the 
demo. 0 jumps straight to each value.

SCALE AND OFFSET:
The value is multiplied by 'scale' and has 'offset' added before it glides, 
so a 0 to 1 control can become, say, a -1 to 1 swing without another node.

SYNTAX:
sig~ <value>

EXAMPLE:
sig~ 0.5

INPUTS and PARAMETERS:

value:
The control value. Send it numbers, or drag it. The argument sets the 
starting value, which the output glides up to from 0 as the patch starts.

glide:
Seconds to cover about two thirds of a change. Default 0.02.

scale / offset:
The output is value times scale, plus offset. Defaults 1 and 0.

enable:
Off fades out over a few milliseconds, then stops computing.

OUTPUTS: 

signal:
The smoothed signal.

RELATED:
snapshot~ goes the other way: a signal's value back out as a number.
ramp~ moves to each value in a straight line over a set time."""

demo = [
    {'key': 'sl', 'init': 'slider 0.0', 'pos': (30, 62), 'w': 220, 'h': 60,
     'props': {'min': 0.0, 'max': 1.0, 'format': '%.2f', 'width': 200}},
    {'key': 'c0', 'comment': True, 'text': 'drag: a value stepping at frame rate',
     'pos': (30, 130)},
    {'key': 'sg', 'init': 'sig~', 'pos': (30, 170), 'w': 200, 'h': 140,
     'props': {'glide': 0.3}},
    {'key': 'c1', 'comment': True, 'text': 'glide 0.3: a slow, smooth follow',
     'pos': (30, 320)},
    {'key': 'sn', 'init': 'snapshot~', 'pos': (300, 170), 'w': 220, 'h': 140},
    {'key': 'f1', 'init': 'float', 'pos': (300, 330), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': 'the signal, read back', 'pos': (300, 380)},
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (30, 360), 'w': 220, 'h': 200},
    {'key': 'c3', 'comment': True, 'text': 'into pitch: up to an octave, gliding',
     'pos': (30, 570)},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 610), 'w': 220, 'h': 220,
     'props': {'fader': 0.0}},
    {'key': 'c4', 'comment': True, 'text': 'raise the fader to hear it', 'pos': (30, 845)},
]
links = [('sl', 'float out', 'sg', 'value'),
         ('sg', 'signal', 'sn', 'in'), ('sn', 'value', 'f1', ''),
         ('sg', 'signal', 'vco', 'pitch'),
         ('vco', 'left out', 'fo', 'left')]
print(build('sig~', 'sig~ - a control value, as a smooth signal', body, demo, links,
            demo_width=560, text_width=790, text_height=700))

# ---------------------------------------------------------------------- noise~
body = """noise~ is the noise source: a leak of air, played by pressure.

'pressure' is the whole interface. Loudness rises steeply with it, as real 
turbulence does, and brightness rises gently; at 0 there is silence. It is 
smoothed on the way in, so an effort stream patched straight to it does not 
zipper - and a body that is still is a leak that is quiet.

THE CONTROLS:
'color' tilts the noise from a dark rumble at 0 to full white at 1.

'aperture' is the size of the hole the air comes through. A small one 
whistles high and narrow; a wide one mostly breathes, low and broad. At 0 
only the whistle gets through.

'whistle' is how tight that whistle is: breathy at 0, piercing and nearly 
pure at 1.

'sputter' breaks the flow up, as if something keeps blocking the hole and 
spitting when it clears: a flutter at low values, hard dropouts near 1. 
'rate' is how often the breaks come, from slow gulps to a buzz - and harder 
pressure makes them come faster.

'level' is the output gain.

ON ITS OWN AND THROUGH THINGS:
Through formant~ it becomes breath, through modal~ a rattling surface, 
through vcf~ classic subtractive hiss.

hiss~ is the same node.

SYNTAX:
noise~ <color>

EXAMPLE:
noise~ 0.5

INPUTS and PARAMETERS:

pressure:
How hard the air is driven, 0 to 2. Default 1. 0 is silence.

color:
0 dark to 1 white. Default 0.85, or the argument.

aperture:
0 pinhole to 1 wide open. Default 0.35.

whistle:
0 breathy to 1 pure. Default 0.5.

sputter:
0 a steady flow to 1 hard dropouts. Default 0.

rate:
How often the sputter breaks come, 0.2 to 200 per second. Default 10.

level:
Output gain, 0 to 2. Default 1.

enable:
Off fades out over a few milliseconds, then stops computing.

Each knob takes a number or a patched signal; where there is a narrow box 
beside it, that is a depth scaling whatever is patched in.

OUTPUTS: 

out:
The signal.

RELATED:
vco~ also has 'noise' and 'pink' shapes: plain noise with none of these 
controls."""

demo = [
    {'key': 'sl', 'init': 'slider 1.0', 'pos': (30, 62), 'w': 220, 'h': 60,
     'props': {'min': 0.0, 'max': 2.0, 'format': '%.2f', 'width': 200}},
    {'key': 'c0', 'comment': True, 'text': 'drag: the pressure, 0 is silence',
     'pos': (30, 130)},
    {'key': 'nz', 'init': 'noise~', 'pos': (30, 170), 'w': 240, 'h': 260},
    {'key': 'sc', 'init': 'scope~', 'pos': (300, 170), 'w': 260, 'h': 220},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 460), 'w': 220, 'h': 220,
     'props': {'fader': 0.0}},
    {'key': 'c1', 'comment': True, 'text': 'raise the fader to hear it', 'pos': (30, 695)},
]
links = [('sl', 'float out', 'nz', 'pressure'),
         ('nz', 'out', 'sc', 'in'),
         ('nz', 'out', 'fo', 'left'), ('nz', 'out', 'fo', 'right')]
print(build('noise~', 'noise~ - a leak, played by pressure', body, demo, links,
            demo_width=590, text_width=790, text_height=760))

# ---------------------------------------------------------------- sampler_osc~
body = """sampler_osc~ treats a recording as an oscillator: a sound file played 
under the control of signals.

Click 'load' to choose a file, or send a file path to 'path' (a bang there 
opens the chooser too). The file is remembered with the patch. When it 
loads, the 'length' outlet sends its duration in seconds.

THE MODES:
They differ in what moves the playhead.

loop      plays round the loop window, with a short crossfade across the 
          seam so the join does not click
oneshot   plays once from the start of the window, then stops. 'start' 
          or the trigger inlet plays it again
scrub     the playhead sits wherever 'position' says, like a record moved 
          by hand. Held still it outputs a steady value, as a stopped 
          record would
follow    'position' is a target the playhead chases on a spring, so 
          movement becomes playback speed: the material sounds while the 
          control moves, and is quiet when it stops. The one for effort 
          data
granular  short overlapping grains, taken from around 'position'

SPEED AND PITCH:
In loop, oneshot and granular, playback speed is 'rate' (a plain 
multiplier: 2 is twice as fast) times 'pitch' in octaves, exactly as on 
vco~ - so the same envelope or LFO that sweeps a vco~ sweeps this the same 
way. In granular they set each grain's speed and pitch while the grains 
keep their place, so pitch and position are independent. scrub and follow 
ignore both: there 'position' alone moves the playhead.

POSITION AND THE WINDOW:
'loop start' and 'loop end' choose a window within the file, as fractions 
from 0 (the start) to 1 (the end). 'position' runs 0 to 1 across that window.

The demo is the classic scan: the file's length sets phasor~'s period, so 
its 0 to 1 ramp sweeps the scrub position through the whole file at its 
natural speed. Change phasor~'s period by hand from there and the speed 
and pitch change together, like tape.

SYNTAX:
sampler_osc~ <path> <mode>

EXAMPLE:
sampler_osc~ granular

INPUTS and PARAMETERS:

rate / pitch:
Speed as a multiplier, and transposition in octaves. Defaults 1 and 0.

position:
0 to 1 across the loop window. Used by scrub, follow and granular.

loop start / loop end:
The window, as fractions of the file. Defaults 0 and 1.

grain size / grain rate / jitter (granular):
Grain length in seconds (default 0.08), grains per second (default 20), 
and how far, as a fraction of the window, each grain may land from 
'position' at random (default 0).

trigger:
A signal; each rising edge past 0.5 restarts from the start of the window.

mode, load, path, start:
The mode; open a file chooser; load a file by path; restart.

enable:
Off fades out over a few milliseconds, then stops computing; the playhead 
waits where it was.

OPTIONS:
reverse plays backwards in loop and granular. In oneshot it plays nothing: 
the playhead starts at the beginning of the window and immediately runs 
off it.
loop crossfade is the seam's fade in seconds (default 0.005); scrub also 
uses it to smooth over jumps in position.
follow speed is how quickly follow mode's spring catches up (default 8).

OUTPUTS: 

left out / right out:
The signal. A mono file comes out of both.

length:
The loaded file's duration in seconds, sent when it loads.

RELATED:
phasor~ is the ramp for scanning position.
polyphonic_sampler~ and granular_sampler~ are the other sample players."""

demo = [
    {'key': 'so', 'init': 'sampler_osc~ scrub', 'pos': (30, 62), 'w': 260, 'h': 400},
    {'key': 'c0', 'comment': True, 'text': "click 'load' and choose a sound file",
     'pos': (320, 62)},
    {'key': 'ph', 'init': 'phasor~', 'pos': (320, 120), 'w': 220, 'h': 180},
    {'key': 'c1', 'comment': True, 'text': 'length sets the period; the ramp scans',
     'pos': (320, 310)},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 490), 'w': 220, 'h': 220,
     'props': {'fader': 0.0}},
    {'key': 'c2', 'comment': True, 'text': 'raise the fader to hear it', 'pos': (30, 725)},
]
links = [('so', 'length', 'ph', 'period'),
         ('ph', 'phase', 'so', 'position'),
         ('so', 'left out', 'fo', 'left'), ('so', 'right out', 'fo', 'right')]
print(build('sampler_osc~', 'sampler_osc~ - a recording, played as an oscillator',
            body, demo, links, demo_width=590, text_width=800, text_height=1100))

# ----------------------------------------------------------------- vcf~ family
body = """These shape a sound by what FREQUENCIES it contains, rather than by its level.

THE NODES:

vcf~       a resonant multimode filter, with per-sample cutoff modulation
formant~   a vowel, as five resonances in parallel
vowel~     the same node
vocoder~   one signal's spectrum imposed on another

vcf~ IS THE GENERAL ONE:
It has two cutoff inlets and the difference matters. 'cutoff' is in Hz. 
'tracking' is exponential, in octaves - so patching the same signal that drives 
a vco~'s pitch inlet into a vcf~'s tracking inlet makes the filter follow the 
oscillator, keeping the same tone at every pitch instead of getting duller as 
the note rises. That is what tracking is for, and it is why the inlet is 
separate.

'drive' saturates into the filter rather than after it, which is a dirtier and 
more instrument-like tone than distorting the output.

formant~ SPEAKS:
Its 'vowel' inlet runs 0 to 1 across a, e, i, o, u - and it RUNS between them 
rather than switching. The formants are interpolated as ratios, so a slow sweep 
is a mouth changing shape rather than a crossfade between two recordings of 
mouths. Patch an envelope, an lfo~, or effort data there and the sound speaks.

vocoder~ IMPOSES ONE THING ON ANOTHER:
The modulator is split into bands, each band's level is followed, and the 
carrier is passed through the same bands at those levels. The result has the 
carrier's pitch and the modulator's shape. Speech through an oscillator is the 
classic case, but nothing about the node requires either to be speech - a 
rhythmic modulator makes the carrier articulate.

SYNTAX:
vcf~ <cutoff>
formant~
vocoder~

EXAMPLE:
vcf~ 800

INPUTS and PARAMETERS:

left in / right in:
The signal to filter.

cutoff:
Where the filter acts, in Hz.

tracking:
The same, exponentially in octaves, for following a pitch.

resonance:
How much the filter emphasises its own cutoff. High enough and it rings.

drive:
Saturation into the filter.

mode:
Low pass, high pass, band pass and so on.

vowel / shift / q (formant~):
Which vowel, how the formants are shifted overall, and how sharp each 
resonance is.

modulator / left carrier / right carrier (vocoder~):
The signal whose shape is taken, and the signal it is imposed on.

attack / release / sibilance (vocoder~):
How quickly the band followers move, and how much of the modulator's high end 
is passed straight through - which is what makes consonants intelligible.

freeze:
Holds the current spectrum, so the shape stops following the modulator.

OUTPUTS: 

left out / right out:
The filtered signal.

bands (vocoder~):
The band levels, as data."""

demo = [
    {'key': 'vco', 'init': 'vco~ 110', 'pos': (30, 62), 'w': 220, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'a bright source to filter', 'pos': (30, 272)},
    {'key': 'lfo', 'init': 'lfo~ 0.3', 'pos': (300, 62), 'w': 200, 'h': 160},
    {'key': 'vcf', 'init': 'vcf~ 600', 'pos': (30, 315), 'w': 240, 'h': 220},
    {'key': 'c1', 'comment': True, 'text': 'the lfo sweeps the cutoff\nraise resonance until it rings', 'pos': (30, 545)},
    {'key': 'sc', 'init': 'scope~', 'pos': (320, 315), 'w': 260, 'h': 220},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 615), 'w': 220, 'h': 220},
]
links = [('vco', 'left out', 'vcf', 'left in'),
         ('lfo', 'signal', 'vcf', 'cutoff'),
         ('vcf', 'left out', 'sc', 'in'),
         ('vcf', 'left out', 'fo', 'left')]
print(build('vcf~', 'vcf~ - shaping by frequency', body, demo, links,
            demo_width=610, text_width=820, text_height=760))

# ------------------------------------------------------------ levels, routing
body = """These set how loud things are and where they sit.

THE NODES:

vca~        a voltage controlled amplifier
mix~        a mixer with per-input levels and a master
pan~        an equal-power panner, -1 hard left to +1 hard right
fader~      a channel fader with a desk taper, meters and a dB readout
fader_out~  a fader and an output socket in one

vca~ VERSUS mult~:
vca~ is an amplifier: it clamps negative gain to zero, and its knob SUMS with
whatever is patched to the gain inlet. The knob starts at 1, so the usual
patch is 'vca~ 0' - the knob at zero - with an adsr~ into gain. Because it
clamps, an LFO into a vca~ loses its negative half. When either signal is
bipolar - ring modulation, amplitude modulation, a shaped modulator - use
mult~ instead, which keeps it.

'response' is 'linear' (the gain as it is) or 'exponential' (the gain cubed,
which sounds closer to an even fade as an envelope closes).

mix~ IS FOR THE LEVELS:
Cords summing into one inlet already mix, so this node exists for the levels,
which are themselves modulatable. It is mono: one 'signal' outlet.

fader~ IS A DESK FADER, NOT A GAIN KNOB:
Unity sits at three quarters of the travel, with +6 dB above it and 60 dB of
dB-linear reach below, and the bottom twentieth fades to true silence.
That taper is why it feels right under the hand where a linear gain does not.
The handle is also an inlet, so automation rides the same taper. Beside the
handle are two level meters (after the fader), and a readout shows its gain in dB.
There is no bypass - the bottom of the throw is its off - but there is a mute,
which silences the channel and leaves the handle where it was, so unmuting
comes back to the balance you had.

fader_out~ exists because almost every patch begins by making a fader~ and an
audio_out~ and joining them. It is fader~ with its output going straight to
the device channels named in its arguments ('channels' option), and still
sending left and right out of its outlets, so a meter or record~ can take the
same signal. The device itself is engine-wide and is chosen on an audio_out~.
The split between fader~ and audio_out~ is still right when a patch has
several sources into one socket - this is just the common pair.

SYNTAX:
vca~ [<gain>] [linear | exponential]
mix~ [<inputs, 2 to 16>]
pan~
fader~
fader_out~ [<channel> <channel>]

EXAMPLE:
vca~ 0
mix~ 6
fader_out~ 3 4

mix~ has 4 inputs by default, fader_out~ lands on channels 1 and 2.

INPUTS and PARAMETERS:

left in / right in (vca~):
The signal. Patch the right inlet and it works in stereo, one gain curve
driving both; unpatched, 'right out' carries the same signal as 'left out'.

gain (vca~):
The knob, summed with whatever is patched here. Patch an envelope.

response (vca~):
linear or exponential.

in 1, in 2, ... / level 1, level 2, ... / master (mix~):
The inputs, their levels and the overall level, all starting at 1.

in / position (pan~):
The signal, and where it sits: -1 to +1.

left / right (fader~, fader_out~):
The signal; patch 'right' for stereo.

fader / pan / mute (fader~, fader_out~):
The handle (0 to 1, unity at 0.75), the equal-power pan (-1 to +1), and the
cut.

bypass (vca~, pan~) / enable (mix~):
The switch on the node's face.

OUTPUTS:

left out / right out (vca~, pan~):
The signal.

signal (mix~):
The mix.

left / right (fader~, fader_out~):
The signal after the fader.

RELATED:
vu~ and meter~ show the level of any signal.
mult~ for bipolar multiplication. audio_out~ is the socket on its own.
place~ positions a source among more than two speakers."""

demo = [
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (30, 62), 'w': 200, 'h': 185},
    {'key': 'adsr', 'init': 'adsr~', 'pos': (30, 290), 'w': 190, 'h': 160},
    {'key': 'c0', 'comment': True, 'text': 'tick the gate to open the envelope',
     'pos': (270, 290)},
    {'key': 'vca', 'init': 'vca~ 0', 'pos': (30, 500), 'w': 170, 'h': 130},
    {'key': 'c1', 'comment': True, 'text': 'knob at zero, envelope into gain',
     'pos': (270, 500)},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 690), 'w': 70, 'h': 310},
    {'key': 'c3', 'comment': True, 'text': 'unity sits three quarters up the throw',
     'pos': (270, 690)},
]
links = [('vco', 'left out', 'vca', 'left in'),
         ('adsr', 'signal', 'vca', 'gain'),
         ('vca', 'left out', 'fo', 'left')]
print(build('vca~', 'vca~ and the level nodes - how loud, and where', body, demo,
            links, demo_width=560, text_width=810, text_height=780))

# ----------------------------------------------------------------- vu~ / meter~
body = """vu~ shows how loud a signal is: two bars and a peak readout in dB.

THE NODES:

vu~     a level meter
meter~  the same node

vu~ IS A TAP, NOT A LINK:
It has no audio outlets at all, so the chain it reads cannot be altered by it.
Branch a cord into it from anywhere and watch. There is no on/off switch
either - bypassing a gauge would mean nothing.

WHAT THE BARS SHOW:
Each bar is one channel's average (RMS) level, moving the way a meter needle
does: quick up (about 25 ms) so a transient registers, slow down (about 300
ms) so it can be read. The scale runs from -60 dB at the left to +6 dB at the
right, coloured green to -12 dB, amber to -3 dB, red above. A thin line marks
the held peak: it stays put for most of a second, then falls back; it turns
red at 0 dB and over.

The readout under the bars is the higher of the two held peaks, in dB.

Patch only 'left in' and both bars show it, rather than one sitting dark.

fader~ and fader_out~ carry the same meters beside their handles, so a strip
needs no separate vu~.

SYNTAX:
vu~
meter~

EXAMPLE:
meter~

INPUTS and PARAMETERS:

left in / right in:
The signal to measure. Patch 'right in' for a stereo reading.

OUTPUTS:

peak:
The held peak of the louder channel, as a plain amplitude (1.0 is full
scale, 0 dB) - not in dB. Sent each frame the meter moves; nothing is sent
while it is still. For patch logic: ducking, auto-gain, a warning light.

RELATED:
fader~ and fader_out~ (on the vca~ help patch) meter what passes through them.
snapshot~ gives the value, peak and RMS of any signal as floats.
scope~ draws the waveform itself."""

demo = [
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (30, 62), 'w': 200, 'h': 185},
    {'key': 'lfo', 'init': 'lfo~ 0.5', 'pos': (30, 290), 'w': 180, 'h': 145},
    {'key': 'c0', 'comment': True, 'text': 'a slow swell', 'pos': (270, 290)},
    {'key': 'vca', 'init': 'vca~ 0', 'pos': (30, 480), 'w': 170, 'h': 130},
    {'key': 'vu', 'init': 'vu~', 'pos': (30, 670), 'w': 160, 'h': 100},
    {'key': 'c1', 'comment': True, 'text': 'a tap: it has no audio outlets',
     'pos': (270, 670)},
    {'key': 'fl', 'init': 'float', 'pos': (30, 820), 'w': 130, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': 'peak, as an amplitude',
     'pos': (270, 820)},
]
links = [('vco', 'left out', 'vca', 'left in'),
         ('lfo', 'signal', 'vca', 'gain'),
         ('vca', 'left out', 'vu', 'left in'),
         ('vu', 'peak', 'fl', '')]
print(build('vu~', 'vu~ - how loud it is', body, demo,
            links, demo_width=560, text_width=810, text_height=700))
