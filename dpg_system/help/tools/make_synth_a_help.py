"""audio_out~ (the framework), sources, filters, levels."""
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
    {'key': 'c1', 'comment': True, 'text': 'fader_out~ is a fader and the socket',
     'pos': (30, 780)},
    {'key': 'c2', 'comment': True, 'text': 'raise its fader to hear it', 'pos': (30, 810)},
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
body = """These are the sources: the nodes that make a signal rather than change one.

THE NODES:

vco~          a band-limited oscillator, with detuned unison. The workhorse
phasor~       a plain 0 to 1 ramp, for driving positions rather than for hearing
lfo~          a low frequency oscillator - though nothing stops it reaching 
              audio rate, where it becomes an unbandlimited modulator for FM and AM
noise~        a leak, played by pressure - the rack's noise source
hiss~         the same node
sig~          a control value brought into the audio graph, with glide
sampler_osc~  recorded material treated as an oscillator

PITCH IS TWO INLETS:
vco~ and sampler_osc~ both take a base 'frequency' in Hz and an exponential 
'pitch' inlet in octaves that scales it. Patch an envelope into pitch for a 
sweep, an LFO for vibrato. Because it is exponential, the same modulation 
produces the same musical interval at any base frequency - which is why the 
two are separate, and why the identical signal that sweeps a vco~ sweeps a 
sampler_osc~ the same way.

There is a third, 'linear fm', in Hz rather than octaves. Linear FM detunes 
rather than transposes, which is what makes it inharmonic and metallic.

sig~ IS THE IMPORTANT ONE HERE:
It is the entry point for effort data. A 60 Hz control stream stepping a gain 
directly will zipper - you hear each step. sig~ glides between the values, 
smoothing them into a continuous signal without ever lagging more than the 
glide time behind.

SYNTAX:
vco~ <frequency>
lfo~ <rate>
sig~ <value>

EXAMPLE:
vco~ 220

INPUTS and PARAMETERS:

frequency / rate:
The base pitch in Hz, or the LFO's rate.

pitch:
Exponential, in octaves. Adds to the base frequency musically.

linear fm:
Linear, in Hz. Adds to it arithmetically.

shape:
The waveform.

width:
Pulse width, for shapes that have one.

detune (vco~):
Spreads the unison voices apart. A little is thickness; a lot is a chorus.

sync / phase mod:
Hard sync and phase modulation inputs.

value / glide (sig~):
The control value, and how long it takes to arrive there.

pressure / color / sputter (noise~):
How hard the leak is blowing, how dark it is, and how irregular. 
Stillness is silence by construction.

OUTPUTS: 

left out / right out / signal / out:
The signal.

RELATED:
additive~ builds an oscillator from a drawn spectrum instead of a chosen wave."""

demo = [
    {'key': 'sl', 'init': 'slider 0.0', 'pos': (30, 62), 'w': 220, 'h': 60,
     'props': {'min': 0.0, 'max': 1.0, 'format': '%.2f', 'width': 200}},
    {'key': 'c0', 'comment': True, 'text': 'a control value, stepping at frame rate',
     'pos': (30, 130)},
    {'key': 'sg', 'init': 'sig~', 'pos': (30, 170), 'w': 200, 'h': 140},
    {'key': 'c1', 'comment': True, 'text': 'sig~ glides it into a smooth signal',
     'pos': (30, 320)},
    {'key': 'vco', 'init': 'vco~ 110', 'pos': (30, 360), 'w': 220, 'h': 200},
    {'key': 'c2', 'comment': True, 'text': 'into pitch: exponential, so octaves',
     'pos': (30, 570)},
    {'key': 'sc', 'init': 'scope~', 'pos': (300, 360), 'w': 260, 'h': 220},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 610), 'w': 220, 'h': 220},
    {'key': 'c3', 'comment': True, 'text': 'raise the fader to hear it', 'pos': (30, 845)},
]
links = [('sl', 'float out', 'sg', 'value'),
         ('sg', 'signal', 'vco', 'pitch'),
         ('vco', 'left out', 'sc', 'in'),
         ('vco', 'left out', 'fo', 'left')]
print(build('vco~', 'vco~ and the other sources - making a signal', body, demo, links,
            demo_width=590, text_width=820, text_height=800))

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
    {'key': 'c1', 'comment': True, 'text': 'the lfo sweeps the cutoff', 'pos': (30, 545)},
    {'key': 'c2', 'comment': True, 'text': 'raise resonance until it rings',
     'pos': (30, 575)},
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
body = """These set how loud things are and where they sit, and let you see it.

THE NODES:

vca~        a voltage controlled amplifier
mix~        a mixer with per-input levels and a master
pan~        an equal-power panner, -1 hard left to +1 hard right
fader~      a channel fader with a desk taper and a dB readout
fader_out~  a fader and an output socket in one
meter~      a level meter
vu~         the same node

vca~ VERSUS mult~:
vca~ is an amplifier: it clamps negative gain, and its knob SUMS with whatever 
is patched to the gain inlet. So the usual patch is the knob at zero with an 
adsr~ into gain. But because it clamps, an LFO into a vca~ loses its negative 
half. When either signal is bipolar - ring modulation, amplitude modulation, 
a shaped modulator - use mult~ instead, which keeps it.

fader~ IS A DESK FADER, NOT A GAIN KNOB:
Unity sits at three quarters of the travel, with +6 dB above it and 60 dB of 
dB-linear reach below, and the bottom twentieth fades to true silence. 
That taper is why it feels right under the hand where a linear gain does not. 
The handle is also an inlet, so automation rides the same taper.

fader_out~ exists because almost every patch begins by making a fader~ and an 
audio_out~ and joining them. The split between them is still right when a 
patch has several sources into one socket - this is just the common pair.

meter~ IS A TAP, NOT A LINK:
It has no audio outlets at all, so the chain it reads cannot be altered by it. 
Branch a cord into it and watch.

SYNTAX:
vca~
mix~ <count: int>
fader_out~ <channel> <channel>

EXAMPLE:
mix~ 6

INPUTS and PARAMETERS:

left in / right in:
The signal. Patch the right inlet and these work in stereo, one gain or 
position driving both.

gain (vca~):
Summed with the knob. Patch an envelope here.

response (vca~):
The curve of the gain - linear or something more perceptual.

in 1, in 2, ... / master (mix~):
The inputs and the overall level. Cords summing into one inlet already mix, 
so this node exists for the levels, which are themselves modulatable.

position (pan~):
-1 to +1.

fader / pan / mute (fader~, fader_out~):
The handle, the position and the cut.

OUTPUTS: 

left out / right out:
The signal.

peak (meter~):
The level, in dB, so the patch can see what a person sees."""

demo = [
    {'key': 'vco', 'init': 'vco~ 220', 'pos': (30, 62), 'w': 220, 'h': 200},
    {'key': 'adsr', 'init': 'adsr~', 'pos': (300, 62), 'w': 220, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'tick the gate to open the envelope',
     'pos': (300, 272)},
    {'key': 'vca', 'init': 'vca~', 'pos': (30, 315), 'w': 220, 'h': 160},
    {'key': 'c1', 'comment': True, 'text': 'knob at zero, envelope into gain',
     'pos': (30, 485)},
    {'key': 'mt', 'init': 'meter~', 'pos': (300, 315), 'w': 180, 'h': 200},
    {'key': 'c2', 'comment': True, 'text': 'a tap: it has no audio outlets',
     'pos': (300, 525)},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 545), 'w': 220, 'h': 220},
    {'key': 'c3', 'comment': True, 'text': 'unity sits three quarters up the throw',
     'pos': (30, 780)},
]
links = [('vco', 'left out', 'vca', 'left in'),
         ('adsr', 'signal', 'vca', 'gain'),
         ('vca', 'left out', 'mt', 'left in'),
         ('vca', 'left out', 'fo', 'left')]
print(build('vca~', 'vca~ and the level nodes - how loud, and where', body, demo,
            links, demo_width=560, text_width=810, text_height=780))
