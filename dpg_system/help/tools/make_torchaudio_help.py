"""torchaudio: sources, playback, processing, analysis."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

TWO_WORLDS = """
TWO AUDIO SYSTEMS, AND WHICH IS WHICH:
This is not the synth. The ~ nodes are a real-time DSP graph compiled into the
audio callback; these carry audio as TENSORS through ordinary patch cords, a
block at a time.

Use the ~ nodes when the audio is being made or heard live, and these when the
audio is DATA - to be analysed, transformed by a torch operation, fed to
whisper or a pitch tracker, or read from a file for something other than
listening.

capture~ is the bridge from the synth world into this one: it hands whole
blocks of a live signal out as arrays.
"""

# --------------------------------------------------------------- t.audio_source
body = """These bring audio in - from a live input, or from a file.

THE NODES:

t.audio_source        a live input: microphone or interface
t.audio.file          load a whole file at once, as one tensor
t.audio.file_stream   play a file out in real-time chunks

WHOLE FILE VERSUS STREAM, AND IT MATTERS:
t.audio.file hands you the entire file as one tensor. That is what you want for 
ANALYSIS - a whole recording you are going to measure, transform or search. 
Everything is available at once, nothing is real-time, and a long file is a 
large tensor.

t.audio.file_stream plays the file out a chunk at a time, in real time, with 
play, pause, rewind, looping and speed. Its output has the same shape as 
t.audio_source's - tensors of (channels, chunk_size) - so anything built for 
the live input works unchanged on a file.

That interchangeability is the point. Build a patch against a recording where 
you can repeat the same passage exactly, then swap in the live source with no 
other change.

CHUNK SIZE IS THE TRADE:
Small chunks mean lower latency and more frames per second for the patch to 
handle. Large chunks mean the opposite, and they also set the shortest event 
anything downstream can resolve. For analysis that is looking at a spectrum, a 
larger chunk is usually better; for anything responding to a transient, smaller.
""" + TWO_WORLDS + """
SYNTAX:
t.audio_source
t.audio.file
t.audio.file_stream

EXAMPLE:
t.audio.file_stream

INPUTS and PARAMETERS:

stream / play:
Start and stop.

path in / load file:
The file. 'load file' opens a dialog.

source / channels / sample_rate / sample format:
Which input, and its format.

chunk_size:
How many samples per block. See above.

rewind:
Back to the start.

OUTPUTS: 

audio tensors:
Blocks of shape (channels, chunk_size).

audio data out / sample_rate (t.audio.file):
The whole file, and the rate it was recorded at.

position / done (t.audio.file_stream):
Where playback has got to, and when it finishes.

A NOTE ON SAMPLE RATE:
Nothing here resamples for you. A 44100 file into an analysis node configured 
for 16000 gives an answer that is wrong by the ratio - a pitch tracker will 
report pitches almost three times too high and look plausible doing it. 
Check that the rate coming out of the source matches what the analysis expects."""

demo = [
    {'key': 'fs', 'init': 't.audio.file_stream', 'pos': (30, 62), 'w': 300, 'h': 300},
    {'key': 'c0', 'comment': True, 'text': 'load a file, then click play', 'pos': (30, 375)},
    {'key': 'c1', 'comment': True, 'text': 'same output shape as the live source,',
     'pos': (30, 405)},
    {'key': 'c2', 'comment': True, 'text': 'so they are interchangeable', 'pos': (30, 435)},
    {'key': 'as', 'init': 't.audio_source', 'pos': (380, 62), 'w': 280, 'h': 240},
    {'key': 'c3', 'comment': True, 'text': 'the live input, same tensors out',
     'pos': (380, 315)},
    {'key': 'ld', 'init': 't.audio.loudness', 'pos': (30, 480), 'w': 240, 'h': 120},
    {'key': 'f1', 'init': 'float', 'pos': (30, 615), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c4', 'comment': True, 'text': 'check the sample rates agree',
     'pos': (30, 665)},
]
links = [('fs', 'audio tensors', 'ld', 'audio tensor in'),
         ('ld', 'loudness out', 'f1', '')]
print(build('t.audio_source', 't.audio sources - live input and files', body, demo,
            links, demo_width=700, text_width=800, text_height=700))

# ------------------------------------------------------------------ t.audio.play
body = """These play audio out, and manage the device it goes to.

THE NODES:

t.audio.play         play one sound
t.audio.multiplayer  hold several sounds and play them by name
audio_mixer          which device, and which engine voices, they play through

audio_mixer IS OPTIONAL, AND THERE SHOULD BE AT MOST ONE:
These players sound through the same audio engine as the sampler and the synth 
- one stream, one device - on a pool of engine voices reserved for them (112 to 
127 by default, clear of polyphonic_sampler's range). audio_mixer chooses the 
output device, which is engine-wide (the same choice audio_out~ offers), and 
moves the voice pool. Without one, the players use those defaults.

Its 'stop all' is the panic button - everything playing stops at once. Worth 
wiring to a key or a foot button when you are working with long sounds.

t.audio.multiplayer IS FOR A SET OF SOUNDS:
Load several files, each keeps its name, and trigger them by name. That is 
better than a t.audio.play per sound once there are more than a couple - one 
node, one place to load things, and adding a sound does not mean rewiring.

'remove wave' and 'clear waves' manage the set while the patch runs.

WHEN TO USE THIS RATHER THAN THE SYNTH:
These play a tensor or a file, and that is all. They do not mix into a signal 
chain, they have no per-voice control, and they cannot be processed by ~ nodes 
on the way out.

For a sound effect fired by an event, that is exactly enough and much simpler. 
For anything that needs to be filtered, enveloped, spatialised or mixed with 
other sound, use sampler_osc~ in the synth system instead - it treats recorded 
material as a modular oscillator, with everything else available to it.
""" + TWO_WORLDS + """
SYNTAX:
t.audio.play
t.audio.multiplayer
audio_mixer

EXAMPLE:
t.audio.multiplayer

INPUTS and PARAMETERS:

trigger:
Play it.

audio tensor in:
Play a tensor directly, rather than a loaded file.

sample_rate:
The rate of a tensor arriving on 'audio tensor in'; a file carries its own. 
Playback is varispeed through the engine, so a 16 kHz tensor and a 48 kHz file 
both play at the right speed.

path in / load file:
The sound.

remove wave / clear waves (t.audio.multiplayer):
Manage the loaded set.

stop / stop all:
Stop this sound, or everything.

output device / first voice / voice count (audio_mixer):
The engine's output device, and which engine voices these players draw on.

OUTPUTS: 

None - these are endpoints.

RELATED:
audio_out~ is the synth system's output. It is the same engine and the same 
stream these players use, so both run at once, and a device chosen on either 
applies to both."""

demo = [
    {'key': 'am', 'init': 'audio_mixer', 'pos': (30, 62), 'w': 280, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'one of these in a patch, not several',
     'pos': (30, 275)},
    {'key': 'btn', 'init': 'button', 'pos': (30, 320), 'w': 88, 'h': 46},
    {'key': 'mp', 'init': 't.audio.multiplayer', 'pos': (30, 385), 'w': 300, 'h': 280},
    {'key': 'c1', 'comment': True, 'text': 'load several, trigger them by name',
     'pos': (30, 680)},
    {'key': 'c2', 'comment': True, 'text': 'adding a sound needs no rewiring',
     'pos': (30, 710)},
    {'key': 'btn2', 'init': 'button', 'pos': (380, 320), 'w': 88, 'h': 46},
    {'key': 'pl', 'init': 't.audio.play', 'pos': (380, 385), 'w': 280, 'h': 240},
    {'key': 'c3', 'comment': True, 'text': 'one sound, fired by an event',
     'pos': (380, 640)},
]
links = [('btn', '', 'mp', 'trigger'), ('btn2', '', 'pl', 'trigger')]
print(build('t.audio.play', 't.audio.play - getting sound out', body, demo, links,
            demo_width=700, text_width=800, text_height=700))

# ------------------------------------------------------------------ t.audio.gain
body = """Three ways of changing how loud something is, which are not the same thing.

THE NODES:

t.audio.gain      scale by an amount in decibels
t.audio.contrast  raise the quiet parts without raising the loud ones
t.audio.overdrive distort by driving it past what it can hold

GAIN IS IN DECIBELS, WHICH IS WHY IT FEELS RIGHT:
Doubling a signal's numbers does not sound twice as loud - hearing is 
logarithmic, and decibels are the scale that matches it. 0 dB leaves the signal 
alone, +6 doubles the amplitude, -6 halves it, and -60 is effectively silence.

Using dB rather than a multiplier is the difference between a fader that feels 
even under the hand and one that does everything in its top tenth.

CONTRAST IS NOT COMPRESSION, THOUGH IT DOES THE SAME JOB:
It increases the apparent loudness by lifting quiet passages toward the loud 
ones - so the difference between the softest and the loudest narrows and the 
whole thing sounds closer and more present.

Where a compressor does that by watching a level and reacting, this does it as 
a fixed shaping curve with no attack or release to set. Fewer controls, no 
pumping, and no way to make it respond over time.

OVERDRIVE IS A DIFFERENT KIND OF LOUD:
It adds harmonics by clipping. 'gain' drives it harder into that clipping and 
'colour' shapes the character. What comes out is not a louder version of the 
input - it is a different sound, with content the input did not have.

For a tensor you are going to analyse, that is usually a mistake: the harmonics 
overdrive adds are real to a pitch tracker or a spectrum, and were not in the 
recording.
""" + TWO_WORLDS + """
SYNTAX:
t.audio.gain <dB>
t.audio.contrast
t.audio.overdrive

EXAMPLE:
t.audio.gain 6.0

INPUTS and PARAMETERS:

audio tensor in:
The audio. Receiving it triggers the node.

gain in dB (t.audio.gain):
How much louder or quieter.

contrast (t.audio.contrast):
How much to lift the quiet parts.

gain / colour (t.audio.overdrive):
How hard into the clipping, and its character.

OUTPUTS: 

audio out:
The processed tensor, in the same shape.

RELATED:
fold~, crush~ and gain in the synth system do the equivalent jobs at audio rate 
inside the DSP graph, with the aliasing handled - see the fold~ help patch for 
why that matters for distortion in particular."""

demo = [
    {'key': 'fs', 'init': 't.audio.file_stream', 'pos': (30, 62), 'w': 300, 'h': 300},
    {'key': 'gn', 'init': 't.audio.gain 6.0', 'pos': (30, 385), 'w': 240, 'h': 120},
    {'key': 'c0', 'comment': True, 'text': 'decibels, so it feels even', 'pos': (30, 520)},
    {'key': 'ct', 'init': 't.audio.contrast', 'pos': (30, 560), 'w': 240, 'h': 120},
    {'key': 'c1', 'comment': True, 'text': 'lifts the quiet toward the loud -',
     'pos': (30, 695)},
    {'key': 'c2', 'comment': True, 'text': 'compression without the time controls',
     'pos': (30, 725)},
    {'key': 'ld', 'init': 't.audio.loudness', 'pos': (330, 385), 'w': 240, 'h': 120},
    {'key': 'p1', 'init': 'plot', 'pos': (330, 520), 'w': 208, 'h': 176,
     'props': PLOT(-60.0, 0.0)},
    {'key': 'c3', 'comment': True, 'text': 'watch the loudness change', 'pos': (330, 705)},
]
links = [('fs', 'audio tensors', 'gn', 'audio tensor in'),
         ('gn', 'audio out', 'ct', 'audio tensor in'),
         ('ct', 'audio out', 'ld', 'audio tensor in'),
         ('ld', 'loudness out', 'p1', 'y')]
print(build('t.audio.gain', 't.audio.gain - three kinds of louder', body, demo, links,
            demo_width=620, text_width=800, text_height=720))

# -------------------------------------------------------------- t.audio.loudness
body = """These measure audio rather than change it.

THE NODES:

t.audio.loudness     how loud it is, properly weighted
t.audio.kaldi_pitch  what pitch it is, and how confidently

LOUDNESS IS NOT AMPLITUDE:
The largest sample in a block tells you almost nothing about how loud it sounds. 
Hearing is far more sensitive around speech frequencies than at the extremes, so 
a quiet midrange sound can be louder to a listener than a large low rumble.

This node applies that weighting. Use it whenever the question is "how loud does 
this SEEM" - a gate, a threshold, anything driven by presence. Use a plain peak 
or RMS only when the question is genuinely about signal level, like avoiding 
clipping.

PITCH COMES WITH A CONFIDENCE, AND YOU SHOULD USE IT:
t.audio.kaldi_pitch has two outlets. 'pitch out' is the estimate. 'nccf out' is 
how strongly periodic the signal was - effectively how much to believe the 
pitch.

That second one matters more than it looks. A pitch tracker always returns a 
number, including during silence, consonants, breath and noise, where there is 
no pitch to find and the value is arbitrary. Wired straight to something, that 
produces confident nonsense in exactly the gaps between the notes.

Gate on the confidence: hold the last pitch, or output nothing, when it falls. 
That single step is the difference between a usable pitch stream and a jumpy one.

NOTE ON AVAILABILITY:
t.audio.kaldi_pitch depends on a function torchaudio removed in version 2.1. On a 
newer torchaudio the node still loads, reports this once, and sends nothing. Use 
speech_pitch (parselmouth or pyin backend) for pitch tracking there.

SAMPLE RATE MUST BE RIGHT:
Both nodes take a 'sample_rate' setting, and neither can detect it from the 
tensor - a tensor is just numbers. Set it to match the source. If it is wrong, 
loudness weighting is applied at the wrong frequencies and every pitch is off by 
the ratio, both while looking entirely plausible.
""" + TWO_WORLDS + """
SYNTAX:
t.audio.loudness
t.audio.kaldi_pitch

EXAMPLE:
t.audio.kaldi_pitch

INPUTS and PARAMETERS:

audio tensor in:
The audio. Receiving it triggers the measurement.

sample_rate:
The rate the audio was recorded at. Set it correctly - see above.

OUTPUTS: 

loudness out:
Perceptual loudness.

pitch out:
The estimated pitch.

nccf out:
How periodic the signal was - the confidence. Gate on this.

RELATED:
The speech_analysis and whisper nodes do more with the same kind of input. 
spectrum and band_pass answer "which frequencies" rather than "which pitch", 
which is a different question and often the more robust one for movement work."""

demo = [
    {'key': 'fs', 'init': 't.audio.file_stream', 'pos': (30, 62), 'w': 300, 'h': 300},
    {'key': 'kp', 'init': 't.audio.kaldi_pitch', 'pos': (30, 385), 'w': 260, 'h': 140},
    {'key': 'c0', 'comment': True, 'text': 'set sample_rate to match the file',
     'pos': (30, 540)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 580), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 400.0)},
    {'key': 'c1', 'comment': True, 'text': 'the pitch - jumpy in the gaps',
     'pos': (30, 765)},
    {'key': 'p2', 'init': 'plot', 'pos': (330, 580), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 1.0)},
    {'key': 'c2', 'comment': True, 'text': 'the confidence - low where there is',
     'pos': (330, 765)},
    {'key': 'c3', 'comment': True, 'text': 'no pitch to find. Gate on this.',
     'pos': (330, 795)},
    {'key': 'gt', 'init': '> 0.5', 'pos': (330, 385), 'w': 130, 'h': 70,
     'props': {'output_type': 'int'}},
    {'key': 'sh', 'init': 'sample_hold', 'pos': (330, 470), 'w': 150, 'h': 80},
    {'key': 'c4', 'comment': True, 'text': 'hold the last good pitch', 'pos': (500, 490)},
]
links = [('fs', 'audio tensors', 'kp', 'audio tensor in'),
         ('kp', 'pitch out', 'p1', 'y'),
         ('kp', 'nccf out', 'p2', 'y'),
         ('kp', 'nccf out', 'gt', 'in'),
         ('gt', 'result', 'sh', 'sample/hold'),
         ('kp', 'pitch out', 'sh', 'input')]
print(build('t.audio.loudness', 't.audio analysis - loudness and pitch', body, demo,
            links, demo_width=620, text_width=800, text_height=740))
