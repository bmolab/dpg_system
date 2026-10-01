"""adc~ / mic~ and record~: sound in, and takes kept."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build

# ------------------------------------------------------------- adc~ / record~
body = """These bring sound in from a microphone or interface, and keep what you play as a sample.

THE NODES:

adc~     an audio input device, as a signal
mic~     the same node
record~  record any signal, and save each take as a WAV file

adc~ IS THE MICROPHONE IN THE GRAPH:
Its outlets are a signal like any other ~ outlet, so a voice can go straight
into vocoder~, vcf~, vst~, string~'s excite inlet - or record~.
'channels' are the device inputs, counted from 1 the way an interface's front
panel counts them: '1 2' is a stereo pair, '1' is mono on both outlets. A
built-in microphone has one input, so '1 2' falls back to mono by itself.

Nothing reaches the speakers unless you patch it there. With speakers rather
than headphones, patching adc~ to an output is a feedback loop.

WHY NOT stream~?
t.audio_source into stream~ also brings a microphone in, but every chunk waits
for a GUI frame on the way, so a busy patch either drops audio or lets it fall
behind. adc~'s device writes straight into the audio engine and the GUI never
touches the sound - which is what makes it safe to record from.

THE HOLD ADAPTS:
'latency' is how much audio adc~ keeps in hand. A built-in microphone delivers
every 10 ms and 25 ms is plenty; some devices (BlackHole, for one) deliver in
bursts with 85 ms gaps. Each time the input runs dry the hold grows by half, up
to 250 ms, so a bursty device settles by itself within a second - usually
before you have pressed record. The status line shows what it is holding.

record~ KEEPS EVERY SAMPLE:
Tick 'record' to start and untick it to stop, or send 1 and 0. The take is
written into a buffer made at the start, by the audio engine itself, so
nothing is lost however busy the patch gets. At 'max seconds' it stops by
itself.

WHAT HAPPENS ON STOP:
The silence before the first sound and after the last is trimmed (anything
under 'threshold'), a few ms of fade go on each end so the sample never starts
or stops on a click, and the take is saved to 'folder' as
<name>_<date>_<time>.wav. Then the rate, the array and - last - the path go
out.

The path is what makes it quick. Patch it into a player and each take is
loaded the moment you stop: granular_sampler's or polyphonic_sampler's 'load'
(into the sound shown in 'sound_id'; send [sound_id, path] to choose), or
sampler_osc~'s 'path'. The players remember the file, so a saved patch keeps
the sound.

MONO OR STEREO:
Patch only 'left in' and the take is mono. Patch 'right in' as well and it is
stereo.

record~ RECORDS ANYTHING:
Not only the microphone. Patch a synth voice, a granular_sampler's outlets or
a whole mix in, and the result becomes new material - resampling.

SYNTAX:
adc~ <channel> [<channel>]
record~ <max seconds>

EXAMPLE:
adc~ 1 2
record~ 30

INPUTS and PARAMETERS:

level (adc~):
Input gain.

channels / device / latency (adc~):
Which inputs, which device (blank for the system default), and the starting
hold in ms. The device list is what was connected when the app started.

left in / right in (record~):
What to record.

record (record~):
On to record, off to stop and save.

folder / name / max seconds (record~):
Where takes go (~/dpg_takes by default), how they are named, and the longest
take.

trim silence / threshold / fade ms (record~):
The clean-up on stop. Turn trimming off to keep a take exactly as recorded,
room tone and all.

OUTPUTS:

left out / right out (adc~):
The input, as a signal.

path (record~):
The saved file, sent last - ready for a player's load inlet.

take (record~):
The samples: one row per frame, 1-D for mono, two columns for stereo.

rate (record~):
The sample rate of the take.

RELATED:
granular_sampler, polyphonic_sampler and sampler_osc~ play what you record.
stream~ and capture~ (on the snapshot~ help patch) are the general bridges
between audio and arrays."""

demo = [
    {'key': 'adc', 'init': 'adc~ 1 2', 'pos': (30, 62), 'w': 260, 'h': 200},
    {'key': 'vu', 'init': 'meter~', 'pos': (330, 62), 'w': 180, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'the microphone, as a signal\nnot sent to the speakers',
     'pos': (330, 280)},
    {'key': 'rec', 'init': 'record~ 30', 'pos': (30, 300), 'w': 260, 'h': 200},
    {'key': 'c1', 'comment': True, 'text': 'tick record, make a sound, untick\nthe take is saved and loaded below',
     'pos': (330, 330)},
    {'key': 'btn', 'init': 'button', 'pos': (30, 540), 'w': 88, 'h': 46},
    {'key': 'gs', 'init': 'granular_sampler', 'pos': (30, 610), 'w': 300, 'h': 320},
    {'key': 'c2', 'comment': True, 'text': 'click to hear the take as grains\nmove grain position to explore it',
     'pos': (380, 610)},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (380, 700), 'w': 220, 'h': 220},
    {'key': 'c3', 'comment': True, 'text': 'raise the fader to hear it',
     'pos': (380, 930)},
]
links = [('adc', 'left out', 'vu', 'left in'),
         ('adc', 'left out', 'rec', 'left in'),
         ('rec', 'path', 'gs', 'load'),
         ('btn', '', 'gs', 'trigger'),
         ('gs', 'left out', 'fo', 'left'),
         ('gs', 'right out', 'fo', 'right')]
print(build('record~', 'record~ - sound in, and takes kept', body,
            demo, links, demo_width=650, text_width=810, text_height=740))
