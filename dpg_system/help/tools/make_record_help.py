"""record~ (takes kept) and adc~ / mic~ (sound in)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build

# ---------------------------------------------------------------------- record~
body = """record~ keeps what you play: it records any signal and saves each take as a WAV file.

THE NODES:

record~  record any signal, and save each take as a WAV file

record~ KEEPS EVERY SAMPLE:
Tick 'record' to start and untick it to stop, or send 1 and 0. The take is
written into a buffer made at the start, by the audio engine itself, so
nothing is lost however busy the patch gets. At 'max seconds' it stops by
itself and the box unticks.

A patch saved in the middle of a take does not start recording again when it
is loaded. Deleting the node in the middle of a take saves what was recorded
rather than losing it.

WHAT HAPPENS ON STOP:
The silence before the first sound and after the last is trimmed (anything
quieter than 'threshold'), keeping a few ms before the onset and a twentieth
of a second after the end so a decay is not cut off. 'fade ms' of fade go on
each end so the sample never starts or stops on a click, and the take is
saved to 'folder' as <name>_<date>_<time>.wav (24-bit). Then the rate, the
array and - last - the path go out. A take that was all silence is not saved.

The path is what makes it quick. Patch it into a player and each take is
loaded the moment you stop: granular_sampler~'s or polyphonic_sampler~'s 'load'
(into the sound shown in 'sound_id'; send [sound_id, path] to choose), or
sampler_osc~'s 'path'. The players remember the file, so a saved patch keeps
the sound.

MONO OR STEREO:
Patch only 'left in' and the take is mono. Patch 'right in' as well and it is
stereo. Which one it is gets decided when recording starts.

record~ RECORDS ANYTHING:
Not only the microphone (adc~). Patch a synth voice, a granular_sampler~'s
outlets or a whole mix in, and the result becomes new material - resampling.

SYNTAX:
record~ [<max seconds>]

EXAMPLE:
record~ 30

With no argument the longest take is 60 seconds.

INPUTS and PARAMETERS:

left in / right in:
What to record.

record:
On to record, off to stop and save.

take:
The status line: the time recorded and the peak level while recording, then
the file name and length of the saved take.

folder / name / max seconds:
Where takes go (~/dpg_takes by default), how they are named ('take' by
default), and the longest take.

trim silence / threshold / fade ms:
The clean-up on stop: trimming on, a threshold of -50 dB, and 3 ms of fade by
default. Turn trimming off to keep a take exactly as recorded, room tone and
all.

OUTPUTS:

path:
The saved file, sent last - ready for a player's load inlet.

take:
The samples: 1-D for mono, two rows (left, right) for stereo - the layout every
sound array in the system uses.

rate:
The sample rate of the take.

RELATED:
adc~ (and mic~) brings a microphone into the graph.
granular_sampler~, polyphonic_sampler~ and sampler_osc~ play what you record.
capture~ hands a live signal to the patch as arrays without saving it."""

demo = [
    {'key': 'adc', 'init': 'adc~ 1 2', 'pos': (30, 62), 'w': 180, 'h': 110},
    {'key': 'c0', 'comment': True, 'text': 'the microphone, as a signal',
     'pos': (400, 62)},
    {'key': 'rec', 'init': 'record~ 30', 'pos': (30, 230), 'w': 260, 'h': 220},
    {'key': 'c1', 'comment': True, 'text': 'tick record, make a sound, untick\nthe take is saved and loaded below',
     'pos': (400, 230)},
    {'key': 'btn', 'init': 'button', 'pos': (30, 520), 'w': 45, 'h': 42},
    {'key': 'cb', 'comment': True, 'text': 'click to hear the take as grains',
     'pos': (400, 520)},
    {'key': 'gs', 'init': 'granular_sampler~', 'pos': (30, 600), 'w': 340, 'h': 540},
    {'key': 'c2', 'comment': True, 'text': 'move grain position to explore it',
     'pos': (400, 600)},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 1200), 'w': 70, 'h': 310,
     'props': {'fader': 0.0}},
    {'key': 'c3', 'comment': True, 'text': 'raise the fader to hear it',
     'pos': (400, 1200)},
]
links = [('adc', 'left out', 'rec', 'left in'),
         ('rec', 'path', 'gs', 'load'),
         ('btn', '', 'gs', 'trigger'),
         ('gs', 'left out', 'fo', 'left'),
         ('gs', 'right out', 'fo', 'right')]
print(build('record~', 'record~ - takes kept', body,
            demo, links, demo_width=680, text_width=810, text_height=740))

# ------------------------------------------------------------------ adc~ / mic~
body = """adc~ brings sound in from a microphone or an audio interface, as a signal in the ~ graph.

THE NODES:

adc~  an audio input device, as a signal
mic~  the same node

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

The device is opened at the engine's sample rate when it accepts that, so
nothing is converted; otherwise at its own rate, converted on the way in.

THE HOLD ADAPTS:
'latency' is how much audio adc~ keeps in hand. A built-in microphone delivers
every 10 ms and 25 ms is plenty; some devices (BlackHole, for one) deliver in
bursts with 85 ms gaps. Each time the input runs dry the hold grows by half, up
to 250 ms, so a bursty device settles by itself within a second - usually
before you have pressed record.

Input and output on separate devices run on separate clocks, so over minutes
one gains on the other: the stream then skips a little, or runs dry once and
refills. The status line counts both and shows what it is holding.

SYNTAX:
adc~ [<channel> [<channel>]]

EXAMPLE:
adc~ 1 2

With no argument it takes inputs 1 and 2. mic~ takes the same arguments.

INPUTS and PARAMETERS:

enable:
Off fades out over a few ms and stops the input.

level:
Input gain, 0 to 2. An inlet, so a signal can ride it; the 'level depth'
option scales whatever is patched there.

in:
The status line: the device, the channels and the rate it opened at - or why
it could not open. Runs dry and skips are added as they happen.

channels / device / latency (options):
Which inputs, which device (blank for the system default), and the starting
hold in ms (25 by default). The device list is what was connected when the
app started.

OUTPUTS:

left out / right out:
The input, as a signal. With one channel, both carry it.

RELATED:
record~ keeps what comes in as WAV takes.
stream~ plays arrays from the patch as a signal - the route for anything that
is not a live input device.
vu~ and scope~ show what is arriving."""

demo = [
    {'key': 'adc', 'init': 'adc~ 1 2', 'pos': (30, 62), 'w': 180, 'h': 110},
    {'key': 'c0', 'comment': True, 'text': 'the microphone, as a signal\nnot sent to the speakers',
     'pos': (370, 62)},
    {'key': 'vu', 'init': 'meter~', 'pos': (30, 240), 'w': 160, 'h': 100},
    {'key': 'c1', 'comment': True, 'text': 'make a sound: the level',
     'pos': (370, 240)},
    {'key': 'sc', 'init': 'scope~', 'pos': (30, 400), 'w': 310, 'h': 250},
    {'key': 'c2', 'comment': True, 'text': 'and the waveform',
     'pos': (370, 400)},
]
links = [('adc', 'left out', 'vu', 'left in'),
         ('adc', 'right out', 'vu', 'right in'),
         ('adc', 'left out', 'sc', 'in')]
print(build('adc~', 'adc~ - sound in from a microphone', body,
            demo, links, demo_width=600, text_width=810, text_height=740))
