"""t.audio_source: the microphone as torch tensors, and the general route to torch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# --------------------------------------------------------------- t.audio_source
body = """This brings a live input into the patch as torch tensors, for torch analysis.

THE NODES:

t.audio_source  a microphone or interface, as tensors

SOUND IS A ~ SIGNAL; TENSORS ARE FOR TORCH:
Everywhere else in the system, sound travels between nodes as a ~ signal:
adc~ for a microphone, the sampler and synth nodes for everything made here,
and the speech nodes, whisper~, nemotron~ and record~ all listen to signals.
Arrays and tensors appear only where audio becomes DATA for something that
wants it that way.

t.audio_source is for one such job: live input straight to torch analysis -
t.rfft, t.cwt, t.energy, a model. Its blocks are tensors of
(channels, chunk_size).

THE GENERAL ROUTE IS capture~:
adc~ -> capture~ with 'format' set to 'torch cpu' (or 'torch mps' for the
Mac's GPU) does the same for ANY ~ signal - the microphone, a synth voice, a
granular_sampler~, the whole mix - and its audio is gathered by the audio engine
itself, so a busy patch does not drop it. 'torch cpu' costs no copy: the tensor
shares the chunk's memory. Prefer it for new work; t.audio_source stays for the
patches already built on it.

CHUNK SIZE IS THE TRADE:
Small chunks mean lower latency and more frames per second for the patch to
handle. Large chunks mean the opposite, and they also set the shortest event
anything downstream can resolve. For analysis that is looking at a spectrum, a
larger chunk is usually better; for anything responding to a transient, smaller.

SYNTAX:
t.audio_source

EXAMPLE:
t.audio_source

INPUTS and PARAMETERS:

stream:
Start and stop.

source / channels / sample_rate / sample format:
Which input, and its format.

chunk_size:
How many samples per block. See above.

OUTPUTS:

audio tensors:
Blocks of shape (channels, chunk_size).

sample_rate:
The rate the stream is really running at, sent when it starts or changes.
Analysis that depends on time - a spectrum's frequency axis, a pitch - needs
it, since nothing here resamples for you.

dropped:
Blocks are handed on from the main thread, one frame behind the device. If the
patch stalls and blocks pile up past a few seconds, the newest are dropped and
counted here.

RELATED:
adc~ (record~ help patch) is the microphone as a signal; capture~ (snapshot~
help patch) turns any signal into arrays or tensors."""

PLOT_CHUNK = {'color': 'none', 'width': 200, 'height': 128, 'style': 'line',
              'update style': 'input is multi-channel sample', 'sample count': 1024,
              'min x': 0.0, 'max x': 1024.0, 'min y': -0.5, 'max y': 0.5}

demo = [
    {'key': 'adc', 'init': 'adc~', 'pos': (30, 62), 'w': 260, 'h': 200},
    {'key': 'cap', 'init': 'capture~ 1024 continuous', 'pos': (30, 290), 'w': 240, 'h': 200,
     'props': {'format': 'torch cpu'}},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 510), 'w': 208, 'h': 176, 'props': PLOT_CHUNK},
    {'key': 'c0', 'comment': True, 'text': 'any ~ signal as tensors:\nthe route for new work',
     'pos': (30, 700)},
    {'key': 'as', 'init': 't.audio_source', 'pos': (380, 62), 'w': 280, 'h': 240},
    {'key': 'p2', 'init': 'plot', 'pos': (380, 330), 'w': 208, 'h': 176, 'props': PLOT_CHUNK},
    {'key': 'c1', 'comment': True, 'text': 'tick stream: the microphone\nalone, as tensors',
     'pos': (380, 520)},
]
links = [('adc', 'left out', 'cap', 'in'), ('cap', 'array', 'p1', 'y'),
         ('as', 'audio tensors', 'p2', 'y')]
print(build('t.audio_source', 't.audio_source - the microphone as tensors', body, demo,
            links, demo_width=700, text_width=800, text_height=700))
