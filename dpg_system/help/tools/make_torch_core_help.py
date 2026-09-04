"""tensor creation and conversion, inspection, buffers."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

HM = lambda n=8, lo=0.0, hi=1.0, fmt='%.2f': {
    'color': 'viridis', 'width': 200, 'height': 100, 'sample count': n,
    'min y': lo, 'max y': hi, 'update_mode': 'heat_map', 'number format': fmt}

# --------------------------------------------------------------------- tensor
body = """These get data into torch, and move it between types and devices.

THE NODES:

tensor     turn anything into a torch tensor
t.to       change a tensor's dtype, device or gradient tracking
t.detach   cut a tensor loose from its gradient history

THE THREE THINGS A TENSOR CARRIES BESIDES ITS NUMBERS:

dtype - what kind of number. float32 is what most torch operations expect;
float64 for precision; the integer types for indices and labels.

device - where it lives. cpu, or a GPU. **Two tensors on different devices
cannot be combined**, and that is the commonest error in torch work by a wide
margin. Everything meeting in one operation must be on one device.

requires_grad - whether operations on it are recorded for automatic
differentiation. Off unless you are training something; it costs memory and time
to keep a history nobody will use.

t.to CHANGES ANY OF THE THREE:
It is the general converter. Moving a tensor to a GPU, casting to float32 for an
operation that demands it, or turning gradient tracking on - all the same node.

t.detach CUTS THE HISTORY:
A tensor produced inside a gradient-tracked computation drags its whole history
along with it. Take such a tensor into the rest of a patch - to plot it, to send
it over OSC, to store it - and you keep that history alive, holding memory and
sometimes producing errors about tensors that need grad.

detach gives you the same numbers with the history removed. Anything leaving a
training computation for the ordinary patch world should go through it.

MOST NODES CONVERT FOR YOU:
The torch nodes in this system generally accept lists and NumPy arrays and
convert on the way in, so an explicit tensor node is often unnecessary. Use it
when you want the conversion to happen once rather than at every node, or when
you need to set the dtype or device deliberately.

SYNTAX:
tensor
t.to
t.detach

EXAMPLE:
t.to

INPUTS and PARAMETERS:

in / tensor in:
The data. Receiving it triggers the node.

dtype / device / requires_grad:
The three properties. See above.

OUTPUTS: 

tensor out:
The converted tensor.

RELATED:
t.info reports what a tensor actually is, which is how you find out that a
device mismatch is what you are looking at.
np.astype does the dtype half of this for NumPy arrays."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 240, 'h': 42,
     'props': {'text in': '1 2 3 4', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'an ordinary list', 'pos': (30, 168)},
    {'key': 'tn', 'init': 'tensor', 'pos': (30, 205), 'w': 240, 'h': 180},
    {'key': 'inf', 'init': 't.info', 'pos': (30, 405), 'w': 240, 'h': 140},
    {'key': 's1', 'init': 'string', 'pos': (30, 560), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 's2', 'init': 'string', 'pos': (250, 560), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'shape and dtype', 'pos': (30, 615)},
    {'key': 'to', 'init': 't.to', 'pos': (30, 655), 'w': 240, 'h': 180},
    {'key': 'inf2', 'init': 't.info', 'pos': (30, 855), 'w': 240, 'h': 140},
    {'key': 's3', 'init': 'string', 'pos': (30, 1010), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'set dtype on t.to and watch it change',
     'pos': (30, 1060)},
]
links = [('btn', '', 'm1', ''), ('m1', 'message out', 'tn', 'in'),
         ('tn', 'tensor out', 'inf', 'in'),
         ('inf', 'shape', 's1', ''), ('inf', 'dtype', 's2', ''),
         ('tn', 'tensor out', 'to', 'in'),
         ('to', 'tensor out', 'inf2', 'in'), ('inf2', 'dtype', 's3', '')]
print(build('tensor', 'tensor and t.to - getting data into torch', body, demo, links,
            demo_width=500, text_width=800, text_height=740))

# --------------------------------------------------------------------- t.info
body = """These tell you what a tensor actually is, which is how most torch problems 
get diagnosed.

THE NODES:

t.info           shape, dtype, device and gradient tracking, on four outlets
t.numel          how many elements it holds in total
t.is_contiguous  whether its memory is laid out straight
t.contiguous     make it so

t.info IS THE FIRST THING TO REACH FOR:
When an operation refuses a tensor, the reason is almost always one of the four
things this reports. The shape is wrong, the dtype is wrong, it is on the wrong
device, or it is carrying gradient history it should not be.

Patch it in beside whatever is failing. It costs nothing and it answers the
question directly, where guessing from the error message often does not.

CONTIGUITY, AND WHY t.view FAILS:
A tensor's numbers live in a flat block of memory, and its shape is a way of
reading that block. Transposing or permuting changes the reading without moving
anything - which is why those operations are free at any size - but it leaves a
tensor whose memory is no longer in the order its shape implies.

That is non-contiguous. Most operations do not care. t.view does: it requires
that the new shape can be read from the existing memory without moving anything,
and after a transpose it cannot, so it fails.

t.contiguous makes a copy laid out straight, after which view works. t.reshape
does this for you when it has to, which is why reshape always works and view
sometimes does not - see the t.reshape help patch.

t.is_contiguous answers the question without changing anything, which is worth
using before inserting a t.contiguous that may be copying a large tensor for no
reason.

t.numel:
The total element count - the product of the shape. Useful for checking that a
reshape is possible before attempting it, since the dimensions must multiply to
this.

SYNTAX:
t.info
t.contiguous

EXAMPLE:
t.info

INPUTS and PARAMETERS:

in / tensor in:
The tensor. Receiving it triggers the node.

OUTPUTS: 

shape / dtype / device / grad (t.info):
The four things worth knowing, separately so each can be tested or displayed.

numel out:
The total element count.

result (t.is_contiguous):
True or false.

tensor out (t.contiguous):
A contiguous copy - or the same tensor unchanged, if it already was."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 3 4', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'ic', 'init': 't.is_contiguous', 'pos': (250, 120), 'w': 220, 'h': 90},
    {'key': 'i1', 'init': 'int', 'pos': (250, 225), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c0', 'comment': True, 'text': 'fresh from t.rand: contiguous',
     'pos': (250, 280)},
    {'key': 'tr', 'init': 't.t', 'pos': (30, 320), 'w': 140, 'h': 70},
    {'key': 'ic2', 'init': 't.is_contiguous', 'pos': (250, 320), 'w': 220, 'h': 90},
    {'key': 'i2', 'init': 'int', 'pos': (250, 425), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 'after a transpose: not contiguous\nthis is why t.view fails there',
     'pos': (250, 480)},
    {'key': 'ct', 'init': 't.contiguous', 'pos': (30, 550), 'w': 200, 'h': 90},
    {'key': 'ic3', 'init': 't.is_contiguous', 'pos': (250, 550), 'w': 220, 'h': 90},
    {'key': 'i3', 'init': 'int', 'pos': (250, 655), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c3', 'comment': True, 'text': 'and contiguous again after a copy',
     'pos': (250, 710)},
    {'key': 'nf', 'init': 't.info', 'pos': (30, 680), 'w': 200, 'h': 140},
    {'key': 's1', 'init': 'string', 'pos': (30, 835), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'ic', 'tensor in'), ('ic', 'result', 'i1', ''),
         ('rnd', 'random tensor', 'tr', 'tensor in'),
         ('tr', 'tensor out', 'ic2', 'tensor in'), ('ic2', 'result', 'i2', ''),
         ('tr', 'tensor out', 'ct', 'tensor in'),
         ('ct', 'tensor out', 'ic3', 'tensor in'), ('ic3', 'result', 'i3', ''),
         ('ct', 'tensor out', 'nf', 'in'), ('nf', 'shape', 's1', '')]
print(build('t.info', 't.info - what a tensor actually is', body, demo, links,
            demo_width=500, text_width=800, text_height=740))

# ------------------------------------------------------------------- t.buffer
body = """These collect a stream of tensors over time into one tensor.

A stream gives you one value per frame, and most of what you want to ask needs
several - an average, a spectrum, a trend, a picture of the recent past.
These hold that history.

THE NODES:

t.buffer          collect samples in one of several ways
t.rolling_buffer  keep the last N, always

t.rolling_buffer IS THE ONE YOU USUALLY WANT:
Feed it a value or a small tensor each frame and it hands back the last N of
them stacked together. That is what turns a stream into something the
statistics, filter and analysis nodes can work on - almost anything you want to
know about "the recent past" starts here.

It only reports what it has actually collected, so a freshly reset buffer is
short until it fills. That is deliberate: it avoids handing you a stretch of
zeros that never happened. It does mean the buffer's length changes for the
first N frames, and anything assuming a fixed size will complain during that
window.

t.buffer HAS SEVERAL FILL MODES:
'update style' decides how incoming data lands - replacing the buffer wholesale,
or filling it circularly one sample at a time. The circular mode is the rolling
behaviour; the replace mode is for when each message is already a complete
buffer and you just want to hold it.

'sample to output' lets you read one position rather than the whole thing.

SAMPLE COUNT IS A TIME WINDOW:
At 60 frames a second, a count of 60 is one second of history, 600 is ten. Think
in seconds rather than samples and the setting stops being arbitrary - it is
"how much of the past should this measurement be about".

Longer is steadier and slower to react. That is the same trade every filter
makes, in a different form.

SYNTAX:
t.rolling_buffer <count>
t.buffer <count>

EXAMPLE:
t.rolling_buffer 60

INPUTS and PARAMETERS:

input:
The value or tensor arriving each frame.

sample count:
How many to keep. See above - this is a time window.

update style:
How incoming data fills the buffer.

sample to output (t.buffer):
Read one position instead of the whole buffer.

reset:
Empty it and start collecting again.

OUTPUTS: 

output:
The collected tensor.

RELATED:
np.rolling_buffer does the same for NumPy arrays.
capture~ fills a buffer from the audio graph at audio rate, which is the right
node when the stream is a signal rather than control data."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'met', 'init': 'metro 30', 'pos': (30, 112), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 30.0, 'units': 'milliseconds'}},
    {'key': 'rnd', 'init': 'random 1.0', 'pos': (30, 192), 'w': 140, 'h': 80,
     'props': {'range': 1.0, 'bipolar': False}},
    {'key': 'c0', 'comment': True, 'text': 'one value per frame', 'pos': (30, 280)},
    {'key': 'rb', 'init': 't.rolling_buffer 64', 'pos': (30, 320), 'w': 260, 'h': 200},
    {'key': 'c1', 'comment': True, 'text': 'at 30ms a frame, 64 is about 2 seconds\nthink in seconds, not samples',
     'pos': (30, 535)},
    {'key': 'mn', 'init': 't.mean', 'pos': (30, 605), 'w': 180, 'h': 100},
    {'key': 'f1', 'init': 'float', 'pos': (30, 720), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'now the stream can be averaged',
     'pos': (30, 770)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (330, 605), 'w': 208, 'h': 148,
     'props': HM(64)},
    {'key': 'c4', 'comment': True, 'text': 'the recent past, as one tensor',
     'pos': (330, 765)},
]
links = [('tog', '', 'met', 'on'), ('met', '', 'rnd', 'trigger'),
         ('rnd', 'out', 'rb', 'input'),
         ('rb', 'output', 'mn', 'tensor in'), ('mn', 'output', 'f1', ''),
         ('rb', 'output', 'hm', 'y')]
print(build('t.buffer', 't.rolling_buffer - the recent past, as a tensor', body,
            demo, links, demo_width=580, text_width=800, text_height=720))
