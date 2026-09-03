"""t tensor makers and sequences."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

HM = lambda n=4, lo=0.0, hi=1.0, fmt='%.2f': {
    'color': 'viridis', 'width': 200, 'height': 100, 'sample count': n,
    'min y': lo, 'max y': hi, 'update_mode': 'heat_map', 'number format': fmt}

# ------------------------------------------------------------------- t.zeros
body = """These make a tensor from nothing - the starting material for tensor work.

You need a tensor before you can do anything to one, and the tensor you need is 
often regular rather than measured: a block of zeros to accumulate into, ones to 
multiply by, noise to test with, an identity matrix to start a transformation 
from.

MAKING ONE FROM A SHAPE:

t.zeros  every element zero
t.ones   every element one
t.rand   random values between limits you set
t.full   every element the same value, which you choose
t.eye    the identity matrix - ones down the diagonal, zeros elsewhere

MAKING ONE THAT MATCHES ANOTHER:

t.zeros_like  zeros, in the same shape as a tensor you give it
t.ones_like   ones, likewise
t.rand_like   random values, likewise

WHY THE _like VERSIONS EXIST:
They take their shape - and their type, and their device - from a tensor you 
hand them, rather than from settings you typed. That matters because the shape 
you need is usually the shape of something you already have, and hard-coding it 
means the patch breaks the moment the data changes size.

Anything that has to match incoming data should use a _like node. Anything that 
is a fixed part of the design can use the plain one.

THE THREE OPTIONS EVERY ONE OF THESE HAS:

dtype:
The number type. float32 is the usual choice and what most torch operations 
want; float64 for precision; the integer types for indices.

device:
Where the tensor lives - cpu, or a GPU. Two tensors on different devices cannot 
be combined, and that is the commonest error in torch work: everything must be 
moved to one device before it meets. The _like nodes inherit the device, which 
avoids the problem entirely.

requires_grad:
Whether to track operations on this tensor for automatic differentiation. 
Leave it off unless you are training something - it costs memory and time to 
record a history nobody is going to use.

SYNTAX:
t.zeros <dim> <dim> ...
t.full <value>
t.eye <n>

EXAMPLE:
t.rand 4 4

INPUTS and PARAMETERS:

input:
A button. Click it, or send anything, to produce a tensor. 
t.rand gives fresh numbers each time; the others give the same thing again.

shape:
The dimensions to produce.

tensor in (the _like nodes):
The tensor whose shape, type and device to copy.

min / max (t.rand):
The range the random values fall in.

value (t.full):
What to fill with.

n (t.eye):
The size of the identity matrix.

OUTPUTS: 

out:
The new tensor.

RELATED:
np.zeros and its family do the same for NumPy arrays. 
The t.dist nodes generate tensors from named statistical distributions rather 
than flat randomness - see the t.dist help patch, which shows all of them side 
by side."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 4 4', 'pos': (30, 120), 'w': 200, 'h': 200},
    {'key': 'hm', 'init': 'heat_map', 'pos': (270, 120), 'w': 208, 'h': 148,
     'props': HM(4)},
    {'key': 'c0', 'comment': True, 'text': 'click for fresh random numbers',
     'pos': (30, 330)},
    {'key': 'ey', 'init': 't.eye 4', 'pos': (30, 375), 'w': 200, 'h': 180},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (270, 375), 'w': 208, 'h': 148,
     'props': HM(4, 0.0, 1.0, '%.0f')},
    {'key': 'c1', 'comment': True, 'text': 'the identity: ones down the diagonal',
     'pos': (30, 565)},
    {'key': 'zl', 'init': 't.zeros_like', 'pos': (30, 610), 'w': 200, 'h': 160},
    {'key': 'hm3', 'init': 'heat_map', 'pos': (270, 610), 'w': 208, 'h': 148,
     'props': HM(4, 0.0, 1.0, '%.0f')},
    {'key': 'c2', 'comment': True, 'text': 'shape taken from the tensor, not typed',
     'pos': (30, 780)},
    {'key': 'c3', 'comment': True, 'text': 'so it follows if the data changes size',
     'pos': (30, 810)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('btn', '', 'ey', '###input'), ('ey', '', 'hm2', 'y', 0),
         ('rnd', 'random tensor', 'zl', 'tensor in'), ('zl', '', 'hm3', 'y', 0)]
print(build('t.zeros', 't tensor makers - something out of nothing', body, demo, links,
            demo_width=500, text_width=810, text_height=780))

# ---------------------------------------------------------------- t.linspace
body = """These make a sequence: evenly spaced values from one number to another.

They differ in what you specify and in how the spacing is measured, and the 
difference matters more than it looks.

THE NODES:

t.linspace  give a start, a stop and a COUNT. Both ends are included, and the 
            spacing is worked out to fit
t.logspace  the same, but the spacing is even in the logarithm - so each step is 
            the same RATIO rather than the same difference
t.arange    give a start, a stop and a STEP. The stop is not included
t.range     the same, including the stop

linspace VERSUS arange, WHICH TO REACH FOR:
If you know how many values you want, use linspace. If you know how far apart 
they should be, use arange. Asking for the wrong one is how you end up with an 
off-by-one - arange from 0 to 1 in steps of 0.1 gives ten values, not eleven, 
because the stop is excluded.

linspace is also the safer one for floating point. arange accumulates its step, 
so rounding error builds up and the last value may not be quite where you 
expect; linspace computes each position from the two ends, so the endpoints are 
exact.

WHY logspace:
Anything perceptual is logarithmic - frequency, loudness, brightness. 
Ten evenly spaced frequencies between 20 and 20000 put almost all of them in 
the top octave, where nobody can tell them apart, and none in the bass. 
Ten logarithmically spaced ones put the same number in each octave, which is 
how the ear actually divides the range.

The same is true of any quantity you are sampling across several orders of 
magnitude - a threshold sweep, a decay time, a learning rate.

SYNTAX:
t.linspace <start> <stop> <steps>
t.arange <start> <stop> <step>

EXAMPLE:
t.linspace 0.0 1.0 16

INPUTS and PARAMETERS:

input:
A button. Click it, or send anything, to produce the sequence.

start / stop:
The two ends.

steps (linspace, logspace):
How many values, both ends included.

step (arange, range):
How far apart, with the stop excluded on arange.

dtype / device / requires_grad:
As on the other tensor makers.

OUTPUTS: 

out:
The sequence, as a 1D tensor.

WHAT A SEQUENCE IS FOR:
Mostly, it is the x axis. Evaluating a curve, sampling a function, sweeping a 
parameter to see what it does, interpolating between two states in a set number 
of steps - all of them start with a sequence, and all of them go wrong at the 
ends if the endpoint handling is not what you assumed."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'ls', 'init': 't.linspace 0.0 1.0 16', 'pos': (30, 120), 'w': 220, 'h': 200,
     'props': {'start': 0.0, 'stop': 1.0, 'steps': 16}},
    {'key': 'hm', 'init': 'heat_map', 'pos': (290, 120), 'w': 208, 'h': 148,
     'props': HM(16)},
    {'key': 'c0', 'comment': True, 'text': 'evenly spaced, both ends included',
     'pos': (30, 330)},
    {'key': 'lg', 'init': 't.logspace 0.0 2.0 16', 'pos': (30, 375), 'w': 220, 'h': 200,
     'props': {'start': 0.0, 'stop': 2.0, 'steps': 16}},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (290, 375), 'w': 208, 'h': 148,
     'props': HM(16, 0.0, 100.0, '%.0f')},
    {'key': 'c1', 'comment': True, 'text': 'even in the logarithm: equal ratios',
     'pos': (30, 585)},
    {'key': 'c2', 'comment': True, 'text': 'crowded at the bottom, spread at the top',
     'pos': (30, 615)},
    {'key': 'ar', 'init': 't.arange 0.0 1.0 0.1', 'pos': (30, 660), 'w': 220, 'h': 200,
     'props': {'start': 0.0, 'stop': 1.0, 'step': 0.1}},
    {'key': 'hm3', 'init': 'heat_map', 'pos': (290, 660), 'w': 208, 'h': 148,
     'props': HM(10)},
    {'key': 'c3', 'comment': True, 'text': 'ten values, not eleven: stop excluded',
     'pos': (30, 870)},
]
links = [('btn', '', 'ls', '###input'), ('ls', '', 'hm', 'y', 0),
         ('btn', '', 'lg', '###input'), ('lg', '', 'hm2', 'y', 0),
         ('btn', '', 'ar', '###input'), ('ar', '', 'hm3', 'y', 0)]
print(build('t.linspace', 't sequences - counts, steps and ratios', body, demo, links,
            demo_width=520, text_width=810, text_height=760))
