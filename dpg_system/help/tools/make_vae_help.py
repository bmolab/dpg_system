"""VAE and VPoser - a small space of plausible poses."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ---------------------------------------------------------------------- vposer
body = """These turn a whole body pose into 32 numbers, and 32 numbers back into a pose.

THE NODES:

vposer     a body pose as 32 numbers, and back again
vposer6D   the same, for a model whose encoder works in a rotation 
           representation that behaves better

WHAT A VARIATIONAL AUTOENCODER GIVES YOU:
An encoder squeezes the input down to a handful of numbers - the LATENT - and a
decoder builds it back. That alone would just be compression. What makes it
worth having is how it was trained: the latent space is made smooth and
continuous, so that every point in it decodes to something PLAUSIBLE, not only
the points that came from real examples.

For VPoser, trained on a very large amount of motion capture, that means any 32
numbers you can think of decode to a pose a human body could actually be in.

WHAT GOES IN AND WHAT COMES OUT:
The pose is SMPL axis-angle, three numbers per joint. The encoder takes the 21 
body joints - 63 numbers. The latent is 32. What comes back is 22 joints - the 
root, then the 21 - so the decoded output is 22 rows of three.

You can send 63 numbers, or 22 joints (66 numbers, root first), or a whole 
SMPL-H pose of 52 joints, whose 30 hand joints are dropped. When the root is 
there it is taken off before encoding.

The root is the direction the whole body faces, which is not part of the pose in
any useful sense: the same crouch facing north and facing south is the same
crouch. So it is kept out of the encoding. 'pass root orientation' puts the 
incoming root back on the decoded pose; unticked, or when the input had no 
root, the decoded root is all zeros. A pose decoded from 'latent in' always has 
a zero root.

THE ROUND TRIP IS NOT LOSSLESS, AND THAT IS THE POINT:
Send a pose in and take the decoded pose out and it will NOT be the same pose.
It will be the nearest pose the model considers plausible.

Measured here, putting an arbitrary made-up pose through repeatedly and asking
how far each pass moved it:

    pass 1   0.4540 rad
    pass 2   0.1008
    pass 3   0.0649
    pass 4   0.0372

The first pass does nearly all the work, and after that it barely moves,
because one pass was enough to land it somewhere the model is happy with.

That is what makes this a POSE PRIOR rather than a compressor. A pose from a
noisy sensor, or one with an impossible joint angle, comes back cleaned up -
not smoothed in time, but corrected towards anatomical sense.

THE THINGS TO DO WITH IT:

Clean up a pose. One pass through removes the impossible parts.

Interpolate between poses. Encode two, cross-fade the 32 numbers, decode - and
everything in between is a real pose. Cross-fading the joint angles directly
does not give you that; it passes through positions no body could take.

Make poses from very little. 32 numbers is few enough to drive from anything -
a sensor, a signal, a hand on a fader - and whatever you send decodes to a body.

Note that 32 zeros is NOT a neutral standing pose. It is the middle of the
learned space, which is a particular pose the model settled on; here it decodes
to joint values spanning about -0.9 to +1.0. There is no "empty" latent.

'mean of dist' - WHETHER IT ANSWERS THE SAME WAY TWICE:
The encoder does not produce a point, it produces a small cloud - a mean and a
spread. Ticked, you get the mean, and the same pose always gives the same 32
numbers. Unticked, you get a sample from the cloud, so the same pose gives
slightly different numbers each time and the decoded pose wobbles.

It starts ticked on vposer and UNTICKED on vposer6D. Tick it unless you want 
that variation deliberately.

vposer6D AND WHY A ROTATION REPRESENTATION MATTERS:
Axis-angle, the three-numbers-per-joint form, has discontinuities - two nearly
identical rotations can have very different numbers - and networks learn badly
across those seams. Both nodes already DECODE through a six-numbers-per-joint 
form that has no such jumps, and convert to axis-angle at the end. vposer6D 
also converts the incoming pose to that form before ENCODING it.

You send and receive exactly the same things; the difference is the model 
inside. A vposer6D needs a model trained that way, and a vposer one trained on 
axis-angle - the two kinds of model are not interchangeable.

YOU MUST GIVE IT A MODEL PATH:
Until 'model path' points at a trained model, the node runs an untrained 
network, and what comes out is meaningless - which is the usual reason one 
seems to produce nonsense. Send the path of the model's directory (for 
example V02_05): it must hold a 'snapshots' folder of .ckpt files, of which the 
newest is used, and the model's .yaml settings file.

THE SIZES:
The two optional arguments are the network width and the latent size, 512 and 
32 by default, and both must be given. The loaded model brings its own sizes, 
but 'latent in' is cut into rows of the node's latent size, so give the 
arguments if your model's latent is not 32.

SYNTAX:
vposer [width latent size]
vposer6D [width latent size]

EXAMPLE:
vposer

INPUTS and PARAMETERS:

input in:
A pose to encode - 63, 66 or 156 numbers.

latent in:
32 numbers to decode into a pose. This is the generative direction.

mean of dist:
The mean, or a sample from the distribution.

pass root orientation:
Put the incoming root back on the decoded pose.

model path:
The trained model directory. Required.

OUTPUTS: 

latents out:
The encoding, 32 numbers. On 'latent in' it repeats what was sent.

decoded out:
The pose, 22 rows of three axis-angle numbers, root first.

Both send torch tensors.

RELATED:
vae is the same idea for a model you trained yourself, on anything.
smpl_take supplies recorded poses to encode.
smpl_to_active turns the decoded pose into one gl_body can draw."""

demo = [
    {'key': 'mp', 'init': 'string', 'pos': (30, 62), 'w': 420, 'h': 42,
     'props': {'text in': '/path/to/vposer/V02_05', 'font size': '24',
               'width': 380}},
    {'key': 'c0', 'comment': True, 'text': 'point this at your model directory and\nclick it - until you do, the output\nis meaningless',
     'pos': (30, 62)},
    {'key': 'take', 'init': 'smpl_take', 'pos': (30, 160), 'w': 125, 'h': 120},
    {'key': 'c1', 'comment': True, 'text': 'recorded SMPL poses', 'pos': (30, 160)},
    {'key': 'v32', 'init': 'vector 32', 'pos': (30, 330), 'w': 160, 'h': 520},
    {'key': 'c9', 'comment': True, 'text': 'or go the other way: 32 numbers you\nchoose, decoded into a body. Anything\nyou send lands on a possible pose',
     'pos': (30, 330)},
    {'key': 'vp', 'init': 'vposer', 'pos': (30, 900), 'w': 155, 'h': 120},
    {'key': 'c2', 'comment': True, 'text': 'a pose in, 32 numbers out\nand the nearest plausible pose back',
     'pos': (30, 900)},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 1070), 'w': 210, 'h': 180,
     'props': PLOT(-3.0, 3.0, 32, 'stem')},
    {'key': 'c3', 'comment': True, 'text': 'the 32 latents. This is the whole pose,\nand cross-fading these gives poses all\nthe way along - which cross-fading the\njoint angles does not',
     'pos': (30, 1070)},
    {'key': 'sa', 'init': 'smpl_to_active', 'pos': (30, 1300), 'w': 200, 'h': 100},
    {'key': 'c4', 'comment': True, 'text': 'SMPL joints to the 20 active joints',
     'pos': (30, 1300)},
    {'key': 'body', 'init': 'gl_body', 'pos': (30, 1450), 'w': 145, 'h': 160},
    {'key': 'c7', 'comment': True, 'text': 'the decoded pose - the nearest one the\nmodel thinks a body could hold',
     'pos': (30, 1450)},
]
links = [('mp', 'string out', 'vp', 'model path'),
         ('take', 'joint_data', 'vp', 'input in'),
         ('v32', 'out', 'vp', 'latent in'),
         ('vp', 'latents out', 'pl', 'y'),
         ('vp', 'decoded out', 'sa', 'smpl pose'),
         ('sa', 'active pose', 'body', 'pose in')]
print(build('vposer', 'VPoser - a body pose as 32 numbers', body,
            demo, links, demo_width=480, text_width=810, text_height=1600))

# ------------------------------------------------------------------------- vae
body = """A variational autoencoder for a model you trained yourself, on anything.

THE NODES:

vae  encode to a few numbers and decode back, with a network of the sizes 
     you give it

WHAT A VARIATIONAL AUTOENCODER GIVES YOU:
An encoder squeezes the input down to a handful of numbers - the LATENT - and a
decoder builds it back. What makes it worth having is how it was trained: the 
latent space is made smooth and continuous, so every point in it decodes to 
something like the training examples, not only the points that came from 
real ones.

THE NETWORK:
The three arguments are the input size, the hidden size and the latent size - 
784, 512 and 4 by default, the shape for 28 by 28 pixel images. The encoder 
narrows from the input through the hidden size, a half, a quarter and an eighth 
of it, to a mean and a spread for each latent number. The decoder runs the 
same path the other way and ends in a sigmoid, so every decoded value is 
between 0 and 1.

Anything sent in is cut into rows of the input size (or, for 'latent in', the 
latent size).

YOU MUST GIVE IT A MODEL PATH:
The node does not train. 'model path' takes the path of a FILE of saved weights 
- a torch state dictionary - for a network of exactly these three sizes. Until 
one is loaded the node runs an untrained network and its output is 
meaningless.

THREE WAYS IN:

input in:
Encode, take the MEAN of what the encoder produces, and decode that. The same 
input always gives the same output. Only the first row of a batch is used.

forward in:
The full training-style pass: encode, draw a random SAMPLE from the encoder's 
distribution, decode it. The same input gives a slightly different result each 
time. Every row of a batch is used.

latent in:
Decode latent numbers you supply - the generative direction.

SYNTAX:
vae [input size] [hidden size] [latent size]

EXAMPLE:
vae 784 512 4

INPUTS and PARAMETERS:

input in / forward in / latent in:
As above.

model path:
The saved weights file. Required.

OUTPUTS: 

distribution out:
The mean of the encoder's distribution - on 'input in' only. Despite the name, 
not the spread; it carries the same numbers as 'latents out' there.

latents out:
The latent numbers: the mean for 'input in', the sample for 'forward in', and 
what was sent for 'latent in'.

decoded out:
The decoded output, values between 0 and 1.

All three send torch tensors.

RELATED:
vposer and vposer6D are trained models of the same kind, for body poses."""

demo = [
    {'key': 'mp', 'init': 'string', 'pos': (30, 62), 'w': 420, 'h': 42,
     'props': {'text in': '/path/to/vae_weights.pt', 'font size': '24',
               'width': 380}},
    {'key': 'c0', 'comment': True, 'text': 'point this at your weights file and\nclick it - until you do, the output\nis meaningless',
     'pos': (30, 62)},
    {'key': 'v4', 'init': 'vector 4', 'pos': (30, 160), 'w': 160, 'h': 120},
    {'key': 'c1', 'comment': True, 'text': 'four latent numbers to decode', 'pos': (30, 160)},
    {'key': 'va', 'init': 'vae 784 512 4', 'pos': (30, 330), 'w': 160, 'h': 200},
    {'key': 'c2', 'comment': True, 'text': '784 in, 4 latent numbers, 784 back',
     'pos': (30, 330)},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 580), 'w': 210, 'h': 180,
     'props': PLOT(0.0, 1.0, 784)},
    {'key': 'c3', 'comment': True, 'text': 'the decoded values, all between 0 and 1',
     'pos': (30, 580)},
]
links = [('mp', 'string out', 'va', 'model path'),
         ('v4', 'out', 'va', 'latent in'),
         ('va', 'decoded out', 'pl', 'y')]
print(build('vae', 'vae - a variational autoencoder of your own', body,
            demo, links, demo_width=480, text_width=810, text_height=1155))
