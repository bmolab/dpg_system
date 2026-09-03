"""VAE and VPoser - a small space of plausible poses."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """These compress something into a few numbers, and turn a few numbers back.

THE NODES:

vposer     a whole body pose as 32 numbers, and back again
vposer6D   the same, built on a rotation representation that behaves better
vae        the general case, for a model you trained yourself

WHAT A VARIATIONAL AUTOENCODER GIVES YOU:
An encoder squeezes the input down to a handful of numbers - the LATENT - and a
decoder builds it back. That alone would just be compression. What makes it
worth having is how it was trained: the latent space is made smooth and
continuous, so that every point in it decodes to something PLAUSIBLE, not only
the points that came from real examples.

For VPoser, trained on a very large amount of motion capture, that means any 32
numbers you can think of decode to a pose a human body could actually be in.

63 NUMBERS IN, 32 OUT, 66 BACK:
The input is 21 body joints as three numbers each. The latent is 32. What comes
back is 22 joints - the 21, plus the root - so the decoded output is (22, 3).

The root is the direction the whole body faces, which is not part of the pose in
any useful sense: the same crouch facing north and facing south is the same
crouch. So it is kept out of the encoding, and 'pass root orientation' decides
whether the incoming root is carried through to the output rather than being
whatever the decoder produced.

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

Leave it ticked unless you want that variation deliberately.

vposer6D AND WHY A ROTATION REPRESENTATION MATTERS:
Axis-angle, the three-numbers-per-joint form, has discontinuities - two nearly
identical rotations can have very different numbers - and networks learn badly
across those seams. vposer6D uses six numbers per joint instead, a form with no
such jumps, and converts back at the end.

Same idea, same latent size, generally better behaved. Use it if you have a
model in that form; the interface is identical.

vae IS THE PLAIN ONE:
Give it input size, hidden size and latent size and it is a VAE over whatever
you like. It has an extra 'distribution out' carrying the mean and spread the
encoder produced, rather than just a point drawn from them.

YOU MUST GIVE IT A MODEL PATH:
Nothing works until 'model path' points at a trained model directory - none of
these carry weights of their own. Send the path and the model loads; until then
the node sits there doing nothing, which is the usual reason one appears dead.

SYNTAX:
vposer
vposer6D
vae <input dim> <hidden> <latent dim>

EXAMPLE:
vposer

INPUTS and PARAMETERS:

input in:
A pose to encode. 63 numbers for the vposer nodes.

latent in:
32 numbers to decode into a pose. This is the generative direction.

mean of dist:
The mean, or a sample from the distribution.

pass root orientation:
Carry the incoming facing direction through to the output.

model path:
Where the trained model is. Required.

OUTPUTS: 

latents out:
The encoding, 32 numbers.

decoded out:
The pose, (22, 3).

distribution out (vae):
The mean and spread, rather than a point.

RELATED:
smpl_body will draw what comes out.
smpl_take supplies recorded poses to encode."""

demo = [
    {'key': 'mp', 'init': 'string', 'pos': (30, 62), 'w': 560, 'h': 42,
     'props': {'text in': '/path/to/vposer/V02_05', 'font size': '24',
               'width': 520}},
    {'key': 'c0', 'comment': True, 'text': 'point this at your model directory and',
     'pos': (30, 112)},
    {'key': 'c1', 'comment': True, 'text': 'click it - nothing works until you do',
     'pos': (30, 142)},

    {'key': 'take', 'init': 'smpl_take', 'pos': (30, 190), 'w': 300, 'h': 200},
    {'key': 'vp', 'init': 'vposer', 'pos': (30, 410), 'w': 320, 'h': 260},
    {'key': 'c2', 'comment': True, 'text': 'a recorded pose in, 32 numbers out',
     'pos': (30, 685)},

    {'key': 'pl', 'init': 'plot', 'pos': (400, 410), 'w': 300, 'h': 180,
     'props': PLOT(-3.0, 3.0, 32, 'stem')},
    {'key': 'c3', 'comment': True, 'text': 'the 32 latents. This is the whole pose,',
     'pos': (400, 600)},
    {'key': 'c4', 'comment': True, 'text': 'and cross-fading these gives poses all',
     'pos': (400, 630)},
    {'key': 'c5', 'comment': True, 'text': 'the way along - which cross-fading the',
     'pos': (400, 660)},
    {'key': 'c6', 'comment': True, 'text': 'joint angles does not', 'pos': (400, 690)},

    {'key': 'body', 'init': 'gl_body', 'pos': (30, 730), 'w': 280, 'h': 220},
    {'key': 'c7', 'comment': True, 'text': 'the decoded pose - the nearest one the',
     'pos': (30, 965)},
    {'key': 'c8', 'comment': True, 'text': 'model thinks a body could hold',
     'pos': (30, 995)},

    {'key': 'v32', 'init': 'vector 32', 'pos': (400, 730), 'w': 300, 'h': 180},
    {'key': 'c9', 'comment': True, 'text': 'or go the other way: 32 numbers you',
     'pos': (400, 925)},
    {'key': 'c10', 'comment': True, 'text': 'choose, decoded into a body. Anything',
     'pos': (400, 955)},
    {'key': 'c11', 'comment': True, 'text': 'you send lands on a possible pose',
     'pos': (400, 985)},
]
links = [('mp', 'string out', 'vp', 'model path'),
         ('take', 'joint_data', 'vp', 'input in'),
         ('vp', 'latents out', 'pl', 'y'),
         ('vp', 'decoded out', 'body', 'pose in'),
         ('v32', 'out', 'vp', 'latent in')]
print(build('vposer', 'VPoser - a body pose as 32 numbers', body,
            demo, links, demo_width=740, text_width=810, text_height=790))
