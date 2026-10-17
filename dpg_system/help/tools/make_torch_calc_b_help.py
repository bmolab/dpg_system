"""t complex, integer/sign utilities, distance & similarity, decompositions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

HM = lambda n=4, lo=0.0, hi=1.0, fmt='%.2f': {
    'color': 'viridis', 'width': 200, 'height': 100, 'sample count': n,
    'min y': lo, 'max y': hi, 'update_mode': 'heat_map', 'number format': fmt}

# ------------------------------------------------------------------ t.complex
body = """These three take complex tensors apart and put them back together.

A complex number is two numbers travelling as one - a real part and an 
imaginary part, or equivalently a magnitude and an angle. You meet them 
wherever something has both a size and a phase, which in practice means 
anywhere a Fourier transform has been.

THE NODES:

t.complex   build a complex tensor from a real part and an imaginary part
t.real      take the real part out
t.imag      take the imaginary part out

WHERE THIS COMES UP:
t.fft returns complex output, and most of what you want to do next needs it 
split. The magnitude - how much of each frequency is present - is the usual 
answer, and the phase is where the timing lives. Splitting with these, working 
on the parts, and rebuilding with t.complex is how you modify a spectrum and 
transform it back.

MAGNITUDE IS NOT t.real:
The real part of a complex number is not its size. A component can have a large 
magnitude and a real part near zero, if its phase happens to sit near a quarter 
turn. If what you want is "how much of this frequency is there", that is the 
magnitude - the length of the two parts taken together - not the real part 
alone. Taking t.real and calling it the spectrum is a common and quiet mistake.

SYNTAX:
t.complex
t.real
t.imag

EXAMPLE:
t.real

INPUTS and PARAMETERS:

real tensor in / imag tensor in (t.complex):
The two parts. They must be the same shape.

tensor in (t.real, t.imag):
The complex tensor to split.

OUTPUTS: 

complex tensor out:
The assembled complex tensor.

out:
The chosen part, as an ordinary real tensor.

RELATED:
See the t.fft help patch for where complex tensors come from."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'r1', 'init': 't.rand 8', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'r2', 'init': 't.rand 8', 'pos': (250, 120), 'w': 180, 'h': 180},
    {'key': 'c0', 'comment': True, 'text': 'a real part and an imaginary part',
     'pos': (30, 310)},
    {'key': 'cx', 'init': 't.complex', 'pos': (30, 350), 'w': 200, 'h': 90},
    {'key': 'rl', 'init': 't.real', 'pos': (30, 460), 'w': 160, 'h': 70},
    {'key': 'im', 'init': 't.imag', 'pos': (250, 460), 'w': 160, 'h': 70},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 550), 'w': 208, 'h': 148,
     'props': HM(8)},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (270, 550), 'w': 208, 'h': 148,
     'props': HM(8)},
    {'key': 'c1', 'comment': True, 'text': 'split apart again, unchanged',
     'pos': (30, 710)},
]
links = [('btn', '', 'r1', '###input'), ('btn', '', 'r2', '###input'),
         ('r1', 'random tensor', 'cx', 'real tensor in'),
         ('r2', 'random tensor', 'cx', 'imag tensor in'),
         ('cx', 'complex tensor out', 'rl', 'tensor in'),
         ('cx', 'complex tensor out', 'im', 'tensor in'),
         ('rl', '', 'hm', 'y', 0), ('im', '', 'hm2', 'y', 0)]
print(build('t.complex', 't.complex - two numbers travelling as one', body, demo,
            links, demo_width=500, text_width=790, text_height=640))

# ---------------------------------------------------------------------- t.gcd
body = """Two number-theory nodes for whole-number tensors, element by element.

THE NODES:

t.gcd   the greatest common divisor - the largest number that divides both
t.lcm   the least common multiple - the smallest number both divide into

Their use in a patch is rhythmic more often than mathematical: the least common
multiple of two cycle lengths is how long it takes before they line up again,
which is the length of the combined pattern. The greatest common divisor is the
largest step that lands on the beats of both.

THEY WORK ON WHOLE NUMBERS:
These are number theory, not arithmetic. Both inlets turn whatever arrives - a
list, a number, a float or integer tensor - into whole numbers before the work
is done, so a list of counts can go straight in.

A value with a fractional part is rounded to the NEAREST whole number first:
2.6 counts as 3, and 9.4 as 9. That keeps a value like 11.9999 that should have
been 12 from quietly turning into 11. If you need a different rule (always
down, always up), put t.floor or t.ceil in front.

A zero is allowed: the gcd of a number and 0 is the number itself, and the lcm
of anything with 0 is 0.

SYNTAX:
t.gcd
t.lcm

INPUTS and PARAMETERS:

tensor a in:
The first tensor, list or number. Receiving it does the work.

tensor b in:
The second tensor, or a single number applied everywhere. It is only stored
when it arrives, so send it BEFORE tensor a.

OUTPUTS:

gcd tensor out / lcm tensor out:
The result, element by element, as whole numbers."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'mb', 'init': 'message', 'pos': (300, 118), 'w': 220, 'h': 42,
     'props': {'text in': '8 12 21 0', 'font size': '24'}},
    {'key': 'ma', 'init': 'message', 'pos': (30, 118), 'w': 220, 'h': 42,
     'props': {'text in': '12 18 7 9', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'b goes first: it is only stored', 'pos': (300, 168)},
    {'key': 'gc', 'init': 't.gcd', 'pos': (30, 205), 'w': 200, 'h': 80},
    {'key': 'l1', 'init': 'list', 'pos': (30, 300), 'w': 220, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'gcd: 4 6 7 9', 'pos': (30, 350)},
    {'key': 'lc', 'init': 't.lcm', 'pos': (300, 205), 'w': 200, 'h': 80},
    {'key': 'l2', 'init': 'list', 'pos': (300, 300), 'w': 220, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'lcm: 24 36 21 0', 'pos': (300, 350)},
]
links = [('btn', '', 'mb', ''), ('btn', '', 'ma', ''),
         ('mb', 'message out', 'gc', 'tensor b in'),
         ('mb', 'message out', 'lc', 'tensor b in'),
         ('ma', 'message out', 'gc', 'tensor a in'),
         ('ma', 'message out', 'lc', 'tensor a in'),
         ('gc', 'gcd tensor out', 'l1', ''), ('lc', 'lcm tensor out', 'l2', '')]
print(build('t.gcd', 't.gcd and t.lcm - whole numbers in common', body,
            demo, links, demo_width=560, text_width=790, text_height=620))

# ----------------------------------------------------------------- t.copysign
body = """Takes the size from one tensor and the sign from another.

WHAT IT IS FOR:
Applying a symmetric curve to a signal that swings both ways. Take the absolute
value, put it through the curve, then copy the original sign back. Without that
last step a curve applied to a bipolar signal folds its negative half the wrong
way - everything comes out positive.

The example does the first and last steps with nothing in between, so you can
see that the original values come back exactly.

NEGATIVE ZERO COUNTS AS NEGATIVE:
The sign is read from the sign bit, so a sign value of -0.0 makes the result
negative. That matters more than it sounds when the sign is carrying information
about a direction that has momentarily stopped.

SEND THE SIGN FIRST:
'sign tensor' is only stored when it arrives; 'tensor in' is what makes the
node work, and it uses whatever sign tensor it last received. If both come from
the same source, connect the sign cord first so it gets there first -
otherwise every result carries the signs from the time before.

SYNTAX:
t.copysign

INPUTS and PARAMETERS:

tensor in:
Where the size comes from. Receiving it does the work.

sign tensor:
Where the sign comes from - a tensor of the same shape, or a single number
applied everywhere.

OUTPUTS:

tensor with copied sign:
The result, in the same shape."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 8', 'pos': (30, 120), 'w': 180, 'h': 180,
     'props': {'min': -1.0, 'max': 1.0}},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 120), 'w': 208, 'h': 148,
     'props': HM(8, -1.0, 1.0)},
    {'key': 'c0', 'comment': True, 'text': 'values with mixed signs', 'pos': (30, 310)},
    {'key': 'ab', 'init': 'abs', 'pos': (30, 350), 'w': 160, 'h': 70},
    {'key': 'c1', 'comment': True, 'text': 'sign discarded, magnitude kept',
     'pos': (30, 430)},
    {'key': 'cp', 'init': 't.copysign', 'pos': (30, 475), 'w': 200, 'h': 90},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (30, 585), 'w': 208, 'h': 148,
     'props': HM(8, -1.0, 1.0)},
    {'key': 'c2', 'comment': True, 'text': 'and the original signs put back',
     'pos': (30, 745)},
]
# the sign cord is connected before the abs cord, so the signs arrive first
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'cp', 'sign tensor'),
         ('rnd', 'random tensor', 'ab', 'in'),
         ('ab', 'result', 'cp', 'tensor in'),
         ('cp', 'tensor with copied sign', 'hm2', 'y')]
print(build('t.copysign', 't.copysign - size from one, sign from another', body,
            demo, links, demo_width=480, text_width=790, text_height=600))

# ------------------------------------------------------------------ t.distance
body = """These measure how big something is, or how alike two things are.

THE NODES:

t.length             the length of a tensor, taken as one long vector - 
                     its distance from the origin
t.dist               the distance between TWO tensors - the length of the 
                     difference between them
t.cdist              the same measurement, by another route
t.normalize          divide by that length, giving a unit vector
t.cosine_similarity  how much two tensors point the same way, regardless of 
                     how big either is
t.corrcoef           the correlation between rows - how much they move together
t.energy             the total absolute CHANGE across a tensor

SIZE AND DIRECTION ARE SEPARATE QUESTIONS:
t.length answers the first, t.normalize removes it so only direction is left, 
and t.cosine_similarity compares two directions with size taken out of both. 
That separation is usually what you want: two poses can be the same shape at 
different scales, and two movements the same gesture at different intensities.

cosine similarity runs from 1 for pointing the same way, through 0 for at right 
angles, to -1 for opposite. It is the right comparison for anything where 
magnitude is a confound.

t.corrcoef IS DIFFERENT:
It asks whether things move TOGETHER over the samples you give it - correlation, 
not alignment. Feed it a tensor whose rows are channels and whose columns are 
time and you get back a matrix of how much each pair of channels co-varies. 
That is the measured prior for which things habitually happen at once.

t.energy IS MOTION, NOT SIZE:
It differences along an axis, takes the absolute value, and sums - so it is the 
total distance travelled rather than how far from the origin the tensor ended 
up. A signal that wanders a long way and comes back has high energy and low 
length. For asking how much a body moved, this is the measure, not t.length.

SYNTAX:
t.length
t.dist
t.cosine_similarity
t.energy <n>

EXAMPLE:
t.cosine_similarity

INPUTS and PARAMETERS:

tensor in / input 1 / input 2:
The tensors. Receiving the first triggers the node.

tensor 2 in (t.dist):
The tensor to measure the distance to. It does not trigger anything, so send 
it before the first - or wire both from the same trigger, the second inlet 
first. Until it has received something t.dist sends nothing.

dim:
Which axis to measure along, where that makes sense. Without one, the whole 
tensor is treated as a single vector.

n (t.energy):
How many times to difference before summing. 1 is total movement; 
2 is total change in movement.

OUTPUTS: 

length (t.length) / distance (t.dist) / output / tensor out:
The measurement - a single number for length, distance and energy, 
a tensor for normalize and corrcoef.

A NOTE ON NORMALIZING NEAR ZERO:
A tensor whose length is nearly zero has no meaningful direction, and dividing 
by that length produces a large and arbitrary unit vector. When normalized data 
starts jumping about during the still moments, that is why - clamp the length 
away from zero, or gate on it."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'r1', 'init': 't.rand 8', 'pos': (30, 120), 'w': 180, 'h': 180,
     'props': {'min': -1.0, 'max': 1.0}},
    {'key': 'r2', 'init': 't.rand 8', 'pos': (250, 120), 'w': 180, 'h': 180,
     'props': {'min': -1.0, 'max': 1.0}},
    {'key': 'c0', 'comment': True, 'text': 'two random directions', 'pos': (30, 310)},
    {'key': 'ln', 'init': 't.length', 'pos': (30, 350), 'w': 160, 'h': 80},
    {'key': 'f1', 'init': 'float', 'pos': (30, 445), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'how far from the origin', 'pos': (30, 495)},
    {'key': 'cs', 'init': 't.cosine_similarity', 'pos': (250, 350), 'w': 220, 'h': 90},
    {'key': 'f2', 'init': 'float', 'pos': (250, 455), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': '1 same way, 0 at right angles, -1 opposite',
     'pos': (250, 505)},
    {'key': 'en', 'init': 't.energy', 'pos': (30, 540), 'w': 180, 'h': 110},
    {'key': 'f3', 'init': 'float', 'pos': (30, 665), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'total movement, not total size',
     'pos': (30, 715)},
    {'key': 'ds', 'init': 't.dist', 'pos': (30, 760), 'w': 180, 'h': 90},
    {'key': 'f4', 'init': 'float', 'pos': (30, 865), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c4', 'comment': True, 'text': 't.dist: how far apart the two are',
     'pos': (30, 915)},
]
links = [('btn', '', 'r2', '###input'), ('btn', '', 'r1', '###input'),
         ('r1', 'random tensor', 'ln', 'tensor in'), ('ln', '', 'f1', '', 0),
         ('r1', 'random tensor', 'cs', 'input 1'),
         ('r2', 'random tensor', 'cs', 'input 2'), ('cs', 'output', 'f2', ''),
         ('r1', 'random tensor', 'en', 'tensor in'), ('en', 'tensor out', 'f3', ''),
         ('r2', 'random tensor', 'ds', 'tensor 2 in'),
         ('r1', 'random tensor', 'ds', 'tensor in'), ('ds', 'distance', 'f4', '')]
print(build('t.distance', 't distance and similarity - how big, how alike', body,
            demo, links, demo_width=520, text_width=810, text_height=860))

# --------------------------------------------------------------- decompositions
body = """These break a matrix apart into the pieces it is made of.

A matrix is a transformation. These nodes answer what that transformation is 
actually DOING - which directions it stretches, which it leaves alone, and how 
much of it could be thrown away without much being lost.

THE NODES:

t.linalg.svd             singular value decomposition
t.linalg.pca_low_rank    principal components, found approximately and quickly
t.linalg.eig             eigenvalues and eigenvectors
t.linalg.qr              QR decomposition

SVD IS THE ONE TO KNOW:
It splits any matrix into a rotation, a set of scalings, and another rotation. 
The scalings - the singular values - come out in descending order, and how 
quickly they fall away tells you how much of the data lives in how few 
dimensions. If the first three are large and the rest are near zero, your 
high-dimensional data is really three-dimensional with noise on top.

That is what makes it the tool for asking "how many things are actually varying 
here". For a set of poses, or joint angles over time, the answer is usually far 
fewer than the number of channels.

pca_low_rank IS SVD WHEN YOU ONLY WANT THE TOP FEW:
Computing a full decomposition of a large matrix is expensive and mostly wasted 
if you are going to keep six components. This finds the leading ones directly. 
It also centres the data first by default, which is what makes it principal 
component analysis rather than a bare decomposition - and centring matters, 
because otherwise the first component just points at where the data sits rather 
than how it varies.

eig FINDS THE DIRECTIONS THAT DO NOT TURN:
An eigenvector is a direction the matrix only stretches, without rotating, 
and its eigenvalue is by how much. For a covariance matrix those directions are 
the axes of variation. Note that eigenvalues can be complex even for real 
matrices, which is the transformation containing a rotation.

qr SPLITS INTO A ROTATION AND A TRIANGLE:
Mostly a building block - it is how least-squares problems get solved stably - 
but Q is also a convenient way to get a set of mutually perpendicular directions 
out of a set of arbitrary ones.

SYNTAX:
t.linalg.svd
t.linalg.pca_low_rank

EXAMPLE:
t.linalg.svd

INPUTS and PARAMETERS:

tensor in:
The matrix. Receiving it triggers the node.

full (t.linalg.svd, pca_low_rank):
Whether to compute the complete decomposition or only the part that carries 
information. The reduced form is smaller and faster and is usually what you want.

center (pca_low_rank):
Subtract the mean first. Leave it on unless you specifically want the 
decomposition of the raw data.

mode (t.linalg.qr):
Complete or reduced.

OUTPUTS: 

S tensor out:
The singular values - the amount of variation along each component, 
in descending order. This is the outlet to look at first; its shape tells you 
how many dimensions your data really has.

U / V / D tensor out:
The component directions.

L / V tensor out (eig):
The eigenvalues and eigenvectors.

Q / R tensor out (qr):
The perpendicular directions and the triangular factor."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 6 6', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 120), 'w': 208, 'h': 148,
     'props': HM(6)},
    {'key': 'c0', 'comment': True, 'text': 'a 6 by 6 matrix', 'pos': (30, 310)},
    {'key': 'sv', 'init': 't.linalg.svd', 'pos': (30, 350), 'w': 200, 'h': 130},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (30, 500), 'w': 208, 'h': 148,
     'props': HM(6, 0.0, 4.0)},
    {'key': 'c1', 'comment': True, 'text': 'the singular values, descending\nhow fast they fall is how few\ndimensions the data really has',
     'pos': (30, 660)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'sv', 'tensor in'),
         ('sv', 'S tensor out', 'hm2', 'y')]
print(build('t.linalg.svd', 't decompositions - what a matrix is made of', body,
            demo, links, demo_width=480, text_width=810, text_height=780))
