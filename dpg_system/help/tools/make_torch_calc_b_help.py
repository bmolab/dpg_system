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
body = """Three nodes that work on whole numbers and on signs.

THE NODES:

t.gcd        the greatest common divisor of two tensors, element by element
t.lcm        the least common multiple
t.copysign   take the size from one tensor and the sign from another

t.gcd and t.lcm want INTEGER tensors - they are number theory, not arithmetic, 
and a float tensor is the wrong kind of thing to ask them about. Their use in a 
patch is rhythmic more often than mathematical: the lowest common multiple of 
two cycle lengths is how long it takes before they line up again, which is the 
length of the combined pattern.

t.copysign IS THE USEFUL ODD ONE:
It gives you the magnitude of one tensor with the sign of another. That is how 
you apply a symmetric curve to a signal that swings both ways: take the 
absolute value, put it through the curve, then copy the original sign back. 
Without that last step a curve applied to a bipolar signal folds its negative 
half the wrong way.

It also handles negative zero correctly, which matters more than it sounds when 
the sign is carrying information about a direction that has momentarily 
stopped.

SYNTAX:
t.gcd
t.copysign

EXAMPLE:
t.copysign

INPUTS and PARAMETERS:

tensor a in / tensor b in:
The two tensors. Receiving the first triggers the node.

tensor in / sign tensor (t.copysign):
Where the magnitude comes from, and where the sign comes from.

OUTPUTS: 

out / tensor with copied sign:
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
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'ab', 'in'),
         ('ab', 'result', 'cp', 'tensor in'),
         ('rnd', 'random tensor', 'cp', 'sign tensor'),
         ('cp', 'tensor with copied sign', 'hm2', 'y')]
print(build('t.gcd', 't.gcd, t.lcm, t.copysign - whole numbers and signs', body,
            demo, links, demo_width=480, text_width=790, text_height=620))

# ------------------------------------------------------------------ t.distance
body = """These measure how big something is, or how alike two things are.

THE NODES:

t.length             the length of a tensor, taken as one long vector - 
                     its distance from the origin
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
t.cosine_similarity
t.energy <n>

EXAMPLE:
t.cosine_similarity

INPUTS and PARAMETERS:

tensor in / input 1 / input 2:
The tensors. Receiving the first triggers the node.

dim:
Which axis to measure along, where that makes sense. Without one, the whole 
tensor is treated as a single vector.

n (t.energy):
How many times to difference before summing. 1 is total movement; 
2 is total change in movement.

OUTPUTS: 

out / output:
The measurement - a single number for the length and energy nodes, 
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
]
links = [('btn', '', 'r1', '###input'), ('btn', '', 'r2', '###input'),
         ('r1', 'random tensor', 'ln', 'tensor in'), ('ln', '', 'f1', '', 0),
         ('r1', 'random tensor', 'cs', 'input 1'),
         ('r2', 'random tensor', 'cs', 'input 2'), ('cs', 'output', 'f2', ''),
         ('r1', 'random tensor', 'en', 'tensor in'), ('en', 'tensor out', 'f3', '')]
print(build('t.distance', 't distance and similarity - how big, how alike', body,
            demo, links, demo_width=520, text_width=810, text_height=760))

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
    {'key': 'c1', 'comment': True, 'text': 'the singular values, descending',
     'pos': (30, 660)},
    {'key': 'c2', 'comment': True, 'text': 'how fast they fall is how few',
     'pos': (30, 690)},
    {'key': 'c3', 'comment': True, 'text': 'dimensions the data really has',
     'pos': (30, 720)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'sv', 'tensor in'),
         ('sv', 'S tensor out', 'hm2', 'y')]
print(build('t.linalg.svd', 't decompositions - what a matrix is made of', body,
            demo, links, demo_width=480, text_width=810, text_height=780))
