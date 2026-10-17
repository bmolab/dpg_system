"""t complex, integer/sign utilities, distance & similarity, normalize, energy, corrcoef, decompositions."""
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
body = """These measure how big a tensor is, how far apart two tensors are, and 
how alike their directions are.

THE NODES:

t.length             the length of a tensor, taken as one long vector - 
                     its distance from the origin
t.dist               the distance between TWO tensors - the length of the 
                     difference between them
t.cdist              the length of a single vector, by another route
t.cosine_similarity  how much two vectors point the same way, regardless of 
                     how big either is

SIZE AND DIRECTION ARE SEPARATE QUESTIONS:
t.length and t.dist answer "how big" and "how far apart". t.cosine_similarity 
answers "how alike" with size taken out of both, so two poses that are the 
same shape at different scales, or two movements that are the same gesture at 
different intensities, still read as alike.

Cosine similarity runs from 1 for pointing the same way, through 0 for at 
right angles, to -1 for opposite. It is the right comparison for anything 
where magnitude would get in the way.

THE WHOLE TENSOR IS ONE VECTOR:
t.length and t.dist do not work along an axis. Every value in the tensor is 
treated as one coordinate of a single long vector, and you get one number 
back. For one length per row, split the tensor first.

t.cdist gives the same answer as t.length for a single vector, but only for 
that: give it a tensor with more than one row and it reports an error and 
sends nothing. Use t.length.

t.cosine_similarity compares one-dimensional vectors. Give it two tables and 
it compares them column by column, which gives more than one number, and it 
reports an error and sends nothing.

SYNTAX:
t.length
t.dist
t.cdist
t.cosine_similarity

EXAMPLE:
t.dist

INPUTS and PARAMETERS:

tensor in (t.length, t.dist) / input (t.cdist):
The tensor. Receiving it triggers the node.

tensor 2 in (t.dist):
The tensor to measure the distance to. It does not trigger anything, and it 
is kept, so send it first - or wire both from the same source, with the 
cord to this inlet made first. Until it has received something t.dist sends 
nothing.

input 1 (t.cosine_similarity):
The first vector. Receiving it triggers the node.

input 2 (t.cosine_similarity):
The second vector. It must arrive afresh before EVERY input 1: the node does 
not reuse an old input 2, so sending input 1 on its own a second time 
gives nothing.

OUTPUTS: 

length (t.length) / distance (t.dist) / output (t.cdist, t.cosine_similarity):
A single number.

RELATED:
t.normalize divides a tensor by its length, leaving only its direction.
t.energy measures how much a tensor changes from value to value."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'r1', 'init': 't.rand 8', 'pos': (30, 120), 'w': 180, 'h': 180,
     'props': {'min': -1.0, 'max': 1.0}},
    {'key': 'r2', 'init': 't.rand 8', 'pos': (250, 120), 'w': 180, 'h': 180,
     'props': {'min': -1.0, 'max': 1.0}},
    {'key': 'c0', 'comment': True, 'text': 'click: two random directions', 'pos': (30, 310)},
    {'key': 'ln', 'init': 't.length', 'pos': (30, 350), 'w': 160, 'h': 80},
    {'key': 'f1', 'init': 'float', 'pos': (30, 445), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'how far from the origin', 'pos': (30, 495)},
    {'key': 'cs', 'init': 't.cosine_similarity', 'pos': (250, 350), 'w': 220, 'h': 90},
    {'key': 'f2', 'init': 'float', 'pos': (250, 455), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': '1 same way, 0 at right angles, -1 opposite',
     'pos': (250, 505)},
    {'key': 'ds', 'init': 't.dist', 'pos': (30, 560), 'w': 180, 'h': 90},
    {'key': 'f4', 'init': 'float', 'pos': (30, 665), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c4', 'comment': True, 'text': 't.dist: how far apart the two are',
     'pos': (30, 715)},
]
links = [('btn', '', 'r2', '###input'), ('btn', '', 'r1', '###input'),
         ('r1', 'random tensor', 'ln', 'tensor in'), ('ln', '', 'f1', '', 0),
         ('r2', 'random tensor', 'cs', 'input 2'),
         ('r1', 'random tensor', 'cs', 'input 1'), ('cs', 'output', 'f2', ''),
         ('r2', 'random tensor', 'ds', 'tensor 2 in'),
         ('r1', 'random tensor', 'ds', 'tensor in'), ('ds', 'distance', 'f4', '')]
print(build('t.distance', 't.length, t.dist, t.cosine_similarity - size, distance, likeness',
            body, demo, links, demo_width=520, text_width=810, text_height=860))

# ----------------------------------------------------------------- t.normalize
body = """t.normalize divides a tensor by its length, so what is left is only its 
direction.

Every result has length 1 - a unit vector - whatever size the input was. 
Two inputs that point the same way but differ in size come out identical. 
That is how you compare shapes or gestures without their intensity getting 
in the way.

IT WORKS ALONG AN AXIS:
'dim' says which axis the length is measured along, and every slice along it 
is scaled separately. On a table of rows with dim 1, each ROW becomes a unit 
vector. 

The default is 1, which suits a table of rows but not a single vector: a 
one-dimensional tensor has no axis 1, so the node reports an error and sends 
nothing. For a single vector set dim to 0 (as in the demo) or -1.

SYNTAX:
t.normalize

EXAMPLE:
t.normalize

INPUTS and PARAMETERS:

tensor in:
The tensor. Receiving it triggers the node.

dim:
The axis to measure length along. Default 1: each row of a table. 
Use 0 or -1 for a single vector.

OUTPUTS: 

normalized tensor out:
The same shape, with every slice along 'dim' scaled to length 1.

A NOTE ON NORMALIZING NEAR ZERO:
A vector whose length is nearly zero has no meaningful direction, yet 
dividing by that tiny length still produces a full-size unit vector - 
pointing wherever the noise happened to point. When normalized data jumps 
about during still moments, that is why: check the length first with 
t.length and gate on it. Only an exactly zero vector comes back as zeros.

RELATED:
t.length gives the length that is divided out here.
t.cosine_similarity compares two directions without normalizing first."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'r1', 'init': 't.rand 8', 'pos': (30, 120), 'w': 180, 'h': 180,
     'props': {'min': -1.0, 'max': 1.0}},
    {'key': 'c0', 'comment': True, 'text': 'click: eight random values', 'pos': (30, 310)},
    {'key': 'l1', 'init': 't.length', 'pos': (250, 120), 'w': 160, 'h': 80},
    {'key': 'f1', 'init': 'float', 'pos': (250, 215), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'its length before', 'pos': (250, 265)},
    {'key': 'nm', 'init': 't.normalize', 'pos': (30, 350), 'w': 200, 'h': 90,
     'props': {'dim': 0}},
    {'key': 'c2', 'comment': True, 'text': 'dim 0: one vector', 'pos': (250, 370)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 460), 'w': 208, 'h': 148,
     'props': HM(8, -1.0, 1.0)},
    {'key': 'c3', 'comment': True, 'text': 'same direction, smaller values', 'pos': (30, 620)},
    {'key': 'l2', 'init': 't.length', 'pos': (250, 460), 'w': 160, 'h': 80},
    {'key': 'f2', 'init': 'float', 'pos': (250, 555), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c4', 'comment': True, 'text': 'its length after: always 1', 'pos': (250, 605)},
]
links = [('btn', '', 'r1', '###input'),
         ('r1', 'random tensor', 'l1', 'tensor in'), ('l1', '', 'f1', '', 0),
         ('r1', 'random tensor', 'nm', 'tensor in'),
         ('nm', 'normalized tensor out', 'hm', 'y'),
         ('nm', 'normalized tensor out', 'l2', 'tensor in'), ('l2', '', 'f2', '', 0)]
print(build('t.normalize', 't.normalize - direction without size', body,
            demo, links, demo_width=470, text_width=790, text_height=660))

# -------------------------------------------------------------------- t.energy
body = """t.energy adds up how much a tensor CHANGES from one value to the next.

It takes the differences between neighbouring values, makes them all 
positive, and sums them. So it measures the total distance travelled, not 
how far from zero the values ended up. A signal that wanders a long way and 
comes back has high energy and a small final value. For asking how much 
something moved over a stretch of samples, this is the measure.

'n' takes the difference more than once before summing. With n = 1 you get 
the total change; with n = 2, the total change in the change - how much the 
movement itself speeds up and slows down, which is a measure of jerkiness 
rather than of travel.

WHICH AXIS:
With no argument, the differences run along the LAST axis. Give an axis as 
the argument - t.energy 0 - and a 'dim' inlet appears so it can be changed. 
Either way the result is one number, summed over everything.

A tensor no longer than n along that axis has no differences to take, and 
gives 0.

SYNTAX:
t.energy <dim>

EXAMPLE:
t.energy

INPUTS and PARAMETERS:

tensor in:
The tensor. Receiving it triggers the node.

n:
How many times to take the difference before summing. 1 is total change; 
2 is total change in the change. Default 1.

dim (only with an argument):
The axis to take the differences along.

OUTPUTS: 

tensor out:
A single number, as a one-value tensor.

RELATED:
t.diff gives the differences themselves, before summing.
t.length is size rather than change: the demo shows both on the same values."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'r1', 'init': 't.rand 8', 'pos': (30, 120), 'w': 180, 'h': 180,
     'props': {'min': -1.0, 'max': 1.0}},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 120), 'w': 208, 'h': 148,
     'props': HM(8, -1.0, 1.0)},
    {'key': 'c0', 'comment': True, 'text': 'click: eight values in a row', 'pos': (30, 310)},
    {'key': 'en', 'init': 't.energy', 'pos': (30, 350), 'w': 180, 'h': 110},
    {'key': 'f1', 'init': 'float', 'pos': (30, 475), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'total change along the row', 'pos': (30, 525)},
    {'key': 'ln', 'init': 't.length', 'pos': (250, 350), 'w': 160, 'h': 80},
    {'key': 'f2', 'init': 'float', 'pos': (250, 475), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': 'size, for contrast', 'pos': (250, 525)},
]
links = [('btn', '', 'r1', '###input'), ('r1', 'random tensor', 'hm', 'y'),
         ('r1', 'random tensor', 'en', 'tensor in'), ('en', 'tensor out', 'f1', ''),
         ('r1', 'random tensor', 'ln', 'tensor in'), ('ln', '', 'f2', '', 0)]
print(build('t.energy', 't.energy - how much it changes', body,
            demo, links, demo_width=480, text_width=790, text_height=640))

# ------------------------------------------------------------------- t.corrcoef
body = """t.corrcoef measures how much the rows of a tensor move TOGETHER.

Feed it a tensor whose rows are channels and whose columns are samples over 
time, and it returns a square table with one entry for every pair of 
channels: their correlation.

  1    the two rise and fall together
  0    no linear relation
 -1    one rises exactly as the other falls

The diagonal is always 1 - every channel moves exactly with itself. The table 
is symmetric, since A with B is B with A.

This is not alignment, which is what t.cosine_similarity measures. 
Correlation removes each channel's average and its scale before comparing, 
so two channels that wobble in step correlate fully even if one is far 
bigger than the other or sits far from zero. Over a long recording, the table 
is a measured picture of which things habitually happen at once.

FEW SAMPLES, LOUD CORRELATIONS:
With only a handful of columns, chance alone produces sizeable correlations - 
the demo's eight random samples per row show it. Give it many samples before 
trusting a number.

A row that never changes has no correlation with anything: its entries come 
out as NaN (not a number). A one-dimensional tensor counts as a single row 
and gives 1.

SYNTAX:
t.corrcoef

EXAMPLE:
t.corrcoef

INPUTS and PARAMETERS:

tensor in:
Rows are channels, columns are samples. Receiving it triggers the node.

OUTPUTS: 

tensor out:
The table of correlations, one row and one column per input row."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'r1', 'init': 't.rand 4 8', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'c0', 'comment': True, 'text': 'click: 4 channels, 8 samples each',
     'pos': (30, 310)},
    {'key': 'cc', 'init': 't.corrcoef', 'pos': (30, 350), 'w': 180, 'h': 70},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 440), 'w': 208, 'h': 148,
     'props': HM(4, -1.0, 1.0)},
    {'key': 'c1', 'comment': True, 'text': 'diagonal is 1; the rest is chance',
     'pos': (30, 600)},
]
links = [('btn', '', 'r1', '###input'), ('r1', 'random tensor', 'cc', 'tensor in'),
         ('cc', 'tensor out', 'hm', 'y')]
print(build('t.corrcoef', 't.corrcoef - what moves together', body,
            demo, links, demo_width=400, text_width=790, text_height=700))

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
