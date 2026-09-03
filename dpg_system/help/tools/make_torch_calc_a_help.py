"""t comparisons, cumulative, rounding, clamping."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

HM = lambda n=4, lo=0.0, hi=1.0, fmt='%.2f': {
    'color': 'viridis', 'width': 200, 'height': 100, 'sample count': n,
    'min y': lo, 'max y': hi, 'update_mode': 'heat_map', 'number format': fmt}

# ------------------------------------------------------------- t comparisons
body = """These compare two tensors element by element and give back a tensor of 
true and false.

The result is not one answer but a whole tensor of them, the same shape as the 
inputs, holding the outcome at every position. That is what makes these useful 
in tensor work: the result is a MASK, and a mask is how you say "these elements, 
not those" to everything that follows.

THE NODES:

t.gt   greater than
t.ge   greater than or equal to
t.lt   less than
t.le   less than or equal to
t.eq   equal to
t.ne   not equal to

WHAT TO DO WITH A MASK:
Multiply by it and everything that failed the test becomes zero. 
Sum it and you get a COUNT of how many passed. Take its mean and you get the 
PROPORTION that passed, which is often the number you actually wanted. 
Feed it to a selection node and you get the values themselves.

Note that the result is a BOOL tensor, and torch will not average one - 
t.mean on it raises "could not infer output dtype". Put a t.to set to float 
in between and the mean works. Summing and multiplying are fine as they are; 
it is averaging that needs the cast.

BROADCASTING:
The two tensors do not have to be the same shape. A single number compared 
against a tensor is compared against every element, and shapes that differ in a 
dimension of length 1 are stretched to match. That is how "every value above 
0.5" and "every row above its own threshold" are both written the same way.

COMPARING FLOATS FOR EQUALITY:
t.eq tests exact equality, and two floats that ought to be the same after 
arithmetic very often are not - the result of a calculation differs from the 
number you would type by a fraction too small to see. If a t.eq is returning 
false when you are sure it should not, that is usually why; compare the 
absolute difference against a small tolerance instead.

SYNTAX:
t.gt
t.eq

EXAMPLE:
t.gt

INPUTS and PARAMETERS:

tensor a in:
The tensor to test. Receiving it triggers the comparison.

tensor b in:
What to test it against - another tensor, or a single number applied everywhere.

OUTPUTS: 

out:
A tensor of true and false values, the same shape as the inputs.

RELATED:
The plain comparison nodes do the same for ordinary numbers and NumPy arrays, 
and let you choose whether the answer comes out as a bool, an int or a float."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 4 4', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 120), 'w': 208, 'h': 148,
     'props': HM(4)},
    {'key': 'c0', 'comment': True, 'text': 'click for a fresh 4 by 4 tensor',
     'pos': (30, 310)},
    {'key': 'gt', 'init': 't.gt', 'pos': (30, 350), 'w': 160, 'h': 80},
    {'key': 'c1', 'comment': True, 'text': 'compared against 0.5 everywhere',
     'pos': (30, 440)},
    {'key': 'm1', 'init': 'message', 'pos': (250, 350), 'w': 100, 'h': 42,
     'props': {'text in': '0.5', 'font size': '24'}},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (30, 480), 'w': 208, 'h': 148,
     'props': HM(4, 0.0, 1.0, '%.0f')},
    {'key': 'c2', 'comment': True, 'text': 'the mask: 1 where it passed', 'pos': (30, 640)},
    {'key': 'tf', 'init': 't.to', 'pos': (280, 480), 'w': 160, 'h': 120,
     'props': {'dtype': 'float32'}},
    {'key': 'mn', 'init': 't.mean', 'pos': (280, 615), 'w': 160, 'h': 100},
    {'key': 'f1', 'init': 'float', 'pos': (280, 730), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'cast to float, then the mean is the',
     'pos': (280, 780)},
    {'key': 'c4', 'comment': True, 'text': 'proportion that passed the test',
     'pos': (280, 810)},
]
links = [('btn', '', 'rnd', '###input'), ('btn', '', 'm1', ''),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('m1', 'message out', 'gt', 'tensor b in'),
         ('rnd', 'random tensor', 'gt', 'tensor a in'),
         ('gt', '', 'hm2', 'y', 0),
         # a bool mask cannot be averaged -- cast it first
         ('gt', '', 'tf', 'in', 0),
         ('tf', 'tensor out', 'mn', 'tensor in'), ('mn', 'output', 'f1', '')]
print(build('t.comparison', 't comparisons - a tensor of answers, not one', body,
            demo, links, demo_width=480, text_width=810, text_height=740))

# ------------------------------------------------------------- t.cumsum family
body = """These look ALONG a tensor and report how it develops, rather than summarising it.

A sum gives you the total. A cumulative sum gives you the total so far at every 
position - the running story rather than the ending. Anything with an order to 
it, which is anything indexed by time or by position along a body, has a 
development worth seeing.

THE NODES:

t.cumsum         the running total
t.cumprod        the running product
t.cummax         the largest value seen so far - a running high-water mark
t.cummin         the smallest so far
t.logcumsumexp   a running sum done in the log domain
t.diff           the difference between neighbouring elements

t.cumsum AND t.diff UNDO EACH OTHER:
diff turns positions into steps; cumsum turns steps back into positions. 
Reach for diff when a tensor holds where something WAS and you want how far it 
MOVED, and cumsum when you have the movements and want the path.

t.cummax IS A DECAY-FREE PEAK HOLD:
Running it along a time axis gives you, at every instant, the largest value 
that has occurred up to then. That is the shape of a peak meter that never 
falls back, and it is a cheap way to ask "has this ever exceeded X, and when 
did it first do so".

WHY logcumsumexp EXISTS:
Adding up many small probabilities directly underflows to zero. Working in the 
log domain keeps them representable, but you cannot simply add logs when you 
want a sum of the underlying values. This node does that correctly, and it is 
the node you want whenever you are accumulating likelihoods.

SYNTAX:
t.cumsum
t.diff <n>

EXAMPLE:
t.cumsum

INPUTS and PARAMETERS:

tensor in:
The tensor. Receiving it triggers the node.

dim:
Which axis to run along. This matters more here than on a reduction - the 
direction IS the ordering you are claiming exists, so getting it wrong gives a 
running total across the wrong thing entirely.

n (t.diff):
How many times to difference. 1 gives the change, 2 the change in the change - 
so from positions, 1 is velocity and 2 is acceleration.

OUTPUTS: 

output:
The running result.

indices (t.cummax, t.cummin):
WHERE each running maximum or minimum came from - the position of the value, 
not just its size.

A NOTE ON LENGTH:
t.diff returns a tensor one element shorter along the differenced axis, 
because there is one fewer gap than there are values. Anything expecting the 
original length will refuse it."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 8', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 120), 'w': 208, 'h': 148,
     'props': HM(8)},
    {'key': 'c0', 'comment': True, 'text': 'eight values in a row', 'pos': (30, 310)},
    {'key': 'cs', 'init': 't.cumsum', 'pos': (30, 350), 'w': 180, 'h': 100},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (250, 350), 'w': 208, 'h': 148,
     'props': HM(8, 0.0, 5.0)},
    {'key': 'c1', 'comment': True, 'text': 'the running total: it only climbs',
     'pos': (30, 465)},
    {'key': 'df', 'init': 't.diff', 'pos': (30, 510), 'w': 180, 'h': 120},
    {'key': 'hm3', 'init': 'heat_map', 'pos': (250, 530), 'w': 208, 'h': 148,
     'props': HM(7, -1.0, 1.0)},
    {'key': 'c2', 'comment': True, 'text': 'the steps between them - seven, not eight',
     'pos': (30, 645)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'cs', 'tensor in'), ('cs', 'output', 'hm2', 'y'),
         ('rnd', 'random tensor', 'df', 'tensor in'), ('df', 'tensor out', 'hm3', 'y')]
print(build('t.cumsum', 't.cumsum and t.diff - the running story', body, demo, links,
            demo_width=480, text_width=810, text_height=740))

# ------------------------------------------------------------ rounding family
body = """These turn tensors of fractions into whole numbers, in the several different 
ways that can mean.

THE NODES:

t.round   to the nearest whole number
t.floor   downward, towards negative
t.ceil    upward, towards positive
t.trunc   towards zero, dropping the fractional part
t.frac    the opposite - keep only the fractional part, discard the whole

THEY DIFFER ONLY FOR NEGATIVES:
Given 2.5 they mostly agree. Given -2.5, floor gives -3, ceil gives -2, and 
trunc gives -2. That is the distinction: floor always goes down the number 
line, trunc always goes towards zero, and for positive numbers those are the 
same direction. Almost every bug involving these is a negative value going the 
way you did not expect.

t.frac IS THE PHASE:
Whatever whole number a value has passed, frac tells you how far it is into the 
next one. Feed it a value that climbs steadily and you get a sawtooth from 0 to 
1 - which is how you make a repeating phase out of a counter, or a position 
within a cell out of a position in space.

ROUNDING TO THE NEAREST EVEN:
t.round rounds half-way cases to the nearest EVEN number, so 0.5 becomes 0 and 
1.5 becomes 2. That is deliberate and standard - always rounding halves upward 
introduces a small bias that accumulates over many values - but it surprises 
people who expect 0.5 to become 1.

SYNTAX:
t.round
t.floor

EXAMPLE:
t.round

INPUTS and PARAMETERS:

tensor in:
The tensor. Receiving it triggers the node.

decimals (t.round):
How many decimal places to keep. The default, 0, rounds to whole numbers; 
2 rounds to hundredths. Negative values round to tens and hundreds.

OUTPUTS: 

output:
The rounded tensor, in the same shape.

RELATED:
The plain round, floor, ceil and trunc nodes do the same for ordinary numbers 
and NumPy arrays - see the math_single help patch."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 8', 'pos': (30, 120), 'w': 180, 'h': 180,
     'props': {'min': -3.0, 'max': 3.0}},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 120), 'w': 208, 'h': 148,
     'props': HM(8, -3.0, 3.0)},
    {'key': 'c0', 'comment': True, 'text': 'values either side of zero', 'pos': (30, 310)},
    {'key': 'fl', 'init': 't.floor', 'pos': (30, 350), 'w': 160, 'h': 70},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (250, 350), 'w': 208, 'h': 148,
     'props': HM(8, -3.0, 3.0, '%.0f')},
    {'key': 'c1', 'comment': True, 'text': 'floor: always down the number line',
     'pos': (30, 430)},
    {'key': 'tr', 'init': 't.trunc', 'pos': (30, 475), 'w': 160, 'h': 70},
    {'key': 'hm3', 'init': 'heat_map', 'pos': (250, 530), 'w': 208, 'h': 148,
     'props': HM(8, -3.0, 3.0, '%.0f')},
    {'key': 'c2', 'comment': True, 'text': 'trunc: always towards zero', 'pos': (30, 555)},
    {'key': 'c3', 'comment': True, 'text': 'compare the negative values', 'pos': (30, 585)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'fl', 'tensor in'), ('fl', 'output', 'hm2', 'y'),
         ('rnd', 'random tensor', 'tr', 'tensor in'), ('tr', 'output', 'hm3', 'y')]
print(build('t.round', 't rounding - whole numbers, several ways', body, demo, links,
            demo_width=480, text_width=800, text_height=680))

# -------------------------------------------------------------------- t.clamp
body = """These hold a tensor's values inside limits.

THE NODES:

t.clamp     limit at both ends
t.maximum   element by element, the larger of two tensors
t.minimum   element by element, the smaller

t.clamp takes a single min and max and applies them everywhere. 
t.maximum and t.minimum compare against another TENSOR, so the limit can differ 
at every position - a per-element floor or ceiling rather than one number. 
That is the difference, and it is why both exist.

Used with a single number, t.maximum is a lower bound and t.minimum an upper 
one - which reads backwards until you see where the names come from. 
t.maximum means "the maximum of the data and this", and that is exactly a floor.

WHY YOU WANT THIS BEFORE A DIVISION:
Clamping a denominator away from zero is the usual way to keep a division from 
producing infinity, and doing it with t.maximum against a very small number is 
cheaper and better behaved than testing for zero and branching.

SYNTAX:
t.clamp <min> <max>

EXAMPLE:
t.clamp 0.0 1.0

INPUTS and PARAMETERS:

tensor in:
The tensor. Receiving it triggers the node.

min / max (t.clamp):
The limits.

tensor b in (t.maximum, t.minimum):
The tensor to compare against - or a single number applied everywhere.

OUTPUTS: 

output:
The limited tensor, in the same shape.

RELATED:
np.clip does the same for NumPy arrays, and clamp for ordinary numbers.

A NOTE ON GRADIENTS:
Where a value is being held at a limit, the gradient through it is zero - the 
output stops responding to the input entirely. If you are training something 
and it has stopped learning, a clamp saturating is worth checking for."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 3.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 3.0)},
    {'key': 'cl', 'init': 't.clamp -0.4 0.4', 'pos': (30, 250), 'w': 180, 'h': 110,
     'props': {'min': -0.4, 'max': 0.4}},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 380), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c0', 'comment': True, 'text': 'the peaks are held at the limits',
     'pos': (30, 645)},
    {'key': 'c1', 'comment': True, 'text': 'drag min and max to move them',
     'pos': (30, 675)},
]
# the torch nodes convert whatever arrives, so a plain float stream is fine
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'cl', 'tensor in'),
         ('cl', 'output', 'p1', 'y')]
print(build('t.clamp', 't.clamp - hold a tensor inside limits', body, demo, links,
            demo_width=430, text_width=790, text_height=620))
