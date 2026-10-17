"""t extremes and ordering, distributions, testing and locating."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

HM = lambda n=8, lo=0.0, hi=1.0, fmt='%.2f': {
    'color': 'viridis', 'width': 200, 'height': 100, 'sample count': n,
    'min y': lo, 'max y': hi, 'update_mode': 'heat_map', 'number format': fmt}

# --------------------------------------------------------------------- t.max
body = """These find the extremes of a tensor: how big the largest or smallest 
value is, and where it is.

THE NODES:

t.max      the largest value - and, along an axis, where it is
t.min      the smallest, likewise
t.argmax   only where the largest is

WITHOUT AN AXIS, ONE ANSWER:
Plain t.max or t.min looks at the whole tensor and sends the single largest 
or smallest value. Plain t.argmax sends where the largest is, counting 
through the tensor as if it were one long row - for a table, read it row by 
row.

WITH AN AXIS, ONE ANSWER PER ROW OR COLUMN, AND TWO OUTLETS:
Give an axis as the argument - t.max 0 - and a 'dim' inlet appears, and 
t.max and t.min grow a second outlet. One sends the extreme VALUE along that 
axis, the other the INDEX it was found at. Those answer different questions 
- "how big was the peak" and "which joint was it" - and getting both from 
one node means they cannot disagree. The index is sent first.

On a table with dim 1 you get the extreme of each ROW; with dim 0, of each 
COLUMN.

t.argmax is the same index without the value, for when you only want to 
know which. There is no t.argmin.

WHY AN INDEX IS OFTEN THE BETTER ANSWER:
For a per-joint measurement, the largest value tells you the intensity and 
the index tells you WHERE the body is working. The second is usually the more 
informative: that something is straining hard matters less than which part 
of it is.

SYNTAX:
t.max <dim>
t.min <dim>
t.argmax <dim>

EXAMPLE:
t.max 0

INPUTS and PARAMETERS:

tensor in:
The tensor. Receiving it triggers the node.

dim (only with an argument):
Which axis to work along.

OUTPUTS: 

max tensor / min tensor:
The extreme value - one number, or one per row or column.

max indices / min indices (only with an argument):
Where along the axis each extreme was found.

max index (t.argmax):
Just the position.

RELATED:
t.argsort gives the whole ranking, not just the top.
t.take_along_dim turns indices back into values - including values from a 
DIFFERENT tensor, which is how you answer "at the moment each joint was 
fastest, what was its angle"."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 8', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 120), 'w': 208, 'h': 148,
     'props': HM(8)},
    {'key': 'c0', 'comment': True, 'text': 'click: eight values', 'pos': (30, 310)},
    {'key': 'mx', 'init': 't.max 0', 'pos': (30, 350), 'w': 180, 'h': 100},
    {'key': 'f1', 'init': 'float', 'pos': (30, 465), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'i1', 'init': 'int', 'pos': (170, 465), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 't.max 0: the largest, and where',
     'pos': (30, 515)},
    {'key': 'mn', 'init': 't.min', 'pos': (320, 350), 'w': 140, 'h': 70},
    {'key': 'f2', 'init': 'float', 'pos': (320, 465), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': 'plain t.min: just the value',
     'pos': (320, 515)},
    {'key': 'am', 'init': 't.argmax', 'pos': (30, 560), 'w': 180, 'h': 70},
    {'key': 'i2', 'init': 'int', 'pos': (30, 645), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c3', 'comment': True, 'text': 'WHERE the largest is, not what',
     'pos': (30, 695)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'mx', 'tensor in'),
         ('mx', 'max tensor', 'f1', ''), ('mx', 'max indices', 'i1', ''),
         ('rnd', 'random tensor', 'mn', 'tensor in'), ('mn', 'min tensor', 'f2', ''),
         ('rnd', 'random tensor', 'am', 'tensor in'), ('am', 'max index', 'i2', '')]
print(build('t.max', 't.max, t.min, t.argmax - the extremes, and where', body, demo, links,
            demo_width=500, text_width=800, text_height=760))

# ------------------------------------------------------------------ t.argsort
body = """t.argsort gives the order that would sort a tensor - as indices, not 
as sorted values.

Entry 0 of the result is the index of the smallest value, entry 1 the index 
of the next smallest, and so on. Tick 'descending' and it is largest first.

AN ORDER IS MORE USEFUL THAN SORTED VALUES:
You can apply the same order to something else - rank joints by speed, then 
reorder their names, or their colours, or a second measurement, by that 
ranking. Take the first few entries and you have "the three fastest joints" 
as indices you can then look up anywhere. The demo applies the order back to 
the values themselves with t.take_along_dim, which gives them sorted.

TIES:
'stable' decides what happens to equal values: with it on, ties keep their 
original relative order. That matters when the ranking drives something that 
would otherwise flicker between two joints that are momentarily equal.

WHICH AXIS:
With no argument it sorts along the LAST axis - each row of a table 
separately. Give an axis as the argument - t.argsort 0 - and a 'dim' inlet 
appears so it can be changed.

SYNTAX:
t.argsort <dim>

EXAMPLE:
t.argsort

INPUTS and PARAMETERS:

tensor in:
The tensor. Receiving it triggers the node.

dim (only with an argument):
Which axis to sort along.

descending:
Largest first.

stable:
Keep equal values in their original order.

OUTPUTS: 

output:
The ordering, as indices, the same shape as the input.

RELATED:
t.take_along_dim turns the indices back into values, from this tensor or 
another.
t.argmax is the first entry of a descending argsort, on its own."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 8', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 120), 'w': 208, 'h': 148,
     'props': HM(8)},
    {'key': 'c0', 'comment': True, 'text': 'click: eight values', 'pos': (30, 310)},
    {'key': 'as', 'init': 't.argsort', 'pos': (30, 350), 'w': 200, 'h': 140},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (30, 505), 'w': 208, 'h': 148,
     'props': HM(8, 0.0, 8.0, '%.0f')},
    {'key': 'c1', 'comment': True, 'text': 'the order: where the smallest is, first',
     'pos': (30, 665)},
    {'key': 'td', 'init': 't.take_along_dim', 'pos': (270, 350), 'w': 200, 'h': 110},
    {'key': 'hm3', 'init': 'heat_map', 'pos': (270, 505), 'w': 208, 'h': 148,
     'props': HM(8)},
    {'key': 'c2', 'comment': True, 'text': 'the values, taken in that order',
     'pos': (270, 665)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'as', 'tensor in'), ('as', 'output', 'hm2', 'y'),
         ('as', 'output', 'td', 'indices in'),
         ('rnd', 'random tensor', 'td', 'tensor in'), ('td', 'output', 'hm3', 'y')]
print(build('t.argsort', 't.argsort - the order that would sort it', body, demo, links,
            demo_width=500, text_width=790, text_height=700))

# -------------------------------------------------------------------- t.histc
body = """These describe how a tensor's values are DISTRIBUTED, rather than what they are.

A mean says where the middle is. A distribution says what the whole spread 
looks like - whether the values cluster, whether there are two groups, whether 
the tail is long. For anything measured rather than computed, that shape is 
usually the interesting thing.

THE NODES:

t.histc      a histogram: how many values fall in each of N equal bins
t.bincount   how many times each integer occurs
t.bucketize  which bin each value belongs to, given boundaries you choose

t.histc IS THE ORDINARY HISTOGRAM:
Give it a bin count and a range and it counts how many values land in each bin. 
The range matters: values outside min and max are not counted at all, so a 
histogram that seems to be missing data usually has its limits set too narrow.

t.bincount IS FOR INTEGERS AND COUNTS EXACTLY:
No bins, no range - it counts occurrences of each whole number. That is the 
right node for labels, indices, note numbers, joint ids - anything where the 
values ARE categories rather than measurements, and where putting them in bins 
would be a mistake.

t.bucketize ASSIGNS RATHER THAN COUNTS:
Given a set of boundaries it tells you which interval each value falls into, 
one answer per input value. That is quantisation with boundaries you choose, 
and it is what you want when the bins are meaningful rather than even - 
'still', 'moving', 'fast' at thresholds that mean something, rather than three 
equal slices of the range.

Its output can then go into t.bincount to count how much time was spent in 
each band, which is the two nodes doing what a histogram cannot: a histogram of 
unequal, meaningful bins.

SYNTAX:
t.histc <bins>
t.bincount
t.bucketize

EXAMPLE:
t.histc

INPUTS and PARAMETERS:

tensor in:
The values.

bin count / min / max (t.histc):
How many bins, and the range they span. Values outside are dropped.

int tensor in (t.bincount):
Whole numbers. Non-integers are not what this node is for.

boundaries tensor in (t.bucketize):
The edges of the intervals, in increasing order.

right (t.bucketize):
Whether a value exactly on a boundary belongs to the interval above or below.

int32 indices:
Return the bin indices as 32-bit rather than 64-bit integers.

OUTPUTS: 

histogram tensor out / bin count tensor out:
The counts, or the per-value bin assignment.

A NOTE ON WATCHING A DISTRIBUTION MOVE:
A histogram of one frame is noise. A histogram accumulated over a window - 
np.rolling_buffer into t.histc - is a picture of what the movement has actually 
been doing, and it changes in ways a running mean does not show: a mean stays 
put while a single peak splits into two."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.dist.normal 256', 'pos': (30, 120), 'w': 240, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'a bell curve, 256 samples', 'pos': (30, 330)},
    {'key': 'hc', 'init': 't.histc', 'pos': (30, 370), 'w': 220, 'h': 160},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 545), 'w': 208, 'h': 148,
     'props': HM(16, 0.0, 60.0, '%.0f')},
    {'key': 'c1', 'comment': True, 'text': 'the shape, not the middle\nset min and max too narrow and\nvalues outside are simply dropped', 'pos': (30, 705)},
    {'key': 'bz', 'init': 't.bucketize', 'pos': (300, 370), 'w': 240, 'h': 160},
    {'key': 'c4', 'comment': True, 'text': 'assigns each value to a band -\nbands you choose, not equal ones',
     'pos': (300, 545)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', '', 'hc', '', 0), ('hc', 'histogram tensor out', 'hm', 'y'),
         ('rnd', '', 'bz', 'tensor in', 0)]
print(build('t.histc', 't.histc - the shape of the values', body, demo, links,
            demo_width=580, text_width=800, text_height=740))

# --------------------------------------------------------------------- t.any
body = """These test a tensor and report what is true, or where.

THE NODES:

t.any            true if ANY value is non-zero
t.all            true only if EVERY value is non-zero
t.count_nonzero  how many are non-zero
t.argwhere       the positions of the non-zero ones
t.non_zero       the same node

THESE ARE WHAT YOU DO WITH A MASK:
A comparison node gives you a tensor of true and false. On its own that is not 
an answer - these four turn it into one, and which you use is which question 
you were asking.

  t.any            is anything above the threshold?
  t.all            is everything?
  t.count_nonzero  how many?
  t.argwhere       which ones?

"Is any joint moving faster than this" is t.any. "How many are" is 
count_nonzero. "Which" is argwhere - and that last one is usually the answer 
you actually wanted, because a count tells you a body is busy while the indices 
tell you what it is doing.

argwhere RETURNS COORDINATES, NOT VALUES:
For a 1D tensor you get a list of positions. For a 2D one you get a pair per 
hit - row and column - so the result is shaped (number of hits, number of 
dimensions). That surprises people expecting a flat list; it is what lets a hit 
in a multi-dimensional tensor be located at all.

The number of hits is not known in advance, which is why this cannot preserve 
the input's shape and why nothing downstream can assume a fixed size.

t.count_nonzero TAKES A dim:
Without one you get a single total. With one you get a count per row or column - 
so for a (frames, joints) mask, a count along frames tells you how long each 
joint spent over the threshold, and along joints how many joints were over it 
at each instant. Two quite different summaries from the same mask.

SYNTAX:
t.any
t.count_nonzero
t.argwhere

EXAMPLE:
t.count_nonzero

INPUTS and PARAMETERS:

tensor in:
The tensor, usually a mask from a comparison. Receiving it triggers the node.

dim (t.count_nonzero):
Which axis to count along.

OUTPUTS: 

out:
True or false, the count, or the positions.

A NOTE ON WHAT COUNTS AS TRUE:
These test for NON-ZERO, not for a boolean. A tensor of measurements passed 
straight in will report almost everything as true, because almost nothing is 
exactly zero. Compare first, then test the result."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 12', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 120), 'w': 208, 'h': 148,
     'props': HM(12)},
    {'key': 'gt', 'init': '> 0.7', 'pos': (30, 320), 'w': 130, 'h': 70,
     'props': {'output_type': 'bool'}},
    {'key': 'c0', 'comment': True, 'text': 'a mask: above 0.7', 'pos': (30, 400)},
    {'key': 'cn', 'init': 't.count_nonzero', 'pos': (30, 440), 'w': 220, 'h': 110},
    {'key': 'i1', 'init': 'int', 'pos': (30, 565), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 'how many passed', 'pos': (30, 615)},
    {'key': 'aw', 'init': 't.argwhere', 'pos': (280, 440), 'w': 220, 'h': 90},
    {'key': 'l1', 'init': 'list', 'pos': (280, 545), 'w': 260, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'WHICH ones - usually the answer\nyou actually wanted',
     'pos': (280, 595)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'gt', 'in'),
         ('gt', 'result', 'cn', 'tensor in'), ('cn', 'tensor out', 'i1', ''),
         ('gt', 'result', 'aw', 'tensor in'),
         ('aw', 'index tensor where non-zero', 'l1', '')]
print(build('t.any', 't.any, t.count_nonzero, t.argwhere - what is true, and where',
            body, demo, links, demo_width=560, text_width=800, text_height=720))
