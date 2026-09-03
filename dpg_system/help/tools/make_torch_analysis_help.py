"""t extremes and ordering, distributions, testing and locating."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

HM = lambda n=8, lo=0.0, hi=1.0, fmt='%.2f': {
    'color': 'viridis', 'width': 200, 'height': 100, 'sample count': n,
    'min y': lo, 'max y': hi, 'update_mode': 'heat_map', 'number format': fmt}

# --------------------------------------------------------------------- t.max
body = """These find the extremes of a tensor, and the order its values are in.

THE NODES:

t.max      the largest value - and where it is
t.min      the smallest, likewise
t.argmax   only where the largest is
t.argsort  the order that would sort the tensor

t.max GIVES YOU BOTH, WHICH IS THE POINT:
When a dim is set it has two outlets: the largest VALUE along that axis, and 
the INDEX it was found at. Those answer different questions - "how big was the 
peak" and "which joint was it" - and having both from one node means they 
cannot disagree, which they can if you compute the value one way and the 
position another.

t.argmax is the same index without the value, for when you only want to know 
which.

WHY AN INDEX IS OFTEN THE BETTER ANSWER:
For a per-joint measurement, the largest value tells you the intensity and the 
index tells you WHERE the body is working. The second is usually the more 
informative: that something is straining hard matters less than which part of 
it is.

t.argsort GIVES AN ORDER, NOT SORTED VALUES:
It returns the indices that WOULD sort the tensor. That is more useful than the 
sorted values, because you can apply the same order to something else - rank 
joints by speed, then reorder their names, or their colours, or a second 
measurement, by that ranking.

Take the first few of what it returns and you have "the three fastest joints" 
as indices you can then look up anywhere.

'stable' decides what happens to equal values: with it on, ties keep their 
original relative order, which matters when the ranking drives something that 
would otherwise flicker between two joints that are momentarily equal.

SYNTAX:
t.max
t.argsort

EXAMPLE:
t.argmax

INPUTS and PARAMETERS:

tensor in:
The tensor. Receiving it triggers the node.

dim:
Which axis to work along. Without it the whole tensor is reduced to one answer; 
with it you get one per row or column.

descending (t.argsort):
Largest first.

stable (t.argsort):
Keep equal values in their original order.

OUTPUTS: 

values and indices (t.max, t.min):
The extreme value, and where it was.

max index (t.argmax):
Just the position.

output (t.argsort):
The ordering, as indices.

RELATED:
t.take_along_dim turns those indices back into values - including values from a 
DIFFERENT tensor, which is how you answer "at the moment each joint was 
fastest, what was its angle"."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 8', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 120), 'w': 208, 'h': 148,
     'props': HM(8)},
    {'key': 'c0', 'comment': True, 'text': 'eight values', 'pos': (30, 310)},
    {'key': 'am', 'init': 't.argmax', 'pos': (30, 350), 'w': 180, 'h': 100},
    {'key': 'i1', 'init': 'int', 'pos': (30, 465), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 'WHERE the largest is, not what',
     'pos': (30, 515)},
    {'key': 'as', 'init': 't.argsort', 'pos': (280, 350), 'w': 200, 'h': 140},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (280, 505), 'w': 208, 'h': 148,
     'props': HM(8, 0.0, 8.0, '%.0f')},
    {'key': 'c2', 'comment': True, 'text': 'the order that would sort it -',
     'pos': (280, 665)},
    {'key': 'c3', 'comment': True, 'text': 'apply it to something else', 'pos': (280, 695)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'am', 'tensor in'), ('am', 'max index', 'i1', ''),
         ('rnd', 'random tensor', 'as', 'tensor in'), ('as', 'output', 'hm2', 'y')]
print(build('t.max', 't.max and t.argsort - extremes and order', body, demo, links,
            demo_width=500, text_width=800, text_height=720))

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
    {'key': 'c1', 'comment': True, 'text': 'the shape, not the middle', 'pos': (30, 705)},
    {'key': 'c2', 'comment': True, 'text': 'set min and max too narrow and',
     'pos': (30, 735)},
    {'key': 'c3', 'comment': True, 'text': 'values outside are simply dropped',
     'pos': (30, 765)},
    {'key': 'bz', 'init': 't.bucketize', 'pos': (300, 370), 'w': 240, 'h': 160},
    {'key': 'c4', 'comment': True, 'text': 'assigns each value to a band -',
     'pos': (300, 545)},
    {'key': 'c5', 'comment': True, 'text': 'bands you choose, not equal ones',
     'pos': (300, 575)},
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
    {'key': 'c2', 'comment': True, 'text': 'WHICH ones - usually the answer',
     'pos': (280, 595)},
    {'key': 'c3', 'comment': True, 'text': 'you actually wanted', 'pos': (280, 625)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'gt', 'in'),
         ('gt', 'result', 'cn', 'tensor in'), ('cn', 'tensor out', 'i1', ''),
         ('gt', 'result', 'aw', 'tensor in'),
         ('aw', 'index tensor where non-zero', 'l1', '')]
print(build('t.any', 't.any, t.count_nonzero, t.argwhere - what is true, and where',
            body, demo, links, demo_width=560, text_width=800, text_height=720))
