"""t selecting, rearranging, scattering, matrix structure."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

HM = lambda n=4, lo=0.0, hi=1.0, fmt='%.2f': {
    'color': 'viridis', 'width': 200, 'height': 100, 'sample count': n,
    'min y': lo, 'max y': hi, 'update_mode': 'heat_map', 'number format': fmt}

# ------------------------------------------------------------ t.index_select
body = """These take part of a tensor out - by position, or by a condition.

There are more of them than seems necessary, and the reason is that "which 
part" can mean several genuinely different things.

THE NODES:

t[]               slice, the way you would write it in code
t.narrow          a contiguous run along one axis: start here, take this many
t.select          one index along one axis, and that axis DISAPPEARS
t.index_select    several indices along one axis, and the axis stays
t.take            treat the tensor as flat and take these positions
t.take_along_dim  take a different index in each row
t.masked_select   take everything a condition is true for

select VERSUS index_select, WHICH IS THE SUBTLE ONE:
Taking row 2 of a (5, 8) tensor with select gives you shape (8,) - the row axis 
is gone, because you asked for one specific row and a single row has no row 
axis left. Taking rows [2] with index_select gives you shape (1, 8) - still a 
matrix, with one row in it.

Which you want depends on what happens next. If something downstream expects a 
matrix, index_select keeps you in the right shape. If you want the vector 
itself, select gets it without a squeeze.

take_along_dim IS THE ONE WORTH KNOWING ABOUT:
The others take the SAME indices from every row. take_along_dim takes a 
different index from each. That is exactly what you need after an argmax: 
argmax tells you where the largest value is in each row, and take_along_dim 
turns those positions back into the values - or pulls the corresponding entries 
out of a DIFFERENT tensor, which is how you answer "at the moment each joint 
was fastest, what was its angle".

masked_select RETURNS A FLAT TENSOR:
However shaped the input, what comes back is one dimension long - because the 
number of elements that passed is not known in advance and cannot be arranged 
into a rectangle. If you want to keep the shape, multiply by the mask instead 
of selecting with it.

SYNTAX:
t[] <indices>
t.narrow <dim> <start> <length>
t.select <dim> <index>

EXAMPLE:
t.select 0 2

INPUTS and PARAMETERS:

tensor in / source tensor:
The tensor. Receiving it triggers the node.

indices:
Which positions to take.

dim:
Which axis to work along.

start / length (t.narrow):
Where the run begins and how long it is.

mask (t.masked_select):
A tensor of true and false the same shape as the source - normally the output 
of a comparison node.

OUTPUTS: 

output / selection tensor:
The selected part.

RELATED:
The comparison nodes make the masks. t.scatter writes values back INTO a 
tensor at given positions, which is the reverse of these."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 4 4', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 120), 'w': 208, 'h': 148,
     'props': HM(4)},
    {'key': 'c0', 'comment': True, 'text': 'a 4 by 4 tensor', 'pos': (30, 310)},
    {'key': 'sl', 'init': 't.select 0 1', 'pos': (30, 350), 'w': 180, 'h': 110},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (250, 350), 'w': 208, 'h': 148,
     'props': HM(4)},
    {'key': 'c1', 'comment': True, 'text': 'one row: the row axis is gone',
     'pos': (30, 475)},
    {'key': 'gt', 'init': '> 0.5', 'pos': (30, 520), 'w': 130, 'h': 70,
     'props': {'output_type': 'bool'}},
    {'key': 'ms', 'init': 't.masked_select', 'pos': (30, 605), 'w': 220, 'h': 90},
    {'key': 'l1', 'init': 'list', 'pos': (30, 715), 'w': 300, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'everything above 0.5, as a flat tensor\nthe shape cannot survive: how many passed\nis not known in advance',
     'pos': (30, 765)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'sl', 'tensor in'), ('sl', 'output', 'hm2', 'y'),
         ('rnd', 'random tensor', 'gt', 'in'),
         ('gt', 'result', 'ms', 'mask'),
         ('rnd', 'random tensor', 'ms', 'source tensor'),
         ('ms', 'selection tensor', 'l1', '')]
print(build('t.index_select', 't selecting - which part, and what shape survives',
            body, demo, links, demo_width=480, text_width=810, text_height=800))

# -------------------------------------------------------------------- t.roll
body = """These move a tensor's contents around, or make more of them.

THE NODES:

t.roll    shift along an axis, with what falls off one end reappearing at the other
t.flip    reverse along an axis
t.repeat  repeat the whole tensor a number of times in each direction
t.tile    the same thing

t.roll LOSES NOTHING:
Everything that falls off the end comes back at the start, which makes it a 
rotation rather than a shift. That is what you want for a circular buffer, 
for a delay of a fixed number of samples, or for comparing a signal against 
itself offset in time. If you want values to fall off and zeros to arrive 
instead, this is not the node.

repeat VERSUS tile:
Both tile the WHOLE tensor rather than repeating individual elements, which 
surprises people coming from NumPy, where repeat does the opposite. 
Given [1, 2, 3] and a count of 2, both give [1, 2, 3, 1, 2, 3] - not 
[1, 1, 2, 2, 3, 3].

The difference between the two is only in how they handle being given fewer 
counts than the tensor has dimensions: tile assumes 1 for the missing leading 
ones, repeat wants at least as many as there are dimensions. tile is the more 
forgiving.

SYNTAX:
t.roll <shift> <dim>
t.flip <dim>
t.tile <count> <count> ...

EXAMPLE:
t.roll 1 0

INPUTS and PARAMETERS:

tensor in:
The tensor. Receiving it triggers the node.

shifts / dims (t.roll):
How far to move and along which axis. Negative shifts go the other way.

flip dims (t.flip):
Which axes to reverse.

repeats / tiling:
How many copies in each direction.

OUTPUTS: 

rolled tensor / output / repeated tensor out / tiled tensor out:
The result."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 8', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 120), 'w': 208, 'h': 148,
     'props': HM(8)},
    {'key': 'c0', 'comment': True, 'text': 'eight values', 'pos': (30, 310)},
    {'key': 'rl', 'init': 't.roll 2', 'pos': (30, 350), 'w': 180, 'h': 110},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (250, 350), 'w': 208, 'h': 148,
     'props': HM(8)},
    {'key': 'c1', 'comment': True, 'text': 'rolled by two: nothing lost, it wraps',
     'pos': (30, 475)},
    {'key': 'fp', 'init': 't.flip', 'pos': (30, 520), 'w': 160, 'h': 90},
    {'key': 'hm3', 'init': 'heat_map', 'pos': (250, 530), 'w': 208, 'h': 148,
     'props': HM(8)},
    {'key': 'c2', 'comment': True, 'text': 'flipped: reversed', 'pos': (30, 625)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'rl', 'tensor in'), ('rl', 'rolled tensor', 'hm2', 'y'),
         ('rnd', 'random tensor', 'fp', 'tensor in'), ('fp', 'output', 'hm3', 'y')]
print(build('t.roll', 't.roll and t.flip - moving contents about', body, demo, links,
            demo_width=480, text_width=800, text_height=680))

# ----------------------------------------------------------------- t.scatter
body = """These write values INTO a tensor at positions you choose - the reverse of selecting.

THE NODES:

t.scatter       write a source tensor's values into a target at given indices
t.scatter_hold  keep a tensor and update parts of it as pairs arrive

t.scatter_hold IS THE USEFUL ONE IN A PATCH:
It holds a tensor of a length you set and remembers it between messages. 
Send it a list of index and value pairs - "3 0.8 7 0.2" - and it writes those 
values at those positions, leaving everything else exactly as it was.

That is how you assemble one tensor from many separate sources. Values arriving 
one at a time, from different nodes, at different rates, accumulate into a 
single tensor that always holds the latest of each - which is what you want 
when per-joint readings arrive separately and something downstream wants them 
all together.

Indices are clamped into range rather than raising, so a stray index writes to 
the nearest valid slot instead of stopping the patch. An odd number of values 
is ignored, since it cannot be read as pairs.

SYNTAX:
t.scatter_hold <length: int>
t.scatter

EXAMPLE:
t.scatter_hold 16

INPUTS and PARAMETERS:

list of index value pairs (t.scatter_hold):
Alternating positions and values. Receiving them triggers the node.

length of target tensor (t.scatter_hold):
How long the held tensor is. Changing it clears the contents.

clear (t.scatter_hold):
Sets everything back to zero.

tensor to scatter into / indices in / source in (t.scatter):
The target, where to write, and what to write.

dim:
Which axis the indices refer to.

OUTPUTS: 

output:
The whole tensor after the write - not just the parts that changed. 
So every update produces a complete current picture, which is what makes this 
usable as a source for anything that wants the full state.

RELATED:
The selecting nodes go the other way, pulling values out at given positions."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 220, 'h': 42,
     'props': {'text in': '2 0.9 5 0.4', 'font size': '24'}},
    {'key': 'btn2', 'init': 'button', 'pos': (280, 62), 'w': 88, 'h': 46},
    {'key': 'm2', 'init': 'message', 'pos': (280, 118), 'w': 220, 'h': 42,
     'props': {'text in': '0 0.7 7 0.2', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'click either: index, value, index, value',
     'pos': (30, 170)},
    {'key': 'sh', 'init': 't.scatter_hold 8', 'pos': (30, 210), 'w': 240, 'h': 140},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 370), 'w': 208, 'h': 148,
     'props': HM(8)},
    {'key': 'c1', 'comment': True, 'text': 'each click updates only its own slots\nthe rest keeps what it had\nclick clear on the node to zero it',
     'pos': (30, 530)},
]
links = [('btn', '', 'm1', ''), ('btn2', '', 'm2', ''),
         ('m1', 'message out', 'sh', 'list of index value pairs'),
         ('m2', 'message out', 'sh', 'list of index value pairs'),
         ('sh', 'output', 'hm', 'y')]
print(build('t.scatter', 't.scatter_hold - assemble a tensor from pieces', body,
            demo, links, demo_width=520, text_width=790, text_height=680))

# -------------------------------------------------------------------- t.diag
body = """These are about a matrix's TRIANGLES and its diagonal.

THE NODES:

t.diag  pull the diagonal out as a vector, or build a matrix from one
t.tril  keep the lower triangle, zero the rest
t.triu  keep the upper triangle, zero the rest

t.diag GOES BOTH WAYS:
Given a matrix it returns the diagonal as a vector. Given a vector it builds a 
matrix with that vector on the diagonal and zeros elsewhere. It works out which 
you meant from the shape of what you send it.

The diagonal of a covariance matrix is the variance of each channel by itself, 
with all the cross-terms removed - so t.diag on a t.corrcoef result separates 
"how much does this vary" from "what does it vary WITH".

WHY TRIANGLES MATTER:
A matrix of relationships between things - a covariance, a distance matrix, 
a similarity - is symmetric: the entry for A against B equals the one for B 
against A. Half of it is therefore redundant, and taking one triangle is how 
you avoid counting every pair twice.

Set "which diag" to 1 and the diagonal itself is excluded as well, which for a 
similarity matrix removes the ones down the middle where everything is compared 
against itself. That is almost always what you want when you are asking about 
pairs.

SYNTAX:
t.diag
t.tril

EXAMPLE:
t.triu

INPUTS and PARAMETERS:

tensor in:
The matrix, or the vector to build one from.

which diag:
Which diagonal to work from. 0 is the main one; positive moves above it, 
negative below. On tril and triu this shifts where the triangle is cut, and 
setting it to 1 on triu excludes the main diagonal.

OUTPUTS: 

output:
The diagonal, the built matrix, or the triangle with the rest zeroed."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 5 5', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 120), 'w': 208, 'h': 148,
     'props': HM(5)},
    {'key': 'c0', 'comment': True, 'text': 'a 5 by 5 matrix', 'pos': (30, 310)},
    {'key': 'tu', 'init': 't.triu', 'pos': (30, 350), 'w': 180, 'h': 90},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (250, 350), 'w': 208, 'h': 148,
     'props': HM(5)},
    {'key': 'c1', 'comment': True, 'text': 'the upper triangle, rest zeroed',
     'pos': (30, 455)},
    {'key': 'dg', 'init': 't.diag', 'pos': (30, 500), 'w': 180, 'h': 90},
    {'key': 'hm3', 'init': 'heat_map', 'pos': (250, 530), 'w': 208, 'h': 148,
     'props': HM(5)},
    {'key': 'c2', 'comment': True, 'text': 'the diagonal, as a vector', 'pos': (30, 605)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'tu', 'tensor in'), ('tu', 'output', 'hm2', 'y'),
         ('rnd', 'random tensor', 'dg', 'tensor in'), ('dg', 'output', 'hm3', 'y')]
print(build('t.diag', 't.diag, t.tril, t.triu - diagonals and triangles', body,
            demo, links, demo_width=480, text_width=800, text_height=660))
