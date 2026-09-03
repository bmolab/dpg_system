"""t shape, joining and splitting, selecting."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

HM = lambda n=4, lo=0.0, hi=1.0, fmt='%.2f': {
    'color': 'viridis', 'width': 200, 'height': 100, 'sample count': n,
    'min y': lo, 'max y': hi, 'update_mode': 'heat_map', 'number format': fmt}

# ----------------------------------------------------------------- t.reshape
body = """These change a tensor's SHAPE without changing the numbers in it.

Twelve numbers can be twelve in a row, three rows of four, four rows of three, 
or a 1 by 12 slab. The numbers are identical; only the arrangement differs, 
and everything downstream cares a great deal about the arrangement. 
Most "this will not connect" problems in tensor work are a shape mismatch.

THE NODES:

t.reshape    rearrange into a shape you give
t.view       the same, when the memory allows it
t.flatten    collapse into one long row
t.ravel      the same as flatten
t.squeeze    remove axes of length 1
t.unsqueeze  add an axis of length 1
t.permute    reorder the axes, however many there are
t.transpose  swap two named axes
t.t          transpose a 2D tensor - the short form
t.adjoint    transpose and conjugate, for complex tensors

view VERSUS reshape:
view demands that the tensor's memory is laid out in a way that allows the new 
shape to be READ from it without moving anything - so it is free, but it fails 
on a tensor that has been transposed or otherwise strided. reshape falls back 
to making a copy when it has to, so it always works and is sometimes not free.

Use view when you know the tensor is contiguous and want it to fail loudly if 
it is not. Use reshape when you just want the shape. If a view is raising 
errors after a transpose, that is exactly this.

WHY LENGTH-1 AXES MATTER:
A tensor of shape (3,) and one of shape (1, 3) hold the same three numbers and 
are not interchangeable - one is a vector, the other a matrix with one row. 
unsqueeze and squeeze convert between them without touching the data. 
When a node insists on a 2D tensor and you have a 1D one, unsqueeze is the 
answer; when a result carries a stray extra dimension, squeeze removes it.

permute VERSUS transpose:
transpose swaps two axes. permute states the whole new order at once, which is 
what you need for anything beyond two dimensions - turning a 
(batch, time, joint) tensor into (batch, joint, time) is one permute and would 
be several transposes.

SYNTAX:
t.reshape <dim> <dim> ...
t.permute <axis> <axis> ...

EXAMPLE:
t.reshape 3 4

INPUTS and PARAMETERS:

tensor in:
The tensor. Receiving it triggers the node.

shape (t.reshape, t.view):
The new shape. The dimensions must multiply to the number of elements you have. 
A -1 means "work this one out", so "6 -1" gives six rows of whatever fits.

dim (t.squeeze, t.unsqueeze):
Where to remove or add the length-1 axis.

permute / dim 1 / dim 2:
The new axis order, or the two axes to swap.

OUTPUTS: 

output / tensor out:
The same numbers, differently arranged.

TRANSPOSING DOES NOT MOVE ANYTHING:
transpose and permute change how the tensor is READ rather than reorganising 
memory, so they are effectively free at any size. What they leave behind is a 
non-contiguous tensor - which is why a view straight after one of them fails."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 3 4', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 120), 'w': 208, 'h': 148,
     'props': HM(4)},
    {'key': 'c0', 'comment': True, 'text': 'three rows of four', 'pos': (30, 310)},
    {'key': 'tr', 'init': 't.t', 'pos': (30, 350), 'w': 140, 'h': 70},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (250, 350), 'w': 208, 'h': 148,
     'props': HM(3)},
    {'key': 'c1', 'comment': True, 'text': 'transposed: four rows of three',
     'pos': (30, 430)},
    {'key': 'fl', 'init': 't.flatten', 'pos': (30, 475), 'w': 160, 'h': 70},
    {'key': 'hm3', 'init': 'heat_map', 'pos': (250, 530), 'w': 208, 'h': 148,
     'props': HM(12)},
    {'key': 'c2', 'comment': True, 'text': 'flattened: twelve in a row', 'pos': (30, 555)},
    {'key': 'c3', 'comment': True, 'text': 'the numbers never changed', 'pos': (30, 585)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'tr', 'tensor in'), ('tr', 'tensor out', 'hm2', 'y'),
         ('rnd', 'random tensor', 'fl', 'tensor in'), ('fl', 'tensor out', 'hm3', 'y')]
print(build('t.reshape', 't shape - the same numbers, arranged differently', body,
            demo, links, demo_width=480, text_width=810, text_height=780))

# --------------------------------------------------------------------- t.cat
body = """These put tensors together, and take them apart again.

THE NODES:

t.cat           join along an EXISTING axis
t.stack         join along a NEW axis
t.hstack        stack side by side
t.vstack        stack one above the other
t.dstack        stack in depth
t.column_stack  as columns
t.row_stack     as rows
t.chunk         cut into a number of pieces
t.tensor_split  the same, with more control over where

cat VERSUS stack, WHICH CATCHES EVERYONE:
Joining two tensors of 3 numbers with cat gives you 6 numbers in a row - 
shape (6,). With stack it gives you shape (2, 3) - the originals kept separate, 
side by side. cat EXTENDS an axis that already exists; stack ADDS one.

Which you want follows from what the pieces are. Two halves of the same thing 
join with cat. Two things that happened at different times, or to different 
joints, stack - because you will want to index them apart again afterwards.

The named stacks - hstack, vstack, dstack - are conveniences that pick the axis 
for you by what it usually means. They are clearer to read than cat with an 
axis number when the tensors are 2D and the intent is obvious.

SPLITTING:
chunk cuts into a number of pieces as evenly as it can. tensor_split does the 
same but lets you say WHERE to cut rather than how many pieces you want, 
which is what you need when the parts are not equal - separating a pose tensor 
into a root and the joints, say.

SYNTAX:
t.cat <dim>
t.stack <dim>
t.chunk <count>

EXAMPLE:
t.stack 0

INPUTS and PARAMETERS:

tensor 1, tensor 2, ...:
The tensors to join. For cat they must match in every dimension except the one 
being joined along; for stack they must match exactly.

dim:
Which axis to join along, or to split along.

split (t.chunk):
How many pieces.

OUTPUTS: 

output:
The joined tensor, or - for chunk and tensor_split - one outlet per piece. 
Those outlets are unnamed, so patch them by position.

A NOTE ON SHAPES:
cat's requirement is easy to violate without noticing: joining a (3, 4) and a 
(3, 5) along axis 0 fails, because they differ in a dimension that is not the 
one being joined. The error names the mismatched dimension, which is usually 
enough to see which tensor is the wrong shape."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'r1', 'init': 't.rand 4', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'r2', 'init': 't.rand 4', 'pos': (250, 120), 'w': 180, 'h': 180},
    {'key': 'c0', 'comment': True, 'text': 'two tensors of four', 'pos': (30, 310)},
    {'key': 'ct', 'init': 't.cat', 'pos': (30, 350), 'w': 180, 'h': 100},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 350), 'w': 208, 'h': 148,
     'props': HM(8)},
    {'key': 'c1', 'comment': True, 'text': 'cat: eight in a row', 'pos': (30, 465)},
    {'key': 'sk', 'init': 't.stack', 'pos': (30, 510), 'w': 180, 'h': 100},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (250, 530), 'w': 208, 'h': 148,
     'props': HM(4)},
    {'key': 'c2', 'comment': True, 'text': 'stack: two rows of four, kept apart',
     'pos': (30, 625)},
]
links = [('btn', '', 'r1', '###input'), ('btn', '', 'r2', '###input'),
         ('r2', 'random tensor', 'ct', 'tensor 2'),
         ('r1', 'random tensor', 'ct', 'tensor 1'), ('ct', 'output', 'hm', 'y'),
         ('r2', 'random tensor', 'sk', 'tensor 2'),
         ('r1', 'random tensor', 'sk', 'tensor 1'), ('sk', 'output', 'hm2', 'y')]
print(build('t.cat', 't.cat and t.stack - extend an axis, or add one', body,
            demo, links, demo_width=480, text_width=810, text_height=760))
