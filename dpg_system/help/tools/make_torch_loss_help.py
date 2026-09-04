"""The three loss functions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """These measure how wrong something is: one number, from two tensors.

THE NODES:

t.mse_loss             squared error - big mistakes dominate
t.l1_loss              plain error - every mistake counts its own size
t.cross_entropy_loss   for choosing between categories

WHAT A LOSS IS FOR:
It compares what you got against what you wanted and reduces the difference to a
single number, where zero means identical and larger means worse. That is what
training minimises - but a loss is just as useful on its own, as a running
measure of how far a signal has drifted from a reference, or how unlike a pose
is from a stored one.

MSE AGAINST L1 - THE DIFFERENCE IS WHAT SQUARING DOES:
Both add up the errors. MSE squares each one first, and that single difference
changes the character of the measure completely. With one element off by 4:

    t.mse_loss   16.0        (4 squared)
    t.l1_loss     4.0

and with the same element off by only 0.4:

    t.mse_loss    0.16
    t.l1_loss     0.40

Notice the order swaps. Squaring makes errors above 1 count for much more and
errors below 1 count for much less, so MSE is dominated by the worst offender
while L1 treats a lot of small errors and one big one as comparable.

Which you want depends on what you are measuring. MSE if a single large
discrepancy is the thing that matters and you want it to shout. L1 if the
occasional wild value is noise you would rather not have swamp the reading -
L1 is the robust one.

BOTH ARE SUMS HERE, NOT AVERAGES:
These use a total rather than a mean, so the number grows with the SIZE of the
tensor. A loss of 16 across four elements and a loss of 16 across four hundred
are not the same thing at all.

That matters if you compare losses between differently-shaped data, or if the
shape can change while a patch runs - the number will jump for reasons that have
nothing to do with the data getting worse. Divide by the element count yourself
if you want something comparable.

CROSS ENTROPY IS A DIFFERENT KIND OF QUESTION:
The other two ask "how far off are these numbers". Cross entropy asks "did you
pick the right category, and how confident were you". The input is one row of
scores per item - one score per category, unnormalised - and the target says
which category was correct.

With scores [2.0, 0.5, 0.1], the model is leaning towards the first category:

    correct answer is 0  ->  0.32     confident and right
    correct answer is 1  ->  1.82
    correct answer is 2  ->  2.22     confident and wrong

Being confidently wrong costs far more than being uncertain, which is the whole
point of it: it rewards well-calibrated confidence, not just correct guesses.

'target' IS A COLD INLET ON ALL THREE:
Setting the target does not compute anything - it is the tensor arriving at
'tensor in' that does. So set the target first. Send them the other way round
and the first answer is either missing or computed against the previous target,
which looks like the loss lagging by one.

THE TARGET IS EITHER INDICES OR PROBABILITIES:
Send one whole number per row - which category is right - or send a full row of
probabilities the same shape as the input, for when the answer is a blend rather
than a single choice. The node works out which you meant from the shape.

SYNTAX:
t.mse_loss
t.l1_loss
t.cross_entropy_loss

EXAMPLE:
t.mse_loss

INPUTS and PARAMETERS:

tensor in:
What you got. Receiving it computes the loss.

target:
What you wanted. It is a cold inlet - set it first, then send the input.

OUTPUTS: 

loss:
One number.

RELATED:
t.dist and the distance nodes if you want the difference as a shape rather than
as a single number.
t.nn.softmax turns cross entropy's kind of scores into probabilities you can
look at."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 8', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'm1', 'init': 'message', 'pos': (240, 120), 'w': 160, 'h': 42,
     'props': {'text in': '0.5', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'how far is a random tensor from 0.5?',
     'pos': (30, 310)},

    {'key': 'mse', 'init': 't.mse_loss', 'pos': (30, 355), 'w': 200, 'h': 110},
    {'key': 'f1', 'init': 'float', 'pos': (30, 480), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'squared: the worst element dominates',
     'pos': (30, 530)},

    {'key': 'l1', 'init': 't.l1_loss', 'pos': (300, 355), 'w': 200, 'h': 110},
    {'key': 'f2', 'init': 'float', 'pos': (300, 480), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': 'plain: every element counts its size\nbelow an error of 1 the two swap over -\nsquaring makes small errors smaller',
     'pos': (300, 530)},

    {'key': 'm3', 'init': 'message', 'pos': (30, 650), 'w': 100, 'h': 42,
     'props': {'text in': '0', 'font size': '24'}},
    {'key': 'm2', 'init': 'message', 'pos': (160, 650), 'w': 220, 'h': 42,
     'props': {'text in': '2.0 0.5 0.1', 'font size': '24'}},
    {'key': 'ce', 'init': 't.cross_entropy_loss', 'pos': (30, 710), 'w': 260, 'h': 110},
    {'key': 'f3', 'init': 'float', 'pos': (30, 835), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c5', 'comment': True, 'text': 'CLICK THE TARGET FIRST, then the scores:\ntarget is a cold inlet, and it is the\nscores arriving that does the work\nchange the 0 to a 2 and click again:\n0.32 becomes 2.22 - confidently wrong\ncosts far more than being unsure',
     'pos': (30, 885)},
]
links = [('btn', '', 'rnd', '###input'),
         ('m1', 'message out', 'mse', 'target'),
         ('m1', 'message out', 'l1', 'target'),
         ('rnd', 'random tensor', 'mse', 'tensor in'), ('mse', 'loss', 'f1', ''),
         ('rnd', 'random tensor', 'l1', 'tensor in'), ('l1', 'loss', 'f2', ''),
         ('m3', 'message out', 'ce', 'target'),
         ('m2', 'message out', 'ce', 'tensor in'), ('ce', 'loss', 'f3', '')]
print(build('t.mse_loss', 't losses - how wrong is this', body,
            demo, links, demo_width=560, text_width=800, text_height=760))
