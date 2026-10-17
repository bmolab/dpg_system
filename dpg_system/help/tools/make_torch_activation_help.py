"""t.nn activations, the softmax family, and the remaining special functions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

HM = lambda n=8, lo=0.0, hi=1.0, fmt='%.2f': {
    'color': 'viridis', 'width': 200, 'height': 100, 'sample count': n,
    'min y': lo, 'max y': hi, 'update_mode': 'heat_map', 'number format': fmt}

# ------------------------------------------------------------------ t.nn.relu
body = """These apply one fixed curve to every element of a tensor.

They come from neural networks, where a layer that was only ever multiplication 
and addition can represent nothing more than a straight line however many 
layers you stack. Bending the signal between layers is what lets a network 
learn something curved - and that bend is all these nodes are.

Outside a network they are simply a library of well-behaved shapes, and they 
are useful as such: a response curve for a control, a soft limit that does not 
have a corner in it, a way to fold negative values away.

THE NODES:

t.nn.relu        zero below zero, unchanged above. The plainest bend there is
t.nn.relu6       the same, but also flat at 6 - clipped at both ends
t.nn.sigmoid     an S from 0 to 1, smooth everywhere
t.nn.tanh        an S from -1 to 1
t.nn.logsigmoid  the logarithm of the sigmoid
t.nn.softsign    a gentler S than tanh, approaching its limits more slowly
t.nn.gelu        relu's shape with the corner rounded off, weighted by how far 
                 from zero the value is
t.nn.silu        the input times its own sigmoid - relu-like, but smooth, and 
                 dipping slightly below zero before it rises
t.nn.mish        similar in spirit, smoother still
t.nn.selu        scaled so that a signal passing through many of them keeps 
                 roughly the same mean and spread
t.nn.hardsigmoid a straight-line approximation of the sigmoid
t.nn.hardswish   a straight-line approximation of silu
t.nn.tanhshrink  the input minus its tanh - almost nothing near zero, and 
                 nearly linear far out. The opposite emphasis to the others
t.nn.glu         gated linear unit: splits the tensor in half and uses one half 
                 to gate the other, so it HALVES the size

WHY SO MANY:
Most of them differ only in how they behave near zero and how sharp the corner 
is. What that costs or buys inside a network is a matter for training. 
As shapes in a patch, the ones worth knowing are relu (fold negatives away), 
tanh (soft symmetric limiting with no corner), and sigmoid (anything into 0 
to 1).

The "hard" versions are straight-line approximations - cheaper, and slightly 
angular. If you are drawing the result, you will see the corners.

SYNTAX:
t.nn.relu

EXAMPLE:
t.nn.tanh

INPUTS and PARAMETERS:

tensor in:
The tensor. Receiving it triggers the node. The curve is applied to every 
element independently - shape is untouched, except by glu.

OUTPUTS: 

output:
The bent tensor, in the same shape.

RELATED:
shaper~ applies a curve you DRAW rather than one from this list, at audio rate. 
When none of these is the shape you want, that is the node.

A NOTE ON RANGE:
Several of these are bounded and several are not. sigmoid and tanh cannot leave 
their limits whatever you feed them, which makes them safe in front of 
something that needs a range. relu, gelu, silu and mish are unbounded above - 
a large input gives a large output, and nothing stops it."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 4.0 saw', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('saw', 4.0, 3.0, True)},
    {'key': 'c0', 'comment': True, 'text': 'a ramp from -3 to 3', 'pos': (30, 215)},
    {'key': 'p0', 'init': 'plot', 'pos': (30, 255), 'w': 208, 'h': 176,
     'props': PLOT(-3.2, 3.2)},
    {'key': 'rl', 'init': 't.nn.relu', 'pos': (280, 255), 'w': 170, 'h': 70},
    {'key': 'p1', 'init': 'plot', 'pos': (280, 340), 'w': 208, 'h': 176,
     'props': PLOT(-3.2, 3.2)},
    {'key': 'c1', 'comment': True, 'text': 'relu: negatives folded to zero',
     'pos': (280, 525)},
    {'key': 'th', 'init': 't.nn.tanh', 'pos': (30, 455), 'w': 170, 'h': 70},
    {'key': 'p2', 'init': 'plot', 'pos': (30, 540), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c2', 'comment': True, 'text': 'tanh: soft limiting, no corner\nswap in any of the others to see its shape',
     'pos': (30, 725)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'p0', 'y'),
         ('sig', '', 'rl', 'tensor in'), ('rl', 'output', 'p1', 'y'),
         ('sig', '', 'th', 'tensor in'), ('th', 'output', 'p2', 'y')]
print(build('t.nn.relu', 't.nn activations - one fixed curve, every element', body,
            demo, links, demo_width=510, text_width=820, text_height=800))

# ------------------------------------------------------------------- t.nn.elu
body = """These are activation curves whose shape you can adjust.

The plain activations have one shape each. These take a parameter or two, 
which usually controls where the curve bends or how steeply - so the same node 
covers a family of shapes rather than one.

THE NODES:

t.nn.leaky_relu  relu, but with a gentle slope below zero instead of flat. 
                 'negative slope' sets how gentle - 0 is exactly relu
t.nn.elu         relu above zero, an exponential curve below it that levels 
                 off at minus 'alpha' rather than at zero
t.nn.celu        elu with the curve scaled so it stays continuous everywhere
t.nn.hardshrink  zero within plus or minus 'lambda', unchanged outside it
t.nn.softshrink  the same dead zone, but the surviving values are pulled 
                 towards zero by lambda rather than passed at full size
t.nn.hardtanh    flat below 'minimum' and above 'maximum', straight between
t.nn.rrelu       leaky_relu with a slope picked at random per element, between 
                 'lower' and 'upper'
t.nn.softplus    a smooth approximation of relu, with no corner at all. 
                 'beta' sets how sharp the bend is
t.nn.Threshold   below 'threshold', substitute 'replacement'; above, pass through

WHY A LEAKY relu EXISTS:
relu's output is exactly zero for every negative input, and so is its gradient - 
so an element that has gone negative stops learning entirely, and stays 
negative. A small slope below zero keeps a little signal alive. 
Whether it helps is arguable; that it is a response to a real failure is not.

hardshrink VERSUS softshrink:
Both silence a band around zero. hardshrink passes what survives untouched, 
so the output JUMPS from zero to lambda the moment a value crosses. softshrink 
subtracts lambda from what survives, so it starts from zero and grows smoothly. 
That is the same trade the noise_gate node's "squeeze" option offers, and for 
the same reason - the smooth one costs a little amplitude and removes the step.

SYNTAX:
t.nn.leaky_relu <slope>
t.nn.hardtanh <minimum> <maximum>

EXAMPLE:
t.nn.hardtanh -1.0 1.0

INPUTS and PARAMETERS:

tensor in:
The tensor. Receiving it triggers the node.

negative slope / alpha / lambda:
The single parameter, named for what it means in that node.

minimum / maximum (t.nn.hardtanh):
The two limits.

lower / upper (t.nn.rrelu):
The range the random slope is drawn from.

beta / threshold (t.nn.softplus):
How sharp the bend is, and above what value the node stops bothering with the 
exponential and passes through linearly.

threshold / replacement (t.nn.Threshold):
The level to test against, and what to substitute below it. 
Note that this substitutes rather than clamping - the replacement need not be 
related to the threshold at all.

OUTPUTS: 

output:
The bent tensor, in the same shape.

A NOTE ON rrelu:
Its slope is random per element and redrawn each time, so the same input does 
not give the same output twice. That is intended - it is a regularizer - but it 
makes the node unsuitable as a plain shaping curve."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 4.0 saw', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('saw', 4.0, 3.0, True)},
    {'key': 'lr', 'init': 't.nn.leaky_relu', 'pos': (30, 232), 'w': 200, 'h': 110},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 360), 'w': 208, 'h': 176,
     'props': PLOT(-3.2, 3.2)},
    {'key': 'c0', 'comment': True, 'text': 'drag negative slope up from zero\nat 0 it is exactly relu',
     'pos': (30, 545)},
    {'key': 'ht', 'init': 't.nn.hardtanh -1.0 1.0', 'pos': (280, 232), 'w': 220, 'h': 140,
     'props': {'minimum': -1.0, 'maximum': 1.0}},
    {'key': 'p2', 'init': 'plot', 'pos': (280, 390), 'w': 208, 'h': 176,
     'props': PLOT(-3.2, 3.2)},
    {'key': 'c2', 'comment': True, 'text': 'flat outside the two limits',
     'pos': (280, 575)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'lr', 'tensor in'), ('lr', 'output', 'p1', 'y'),
         ('sig', '', 'ht', 'tensor in'), ('ht', 'output', 'p2', 'y')]
print(build('t.nn.elu', 't.nn adjustable activations - curves with a knob', body,
            demo, links, demo_width=530, text_width=820, text_height=780))

# --------------------------------------------------------------- t.nn.softmax
body = """These turn a tensor of arbitrary numbers into something that behaves like a set 
of probabilities.

Every value comes out positive, and they all add up to 1. That makes the result 
a distribution over the elements: not "how big is each one" but "what SHARE of 
the whole does each one have". Feeding it the same numbers scaled up or down 
gives a different answer, because what matters is how far apart they are, 
not how large.

THE NODES:

t.nn.softmax          the shares, positive and summing to 1
t.nn.softmin          the same, but small values get the large shares
t.nn.log_softmax      the logarithm of softmax, computed stably
t.special.softmax     the same as t.nn.softmax
t.special.log_softmax the same as t.nn.log_softmax
t.nn.gumbel_softmax   softmax with randomness added, so it can be used to 
                      SAMPLE a choice rather than describe one

softmax IS A SOFT argmax:
The name is the clue. argmax tells you which element is largest, as a single 
index. softmax tells you the same thing spread out - the largest element gets 
the biggest share, and how much bigger depends on how far ahead it is. 
Values close together give a flat, undecided distribution; one value well ahead 
gives a spiky one that is nearly a choice.

That is why it is useful for weighting rather than deciding. Given a set of 
scores - how well each candidate matches, how active each joint is - softmax 
turns them into weights you can blend with, and the blend is smooth as the 
scores change where an argmax would jump.

WHY log_softmax EXISTS SEPARATELY:
Taking the softmax and then the logarithm loses precision badly, because the 
small shares underflow before the log can rescue them. log_softmax does both in 
one step, staying accurate. If you want log probabilities, always use this 
rather than composing the two.

gumbel_softmax SAMPLES:
Adding a particular kind of noise before the softmax makes drawing from it 
equivalent to sampling from the distribution the scores describe. 'tau' 
controls how spiky the result is - low values approach a hard one-of-N choice, 
high values stay blended. 'hard' snaps the output to exactly one-of-N while 
keeping it differentiable.

SYNTAX:
t.nn.softmax <dim>

EXAMPLE:
t.nn.softmax -1

INPUTS and PARAMETERS:

tensor in:
The tensor. Receiving it triggers the node.

dim:
WHICH AXIS the shares are taken across, and this is the setting that matters. 
The values along that axis sum to 1; every other axis is treated as a separate 
set. For a (frames, joints) tensor, dim over joints gives each frame's 
distribution across joints; dim over frames gives each joint's distribution 
over time. Those are entirely different questions, and getting it wrong is 
the usual softmax bug.

tau / hard (t.nn.gumbel_softmax):
How spiky, and whether to snap to a single choice.

OUTPUTS: 

output:
The distribution, in the same shape as the input.

A NOTE ON SCALE:
Because only the differences matter, multiplying the input by a constant 
sharpens or flattens the result - a large multiplier makes it nearly a hard 
choice, a small one makes it nearly uniform. That multiplier is the 
"temperature" you see referred to elsewhere, and if a softmax is too decisive 
or too vague, scaling its input is the adjustment."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 8', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 120), 'w': 208, 'h': 148,
     'props': HM(8, 0.0, 1.0)},
    {'key': 'c0', 'comment': True, 'text': 'eight arbitrary scores', 'pos': (30, 310)},
    {'key': 'sm', 'init': 't.nn.softmax', 'pos': (30, 350), 'w': 190, 'h': 100},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (250, 350), 'w': 208, 'h': 148,
     'props': HM(8, 0.0, 0.4)},
    {'key': 'c1', 'comment': True, 'text': 'shares of the whole: they sum to 1',
     'pos': (30, 465)},
    {'key': 'mul', 'init': '* 8.0', 'pos': (30, 510), 'w': 140, 'h': 70,
     'props': {'operand': 8.0}},
    {'key': 'sm2', 'init': 't.nn.softmax', 'pos': (30, 595), 'w': 190, 'h': 100},
    {'key': 'hm3', 'init': 'heat_map', 'pos': (250, 595), 'w': 208, 'h': 148,
     'props': HM(8, 0.0, 1.0)},
    {'key': 'c2', 'comment': True, 'text': 'scaled up first: nearly a hard choice\nthat multiplier is the temperature',
     'pos': (30, 710)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'sm', 'tensor in'), ('sm', 'output', 'hm2', 'y'),
         ('rnd', 'random tensor', 'mul', 'in'),
         ('mul', 'result', 'sm2', 'tensor in'), ('sm2', 'output', 'hm3', 'y')]
print(build('t.nn.softmax', 't.nn.softmax - scores into shares', body, demo, links,
            demo_width=480, text_width=820, text_height=780))

# ---------------------------------------------------------- t.special.gammainc
body = """The gamma function and its relatives.

The gamma function extends the factorial to values that are not whole numbers. 
Where a factorial only makes sense at 1, 2, 3, gamma is defined smoothly in 
between, and it turns up throughout statistics because it is what normalises 
most of the continuous distributions.

THE NODES:

t.special.gammainc      the regularised lower incomplete gamma function
t.special.gammaincc     the upper one - the complement, so the two sum to 1
t.special.polygamma     the nth derivative of the log gamma function
t.special.multigammaln  the log of the multivariate gamma function

WHAT gammainc IS ACTUALLY FOR:
It is the cumulative distribution of a gamma-distributed quantity - how much of 
the distribution lies below a given point. So given a waiting time, or a 
duration, or an accumulated amount, it answers "what fraction of the time would 
we expect to see a value at least this small". 

That makes it the node for turning a measurement into a probability, which is 
how you say "this reading is unusually large" in a way that does not depend on 
the units. gammaincc answers the other half - "how surprising is a value this 
big" - and is the one you want for a tail probability.

polygamma AND multigammaln:
Both are machinery rather than everyday tools. polygamma at n=0 is the digamma 
function, which is the derivative of log gamma and appears whenever you 
differentiate something involving a gamma distribution. multigammaln is the 
multivariate version's logarithm, needed for distributions over matrices.

SYNTAX:
t.special.gammainc
t.special.polygamma <n>

EXAMPLE:
t.special.gammainc

INPUTS and PARAMETERS:

tensor 1 in / tensor 2 in (gammainc, gammaincc):
The shape parameter and the value to evaluate at.

tensor in / n (polygamma):
The values, and which derivative - 0 is digamma, 1 is trigamma.

tensor in / p (multigammaln):
The values, and the dimension.

OUTPUTS: 

tensor out:
The result, element by element.

WORKING IN LOGS:
Gamma values become enormous very quickly - the same reason factorials do - and 
overflow to infinity for quite modest inputs. That is why multigammaln returns 
the logarithm rather than the value, and why t.special.gammaln (on the 
t.special help patch) exists alongside the plain gamma. If a gamma calculation 
is returning infinity, working in logs is the fix rather than a workaround."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'r1', 'init': 't.rand 8', 'pos': (30, 120), 'w': 180, 'h': 180,
     'props': {'min': 0.5, 'max': 3.0}},
    {'key': 'r2', 'init': 't.rand 8', 'pos': (250, 120), 'w': 180, 'h': 180,
     'props': {'min': 0.5, 'max': 3.0}},
    {'key': 'c0', 'comment': True, 'text': 'a shape parameter and a value',
     'pos': (30, 310)},
    {'key': 'gi', 'init': 't.special.gammainc', 'pos': (30, 350), 'w': 220, 'h': 90},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 455), 'w': 208, 'h': 148,
     'props': HM(8, 0.0, 1.0)},
    {'key': 'c1', 'comment': True, 'text': 'the fraction of the distribution below',
     'pos': (30, 615)},
    {'key': 'gc', 'init': 't.special.gammaincc', 'pos': (280, 350), 'w': 220, 'h': 90},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (280, 455), 'w': 208, 'h': 148,
     'props': HM(8, 0.0, 1.0)},
    {'key': 'c2', 'comment': True, 'text': 'and the tail above: the two sum to 1',
     'pos': (280, 615)},
]
links = [('btn', '', 'r1', '###input'), ('btn', '', 'r2', '###input'),
         ('r2', 'random tensor', 'gi', 'tensor 2 in'),
         ('r1', 'random tensor', 'gi', 'tensor 1 in'), ('gi', 'tensor out', 'hm', 'y'),
         ('r2', 'random tensor', 'gc', 'tensor 2 in'),
         ('r1', 'random tensor', 'gc', 'tensor 1 in'), ('gc', 'tensor out', 'hm2', 'y')]
print(build('t.special.gammainc', 't.special gamma - factorials, continued', body,
            demo, links, demo_width=520, text_width=810, text_height=720))

# ------------------------------------------------------------- t.special.xlogy
body = """x times a logarithm, made safe for x = 0.

THE NODES:

t.special.xlogy    x times the log of y, defined as 0 when x is 0
t.special.xlog1py  x times the log of (1 plus y), same protection

xlogy IS NOT JUST A MULTIPLY:
Written out directly, x times log(y) gives not-a-number when x is 0 and y is 0,
because log(0) is minus infinity and 0 times infinity is undefined. But the
limit is 0, and 0 is the answer you want. This node returns it: whenever x is 0
the result is 0, whatever y is (unless y is itself not-a-number).

That matters because x times log(y) is the shape of every entropy and
cross-entropy calculation - a probability multiplied by its own log - and zero
probabilities are entirely normal there. Doing it by hand produces NaN that then
poisons every subsequent sum. Using this node does not.

xlog1py:
The same, with log(1 plus y) in place of log(y). Taking log(1 plus y) directly
loses accuracy when y is very small, because adding 1 first throws away y's
digits; this computes it without that loss - the same reason log1p exists. Here
the protected case is y = -1, where log(1 plus y) is minus infinity.

SYNTAX:
t.special.xlogy
t.special.xlog1py

INPUTS and PARAMETERS:

tensor 1 or number in:
x, the multiplier. Receiving it does the work.

tensor 2 or number in:
y, the value whose logarithm is taken. Either input may be a single number
rather than a tensor, applied everywhere. This one is only stored when it
arrives, so send it BEFORE x - the example connects it first for that reason.

OUTPUTS:

tensor out:
The result, element by element.

WHEN TO SUSPECT YOU NEED THIS:
When a calculation over a tensor produces NaN somewhere in the middle and you
cannot see where, a zero going through a logarithm is the first thing to
suspect.

RELATED:
t.special.logits - the inverse of the sigmoid, the other log near the edges."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 260, 'h': 42,
     'props': {'text in': '0.0 0.25 0.5 0.75', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'the same list goes to x and y -\nnote the zero at the start', 'pos': (30, 168)},
    {'key': 'xl', 'init': 't.special.xlogy', 'pos': (30, 225), 'w': 240, 'h': 90},
    {'key': 'l1', 'init': 'list', 'pos': (30, 330), 'w': 320, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'x times log x, with 0 log 0 giving 0\ndoing it by hand would give NaN there',
     'pos': (30, 380)},
    {'key': 'x1', 'init': 't.special.xlog1py', 'pos': (30, 450), 'w': 240, 'h': 90},
    {'key': 'l2', 'init': 'list', 'pos': (30, 555), 'w': 320, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'x times log (1 plus x)',
     'pos': (30, 605)},
]
links = [('btn', '', 'm1', ''),
         ('m1', 'message out', 'xl', 'tensor 2 or number in'),
         ('m1', 'message out', 'xl', 'tensor 1 or number in'),
         ('xl', 'tensor out', 'l1', ''),
         ('m1', 'message out', 'x1', 'tensor 2 or number in'),
         ('m1', 'message out', 'x1', 'tensor 1 or number in'),
         ('x1', 'tensor out', 'l2', '')]
print(build('t.special.xlogy', 't.special.xlogy - x log y without the NaN', body,
            demo, links, demo_width=480, text_width=800, text_height=660))

# ------------------------------------------------------------ t.special.logits
body = """The inverse of the sigmoid: turns a probability back into a score.

WHAT IT GIVES YOU:
Given a probability between 0 and 1, it returns the unbounded score that a
sigmoid would have turned into that probability - the log of p over (1 minus p).
0.5 becomes 0, values above 0.5 become positive, values below become negative,
and the ends stretch out without limit.

That is useful whenever you want to add, average or smooth probabilities: do it
on the logits, where 0.99 and 0.999 are properly far apart, then put the result
back through a sigmoid.

THE ENDS ARE INFINITE - eps IS THE SAFETY MARGIN:
At exactly 0 the answer is minus infinity, and at exactly 1 it is plus infinity.
'eps' clamps the input to between eps and 1 minus eps first, so the ends come
out as large finite numbers instead.

THE DEFAULT eps IS TOO SMALL FOR THE TOP END:
eps starts at 0.00000001. Tensors here are ordinary 32-bit floats, and in those
1 minus 0.00000001 rounds back to exactly 1 - so a probability of 1 still comes
out as infinity. The bottom end is protected (0 gives about -18.4). To keep both
ends finite, set eps to 0.000001 or larger; the example shows both settings.

SYNTAX:
t.special.logits

INPUTS and PARAMETERS:

tensor in:
The probabilities, 0 to 1. Receiving them does the work. Values outside 0..1
are clamped by eps too, so they come out at the ends rather than as errors.

eps:
How far from 0 and 1 to hold the input. Default 0.00000001.

OUTPUTS:

tensor out:
The scores, element by element.

RELATED:
t.special.expit (and t.nn.sigmoid) go the other way, score to probability."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 300, 'h': 42,
     'props': {'text in': '0.0 0.25 0.5 0.75 1.0', 'font size': '24'}},
    {'key': 'lg', 'init': 't.special.logits', 'pos': (30, 205), 'w': 220, 'h': 90},
    {'key': 'l1', 'init': 'list', 'pos': (30, 310), 'w': 360, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'default eps: the 1.0 still comes out as inf',
     'pos': (30, 360)},
    {'key': 'lg2', 'init': 't.special.logits', 'pos': (30, 420), 'w': 220, 'h': 90,
     'props': {'eps': 0.000001}},
    {'key': 'l2', 'init': 'list', 'pos': (30, 525), 'w': 360, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'eps 0.000001: both ends finite',
     'pos': (30, 575)},
]
links = [('btn', '', 'm1', ''),
         ('m1', 'message out', 'lg', 'tensor in'), ('lg', 'tensor out', 'l1', ''),
         ('m1', 'message out', 'lg2', 'tensor in'), ('lg2', 'tensor out', 'l2', '')]
print(build('t.special.logits', 't.special.logits - probability back to score', body,
            demo, links, demo_width=480, text_width=800, text_height=660))

# ------------------------------------------------------------- t.special.zeta
body = """The Hurwitz zeta function.

WHAT IT COMPUTES:
zeta(x, q) is the sum, over k = 0, 1, 2, ..., of 1 divided by (k plus q) raised
to the power x. With q = 1 it is the ordinary Riemann zeta function: zeta(2, 1)
is pi squared over 6, about 1.645.

The sum only settles to a finite value when x is greater than 1. At x = 1 the
result is infinity, and below 1 it is not-a-number. q should be positive.

WHERE IT TURNS UP:
It is the normalising constant of discrete power laws - Zipf-like
distributions, where the k-th most common thing occurs in proportion to 1 over
k to the x - and it appears in the derivatives of the gamma function
(t.special.polygamma is built from it).

SYNTAX:
t.special.zeta

INPUTS and PARAMETERS:

tensor 1 or number in:
x, the power. Receiving it does the work.

tensor 2 or number in:
q, the offset. Either input may be a single number rather than a tensor,
applied everywhere. This one is only stored when it arrives, so send it
BEFORE x.

OUTPUTS:

tensor out:
The result, element by element."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'mq', 'init': 'message', 'pos': (300, 118), 'w': 120, 'h': 42,
     'props': {'text in': '1.0', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'q = 1: the Riemann zeta', 'pos': (300, 168)},
    {'key': 'mx', 'init': 'message', 'pos': (30, 118), 'w': 220, 'h': 42,
     'props': {'text in': '2.0 3.0 4.0', 'font size': '24'}},
    {'key': 'zt', 'init': 't.special.zeta', 'pos': (30, 225), 'w': 240, 'h': 90},
    {'key': 'l1', 'init': 'list', 'pos': (30, 330), 'w': 320, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': '1.645 (pi squared over 6), 1.202, 1.082',
     'pos': (30, 380)},
]
links = [('btn', '', 'mq', ''), ('btn', '', 'mx', ''),
         ('mq', 'message out', 'zt', 'tensor 2 or number in'),
         ('mx', 'message out', 'zt', 'tensor 1 or number in'),
         ('zt', 'tensor out', 'l1', '')]
print(build('t.special.zeta', 't.special.zeta - the Hurwitz zeta function', body,
            demo, links, demo_width=480, text_width=800, text_height=560))

