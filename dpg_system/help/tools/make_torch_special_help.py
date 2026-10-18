"""t.special.*: one page per special function.

These were the hand-made t.special family page (one t.linspace feeding all
of them into a wall of plots) and four pages that each bundled a few
(t.special.gammainc, t.special.xlogy) or documented one (t.special.logits,
t.special.zeta). Each node now has its own page: a plot of the input range
beside a plot of the function over that range, a live range or second
argument, verified prose and a SEE ALSO row.

t.special.softmax and t.special.log_softmax are documented elsewhere.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import PLOT, FLT


def see_also(y, stems, x=30, right=580):
    """A SEE ALSO row: one help_link per related page, left to right, wrapping
    onto a second row rather than running under the prose."""
    row = []
    x0 = x
    for s in stems:
        w = int(len(s) * 7.7) + 30
        if x > x0 and x + w > right:
            x, y = x0, y + 44
        row.append({'key': 'sa_' + s, 'init': 'help_link ' + s, 'pos': (x, y),
                    'w': w, 'h': 30})
        x += w + 14
    if x + 200 > right:
        x, y = x0, y + 44
    row.append({'key': 'sa_c', 'comment': True, 'text': 'SEE ALSO - click to open',
                'pos': (x, y)})
    return row


def ARRAY_PLOT(miny, maxy, count=200, width=300):
    """A plot that draws one whole 1-D tensor as a curve, first value on the left."""
    return dict(PLOT(miny, maxy, count),
                **{'width': width, 'update style': 'buffer holds one sample of input'})


METRO = lambda ms: {'on': True, 'period': float(ms), 'units': 'milliseconds'}

STEPS = 200


def text_h(body):
    """Height of the prose block: 15px per line at text size 30, plus a margin."""
    return (body.count('\n') + 1) * 15 + 30


def fmt(v):
    """A number as the prose should show it: no trailing .0 on whole numbers."""
    return ('%g' % v)


def linspace(a, b):
    return {'key': 'ls', 'init': 't.linspace %s %s %d' % (fmt(a), fmt(b), STEPS),
            'pos': (30, 150), 'w': 190, 'h': 170,
            'props': {'start': float(a), 'stop': float(b), 'steps': STEPS}}


INPUT_ONE = """tensor in:
A tensor - or a number or a list, which is turned into one. Receiving it
triggers the node. Each element is worked on separately, so the output is
the same shape as the input. Whole-number input comes out as floats."""

READ_ONE = """READING THE DEMO:
The metro re-clicks t.linspace ten times a second. t.linspace makes 200
evenly spaced values from its start to its stop ({a} to {b} here), and the
upper plot draws them: a straight ramp. The lower plot is {short} of each
of those 200 values, so it is the graph of the function from {a} to {b},
drawn from {lo} (bottom) to {hi} (top). Drag start and stop on t.linspace
to look at another stretch{hint}."""


# ---------------------------------------------------------------------------
# single-input functions (TorchSpecialNode)
# Each: what (first line), use (one sentence), curve, edges, demo range,
# plot range, what the lower plot shows, see also, related paragraph.

SINGLE = [
 dict(name='t.special.airy_ai', short='Ai(x)', title='the Airy function',
  what="""Applies the Airy function Ai to every element of a tensor.""",
  use="""It is used for the light near a caustic - the bright edge of a rainbow,
the bright curve in a coffee cup - and for a quantum particle in a uniform
field: it is the shape a wave takes where it turns back.""",
  curve="""Ai is the solution of y'' = x y that dies away for positive x. To the
right of zero it falls smoothly towards 0, faster than any exponential. To
the left it oscillates, and the further left you go the faster it wiggles
and the smaller the swings get. Ai(0) is about 0.355; its first zero is at
about -2.338.""",
  edges="""Defined for every real number. Very large positive inputs give 0; very
large negative ones oscillate too fast to be useful in 32-bit floats.""",
  a=-10, b=4, lo=-0.5, hi=0.6, out='out: Ai - wiggling on the left, dying away on the right',
  hint=' (start -20 shows the wiggles getting faster)',
  see=['t.special.bessel_j0', 't.special.bessel_j1', 't.special.spherical_bessel_j0',
       't.special.scaled_modified_bessel_k0'],
  related="""t.special.bessel_j0 and t.special.bessel_j1 are the other wave
shapes in this set; Ai is the one for a wave meeting a turning point,
Bessel functions are the ones for circular or cylindrical waves."""),

 dict(name='t.special.bessel_j0', short='J0(x)', title='Bessel function of order 0',
  what="""Applies the Bessel function of the first kind of order 0, J0, to every
element of a tensor.""",
  use="""It is the shape of waves on anything circular - the ripples on a pond,
the way a round drum skin vibrates - and in FM synthesis J0 of the
modulation index is how loud the carrier stays.""",
  curve="""J0 starts at 1 at x = 0 and then oscillates like a cosine whose swings
slowly shrink (by about 1 over the square root of x). Its zeros are not
evenly spaced at first - 2.405, 5.520, 8.654, 11.79 - but settle towards a
gap of pi. It is symmetric: J0(-x) = J0(x).""",
  edges="""Defined for every real number; very large inputs give small values that
keep oscillating.""",
  a=0, b=20, lo=-0.5, hi=1.05, out='out: J0 - starts at 1, swings that slowly shrink',
  hint='',
  see=['t.special.bessel_j1', 't.special.spherical_bessel_j0', 't.special.i0',
       't.special.airy_ai', 't.special.sinc'],
  related="""t.special.bessel_j1 is the next order. t.special.spherical_bessel_j0
is the spherical (3-D) version, sin(x) / x. t.special.i0 is the
'modified' Bessel function - the same equation with the sign flipped,
which grows instead of oscillating."""),

 dict(name='t.special.bessel_j1', short='J1(x)', title='Bessel function of order 1',
  what="""Applies the Bessel function of the first kind of order 1, J1, to every
element of a tensor.""",
  use="""It gives the size of the bright rings around a point of light seen
through a round lens (the Airy disk), and in FM synthesis J1 of the
modulation index is how loud the first pair of sidebands is.""",
  curve="""J1 starts at 0, rises to its peak of about 0.582 at x = 1.84, then
oscillates with slowly shrinking swings, a quarter-cycle behind J0. Its
zeros are 0, 3.832, 7.016, 10.17 ... It is antisymmetric: J1(-x) = -J1(x).""",
  edges="""Defined for every real number.""",
  a=0, b=20, lo=-0.4, hi=0.7, out='out: J1 - starts at 0, peaks near 1.84, then swings',
  hint='',
  see=['t.special.bessel_j0', 't.special.i1', 't.special.spherical_bessel_j0',
       't.special.airy_ai'],
  related="""t.special.bessel_j0 is order 0 (carrier where this is the first
sideband). t.special.i1 is the modified version, which grows instead of
oscillating."""),

 dict(name='t.special.digamma', short='digamma', title='the derivative of log gamma',
  what="""Applies the digamma function - the slope of the log of the gamma
function - to every element of a tensor.""",
  use="""It turns up whenever you fit or differentiate a gamma, beta or Dirichlet
distribution (the average log of a gamma-distributed value is digamma of its
shape), and digamma(n + 1) + 0.5772 is the harmonic sum 1 + 1/2 + ... + 1/n.""",
  curve="""For positive x it rises steadily, like a logarithm: very negative near 0,
crossing zero at about 1.4616 (where log gamma is lowest), -0.5772 at 1,
about 1.7 at 6. Each step of 1 adds 1/x: digamma(x + 1) = digamma(x) + 1/x.""",
  edges="""0 gives minus infinity. Negative whole numbers give NaN (not a number),
and between them the curve swoops from plus to minus infinity, so keep
the input positive unless you mean it.""",
  a=0.25, b=6, lo=-5, hi=2, out='out: digamma - a log-like rise, crossing 0 near 1.46',
  hint=' (keep start above 0)',
  see=['t.special.gammaln', 't.special.polygamma', 't.special.multigammaln',
       't.special.zeta'],
  related="""t.special.gammaln is the function this is the slope of.
t.special.polygamma with n = 0 is the same thing, and higher n gives its
further derivatives."""),

 dict(name='t.special.entr', short='entr', title='one term of an entropy',
  what="""Computes minus x times the natural log of x for every element of a
tensor - one term of an entropy.""",
  use="""Summed over a probability distribution (t.sum after this node) it gives
the distribution's entropy in nats: how spread out, or how unpredictable,
it is.""",
  curve="""Over 0 to 1 it is a hump: 0 at 0, rising to its peak of 1/e (0.368) at
x = 1/e (0.368), back to 0 at 1, and negative beyond 1. A probability of 0
contributes nothing and so does a certainty of 1; middling probabilities
contribute most.""",
  edges="""0 gives exactly 0 (not the NaN you get computing 0 times log 0 by hand).
Negative inputs give minus infinity.""",
  a=0, b=1, lo=-0.05, hi=0.4, out='out: a hump, 0 at both ends, 0.368 at its peak',
  hint='',
  see=['t.special.xlogy', 't.special.logits', 't.nn.softmax', 't.sum'],
  related="""t.special.xlogy computes x times log y with the same care at 0 -
entr(p) is minus xlogy(p, p). t.nn.softmax turns scores into the
probabilities you would take the entropy of."""),

 dict(name='t.special.erf', short='erf', title='the error function',
  what="""Applies the error function, erf, to every element of a tensor.""",
  use="""It is the area under a bell curve, so it turns a distance measured in
standard deviations into a probability, and it makes a smooth S-shaped step
for edges and transitions.""",
  curve="""An S from -1 to 1 through 0, steepest at 0 and flattening out quickly:
erf(1) = 0.843, erf(2) = 0.995, erf(3) = 0.99998. It is antisymmetric:
erf(-x) = -erf(x). erf(x / 1.414) is the chance a normal value lies
within x standard deviations of the mean.""",
  edges="""Defined for every real number; far out it is exactly 1 or -1 in 32-bit
floats.""",
  a=-3, b=3, lo=-1.1, hi=1.1, out='out: erf - an S from -1 to 1',
  hint='',
  see=['t.special.erfc', 't.special.erfinv', 't.special.ndtr', 't.special.erfcx',
       't.dist.normal'],
  related="""t.special.erfc is 1 minus this, kept accurate far out.
t.special.erfinv undoes it. t.special.ndtr is the same curve rescaled to
the normal distribution's 0 to 1 probability."""),

 dict(name='t.special.erfc', short='erfc', title='the complementary error function',
  what="""Applies the complementary error function, erfc = 1 - erf, to every
element of a tensor.""",
  use="""It gives tail probabilities - how likely a normal value is to land this
far out or further - and stays accurate where 1 - erf would have rounded to
nothing.""",
  curve="""A falling S from 2 to 0, passing 1 at x = 0. erfc(2) = 0.0047,
erfc(3) = 0.000022. The point of the separate node is the right-hand tail:
at x = 5, 1 - erf(5) rounds to exactly 0 in 32-bit floats, but erfc(5)
correctly gives 0.0000000000015.""",
  edges="""Beyond about x = 10 even erfc is too small for a 32-bit float and
becomes 0; t.special.erfcx keeps going.""",
  a=-3, b=3, lo=-0.1, hi=2.1, out='out: erfc - a falling S from 2 to 0',
  hint='',
  see=['t.special.erf', 't.special.erfcx', 't.special.ndtr', 't.special.logndtr',
       't.special.erfinv'],
  related="""t.special.erf is the rising version. t.special.erfcx is erfc
multiplied by e to the x squared, which keeps the far tail in range.
t.special.logndtr gives the log of a normal tail probability directly."""),

 dict(name='t.special.erfcx', short='erfcx', title='the scaled complementary error function',
  what="""Applies the scaled complementary error function - erfc(x) times e to the
power x squared - to every element of a tensor.""",
  use="""It is how you compute with far-out tail probabilities without them
underflowing to 0: carry erfcx and the e to the x squared factor
separately, or take logs.""",
  curve="""1 at x = 0, then a slow decline, approaching 1 / (x times the square
root of pi): 0.43 at 1, 0.056 at 10. Compare erfc(10), which is 2.8e-45 -
the smallest number a 32-bit float can hold - and gives exactly 0 a little
further on.""",
  edges="""For negative x it grows very fast (16206 at -3; about 2 times e to the x
squared) and overflows to infinity before -10. Use it on the positive
side.""",
  a=0, b=10, lo=0, hi=1.05, out='out: erfcx - a slow decline from 1',
  hint=' (start below -1 shoots off the top)',
  see=['t.special.erfc', 't.special.erf', 't.special.logndtr', 't.special.i0e'],
  related="""t.special.erfc is the unscaled version. t.special.i0e and
t.special.i1e use the same trick (divide out the exponential) for the
Bessel functions."""),

 dict(name='t.special.erfinv', short='erfinv', title='the inverse error function',
  what="""Applies the inverse error function to every element of a tensor: given
erf(x), it returns x.""",
  use="""It turns evenly spread random numbers into normally distributed ones -
1.414 times erfinv(2u - 1), for u between 0 and 1 - and in general turns a
probability back into a distance in standard deviations.""",
  curve="""Defined only between -1 and 1. An S lying on its side: nearly straight
through the middle (erfinv(0.5) = 0.477), then shooting up near 1 and
down near -1. erfinv(0.99) = 1.82, erfinv(0.9999) = 2.75.""",
  edges="""1 and -1 give plus and minus infinity; anything outside -1 to 1 gives NaN
(not a number). That is why the demo runs from -0.99 to 0.99.""",
  a=-0.99, b=0.99, lo=-2, hi=2, out='out: erfinv - steepening towards both ends',
  hint=' (inside -1 to 1)',
  see=['t.special.erf', 't.special.ndtri', 't.special.erfc', 't.special.ndtr'],
  related="""t.special.erf is the function this undoes. t.special.ndtri is the
same idea for the normal distribution, scaled so that a probability 0 to 1
gives a value in standard deviations."""),

 dict(name='t.special.exp2', short='2 to the x', title='2 to the power x',
  what="""Raises 2 to the power of every element of a tensor.""",
  use="""It is the natural exponential for anything measured in octaves or
doublings: a pitch n semitones up has frequency ratio 2 to the n/12, and a
value that doubles every step grows as 2 to the step.""",
  curve="""Always positive, rising faster and faster: 1 at 0, 2 at 1, 16 at 4, and
halving with each step left: 0.5 at -1, 0.0625 at -4. Whole-number inputs
give exact powers of 2.""",
  edges="""Overflows to infinity above about 128 in 32-bit floats, and underflows to
0 below about -149.""",
  a=-4, b=4, lo=0, hi=16.5, out='out: 2 to the x - 0.0625 up to 16',
  hint='',
  see=['t.special.expm1', 'exp', 'log2', 't.logspace'],
  related="""exp raises e rather than 2; log2 undoes this. t.logspace makes a
series of evenly spaced powers directly."""),

 dict(name='t.special.expit', short='expit', title='the logistic sigmoid',
  what="""Applies the logistic sigmoid, 1 / (1 + e to the minus x), to every
element of a tensor.""",
  use="""It squeezes any score into a probability between 0 and 1 - the last step
of a classifier, or a smooth switch that is off for very negative input and
on for very positive input.""",
  curve="""An S from 0 to 1, passing 0.5 at x = 0. It is never quite 0 or 1:
expit(4) = 0.982, expit(8) = 0.99966. expit(-x) = 1 - expit(x).""",
  edges="""Defined for every real number and always strictly between 0 and 1 (in
32-bit floats it reaches exactly 1 near x = 17).""",
  a=-8, b=8, lo=-0.05, hi=1.05, out='out: an S from 0 to 1, 0.5 in the middle',
  hint='',
  see=['t.nn.sigmoid', 't.special.logits', 't.nn.tanh', 't.special.ndtr',
       't.nn.softmax'],
  related="""t.nn.sigmoid gives exactly the same numbers (torch implements expit as
the sigmoid) - it is the same curve under its neural-network name.
t.special.logits undoes it. t.nn.tanh is the same S stretched to -1..1,
and t.special.ndtr is a similar S with slightly thinner tails."""),

 dict(name='t.special.expm1', short='e to the x minus 1', title='e to the x, minus 1',
  what="""Computes e to the power x, minus 1, for every element of a tensor - kept
accurate when x is tiny.""",
  use="""It is the growth from a small rate: compound interest, a small gain in
decibels or a small log-ratio turned back into a fractional change, where
writing exp(x) - 1 would lose the digits that matter.""",
  curve="""An exponential shifted down by 1: 0 at 0, approaching -1 on the left,
rising steeply on the right (6.39 at 2). Near 0 it is almost exactly x.
For x = 0.00000001, exp then subtract gives 0 in 32-bit floats; this node
gives 0.00000001.""",
  edges="""Overflows to infinity above about 88 in 32-bit floats.""",
  a=-3, b=2, lo=-1.5, hi=6.5, out='out: e to the x minus 1, flattening at -1 on the left',
  hint='',
  see=['t.special.log1p', 't.special.exp2', 'exp', 't.special.xlog1py'],
  related="""t.special.log1p is its inverse, with the same care near zero. exp is
the plain exponential."""),

 dict(name='t.special.gammaln', short='log gamma', title='log of the gamma function',
  what="""Computes the natural log of the absolute value of the gamma function for
every element of a tensor.""",
  use="""It gives logs of factorials - gammaln(n + 1) is log(n!) - so it is how
you compute counts of combinations and probabilities that involve huge
factorials without overflowing.""",
  curve="""For positive x: high near 0, falling to 0 at x = 1, a shallow dip to
-0.121 at 1.46, back to 0 at x = 2, then rising steadily and a little
faster than a straight line. gammaln(11) = 15.104 = log(10!). The gamma
function itself would overflow past x = 35; its log is still a modest
number there.""",
  edges="""0 and negative whole numbers give infinity. Between the negative whole
numbers it gives the log of the absolute value, so the sign of gamma is
lost there.""",
  a=0.1, b=10, lo=-1, hi=13, out='out: log gamma - a dip below 0 between 1 and 2, then rising',
  hint=' (keep start above 0)',
  see=['t.special.digamma', 't.special.multigammaln', 't.special.polygamma',
       't.special.gammainc', 't.dist.gamma'],
  related="""t.special.digamma is its slope. t.special.multigammaln is the
version for several dimensions (with p = 1 it equals this)."""),

 dict(name='t.special.i0', short='I0(x)', title='modified Bessel function of order 0',
  what="""Applies the modified Bessel function of the first kind of order 0, I0,
to every element of a tensor.""",
  use="""It shapes the Kaiser window used in filter design and is the normalising
constant of the von Mises distribution (a bell curve wrapped around a
circle, for angles and directions).""",
  curve="""1 at 0, symmetric, curving up on both sides and then growing roughly like
e to the |x| (divided by the square root of 2 pi |x|): I0(2) = 2.28,
I0(4) = 11.3. It never oscillates - it is the Bessel equation with a sign
flipped.""",
  edges="""Overflows to infinity past about |x| = 90 in 32-bit floats;
t.special.i0e avoids that.""",
  a=-4, b=4, lo=0, hi=12, out='out: I0 - 1 in the middle, rising steeply both sides',
  hint='',
  see=['t.special.i0e', 't.special.i1', 't.special.bessel_j0', 't.window.kaiser',
       't.dist.von_mises'],
  related="""t.special.i0e is the same divided by e to the |x|, which stays in range.
t.special.i1 is order 1. t.special.bessel_j0 is the oscillating
cousin. t.window.kaiser builds the window this function shapes."""),

 dict(name='t.special.i0e', short='I0e(x)', title='exponentially scaled I0',
  what="""Applies the exponentially scaled modified Bessel function of order 0 -
I0(x) times e to the minus |x| - to every element of a tensor.""",
  use="""It is I0 for large inputs: wherever I0 would overflow (a von Mises
distribution with a high concentration, say), work with this and keep the
exponential factor separately, or in logs.""",
  curve="""1 at 0, symmetric, falling slowly on both sides, about like 1 over the
square root of 2 pi |x|: 0.47 at 2, 0.128 at 10, 0.04 at 100 - where I0
itself is already infinite in 32-bit floats.""",
  edges="""Defined and finite for every real number.""",
  a=-10, b=10, lo=0, hi=1.05, out='out: a peak of 1 at 0, a slow fall either side',
  hint='',
  see=['t.special.i0', 't.special.i1e', 't.special.scaled_modified_bessel_k0',
       't.special.erfcx'],
  related="""t.special.i0 is the unscaled function. t.special.i1e is the order-1
version, and t.special.scaled_modified_bessel_k0 the same trick for the
decaying K0."""),

 dict(name='t.special.i1', short='I1(x)', title='modified Bessel function of order 1',
  what="""Applies the modified Bessel function of the first kind of order 1, I1,
to every element of a tensor.""",
  use="""It is the slope of I0, so it appears wherever I0 does and something is
being fitted or differentiated - the mean direction of a von Mises
distribution, for example, involves I1 / I0.""",
  curve="""0 at 0 and antisymmetric (I1(-x) = -I1(x)): nearly a straight line x/2
near 0, then growing roughly like e to the |x|: I1(2) = 1.59, I1(4) = 9.76.
It never oscillates.""",
  edges="""Overflows to infinity past about |x| = 90 in 32-bit floats;
t.special.i1e avoids that.""",
  a=-4, b=4, lo=-10, hi=10, out='out: I1 - through 0, curving away both sides',
  hint='',
  see=['t.special.i1e', 't.special.i0', 't.special.bessel_j1'],
  related="""t.special.i1e is the version divided by e to the |x|. t.special.i0 is
order 0. t.special.bessel_j1 is the oscillating cousin."""),

 dict(name='t.special.i1e', short='I1e(x)', title='exponentially scaled I1',
  what="""Applies the exponentially scaled modified Bessel function of order 1 -
I1(x) times e to the minus |x| - to every element of a tensor.""",
  use="""It is I1 for large inputs, where I1 itself would overflow; the ratio
I1 / I0 is i1e / i0e and stays finite.""",
  curve="""0 at 0 and antisymmetric, rising to a peak of about 0.219 at x = 1.55,
then falling slowly back towards 0 (0.12 at 10).""",
  edges="""Defined and finite for every real number.""",
  a=-10, b=10, lo=-0.25, hi=0.25, out='out: through 0, peaks near plus and minus 1.55',
  hint='',
  see=['t.special.i1', 't.special.i0e', 't.special.scaled_modified_bessel_k0'],
  related="""t.special.i1 is the unscaled function, t.special.i0e the order-0
version."""),

 dict(name='t.special.log1p', short='log(1 + x)', title='the log of 1 plus x',
  what="""Computes the natural log of (1 + x) for every element of a tensor - kept
accurate when x is tiny.""",
  use="""It turns small fractional changes (a 0.1% gain, a tiny probability) into
logs without losing them: log(1 + 0.00000001) computed by adding first
gives 0 in 32-bit floats, this gives 0.00000001.""",
  curve="""0 at 0, rising ever more slowly to the right (1.79 at 5), plunging
towards minus infinity as x approaches -1. Near 0 it is almost exactly x.""",
  edges="""-1 gives minus infinity; anything below -1 gives NaN (not a number). The
demo starts at -0.9 for that reason.""",
  a=-0.9, b=5, lo=-2.5, hi=2, out='out: log of 1 plus x, plunging as x nears -1',
  hint=' (keep start above -1)',
  see=['t.special.expm1', 't.special.xlog1py', 'log2', 't.special.entr'],
  related="""t.special.expm1 is its inverse, with the same care near zero.
t.special.xlog1py multiplies the result by a second value, safely when
that value is 0."""),

 dict(name='t.special.logndtr', short='log of ndtr', title='log of the normal distribution function',
  what="""Computes the natural log of the standard normal cumulative distribution
function for every element of a tensor.""",
  use="""It gives log-probabilities of far-out events directly - the chance of a
reading 10 standard deviations below the mean is far too small for a
float, but its log is a perfectly ordinary number.""",
  curve="""Close to 0 (just below it) on the right, where the probability is
nearly 1, and curving down ever more steeply to the left, like a downward
parabola: -0.69 at 0 (log of 0.5), -15.06 at -5, -804.6 at -40 - where
ndtr itself has long since underflowed to exactly 0.""",
  edges="""Defined for every real number and never above 0.""",
  a=-5, b=3, lo=-16, hi=1, out='out: near 0 on the right, curving steeply down to the left',
  hint=' (start -40 still gives finite numbers)',
  see=['t.special.ndtr', 't.special.ndtri', 't.special.erfc', 't.special.erfcx'],
  related="""t.special.ndtr is the probability itself. t.special.ndtri goes from
probability back to standard deviations."""),

 dict(name='t.special.ndtr', short='ndtr', title='the normal distribution function',
  what="""Computes the standard normal cumulative distribution function for every
element of a tensor: the probability that a normal value with mean 0 and
standard deviation 1 is below x.""",
  use="""It turns a z-score (a distance from the mean in standard deviations) into
a percentile.""",
  curve="""An S from 0 to 1, passing 0.5 at 0: ndtr(1) = 0.841, ndtr(1.96) = 0.975,
ndtr(3) = 0.99865. It is erf rescaled: (1 + erf(x / 1.414)) / 2.""",
  edges="""Defined for every real number. Below about -14 it underflows to 0 in
32-bit floats; t.special.logndtr keeps going.""",
  a=-4, b=4, lo=-0.05, hi=1.05, out='out: an S from 0 to 1 - the bell curve, added up',
  hint='',
  see=['t.special.ndtri', 't.special.logndtr', 't.special.erf', 't.dist.normal',
       't.special.expit'],
  related="""t.special.ndtri undoes it. t.special.logndtr is its log, for the far
tail. t.special.erf is the same curve on a different scale."""),

 dict(name='t.special.ndtri', short='ndtri', title='the inverse normal distribution function',
  what="""Computes the inverse of the standard normal cumulative distribution
function for every element of a tensor: given a probability p, the value
that a normal variable falls below with probability p.""",
  use="""It turns a percentile into a z-score - ndtri(0.975) = 1.96, the familiar
95% two-sided cut-off - and turns evenly spread random numbers into normally
distributed ones.""",
  curve="""Defined only between 0 and 1: 0 at 0.5, nearly straight through the
middle and shooting off near both ends - -3.09 at 0.001, 3.09 at 0.999.""",
  edges="""0 and 1 give minus and plus infinity; outside 0 to 1 gives NaN (not a
number). The demo runs from 0.001 to 0.999.""",
  a=0.001, b=0.999, lo=-3.5, hi=3.5, out='out: a sideways S, 0 at 0.5',
  hint=' (inside 0 to 1)',
  see=['t.special.ndtr', 't.special.erfinv', 't.special.logndtr', 't.dist.normal'],
  related="""t.special.ndtr is the function it undoes. t.special.erfinv is the same
shape on the error function's scale."""),

 dict(name='t.special.scaled_modified_bessel_k0', short='e to the x times K0(x)',
  title='exponentially scaled K0',
  what="""Applies the modified Bessel function of the second kind of order 0, K0,
multiplied by e to the x, to every element of a tensor.""",
  use="""K0 describes how something spreading out from a long thin source falls
off with distance - heat from a wire, or a field that is screened as it
spreads - and this scaled form keeps it from underflowing far from the
source.""",
  curve="""Defined for positive x. Infinite at 0, falling steeply at first and then
very slowly, approaching the square root of (pi / 2x): 2.68 at 0.1, 0.39 at
10. K0 itself dies away like e to the minus x; multiplying by e to the x
removes that so the numbers stay ordinary.""",
  edges="""0 gives infinity, negative inputs give NaN (not a number). Keep start
above 0.""",
  a=0.1, b=10, lo=0, hi=3, out='out: high near 0, a long slow fall',
  hint=' (keep start above 0)',
  see=['t.special.i0e', 't.special.i0', 't.special.erfcx'],
  related="""t.special.i0e is the matching scaled version of I0, the growing
partner of K0."""),

 dict(name='t.special.sinc', short='sinc', title='the normalised sinc function',
  what="""Applies the normalised sinc function, sin(pi x) / (pi x), to every
element of a tensor.""",
  use="""It is the impulse response of an ideal low-pass filter, which makes it
the kernel for resampling and band-limited interpolation (usually with a
window over it).""",
  curve="""1 at x = 0, symmetric, with ripples that shrink like 1/x on either side.
It crosses zero at every nonzero whole number - 1, 2, 3 and -1, -2, -3 -
because this is the NORMALISED sinc, with pi inside.""",
  edges="""Defined for every real number; at exactly 0 it gives 1, not the 0/0 a
hand-written version would.""",
  a=-8, b=8, lo=-0.3, hi=1.05, out='out: a peak of 1 at 0, ripples crossing 0 at each whole number',
  hint='',
  see=['t.special.spherical_bessel_j0', 't.special.bessel_j0', 't.window.kaiser',
       't.window.hann'],
  related="""t.special.spherical_bessel_j0 is the UNnormalised sinc, sin(x) / x:
the same curve stretched by pi. A window such as t.window.kaiser or
t.window.hann is what tames sinc's long ripples in practice."""),

 dict(name='t.special.spherical_bessel_j0', short='j0(x)', title='spherical Bessel function of order 0',
  what="""Applies the spherical Bessel function of the first kind of order 0,
sin(x) / x, to every element of a tensor.""",
  use="""It describes waves spreading from a point in three dimensions - sound
standing in a spherical room, scattering from a small ball - and it is the
sinc function of signal processing without the pi.""",
  curve="""1 at x = 0, symmetric, with ripples that shrink like 1/x. It crosses zero
at every multiple of pi (3.14, 6.28, 9.42 ...). It equals
t.special.sinc(x / pi).""",
  edges="""Defined for every real number; at exactly 0 it gives 1.""",
  a=0, b=20, lo=-0.3, hi=1.05, out='out: 1 at 0, ripples crossing 0 at each multiple of pi',
  hint='',
  see=['t.special.sinc', 't.special.bessel_j0', 't.special.bessel_j1'],
  related="""t.special.sinc is the same curve with zeros at whole numbers instead.
t.special.bessel_j0 is the cylindrical (2-D) version, whose swings shrink
more slowly."""),
]


def single_page(s):
    name, a, b, lo, hi = s['name'], s['a'], s['b'], s['lo'], s['hi']
    read = READ_ONE.format(a=fmt(a), b=fmt(b), lo=fmt(lo), hi=fmt(hi),
                           short=s['short'], hint=s['hint'])
    body = '\n'.join([
        s['what'], '', s['use'], '',
        'THE CURVE:', s['curve'], '',
        'SYNTAX:', name, '',
        'INPUTS and PARAMETERS:', '', INPUT_ONE, '',
        'OUTPUTS:', '', 'tensor out:',
        'The result, element by element, in the same shape. On the same device\n'
        'as the input; float32 in gives float32 out.', '',
        'WHERE IT IS DEFINED:', s['edges'], '',
        read, '',
        'RELATED:', s['related']])
    short_node = name.split('.')[-1]
    demo = [
        {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
         'props': METRO(100)},
        {'key': 'c0', 'comment': True, 'text': 'redraws ten times a second',
         'pos': (180, 70)},
        linspace(a, b),
        {'key': 'c1', 'comment': True,
         'text': '200 values from start to stop - drag them', 'pos': (240, 160)},
        {'key': 'p1', 'init': 'plot', 'pos': (30, 340), 'w': 308, 'h': 178,
         'props': ARRAY_PLOT(min(a, b), max(a, b))},
        {'key': 'c2', 'comment': True,
         'text': 'in: a ramp from %s to %s' % (fmt(a), fmt(b)), 'pos': (360, 345)},
        {'key': 'fn', 'init': name, 'pos': (30, 540), 'w': 240, 'h': 70},
        {'key': 'c3', 'comment': True, 'text': short_node + ' of each value',
         'pos': (290, 550)},
        {'key': 'p2', 'init': 'plot', 'pos': (30, 630), 'w': 308, 'h': 178,
         'props': ARRAY_PLOT(lo, hi)},
        {'key': 'c4', 'comment': True,
         'text': '%s\n(plot from %s to %s)' % (s['out'], fmt(lo), fmt(hi)),
         'pos': (360, 635)},
    ] + see_also(840, s['see'])
    links = [('met', '', 'ls', '###input'),
             ('ls', 'linspace out', 'p1', 'y'),
             ('ls', 'linspace out', 'fn', 'tensor in'),
             ('fn', 'tensor out', 'p2', 'y')]
    print(build(name, '%s - %s' % (name, s['title']), body, demo, links,
                demo_width=640, text_width=760, text_height=text_h(body)))


for s in SINGLE:
    single_page(s)


# ---------------------------------------------------------------------------
# functions with a parameter widget on the node (polygamma n, multigammaln p,
# logits eps): the same demo, with the parameter dragged on the node itself.

PARAM = [
 dict(name='t.special.polygamma', title='derivatives of digamma',
  a=1, b=6, lo=-3, hi=3, props={'n': 0},
  out='out: n = 0 is digamma; try n = 1 and 2',
  ctext='drag n: which derivative (0 = digamma)',
  body="""Computes the n-th derivative of the digamma function for every element
of a tensor.

It is used when fitting gamma, beta and Dirichlet distributions (n = 1, the
trigamma function, gives how certain such a fit is) and for sums of 1/(x + k)
to a power, since polygamma(n, x) is a Hurwitz zeta value in disguise.

THE CURVE:
n = 0 is digamma itself: a log-like rise, -0.577 at 1. n = 1 (trigamma) is
positive and falling, pi squared over 6 (1.645) at 1. n = 2 is negative and
rising towards 0 from -2.40. Each further n alternates in sign and grows
roughly like n factorial near x = 1 (n = 3 gives 6.49, n = 4 gives -24.9),
so for n of 3 or more the curve leaves the demo's plot range near the left.

SYNTAX:
t.special.polygamma

There is no argument: n starts at 0 and is set on the node.

INPUTS and PARAMETERS:

tensor in:
A tensor - or a number or a list, which is turned into one. Receiving it
triggers the node. Each element is worked on separately.

n (int, also an inlet):
Which derivative of digamma: 0 is digamma, 1 trigamma, 2 tetragamma ...
Default 0, minimum 0. Changing it does not send by itself; the next tensor
uses it.

OUTPUTS:

tensor out:
The result, element by element, in the same shape as the input.

WHERE IT IS DEFINED:
Positive x is the useful range. At 0 and the negative whole numbers every
order is infinite (n = 0 gives minus infinity; higher n plus or minus).

READING THE DEMO:
The metro re-clicks t.linspace ten times a second; it makes 200 values from
1 to 6, drawn in the upper plot. The lower plot is polygamma of order n of
each, from -3 (bottom) to 3 (top). Drag n on the node to step through the
derivatives; drag start and stop on t.linspace to look elsewhere (keep
start above 0).

RELATED:
t.special.digamma is order 0 on its own. t.special.gammaln is what digamma
is the derivative of. t.special.zeta is the Hurwitz zeta function that the
higher orders are built from.""",
  see=['t.special.digamma', 't.special.gammaln', 't.special.zeta',
       't.special.multigammaln']),

 dict(name='t.special.multigammaln', title='log of the multivariate gamma',
  a=2.6, b=10, lo=0, hi=60, props={'p': 1},
  out='out: grows with p; p = 1 is plain log gamma',
  ctext='drag p: the dimension (1 or more)',
  body="""Computes the log of the multivariate gamma function of dimension p for
every element of a tensor.

It is the normalising constant of the Wishart distribution - the
distribution of random covariance matrices - so it appears when you work
with uncertainty about a whole matrix rather than a single number.

THE CURVE:
It is a sum of p ordinary log gammas, at x, x - 0.5, x - 1 and so on down
to x - (p - 1)/2, plus a constant p(p - 1)/4 times log pi. So p = 1 is
exactly t.special.gammaln, and each increase in p adds another rising log
gamma curve: at x = 10 it is 12.8 for p = 1, 25.1 for p = 2, 36.8 for
p = 3, 58.9 for p = 5.

SYNTAX:
t.special.multigammaln

There is no argument: p starts at 1 and is set on the node.

INPUTS and PARAMETERS:

tensor in:
A tensor - or a number or a list, which is turned into one. Receiving it
triggers the node. Each element is worked on separately.

p (int, also an inlet):
The dimension. Default 1, minimum 1. Changing it does not send by itself;
the next tensor uses it.

OUTPUTS:

tensor out:
The result, element by element, in the same shape as the input.

WHERE IT IS DEFINED:
Every element must be greater than (p - 1) / 2. Values at or below that
come out as infinity. The demo starts at 2.6 so that p up to 5 is safe.

READING THE DEMO:
The metro re-clicks t.linspace ten times a second; it makes 200 values from
2.6 to 10, drawn in the upper plot. The lower plot is multigammaln of each
for dimension p, from 0 (bottom) to 60 (top). Drag p on the node and the
curve climbs.

RELATED:
t.special.gammaln is the one-dimensional case. t.special.digamma and
t.special.polygamma are its derivatives.""",
  see=['t.special.gammaln', 't.special.digamma', 't.special.polygamma']),

 dict(name='t.special.logits', title='probability back to score',
  a=0, b=1, lo=-15, hi=15, props={'eps': 1e-6},
  out='out: 0 at 0.5, ends held at -13.8 and 13.8 by eps',
  ctext='drag eps: try 0.001 and the ends come in to 6.9',
  body="""The inverse of the sigmoid: turns probabilities between 0 and 1 back into
unbounded scores, the log of p / (1 - p), for every element of a tensor.

Use it whenever you want to add, average or smooth probabilities: do it on
the logits, where 0.99 and 0.999 are properly far apart, then turn the result
back with t.special.expit.

THE CURVE:
0.5 becomes 0, values above 0.5 become positive and values below negative.
The middle is nearly straight (0.25 gives -1.10, 0.75 gives 1.10) and the
ends shoot off towards plus and minus infinity - held back by eps.

SYNTAX:
t.special.logits

INPUTS and PARAMETERS:

tensor in:
The probabilities, 0 to 1 - a tensor, or a number or list, which is turned
into one. Receiving it triggers the node. Values outside 0..1 are clamped
by eps too, so they come out at the ends rather than as errors.

eps (float, also an inlet):
The input is clamped to between eps and 1 - eps before the log is taken,
so 0 and 1 give large finite numbers instead of infinities. Default
0.000001, which holds the ends at -13.8 and 13.8; 0.001 gives -6.9 and 6.9.
At 0 there is no protection and 0 and 1 give minus and plus infinity.
Do not set it below about 0.0000001: in 32-bit floats 1 minus anything
smaller rounds back to exactly 1, so the top end is unprotected again (1
gives infinity) while the bottom end still looks fine. Changing eps does
not send by itself; the next tensor uses it.

OUTPUTS:

tensor out:
The scores, element by element, in the same shape as the input.

READING THE DEMO:
The metro re-clicks t.linspace ten times a second; it makes 200
probabilities from 0 to 1, drawn in the upper plot. The lower plot is the
logit of each, from -15 (bottom) to 15 (top): a slow climb through 0 in the
middle and the two ends flicking out to the eps limit. Drag eps on the node
to move those limits.

RELATED:
t.special.expit (and t.nn.sigmoid) go the other way, score to probability.
t.special.ndtri is the same idea for the normal distribution's S curve.""",
  see=['t.special.expit', 't.nn.sigmoid', 't.special.ndtri', 't.special.entr',
       't.special.xlogy']),
]


def param_page(s):
    name, a, b, lo, hi = s['name'], s['a'], s['b'], s['lo'], s['hi']
    body = s['body']
    demo = [
        {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
         'props': METRO(100)},
        {'key': 'c0', 'comment': True, 'text': 'redraws ten times a second',
         'pos': (180, 70)},
        linspace(a, b),
        {'key': 'c1', 'comment': True,
         'text': '200 values from start to stop - drag them', 'pos': (240, 160)},
        {'key': 'p1', 'init': 'plot', 'pos': (30, 340), 'w': 308, 'h': 178,
         'props': ARRAY_PLOT(min(a, b), max(a, b))},
        {'key': 'c2', 'comment': True,
         'text': 'in: a ramp from %s to %s' % (fmt(a), fmt(b)), 'pos': (360, 345)},
        {'key': 'fn', 'init': name, 'pos': (30, 540), 'w': 240, 'h': 100,
         'props': s['props']},
        {'key': 'c3', 'comment': True, 'text': s['ctext'], 'pos': (290, 550)},
        {'key': 'p2', 'init': 'plot', 'pos': (30, 660), 'w': 308, 'h': 178,
         'props': ARRAY_PLOT(lo, hi)},
        {'key': 'c4', 'comment': True,
         'text': '%s\n(plot from %s to %s)' % (s['out'], fmt(lo), fmt(hi)),
         'pos': (360, 665)},
    ] + see_also(870, s['see'])
    links = [('met', '', 'ls', '###input'),
             ('ls', 'linspace out', 'p1', 'y'),
             ('ls', 'linspace out', 'fn', 'tensor in'),
             ('fn', 'tensor out', 'p2', 'y')]
    print(build(name, '%s - %s' % (name, s['title']), body, demo, links,
                demo_width=640, text_width=760, text_height=text_h(body)))


for s in PARAM:
    param_page(s)


# ---------------------------------------------------------------------------
# gammainc, gammaincc, xlog1py: the FIRST input (hot) is the parameter, the
# second (cold) carries the x range. A load_bang sends the range and then a
# starting parameter; dragging the float re-draws.

GAMMA_BODY = """Computes the regularised {which} incomplete gamma function, element by
element, of a shape a (first input) and a value x (second input).

{use}

THE CURVE:
{curve}

SYNTAX:
{name}

INPUTS and PARAMETERS:

tensor 1 in:
a, the shape - a tensor, or a single number applied to every x. Receiving
it triggers the node. a should be positive: a = 0 gives a flat line (NaN
at x = 0), and negative a gives NaN (not a number) everywhere.

tensor 2 in:
x, where to evaluate - a tensor, or a single number. It is only stored when
it arrives, so send it BEFORE a. It is converted to a's dtype and device.
The two inputs are matched element by element, or broadcast when one is a
single value. x must be 0 or more; negative x gives NaN.

OUTPUTS:

tensor out:
The result, between 0 and 1, in the broadcast shape of the two inputs.

READING THE DEMO:
When the page opens, load_bang sends a bang to t.linspace first (200 values
of x from 0 to 10 into the cold second input, drawn in the upper plot) and
then 1.0 to the float, which sends it to the first input and draws the
curve. Drag a and the curve redraws: small a rises early and fast, large a
later and more gently. To change the x range, drag start and stop on
t.linspace, click its button, then nudge a - x alone does not trigger.

RELATED:
{related}"""

TWO_HOT = [
 dict(name='t.special.gammainc', which='lower', title='the gamma distribution, added up',
  use="""It is the cumulative distribution of a gamma-distributed quantity with
shape a: the chance that a waiting time, a duration or an accumulated amount
comes out below x. The chi-squared test uses it too: the chance a
chi-squared value with k degrees of freedom is below c is gammainc(k/2, c/2)
- gammainc(0.5, 1.92) = 0.95, the familiar 3.84 cut-off for one degree.""",
  curve="""For each a, an S-shaped rise from 0 at x = 0 to 1 as x grows. With a = 1
it is 1 - e to the minus x (0.632 at x = 1). Larger a moves the rise
later and makes it gentler: the half-way point sits a little below x = a.""",
  out='out: rises from 0 to 1; larger a rises later',
  related="""t.special.gammaincc is the complement, 1 minus this - the upper tail.
t.special.gammaln is the log of the complete gamma function.
t.dist.gamma and t.dist.chi2 sample from the distributions this adds up.""",
  see=['t.special.gammaincc', 't.special.gammaln', 't.dist.gamma', 't.dist.chi2',
       't.special.ndtr']),

 dict(name='t.special.gammaincc', which='upper', title='the gamma distribution\'s upper tail',
  use="""It is the upper tail of a gamma distribution with shape a: the chance of a
value of x or more - how surprising a reading this large is. For whole
numbers it is also the Poisson distribution added up: with an average rate
m, the chance of fewer than k events is gammaincc(k, m).""",
  curve="""For each a, a falling S from 1 at x = 0 to 0 as x grows - the mirror of
gammainc, and the two always sum to 1. With a = 1 it is e to the minus x
(0.368 at x = 1). Larger a holds it near 1 for longer. Computed directly,
so far-out tails stay accurate instead of being 1 minus nearly 1.""",
  out='out: falls from 1 to 0; larger a falls later',
  related="""t.special.gammainc is the lower part; the two sum to 1.
t.special.gammaln is the log of the complete gamma function.
t.dist.gamma and t.dist.poisson sample from the distributions involved.""",
  see=['t.special.gammainc', 't.special.gammaln', 't.dist.gamma', 't.dist.poisson']),
]

XLOG1PY_BODY = """Computes x times the natural log of (1 + y), element by element - and
gives 0 whenever x is 0, even where the log is infinite.

It is the term that appears in log-likelihoods and entropies of the form
x log(1 + y), kept accurate for tiny y (as t.special.log1p is) and free of
the NaN that 0 times minus infinity would give.

THE CURVE:
For a fixed x it is just log(1 + y) scaled by x: 0 at y = 0, rising ever
more slowly to the right, plunging towards minus infinity as y approaches
-1. Negative x flips it upside down. At x = 0 it is flat 0 everywhere -
including y = -1, where a hand-written x * log(1 + y) gives NaN.

SYNTAX:
t.special.xlog1py

INPUTS and PARAMETERS:

tensor 1 or number in:
x, the multiplier - a tensor, or a single number applied to every y.
Receiving it triggers the node.

tensor 2 or number in:
y - a tensor, or a single number. It is only stored when it arrives, so
send it BEFORE x. It is converted to x's dtype and device. The two are
matched element by element, or broadcast when one is a single value.

OUTPUTS:

tensor out:
The result, in the broadcast shape of the two inputs.

WHERE IT IS DEFINED:
y below -1 gives NaN (not a number) unless x is 0. y = -1 gives minus
infinity for positive x and 0 for x = 0.

READING THE DEMO:
When the page opens, load_bang sends a bang to t.linspace first (200 values
of y from -0.9 to 4 into the cold second input, drawn in the upper plot)
and then 1.0 to the float, which sends x and draws log(1 + y). Drag x: the
curve scales, flattens to 0 at x = 0 and turns over for negative x. To
change the y range, drag start and stop on t.linspace, click its button,
then nudge x - y alone does not trigger.

RELATED:
t.special.xlogy is the same with log(y). t.special.log1p is log(1 + y)
without the multiplier; t.special.expm1 is its inverse."""


def two_hot_page(name, title, body, a, b, lo, hi, a0, out, see, fin, fout):
    """t 1.0 bang: the bang (right, first) sends the range to the cold
    inlet, then 1.0 goes to the float, which sends the parameter to the hot
    inlet and draws."""
    demo = [
        {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
        {'key': 'tt', 'init': 't %s bang' % fmt(a0), 'pos': (140, 62), 'w': 110, 'h': 46},
        {'key': 'c0', 'comment': True, 'text': 'on open: the range, then a first value',
         'pos': (270, 70)},
        dict(linspace(a, b), pos=(250, 150)),
        {'key': 'c1', 'comment': True, 'text': 'the range: drag, then click its button',
         'pos': (460, 160)},
        {'key': 'fl', 'init': 'float', 'pos': (30, 150), 'w': 127, 'h': 42,
         'props': dict(FLT, **{'': float(a0)})},
        {'key': 'cf', 'comment': True, 'text': 'drag me', 'pos': (30, 200)},
        {'key': 'p1', 'init': 'plot', 'pos': (250, 340), 'w': 308, 'h': 178,
         'props': ARRAY_PLOT(min(a, b), max(a, b))},
        {'key': 'c2', 'comment': True,
         'text': 'in (second input): %s to %s' % (fmt(a), fmt(b)), 'pos': (580, 345)},
        {'key': 'fn', 'init': name, 'pos': (30, 540), 'w': 280, 'h': 90},
        {'key': 'p2', 'init': 'plot', 'pos': (30, 650), 'w': 308, 'h': 178,
         'props': ARRAY_PLOT(lo, hi)},
        {'key': 'c4', 'comment': True,
         'text': '%s\n(plot from %s to %s)' % (out, fmt(lo), fmt(hi)),
         'pos': (360, 655)},
    ] + see_also(860, see)
    links = [('lb', 'out', 'tt', ''),
             ('tt', 'bang', 'ls', '###input'),
             ('tt', fmt(a0), 'fl', ''),
             ('ls', 'linspace out', 'p1', 'y'),
             ('ls', 'linspace out', 'fn', fin),
             ('fl', 'float out', 'fn', fout),
             ('fn', 'tensor out', 'p2', 'y')]
    print(build(name, '%s - %s' % (name, title), body, demo, links,
                demo_width=760, text_width=760, text_height=text_h(body)))


for s in TWO_HOT:
    body = GAMMA_BODY.format(which=s['which'], use=s['use'], curve=s['curve'],
                             name=s['name'], related=s['related'])
    two_hot_page(s['name'], s['title'], body, 0, 10, -0.05, 1.05, 1.0, s['out'],
                 s['see'], 'tensor 2 in', 'tensor 1 in')

two_hot_page('t.special.xlog1py', 'x log (1 + y) without the NaN', XLOG1PY_BODY,
             -0.9, 4, -5, 5, 1.0, 'out: x times log(1 + y); 0 everywhere when x is 0',
             ['t.special.xlogy', 't.special.log1p', 't.special.expm1'],
             'tensor 2 or number in', 'tensor 1 or number in')


# ---------------------------------------------------------------------------
# t.special.zeta: x (hot) from the linspace, q (cold) from a float.

body = """Computes the Hurwitz zeta function, element by element, of a power x
(first input) and an offset q (second input).

It is the normalising constant of discrete power laws - Zipf-like
distributions, where the k-th most common thing turns up in proportion to 1
over k to the x - and with q = 1 it is the Riemann zeta function.

WHAT IT COMPUTES:
zeta(x, q) is the sum, over k = 0, 1, 2 ..., of 1 / (k + q) to the power x.
With q = 1 that is 1 + 1/2 to the x + 1/3 to the x + ...: zeta(2, 1) is pi
squared over 6 (1.645), zeta(3, 1) = 1.202.

THE CURVE:
Over x, for a fixed q: very large just above x = 1, falling quickly and
levelling off at 1 / q to the x - the first term - since the others vanish
as x grows. So with q = 1 it settles towards 1, with q = 2 towards 0, and
with q below 1 it climbs instead (q = 0.5 gives 2 to the x).

SYNTAX:
t.special.zeta

INPUTS and PARAMETERS:

tensor 1 or number in:
x, the power - a tensor, or a single number. Receiving it triggers the node.

tensor 2 or number in:
q, the offset - a tensor, or a single number applied everywhere. It is only
stored when it arrives, so send it BEFORE x. It is converted to x's dtype
and device. The two are matched element by element, or broadcast when one
is a single value.

OUTPUTS:

tensor out:
The result, in the broadcast shape of the two inputs.

WHERE IT IS DEFINED:
The sum only settles when x is greater than 1. x = 1 gives infinity, and x
below 1 gives NaN (not a number). q should be positive: q = 0 and negative
whole numbers give infinity.

READING THE DEMO:
The metro re-clicks t.linspace ten times a second; it makes 200 values of x
from 1.5 to 8, drawn in the upper plot, and sends them to the first input.
The float holds q in the second input; load_bang sets it to 1 when the
page opens (the Riemann zeta). The
lower plot is zeta of each x, from 0 (bottom) to 3 (top): 2.61 at x = 1.5,
settling towards 1. Drag q to 2 and it settles towards 0; below 1 it climbs
off the top. Drag start and stop to move the range (keep start above 1).

RELATED:
t.special.polygamma of order n is built from zeta(n + 1, x).
t.special.digamma and t.special.gammaln are its gamma-function relatives."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': METRO(100)},
    {'key': 'c0', 'comment': True, 'text': 'redraws ten times a second', 'pos': (180, 70)},
    linspace(1.5, 8),
    {'key': 'c1', 'comment': True, 'text': 'x: 200 values from start to stop',
     'pos': (240, 160)},
    {'key': 'lb', 'init': 'load_bang', 'pos': (360, 62), 'w': 88, 'h': 46},
    {'key': 'tq', 'init': 't 1', 'pos': (470, 62), 'w': 22, 'h': 46},
    {'key': 'cl', 'comment': True, 'text': 'q = 1 on open', 'pos': (520, 70)},
    {'key': 'fq', 'init': 'float', 'pos': (360, 470), 'w': 127, 'h': 42,
     'props': dict(FLT, **{'': 1.0})},
    {'key': 'cq', 'comment': True, 'text': 'q - drag me', 'pos': (500, 475)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 340), 'w': 308, 'h': 178,
     'props': ARRAY_PLOT(1.5, 8)},
    {'key': 'c2', 'comment': True, 'text': 'in: x from 1.5 to 8', 'pos': (360, 345)},
    {'key': 'fn', 'init': 't.special.zeta', 'pos': (30, 540), 'w': 280, 'h': 90},
    {'key': 'p2', 'init': 'plot', 'pos': (30, 650), 'w': 308, 'h': 178,
     'props': ARRAY_PLOT(0, 3)},
    {'key': 'c4', 'comment': True,
     'text': 'out: zeta(x, q) - with q = 1, 2.61 falling to 1\n(plot from 0 to 3)',
     'pos': (360, 655)},
] + see_also(860, ['t.special.polygamma', 't.special.digamma', 't.special.gammaln'])
links = [('met', '', 'ls', '###input'),
         ('lb', 'out', 'tq', ''), ('tq', '1', 'fq', ''),
         ('ls', 'linspace out', 'p1', 'y'),
         ('fq', 'float out', 'fn', 'tensor 2 or number in'),
         ('ls', 'linspace out', 'fn', 'tensor 1 or number in'),
         ('fn', 'tensor out', 'p2', 'y')]
print(build('t.special.zeta', 't.special.zeta - the Hurwitz zeta function', body,
            demo, links, demo_width=640, text_width=760, text_height=text_h(body)))


# ---------------------------------------------------------------------------
# t.special.xlogy: p log p over 0..1 - the same tensor into both inputs.

body = """Computes x times the natural log of y, element by element - and gives 0
whenever x is 0, even where log y is minus infinity.

It is the safe way to write the terms of an entropy or cross-entropy (a
probability times the log of a probability), where zero probabilities are
normal and a hand-written x * log y would turn them into NaN that then
spoils every sum after it.

WHY IT IS NOT JUST A MULTIPLY:
log(0) is minus infinity, and 0 times minus infinity is undefined, so x *
log(y) gives NaN (not a number) at x = 0, y = 0. The limit is 0, and 0 is
the answer you want. This node returns it: whenever x is 0 the result is 0,
whatever y is - unless y is itself NaN.

SYNTAX:
t.special.xlogy

INPUTS and PARAMETERS:

tensor 1 or number in:
x, the multiplier - a tensor, or a single number applied to every y.
Receiving it triggers the node.

tensor 2 or number in:
y, the value whose log is taken - a tensor, or a single number. It is only
stored when it arrives, so send it BEFORE x. It is converted to x's dtype
and device. The two are matched element by element, or broadcast when one
is a single value.

OUTPUTS:

tensor out:
The result, in the broadcast shape of the two inputs.

WHERE IT IS DEFINED:
With x not 0: y = 0 gives minus infinity (for positive x) and negative y
gives NaN, as log does.

READING THE DEMO:
The metro re-clicks t.linspace ten times a second; it makes 200 values of p
from 0 to 1. 't array array' sends them to the second input and then the
first (right outlet first), so the node computes p log p. The lower plot is
that, from -0.4 (bottom) to 0.05 (top): a dip to -0.368 at p = 0.368
(1/e), and exactly 0 at BOTH ends - at p = 1 because log 1 is 0, and at
p = 0 because of the protection; written by hand that first point would be
NaN. Minus the sum of these over a distribution is its entropy. Drag start
and stop on t.linspace to look elsewhere (negative values give NaN, which
the plot leaves out).

RELATED:
t.special.xlog1py is the same with log(1 + y). t.special.entr computes
minus p log p in one step. t.special.logits is the other place a log near
0 and 1 needs care."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': METRO(100)},
    {'key': 'c0', 'comment': True, 'text': 'redraws ten times a second', 'pos': (180, 70)},
    linspace(0, 1),
    {'key': 'c1', 'comment': True, 'text': 'p: 200 values from 0 to 1', 'pos': (240, 160)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 340), 'w': 308, 'h': 178,
     'props': ARRAY_PLOT(0, 1)},
    {'key': 'c2', 'comment': True, 'text': 'in: p from 0 to 1', 'pos': (360, 345)},
    {'key': 'tr', 'init': 't array array', 'pos': (30, 540), 'w': 140, 'h': 46},
    {'key': 'ct', 'comment': True, 'text': 'p to y first (right), then to x',
     'pos': (190, 545)},
    {'key': 'fn', 'init': 't.special.xlogy', 'pos': (30, 610), 'w': 280, 'h': 90},
    {'key': 'p2', 'init': 'plot', 'pos': (30, 720), 'w': 308, 'h': 178,
     'props': ARRAY_PLOT(-0.4, 0.05)},
    {'key': 'c4', 'comment': True,
     'text': 'out: p log p - 0 at both ends, -0.368 at its lowest\n(plot from -0.4 to 0.05)',
     'pos': (360, 725)},
] + see_also(930, ['t.special.xlog1py', 't.special.entr', 't.special.logits',
                   't.special.log1p'])
links = [('met', '', 'ls', '###input'),
         ('ls', 'linspace out', 'p1', 'y'),
         ('ls', 'linspace out', 'tr', ''),
         ('tr', 'array', 'fn', 'tensor 2 or number in', 1),
         ('tr', 'array', 'fn', 'tensor 1 or number in', 0),
         ('fn', 'tensor out', 'p2', 'y')]
print(build('t.special.xlogy', 't.special.xlogy - x log y without the NaN', body,
            demo, links, demo_width=640, text_width=760, text_height=text_h(body)))
