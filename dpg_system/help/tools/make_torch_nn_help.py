"""t.nn activation curves and the softmax family: one page per node.

The old make_torch_activation_help.py wrote three family pages (t.nn.relu for
the fixed curves, t.nn.elu for the adjustable ones, t.nn.softmax for the
softmax family). Each node now has its own page. A curve page draws a ramp
(t.linspace) and the curve applied to it in ONE plot, so the bend reads
against the straight line of the input; adjustable curves have a slider on
their parameter and a metro re-sending the ramp, so the curve moves as you
drag. Softmax pages show a vector of scores before and after, with a scale
slider standing in for temperature.

t.special.softmax and t.special.log_softmax are the same operations as
t.nn.softmax and t.nn.log_softmax (torch.special.softmax IS torch.softmax,
same default dim 0), so they are documented on those pages.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import PLOT, INT, FLT


def HM(width, height, count, miny=0.0, maxy=1.0, fmt='%.2f', color='viridis'):
    """Property block for a heat_map that draws a whole array as one picture."""
    return {'color': color, 'width': width, 'height': height, 'sample count': count,
            'min y': miny, 'max y': maxy, 'update_mode': 'heat_map',
            'number format': fmt}


LIST = {'text in': '', 'font size': '24'}


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


def MULTI_PLOT(miny=-1.0, maxy=1.0, count=200, width=300, height=128):
    """A plot that draws each row of a 2-D array as its own line."""
    return dict(PLOT(miny, maxy, count),
                **{'width': width, 'height': height,
                   'update style': 'input is multi-channel sample'})


def ARRAY_PLOT(miny, maxy, count, width=300, height=128):
    """A plot that draws one whole 1-D array as a curve, first value on the left."""
    return dict(PLOT(miny, maxy, count),
                **{'width': width, 'height': height,
                   'update style': 'buffer holds one sample of input'})


METRO = lambda ms: {'on': True, 'period': float(ms), 'units': 'milliseconds'}


def SLIDER(v, lo, hi, width=200):
    return {'': float(v), 'min': float(lo), 'max': float(hi), 'format': '%.3f',
            'width': width}


def text_h(body):
    """Height of the prose block: 15px per line at text size 30, plus a margin."""
    return (body.count('\n') + 1) * 15 + 30


def fmt_num(v):
    return '%g' % v


# ------------------------------------------------------------- shared prose
TENSOR_IN = """tensor in:
The tensor to bend. Receiving it triggers the node. A list, number or numpy
array is turned into a float32 tensor on the cpu first; a tensor arriving as
a tensor keeps its own dtype and device. The curve is applied to every
element on its own, so the output has the same shape as the input."""

INT_REFUSED = """
An INTEGER tensor (t.arange with whole numbers makes one) is refused: the
console reports "not implemented for 'Long'" and nothing is sent. Put a
t.to float32 in front of it."""

INT_OK = """
An integer tensor is accepted."""

OUTPUT = """OUTPUTS:

output:
The bent tensor, the same shape as the input."""


def reading(x0, x1, y0, y1):
    return f"""READING THE DEMO:
t.linspace makes 128 values running evenly from {fmt_num(x0)} (left edge of the
plot) to {fmt_num(x1)} (right edge); the middle of the plot is 0. The plot
draws two lines over each other: the straight diagonal is the input itself,
and the other is the node's output for the same values. Where the two lines
coincide the node passes the value unchanged; where they part, that is the
bend. The plot's height runs from {fmt_num(y0)} to {fmt_num(y1)}; the input
line runs off the top and bottom where it leaves that range."""


# ----------------------------------------------------- the curve demo itself
def curve_demo(init, x0, x1, y0, y1, see, params=(), node_h=70, note=None, node_props=None):
    """metro -> t.linspace ramp -> t array array: right to t.stack (the
    reference), left through the node to t.stack, then one plot of both rows.

    params: [(inlet name, default, slider min, slider max)] -- a slider wired
    to each parameter inlet, so the curve moves while it is dragged."""
    demo = [
        {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
         'props': METRO(100)},
        {'key': 'cmet', 'comment': True,
         'text': 're-sends the ramp ten times a second', 'pos': (180, 72)},
        {'key': 'ls', 'init': f't.linspace {fmt_num(x0)} {fmt_num(x1)} 128',
         'pos': (30, 150), 'w': 190, 'h': 150,
         'props': {'start': float(x0), 'stop': float(x1), 'steps': 128}},
        {'key': 'cls', 'comment': True,
         'text': f'the input: 128 values from {fmt_num(x0)} to {fmt_num(x1)}',
         'pos': (240, 165)},
        {'key': 'tr', 'init': 't array array', 'pos': (30, 320), 'w': 140, 'h': 46},
        {'key': 'ctr', 'comment': True,
         'text': 'right first: the ramp as it is, then through the node',
         'pos': (190, 325)},
    ]
    links = [('met', '', 'ls', '###input'),
             ('ls', 'linspace out', 'tr', ''),
             ('tr', 'array', 'st', 'tensor 2', 1),
             ('tr', 'array', 'nd', 'tensor in', 0),
             ('nd', 'output', 'st', 'tensor 1'),
             ('st', 'output', 'pl', 'y')]
    y = 390
    for i, (name, dflt, lo, hi) in enumerate(params):
        k = f'sl{i}'
        demo += [
            {'key': k, 'init': 'slider', 'pos': (30 + 250 * i, y), 'w': 220, 'h': 42,
             'props': SLIDER(dflt, lo, hi)},
            {'key': 'c' + k, 'comment': True,
             'text': f'{name}: {fmt_num(lo)} to {fmt_num(hi)}',
             'pos': (30 + 250 * i, y + 48)},
        ]
        links.append((k, '', 'nd', name))
    if params:
        y += 90
    demo += [
        dict({'key': 'nd', 'init': init, 'pos': (30, y), 'w': 220, 'h': node_h},
             **({'props': node_props} if node_props else {})),
        {'key': 'st', 'init': 't.stack', 'pos': (30, y + node_h + 30), 'w': 170, 'h': 100},
        {'key': 'cst', 'comment': True,
         'text': 'output on top of the input: two rows, one line each',
         'pos': (220, y + node_h + 40)},
        {'key': 'pl', 'init': 'plot', 'pos': (30, y + node_h + 150), 'w': 440, 'h': 260,
         'props': MULTI_PLOT(y0, y1, 128, width=420, height=230)},
    ]
    py = y + node_h + 150 + 270
    demo.append({'key': 'cpl', 'comment': True,
                 'text': note or 'the diagonal is the input; the other line is the output',
                 'pos': (30, py)})
    demo += see_also(py + 50 + 15 * (note or '').count('\n'), see)
    return demo, links


def curve_page(name, title, intro, curve, syntax, params_text, int_ok,
               x0, x1, y0, y1, see, related, params=(), node_h=70, note=None,
               extra='', node_props=None, init=None):
    body = intro.strip('\n') + '\n\n' + curve.strip('\n') + '\n\n' + \
        'SYNTAX:\n' + syntax.strip('\n') + '\n\n' + \
        'INPUTS and PARAMETERS:\n\n' + TENSOR_IN + (INT_OK if int_ok else INT_REFUSED) + \
        ('\n\n' + params_text.strip('\n') if params_text else '') + '\n\n' + \
        OUTPUT + '\n\n' + reading(x0, x1, y0, y1) + \
        ('\n\n' + extra.strip('\n') if extra else '') + \
        '\n\nRELATED:\n' + related.strip('\n')
    demo, links = curve_demo(init or name, x0, x1, y0, y1, see, params, node_h, note,
                             node_props)
    print(build(name, title, body, demo, links, demo_width=480, text_width=780,
                text_height=text_h(body)))


# =================================================================== FIXED
curve_page(
    't.nn.relu', 't.nn.relu - zero below zero, unchanged above',
    """Passes every positive value unchanged and replaces every negative value
with zero.

It is the plainest bend there is, and the most used activation in neural
networks: a straight line with one corner, at zero. In a patch it folds
negatives away - half-wave rectifying a signal, keeping only the increases
of a difference, making sure a level never goes below zero while leaving
everything above zero exactly as it was.""",
    """THE CURVE:
Flat along zero for every negative input, then the input itself from zero
up. There is no upper limit: a large input comes out just as large. The
corner at zero is sharp - if that matters, gelu, silu and softplus are
rounded versions of the same shape.""",
    't.nn.relu', '', True, -4, 4, -4, 4,
    ['t.nn.leaky_relu', 't.nn.relu6', 't.nn.elu', 't.nn.gelu', 't.nn.softplus', 't.clamp'],
    """t.nn.leaky_relu keeps a small slope below zero instead of flat zero.
t.nn.relu6 also caps the top at 6. t.nn.elu and t.nn.gelu are smooth
versions of the corner; t.nn.softplus is relu with no corner at all.
t.clamp with a minimum of 0 does the same job and lets you choose the
floor.""")

curve_page(
    't.nn.relu6', 't.nn.relu6 - relu, capped at 6',
    """Replaces negative values with zero and values above 6 with 6, passing
everything in between unchanged.

It is relu with a ceiling: a clamp between 0 and 6. The 6 is fixed - it
comes from networks built for low-precision hardware, where a known
maximum lets the numbers be stored in fewer bits. In a patch it is a quick
way to keep a level inside 0..6; for any other range use t.nn.hardtanh or
t.clamp, which take the limits as parameters.""",
    """THE CURVE:
Flat at 0 below zero, the input itself from 0 to 6, then flat at 6. Two
corners, both sharp. The demo's ramp runs from -8 to 8 so that both flat
stretches show.""",
    't.nn.relu6', '', True, -8, 8, -8, 8,
    ['t.nn.relu', 't.nn.hardtanh', 't.clamp', 't.nn.hardsigmoid', 't.nn.hardswish'],
    """t.nn.relu has no ceiling. t.nn.hardtanh is the same flat-straight-flat
shape with limits you set (default -1 and 1); t.clamp does it for any
tensor. t.nn.hardsigmoid and t.nn.hardswish are built from relu6.""")

curve_page(
    't.nn.sigmoid', 't.nn.sigmoid - an S from 0 to 1',
    """Squeezes every value into the range 0 to 1 along a smooth S-shaped curve:
1 divided by (1 plus e to the minus x).

Reach for it to turn any number - a score, a difference, an unbounded
control - into something between 0 and 1: a probability, a mix amount, a
gate that opens gradually. However large the input, the output never
leaves 0..1, which makes it safe in front of anything that needs that
range.""",
    """THE CURVE:
0 gives exactly 0.5. Positive values rise towards 1 and negative values
fall towards 0, but neither limit is ever reached: 2 gives 0.88, 4 gives
0.98, 6 gives 0.998. The curve is steepest at 0 (slope 0.25) and flattens
out on both sides, so it is most sensitive to changes near zero and nearly
deaf to changes far from it. To make the S sharper or gentler, scale the
input before it (multiply by 4 and the S is four times as steep).""",
    't.nn.sigmoid', '', True, -6, 6, -1.5, 1.5,
    ['t.special.expit', 't.nn.tanh', 't.nn.hardsigmoid', 't.nn.logsigmoid',
     't.nn.softsign', 't.special.logits'],
    """t.special.expit is the same function under its statistics name.
t.nn.tanh is the same S stretched to run from -1 to 1. t.nn.hardsigmoid is
a straight-line imitation. t.nn.logsigmoid is its logarithm, computed
safely. t.special.logits goes the other way, from 0..1 back to a score.""")

curve_page(
    't.nn.tanh', 't.nn.tanh - an S from -1 to 1',
    """Squeezes every value into the range -1 to 1 along a smooth S-shaped
curve, the hyperbolic tangent.

It is the classic soft limiter: small values pass almost unchanged (near
zero the curve follows the input), large values are held just inside -1
or 1, and there is no corner anywhere. Use it to tame a signal that
sometimes runs too large without clipping it flat, or to map an unbounded
control onto a symmetric range.""",
    """THE CURVE:
0 gives 0, and near zero the output is almost the input (0.1 gives 0.0997).
Further out it bends over: 1 gives 0.76, 2 gives 0.96, 4 gives 0.9993.
It is symmetric - minus x gives minus the answer for x. It is the sigmoid
stretched to twice the height and shifted down: tanh of x equals 2 times
sigmoid of 2x, minus 1.""",
    't.nn.tanh', '', True, -4, 4, -2, 2,
    ['t.nn.sigmoid', 't.nn.softsign', 't.nn.hardtanh', 't.nn.tanhshrink', 't.special.erf'],
    """t.nn.sigmoid is the same S from 0 to 1. t.nn.softsign is a gentler S
that approaches its limits more slowly. t.nn.hardtanh is the straight-line
version, flat outside -1..1. t.nn.tanhshrink is what tanh takes away (x
minus tanh x). t.special.erf is another S from -1 to 1, a little
steeper.""")

curve_page(
    't.nn.logsigmoid', 't.nn.logsigmoid - the logarithm of the sigmoid',
    """Takes the natural logarithm of the sigmoid of every value, computed in
one step so that it stays accurate where the sigmoid is tiny.

The sigmoid turns a score into a probability; logsigmoid turns it into a
log probability. Doing it as two nodes (sigmoid then log) fails for very
negative inputs, because the sigmoid rounds to exactly 0 and the log of 0
is minus infinity - logsigmoid of -100 is -100, which the two-step version
cannot give you. Use it when you add up log probabilities, or want a
curve that is flat at the top and straight at the bottom.""",
    """THE CURVE:
Always negative. Large positive inputs give values just below 0 (4 gives
-0.018); 0 gives -0.693 (the log of one half); large negative inputs give
almost the input itself (-4 gives -4.018). So it runs along the diagonal
on the left and flattens onto zero on the right. It is the mirror image of
softplus: logsigmoid of x equals minus softplus of minus x.""",
    't.nn.logsigmoid', '', False, -4, 4, -4, 4,
    ['t.nn.sigmoid', 't.nn.softplus', 't.nn.log_softmax', 't.special.expit'],
    """t.nn.sigmoid is the curve before the logarithm. t.nn.softplus is the same
shape turned upside down and back to front. t.nn.log_softmax is the
many-choice version: log probabilities across a whole vector.""")

curve_page(
    't.nn.softsign', 't.nn.softsign - a gentle S from -1 to 1',
    """Squeezes every value into -1 to 1 by dividing it by (1 plus its own
size): x divided by (1 plus the absolute value of x).

It is a soft limiter like tanh, but it approaches its limits much more
slowly, so large values keep more of their differences. Where tanh makes
3 and 6 almost the same (0.995 and 0.99999), softsign keeps them apart
(0.75 and 0.86). It is also cheap: no exponential.""",
    """THE CURVE:
0 gives 0, 1 gives 0.5, 3 gives 0.75, 9 gives 0.9 - the output reaches a
fraction n/(n+1) of the way to 1 when the input is n. Symmetric about 0.
Near zero its slope is 1, so small values pass almost unchanged, as with
tanh. The demo's ramp runs from -6 to 6 so you can see how far from 1 it
still is at the edges.""",
    't.nn.softsign', '', True, -6, 6, -2, 2,
    ['t.nn.tanh', 't.nn.sigmoid', 't.nn.hardtanh', 't.nn.tanhshrink'],
    """t.nn.tanh is the usual soft limiter, reaching its limits much sooner.
t.nn.sigmoid is an S from 0 to 1. t.nn.hardtanh is the straight-line
version with corners.""")

curve_page(
    't.nn.gelu', 't.nn.gelu - relu with the corner rounded',
    """Multiplies every value by the probability that a normally distributed
random number falls below it: x times the Gaussian cumulative distribution
of x.

It has relu's overall shape - near zero on the left, the input itself on
the right - with the corner rounded off and a small dip below zero just
left of the origin. It is the activation used in most transformer
language models. In a patch it is a relu whose output changes smoothly as
the input crosses zero.""",
    """THE CURVE:
0 gives 0. Large positive values pass almost unchanged (3 gives 2.996).
Large negative values come out as almost 0 (-3 gives -0.004). In between it
dips: the lowest point is about -0.17, at an input of about -0.75, and then
it climbs back to 0. This node uses the exact form, not the tanh
approximation some libraries offer.""",
    't.nn.gelu', '', False, -3, 3, -1, 3,
    ['t.nn.relu', 't.nn.silu', 't.nn.mish', 't.nn.elu', 't.special.ndtr'],
    """t.nn.relu is the sharp-cornered original. t.nn.silu and t.nn.mish are
very close relatives with slightly deeper dips. t.nn.elu rounds the corner
differently and levels off at -1 rather than coming back to 0.
t.special.ndtr is the Gaussian cumulative distribution gelu multiplies
by.""",
    note='the dip just left of 0 is the difference from relu')

curve_page(
    't.nn.silu', 't.nn.silu - the input times its own sigmoid',
    """Multiplies every value by its own sigmoid: x times 1 over (1 plus e to the
minus x). Also known as swish.

The sigmoid acts as a soft gate on each value: nearly shut for negative
inputs, nearly open for positive ones. The result is relu-shaped but
smooth, with a dip below zero before it rises.""",
    """THE CURVE:
0 gives 0. Positive values come through a little reduced (1 gives 0.73, 4
gives 3.93) and approach the input as they grow. Negative values are
pulled towards 0, but not all the way: the lowest point is about -0.28, at
an input of about -1.28, and very negative inputs come out as almost 0.""",
    't.nn.silu', '', False, -4, 4, -1, 4,
    ['t.nn.gelu', 't.nn.mish', 't.nn.hardswish', 't.nn.sigmoid', 't.nn.relu'],
    """t.nn.gelu is almost the same curve, gated by the Gaussian rather than the
sigmoid. t.nn.mish is smoother still. t.nn.hardswish is its straight-line
imitation. t.nn.sigmoid is the gate on its own.""")

curve_page(
    't.nn.mish', 't.nn.mish - a smooth relu with a dip',
    """Multiplies every value by the tanh of its softplus: x times tanh(log(1
plus e to the x)).

One more member of the smooth-relu family, very close to silu: near zero
for large negative inputs, the input itself for large positive ones, a
small dip below zero in between, and no corner anywhere.""",
    """THE CURVE:
0 gives 0. 1 gives 0.87, 4 gives 3.997. The lowest point is about -0.31,
at an input of about -1.19 - a slightly deeper dip than silu's, and a
slightly quicker climb to the diagonal on the right.""",
    't.nn.mish', '', False, -4, 4, -1, 4,
    ['t.nn.silu', 't.nn.gelu', 't.nn.softplus', 't.nn.tanh', 't.nn.relu'],
    """t.nn.silu and t.nn.gelu are the nearest relatives. t.nn.softplus and
t.nn.tanh are the two pieces it is built from.""")

curve_page(
    't.nn.selu', 't.nn.selu - elu with fixed scale',
    """A scaled elu: for positive values, 1.0507 times the input; for negative
values, 1.0507 times 1.6733 times (e to the x, minus 1).

The two constants are chosen so that a signal passed through many layers
of a network keeps roughly the same average and spread instead of growing
or dying away - "self-normalising". They are fixed; there is nothing to
adjust. Outside a network it is simply elu, slightly stretched.""",
    """THE CURVE:
Above zero a straight line slightly STEEPER than the input (4 gives 4.20),
so it pulls away from the diagonal on the right. Below zero an exponential
that levels off at about -1.76 (-4 gives -1.73). There is a corner at 0:
the slope is 1.05 on the right and 1.76 on the left.""",
    't.nn.selu', '', False, -4, 4, -4, 4,
    ['t.nn.elu', 't.nn.celu', 't.nn.relu', 't.nn.leaky_relu'],
    """t.nn.elu is the same shape with an adjustable floor and no scaling.
t.nn.celu is elu without the corner. t.nn.relu is flat at 0 below zero.""",
    note='right of 0 the output climbs a little faster than the input')

curve_page(
    't.nn.hardsigmoid', 't.nn.hardsigmoid - a straight-line sigmoid',
    """A straight-line imitation of the sigmoid: 0 below -3, 1 above 3, and a
straight ramp from 0 to 1 in between (the input divided by 6, plus 0.5).

It is cheaper than the sigmoid (no exponential) and reaches its limits
exactly, rather than only approaching them. As a shape in a patch it maps
-3..3 onto 0..1 and holds everything outside.""",
    """THE CURVE:
0 gives 0.5. Each unit of input moves the output by one sixth. At -3 and 3
the ramp meets its two flat stretches with a sharp corner each - the
corners a real sigmoid does not have.""",
    't.nn.hardsigmoid', '', False, -6, 6, -1.5, 1.5,
    ['t.nn.sigmoid', 't.nn.hardswish', 't.nn.hardtanh', 't.nn.relu6'],
    """t.nn.sigmoid is the smooth original. t.nn.hardswish multiplies the
input by this. t.nn.hardtanh is the same flat-ramp-flat shape from -1 to 1
with limits you set. t.nn.relu6 is the same shape again, from 0 to 6.""")

curve_page(
    't.nn.hardswish', 't.nn.hardswish - a straight-line silu',
    """An imitation of silu made from straight-line pieces: the input times
hardsigmoid of the input. 0 below -3, the input itself above 3, and a
curve (a parabola) in between.

Like hardsigmoid it is cheap to compute and reaches its limits exactly. It
was designed for networks running on phones. In a patch it is a relu
whose corner is replaced by a short bend.""",
    """THE CURVE:
0 gives 0. Below -3 it is exactly 0; above 3 exactly the input. Between,
it dips to -0.375 at -1.5 and climbs back up - close to silu's shape, with
a visible join at -3 and at 3 where the curve meets the straight
stretches.""",
    't.nn.hardswish', '', False, -5, 5, -1, 5,
    ['t.nn.silu', 't.nn.hardsigmoid', 't.nn.relu6', 't.nn.relu'],
    """t.nn.silu is the smooth original. t.nn.hardsigmoid is the gate it
multiplies by. t.nn.relu6 is what that gate is made from.""")

curve_page(
    't.nn.tanhshrink', 't.nn.tanhshrink - the input minus its tanh',
    """Subtracts the tanh of every value from the value itself: x minus tanh x.

tanh keeps small values and limits large ones; tanhshrink keeps what tanh
throws away. Near zero almost nothing comes through, and far from zero the
value comes through less 1. It is a soft dead zone: small wobbles around
zero are suppressed smoothly, while large values pass with a constant
offset.""",
    """THE CURVE:
0 gives 0, and near zero the curve is very flat (0.5 gives 0.04, 1 gives
0.24). Further out it straightens into a line parallel to the input, 1
below it on the right and 1 above it on the left (4 gives 3.0007).
Symmetric about 0.""",
    't.nn.tanhshrink', '', True, -4, 4, -4, 4,
    ['t.nn.tanh', 't.nn.softshrink', 't.nn.hardshrink', 't.nn.softsign'],
    """t.nn.tanh is the part it subtracts. t.nn.softshrink and t.nn.hardshrink
are dead zones with corners and a width you set.""",
    note='flat near 0, then parallel to the input, 1 inside it')

# ================================================================ ADJUSTABLE
curve_page(
    't.nn.leaky_relu', 't.nn.leaky_relu - relu with a slope below zero',
    """Passes positive values unchanged and multiplies negative values by a
small slope, instead of zeroing them as relu does.

Inside a network the slope keeps a little signal (and gradient) alive for
values that have gone negative, which relu would silence completely. In a
patch it is a two-slope line: full strength above zero, reduced strength
below - a soft rectifier whose amount you choose.""",
    """THE CURVE:
Two straight lines meeting at 0. On the right, the input itself. On the
left, the input times 'negative slope'. Slope 0 is exactly relu; 1 is the
input unchanged; the default 0.01 is so close to relu that you will barely
see it. A negative slope folds negative values up into positive ones (-1
gives the absolute value).""",
    """t.nn.leaky_relu
No arguments; default negative slope 0.01.""",
    """negative slope (drag float on the node, and an inlet):
What negative values are multiplied by. Default 0.01. It does not trigger
the node: the next tensor uses it. In the demo the metro re-sends the
ramp ten times a second, so the curve follows the slider.""",
    False, -4, 4, -4, 4,
    ['t.nn.relu', 't.nn.rrelu', 't.nn.elu', 't.nn.Threshold'],
    """t.nn.relu is slope 0. t.nn.rrelu draws a fresh random slope for every
element from a range instead of using one fixed slope. t.nn.elu bends
smoothly below zero instead of running straight.""",
    params=[('negative slope', 0.01, 0.0, 1.0)], node_h=90,
    note='drag negative slope: 0 is relu, 1 is the input unchanged')

curve_page(
    't.nn.elu', 't.nn.elu - relu with an exponential floor',
    """Passes positive values unchanged and bends negative values along an
exponential curve that levels off at minus alpha: alpha times (e to the x,
minus 1).

Where relu cuts every negative value to exactly 0, elu lets them through
in a compressed form that never goes below -alpha. It is relu with the
negative side softened and given a floor you choose.""",
    """THE CURVE:
On the right, the input itself. On the left, a curve that starts at 0 and
sinks towards -alpha: with the default alpha 1, -1 gives -0.63, -2 gives
-0.86, -4 gives -0.98. Just left of zero its slope is alpha, so at alpha 1
the two sides join smoothly, and at any other alpha there is a corner at
0 (t.nn.celu is the version that never has one). Alpha 0 is exactly relu.""",
    """t.nn.elu
No arguments; default alpha 1.0.""",
    """alpha (drag float on the node, and an inlet):
The depth of the floor: very negative inputs come out as minus alpha.
Default 1.0. It does not trigger the node: the next tensor uses it. In the
demo the metro re-sends the ramp ten times a second, so the curve follows
the slider.""",
    False, -4, 4, -4, 4,
    ['t.nn.celu', 't.nn.selu', 't.nn.relu', 't.nn.leaky_relu', 't.nn.softplus'],
    """t.nn.celu is elu reshaped so it has no corner at any alpha. t.nn.selu is
elu with fixed constants and a scale. t.nn.relu is alpha 0.
t.nn.leaky_relu runs straight below zero instead of levelling off.""",
    params=[('alpha', 1.0, 0.0, 3.0)], node_h=90,
    note='drag alpha: the left side levels off at minus alpha')

curve_page(
    't.nn.celu', 't.nn.celu - elu with no corner',
    """Passes positive values unchanged and bends negative values along an
exponential that levels off at minus alpha: alpha times (e to the x over
alpha, minus 1).

It is elu with the exponential stretched by alpha, so the curve leaves 0
with slope 1 whatever alpha is - the two sides always join smoothly, with
no corner. ("Continuously differentiable": the slope never jumps.)""",
    """THE CURVE:
On the right, the input itself. On the left, a curve that starts along the
diagonal and levels off at -alpha. A small alpha bends it over sharply and
close to zero (alpha 0.25 floors at -0.25, close to relu with a rounded
corner); a large alpha follows the diagonal further before levelling off.
At alpha 1 it is identical to elu.""",
    """t.nn.celu
No arguments; default alpha 1.0.""",
    """alpha (drag float on the node, and an inlet):
The depth of the floor, and how far the curve follows the diagonal before
it bends. Default 1.0. Alpha 0 is not allowed (it divides by alpha): the
console reports "alpha cannot be 0 for CELU" and nothing is sent, which
is why the demo slider starts at 0.1. It does not trigger the node: the
next tensor uses it. The demo's metro re-sends the ramp ten times a
second, so the curve follows the slider.""",
    False, -4, 4, -4, 4,
    ['t.nn.elu', 't.nn.selu', 't.nn.softplus', 't.nn.relu'],
    """t.nn.elu is the same at alpha 1 and has a corner at other values.
t.nn.selu is elu with fixed constants. t.nn.softplus is another relu
without a corner, one that never goes below zero.""",
    params=[('alpha', 1.0, 0.1, 3.0)], node_h=90,
    note='drag alpha: the floor moves, the join at 0 stays smooth')

curve_page(
    't.nn.hardshrink', 't.nn.hardshrink - a dead zone around zero',
    """Sets every value within lambda of zero to exactly 0, and passes
everything outside that band unchanged.

It is a dead zone, or a hard noise gate on values: small wobbles around
zero disappear, and anything big enough survives at full size. Use it to
ignore jitter in a control that rests near zero, or to keep only the large
coefficients of a transform.""",
    """THE CURVE:
Flat at 0 from -lambda to lambda, then the input itself outside. At the
edges of the band the output JUMPS from 0 straight to plus or minus
lambda - a value just past the threshold comes out at full size. Lambda 0
passes everything. (A negative lambda is accepted and also passes
everything.)""",
    """t.nn.hardshrink
No arguments; default lambda 0.5.""",
    """lambda (drag float on the node, and an inlet):
The half-width of the dead zone. Default 0.5. It does not trigger the
node: the next tensor uses it. In the demo the metro re-sends the ramp ten
times a second, so the curve follows the slider.""",
    False, -4, 4, -4, 4,
    ['t.nn.softshrink', 't.nn.Threshold', 't.nn.tanhshrink', 'noise_gate'],
    """t.nn.softshrink has the same dead zone without the jump: survivors are
pulled towards zero by lambda. t.nn.Threshold zeroes (or replaces) only
the values below a level, on one side. t.nn.tanhshrink is a dead zone with
no corners at all. noise_gate does this to a stream of numbers, with
smoothing.""",
    params=[('lambda', 0.5, 0.0, 3.0)], node_h=90,
    note='drag lambda: the flat band widens, with a jump at each edge')

curve_page(
    't.nn.softshrink', 't.nn.softshrink - a dead zone without the jump',
    """Sets every value within lambda of zero to 0, and moves everything outside
that band towards zero by lambda.

Like hardshrink it silences a band around zero, but the survivors are
shrunk rather than passed whole, so the output starts from 0 at the edge
of the band and grows smoothly - no jump. That is the usual way to remove
small noise from a signal (it is the "soft thresholding" of wavelet
denoising): the step in hardshrink is itself a kind of noise.""",
    """THE CURVE:
Flat at 0 from -lambda to lambda. Outside, two lines parallel to the input
but lambda closer to zero: with lambda 0.5, 2 gives 1.5 and -2 gives -1.5.
Two corners, no jumps.""",
    """t.nn.softshrink
No arguments; default lambda 0.5.""",
    """lambda (drag float on the node, and an inlet):
The half-width of the dead zone, and how much every survivor is shrunk.
Default 0.5. It must not be negative: the console reports "lambda must be
greater or equal to 0" and nothing is sent. It does not trigger the node:
the next tensor uses it. In the demo the metro re-sends the ramp ten times
a second, so the curve follows the slider.""",
    False, -4, 4, -4, 4,
    ['t.nn.hardshrink', 't.nn.tanhshrink', 't.nn.Threshold', 'noise_gate'],
    """t.nn.hardshrink keeps the survivors at full size, with a jump at the
edges. t.nn.tanhshrink is a dead zone with no corners. noise_gate does a
similar job on a stream of numbers.""",
    params=[('lambda', 0.5, 0.0, 3.0)], node_h=90,
    note='drag lambda: the band widens and the outer lines move in')

curve_page(
    't.nn.hardtanh', 't.nn.hardtanh - hold values between two limits',
    """Holds every value between a minimum and a maximum: values below the
minimum come out as the minimum, values above the maximum as the maximum,
and everything between passes unchanged.

Despite the name it is simply a clamp - a straight-line imitation of tanh
when the limits are -1 and 1 (its defaults). Use it wherever a tensor must
stay inside a range.""",
    """THE CURVE:
Flat at the minimum, the input itself between the limits, flat at the
maximum. Two sharp corners. Drag the limits and the flat stretches move
up and down.""",
    """t.nn.hardtanh
No arguments; defaults minimum -1, maximum 1.""",
    """minimum / maximum (drag floats on the node, and inlets):
The two limits. Defaults -1 and 1. The minimum must not be above the
maximum: the console reports "min_val cannot be greater than max_val" and
nothing is sent until they are put back in order. Neither triggers the
node: the next tensor uses them. In the demo the metro re-sends the ramp
ten times a second, so the curve follows the sliders.""",
    True, -4, 4, -4, 4,
    ['t.clamp', 'np.clip', 't.nn.tanh', 't.nn.relu6', 't.nn.hardsigmoid'],
    """t.clamp is the general clamp for tensors; np.clip is the numpy one.
t.nn.tanh is the smooth limiter this imitates. t.nn.relu6 is the same
shape fixed at 0 and 6.""",
    params=[('minimum', -1.0, -4.0, 0.0), ('maximum', 1.0, 0.0, 4.0)], node_h=110,
    note='drag the two limits: the flat stretches follow')

curve_page(
    't.nn.rrelu', 't.nn.rrelu - leaky relu with a random slope',
    """Passes positive values unchanged and multiplies each negative value by
its own RANDOM slope, drawn evenly between lower and upper - a fresh draw
for every element, every time a tensor arrives.

This is "randomised leaky relu" as torch uses it while a network is
training: the random slope is a regulariser, noise that keeps a network
from leaning on exact negative values. In a patch it is a leaky rectifier
with a controlled amount of grain below zero: positive values come
through clean, negative ones come out scaled down by a slightly different
amount each time. The same input does NOT give the same output twice. For
one fixed slope use t.nn.leaky_relu.""",
    """THE CURVE:
On the right, the input itself - a clean straight line. On the left, a
ragged band rather than a line: each value is somewhere between the input
times lower and the input times upper, and the band redraws on every
frame. With the defaults (0.125 and 0.333) -4 comes out anywhere from
-0.5 to -1.33. The band is narrow near 0 and widens towards the left, and
lower equal to upper closes it into a single leaky relu line.""",
    """t.nn.rrelu
No arguments; defaults lower 0.125, upper 0.3333.""",
    """lower / upper (drag floats on the node, and inlets):
The ends of the range the random slopes are drawn from. Defaults 0.125
and 0.3333. lower must not be above upper: the console reports
"Lower bound should be less than or equal to the upper bound" and nothing
is sent until they are back in order. Neither triggers the node: the next
tensor uses them. In the demo the metro re-sends the ramp ten times a
second, so the band redraws with new slopes each time and follows the
sliders: drag lower and upper together to close the band into one line,
apart to widen it.""",
    False, -4, 4, -4, 4,
    ['t.nn.leaky_relu', 't.nn.relu', 't.nn.elu', 't.rand'],
    """t.nn.leaky_relu uses one fixed slope, the same for every element.
t.nn.relu is slope 0. t.rand and t.dist.uniform make the kind of random
numbers the slopes are drawn from.""",
    params=[('lower', 0.125, 0.0, 1.0), ('upper', 0.3333, 0.0, 1.0)], node_h=110,
    note='below zero each value gets its own random slope\nbetween lower and upper, new every frame')

curve_page(
    't.nn.softplus', 't.nn.softplus - relu with no corner',
    """A smooth version of relu: the logarithm of (1 plus e to the x), with beta
controlling how sharp the bend is.

It is never negative and never has a corner: nearly 0 for negative
inputs, nearly the input for positive ones, and a smooth curve between.
Use it where a value must stay positive but a hard cut at zero would put
a kink in the response - a positive-only rate, a width, a variance.""",
    """THE CURVE:
With beta 1: 0 gives 0.69 (log 2), -4 gives 0.018, 4 gives 4.018. The
curve lies a little ABOVE relu everywhere and approaches it on both sides.
Beta sharpens it: the formula becomes log(1 plus e to the (beta times x))
divided by beta, so a large beta hugs relu closely (0 gives 0.69 divided
by beta) and a small beta makes a wide, lazy bend.""",
    """t.nn.softplus
No arguments; defaults beta 1.0, threshold 20.""",
    """beta (drag float on the node, and an inlet):
How sharp the bend is. Default 1.0. Larger is closer to relu.

threshold (drag float on the node, and an inlet):
Where the node stops computing the logarithm and passes the input
straight through: wherever beta times x is above threshold. Default 20,
where the difference is far too small to see. It is a shortcut against
overflow, not a shape control - set it low (say 1) and the curve gets a
visible step where the shortcut begins.

Neither triggers the node: the next tensor uses them. In the demo the
metro re-sends the ramp ten times a second, so the curve follows the
slider.""",
    False, -4, 4, -1, 4,
    ['t.nn.relu', 't.nn.elu', 't.nn.logsigmoid', 't.nn.sigmoid', 't.nn.mish'],
    """t.nn.relu is the sharp-cornered limit of large beta. t.nn.logsigmoid is
softplus turned upside down and back to front. t.nn.sigmoid is its slope.
t.nn.elu and t.nn.mish are other smooth relus, both dipping below 0.""",
    params=[('beta', 1.0, 0.25, 8.0)], node_h=110,
    note='drag beta: high hugs relu, low makes a wide bend')

curve_page(
    't.nn.Threshold', 't.nn.Threshold - replace values at or below a level',
    """Passes every value above a threshold unchanged and replaces every value
at or below it with a fixed replacement value.

It is a one-sided substitution: the replacement does not have to be the
threshold, or anything to do with it. Threshold 0 with replacement 0 is
exactly relu; replacement -1 marks the cut values so they can be found
later; a high threshold keeps only the peaks.""",
    """THE CURVE:
Flat at the replacement up to the threshold, then the input itself. If
the replacement differs from the threshold the output jumps at the
threshold.""",
    """t.nn.Threshold <threshold> <replacement>
Defaults 0 and 0. Either may be fractional: the demo's t.nn.Threshold
1.5 -1 passes everything above 1.5 and turns the rest into -1.""",
    """threshold (drag float on the node, and an inlet):
The level. A value must be ABOVE it to pass; a value equal to it is
replaced.

replacement (drag float on the node, and an inlet):
What every value at or below the threshold becomes.

Neither triggers the node: the next tensor uses them. In the demo the
metro re-sends the ramp ten times a second, so the curve follows the
sliders.""",
    True, -4, 4, -4, 4,
    ['t.nn.relu', 't.nn.hardshrink', 't.clamp', 't.gt'],
    """t.nn.relu is Threshold 0 0. t.nn.hardshrink zeroes a band on BOTH sides
of zero. t.clamp replaces out-of-range values with the limit itself.
t.gt makes the mask (above or not) without replacing anything.""",
    params=[('threshold', 1.5, -4.0, 4.0), ('replacement', -1.0, -4.0, 4.0)], node_h=110,
    init='t.nn.Threshold 1.5 -1',
    note='drag threshold and replacement: below the threshold\nthe line is flat at the replacement')

# =================================================================== t.nn.glu
body = """Splits a tensor in half along its last axis and multiplies the first half
by the sigmoid of the second: the gated linear unit.

The second half acts as a set of gates, one per element of the first half:
where a gate value is large and positive the matching value passes almost
whole, where it is large and negative the value is shut off, and at 0 it
passes at half strength. Because half the tensor is used up as gates, the
output is HALF THE SIZE of the input along that axis - 128 values in, 64
out.

Use it when one set of numbers should control how much of another set
gets through - a bank of values and a bank of soft switches carried in
one tensor.

THE OPERATION:
For an input of 2n values, a = the first n, b = the last n, and output =
a times sigmoid(b), element by element. The split is always on the LAST
axis (this node has no dim setting), and that axis must have an even
length: an odd length makes the console report "Halving dimension must be
even" and nothing is sent.

SYNTAX:
t.nn.glu

INPUTS and PARAMETERS:

tensor in:
The tensor, values first and gates second along its last axis. Receiving
it triggers the node. A list, number or numpy array is turned into a
float32 tensor first. An integer tensor is refused (the sigmoid needs
fractions): put a t.to float32 in front of it.

OUTPUTS:

output:
The gated values, half the length of the input along the last axis.

READING THE DEMO:
The input is built from two pieces joined by t.cat: a ramp of 64 values
from -4 to 4 (the values) followed by 64 copies of one gate value (from
t.full - drag 'value' on that node). The top plot shows all 128: the ramp
on the left, the flat gate level on the right. The bottom plot shows the
64 outputs: the ramp scaled by sigmoid(gate). The page opens at gate 1,
where the ramp comes through at 73% (-2.9 to 2.9). At gate 0 it comes
through at half height (-2 to 2); at 4 almost whole (98%); at -4 almost
flat (2%).

RELATED:
t.nn.sigmoid is the gate curve. t.chunk splits a tensor in two the way
glu does, if you want to gate by hand. t.nn.silu is the case where each
value gates itself."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': METRO(100)},
    {'key': 'cmet', 'comment': True, 'text': 'ten times a second', 'pos': (180, 72)},
    {'key': 'tb', 'init': 't bang bang', 'pos': (30, 150), 'w': 140, 'h': 46},
    {'key': 'ctb', 'comment': True, 'text': 'right first: the gates, then the values',
     'pos': (190, 155)},
    {'key': 'ls', 'init': 't.linspace -4 4 64', 'pos': (30, 220), 'w': 190, 'h': 150,
     'props': {'start': -4.0, 'stop': 4.0, 'steps': 64}},
    {'key': 'cls', 'comment': True, 'text': 'the values: 64 from -4 to 4', 'pos': (30, 380)},
    {'key': 'fl', 'init': 't.full 64', 'pos': (260, 220), 'w': 190, 'h': 130,
     'props': {'value': 1.0, 'dim 0': 64}},
    {'key': 'cfl', 'comment': True, 'text': 'the gates: 64 copies of value\ndrag value: -4 to 4',
     'pos': (260, 360)},
    {'key': 'cat', 'init': 't.cat', 'pos': (30, 420), 'w': 170, 'h': 100},
    {'key': 'ccat', 'comment': True, 'text': 'joined: 128 values, gates last', 'pos': (220, 430)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 540), 'w': 440, 'h': 200,
     'props': ARRAY_PLOT(-4.5, 4.5, 128, width=420, height=170)},
    {'key': 'cp1', 'comment': True, 'text': 'the input: ramp, then the flat gate level',
     'pos': (30, 750)},
    {'key': 'glu', 'init': 't.nn.glu', 'pos': (30, 790), 'w': 170, 'h': 70},
    {'key': 'p2', 'init': 'plot', 'pos': (30, 880), 'w': 440, 'h': 200,
     'props': ARRAY_PLOT(-4.5, 4.5, 64, width=420, height=170)},
    {'key': 'cp2', 'comment': True,
     'text': 'the output: 64 values, the ramp times sigmoid(gate)',
     'pos': (30, 1090)},
] + see_also(1140, ['t.nn.sigmoid', 't.nn.silu', 't.chunk', 't.cat'])
links = [('met', '', 'tb', ''),
         ('tb', 'bang', 'fl', '###input', 1), ('tb', 'bang', 'ls', '###input', 0),
         ('fl', 'filled tensor', 'cat', 'tensor 2'),
         ('ls', 'linspace out', 'cat', 'tensor 1'),
         ('cat', 'output', 'p1', 'y'),
         ('cat', 'output', 'glu', 'tensor in'),
         ('glu', 'output', 'p2', 'y')]
print(build('t.nn.glu', 't.nn.glu - half the tensor gates the other half', body,
            demo, links, demo_width=480, text_width=780, text_height=text_h(body)))


# ============================================================ SOFTMAX FAMILY
SCORES = '1.0 2.0 0.5 3.0 1.5 0.0 2.5 1.0'


def softmax_demo(init, out_lo, out_hi, see, node_h=90, extra=None, extra_links=None,
                 note='the output: positive, and summing to 1', sum_node=True,
                 out_fmt='%.2f'):
    """metro -> message (eight scores) -> * scale -> the node -> heat_map,
    with the raw scores in a heat_map above and the sum of the output."""
    demo = [
        {'key': 'met', 'init': 'metro 200', 'pos': (30, 62), 'w': 129, 'h': 70,
         'props': METRO(200)},
        {'key': 'cmet', 'comment': True, 'text': 're-sends the scores five times a second',
         'pos': (180, 72)},
        {'key': 'msg', 'init': 'message', 'pos': (30, 150), 'w': 330, 'h': 42,
         'props': {'text in': SCORES, 'font size': '24', 'width': 310}},
        {'key': 'cmsg', 'comment': True, 'text': 'eight scores - edit them', 'pos': (30, 200)},
        {'key': 'hm0', 'init': 'heat_map', 'pos': (30, 240), 'w': 460, 'h': 80,
         'props': HM(440, 60, 1, 0.0, 3.0)},
        {'key': 'chm0', 'comment': True, 'text': 'the scores: dark 0, bright 3', 'pos': (30, 330)},
        {'key': 'sl', 'init': 'slider', 'pos': (30, 370), 'w': 220, 'h': 42,
         'props': SLIDER(1.0, 0.0, 5.0)},
        {'key': 'csl', 'comment': True, 'text': 'scale: 0 to 5', 'pos': (270, 375)},
        {'key': 'mul', 'init': '* 1.0', 'pos': (30, 430), 'w': 120, 'h': 46,
         'props': {'operand': 1.0}},
        {'key': 'cmul', 'comment': True, 'text': 'the scores times scale', 'pos': (170, 435)},
        {'key': 'nd', 'init': init, 'pos': (30, 500), 'w': 220, 'h': node_h},
        {'key': 'hm1', 'init': 'heat_map', 'pos': (30, 520 + node_h), 'w': 460, 'h': 80,
         'props': HM(440, 60, 1, out_lo, out_hi, fmt=out_fmt)},
        {'key': 'chm1', 'comment': True, 'text': note, 'pos': (30, 610 + node_h)},
    ]
    links = [('met', '', 'msg', ''),
             ('msg', 'message out', 'hm0', 'y'),
             ('msg', 'message out', 'mul', 'in'),
             ('sl', '', 'mul', 'operand'),
             ('mul', 'result', 'nd', 'tensor in'),
             ('nd', '@OUT', 'hm1', 'y')]
    y = 650 + node_h + 15 * note.count('\n')
    if sum_node:
        demo += [
            {'key': 'sum', 'init': 't.sum', 'pos': (30, y), 'w': 170, 'h': 90},
            {'key': 'fs', 'init': 'float', 'pos': (230, y + 20), 'w': 127, 'h': 42,
             'props': FLT},
            {'key': 'cfs', 'comment': True, 'text': 'their sum, whatever the scores',
             'pos': (230, y + 70)},
        ]
        links += [('nd', '@OUT', 'sum', 'tensor in'), ('sum', '@SUM', 'fs', '')]
        y += 120
    demo += (extra or [])
    links += (extra_links or [])
    if extra:
        y = max(e['pos'][1] for e in extra) + 70
    demo += see_also(y, see)
    return demo, links


import json as _json
_IFACE = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iface.json')))


def _out_name(label):
    for e in _IFACE[label]['elements']:
        if e['kind'] == 'output' and e['label'] != '<dynamic>':
            return e['label']
    return ''


def fix_links(links, init):
    out = []
    for l in links:
        l = list(l)
        if l[1] == '@OUT':
            l[1] = _out_name(init.split()[0])
        if l[1] == '@SUM':
            l[1] = _out_name('t.sum')
        out.append(tuple(l))
    return out


DIM_TEXT = """dim (an inlet, with a box on the node):
WHICH AXIS the shares are taken across. Default 0, or the first argument.
The values along that axis sum to 1; every other axis is a separate set.
For a (frames, joints) tensor, dim 1 gives each frame's distribution
across the joints; dim 0 gives each joint's distribution over time.
Those are entirely different questions, and getting dim wrong is the
usual softmax mistake. For a 1-D tensor, as in the demo, 0 (or -1) is the
only axis. Changing dim does not trigger the node."""

SPECIAL_ALIAS = """ALSO DOCUMENTED HERE: {special}
{special} is the same operation (torch.special.{op} is torch's {op}
under another name) with the same default, dim 0. Two differences in the
node: it has NO dim inlet unless you type a dim as its argument
({special} 1 shows one), and its outlet is called 'tensor out' rather
than 'output'."""

TEMP_TEXT = """THE SCALE SLIDER IS THE TEMPERATURE:
Only the DIFFERENCES between the scores matter - adding the same amount to
every score changes nothing. Multiplying them all by a number does: a
large multiplier stretches the differences, so the largest score takes
nearly everything and the result is almost a hard choice; a multiplier
near 0 shrinks them, so every element gets nearly the same share (1/8 =
0.125 here). Dividing by a "temperature" before a softmax is the same
adjustment; the slider multiplies by 1 over the temperature."""

# --------------------------------------------------------------- t.nn.softmax
body = """Turns a tensor of arbitrary numbers into shares of a whole: every value
comes out positive, and they add up to 1.

Each value becomes e to that value, divided by the sum of e to all the
values. The result is a distribution over the elements: the largest score
gets the biggest share, and how much bigger depends on how far ahead it
is. Use it to turn a set of scores - how well each candidate matches, how
active each joint is - into weights you can blend with. The blend changes
smoothly as the scores change, where picking the largest (argmax) would
jump from one to another.

SOFTMAX IS A SOFT ARGMAX:
argmax answers "which one is largest" with a single index. softmax answers
the same question spread out. Scores close together give a flat,
undecided distribution; one score well ahead gives a spiky one that is
nearly a choice.

SYNTAX:
t.nn.softmax <dim: int>
Default dim 0.

EXAMPLE:
t.nn.softmax 1

INPUTS and PARAMETERS:

tensor in:
The scores. Receiving them triggers the node. A list, number or numpy
array is turned into a float32 tensor first. An integer tensor is refused
(the console reports "not implemented for 'Long'"): convert it with t.to
float32.

""" + DIM_TEXT + """

OUTPUTS:

output:
The shares, the same shape as the input, each between 0 and 1, summing
to 1 along dim.

READING THE DEMO:
The eight scores are 1, 2, 0.5, 3, 1.5, 0, 2.5, 1. The top strip shows
them; the bottom strip shows the softmax. At scale 1 the 3 gets 0.38, the
2.5 gets 0.23, and the 0 only 0.02 - the order is kept, the gaps are
exaggerated. The float shows the sum: 1.00 always.

""" + TEMP_TEXT + """

""" + SPECIAL_ALIAS.format(special='t.special.softmax', op='softmax') + """

RELATED:
t.nn.softmin is the same with the smallest score winning. t.nn.log_softmax
gives the logarithm of the shares, accurately. t.nn.gumbel_softmax draws a
random choice weighted by them. t.argmax gives only the winner's index.
t.nn.sigmoid is the two-choice case: sigmoid of x is the softmax of
[x, 0]."""

demo, links = softmax_demo('t.nn.softmax', 0.0, 1.0,
                           ['t.nn.softmin', 't.nn.log_softmax', 't.nn.gumbel_softmax',
                            't.argmax', 't.nn.sigmoid', 't.special.softmax'])
print(build('t.nn.softmax', 't.nn.softmax - scores into shares that sum to 1', body,
            demo, fix_links(links, 't.nn.softmax'), demo_width=500, text_width=780,
            text_height=text_h(body)))

# --------------------------------------------------------------- t.nn.softmin
body = """Turns a tensor of numbers into shares of a whole that favour the SMALLEST
values: every value comes out positive, they add up to 1, and the lowest
score gets the biggest share.

It is softmax of the negated scores - e to minus each value, divided by
the sum of e to minus all of them. Use it when the scores are costs or
distances rather than merits: the nearest target, the cheapest option,
the joint with the least error should get the most weight.

SYNTAX:
t.nn.softmin <dim: int>
Default dim 0.

INPUTS and PARAMETERS:

tensor in:
The scores. Receiving them triggers the node. A list, number or numpy
array is turned into a float32 tensor first. An integer tensor is refused
(the console reports "not implemented for 'Long'"): convert it with t.to
float32.

""" + DIM_TEXT + """

OUTPUTS:

output:
The shares, the same shape as the input, summing to 1 along dim.

READING THE DEMO:
The same eight scores as on the softmax page: 1, 2, 0.5, 3, 1.5, 0, 2.5,
1. Here the 0 (sixth) wins with 0.35 at scale 1, the 0.5 is next with
0.21, and the 3 gets only 0.02 - the softmax picture reversed. The float
shows the sum: always 1.

""" + TEMP_TEXT.replace('the largest score', 'the SMALLEST score') + """

RELATED:
t.nn.softmax favours the largest. t.min finds the smallest value
itself. t.cdist gives the distances softmin is often fed. t.nn.log_softmax
is the logarithm of softmax."""
demo, links = softmax_demo('t.nn.softmin', 0.0, 1.0,
                           ['t.nn.softmax', 't.nn.log_softmax', 't.min', 't.cdist'],
                           note='the output: the SMALLEST scores get the most')
print(build('t.nn.softmin', 't.nn.softmin - shares that favour the smallest', body,
            demo, fix_links(links, 't.nn.softmin'), demo_width=500, text_width=780,
            text_height=text_h(body)))

# ----------------------------------------------------------- t.nn.log_softmax
body = """Gives the natural logarithm of the softmax of a tensor, computed in one
step so that it stays accurate.

Each value becomes that value minus the log of the sum of e to all the
values. The results are log probabilities: all zero or negative, the
largest score closest to 0, and e to the outputs summing to 1. Use it
when you will ADD probabilities in log form (scoring a sequence, a loss),
or need to tell apart shares far too small to see in plain softmax.

WHY NOT softmax THEN A LOG:
Small shares underflow to exactly 0 in plain softmax, and the log of 0 is
minus infinity. With scores 0 and 200, softmax gives the 0 a share of
exactly 0 in float32 and the log comes out minus infinity;
log_softmax gives -200, the right answer.

A useful property: the GAPS between outputs are the gaps between the
scores. Only the whole set moves, down by one amount, so that the shares
it stands for add up to 1.

SYNTAX:
t.nn.log_softmax <dim: int>
Default dim 0.

INPUTS and PARAMETERS:

tensor in:
The scores. Receiving them triggers the node. A list, number or numpy
array is turned into a float32 tensor first. An integer tensor is refused
(the console reports "not implemented for 'Long'"): convert it with t.to
float32.

""" + DIM_TEXT.replace('The values along that axis sum to 1',
                 'e to the values along that axis sums to 1') + """

OUTPUTS:

output:
The log shares, the same shape as the input, all 0 or below.

READING THE DEMO:
The same eight scores as on the softmax page. At scale 1 the 3 comes out
-0.96 (the log of its 0.38 share), and every other output sits exactly as
far below it as its score did: the 2.5 at -1.46, the 0 at -3.96. The
bottom strip runs from dark at -6 to bright at 0. The list shows the
values. Raise the scale and the low scores sink much further while the
winner climbs towards 0.

""" + SPECIAL_ALIAS.format(special='t.special.log_softmax', op='log_softmax') + """

RELATED:
t.nn.softmax is the plain shares. t.nn.logsigmoid is the two-choice case.
t.cross_entropy_loss applies log_softmax internally. t.logcumsumexp is the
running log of summed exponentials."""

demo, links = softmax_demo(
    't.nn.log_softmax', -6.0, 0.0,
    ['t.nn.softmax', 't.nn.logsigmoid', 't.cross_entropy_loss', 't.special.log_softmax'],
    note='the output: log shares, all below 0', sum_node=False,
    extra=[{'key': 'ls', 'init': 'list', 'pos': (30, 740), 'w': 460, 'h': 42,
            'props': dict(LIST, width=440)},
           {'key': 'cls', 'comment': True, 'text': 'the same values as numbers',
            'pos': (30, 790)}],
    extra_links=[('nd', '@OUT', 'ls', '')])
print(build('t.nn.log_softmax', 't.nn.log_softmax - log shares, computed safely', body,
            demo, fix_links(links, 't.nn.log_softmax'), demo_width=500, text_width=780,
            text_height=text_h(body)))

# ------------------------------------------------------- t.nn.gumbel_softmax
body = """Makes a random choice among the elements, weighted by their scores, as a
softmax: each call adds fresh random noise (Gumbel noise) to the scores
and then takes a softmax.

Run it repeatedly on the same scores and each element comes out on top
as often as its plain softmax share says it should - the 3 in the demo
wins about 38% of the time, the 0 about 2%. So where t.nn.softmax
DESCRIBES the distribution, gumbel_softmax SAMPLES from it. With 'hard'
on, the output is exactly one 1 and the rest 0: a one-of-N pick, the way
you would choose a random state, sound or pose by weight.

SYNTAX:
t.nn.gumbel_softmax
No arguments.

INPUTS and PARAMETERS:

tensor in:
The scores (unnormalised log probabilities). Receiving them triggers the
node, and every arrival draws new noise, so the same input gives a
different output each time. A list, number or numpy array is turned into
a float32 tensor first; an integer tensor is refused.

tau (drag float on the node, and an inlet):
The temperature of the softmax taken after the noise. Default 1. Low tau
(0.1) makes each draw nearly one-of-N - one element close to 1, the
others close to 0. High tau (5) blurs each draw towards even shares.
Must be above 0. Does not trigger.

hard (checkbox on the node, and an inlet):
On: the output is exactly one-of-N, a single 1 at the winner. Default off.
Does not trigger.

dim (box on the node, and an inlet):
The axis the choice is made along. Default -1, the last axis. For a 2-D
tensor, -1 (or 1) makes one choice per ROW, each row summing to 1; 0
makes one choice per COLUMN. The demo's scores are 1-D, so there is only
one axis (0 and -1 are the same). An axis the tensor does not have is
refused with a console message and nothing is sent. Does not trigger.

OUTPUTS:

output:
One random draw, the same shape as the input, summing to 1 along the
dim axis.

READING THE DEMO:
The same eight scores as on the softmax page, re-sent five times a
second, so the bottom strip changes on every draw. Watch which cell lights
up: mostly the fourth (score 3) and seventh (2.5), sometimes others,
rarely the sixth (score 0). The int shows the winner's index (t.argmax),
counting from 0. Tick 'hard' on the node to see clean one-of-N picks;
drag tau up and the draws turn soft and even. The scale slider is a
second temperature, applied before the noise: it changes how often each
element wins, where tau changes only how decisive each draw looks.

RELATED:
t.nn.softmax is the distribution being sampled. t.argmax picks the
winner's index. t.dist.gumbel is the noise itself. With hard on, this node
is the patch's way to make a weighted random choice."""
demo, links = softmax_demo(
    't.nn.gumbel_softmax', 0.0, 1.0,
    ['t.nn.softmax', 't.argmax', 't.dist.gumbel', 't.dist.bernoulli'],
    node_h=140, sum_node=False, note='one random draw: changes every time',
    extra=[{'key': 'am', 'init': 't.argmax', 'pos': (30, 800), 'w': 170, 'h': 90},
           {'key': 'ia', 'init': 'int', 'pos': (230, 820), 'w': 110, 'h': 42, 'props': INT},
           {'key': 'cia', 'comment': True, 'text': 'the winner: 0 is the first cell',
            'pos': (230, 870)}],
    extra_links=[('nd', '@OUT', 'am', 'tensor in'), ('am', 'max index', 'ia', '')])
print(build('t.nn.gumbel_softmax', 't.nn.gumbel_softmax - a weighted random choice', body,
            demo, fix_links(links, 't.nn.gumbel_softmax'), demo_width=500, text_width=780,
            text_height=text_h(body)))
