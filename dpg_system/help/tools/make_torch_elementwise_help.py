"""torch element-by-element operations: one page per node.

t.clamp, t.maximum, t.minimum (limits); t.round, t.ceil, t.floor, t.trunc,
t.frac (rounding); t.eq, t.ge, t.gt, t.le, t.lt, t.ne (comparisons);
t.gcd, t.lcm; t.copysign; t.complex, t.real, t.imag.

These were the t.clamp, t.round, t.comparison, t.gcd, t.complex and
t.copysign family pages. Each node now has its own page whose demo shows
the input and the output in the same display, with a SEE ALSO row under it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT


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


def MULTI_PLOT(miny=-1.0, maxy=1.0, count=200, width=300):
    """A plot that draws each element of a list (or each row) as its own line."""
    return dict(PLOT(miny, maxy, count),
                **{'width': width, 'update style': 'input is multi-channel sample'})


METRO = lambda ms: {'on': True, 'period': float(ms), 'units': 'milliseconds'}


def text_h(body):
    """Height of the prose block: 15px per line at text size 30, plus a margin."""
    return (body.count('\n') + 1) * 15 + 30


def page(name, title, body, demo, links, subpatch=None, demo_width=620):
    print(build(name, title, body, demo, links, demo_width=demo_width, text_width=760,
                text_height=text_h(body), subpatch=subpatch))


# --------------------------------------------------------- demo subpatchers
# make_sine: metro 100 -> t.linspace 0 720 128 -> sin (degrees): two cycles
# of a sine as one tensor of 128 values, re-sent ten times a second.
SINE_SUB = {
    'name': 'make_sine',
    'host': 'mk',
    'demo': [
        {'key': 'met', 'init': 'metro 100', 'pos': (40, 40), 'w': 129, 'h': 70,
         'props': METRO(100)},
        {'key': 'cm', 'init': 'comment re-sends the wave ten times a second',
         'pos': (190, 50), 'w': 320, 'h': 30},
        {'key': 'ls', 'init': 't.linspace 0 720 128', 'pos': (40, 130), 'w': 190, 'h': 150,
         'props': {'start': 0.0, 'stop': 720.0, 'steps': 128}},
        {'key': 'cs', 'init': 'comment 128 angles from 0 to 720 degrees',
         'pos': (250, 145), 'w': 300, 'h': 30},
        {'key': 'sn', 'init': 'sin', 'pos': (40, 300), 'w': 120, 'h': 70,
         'props': {'degrees': True}},
        {'key': 'cn', 'init': 'comment their sines: two cycles, from -1 to 1',
         'pos': (180, 315), 'w': 340, 'h': 30},
        {'key': 'out', 'init': 'out sine', 'pos': (40, 400), 'w': 150, 'h': 80},
    ],
    'links': [('met', '', 'ls', '###input'),
              ('ls', 'linspace out', 'sn', 'in'),
              ('sn', 'out', 'out', 'sine')],
}

# make_sine_on_bang: the same wave, made whenever a bang arrives, so one metro
# outside can send the wave and a second tensor in a known order.
SINE_BANG_SUB = {
    'name': 'make_sine',
    'host': 'mk',
    'demo': [
        {'key': 'in', 'init': 'in bang', 'pos': (40, 40), 'w': 150, 'h': 80},
        {'key': 'ls', 'init': 't.linspace 0 720 128', 'pos': (40, 140), 'w': 190, 'h': 150,
         'props': {'start': 0.0, 'stop': 720.0, 'steps': 128}},
        {'key': 'cs', 'init': 'comment 128 angles from 0 to 720 degrees',
         'pos': (250, 155), 'w': 300, 'h': 30},
        {'key': 'sn', 'init': 'sin', 'pos': (40, 310), 'w': 120, 'h': 70,
         'props': {'degrees': True}},
        {'key': 'cn', 'init': 'comment their sines: two cycles, from -1 to 1',
         'pos': (180, 325), 'w': 340, 'h': 30},
        {'key': 'out', 'init': 'out sine', 'pos': (40, 410), 'w': 150, 'h': 80},
    ],
    'links': [('in', 'bang', 'ls', '###input'),
              ('ls', 'linspace out', 'sn', 'in'),
              ('sn', 'out', 'out', 'sine')],
}

# make_counts: the whole numbers 1 to 12 twice - as a row (12,) and as a
# column (12, 1). An operation between them broadcasts to a 12 x 12 table.
COUNTS_SUB = {
    'name': 'make_counts',
    'host': 'mc',
    'demo': [
        {'key': 'in', 'init': 'in bang', 'pos': (40, 40), 'w': 150, 'h': 80},
        {'key': 'ls', 'init': 't.linspace 1 12 12', 'pos': (40, 140), 'w': 190, 'h': 150,
         'props': {'start': 1.0, 'stop': 12.0, 'steps': 12}},
        {'key': 'cs', 'init': 'comment the numbers 1 to 12, as a row',
         'pos': (250, 155), 'w': 300, 'h': 30},
        {'key': 'rs', 'init': 't.reshape 12 1', 'pos': (240, 320), 'w': 230, 'h': 70,
         'props': {'shape': '12, 1'}},
        {'key': 'cr', 'init': 'comment and stood up as a column, 12 rows of 1',
         'pos': (240, 400), 'w': 360, 'h': 30},
        {'key': 'ro', 'init': 'out row', 'pos': (40, 450), 'w': 150, 'h': 80},
        {'key': 'co', 'init': 'out column', 'pos': (240, 450), 'w': 150, 'h': 80},
    ],
    # the row leaves first: it goes to the cold inlet outside
    'links': [('in', 'bang', 'ls', '###input'),
              ('ls', 'linspace out', 'ro', 'row'),
              ('ls', 'linspace out', 'rs', 'tensor in'),
              ('rs', 'output', 'co', 'column')],
}

# make_turning: a complex number of size 1 turning once every 4 seconds.
# One saw (0 to 360 degrees) feeds both sin and cos, so the two parts always
# belong to the same angle; the imaginary part (right outlet) goes first.
TURN_SUB = {
    'name': 'make_turning',
    'host': 'mt',
    'demo': [
        {'key': 'sg', 'init': 'signal 4.0 saw', 'pos': (40, 40), 'w': 129, 'h': 78,
         'props': SIG('saw', 4.0, 360.0, False)},
        {'key': 'cm', 'init': 'comment an angle, 0 to 360 degrees every 4 s',
         'pos': (190, 50), 'w': 360, 'h': 30},
        {'key': 'tr', 'init': 't float float', 'pos': (40, 140), 'w': 150, 'h': 46},
        {'key': 'co', 'init': 'cos', 'pos': (40, 210), 'w': 120, 'h': 70,
         'props': {'degrees': True}},
        {'key': 'si', 'init': 'sin', 'pos': (220, 210), 'w': 120, 'h': 70,
         'props': {'degrees': True}},
        {'key': 'cx', 'init': 't.complex', 'pos': (40, 300), 'w': 200, 'h': 90},
        {'key': 'cc', 'init': 'comment one complex number of size 1, going round',
         'pos': (260, 315), 'w': 400, 'h': 30},
        {'key': 'out', 'init': 'out complex', 'pos': (40, 420), 'w': 150, 'h': 80},
    ],
    'links': [('sg', '', 'tr', ''),
              ('tr', 'float', 'si', 'in', 1), ('tr', 'float', 'co', 'in', 0),
              ('si', 'out', 'cx', 'imag tensor in'), ('co', 'out', 'cx', 'real tensor in'),
              ('cx', 'complex tensor out', 'out', 'complex')],
}


# =================================================================== limits
# ------------------------------------------------------------------- t.clamp
body = """Holds every value of a tensor inside two limits: anything below min
comes out as min, anything above max comes out as max, and everything in
between passes untouched.

Use it to keep values in range before they drive something that would
misbehave outside it - colours inside 0 to 1, a gain that must never go
negative, a denominator kept away from zero. It works on every element
separately, so the output has the same shape as the input.

SYNTAX:
t.clamp <min: float> <max: float>
Defaults: min 0.0, max 1.0. Give BOTH numbers: with only one argument,
both limits keep their defaults.

EXAMPLE:
t.clamp -0.5 0.7

INPUTS and PARAMETERS:

tensor in:
The tensor to limit. A list, a NumPy array or a single number is turned
into a float32 tensor first (a single number becomes a one-value tensor).
Every input is clamped and sent - this is the only inlet that makes the
node send. An empty tensor sends nothing.

min / max (drag floats, also inlets):
The lower and upper limits, applied to every element. Because they are
inlets they can be driven from the patch. Changing one does NOT send by
itself - the next tensor uses it. In the demo make_sine re-sends the wave
ten times a second, so the plot follows your drags. If min is above max,
every value comes out as max.

OUTPUTS:

output:
The clamped tensor, the same shape as the input. An integer tensor comes
out as float, because the limits are floats.

READING THE DEMO:
The plot draws the wave and the clamped copy over each other. Inside the
limits the two lines are one; past them the wave carries on to 1 or -1
while the copy runs flat along the limit - the top cut at 0.7, the bottom
at -0.5. Drag max down towards min and the wave is squeezed into a band.

A NOTE ON GRADIENTS:
Where a value is held at a limit the gradient through it is zero - the
output stops responding to the input. If something you are training has
stopped learning, a saturated clamp is worth checking for.

RELATED:
t.maximum and t.minimum limit one side only, and against another tensor,
so the limit can differ at every position. np.clip is the NumPy version;
clamp does the same for single numbers and lists."""

demo = [
    {'key': 'mk', 'init': 'p make_sine', 'pos': (30, 62), 'w': 170, 'h': 70},
    {'key': 'c0', 'comment': True,
     'text': 'a sine as a tensor of 128 values, sent 10 times a second',
     'pos': (30, 150)},
    {'key': 'cl', 'init': 't.clamp -0.5 0.7', 'pos': (30, 200), 'w': 190, 'h': 110,
     'props': {'min': -0.5, 'max': 0.7}},
    {'key': 'c1', 'comment': True, 'text': 'drag min and max on the node', 'pos': (240, 220)},
    {'key': 'pk', 'init': 'pack 2', 'pos': (30, 340), 'w': 130, 'h': 100},
    {'key': 'cp', 'comment': True, 'text': 'the clamped copy (in 1) and the wave (in 2)',
     'pos': (180, 360)},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 460), 'w': 308, 'h': 178,
     'props': MULTI_PLOT(-1.2, 1.2, 128)},
    {'key': 'c2', 'comment': True,
     'text': 'the wave and its clamped copy, drawn over each other:\nwhere they part, the copy runs flat at the limit',
     'pos': (30, 650)},
] + see_also(740, ['t.maximum', 't.minimum', 'np.clip', 'clamp', 't.round'])
# the raw wave reaches pack's cold inlet before the clamped copy fires it
links = [('mk', 'sine', 'pk', 'in 2'),
         ('mk', 'sine', 'cl', 'tensor in'),
         ('cl', 'output', 'pk', 'in 1'),
         ('pk', 'out', 'pl', 'y')]
page('t.clamp', 't.clamp - hold a tensor between two limits', body, demo, links,
     subpatch=SINE_SUB)


# --------------------------------------------------------- t.maximum / minimum
def maxmin_demo(node, out):
    """metro -> ramp (b, first) and make_sine (a); pack 3 -> plot of result,
    wave and ramp, so the result can be seen tracing one or the other."""
    demo = [
        {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
         'props': METRO(100)},
        {'key': 'c0', 'comment': True, 'text': 'ten times a second: the ramp, then the wave',
         'pos': (180, 72)},
        {'key': 'rp', 'init': 't.linspace -1 1 128', 'pos': (330, 150), 'w': 190, 'h': 150,
         'props': {'start': -1.0, 'stop': 1.0, 'steps': 128}},
        {'key': 'cr', 'comment': True, 'text': 'tensor b: a ramp - drag start and stop',
         'pos': (330, 310)},
        {'key': 'mk', 'init': 'p make_sine', 'pos': (30, 150), 'w': 170, 'h': 70},
        {'key': 'cs', 'comment': True, 'text': 'tensor a: two cycles of a sine',
         'pos': (30, 230)},
        {'key': 'nd', 'init': node, 'pos': (30, 350), 'w': 190, 'h': 90},
        {'key': 'pk', 'init': 'pack 3', 'pos': (30, 470), 'w': 130, 'h': 120},
        {'key': 'cp', 'comment': True, 'text': 'result (in 1), wave (in 2), ramp (in 3)',
         'pos': (180, 490)},
        {'key': 'pl', 'init': 'plot', 'pos': (30, 610), 'w': 308, 'h': 178,
         'props': MULTI_PLOT(-1.2, 1.2, 128)},
    ]
    links = [('met', '', 'rp', '###input'),
             ('met', '', 'mk', 'bang'),
             ('rp', 'linspace out', 'nd', 'tensor b in'),
             ('rp', 'linspace out', 'pk', 'in 3'),
             ('mk', 'sine', 'pk', 'in 2'),
             ('mk', 'sine', 'nd', 'tensor a in'),
             ('nd', out, 'pk', 'in 1'),
             ('pk', 'out', 'pl', 'y')]
    return demo, links


body = """Compares two tensors element by element and keeps the LARGER value
at every position.

Where t.clamp has one floor for the whole tensor, t.maximum's floor is
another tensor, so it can be different at every position: a per-element
lower bound, a noise floor that varies across a spectrum, or simply the
upper envelope of two signals. With a single number in tensor b it is a
plain floor, the same everywhere (t.maximum against 0 keeps only the
positive part of a signal).

THE NAME AND WHAT IT DOES:
t.maximum does not find the largest value in a tensor - that is t.max,
the reduction. It takes the larger of two values, one pair at a time,
which keeps the result at or ABOVE tensor b: a floor.

SYNTAX:
t.maximum
It takes no arguments.

INPUTS and PARAMETERS:

tensor a in:
The first tensor (a list or number is turned into a float32 tensor).
Receiving it makes the node compare and send.

tensor b in:
The second tensor, or a single number used everywhere. It is only stored
when it arrives, so send it FIRST; until it has received something the
node sends nothing. It is converted to tensor a's type and device, so
against an integer tensor a b of 0.5 becomes 0.

The two shapes must broadcast: the same shape, a single number, or shapes
that differ only where one of them has length 1. Shapes that cannot be
matched raise an error and send nothing.

OUTPUTS:

maximum tensor:
The larger of a and b at every position, in the broadcast shape.

READING THE DEMO:
The plot draws the result over the two inputs: the sine (a) and a ramp
from -1 to 1 (b). Wherever the sine is above the ramp the result follows
the sine; wherever it dips below, the result rides along the ramp. So on
the left, where the ramp is low, almost all of the wave survives, and on
the right only the peaks that rise above the ramp are left. Drag the
ramp's start and stop (try both 0: a flat floor at 0 that cuts off every
trough).

RELATED:
t.minimum keeps the smaller value instead. t.clamp limits both sides
with single numbers. t.max finds the largest value in one tensor.
np.max and max do the same floor for NumPy arrays and numbers."""

demo, links = maxmin_demo('t.maximum', 'maximum tensor')
demo += [{'key': 'c2', 'comment': True,
          'text': 'the result follows whichever is higher:\nthe sine where it is above the ramp, the ramp elsewhere',
          'pos': (30, 800)}] + see_also(890, ['t.minimum', 't.clamp', 't.max', 'np.max', 'max'])
page('t.maximum', 't.maximum - the larger of two tensors, element by element',
     body, demo, links, subpatch=SINE_BANG_SUB)

body = """Compares two tensors element by element and keeps the SMALLER value
at every position.

Where t.clamp has one ceiling for the whole tensor, t.minimum's ceiling
is another tensor, so it can be different at every position: a
per-element upper limit, a cap that varies across a picture, or simply
the lower envelope of two signals. With a single number in tensor b it
is a plain ceiling, the same everywhere.

THE NAME AND WHAT IT DOES:
t.minimum does not find the smallest value in a tensor - that is t.min,
the reduction. It takes the smaller of two values, one pair at a time,
which keeps the result at or BELOW tensor b: a ceiling.

SYNTAX:
t.minimum
It takes no arguments.

INPUTS and PARAMETERS:

tensor a in:
The first tensor (a list or number is turned into a float32 tensor).
Receiving it makes the node compare and send.

tensor b in:
The second tensor, or a single number used everywhere. It is only stored
when it arrives, so send it FIRST; until it has received something the
node sends nothing. It is converted to tensor a's type and device, so
against an integer tensor a b of 0.5 becomes 0.

The two shapes must broadcast: the same shape, a single number, or shapes
that differ only where one of them has length 1. Shapes that cannot be
matched raise an error and send nothing.

OUTPUTS:

minimum tensor:
The smaller of a and b at every position, in the broadcast shape.

READING THE DEMO:
The plot draws the result over the two inputs: the sine (a) and a ramp
from -1 to 1 (b). Wherever the sine is below the ramp the result follows
the sine; wherever it rises above, the result is held down along the
ramp. So on the left, where the ramp is low, only the troughs that dip
beneath it are left, and on the right almost all of the wave survives.
Drag the ramp's start and stop (try both 0: a flat ceiling at 0 that cuts
off every peak).

RELATED:
t.maximum keeps the larger value instead. t.clamp limits both sides with
single numbers. t.min finds the smallest value in one tensor. np.min and
min do the same ceiling for NumPy arrays and numbers."""

demo, links = maxmin_demo('t.minimum', 'minimum tensor')
demo += [{'key': 'c2', 'comment': True,
          'text': 'the result follows whichever is lower:\nthe sine where it is below the ramp, the ramp elsewhere',
          'pos': (30, 800)}] + see_also(890, ['t.maximum', 't.clamp', 't.min', 'np.min', 'min'])
page('t.minimum', 't.minimum - the smaller of two tensors, element by element',
     body, demo, links, subpatch=SINE_BANG_SUB)


# ================================================================= rounding
def round_demo(node, props, see, halves, halves_note, plot_range=(-3.0, 3.0), note=''):
    """metro -> t.linspace -2.5 2.5 128 (the ramp) -> pack in 2, and through
    the node -> pack in 1: one plot of the ramp and the staircase it makes.
    Below: a message of chosen values -> node -> list, numbers you can read."""
    demo = [
        {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
         'props': METRO(100)},
        {'key': 'c0', 'comment': True, 'text': 're-sends the ramp ten times a second',
         'pos': (180, 72)},
        {'key': 'rp', 'init': 't.linspace -2.5 2.5 128', 'pos': (30, 150), 'w': 200, 'h': 150,
         'props': {'start': -2.5, 'stop': 2.5, 'steps': 128}},
        {'key': 'cr', 'comment': True,
         'text': '128 values climbing evenly from -2.5 to 2.5\ndrag start and stop to move the ramp',
         'pos': (250, 165)},
        {'key': 'nd', 'init': node, 'pos': (30, 320), 'w': 160, 'h': 70 if not props else 100,
         'props': props},
        {'key': 'cn', 'comment': True, 'text': note or 'every value, rounded on its own',
         'pos': (250, 335)},
        {'key': 'pk', 'init': 'pack 2', 'pos': (30, 420), 'w': 130, 'h': 100},
        {'key': 'cp', 'comment': True, 'text': 'the result (in 1) and the ramp (in 2)',
         'pos': (180, 440)},
        {'key': 'pl', 'init': 'plot', 'pos': (30, 540), 'w': 308, 'h': 178,
         'props': MULTI_PLOT(plot_range[0], plot_range[1], 128)},
        {'key': 'c2', 'comment': True,
         'text': 'the ramp, and the result drawn over it',
         'pos': (30, 730)},
        {'key': 'msg', 'init': 'message', 'pos': (30, 790), 'w': 330, 'h': 42,
         'props': {'text in': halves, 'font size': '24', 'width': 300}},
        {'key': 'lb', 'init': 'load_bang', 'pos': (380, 740), 'w': 88, 'h': 46},
        {'key': 'cm', 'comment': True, 'text': 'sent on opening; click to send again',
         'pos': (380, 795)},
        {'key': 'n2', 'init': node, 'pos': (30, 860), 'w': 160, 'h': 70 if not props else 100,
         'props': props},
        {'key': 'l1', 'init': 'list', 'pos': (30, 970), 'w': 330, 'h': 42, 'props': LIST},
        {'key': 'cl', 'comment': True, 'text': halves_note, 'pos': (30, 1025)},
    ] + see_also(1100, see)
    links = [('met', '', 'rp', '###input'),
             ('rp', 'linspace out', 'pk', 'in 2'),
             ('rp', 'linspace out', 'nd', 'tensor in'),
             ('nd', 'output', 'pk', 'in 1'),
             ('pk', 'out', 'pl', 'y'),
             ('lb', 'out', 'msg', ''),
             ('msg', 'message out', 'n2', 'tensor in'),
             ('n2', 'output', 'l1', '')]
    return demo, links


ROUND_COMMON_IN = """tensor in:
The tensor. A list, a NumPy array or a single number is turned into a
float32 tensor first. Every input is processed and sent - this is the
only inlet that makes the node send. An empty tensor sends nothing."""

# ------------------------------------------------------------------- t.round
body = """Rounds every value of a tensor to the nearest whole number - or,
with decimals, to the nearest tenth, hundredth, ten or hundred.

Use it to snap continuous values onto a grid: a fader position onto
whole steps, a pitch onto semitones, a coordinate onto a cell. Each
element is rounded on its own, so the output has the shape of the input.

HALVES GO TO THE EVEN NEIGHBOUR:
A value exactly half-way, like 0.5 or 2.5, goes to whichever neighbour is
EVEN: 0.5 becomes 0, 1.5 becomes 2, 2.5 becomes 2, -0.5 becomes -0. This
is deliberate and standard (always rounding halves up adds a small bias
that builds up over many values), but it surprises people who expect 0.5
to become 1. If you want halves always up, add 0.5 and use t.floor.

SYNTAX:
t.round
It takes no arguments.

EXAMPLE:
t.round

INPUTS and PARAMETERS:

""" + ROUND_COMMON_IN + """

decimals (int box, also an inlet):
How many decimal places to keep. Default 0, whole numbers. 1 rounds to
tenths, 2 to hundredths. Negative values round to the left of the point:
-1 to tens, -2 to hundreds (halves still go even: 15 and 25 both become
20). Changing it does not send by itself; the next tensor uses it.

OUTPUTS:

output:
The rounded tensor, the same shape and type as the input - still floats,
just with whole-number values.

READING THE DEMO:
The ramp climbs straight from -2.5 to 2.5; t.round turns it into a
staircase of flat steps at -2, -1, 0, 1, 2, each step centred on its
whole number and rising half-way between. Set decimals to 1 and the
steps become ten times finer, almost hugging the ramp. The message
below (sent when the page opens, and again when clicked) shows the
half-way values going to even numbers.

RELATED:
t.floor, t.ceil and t.trunc turn fractions into whole numbers in the
other three ways - they differ in which way they go. t.frac keeps the
part t.trunc throws away. round does the same for numbers and NumPy
arrays (np.round shares its page)."""

demo, links = round_demo('t.round', {'decimals': 0},
                         ['t.floor', 't.ceil', 't.trunc', 't.frac', 'round'],
                         '0.5 1.5 2.5 3.5 -0.5 -1.5', 'halves go to the even neighbour: 0 2 2 4 -0 -2',
                         note='drag decimals: 0 whole numbers, 1 tenths')
page('t.round', 't.round - to the nearest whole number', body, demo, links)

# ------------------------------------------------------------------- t.floor
body = """Rounds every value of a tensor DOWN to the whole number at or below
it - down the number line, so 2.7 becomes 2 and -2.3 becomes -3.

Use it to find which whole-numbered cell, bin or step a value falls in:
floor of a position divided by a cell size is the cell's index, and that
stays right for negative positions too. Each element is rounded on its
own, so the output has the shape of the input.

FLOOR, CEIL AND TRUNC DIFFER FOR NEGATIVES:
For positive values floor and trunc agree. For negative values floor
still goes DOWN (-2.3 becomes -3) while trunc goes towards zero (-2.3
becomes -2). If a value on the negative side lands one step lower than
you expected, that is floor doing its job - use t.trunc if you meant
"drop the fraction".

SYNTAX:
t.floor
It takes no arguments and has no parameters.

INPUTS and PARAMETERS:

""" + ROUND_COMMON_IN + """

OUTPUTS:

output:
The floored tensor, the same shape and type as the input - still floats,
just with whole-number values. An integer tensor comes back unchanged.

READING THE DEMO:
The ramp climbs straight from -2.5 to 2.5; t.floor turns it into a
staircase that always sits ON or UNDER the ramp: each step starts where
the ramp crosses a whole number and stays flat until the next one. The
message shows values either side of zero going down.

RELATED:
t.ceil goes up instead, t.trunc towards zero, t.round to the nearest.
t.frac keeps the fractional part. floor does the same for numbers and
NumPy arrays (np.floor shares its page)."""

demo, links = round_demo('t.floor', {},
                         ['t.ceil', 't.trunc', 't.round', 't.frac', 'floor'],
                         '2.7 2.3 -2.3 -2.7 -0.5', 'always down: 2 2 -3 -3 -1',
                         note='every value, down to the whole number below')
page('t.floor', 't.floor - round down', body, demo, links)

# -------------------------------------------------------------------- t.ceil
body = """Rounds every value of a tensor UP to the whole number at or above
it - up the number line, so 2.3 becomes 3 and -2.7 becomes -2.

Use it when a part counts as a whole: how many pages, buffers or beats
are needed to hold a length, where anything over an exact number needs
one more. Each element is rounded on its own, so the output has the
shape of the input.

CEIL IS NOT "AWAY FROM ZERO":
For negative values ceil still goes UP, which is towards zero: -2.7
becomes -2. There is no node that rounds away from zero directly; t.ceil
of the size with t.copysign to restore the sign does it.

SYNTAX:
t.ceil
It takes no arguments and has no parameters.

INPUTS and PARAMETERS:

""" + ROUND_COMMON_IN + """

OUTPUTS:

output:
The tensor rounded up, the same shape and type as the input - still
floats, just with whole-number values. An integer tensor comes back
unchanged.

READING THE DEMO:
The ramp climbs straight from -2.5 to 2.5; t.ceil turns it into a
staircase that always sits ON or ABOVE the ramp: each step jumps up as
soon as the ramp passes a whole number. Compare it with t.floor's page,
where the same staircase sits one step lower. The message shows values
either side of zero going up.

RELATED:
t.floor goes down instead, t.trunc towards zero, t.round to the nearest.
ceil does the same for numbers and NumPy arrays (np.ceil shares its
page)."""

demo, links = round_demo('t.ceil', {},
                         ['t.floor', 't.trunc', 't.round', 't.copysign', 'ceil'],
                         '2.3 2.7 -2.3 -2.7 0.5', 'always up: 3 3 -2 -2 1',
                         note='every value, up to the whole number above')
page('t.ceil', 't.ceil - round up', body, demo, links)

# ------------------------------------------------------------------- t.trunc
body = """Drops the fractional part of every value in a tensor, keeping the
whole-number part: 2.7 becomes 2 and -2.7 becomes -2 - always towards
zero.

This is "the whole part" in the everyday sense, and the same as
converting a float to an integer in most programming languages. Each
element is processed on its own, so the output has the shape of the
input.

TRUNC AND FLOOR DIFFER FOR NEGATIVES:
For positive values trunc and floor agree. For negative values trunc
goes UP towards zero (-2.7 becomes -2), floor goes down (-3). So trunc
is symmetric about zero: -x gives exactly minus what x gives. Around
zero this makes one step twice as wide as the others - every value from
-0.99 to 0.99 comes out as 0 - which is easy to see in the demo.

SYNTAX:
t.trunc
It takes no arguments and has no parameters.

INPUTS and PARAMETERS:

""" + ROUND_COMMON_IN + """

OUTPUTS:

output:
The truncated tensor, the same shape and type as the input - still
floats, just with whole-number values. An integer tensor comes back
unchanged.

READING THE DEMO:
The ramp climbs straight from -2.5 to 2.5; t.trunc turns it into a
staircase that sits UNDER the ramp on the right and ABOVE it on the
left - always between the ramp and zero - with a double-width step at 0
in the middle. The message shows the same sizes either side of zero
giving the same sizes back.

RELATED:
t.frac keeps exactly the part t.trunc throws away: the two add up to
the original value. t.floor always goes down, t.ceil always up, t.round
to the nearest. trunc does the same for numbers and NumPy arrays
(np.trunc shares its page)."""

demo, links = round_demo('t.trunc', {},
                         ['t.frac', 't.floor', 't.ceil', 't.round', 'trunc'],
                         '2.7 2.3 -2.3 -2.7 -0.5', 'towards zero: 2 2 -2 -2 -0',
                         note='every value, its fraction dropped')
page('t.trunc', 't.trunc - drop the fraction', body, demo, links)

# -------------------------------------------------------------------- t.frac
body = """Keeps only the fractional part of every value in a tensor - what is
left after the whole number is taken away: 2.7 gives 0.7, 12.25 gives
0.25.

Feed it a value that climbs steadily and the output is a sawtooth: it
rises from 0 towards 1, drops back to 0 at every whole number, and rises
again. That is how to make a repeating phase out of a counter or a
clock, or a position within a cell out of a position in space. Each
element is processed on its own, so the output has the shape of the
input.

THE SIGN STAYS:
t.frac is the value minus t.trunc of it, so the fraction of a negative
value is negative: -2.3 gives -0.3, not 0.7. The sawtooth below zero
therefore runs from 0 down towards -1. If you want a phase that is
always between 0 and 1, use the value minus t.floor of it instead (or
the mod node by 1, for numbers and NumPy arrays).

SYNTAX:
t.frac
It takes no arguments and has no parameters.

INPUTS and PARAMETERS:

tensor in:
The tensor. A list, a NumPy array or a single number is turned into a
float32 tensor first. Every input is processed and sent - this is the
only inlet that makes the node send. It must be a floating-point tensor:
an integer tensor raises an error and sends nothing.

OUTPUTS:

output:
The fractional parts, the same shape as the input, each between -1 and
1 with the sign of the value it came from.

READING THE DEMO:
The ramp climbs straight from -2.5 to 2.5; t.frac turns it into a
sawtooth. Right of zero each tooth climbs from 0 towards 1 and drops
back at every whole number. Left of zero the teeth hang BELOW the line,
from 0 down towards -1 - the sign kept. The message shows the same
fraction coming out with opposite signs.

RELATED:
t.trunc is the other half: the whole part. t.trunc plus t.frac is the
original value. t.floor and t.round are the other ways to a whole
number. mod 1 gives an always-positive fraction for numbers and NumPy
arrays."""

demo, links = round_demo('t.frac', {},
                         ['t.trunc', 't.floor', 't.round', 'mod', 'trunc'],
                         '2.7 12.25 0.5 -0.5 -2.3', 'the fraction, sign kept: 0.7 0.25 0.5 -0.5 -0.3',
                         plot_range=(-3.0, 3.0), note='every value, only its fraction kept')
page('t.frac', 't.frac - only the fractional part', body, demo, links)


# ============================================================== comparisons
def compare_demo(node, see, mask_note, start=5):
    """metro -> t.linspace 0 11 12 (the gradient, a strip with its numbers)
    -> the comparison against a slider threshold -> the mask, a strip in
    black and white directly under it."""
    demo = [
        {'key': 'lb', 'init': 'load_bang', 'pos': (330, 62), 'w': 88, 'h': 46},
        {'key': 'tt', 'init': 't %d' % start, 'pos': (450, 62), 'w': 40, 'h': 46},
        {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
         'props': METRO(100)},
        {'key': 'c0', 'comment': True, 'text': 're-sends ten times a second',
         'pos': (30, 140)},
        {'key': 'gr', 'init': 't.linspace 0 11 12', 'pos': (30, 180), 'w': 200, 'h': 150,
         'props': {'start': 0.0, 'stop': 11.0, 'steps': 12}},
        {'key': 'sl', 'init': 'slider 11', 'pos': (330, 180), 'w': 200, 'h': 42,
         'props': {'min': 0, 'max': 11, 'format': '%d', 'width': 180}},
        {'key': 'cs', 'comment': True, 'text': 'the threshold - drag it\n(starts at %d)' % start,
         'pos': (330, 230)},
        {'key': 'hg', 'init': 'heat_map', 'pos': (30, 350), 'w': 520, 'h': 90,
         'props': HM(504, 50, 1, 0.0, 11.0, fmt='%.0f')},
        {'key': 'cg', 'comment': True, 'text': 'tensor a: the whole numbers 0 to 11, a gradient',
         'pos': (30, 450)},
        {'key': 'nd', 'init': node, 'pos': (30, 490), 'w': 190, 'h': 90},
        {'key': 'hm', 'init': 'heat_map', 'pos': (30, 600), 'w': 520, 'h': 90,
         'props': HM(504, 50, 1, 0.0, 1.0, fmt='%.0f', color='hot')},
        {'key': 'cm', 'comment': True, 'text': mask_note, 'pos': (30, 700)},
    ] + see_also(790, see)
    links = [('lb', 'out', 'tt', ''), ('tt', '', 'sl', '', 0),
             ('sl', 'int out', 'nd', 'tensor b in'),
             ('met', '', 'gr', '###input'),
             ('gr', 'linspace out', 'hg', 'y'),
             ('gr', 'linspace out', 'nd', 'tensor a in'),
             ('nd', 'tensor result', 'hm', 'y')]
    return demo, links


def compare_body(node, words, symbol, example, reading, related, extra=''):
    return f"""Compares two tensors element by element and gives back a tensor of
true and false: true wherever a is {words} b.

The answer is not one yes or no but a whole tensor of them, the same
shape as the inputs, holding the outcome at every position. That tensor
is a MASK - the way to say "these elements, not those" to everything that
follows.
{extra}
WHAT TO DO WITH A MASK:
Multiply by it and everything that failed becomes zero. t.masked_select
takes out just the values where it is true. t.count_nonzero (or t.sum)
counts how many passed. To get the PROPORTION that passed, put t.to
float32 in front of t.mean: torch will not average a bool tensor (t.mean
on it raises 'could not infer output dtype'). Summing and multiplying
work on it as it is.

SYNTAX:
{node}
It takes no arguments.

EXAMPLE:
{example}

INPUTS and PARAMETERS:

tensor a in:
The tensor to test (a list or number is turned into a float32 tensor).
Receiving it makes the node compare and send.

tensor b in:
What to test against: another tensor, or a single number used for every
element. It is only stored when it arrives, so send it FIRST; until it
has received something the node sends nothing. It is converted to tensor
a's type, so against an integer tensor a threshold of 4.5 becomes 4.

The shapes must broadcast: the same shape, a single number, or shapes
that differ only where one has length 1 - so a column of thresholds
tests each row against its own.

OUTPUTS:

tensor result:
A bool tensor, true where a {symbol} b, in the broadcast shape.

READING THE DEMO:
{reading}

RELATED:
{related}"""


CMP = [
    ('t.gt', 'greater than', '>', 't.gt',
     """The top strip is tensor a: the whole numbers 0 to 11. The strip under
it is the mask - white (1) where the number is greater than the slider,
black (0) elsewhere. At 5 the cells 6 to 11 are white; the cell holding 5
itself is black, because 5 is not greater than 5 (compare t.ge). Drag the
slider and the edge moves.""",
     """t.ge also counts equality as true. t.lt is the other direction. > does
the same for numbers and NumPy arrays, with a choice of bool, int or
float output. np.where uses a mask to choose between two arrays.""",
     ['t.ge', 't.lt', 't.le', 'greater', 't.masked_select', 'np.where'],
     'white where a > threshold'),
    ('t.ge', 'greater than or equal to', '>=', 't.ge',
     """The top strip is tensor a: the whole numbers 0 to 11. The strip under
it is the mask - white (1) where the number is at least the slider, black
(0) elsewhere. At 5 the cells 5 to 11 are white: the cell holding 5 is
included, which is the one difference from t.gt. Drag the slider and the
edge moves.""",
     """t.gt leaves equality out. t.le is the other direction. >= does the same
for numbers and NumPy arrays. np.where uses a mask to choose between two
arrays.""",
     ['t.gt', 't.le', 't.lt', 'greater_equal', 't.masked_select', 'np.where'],
     'white where a >= threshold'),
    ('t.lt', 'less than', '<', 't.lt',
     """The top strip is tensor a: the whole numbers 0 to 11. The strip under
it is the mask - white (1) where the number is less than the slider,
black (0) elsewhere. At 5 the cells 0 to 4 are white; the cell holding 5
itself is black, because 5 is not less than 5 (compare t.le). Drag the
slider and the edge moves.""",
     """t.le also counts equality as true. t.gt is the other direction. < does
the same for numbers and NumPy arrays. np.where uses a mask to choose
between two arrays.""",
     ['t.le', 't.gt', 't.ge', 'less', 't.masked_select', 'np.where'],
     'white where a < threshold'),
    ('t.le', 'less than or equal to', '<=', 't.le',
     """The top strip is tensor a: the whole numbers 0 to 11. The strip under
it is the mask - white (1) where the number is at most the slider, black
(0) elsewhere. At 5 the cells 0 to 5 are white: the cell holding 5 is
included, which is the one difference from t.lt. Drag the slider and the
edge moves.""",
     """t.lt leaves equality out. t.ge is the other direction. <= does the same
for numbers and NumPy arrays. np.where uses a mask to choose between two
arrays.""",
     ['t.lt', 't.ge', 't.gt', 'less_equal', 't.masked_select', 'np.where'],
     'white where a <= threshold'),
    ('t.eq', 'equal to', '==', 't.eq',
     """The top strip is tensor a: the whole numbers 0 to 11. The strip under
it is the mask - white (1) only where the number equals the slider. At 5
a single cell is lit, the one holding 5. Drag the slider and the lit cell
moves along. The gradient holds exact whole numbers, which is why
equality works here.""",
     """t.ne is the exact opposite. t.ge and t.le include equality in a range.
== does the same for numbers and NumPy arrays. np.where uses a mask to
choose between two arrays.""",
     ['t.ne', 't.ge', 't.le', 'equal', 't.masked_select', 'np.where'],
     'white only where a == threshold'),
    ('t.ne', 'not equal to', '!=', 't.ne',
     """The top strip is tensor a: the whole numbers 0 to 11. The strip under
it is the mask - white (1) wherever the number differs from the slider.
At 5 every cell is white except the one holding 5. Drag the slider and
the dark cell moves along. The gradient holds exact whole numbers, which
is why equality is meaningful here.""",
     """t.eq is the exact opposite. t.gt and t.lt split a range at a value.
!= does the same for numbers and NumPy arrays. np.where uses a mask to
choose between two arrays.""",
     ['t.eq', 't.gt', 't.lt', 'not_equal', 't.masked_select', 'np.where'],
     'white everywhere except where a == threshold'),
]

FLOAT_EQ = """
COMPARING FLOATS FOR EQUALITY:
This tests EXACT equality, and two floats that ought to be the same after
arithmetic very often are not - 0.1 + 0.2 is not exactly 0.3. If it
answers the opposite of what you are sure is right, that is usually why.
Compare whole numbers (t.round first), or test the size of the
difference against a small tolerance with t.lt instead.
"""

for node, words, symbol, example, reading, related, see, note in CMP:
    extra = FLOAT_EQ if node in ('t.eq', 't.ne') else ''
    body = compare_body(node, words, symbol, example, reading, related, extra)
    demo, links = compare_demo(node, see, note)
    page(node, f'{node} - true where a is {words} b, element by element',
         body, demo, links)


# ============================================================= gcd and lcm
def gcd_demo(node, out, see, table_max, msg_note, table_note):
    demo = [
        {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
        {'key': 'lb', 'init': 'load_bang', 'pos': (150, 62), 'w': 88, 'h': 46},
        {'key': 'c0', 'comment': True, 'text': 'click, or open the page, to send both',
         'pos': (270, 67)},
        {'key': 'mb', 'init': 'message', 'pos': (330, 130), 'w': 220, 'h': 42,
         'props': {'text in': '8 12 21 0', 'font size': '24'}},
        {'key': 'ma', 'init': 'message', 'pos': (30, 130), 'w': 220, 'h': 42,
         'props': {'text in': '12 18 7 9', 'font size': '24'}},
        {'key': 'cb', 'comment': True, 'text': 'b (sent first) and a', 'pos': (330, 180)},
        {'key': 'n1', 'init': node, 'pos': (30, 220), 'w': 200, 'h': 80},
        {'key': 'l1', 'init': 'list', 'pos': (30, 320), 'w': 260, 'h': 42, 'props': LIST},
        {'key': 'c1', 'comment': True, 'text': msg_note, 'pos': (30, 372)},
        {'key': 'mc', 'init': 'p make_counts', 'pos': (30, 430), 'w': 190, 'h': 70},
        {'key': 'cc', 'comment': True,
         'text': '1 to 12 as a row (to b) and as a column (to a)', 'pos': (240, 445)},
        {'key': 'n2', 'init': node, 'pos': (30, 520), 'w': 200, 'h': 80},
        {'key': 'hm', 'init': 'heat_map', 'pos': (30, 620), 'w': 520, 'h': 330,
         'props': HM(504, 288, 12, 0.0, float(table_max), fmt='%.0f')},
        {'key': 'ct', 'comment': True, 'text': table_note, 'pos': (30, 960)},
    ] + see_also(1050, see)
    links = [('btn', '', 'mb', ''), ('btn', '', 'ma', ''), ('btn', '', 'mc', 'bang'),
             ('lb', 'out', 'mb', ''), ('lb', 'out', 'ma', ''), ('lb', 'out', 'mc', 'bang'),
             ('mb', 'message out', 'n1', 'tensor b in'),
             ('ma', 'message out', 'n1', 'tensor a in'),
             ('n1', out, 'l1', ''),
             ('mc', 'row', 'n2', 'tensor b in'),
             ('mc', 'column', 'n2', 'tensor a in'),
             ('n2', out, 'hm', 'y')]
    return demo, links


GCD_INPUTS = """INPUTS and PARAMETERS:

tensor a in:
The first tensor, list or number. Receiving it does the work.

tensor b in:
The second tensor, list or a single number used for every element. It is
only stored when it arrives, so send it FIRST; until it has received
something the node sends nothing.

The shapes must broadcast: the same shape, a single number, or shapes
that differ only where one has length 1. The demo's table is exactly
that - a column of 12 against a row of 12 gives all 144 pairs.

WHOLE NUMBERS ONLY:
This is number theory, not arithmetic. Both inputs are turned into whole
numbers (64-bit integers) before the work is done, so lists and float
tensors can go straight in. A value with a fraction is rounded to the
NEAREST whole number first: 2.6 counts as 3 and 9.4 as 9, so a value
like 11.9999 that should have been 12 does not quietly become 11. For a
different rule put t.floor or t.ceil in front."""

body = """Finds, element by element, the greatest common divisor of two
tensors of whole numbers: the largest number that divides both exactly.

The gcd of 12 and 8 is 4 - the biggest step that lands on both. In a
patch that is often rhythmic: the gcd of two cycle lengths is the
largest pulse both are built from, and dividing each by it reduces a
ratio to its simplest form (12:8 is 3:2).

SYNTAX:
t.gcd
It takes no arguments.

""" + GCD_INPUTS + """

ZERO:
The gcd of a number and 0 is the number itself (every number divides 0),
and the gcd of 0 and 0 is 0.

OUTPUTS:

gcd tensor out:
The greatest common divisors, as 64-bit integers, in the broadcast shape.

READING THE DEMO:
Top: four pairs at once. 12 and 8 give 4, 18 and 12 give 6, 7 and 21
give 7, 9 and 0 give 9.

Bottom: every pair from 1 to 12. Row r, column c holds the gcd of r+1
and c+1 (the top-left cell is 1 and 1). The diagonal is bright - a
number shares everything with itself - while the first row and column
are all 1, and so are the rows and columns of the primes 5, 7 and 11
except where they meet themselves. The bright lattice of 2s, 3s, 4s and
6s is the numbers that share factors.

RELATED:
t.lcm is the other half: the smallest number both divide into. For any
pair, gcd times lcm is the pair multiplied together. t.round and t.floor
make whole numbers from fractions."""

demo, links = gcd_demo('t.gcd', 'gcd tensor out',
                       ['t.lcm', 't.round', 't.floor', 't.ceil'], 12,
                       'gcd: 4 6 7 9',
                       'gcd of every pair from 1 to 12 - the diagonal is the numbers themselves')
page('t.gcd', 't.gcd - the greatest common divisor', body, demo, links,
     subpatch=COUNTS_SUB)

body = """Finds, element by element, the least common multiple of two tensors
of whole numbers: the smallest number both divide into exactly.

The lcm of 4 and 6 is 12. In a patch it answers "when do these line up
again?": two loops of 4 and 6 beats coincide every 12 beats, so the lcm
of two cycle lengths is the length of the combined pattern before it
repeats.

SYNTAX:
t.lcm
It takes no arguments.

""" + GCD_INPUTS + """

ZERO:
The lcm of anything with 0 is 0 (0 is the only multiple of 0).

OUTPUTS:

lcm tensor out:
The least common multiples, as 64-bit integers, in the broadcast shape.
They grow quickly - the lcm of 11 and 12 is already 132.

READING THE DEMO:
Top: four pairs at once. 12 and 8 give 24, 18 and 12 give 36, 7 and 21
give 21, 9 and 0 give 0.

Bottom: every pair from 1 to 12. Row r, column c holds the lcm of r+1
and c+1 (the top-left cell is 1 and 1). Pairs with nothing in common
multiply out in full and glow brightest towards the bottom right (11 and
12 give 132); pairs that share factors stay dark, and the diagonal - a
number with itself - is only the number.

RELATED:
t.gcd is the other half: the largest number dividing both. For any pair,
gcd times lcm is the pair multiplied together. t.round and t.floor make
whole numbers from fractions."""

demo, links = gcd_demo('t.lcm', 'lcm tensor out',
                       ['t.gcd', 't.round', 't.floor', 't.ceil'], 132,
                       'lcm: 24 36 21 0',
                       'lcm of every pair from 1 to 12 - pairs sharing no factor are brightest')
page('t.lcm', 't.lcm - the least common multiple', body, demo, links,
     subpatch=COUNTS_SUB)


# ================================================================ t.copysign
body = """Takes the size of each value from one tensor and its sign from
another: the result is as big as 'tensor in' and points the way 'sign
tensor' points.

Its everyday use is applying a curve to a signal that swings both ways.
Take the size (abs), put it through the curve, and copy the original
sign back. Without that last step a curve applied to a two-sided signal
folds the negative half over, and everything comes out positive. The
demo does exactly this with a power curve.

NEGATIVE ZERO COUNTS AS NEGATIVE:
The sign is read from the sign bit, so a sign value of -0.0 makes the
result negative and 0.0 makes it positive. That matters when the sign
carries a direction that has momentarily stopped.

SYNTAX:
t.copysign
It takes no arguments.

INPUTS and PARAMETERS:

tensor in:
Where the size comes from (a list or number is turned into a float32
tensor). Receiving it does the work; its own sign is ignored.

sign tensor:
Where the sign comes from - a tensor that broadcasts against tensor in,
or a single number used for every element. It is converted to tensor
in's type. It is only stored when it arrives, so send it FIRST; until it
has received something the node sends nothing. If both come from the
same source, connect the sign cord first, or every result carries the
signs from the time before.

OUTPUTS:

tensor with copied sign:
The sizes of tensor in with the signs of sign tensor, in the broadcast
shape.

READING THE DEMO:
make_sine sends a sine (the thin original line on the plot). abs takes
its size and ^ 2 squares it: a curve that should make small values
smaller and leave the peaks at 1. Alone, that folds the troughs up - the
squared line has every hump above zero. t.copysign gives each value back
the sign of the original sine, so the result is a squared sine that
still swings below zero. Drag the exponent on ^ (0.5 rounds the wave
out, 3 sharpens it); the shape changes and the signs stay right.

RELATED:
abs gives the size and throws the sign away. ^ (power) is the curve in
the demo. t.maximum and t.clamp limit values instead of reshaping them.
t.ceil with t.copysign rounds away from zero."""

demo = [
    {'key': 'mk', 'init': 'p make_sine', 'pos': (30, 62), 'w': 170, 'h': 70},
    {'key': 'c0', 'comment': True,
     'text': 'a sine as a tensor of 128 values, sent 10 times a second',
     'pos': (30, 150)},
    {'key': 'ab', 'init': 'abs', 'pos': (30, 200), 'w': 120, 'h': 70},
    {'key': 'ca', 'comment': True, 'text': 'its size, the sign thrown away', 'pos': (170, 215)},
    {'key': 'pw', 'init': '^ 2.0', 'pos': (30, 290), 'w': 140, 'h': 70,
     'props': {'': 2.0}},
    {'key': 'cw', 'comment': True, 'text': 'a curve: drag the exponent', 'pos': (190, 305)},
    {'key': 'cp', 'init': 't.copysign', 'pos': (30, 380), 'w': 200, 'h': 90},
    {'key': 'cc', 'comment': True, 'text': 'size from the curve, sign from the sine',
     'pos': (250, 395)},
    {'key': 'pk', 'init': 'pack 3', 'pos': (30, 500), 'w': 130, 'h': 120},
    {'key': 'cp2', 'comment': True, 'text': 'result (in 1), curve alone (in 2), sine (in 3)',
     'pos': (180, 520)},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 640), 'w': 308, 'h': 178,
     'props': MULTI_PLOT(-1.2, 1.2, 128)},
    {'key': 'c2', 'comment': True,
     'text': 'the curve alone folds every trough above zero;\nwith the sign copied back, it swings both ways',
     'pos': (30, 830)},
] + see_also(920, ['abs', 'power', 't.maximum', 't.clamp', 't.ceil'])
# the sign cord leaves the wave first, so the signs are stored before the
# curve arrives; the curve reaches pack's cold inlet before the result
links = [('mk', 'sine', 'cp', 'sign tensor'),
         ('mk', 'sine', 'pk', 'in 3'),
         ('mk', 'sine', 'ab', 'in'),
         ('ab', 'result', 'pw', 'in'),
         ('pw', 'result', 'pk', 'in 2'),
         ('pw', 'result', 'cp', 'tensor in'),
         ('cp', 'tensor with copied sign', 'pk', 'in 1'),
         ('pk', 'out', 'pl', 'y')]
page('t.copysign', 't.copysign - size from one tensor, sign from another', body,
     demo, links, subpatch=SINE_SUB)


# ================================================================== complex
body = """Builds a complex tensor from two real tensors: one holding the real
parts, one the imaginary parts.

A complex number is two numbers travelling as one - a real part and an
imaginary part, which you can also read as a point on a plane, or as a
size and an angle. You meet them wherever something has both an amount
and a phase, which in practice means anywhere near a Fourier transform:
t.fft and t.rfft send complex tensors, and t.ifft and t.irfft take them
back. Splitting a spectrum with t.real and t.imag, working on the parts,
and rebuilding it here is how you edit one.

SYNTAX:
t.complex
It takes no arguments.

INPUTS and PARAMETERS:

real tensor in:
The real parts (a list or number is turned into a float32 tensor). It
must be a float tensor - float16, float32 or float64.

imag tensor in:
The imaginary parts. It is converted to the real tensor's type.

BOTH inlets make the node send, so when both change it sends twice -
once with the old other half. Send the imaginary part first and the
second send is the right one. It sends nothing until both have arrived,
and nothing at all if the two shapes differ: there is no broadcasting,
so a single number cannot stand for every imaginary part.

OUTPUTS:

complex tensor out:
The complex tensor, the same shape as the inputs: complex64 from
float32 parts, complex128 from float64.

READING THE DEMO:
Top: cos and sin of one turning angle go in as the real and imaginary
parts, so the complex number goes round a circle once every 4 seconds.
The trigger sends the angle to sin first, so the imaginary part is
already stored when the real part arrives. The list shows the number as
real+imaginary j; abs gives its size, which stays 1.0 however it turns.

Bottom: click the button for four numbers at once. The imaginary parts
go first. 3 and 4 make 3+4j, whose size is 5 - the long side of a 3, 4,
5 triangle.

RELATED:
t.real and t.imag take the two parts out again. abs gives the size of
each complex value. t.fft and t.rfft are where complex tensors usually
come from."""

demo = [
    {'key': 'sg', 'init': 'signal 4.0 saw', 'pos': (30, 62), 'w': 129, 'h': 78,
     'props': SIG('saw', 4.0, 360.0, False)},
    {'key': 'cg', 'comment': True, 'text': 'an angle, 0 to 360 degrees every 4 s',
     'pos': (180, 72)},
    {'key': 'tr', 'init': 't float float', 'pos': (30, 160), 'w': 150, 'h': 46},
    {'key': 'ct', 'comment': True, 'text': 'right first: the imaginary part goes first',
     'pos': (200, 165)},
    {'key': 'co', 'init': 'cos', 'pos': (30, 225), 'w': 120, 'h': 70,
     'props': {'degrees': True}},
    {'key': 'si', 'init': 'sin', 'pos': (220, 225), 'w': 120, 'h': 70,
     'props': {'degrees': True}},
    {'key': 'fr', 'init': 'float', 'pos': (30, 310), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'fi', 'init': 'float', 'pos': (220, 310), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c0', 'comment': True, 'text': 'real part (cos) and imaginary part (sin)',
     'pos': (30, 362)},
    {'key': 'cx', 'init': 't.complex', 'pos': (30, 400), 'w': 200, 'h': 90},
    {'key': 'l1', 'init': 'list', 'pos': (30, 510), 'w': 260, 'h': 42, 'props': LIST},
    {'key': 'c1', 'comment': True, 'text': 'the complex number, going round', 'pos': (30, 562)},
    {'key': 'ab', 'init': 'abs', 'pos': (330, 510), 'w': 120, 'h': 70},
    {'key': 'l2', 'init': 'list', 'pos': (330, 600), 'w': 200, 'h': 42, 'props': LIST},
    {'key': 'c2', 'comment': True, 'text': 'its size: always 1', 'pos': (330, 652)},
    {'key': 'btn', 'init': 'button', 'pos': (30, 710), 'w': 88, 'h': 46},
    {'key': 'mi', 'init': 'message', 'pos': (330, 770), 'w': 200, 'h': 42,
     'props': {'text in': '4 1 0 -2', 'font size': '24'}},
    {'key': 'mr', 'init': 'message', 'pos': (30, 770), 'w': 200, 'h': 42,
     'props': {'text in': '3 0 -1 2', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'real parts, and imaginary parts (sent first)',
     'pos': (30, 822)},
    {'key': 'cx2', 'init': 't.complex', 'pos': (30, 860), 'w': 200, 'h': 90},
    {'key': 'l3', 'init': 'list', 'pos': (30, 970), 'w': 360, 'h': 42, 'props': LIST},
    {'key': 'ab2', 'init': 'abs', 'pos': (420, 970), 'w': 120, 'h': 70},
    {'key': 'l4', 'init': 'list', 'pos': (330, 1060), 'w': 260, 'h': 42, 'props': LIST},
    {'key': 'c4', 'comment': True, 'text': '3+4j 1j -1 2-2j, and their sizes: 5 1 1 2.83',
     'pos': (30, 1120)},
] + see_also(1200, ['t.real', 't.imag', 't.fft', 't.rfft', 'abs'])
links = [('sg', '', 'tr', ''),
         ('tr', 'float', 'si', 'in', 1), ('tr', 'float', 'co', 'in', 0),
         ('co', 'out', 'fr', ''), ('si', 'out', 'fi', ''),
         ('si', 'out', 'cx', 'imag tensor in'), ('co', 'out', 'cx', 'real tensor in'),
         ('cx', 'complex tensor out', 'l1', ''),
         ('cx', 'complex tensor out', 'ab', 'in'), ('ab', 'result', 'l2', ''),
         ('btn', '', 'mi', ''), ('btn', '', 'mr', ''),
         ('mi', 'message out', 'cx2', 'imag tensor in'),
         ('mr', 'message out', 'cx2', 'real tensor in'),
         ('cx2', 'complex tensor out', 'l3', ''),
         ('cx2', 'complex tensor out', 'ab2', 'in'), ('ab2', 'result', 'l4', '')]
page('t.complex', 't.complex - a complex tensor from two real ones', body, demo, links)


def part_demo(node, out, see, part_note):
    demo = [
        {'key': 'mt', 'init': 'p make_turning', 'pos': (30, 62), 'w': 190, 'h': 70},
        {'key': 'c0', 'comment': True,
         'text': 'a complex number of size 1 going round once every 4 s',
         'pos': (30, 150)},
        {'key': 'l1', 'init': 'list', 'pos': (300, 190), 'w': 260, 'h': 42, 'props': LIST},
        {'key': 'ab', 'init': 'abs', 'pos': (300, 260), 'w': 120, 'h': 70},
        {'key': 'ca', 'comment': True, 'text': 'its size', 'pos': (440, 275)},
        {'key': 'nd', 'init': node, 'pos': (30, 260), 'w': 160, 'h': 70},
        {'key': 'pk', 'init': 'pack 2', 'pos': (30, 360), 'w': 130, 'h': 100},
        {'key': 'cp', 'comment': True, 'text': '%s (in 1) and the size (in 2)' % part_note,
         'pos': (180, 380)},
        {'key': 'pl', 'init': 'plot', 'pos': (30, 480), 'w': 308, 'h': 178,
         'props': MULTI_PLOT(-1.2, 1.2, 200)},
        {'key': 'c2', 'comment': True,
         'text': 'the size stays at 1 while the %s part swings from -1 to 1' % part_note,
         'pos': (30, 670)},
    ] + see_also(760, see)
    links = [('mt', 'complex', 'l1', ''),
             ('mt', 'complex', 'ab', 'in'),
             ('ab', 'result', 'pk', 'in 2'),
             ('mt', 'complex', 'nd', 'tensor in'),
             ('nd', out, 'pk', 'in 1'),
             ('pk', 'out', 'pl', 'y')]
    return demo, links


PART_TEXT = """Takes the {part} part out of a complex tensor, as an ordinary real
tensor of the same shape.

Complex tensors come out of t.fft and t.rfft, and most of what you want
to do next needs real numbers: plotting, thresholding, feeding anything
that is not complex-aware. t.real and t.imag split the two halves;
t.complex puts them back together.

THE {PART} PART IS NOT THE SIZE:
A complex value's size (its magnitude, what abs gives) is the length of
the two parts taken together. The {part} part alone depends on the
phase: a component can be large and still have a {part} part near zero,
if its angle happens to sit {where}. For "how much of this frequency is
there", use abs on the complex tensor, not t.{short}. The demo shows the
difference: a value of constant size whose {part} part swings.

SYNTAX:
t.{short}
It takes no arguments.

INPUTS and PARAMETERS:

tensor in:
The complex tensor. A tensor that is not complex - including an ordinary
float tensor, a list or a number - sends NOTHING, silently, rather than
an answer of {real_zero}.

OUTPUTS:

{outname}:
The {part} parts, as float32 from complex64 (float64 from complex128),
the same shape as the input.

READING THE DEMO:
make_turning sends one complex number of size 1, going round a circle
once every 4 seconds (the list shows it). The plot draws two lines: abs,
flat at 1, and the {part} part from t.{short}, swinging as a {wave}
between -1 and 1. The size never changed; only the angle did.

RELATED:
{other} takes the other half. t.complex builds a complex tensor from the
two parts. abs gives the size. t.fft and t.rfft send complex tensors."""

for short, part, where, wave, outname, other, real_zero in [
        ('real', 'real', 'near a quarter turn', 'cosine', 'real tensor', 't.imag',
         'the value itself'),
        ('imag', 'imaginary', 'near zero or a half turn', 'sine', 'imaginary tensor',
         't.real', 'zeros')]:
    body = PART_TEXT.format(part=part, PART=part.upper(), where=where, short=short,
                            wave=wave, outname=outname, other=other, real_zero=real_zero)
    see = [other, 't.complex', 'abs', 't.fft', 't.rfft']
    demo, links = part_demo('t.' + short, outname, see, part)
    page('t.' + short, f't.{short} - the {part} part of a complex tensor', body, demo,
         links, subpatch=TURN_SUB)
