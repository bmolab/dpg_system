"""t tensor makers, one page each.

t.linspace, t.logspace, t.arange, t.range, t.zeros, t.ones, t.rand, t.full,
t.eye, t.zeros_like, t.ones_like, t.rand_like (torch_generator_nodes.py).
These were two family pages (t.linspace, t.zeros); each now has its own page
whose demo SHOWS the tensor, with a SEE ALSO row of help_links under it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import INT, FLT


def HM(width, height, count, miny=0.0, maxy=1.0, fmt='%.2f', color='viridis'):
    """Property block for a heat_map that draws a whole array as one picture."""
    return {'color': color, 'width': width, 'height': height, 'sample count': count,
            'min y': miny, 'max y': maxy, 'update_mode': 'heat_map',
            'number format': fmt}


LIST = {'text in': '', 'font size': '24'}


def see_also(y, stems, x=30):
    """A SEE ALSO row: one help_link per related page, left to right."""
    row = []
    for s in stems:
        w = int(len(s) * 7.7) + 30
        row.append({'key': 'sa_' + s, 'init': 'help_link ' + s, 'pos': (x, y),
                    'w': w, 'h': 30})
        x += w + 14
    row.append({'key': 'sa_c', 'comment': True, 'text': 'SEE ALSO - click to open',
                'pos': (x, y)})
    return row


# Shared prose for the three settings every tensor maker carries.
DTYPE_DEVICE_GRAD = """dtype (menu):
The number type. float32 is the default and what most torch operations
want. The menu also offers float16, int64, uint8, bool, complex64 and
complex128; 'float' is the same as float32, and each name appears a
second time with a 'torch.' prefix, which means the same thing. There
is no 64-bit float here.

device (menu):
Where the tensor is made: cpu (the default), and mps or cuda when the
machine has them. Tensors on different devices cannot be combined, so
keep a chain on one device.

requires_grad (checkbox):
Records operations for automatic differentiation. Leave it off unless
you are training something. Only float and complex tensors can carry it:
ticked with an integer or bool dtype, the node raises an error and sends
nothing.

A dtype or device name can also be typed as an argument, after the
numbers - for example %s."""


# ------------------------------------------------------------------ t.linspace
body = """Makes a 1-D tensor of evenly spaced values running from start to stop.

Reach for it whenever you need a regular series rather than a measured
one: an x axis to evaluate a function across, evenly spaced positions,
the steps of a fade, or a smooth path from one state to another.

BOTH ENDS ARE INCLUDED:
steps is the number of VALUES, so there is one gap fewer than there are
values. With start 0, stop 1 and 9 steps the gap is (1 - 0) / 8 = 0.125,
and you get 0, 0.125, 0.25 ... 0.875, 1. In general the gap is
(stop - start) / (steps - 1). Start can be larger than stop: the values
then count down.

If you know the SPACING you want rather than the count, use t.arange
instead.

SYNTAX:
t.linspace <start: float> <stop: float> <steps: int> <dtype> <device>
Defaults: 0.0, 1.0 and 50, float32 on cpu.

EXAMPLE:
t.linspace 0 1 9

INPUTS and PARAMETERS:

in (button):
Click it, or send it anything, to produce the tensor. This is the only
thing that makes the node send. Changing start, stop or steps does NOT
send by itself - in the demo a metro clicks the node 20 times a second,
so the strip follows your drags.

start / stop (drag floats):
The first and last values.

steps (drag int):
How many values. 1 gives just start; 0 sends an empty tensor (the
heat_map keeps its last picture); a negative count is an error and
nothing is sent.

""" + DTYPE_DEVICE_GRAD % "t.linspace 0 1 9 int64" + """
An integer dtype cuts each value down to a whole number, so t.linspace
0 1 9 as int64 is eight 0s and a single 1.

OUTPUTS:

linspace out:
The 1-D tensor, steps values long.

READING THE DEMO:
The heat_map shows the ramp as a strip of cells, start on the left and
stop on the right, each coloured from the heat_map's min y (dark purple)
to its max y (yellow) - here 0 and 1. Lower stop to 0.5 and the ramp only
gets half way to yellow; swap start and stop and it runs the other way;
drag steps and the strip divides more finely. The list shows the same
values as numbers.

The heat_map's min y and max y are the scale, not the data: values
outside them saturate. If you change the range a lot, change them too.

RELATED:
np.linspace is the NumPy version. t.logspace spaces the values evenly in
the logarithm instead. t.arange and t.range take a step size rather than
a count. t.rand gives values anywhere in a range instead of evenly
spaced."""

demo = [
    {'key': 'met', 'init': 'metro 50', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 50.0, 'units': 'milliseconds'}},
    {'key': 'c0', 'comment': True, 'text': 'clicks t.linspace 20 times a second',
     'pos': (30, 70)},
    {'key': 'ls', 'init': 't.linspace 0 1 9', 'pos': (30, 150), 'w': 220, 'h': 210,
     'props': {'start': 0.0, 'stop': 1.0, 'steps': 9}},
    {'key': 'c1', 'comment': True, 'text': 'drag start, stop or steps on the node',
     'pos': (30, 155)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 380), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 'one cell per value, start on the left:',
     'pos': (30, 385)},
    {'key': 'c3', 'comment': True, 'text': '0 is dark purple, 1 is yellow',
     'pos': (30, 407)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 510), 'w': 468, 'h': 42, 'props': LIST},
    {'key': 'c4', 'comment': True, 'text': 'the same values as numbers',
     'pos': (30, 515)},
] + see_also(590, ['np.linspace', 't.logspace', 't.arange', 't.range', 't.rand'])
links = [('met', '', 'ls', '###input'), ('ls', 'linspace out', 'hm', 'y'),
         ('ls', 'linspace out', 'l1', '')]
print(build('t.linspace', 't.linspace - evenly spaced values from start to stop',
            body, demo, links, demo_width=520, text_width=760, text_height=820))

# ------------------------------------------------------------------ t.logspace
body = """Makes a 1-D tensor whose values are spaced evenly in the logarithm.

Each step MULTIPLIES by the same amount instead of adding the same
amount. That is the right spacing for anything that spans several orders
of magnitude or that we perceive in ratios - frequency, loudness,
brightness, a decay time, a learning rate to sweep. Ten evenly spaced
frequencies from 20 to 20000 Hz put nearly all of them in the top octave;
ten logarithmically spaced ones put the same number in every decade.

START AND STOP ARE POWERS OF TEN:
The values run from 10 to the power start up to 10 to the power stop.
t.logspace 0 2 9 runs from 1 (10 to the 0) to 100 (10 to the 2): 1,
1.78, 3.16, 5.62, 10, 17.8, 31.6, 56.2, 100. Every value is about 1.78
times the one before, and 10 sits exactly in the middle. Both ends are
included, and steps is the number of values, as in t.linspace. The base
is always 10 - to go from 20 to 20000 use start 1.301 (the logarithm of
20) and stop 4.301.

SYNTAX:
t.logspace <start: float> <stop: float> <steps: int> <dtype> <device>
Defaults: 0.0, 1.0 and 50, float32 on cpu - that is 50 values from 1 to
10.

EXAMPLE:
t.logspace 0 2 9

INPUTS and PARAMETERS:

in (button):
Click it, or send it anything, to produce the tensor. Changing start,
stop or steps does not send by itself - the metro in the demo clicks the
node 20 times a second.

start / stop (drag floats):
The powers of ten at the two ends.

steps (drag int):
How many values. 0 sends an empty tensor; a negative count is an error.

""" + DTYPE_DEVICE_GRAD % "t.logspace 0 2 9 float16" + """
OUTPUTS:

logspace out:
The 1-D tensor, steps values long.

READING THE DEMO:
Both strips cover 1 to 100 in 9 values on the same colour scale (0 dark
purple, 100 yellow). The top one is t.logspace: it stays dark for most of
its length and brightens only at the end, because the values crowd
together at the bottom of the range and spread out at the top. The
bottom one is t.linspace 1 100 9 for comparison - a straight ramp. The
list shows t.logspace's values. Drag stop on t.logspace down to 1 and the
top strip ends at 10, mostly dark purple on this scale.

RELATED:
t.linspace is the even-difference version. t.arange and t.range take a
step size. np.linspace is the NumPy linspace; there is no np.logspace
node."""

demo = [
    {'key': 'met', 'init': 'metro 50', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 50.0, 'units': 'milliseconds'}},
    {'key': 'c0', 'comment': True, 'text': 'clicks both nodes 20 times a second',
     'pos': (30, 70)},
    {'key': 'lg', 'init': 't.logspace 0 2 9', 'pos': (30, 150), 'w': 220, 'h': 210,
     'props': {'start': 0.0, 'stop': 2.0, 'steps': 9}},
    {'key': 'c1', 'comment': True, 'text': '1 to 100: start and stop are powers of ten',
     'pos': (30, 155)},
    {'key': 'ls', 'init': 't.linspace 1 100 9', 'pos': (280, 150), 'w': 220, 'h': 210,
     'props': {'start': 1.0, 'stop': 100.0, 'steps': 9}},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 380), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, 0.0, 100.0, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 't.logspace: crowded low, spread out high',
     'pos': (30, 385)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 510), 'w': 468, 'h': 42, 'props': LIST},
    {'key': 'c3', 'comment': True, 'text': "t.logspace's values: each about 1.78 x the last",
     'pos': (30, 515)},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (30, 570), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, 0.0, 100.0, fmt='none')},
    {'key': 'c4', 'comment': True, 'text': 't.linspace 1 100 9 on the same scale: a straight ramp',
     'pos': (30, 575)},
] + see_also(720, ['t.linspace', 't.arange', 't.range', 'np.linspace'])
links = [('met', '', 'lg', '###input'), ('met', '', 'ls', '###input'),
         ('lg', 'logspace out', 'hm', 'y'), ('lg', 'logspace out', 'l1', ''),
         ('ls', 'linspace out', 'hm2', 'y')]
print(build('t.logspace', 't.logspace - values spaced by equal ratios',
            body, demo, links, demo_width=520, text_width=760, text_height=880))

# -------------------------------------------------------------------- t.arange
body = """Makes a 1-D tensor counting from start towards stop in steps of a fixed size.

The stop is NOT included. Use it when you know how far apart the values
should be rather than how many you want - whole-number indices 0, 1, 2
..., every tenth of a second, every 5 degrees.

THE STOP IS LEFT OUT:
t.arange 0 1 0.1 gives TEN values, 0 to 0.9, not eleven. The count is
(stop - start) / step, rounded up. This is the same rule as a Python
range, and the reason t.arange 0 n 1 gives exactly n indices from 0 to
n - 1. If you want the stop as well, use t.range; if you want an exact
count with both ends exact, use t.linspace.

A negative step counts down, but then start must be larger than stop.

SYNTAX:
t.arange <start: float> <stop: float> <step: float> <dtype> <device>
Defaults: 0.0, 1.0 and 0.01 - 100 values, 0 to 0.99.

EXAMPLE:
t.arange 0 1 0.1

INPUTS and PARAMETERS:

in (button):
Click it, or send it anything, to produce the tensor. Changing start,
stop or step does not send by itself - the metro in the demo clicks the
node 20 times a second.

start / stop (drag floats):
Where to begin, and the value it stops short of.

step (drag float):
The distance between values. 0 is an error, and so is a step pointing
away from stop (a positive step with stop below start): either way
nothing is sent.

""" + DTYPE_DEVICE_GRAD % "t.arange 0 16 1 int64" + """
With an integer dtype use whole-number steps: a fractional step there
gives an empty or surprising tensor.

OUTPUTS:

arange out:
The 1-D tensor.

READING THE DEMO:
The strip shows t.arange 0 1 0.1, one cell per value, coloured from 0
(dark purple) to 1 (yellow). It never reaches yellow, because 1 is
left out - the list ends at 0.9 and t.info says [10]. Below, t.range with
the same three numbers does include 1: eleven values. Drag step on
t.arange and watch the count in t.info.

RELATED:
t.range is the same with the stop included. t.linspace takes a count
instead of a step. t.logspace spaces values by ratio. np.linspace is the
NumPy linspace; there is no np.arange node."""

demo = [
    {'key': 'met', 'init': 'metro 50', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 50.0, 'units': 'milliseconds'}},
    {'key': 'c0', 'comment': True, 'text': 'clicks both nodes 20 times a second',
     'pos': (30, 70)},
    {'key': 'ar', 'init': 't.arange 0 1 0.1', 'pos': (30, 150), 'w': 220, 'h': 210,
     'props': {'start': 0.0, 'stop': 1.0, 'step': 0.1}},
    {'key': 'c1', 'comment': True, 'text': 'drag step: the count changes',
     'pos': (30, 155)},
    {'key': 'inf', 'init': 't.info', 'pos': (290, 150), 'w': 150, 'h': 130},
    {'key': 'c1b', 'comment': True, 'text': 'shape [10]: 0 to 0.9', 'pos': (290, 155)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 380), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 'never quite yellow: the stop, 1, is left out',
     'pos': (30, 385)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 510), 'w': 468, 'h': 42, 'props': LIST},
    {'key': 'c3', 'comment': True, 'text': 't.arange 0 1 0.1 as numbers',
     'pos': (30, 515)},
    {'key': 'rg', 'init': 't.range 0 1 0.1', 'pos': (30, 580), 'w': 220, 'h': 210,
     'props': {'start': 0.0, 'stop': 1.0, 'step': 0.1}},
    {'key': 'l2', 'init': 'list', 'pos': (30, 810), 'w': 468, 'h': 42, 'props': LIST},
    {'key': 'c4', 'comment': True, 'text': 'compare t.range: the same numbers, 1 included',
     'pos': (30, 815)},
] + see_also(890, ['t.range', 't.linspace', 't.logspace', 'np.linspace'])
links = [('met', '', 'ar', '###input'), ('met', '', 'rg', '###input'),
         ('ar', 'arange out', 'hm', 'y'), ('ar', 'arange out', 'l1', ''),
         ('ar', 'arange out', 'inf', 'in'), ('rg', 'range out', 'l2', '')]
print(build('t.arange', 't.arange - values a fixed step apart, stop left out',
            body, demo, links, demo_width=520, text_width=760, text_height=860))

# --------------------------------------------------------------------- t.range
body = """Makes a 1-D tensor counting from start to stop in steps of a fixed size, stop included.

It is t.arange with the end kept: t.range 0 1 0.1 gives eleven values,
0, 0.1 ... 0.9, 1. Use it when you think of the series as running from
one value TO another - 0 to 360 degrees in 15 degree steps, 1 to 10 -
and want both ends.

HOW THE STOP IS INCLUDED:
PyTorch dropped its own range function, so this node asks for an
arange whose stop is moved out by half a step. A value that lands on
the stop is then kept, and floating-point rounding cannot lose it. The
side effect: a value can pass stop by LESS than half a step and still
be included. t.range 0 1 0.6 gives 0, 0.6, 1.2 - 1.2 is over the stop
by 0.2, which is under half of 0.6. When stop is a whole number of steps
from start, as it usually is, this never comes up.

A negative step counts down from a larger start.

SYNTAX:
t.range <start: float> <stop: float> <step: float> <dtype> <device>
Defaults: 0.0, 1.0 and 0.01 - 101 values, 0 to 1.

EXAMPLE:
t.range 0 1 0.1

INPUTS and PARAMETERS:

in (button):
Click it, or send it anything, to produce the tensor. Changing start,
stop or step does not send by itself - the metro in the demo clicks the
node 20 times a second.

start / stop (drag floats):
The two ends.

step (drag float):
The distance between values. 0 is an error, and so is a step pointing
away from stop: either way nothing is sent.

""" + DTYPE_DEVICE_GRAD % "t.range 1 10 1 int64" + """
With an integer dtype use whole-number steps.

OUTPUTS:

range out:
The 1-D tensor.

READING THE DEMO:
The strip runs all the way to yellow: the last value is the stop, 1.
t.info says [11]. Below, t.arange with the same three numbers stops at
0.9 - ten values. Drag step to 0.25 and t.range gives five values ending
in 1; t.arange gives four.

RELATED:
t.arange is the same with the stop left out. t.linspace takes a count
instead of a step. t.logspace spaces values by ratio. np.linspace is the
NumPy linspace."""

demo = [
    {'key': 'met', 'init': 'metro 50', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 50.0, 'units': 'milliseconds'}},
    {'key': 'c0', 'comment': True, 'text': 'clicks both nodes 20 times a second',
     'pos': (30, 70)},
    {'key': 'rg', 'init': 't.range 0 1 0.1', 'pos': (30, 150), 'w': 220, 'h': 210,
     'props': {'start': 0.0, 'stop': 1.0, 'step': 0.1}},
    {'key': 'c1', 'comment': True, 'text': 'drag step: the count changes',
     'pos': (30, 155)},
    {'key': 'inf', 'init': 't.info', 'pos': (290, 150), 'w': 150, 'h': 130},
    {'key': 'c1b', 'comment': True, 'text': 'shape [11]: 0 to 1', 'pos': (290, 155)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 380), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 'ends in yellow: the stop, 1, is included',
     'pos': (30, 385)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 510), 'w': 468, 'h': 42, 'props': LIST},
    {'key': 'c3', 'comment': True, 'text': 't.range 0 1 0.1 as numbers',
     'pos': (30, 515)},
    {'key': 'ar', 'init': 't.arange 0 1 0.1', 'pos': (30, 580), 'w': 220, 'h': 210,
     'props': {'start': 0.0, 'stop': 1.0, 'step': 0.1}},
    {'key': 'l2', 'init': 'list', 'pos': (30, 810), 'w': 468, 'h': 42, 'props': LIST},
    {'key': 'c4', 'comment': True, 'text': 'compare t.arange: the same numbers, 1 left out',
     'pos': (30, 815)},
] + see_also(890, ['t.arange', 't.linspace', 't.logspace', 'np.linspace'])
links = [('met', '', 'rg', '###input'), ('met', '', 'ar', '###input'),
         ('rg', 'range out', 'hm', 'y'), ('rg', 'range out', 'l1', ''),
         ('rg', 'range out', 'inf', 'in'), ('ar', 'arange out', 'l2', '')]
print(build('t.range', 't.range - values a fixed step apart, stop included',
            body, demo, links, demo_width=520, text_width=760, text_height=880))

# --------------------------------------------------------------------- t.zeros
body = """Makes a tensor filled with zeros, in the shape you give it.

On its own it is the dullest tensor there is, and that is its use: a
blank to start from. Accumulate into it, add values into part of it,
or hold a correctly shaped place in a chain until real data arrives.

SYNTAX:
t.zeros <dim 0: int> <dim 1: int> ... <dtype> <device>
One whole number per dimension: t.zeros 8 8 is an 8 by 8 tensor,
t.zeros 16 a row of 16. With no numbers you get a 1-element tensor.

EXAMPLE:
t.zeros 8 8

INPUTS and PARAMETERS:

in (button):
The small button at the top left of the node. Click it, or send it
anything, to send the tensor. Nothing else makes the node send; each
click sends a new tensor.

shape (text):
The dimensions, typed as numbers separated by spaces or commas - '8 8',
'[4, 16]' or '2 3 4' for 3-D. Unlike np.zeros, the number of dimensions
can change here without retyping the node. Editing it does not send;
the next click uses it.

""" + DTYPE_DEVICE_GRAD % "t.zeros 8 8 int64" + """
OUTPUTS:

tensor of zeros:
The tensor. A bool dtype gives all false.

READING THE DEMO:
The load_bang clicks t.zeros once when the page opens. The heat_map is
the tensor - every cell 0, one flat colour, each printing its 0. t.info
reports its shape, dtype and device: [8, 8], torch.float32, cpu. Type
'4 8' into shape and click the node's button: the picture becomes 4
rows of 8 and t.info follows. Choose int64 as dtype and click again to
see t.info change.

RELATED:
t.ones is the same with every value 1, t.full with any value you pick.
t.zeros_like takes its shape from another tensor. t.eye puts ones on the
diagonal of a square of zeros. np.zeros is the NumPy version."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'clicks t.zeros once when the page opens',
     'pos': (30, 70)},
    {'key': 'zr', 'init': 't.zeros 8 8', 'pos': (30, 125), 'w': 220, 'h': 160},
    {'key': 'c1', 'comment': True,
     'text': "type another shape, then click the node's\nown button (top left)",
     'pos': (30, 130)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 305), 'w': 280, 'h': 290,
     'props': HM(260, 260, 8, fmt='%.0f')},
    {'key': 'c2', 'comment': True, 'text': 'every cell 0: one flat colour',
     'pos': (30, 310)},
    {'key': 'inf', 'init': 't.info', 'pos': (340, 305), 'w': 160, 'h': 130},
    {'key': 'c3', 'comment': True, 'text': 'shape, dtype, device', 'pos': (340, 310)},
] + see_also(620, ['t.ones', 't.full', 't.zeros_like', 't.eye', 'np.zeros'])
links = [('lb', 'out', 'zr', '###input'), ('zr', 'tensor of zeros', 'hm', 'y'),
         ('zr', 'tensor of zeros', 'inf', 'in')]
print(build('t.zeros', 't.zeros - a tensor of zeros', body, demo, links,
            demo_width=520, text_width=760, text_height=720))

# ---------------------------------------------------------------------- t.ones
body = """Makes a tensor filled with ones, in the shape you give it.

Multiplying by one changes nothing, which makes ones the neutral start
for many things: times a number they become a tensor full of that
number, and as weights or a mask they let everything through until you
change some of them.

SYNTAX:
t.ones <dim 0: int> <dim 1: int> ... <dtype> <device>
One whole number per dimension: t.ones 8 is a row of 8, t.ones 4 4 a 4
by 4 tensor. With no numbers you get a 1-element tensor.

EXAMPLE:
t.ones 8

INPUTS and PARAMETERS:

in (button):
The small button at the top left of the node. Click it, or send it
anything, to send the tensor. Nothing else makes the node send.

shape (text):
The dimensions, typed as numbers separated by spaces or commas ('8',
'4 4', '[2, 3, 4]'). Editing it does not send; the next click uses it.

""" + DTYPE_DEVICE_GRAD % "t.ones 8 int64" + """
OUTPUTS:

tensor of ones:
The tensor. A bool dtype gives all true.

READING THE DEMO:
The top list is the tensor itself: eight ones. Below, '*' scales it by
the float. Drag the float: 't bang float' sets the multiplier first and
THEN clicks t.ones, so you get a row of whatever number you drag to.

RELATED:
t.zeros is the same with every value 0. t.full makes a tensor of any one
value in a single step - the same result as ones times a number.
t.ones_like takes its shape from another tensor. np.ones is the NumPy
version."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'fv', 'init': 'float', 'pos': (160, 62), 'w': 110, 'h': 42,
     'props': dict(FLT, **{'': 0.5, 'width': 80})},
    {'key': 'c0', 'comment': True, 'text': 'drag the float', 'pos': (160, 70)},
    {'key': 'tv', 'init': 't bang float', 'pos': (160, 120), 'w': 110, 'h': 46},
    {'key': 'on', 'init': 't.ones 8', 'pos': (30, 185), 'w': 220, 'h': 160},
    {'key': 'l1', 'init': 'list', 'pos': (30, 365), 'w': 360, 'h': 42, 'props': LIST},
    {'key': 'c1', 'comment': True, 'text': 'the tensor of ones', 'pos': (30, 370)},
    {'key': 'mul', 'init': '*', 'pos': (30, 425), 'w': 110, 'h': 70},
    {'key': 'l2', 'init': 'list', 'pos': (30, 515), 'w': 360, 'h': 42, 'props': LIST},
    {'key': 'c2', 'comment': True, 'text': 'ones times the float: a constant tensor',
     'pos': (30, 520)},
] + see_also(590, ['t.zeros', 't.full', 't.ones_like', 'np.ones'])
links = [('lb', 'out', 'fv', ''), ('fv', 'float out', 'tv', ''),
         ('tv', 'float', 'mul', 'operand'), ('tv', 'bang', 'on', '###input'),
         ('on', 'tensor of ones', 'l1', ''), ('on', 'tensor of ones', 'mul', 'in'),
         ('mul', 'result', 'l2', '')]
print(build('t.ones', 't.ones - a tensor of ones', body, demo, links,
            demo_width=420, text_width=760, text_height=640))

# ---------------------------------------------------------------------- t.rand
body = """Makes a tensor of random numbers, spread evenly between min and max.

Each click gives fresh numbers. Use it to test a chain with data that
changes, as noise to mix into a signal, for random positions or colours,
or wherever a starting point should be different every time.

The numbers are UNIFORM: every value between min and max is equally
likely, so the picture has no structure at all. For randomness with a
shape (a bell curve, say) see the t.dist nodes, such as t.dist.normal.

SYNTAX:
t.rand <dim 0: int> <dim 1: int> ... <dtype> <device>
One whole number per dimension. t.rand 12 12 is a 12 by 12 tensor;
t.rand 100 a row of 100. With no numbers you get a 1-element tensor.

EXAMPLE:
t.rand 12 12

INPUTS and PARAMETERS:

in (button):
The small button at the top left of the node. Click it, or send it
anything, for a new tensor - there is no need for a separate button
node. Only this inlet makes the node send.

shape (text):
The dimensions, typed as numbers separated by spaces or commas. Editing
it does not send; the next click uses it.

min / max (drag floats, also inlets):
The range. Setting them does NOT make a new tensor by itself - the next
click uses them. In the demo each float goes through 't bang float',
which passes the number to the inlet first and THEN clicks the node, so
dragging it gives you new noise at once.

""" + DTYPE_DEVICE_GRAD % "t.rand 12 12 int64" + """
The dtype changes what the random numbers are:
float32 and float16: any value from min up to, but never
exactly, max.
int64: whole numbers from min up to max - 1 - max itself never appears.
Choosing int64 while max is 1 changes max to 32767.
uint8: whole numbers. Choosing it raises max to 255 and a negative min
to 0 - and since max is left out, the largest value is 254.
bool: true or false at random; min and max are ignored.

OUTPUTS:

random tensor:
The new tensor, in the shape given.

READING THE DEMO:
load_bang makes the first tensor when the page opens; click the button
on t.rand itself for another. The heat_map colours 0 dark purple and 1
yellow. Drag min and max towards each other and the noise flattens to a
single colour; push max past 1 and cells saturate yellow. t.min and
t.max report the smallest and largest value in each new tensor - always
inside min and max.

RELATED:
np.rand is the NumPy version. t.rand_like takes its shape from another
tensor. t.dist.uniform does the same job as a distribution node, and
t.dist.normal gives a bell curve instead. t.zeros makes a tensor of one
fixed value."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'fills the heat_map once when the page opens',
     'pos': (30, 70)},
    {'key': 'fmin', 'init': 'float', 'pos': (270, 62), 'w': 110, 'h': 42,
     'props': dict(FLT, **{'': 0.0, 'width': 80})},
    {'key': 'fmax', 'init': 'float', 'pos': (400, 62), 'w': 110, 'h': 42,
     'props': dict(FLT, **{'': 1.0, 'width': 80})},
    {'key': 'tmin', 'init': 't bang float', 'pos': (270, 120), 'w': 110, 'h': 46},
    {'key': 'tmax', 'init': 't bang float', 'pos': (400, 120), 'w': 110, 'h': 46},
    {'key': 'c1', 'comment': True, 'text': 'drag min and max: new noise as you drag',
     'pos': (270, 125)},
    {'key': 'rnd', 'init': 't.rand 12 12', 'pos': (30, 190), 'w': 220, 'h': 230,
     'props': {'min': 0.0, 'max': 1.0}},
    {'key': 'c0b', 'comment': True,
     'text': "click t.rand's own button (top left of the node)\nfor a fresh tensor",
     'pos': (30, 195)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (270, 190), 'w': 260, 'h': 280,
     'props': HM(240, 240, 12, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 'a 12 x 12 tensor: 0 purple, 1 yellow',
     'pos': (270, 200)},
    {'key': 'tmn', 'init': 't.min', 'pos': (30, 490), 'w': 110, 'h': 70},
    {'key': 'tmx', 'init': 't.max', 'pos': (270, 490), 'w': 110, 'h': 70},
    {'key': 'f1', 'init': 'float', 'pos': (30, 580), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'f2', 'init': 'float', 'pos': (270, 580), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'smallest and largest: inside min..max',
     'pos': (30, 585)},
] + see_also(660, ['np.rand', 't.rand_like', 't.dist.uniform', 't.dist.normal',
                   't.zeros'])
links = [('lb', 'out', 'rnd', '###input'),
         ('fmin', 'float out', 'tmin', ''), ('fmax', 'float out', 'tmax', ''),
         ('tmin', 'float', 'rnd', 'min'), ('tmin', 'bang', 'rnd', '###input'),
         ('tmax', 'float', 'rnd', 'max'), ('tmax', 'bang', 'rnd', '###input'),
         ('rnd', 'random tensor', 'hm', 'y'),
         ('rnd', 'random tensor', 'tmn', 'tensor in'),
         ('rnd', 'random tensor', 'tmx', 'tensor in'),
         ('tmn', 'min tensor', 'f1', ''), ('tmx', 'max tensor', 'f2', '')]
print(build('t.rand', 't.rand - a tensor of random numbers', body, demo, links,
            demo_width=540, text_width=760, text_height=900))

# ---------------------------------------------------------------------- t.full
body = """Makes a tensor in which every element is the same value, which you choose.

A constant of any shape: a level to compare against, a bias to add, a
starting value other than zero or one. It is the one-step version of
t.ones times a number.

SYNTAX:
t.full <dim 0: int> <dim 1: int> ... <dtype> <device>
One whole number per dimension - t.full 4 4 is 4 by 4. The VALUE is not
an argument: set it on the node. With no numbers you get a 1-element
tensor. The number of dimensions is fixed when the node is made (one
dim property per number typed); retype the node to change it.

EXAMPLE:
t.full 4 4

INPUTS and PARAMETERS:

in (button):
The small button at the top left of the node. Click it, or send it
anything, to send the tensor. Nothing else makes the node send.

value (drag float, also an inlet):
What to fill with. Default 1.0. Setting it does not send by itself - in
the demo 't bang float' sets the value and then clicks the node.

dim 0, dim 1, ... (whole numbers):
The size of each dimension, used on the next click.

""" + DTYPE_DEVICE_GRAD % "t.full 4 4 int64" + """
An integer dtype cuts the value down to a whole number (2.7 fills with
2); a value that does not fit (300 as uint8) is an error and nothing is
sent.

OUTPUTS:

filled tensor:
The tensor.

READING THE DEMO:
Drag the float from 0 to 1: the whole 4 by 4 heat_map changes colour
together, every cell printing the same number, because every element is
that value.

RELATED:
t.ones and t.zeros are the two commonest fills. t.eye puts ones only on
the diagonal. np.ones (times a number) is the NumPy route to the same
thing."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'fv', 'init': 'float', 'pos': (160, 62), 'w': 110, 'h': 42,
     'props': dict(FLT, **{'': 0.3, 'width': 80, 'min': 0.0, 'max': 1.0})},
    {'key': 'c0', 'comment': True, 'text': 'drag the float between 0 and 1',
     'pos': (160, 70)},
    {'key': 'tv', 'init': 't bang float', 'pos': (160, 120), 'w': 110, 'h': 46},
    {'key': 'fl', 'init': 't.full 4 4', 'pos': (30, 185), 'w': 220, 'h': 220,
     'props': {'value': 0.3, 'dim 0': 4, 'dim 1': 4}},
    {'key': 'hm', 'init': 'heat_map', 'pos': (270, 185), 'w': 240, 'h': 250,
     'props': HM(220, 220, 4, fmt='%.2f')},
    {'key': 'c1', 'comment': True, 'text': 'every cell the same value: 0 purple, 1 yellow',
     'pos': (270, 190)},
] + see_also(470, ['t.ones', 't.zeros', 't.eye', 'np.ones'])
links = [('lb', 'out', 'fv', ''), ('fv', 'float out', 'tv', ''),
         ('tv', 'float', 'fl', 'value'), ('tv', 'bang', 'fl', '###input'),
         ('fl', 'filled tensor', 'hm', 'y')]
print(build('t.full', 't.full - a tensor of one value', body, demo, links,
            demo_width=530, text_width=760, text_height=700))

# ----------------------------------------------------------------------- t.eye
body = """Makes the identity matrix: an n by n square of zeros with ones down the diagonal.

Multiplying by the identity changes nothing, the way multiplying a
number by 1 does - so it is the starting point for a transformation
(a 4 by 4 identity is "no rotation, no move, no scale") and the neutral
element when you build one up step by step. As a picture it is a single
diagonal line, which also makes it a handy mask: it keeps the diagonal
of whatever you multiply it with element by element.

SYNTAX:
t.eye <n: int> <dtype> <device>
Default n is 4. It is always square. Give n FIRST: a dtype typed in
its place ('t.eye bool') is read as n = 0 and the node sends an empty
tensor.

EXAMPLE:
t.eye 6

INPUTS and PARAMETERS:

in (button):
The small button at the top left of the node. Click it, or send it
anything, to send the matrix. Nothing else makes the node send.

n (whole number, also an inlet):
The number of rows and columns. Setting it does not send - the metro in
the demo clicks the node 10 times a second, so a new n shows at once.
0 sends an empty tensor.

""" + DTYPE_DEVICE_GRAD % "t.eye 6 bool" + """
A bool dtype gives true on the diagonal, false elsewhere.

OUTPUTS:

eye tensor:
The n by n tensor.

READING THE DEMO:
The heat_map uses colour 'hot': 0 is black, 1 white, so the diagonal is
a white line from top left to bottom right - row 0 is at the top. Each
cell prints its value. Type a different n into the node and the square
grows or shrinks.

RELATED:
t.diag puts any values on the diagonal (or reads them off). t.zeros and
t.ones fill the whole square; t.full fills it with one value. t.tril
and t.triu keep the lower or upper triangle of a matrix."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 100.0, 'units': 'milliseconds'}},
    {'key': 'c0', 'comment': True, 'text': 'clicks t.eye 10 times a second',
     'pos': (30, 70)},
    {'key': 'ey', 'init': 't.eye 6', 'pos': (30, 150), 'w': 220, 'h': 190,
     'props': {'n': 6}},
    {'key': 'c1', 'comment': True, 'text': 'type a new n into the node',
     'pos': (30, 155)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (270, 150), 'w': 270, 'h': 280,
     'props': HM(250, 250, 6, fmt='%.0f', color='hot')},
    {'key': 'c2', 'comment': True, 'text': 'ones (white) down the diagonal, zeros (black) elsewhere',
     'pos': (270, 155)},
] + see_also(460, ['t.diag', 't.zeros', 't.full', 't.tril', 't.triu'])
links = [('met', '', 'ey', '###input'), ('ey', 'eye tensor', 'hm', 'y')]
print(build('t.eye', 't.eye - the identity matrix', body, demo, links,
            demo_width=560, text_width=760, text_height=700))


# ---------------------------------------------------------- the _like nodes
def LIKE_COMMON(name, fill):
    return """INPUTS and PARAMETERS:

tensor in:
The tensor whose SHAPE to copy. Anything arriving here makes the node
send - a stream of tensors gives a stream of outputs, each matching the
one that came in. Only torch tensors work: a NumPy array, list or number
is ignored and nothing is sent (turn it into a tensor first, for example
with t.to).

""" + ("""min / max (drag floats, also inlets):
The range of the random values, as on t.rand. They do not send by
themselves; the next tensor in uses them.

""" if name == 't.rand_like' else '') + DTYPE_DEVICE_GRAD % (name + ' int64') + """
NOTE - ONLY THE SHAPE IS COPIED:
PyTorch's own %s also copies the input's dtype and
device. This node does not: the output is always the dtype and device
set on the node (float32 on cpu unless you change them). Feed it a
float16 tensor on mps and you get float32 on cpu back. Set dtype and
device here to match if the two tensors will meet later.

OUTPUTS:

%s:
A tensor the same shape as the input.""" % (name.replace('t.', 'torch.'), fill)


body = """Makes a tensor of zeros the same shape as the tensor you send it.

Use it when the blank you need must match data you already have: a
running total for incoming frames, an empty mask the size of an image,
a reset value for a buffer. Typing the shape into t.zeros breaks the
moment the data changes size; t.zeros_like follows it.

SYNTAX:
t.zeros_like <dtype> <device>
No arguments are needed.

EXAMPLE:
t.zeros_like

""" + LIKE_COMMON('t.zeros_like', 'tensor of zeros') + """

READING THE DEMO:
A metro clicks t.linspace, whose ramp (top strip) feeds t.zeros_like.
The bottom strip is the result: the same number of cells, all 0, one
flat colour. Drag steps on t.linspace: both strips divide together, and
t.info shows the shape following - no shape was ever typed into
t.zeros_like.

RELATED:
t.zeros makes zeros in a shape you type. t.ones_like and t.rand_like
copy the shape the same way with ones or random values. np.zeros is the
NumPy blank."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 100.0, 'units': 'milliseconds'}},
    {'key': 'ls', 'init': 't.linspace 0 1 12', 'pos': (30, 150), 'w': 220, 'h': 210,
     'props': {'start': 0.0, 'stop': 1.0, 'steps': 12}},
    {'key': 'c0', 'comment': True, 'text': 'the input: drag steps to change its length',
     'pos': (30, 155)},
    {'key': 'hm0', 'init': 'heat_map', 'pos': (30, 380), 'w': 468, 'h': 100,
     'props': HM(460, 50, 1, fmt='none')},
    {'key': 'c1', 'comment': True, 'text': 'input: a ramp, 0 to 1', 'pos': (30, 385)},
    {'key': 'zl', 'init': 't.zeros_like', 'pos': (30, 500), 'w': 220, 'h': 140},
    {'key': 'inf', 'init': 't.info', 'pos': (290, 500), 'w': 160, 'h': 130},
    {'key': 'c2', 'comment': True, 'text': 'shape of the output', 'pos': (290, 505)},
    {'key': 'hm1', 'init': 'heat_map', 'pos': (30, 660), 'w': 468, 'h': 100,
     'props': HM(460, 50, 1, fmt='none')},
    {'key': 'c3', 'comment': True, 'text': 'output: the same length, every cell 0',
     'pos': (30, 665)},
] + see_also(790, ['t.zeros', 't.ones_like', 't.rand_like', 'np.zeros'])
links = [('met', '', 'ls', '###input'), ('ls', 'linspace out', 'hm0', 'y'),
         ('ls', 'linspace out', 'zl', 'tensor in'),
         ('zl', 'tensor of zeros', 'hm1', 'y'), ('zl', 'tensor of zeros', 'inf', 'in')]
print(build('t.zeros_like', 't.zeros_like - zeros in the shape of another tensor',
            body, demo, links, demo_width=520, text_width=760, text_height=860))

body = """Makes a tensor of ones the same shape as the tensor you send it.

Use it when a neutral tensor - weights that change nothing, a mask that
lets everything through - must match data you already have. Times a
number, it becomes a constant of that shape. Typing the shape into
t.ones breaks the moment the data changes size; t.ones_like follows it.

SYNTAX:
t.ones_like <dtype> <device>
No arguments are needed.

EXAMPLE:
t.ones_like

""" + LIKE_COMMON('t.ones_like', 'tensor of ones') + """

READING THE DEMO:
A metro clicks t.linspace, whose ramp (top strip) feeds t.ones_like.
The bottom strip is the result: the same number of cells, all 1 (all
yellow). Drag steps on t.linspace: both strips divide together, and
t.info shows the shape following.

RELATED:
t.ones makes ones in a shape you type. t.zeros_like and t.rand_like
copy the shape the same way with zeros or random values. t.full fills
with any value."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 100.0, 'units': 'milliseconds'}},
    {'key': 'ls', 'init': 't.linspace 0 1 12', 'pos': (30, 150), 'w': 220, 'h': 210,
     'props': {'start': 0.0, 'stop': 1.0, 'steps': 12}},
    {'key': 'c0', 'comment': True, 'text': 'the input: drag steps to change its length',
     'pos': (30, 155)},
    {'key': 'hm0', 'init': 'heat_map', 'pos': (30, 380), 'w': 468, 'h': 100,
     'props': HM(460, 50, 1, fmt='none')},
    {'key': 'c1', 'comment': True, 'text': 'input: a ramp, 0 to 1', 'pos': (30, 385)},
    {'key': 'ol', 'init': 't.ones_like', 'pos': (30, 500), 'w': 220, 'h': 140},
    {'key': 'inf', 'init': 't.info', 'pos': (290, 500), 'w': 160, 'h': 130},
    {'key': 'c2', 'comment': True, 'text': 'shape of the output', 'pos': (290, 505)},
    {'key': 'hm1', 'init': 'heat_map', 'pos': (30, 660), 'w': 468, 'h': 100,
     'props': HM(460, 50, 1, fmt='none')},
    {'key': 'c3', 'comment': True, 'text': 'output: the same length, every cell 1',
     'pos': (30, 665)},
] + see_also(790, ['t.ones', 't.zeros_like', 't.rand_like', 't.full'])
links = [('met', '', 'ls', '###input'), ('ls', 'linspace out', 'hm0', 'y'),
         ('ls', 'linspace out', 'ol', 'tensor in'),
         ('ol', 'tensor of ones', 'hm1', 'y'), ('ol', 'tensor of ones', 'inf', 'in')]
print(build('t.ones_like', 't.ones_like - ones in the shape of another tensor',
            body, demo, links, demo_width=520, text_width=760, text_height=860))

body = """Makes a tensor of random numbers the same shape as the tensor you send it.

Use it to add noise to incoming data - the noise must be the same shape
as the data - to jitter positions, or to test a chain with random data
in exactly the shape real data will have. Because every tensor in
triggers a new one, a stream of frames gets fresh noise every frame.

The numbers are uniform between min and max, as on t.rand.

SYNTAX:
t.rand_like <dtype> <device>
No arguments are needed.

EXAMPLE:
t.rand_like

""" + LIKE_COMMON('t.rand_like', 'random tensor') + """

The dtype changes what the numbers are, as on t.rand: float types give
any value from min up to (not including) max; int64 and uint8 give whole
numbers from min to max - 1 (choosing int64 while max is 1 raises max to
32767; uint8 raises it to 255); bool gives true or false at random.

READING THE DEMO:
A metro clicks t.linspace 10 times a second, and each ramp it sends
(top strip) makes t.rand_like send a fresh row of noise the same length
(bottom strip) - new numbers every time. Drag steps on t.linspace and
the noise follows its length; drag min and max on t.rand_like to change
the noise range (the strip is scaled 0 to 1).

RELATED:
t.rand makes noise in a shape you type. t.zeros_like and t.ones_like
copy the shape the same way with zeros or ones. np.rand is the NumPy
version of t.rand."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 100.0, 'units': 'milliseconds'}},
    {'key': 'ls', 'init': 't.linspace 0 1 12', 'pos': (30, 150), 'w': 220, 'h': 210,
     'props': {'start': 0.0, 'stop': 1.0, 'steps': 12}},
    {'key': 'c0', 'comment': True, 'text': 'the input: drag steps to change its length',
     'pos': (30, 155)},
    {'key': 'hm0', 'init': 'heat_map', 'pos': (30, 380), 'w': 468, 'h': 100,
     'props': HM(460, 50, 1, fmt='none')},
    {'key': 'c1', 'comment': True, 'text': 'input: a ramp, 0 to 1', 'pos': (30, 385)},
    {'key': 'rl', 'init': 't.rand_like', 'pos': (30, 500), 'w': 220, 'h': 200,
     'props': {'min': 0.0, 'max': 1.0}},
    {'key': 'c2b', 'comment': True, 'text': 'drag min and max', 'pos': (30, 505)},
    {'key': 'inf', 'init': 't.info', 'pos': (290, 500), 'w': 160, 'h': 130},
    {'key': 'c2', 'comment': True, 'text': 'shape of the output', 'pos': (290, 505)},
    {'key': 'hm1', 'init': 'heat_map', 'pos': (30, 720), 'w': 468, 'h': 100,
     'props': HM(460, 50, 1, fmt='none')},
    {'key': 'c3', 'comment': True, 'text': 'output: the same length, fresh noise each time',
     'pos': (30, 725)},
] + see_also(850, ['t.rand', 't.zeros_like', 't.ones_like', 'np.rand'])
links = [('met', '', 'ls', '###input'), ('ls', 'linspace out', 'hm0', 'y'),
         ('ls', 'linspace out', 'rl', 'tensor in'),
         ('rl', 'random tensor', 'hm1', 'y'), ('rl', 'random tensor', 'inf', 'in')]
print(build('t.rand_like', 't.rand_like - random numbers in the shape of another tensor',
            body, demo, links, demo_width=520, text_width=760, text_height=920))
