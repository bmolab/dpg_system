"""np limits, running totals and targets: one page per node.

np.clip, np.min, np.max (one class, NumpyClipNode) were the np_clip family
page; their pages are hand-made now (see tools/README.md), so this script no
longer builds them. np.cumsum, np.cumprod, np.diff were np_cumulative;
np.distance_from_target, np.proximity_to_target, proximity_trigger were
np_target. Each now has its own page whose demo shows the input and the
output, with a SEE ALSO row under it.
"""
import sys, os, json
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


def ARRAY_PLOT(miny, maxy, count, width=300):
    """A plot that draws one whole 1-D array as a curve, first value on the left."""
    return dict(PLOT(miny, maxy, count),
                **{'width': width, 'update style': 'buffer holds one sample of input'})


METRO = lambda ms: {'on': True, 'period': float(ms), 'units': 'milliseconds'}


def text_h(body):
    """Height of the prose block: 15px per line at text size 30, plus a margin."""
    return (body.count('\n') + 1) * 15 + 30


def set_state(path, node_name, state):
    """Add saved node state (what a node's save_custom writes) to a built page.

    build() has no slot for it; the target nodes keep their targets this way,
    so the demo works the moment the page opens."""
    doc = json.load(open(path))
    patches = list(doc['patches'].values()) if 'patches' in doc else [doc]
    for p in patches:
        for n in p['nodes'].values():
            if n.get('name') == node_name:
                n.update(state)
    json.dump(doc, open(path, 'w'), indent=4)


# --------------------------------------------------------- demo subpatchers
# Inside make_point: two sines at different speeds make an x y point that
# wanders over the square from -1 to 1, coming back near any spot every few
# seconds (the path repeats every 12 seconds).
POINT_SUB = {
    'name': 'make_point',
    'host': 'mp',
    'demo': [
        {'key': 'sx', 'init': 'signal 3.0 sin', 'pos': (40, 40), 'w': 129, 'h': 78,
         'props': SIG('sin', 3.0)},
        {'key': 'sy', 'init': 'signal 4.0 cos', 'pos': (200, 40), 'w': 129, 'h': 78,
         'props': SIG('cos', 4.0)},
        {'key': 'cx', 'init': 'comment x: once every 3 s, y: once every 4 s, both -1 to 1',
         'pos': (40, 135), 'w': 460, 'h': 30},
        {'key': 'pk', 'init': 'pack 2', 'pos': (40, 180), 'w': 130, 'h': 100},
        {'key': 'cp', 'init': 'comment the point as a list: x y',
         'pos': (200, 200), 'w': 260, 'h': 30},
        {'key': 'out', 'init': 'out point', 'pos': (40, 310), 'w': 150, 'h': 80},
    ],
    'links': [('sx', '', 'pk', 'in 1'), ('sy', '', 'pk', 'in 2'),
              ('pk', 'out', 'out', 'point')],
}

TARGET = [0.5, 0.7]
TARGETS = [[0.5, 0.7], [-0.5, -0.7], [0.0, 0.0]]


# np.clip, np.min, np.max: their pages are hand-made now (see tools/README.md).

# -------------------------------------------------------------------- np.cumsum
body = """Replaces each value of an array with the running total up to and
including it: the first value, then the first two added, then the first
three, and so on.

A sum gives you the total; a cumulative sum gives you the total SO FAR at
every point. Use it to turn steps into a path (movements into positions,
speeds into distance travelled), to turn a histogram into a cumulative
distribution, or to turn a list of durations into start times.

SYNTAX:
np.cumsum <axis: int>
Default axis -1, the last axis: along each row.

EXAMPLE:
np.cumsum

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input is summed and sent -
the only inlet that makes the node send. A single number has no axis to
run along, and sends nothing.

axis (int box, also an inlet):
Which way to run. For a 2-D array, -1 or 1 runs along each row on its
own, 0 or -2 runs down each column. Negative counts from the end. An axis
the array does not have sends nothing. Changing it does not send by
itself; the next array uses it.

OUTPUTS:

cumsum:
The running totals, the same shape as the input. The last value is the
sum of the whole row (what np.sum would give).

READING THE DEMO:
Top: np.rand makes 200 random steps between -1 and 1, a new set every
half second. The upper plot shows them - noise, with no shape. The lower plot
is their cumsum: a random walk, the path you get by taking those steps
one after another. Drag np.rand's max up to 1.2 (or min down) and the
steps are no longer balanced: the walk drifts steadily up (or down).

Bottom: ten 1s summed give the ramp 1 2 3 ... 10 - the cumsum of a
constant is a straight line whose slope is the constant. Edit the
message (2 2 2, or 1 -1 1 -1) and click it.

RELATED:
np.diff is the reverse: it turns a path back into its steps. np.cumprod
multiplies instead of adding. accumulate keeps a running total of a
STREAM, one number at a time. np.sum gives only the final total.
t.cumsum is the PyTorch version."""

demo = [
    {'key': 'met', 'init': 'metro 500', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': METRO(500)},
    {'key': 'c0', 'comment': True, 'text': 'new steps twice a second', 'pos': (30, 70)},
    {'key': 'rnd', 'init': 'np.rand 200', 'pos': (30, 150), 'w': 160, 'h': 140,
     'props': {'min': -1.0, 'max': 1.0, 'dim 0': 200, 'dtype': 'float32'}},
    {'key': 'c1', 'comment': True, 'text': '200 steps between min and max - drag them',
     'pos': (30, 155)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 310), 'w': 308, 'h': 178,
     'props': ARRAY_PLOT(-1.5, 1.5, 200)},
    {'key': 'c2', 'comment': True, 'text': 'the steps: noise', 'pos': (30, 315)},
    {'key': 'cs', 'init': 'np.cumsum', 'pos': (30, 510), 'w': 150, 'h': 70,
     'props': {'axis': -1}},
    {'key': 'p2', 'init': 'plot', 'pos': (30, 600), 'w': 308, 'h': 178,
     'props': ARRAY_PLOT(-30.0, 30.0, 200)},
    {'key': 'c3', 'comment': True, 'text': 'their running total: a random walk (-30 to 30)',
     'pos': (30, 605)},
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 820), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 880), 'w': 300, 'h': 42,
     'props': {'text in': '1 1 1 1 1 1 1 1 1 1', 'font size': '24'}},
    {'key': 'c4', 'comment': True, 'text': 'ten 1s - edit and click', 'pos': (30, 885)},
    {'key': 'h1', 'init': 'heat_map', 'pos': (30, 940), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, miny=0.0, maxy=10.0, fmt='%.0f')},
    {'key': 'c5', 'comment': True, 'text': 'in: ten 1s', 'pos': (30, 945)},
    {'key': 'cs2', 'init': 'np.cumsum', 'pos': (30, 1070), 'w': 150, 'h': 70,
     'props': {'axis': -1}},
    {'key': 'h2', 'init': 'heat_map', 'pos': (30, 1160), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, miny=0.0, maxy=10.0, fmt='%.0f')},
    {'key': 'c6', 'comment': True, 'text': 'out: 1 to 10, a ramp', 'pos': (30, 1165)},
] + see_also(1300, ['np.diff', 'np.cumprod', 'accumulate', 'np.sum', 't.cumsum'])
links = [('met', '', 'rnd', ''), ('rnd', 'random array', 'p1', 'y'),
         ('rnd', 'random array', 'cs', 'in'), ('cs', 'cumsum', 'p2', 'y'),
         ('lb', 'out', 'm1', ''), ('m1', 'message out', 'h1', 'y'),
         ('m1', 'message out', 'cs2', 'in'), ('cs2', 'cumsum', 'h2', 'y')]
print(build('np.cumsum', 'np.cumsum - the running total',
            body, demo, links, demo_width=600, text_width=760, text_height=text_h(body)))

# ------------------------------------------------------------------- np.cumprod
body = """Replaces each value of an array with the running product up to and
including it: the first value, then the first two multiplied, then the
first three, and so on.

It is cumsum with multiplying instead of adding. Use it for anything that
compounds: growth by a rate per step (interest, a population, a pitch
rising by a fixed ratio per note), decay by a factor per step (an echo
fading), or a chain of gains or scale factors applied one after another.

SYNTAX:
np.cumprod <axis: int>
Default axis -1, the last axis: along each row.

EXAMPLE:
np.cumprod

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input is multiplied out and
sent - the only inlet that makes the node send. A single number has no
axis to run along, and sends nothing.

axis (int box, also an inlet):
Which way to run. For a 2-D array, -1 or 1 runs along each row on its
own, 0 or -2 runs down each column. Negative counts from the end. An axis
the array does not have sends nothing. Changing it does not send by
itself; the next array uses it.

OUTPUTS:

cumprod:
The running products, the same shape as the input. A single 0 anywhere
makes every value after it 0. Whole numbers stay whole numbers (int64),
which silently overflow past about 9 x 10 to the 18th - beyond 20
factorial.

READING THE DEMO:
Top: np.ones makes 40 ones and * turns them into forty copies of the
growth rate, 1.05. The plot shows that flat input line and its cumprod:
1.05, then 1.05 times 1.05, ... - compound growth, a curve that bends
upward ever faster and reaches about 7 after 40 steps. Change the * value
on the node: 1.02 grows slowly, 0.95 decays towards 0, 1 stays flat.
Above about 1.055 the curve runs off the top of the plot (0 to 8).

Bottom: 1 2 3 4 5 6 multiplied out gives 1 2 6 24 120 720 - the
factorials.

RELATED:
np.cumsum adds instead of multiplying. np.diff is the reverse of
cumsum, not of cumprod (for cumprod, divide each value by the one
before). t.cumprod is the PyTorch version."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': METRO(100)},
    {'key': 'c0', 'comment': True, 'text': 'ten times a second', 'pos': (30, 70)},
    {'key': 'on', 'init': 'np.ones 40', 'pos': (30, 150), 'w': 160, 'h': 140,
     'props': {'dim 0': 40, 'dtype': 'float32'}},
    {'key': 'c1', 'comment': True, 'text': '40 ones', 'pos': (30, 155)},
    {'key': 'ml', 'init': '* 1.05', 'pos': (30, 310), 'w': 120, 'h': 56,
     'props': {'operand': 1.05}},
    {'key': 'c2', 'comment': True, 'text': 'the growth rate - change it', 'pos': (30, 315)},
    {'key': 'tr', 'init': 't array array', 'pos': (30, 390), 'w': 140, 'h': 46},
    {'key': 'ct', 'comment': True, 'text': 'right first: the rate as it is, then',
     'pos': (200, 395)},
    {'key': 'cp', 'init': 'np.cumprod', 'pos': (30, 460), 'w': 150, 'h': 70,
     'props': {'axis': -1}},
    {'key': 'pk', 'init': 'pack 2', 'pos': (30, 550), 'w': 130, 'h': 100},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 670), 'w': 308, 'h': 178,
     'props': MULTI_PLOT(0.0, 8.0, 40)},
    {'key': 'c3', 'comment': True,
     'text': 'the flat line is the input (1.05 forty times);\nthe curve is its cumprod: compound growth',
     'pos': (30, 860)},
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 930), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 990), 'w': 200, 'h': 42,
     'props': {'text in': '1 2 3 4 5 6', 'font size': '24'}},
    {'key': 'cp2', 'init': 'np.cumprod', 'pos': (30, 1050), 'w': 150, 'h': 70,
     'props': {'axis': -1}},
    {'key': 'l1', 'init': 'list', 'pos': (30, 1140), 'w': 300, 'h': 42, 'props': LIST},
    {'key': 'c4', 'comment': True, 'text': '1 x 2 x 3 ...: the factorials', 'pos': (30, 1145)},
] + see_also(1220, ['np.cumsum', 'np.diff', 't.cumprod'])
links = [('met', '', 'on', ''), ('on', 'array of ones', 'ml', 'in'),
         ('ml', 'result', 'tr', ''),
         ('tr', 'array', 'pk', 'in 2', 1),
         ('tr', 'array', 'cp', 'in', 0), ('cp', 'cumprod', 'pk', 'in 1'),
         ('pk', 'out', 'pl', 'y'),
         ('lb', 'out', 'm1', ''), ('m1', 'message out', 'cp2', 'in'),
         ('cp2', 'cumprod', 'l1', '')]
print(build('np.cumprod', 'np.cumprod - the running product',
            body, demo, links, demo_width=600, text_width=760, text_height=text_h(body)))

# ---------------------------------------------------------------------- np.diff
body = """Replaces an array with the differences between neighbouring values:
each value minus the one before it.

It turns positions into steps - where something WAS into how far it
MOVED between samples. Applied twice it gives the change in the change:
from positions, order 1 is speed and order 2 is acceleration. Use it to
find where a sequence jumps, to get velocities from a recorded path, or
to undo a cumsum.

THE OUTPUT IS SHORTER:
Nine values have only eight gaps between them, so the result is one
value shorter than the input along the axis - and each further order
loses one more. Anything expecting the original length will notice.

SYNTAX:
np.diff <order: int> <axis: int>
Defaults: order 1, axis -1 (the last axis: along each row).

EXAMPLE:
np.diff 2

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input is differenced and
sent - the only inlet that makes the node send.

order (int box, also an inlet):
How many times to difference. Below 1 is treated as 1. If the array is
not longer than the order along the axis (6 values at order 6), there is
nothing left to send, and nothing is sent. Changing it does not send by
itself - in the demo a metro re-sends the input five times a second.

axis (int box, also an inlet):
Which way to difference. For a 2-D array, -1 or 1 works along each row,
0 or -2 down each column. An axis the array does not have sends nothing.

OUTPUTS:

differences:
The differences, shorter by order along the axis.

READING THE DEMO:
Top: 0 1 4 9 16 25 36 49 64 are the positions of something speeding up
(the squares). Order 1 gives 1 3 5 7 9 11 13 15 - the distance covered
in each step, growing steadily: the speed. Type 2 into order and you get
2 2 2 2 2 2 2 - the speed grows by the same amount every step, a
constant acceleration. Order 3 gives zeros: the acceleration never
changes.

Bottom, the inverse: 3 1 4 1 5 9 summed by np.cumsum is the path
3 4 8 9 14 23, and np.diff of that gives back 1 4 1 5 9 - the steps, all
but the first, which had nothing before it to differ from. To rebuild a
path from its steps exactly you also need where it started.

RELATED:
np.cumsum is the reverse: steps back into a path. diff does the same
for a STREAM of numbers, one sample at a time, and diff_filter measures
movement on a smoother timescale. t.diff is the PyTorch version."""

demo = [
    {'key': 'met', 'init': 'metro 200', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': METRO(200)},
    {'key': 'c0', 'comment': True, 'text': 're-sends the positions five times a second',
     'pos': (30, 70)},
    {'key': 'm1', 'init': 'message', 'pos': (30, 150), 'w': 300, 'h': 42,
     'props': {'text in': '0 1 4 9 16 25 36 49 64', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'positions of something speeding up',
     'pos': (30, 155)},
    {'key': 'h1', 'init': 'heat_map', 'pos': (30, 210), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, miny=0.0, maxy=64.0, fmt='%.0f')},
    {'key': 'c2', 'comment': True, 'text': 'in: 9 positions', 'pos': (30, 215)},
    {'key': 'df', 'init': 'np.diff 1', 'pos': (30, 340), 'w': 150, 'h': 90,
     'props': {'order': 1, 'axis': -1}},
    {'key': 'c3', 'comment': True, 'text': 'type 1, 2 or 3 into order', 'pos': (30, 345)},
    {'key': 'h2', 'init': 'heat_map', 'pos': (30, 450), 'w': 468, 'h': 110,
     'props': HM(460, 60, 1, miny=0.0, maxy=16.0, fmt='%.0f')},
    {'key': 'c4', 'comment': True, 'text': 'out: one value fewer per order',
     'pos': (30, 455)},
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 600), 'w': 88, 'h': 46},
    {'key': 'm2', 'init': 'message', 'pos': (30, 660), 'w': 200, 'h': 42,
     'props': {'text in': '3 1 4 1 5 9', 'font size': '24'}},
    {'key': 'c5', 'comment': True, 'text': 'six steps - click to re-send', 'pos': (30, 665)},
    {'key': 'cs', 'init': 'np.cumsum', 'pos': (30, 720), 'w': 150, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (30, 810), 'w': 300, 'h': 42, 'props': LIST},
    {'key': 'c6', 'comment': True, 'text': 'the path they make', 'pos': (30, 815)},
    {'key': 'df2', 'init': 'np.diff', 'pos': (30, 870), 'w': 150, 'h': 90},
    {'key': 'l2', 'init': 'list', 'pos': (30, 980), 'w': 300, 'h': 42, 'props': LIST},
    {'key': 'c7', 'comment': True, 'text': 'the steps again, minus the first',
     'pos': (30, 985)},
] + see_also(1060, ['np.cumsum', 'diff', 'diff_filter', 't.diff'])
links = [('met', '', 'm1', ''), ('m1', 'message out', 'h1', 'y'),
         ('m1', 'message out', 'df', 'in'), ('df', 'differences', 'h2', 'y'),
         ('lb', 'out', 'm2', ''), ('m2', 'message out', 'cs', 'in'),
         ('cs', 'cumsum', 'l1', ''), ('cs', 'cumsum', 'df2', 'in'),
         ('df2', 'differences', 'l2', '')]
print(build('np.diff', 'np.diff - the differences between neighbours',
            body, demo, links, demo_width=600, text_width=760, text_height=text_h(body)))

# ------------------------------------------------------- np.distance_from_target
body = """Measures how far its input is from a stored target position, and
sends that distance every time an input arrives.

The target is captured rather than typed: put the thing where you want
the target to be, click "set target", and from then on every input is
measured against that spot. Use it to know how far a tracked hand is from
a place in the room, how far a pose has moved from a reference pose, or
how near a moving point is to where it started.

The distance is the straight-line one: the square root of the sum of the
squared differences (np.linalg.norm of input minus target), in the same
units as the input. It is 0 at the target and never negative.

SYNTAX:
np.distance_from_target

INPUTS and PARAMETERS:

input:
The current position, as a list or array of any length (x y, x y z, a
whole pose...). Every input is measured and sent. Nothing is sent until a
target has been set. An input whose shape does not fit the target sends
nothing.

set target (button):
Captures whatever was last received at input as the target, and sends
the distance at once (0). If nothing has arrived yet, it does nothing.
The target is saved with the patch - this page opens with 0.5 0.7.

axis (menu, also an inlet):
Only matters when the input holds several positions, one per row.
None (the default) measures the whole difference as ONE distance. 1
gives one distance per row. A captured set of rows is measured row
against row; a single captured position is measured against every row.
The menu offers as many axes as the input has.

OUTPUTS:

norm:
The distance: one number, or one per row with axis 1.

READING THE DEMO:
make_point moves an x y point around the square from -1 to 1 (open it
to see how). The plot and the number show its distance from the target,
0.5 0.7: the curve dips to near 0 each time the point passes the target,
and peaks when it is in the far corner. Click "set target" and wherever
the point is at that moment becomes the target - the curve starts again
from 0.

RELATED:
proximity_trigger uses the same distance to fire an event on arrival.
np.proximity_to_target turns distance from several targets into 0 to 1
levels for driving sounds. np.linalg.norm measures the length of an
array (its distance from zero). t.dist is the PyTorch version, between
two tensors."""

demo = [
    {'key': 'mp', 'init': 'p make_point', 'pos': (30, 62), 'w': 170, 'h': 70},
    {'key': 'c0', 'comment': True, 'text': 'an x y point wandering from -1 to 1',
     'pos': (30, 150)},
    {'key': 'dt', 'init': 'np.distance_from_target', 'pos': (30, 190), 'w': 240, 'h': 120,
     'props': {'axis': 'None'}},
    {'key': 'c1', 'comment': True, 'text': 'click set target to capture the point',
     'pos': (30, 195)},
    {'key': 'f1', 'init': 'float', 'pos': (30, 330), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': 'the distance now', 'pos': (30, 335)},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 400), 'w': 308, 'h': 178,
     'props': dict(PLOT(0.0, 2.5), width=300)},
    {'key': 'c3', 'comment': True,
     'text': 'distance from 0.5 0.7 over the last few seconds:\nnear 0 each time the point passes the target',
     'pos': (30, 590)},
] + see_also(670, ['proximity_trigger', 'np.proximity_to_target', 'np.linalg.norm', 't.dist'])
links = [('mp', 'point', 'dt', 'input'), ('dt', 'norm', 'f1', ''), ('dt', 'norm', 'pl', 'y')]
p = build('np.distance_from_target', 'np.distance_from_target - how far from a stored spot',
          body, demo, links, demo_width=600, text_width=760, text_height=text_h(body),
          subpatch=POINT_SUB)
set_state(p, 'np.distance_from_target', {'target': TARGET})
print(p)

# ------------------------------------------------------------- proximity_trigger
body = """Fires an event when its input arrives near a stored target, and
counts the arrivals.

It turns "how far is it" into "it got there": when the distance from the
input to the target drops below threshold, state sends true and count
goes up by one. It cannot fire again until the input has gone further
away than release_threshold, when state sends false. Use it to trigger a
cue when a dancer's hand reaches a spot, or to count how often a tracked
point visits a place.

TWO THRESHOLDS:
The gap between threshold and release_threshold is hysteresis, the same
idea as the hysteresis node: without it, a hand hovering at the boundary
would fire over and over as tracking noise carries it back and forth.
Keep release_threshold larger than threshold; if it is smaller, the
trigger releases on the very next input and re-fires while you stay
near.

SYNTAX:
proximity_trigger

INPUTS and PARAMETERS:

input:
The current position, as a list or array of any length. Every input is
measured against the target. The node does nothing at all until a
target is set and arm is ticked. An input whose shape does not fit the
target sends nothing.

arm (checkbox):
Switches the trigger on. It starts OFF, and while it is off nothing is
measured or sent. Ticking or unticking it also sets the count back to
0 and sends that 0 from count.

reset_count (button):
Sets the count back to 0 and sends the 0 from count at once.

set target (button):
Captures whatever was last received at input as the target. The target
is saved with the patch - this page opens with 0.5 0.7.

threshold (drag float, default 0.3):
How close counts as arrived. Strictly closer than this fires.

release_threshold (drag float, default 0.5):
How far it must go (this far or further) before it can fire again.

axis (menu, also an inlet):
Only matters when the input holds several positions, one per row. With
None (the default) the whole difference is one distance. With 1 each row
has its own distance, and the NEAREST decides: any row inside threshold
fires, and it releases only when every row is past release_threshold.

OUTPUTS:

state:
true on arrival, false on release - sent only when it changes.

count:
How many arrivals since the count was last reset, sent on each arrival
and on each reset.

proximity:
Despite its name, the DISTANCE to the target, sent on every input while
armed (one per row with axis 1). It gets smaller as you approach, and is
0 at the target.

READING THE DEMO:
make_point moves an x y point around the square from -1 to 1. The plot
shows its distance from the target 0.5 0.7, with two flat lines at 0.3
and 0.5 - the thresholds as saved. Each time the curve dips below the
lower line the toggle turns on and the count goes up; the toggle turns
off only when the curve climbs back past the upper line. The point
arrives about every three seconds. Drag the thresholds on the node (the
lines stay at the saved values), or click set target to move the target.

RELATED:
np.distance_from_target sends the same distance without the event.
np.proximity_to_target grades closeness to several targets from 0 to 1.
trigger and hysteresis do the same two-threshold firing on a single
number."""

demo = [
    {'key': 'mp', 'init': 'p make_point', 'pos': (30, 62), 'w': 170, 'h': 70},
    {'key': 'c0', 'comment': True, 'text': 'an x y point wandering from -1 to 1',
     'pos': (30, 150)},
    {'key': 'pt', 'init': 'proximity_trigger', 'pos': (30, 190), 'w': 240, 'h': 220,
     'props': {'arm': True, 'threshold': 0.3, 'release_threshold': 0.5, 'axis': 'None'}},
    {'key': 'c1', 'comment': True, 'text': 'armed; target 0.5 0.7', 'pos': (30, 195)},
    {'key': 'tg', 'init': 'toggle', 'pos': (30, 440), 'w': 45, 'h': 42},
    {'key': 'i1', 'init': 'int', 'pos': (110, 440), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'state: on while arrived.  count of arrivals',
     'pos': (30, 445)},
    {'key': 'pk', 'init': 'pack f 0.3 0.5', 'pos': (30, 510), 'w': 150, 'h': 100},
    {'key': 'c3', 'comment': True, 'text': 'the distance, with lines at 0.3 and 0.5',
     'pos': (30, 515)},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 630), 'w': 308, 'h': 178,
     'props': MULTI_PLOT(0.0, 2.5)},
    {'key': 'c4', 'comment': True,
     'text': 'fires when the curve dips under 0.3,\nre-arms when it climbs back past 0.5',
     'pos': (30, 820)},
] + see_also(890, ['np.distance_from_target', 'np.proximity_to_target', 'trigger', 'hysteresis'])
links = [('mp', 'point', 'pt', 'input'), ('pt', 'state', 'tg', ''),
         ('pt', 'count', 'i1', ''), ('pt', 'proximity', 'pk', 'float'),
         ('pk', 'out', 'pl', 'y')]
p = build('proximity_trigger', 'proximity_trigger - an event on arrival at a target',
          body, demo, links, demo_width=600, text_width=760, text_height=text_h(body),
          subpatch=POINT_SUB)
set_state(p, 'proximity_trigger', {'target': TARGET})
print(p)

# -------------------------------------------------------- np.proximity_to_target
body = """Measures how close its input is to each of a set of targets, as a
level from 1 (at the target) falling to 0 at the edge of the target's
zone - the shape you want for fading sounds in and out as a tracked point
moves about.

Each target is a spot with a round zone around it, radius wide. Inside
the zone the level rises as you approach; outside it the level is 0.
The output is made to plug straight into the fade inlet of
polyphonic_sampler~: target 0 drives one sound, target 1 the next, and
so on, so walking from spot to spot crossfades between sounds.

SETTING THE TARGETS:
Choose a slot with target index, put the tracker where that target
should be, and click capture. Choose another index and capture again.
Skipping ahead (index 5 when there are 2) leaves the slots in between
unset; they report 0. The targets are saved with the patch - this page
opens with three: 0.5 0.7, -0.5 -0.7 and 0 0. Capturing a point of a
different length (x y z into a set of x y targets) throws the whole set
away and starts a new one.

SYNTAX:
np.proximity_to_target

INPUTS and PARAMETERS:

input:
The tracker position, as a list or array (x y, x y z...), or several
trackers, one per row. Every input is measured and sent. Nothing is sent
until there are targets, or when the input's length does not match
theirs.

target index (int box) / capture (button):
The slot to capture into, and the button that copies the last input into
it. capture sends the new set from targets and measures at once.

targets:
Replaces the whole set: a list of positions, one per row ([[x, y], ...]).
A flat list is taken as ONE target.

clear (button):
Forgets every target, fades out whatever is sounding (one last 0 for
each), and sends an empty list from targets.

radius (drag float, default 0.5):
The size of each zone, in the input's units. The level is 1 minus
(distance / radius), so it falls in a straight line from 1 to 0.

exponent (drag float, default 1.0):
Shapes that fall. Above 1 the level stays low until you are close and
then rises sharply - a tighter spot. Below 1 it rises quickly at the
edge and stays high across the zone - a broad plateau.

first sound (int box, default 0):
The sound number target 0 drives; target i drives first sound + i.
Changing it fades out the sounds that were on.

OUTPUTS:

fade:
[[sound, level], ...] for every target whose zone you are in. It is
sparse: a sound is listed while its level is above 0, gets a single 0
the moment you leave its zone (so the sampler stops it), and is then
left out. With nothing in range and nothing just left, nothing is sent.
With several trackers the closest one sets each target's level. Levels
are rounded to 4 decimals.

nearest:
The index of the target with the highest level, or -1 when the tracker
is in no zone (one per tracker with several).

targets:
The whole set of targets, sent when it changes (capture, the targets
inlet, clear). Unset slots are nan.

READING THE DEMO:
make_point moves an x y point around the square from -1 to 1. As it
passes through a zone, fade lists that target's sound with a level that
rises to near 1 at the centre and falls away, then sends one last 0;
nearest switches between 0, 1 and 2, and is -1 between zones, as the
plot shows. Where two zones overlap (0.5 0.7 and 0 0 just do), fade
lists two sounds at once; drag radius up and the zones grow until they
overlap widely. Raise exponent and the levels stay
lower except near the centres.

RELATED:
np.distance_from_target gives the plain distance to one target.
proximity_trigger fires an event on arrival instead of a level.
polyphonic_sampler~ plays the sounds this fades. effort_fader sends
fades in the same sparse format."""

demo = [
    {'key': 'mp', 'init': 'p make_point', 'pos': (30, 62), 'w': 170, 'h': 70},
    {'key': 'c0', 'comment': True, 'text': 'an x y point wandering from -1 to 1',
     'pos': (30, 150)},
    {'key': 'px', 'init': 'np.proximity_to_target', 'pos': (30, 190), 'w': 240, 'h': 280,
     'props': {'target index': 0, 'radius': 0.5, 'exponent': 1.0, 'first sound': 0}},
    {'key': 'c1', 'comment': True, 'text': 'three targets saved; drag radius, exponent',
     'pos': (30, 195)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 490), 'w': 360, 'h': 42, 'props': LIST},
    {'key': 'c2', 'comment': True, 'text': 'fade: [[sound, level], ...]', 'pos': (30, 495)},
    {'key': 'i1', 'init': 'int', 'pos': (30, 550), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c3', 'comment': True, 'text': 'nearest target, -1 for none', 'pos': (30, 555)},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 610), 'w': 308, 'h': 178,
     'props': dict(PLOT(-1.5, 2.5, style='stair'), width=300)},
    {'key': 'c4', 'comment': True, 'text': 'nearest over time: 0, 1, 2, or -1 between zones',
     'pos': (30, 800)},
    {'key': 'l2', 'init': 'list', 'pos': (30, 840), 'w': 360, 'h': 42, 'props': LIST},
    {'key': 'c5', 'comment': True, 'text': 'targets: sent when the set changes (capture)',
     'pos': (30, 845)},
] + see_also(920, ['np.distance_from_target', 'proximity_trigger',
                   'polyphonic_sampler~', 'effort_fader'])
links = [('mp', 'point', 'px', 'input'), ('px', 'fade', 'l1', ''),
         ('px', 'nearest', 'i1', ''), ('px', 'nearest', 'pl', 'y'),
         ('px', 'targets', 'l2', '')]
p = build('np.proximity_to_target', 'np.proximity_to_target - closeness to several targets, 1 to 0',
          body, demo, links, demo_width=600, text_width=760, text_height=text_h(body),
          subpatch=POINT_SUB)
set_state(p, 'np.proximity_to_target', {'targets': TARGETS})
print(p)
