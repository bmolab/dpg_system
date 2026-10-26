"""Seed the help pages for the learning nodes (2026-10-26): one page per node.

    python3 dpg_system/help/tools/seed_torch_learning_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

t.param, t.optimize, quaternion_multiply. The t.optimize page builds its points in a
make_points subpatcher. Both learning demos train every frame from the moment the page
opens, so the loss can be watched falling. Not a make_* script: rebuild.py never runs
it, and it will not write over an existing page without --force. Treat the pages as
hand-made once written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from seed_numpy_more_help import LIST, MSG, C, N, see_also
from seed_scipy_help import S
from seed_torch_batch4_help import metro, held, W
from help_common import PLOT


PAGES = {}


def page(stem, title, body, demo, links, also, also_y, text_height=820, subpatch=None):
    PAGES[stem] = dict(title=title, body=body.strip('\n'), demo=demo + see_also(also_y, also),
                       links=links, text_height=text_height, subpatch=subpatch)


EVERY_FRAME = {'step on': 'every frame', 'learning rate': 0.05}


# -------------------------------------------------------------------- t.param
TARGET = '1 2 3'
KICK = '10 -4 0.5'

page('t.param', 't.param - numbers for t.optimize to tune',
"""
Holds a set of numbers that are to be learned rather than set by hand: the strength of
a correction, the offsets of a set of sensors, the coefficients of a curve. It sends
them as a tensor that remembers where it came from, so whatever is worked out from them
by torch nodes - a corrected pose, a fitted curve, and finally a loss that measures how
wrong it is - can be traced back to them. t.optimize does that tracing and nudges the
numbers to make the loss smaller.

The numbers can be kept in a .pt file: give a path and they are read back whenever the
patch is opened, so what was learned survives a restart. Save writes them; t.optimize
can also save them as it goes.

SYNTAX:
t.param <shape ...> <device> <path>
Default: one number, 0, on the cpu. t.param 3 is three numbers; t.param 18 5 a table
of 18 rows of 5.

INPUTS and PARAMETERS:

in:
Anything sends the numbers. A list or tensor of numbers sets them first (and gives them
its shape, if that differs), then sends them. Wire t.optimize's 'again' outlet here.

learn (checkbox, also an inlet):
On (the default), t.optimize may change these numbers. Off, they are held still while
others learn.

save (button):
Writes the numbers to the path, or asks where if there is none.

reset (button):
Puts every number back to the initial value.

initial value:
An option, under the node. What reset puts back; 0 by default.

low, high:
Options. If high is above low, the numbers are kept between them: they ease toward
either end rather than being cut off there, so they can approach a limit but never
pass it. Both 0 (the default): no limits.

device:
An option. Where the numbers live: cpu, or mps for the GPU. Everything worked out from
them follows.

path:
An option. The .pt file the numbers are kept in. If it exists when the patch opens, or
when a path is entered, the numbers are read from it.

OUTPUTS:

values:
The numbers, as a tensor. While learning, it carries the trail back to this node;
while not, it is plain values and costs nothing extra downstream.

READING THE DEMO:
Three numbers learn to match 1 2 3: t.mse_loss measures how far they are from the
target, and t.optimize moves them closer every frame, asking for a new loss through
'again'. Click the message to throw them to 10 -4 0.5 and watch them find their way
back.

RELATED:
t.optimize does the learning. t.mse_loss and t.l1_loss measure how wrong a result is.
t.save and t.load keep any tensor in a file.
""",
held(53) + [
    C('c1', 'click: throw the numbers off', 28, 160),
    N('mk', 'message', 28, 195, W(KICK), 42, MSG(KICK)),
    C('c2', 'three numbers', 260, 53),
    N('pa', 't.param 3', 260, 92, 170, 150),
    C('c3', 'the target, held', 260, 300),
    N('mt', 'message', 260, 335, W(TARGET), 42, MSG(TARGET)),
    C('c4', 'how far off', 450, 53),
    N('ls', 't.mse_loss', 450, 92, 150, 90),
    C('c5', 'nudge them, every frame', 450, 230),
    N('op', 't.optimize', 450, 265, 170, 150, EVERY_FRAME),
    C('c6', 'the numbers now', 640, 53),
    N('l1', 'list', 640, 92, 160, 42, LIST),
],
[('lb', 'out', 'mt', ''), ('mt', 'message out', 'ls', 'target'), ('mk', 'message out', 'pa', 'in'),
 ('pa', 'values', 'ls', 'tensor in'), ('pa', 'values', 'l1', ''), ('ls', 'loss', 'op', 'loss'),
 ('op', 'again', 'pa', 'in')],
['t.optimize', 't.mse_loss', 't.save', 't.load'], 480, 1100)


# ----------------------------------------------------------------- t.optimize
# Inside make_points: 50 positions from -1 to 1 and the line 3x + 0.5 through them.
POINTS_SUB = {
    'name': 'make_points',
    'host': 'mp',
    'demo': [
        S('lb', 'load_bang', 40, 40, 88, 46),
        S('ls', 't.linspace -1 1 50', 40, 120, 190, 150),
        S('cl', 'comment 50 positions, -1 to 1', 250, 135, 260, 30),
        S('m1', '* 3', 40, 300, 129, 70),
        S('a1', '+ 0.5', 40, 390, 129, 70),
        S('c1', 'comment the line 3x + 0.5', 190, 405, 230, 30),
        S('ox', 'out x', 300, 300, 150, 80),
        S('oy', 'out y', 40, 490, 150, 80),
    ],
    'links': [('lb', 'out', 'ls', '###input'), ('ls', 'linspace out', 'ox', 'x'),
              ('ls', 'linspace out', 'm1', 'in'), ('m1', 'result', 'a1', 'in'), ('a1', 'result', 'oy', 'y')],
}

page('t.optimize', 't.optimize - learn the numbers that make a loss smallest',
"""
Tunes t.param numbers by gradient descent. Build, from t.param and ordinary torch
nodes, a measure of how wrong the result is - a loss, one number, smaller is better -
and send it here. t.optimize traces it back to every t.param it was worked out from,
finds which way each number should move to shrink it, and moves them a little. Repeat,
and the numbers settle where the loss is smallest.

Nothing needs to be wired back to the t.param nodes except 'again'. After a step the
numbers have changed, so the loss must be worked out afresh before the next step:
'again' asks for that. Wire it to the t.param nodes - to the ones on cold inlets first -
or to whatever starts the calculation. However many steps are asked for at once, they
run one after another, not nested.

The trail back from the loss must stay in torch: a node that turns it into a plain
number, a list or a numpy array cuts it, and t.optimize says so. Most t. nodes keep it,
as do + - * / and the quaternion conversions, quaternion_multiply and swing_twist.

SYNTAX:
t.optimize <learning rate>
Default: learning rate 0.01.

INPUTS and PARAMETERS:

loss:
The loss. What happens next depends on 'step on'.

train (checkbox, also an inlet):
On (the default), it learns. Off, it only shows the loss, and the t.param nodes stop
keeping their trail - pure use of what was learned, at no extra cost.

step (button):
In 'bang' mode: ask for a fresh loss through 'again' and take the steps.

reset (button):
Forgets the optimizer's momentum and the step count. The numbers are kept; t.param's
own reset puts those back.

step on:
An option. each loss (the default): every loss that arrives is stepped on. bang: only
the step button (or a bang) steps. every frame: it steps by itself every frame, asking
for each loss through 'again'.

steps:
An option. Steps per loss, bang or frame: 1 by default. More learns faster per frame,
at more work per frame.

optimizer:
An option: adam (the default), adamw, sgd or rmsprop.

learning rate:
An option. How big a step is. Too small learns slowly; too big overshoots and jitters.

clip gradient:
An option. Above 0, a step is never steeper than this, which steadies a loss with
occasional wild frames.

autosave every:
An option. Above 0, every this many steps the t.param nodes that have a path write
their numbers.

OUTPUTS:

loss:
The loss as a plain number, every time one arrives.

steps:
How many steps have been taken.

gradient size:
How steeply the loss changes with the numbers - near 0 when they have settled.

again:
A bang when a fresh loss is needed.

READING THE DEMO:
The make_points subpatcher makes 50 points on the line 3x + 0.5. Two t.param numbers,
a slope and an offset, start at 0; the * and + nodes draw their line through the same
x, and t.mse_loss measures how far it lies from the points. t.optimize steps every
frame: the loss falls to 0 as the slope reaches 3 and the offset 0.5. Click reset on
either t.param to watch it learn again.

RELATED:
t.param holds the numbers. t.mse_loss and t.l1_loss measure how wrong a result is.
quaternion_multiply and swing_twist keep the trail through rotations.
""",
[
    C('c1', 'points on a line', 28, 53),
    N('mp', 'p make_points', 28, 92, 170, 70),
    C('c2', 'slope', 230, 53),
    N('pa', 't.param', 230, 92, 150, 150),
    C('c3', 'offset', 230, 290),
    N('pb', 't.param', 230, 325, 150, 150),
    C('c4', 'their line', 420, 53),
    N('mu', '*', 420, 92, 129, 70),
    N('ad', '+', 420, 200, 129, 70),
    C('c5', 'how far off', 600, 53),
    N('ls', 't.mse_loss', 600, 92, 150, 90),
    C('c6', 'learn, every frame', 600, 230),
    N('op', 't.optimize', 600, 265, 170, 150, EVERY_FRAME),
    C('c7', 'the loss', 420, 520),
    N('pl', 'plot', 420, 555, 228, 160, PLOT(0.0, 4.0, 200)),
    C('c8', 'slope and offset now', 28, 200),
    N('la', 'list', 28, 235, 140, 42, LIST),
    N('lb2', 'list', 28, 295, 140, 42, LIST),
],
[('mp', 'x', 'mu', 'operand'), ('mp', 'y', 'ls', 'target'),
 ('op', 'again', 'pb', 'in'), ('op', 'again', 'pa', 'in'),
 ('pb', 'values', 'ad', 'operand'), ('pa', 'values', 'mu', 'in'), ('mu', 'result', 'ad', 'in'),
 ('ad', 'result', 'ls', 'tensor in'), ('ls', 'loss', 'op', 'loss'), ('op', 'loss', 'pl', 'y'),
 ('pa', 'values', 'la', ''), ('pb', 'values', 'lb2', '')],
['t.param', 't.mse_loss', 'quaternion_multiply', 'swing_twist'], 760, 1500, subpatch=POINTS_SUB)


# --------------------------------------------------------- quaternion_multiply
TURN = '0.7071 0 0 0.7071'

page('quaternion_multiply', 'quaternion_multiply - one rotation after another',
"""
Composes two rotations into one: the product a * b turns by b first, then by a. Both
are quaternions, scalar first: w x y z. Either can be inverted before multiplying, so
the same node takes a rotation away as well as adding one.

This is how an offset is applied: a calibration correction times a sensor's reading
gives the corrected reading. Many at once work alike: a row of one offset per sensor
times a whole take of frames corrects every frame of every sensor. With tensors in, the
result stays on their device, and a t.param offset keeps its trail for t.optimize.

The product is not flipped to keep its w positive, so a smoothly turning stream stays
smooth.

SYNTAX:
quaternion_multiply

INPUTS and PARAMETERS:

a:
The rotation applied second. Sends the product.

b:
The rotation applied first, held. Lists, arrays and tensors broadcast against a: one
quaternion against many, or (sensors, 4) against (frames, sensors, 4).

invert a, invert b:
Options, under the node. Use the opposite rotation of that side: invert a with b gives
the turn from a to b.

OUTPUTS:

a * b:
The combined rotation, as a tensor.

READING THE DEMO:
0.7071 0 0 0.7071 is a quarter turn about z. A quarter turn after a quarter turn is a
half turn: 0 0 0 1.

RELATED:
quaternion_relative gives the turn from one rotation to another. quaternion_norm keeps
a quaternion unit length. swing_twist splits a rotation into pointing and twist.
""",
metro() + held() + [
    C('c1', 'a quarter turn about z', 180, 53),
    N('ma', 'message', 180, 92, W(TURN), 42, MSG(TURN)),
    C('c2', 'and another, held', 180, 175),
    N('mb', 'message', 180, 210, W(TURN), 42, MSG(TURN)),
    C('c3', 'one after the other', 440, 53),
    N('qm', 'quaternion_multiply', 440, 92, 200, 90),
    C('c4', 'a half turn', 680, 53),
    N('l1', 'list', 680, 92, 180, 42, LIST),
],
[('lb', 'out', 'mb', ''), ('mb', 'message out', 'qm', 'b'), ('mt', '', 'ma', ''),
 ('ma', 'message out', 'qm', 'a'), ('qm', 'a * b', 'l1', '')],
['quaternion_relative', 'quaternion_norm', 'swing_twist', 't.optimize'], 330, 900)


def main():
    args = sys.argv[1:]
    force = '--force' in args
    wanted = [a for a in args if not a.startswith('--')] or list(PAGES)
    for stem in wanted:
        p = PAGES[stem]
        path = os.path.join(HELP_DIR, stem + '_help.json')
        if os.path.exists(path) and not force:
            print('exists, skipped (use --force):', path)
            continue
        print(build(stem, p['title'], p['body'], p['demo'], p['links'], demo_width=880,
                    text_width=760, text_height=p['text_height'], subpatch=p['subpatch']))


if __name__ == '__main__':
    main()
