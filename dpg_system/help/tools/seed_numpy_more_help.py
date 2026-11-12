"""Seed the help pages for the numpy nodes added 2026-10-24: one page per node.

    python3 dpg_system/help/tools/seed_numpy_more_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

atan2, np.logical_and, np.logical_or, np.logical_xor, np.logical_not, np.isnan,
np.isfinite, np.nan_to_num, np.percentile, np.histogram, np.interp, np.convolve,
np.correlate, np.linalg.pinv, np.linalg.lstsq, np.einsum. (log is in
derive_math_single_help.py, with the other single-input functions.)

Deliberately not a make_* script: the demos flow left to right with each
comment above its node, which plain relayout would move into a gutter, so
rebuild.py must not run it. Treat the pages as hand-made once written. It
refuses to write over a page that already exists unless given --force, so
it cannot undo later work on them by hand.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from help_common import SIG, PLOT


LIST = {'text in': '', 'font size': '24'}


def MSG(text):
    return {'text in': text, 'font size': '24'}


def HM(width, height, count, miny=0.0, maxy=1.0, fmt='%.2f', color='viridis'):
    """Property block for a heat_map that draws a whole array as one picture."""
    return {'color': color, 'width': width, 'height': height, 'sample count': count,
            'min y': miny, 'max y': maxy, 'update_mode': 'heat_map',
            'number format': fmt}


def BARS(count, maxy, width=520, height=200, style='bar'):
    """Property block for a plot that draws one array across its width."""
    return {'color': 'none', 'width': width, 'height': height, 'style': style,
            'update style': 'buffer holds one sample of input', 'sample count': count,
            'min x': 0.0, 'max x': float(count), 'min y': 0.0, 'max y': maxy}


def C(key, text, x, y):
    return {'key': key, 'comment': True, 'text': text, 'pos': (x, y)}


def N(key, init, x, y, w=150, h=70, props=None):
    d = {'key': key, 'init': init, 'pos': (x, y), 'w': w, 'h': h}
    if props:
        d['props'] = props
    return d


def see_also(y, stems, x=30, right=820):
    """A SEE ALSO row: one help_link per related page, left to right."""
    row = [C('sa_c', 'SEE ALSO - click to open', x, y)]
    y += 34
    x0 = x
    for s in stems:
        w = int(len(s) * 7.7) + 30
        if x > x0 and x + w > right:
            x, y = x0, y + 44
        row.append({'key': 'sa_' + s, 'init': 'help_link ' + s, 'pos': (x, y),
                    'w': w, 'h': 30})
        x += w + 14
    return row


PAGES = {}


def page(stem, title, body, demo, links, also, also_y, text_height=820):
    PAGES[stem] = dict(title=title, body=body.strip('\n'), demo=demo + see_also(also_y, also),
                       links=links, text_height=text_height)


# ------------------------------------------------------------------- atan2
page('atan2', 'atan2 - the angle of a point',
"""
Gives the angle of the point (x, y): the direction you would face, standing at the
origin, to look at it. The angle is measured from the positive x axis,
counter-clockwise, from -180 to 180 degrees.

atan of y / x tries to do the same and cannot: (1, 1) and (-1, -1) both have y / x =
1, so atan gives 45 for both, though one is up-right and the other down-left. atan2
is given y and x separately, so it knows which quadrant the point is in, and it
copes with x = 0 (straight up or down), where y / x has no answer.

Use it to get a heading from a velocity, the angle of a joystick or a hand around
a centre, the direction from one point to another (atan2 of their difference), or
the phase of a point going round a circle.

SYNTAX:
atan2 <x: float>
Default: x 1.

EXAMPLE:
atan2 -1

INPUTS and PARAMETERS:

y:
The vertical part of the point. A number, list, array or tensor - each element is
paired with the matching element of x. Every input sends: the only inlet that makes
the node send.

x (float box, also an inlet):
The horizontal part. A single number pairs with every y; a list, array or tensor
must have the same shape as y. It does not send by itself: send x first, then y.

degrees:
A checkbox property, on by default. Off gives radians (-pi to pi).

OUTPUTS:

angle:
The angle in degrees (or radians), the same shape as y.

ZERO:
atan2 of y 0, x 0 is 0: the origin has no direction, and 0 is what it says.

READING THE DEMO:
Top: cos and sin of a steadily turning angle trace a circle; atan2 recovers the
angle, which climbs steadily from -180 to 180 and then jumps back - the same
direction, just counted from the other side.

Bottom: the point x -1, y 1 is up and to the left, 135 degrees. atan of 1 / -1 sees
only -1 and answers -45, which is down and to the right.

RELATED:
atan and tan work on one ratio; sin and cos go the other way, from an angle to a
point. np.linalg.norm gives the point's distance from the origin, which with atan2
is the whole of the point in polar form.
""",
[
    C('c1', 'a point going round a circle every 4 seconds', 28, 53),
    N('sx', 'signal 4.0 cos', 28, 92, 129, 78, SIG('cos', 4.0)),
    N('sy', 'signal 4.0 sin', 180, 92, 129, 78, SIG('sin', 4.0)),
    C('c2', 'x from cos, y from sin', 340, 53),
    N('at', 'atan2', 340, 92, 140, 90),
    C('c3', 'its angle: a steady climb', 520, 53),
    N('pl', 'plot', 520, 92, 208, 178, PLOT(-180.0, 180.0, 300)),
    C('c4', 'once at load', 28, 335),
    N('lb', 'load_bang', 28, 370, 88, 46),
    C('c5', 'y 1 ...', 140, 335),
    N('m1', 'message', 140, 370, 60, 42, MSG('1')),
    C('c6', '... and x -1: up and left', 330, 335),
    N('a2', 'atan2 -1', 330, 370, 140, 90),
    C('c7', 'degrees', 520, 335),
    N('l1', 'list', 520, 370, 120, 42, LIST),
    C('c8', 'atan of y / x = -1 cannot tell which side', 140, 490),
    N('m2', 'message', 140, 525, 60, 42, MSG('-1')),
    N('a3', 'atan', 330, 525, 129, 70),
    N('l2', 'list', 520, 525, 120, 42, LIST),
],
[('sx', '', 'at', 'x'), ('sy', '', 'at', 'y'), ('at', 'angle', 'pl', 'y'),
 ('lb', 'out', 'm1', ''), ('lb', 'out', 'm2', ''),
 ('m1', 'message out', 'a2', 'y'), ('a2', 'angle', 'l1', ''),
 ('m2', 'message out', 'a3', 'in'), ('a3', 'out', 'l2', '')],
['atan', 'tan', 'sin', 'cos', 'np.linalg.norm'], 640, 900)


# ---------------------------------------------------------- logical and/or/xor
LOGIC = {
    'and': ('both are true', '1 0 0 0', 'true only where A AND B are both true',
            """
Use it to combine conditions that must all hold: inside a range (above the low limit
and below the high one), a hand both raised and still, a frame both bright and
moving. Chain two to require three conditions.""",
            'np.logical_or', 'np.logical_xor'),
    'or': ('either is true', '1 1 1 0', 'true where A OR B (or both) is true',
           """
Use it to combine conditions any one of which is enough: outside a range (below the
low limit or above the high one), either hand raised, any of several triggers.""",
           'np.logical_and', 'np.logical_xor'),
    'xor': ('exactly one is true', '0 1 1 0', 'true where A and B DIFFER',
            """
Use it to find disagreement: where two masks differ, which cells changed between
two thresholded frames, which of two sensors is reporting alone. It is true where
one is true and the other is not - "one or the other, but not both".""",
            'np.logical_and', 'np.logical_or'),
}

for op, (gist, result, says, use, sib1, sib2) in LOGIC.items():
    stem = 'np.logical_' + op
    page(stem, stem + ' - true where ' + gist,
f"""
Combines two arrays of true and false element by element, giving true {says[5:]}.

A     B     {op}
1     1     {'1' if op != 'xor' else '0'}
1     0     {'0' if op == 'and' else '1'}
0     1     {'0' if op == 'and' else '1'}
0     0     0
{use}

The inputs need not be true and false: any number other than 0 counts as true, and
0 as false, so 2.5 and -1 are both true. The arrays from the comparison nodes (>, <,
==) are ready to use as they are.

SYNTAX:
{stem}

INPUTS and PARAMETERS:

in:
The first array, A (a list is turned into one). Every input combines and sends -
the only inlet that makes the node send.

in 2:
The second array, B. It is held, and used with every A that follows; it does not
send by itself, so send it first. It must have the same shape as A, or a shape that
broadcasts against it (a single value goes with every element).

output_type:
An option, under the node: bool (the default), int (1 and 0) or float (1.0 and
0.0). Choose int or float to plot the result, or to multiply with it.

OUTPUTS:

result:
True {says[5:]}, the shape of A and B together.

READING THE DEMO:
A is 1 1 0 0 and B is 1 0 1 0 - every pairing of true and false once - so the result,
{result}, is the whole table above in one go (output_type is set to int here, so it
shows 1s and 0s). A metro re-sends A five times a second;
type other 1s and 0s into either message and click it.

RELATED:
{sib1} and {sib2} combine the same way with other rules; np.logical_not turns true
into false. np.any and np.all ask whether ANY or ALL of one array is true.
""",
    [
        C('c1', 'five times a second', 28, 53),
        N('mt', 'metro 200', 28, 92, 129, 70, {'on': True, 'period': 200.0, 'units': 'milliseconds'}),
        C('c2', 'A', 180, 53),
        N('ma', 'message', 180, 92, 140, 42, MSG('1 1 0 0')),
        C('c3', 'once at load', 28, 175),
        N('lb', 'load_bang', 28, 210, 88, 46),
        C('c4', 'B, held', 180, 175),
        N('mb', 'message', 180, 210, 140, 42, MSG('1 0 1 0')),
        C('c5', 'A ' + op + ' B', 370, 53),
        N('op', stem, 370, 92, 170, 90, {'output_type': 'int'}),
        C('c6', 'result', 590, 53),
        N('ls', 'list', 590, 92, 220, 42, LIST),
    ],
    [('mt', '', 'ma', ''), ('ma', 'message out', 'op', 'in'),
     ('lb', 'out', 'mb', ''), ('mb', 'message out', 'op', 'in 2'),
     ('op', 'result', 'ls', '')],
    [sib1, sib2, 'np.logical_not', 'np.any', 'np.all', 'greater'], 340, 760)


page('np.logical_not', 'np.logical_not - true becomes false',
"""
Turns every true into false and every false into true, element by element.

Use it to flip a mask: the cells NOT above a threshold, the frames where a hand is
NOT moving, everything outside a selected region. With np.where it selects the
other branch; with np.logical_and it subtracts one condition from another (A and
not B: true in A but not in B).

Any number other than 0 counts as true and 0 as false, so not of 2.5 is false and not
of 0 is true.

SYNTAX:
np.logical_not

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input is flipped and sent.

output_type:
An option, under the node: bool (the default), int (1 and 0) or float (1.0 and
0.0). Choose int or float to plot the result, or to multiply with it.

OUTPUTS:

result:
The opposite of each element, the same shape as the input.

READING THE DEMO:
Top: 1 0 0 1 becomes 0 1 1 0. Bottom: 0 2.5 -1 0 becomes 1 0 0 1 - only the zeros
were false (output_type is set to int here, so they show as 1s and 0s). A metro
re-sends both five times a second; edit a message and click it.

RELATED:
np.logical_and, np.logical_or and np.logical_xor combine two arrays. != 0 makes a
mask from numbers; np.where picks between two arrays by a mask.
""",
[
    C('c1', 'five times a second', 28, 53),
    N('mt', 'metro 200', 28, 92, 129, 70, {'on': True, 'period': 200.0, 'units': 'milliseconds'}),
    C('c2', 'true and false', 180, 53),
    N('m1', 'message', 180, 92, 140, 42, MSG('1 0 0 1')),
    C('c3', 'flip each', 370, 53),
    N('n1', 'np.logical_not', 370, 92, 170, 70, {'output_type': 'int'}),
    C('c4', 'result', 590, 53),
    N('l1', 'list', 590, 92, 220, 42, LIST),
    C('c5', 'any number but 0 is true', 180, 190),
    N('m2', 'message', 180, 225, 160, 42, MSG('0 2.5 -1 0')),
    N('n2', 'np.logical_not', 370, 225, 170, 70, {'output_type': 'int'}),
    N('l2', 'list', 590, 225, 220, 42, LIST),
],
[('mt', '', 'm1', ''), ('mt', '', 'm2', ''),
 ('m1', 'message out', 'n1', 'in'), ('n1', 'result', 'l1', ''),
 ('m2', 'message out', 'n2', 'in'), ('n2', 'result', 'l2', '')],
['np.logical_and', 'np.logical_or', 'np.logical_xor', 'np.where', 'not_equal'], 350, 700)


# ------------------------------------------------------------ isnan/isfinite
BAD = '1 nan 3 inf -inf 0'

page('np.isnan', 'np.isnan - where the values are not numbers',
"""
Marks the elements that are nan ("not a number"), and counts them.

nan is what a calculation gives when it has no answer: 0 divided by 0, the mean of
nothing, the square root of a negative in some libraries, a sensor that dropped out
and was filled with a placeholder. It spreads - anything added to nan is nan - so a
single one can quietly turn a sum, a mean or a whole filter's output into nan.
np.isnan finds them so you can mask them, replace them, or just notice them.

Infinity (inf, -inf) is NOT nan: it is a number, only too large. np.isfinite
catches both kinds.

SYNTAX:
np.isnan

INPUTS and PARAMETERS:

in:
The array (a list is turned into one; a message can say nan and inf). Every input
is tested and sent.

OUTPUTS:

mask:
True where the element is nan, the same shape as the input.

count:
How many elements are nan. Sent before the mask.

READING THE DEMO:
1 nan 3 inf -inf 0: only the second value is nan, so the mask is false true false
false false false and the count is 1. A metro re-sends the message five times a
second; type nan somewhere else and click it.

RELATED:
np.isfinite marks every value that is an ordinary number - neither nan nor
infinite. np.nan_to_num replaces nan and infinity with numbers you choose.
np.nanmean, np.nanmedian and np.nansum skip the nans instead.
""",
[
    C('c1', 'five times a second', 28, 53),
    N('mt', 'metro 200', 28, 92, 129, 70, {'on': True, 'period': 200.0, 'units': 'milliseconds'}),
    C('c2', 'a list with gaps in it', 180, 53),
    N('m1', 'message', 180, 92, 240, 42, MSG(BAD)),
    C('c3', 'test each', 460, 53),
    N('n1', 'np.isnan', 460, 92, 150, 90),
    C('c4', 'nan here', 28, 215),
    N('l1', 'list', 28, 250, 420, 42, LIST),
    C('c5', 'how many', 460, 215),
    N('l2', 'list', 460, 250, 80, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'n1', 'in'),
 ('n1', 'mask', 'l1', ''), ('n1', 'count', 'l2', '')],
['np.isfinite', 'np.nan_to_num', 'np.nanmean', 'np.nansum'], 340, 760)


page('np.isfinite', 'np.isfinite - where the values are ordinary numbers',
"""
Marks the elements that are ordinary, finite numbers - not nan ("not a number") and
not infinite - and counts them.

Infinity comes from dividing by zero, from exp of something large, from log of 0;
nan from 0 divided by 0 and other calculations with no answer. Either one wrecks
what follows: a plot's range, a mean, a filter's memory. np.isfinite gives a mask
of the values you can trust, ready for np.where (keep the good, replace the rest)
or np.all (is this whole frame usable?).

SYNTAX:
np.isfinite

INPUTS and PARAMETERS:

in:
The array (a list is turned into one; a message can say nan and inf). Every input
is tested and sent.

OUTPUTS:

mask:
True where the element is an ordinary number, the same shape as the input.

count:
How many elements are ordinary numbers. Sent before the mask.

READING THE DEMO:
1 nan 3 inf -inf 0: nan, inf and -inf all fail, so the mask is true false true
false false true and the count is 3. A metro re-sends the message five times a
second; edit it and click it.

RELATED:
np.isnan marks only the nans. np.nan_to_num replaces nan and infinity with numbers
you choose. np.all of the mask says whether every value is good.
""",
[
    C('c1', 'five times a second', 28, 53),
    N('mt', 'metro 200', 28, 92, 129, 70, {'on': True, 'period': 200.0, 'units': 'milliseconds'}),
    C('c2', 'a list with gaps in it', 180, 53),
    N('m1', 'message', 180, 92, 240, 42, MSG(BAD)),
    C('c3', 'test each', 460, 53),
    N('n1', 'np.isfinite', 460, 92, 150, 90),
    C('c4', 'ordinary numbers here', 28, 215),
    N('l1', 'list', 28, 250, 420, 42, LIST),
    C('c5', 'how many', 460, 215),
    N('l2', 'list', 460, 250, 80, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'n1', 'in'),
 ('n1', 'mask', 'l1', ''), ('n1', 'count', 'l2', '')],
['np.isnan', 'np.nan_to_num', 'np.all', 'np.where'], 340, 700)


page('np.nan_to_num', 'np.nan_to_num - replace nan and infinity',
"""
Replaces every nan ("not a number") and every infinity with an ordinary number of
your choosing, leaving all the other values as they are.

One nan or infinity can wreck everything after it: a plot loses its range, a sum
or mean turns to nan, a filter remembers the bad value for ever. Put np.nan_to_num
in front of anything that cannot cope - a quick, blunt repair. When the gaps
matter, np.isnan or np.isfinite will show you where they are first.

Here all three default to 0. NumPy's own np.nan_to_num turns infinity into the
largest number a float can hold instead, which is rarely useful in a patch; set
posinf and neginf to large values if you want that.

SYNTAX:
np.nan_to_num <nan: float>
Default: nan 0.

EXAMPLE:
np.nan_to_num -1

INPUTS and PARAMETERS:

in:
The array (a list is turned into one; a message can say nan and inf). Every input
is repaired and sent. Whole-number arrays cannot hold nan or infinity and go
through unchanged.

nan (float box, also an inlet):
What each nan becomes.

posinf (float box, also an inlet):
What inf (positive infinity) becomes.

neginf (float box, also an inlet):
What -inf becomes.

OUTPUTS:

out:
The input with nan and infinity replaced, the same shape.

READING THE DEMO:
1 nan 3 inf -inf 0 becomes 1 0 3 0 0 0 with the defaults. The second node marks
the gaps instead: -1 for nan, 999 for inf and -999 for -inf. A metro re-sends the
message five times a second; change the numbers in the second node.

RELATED:
np.isnan and np.isfinite find the gaps rather than filling them. np.clip limits
ordinary numbers to a range (it leaves nan as nan).
""",
[
    C('c1', 'five times a second', 28, 53),
    N('mt', 'metro 200', 28, 92, 129, 70, {'on': True, 'period': 200.0, 'units': 'milliseconds'}),
    C('c2', 'a list with gaps in it', 180, 53),
    N('m1', 'message', 180, 92, 240, 42, MSG(BAD)),
    C('c3', 'every gap to 0', 400, 53),
    N('n1', 'np.nan_to_num', 400, 92, 160, 140),
    C('c4', 'repaired', 570, 53),
    N('l1', 'list', 570, 92, 250, 42, LIST),
    C('c5', 'or mark each kind of gap', 400, 270),
    N('n2', 'np.nan_to_num -1', 400, 305, 160, 140,
      {'nan': -1.0, 'posinf': 999.0, 'neginf': -999.0}),
    C('c6', 'marked', 570, 270),
    N('l2', 'list', 570, 305, 250, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'n1', 'in'), ('m1', 'message out', 'n2', 'in'),
 ('n1', 'out', 'l1', ''), ('n2', 'out', 'l2', '')],
['np.isnan', 'np.isfinite', 'np.clip', 'np.nanmean'], 490, 900)


# -------------------------------------------------------------- percentile
page('np.percentile', 'np.percentile - the value a share of the data lies below',
"""
Gives the value below which a given percentage of the data lies. percentile 50 is the
median: half the values are below it. percentile 90 has nine values in ten below it;
percentile 0 is the smallest value and 100 the largest.

Percentiles describe a spread of values without being thrown by a few wild ones.
The mean of 1 to 9 plus one glitch of 100 is 14.5, a value none of the data is
near; the median is 5.5, and percentile 90 is still only 18. Use them for a robust
range (percentile 5 to 95 ignores the extremes), an automatic threshold (call the
top 10% of activity "loud"), or to normalize a signal whose peaks are unreliable.

SYNTAX:
np.percentile <percentile: float> <axis: int>
Defaults: percentile 50, axis None (the whole array).

EXAMPLE:
np.percentile 90

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input is measured and sent - the only
inlet that makes the node send.

percentile (float box, also an inlet):
Which percentile, 0 to 100 (values outside are clamped). Send it a list - 5 50 95 -
to get several at once. It does not send by itself.

axis (menu, also an inlet):
None takes the percentile over every element. A number takes it along that axis: for
a 2-D array, 0 gives one value per column, 1 one per row.

method:
An option, under the node. When the percentile falls between two values, linear
(the default) blends them; lower, higher and nearest pick one of the two values;
midpoint takes their average.

ignore nan:
An option, under the node. Off, a single nan in the data makes the result nan. On,
the nans are skipped.

OUTPUTS:

result:
The percentile: a single value, or one per percentile asked for, or an array of
them along the axis.

READING THE DEMO:
Top: ten values, one of them a glitch. The mean is dragged up to 14.5; percentile 50
- the median - stays at 5.5 among the ordinary values.

Bottom: 1000 random values, evenly spread from 0 to 1. Percentile 90 comes out near
0.9, since nine values in ten are below that (round trims it to two decimal places for
reading). The second node was sent 5 50 95 at
load, so it gives three percentiles at once: about 0.05 0.5 0.95. The metro makes a
new random array twice a second; the answers wobble a little each time.

RELATED:
np.median is percentile 50. np.amin and np.amax are percentile 0 and 100. np.mean
and np.std summarize the same data in a way that outliers do pull around;
np.robust_mean is a mean that resists them. np.histogram shows the whole spread.
""",
[
    C('c1', 'five times a second', 28, 53),
    N('mt', 'metro 200', 28, 92, 129, 70, {'on': True, 'period': 200.0, 'units': 'milliseconds'}),
    C('c2', 'nine values and a glitch', 180, 53),
    N('m1', 'message', 180, 92, 280, 42, MSG('1 2 3 4 5 6 7 8 9 100')),
    C('c3', 'the mean is pulled up', 500, 53),
    N('mn', 'np.mean', 500, 92, 150, 70),
    N('l1', 'list', 680, 92, 120, 42, LIST),
    C('c4', 'percentile 50 is not', 500, 180),
    N('p1', 'np.percentile 50', 500, 215, 170, 120),
    N('l2', 'list', 680, 215, 120, 42, LIST),
    C('c5', 'twice a second', 28, 375),
    N('m2', 'metro 500', 28, 410, 129, 70, {'on': True, 'period': 500.0, 'units': 'milliseconds'}),
    C('c6', '1000 random values', 180, 375),
    N('rn', 'np.rand 1000', 180, 410, 150, 110),
    C('c7', 'nine in ten are below', 370, 375),
    N('p2', 'np.percentile 90', 370, 410, 170, 120),
    C('c9', 'to 2 places', 570, 375),
    N('r1', 'round', 570, 410, 100, 70, {'decimals': 2}),
    N('l3', 'list', 690, 410, 130, 42, LIST),
    C('c8', 'several at once', 180, 565),
    N('lb', 'load_bang', 28, 600, 88, 46),
    N('mq', 'message', 180, 600, 140, 42, MSG('5 50 95')),
    N('p3', 'np.percentile', 370, 600, 170, 120),
    N('r2', 'round', 570, 600, 100, 70, {'decimals': 2}),
    N('l4', 'list', 690, 600, 130, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'mn', 'in'), ('m1', 'message out', 'p1', 'in'),
 ('mn', 'mean', 'l1', ''), ('p1', 'result', 'l2', ''),
 ('m2', '', 'rn', ''), ('rn', 'random array', 'p2', 'in'), ('rn', 'random array', 'p3', 'in'),
 ('p2', 'result', 'r1', 'in'), ('r1', 'result', 'l3', ''), ('lb', 'out', 'mq', ''), ('mq', 'message out', 'p3', 'percentile'),
 ('p3', 'result', 'r2', 'in'), ('r2', 'result', 'l4', '')],
['np.median', 'np.mean', 'np.robust_mean', 'np.histogram', 'np.amax'], 770, 1100)


# --------------------------------------------------------------- histogram
page('np.histogram', 'np.histogram - how many values fall in each range',
"""
Divides a range of values into equal bins and counts how many of the input's values
fall into each. The counts are the shape of the data: where it piles up, where it
is sparse, whether it has one peak or two, how far it spreads.

Use it to see what a sensor actually reports over time, to find the levels a
signal sits at, to choose a threshold between two clusters, or as a feature in its
own right - the distribution of speeds in a movement, of brightness in an image.

SYNTAX:
np.histogram <bins: int> <min: float> <max: float>
Defaults: 10 bins, auto range. Giving min and max turns auto range off.

EXAMPLE:
np.histogram 20 0 4

INPUTS and PARAMETERS:

in:
The values: a list or array of any shape (every element is counted). nan and
infinity belong in no bin and are left out. Every input is counted and sent.

bins (int box, also an inlet):
How many bins.

min, max (float boxes, also inlets):
The range the bins cover, when auto range is off. Values outside it are not
counted; a value exactly at max goes in the last bin.

auto range:
An option, under the node. On, the bins span the input's own smallest to largest
value, so they move with the data; off, they stay at min to max, so one count can
be compared with the next.

density:
An option, under the node. On, the counts are scaled so that the bars' total AREA is
1 - a probability density, comparable between inputs of different sizes.

OUTPUTS:

counts:
One count per bin.

bin edges:
Where the bins start and end: one more value than there are bins. Sent before the
counts.

READING THE DEMO:
Top: four random numbers, each anywhere from 0 to 1, are added together a thousand
times. Each sum could be anything from 0 to 4, but most land near 2 - it takes four
small numbers to make a sum near 0, and many different mixtures make one near 2.
The histogram shows the pile in the middle: twenty bins from 0 to 4, redrawn five
times a second.

Bottom: 1 1 2 2 2 3 10 in 3 bins with auto range. The range is 1 to 10, so the edges
are 1 4 7 10: six values in the first bin, none in the middle, one - the 10 - in the
last.

RELATED:
np.percentile reads a single point off the same distribution. t.histc is the
PyTorch version. np.rand makes evenly spread random values; adding several of them
together is a quick bell curve.
""",
[
    C('c1', 'five times a second', 28, 53),
    N('mt', 'metro 200', 28, 92, 129, 70, {'on': True, 'period': 200.0, 'units': 'milliseconds'}),
    C('c2', '4 x 1000 random values', 180, 53),
    N('rn', 'np.rand 4 1000', 180, 92, 150, 130),
    C('c3', 'add each four', 360, 53),
    N('sm', 'np.sum 0', 360, 92, 150, 70, {'axis': '0'}),
    C('c4', 'count them in 20 bins', 540, 53),
    N('hg', 'np.histogram 20 0 4', 540, 92, 190, 170),
    C('c5', 'most sums land near 2', 28, 300),
    N('pl', 'plot', 28, 335, 528, 250, BARS(20, 160.0)),
    C('c6', 'once at load', 28, 640),
    N('lb', 'load_bang', 28, 675, 88, 46),
    C('c7', 'seven values', 140, 640),
    N('m1', 'message', 140, 675, 180, 42, MSG('1 1 2 2 2 3 10')),
    C('c8', '3 bins over their range', 350, 640),
    N('h2', 'np.histogram 3', 350, 675, 190, 170),
    C('c9', 'counts', 580, 640),
    N('l1', 'list', 580, 675, 120, 42, LIST),
    C('c10', 'bin edges', 580, 735),
    N('l2', 'list', 580, 770, 240, 42, LIST),
],
[('mt', '', 'rn', ''), ('rn', 'random array', 'sm', 'in'), ('sm', 'sum', 'hg', 'in'),
 ('hg', 'counts', 'pl', 'y'), ('lb', 'out', 'm1', ''), ('m1', 'message out', 'h2', 'in'),
 ('h2', 'counts', 'l1', ''), ('h2', 'bin edges', 'l2', '')],
['np.percentile', 't.histc', 'np.rand', 'np.sum'], 880, 1150)


# ------------------------------------------------------------------ interp
page('np.interp', 'np.interp - look up values between the points of a table',
"""
Draws straight lines between the points of a table and reads values off them.
The table is a set of positions (xp) and the values at them (fp); for each x you
send, np.interp finds where it falls among the positions and blends the two
nearest values in proportion.

Use it as a breakpoint curve - a control that rises fast, levels off and rises
again, drawn with a few points - or as a lookup table, a transfer function, a
calibration from raw sensor readings to real units. Send it many x at once to
resample: a path recorded at 30 frames a second read off at 60, or a curve stretched
to any length.

SYNTAX:
np.interp

INPUTS and PARAMETERS:

x:
Where to read the table: a number, list or array of any shape. Every input is looked
up and sent - the only inlet that makes the node send.

xp:
The positions of the table's points. They are held; send them first. If they are not
in increasing order they are sorted (with their values) for you. If xp is never sent,
the positions are 0, 1, 2 ... - the index of each value - so x 1.5 reads halfway
between the second and third values.

fp:
The values at those positions, held. One per position. A 2-D fp - one row per
position, with several columns - is read column by column, so a table of xyz points
gives back xyz points.

OUTPUTS:

values:
The value read at each x, the shape of x (plus fp's columns). Below the first
position the first value is given, and above the last the last: the curve is held
flat beyond its ends.

READING THE DEMO:
Top: a curve with three points: at 0 it is 0, at 0.25 it has already reached 0.9,
and at 1 it is 1. A ramp from 0 to 1 read through it rises steeply then levels off -
the shape of the table, not a straight line.

Bottom: with no xp, the table 0 10 30 is indexed by position. 0.5 is halfway
between 0 and 10, so 5; 1.5 halfway between 10 and 30, so 20; 2 and beyond are 30.

RELATED:
lerp blends between two values by a fraction; np.interp does that across a whole
table. np.linspace makes evenly spaced x positions for resampling. ranger maps one
range onto another in a straight line.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', 'positions', 140, 53),
    N('mx', 'message', 140, 92, 140, 42, MSG('0 0.25 1')),
    C('c3', 'values at them', 140, 150),
    N('mf', 'message', 140, 185, 140, 42, MSG('0 0.9 1')),
    C('c4', 'a ramp, 0 to 1', 320, 53),
    N('sg', 'signal 2.0 saw', 320, 92, 129, 78, SIG('saw', 2.0, 1.0, False)),
    C('c5', 'read off the curve', 320, 230),
    N('ip', 'np.interp', 320, 265, 150, 110),
    C('c6', 'steep, then level', 520, 53),
    N('pl', 'plot', 520, 92, 208, 178, PLOT(0.0, 1.0, 240)),
    C('c7', 'with no positions: by index', 140, 430),
    N('m3', 'message', 140, 465, 140, 42, MSG('0 10 30')),
    C('c8', 'five times a second', 28, 540),
    N('mt', 'metro 200', 28, 575, 129, 70, {'on': True, 'period': 200.0, 'units': 'milliseconds'}),
    C('c9', 'x', 180, 540),
    N('m4', 'message', 180, 575, 180, 42, MSG('0 0.5 1.5 2 3')),
    N('i2', 'np.interp', 400, 465, 150, 110),
    C('c10', 'values', 600, 430),
    N('l1', 'list', 600, 465, 220, 42, LIST),
],
[('lb', 'out', 'mx', ''), ('lb', 'out', 'mf', ''), ('lb', 'out', 'm3', ''),
 ('mx', 'message out', 'ip', 'xp'), ('mf', 'message out', 'ip', 'fp'),
 ('sg', '', 'ip', 'x'), ('ip', 'values', 'pl', 'y'),
 ('m3', 'message out', 'i2', 'fp'), ('mt', '', 'm4', ''), ('m4', 'message out', 'i2', 'x'),
 ('i2', 'values', 'l1', '')],
['lerp', 'np.linspace', 'ranger', 'crossfade'], 700, 1050)


# ---------------------------------------------------------------- convolve
page('np.convolve', 'np.convolve - slide a kernel along an array',
"""
Slides a short list of weights - the kernel - along an array, and at each position
adds up the neighbouring values, each multiplied by its weight.

The kernel decides what that does. Equal weights (0.2 0.2 0.2 0.2 0.2) average each
value with its neighbours: smoothing. 1 2 1 smooths more gently, favouring the
centre. -1 0 1 gives the slope at each point: an edge or change detector. And a
single spike in the input comes out as a copy of the kernel, which is the easiest
way to see what any kernel does.

Use it to smooth a recorded curve or a frame of sensor values without any lag, to
blur, to detect edges, or to apply any filter you can write down as weights. Unlike
the streaming filters, it works on a whole array at once, so it can look ahead as
well as behind.

SYNTAX:
np.convolve <kernel weights: floats ...> <mode>
Defaults: no kernel (send one to the kernel inlet), mode same.

EXAMPLES:
np.convolve 0.2 0.2 0.2 0.2 0.2
np.convolve 1 2 1 full

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input is convolved and sent - the only
inlet that makes the node send.

kernel:
The weights, replacing any given as arguments. Held: send them first.

axis (int box, also an inlet):
For a 2-D or larger array, the axis to slide along: -1 (the default) along each
row, 0 down each column. Each row or column is convolved separately.

mode:
An option, under the node. same (the default) gives an output the length of the
input, the kernel centred on each value. full gives every position where the kernel
touches the input at all - longer by the kernel's length less one. valid gives only
positions where the kernel lies wholly inside the input - shorter by the same
amount, with no edge effects.

OUTPUTS:

convolved:
The result.

AT THE EDGES:
In same and full modes the input is treated as if it were zero beyond its ends, so
values near the ends are pulled towards zero. Use valid to drop them.

READING THE DEMO:
Top: 64 random values (left) averaged five at a time (right): the jagged noise
becomes a gently wandering line, and its spikes are flattened. A new array comes
twice a second.

Bottom: two spikes, 1 and 2, convolved with 1 2 1. Each spike becomes a copy of the
kernel scaled by its height: 1 2 1, and 2 4 2.

RELATED:
np.correlate slides a pattern WITHOUT reversing it, to find where it occurs.
smooth and the one_euro_filter smooth a stream one sample at a time. t.sav_gol_filter
smooths while keeping peaks sharp.
""",
[
    C('c1', 'twice a second', 28, 53),
    N('mt', 'metro 500', 28, 92, 129, 70, {'on': True, 'period': 500.0, 'units': 'milliseconds'}),
    C('c2', '64 random values', 180, 53),
    N('rn', 'np.rand 64', 180, 92, 150, 110),
    C('c3', 'each averaged with 4 neighbours', 360, 53),
    N('cv', 'np.convolve 0.2 0.2 0.2 0.2 0.2', 360, 92, 270, 110),
    C('c4', 'in', 28, 250),
    N('p1', 'plot', 28, 285, 388, 180, BARS(64, 1.0, 380, 130, 'line')),
    C('c5', 'out: smoothed', 440, 250),
    N('p2', 'plot', 440, 285, 388, 180, BARS(64, 1.0, 380, 130, 'line')),
    C('c6', 'once at load', 28, 520),
    N('lb', 'load_bang', 28, 555, 88, 46),
    C('c7', 'two spikes', 140, 520),
    N('m1', 'message', 140, 555, 220, 42, MSG('0 0 1 0 0 0 2 0')),
    C('c8', 'each becomes the kernel', 400, 520),
    N('c2', 'np.convolve 1 2 1', 440, 555, 190, 110),
    C('c9', 'out: 1 2 1 at the first spike, 2 4 2 at the second', 140, 690),
    N('l1', 'list', 140, 725, 340, 42, LIST),
],
[('mt', '', 'rn', ''), ('rn', 'random array', 'cv', 'in'), ('rn', 'random array', 'p1', 'y'),
 ('cv', 'convolved', 'p2', 'y'), ('lb', 'out', 'm1', ''), ('m1', 'message out', 'c2', 'in'),
 ('c2', 'convolved', 'l1', '')],
['np.correlate', 'smooth', 'one_euro_filter', 't.sav_gol_filter', 'np.diff'], 830, 1250)


page('np.correlate', 'np.correlate - find where a pattern occurs',
"""
Slides a short pattern along an array and, at each position, multiplies the
overlapping values together and adds them up. The sum is large where the array
looks like the pattern, so the peaks of the output mark where the pattern occurs.

Use it to find a known shape in a recording - a gesture, a beat, a pulse - or, with
two signals, to measure how far one lags behind the other: correlate one against
the other and the peak's distance from the centre is the lag.

It is np.convolve without the kernel reversed. For a symmetric pattern (1 2 1) the
two are the same; for 1 2 3 they differ, and np.correlate is the one that matches
the pattern as written.

SYNTAX:
np.correlate <pattern: floats ...> <mode>
Defaults: no pattern (send one to the pattern inlet), mode same.

EXAMPLE:
np.correlate 1 2 3

INPUTS and PARAMETERS:

in:
The array to search (a list is turned into one). Every input is correlated and
sent - the only inlet that makes the node send.

pattern:
What to look for, replacing any given as arguments. Held: send it first.

axis (int box, also an inlet):
For a 2-D or larger array, the axis to slide along: -1 (the default) along each
row, 0 down each column.

mode:
An option, under the node: same (the default, output the length of the input),
full (every overlap) or valid (only where the pattern lies wholly inside), as for
np.convolve.

OUTPUTS:

correlated:
The match score at each position.

HEIGHT IS NOT ONLY LIKENESS:
The score grows with the size of the values as well as with how alike they are, so
a loud passage scores higher than a quiet one that matches better. To compare shape
alone, subtract the mean and divide by the size of both first.

READING THE DEMO:
1 2 3 appears twice in the array; the correlation peaks at 14 - 1x1 + 2x2 + 3x3 -
over the centre of each copy, positions 3 and 9. np.argmax picks the first, best
match. A metro re-sends the array five times a second; move the 1 2 3 and click.

RELATED:
np.convolve is the same sliding sum with the kernel reversed: a filter. t.corrcoef
and t.cosine_similarity compare two whole arrays at once. np.argmax finds the peak.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', 'the pattern, held', 140, 53),
    N('mp', 'message', 140, 92, 120, 42, MSG('1 2 3')),
    C('c3', 'five times a second', 28, 170),
    N('mt', 'metro 200', 28, 205, 129, 70, {'on': True, 'period': 200.0, 'units': 'milliseconds'}),
    C('c4', 'an array with the pattern in it twice', 180, 170),
    N('ma', 'message', 180, 205, 330, 42, MSG('0 0 1 2 3 0 0 0 1 2 3 0')),
    C('c5', 'slide and match', 550, 170),
    N('cr', 'np.correlate', 550, 205, 160, 110),
    C('c6', 'the score at each position: peaks at the copies', 28, 360),
    N('l1', 'list', 28, 395, 520, 42, LIST),
    C('c7', 'the best match', 580, 360),
    N('am', 'np.argmax', 580, 395, 150, 70),
    N('l2', 'list', 580, 480, 80, 42, LIST),
],
[('lb', 'out', 'mp', ''), ('mp', 'message out', 'cr', 'pattern'),
 ('mt', '', 'ma', ''), ('ma', 'message out', 'cr', 'in'),
 ('cr', 'correlated', 'l1', ''), ('cr', 'correlated', 'am', 'in'), ('am', 'argmax', 'l2', '')],
['np.convolve', 't.corrcoef', 't.cosine_similarity', 'np.argmax'], 580, 1100)


# -------------------------------------------------------------------- pinv
page('np.linalg.pinv', 'np.linalg.pinv - the best available inverse of a matrix',
"""
Gives the pseudo-inverse of a matrix: the nearest thing to an inverse that a matrix
has, even when it has no true inverse.

Only a square matrix with no redundancy can be inverted (np.linalg.inv). Most
matrices that turn up in practice are not like that: more equations than unknowns
(ten measurements of three values), fewer, or rows that repeat each other. The
pseudo-inverse still undoes as much of the matrix as can be undone, and where an
exact answer is impossible it gives the closest one.

Multiplying it by b gives the least-squares solution of A x = b - the x that comes
closest to satisfying every equation at once - which is the heart of fitting a line
or a plane to measurements, calibrating a sensor against reference readings, or
solving for joint angles that best reach a target. For a square matrix that does
have an inverse, it is exactly that inverse.

SYNTAX:
np.linalg.pinv

INPUTS and PARAMETERS:

input:
The matrix: any 2-D array, of any shape. Every input is inverted and sent.

OUTPUTS:

pseudo-inverse:
The pseudo-inverse. An m x n matrix gives an n x m one.

READING THE DEMO:
A is 3 x 2: three rows, two columns - more equations than unknowns, so it has no
inverse. Its pseudo-inverse is 2 x 3, and multiplied by A it gives the 2 x 2
identity (1 0 / 0 1): it undoes A completely from that side. (A times its
pseudo-inverse is NOT the 3 x 3 identity - three rows cannot all be recovered from
two numbers.)

RELATED:
np.linalg.inv is the exact inverse of a square matrix. np.linalg.lstsq solves A x = b
by least squares directly, and also says how well. np.matmul multiplies matrices.
""",
[
    C('c1', 'five times a second', 28, 53),
    N('mt', 'metro 200', 28, 92, 129, 70, {'on': True, 'period': 200.0, 'units': 'milliseconds'}),
    C('c2', 'six numbers ...', 180, 53),
    N('m1', 'message', 180, 92, 160, 42, MSG('1 2 3 4 5 6')),
    C('c3', '... as A: 3 rows of 2', 370, 53),
    N('rs', 'np.reshape 3 2', 370, 92, 150, 90),
    C('c4', 'A', 650, 53),
    N('ha', 'heat_map', 650, 92, 168, 150, HM(150, 90, 1, 0.0, 6.0, '%.0f')),
    C('c5', 'its pseudo-inverse: 2 rows of 3', 28, 290),
    N('pi', 'np.linalg.pinv', 28, 325, 170, 70),
    N('hp', 'heat_map', 28, 420, 248, 110, HM(230, 60, 1, -1.5, 1.5, '%.2f', 'red-blue')),
    C('c6', 'pinv times A: the identity', 370, 290),
    N('mm', 'np.matmul', 370, 325, 150, 90),
    N('hi', 'heat_map', 370, 420, 168, 110, HM(150, 60, 1, 0.0, 1.0, '%.2f')),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'rs', 'tensor in'),
 ('rs', 'output', 'ha', 'y'), ('rs', 'output', 'mm', 'in 2'), ('rs', 'output', 'pi', 'input'),
 ('pi', 'pseudo-inverse', 'hp', 'y'), ('pi', 'pseudo-inverse', 'mm', 'in 1'),
 ('mm', 'mat mul result', 'hi', 'y')],
['np.linalg.inv', 'np.linalg.lstsq', 'np.linalg.solve', 'np.matmul'], 590, 1000)


# ------------------------------------------------------------------- lstsq
page('np.linalg.lstsq', 'np.linalg.lstsq - the best fit to more equations than unknowns',
"""
Solves A x = b by least squares: finds the x that makes A x come as close to b as
possible, when no x can make them exactly equal.

That is the situation whenever you fit a model to measurements. Five noisy points
cannot all lie on one line, but some line passes closest to all of them - the one
whose squared misses add up to the least. Write the model as a matrix (one row per
measurement), and lstsq gives the line, plane, scale-and-offset or calibration
that fits best.

The matrix holds what you know about each measurement, and x is what you want to
find. To fit y = slope * t + offset to points (t, y), each row of A is t 1, b
holds the y values, and x comes out as slope offset. Add a t*t column and it fits
a curve instead.

SYNTAX:
np.linalg.lstsq

INPUTS and PARAMETERS:

A matrix:
The 2-D matrix, one row per equation (measurement). Every input solves and sends -
the only inlet that makes the node send.

b:
The right-hand side: one value per row of A (or one row of values per row of A, to
solve several fits at once). Held: send it first.

OUTPUTS:

x:
The best-fitting solution: one value per column of A.

residuals:
The sum of the squared misses: 0 for a perfect fit, larger the worse the fit. It
comes back empty when there are no more equations than unknowns, or when the
columns of A are not independent.

rank:
How many independent columns A has. Fewer than its number of columns means some
unknowns cannot be told apart (two columns that are the same), and x is only one
of many equally good answers. Sent first, then residuals, then x.

READING THE DEMO:
Five points measured at t = 0 1 2 3 4: y = 1.1 2.9 5.2 6.8 9.1 - roughly a straight
line. Each row of A is t 1. The fit gives a slope of about 2 and an offset of about
1: y = 2t + 1, with a small residual for the noise. Change a y value and click the
b message to see the line move. The round nodes only tidy the numbers for reading.

RELATED:
np.linalg.solve solves a square system exactly. np.linalg.pinv is the matrix that
lstsq effectively uses: pinv of A times b is the same x.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', 'b: the measured y values', 140, 53),
    N('mb', 'message', 140, 92, 240, 42, MSG('1.1 2.9 5.2 6.8 9.1')),
    C('c3', 'five times a second', 28, 170),
    N('mt', 'metro 200', 28, 205, 129, 70, {'on': True, 'period': 200.0, 'units': 'milliseconds'}),
    C('c4', 'rows of t 1, for t = 0 to 4', 180, 170),
    N('ma', 'message', 180, 205, 240, 42, MSG('0 1 1 1 2 1 3 1 4 1')),
    C('c5', 'as A: 5 rows of 2', 450, 170),
    N('rs', 'np.reshape 5 2', 450, 205, 150, 90),
    C('c6', 'fit', 640, 170),
    N('ls', 'np.linalg.lstsq', 640, 205, 170, 110),
    C('c7', 'x: slope and offset, to 3 places', 28, 360),
    N('r1', 'round', 28, 395, 100, 70, {'decimals': 3}),
    N('l1', 'list', 28, 490, 200, 42, LIST),
    C('c8', 'residuals', 320, 360),
    N('r2', 'round', 320, 395, 100, 70, {'decimals': 3}),
    N('l2', 'list', 320, 490, 120, 42, LIST),
    C('c9', 'rank', 510, 360),
    N('l3', 'list', 510, 395, 60, 42, LIST),
],
[('lb', 'out', 'mb', ''), ('mb', 'message out', 'ls', 'b'),
 ('mt', '', 'ma', ''), ('ma', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'ls', 'A matrix'),
 ('ls', 'x', 'r1', 'in'), ('r1', 'result', 'l1', ''),
 ('ls', 'residuals', 'r2', 'in'), ('r2', 'result', 'l2', ''), ('ls', 'rank', 'l3', '')],
['np.linalg.pinv', 'np.linalg.solve', 'np.linalg.inv', 'np.matmul'], 580, 1050)


# ------------------------------------------------------------------ einsum
page('np.einsum', 'np.einsum - multiply and sum arrays by naming their axes',
"""
Combines arrays by naming their axes with letters. A short formula says which axes
line up, which are summed away, and which make up the result - so one node can do a
dot product, a matrix product, a transpose, a trace, a sum along an axis, or any
of these done row by row over a whole batch.

Read a formula like ij,ij->i as: the first array has axes i and j, so does the
second; multiply them element by element; keep only i in the result, so sum over j.
That is a dot product of each row with the matching row - for 100 joints' velocity
vectors against 100 direction vectors, 100 dot products in one step.

A few formulas:
ij,jk->ik    matrix product (np.matmul)
ij,ij->i     dot product of each row with the matching row
i,j->ij      outer product (np.outer)
ij->ji       transpose
ij->j        sum down each column
ii->         trace (the sum of the diagonal)
bij,bjk->bik a matrix product for each item of a batch

SYNTAX:
np.einsum <subscripts>
Default: ij,jk->ik. Spaces in the formula are ignored, so ij, jk -> ik is fine.

EXAMPLE:
np.einsum ij,ij->i

INPUTS and PARAMETERS:

in:
The first array. Every input computes and sends - the only inlet that makes the
node send.

in 2, in 3 ...:
One more inlet for each further array the formula names (one per comma). Held:
send them first.

subscripts:
An option, under the node: the formula. It can be changed to another formula with
the same number of arrays - the inlets are made from the formula the node was
created with.

OUTPUTS:

result:
The result, with the axes named after the arrow. With nothing after the arrow, a
single number.

READING THE DEMO:
Top: three 2-D vectors in A, three in B. ij,ij->i gives each row's dot product:
1x2 + 0x3 = 2, 0x4 + 1x5 = 5, 1x1 + 1x1 = 2.

Bottom: ij->j has one array and keeps only j: it sums down each column of A, giving
2 2.

RELATED:
np.matmul, np.dot, np.inner and np.outer are fixed cases of einsum, and are easier
to read when they are what you need. np.transpose and np.sum do the single-array
cases.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', 'B: 3 rows of 2', 140, 53),
    N('mb', 'message', 140, 92, 160, 42, MSG('2 3 4 5 1 1')),
    N('rb', 'np.reshape 3 2', 330, 92, 150, 90),
    C('c3', 'five times a second', 28, 215),
    N('mt', 'metro 200', 28, 250, 129, 70, {'on': True, 'period': 200.0, 'units': 'milliseconds'}),
    C('c4', 'A: 3 rows of 2', 180, 215),
    N('ma', 'message', 180, 250, 160, 42, MSG('1 0 0 1 1 1')),
    N('ra', 'np.reshape 3 2', 370, 250, 150, 90),
    C('c5', 'dot product of each row', 560, 53),
    N('e1', 'np.einsum ij,ij->i', 560, 92, 200, 110),
    N('l1', 'list', 560, 250, 160, 42, LIST),
    C('c6', 'sum down each column of A', 370, 380),
    N('e2', 'np.einsum ij->j', 370, 415, 200, 90),
    N('l2', 'list', 600, 415, 120, 42, LIST),
],
[('lb', 'out', 'mb', ''), ('mb', 'message out', 'rb', 'tensor in'), ('rb', 'output', 'e1', 'in 2'),
 ('mt', '', 'ma', ''), ('ma', 'message out', 'ra', 'tensor in'),
 ('ra', 'output', 'e1', 'in'), ('ra', 'output', 'e2', 'in'),
 ('e1', 'result', 'l1', ''), ('e2', 'result', 'l2', '')],
['np.matmul', 'np.dot', 'np.inner', 'np.outer', 'np.transpose', 'np.sum'], 560, 1150)


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
        print(build(stem, p['title'], p['body'], p['demo'], p['links'], demo_width=840,
                    text_width=760, text_height=p['text_height']))


if __name__ == '__main__':
    main()
