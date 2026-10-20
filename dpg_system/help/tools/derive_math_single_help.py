"""Derive one help patch per single-input math function from the hand-made arithmetic_help.json.

    python3 dpg_system/help/tools/derive_math_single_help.py

abs, sqrt, exp, log2, log10, inverse, norm, round, floor, ceil and trunc take no
operand, so their pages use the arithmetic layout with the operand parts taken
out: a sine wave through the function into a plot, a number box through it into
a number box, and one random array through it into a heat map. The NumPy names
np.round, np.floor, np.ceil and np.trunc are the same nodes and share their
pages. math_single_help.json remains the family overview. After running this, use

    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json
"""
import copy, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HELP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from highlight_featured import mark_featured
from derive_arithmetic_help import (
    TITLE, TEXT, SIGNAL, SIG_OP, PLOT, SIG_NOTE, SIGNAL_DEMO, INT_IN, INT_OP, INT_OUT,
    INT_NOTE, CLOSE, OPERAND_FLOAT, OPERAND_NOTE, RAND_A, RAND_B, MAP_A, MAP_B, MAP_OUT,
    NP_BANG, NP_NOTE, NP_OP, NP_T, set_prop, set_comment, set_block, set_number, link)

ARRAYS = """Lists, NumPy arrays and PyTorch tensors go through element by element, keeping their
shape. In the NumPy example at the bottom left, a random array (left) goes through
{label} cell by cell (right)."""

EVERY_INPUT = 'The data to be operated on. Receiving data here triggers the calculation.'

ROUNDERS = """THE FOUR ROUNDING NODES:
They differ in which way a fraction goes. Given -2.5 and 2.7:
round   -2   3     to the nearest whole number
floor   -3   2     down, towards negative
ceil    -2   3     up, towards positive
trunc   -2   2     towards zero - the fraction is simply dropped
For positive numbers trunc is the same as floor; for negative numbers it is the
same as ceil."""

HALVES = """HALFWAY:
A value exactly halfway between two whole numbers rounds to the EVEN one: 2.5 gives 2,
3.5 gives 4, -2.5 gives -2. This is the NumPy and Python rule (it keeps a long run
of roundings from drifting upward); it is not the "always round .5 up" taught at school."""

NP_NAME = """{np} is the same node under its NumPy name, and behaves identically."""


def page(intro, syntax, examples, inputs, result_text, notes, see_also, out='result'):
    parts = [intro.strip(), 'SYNTAX:\n' + syntax, 'EXAMPLES:\n' + examples.strip('\n'),
             'INPUTS and PARAMETERS:\n\nin:\n' + EVERY_INPUT + ''.join('\n\n' + i for i in inputs),
             'OUTPUTS:\n\n' + out + ':\n' + result_text]
    parts += [n.strip() for n in notes]
    parts.append('SEE ALSO:\n' + see_also.strip('\n'))
    return '\n\n'.join(parts)


# One entry per page. 'sig': the signal's shape, range, bipolar flag, an optional
# 'pre' node between signal and function, the plot range and the note beside it
# (None drops the sine demo). 'num': the number demo's input, and its drag speed.
# 'np': np.rand min/max, the input heat map's range, the result heat map's range.
OPS = [
    dict(stem='abs', label='abs', title='abs (absolute value)',
         fn=abs,
         sig=dict(range=1.0, bipolar=True, y=(-1.0, 1.0),
                  note='a sine wave through abs:\nthe lower half folded up'),
         num=dict(value=-7.5, speed=0.1),
         np=dict(rand=(-1.0, 1.0), a=(-1.0, 1.0), out=(-1.0, 1.0)),
         text=page("""
abs gives the distance of each value from zero, throwing away the sign.

-7.5 and 7.5 both come out as 7.5; 0 stays 0.
In the demo a sine wave swinging from -1 to 1 goes through abs: the half below zero
is folded up, so it bounces between 0 and 1, twice as often.

Use it when only the size of something matters, not its direction: how fast,
not which way; how far from the centre, not to which side; how loud a signal is
whichever way the speaker cone is moving.

""" + ARRAYS.format(label='abs') + """ Dark (negative) cells
turn bright.""",
             'abs', """
abs        a speed from a velocity that can be negative
- 0.5 then abs     how far a 0 to 1 control is from its middle""",
             [], 'The input with its sign removed: always zero or positive.', [], """
sqrt    square root
-       subtract, to measure the distance from a value other than zero""")),

    dict(stem='sqrt', label='sqrt', title='sqrt (square root)',
         fn=lambda a: math.copysign(math.sqrt(abs(a)), a),
         sig=dict(range=1.0, bipolar=True, y=(-1.0, 1.0),
                  note='a sine wave through sqrt:\nfuller, squarer humps - the sign is kept'),
         num=dict(value=16.0, speed=0.1),
         np=dict(rand=(0.0, 1.0), a=(0.0, 1.0), out=(0.0, 1.0)),
         text=page("""
sqrt gives the square root of each value: the number that, multiplied by itself,
gives the input.

16 comes out as 4, 2 as 1.414, 0.25 as 0.5. Between 0 and 1 the square root is
LARGER than the input, so in the demo the sine wave's humps swell out towards 1,
and the NumPy example's cells all get brighter.

Use it to undo a square: a distance from squared distances, a level from a power.
It is also a gentle curve for controls - a 0 to 1 slider through sqrt rises quickly
then levels off.

""" + ARRAYS.format(label='sqrt'),
             'sqrt', """
sqrt       a distance from the sum of squares of x and y
sqrt       a fast-rising curve for a 0 to 1 control""",
             [], 'The square root of the input, with the input\'s sign.',
             ["""
NEGATIVE NUMBERS:
A negative number has no ordinary square root. Rather than stop the patch, sqrt
keeps the sign: -16 gives -4. That is why the sine wave in the demo still dips below
zero. If you need true square roots only, put abs before sqrt."""], """
^       raise to a power ("^ 0.5" is also a square root)
abs     absolute value
exp     e to the power of the input""")),

    dict(stem='exp', label='exp', title='exp (e to the power)',
         fn=math.exp,
         sig=dict(range=1.0, bipolar=True, y=(0.0, 3.0),
                  note='e to the power of a sine wave (-1 to 1):\nfrom 0.37 up to 2.72, lopsided'),
         num=dict(value=1.0, speed=0.01),
         np=dict(rand=(-1.0, 1.0), a=(-1.0, 1.0), out=(0.0, 3.0)),
         text=page("""
exp raises the number e (2.71828...) to the power of each value.

0 comes out as 1, 1 as 2.718, -1 as 0.368, 2 as 7.389. The result is always
positive, and each step of 1 in the input MULTIPLIES the output by 2.718 - which
is why the sine wave in the demo comes out lopsided: tall peaks, shallow troughs.

Use it to turn an even control into one that grows by ratios, the way pitch,
loudness and many perceptions do: a slider through exp feels even across its
whole range where a straight one feels crowded at one end. It undoes log.

""" + ARRAYS.format(label='exp'),
             'exp', """
exp        an even slider into a growing ratio
* 0.693 then exp    the same as 2 to the power of the input""",
             [], 'e to the power of the input. Always greater than zero.',
             ["""
LARGE INPUTS:
exp grows very fast: exp of 100 is a 44-digit number, and above about 709 a single
number is too large to hold: the node reports an error and sends nothing.
Scale inputs down first."""], """
log2, log10    the reverse of exp (in other bases)
^              raise to any power
sqrt           square root""")),

    dict(stem='log2', label='log2', title='log2 (logarithm base 2)',
         fn=math.log2,
         sig=dict(range=8.0, bipolar=False, pre=('+', 0.125), y=(-3.5, 3.5),
                  note='log2 of a sine wave from 0.125 to 8.125:\n-3 to 3, the small values spread out'),
         num=dict(value=8.0, speed=0.1),
         np=dict(rand=(0.25, 4.0), a=(0.0, 4.0), out=(-2.0, 2.0)),
         text=page("""
log2 asks: 2 to the power of what gives this value? It is the logarithm base 2.

8 comes out as 3 (2 x 2 x 2), 1 as 0, 0.5 as -1, 0.125 as -3.
Each DOUBLING of the input adds 1 to the output. In the demo the sine wave runs from
0.125 to 8.125, and log2 spreads its small values out and squeezes its large ones
together, giving -3 to 3.

Use it to count doublings: octaves from a frequency ratio (a ratio of 4 is 2 octaves),
bits needed for a number of values, levels in a halving or doubling scale.

""" + ARRAYS.format(label='log2'),
             'log2', """
log2       octaves from a frequency ratio
log2       the number of times something has doubled""",
             [], 'The base 2 logarithm of the input.',
             ["""
ZERO AND NEGATIVE NUMBERS:
These have no logarithm. A single number of zero or less gives -inf (negative
infinity) rather than stopping the patch. Lists, arrays and tensors take the
logarithm of the absolute value instead, so -8 gives 3 and only 0 gives -inf.
The demo adds 0.125 to the wave so it never reaches zero."""], """
log10   logarithm base 10
exp     e to the power of the input
^       raise to a power""")),

    dict(stem='log10', label='log10', title='log10 (logarithm base 10)',
         fn=math.log10,
         sig=dict(range=10.0, bipolar=False, pre=('+', 0.1), y=(-1.2, 1.2),
                  note='log10 of a sine wave from 0.1 to 10.1:\n-1 to 1, the small values spread out'),
         num=dict(value=1000.0, speed=1.0),
         np=dict(rand=(0.1, 10.0), a=(0.0, 10.0), out=(-1.0, 1.0)),
         text=page("""
log10 asks: 10 to the power of what gives this value? It is the logarithm base 10.

1000 comes out as 3, 100 as 2, 1 as 0, 0.1 as -1. It counts the digits, roughly:
each factor of TEN in the input adds 1 to the output. In the demo the sine wave runs
from 0.1 to 10.1; log10 spreads the stretch from 0.1 to 1 over as much of the plot as
the stretch from 1 to 10.

Use it to tame a value that spans many sizes - a sound level, a distance, a count -
into an even scale. Decibels are 20 times the log10 of an amplitude ratio.

""" + ARRAYS.format(label='log10'),
             'log10', """
log10 then * 20     decibels from an amplitude ratio
log10      how many digits a count has""",
             [], 'The base 10 logarithm of the input.',
             ["""
ZERO AND NEGATIVE NUMBERS:
These have no logarithm. A single number of zero or less gives -inf (negative
infinity) rather than stopping the patch. Lists, arrays and tensors take the
logarithm of the absolute value instead, so -100 gives 2 and only 0 gives -inf.
The demo adds 0.1 to the wave so it never reaches zero."""], """
log2    logarithm base 2
exp     e to the power of the input
^       raise to a power""")),

    dict(stem='inverse', label='inverse', title='inverse (one divided by)',
         fn=lambda a: 1 / a,
         sig=dict(range=2.0, bipolar=False, pre=('+', 0.5), y=(0.0, 2.2),
                  note='one over a sine wave from 0.5 to 2.5:\nsharp peaks where the input is small'),
         num=dict(value=4.0, speed=0.1),
         np=dict(rand=(0.5, 2.0), a=(0.0, 2.0), out=(0.0, 2.0)),
         text=page("""
inverse divides 1 by each value: the reciprocal.

4 comes out as 0.25, 0.25 as 4, 2 as 0.5, -2 as -0.5.
Small inputs give large outputs and large inputs small ones. In the demo the sine
wave runs from 0.5 to 2.5: where it is low, its inverse shoots up to 2; where it is
high, the inverse sags to 0.4. In the NumPy example bright cells turn dark and dark
ones bright.

Use it to convert between a rate and a time: a frequency of 4 per second is a period
of 0.25 seconds, and back again. Or when something should grow as another shrinks.

""" + ARRAYS.format(label='inverse'),
             'inverse', """
inverse    a period in seconds from a frequency in Hz
inverse    a zoom factor from a size""",
             [], 'One divided by the input.',
             ["""
ZERO:
1 divided by 0 has no answer. A single number of 0 gives inf (infinity), and so
does a 0 in an array, rather than stopping the patch. Keep inputs away from zero -
the demo adds 0.5 to the wave."""], """
!/      divide the operand by the input - "!/ 1" is inverse
/       divide by an operand""")),

    dict(stem='norm', label='norm', title='norm (scale to length 1)',
         fn=None,
         sig=None,
         msg=dict(value='3 4', result='[0.6, 0.8]'),
         np=dict(rand=(0.0, 1.0), a=(0.0, 1.0), out=(0.0, 0.5),
                 note='an array divided by its own length: the squares of the cells add up to 1'),
         text=page("""
norm normalizes a list, array or tensor: it divides every element by the length of
the whole, so the result has a length of exactly 1 and points the same way.

The list [3, 4] has length 5 (the square root of 3 x 3 + 4 x 4), so norm gives
[0.6, 0.8]. [30, 40] gives the same [0.6, 0.8] - only the direction survives, the
size is gone. Click the message in the example to see it.

Use it to keep a direction and throw away a magnitude: which way a hand is moving
regardless of how fast, a direction to point a light, a colour's balance regardless
of its brightness.

An array of any shape is divided by its length taken over ALL of its elements. In the
NumPy example at the bottom left, a random 4 x 4 array (left) is normalized (right):
the pattern is the same, only scaled down so that the squares of all 16 cells add
up to 1.""",
             'norm', """
norm       a direction from a velocity
norm       balance a set of weights to a fixed overall size""",
             [], 'The input divided by its length, the same shape as the input.',
             ["""
SINGLE NUMBERS AND ZEROS:
A single number has no direction, so norm of any single number gives 1.0.
A list or array of all zeros has no length to divide by and gives nan (not a
number) in every element."""], """
abs       the size of a single number
sqrt      square root, for working out a length yourself""")),

    dict(stem='round', label='round', np_name='np.round',
         title='round (to the nearest whole number)',
         fn=lambda a: round(a, 2),
         decimals=2,
         sig=dict(range=3.0, bipolar=True, y=(-3.5, 3.5),
                  note='a sine wave from -3 to 3, rounded:\neach step centred on a whole number'),
         num=dict(value=3.14159, speed=0.01),
         np=dict(rand=(0.0, 4.0), a=(0.0, 4.0), out=(0.0, 4.0)),
         text=page("""
round moves each value to the nearest whole number - or, with the decimals option,
to the nearest hundredth, tenth and so on.

2.7 comes out as 3, 2.2 as 2, -2.7 as -3. In the demo a smooth sine wave from -3 to 3
becomes a staircase, and the NumPy example's smooth random values become five flat
levels, 0 to 4. The number example is set to 2 decimals, so 3.14159 comes out as 3.14.

Use it to snap a continuous control to steps - whole notes, whole frames, menu
items - or to tidy a long decimal for display.

""" + ARRAYS.format(label='round') + '\n\n' + NP_NAME.format(np='np.round'),
             'round', """
round      snap a control to whole numbers
round      with decimals 2 - keep two decimal places""",
             ["""decimals:
An option, under the node. How many decimal places to keep: 0 (the default) gives
whole numbers, 2 keeps hundredths. A negative number rounds to tens (-1), hundreds
(-2) and so on."""],
             'The input rounded. With decimals 0 a single number comes out as a whole number.',
             [HALVES, ROUNDERS], """
floor   round down
ceil    round up
trunc   round towards zero""")),

    dict(stem='floor', label='floor', np_name='np.floor', title='floor (round down)',
         fn=math.floor,
         sig=dict(range=3.0, bipolar=True, y=(-3.5, 3.5),
                  note='the same sine wave rounded DOWN:\nevery step at or below the wave'),
         num=dict(value=-2.5, speed=0.1),
         np=dict(rand=(-2.0, 2.0), a=(-2.0, 2.0), out=(-2.0, 2.0)),
         text=page("""
floor rounds each value DOWN to the whole number at or below it - towards negative.

2.7 comes out as 2, 2.0 as 2, -2.5 as -3 (down, not towards zero).
In the demo a sine wave from -3 to 3 becomes a staircase that always sits at or
below the wave.

Use it to find which whole slot a value falls in: the bar of a beat counter, the
cell of a grid from a position, the frame of an animation from a time. Unlike round,
a slot runs from its whole number up to just below the next.

""" + ARRAYS.format(label='floor') + '\n\n' + NP_NAME.format(np='np.floor'),
             'floor', """
* 8 then floor     which of 8 slots a 0 to 1 control is in
floor      the whole part of a positive number""",
             [], 'The input rounded down. A single number comes out as a whole number.',
             [ROUNDERS], """
ceil    round up
trunc   round towards zero
round   round to the nearest
//      divide and round down in one step""")),

    dict(stem='ceil', label='ceil', np_name='np.ceil', title='ceil (round up)',
         fn=math.ceil,
         sig=dict(range=3.0, bipolar=True, y=(-3.5, 3.5),
                  note='the same sine wave rounded UP:\nevery step at or above the wave'),
         num=dict(value=-2.5, speed=0.1),
         np=dict(rand=(-2.0, 2.0), a=(-2.0, 2.0), out=(-2.0, 2.0)),
         text=page("""
ceil (ceiling) rounds each value UP to the whole number at or above it - towards
positive.

2.2 comes out as 3, 2.0 as 2, -2.5 as -2 (up, which for a negative number is
towards zero). In the demo a sine wave from -3 to 3 becomes a staircase that always
sits at or above the wave.

Use it when a part needs a whole one: how many pages for 41 lines at 10 a page
(4.1, so 5), how many buses for a group, how many tiles to cover a length.

""" + ARRAYS.format(label='ceil') + '\n\n' + NP_NAME.format(np='np.ceil'),
             'ceil', """
/ 10 then ceil     pages needed for a number of lines, 10 to a page
ceil       the next whole number up""",
             [], 'The input rounded up. A single number comes out as a whole number.',
             [ROUNDERS], """
floor   round down
trunc   round towards zero
round   round to the nearest""")),

    dict(stem='trunc', label='trunc', np_name='np.trunc', title='trunc (round towards zero)',
         fn=math.trunc,
         sig=dict(range=3.0, bipolar=True, y=(-3.5, 3.5),
                  note='the same sine wave with the fraction dropped:\nthe step at 0 is twice as wide'),
         num=dict(value=-2.5, speed=0.1),
         np=dict(rand=(-2.0, 2.0), a=(-2.0, 2.0), out=(-2.0, 2.0)),
         text=page("""
trunc (truncate) drops the fractional part of each value, leaving the whole number
part. That rounds towards zero.

2.7 comes out as 2, -2.7 as -2. For positive numbers it is the same as floor; for
negative ones it is the same as ceil. In the demo a sine wave from -3 to 3 becomes a
staircase pulled in towards zero, with a step twice as wide at 0 - everything between
-1 and 1 becomes 0.

Use it when the whole part is what you mean and the sign should be kept as it is:
the whole seconds in a time that can run backwards, the integer part of a reading.

""" + ARRAYS.format(label='trunc') + '\n\n' + NP_NAME.format(np='np.trunc'),
             'trunc', """
trunc      the whole number part, keeping the sign
trunc      whole seconds from a time in seconds""",
             [], 'The input with its fraction dropped. A single number comes out as a whole number.',
             [ROUNDERS], """
floor   round down
ceil    round up
round   round to the nearest
%       "% 1" gives the fractional part (of a positive number)""")),
]


def float_box(template, node_id, sid, x, y, value, speed):
    """A float box in place of one of the template's int boxes."""
    n = copy.deepcopy(template)
    n['id'], n['sid'] = node_id, sid
    n['position_x'], n['position_y'] = x, y
    set_number(n, value)
    set_prop(n, 'speed_property', speed)
    return n


def message(node_id, sid, x, y, text):
    return {'name': 'message', 'id': node_id, 'sid': sid, 'position_x': x, 'position_y': y,
            'width': 130, 'height': 42, 'visibility': 'show_all', 'draggable': True,
            'presentation_state': 'show_all',
            'properties': {'0': {'name': 'text in', 'value': text, 'value_type': 'string'},
                           '1': {'name': 'adapt_width', 'value': 'grow_to_fit', 'value_type': 'string'},
                           '2': {'name': 'bind to', 'value': '', 'value_type': 'string'},
                           '3': {'name': 'width', 'value': 100, 'value_type': 'int'},
                           '4': {'name': 'font size', 'value': '24', 'value_type': 'string'}}}


def set_function(node, op, decimals=False):
    """A single-input node saves only its own settings: round's decimals, trig's degrees."""
    node['name'] = op['label']
    node['init'] = op['label']
    node['properties'] = {}
    for name, value in op.get('props', {}).items():
        node['properties'][str(len(node['properties']))] = {
            'name': name, 'value': value, 'value_type': type(value).__name__}
    if decimals and 'decimals' in op:
        node['properties']['0'] = {'name': 'decimals', 'value': op['decimals'], 'value_type': 'int'}


def drop(nodes, links, ids):
    for i in ids:
        nodes.pop(i, None)     # the template may have no load_bang / t 1
    return [l for l in links if l['source_node'] not in ids and l['dest_node'] not in ids]


def derive(template, op):
    d = copy.deepcopy(template)
    d['name'] = op['stem'] + '_help'
    nodes = {n['id']: n for n in d['nodes'].values()}
    links = list(d['links'].values())
    next_id = max(nodes) + 1
    next_sid = max(n.get('sid', 0) for n in nodes.values()) + 1
    label = op['label']
    float_template = nodes[OPERAND_FLOAT]

    set_comment(nodes[TITLE], op['title'])
    set_block(nodes[TEXT], op['text'], '30')
    nodes[TEXT]['position_x'], nodes[TEXT]['position_y'] = 729, 45

    # no operand: its number box and label go, and the NumPy demo's second array
    links = drop(nodes, links, {OPERAND_FLOAT, OPERAND_NOTE, RAND_B, MAP_B, NP_T})
    links.append(link(NP_BANG, 'load_bang', 0, 'out', RAND_A, 'np.rand', 0, ''))

    sig = op['sig']
    dy = 0
    if sig is None:
        # norm: a single number always gives 1.0, so there is no sine demo
        links = drop(nodes, links, SIGNAL_DEMO - {OPERAND_FLOAT, OPERAND_NOTE})
        dy = -190
    else:
        sop = nodes[SIG_OP]
        set_function(sop, op)
        set_prop(nodes[SIGNAL], 'range', sig['range'])
        set_prop(nodes[SIGNAL], 'bipolar', sig['bipolar'])
        for name in ('shape', 'period'):
            if name in sig:
                set_prop(nodes[SIGNAL], name, sig[name])
        set_prop(nodes[PLOT], 'min y', sig['y'][0])
        set_prop(nodes[PLOT], 'max y', sig['y'][1])
        set_block(nodes[SIG_NOTE], sig['note'], '24')
        if 'pre' in sig:
            # signal -> + offset -> function, so a log or inverse never sees zero
            pre = copy.deepcopy(sop)
            name, operand = sig['pre']
            pre['id'], pre['sid'] = next_id, next_sid
            pre['name'], pre['init'] = name, f'{name} {operand}'
            pre['properties'] = {'0': {'name': 'operand', 'value': operand, 'value_type': 'float'}}
            pre['position_x'], pre['position_y'] = 333, 92
            sop['position_y'] = 165
            nodes[pre['id']] = pre
            for l in links:
                if l['source_node'] == SIGNAL and l['dest_node'] == SIG_OP:
                    l['dest_node'], l['dest_node_name'] = pre['id'], name
            links.append(link(pre['id'], name, 0, 'result', SIG_OP, label, 0, 'in'))
        next_id, next_sid = next_id + 1, next_sid + 1

    # the number demo: a number in, the function, a number out
    set_function(nodes[INT_OP], op, decimals=True)
    nodes[INT_OP]['position_x'] = 190
    if 'msg' in op:
        m = op['msg']
        for nid, text in ((INT_IN, m['value']), (INT_OUT, m['result'])):
            old = nodes[nid]
            nodes[nid] = message(nid, old['sid'], old['position_x'], old['position_y'], text)
        set_comment(nodes[INT_NOTE], 'click to send the list [3, 4]')
        for l in links:
            if l['source_node'] == INT_IN:
                l['source_node_name'], l['source_output_name'] = 'message', 'message out'
            if l['dest_node'] == INT_OUT:
                l['dest_node_name'], l['dest_input_name'] = 'message', '###text in'
    else:
        num = op['num']
        for nid, value in ((INT_IN, num['value']), (INT_OUT, op['fn'](num['value']))):
            old = nodes[nid]
            nodes[nid] = float_box(float_template, nid, old['sid'], old['position_x'],
                                   old['position_y'], value, num['speed'])
        nodes[INT_OUT]['position_x'] = 310
        for l in links:
            if l['source_node'] == INT_IN:
                l['source_node_name'], l['source_output_name'] = 'float', 'float out'
            if l['dest_node'] == INT_OUT:
                l['dest_node_name'], l['dest_input_name'] = 'float', ''

    # the NumPy demo: one random array through the function
    npd = op['np']
    set_function(nodes[NP_OP], op)
    nodes[NP_OP].pop('init')
    set_prop(nodes[RAND_A], 'min', npd['rand'][0])
    set_prop(nodes[RAND_A], 'max', npd['rand'][1])
    set_prop(nodes[MAP_A], 'min y', npd['a'][0])
    set_prop(nodes[MAP_A], 'max y', npd['a'][1])
    set_prop(nodes[MAP_OUT], 'min y', npd['out'][0])
    set_prop(nodes[MAP_OUT], 'max y', npd['out'][1])
    set_comment(nodes[NP_NOTE], npd.get('note', 'arrays and tensors go through cell by cell: '
                                              'the input on the left, the result on the right'))
    for nid, x, y in ((NP_BANG, 25, 409), (RAND_A, 25, 462), (MAP_A, 170, 440),
                      (NP_OP, 375, 486), (MAP_OUT, 485, 440), (CLOSE, 24, 600)):
        nodes[nid]['position_x'], nodes[nid]['position_y'] = x, y

    for n in nodes.values():
        if n['id'] not in (TITLE, TEXT, 1444) and n['position_y'] > 250:
            n['position_y'] += dy

    for l in links:
        for end in ('source', 'dest'):
            if l[end + '_node'] in (SIG_OP, INT_OP, NP_OP):
                l[end + '_node_name'] = label
                if end == 'source':
                    l['source_output_name'] = op.get('out', 'result')

    d['nodes'] = {str(i): n for i, n in enumerate(nodes.values())}
    d['links'] = {str(i): l for i, l in enumerate(links)}
    return d


def main():
    with open(os.path.join(HELP, 'arithmetic_help.json')) as f:
        template = json.load(f)
    for op in OPS:
        path = os.path.join(HELP, op['stem'] + '_help.json')
        with open(path, 'w') as f:
            doc = derive(template, op)
            mark_featured(doc, op['stem'], clear=True)   # the page's own node, highlighted
            json.dump(doc, f, indent=4)
        print(path)


if __name__ == '__main__':
    main()
