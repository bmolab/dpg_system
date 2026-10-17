"""Derive one help patch per arithmetic operation from the hand-made arithmetic_help.json.

    python3 dpg_system/help/tools/derive_arithmetic_help.py

arithmetic_help.json is laid out by hand, and is the overview of the family. This
script copies its layout into a page per operation (add_help.json, modulo_help.json,
...), swapping in that operation's node, operand, plot range and prose. Rearrange
the overview and rerun this to carry the new layout to every page.

It is deliberately NOT a make_*_help.py: rebuild.py runs a full relayout over what
those generate, which would re-flow this hand layout. After running it, use

    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

The template is read by node id, so the ids below must match arithmetic_help.json.
"""
import copy, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HELP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build_help import annotation_box

# node ids in the template
TITLE, TEXT = 1462, 1473
LOAD_BANG, T1, SIGNAL, SIG_OP, PLOT, SIG_NOTE = 1490, 1497, 1507, 1529, 1539, 1590
INT_IN, INT_OP, INT_OUT, INT_NOTE = 1607, 1643, 1653, 1686
CLOSE, OPERAND_FLOAT, OPERAND_NOTE = 1697, 1714, 1756
RAND_A, RAND_B, MAP_A, MAP_B, MAP_OUT, NP_BANG, NP_NOTE, NP_OP, NP_T = 1783, 1805, 2002, 2078, 2116, 2162, 2198, 2217, 2300

SIGNAL_DEMO = {LOAD_BANG, T1, SIGNAL, SIG_OP, PLOT, SIG_NOTE, OPERAND_FLOAT, OPERAND_NOTE}
NUMPY_DEMO = {RAND_A, RAND_B, MAP_A, MAP_B, MAP_OUT, NP_BANG, NP_NOTE, NP_OP, NP_T}

ARRAYS = """Arrays and tensors are handled element by element, keeping their shape. A single-number
operand applies to every element; an array operand of the same shape pairs element with
element."""

HOLD = """Setting it does not send anything - it waits for the next value at "in"."""

INT_OR_FLOAT = """"{e}" gives a stepped integer slider; "{e}.0" a smooth floating point one."""

DIV_ZERO = """DIVIDING BY ZERO:
This does not raise an error or stop your patch. A single number is divided by
0.00000001 instead, giving a very large number of the right sign. Arrays and tensors
give inf (or nan for 0 divided by 0)."""


def page(intro, syntax, examples, in_text, operand_text, result_text, notes, see_also):
    """Assemble a page's prose in the house order."""
    parts = [intro.strip(), 'SYNTAX:\n' + syntax, 'EXAMPLES:\n' + examples.strip('\n'),
             'INPUTS and PARAMETERS:\n\nin:\n' + in_text + '\n\noperand:\n' + operand_text,
             'OUTPUTS:\n\nresult:\n' + result_text]
    parts += [n.strip() for n in notes]
    parts.append('SEE ALSO:\n' + see_also.strip('\n'))
    return '\n\n'.join(parts)


# One entry per help page. 'sig' is the sine demo (operand, plot range, note);
# 'int' the number demo; 'np' the heat-map range for the NumPy demo's result.
OPS = [
    dict(stem='add', label='+', title='+ (add)',
         sig=dict(init='+ 0.5', operand=0.5, y=(-2.0, 2.0),
                  note='a sine wave, raised by 0.5\ndrag the operand and watch it move'),
         int=dict(init='+ 100', operand=100, value=12),
         np=(0.0, 2.0),
         text=page("""
+ adds the operand to every value that arrives at its left inlet.

7 arriving at "+ 3" sends out 10. A sine wave swinging from -1 to 1, sent through
"+ 0.5", swings from -0.5 to 1.5 instead: the same shape, lifted.

Use it to offset things: transpose a MIDI note ("+ 12" is up an octave), centre a
sensor reading, or lift a signal into a positive range before it becomes a size or
a brightness.

""" + ARRAYS + """ In the NumPy example at the bottom left,
two random arrays are added cell by cell, so the sum runs from 0 to 2.""",
             '+ <operand: int or float>',
             """
+ 12       raise a note number by an octave
+ 1.0      move a -1 to 1 signal into the range 0 to 2""",
             'The value to add to. Receiving data here triggers the calculation.',
             'The amount added. A negative operand subtracts. \n' + HOLD + '\n'
             + INT_OR_FLOAT.format(e='+ 1'),
             'The input plus the operand.',
             [],
             """
-     subtract the operand
!-    subtract the input from the operand""")),

    dict(stem='subtract', label='-', title='- (subtract)',
         sig=dict(init='- 0.5', operand=0.5, y=(-2.0, 2.0),
                  note='a sine wave, lowered by 0.5\ndrag the operand and watch it move'),
         int=dict(init='- 5', operand=5, value=12),
         np=(-1.0, 1.0),
         text=page("""
- subtracts the operand from every value that arrives at its left inlet.

7 arriving at "- 3" sends out 4. A sine wave swinging from -1 to 1, sent through
"- 0.5", swings from -1.5 to 0.5: the same shape, lowered.

Use it to remove an offset: subtract a sensor's resting value so it reads 0 at rest,
or subtract a threshold so that only the excess above it stays positive.

""" + ARRAYS + """ In the NumPy example at the bottom left,
one random array is subtracted from the other cell by cell, so the result runs
from -1 to 1.""",
             '- <operand: int or float>',
             """
- 60       a note number measured from middle C
- 0.5      centre a 0 to 1 control around zero""",
             'The value to subtract from. Receiving data here triggers the calculation.',
             'The amount taken away. \n' + HOLD + '\n' + INT_OR_FLOAT.format(e='- 1'),
             'The input minus the operand.',
             ["""
ORDER MATTERS:
- always takes the operand away from the input. To take the input away from
the operand instead - to turn a 0 to 1 control into 1 to 0, say - use !-."""],
             """
+     add the operand
!-    the operand minus the input""")),

    dict(stem='multiply', label='*', title='* (multiply)',
         sig=dict(init='* 0.5', operand=0.5, y=(-1.0, 1.0),
                  note='a sine wave, halved\ndrag the operand and watch it change'),
         int=dict(init='* 3', operand=3, value=12),
         np=(0.0, 1.0),
         text=page("""
* multiplies every value that arrives at its left inlet by the operand.

7 arriving at "* 3" sends out 21. A sine wave swinging from -1 to 1, sent through
"* 0.5", swings from -0.5 to 0.5: the same shape at half the size.

Use it to scale: stretch a 0 to 1 control to the 0 to 127 MIDI range, convert units
(metres to centimetres is "* 100"), or turn a level up and down. An operand between
0 and 1 shrinks, above 1 enlarges, a negative one also flips the sign, and 0 silences.

""" + ARRAYS + """ This is NOT matrix
multiplication - use np.matmul for that. In the NumPy example at the bottom left, two
random arrays are multiplied cell by cell; every cell is between 0 and 1, so each
product is smaller than either of the values that made it.""",
             '* <operand: int or float>',
             """
* 127      a 0 to 1 control -> the 0 to 127 MIDI range
* -1.0     flip the sign""",
             'The value to multiply. Receiving data here triggers the calculation.',
             'The amount to multiply by. \n' + HOLD + '\n' + INT_OR_FLOAT.format(e='* 2'),
             'The input times the operand.',
             [],
             """
/          divide by the operand
np.matmul  matrix multiplication""")),

    dict(stem='divide', label='/', title='/ (divide)',
         sig=dict(init='/ 2.0', operand=2.0, y=(-2.0, 2.0),
                  note='a sine wave, divided by 2\ndrag the operand toward 0 and watch it grow'),
         int=dict(init='/ 4', operand=4, value=7, out='float'),
         np=(0.0, 4.0),
         text=page("""
/ divides every value that arrives at its left inlet by the operand.

7 arriving at "/ 2" sends out 3.5 - the result is a fraction even when both numbers
are whole. A sine wave swinging from -1 to 1, sent through "/ 2.0", swings from -0.5
to 0.5. As the operand nears zero the result grows without limit.

Use it to convert units (milliseconds to seconds is "/ 1000.0"), to bring a value
with a known maximum into the range 0 to 1 ("/ 127.0" for MIDI), or to turn a total
into an average.

""" + ARRAYS + """ In the NumPy example at the bottom left,
one random array is divided by the other; wherever the divisor cell is close to 0
the result is very large, which is why some cells sit at the top of the colour range.""",
             '/ <operand: int or float>',
             """
/ 1000.0   milliseconds -> seconds
/ 127.0    a MIDI value -> the range 0 to 1""",
             'The value to divide. Receiving data here triggers the calculation.',
             'The amount to divide by. \n' + HOLD + '\n' + INT_OR_FLOAT.format(e='/ 2'),
             'The input divided by the operand. To keep only the whole part, use //.',
             [DIV_ZERO,
              """
A fixed division is often clearer as a multiplication: "* 0.5" is the same as "/ 2"
and cannot divide by zero."""],
             """
//    divide and round down to a whole number
!/    divide the operand by the input
*     multiply""")),

    dict(stem='floor_divide', label='//', title='// (divide, rounding down)',
         sig=dict(init='// 0.25', operand=0.25, y=(-5.0, 5.0),
                  note='a sine wave counted in steps of 0.25\nthe output is the whole number of steps'),
         int=dict(init='// 3', operand=3, value=7),
         np=(0.0, 4.0),
         text=page("""
// divides every value that arrives at its left inlet by the operand and rounds
the result DOWN to a whole number. It answers: how many whole times does the
operand fit into the input?

7 arriving at "// 3" sends out 2 - 3 fits twice, with 1 left over.
A sine wave sent through "// 0.25" becomes a staircase: it counts quarter-steps,
from -4 up to 3.

Use it to count and to quantise: which bar a beat count is in ("// 4"), which row
of an 8-wide grid a cell number is on ("// 8"), which octave a note is in ("// 12").

Two whole numbers give a whole number; a float gives a whole-numbered float (3.0).
""" + ARRAYS,
             '// <operand: int or float>',
             """
// 4      beat count -> bar number
// 12     note number -> octave""",
             'The value to divide. Receiving data here triggers the calculation.',
             'The amount to divide by - the size of one step. \n' + HOLD + '\n'
             + INT_OR_FLOAT.format(e='// 2'),
             'The number of whole steps.',
             ["""
ROUNDING DOWN:
// rounds toward minus infinity, not toward zero, so -7 // 2 is -4, not -3.
That keeps every step the same size as the input crosses zero.""",
              """
// AND % ARE A PAIR:
// counts the whole steps and % gives what is left over. "// 8" on a cell number
gives its row in an 8-wide grid; "% 8" gives its column.""",
              DIV_ZERO],
             """
%     the remainder that // leaves behind
/     divide, keeping the fraction""")),

    dict(stem='modulo', label='%', title='% (remainder)',
         sig=dict(init='% 0.5', operand=0.5, y=(-1.0, 1.0),
                  note='a sine wave, wrapped into 0 to 0.5\ndrag the operand to change the wrap'),
         int=dict(init='% 12', operand=12, value=14),
         np=(0.0, 1.0),
         text=page("""
% divides every value that arrives at its left inlet by the operand and sends out
the remainder: what is left once the operand has been taken out as many whole
times as it fits. mod is the same node under another name.

7 arriving at "% 3" sends out 1. 14 arriving at "% 12" sends out 2.
A sine wave sent through "% 0.5" is wrapped into the range 0 to 0.5: each time
it climbs past 0.5 it drops back to 0 and carries on.

Its main use is WRAPPING - turning something that counts up forever into a cycle:
a beat count through "% 4" gives the beat in the bar (0 1 2 3 0 1 2 3 ...);
a note number through "% 12" gives the pitch class (C is 0, C# is 1 ... B is 11);
an angle through "% 360.0" stays between 0 and 360.

""" + ARRAYS,
             '% <operand: int or float>      or      mod <operand>',
             """
% 4       beat count -> beat within the bar
% 360.0   keep an angle between 0 and 360""",
             'The value to wrap. Receiving data here triggers the calculation.',
             'The size of the cycle. \n' + HOLD + '\n' + INT_OR_FLOAT.format(e='% 2'),
             'The remainder.',
             ["""
NEGATIVE INPUTS:
The result always takes the sign of the operand, so with a positive operand it runs
from 0 up to (but not including) the operand even when the input is negative:
-1 % 12 is 11. That is what you want for wrapping - counting down past 0 lands
at the top of the cycle.""",
              """
An operand of 0 gives 0 rather than an error."""],
             """
//    the number of whole times the operand fits - the other half of the division""")),

    dict(stem='power', label='^', title='^ (raise to a power)',
         sig=dict(init='^ 2.0', operand=2.0, y=(0.0, 1.0), bipolar=False,
                  note='a sine wave from 0 to 1, raised to a power\nabove 1 the peaks sharpen, below 1 they swell'),
         int=dict(init='^ 2', operand=2, value=3),
         np=(0.0, 1.0),
         text=page("""
^ raises every value that arrives at its left inlet to the power of the operand.
pow is the same node under another name.

3 arriving at "^ 2" sends out 9.0 (3 squared); 2 arriving at "^ 3" sends out 8.0.
A power of 0.5 is a square root: 9 through "^ 0.5" gives 3.0.

Its main use is SHAPING a range from 0 to 1. 0 and 1 stay put, but everything
between them bends: above 1 values sag toward 0, below 1 they bulge toward 1.
"^ 2.0" on a fader gives finer control at the quiet end, which is why it is the
usual way to make a volume control feel even. In the demo, a 0 to 1 sine wave is
raised to a power - drag the operand either side of 1 and watch the shape change.

""" + ARRAYS,
             '^ <operand>      or      pow <operand>',
             """
^ 2.0     a fader curve with finer control near 0
^ 0.5     square root""",
             'The value to raise. Receiving data here triggers the calculation.',
             'The power. \n' + HOLD + '\n'
             'It is always a smooth floating point slider, whether or not you write a \n'
             'decimal point, so you can drag between whole-number powers.',
             'The input raised to the operand. For a single number this is always a float.',
             ["""
NO REAL ANSWER:
A negative number raised to a fractional power, or 0 raised to a negative one,
has no real-number result. For a single number the node reports an error and sends
nothing; arrays and tensors give nan (or inf) in those elements. To shape a signal
that crosses zero, use abs first, raise that, and restore the sign afterwards."""],
             """
sqrt, exp, log2, log10    the single-input math functions""")),

    dict(stem='min', label='min', title='min (the smaller value)',
         sig=dict(init='min 0.5', operand=0.5, y=(-1.0, 1.0),
                  note='a sine wave with its peaks cut off at 0.5\nthe operand is a ceiling'),
         int=dict(init='min 10', operand=10, value=12),
         np=(0.0, 1.0),
         text=page("""
min compares every value that arrives at its left inlet with the operand and
sends out whichever is SMALLER.

7 arriving at "min 5" sends out 5; 3 arriving sends out 3.
So the operand is a CEILING: nothing above it gets through. In the demo the sine
wave is flattened wherever it would rise above 0.5.

Use it to protect whatever comes next from values that are too large - cap a
volume at a safe level, or stop an index running past the end of a list.

Arrays and tensors are compared element by element, keeping their shape. In the
NumPy example at the bottom left, each cell of the result is the smaller of the two
cells in the same place.""",
             'min <operand: int or float>',
             """
min 1.0   never let a level exceed 1
min 127   keep a value inside the MIDI range""",
             'The value to limit. Receiving data here triggers the calculation.',
             'The ceiling. \n' + HOLD + '\n' + INT_OR_FLOAT.format(e='min 1'),
             'The input, or the operand if the input is larger.',
             ["""
min does not find the smallest element of an array - use np.min for that.
To limit a value at both ends, use clamp, or send it through max and then min."""],
             """
max     the larger value - a floor rather than a ceiling
clamp   limit at both ends at once
np.min  the smallest element in an array""")),

    dict(stem='max', label='max', title='max (the larger value)',
         sig=dict(init='max -0.5', operand=-0.5, y=(-1.0, 1.0),
                  note='a sine wave with its troughs cut off at -0.5\nthe operand is a floor'),
         int=dict(init='max 10', operand=10, value=7),
         np=(0.0, 1.0),
         text=page("""
max compares every value that arrives at its left inlet with the operand and
sends out whichever is LARGER.

3 arriving at "max 5" sends out 5; 7 arriving sends out 7.
So the operand is a FLOOR: nothing below it gets through. In the demo the sine
wave is flattened wherever it would fall below -0.5.

Use it to keep values from going too low - "max 0" stops a value going negative,
and a small positive floor keeps a size or a divisor from ever reaching zero.

Arrays and tensors are compared element by element, keeping their shape. In the
NumPy example at the bottom left, each cell of the result is the larger of the two
cells in the same place.""",
             'max <operand: int or float>',
             """
max 0     never let a value go negative
max 0.01  keep a value safely away from zero""",
             'The value to limit. Receiving data here triggers the calculation.',
             'The floor. \n' + HOLD + '\n' + INT_OR_FLOAT.format(e='max 1'),
             'The input, or the operand if the input is smaller.',
             ["""
max does not find the largest element of an array - use np.max for that.
To limit a value at both ends, use clamp, or send it through max and then min."""],
             """
min     the smaller value - a ceiling rather than a floor
clamp   limit at both ends at once
np.max  the largest element in an array""")),

    dict(stem='reverse_subtract', label='!-', title='!- (subtract from the operand)',
         sig=dict(init='!- 0.5', operand=0.5, y=(-2.0, 2.0),
                  note='0.5 minus a sine wave\nflipped upside down, then raised'),
         int=dict(init='!- 100', operand=100, value=12),
         np=(-1.0, 1.0),
         text=page("""
!- subtracts every value that arrives at its left inlet FROM the operand.
It is - with the two sides swapped: operand minus input.

3 arriving at "!- 10" sends out 7 (10 minus 3), where "- 10" would send -7.
A sine wave swinging from -1 to 1, sent through "!- 0.5", is turned upside down
and lifted: it swings from 1.5 to -0.5.

Its main use is INVERTING a range: "!- 1.0" turns a 0 to 1 control into 1 to 0,
so one fader can bring one thing up while it takes another down; "!- 127" does the
same for MIDI values. It also gives "how much is left": total minus amount used.

Why not just -? The changing value has to arrive at the left inlet, because that
is the one that triggers. !- keeps that arrangement and subtracts the other way round.

""" + ARRAYS,
             '!- <operand: int or float>',
             """
!- 1.0    turn a 0 to 1 control upside down
!- 127    turn a MIDI value upside down""",
             'The amount to subtract. Receiving data here triggers the calculation.',
             'The value subtracted from. \n' + HOLD + '\n' + INT_OR_FLOAT.format(e='!- 1'),
             'The operand minus the input.',
             [],
             """
-     the input minus the operand
!/    the other swapped operation: the operand divided by the input""")),

    dict(stem='reverse_divide', label='!/', title='!/ (divide the operand)',
         sig=dict(init='!/ 1.0', operand=1.0, y=(0.0, 1.2), offset=2,
                  note='1 divided by a sine wave that runs from 1 to 3\n(the + 2 keeps it away from zero)'),
         int=dict(init='!/ 60.0', operand=60.0, value=120, out='float'),
         np=(0.0, 4.0),
         text=page("""
!/ divides the operand BY every value that arrives at its left inlet.
It is / with the two sides swapped: operand divided by input.

4 arriving at "!/ 1.0" sends out 0.25 - one divided by the input, its reciprocal.
In the demo, 1 is divided by a sine wave that runs from 1 to 3, giving a curve
between 1/3 and 1 that rises highest where the sine dips lowest.

Its main use is converting between a RATE and a PERIOD, which are reciprocals:
"!/ 1.0" turns a frequency in hertz into a period in seconds, and "!/ 60.0" turns
beats per minute into seconds per beat (120 bpm gives 0.5). Each works in both
directions. As the input nears zero the result grows without limit.

""" + ARRAYS,
             '!/ <operand: int or float>',
             """
!/ 1.0    the reciprocal: hertz -> seconds per cycle
!/ 60.0   beats per minute -> seconds per beat""",
             'The amount to divide by. Receiving data here triggers the calculation.',
             'The value divided. \n' + HOLD + '\n'
             'Write it with a decimal point ("!/ 1.0"): with a whole-number operand the \n'
             'slider is stepped.',
             'The operand divided by the input. It is a fraction even for whole numbers.',
             [DIV_ZERO.replace('A single number is divided by \n0.00000001 instead',
                               'A zero input is replaced by \n0.00000001')],
             """
/     the input divided by the operand
!-    the other swapped operation: the operand minus the input""")),

    dict(stem='perm', label='perm', title='perm (ordered picks)',
         single=dict(init='perm 2', operand=2, value=7, in_note='how many items', op_note='how many to pick'),
         text=page("""
perm counts the ORDERED ways of picking some items from a larger set: how many
different sequences you could make by picking operand items from input items.

7 arriving at "perm 2" sends out 42: there are 7 choices for the first pick and 6
left for the second, and 7 x 6 = 42. Order counts, so gold-then-silver is a different
result from silver-then-gold - 42 possible podiums for the top two of 7 runners.
"perm 3" on 7 gives 210 (7 x 6 x 5).

Picking every item (operand equal to the input) counts every possible arrangement:
5 items can be put in order 120 ways.

perm works on single whole numbers only. Fractions are cut down to whole numbers;
an array or tensor does not raise an error, but the result is a meaningless 0 or 1.""",
             'perm <operand: int>',
             """
perm 2    ordered pairs
perm 3    first, second and third places""",
             'How many items there are to choose from. \n'
             'Receiving data here triggers the calculation.',
             'How many of them to pick. \n' + HOLD,
             'The number of different ordered picks.',
             ["""
If you pick more than there are, or either number is negative, the result is 0.
The count grows very fast: 13 items in order already exceed six billion, more than
a number box can show."""],
             """
combination   the same count when order does not matter""")),

    dict(stem='combination', label='combination', title='combination (unordered picks)',
         single=dict(init='combination 2', operand=2, value=7, in_note='how many items', op_note='how many to choose'),
         text=page("""
combination counts the ways of CHOOSING some items from a larger set when order
does not matter: how many different groups of operand items you could take from
input items.

7 arriving at "combination 2" sends out 21. perm counts 42 ordered pairs from 7,
but choosing A and B is the same group as choosing B and A, so each pair is counted
once. A five-card hand from a 52-card deck: 52 into "combination 5" gives 2598960.

Choosing none, or all of them, can only be done one way, so both give 1.

combination works on single whole numbers only. Fractions are cut down to whole
numbers; an array or tensor does not raise an error, but the result is a
meaningless 0 or 1.""",
             'combination <operand: int>',
             """
combination 2   how many pairs
combination 5   how many five-card hands""",
             'How many items there are to choose from. \n'
             'Receiving data here triggers the calculation.',
             'How many of them to choose. \n' + HOLD,
             'The number of different groups.',
             ["""
If you choose more than there are, or either number is negative, the result is 0.
The count grows quickly, though more slowly than perm."""],
             """
perm   the same count when order matters""")),
]


# what each node sends, so a saved output box shows the true result on opening
RESULT = {
    '+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b,
    '/': lambda a, b: a / b, '//': lambda a, b: a // b, '%': lambda a, b: a % b,
    '^': lambda a, b: float(a ** b), 'min': min, 'max': max,
    '!-': lambda a, b: b - a, '!/': lambda a, b: b / a,
    'perm': lambda a, b: math.perm(a, b), 'combination': lambda a, b: math.comb(a, b),
}


def set_number(node, value):
    """The value an int or float box shows: its unnamed first property."""
    set_prop(node, '', float(value) if node['name'] == 'float' else int(value))


def prop(node, name):
    for p in node['properties'].values():
        if p['name'] == name:
            return p
    raise KeyError(name)


def set_prop(node, name, value):
    p = prop(node, name)
    p['value'] = value
    if value is not None:
        p['value_type'] = type(value).__name__ if not isinstance(value, str) else 'string'


def set_comment(node, text):
    node['init'] = 'comment ' + text
    node['comment'] = text
    set_prop(node, 'text', text)


def set_block(node, text, size):
    """Fill a locked text_block and size it to fit, as build_help does."""
    w, h = annotation_box(text, size)
    # A text_block saves its text twice, and the app loads the top-level
    # 'text', not the 'block' property - set only one and the template's
    # prose comes back on load.
    set_prop(node, 'block', text)
    node['text'] = text
    set_prop(node, 'width', w)
    set_prop(node, 'height', h)
    node['width'], node['height'] = w + 8, h + 16


def set_op(node, label, init, operand):
    node['name'] = label
    if init is None:
        node.pop('init', None)
    else:
        node['init'] = init
    set_prop(node, 'operand', operand)


def link(src, src_name, out_i, out_name, dst, dst_name, in_i, in_name):
    return {'source_node': src, 'source_node_name': src_name, 'source_output_index': out_i,
            'source_output_name': out_name, 'dest_node': dst, 'dest_node_name': dst_name,
            'dest_input_index': in_i, 'dest_input_name': in_name}


def int_node(template, node_id, sid, x, y):
    n = copy.deepcopy(template)
    n['id'], n['sid'] = node_id, sid
    n['position_x'], n['position_y'] = x, y
    return n


def derive(template, op):
    d = copy.deepcopy(template)
    d['name'] = op['stem'] + '_help'
    nodes = {n['id']: n for n in d['nodes'].values()}
    links = list(d['links'].values())
    next_id = max(nodes) + 1
    next_sid = max(n.get('sid', 0) for n in nodes.values()) + 1

    set_comment(nodes[TITLE], op['title'])
    set_block(nodes[TEXT], op['text'], '30')
    nodes[TEXT]['position_x'], nodes[TEXT]['position_y'] = 729, 45

    if 'single' in op:
        # perm / combination: whole numbers only, so just the number demo,
        # with a second int box driving the operand.
        s = op['single']
        drop = SIGNAL_DEMO | NUMPY_DEMO
        for i in drop:
            nodes.pop(i, None)     # the template may have no load_bang / t 1
        links = [l for l in links if l['source_node'] not in drop and l['dest_node'] not in drop]
        set_op(nodes[INT_OP], op['label'], s['init'], s['operand'])
        for nid, x, y in ((INT_IN, 29, 160), (INT_OP, 183, 156), (INT_OUT, 330, 163),
                          (INT_NOTE, 29, 122), (CLOSE, 24, 280)):
            nodes[nid]['position_x'], nodes[nid]['position_y'] = x, y
        set_comment(nodes[INT_NOTE], s['in_note'])
        nodes[INT_NOTE]['width'] = int(len(s['in_note']) * 6.16) + 30
        opnd = int_node(nodes[INT_IN], next_id, next_sid, 183, 95)
        note = copy.deepcopy(nodes[INT_NOTE])
        note['id'], note['sid'] = next_id + 1, next_sid + 1
        note['position_x'], note['position_y'] = 183, 58
        set_comment(note, s['op_note'])
        nodes[opnd['id']], nodes[note['id']] = opnd, note
        links.append(link(opnd['id'], 'int', 0, 'int out', INT_OP, op['label'], 1, '###operand'))
        set_number(nodes[INT_IN], s['value'])
        set_number(opnd, s['operand'])
        set_number(nodes[INT_OUT], RESULT[op['label']](s['value'], s['operand']))
    else:
        sig, num = op['sig'], op['int']
        set_op(nodes[SIG_OP], op['label'], sig['init'], sig['operand'])
        set_prop(nodes[SIGNAL], 'bipolar', sig.get('bipolar', True))
        set_prop(nodes[PLOT], 'min y', sig['y'][0])
        set_prop(nodes[PLOT], 'max y', sig['y'][1])
        set_block(nodes[SIG_NOTE], sig['note'], '24')

        set_op(nodes[INT_OP], op['label'], num['init'], num['operand'])
        if num.get('out') == 'float':
            # a fractional result needs a float box to show it
            out = copy.deepcopy(nodes[OPERAND_FLOAT])
            out['id'], out['sid'] = INT_OUT, nodes[INT_OUT]['sid']
            out['position_x'], out['position_y'] = nodes[INT_OUT]['position_x'], nodes[INT_OUT]['position_y']
            nodes[INT_OUT] = out
            for l in links:
                if l['dest_node'] == INT_OUT:
                    l['dest_node_name'] = 'float'

        set_number(nodes[OPERAND_FLOAT], sig['operand'])
        set_number(nodes[INT_IN], num['value'])
        set_number(nodes[INT_OUT], RESULT[op['label']](num['value'], num['operand']))

        set_op(nodes[NP_OP], op['label'], None, 0.0)
        for m in (MAP_A, MAP_B):
            set_prop(nodes[m], 'min y', 0.0)
            set_prop(nodes[m], 'max y', 1.0)
        set_prop(nodes[MAP_OUT], 'min y', op['np'][0])
        set_prop(nodes[MAP_OUT], 'max y', op['np'][1])

        if 'offset' in sig:
            # signal -> + offset -> op, so the op never sees zero
            add = copy.deepcopy(nodes[SIG_OP])
            add['id'], add['sid'] = next_id, next_sid
            set_op(add, '+', '+ %d' % sig['offset'], sig['offset'])
            add['position_x'], add['position_y'] = 333, 92
            nodes[SIG_OP]['position_y'] = 165
            nodes[add['id']] = add
            for l in links:
                if l['source_node'] == SIGNAL and l['dest_node'] == SIG_OP:
                    l['dest_node'], l['dest_node_name'] = add['id'], '+'
            links.append(link(add['id'], '+', 0, 'result', SIG_OP, op['label'], 0, 'in'))

    for l in links:
        for end in ('source', 'dest'):
            nid = l[end + '_node']
            if nid in (SIG_OP, INT_OP, NP_OP):
                l[end + '_node_name'] = op['label']

    d['nodes'] = {str(i): n for i, n in enumerate(nodes.values())}
    d['links'] = {str(i): l for i, l in enumerate(links)}
    return d


def main():
    with open(os.path.join(HELP, 'arithmetic_help.json')) as f:
        template = json.load(f)
    for op in OPS:
        path = os.path.join(HELP, op['stem'] + '_help.json')
        with open(path, 'w') as f:
            json.dump(derive(template, op), f, indent=4)
        print(path)


if __name__ == '__main__':
    main()
