import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter


# arithmetic is no longer generated here: arithmetic_help.json is laid out by
# hand, and derive_arithmetic_help.py copies its layout into one page per
# operation (add, subtract, ..., combination). Regenerating it here would
# overwrite the hand layout on every rebuild.

# ----------------------------------------------------------------- math_single
body = """These nodes each apply one mathematical function to whatever arrives at their inlet.

They take no operand - the function is the whole node. 
All of them accept single numbers, lists, NumPy arrays and PyTorch tensors, 
and an array goes through element by element, keeping its shape.

THE NODES:

abs       distance from zero, discarding the sign
sqrt      square root
exp       e raised to the power of the input
log2      logarithm base 2
log10     logarithm base 10
inverse   one divided by the input
norm      divide a vector by its own length, giving a unit vector
round     round to the nearest whole number
floor     round down, towards negative
ceil      round up, towards positive
trunc     drop the fractional part, rounding towards zero

The four rounding nodes also exist under NumPy names - np.round, np.floor, 
np.ceil and np.trunc - which behave identically.

Note that floor, ceil and trunc differ only for negative numbers. 
Given -2.5: floor gives -3, ceil gives -2, and trunc gives -2.

SYNTAX:
<function>

EXAMPLE:
sqrt

INPUTS and PARAMETERS:

in:
The data to be operated on. Receiving data here triggers the calculation.

decimals:
An option that appears on the round node only. 
It sets how many decimal places to keep. The default, 0, rounds to whole numbers.

OUTPUTS: 

result:
The result, in the same shape as the input.

STAYING ALIVE ON BAD INPUT:
These nodes are built not to interrupt a running patch. 
log2 and log10 of zero or a negative number return negative infinity for single 
numbers, and for arrays they take the logarithm of the absolute value. 
inverse of zero returns infinity. 
norm of a single number returns 1.0, since one number has no direction."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 2.0)},
    {'key': 'ab', 'init': 'abs', 'pos': (30, 232), 'w': 100, 'h': 50},
    {'key': 'p1', 'init': 'plot', 'pos': (200, 200), 'w': 208, 'h': 176,
     'props': PLOT(-1.0, 1.0)},
    {'key': 'c1', 'comment': True, 'text': 'the sine folded onto the positive side',
     'pos': (30, 292)},
    {'key': 'sig2', 'init': 'signal', 'pos': (30, 345), 'w': 129, 'h': 78,
     'props': SIG('saw', 3.0, 4.0, False)},
    {'key': 'fl', 'init': 'floor', 'pos': (30, 445), 'w': 100, 'h': 50},
    {'key': 'p2', 'init': 'plot', 'pos': (200, 413), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 4.0)},
    {'key': 'c2', 'comment': True, 'text': 'a ramp turned into a staircase',
     'pos': (30, 505)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'), ('tt', '1', 'sig2', 'on'),
         ('sig', '', 'ab', 'in'), ('ab', 'result', 'p1', 'y'),
         ('sig2', '', 'fl', 'in'), ('fl', 'result', 'p2', 'y')]
print(build('math_single', 'math_single - one function, no operand', body,
            demo, links, demo_width=430, text_width=800, text_height=640))


# ----------------------------------------------------------------------- trig
body = """The trigonometry nodes convert between angles and ratios.

sin, cos and tan take an angle and give you a number. 
asin, acos and atan go the other way, taking a number and giving you back an angle.

By default these nodes work in DEGREES, not radians. 
This is unusual for a programming environment, and it is deliberate - most of the 
angles you deal with in a patch come from sensors, rotations and interfaces that 
speak in degrees. Uncheck the "degrees" box to work in radians instead.

They accept single numbers, lists, NumPy arrays and PyTorch tensors, 
and an array is processed element by element.

THE NODES:

sin    the sine of an angle, between -1 and 1
cos    the cosine of an angle, between -1 and 1
tan    the tangent of an angle - unbounded, and very large near 90 and 270 degrees
asin   the angle whose sine is the input; the input must be between -1 and 1
acos   the angle whose cosine is the input; the input must be between -1 and 1
atan   the angle whose tangent is the input; any input is valid

SYNTAX:
<function>

EXAMPLE:
sin

INPUTS and PARAMETERS:

in:
The value to be operated on. Receiving data here triggers the calculation. 
For sin, cos and tan this is an angle. For asin, acos and atan it is a ratio.

degrees:
When checked, angles are read and written in degrees. 
When unchecked, they are in radians. Checked is the default.

OUTPUTS: 

out:
The result. For asin, acos and atan this is an angle, in whichever unit 
the degrees checkbox is set to."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('saw', 3.0, 360.0, False)},
    {'key': 'c0', 'comment': True, 'text': 'a ramp from 0 to 360 degrees', 'pos': (30, 215)},
    {'key': 'sn', 'init': 'sin', 'pos': (30, 252), 'w': 100, 'h': 60,
     'props': {'degrees': True}},
    {'key': 'p1', 'init': 'plot', 'pos': (200, 220), 'w': 208, 'h': 176,
     'props': PLOT()},
    {'key': 'cs', 'init': 'cos', 'pos': (30, 340), 'w': 100, 'h': 60,
     'props': {'degrees': True}},
    {'key': 'p2', 'init': 'plot', 'pos': (200, 400), 'w': 208, 'h': 176,
     'props': PLOT()},
    {'key': 'c1', 'comment': True, 'text': 'the same ramp, a quarter turn apart',
     'pos': (30, 420)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'sn', 'in'), ('sn', 'out', 'p1', 'y'),
         ('sig', '', 'cs', 'in'), ('cs', 'out', 'p2', 'y')]
print(build('trig', 'trig - angles in, ratios out (and back again)', body,
            demo, links, demo_width=430, text_width=790, text_height=600))

# ----------------------------------------------------------------- comparison
body = """The comparison nodes test the incoming value against an operand and report 
whether the test is true or false.

The left inlet takes the value and triggers the test; the right inlet holds the 
value to compare against. The result comes out as true or false - or, if you 
prefer, as a 1/0 number.

Feeding in a NumPy array or a PyTorch tensor compares every element, 
and you get back an array of results the same shape as the input. 
That makes these nodes useful as masks, not just as switches.

THE NODES:

>     greater than
>=    greater than or equal to
<     less than
<=    less than or equal to
==    equal to
!=    not equal to

SYNTAX:
<operator> <operand: int or float>

EXAMPLE:
> 0.5

INPUTS and PARAMETERS:

in:
The value to be tested. Receiving data here triggers the comparison.

operand:
The value to compare against. 
If you supply the operand as an argument when creating the node, this inlet 
appears as a draggable number you can adjust by hand. 
If you do not, it is a plain inlet expecting the operand from elsewhere.

output_type:
Chooses what the answer looks like. 
"bool" gives true or false, "int" gives 1 or 0, and "float" gives 1.0 or 0.0. 
Use int or float when you want to feed the result into arithmetic or a plot.

OUTPUTS: 

result:
The outcome of the test, in the type chosen by output_type. 
For array input, an array of outcomes of the same shape.

RELATED:
For a node that passes the value through when the test succeeds, rather than 
reporting true or false, see the change / increasing / decreasing / pass family."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 2.0)},
    {'key': 'gt', 'init': '> 0.0', 'pos': (30, 235), 'w': 120, 'h': 70,
     'props': {'output_type': 'int'}},
    {'key': 'c0', 'comment': True, 'text': 'output_type set to int, so it plots',
     'pos': (30, 315)},
    {'key': 'p1', 'init': 'plot', 'pos': (215, 205), 'w': 208, 'h': 176,
     'props': PLOT(-0.2, 1.2)},
    {'key': 'c1', 'comment': True, 'text': 'a sine wave becomes a square wave\ndrag the operand to move the threshold',
     'pos': (30, 345)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'gt', 'in'), ('gt', 'result', 'p1', 'y')]
print(build('comparison', 'comparison - test a value, report true or false', body,
            demo, links, demo_width=445, text_width=790, text_height=620))


# change, increasing, decreasing and pass are no longer generated here:
# increasing_help.json is laid out by hand, and derive_gate_help.py copies its
# layout into the other three. Regenerating them here would undo that.


# ------------------------------------------------------------------- crossfade
body = """The crossfade node blends smoothly between two inputs.

At a mix of 0 you get input A alone. At 1 you get input B alone. 
Anywhere in between you get a proportional blend of the two: 
the result is A times (1 minus mix), plus B times mix.

lerp is the same node under a different name - short for linear interpolation - 
for when you are thinking of it as moving between two values rather than 
fading between two sources.

It works on single numbers, lists, NumPy arrays and PyTorch tensors, 
and A and B do not have to be the same kind of thing: B is converted to match A. 
Blending two arrays blends them element by element.

The mix itself can be an array too. Feed an array of mix values the same shape 
as your data and each element gets its own blend - a gradient, a mask, or 
a per-joint weighting.

SYNTAX:
crossfade <mix: float>
lerp <mix: float>

EXAMPLE:
crossfade 0.5

INPUTS and PARAMETERS:

A:
The first input, heard alone when mix is 0. 
Receiving data here triggers the blend.

B:
The second input, heard alone when mix is 1. 
This inlet stores its value; it does not trigger output on its own.

mix:
The blend position, from 0.0 to 1.0. It defaults to 0.5, an even mix. 
The slider moves in steps of 0.01. Values outside 0 to 1 are not clipped, 
so you can push past either end to extrapolate beyond A or B.

OUTPUTS: 

out:
The blended result, in the same shape as input A."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 2.0)},
    {'key': 'sig2', 'init': 'signal', 'pos': (185, 132), 'w': 129, 'h': 78,
     'props': SIG('square', 0.7)},
    {'key': 'ca', 'comment': True, 'text': 'A: a sine', 'pos': (30, 215)},
    {'key': 'cb', 'comment': True, 'text': 'B: a square', 'pos': (185, 215)},
    {'key': 'xf', 'init': 'crossfade 0.5', 'pos': (30, 255), 'w': 150, 'h': 100,
     'props': {'mix': 0.5}},
    {'key': 'c1', 'comment': True, 'text': 'drag mix from 0 to 1', 'pos': (200, 285)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 375), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'), ('tt', '1', 'sig2', 'on'),
         ('sig', '', 'xf', 'A'), ('sig2', '', 'xf', 'B'), ('xf', 'out', 'p1', 'y')]
print(build('crossfade', 'crossfade - blend between two inputs', body,
            demo, links, demo_width=400, text_width=780, text_height=580))


# accumulate is no longer generated here: accumulate_help.json is laid out by
# hand. Regenerating it here would overwrite that layout on every rebuild.

# --------------------------------------------------------- continuous_rotation
body = """The continuous_rotation node removes the jump that happens when an angle wraps around.

Angles usually arrive already folded into a range - a sensor that reads 359 degrees 
and then turns a little further reports 1, not 361. That wrap is a lie about the 
motion: nothing jumped 358 degrees backwards, the reading simply ran off the end 
of its scale.

This node undoes that. It compares each angle to the previous one and adds or 
subtracts whole turns until the change between them is the smallest one possible. 
What comes out is an angle that keeps climbing past 360, or falling below zero, 
following the real motion.

You use it before differentiating an angle, before smoothing one, or any time a 
wrap would otherwise show up as a huge spurious spike.

It handles a list of angles at once, unwrapping each element against its own history.

SYNTAX:
continuous_rotation

EXAMPLE:
continuous_rotation

INPUTS and PARAMETERS:

rotation in:
The angle or list of angles, in degrees. 
Receiving data here triggers the node.

clear input:
Forgets the accumulated turns and starts again from the next value that arrives. 
Send anything here to clear. 
Use it when the source jumps somewhere genuinely new and you do not want the 
node to interpret that as many turns.

OUTPUTS: 

out:
The unwrapped angle, as a NumPy array. 
It is continuous with the previous output, so it may lie far outside 0 to 360.

THE LIMIT OF THIS:
The node assumes the input never really moves more than half a turn between two 
frames - that is what lets it decide which way the angle went. 
If your source genuinely spins faster than 180 degrees per frame, 
that assumption breaks and the unwrapping will follow the wrong direction. 
Sample faster, or do not use this node."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('saw', 3.0, 360.0, False)},
    {'key': 'c0', 'comment': True, 'text': 'an angle that wraps 0 -> 360 -> 0',
     'pos': (30, 215)},
    {'key': 'p0', 'init': 'plot', 'pos': (200, 118), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 360.0)},
    {'key': 'cr', 'init': 'continuous_rotation', 'pos': (30, 320), 'w': 190, 'h': 70},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 430), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 2000.0)},
    {'key': 'c1', 'comment': True, 'text': 'the same angle, unwrapped:\nit just keeps climbing', 'pos': (30, 430)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'p0', 'y'),
         ('sig', '', 'cr', 'rotation in'), ('cr', 'out', 'p1', 'y')]
print(build('continuous_rotation',
            'continuous_rotation - undo the wrap in an angle', body,
            demo, links, demo_width=480, text_width=790, text_height=650))
