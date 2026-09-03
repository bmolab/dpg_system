import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter


# ------------------------------------------------------------------ arithmetic
body = """The arithmetic nodes each perform one calculation on the value arriving at the left inlet, 
using a second value - the operand - held in the right inlet.

They all work the same way. The left inlet takes the data and triggers the calculation; 
the right inlet holds the other number, which you can type, drag, or send from elsewhere. 
The operand keeps its value between calculations, so you set it once and it stays put.

All of them accept single numbers, lists, NumPy arrays and PyTorch tensors. 
When the input and the operand are different kinds of thing, the operand is converted 
to match the input, so feeding an array into a node with a single number as its operand 
applies that number to every element.

You give the starting operand as an argument when you create the node. 
Whether you write it with a decimal point matters: "* 2" gives you an integer operand 
and a stepped slider, "* 2.0" gives you a floating point one.

THE NODES:

+     add                    -     subtract
*     multiply               /     divide
//    divide, discarding any remainder
%     modulo - the remainder after division
^     raise to a power       pow   the same as ^
min   the smaller of the two values
max   the larger of the two values
mod   the same as %
!-    reverse subtract - operand minus input, rather than input minus operand
!/    reverse divide - operand divided by input
perm  permutations: the number of ordered ways to pick operand items from input items
combination   the number of unordered ways to make that same pick

SYNTAX:
<operator> <operand: int or float>

EXAMPLE:
* 0.5

INPUTS and PARAMETERS:

in:
The data to be operated on. Receiving data here triggers the calculation.

operand:
The second value in the calculation. 
Setting it does not itself produce output - the node waits for the next value at "in".

OUTPUTS: 

result:
The result of the calculation, in the same shape as the input.

A NOTE ON DIVISION:
Dividing by zero does not raise an error and does not stop your patch. 
For single numbers the result is 0; for arrays the division is done with 
NumPy's error reporting switched off. Likewise "%" and "mod" return 0 
when the operand is zero. The idea is that a live patch should keep running."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 2.0)},
    {'key': 'mul', 'init': '* 0.5', 'pos': (30, 232), 'w': 120, 'h': 70,
     'props': {'operand': 0.5}},
    {'key': 'p1', 'init': 'plot', 'pos': (210, 205), 'w': 208, 'h': 176,
     'props': PLOT()},
    {'key': 'c1', 'comment': True, 'text': 'a sine wave, halved', 'pos': (30, 310)},
    {'key': 'c2', 'comment': True, 'text': 'drag the operand and watch it change',
     'pos': (30, 340)},
    {'key': 'i1', 'init': 'int', 'pos': (30, 405), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'add', 'init': '+ 100', 'pos': (30, 462), 'w': 120, 'h': 70,
     'props': {'operand': 100}},
    {'key': 'i2', 'init': 'int', 'pos': (30, 552), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c3', 'comment': True, 'text': 'drag this number', 'pos': (172, 410)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'mul', 'in'), ('mul', 'result', 'p1', 'y'),
         ('i1', 'int out', 'add', 'in'), ('add', 'result', 'i2', '')]
print(build('arithmetic', 'arithmetic - one calculation, one operand', body,
            demo, links, demo_width=440, text_width=800, text_height=660))

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
    {'key': 'c1', 'comment': True, 'text': 'a sine wave becomes a square wave',
     'pos': (30, 345)},
    {'key': 'c2', 'comment': True, 'text': 'drag the operand to move the threshold',
     'pos': (30, 375)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'gt', 'in'), ('gt', 'result', 'p1', 'y')]
print(build('comparison', 'comparison - test a value, report true or false', body,
            demo, links, demo_width=445, text_width=790, text_height=620))


# ---------------------------------------------------------------------- change
body = """These nodes act as gates that let a value through only when it passes a test. 
Unlike the comparison nodes, what comes out is the value itself, not true or false.

change, increasing and decreasing all compare the incoming value against the 
PREVIOUS one. They are filters on a stream: they suppress the frames where 
nothing interesting happened, and pass the ones where something did.

pass is the same machinery with the test switched off, so everything goes through. 
On its own it is a patch-cord tidy; given arguments it becomes a gate with 
whatever comparison you choose.

THE NODES:

change       passes a value only when it differs from the one before it
increasing   passes a value only when it is larger than the one before it
decreasing   passes a value only when it is smaller than the one before it
pass         passes everything, unless you give it a comparison to apply

You use change to stop a slow part of the patch being retriggered by a stream 
that is repeating the same number, and increasing or decreasing to catch the 
moment a reading starts to move one way.

SYNTAX:
change
pass <comparison> <operand: int or float>

EXAMPLE:
change
pass >= 0.5

INPUTS and PARAMETERS:

in:
The value to be tested and possibly passed on. Receiving data here triggers the node.

comparison_property:
The test to apply, chosen from a menu: !=, ==, >, >=, <, <= or always. 
change, increasing and decreasing set this for you; you can still change it by hand.

operand_property:
The value being compared against. 
While self_compare is checked this is filled in automatically after each value 
arrives, and shows you what the node will compare against next.

self_compare:
When checked, the node compares each value against the previous one - this is 
what makes change, increasing and decreasing behave the way their names suggest. 
When unchecked, it compares against a fixed operand you set yourself, 
turning the node into a threshold gate.

force_int:
When checked, values are rounded to whole numbers before being compared. 
Useful when a noisy float stream would otherwise register as "changed" on every frame. 
It affects only the comparison - what comes out is the unrounded value.

OUTPUTS: 

result:
The input value, unchanged, on the occasions when the test succeeds. 
When the test fails, nothing at all is sent. 

For arrays and tensors the test passes if ANY element satisfies it, 
and the whole array is then sent."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('saw', 4.0, 5.0, False)},
    {'key': 'fl', 'init': 'floor', 'pos': (30, 232), 'w': 100, 'h': 50},
    {'key': 'c0', 'comment': True, 'text': 'a staircase - the same number for many frames',
     'pos': (30, 290)},
    {'key': 'i0', 'init': 'int', 'pos': (215, 235), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'ch', 'init': 'change', 'pos': (30, 330), 'w': 120, 'h': 100,
     'props': {'self_compare': True, 'force_int': False}},
    {'key': 'cnt', 'init': 'counter', 'pos': (215, 335), 'w': 123, 'h': 84,
     'props': {'step': 1}},
    {'key': 'i1', 'init': 'int', 'pos': (215, 430), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 'counts steps, not frames', 'pos': (30, 442)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'fl', 'in'), ('fl', 'result', 'i0', ''),
         ('fl', 'result', 'ch', 'in'), ('ch', 'result', 'cnt', 'input'),
         ('cnt', 'count out', 'i1', '')]
print(build('change', 'change - pass a value on only when it has moved', body,
            demo, links, demo_width=380, text_width=800, text_height=690))

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


# ------------------------------------------------------------------ accumulate
body = """The accumulate node adds up everything it is given and reports the running total.

Each value that arrives is added to the total so far, and the new total is sent out. 
It is a running sum, kept between frames - the node remembers.

You use it to integrate a stream: turning a rate into a distance, a speed into a 
position, or a series of increments into a count. 
Feeding it a stream of 1s makes it a simple counter.

SYNTAX:
accumulate

EXAMPLE:
accumulate

INPUTS and PARAMETERS:

in:
The value to add to the running total. 
Receiving data here triggers the node - it adds, then sends the new total. 
Negative values subtract, so a stream that goes both ways will wander up and down.

set:
Sets the running total directly, discarding what was there. 
The new total is sent out immediately. 
Use this to jump the accumulator to a known starting point.

reset:
A button that returns the total to zero and sends that zero out. 
Click it, or send it anything from elsewhere in the patch.

OUTPUTS: 

sum:
The running total after the latest value has been added.

A NOTE ON TYPE:
Incoming values are converted to a single number - an integer if they look like 
one, otherwise a float. This node keeps a scalar total; 
it does not accumulate arrays element by element."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 100.0, 'units': 'milliseconds'}},
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 20), 'w': 45, 'h': 42,
     'props': {'': True}},
    {'key': 'c0', 'comment': True, 'text': 'ten values a second', 'pos': (172, 80)},
    {'key': 'tt', 'init': 't 1', 'pos': (30, 148), 'w': 22, 'h': 46},
    {'key': 'c1', 'comment': True, 'text': 'each tick becomes the number 1', 'pos': (68, 152)},
    {'key': 'acc', 'init': 'accumulate', 'pos': (30, 215), 'w': 140, 'h': 100},
    {'key': 'i1', 'init': 'int', 'pos': (30, 340), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'click reset on the node to zero it', 'pos': (185, 250)},
]
links = [('tog', '', 'met', 'on'), ('met', '', 'tt', ''),
         ('tt', '1', 'acc', 'in'), ('acc', 'sum', 'i1', '')]
print(build('accumulate', 'accumulate - a running total', body,
            demo, links, demo_width=430, text_width=780, text_height=520))

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
    {'key': 'c1', 'comment': True, 'text': 'the same angle, unwrapped:', 'pos': (30, 430)},
    {'key': 'c2', 'comment': True, 'text': 'it just keeps climbing', 'pos': (30, 460)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'p0', 'y'),
         ('sig', '', 'cr', 'rotation in'), ('cr', 'out', 'p1', 'y')]
print(build('continuous_rotation',
            'continuous_rotation - undo the wrap in an angle', body,
            demo, links, demo_width=480, text_width=790, text_height=650))
