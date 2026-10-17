"""Derive one help patch per trigonometry node from the hand-made arithmetic_help.json.

    python3 dpg_system/help/tools/derive_trig_help.py

sin, cos, tan, asin, acos and atan get the single-input layout of
derive_math_single_help.py, whose derive() this reuses: an angle (or a ratio)
through the function into a plot, a number box through it into a number box, and
a random array through it into a heat map. Their outlet is 'out', not 'result',
and they save their 'degrees' checkbox. trig_help.json remains the family
overview. After running this, use

    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HELP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from derive_math_single_help import ARRAYS, page, derive

D = math.pi / 180

DEGREES = """degrees:
A checkbox on the node, checked by default. Checked, angles are in degrees (a full
turn is 360); unchecked, in radians (a full turn is 2 x pi, about 6.283)."""

ANGLE_IN = 'The angle, in degrees or radians. Receiving data here triggers the calculation.'
RATIO_IN = 'The ratio. Receiving data here triggers the calculation.'

WHY_DEGREES = """DEGREES, NOT RADIANS:
Most programming languages work in radians. These nodes default to degrees, because
most angles in a patch - from sensors, rotations and interfaces - arrive in degrees.
Uncheck "degrees" when the angle comes from something that speaks radians."""

CLAMPED = """OUT OF RANGE:
Only values from -1 to 1 have an {f}. Rather than stop the patch, anything beyond
is treated as -1 or 1, so 1.5 gives the same answer as 1."""

RAMP = dict(shape='saw', period=3.0, range=360.0, bipolar=False)
WAVE = dict(shape='sin', period=2.0, range=1.0, bipolar=True)


def trig_page(intro, syntax_ex, examples, angle_in, result_text, notes, see_also):
    return page(intro, syntax_ex, examples,
                [DEGREES], result_text, notes + [WHY_DEGREES], see_also, out='out') \
        .replace('in:\nThe data to be operated on. Receiving data here triggers the calculation.',
                 'in:\n' + (ANGLE_IN if angle_in else RATIO_IN))


OPS = [
    dict(stem='sin', label='sin', title='sin (sine)', out='out', props={'degrees': True},
         fn=lambda a: math.sin(a * D),
         sig=dict(RAMP, y=(-1.2, 1.2),
                  note='an angle ramping from 0 to 360 degrees, through sin:\none full wave for each turn'),
         num=dict(value=30.0, speed=1.0),
         np=dict(rand=(0.0, 360.0), a=(0.0, 360.0), out=(-1.0, 1.0)),
         text=trig_page("""
sin gives the sine of an angle: how far up a point on a circle of radius 1 is, after
turning that angle from the right-hand side.

0 degrees comes out as 0, 30 as 0.5, 90 as 1, 180 as 0, 270 as -1, 360 as 0 again.
Every value is between -1 and 1. In the demo an angle ramps steadily round the circle
from 0 to 360 degrees, and sin turns it into a smooth wave - one wave for each turn.

Use it to turn a steady rotation into smooth back-and-forth movement, or to find the
vertical part of a direction: a thing moving at speed s in direction a moves up by
s x sin(a). sin and cos together turn an angle into a position on a circle.

""" + ARRAYS.format(label='sin'),
             'sin', """
sin        a smooth wave from a steadily turning angle
sin then * 100     the height of a point on a circle of radius 100""",
             True, 'The sine of the angle, between -1 and 1.', [], """
cos     the same wave a quarter turn earlier
asin    the reverse: the angle from a sine
signal  makes sine waves directly, at a frequency""")),

    dict(stem='cos', label='cos', title='cos (cosine)', out='out', props={'degrees': True},
         fn=lambda a: math.cos(a * D),
         sig=dict(RAMP, y=(-1.2, 1.2),
                  note='the same ramp through cos:\nthe wave starts at 1, a quarter turn ahead of sin'),
         num=dict(value=60.0, speed=1.0),
         np=dict(rand=(0.0, 360.0), a=(0.0, 360.0), out=(-1.0, 1.0)),
         text=trig_page("""
cos gives the cosine of an angle: how far across to the right a point on a circle of
radius 1 is, after turning that angle.

0 degrees comes out as 1, 60 as 0.5, 90 as 0, 180 as -1, 270 as 0, 360 as 1 again.
It is the same wave as sin, a quarter turn (90 degrees) ahead. In the demo an angle
ramps from 0 to 360 degrees, and cos starts at the top, where sin starts in the middle.

Use it with sin to place things on a circle - x is radius x cos(a), y is radius x
sin(a) - or to find how much of a movement lies along a direction: a vector at angle
a to an axis has cos(a) of its length along that axis.

""" + ARRAYS.format(label='cos'),
             'cos', """
cos then * 100     the x of a point on a circle of radius 100
cos        how much two directions agree: 1 the same, 0 at right angles, -1 opposite""",
             True, 'The cosine of the angle, between -1 and 1.', [], """
sin     the same wave a quarter turn later
acos    the reverse: the angle from a cosine""")),

    dict(stem='tan', label='tan', title='tan (tangent)', out='out', props={'degrees': True},
         fn=lambda a: math.tan(a * D),
         sig=dict(RAMP, y=(-5.0, 5.0),
                  note='the same ramp through tan:\nit shoots off to infinity at 90 and 270 degrees'),
         num=dict(value=45.0, speed=1.0),
         np=dict(rand=(-60.0, 60.0), a=(-60.0, 60.0), out=(-2.0, 2.0)),
         text=trig_page("""
tan gives the tangent of an angle: the slope of a line at that angle, how far it
rises for each step across. It is sin divided by cos.

0 degrees comes out as 0, 45 as 1 (one up for one across), -45 as -1, 60 as 1.732.
Near 90 degrees the line is nearly vertical and the slope grows without limit: 89
degrees gives 57, 89.9 gives 573. In the demo the ramp's tangent climbs off the top
of the plot at 90 degrees and comes back from the bottom - twice for each turn.

Use it for slopes and for heights from angles: something seen at an angle a above
the horizon, d away, is d x tan(a) high.

""" + ARRAYS.format(label='tan') + """ The random angles there run from
-60 to 60 degrees, safely away from the jumps at -90 and 90.""",
             'tan', """
tan        the slope of a line at an angle
tan then * 10      the height of something seen at that angle, 10 away""",
             True, 'The tangent of the angle. Any size; very large near 90 and 270 degrees.',
             ["""
AT 90 DEGREES:
The true tangent at exactly 90 or 270 degrees is infinite. Because the angle is
converted to radians first, the node gives a huge number (about 16 thousand trillion)
rather than infinity or an error. Keep inputs away from these angles, or clip the
result."""], """
atan    the reverse: the angle from a slope
sin, cos    tan is sin divided by cos""")),

    dict(stem='asin', label='asin', title='asin (arcsine)', out='out', props={'degrees': True},
         fn=lambda a: math.asin(a) / D,
         sig=dict(WAVE, y=(-100.0, 100.0),
                  note='a sine wave (-1 to 1) through asin:\nthe angle back again - a straight-sided triangle'),
         num=dict(value=0.5, speed=0.01),
         np=dict(rand=(-1.0, 1.0), a=(-1.0, 1.0), out=(-90.0, 90.0)),
         text=trig_page("""
asin is the reverse of sin: given a sine, between -1 and 1, it gives back the angle.

0.5 comes out as 30 degrees, 1 as 90, 0 as 0, -1 as -90.
The answer is always between -90 and 90 degrees. Many angles share a sine - 30 and
150 both have a sine of 0.5 - and asin always picks the one in that range.

In the demo a sine wave goes through asin. Undoing the sine straightens its curves,
so the plot shows the angle itself rising and falling at a steady rate - a triangle
wave between -90 and 90.

Use it to recover an angle from a height: how far a tilt sensor's reading of
gravity along one axis (as a fraction of 1 g) says it is tipped.

""" + ARRAYS.format(label='asin'),
             'asin', """
asin       tilt in degrees from an accelerometer axis reading, in g
asin       turn a sine wave into a triangle wave""",
             False, 'The angle, between -90 and 90 degrees (or -pi/2 and pi/2 radians).',
             [CLAMPED.format(f='arcsine')], """
sin     the forward direction: the sine of an angle
acos    the angle from a cosine
atan    the angle from a slope, and any input is valid""")),

    dict(stem='acos', label='acos', title='acos (arccosine)', out='out', props={'degrees': True},
         fn=lambda a: math.acos(a) / D,
         sig=dict(WAVE, y=(-10.0, 190.0),
                  note='a sine wave (-1 to 1) through acos:\nan angle swinging from 180 down to 0 and back'),
         num=dict(value=0.5, speed=0.01),
         np=dict(rand=(-1.0, 1.0), a=(-1.0, 1.0), out=(0.0, 180.0)),
         text=trig_page("""
acos is the reverse of cos: given a cosine, between -1 and 1, it gives back the angle.

0.5 comes out as 60 degrees, 1 as 0, 0 as 90, -1 as 180.
The answer is always between 0 and 180 degrees - as an angle between two directions
always is, which is what acos is most often used for.

In the demo a sine wave goes through acos: at -1 the angle is 180, at 1 it is 0, and
the curves straighten into a steady swing between them.

Use it for the angle between two directions: take the dot product of two unit
vectors (norm each first), and acos of it is the angle between them - 0 when they
point the same way, 90 at right angles, 180 when opposite.

""" + ARRAYS.format(label='acos'),
             'acos', """
acos       the angle between two unit vectors, from their dot product
acos       how far apart two directions are, 0 to 180 degrees""",
             False, 'The angle, between 0 and 180 degrees (or 0 and pi radians).',
             [CLAMPED.format(f='arccosine') + """ This matters for acos in
particular: a dot product of two unit vectors can come out as 1.0000001 through
rounding, and it should still give 0 degrees, not fail."""], """
cos     the forward direction: the cosine of an angle
asin    the angle from a sine
norm    make vectors unit length before a dot product""")),

    dict(stem='atan', label='atan', title='atan (arctangent)', out='out', props={'degrees': True},
         fn=lambda a: math.atan(a) / D,
         sig=dict(WAVE, range=10.0, y=(-100.0, 100.0),
                  note='a sine wave from -10 to 10 through atan:\nsqueezed into -90 to 90, the peaks flattened'),
         num=dict(value=1.0, speed=0.01),
         np=dict(rand=(-5.0, 5.0), a=(-5.0, 5.0), out=(-90.0, 90.0)),
         text=trig_page("""
atan is the reverse of tan: given a slope, it gives back the angle.

1 comes out as 45 degrees, 0 as 0, -1 as -45, 10 as 84.3, 1000 as 89.9.
Any number is a valid input, and the answer is always between -90 and 90 degrees.
In the demo a sine wave from -10 to 10 goes through atan: small values pass through
almost in proportion, large ones are squeezed towards 90 - the peaks flatten out.

Use it to find the angle of a slope, or as a soft limiter: atan of any value, however
large, stays within -90 and 90, so it tames a control that can run away, without the
hard corner of clamping it.

""" + ARRAYS.format(label='atan'),
             'atan', """
atan       the angle of a slope (rise divided by run)
* 2 then atan      gently squash an unbounded value into -90 to 90""",
             False, 'The angle, between -90 and 90 degrees (or -pi/2 and pi/2 radians).',
             ["""
DIRECTION FROM X AND Y:
atan of y / x gives the angle of a point x, y - but only for points on the right
(x above 0). Opposite points share a slope (1, 1 and -1, -1 both give 45), so when x
is negative add 180 degrees to get the full-circle direction. At x = 0 the slope is
infinite: the direction is straight up or down."""], """
tan     the forward direction: the slope of an angle
asin, acos    angles from a sine or cosine
clamp   a hard limit, where atan is a soft one""")),
]


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
