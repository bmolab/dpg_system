"""rotation representations, 6D, comparing rotations, normalising and aligning."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# -------------------------------------------------------- quaternion_to_euler
body = """These convert a rotation between the four ways of writing one down.

A rotation is one thing. There are several ways to describe it, each good at 
something and bad at something else, and most of the trouble in orientation work 
is a value in the wrong one.

THE FOUR REPRESENTATIONS:

quaternion        four numbers. No gimbal lock, composes cleanly, interpolates 
                  properly. Unreadable by eye. This is the one to STORE and 
                  COMPUTE in
euler angles      three angles about three axes. Readable, and what a person 
                  can type. Gimbal locks, and the order of the axes changes 
                  what the numbers mean
rotation matrix   nine numbers. What actually gets applied to a vector, and how 
                  a graphics system wants it
axis-angle        an axis and how far round it. The most physical description - 
                  it is what a rotation IS
rotation vector   the same thing packed into three numbers, with the angle 
                  carried in the length

THE NODES:

quaternion_to_euler      matrix_to_quaternion       quaternion_to_rotvec
euler_to_quaternion      quaternion_to_matrix       rotvec_to_quaternion
quaternion_to_axis_angle matrix_to_axis_angle       matrix_to_rotvec
axis_angle_to_quaternion

WHAT GIMBAL LOCK ACTUALLY COSTS YOU:
When two of an Euler triple's axes line up, a degree of freedom disappears - 
two of the three angles then do the same job, and the values jump about wildly 
for a rotation that is moving smoothly. Near that configuration, small real 
movements produce enormous changes in the numbers.

This is why converting sensor quaternions to Euler angles to do arithmetic on 
them, and converting back, is a mistake even though it reads more naturally. 
Whatever you were computing behaves badly in exactly the postures where the 
angles broke - typically arms overhead, which is where the interesting movement 
usually is.

Convert to Euler for DISPLAY, and to let a person type a value. Do the work in 
quaternions.

ORDER AND DEGREES:
euler_to_quaternion has an 'order' option because rotating about x then y then 
z gives a different result from z then y then x. There is no universal 
convention, so two systems that both say "Euler angles" will usually disagree. 
If a conversion is nearly right but wrong in a way that grows with the angle, 
the order is the first thing to check.

The 'degrees' option decides whether the angles are in degrees or radians. 
As elsewhere in this system, degrees is the default.

SYNTAX:
quaternion_to_euler
euler_to_quaternion

EXAMPLE:
quaternion_to_euler

INPUTS and PARAMETERS:

quaternion / rotation matrix / xyz rotation / rotation vector:
The rotation, in whichever form the node takes. Receiving it triggers the 
conversion.

degrees:
Whether angles are in degrees. On by default.

order (euler_to_quaternion):
Which axis order the three angles apply in.

offset x / offset y (quaternion_to_euler):
Constant offsets added to the result, for matching a convention that differs 
from this one.

OUTPUTS: 

The rotation in the target representation.

A NOTE ON QUATERNION ORDER:
The quaternions here are scalar-first - w, x, y, z. Several libraries use 
scalar-last, and a quaternion read in the wrong order is not an error, just a 
different and wrong rotation. If an imported orientation is confidently 
incorrect rather than noisy, suspect this."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 5.0 saw', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('saw', 5.0, 180.0, True)},
    {'key': 'pk', 'init': 'pack 3', 'pos': (30, 232), 'w': 140, 'h': 100},
    {'key': 'c0', 'comment': True, 'text': 'a sweeping angle on one axis',
     'pos': (30, 345)},
    {'key': 'eq', 'init': 'euler_to_quaternion', 'pos': (30, 385), 'w': 260, 'h': 120,
     'props': {'degrees': True}},
    {'key': 'l1', 'init': 'list', 'pos': (30, 520), 'w': 300, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'as a quaternion: w x y z', 'pos': (30, 570)},
    {'key': 'qe', 'init': 'quaternion_to_euler', 'pos': (30, 615), 'w': 260, 'h': 160,
     'props': {'degrees': True}},
    {'key': 'l2', 'init': 'list', 'pos': (30, 790), 'w': 300, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'and back again', 'pos': (30, 840)},
    {'key': 'qm', 'init': 'quaternion_to_matrix', 'pos': (350, 615), 'w': 240, 'h': 90},
    {'key': 'hm', 'init': 'heat_map', 'pos': (350, 720), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 3,
               'min y': -1.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c3', 'comment': True, 'text': 'the same rotation as a matrix',
     'pos': (350, 880)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'pk', 'in 1'),
         ('pk', 'out', 'eq', 'xyz rotation'),
         ('eq', 'quaternion rotation', 'l1', ''),
         ('eq', 'quaternion rotation', 'qe', 'quaternion'),
         ('qe', 'euler angles', 'l2', ''),
         ('eq', 'quaternion rotation', 'qm', 'quaternion'),
         ('qm', 'rotation matrix', 'hm', 'y')]
print(build('quaternion_to_euler', 'rotation conversions - four ways to say one thing',
            body, demo, links, demo_width=620, text_width=820, text_height=800))

# ----------------------------------------------------------- quaternion_to_6d
body = """The 6D representation: rotations in a form that does not jump.

WHY IT EXISTS:
Every compact way of writing a rotation is discontinuous somewhere. 
Euler angles jump at gimbal lock. Quaternions have the double-cover problem - 
q and minus q are the SAME rotation, so a smooth motion can flip sign and 
produce an enormous apparent change for no movement at all. Axis-angle wraps 
at a full turn.

Those discontinuities are harmless when you are just applying a rotation. 
They are ruinous when something has to LEARN or PREDICT one: a network trying 
to output a rotation cannot represent a jump smoothly, so it hedges near the 
discontinuity and produces wrong answers in that whole region.

The 6D form has no discontinuity. It is the first two columns of the rotation 
matrix - six numbers - and the third column is recovered by taking their cross 
product. Any six numbers can be turned back into a valid rotation, so nothing 
can be out of range and every path between two rotations is smooth.

THE NODES:

quaternion_to_6d  from a quaternion
matrix_to_6d      from a rotation matrix
6d_to_matrix      back to a matrix
6d_to_axis_angle  back to an axis and angle
6d_to_rotvec      back to a rotation vector

WHEN TO USE IT:
Whenever a rotation is the output or the target of something learned, and 
whenever you are interpolating or filtering rotations and want no surprises. 
For storing, applying and composing rotations, quaternions remain better - 
they are smaller and compose directly.

SYNTAX:
quaternion_to_6d
6d_to_matrix

EXAMPLE:
quaternion_to_6d

INPUTS and PARAMETERS:

quaternion / rotation matrix / 6d rotation:
The rotation. Receiving it triggers the conversion.

OUTPUTS: 

The rotation in the target form.

THE RECOVERY IS A NORMALISATION:
Going back from 6D does not simply reshape the numbers. The first column is 
normalised, the second is made perpendicular to it and normalised, and the 
third is their cross product. That is why arbitrary values still give a valid 
rotation - and it means the six numbers you get back after a round trip may not 
be the six you put in, while describing the identical rotation."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 5.0 saw', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('saw', 5.0, 180.0, True)},
    {'key': 'pk', 'init': 'pack 3', 'pos': (30, 232), 'w': 140, 'h': 100},
    {'key': 'eq', 'init': 'euler_to_quaternion', 'pos': (30, 350), 'w': 260, 'h': 120,
     'props': {'degrees': True}},
    {'key': 'q6', 'init': 'quaternion_to_6d', 'pos': (30, 490), 'w': 240, 'h': 90},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 595), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 6,
               'min y': -1.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c0', 'comment': True, 'text': 'six numbers, moving smoothly',
     'pos': (30, 755)},
    {'key': 'c1', 'comment': True, 'text': 'no jump anywhere in the sweep',
     'pos': (30, 785)},
    {'key': 'sm', 'init': '6d_to_matrix', 'pos': (300, 490), 'w': 220, 'h': 90},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (300, 595), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 3,
               'min y': -1.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c2', 'comment': True, 'text': 'and back to a full matrix',
     'pos': (300, 755)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'pk', 'in 1'), ('pk', 'out', 'eq', 'xyz rotation'),
         ('eq', 'quaternion rotation', 'q6', 'quaternion'),
         ('q6', 'rotation matrix', 'hm', 'y'),
         ('q6', 'rotation matrix', 'sm', '6d rotation'),
         ('sm', 'rotation matrix', 'hm2', 'y')]
print(build('quaternion_to_6d', '6D rotations - the form that does not jump', body,
            demo, links, demo_width=560, text_width=810, text_height=740))

# --------------------------------------------------------------- quaternion_diff
body = """These ask how two rotations differ, or how fast one is changing.

"The difference between two rotations" is itself a rotation - the one that 
takes you from the first to the second. That is what these produce, and it is 
more useful than subtracting the numbers, which is meaningless for quaternions.

THE NODES:

quaternion_diff        how much a rotation changed since the last frame
rotation_matrix_diff   the same, for matrices
quaternion_relative    the rotation from one quaternion to another
quaternion_distance    how far apart two rotations are, as a single number

diff VERSUS relative:
quaternion_diff compares each incoming rotation against the PREVIOUS one, so it 
is a rate - how much this joint turned during this frame. 
quaternion_relative compares two rotations you give it, so it is a relationship - 
how the forearm sits relative to the upper arm, how the head sits relative to 
the chest.

Both matter, and mixing them up gives you something that looks plausible and 
means nothing.

quaternion_distance GIVES A NUMBER:
A single value for how far apart two orientations are - the angle of the 
rotation between them. That is the right measure for "how close is this pose to 
that one", because it is invariant: it does not care which representation you 
used or which way round the quaternions happen to be signed.

Its 'reference' inlet is the rotation to compare against, and 'freeze ref' 
holds the current one as the reference so you can capture a target and then 
watch the distance from it.

'distance squared' skips the square root - cheaper, and monotonic with the 
distance, so it works identically for ranking or thresholding.

WHY NOT JUST SUBTRACT:
Quaternion components are not independent, and q and minus q are the same 
rotation. Subtracting two quaternions element by element gives a large answer 
for two identical rotations that happen to be signed oppositely, and a small one 
for genuinely different rotations that happen to be signed alike. These nodes 
handle the sign; arithmetic on the raw components does not.

SYNTAX:
quaternion_diff
quaternion_distance

EXAMPLE:
quaternion_relative

INPUTS and PARAMETERS:

quaternion:
The rotation, or the stream of them.

q1 / q2 (quaternion_relative):
The two rotations - the result takes you from q1 to q2.

reference / freeze ref (quaternion_distance):
What to measure against, and a way to capture it.

distanceAxis:
Which axis to measure along, when you want the difference in one direction 
rather than overall.

distance squared:
Skip the square root.

OUTPUTS: 

quaternion difference / q_diff:
The rotation between the two.

distance:
How far apart, as a number.

RELATED:
quaternion_diff_and_axis in the motion capture nodes does the same per-frame 
difference for a whole pose at once, and reports the axis as well as the size."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 4.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 4.0, 90.0, True)},
    {'key': 'pk', 'init': 'pack 3', 'pos': (30, 232), 'w': 140, 'h': 100},
    {'key': 'eq', 'init': 'euler_to_quaternion', 'pos': (30, 350), 'w': 260, 'h': 120,
     'props': {'degrees': True}},
    {'key': 'qd', 'init': 'quaternion_diff', 'pos': (30, 490), 'w': 240, 'h': 90},
    {'key': 'l1', 'init': 'list', 'pos': (30, 595), 'w': 300, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'how much it turned this frame',
     'pos': (30, 645)},
    {'key': 'qs', 'init': 'quaternion_distance', 'pos': (30, 690), 'w': 260, 'h': 160},
    {'key': 'f1', 'init': 'float', 'pos': (30, 865), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'click freeze ref to capture a target',
     'pos': (30, 915)},
    {'key': 'c2', 'comment': True, 'text': 'then watch the distance from it',
     'pos': (30, 945)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'pk', 'in 1'), ('pk', 'out', 'eq', 'xyz rotation'),
         ('eq', 'quaternion rotation', 'qd', 'quaternion'),
         ('qd', 'quaternion difference', 'l1', ''),
         ('eq', 'quaternion rotation', 'qs', 'quaternion'),
         ('qs', 'distance', 'f1', '')]
print(build('quaternion_diff', 'comparing rotations - how much, and relative to what',
            body, demo, links, demo_width=560, text_width=810, text_height=760))

# -------------------------------------------------------------- quaternion_norm
body = """Two nodes for keeping rotations valid and putting them in the right frame.

THE NODES:

quaternion_norm  scale a quaternion back to unit length
tracker_align    align an inertial body with an external tracker's world

quaternion_norm AND WHY DRIFT HAPPENS:
Only unit-length quaternions represent rotations. Composing many of them, or 
interpolating, or simply accumulating floating point error over a long run, 
lets the length wander away from 1 - and a quaternion that is not unit length 
scales as well as rotates, so things slowly grow or shrink.

The symptom is a body that gradually inflates or deflates over minutes rather 
than anything obviously wrong. Normalising in the chain costs nothing and 
removes the possibility.

tracker_align AND THE TWO WORLDS PROBLEM:
An inertial suit knows which way is down, and takes its heading from the 
magnetic field. An external tracker knows where things are in ITS room 
coordinates. Neither is wrong, and they disagree about which way is north - so 
the suit's body faces one way and the tracker's position moves in another, and 
the two drift apart as the performer turns.

This node measures that disagreement as a yaw offset and corrects it. 
'calibrate' captures the current offset. 'continuous' keeps tracking it, which 
matters because magnetometer heading drifts over a session rather than staying 
put. 'smoothing' controls how fast the correction follows - high enough that a 
momentary disagreement does not swing the body, low enough that real drift is 
followed.

SYNTAX:
quaternion_norm
tracker_align

EXAMPLE:
quaternion_norm

INPUTS and PARAMETERS:

quaternions (quaternion_norm):
One quaternion or a whole pose. All of them are normalised.

imu root quat (tracker_align):
The suit's root orientation - a single quaternion, or a full 37-joint Shadow 
pose, or a 20-joint active pose, from which the root is taken.

tracker pos / tracker quat:
What the external tracker reports.

body offset:
Where the tracker sits relative to the suit's root, in body-local coordinates.

calibrate:
Capture the current yaw disagreement as the offset.

continuous:
Keep following it. On by default.

smoothing:
How quickly the correction follows. Default 0.88.

OUTPUTS: 

normalized:
Unit-length quaternions.

corrected pos:
The tracker position, brought into the suit's world.

correction quat / yaw offset:
The correction being applied, and its size - worth watching, because a yaw 
offset that keeps growing means the magnetometer heading is drifting, which is 
a sensor problem rather than something to keep correcting.

RELATED:
mag_yaw_correct addresses the same magnetic heading error at the sensor level, 
per limb. This one reconciles a whole body with an external reference."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 4.0 saw', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('saw', 4.0, 180.0, True)},
    {'key': 'pk', 'init': 'pack 3', 'pos': (30, 232), 'w': 140, 'h': 100},
    {'key': 'eq', 'init': 'euler_to_quaternion', 'pos': (30, 350), 'w': 260, 'h': 120,
     'props': {'degrees': True}},
    {'key': 'mul', 'init': '* 1.02', 'pos': (30, 490), 'w': 140, 'h': 70,
     'props': {'operand': 1.02}},
    {'key': 'c0', 'comment': True, 'text': 'scaled slightly: no longer unit length',
     'pos': (30, 570)},
    {'key': 'qn', 'init': 'quaternion_norm', 'pos': (30, 610), 'w': 240, 'h': 90},
    {'key': 'l1', 'init': 'list', 'pos': (30, 715), 'w': 320, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'back to length 1, same rotation',
     'pos': (30, 765)},
    {'key': 'ta', 'init': 'tracker_align', 'pos': (30, 810), 'w': 280, 'h': 240},
    {'key': 'c2', 'comment': True, 'text': 'reconciles the suit and a tracker',
     'pos': (30, 1065)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'pk', 'in 1'), ('pk', 'out', 'eq', 'xyz rotation'),
         ('eq', 'quaternion rotation', 'mul', 'in'),
         ('mul', 'result', 'qn', 'quaternions'),
         ('qn', 'normalized', 'l1', ''),
         ('eq', 'quaternion rotation', 'ta', 'imu root quat')]
print(build('quaternion_norm', 'quaternion_norm and tracker_align - staying valid',
            body, demo, links, demo_width=580, text_width=800, text_height=740))
