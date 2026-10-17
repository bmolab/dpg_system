"""shadow capture, pose handling, drawing, skeleton conversion."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ---------------------------------------------------------------------- shadow
body = """These read live data from a Shadow suit.

THE NODES:

shadow         the suit: joint orientations and positions, for up to four bodies
shadow_sensor  the raw sensor readings from one sensor - magnetometer, 
               accelerometer and gyroscope, before any fusion

shadow_pose is not a suit node: it builds a 37-row pose from single joints, 
and is described on the pose help patch.

WHAT THE SUIT GIVES YOU:
Each body has a quaternions outlet and a positions outlet. The quaternions are 
the joint ORIENTATIONS, which is the real measurement - the suit is a set of 
inertial sensors, and orientation is what they produce. The positions are 
inferred from those orientations plus assumed limb lengths, so they are a 
derived quantity and only as good as the assumptions behind them.

That distinction matters constantly. An orientation error is a sensor problem; 
a position error may be an orientation error, or a limb length, or the root 
inference - see the sensor_to_root help patch.

shadow_sensor IS FOR DIAGNOSING, NOT PERFORMING:
It gives you what one sensor actually reads before the fusion combines them. 
The magnetometer outlet is the one that matters for troubleshooting: a sensor 
sitting in a distorted field produces a yaw error that no amount of filtering 
downstream will fix, and the only way to see it is to look at the raw field. 
Patch it into mag_offset to measure that.

SYNTAX:
shadow
shadow_sensor

EXAMPLE:
shadow

INPUTS and PARAMETERS:

flush (shadow):
Discard whatever is queued and start from the current frame. Use it when the 
stream has fallen behind and you would rather lose data than lag.

reconnect:
Re-establish the connection to the suit.

OUTPUTS: 

body N quaternions:
The joint orientations for that body - the measurement.

body N positions:
Joint positions, inferred from the orientations and the limb lengths.

magnetometer / accelerometer / gyroscope (shadow_sensor):
The raw readings from one sensor.

A NOTE ON MULTIPLE BODIES:
The four bodies are separate suits, and the outlets are static - body 3's 
outlets exist whether or not a third suit is connected, and simply stay quiet. 
So a patch built for two performers works unchanged with one."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'c0', 'comment': True, 'text': 'quaternions are the measurement\npositions are inferred from them',
     'pos': (30, 395)},
    {'key': 'aj', 'init': 'active_joints', 'pos': (30, 470), 'w': 220, 'h': 90},
    {'key': 'c2', 'comment': True, 'text': 'the 20 joints most patches want',
     'pos': (30, 575)},
    {'key': 'ss', 'init': 'shadow_sensor', 'pos': (350, 62), 'w': 240, 'h': 180},
    {'key': 'mo', 'init': 'mag_offset', 'pos': (350, 260), 'w': 240, 'h': 220},
    {'key': 'c3', 'comment': True, 'text': 'raw field, for diagnosing a bad sensor',
     'pos': (350, 495)},
]
links = [('sh', 'body 1 quaternions', 'aj', 'full pose quats in'),
         ('ss', 'magnetometer', 'mo', 'magnetometer')]
print(build('shadow', 'shadow - live data from the suit', body, demo, links,
            demo_width=620, text_width=800, text_height=700))

# ------------------------------------------------------------------------ pose
# A quaternion stream every pose demo below shares: a swinging angle about x,
# packed to x y z and turned into one w x y z quaternion (an array, which is
# what the pose nodes need - they ignore plain lists).
def swing(x=30, y=62):
    return [
        {'key': 'sig', 'init': 'signal 4.0 sin', 'pos': (x, y), 'w': 129, 'h': 78,
         'props': SIG('sin', 4.0, 90.0, True)},
        {'key': 'pk', 'init': 'pack 3', 'pos': (x, y + 100), 'w': 140, 'h': 100},
        {'key': 'eq', 'init': 'euler_to_quaternion', 'pos': (x, y + 215), 'w': 260, 'h': 120,
         'props': {'degrees': True}},
        {'key': 'cs', 'comment': True, 'text': 'one joint swinging 90 degrees\neach way about x',
         'pos': (x + 300, y + 215)},
    ]

SWING_LINKS = [('sig', '', 'pk', 'in 1'),
               ('pk', 'out', 'eq', 'xyz rotation')]

body = """pose builds a whole pose out of single joints. It has one inlet per joint, and
whenever any of them receives a rotation, the whole pose goes out.

THE NODES:

pose         20 joint inlets in, a 20-joint pose out
shadow_pose  the same 20 inlets, but the pose out has the 37 rows of the
             raw Shadow layout

WHAT A POSE IS:
An array of joint orientations, one row per joint, each a quaternion
w x y z with the real part first. pose sends the 20 active joints in the
order active_joints uses: base of skull, upper vertebrae, mid vertebrae,
lower vertebrae, spine pelvis, pelvis anchor, then left hip, knee and ankle,
right hip, knee and ankle, left shoulder blade, shoulder, elbow and wrist,
and the same four on the right. So a pose made here can go anywhere a
reduced suit pose goes.

shadow_pose puts each of the same 20 inlets at its row in the 37-row layout
the shadow node sends. The 17 rows without an inlet are always the identity.

JOINTS YOU DO NOT SET:
An inlet that has received nothing counts as the identity - no rotation - and
an inlet keeps the last rotation it received. So you can drive two or three
joints and leave the rest of the body at rest.

WHAT AN INLET TAKES:
Four numbers, w x y z: a numpy array, a torch tensor, or a plain list - from
a message box, say. euler_to_quaternion turns angles into one, as the example
does.

SYNTAX:
pose
shadow_pose

EXAMPLE:
pose

INPUTS:

base of skull ... right wrist:
One quaternion each, w x y z. Any of them arriving sends the pose.

OUTPUT:

pose out:
The pose as a numpy array: 20 rows of four for pose, 37 rows of four for
shadow_pose.

RELATED:
active_pose shows a whole pose row by row, and lets you edit it.
active_joints reduces a 37-row Shadow pose to these 20 joints.
gl_body draws a pose of either size."""

demo = swing() + [
    {'key': 'po', 'init': 'pose', 'pos': (30, 400), 'w': 160, 'h': 300},
    {'key': 'c0', 'comment': True, 'text': 'into the left elbow inlet only:\nevery other joint stays at rest',
     'pos': (330, 400)},
    {'key': 'ap', 'init': 'active_pose', 'pos': (30, 720), 'w': 300, 'h': 520},
    {'key': 'c1', 'comment': True, 'text': 'the pose that comes out - 20 rows,\nonly left_elbow moving', 'pos': (330, 720)},
]
links = SWING_LINKS + [('eq', 'quaternion rotation', 'po', 'left elbow'),
                       ('po', 'pose out', 'ap', 'in')]
print(build('pose', 'pose - a pose built from single joints', body, demo, links,
            demo_width=620, text_width=810, text_height=870))

# ----------------------------------------------------------------- active_pose
body = """active_pose shows a 20-joint pose one row per joint, and lets you edit it.

WHAT IT SHOWS:
Twenty rows, labelled base_of_skull through right_wrist in the order
active_joints and pose use. Each row is one joint's rotation as a quaternion,
w x y z with the real part first. Every joint starts at the identity, 1 0 0 0,
which is no rotation.

EDITING:
Change any number in a row and the whole pose goes out of 'out'. The numbers
go out exactly as typed - nothing normalises them - so keep each row a unit
quaternion, its four squares adding up to one. 0.707 0.707 0 0 is a quarter
turn about x; 1 1 0 0 is not a rotation at all.

A POSE ARRIVING:
A pose sent into 'in' replaces the rows and goes straight out, so active_pose
also works as a readout of a live pose. While a stream is arriving, it
overwrites your edits on the next frame. A bang into 'in' sends the rows as
they are.

Send it 20 rows. It shows whatever shape arrives, but the joint labels belong
to the first 20 rows, so a pose of another size is labelled wrongly.

The rows are saved with the patch.

SYNTAX:
active_pose

EXAMPLE:
active_pose

INPUTS:

in:
A 20-joint pose, or a bang to send the current rows.

base_of_skull ... right_wrist:
The rows themselves.

reset:
Set every joint back to the identity, and send the pose.

output type:
numpy (the default), torch or list.

OPTIONS:
save and load write the rows to a .npy file and read them back.

OUTPUT:

out:
The pose, 20 rows of four.

RELATED:
pose builds a pose from single joints.
pose_adjust applies fixed corrections to a pose, joint by joint.
quaternion_to_euler shows the rows as angles, as in the example."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click to send the pose', 'pos': (330, 62)},
    {'key': 'ap', 'init': 'active_pose', 'pos': (30, 130), 'w': 300, 'h': 520,
     'props': {'[14]': [0.7071, 0.7071, 0.0, 0.0]}},
    {'key': 'c1', 'comment': True, 'text': 'left_elbow is set to a quarter\nturn about x - edit any row\nand the pose goes out',
     'pos': (330, 130)},
    {'key': 'qe', 'init': 'quaternion_to_euler', 'pos': (30, 670), 'w': 260, 'h': 160,
     'props': {'degrees': True}},
    {'key': 've', 'init': 'vector', 'pos': (30, 850), 'w': 260, 'h': 520},
    {'key': 'c2', 'comment': True, 'text': 'the same pose as x y z angles\nin degrees, one row per joint',
     'pos': (330, 850)},
]
links = [('btn', '', 'ap', 'in'),
         ('ap', 'out', 'qe', 'quaternion'),
         ('qe', 'euler angles', 've', 'in')]
print(build('active_pose', 'active_pose - see and edit a 20-joint pose', body, demo, links,
            demo_width=620, text_width=810, text_height=920))

# --------------------------------------------------------------- active_joints
body = """active_joints reduces a full 37-row Shadow pose to the 20 joints most work
uses.

WHY 20:
The shadow node sends 37 rows per body. Twenty of them are the joints that
carry the movement of the body: the spine and head, the hips, knees and
ankles, the shoulder blades, shoulders, elbows and wrists. Reducing early
keeps everything downstream smaller and in one agreed order.

WHAT IT TAKES:
A pose of 37 rows, as a numpy array or a torch tensor - straight from the
shadow node's 'body N quaternions' outlet, from a take, or from shadow_pose.
Anything else produces nothing: a pose that is already 20 rows, a flat array
of 148 numbers, or a list.

WHAT IT SENDS:
The 20 rows, in this order: base of skull, upper vertebrae, mid vertebrae,
lower vertebrae, spine pelvis, pelvis anchor, left hip, knee and ankle, right
hip, knee and ankle, left shoulder blade, shoulder, elbow and wrist, then the
same four on the right.

It picks rows and does not change them, and sends the same kind it received:
numpy in, numpy out; torch in, torch out.

SYNTAX:
active_joints

EXAMPLE:
active_joints

INPUTS:

full pose quats in:
A 37-row Shadow pose.

OUTPUT:

active joint quats out:
The 20-row pose.

RELATED:
shadow_pose builds a 37-row pose from single joints, as in the example.
active_pose shows a 20-joint pose with its joint names.
body_to_joints splits a pose into one outlet per joint."""

demo = swing() + [
    {'key': 'sp', 'init': 'shadow_pose', 'pos': (30, 400), 'w': 160, 'h': 300},
    {'key': 'c0', 'comment': True, 'text': 'a 37-row pose with the left elbow moving\n- the same shape the shadow node sends',
     'pos': (330, 400)},
    {'key': 'aj', 'init': 'active_joints', 'pos': (30, 720), 'w': 220, 'h': 90},
    {'key': 'c1', 'comment': True, 'text': '37 rows in, 20 out', 'pos': (330, 720)},
    {'key': 'ap', 'init': 'active_pose', 'pos': (30, 830), 'w': 300, 'h': 520},
    {'key': 'c2', 'comment': True, 'text': 'the left elbow lands on its\nown row of the 20', 'pos': (330, 830)},
]
links = SWING_LINKS + [('eq', 'quaternion rotation', 'sp', 'left elbow'),
                       ('sp', 'pose out', 'aj', 'full pose quats in'),
                       ('aj', 'active joint quats out', 'ap', 'in')]
print(build('active_joints', 'active_joints - 37 Shadow joints down to 20', body, demo, links,
            demo_width=620, text_width=810, text_height=720))

# ----------------------------------------------------------------- target_pose
body = """target_pose compares each pose that arrives with a stored target pose, and says
how far apart they are - in total, and joint by joint.

HOW FAR:
For each joint it measures the angle between the joint's rotation in the pose
and its rotation in the target: the size of the single turn that would take
one to the other. It is in radians, from 0 when they match to 3.14 (half a
turn) at most. A quaternion and its negative are the same rotation, and
count as a match.

'score out' is the sum of those angles over all the joints.

'axis distances out' sends the angles themselves, one per joint in the pose's
row order. Despite the name, it is one number per JOINT, not per axis: it
tells you which joints are off, rather than only how much in all.

Every row of both is normalised before comparing, so a quaternion that is
not quite unit length does not distort the result.

CAPTURING THE TARGET:
Click 'capture in', and the NEXT pose to arrive becomes the target. Nothing
is sent for that pose; the poses after it are scored against it.

A bang sent into 'capture in' does the same as a click.

A number - a toggle, say - switches CONTINUOUS capture: while it is non-zero,
every pose that arrives becomes the target and nothing is scored; 0 switches
it off, and the poses after that are scored against the last one captured.

The target is held in memory only. It is not saved with the patch.

BEFORE THE FIRST CAPTURE:
The target starts as 20 joints at rest, so the score is how far the pose is
from the rest pose. A pose with more than 20 rows - a 37-row Shadow pose -
fails with an error until a pose of that size has been captured.

SYNTAX:
target_pose

EXAMPLE:
target_pose

INPUTS:

pose in:
A pose: rows of w x y z quaternions.

capture in:
Click (or bang) to take the next pose as the target; 1 / 0 to capture
continuously or stop.

OUTPUTS:

score out:
The total angle between pose and target, in radians.

axis distances out:
The angle for each joint, in radians.

RELATED:
calibrate_pose captures a pose and measures every later one FROM it.
quaternion_distance measures between two single rotations."""

demo = swing() + [
    {'key': 'po', 'init': 'pose', 'pos': (30, 400), 'w': 160, 'h': 300},
    {'key': 'btn', 'init': 'button', 'pos': (250, 720), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click: the next pose becomes the target',
     'pos': (360, 720)},
    {'key': 'tp', 'init': 'target_pose', 'pos': (30, 780), 'w': 240, 'h': 140},
    {'key': 'f1', 'init': 'float', 'pos': (30, 940), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'total angle from the target, radians\nat rest until you capture',
     'pos': (360, 940)},
    {'key': 've', 'init': 'vector', 'pos': (30, 1000), 'w': 200, 'h': 520},
    {'key': 'c2', 'comment': True, 'text': 'one angle per joint: only the\nleft elbow (row 14) is ever off',
     'pos': (360, 1000)},
]
links = SWING_LINKS + [('eq', 'quaternion rotation', 'po', 'left elbow'),
                       ('po', 'pose out', 'tp', 'pose in'),
                       ('btn', '', 'tp', 'capture in'),
                       ('tp', 'score out', 'f1', ''),
                       ('tp', 'axis distances out', 've', 'in')]
print(build('target_pose', 'target_pose - how far from a captured pose', body, demo, links,
            demo_width=620, text_width=810, text_height=930))

# ----------------------------------------------------------------- pose_adjust
body = """pose_adjust applies a fixed rotation to chosen joints of a 20-joint pose - for a
pose that is right in shape but wrong in detail, such as a sensor that sits a
little off on one limb.

WHAT IT TAKES:
A 20-joint active pose: 20 rows of w x y z, or a flat list of 80 numbers. A
pose of any other size produces nothing - reduce a 37-row Shadow pose with
active_joints first.

THE CORRECTIONS:
One row of four numbers per joint, a quaternion w x y z. 1 0 0 0, the
identity, means no correction. Keep each row unit length - its four squares
adding to one - because nothing normalises it, and a longer or shorter row
scales the joint's quaternion as well as turning it.

The rows have short names, in the same order as the active joints:

head     base of skull         l_hip ... l_ankle      left leg
neck     upper vertebrae       r_hip ... r_ankle      right leg
spine3   mid vertebrae         l_collar ... l_wrist   left arm, from the
spine2   lower vertebrae                              shoulder blade out
spine1   spine pelvis          r_collar ... r_wrist   right arm
pelvis   pelvis anchor

Changing a correction does not send anything; the next pose to arrive is
corrected with it.

WHICH FRAME A CORRECTION IS IN:
The frame the rotation is applied in decides whether the correction holds as
the body moves.

The pelvis correction is always in the body's own frame. Every other joint's
correction is, by default, in its PARENT bone's frame. The 'child frame'
checkbox switches them all to the joint's OWN frame.

A sensor mounted on the child side of a joint produces an error in the
child's frame, and correcting it in the parent's will drift as the joint
bends. If a correction looks right in one posture and wrong in another, this
is the setting.

In the example the left elbow swings about x and is corrected by a quarter
turn about z. Tick 'child frame' and the corrected angles change, because
the two frames differ once the elbow has turned.

SYNTAX:
pose_adjust

EXAMPLE:
pose_adjust

INPUTS:

pose in:
The 20-joint pose.

child frame:
Off: corrections in the parent bone's frame. On: in the joint's own frame.

reset:
Set every correction back to the identity.

head ... r_wrist:
The correction for each joint. Saved with the patch and with presets.

OUTPUT:

pose out:
The corrected pose, 20 rows. Joints without a correction come out exactly
as they went in.

RELATED:
smpl_pose_adjust does the same for an SMPL pose, with the corrections in
axis-angle form.
calibrate_pose works out a correction from a captured pose rather than
taking one you set."""

demo = swing() + [
    {'key': 'po', 'init': 'pose', 'pos': (30, 400), 'w': 160, 'h': 300},
    {'key': 'pa', 'init': 'pose_adjust', 'pos': (30, 720), 'w': 300, 'h': 560,
     'props': {'l_elbow': [0.7071, 0.0, 0.0, 0.7071]}},
    {'key': 'c0', 'comment': True, 'text': 'l_elbow: a quarter turn about z\ntick child frame and watch row 14 change',
     'pos': (360, 720)},
    {'key': 'qe', 'init': 'quaternion_to_euler', 'pos': (30, 1300), 'w': 260, 'h': 160,
     'props': {'degrees': True}},
    {'key': 've', 'init': 'vector', 'pos': (30, 1480), 'w': 260, 'h': 520},
    {'key': 'c1', 'comment': True, 'text': 'the corrected pose, as x y z\nangles in degrees', 'pos': (360, 1480)},
]
links = SWING_LINKS + [('eq', 'quaternion rotation', 'po', 'left elbow'),
                       ('po', 'pose out', 'pa', 'pose in'),
                       ('pa', 'pose out', 'qe', 'quaternion'),
                       ('qe', 'euler angles', 've', 'in')]
print(build('pose_adjust', 'pose_adjust - fixed corrections, joint by joint', body, demo, links,
            demo_width=640, text_width=810, text_height=1190))

# -------------------------------------------------------------- calibrate_pose
body = """calibrate_pose captures a reference pose and, from then on, sends every pose as
it stands relative to that reference.

WHAT CALIBRATING DOES:
Click 'calibrate input' and it takes the pose currently at 'pose input' and
stores the opposite of each joint's rotation. Every pose that arrives after
that has each joint combined with its stored opposite. So in the captured
posture every joint reads as the identity, 1 0 0 0, and away from it each
joint reads as the turn it has made since, in its parent bone's frame.

That is how to make a performer's own stance the zero: have them hold it,
click, and the pose becomes a measure of departure from it.

NOTHING COMES OUT UNTIL IT IS CALIBRATED:
Before the first calibration, 'pose out' sends nothing at all - not the pose
unchanged. Click 'calibrate input' while a pose is arriving, or send it a
stored calibration.

KEEPING A CALIBRATION:
The capture is not saved with the patch. It goes out of 'calibration output'
as soon as it is taken - the stored opposites, one row per joint, not the
captured pose itself - and sending those rows into 'calibration input' later
restores it.

ANY NUMBER OF JOINTS:
It works row by row, so a 20-joint pose, a 37-row Shadow pose or a 22-joint
SMPL quaternion pose all work, as long as the poses that follow have the same
number of rows as the one captured. Each row is a quaternion w x y z; it must
be a numpy array.

SYNTAX:
calibrate_pose

EXAMPLE:
calibrate_pose

INPUTS:

pose input:
The pose. Each one is sent on, corrected, once calibrated.

calibration input:
A stored calibration, from 'calibration output'. Replaces the current one.

calibrate input:
Click to capture the pose arriving now.

OUTPUTS:

pose out:
The pose relative to the captured one.

calibration output:
The calibration, sent when it is captured.

RELATED:
target_pose also captures a pose, but scores later poses against it rather
than correcting them.
pose_adjust applies corrections you set by hand."""

demo = swing() + [
    {'key': 'po', 'init': 'pose', 'pos': (30, 400), 'w': 160, 'h': 300},
    {'key': 'btn', 'init': 'button', 'pos': (250, 720), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click: the pose arriving now becomes zero',
     'pos': (360, 720)},
    {'key': 'cp', 'init': 'calibrate_pose', 'pos': (30, 780), 'w': 240, 'h': 140},
    {'key': 'qe', 'init': 'quaternion_to_euler', 'pos': (30, 940), 'w': 260, 'h': 160,
     'props': {'degrees': True}},
    {'key': 've', 'init': 'vector', 'pos': (30, 1120), 'w': 260, 'h': 520},
    {'key': 'c1', 'comment': True, 'text': 'nothing until you click; then the left\nelbow reads as its turn since the click',
     'pos': (360, 1120)},
]
links = SWING_LINKS + [('eq', 'quaternion rotation', 'po', 'left elbow'),
                       ('po', 'pose out', 'cp', 'pose input'),
                       ('btn', '', 'cp', 'calibrate input'),
                       ('cp', 'pose out', 'qe', 'quaternion'),
                       ('qe', 'euler angles', 've', 'in')]
print(build('calibrate_pose', 'calibrate_pose - a captured pose as the zero', body, demo,
            links, demo_width=640, text_width=810, text_height=950))

# --------------------------------------------------------------------- gl_body
body = """These draw a skeleton using the older GL system.

THE NODES:

gl_body         the full skeleton, with joint selection and per-joint data
gl_simple_body  a lighter version
gl_alt_body     an alternative drawing style

WHICH BODY NODE:
These predate the mgl system. mgl_body draws the same thing in the newer scene 
graph, with materials, lighting and the rest of the mgl chain available to it. 
For anything new, that is the one to use.

These remain because they carry the joint-selection and per-joint-data 
machinery that patches have been built around, and because gl_body's 
'limb_sizes' outlet pairs with the limb_size node for adjusting proportions 
interactively.

SEEING A NUMBER ON THE BODY:
'joint data' takes a value per joint and colours the skeleton by it. That is 
the point of these nodes rather than a decoration: a per-joint measurement - 
speed, torque, error, contact - is far easier to read drawn on the body than 
plotted in twenty stacked graphs. Which joint is doing something is immediately 
obvious, and the ones that are not doing anything take up no attention.

'current_joint_name' and 'current_joint_data' report what you clicked on, so 
the drawing can also be the selector.

SYNTAX:
gl_body

EXAMPLE:
gl_body

INPUTS and PARAMETERS:

pose in:
The joint orientations.

gl chain:
The GL chain, as with the other gl nodes.

joint data:
A value per joint, used to colour them.

clear joint data:
Remove the colouring.

capture pose:
Store the current pose.

OUTPUTS: 

gl_chain:
The chain, continuing.

current_joint_name / current_joint_data:
What was clicked.

limb_sizes:
The current proportions - patch this to a limb_size node to adjust them.

RELATED:
mgl_body is the current way to draw a body. 
limb_size adjusts the proportions this node draws with."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'gb', 'init': 'gl_body', 'pos': (30, 400), 'w': 280, 'h': 260},
    {'key': 'c0', 'comment': True, 'text': 'click a joint to select it', 'pos': (30, 675)},
    {'key': 's1', 'init': 'string', 'pos': (350, 400), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'which joint was clicked', 'pos': (350, 450)},
    {'key': 'ls', 'init': 'limb_size', 'pos': (350, 495), 'w': 260, 'h': 200},
    {'key': 'c2', 'comment': True, 'text': 'adjust the proportions it draws with',
     'pos': (350, 710)},
]
links = [('sh', 'body 1 quaternions', 'gb', 'pose in'),
         ('gb', 'current_joint_name', 's1', ''),
         ('gb', 'limb_sizes', 'ls', 'limb_sizes_dict'),
         ('ls', 'out to gl_body', 'gb', 'pose in')]
print(build('gl_body', 'gl_body - drawing a skeleton, the older way', body, demo,
            links, demo_width=650, text_width=790, text_height=680))

# -------------------------------------------------------------- body_to_joints
body = """These convert between the ways a skeleton's pose can be expressed.

THE NODES:

body_to_joints         split a pose into one outlet per joint
shadow_body_to_joints  the same split, for a pose in the raw 37-joint 
                       Shadow layout
local_to_global_body   turn joint-relative rotations into world-space ones
global_to_local_body   turn them back

ONE OUTLET PER JOINT:
body_to_joints and shadow_body_to_joints have the same 20 outlets, one per 
active joint, named for the joint: 'base of skull', 'upper vertebrae', 
'mid vertebrae', 'lower vertebrae', 'spine pelvis', 'pelvis anchor', then the 
hips, knees and ankles, the shoulder blades, shoulders, elbows and wrists. Each 
sends that joint's quaternion, so one joint can go somewhere on its own without 
unpacking the whole array.

Both tell the two pose sizes apart by the number of rows: 37 rows is a full 
Shadow pose, straight from the shadow node; 20 rows is an active pose, from 
active_joints. The pose must be a numpy array or a torch tensor - a list is 
ignored.

Both handle both sizes, and give the same 20 outputs for the same pose.

LOCAL AND GLOBAL ARE THE IMPORTANT PAIR:
A pose is normally stored as LOCAL rotations - each joint's rotation relative 
to its parent. That is the right representation for a skeleton, because it is 
what stays constant when the body moves as a whole: bend an elbow and only the 
elbow's local rotation changes.

A GLOBAL rotation is where that bone actually points in the world, which is 
what you need to ask about direction: is the forearm horizontal, are the two 
hands parallel, which way is the head facing. Those questions are unanswerable 
from local rotations without walking up the chain, which is what 
local_to_global_body does for you - from the pelvis anchor (taken as already 
global) up the spine to the head, out through each shoulder blade to the wrist, 
and down each hip to the ankle.

The rule of thumb: measure in global, edit in local. A correction applied to a 
global rotation moves every child of that joint as well, which is almost never 
what you meant.

WHAT THE GLOBAL CONVERSION EXPECTS:
Both conversion nodes take a 20-joint active pose - 20 rows of four numbers in 
the active_joints order - and send 20 rows back.

Each quaternion is read as w, x, y, z - the real part FIRST, as the shadow node 
and active_joints send it. A joint's world-space rotation is its parent's 
world-space rotation combined with its own local one, working out from the 
pelvis.

global_to_local_body undoes local_to_global_body exactly: a pose sent through 
one and then the other comes back as it started.

SYNTAX:
body_to_joints
shadow_body_to_joints
local_to_global_body
global_to_local_body

EXAMPLE:
local_to_global_body

INPUTS and PARAMETERS:

pose in (body_to_joints, shadow_body_to_joints):
A pose of 37 or 20 quaternions.

gl chain (body_to_joints, shadow_body_to_joints):
Present but unused - anything arriving here does nothing.

pose in (local_to_global_body):
A 20-joint pose of joint-relative rotations.

absolute pose in (global_to_local_body):
A 20-joint pose of world-space rotations.

OUTPUTS: 

base of skull ... right wrist (body_to_joints, shadow_body_to_joints):
One quaternion per outlet, all twenty sent for every pose that arrives.

absolute pose out (local_to_global_body):
The world-space rotations, 20 rows in the same joint order.

relative pose out (global_to_local_body):
The joint-relative rotations, 20 rows.

RELATED:
active_joints reduces a 37-joint Shadow pose to the 20 these nodes use.
limb_size sets the proportions a gl_body skeleton is drawn with.
The quaternion nodes work on the rotations once you have them out."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 120, 'h': 170},
    {'key': 'aj', 'init': 'active_joints', 'pos': (30, 270), 'w': 145, 'h': 42},
    {'key': 'c0', 'comment': True, 'text': 'the 20 active joints', 'pos': (30, 270)},
    {'key': 'lg', 'init': 'local_to_global_body', 'pos': (30, 350), 'w': 135, 'h': 42},
    {'key': 'c1', 'comment': True, 'text': 'world-space: where each bone points\nmeasure in global, edit in local',
     'pos': (30, 350)},
    {'key': 'gl', 'init': 'global_to_local_body', 'pos': (30, 440), 'w': 135, 'h': 42},
    {'key': 'c2', 'comment': True, 'text': 'and back again', 'pos': (30, 440)},
    {'key': 'bj', 'init': 'body_to_joints', 'pos': (30, 530), 'w': 135, 'h': 330},
    {'key': 'c3', 'comment': True, 'text': 'one outlet per joint', 'pos': (30, 530)},
]
links = [('sh', 'body 1 quaternions', 'aj', 'full pose quats in'),
         ('aj', 'active joint quats out', 'lg', 'pose in'),
         ('lg', 'absolute pose out', 'gl', 'absolute pose in'),
         ('aj', 'active joint quats out', 'bj', 'pose in')]
print(build('body_to_joints', 'body_to_joints - local, global, and per joint', body,
            demo, links, demo_width=480, text_width=800, text_height=1480))

# ------------------------------------------------------------------- limb_size
body = """This adjusts the limb proportions a gl_body skeleton is drawn with.

THE NODES:

limb_size  a slider per limb, wired between a gl_body and itself

WHY PROPORTIONS MATTER:
The suit measures orientations, and positions are worked out from those plus the 
limb lengths. If the lengths are wrong the positions are wrong, even though 
every orientation is correct - and the error shows up as feet that do not reach 
the floor or hands that do not meet.

HOW IT IS WIRED - A LOOP:
gl_body's 'limb_sizes' outlet sends a dictionary of its current proportions. 
limb_size takes that in, sets its sliders from it, and remembers it as the 
values 'reset' goes back to. Moving a slider sends a message to gl_body's 
'pose in' inlet, which is where gl_body takes commands:

    limb_size <joint name> <length> <thickness> <thickness>

So the cord goes out of gl_body and back into it.

NOTHING HAPPENS UNTIL THE DICTIONARY ARRIVES:
The sliders only act on limbs that are in the dictionary, so until one has come 
in, moving them does nothing, and neither does 'reset'. 
gl_body sends its dictionary once as it is created, which in a loaded patch is 
before the cord to limb_size exists. Send gl_body the message 'dump_limb_sizes' 
to have it send again; the example does this with the message box.

LENGTH, AND THICKNESS:
Each limb has three sizes: its length and two thicknesses. A slider sets the 
LENGTH. A list of three numbers sent into a slider's inlet sets all three.

'symmetric' keeps left and right the same - moving a left slider moves the 
matching right one, and the other way round - which is usually right and saves 
half the adjusting. The head, neck and spine have no partner.

SYNTAX:
limb_size

EXAMPLE:
limb_size

INPUTS and PARAMETERS:

limb_sizes_dict:
The current proportions, from gl_body's 'limb_sizes' outlet.

symmetric:
Mirror changes between left and right. On by default.

head, neck, spine4, spine3, spine2, spine1:
The head and the spine, from the top down.

left_hip, left_upper_leg, left_lower_leg, left_foot, left_toes:
The left leg - and the same five for the right.

left_shoulder_blade, left_shoulder, left_upper_arm, left_lower_arm, 
left_hand, left_fingers:
The left arm - and the same six for the right.

reset:
Send every limb back to the values in the last dictionary received.

OUTPUTS: 

out to gl_body:
One 'limb_size' message per change, for gl_body's 'pose in' inlet.

RELATED:
gl_body draws the skeleton these proportions apply to.
body_to_joints and local_to_global_body work on the pose itself."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 120, 'h': 170},
    {'key': 'm1', 'init': 'message', 'pos': (30, 270), 'w': 200, 'h': 42,
     'props': {'text in': 'dump_limb_sizes', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'click to fill the sliders\nfrom the body',
     'pos': (30, 270)},
    {'key': 'gb', 'init': 'gl_body', 'pos': (30, 350), 'w': 145, 'h': 160},
    {'key': 'c1', 'comment': True, 'text': 'the body being drawn', 'pos': (30, 350)},
    {'key': 'ls', 'init': 'limb_size', 'pos': (30, 560), 'w': 220, 'h': 470},
    {'key': 'c2', 'comment': True, 'text': 'a slider per limb - its\nchanges go back into gl_body',
     'pos': (30, 560)},
]
links = [('sh', 'body 1 quaternions', 'gb', 'pose in'),
         ('m1', 'message out', 'gb', 'pose in'),
         ('gb', 'limb_sizes', 'ls', 'limb_sizes_dict'),
         ('ls', 'out to gl_body', 'gb', 'pose in')]
print(build('limb_size', 'limb_size - adjusting the proportions of a gl_body', body,
            demo, links, demo_width=480, text_width=800, text_height=1095))
