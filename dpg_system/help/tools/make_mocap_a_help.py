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
shadow_pose    a named pose from a Shadow stream

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
body = """These carry, edit and compare poses.

A pose is a set of joint orientations as quaternions. Everything else in this 
part of the system is a transformation of one.

THE NODES:

pose            hold a pose, and send it
shadow_pose     the same, for the 37-joint Shadow layout
active_pose     an editor for the 20 active joints - see and set each one
active_joints   reduce a full pose to the 20 active joints
target_pose     compare a pose against a captured one and score the match
pose_adjust     apply a per-joint correction to a pose
calibrate_pose  capture a reference pose and correct against it

TWENTY JOINTS, NOT THIRTY-SEVEN:
The Shadow layout has 37 joints; most work uses the 20 that carry the movement. 
active_joints does that reduction, and it is worth doing early - the other 17 
are mostly fixed relationships that add nothing and cost attention.

pose_adjust IS WHERE FRAME CONVENTION BITES:
It applies a per-joint quaternion correction, and by default each non-root joint 
is PRE-multiplied, so the correction is expressed in the PARENT bone's frame. 
The root is post-multiplied, in body-local terms.

The 'child frame' checkbox switches every non-root joint to post-multiply, 
putting the correction in the joint's OWN frame. Which you want depends on where 
the error is: a sensor mounted on the child side of a joint produces an error in 
the child's frame, and correcting it in the parent's will not hold as the joint 
moves.

If a correction looks right in one posture and wrong in another, that is almost 
always this setting.

target_pose SCORES A MATCH:
Capture a pose, then feed the live stream in, and it reports how close the 
current pose is. The 'axis distances out' outlet breaks the score down per axis, 
which tells you WHICH way the difference lies rather than only how much - so a 
performer can be guided towards a target rather than just told they are wrong.

SYNTAX:
pose
active_pose
calibrate_pose

EXAMPLE:
active_joints

INPUTS and PARAMETERS:

pose in:
The pose, as quaternions.

capture in (target_pose):
Store the current pose as the target.

calibration input / calibrate input (calibrate_pose):
The reference, and the command to capture it.

child frame (pose_adjust):
Which frame the correction is applied in - see above.

reset:
Clear the corrections.

zero (active_pose):
Set every joint to identity.

OUTPUTS: 

pose out:
The pose, adjusted or passed through.

score out / axis distances out (target_pose):
How close, overall and per axis.

calibration output:
The captured reference, so it can be stored and reloaded."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'aj', 'init': 'active_joints', 'pos': (30, 400), 'w': 220, 'h': 90},
    {'key': 'ap', 'init': 'active_pose', 'pos': (30, 505), 'w': 260, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'see and set each of the 20 joints',
     'pos': (30, 720)},
    {'key': 'pa', 'init': 'pose_adjust', 'pos': (30, 875), 'w': 260, 'h': 200},
    {'key': 'c1', 'comment': True, 'text': 'child frame changes which frame the\ncorrection is applied in - it matters',
     'pos': (30, 1090)},
    {'key': 'tp', 'init': 'target_pose', 'pos': (371, 505), 'w': 240, 'h': 140},
    {'key': 'f1', 'init': 'float', 'pos': (371, 660), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'how close to the captured target',
     'pos': (340, 710)},
]
links = [('sh', 'body 1 quaternions', 'aj', 'full pose quats in'),
         ('aj', 'active joint quats out', 'ap', 'in'),
         ('ap', 'out', 'pa', 'pose in'),
         ('aj', 'active joint quats out', 'tp', 'pose in'),
         ('tp', 'score out', 'f1', '')]
print(build('pose', 'pose - carrying, editing and comparing poses', body, demo, links,
            demo_width=620, text_width=810, text_height=780))

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
         ('ls', 'out to gl_body', 'gb', 'joint data')]
print(build('gl_body', 'gl_body - drawing a skeleton, the older way', body, demo,
            links, demo_width=650, text_width=790, text_height=680))

# -------------------------------------------------------------- body_to_joints
body = """These convert between the ways a skeleton can be expressed.

THE NODES:

body_to_joints         split a pose into one outlet per joint
shadow_body_to_joints  the same for the 37-joint Shadow layout
local_to_global_body   turn joint-relative rotations into world-space ones
global_to_local_body   turn them back
limb_size              set the limb proportions a body is drawn with

LOCAL AND GLOBAL ARE THE IMPORTANT PAIR:
A pose is normally stored as LOCAL rotations - each joint's rotation relative 
to its parent. That is the right representation for a skeleton, because it is 
what stays constant when the body moves as a whole: bend an elbow and only the 
elbow's local rotation changes.

A GLOBAL rotation is where that bone actually points in the world, which is 
what you need to ask about direction: is the forearm horizontal, are the two 
hands parallel, which way is the head facing. Those questions are unanswerable 
from local rotations without walking up the chain, which is what 
local_to_global_body does for you.

The rule of thumb: measure in global, edit in local. A correction applied to a 
global rotation moves every child of that joint as well, which is almost never 
what you meant.

limb_size AND WHY PROPORTIONS MATTER:
The suit measures orientations, and positions are worked out from those plus the 
limb lengths. If the lengths are wrong the positions are wrong, even though 
every orientation is correct - and the error shows up as feet that do not reach 
the floor or hands that do not meet.

'symmetric' keeps left and right the same, which is usually right and saves 
half the adjusting.

SYNTAX:
body_to_joints
local_to_global_body

EXAMPLE:
local_to_global_body

INPUTS and PARAMETERS:

pose in:
The pose.

limb_sizes_dict (limb_size):
The current proportions, from gl_body.

symmetric (limb_size):
Mirror left and right.

reset:
Back to the defaults.

OUTPUTS: 

absolute pose out:
World-space rotations.

one outlet per joint (body_to_joints):
The joints individually, for patching to different places.

out to gl_body (limb_size):
The proportions, back to the body being drawn.

RELATED:
The quaternion nodes work on the rotations once you have them out."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'aj', 'init': 'active_joints', 'pos': (30, 400), 'w': 220, 'h': 90},
    {'key': 'lg', 'init': 'local_to_global_body', 'pos': (30, 505), 'w': 260, 'h': 90},
    {'key': 'c0', 'comment': True, 'text': 'world-space: where each bone points\nmeasure in global, edit in local',
     'pos': (30, 610)},
    {'key': 'gl', 'init': 'global_to_local_body', 'pos': (30, 685), 'w': 260, 'h': 90},
    {'key': 'c2', 'comment': True, 'text': 'and back again', 'pos': (30, 790)},
    {'key': 'bj', 'init': 'body_to_joints', 'pos': (340, 505), 'w': 240, 'h': 200},
    {'key': 'c3', 'comment': True, 'text': 'one outlet per joint', 'pos': (340, 720)},
]
links = [('sh', 'body 1 quaternions', 'aj', 'full pose quats in'),
         ('aj', 'active joint quats out', 'lg', 'pose in'),
         ('lg', 'absolute pose out', 'gl', 'absolute pose in'),
         ('aj', 'active joint quats out', 'bj', 'pose in')]
print(build('body_to_joints', 'body_to_joints - local, global, and per joint', body,
            demo, links, demo_width=620, text_width=800, text_height=720))
