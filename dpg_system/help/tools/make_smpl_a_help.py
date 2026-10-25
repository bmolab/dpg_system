"""SMPL body and data, conversions, corrections."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# -------------------------------------------------------------------- smpl_pose
body = """smpl_pose shows a 22-joint SMPL pose one row per joint, and lets you edit it.

WHAT IT SHOWS:
Twenty-two rows in SMPL's joint order: pelvis, left_hip, right_hip, spine1,
left_knee, right_knee, spine2, left_ankle, right_ankle, spine3, left_foot,
right_foot, neck, left_collar, right_collar, head, left_shoulder,
right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist. These are
SMPL's body joints; the hand joints are not included.

Each row is one joint's rotation as a quaternion, w x y z with the real part
first. Every joint starts at the identity, 1 0 0 0, which is no rotation.

QUATERNIONS, NOT AXIS-ANGLE:
SMPL files store a pose as axis-angle - three numbers per joint - and that is
what smpl_take sends. Put smpl_to_quats in between. A three-column pose sent
straight in is shown as three-number rows, and the joint names disappear.

EDITING:
Change any number in a row and the whole pose goes out of 'out'. The numbers
go out exactly as typed - nothing normalises them - so keep each row a unit
quaternion, its four squares adding up to one. 0.707 0 0.707 0 is a quarter
turn about y; 1 1 0 0 is not a rotation at all.

A POSE ARRIVING:
A pose sent into 'in' replaces the rows and goes straight out, so smpl_pose
also works as a readout. While a stream is arriving, it overwrites your edits
on the next frame. A bang into 'in' sends the rows as they are.

The rows are saved with the patch.

SYNTAX:
smpl_pose

EXAMPLE:
smpl_pose

INPUTS:

in:
A 22-joint quaternion pose, or a bang to send the current rows.

pelvis ... right_wrist:
The rows themselves.

reset:
Set every joint back to the identity, and send the pose.

output type:
numpy (the default), torch or list.

OPTIONS:
save and load write the rows to a .npy file and read them back.

OUTPUT:

out:
The pose, 22 rows of four.

RELATED:
gl_body draws a 22-row SMPL quaternion pose directly.
smpl_quats_to_joints splits one into an outlet per joint.
active_pose is the same editor for the 20 Shadow active joints."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click to send the pose', 'pos': (330, 62)},
    {'key': 'sp', 'init': 'smpl_pose', 'pos': (30, 130), 'w': 300, 'h': 570,
     'props': {'[18]': [0.7071, 0.0, 0.7071, 0.0]}},
    {'key': 'c1', 'comment': True, 'text': 'left_elbow is set to a quarter\nturn about y - edit any row\nand the pose goes out',
     'pos': (330, 130)},
    {'key': 'qe', 'init': 'quaternion_to_euler', 'pos': (30, 720), 'w': 260, 'h': 160,
     'props': {'degrees': True}},
    {'key': 've', 'init': 'vector', 'pos': (30, 900), 'w': 260, 'h': 570},
    {'key': 'c2', 'comment': True, 'text': 'the same pose as x y z angles\nin degrees, one row per joint',
     'pos': (330, 900)},
]
links = [('btn', '', 'sp', 'in'),
         ('sp', 'out', 'qe', 'quaternion'),
         ('qe', 'euler angles', 've', 'in')]
print(build('smpl_pose', 'smpl_pose - see and edit an SMPL pose', body,
            demo, links, demo_width=640, text_width=810, text_height=1010))

# -------------------------------------------------------------------- smpl_body
body = """smpl_body reads an SMPL body model file and sends its skeleton: the offset from
each joint to the next, at rest.

WHAT IT SENDS:
One message per bone on 'skeleton_data': the bone's name followed by x y z,
the offset from the joint at its start to the joint at its end, in metres,
in the model's rest pose. The bones are left_hip, right_hip, lower_back,
left_thigh, right_thigh, mid_back, left_lower_leg, right_lower_leg,
upper_back, left_foot, right_foot, lower_neck, left_shoulder_blade,
right_shoulder_blade, upper_neck, left_shoulder, right_shoulder,
left_upper_arm, right_upper_arm, left_forearm, right_forearm, left_hand and
right_hand. An SMPL-H model's left_hand and right_hand run from the wrist to
the first knuckle of the index finger; its finger bones are not sent.

gl_body understands these messages. Sent into its 'pose in' inlet, each one
sets the matching bone of the gl_body skeleton, so the skeleton takes on the
SMPL body's proportions.

WHICH MODEL FILE:
An SMPL or SMPL-H body model: the 24-joint SMPL .pkl, or an SMPL-H .pkl such
as the two in dpg_system/smplh. 'load' offers .pkl files; the 'path' option
also takes an SMPL-H .npz. A file that cannot be read prints a message saying
so, and nothing is sent.

BETAS RESHAPE IT:
'betas' takes a list of shape numbers. Each arrival recomputes the rest
joints for that body shape and sends the whole skeleton again. A model has a
fixed number of betas (SMPL 10, SMPL-H 16): a shorter list leaves the rest at
zero, and extra numbers are ignored. Loading a model starts from its own
default body, all betas zero.

WHEN IT SENDS:
When a model loads, and again whenever betas arrive. The 'path' is saved with
the patch, so the model reloads when the patch opens, and the skeleton goes
out once the patch's cords are connected.

SYNTAX:
smpl_body

EXAMPLE:
smpl_body

INPUTS and PROPERTIES:

betas:
Shape numbers. Arriving reshapes the skeleton and sends it.

load:
Choose a model file.

path (option):
The model file. Setting it loads the file.

OUTPUT:

skeleton_data:
One message per bone: name, x, y, z.

RELATED:
gl_body draws a skeleton with these proportions.
smpl_beta_editor edits betas, and gives bone lengths for any set of them."""

demo = [
    {'key': 'be', 'init': 'smpl_beta_editor', 'pos': (30, 62), 'w': 300, 'h': 420},
    {'key': 'c0', 'comment': True, 'text': 'change a beta and smpl_body\nsends the reshaped skeleton',
     'pos': (360, 62)},
    {'key': 'sb', 'init': 'smpl_body', 'pos': (30, 500), 'w': 240, 'h': 120},
    {'key': 'c1', 'comment': True, 'text': 'click load and choose an SMPL .pkl\n(SMPL or SMPL-H - dpg_system/smplh has two)',
     'pos': (360, 500)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 640), 'w': 300, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'one message per bone: name x y z\n(this shows the last one)',
     'pos': (360, 640)},
]
links = [('be', 'betas', 'sb', 'betas'),
         ('sb', 'skeleton_data', 'l1', '')]
print(build('smpl_body', 'smpl_body - the skeleton of an SMPL model', body,
            demo, links, demo_width=680, text_width=810, text_height=1010))

# -------------------------------------------------------------------- smpl_take
body = """smpl_take plays back a recorded SMPL motion file, frame by frame.

THE FILE:
An .npz file holding two arrays: 'poses', one row per frame of axis-angle
joint rotations, and 'trans', the root translation for each frame. AMASS
'_poses.npz' files are in this form. 'trans' is optional: a file with only
'poses' plays, and 'root_position' sends nothing. A file without 'poses' does
not load.

Only the first 22 joints are kept - SMPL's body joints. The hand joints in an
SMPL-H file are dropped.

WHAT IT SENDS:
'joint_data' is the pose: 22 rows of three numbers in axis-angle form. The
direction of the three numbers is the axis, and their length is the angle in
radians. 'root_position' is x y z.

Both are exactly as the file has them. Nothing is converted: an AMASS file is
Z-up, and so is what comes out.

PLAYING:
Click 'load' to choose a file, or type its path into the 'path' option; the
path is saved with the patch. 'on/off' plays, once a file is loaded - ticking
it first starts playback as soon as a file loads. At the end it loops back to
frame 0.

'speed' is how many file frames to move on per display frame, about 60 a
second. So speed 1 plays one file frame per display frame whatever rate the
file was recorded at: a file recorded at 120 frames a second plays at half
its real pace, and needs speed 2. Fractions slow it down. Negative speeds play
backwards, looping from frame 0 round to the last frame.

'frame' shows where playback is. Setting it - by hand or by sending an
integer - jumps there and sends that frame, so you can step through a take
while it is stopped.

SYNTAX:
smpl_take

EXAMPLE:
smpl_take

INPUTS and PROPERTIES:

on/off:
Play or stop.

speed:
File frames per display frame.

frame:
The current frame. Setting it sends that frame.

load:
Choose a file.

path (option):
The file. Setting it loads the file.

OUTPUTS:

joint_data:
The pose, 22 rows of axis-angle.

root_position:
The root translation, x y z (nothing if the file has no 'trans').

RELATED:
smpl_to_quats turns the pose into quaternions, as in the example.
smpl_torque, mgl_smpl_mesh and floor_zero take axis-angle poses and
translations like these."""

demo = [
    {'key': 'tk', 'init': 'smpl_take', 'pos': (30, 62), 'w': 260, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'click load and choose an AMASS\n_poses.npz, then tick on/off',
     'pos': (360, 62)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 280), 'w': 300, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'root position, x y z (Z is up)', 'pos': (360, 280)},
    {'key': 'sq', 'init': 'smpl_to_quats', 'pos': (30, 340), 'w': 240, 'h': 90},
    {'key': 'c2', 'comment': True, 'text': 'axis-angle in, quaternions out', 'pos': (360, 340)},
    {'key': 'sp', 'init': 'smpl_pose', 'pos': (30, 450), 'w': 300, 'h': 570},
    {'key': 'c3', 'comment': True, 'text': 'the playing pose, joint by joint', 'pos': (360, 450)},
]
links = [('tk', 'root_position', 'l1', ''),
         ('tk', 'joint_data', 'sq', 'smpl pose'),
         ('sq', 'smpl pose as quaternions', 'sp', 'in')]
print(build('smpl_take', 'smpl_take - play a recorded SMPL take', body,
            demo, links, demo_width=680, text_width=810, text_height=1110))

# ------------------------------------------------------------- smpl_beta_editor
body = """smpl_beta_editor sets the shape of an SMPL body by hand, and sends it in the
forms the other SMPL nodes want.

WHAT BETAS ARE:
SMPL describes a body's shape with a handful of numbers called betas. They
are not measurements. They are the main ways in which the bodies of a large
scanned population differ from the average, so each one changes several
proportions at once. 0 is the average body; the sliders run from -5 to 5.

The names on the first few sliders are rough guides to what each one does:
size, weight, arm ratio, leg ratio, and so on. Check by eye. Note that a
POSITIVE beta_0 makes this body SMALLER: at 2, the thigh is about 8 percent
shorter than at 0, and at -2 about 8 percent longer.

GENDER:
male, female or neutral - which body model to use. Only the male and female
SMPL-H models are installed here, and for the limb lengths this node works
out, neutral uses the male model.

WHAT IT SENDS:
All three outlets send whenever anything changes, and again when the patch
opens.

config is a dictionary with 'betas' (the ten numbers) and 'gender'. It is
what smpl_torque, floor_zero and the other SMPL nodes take on their config
inlets. It does not carry the mass.

betas is the ten numbers on their own.

limb_lengths is a dictionary: 'lengths' (bone lengths in metres - upper_leg,
lower_leg, upper_arm, lower_arm, foot, hand, collar, neck, head, the spine
segments, pelvis_width, shoulder_width), 'masses' (per segment, from
total_mass and the shape), 'offsets', and 'total_mass_scaled' (total_mass
adjusted for the first two betas). mgl_body and tracker_root_inference take
it on their 'limb_lengths' inlets.

total_mass, in kilograms, is used only for limb_lengths.

CHANGING WEIGHT WITHOUT CHANGING BONES:
'solve β1' sets the weight beta (beta_1) to 'solve_target β1' while keeping
the major bone lengths where they were: it re-fits the other nine betas to
hold them. The first solve remembers the betas at that moment as its
reference, and later solves start from the same reference, so repeated
solves do not drift. 'clear solve ref' forgets it - and so do reset, load,
and new betas arriving. The 'solve_reg' option sets how strongly the other
betas are held near the reference while fitting.

SAVING:
save writes the betas, gender and total_mass to a .npy file, and load reads
one back. With the 'file_path' option set they use that file without asking;
otherwise a file dialog opens.

SYNTAX:
smpl_beta_editor

EXAMPLE:
smpl_beta_editor

INPUTS and PROPERTIES:

beta_0 size ... beta_9 breadth:
The ten betas.

gender / total_mass:
As above.

betas_in:
Up to ten numbers, or a dictionary with 'betas' and optionally 'gender' and
'total_mass' - such as a config. Sets the sliders and sends.

reset:
Set every beta to 0. Gender and mass are left alone.

save / load:
Store a body, and bring it back.

solve_target β1 / solve β1 / clear solve ref:
Change weight while holding the bones, as above.

OUTPUTS:

config:
betas and gender, for the other SMPL nodes.

betas:
The ten numbers.

limb_lengths:
Bone lengths, segment masses and offsets for this body.

RELATED:
smpl_torque, floor_zero and smpl_ragdoll take the config.
mgl_body takes the limb lengths."""

demo = [
    {'key': 'be', 'init': 'smpl_beta_editor', 'pos': (30, 62), 'w': 300, 'h': 420},
    {'key': 'c0', 'comment': True, 'text': 'move a slider: everything is sent again',
     'pos': (360, 62)},
    {'key': 've', 'init': 'vector', 'pos': (30, 500), 'w': 200, 'h': 280},
    {'key': 'c1', 'comment': True, 'text': 'the ten betas', 'pos': (360, 500)},
    {'key': 'dx', 'init': 'dict_extract lengths', 'pos': (30, 800), 'w': 200, 'h': 60},
    {'key': 'dl', 'init': 'dict_extract upper_leg upper_arm', 'pos': (30, 880), 'w': 260, 'h': 80},
    {'key': 'f0', 'init': 'float', 'pos': (30, 980), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'f1', 'init': 'float', 'pos': (180, 980), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': 'thigh and upper arm length, metres\n- watch them shrink as beta_0 rises',
     'pos': (360, 980)},
]
links = [('be', 'betas', 've', 'in'),
         ('be', 'limb_lengths', 'dx', 'dict in'),
         ('dx', '', 'dl', 'dict in', 0),
         ('dl', '', 'f0', '', 0),
         ('dl', '', 'f1', '', 1)]
print(build('smpl_beta_editor', 'smpl_beta_editor - the shape of the body', body,
            demo, links, demo_width=680, text_width=810, text_height=1460))

# --------------------------------------------------------------- shadow_to_smpl
body = """These convert poses into and out of the SMPL convention.

Every system has its own joint order, its own quaternion order, and its own 
idea of which way is up. Most of the work of getting motion capture into SMPL 
is those three disagreements, and these nodes are them.

THE NODES:

shadow_to_smpl      37-joint Shadow to 22-joint SMPL, plus the root translation
active_to_smpl      the 20 active joints to SMPL
smpl_to_active      back the other way
smpl_to_quats       SMPL axis-angle to quaternions
quats_flip_y_z      swap the y and z axes of a pose
smpl_trans_to_y_up  SMPL's Z-up translation to a normal Y-up one

THE THREE DISAGREEMENTS:

JOINT ORDER. Shadow has 37 joints, the active set has 20, SMPL has 22, and 
they are not subsets of each other in any simple way. shadow_to_smpl and its 
relatives hold the mapping so you do not.

ROTATION FORMAT. SMPL stores poses as AXIS-ANGLE - three numbers per joint - 
while the suit produces quaternions. smpl_to_quats converts. Both describe the 
same rotation; axis-angle is more compact and quaternions are better behaved 
for arithmetic, which is why both exist.

UP AXIS. SMPL is Z-up. Most of the rest of this system, and most graphics, is 
Y-up. smpl_trans_to_y_up converts a translation by [x, y, z] -> [x, z, -y], 
and quats_flip_y_z does the equivalent for rotations.

DO NOT CONVERT WHAT ALREADY HANDLES IT:
smpl_torque and smpl_ragdoll take an 'axis_permutation' setting and do the 
conversion themselves. Converting before them AND leaving their setting at the 
default applies it twice, which produces a body that is subtly and confusingly 
wrong rather than obviously upside down. Feed those two the raw SMPL 
convention and set their permutation; use these nodes for everything else.

SYNTAX:
shadow_to_smpl
smpl_to_quats

EXAMPLE:
shadow_to_smpl

INPUTS and PARAMETERS:

pose / positions:
The source pose, and the positions the root translation is taken from - 
shadow_to_smpl reads the root from positions[4].

config:
The body config, where the conversion needs to know the body's size.

output_format:
Quaternions or axis-angle.

OUTPUTS: 

pose / trans:
The converted pose and root translation.

RELATED:
The quaternion nodes convert rotation representations in general; these are the 
SMPL-specific joint-order and axis conventions on top of that."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'be', 'init': 'smpl_beta_editor', 'pos': (350, 62), 'w': 300, 'h': 340},
    {'key': 'ss', 'init': 'shadow_to_smpl', 'pos': (30, 400), 'w': 260, 'h': 140},
    {'key': 'c0', 'comment': True, 'text': '37 joints in, 22 out, plus the root',
     'pos': (30, 555)},
    {'key': 'sq', 'init': 'smpl_to_quats', 'pos': (30, 595), 'w': 240, 'h': 90},
    {'key': 'c1', 'comment': True, 'text': 'SMPL stores axis-angle; this gives quats',
     'pos': (30, 700)},
    {'key': 'ty', 'init': 'smpl_trans_to_y_up', 'pos': (350, 595), 'w': 260, 'h': 90},
    {'key': 'c2', 'comment': True, 'text': 'SMPL is Z-up; most else is Y-up\nbut do NOT convert before smpl_torque -\nit does this itself',
     'pos': (350, 700)},
]
links = [('sh', 'body 1 quaternions', 'ss', 'pose'),
         ('sh', 'body 1 positions', 'ss', 'positions'),
         ('be', 'config', 'ss', 'config'),
         ('ss', 'pose', 'sq', 'smpl pose'),
         ('ss', 'trans', 'ty', 'z-up trans')]
print(build('shadow_to_smpl', 'shadow_to_smpl - joint order, format, up axis', body,
            demo, links, demo_width=690, text_width=800, text_height=740))

# ------------------------------------------------------------- smpl_pose_adjust
body = """smpl_pose_adjust adds a fixed rotation to chosen joints of an SMPL pose - for a
pose that is right in shape but wrong in detail, such as a sensor that sits a
little off on one limb.

WHAT IT TAKES:
An SMPL pose in AXIS-ANGLE form: three numbers per joint, at least 22 joints,
flat or one row per joint. Joints past the 22nd (the hands) pass through
untouched. A pose in quaternions produces no output at all - from
shadow_to_smpl, set 'output_format' to axis_angle.

THE CORRECTIONS:
There is one row of three numbers per body joint, named pelvis, left_hip, and
so on to right_wrist. Each row is a rotation in axis-angle form: the direction
of the three numbers is the axis, and their length is the angle in radians.
All zeros means no correction. A corrected joint carries everything below it
with it, as a real joint would.

WHICH FRAME A CORRECTION IS IN:
The frame the rotation is applied in decides whether the correction holds as
the body moves.

The pelvis correction is always in the body's own frame. Every other joint's
correction is, by default, in its PARENT bone's frame. The 'child frame'
checkbox switches them to the joint's OWN frame.

A sensor mounted on the child side of a joint produces an error in the child's
frame, and correcting it in the parent's will drift as the joint bends. If a
correction looks right in one posture and wrong in another, this is the
setting.

SYNTAX:
smpl_pose_adjust

EXAMPLE:
smpl_pose_adjust

INPUTS:

pose in:
The axis-angle pose.

child frame:
Off: corrections in the parent bone's frame. On: in the joint's own frame.

reset:
Set every correction back to zero.

pelvis ... right_wrist:
The correction for each joint, as above. The values are saved with the patch
and with presets.

OUTPUT:

pose out:
The corrected pose, one row per joint. Joints without a
correction come out exactly as they went in.

RELATED:
smpl_mag_yaw_correct corrects magnetometer heading errors joint by joint.
floor_zero puts the feet on the floor."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'ss', 'init': 'shadow_to_smpl', 'pos': (30, 400), 'w': 260, 'h': 140,
     'props': {'output_format': 'axis_angle'}},
    {'key': 'c0', 'comment': True, 'text': 'output_format axis_angle:\nquaternions are ignored',
     'pos': (330, 400)},
    {'key': 'pa', 'init': 'smpl_pose_adjust', 'pos': (30, 560), 'w': 280, 'h': 200},
    {'key': 'c1', 'comment': True, 'text': 'one rotation per joint, in radians\nchild frame: which frame it applies in',
     'pos': (330, 560)},
]
links = [('sh', 'body 1 quaternions', 'ss', 'pose'),
         ('sh', 'body 1 positions', 'ss', 'positions'),
         ('ss', 'pose', 'pa', 'pose in')]
print(build('smpl_pose_adjust', 'smpl_pose_adjust - per-joint corrections', body,
            demo, links, demo_width=650, text_width=810, text_height=780))

# --------------------------------------------------------- smpl_mag_yaw_correct
body = """smpl_mag_yaw_correct turns joints of an SMPL pose about the vertical, to undo
heading errors that come from the suit's magnetometers.

WHAT IT TAKES:
An SMPL pose in AXIS-ANGLE form, at least 22 joints. Joints past the 22nd pass
through untouched. A pose in quaternions produces no output at all - from
shadow_to_smpl, set 'output_format' to axis_angle.

Set 'up axis' to match the pose. It defaults to Y, but shadow_to_smpl sends
Z-up by default, and a heading correction about the wrong axis is a tilt.

EACH JOINT IS CORRECTED WHERE IT IS IN THE ROOM:
The node works out every joint's orientation in the room, turns each one, and
then works the pose back out from the corrected orientations. So a correction
on the upper arm turns the upper arm alone: the forearm stays where it was in
the room. That matches the problem, because each sensor has its own heading
error.

TWO CORRECTIONS PER JOINT:
Each joint has two values, in degrees.

The first, named after the joint (left_shoulder), turns the joint about the
room's vertical. This is the one for heading drift: the whole sensor reading
rotated in the room.

The second, named with _cal (left_shoulder_cal), turns it about the vertical
axis in the joint's own frame - so its effect follows the joint as it moves.

'sync local/global' keeps the two equal. Equal values leave a joint that is in
the rest orientation - or turned from it only about the vertical - exactly as
it is; the correction shows as the joint tilts away. That is right when a
sensor's heading error is the same whichever way it faces, and not otherwise,
so it is a convenience rather than a setting to leave on regardless.

After the conversion from the suit to SMPL the _cal value is no longer the
sensor's own calibration offset; treat it as a heading adjustment that travels
with the joint.

SYNTAX:
smpl_mag_yaw_correct

EXAMPLE:
smpl_mag_yaw_correct

INPUTS and PROPERTIES:

pose in:
The axis-angle pose.

up axis:
Y or Z: which way is up in the pose.

symmetric:
Editing a left joint copies the same value to the right one, and the other
way round.

sync local/global:
Editing either value of a joint copies it to the other.

Both act when you edit a value; ticking them does not change values already
set.

reset:
Set every value back to zero.

pelvis ... right_wrist, pelvis_cal ... right_wrist_cal:
The two corrections for each joint, in degrees.

OUTPUT:

pose out:
The corrected pose, one row per joint.

RELATED:
mag_yaw_correct does the same correction on the suit side, per sensor, before
the pose becomes SMPL at all. Correcting there is better when you can, because
it fixes the measurement rather than its consequences.
smpl_pose_adjust adds a fixed rotation to any joint, in the joint's frame or
its parent's."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'ss', 'init': 'shadow_to_smpl', 'pos': (30, 400), 'w': 260, 'h': 140,
     'props': {'output_format': 'axis_angle'}},
    {'key': 'c0', 'comment': True, 'text': 'axis_angle out, Z-up', 'pos': (330, 400)},
    {'key': 'my', 'init': 'smpl_mag_yaw_correct', 'pos': (30, 560), 'w': 280, 'h': 220,
     'props': {'up axis': 'Z'}},
    {'key': 'c1', 'comment': True, 'text': 'up axis Z, to match\nfirst value: turn in the room\n_cal: turn in the joint\'s own frame',
     'pos': (330, 560)},
]
links = [('sh', 'body 1 quaternions', 'ss', 'pose'),
         ('sh', 'body 1 positions', 'ss', 'positions'),
         ('ss', 'pose', 'my', 'pose in')]
print(build('smpl_mag_yaw_correct', 'smpl_mag_yaw_correct - heading, joint by joint',
            body, demo, links, demo_width=650, text_width=810, text_height=780))

# ------------------------------------------------------------------ floor_zero
body = """floor_zero shifts the root translation up or down by a constant, so that the
feet are on the floor.

WHAT IT MEASURES:
Click 'calibrate'. On the next translation that arrives, it builds the SMPL
skeleton - from the betas and gender in 'config', or a neutral body if there
is none - puts it in the current pose, and finds the lower of the two foot
joints. It stores the offset that brings that foot, less 'sole_offset', down
or up to 'floor_height'.

Click it while the performer is standing, with a foot on the floor.

THE OFFSET IS CONSTANT:
After calibration the same offset is added to every translation that comes
through. Real vertical motion - jumps, crouches - is preserved exactly. A node
that kept re-zeroing would flatten precisely the movement you care about.
Click 'calibrate' again whenever you want a new measurement; until the first
one, translations pass through unchanged.

SYNTAX:
floor_zero

EXAMPLE:
floor_zero

INPUTS:

trans:
The root translation, x y z. Each one is sent on, shifted.

pose:
The current pose: 22 joints as axis-angle (three numbers each) or as
quaternions (four, w first). Only used when calibrating. If no pose has
arrived, clicking 'calibrate' prints a message, and the calibration waits for
the first translation after a pose arrives.

config:
A dictionary with 'betas' and 'gender', as smpl_beta_editor sends. Optional.

PROPERTIES and OPTIONS:

calibrate:
Measure on the next translation.

up_axis:
Which axis is up: Z (the default, as shadow_to_smpl sends) or Y.

floor_height:
Where the floor is. Usually 0.

sole_offset:
How far the sole is below the foot joint. The foot joint is inside the foot,
so without this the joint, not the sole, is put on the floor.

OUTPUTS:

trans:
The shifted translation.

offset:
The shift, as a single number, on every frame - so the same shift can be
added to a translation elsewhere, for example a Y-up copy being saved.

measured sole height:
Sent once per calibration: where the sole was before the shift. Worth checking,
because an implausible value means the pose or the betas are wrong rather than
the floor."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'be', 'init': 'smpl_beta_editor', 'pos': (350, 62), 'w': 300, 'h': 340},
    {'key': 'ss', 'init': 'shadow_to_smpl', 'pos': (30, 420), 'w': 260, 'h': 140},
    {'key': 'fz', 'init': 'floor_zero', 'pos': (30, 590), 'w': 260, 'h': 160},
    {'key': 'c0', 'comment': True, 'text': 'stand still, click calibrate\na CONSTANT offset, so jumps survive',
     'pos': (330, 590)},
    {'key': 'f0', 'init': 'float', 'pos': (30, 780), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'f1', 'init': 'float', 'pos': (200, 780), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'offset, and the sole height it measured',
     'pos': (30, 835)},
]
links = [('sh', 'body 1 quaternions', 'ss', 'pose'),
         ('sh', 'body 1 positions', 'ss', 'positions'),
         ('ss', 'pose', 'fz', 'pose'),
         ('ss', 'trans', 'fz', 'trans'),
         ('be', 'config', 'fz', 'config'),
         ('fz', 'offset', 'f0', ''),
         ('fz', 'measured sole height', 'f1', '')]
print(build('floor_zero', 'floor_zero - put the feet on the floor', body,
            demo, links, demo_width=690, text_width=810, text_height=780))
