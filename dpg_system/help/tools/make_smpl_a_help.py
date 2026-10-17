"""SMPL body and data, conversions, corrections."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# -------------------------------------------------------------------- smpl_pose
body = """These hold an SMPL body: its shape, its pose, and recorded takes of both.

WHAT SMPL IS:
A parametric model of a human body. Its shape is a handful of numbers - the 
BETAS - and its pose is 22 joint rotations. Between them they describe a 
specific person in a specific posture, and from that the model produces the 
actual surface: where every vertex of the body is.

That is the value of it. A skeleton tells you where the joints are; SMPL tells 
you where the BODY is, which is what you need for contact, for volume, for 
anything about the physical thing rather than the stick figure.

THE NODES:

smpl_pose        hold and edit a 22-joint SMPL pose
smpl_body        the skeleton for a given set of betas
smpl_take        play back a recorded SMPL take
smpl_beta_editor tune the betas by hand and produce a config

BETAS ARE THE PERSON:
The first beta is roughly overall size, the second roughly the tall-versus-wide 
axis, and the rest progressively subtler. They are not anatomical measurements - 
they are the axes a model found in a population, so no single one means 
"leg length". You tune them by eye against the person you are working with.

smpl_beta_editor produces a CONFIG dict, and that config is what the other 
nodes want. Almost everything downstream - torque, ragdoll, floor calibration - 
takes it, because none of them can work out where the body is without knowing 
its size and mass.

Its 'limb_lengths' outlet feeds the motion capture side, so the skeleton the 
suit assumes and the body SMPL builds can be made to agree.

SYNTAX:
smpl_pose
smpl_beta_editor
smpl_take

EXAMPLE:
smpl_beta_editor

INPUTS and PARAMETERS:

in / betas_in:
The pose, or the shape numbers.

gender / total_mass (smpl_beta_editor):
Which body model, and the mass the physics nodes should assume.

solve β1 / solve_target β1 / solve_reg:
Fit the first beta to a target rather than setting it by hand.

save / load / file_path:
Store a set of betas.

on/off / speed / frame (smpl_take):
Transport for a recorded take.

OUTPUTS: 

out / pose:
The pose.

config:
The dict every other SMPL node wants - betas, gender, mass.

betas / limb_lengths:
The shape numbers, and the resulting bone lengths.

joint_data / root_position (smpl_take):
The playing frames.

skeleton_data (smpl_body):
The skeleton for those betas."""

demo = [
    {'key': 'be', 'init': 'smpl_beta_editor', 'pos': (30, 62), 'w': 300, 'h': 340},
    {'key': 'c0', 'comment': True, 'text': 'tune by eye against the person\nconfig is what everything else wants',
     'pos': (30, 415)},
    {'key': 'sb', 'init': 'smpl_body', 'pos': (30, 490), 'w': 240, 'h': 120},
    {'key': 'tk', 'init': 'smpl_take', 'pos': (380, 62), 'w': 260, 'h': 200},
    {'key': 'sp', 'init': 'smpl_pose', 'pos': (380, 285), 'w': 260, 'h': 200},
    {'key': 'c2', 'comment': True, 'text': 'a recorded take driving a pose',
     'pos': (380, 500)},
]
links = [('be', 'betas', 'sb', 'betas'),
         ('tk', 'joint_data', 'sp', 'in')]
print(build('smpl_pose', 'smpl_pose - the body, its shape and its posture', body,
            demo, links, demo_width=680, text_width=800, text_height=700))

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
arrived, calibrating does nothing.

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
