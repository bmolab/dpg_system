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
body = """These correct an SMPL pose that is right in shape but wrong in detail.

THE NODES:

smpl_pose_adjust      per-joint corrections
smpl_mag_yaw_correct  magnetometer yaw errors, per joint
floor_zero            put the feet on the floor

WHICH FRAME A CORRECTION IS IN:
smpl_pose_adjust applies a per-joint axis-angle adjustment, and the frame it 
applies in decides whether the correction holds as the body moves.

By default the pelvis is post-multiplied - body-local - and every other joint 
is pre-multiplied, so the correction sits in the PARENT bone's frame. 
The 'child frame' checkbox switches the others to post-multiply, putting the 
correction in the joint's OWN frame.

A sensor mounted on the child side of a joint produces an error in the child's 
frame, and correcting it in the parent's will drift as the joint bends. 
If a correction looks right in one posture and wrong in another, this is the 
setting.

smpl_mag_yaw_correct HAS THE SAME TWO CORRECTIONS AS ITS SUIT COUNTERPART:
'Global yaw' is pre-multiplied in the world frame and addresses ongoing 
magnetometer drift. 'Local yaw' is post-multiplied in the joint's body frame 
and addresses a residual calibration offset baked into the T-pose.

'sync' keeps the two equal, which is correct when a hard-iron bias produces the 
same error in both - and wrong when it does not, so it is a convenience rather 
than a default to leave on.

floor_zero IS A ONE-TIME CALIBRATION:
On 'calibrate' it runs a single forward-kinematics pass on the current pose, 
finds the lowest foot relative to the root, and stores a CONSTANT vertical 
offset that puts that foot on the floor.

Constant is the point. The offset is applied to every subsequent translation, 
so real vertical motion - jumps, crouches - is preserved exactly. A node that 
kept re-zeroing would flatten precisely the movement you care about.

SYNTAX:
smpl_pose_adjust
floor_zero

EXAMPLE:
floor_zero

INPUTS and PARAMETERS:

pose in / pose:
The pose.

trans / config (floor_zero):
The root translation, and the body config it needs to run the kinematics.

child frame (smpl_pose_adjust):
Which frame the correction is applied in.

symmetric / sync local/global (smpl_mag_yaw_correct):
Mirror left to right, and tie the two corrections together.

reset:
Clear the corrections.

OUTPUTS: 

pose out / trans:
The corrected pose or translation.

offset / measured sole height (floor_zero):
What was applied, and what it measured - worth checking, because an 
implausible sole height means the pose or the betas are wrong rather than the 
floor.

RELATED:
mag_yaw_correct does the same magnetic correction on the suit side, per sensor, 
before the pose becomes SMPL at all. Correcting there is better when you can, 
because it fixes the measurement rather than its consequences."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'be', 'init': 'smpl_beta_editor', 'pos': (350, 62), 'w': 300, 'h': 340},
    {'key': 'ss', 'init': 'shadow_to_smpl', 'pos': (30, 400), 'w': 260, 'h': 140},
    {'key': 'pa', 'init': 'smpl_pose_adjust', 'pos': (30, 560), 'w': 280, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'child frame: which frame it applies in',
     'pos': (30, 775)},
    {'key': 'my', 'init': 'smpl_mag_yaw_correct', 'pos': (350, 560), 'w': 280, 'h': 220},
    {'key': 'c1', 'comment': True, 'text': 'global = drift in the room;\nlocal = calibration baked into T-pose',
     'pos': (350, 795)},
    {'key': 'fz', 'init': 'floor_zero', 'pos': (30, 958), 'w': 260, 'h': 160},
    {'key': 'c3', 'comment': True, 'text': 'a CONSTANT offset, so jumps survive',
     'pos': (30, 1133)},
]
links = [('sh', 'body 1 quaternions', 'ss', 'pose'),
         ('sh', 'body 1 positions', 'ss', 'positions'),
         ('be', 'config', 'ss', 'config'),
         ('ss', 'pose', 'pa', 'pose in'),
         ('ss', 'pose', 'my', 'pose in'),
         ('ss', 'trans', 'fz', 'trans'), ('ss', 'pose', 'fz', 'pose'),
         ('be', 'config', 'fz', 'config')]
print(build('smpl_pose_adjust', 'smpl corrections - frames, yaw and the floor', body,
            demo, links, demo_width=690, text_width=810, text_height=780))
