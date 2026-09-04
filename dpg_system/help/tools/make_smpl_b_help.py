"""smpl_torque, the ragdoll, and the per-joint splitters."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ------------------------------------------------------------------ smpl_torque
body = """smpl_torque works out the forces inside a moving body.

A pose tells you where the body is. This tells you what it had to DO to be 
there - the torque at every joint, frame by frame. That is a different kind of 
information: position is geometry, torque is effort, and effort is what a body 
is actually spending.

The calculation is inverse dynamics. Given the pose, the translation, and the 
body's size and mass from the config, it works backwards from the motion to the 
forces that must have produced it - gravity acting on every segment, the 
acceleration of each mass, the passive resistance of joints near their limits, 
and the reaction from the floor where the body is in contact with it.

THE OUTLETS, AND WHY THERE ARE SO MANY:
The total torque at a joint is a sum of causes, and separating them is most of 
the value.

torque_vectors           the total at each joint
gravity_torque_vectors   the part that is just holding the body up
dynamic_torque_vectors   the part that is accelerating it
passive_torque_vectors   the part the joint's own limits are contributing
root_torque              what the whole body is doing about its root
combined_effort          a single summary figure
joint_positions          where the joints ended up
inertias                 what each segment is resisting with

Holding an arm out and swinging an arm can produce the same total, and they are 
not the same act. Gravity torque is the cost of the posture; dynamic torque is 
the cost of the change. Passive torque is the body resisting itself, and it 
rising is a sign of a joint near the end of its range.

THE SETTING TO GET RIGHT FIRST:
'axis_permutation' - and it defaults to 'x, z, -y', which converts SMPL's Z-up 
into Y-up. This node does the conversion ITSELF. Do not convert before it and 
leave the setting at its default, or it is applied twice.

Match the permutation to what you are actually feeding in, rather than 
converting upstream to suit it. Everything else in the node - gravity, floor 
contact, which way is down - depends on this being right, and getting it wrong 
gives plausible-looking numbers that are wrong.

THE REST OF THE OPTIONS, BY GROUP:
There are a great many. They divide into:

  what to include   add_gravity, enable_passive_limits, enable_apparent_gravity,
                    zero_root_torque, world_frame_dynamics
  the floor         floor_contact_enable, floor_height, floor_tolerance,
                    contact_method, enable_body_contacts
  contact sensing   the lo_ group - a log-odds estimate of which feet are
                    bearing weight, fusing height, kinematics, structural load,
                    divergence and touchdown
  smoothing         the com_ and smooth_ groups, and torque_smooth_window
  gating            the coherence_gate_ group, which suppresses torque that is
                    not coherent across the body

Most have sensible defaults. The ones worth touching early are 
axis_permutation, the floor settings, and torque_smooth_window.

SYNTAX:
smpl_torque

EXAMPLE:
smpl_torque

INPUTS and PARAMETERS:

pose:
The SMPL pose. Receiving it triggers the calculation.

trans:
The root translation. Without it the body has no motion through space, so 
dynamic torque is only what the limbs do relative to the root.

config:
Betas, gender and total mass, from smpl_beta_editor. The mass matters directly - 
every torque scales with it.

reset_state / reset_noise_stats / reset_floor:
Clear the running estimates.

OUTPUTS: 

See above. All the vector outlets are per joint.

A CAUTION ON TRUSTING THE NUMBERS:
Inverse dynamics amplifies noise, because it depends on acceleration and 
acceleration is the second derivative of a measured position. A jitter too 
small to see in the pose becomes a large spike in the torque. That is what the 
smoothing and gating options are for, and it is why a torque trace should be 
looked at alongside the movement that produced it rather than on its own."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'be', 'init': 'smpl_beta_editor', 'pos': (350, 62), 'w': 300, 'h': 340},
    {'key': 'ss', 'init': 'shadow_to_smpl', 'pos': (30, 400), 'w': 260, 'h': 140},
    {'key': 'st', 'init': 'smpl_torque', 'pos': (30, 560), 'w': 300, 'h': 400},
    {'key': 'c0', 'comment': True, 'text': 'set axis_permutation to match your input\nit does the up-axis conversion itself',
     'pos': (30, 975)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (380, 560), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 22,
               'min y': 0.0, 'max y': 50.0, 'update_mode': 'heat_map',
               'number format': '%.0f'}},
    {'key': 'c2', 'comment': True, 'text': 'total torque per joint', 'pos': (380, 720)},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (380, 760), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 22,
               'min y': 0.0, 'max y': 50.0, 'update_mode': 'heat_map',
               'number format': '%.0f'}},
    {'key': 'c3', 'comment': True, 'text': 'gravity only: the cost of the posture',
     'pos': (380, 920)},
    {'key': 'f1', 'init': 'float', 'pos': (380, 960), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c4', 'comment': True, 'text': 'combined effort, as one number',
     'pos': (380, 1010)},
]
links = [('sh', 'body 1 quaternions', 'ss', 'pose'),
         ('sh', 'body 1 positions', 'ss', 'positions'),
         ('be', 'config', 'ss', 'config'),
         ('ss', 'pose', 'st', 'pose'), ('ss', 'trans', 'st', 'trans'),
         ('be', 'config', 'st', 'config'),
         ('st', 'torque_vectors', 'hm', 'y'),
         ('st', 'gravity_torque_vectors', 'hm2', 'y'),
         ('st', 'combined_effort', 'f1', '')]
print(build('smpl_torque', 'smpl_torque - what the body had to do', body, demo, links,
            demo_width=690, text_width=830, text_height=840))

# ----------------------------------------------------------------- smpl_ragdoll
body = """smpl_ragdoll lets go of part of the body and hands it to physics.

The pose coming in is what the performer did. This node picks joints to RELEASE 
and simulates them instead - they fall, swing, collide and settle under gravity, 
while the rest of the body keeps following the performer.

That makes it a blend rather than a switch. A released arm is still attached to 
a driven shoulder, so it swings from a body that is moving as the person moves. 
'blend_weight' sets how far each released joint has gone from driven to free, 
and 'ramp_ms' how long that transition takes - so a limb can be let go and 
caught again without a discontinuity.

THE NODES:

smpl_ragdoll     the simulation
ragdoll_blend_ui a bank of named sliders for the per-joint blend weights

ragdoll_blend_ui EXISTS BECAUSE THE WEIGHTS ARE PER JOINT:
It is a slider_bank with the joint names filled in, sending the messages 
smpl_ragdoll expects. Twenty-odd separate sliders and cords would be unusable; 
this is one node and one cord.

RELEASE AND CATCH:
'release' hands the named joints to physics. 'catch' takes them back, ramping 
so the limb returns to the performer's pose rather than snapping to it.

'auto_release_unsupported' releases a limb when nothing is holding it up, after 
'auto_release_delay'. That is how a body goes limp when it stops supporting 
itself, without anyone deciding the moment.

WHAT COMES BACK OUT:
Besides the blended pose, the node reports what the physics found - 
'contact_forces' where the body is touching, 'support' for whether it is held 
up, 'energy_removed' for what the collisions absorbed, and 
'free_torque_vectors' for what the released joints are experiencing.

Those are worth watching rather than only using: 'support' falling is the 
physical fact behind a body giving way, and it is available before anything 
visibly collapses.

SYNTAX:
smpl_ragdoll

EXAMPLE:
smpl_ragdoll

INPUTS and PARAMETERS:

pose / trans / config:
The driven pose, its translation, and the body's size and mass.

weights:
Per-joint blend, from ragdoll_blend_ui.

release / catch / reset:
Hand joints to physics, take them back, start again.

free_joints:
Which joints may be released.

blend_weight / ramp_ms / blend_soft / blend_firm:
How far from driven to free, and how the transition is shaped.

gravity / total_mass / limit_stiffness / friction / self_collision:
The physics.

floor_enable / floor_height / floor_auto:
The ground it lands on.

motor_strength / spring_rate / partial_damping / gravity_comp:
How strongly a partly-released joint is still driven.

substeps / substep_rate:
Simulation accuracy. More is stabler and slower - raise it if limbs jitter or 
pass through each other.

up_axis / axis_permutation / quat_format:
As on smpl_torque, and with the same warning: this node does the conversion 
itself, so do not also do it upstream.

OUTPUTS: 

pose / smpl_pose / trans:
The blended result.

blend_weights / support / contact_forces / energy_removed / free_torque_vectors:
What the simulation is doing and what it found."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'be', 'init': 'smpl_beta_editor', 'pos': (350, 62), 'w': 300, 'h': 340},
    {'key': 'ss', 'init': 'shadow_to_smpl', 'pos': (30, 400), 'w': 260, 'h': 140},
    {'key': 'ui', 'init': 'ragdoll_blend_ui', 'pos': (350, 420), 'w': 300, 'h': 300},
    {'key': 'c0', 'comment': True, 'text': 'one cord instead of twenty', 'pos': (350, 735)},
    {'key': 'rd', 'init': 'smpl_ragdoll', 'pos': (30, 560), 'w': 300, 'h': 400},
    {'key': 'c1', 'comment': True, 'text': 'release a limb and it swings from a\nshoulder the performer is still moving',
     'pos': (30, 975)},
    {'key': 'f1', 'init': 'float', 'pos': (350, 780), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'support falling is the body giving way,\navailable before it visibly does',
     'pos': (350, 830)},
]
links = [('sh', 'body 1 quaternions', 'ss', 'pose'),
         ('sh', 'body 1 positions', 'ss', 'positions'),
         ('be', 'config', 'ss', 'config'),
         ('ss', 'pose', 'rd', 'pose'), ('ss', 'trans', 'rd', 'trans'),
         ('be', 'config', 'rd', 'config'),
         ('ui', 'messages', 'rd', 'weights'),
         ('rd', 'support', 'f1', '')]
print(build('smpl_ragdoll', 'smpl_ragdoll - letting go of part of the body', body,
            demo, links, demo_width=690, text_width=810, text_height=800))

# ----------------------------------------------------------- smpl_pose_to_joints
body = """These split an SMPL pose into one outlet per joint.

A pose is 22 rotations travelling together. Most of the time that is what you 
want - it stays one thing, and every node downstream takes it whole. 
Sometimes you want ONE joint: the left knee to drive something, the head to 
point a camera, the spine to set a mood.

THE NODES:

smpl_pose_to_joints   split an axis-angle pose, one outlet per joint
smpl_quats_to_joints  the same for a quaternion pose

The outlets are named for the joints, so the patch says what it is doing - a 
cord from "left elbow" is self-documenting in a way that indexing into an array 
is not.

WHEN NOT TO USE THESE:
If you are going to do the same thing to every joint, keep the pose whole and 
work on the array - the numpy and torch nodes will do it in one operation, and 
twenty-two parallel branches of identical patching is both slower and much 
harder to change.

These are for when joints are being treated DIFFERENTLY.

SYNTAX:
smpl_pose_to_joints
smpl_quats_to_joints

EXAMPLE:
smpl_quats_to_joints

INPUTS and PARAMETERS:

pose in:
The pose. Receiving it triggers the node.

OUTPUTS: 

One outlet per SMPL joint, named.

RELATED:
body_to_joints does the same for a Shadow or active pose. 
t.index_select and np.[] pull one joint out of a pose kept whole, which is the 
better choice when the index is decided by the patch rather than by the 
patching."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'be', 'init': 'smpl_beta_editor', 'pos': (350, 62), 'w': 300, 'h': 340},
    {'key': 'ss', 'init': 'shadow_to_smpl', 'pos': (30, 400), 'w': 260, 'h': 140},
    {'key': 'sq', 'init': 'smpl_to_quats', 'pos': (30, 560), 'w': 240, 'h': 90},
    {'key': 'pj', 'init': 'smpl_quats_to_joints', 'pos': (30, 670), 'w': 280, 'h': 260},
    {'key': 'c0', 'comment': True, 'text': 'outlets named for the joints\nuse these when joints are treated\ndifferently - not for the same thing to all', 'pos': (30, 945)},
]
links = [('sh', 'body 1 quaternions', 'ss', 'pose'),
         ('sh', 'body 1 positions', 'ss', 'positions'),
         ('be', 'config', 'ss', 'config'),
         ('ss', 'pose', 'sq', 'smpl pose'),
         ('sq', 'smpl pose as quaternions', 'pj', 'pose in')]
print(build('smpl_pose_to_joints', 'smpl_pose_to_joints - one outlet per joint', body,
            demo, links, demo_width=690, text_width=790, text_height=640))
