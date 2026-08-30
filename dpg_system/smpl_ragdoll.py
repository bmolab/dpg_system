"""smpl_ragdoll: an SMPL body driven by motion capture, with any subset of
its joints -- and the root -- handed to physics.

The physics is pybullet, in smpl_bullet.py: a URDF built from the
SMPLProcessor skeleton (segment masses, lengths, inertia; the virtual toe,
finger and heel tips), spherical joints, capsule collision, a floor plane.
This file is the node around it: the control surface, pose formats, the
blend weights, the support measure, and the tables the core is built from
(joint limits, named joint groups).

Blend weights
-------------
Every joint, and the root, has a weight from 0 to 1.

    1.0   the capture prescribes the joint outright: a position motor with
          the captured rate fed forward, whatever force it takes; the root is
          a fixed constraint retargeted along the captured path each substep.
    0.0   free: gravity, contact, limits, viscous damping.
    between   a spring-damper toward the capture whose stiffness follows the
          weight: the joint gives blend_soft degrees under its typical load
          at 0 and blend_firm at 1 (log-spaced), with gravity compensation
          above a quarter weight so a torso on springs is not an inverted
          pendulum.  The root hangs on the same spring against body weight.
          Contacts follow the weight too -- compliant when guided, firm when
          free -- so inexact captured foot heights are absorbed, not fought.

Weights are set per joint or per named group (`weight <joints> <v>`), by the
`weights` input (22 SMPL or 20 active values), or all at once by `release`
and `catch`, and ramp over `ramp_ms`.  A catch reels both the root and the
joints from where they are to the capture at a bounded rate, at full
strength, so nothing snaps.

Losing support
--------------
A driven root goes where the capture went, whatever the simulation is doing
under it.  With `auto_release_unsupported` on, once something is released
the node judges support on the captured pose -- how much nearer the pelvis
stands over a driven ground point than a released one, or the force the
free links actually push with -- and after `auto_release_delay` of loss
lets the root go in proportion.  Only a catch takes it back.

Pose formats
------------
Both of this system's layouts are accepted, told apart by shape: SMPL /
SMPL-H axis-angle (24 or 52 joints x 3) and the 20-joint Shadow "active"
quaternion layout.  The `pose` output is in the incoming layout; `smpl_pose`
is always SMPL axis-angle; `trans` is the simulated root position.

Frame rate
----------
Nothing in a pose stream reveals its own rate; the `framerate` property must
match the capture, or every velocity the physics inherits is wrong by the
same factor.  The integration runs at `substep_rate` regardless.

Floor
-----
The floor follows the captured body's lowest point -- the tenth percentile
over `floor_tau` seconds, moved toward at no more than `floor_rate` -- plus
`floor_height`; or, with `floor_auto` off, sits at `floor_height`.
"""
import math
import warnings
import numpy as np
from scipy.spatial.transform import Rotation as R

# Joint limits are read as Euler angles with the tightest axis in the middle,
# so its singularity lies outside the joint's range; scipy still warns on the
# rare exact hit, and the fallback there is harmless.
warnings.filterwarnings('ignore', message='Gimbal lock detected')


RAGDOLL_JOINT_LIMITS = {
    # X flex(-)/extend(+), Y twist, Z ab/adduct.  Z mirrors between legs.
    'left_hip':   ((-2.10, -0.70, -0.35), (0.52, 0.70, 0.79)),
    'right_hip':  ((-2.10, -0.70, -0.79), (0.52, 0.70, 0.35)),
    # X flexion only -- a knee does not bend forwards, so the minimum is zero.
    # Secondary axes wider than a textbook knee: a walk-and-hop capture put
    # the fitted knee 35 degrees in twist and 20 in varus, and a limit the
    # performer exceeds is the model's error.
    'left_knee':  ((0.0, -0.60, -0.35), (2.60, 0.60, 0.35)),
    'right_knee': ((0.0, -0.60, -0.35), (2.60, 0.60, 0.35)),
    # X dorsiflex(-)/plantarflex(+), Y inversion/eversion.
    'left_ankle':  ((-0.60, -0.45, -0.45), (0.87, 0.45, 0.45)),
    'right_ankle': ((-0.60, -0.45, -0.45), (0.87, 0.45, 0.45)),
    'left_foot':   ((-0.35, -0.12, -0.12), (0.87, 0.12, 0.12)),
    'right_foot':  ((-0.35, -0.12, -0.12), (0.87, 0.12, 0.12)),
    # Three spine segments, each carrying a third of the trunk's range: 120
    # degrees of flexion in all, which is what a limp trunk curls to.  At 30
    # per segment the lowest one sat on its flexion stop 95 percent of the
    # time through a released cartwheel -- a torso braced against a spring.
    'spine1': ((-0.35, -0.35, -0.45), (0.87, 0.35, 0.45)),
    'spine2': ((-0.35, -0.35, -0.45), (0.87, 0.35, 0.45)),
    'spine3': ((-0.35, -0.35, -0.45), (0.87, 0.35, 0.45)),
    # Neck and head together give about 65 degrees each way; the head takes
    # the larger share because, being light, it is the one that lolls.
    'neck':   ((-0.61, -0.70, -0.61), (0.61, 0.70, 0.61)),
    'head':   ((-0.52, -0.52, -0.52), (0.52, 0.52, 0.52)),
    # The collars barely move; they carry shoulder-blade shrug and protraction.
    'left_collar':  ((-0.26, -0.70, -0.55), (0.26, 0.70, 0.55)),
    'right_collar': ((-0.26, -0.70, -0.55), (0.26, 0.70, 0.55)),
    # X humeral twist, Y flex(-)/extend(+) for the left, Z elevation.
    # Both Y and Z mirror between arms.
    'left_shoulder':  ((-1.20, -2.00, -1.80), (1.20, 1.20, 1.80)),
    'right_shoulder': ((-1.20, -1.20, -1.80), (1.20, 2.00, 1.80)),
    # Y flexion, one way only; X is forearm pronation, which is real.
    'left_elbow':  ((-1.40, -2.60, -0.35), (1.40, 0.0, 0.35)),
    'right_elbow': ((-1.40, 0.0, -0.35), (1.40, 2.60, 0.35)),
    'left_wrist':  ((-0.35, -1.20, -0.50), (0.35, 1.20, 0.50)),
    'right_wrist': ((-0.35, -1.20, -0.50), (0.35, 1.20, 0.50)),
}


# Self-collision colliders: capsules spanning a pair of joints, with a radius.
#
# The radii are not anatomical, they are the largest that leave every modelled
# pair clear of itself in ordinary poses -- measured off the model rather than
# guessed, because a collider that already overlaps at rest pushes the body
# apart permanently.  The binding constraints are tight: the thighs pass within
# 0.119 m of each other standing, and the spine axis runs 0.103 m from the
# thigh line, so a torso of realistic girth would swallow the hips.  Thighs are
# therefore slimmer than real ones (real thighs touch, and a model that keeps
# them 0.17 m apart would splay the legs), and the pelvis-to-thigh pair is not
# modelled at all -- the hip attaches there, and the hip's own limits are what
# keep a leg out of the pelvis.
JOINT_GROUPS = {
    'left_arm':   [16, 18, 20],
    'right_arm':  [17, 19, 21],
    'arms':       [16, 18, 20, 17, 19, 21],
    'neck':       [12],
    'head':       [12, 15],
    'spine':      [3, 6, 9],
    'left_leg':   [1, 4, 7, 10],
    'right_leg':  [2, 5, 8, 11],
    'legs':       [1, 4, 7, 10, 2, 5, 8, 11],
    'upper_body': [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    # Index 0 is the root's six degrees of freedom, not an ordinary joint --
    # it is simulated by conservation rather than by a joint torque.  It is in
    # 'all', because a set called "all" that quietly leaves out the one degree
    # of freedom that lets a body fall is a trap: every joint goes limp, the
    # root keeps following the capture, and the body sails on through the air
    # with the translation output passing straight through unchanged.
    'root':       [0],
    'joints':     list(range(1, 22)),      # every joint, root still driven
    'all':        list(range(0, 22)),
    'everything': list(range(0, 22)),      # synonym for 'all'
}


class RagdollParams:
    """Plain holder for the per-frame simulation knobs (see smpl_bullet)."""

    def __init__(self):
        self.dt = 1.0 / 60.0
        self.ramp_s = 0.12            # the node's ramp, for a catch that completes over it
        self.gravity = 1.0            # scale on the true gravity field
        # -- the blended regime
        self.motor_strength = 1.0     # multiplier on a partial joint's spring stiffness
        self.blend_soft = 180.0       # degrees a partial joint gives under its typical load at weight 0
        self.blend_firm = 1.0         # ... and at weight 1; log-spaced between (root: half a metre per radian)
        self.partial_damping = 0.5    # damping ratio of a partially driven joint's spring (1 = critical)
        self.spring_rate = 60.0       # 1/s: a spring closes at most this fraction of its error per second
        self.gravity_comp = 1.0       # share of the weight below a partial joint its 'muscle' carries
        self.joint_damping_fraction = 0.05  # a free joint's viscous damping: this fraction of its torque scale at 3 rad/s
        # -- the driven regime
        self.drive_kp = 0.9           # bullet motor gains for driven joints
        self.drive_kd = 0.9
        self.drive_force = 3000.0     # N m: a driven joint tracks the capture, whatever it takes
        self.spike_ratio = 3.0        # feed-forward rate clipped at this multiple of the joint's running speed
        self.spike_floor = 6.0        # rad/s, never clipped below this
        # -- the root
        self.root_hold_force = 1.0e5  # N: the constraint holding a driven root
        self.root_tether = 1.0        # multiplier on the sag a partial root is allowed
        self.root_erp = 1.0           # how much of the root's tracking error the constraint closes per step
        self.root_catch_speed = 2.5   # m/s floor on the speed a catch reels the root in at
        self.root_catch_rate = 6.0    # rad/s
        # -- limits
        self.limit_stiffness = 1.0    # multiplier on the limit springs
        self.limit_entry_s = 0.5      # s over which a joint let go outside its box is eased in
        # -- integration
        self.substeps = 4             # ceiling; the rate below sets the need
        self.substep_rate = 240.0     # Hz the integration should effectively run at
        self.solver_iterations = 40   # bullet constraint solver iterations per substep
        # -- contact
        self.self_collision = True
        self.floor_enable = True
        self.floor_height = 0.0       # m: the floor, or the offset on the estimate with floor_auto
        self.floor_auto = True        # the floor follows the captured body's lowest point
        self.floor_tau = 2.0          # s, window over which the floor is the tenth percentile of the lowest point
        self.floor_rate = 0.05        # m/s, the most the floor estimate may move
        self.support_tolerance = 0.05 # m: captured points this close to the lowest one are ground points
        self.friction = 0.8
        # -- capture discontinuities (a looping file, a seeking stream)
        self.jump_trans = 0.25        # m in one frame that reads as a teleport, not motion
        self.jump_rot = 1.2           # rad in one frame, root or a major joint


import os
from dpg_system.node import Node
from dpg_system.interface_nodes import SliderBankNode
from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions
from dpg_system.body_defs import JointTranslator


# This system's other pose format: the twenty "active" joints derived from the
# Shadow suit, carried as quaternions.  Twenty is unambiguous -- no SMPL layout
# has twenty joints -- so the two can be told apart by shape alone and neither
# source needs a converter bolted on in front of the node.
ACTIVE_JOINT_COUNT = 20

# Active joint names, as aliases for naming free joints.  Most match SMPL, but
# the spine and collars are named differently in the two conventions.
_SMPL_NAME_TO_INDEX = {
    'pelvis': 0, 'left_hip': 1, 'right_hip': 2, 'spine1': 3, 'left_knee': 4,
    'right_knee': 5, 'spine2': 6, 'left_ankle': 7, 'right_ankle': 8,
    'spine3': 9, 'left_foot': 10, 'right_foot': 11, 'neck': 12,
    'left_collar': 13, 'right_collar': 14, 'head': 15, 'left_shoulder': 16,
    'right_shoulder': 17, 'left_elbow': 18, 'right_elbow': 19,
    'left_wrist': 20, 'right_wrist': 21,
}
ACTIVE_NAME_TO_SMPL_INDEX = {
    active: _SMPL_NAME_TO_INDEX[smpl]
    for smpl, active in JointTranslator.smpl_from_bmolab_active_joint_map.items()
    if smpl in _SMPL_NAME_TO_INDEX and active != 'empty'
}
# Active slot -> SMPL index, for per-joint scalars like weights.  (The pose
# translator is not used for these: it pads with an identity rotation, which
# has no meaning for a scalar and the wrong width.)
ACTIVE_INDEX_TO_SMPL_INDEX = {
    JointTranslator.bmolab_active_joints[active]: smpl_idx
    for active, smpl_idx in ACTIVE_NAME_TO_SMPL_INDEX.items()
    if active in JointTranslator.bmolab_active_joints
}


class SMPLRagdollNode(Node):
    """Drive an SMPL body from motion capture while a chosen set of joints
    falls under physics instead.

    Pose in, pose out, in whatever format arrived -- so this drops between a
    capture source and smpl_body, mgl_smpl_mesh or smpl_torque unchanged.
    """

    joint_names = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
        'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    ]

    @staticmethod
    def factory(name, data, args=None):
        return SMPLRagdollNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # State that property callbacks may touch must exist before the
        # widgets do -- patch load fires those callbacks during restore.
        self.processor = None
        self.sim = None
        self.framerate = 60.0
        self.gender = 'neutral'
        self.betas = np.zeros(10)
        self.total_mass = 75.0
        self.free_indices = []
        self.root_free = False
        self._unsupported_time = 0.0
        self.weights = np.zeros(22)
        self.weight_targets = np.zeros(22)
        self.params = RagdollParams()

        self.pose_input = self.add_input('pose', triggers_execution=True)
        self.trans_input = self.add_input('trans')
        self.config_input = self.add_input('config')
        self.weights_input = self.add_input('weights')

        self.pose_output = self.add_output('pose')
        self.smpl_pose_output = self.add_output('smpl_pose')
        self.trans_output = self.add_output('trans')
        self.weights_output = self.add_output('blend_weights')
        self.contact_force_output = self.add_output('contact_forces')
        self.energy_output = self.add_output('energy_removed')
        self.support_output = self.add_output('support')
        self.torque_output = self.add_output('free_torque_vectors')

        # Everything free by default: weights decide what physics gets, and
        # changing the free set resets the simulation, so it should not need
        # changing mid-performance.
        self.free_joints_prop = self.add_property(
            'free_joints', widget_type='text_input', default_value='all',
            callback=self._on_free_joints_changed)
        # Defaults to 1: the node arrives armed rather than already limp.
        # `catch` returns the weights to this value, so a default of 0 made the
        # catch button silently do nothing -- released and held were the same
        # state.  Set it to 0 for a limb that should just hang, in which case
        # release and catch have nothing to do and that is self-consistent.
        self.weight_prop = self.add_property(
            'blend_weight', widget_type='drag_float', default_value=1.0,
            callback=self._on_weight_changed)
        self.release_input = self.add_input(
            'release', widget_type='button', callback=self._release)
        self.catch_input = self.add_input(
            'catch', widget_type='button', callback=self._catch)
        self.ramp_prop = self.add_property(
            'ramp_ms', widget_type='drag_float', default_value=120.0)
        self.reset_input = self.add_input(
            'reset', widget_type='button', callback=self._reset_sim)

        self.gravity_prop = self.add_property(
            'gravity', widget_type='drag_float', default_value=1.0)
        self.auto_release_prop = self.add_property(
            'auto_release_unsupported', widget_type='checkbox', default_value=True)
        self.auto_release_delay_prop = self.add_option(
            'auto_release_delay', widget_type='drag_float', default_value=0.15)
        self.limit_stiffness_prop = self.add_property(
            'limit_stiffness', widget_type='drag_float', default_value=1.0)

        self.self_collision_prop = self.add_property(
            'self_collision', widget_type='checkbox', default_value=True)
        self.floor_enable_prop = self.add_property(
            'floor_enable', widget_type='checkbox', default_value=True)
        self.floor_height_prop = self.add_property(
            'floor_height', widget_type='drag_float', default_value=0.0)
        self.floor_auto_prop = self.add_property(
            'floor_auto', widget_type='checkbox', default_value=True)
        self.floor_tau_prop = self.add_option('floor_tau', widget_type='drag_float', default_value=2.0)
        self.floor_rate_prop = self.add_option('floor_rate', widget_type='drag_float', default_value=0.05)
        self.friction_prop = self.add_property(
            'friction', widget_type='drag_float', default_value=0.8)

        self.motor_strength_prop = self.add_option(
            'motor_strength', widget_type='drag_float', default_value=1.0)
        self.partial_damping_prop = self.add_option('partial_damping', widget_type='drag_float', default_value=0.5)
        self.spring_rate_prop = self.add_option('spring_rate', widget_type='drag_float', default_value=60.0)
        self.gravity_comp_prop = self.add_option('gravity_comp', widget_type='drag_float', default_value=1.0)
        self.blend_soft_prop = self.add_option('blend_soft', widget_type='drag_float', default_value=180.0)
        self.blend_firm_prop = self.add_option('blend_firm', widget_type='drag_float', default_value=1.0)
        self.substeps_prop = self.add_option(
            'substeps', widget_type='drag_int', default_value=4)
        self.substep_rate_prop = self.add_option(
            'substep_rate', widget_type='drag_float', default_value=240.0)
        self.total_mass_prop = self.add_option(
            'total_mass', widget_type='drag_float', default_value=75.0)

        self.up_axis_prop = self.add_option('up_axis', widget_type='combo', default_value='Y')
        self.up_axis_prop.widget.combo_items = ['Y', 'Z']
        self.axis_perm_prop = self.add_option(
            'axis_permutation', widget_type='text_input', default_value='x, z, -y')
        self.quat_format_prop = self.add_option('quat_format', widget_type='combo', default_value='wxyz')
        self.quat_format_prop.widget.combo_items = ['xyzw', 'wxyz']

        self._parse_free_joints('all')

        # Take over the button names as a superset: with no arguments they do
        # what the button does.
        self.message_handlers['weight'] = self._weight_message
        self.message_handlers['release'] = self._release_message
        self.message_handlers['catch'] = self._catch_message

    # ------------------------------------------------------------------
    # Free set and weights
    # ------------------------------------------------------------------

    def _resolve_joints(self, tokens):
        """Joint indices for a list of names: groups ('left_arm'), SMPL joint
        names ('left_elbow'), active-convention names ('left_shoulder_blade')
        or indices.  Returns (indices, unknown tokens)."""
        indices = []
        unknown = []
        for token in tokens:
            tok = str(token).strip().lower()
            if not tok:
                continue
            if tok in JOINT_GROUPS:
                indices.extend(JOINT_GROUPS[tok])
            elif tok in self.joint_names:
                indices.append(self.joint_names.index(tok))
            elif tok in ACTIVE_NAME_TO_SMPL_INDEX:
                indices.append(ACTIVE_NAME_TO_SMPL_INDEX[tok])
            elif tok.isdigit():
                indices.append(int(tok))
            else:
                unknown.append(tok)
        return sorted({i for i in indices if 0 <= i < 22}), unknown

    def _parse_free_joints(self, text):
        """Accept group names ('left_arm'), joint names ('left_elbow') or
        indices, comma separated."""
        indices, unknown = self._resolve_joints(
            str(text).replace(';', ',').split(','))
        if unknown:
            print(f'smpl_ragdoll: unknown joint or group {unknown}; '
                  f'known groups: {sorted(JOINT_GROUPS)}')

        self.free_indices = sorted({i for i in indices if 0 <= i < 22})
        self.root_free = 0 in self.free_indices
        # Reported once per change, because whether the root is in the set is
        # not visible anywhere else and decides whether the body can fall at
        # all -- with it driven, the translation output passes the capture
        # straight through however limp the joints go.
        joints = [self.joint_names[i] for i in self.free_indices if i > 0]
        listed = ', '.join(joints) if len(joints) <= 6 else '%d joints' % len(joints)
        print('smpl_ragdoll: free = %s; root %s'
              % (listed or 'nothing',
                 'FREE (the body can fall)' if self.root_free
                 else 'DRIVEN (translation follows the capture)'))
        if self.sim is not None:
            self.sim.set_free_joints([j for j in self.free_indices if j > 0])
            self.sim.set_root_free(self.root_free)
        self._apply_weight_immediately()

    def _base_weight(self):
        try:
            return float(np.clip(self.weight_prop(), 0.0, 1.0))
        except Exception:
            return 1.0        # matches the property's default

    def _apply_weight_immediately(self):
        w = self._base_weight()
        self.weights[:] = 1.0
        self.weight_targets[:] = 1.0
        for j in self.free_indices:
            self.weights[j] = w
            self.weight_targets[j] = w

    def _on_free_joints_changed(self):
        try:
            self._parse_free_joints(self.free_joints_prop())
        except Exception as e:
            print(f'smpl_ragdoll: free_joints parse failed: {e}')

    def _on_weight_changed(self):
        self._apply_weight_immediately()

    # ------------------------------------------------------------------
    # Per-joint weights
    # ------------------------------------------------------------------
    #
    # The simulation has always carried a weight per joint; this is the
    # control surface for it.  Messages, sent to any input as a string or a
    # list:
    #
    #     weight <joints...> <value>     ramp the named joints to a weight
    #     release <joints...>            ramp them to 0   (no names: all)
    #     catch <joints...>              ramp them back to blend_weight
    #
    # with joints named as groups or in either joint-name convention, e.g.
    # "weight arms 0", "release left_leg", "catch base_of_skull".  The
    # `weights` input takes a whole array instead -- 22 in SMPL order (root
    # first), or 20 in the active order.  All of it goes through the ramp.
    #
    # A weight means something only for a joint in the free set, and adding a
    # joint to that set resets the simulation -- so for live work name
    # everything free up front (the default) and let the weights decide.  A
    # message naming a joint that is not free says so and leaves it alone.
    #
    # The `blend_weight` slider and a bare `catch` still act on the whole set.

    def _targets_for(self, tokens, what):
        """Free joints named by tokens, warning about the rest."""
        indices, unknown = self._resolve_joints(tokens)
        if unknown:
            print(f'smpl_ragdoll: {what}: unknown joint or group {unknown}')
        free = set(self.free_indices)
        not_free = [self.joint_names[i] for i in indices if i not in free]
        if not_free:
            print(f'smpl_ragdoll: {what}: {not_free} not in free_joints -- '
                  f'ignored (changing free_joints resets the simulation; '
                  f'set it to "all" and use weights instead)')
        return [i for i in indices if i in free]

    def _weight_message(self, message='', args=None):
        args = list(args or [])
        if len(args) < 2:
            print('smpl_ragdoll: usage: weight <joints...> <value>')
            return
        try:
            value = float(np.clip(float(args[-1]), 0.0, 1.0))
        except (TypeError, ValueError):
            print(f'smpl_ragdoll: weight: last argument must be a number, got {args[-1]!r}')
            return
        for j in self._targets_for(args[:-1], 'weight'):
            self.weight_targets[j] = value

    def _release_message(self, message='', args=None):
        args = list(args or [])
        if not args:
            self._release()
            return
        for j in self._targets_for(args, 'release'):
            self.weight_targets[j] = 0.0

    def _catch_message(self, message='', args=None):
        args = list(args or [])
        if not args:
            self._catch()
            return
        self._unsupported_time = 0.0
        w = self._base_weight()
        for j in self._targets_for(args, 'catch'):
            self.weight_targets[j] = w

    def _apply_weights_array(self, data):
        """A whole per-joint array of weight targets: 22 (SMPL, root first) or
        20 (active order)."""
        a = np.asarray(data, dtype=float).reshape(-1)
        if a.size == ACTIVE_JOINT_COUNT:
            # Joints with no active counterpart (the feet) stay driven.
            mapped = np.ones(22)
            for act_i, smpl_i in ACTIVE_INDEX_TO_SMPL_INDEX.items():
                if act_i < ACTIVE_JOINT_COUNT and smpl_i < 22:
                    mapped[smpl_i] = a[act_i]
            a = mapped
        elif a.size >= 22:
            a = a[:22]
        else:
            print(f'smpl_ragdoll: weights input needs 22 (SMPL) or 20 (active) '
                  f'values, got {a.size}')
            return
        a = np.clip(a, 0.0, 1.0)
        for j in self.free_indices:
            self.weight_targets[j] = a[j]

    def _release(self):
        """Ramp every free joint's weight to zero -- let go."""
        for j in self.free_indices:
            self.weight_targets[j] = 0.0

    def _catch(self):
        """Ramp back to the set blend weight -- take hold again."""
        # An explicit catch outranks the automatic release: the timer restarts
        # so the ramp can finish before support is judged again.
        self._unsupported_time = 0.0
        w = self._base_weight()
        if w <= 1e-6:
            # Never leave the button silently inert.
            print('smpl_ragdoll: catch returns the joints to blend_weight, '
                  'which is 0 -- raise it (1 = fully driven by the capture) '
                  'for catch to take hold.')
            return
        for j in self.free_indices:
            self.weight_targets[j] = w

    def _reset_sim(self):
        if self.sim is not None:
            self.sim.reset()
        self._unsupported_time = 0.0
        self._apply_weight_immediately()

    def _advance_weights(self, dt):
        # A driven root that the simulated body can no longer be holding up is
        # let go, in proportion to how much of its support has actually gone --
        # what a cartwheel with the arms released needs, since the capture
        # keeps the root travelling over hands that are no longer touching.
        #
        # Four conditions before it may act, each learned from a way this went
        # wrong:
        #
        #   something must already be released -- with every joint driven the
        #     simulated body *is* the captured body, so a shortfall is not lost
        #     support but a floor set wrong, and acting on it drops the body
        #     the instant the patch starts;
        #   no weight may be ramping up, and a catch restarts the timer, or
        #     taking hold again is undone on the very next frame and the root
        #     can never be recovered at all;
        #   the loss must persist, because contact force is present or absent
        #     with nothing in between, and a capture a few centimetres high
        #     reads as unsupported the moment it is not quite touching.
        if not bool(self.auto_release_prop()):
            self._unsupported_time = 0.0
        else:
            # released means meaningfully so: a joint at 0.999 is driven, and
            # counting it let a body at 0.999 be dropped for lack of support
            released = bool(np.any(self.weights[1:] < 0.5))
            # only a rising weight (a catch) blocks it: a release in progress
            # is exactly when the loss it is measuring begins
            ramping = bool(np.any(self.weight_targets - self.weights > 1e-6))
            if self.root_free and self.sim is not None and released and not ramping:
                support = float(getattr(self.sim, 'last_support', 1.0))
                if support < 0.5:
                    self._unsupported_time += dt
                else:
                    self._unsupported_time = 0.0
                if (self._unsupported_time >= float(self.auto_release_delay_prop())
                        and support < self.weight_targets[0]):
                    self.weight_targets[0] = support
            else:
                self._unsupported_time = 0.0

        ramp_s = max(float(self.ramp_prop()), 1e-3) / 1000.0
        rate = dt / ramp_s
        delta = self.weight_targets - self.weights
        step = np.clip(delta, -rate, rate)
        self.weights += step

    # ------------------------------------------------------------------
    # Processor
    # ------------------------------------------------------------------

    def _to_array(self, d):
        return np.asarray(d, dtype=float) if not isinstance(d, np.ndarray) else d.astype(float)

    def _ensure_processor(self, rebuild=False):
        if self.processor is None or rebuild:
            self.processor = SMPLProcessor(
                framerate=self.framerate,
                betas=self.betas,
                gender=self.gender,
                total_mass_kg=float(self.total_mass_prop()),
                model_path=os.path.dirname(os.path.abspath(__file__)))
            self.sim = self._make_sim()
            self.sim.set_free_joints([j for j in self.free_indices if j > 0])
            self.sim.set_root_free(self.root_free)
            # Reported because nothing else shows it and the frame rate is not
            # guessable from the pose stream.  Every velocity in the simulation
            # is differenced against it, so a body captured at 120 Hz and run
            # as 60 is handed half its real speed and the physics degenerates
            # from there.  Connect `config` (smpl_beta_editor emits it, or
            # build it from the take) to set it.
            print('smpl_ragdoll: body = %s, %g Hz, betas %s'
                  % (self.gender, self.framerate,
                     'supplied' if np.any(self.betas) else 'ZERO (no config?)'))

    def _make_sim(self):
        """The physics core: pybullet."""
        from dpg_system.smpl_bullet import BulletRagdollSim, PYBULLET_AVAILABLE
        if not PYBULLET_AVAILABLE:
            raise ImportError('smpl_ragdoll needs pybullet (conda install -c conda-forge pybullet)')
        return BulletRagdollSim(self.processor)

    def _handle_config(self):
        if not self.config_input.fresh_input:
            return
        cfg = self.config_input()
        if not isinstance(cfg, dict):
            return
        changed = False
        for k in ('motioncapture_framerate', 'mocap_framerate', 'framerate'):
            if k in cfg:
                fr = float(cfg[k])
                if fr != self.framerate:
                    self.framerate = fr
                    changed = True
                break
        if 'gender' in cfg and str(cfg['gender']) != self.gender:
            self.gender = str(cfg['gender'])
            changed = True
        if 'betas' in cfg:
            b = self._to_array(cfg['betas'])
            if self.betas is None or not np.array_equal(self.betas, b):
                self.betas = b
                changed = True
        if changed:
            self._ensure_processor(rebuild=True)

    # ------------------------------------------------------------------
    # Pose shape handling
    # ------------------------------------------------------------------

    @staticmethod
    def _split_pose(orig):
        """Normalize any accepted pose layout to (F, n_joints, C).

        Returns (view, F, n_joints, C) or None if the layout is unrecognised.
        The returned array is a fresh copy -- the caller's buffer is not
        touched.
        """
        a = np.array(orig, dtype=float)
        if a.ndim == 1:
            sizes = {60: (20, 3), 80: (20, 4),          # active joints
                     66: (22, 3), 72: (24, 3), 88: (22, 4),
                     96: (24, 4), 156: (52, 3), 208: (52, 4)}
            if a.size not in sizes:
                return None
            n, c = sizes[a.size]
            return a.reshape(1, n, c), 1, n, c
        if a.ndim == 2:
            if a.shape[1] in (3, 4) and (a.shape[0] >= 22
                                         or a.shape[0] == ACTIVE_JOINT_COUNT):
                return a.reshape(1, a.shape[0], a.shape[1]), 1, a.shape[0], a.shape[1]
            return None
        if a.ndim == 3 and a.shape[2] in (3, 4):
            return a, a.shape[0], a.shape[1], a.shape[2]
        return None

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self):
        self._handle_config()
        self._ensure_processor()

        if not self.pose_input.fresh_input:
            return

        raw = self.pose_input()
        split = self._split_pose(raw)
        trans = self.trans_input()
        trans_raw = trans
        trans = np.zeros(3) if trans is None else self._to_array(trans)
        trans = np.asarray(trans, dtype=float).reshape(-1)
        root_trans = np.zeros(3)
        root_trans[:min(3, trans.size)] = trans[:3]

        if split is None:
            print('smpl_ragdoll: unrecognised pose layout, passing through')
            self.pose_output.send(raw)
            self.trans_output.send(trans_raw if trans_raw is not None else root_trans)
            return
        out_pose, F, n_joints, C = split

        # An active-joint stream is lifted into SMPL here and put back on the
        # way out, so the simulation only ever deals in one layout and the
        # patch does not need converters wrapped around the node.
        is_active = (n_joints == ACTIVE_JOINT_COUNT)
        if is_active:
            work = np.zeros((F, 24, C))
            if C == 4:
                work[:, :, 0 if self.quat_format_prop() == 'wxyz' else 3] = 1.0
            for f in range(F):
                work[f, :22] = JointTranslator.translate_from_bmolab_active_to_smpl(
                    out_pose[f])
            work_joints = 24
        else:
            work = out_pose
            work_joints = n_joints

        if self.weights_input.fresh_input:
            self._apply_weights_array(self.weights_input())

        dt = 1.0 / max(self.framerate, 1.0)
        self._advance_weights(dt)

        # Nothing configured as free: the node is inert, pass through.
        #
        # Note this deliberately does NOT skip out when everything is merely
        # prescribed (weight 1) and waiting for a release.  While prescribed,
        # the simulation shadows the capture and accumulates the velocity and
        # spin that a release inherits; skipping those frames would make the
        # body let go from rest, with no momentum at all.
        if not self.free_indices:
            self.weights_output.send(self.weights.copy())
            self.pose_output.send(raw)
            # Still emitted, so an inert node is a working format converter
            # rather than a dead end in the chain.
            self.smpl_pose_output.send(self._as_smpl_axis_angle(work, F, C))
            self.trans_output.send(trans_raw if trans_raw is not None else root_trans)
            return

        options = SMPLProcessingOptions(
            input_type='quat' if C == 4 else 'axis_angle',
            input_up_axis=self.up_axis_prop(),
            axis_permutation=self.axis_perm_prop(),
            quat_format=self.quat_format_prop(),
            dt=dt)

        p = self.params
        p.dt = dt
        p.ramp_s = max(float(self.ramp_prop()), 1e-3) / 1000.0
        p.motor_strength = float(self.motor_strength_prop())
        p.partial_damping = float(self.partial_damping_prop())
        p.spring_rate = float(self.spring_rate_prop())
        p.gravity_comp = float(self.gravity_comp_prop())
        p.blend_soft = float(self.blend_soft_prop())
        p.blend_firm = float(self.blend_firm_prop())
        p.substeps = max(1, int(self.substeps_prop()))
        p.substep_rate = float(self.substep_rate_prop())
        p.gravity = float(self.gravity_prop())
        p.limit_stiffness = float(self.limit_stiffness_prop())
        p.self_collision = bool(self.self_collision_prop())
        p.floor_enable = bool(self.floor_enable_prop())
        p.floor_height = float(self.floor_height_prop())
        p.floor_tau = float(self.floor_tau_prop())
        p.floor_rate = float(self.floor_rate_prop())
        p.floor_auto = bool(self.floor_auto_prop())
        p.friction = float(self.friction_prop())

        proc = self.processor
        out_trans = root_trans
        for f in range(F):
            # Crop to the SMPL body joints. An AMASS / SMPL-H stream carries
            # 52, and the processor's quaternion reshape assumes 24.
            frame = work[f:f + 1, :24].copy()
            try:
                t_int, aa_int, _quats = proc._prepare_trans_and_pose(
                    frame, root_trans.reshape(1, 3), options)
                result, root_rot, trans_int = self.sim.advance(
                    aa_int[0], t_int[0], self.weights, p)
            except Exception as e:
                print(f'smpl_ragdoll: simulation failed ({e}); passing through')
                self.pose_output.send(raw)
                self.trans_output.send(root_trans)
                return

            # A non-root joint's local rotation is the same in the incoming
            # frame and the internal one -- the axis permutation and the
            # up-axis conversion touch only the root -- so it can be written
            # straight back.  The root cannot, and is converted below.
            for j, aa in result.items():
                self._write_joint(work, f, j, work_joints, C, aa)

            if self.root_free:
                in_rot, out_trans = self._to_input_frame(root_rot, trans_int)
                self._write_joint(work, f, 0, work_joints, C, in_rot.as_rotvec())

        if is_active:
            for f in range(F):
                back = JointTranslator.translate_from_smpl_to_bmolab_active(work[f])
                out_pose[f] = back[:ACTIVE_JOINT_COUNT]
        shaped = out_pose.reshape(np.shape(raw)) if np.ndim(raw) != 3 else out_pose
        self.pose_output.send(shaped)
        self.smpl_pose_output.send(self._as_smpl_axis_angle(work, F, C))
        self.trans_output.send(out_trans)
        self.weights_output.send(self.weights.copy())
        self.torque_output.send(self.sim.last_torque.copy())
        self.contact_force_output.send(self.sim.last_contact_force.copy())
        self.energy_output.send(self.sim.last_energy_injected)
        self.support_output.send(self.sim.last_support)

    def _as_smpl_axis_angle(self, work, F, C):
        """The pose as SMPL axis-angle, whatever came in.

        The `pose` output matches the incoming layout so the node drops into an
        existing chain unchanged; this one is always SMPL, which is what
        mgl_smpl_mesh, smpl_body and smpl_torque want.  Both are emitted every
        frame, so neither consumer needs a converter and neither has to be
        chosen in advance.
        """
        out = np.zeros((F, 24, 3))
        # The working array may carry 22 joints (a 66-float SMPL stream), 24,
        # or 52 (SMPL-H); pad or crop to the 24 the SMPL body expects.
        n = min(24, work.shape[1])
        for f in range(F):
            if C == 3:
                out[f, :n] = work[f, :n, :3]
            else:
                q = np.array(work[f, :n, :4], dtype=float)
                if self.quat_format_prop() == 'wxyz':
                    q = np.roll(q, -1, axis=-1)          # -> xyzw for scipy
                norm = np.linalg.norm(q, axis=-1, keepdims=True)
                q = np.where(norm > 1e-9, q / np.maximum(norm, 1e-9),
                             np.array([0.0, 0.0, 0.0, 1.0]))
                out[f, :n] = R.from_quat(q).as_rotvec()
        return out[0] if F == 1 else out

    def _write_joint(self, out_pose, f, j, n_joints, C, aa):
        """Write one local rotation back in the layout that arrived."""
        if j >= n_joints:
            return
        if C == 3:
            out_pose[f, j] = aa
        else:
            q = R.from_rotvec(aa).as_quat()              # xyzw
            if self.quat_format_prop() == 'wxyz':
                q = np.roll(q, 1)
            out_pose[f, j] = q

    def _to_input_frame(self, root_rot, trans):
        """Invert the processor's frame conversion for the root.

        Mirrors _prepare_trans_and_pose, which applies the axis permutation and
        then, for Z-up input, a further -90 degree rotation about X.  Only the
        root's rotation and the translation are affected, which is why every
        other joint needs no inverse.
        """
        proc = self.processor
        t = np.asarray(trans, dtype=float).reshape(3)
        rot = root_rot

        if self.up_axis_prop() == 'Z':
            # Forward was  internal = (x, z, -y)  on the permuted translation.
            t = np.array([t[0], -t[2], t[1]])
            rot = R.from_euler('x', -90, degrees=True).inv() * rot

        basis = getattr(proc, 'perm_basis', None)
        if basis is not None:
            # Forward was  permuted = t @ basis.T, and a signed permutation is
            # orthogonal, so the inverse is a right-multiply by the basis.
            t = t @ basis
            rot = R.from_matrix(proc.perm_basis_rot).inv() * rot
        return rot, t


class RagdollBlendUINode(SliderBankNode):
    """ragdoll_blend_ui: one slider per body region, each sending
    'weight <region> <value>' -- connect its output to any input of an
    smpl_ragdoll.  Names and the message template are the slider_bank
    options, so a slider can be renamed to a single joint ('left_elbow') or
    the template changed to drive something else."""
    default_names = ['root', 'spine', 'head', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
    default_template = 'weight {name} {value}'
    default_min = 0.0
    default_max = 1.0
    default_value = 1.0

    @staticmethod
    def factory(name, data, args=None):
        return RagdollBlendUINode(name, data, args)
