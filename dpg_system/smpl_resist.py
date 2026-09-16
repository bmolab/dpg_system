"""smpl_resist: a body that resists going with the flow.

smpl_ragdoll runs one way: a joint's control is taken from the performer and
gravity and momentum fill in.  This runs the other way.  It finds how much
of each joint's captured motion is the body going along with gravity and
with the momentum handed to it by the rest of the body -- the arm swing a
walk pumps, the head lagging a turn, the follow-through after a throw, a
limb dropping -- and pushes back against exactly that share, leaving the
performer's own action alone.

The split
---------
Inverse dynamics on the capture, joint by joint.  Everything below a joint
is taken as one rigid body pinned there; the torque gravity and the joint
point's own acceleration exert on it is the passive forcing, and the
subtree's inertia turns that into the angular acceleration the flow wants.
The joint's captured acceleration, relative to its parent and with its
twist left out (the flow cannot turn a segment about its own bone), is
then compared with that: how much of it lies along the passive direction
and how much across, above a quiet floor; moving with the flow at any
fraction of its rate is being carried by it, and only what exceeds the
flow's rate counts as push.  The joint must also actually be moving.
That is the joint's passivity: a valve that opens over `attack_s` -- so
one noisy or ringing frame cannot open it, which showed as hesitations --
and closes over a quarter of a second.  A held-out arm has a large
passive forcing and no motion, so it is not passive; an arm the shoulder
is pumping accelerates as the forcing says, and is; an arm swung against
gravity is muscle.  What it cannot tell is a flop from a flexion the
muscles assist in the same direction -- a fist brought up to guard while
the upper arm swings back reads as passive -- and that is the limit of a
decomposition made from kinematics alone.

The passivity is a valve, not the thing resisted.  Two things were tried
as the thing resisted before this.  The passive share of the acceleration
itself is rectified for anything hanging the wrong way up -- gravity
pushes a head forward on every frame, so only the forward half of a bob
counts, and resisting that alone is a sustained push that threw the head
back.  The joint's whole relative acceleration, scaled by the valve, fails
differently: the valve moves, and cancelling only the deceleration of a
punch it opened on means the deviation keeps going -- follow-through, the
opposite of what was asked.  Anything integrated from accelerations
inherits the valve's history.

The response
------------
What resists is a pull toward a hold posture, by an amount the valve
sets.  The hold posture is a latch: it follows the capture over `leak_s`
while the valve is shut and freezes as it opens, drifting only over
`hold_s`.  So a passive swing is clipped where it began.  (A running
average of the pose was tried as the hold first; pulling toward an
average is a low-pass filter, and that is exactly what it looked like.)

    resist   how far toward the hold posture a fully passive joint is
             pulled: the pull is resist times the passivity, capped at
             one, so at 1 a wholly passive swing is pinned and a joint the
             valve finds a tenth passive is pulled a tenth of the way.
    brake    viscous damping on the output's own velocity relative to
             the parent, times the passivity.  Off by default: it reads
             as smoothing, not holding.

The pull is a blend, not a spring: a spring against a leak saturated --
at a tenth passivity it already pulled more than halfway, so every
flicker of the valve was a full hold, which read as hesitation, and a
hold lagging a fast move yanked the joint after its stop.  A deviation
from the capture follows the blend through a critically damped tracker
with time constant `leak_s`, which is how fast a hold engages and
releases; when the valve closes the joint is back on the performer in a
few of those.  At zero gain the output is the input.  The output is the
capture composed with the deviation, joint by joint; the root is
untouched.

The `legs` property scales the legs and starts at 0: with the root still
following the capture, resisting the legs' swing makes the feet skate.

Signals
-------
Every acceleration here is differenced from motion capture and low-passed
at `dynamics_cutoff`; the passive share is only as good as that and the
segment mass model (the same one smpl_ragdoll's body is built from).
`passive_floor` is the acceleration below which a joint reads as quiet
rather than passive, and `quiet_speed` the relative speed below which it
is standing rather than carried, so stillness is not braked and the
ringing after an abrupt stop does not open the valve.
"""
import math
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter

from dpg_system.node import Node
from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions
from dpg_system.smpl_ragdoll import (SMPLRagdollNode, JOINT_GROUPS, ACTIVE_JOINT_COUNT,
                                     ACTIVE_NAME_TO_SMPL_INDEX)
from dpg_system.body_defs import JointTranslator

GRAVITY = np.array([0.0, -9.81, 0.0])       # the processor's internal frame is Y-up
SEG_RADIUS_DEFAULT = 0.04


class _LowPass:
    """Second-order Butterworth low-pass, streaming, vectorised over
    channels (transposed direct form II)."""

    def __init__(self, cutoff_hz, rate_hz, shape):
        self.shape = shape
        self.set(cutoff_hz, rate_hz)
        self.s1 = np.zeros(shape); self.s2 = np.zeros(shape)
        self.primed = False

    def set(self, cutoff_hz, rate_hz):
        fc = min(max(float(cutoff_hz), 0.1), 0.45 * float(rate_hz))
        b, a = butter(2, fc, fs=float(rate_hz))
        self.b0, self.b1, self.b2 = float(b[0]), float(b[1]), float(b[2])
        self.a1, self.a2 = float(a[1]), float(a[2])
        self.key = (fc, float(rate_hz))

    def reset(self):
        self.s1[:] = 0.0; self.s2[:] = 0.0; self.primed = False

    def push(self, x):
        x = np.asarray(x, dtype=float)
        if not self.primed:
            # start at the first value, not at zero: a step from zero is a
            # transient the differencing downstream turns into a spike
            g = (1.0 + self.a1 + self.a2)
            self.s1 = x * (self.b1 + self.b2 - (self.a1 + self.a2) * (self.b0 + self.b1 + self.b2) / g)
            self.s2 = x * (self.b2 - self.a2 * (self.b0 + self.b1 + self.b2) / g)
            self.primed = True
        y = self.b0 * x + self.s1
        self.s1 = self.b1 * x - self.a1 * y + self.s2
        self.s2 = self.b2 * x - self.a2 * y
        return y


class ResistParams:
    def __init__(self):
        self.dt = 1.0 / 60.0
        self.resist = 1.0            # pull toward the hold posture: this x passivity, capped at 1
        self.brake = 0.0             # viscous damping on the output's relative velocity: this x damp_c0, x passivity (a low-pass look; off by default)
        self.damp_c0 = 20.0          # 1/s behind brake 1
        self.hold_s = 10.0           # s, how fast the latched hold posture drifts while a joint is held
        self.leak_s = 0.05           # s, the tracker behind the pull: how fast a hold engages and releases
        self.dynamics_cutoff = 8.0   # Hz, low-pass on the differenced kinematics
        self.passive_floor = 2.0     # rad/s^2, below this a joint is quiet, not passive
        self.quiet_speed = 0.5       # rad/s, relative speed below which a joint is standing, not carried
        self.attack_s = 0.12         # s, how fast the valve can open (0.08 opened it 0.13 times a second per joint on contemporary dance, 0.15 almost never)
        self.max_deviation = 1.5     # rad, hard ceiling on the deviation


class ResistFilter:
    """The body model and the per-frame step.  Poses are SMPL axis-angle
    (24, 3) in the processor's internal Y-up frame, translation in metres."""

    def __init__(self, processor):
        from dpg_system.smpl_bullet import SEG_RADIUS
        hierarchy = list(processor._get_hierarchy())
        self.parents = hierarchy[:24]
        offsets = np.asarray(processor.skeleton_offsets)
        self.offsets = offsets[:24].copy()
        seg_mass = np.maximum(np.asarray(processor._seg_mass, dtype=float)[:24], 0.05)
        seg_len = np.asarray(processor._seg_length, dtype=float)[:24]
        children = {j: [] for j in range(24)}
        for j in range(1, 24):
            children[self.parents[j]].append(j)
        tips = {}
        for v in range(24, min(30, offsets.shape[0])):
            if v < len(hierarchy):
                tips.setdefault(hierarchy[v], []).append(v)
        # each segment: centre of mass in its own frame, rod inertia about it
        self.mass = seg_mass
        self.com_local = np.zeros((24, 3))
        self.inertia_local = np.zeros((24, 3, 3))
        for j in range(24):
            kids = children[j]
            d = None
            if kids:
                end = np.mean([offsets[c] for c in kids], axis=0)
                n = np.linalg.norm(end)
                if n > 1e-6:
                    d, L = end / n, float(n)
            if d is None:
                for v in tips.get(j, []):
                    if v in (24, 25, 26, 27):
                        e = offsets[v]; n = np.linalg.norm(e)
                        if n > 1e-6:
                            d, L = e / n, float(max(n, 0.03)); break
            if d is None:
                e = offsets[j] if j > 0 else np.array([0.0, 1.0, 0.0])
                n = np.linalg.norm(e)
                d = e / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
                L = float(max(seg_len[j], 0.05))
            r = SEG_RADIUS.get(j, SEG_RADIUS_DEFAULT)
            m = float(seg_mass[j])
            self.com_local[j] = d * (0.5 * L)
            i_ax = 0.5 * m * r * r
            i_pe = m * (L * L / 12.0 + r * r / 4.0)
            # rotate diag(i_pe, i_pe, i_ax) so its axis lies along the bone
            z = np.array([0.0, 0.0, 1.0])
            v = np.cross(z, d); s = np.linalg.norm(v); c = float(np.dot(z, d))
            if s < 1e-9:
                rot = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
            else:
                rot = R.from_rotvec(v / s * math.atan2(s, c)).as_matrix()
            self.inertia_local[j] = rot @ np.diag([i_pe, i_pe, i_ax]) @ rot.T
        # subtree membership
        self.sub = np.zeros((24, 24), dtype=bool)
        for k in range(24):
            j = k
            while j >= 0:
                self.sub[j, k] = True
                j = self.parents[j] if j > 0 else -1
        self.joints = list(range(1, 22))
        self.rate = None
        self.reset()

    def reset(self):
        self.prev_G = None
        self.prev_c = None
        self.prev_p = None
        self.prev_w = None
        self.prev_L = None
        self.prev_pdot = None
        self.prev_wrel = None
        self.prev_wf = None
        self.filters = None
        self.dev = np.zeros((24, 3))       # deviation, parent frame
        self.dev_rate = np.zeros((24, 3))
        self.hold = None                   # hold posture per joint, scipy Rotation (24,)
        self.passivity = np.zeros(24)      # recent passive share of motion, 0..1
        self.last_passive_acc = np.zeros((24, 3))
        self.last_arel = np.zeros((24, 3))         # relative angular acceleration, world (diagnostics)
        self.last_passive_rate = np.zeros((24, 3)) # the acceleration the flow wants, world (diagnostics)
        self.last_share = np.zeros(24)

    def _ensure_filters(self, p_):
        rate = 1.0 / max(p_.dt, 1e-4)
        key = (float(p_.dynamics_cutoff), rate)
        if self.filters is None or self.filters['key'] != key:
            fc = p_.dynamics_cutoff
            self.filters = {
                'key': key,
                'w': _LowPass(fc, rate, (24, 3)),       # segment angular velocity, world
                'v': _LowPass(fc, rate, (24, 3)),       # segment com velocity
                'pdot': _LowPass(fc, rate, (24, 3)),    # joint point velocity
                'L': _LowPass(fc, rate, (24, 3)),       # subtree angular momentum about the joint
                'a': _LowPass(fc, rate, (24, 3)),       # joint point acceleration
                'wdot': _LowPass(fc, rate, (24, 3)),    # segment angular acceleration, world
                'wrel': _LowPass(fc, rate, (24, 3)),    # relative angular velocity, world
                'arel': _LowPass(fc, rate, (24, 3)),    # relative angular acceleration, world
                'Ldot': _LowPass(fc, rate, (24, 3)),
            }

    def fk(self, aa, trans):
        G = np.zeros((24, 3, 3)); p = np.zeros((24, 3))
        Rl = R.from_rotvec(aa[:24]).as_matrix()
        G[0] = Rl[0]; p[0] = trans
        for j in range(1, 24):
            par = self.parents[j]
            G[j] = G[par] @ Rl[j]
            p[j] = p[par] + G[par] @ self.offsets[j]
        c = p + np.einsum('jab,jb->ja', G, self.com_local)
        return G, p, c

    def step(self, aa, trans, gains, p_):
        """One frame.  gains: per-joint multiplier on resist and brake (24,).
        Returns the output pose (24, 3)."""
        self._ensure_filters(p_)
        dt = p_.dt
        aa = np.asarray(aa, dtype=float).reshape(-1, 3)[:24]
        G, p, c = self.fk(aa, np.asarray(trans, dtype=float).reshape(3))
        f = self.filters
        out = aa.copy()
        Rl = R.from_rotvec(aa)
        if self.hold is None:
            self.hold = Rl
        else:
            # the hold is a latch: it follows the capture over leak_s while
            # the valve is shut and freezes as it opens (drifting only over
            # hold_s), so a passive swing is clipped where it began rather
            # than pulled toward a running average of itself -- an average
            # is a low-pass filter, and that is what it looked like
            held = self.passivity ** 2 / (self.passivity ** 2 + 0.15 ** 2)
            tau = float(p_.leak_s) * (1.0 - held) + float(p_.hold_s) * held
            k = np.clip(dt / np.maximum(tau, dt), 0.0, 1.0)
            rel = (Rl * self.hold.inv()).as_rotvec()
            self.hold = R.from_rotvec(rel * k[:, None]) * self.hold
        if self.prev_G is None:
            self.prev_G, self.prev_c, self.prev_p = G, c, p
            return out
        # -- world kinematics of every segment, differenced and filtered
        w_raw = np.zeros((24, 3))
        for k in range(24):
            w_raw[k] = R.from_matrix(G[k] @ self.prev_G[k].T).as_rotvec() / dt
        # a cut in the capture (a file looping, a seek) is a teleport, not
        # motion: start the kinematics again from here, keep the deviation
        if np.max(np.linalg.norm(w_raw, axis=1)) > 50.0 or np.linalg.norm(p[0] - self.prev_p[0]) > 0.25:
            for lp in f.values():
                if isinstance(lp, _LowPass):
                    lp.reset()
            self.prev_G, self.prev_c, self.prev_p = G, c, p
            self.prev_pdot = None; self.prev_L = None; self.prev_wrel = None; self.prev_wf = None
            return out
        w = f['w'].push(w_raw)
        wdot = f['wdot'].push((w - self.prev_wf) / dt) if self.prev_wf is not None else np.zeros((24, 3))
        v = f['v'].push((c - self.prev_c) / dt)
        pdot = f['pdot'].push((p - self.prev_p) / dt)
        a = f['a'].push((pdot - self.prev_pdot) / dt) if self.prev_pdot is not None else np.zeros((24, 3))
        # -- subtree angular momentum about each joint point, and its inertia
        Iw = np.einsum('jab,jbc,jdc->jad', G, self.inertia_local, G)      # G I G^T
        seg_L = np.einsum('jab,jb->ja', Iw, w) + self.mass[:, None] * np.cross(c, v)
        m_v = self.mass[:, None] * v
        m_c = self.mass[:, None] * c
        L = np.zeros((24, 3)); M = np.zeros(24); C = np.zeros((24, 3)); I = np.zeros((24, 3, 3))
        for j in self.joints:
            S = self.sub[j]
            M[j] = self.mass[S].sum()
            C[j] = m_c[S].sum(0) / M[j]
            L[j] = (seg_L[S].sum(0) - np.cross(p[j], m_v[S].sum(0))
                    - np.cross(m_c[S].sum(0), pdot[j]) + M[j] * np.cross(p[j], pdot[j]))
            r = c[S] - p[j]
            I[j] = Iw[S].sum(0) + np.einsum('k,kab->ab', self.mass[S],
                                            np.einsum('k,ab->kab', np.einsum('ka,ka->k', r, r), np.eye(3))
                                            - np.einsum('ka,kb->kab', r, r))
        Lf = f['L'].push(L)
        Ldot = f['Ldot'].push((Lf - self.prev_L) / dt) if self.prev_L is not None else np.zeros((24, 3))
        # -- relative angular velocity and acceleration, child against parent
        par = np.array(self.parents)
        wrel_raw = w - w[par]
        wrel_raw[0] = 0.0
        wrel = f['wrel'].push(wrel_raw)
        arel = f['arel'].push((wrel - self.prev_wrel) / dt) if self.prev_wrel is not None else np.zeros((24, 3))
        self.last_arel = arel
        # -- the passive forcing and the passive share of the motion
        leak = max(float(p_.leak_s), 0.02)
        floor = max(float(p_.passive_floor), 1e-3)
        for j in self.joints:
            tau_p = np.cross(C[j] - p[j], M[j] * (GRAVITY - a[j]))
            try:
                alpha_p = np.linalg.solve(I[j], tau_p) - wdot[par[j]]   # relative to the parent
            except np.linalg.LinAlgError:
                alpha_p = np.zeros(3)
            self.last_passive_rate[j] = alpha_p
            n_p = float(np.linalg.norm(alpha_p))
            # twist is left out of the comparison: the flow cannot turn a
            # segment about its own bone, so twist is muscle or noise either
            # way, and counting it against the swing halved every valve
            bone = G[j] @ self.com_local[j]; nb = float(np.linalg.norm(bone))
            a_sw = arel[j] - bone * (float(np.dot(arel[j], bone)) / (nb * nb)) if nb > 1e-9 else arel[j]
            if n_p > 1e-9:
                along = float(np.dot(a_sw, alpha_p)) / n_p
                across = float(np.linalg.norm(a_sw - alpha_p * (along / n_p)))
            else:
                along = 0.0; across = float(np.linalg.norm(a_sw))
            n_rel = float(np.linalg.norm(a_sw))
            if along > 0.0:
                # moving with the flow at any fraction of its rate is being
                # carried by it (the muscle let it happen); beyond its rate
                # the excess is push, and only that counts against.  The
                # across component counts against, above the quiet floor.
                f = min(n_p / along, 1.0)
                s = f * along * along / (along * along + across * across + floor * floor)
                # and the joint must actually be moving: the ringing in the
                # differenced accelerations after an abrupt stop looked
                # passive by direction while the joint stood still, and the
                # valve re-opening on it yanked the joint about
                v2 = float(np.dot(wrel[j], wrel[j])); q2 = float(p_.quiet_speed) ** 2
                s *= v2 / (v2 + q2)
            else:
                s = 0.0
            # the valve opens over attack_s (one ringing frame cannot open
            # it) and closes over 0.25 s
            rate = dt / max(float(p_.attack_s), dt) if s > self.passivity[j] else dt / 0.25
            self.passivity[j] += min(rate, 1.0) * (s - self.passivity[j])
            self.last_passive_acc[j] = alpha_p * (min(max(along, 0.0), n_p) / n_p) if n_p > 1e-9 else np.zeros(3)
            self.last_share[j] = s
            g = float(gains[j]) * self.passivity[j]
            Gp = G[par[j]].T                                      # world -> parent frame
            d, dr = self.dev[j], self.dev_rate[j]
            # where the blend puts the joint: lam of the way from the
            # capture to the hold posture, parent frame
            lam = min(p_.resist * g, 1.0)
            d_eq = -lam * (Rl[j] * self.hold[j].inv()).as_rotvec()
            c_s = float(p_.damp_c0) * p_.brake * g
            acc = ((d_eq - d) / (leak * leak) - (2.0 / leak) * dr
                   - c_s * (Gp @ wrel[j] + dr))
            dr = dr + acc * dt
            d = d + dr * dt
            n_d = float(np.linalg.norm(d))
            if n_d > p_.max_deviation:
                d = d * (p_.max_deviation / n_d); dr = dr * 0.0
            self.dev[j], self.dev_rate[j] = d, dr
            if n_d > 1e-9:
                out[j] = (R.from_rotvec(d) * R.from_rotvec(aa[j])).as_rotvec()
        self.prev_G, self.prev_c, self.prev_p = G, c, p
        self.prev_pdot = pdot; self.prev_L = Lf; self.prev_wrel = wrel; self.prev_wf = w
        return out


class SMPLResistNode(Node):
    """Pose in, pose out, in whatever layout arrived (SMPL / SMPL-H axis-angle
    or the 20-joint active quaternions), like smpl_ragdoll."""

    joint_names = SMPLRagdollNode.joint_names

    @staticmethod
    def factory(name, data, args=None):
        return SMPLResistNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.processor = None
        self.filter = None
        self.framerate = 60.0
        self.gender = 'neutral'
        self.betas = np.zeros(10)
        self.gains = np.ones(24)
        self.params = ResistParams()

        self.pose_input = self.add_input('pose', triggers_execution=True)
        self.trans_input = self.add_input('trans')
        self.config_input = self.add_input('config')

        self.pose_output = self.add_output('pose')
        self.smpl_pose_output = self.add_output('smpl_pose')
        self.trans_output = self.add_output('trans')
        self.share_output = self.add_output('passive_share')
        self.deviation_output = self.add_output('deviation')

        self.resist_prop = self.add_property('resist', widget_type='drag_float', default_value=1.0)
        self.brake_prop = self.add_property('brake', widget_type='drag_float', default_value=0.0)
        self.hold_prop = self.add_property('hold_s', widget_type='drag_float', default_value=10.0)
        # visible, because a hidden zero that swallows any resist setting is a trap
        self.legs_prop = self.add_property('legs', widget_type='drag_float', default_value=0.0)
        self.leak_prop = self.add_property('leak_s', widget_type='drag_float', default_value=0.05)
        self.reset_input = self.add_input('reset', widget_type='button', callback=self._reset)
        self.cutoff_prop = self.add_option('dynamics_cutoff', widget_type='drag_float', default_value=8.0)
        self.floor_prop = self.add_option('passive_floor', widget_type='drag_float', default_value=2.0)
        self.quiet_speed_prop = self.add_option('quiet_speed', widget_type='drag_float', default_value=0.5)
        self.attack_prop = self.add_option('attack_s', widget_type='drag_float', default_value=0.12)
        self.damp_prop = self.add_option('damp_c0', widget_type='drag_float', default_value=20.0)
        self.max_dev_prop = self.add_option('max_deviation', widget_type='drag_float', default_value=1.5)
        self.total_mass_prop = self.add_option('total_mass', widget_type='drag_float', default_value=75.0)
        self.up_axis_prop = self.add_option('up_axis', widget_type='combo', default_value='Y')
        self.up_axis_prop.widget.combo_items = ['Y', 'Z']
        self.axis_perm_prop = self.add_option(
            'axis_permutation', widget_type='text_input', default_value='x, z, -y')
        self.quat_format_prop = self.add_option('quat_format', widget_type='combo', default_value='wxyz')
        self.quat_format_prop.widget.combo_items = ['xyzw', 'wxyz']

        # gain <joints...> <value>: per-joint multiplier on resist and brake
        self.message_handlers['gain'] = self._gain_message

    # the ragdoll's layout handling, shared
    _split_pose = staticmethod(SMPLRagdollNode._split_pose)
    _resolve_joints = SMPLRagdollNode._resolve_joints
    _write_joint = SMPLRagdollNode._write_joint
    _as_smpl_axis_angle = SMPLRagdollNode._as_smpl_axis_angle
    _to_array = SMPLRagdollNode._to_array

    def _gain_message(self, message='', args=None):
        args = list(args or [])
        if len(args) < 2:
            print('smpl_resist: usage: gain <joints...> <value>')
            return
        try:
            value = max(float(args[-1]), 0.0)
        except (TypeError, ValueError):
            print(f'smpl_resist: gain: last argument must be a number, got {args[-1]!r}')
            return
        indices, unknown = self._resolve_joints(args[:-1])
        if unknown:
            print(f'smpl_resist: gain: unknown joint or group {unknown}')
        for j in indices:
            if 0 < j < 22:
                self.gains[j] = value

    def _reset(self):
        if self.filter is not None:
            self.filter.reset()

    def _ensure_processor(self, rebuild=False):
        if self.processor is None or rebuild:
            self.processor = SMPLProcessor(
                framerate=self.framerate, betas=self.betas, gender=self.gender,
                total_mass_kg=float(self.total_mass_prop()),
                model_path=os.path.dirname(os.path.abspath(__file__)))
            self.filter = ResistFilter(self.processor)
            print('smpl_resist: body = %s, %g Hz, betas %s'
                  % (self.gender, self.framerate, 'supplied' if np.any(self.betas) else 'ZERO (no config?)'))

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
                    self.framerate = fr; changed = True
                break
        if 'gender' in cfg and str(cfg['gender']) != self.gender:
            self.gender = str(cfg['gender']); changed = True
        if 'betas' in cfg:
            b = self._to_array(cfg['betas'])
            if self.betas is None or not np.array_equal(self.betas, b):
                self.betas = b; changed = True
        if changed:
            self._ensure_processor(rebuild=True)

    def execute(self):
        self._handle_config()
        self._ensure_processor()
        if not self.pose_input.fresh_input:
            return
        raw = self.pose_input()
        split = self._split_pose(raw)
        trans_raw = self.trans_input()
        trans = np.zeros(3) if trans_raw is None else np.asarray(self._to_array(trans_raw), dtype=float).reshape(-1)
        root_trans = np.zeros(3); root_trans[:min(3, trans.size)] = trans[:3]
        if split is None:
            print('smpl_resist: unrecognised pose layout, passing through')
            self.pose_output.send(raw)
            self.trans_output.send(trans_raw if trans_raw is not None else root_trans)
            return
        out_pose, F, n_joints, C = split
        is_active = (n_joints == ACTIVE_JOINT_COUNT)
        if is_active:
            work = np.zeros((F, 24, C))
            if C == 4:
                work[:, :, 0 if self.quat_format_prop() == 'wxyz' else 3] = 1.0
            for f in range(F):
                work[f, :22] = JointTranslator.translate_from_bmolab_active_to_smpl(out_pose[f])
            work_joints = 24
        else:
            work = out_pose; work_joints = n_joints

        dt = 1.0 / max(self.framerate, 1.0)
        options = SMPLProcessingOptions(
            input_type='quat' if C == 4 else 'axis_angle',
            input_up_axis=self.up_axis_prop(), axis_permutation=self.axis_perm_prop(),
            quat_format=self.quat_format_prop(), dt=dt)
        p = self.params
        p.dt = dt
        p.resist = float(self.resist_prop()); p.brake = float(self.brake_prop())
        p.leak_s = float(self.leak_prop()); p.hold_s = float(self.hold_prop())
        p.dynamics_cutoff = float(self.cutoff_prop())
        p.passive_floor = float(self.floor_prop()); p.max_deviation = float(self.max_dev_prop())
        p.attack_s = float(self.attack_prop()); p.quiet_speed = float(self.quiet_speed_prop())
        p.damp_c0 = float(self.damp_prop())
        gains = self.gains.copy()
        gains[JOINT_GROUPS['legs']] *= max(float(self.legs_prop()), 0.0)

        proc = self.processor
        for f in range(F):
            frame = work[f:f + 1, :24].copy()
            try:
                t_int, aa_int, _q = proc._prepare_trans_and_pose(frame, root_trans.reshape(1, 3), options)
                out = self.filter.step(aa_int[0], t_int[0], gains, p)
            except Exception as e:
                print(f'smpl_resist: failed ({e}); passing through')
                self.pose_output.send(raw)
                self.trans_output.send(trans_raw if trans_raw is not None else root_trans)
                return
            # non-root local rotations are frame-independent: written straight back
            for j in range(1, 22):
                self._write_joint(work, f, j, work_joints, C, out[j])
        if is_active:
            for f in range(F):
                back = JointTranslator.translate_from_smpl_to_bmolab_active(work[f])
                out_pose[f] = back[:ACTIVE_JOINT_COUNT]
        shaped = out_pose.reshape(np.shape(raw)) if np.ndim(raw) != 3 else out_pose
        self.pose_output.send(shaped)
        self.smpl_pose_output.send(self._as_smpl_axis_angle(work, F, C))
        self.trans_output.send(trans_raw if trans_raw is not None else root_trans)
        self.share_output.send(self.filter.last_share[:22].copy())
        self.deviation_output.send(np.linalg.norm(self.filter.dev[:22], axis=1))
