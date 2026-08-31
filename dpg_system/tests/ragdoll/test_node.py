import os, sys
from _env import ROOT, HERE
"""Drive SMPLRagdollNode.execute with stubbed widgets."""
import sys
import numpy as np
from dpg_system.smpl_ragdoll import SMPLRagdollNode, RagdollParams

class P:
    def __init__(self, v): self.v = v
    def __call__(self): return self.v
class In(P):
    def __init__(self, v, fresh=True):
        super().__init__(v); self.fresh_input = fresh
class Out:
    def __init__(self): self.last = None
    def send(self, v): self.last = v

class StubNode(SMPLRagdollNode):
    def __init__(self, pose, trans, free='left_arm', weight=0.0):
        self.processor = None; self.sim = None
        self.framerate = 60.0; self.gender = 'neutral'
        self.betas = np.zeros(10); self.total_mass = 75.0
        self.free_indices = []; self.weights = np.zeros(22)
        self.weight_targets = np.zeros(22); self.params = RagdollParams()
        self.pose_input = In(pose); self.trans_input = In(trans)
        self.config_input = In(None, fresh=False); self.weights_input = In(None, fresh=False)
        self.pose_output = Out(); self.weights_output = Out(); self.torque_output = Out()
        self.trans_output = Out(); self.smpl_pose_output = Out()
        self.support_output = Out(); self.contact_output=Out()
        self.free_joints_prop = P(free); self.weight_prop = P(weight)
        self.ramp_prop = P(120.0)
        self.gravity_prop = P(1.0); self.transport_prop = P(1.0)
        self.damping_prop = P(1.5); self.limit_stiffness_prop = P(1.0); self.stop_softness_prop=P(0.087)
        self.drag_prop = P(0.15); self.contact_damp_boost_prop=P(0.0)
        self.floor_enable_prop = P(True); self.self_collision_prop = P(True)
        self.self_depth_prop = P(0.04); self.self_max_g_prop = P(2.0); self.floor_height_prop = P(0.0); self.floor_auto_prop=P(True); self.floor_tau_prop=P(2.0); self.floor_rate_prop=P(0.05)
        self.friction_prop=P(0.8); self.contact_sense_prop=P(True); self.contact_depth_prop = P(0.01)
        self.contact_damping_prop=P(4.0); self.partial_damping_prop=P(0.5); self.spring_rate_prop=P(60.0); self.gravity_comp_prop=P(1.0); self.blend_soft_prop=P(180.0); self.blend_firm_prop=P(1.0); self.slip_velocity_prop = P(0.02)
        self.max_penetration_prop = P(0.05); self.max_contact_g_prop = P(50.0)
        self.max_point_g_prop = P(10.0); self.max_point_accel_prop = P(300.0); self.recovery_speed_prop=P(0.1)
        self.damping_speed_prop = P(2.0)
        self.passivity_prop = P(True); self.passivity_rate_prop=P(0.0); self.passivity_rate_contact_prop=P(0.25)
        self.passivity_deadband_prop = P(0.01)
        self.auto_release_prop = P(False); self.auto_release_delay_prop = P(0.15)
        self._unsupported_time = 0.0; self.energy_output = Out()
        self.contact_force_output = Out()
        self.kp_prop = P(120.0); self.kd_prop = P(12.0)
        self.substeps_prop = P(4); self.engine_prop=P('bullet'); self.motor_strength_prop=P(1.0); self.motor_kp_prop=P(0.6); self.motor_kd_prop=P(0.3); self.root_spring_prop=P(40.0); self.substep_rate_prop = P(240.0); self.locked_stiffness_prop = P(1.0)
        self.pivot_smoothing_prop = P(0.25); self.max_ang_vel_prop = P(40.0)
        self.root_seed_smoothing_prop = P(0.3)
        self.total_mass_prop = P(75.0)
        self.up_axis_prop = P('Y'); self.axis_perm_prop = P('')
        self.quat_format_prop = P('wxyz')
        self._parse_free_joints(free)

print('--- axis-angle 24x3, free left arm, weight 0 ---')
pose = np.zeros((24, 3))
n = StubNode(pose.copy(), np.zeros(3))
for f in range(120):
    n.pose_input.v = pose.copy()
    n.trans_input.v = np.array([0.3 * np.sin(2 * np.pi * 1.5 * f / 60.0), 0.0, 0.0])
    n.execute()
out = n.pose_output.last
print('out shape', np.shape(out), 'finite', bool(np.all(np.isfinite(out))))
changed = [j for j in range(24) if np.linalg.norm(np.asarray(out)[j] - pose[j]) > 1e-9]
print('joints changed:', changed, '(expected [16, 18, 20])')
assert changed == [16, 18, 20], changed
print('weights out:', np.round(n.weights_output.last[[16, 18, 20, 17]], 3))

print('\n--- flattened 66, shape preserved ---')
flat = np.zeros(66)
n2 = StubNode(flat.copy(), np.zeros(3))
for f in range(30):
    n2.pose_input.v = flat.copy(); n2.execute()
o2 = n2.pose_output.last
print('out shape', np.shape(o2), '(expected (66,))  finite', bool(np.all(np.isfinite(o2))))
assert np.shape(o2) == (66,)

print('\n--- SMPL-H 156 flattened (52 joints) ---')
sh = np.zeros(156)
n3 = StubNode(sh.copy(), np.zeros(3))
for f in range(30):
    n3.pose_input.v = sh.copy(); n3.execute()
o3 = n3.pose_output.last
print('out shape', np.shape(o3), '(expected (156,))  finite', bool(np.all(np.isfinite(o3))))
assert np.shape(o3) == (156,)
o3r = np.asarray(o3).reshape(52, 3)
print('body joints changed:', [j for j in range(52) if np.linalg.norm(o3r[j]) > 1e-9])

print('\n--- quats 24x4 wxyz ---')
q = np.zeros((24, 4)); q[:, 0] = 1.0
n4 = StubNode(q.copy(), np.zeros(3))
for f in range(60):
    n4.pose_input.v = q.copy(); n4.execute()
o4 = np.asarray(n4.pose_output.last)
print('out shape', o4.shape, 'finite', bool(np.all(np.isfinite(o4))))
print('norms of changed quats:', np.round([np.linalg.norm(o4[j]) for j in (16, 18, 20)], 5))
assert np.allclose([np.linalg.norm(o4[j]) for j in range(24)], 1.0, atol=1e-6)

print('\n--- release: whole body prescribed, then let go ---')
n5 = StubNode(pose.copy(), np.zeros(3), free='all', weight=1.0)
n5.weight_prop = P(1.0); n5._apply_weight_immediately()
heights = []
for f in range(200):
    t = f / 60.0
    ps = np.zeros((24, 3))
    ps[16] = np.array([0.0, 0.0, -0.8]); ps[17] = np.array([0.0, 0.0, 0.8])
    ps[4] = np.array([-0.5, 0.0, 0.0]); ps[5] = np.array([-0.5, 0.0, 0.0])
    n5.pose_input.v = ps
    n5.trans_input.v = np.array([0.0, max(0.0, 1.2 * np.sin(np.pi * t / 1.0)), 0.0])
    if f == 60:
        n5._release()
    n5.execute()
    heights.append(float(np.linalg.norm(np.asarray(n5.pose_output.last) - ps)))
print('pose divergence from capture at f59/61/70/100/199: %s'
      % np.round([heights[i] for i in (59, 61, 70, 100, 199)], 4))
print('all finite:', bool(np.all(np.isfinite(heights))))
print('weights after release:', round(float(n5.weights[16]), 4))

print("\n--- root released mid-leap, checked in the caller's own frame ---")

def leap(free, perm, up, up_axis_idx, spin, frames=200):
    n6 = StubNode(np.zeros((24, 3)), np.zeros(3), free=free, weight=1.0)
    n6.axis_perm_prop = P(perm); n6.up_axis_prop = P(up)
    n6.floor_enable_prop = P(False)   # this checks free fall, not landing
    n6.weight_prop = P(1.0); n6._apply_weight_immediately()
    pos, com = [], []
    for f in range(frames):
        t = f / 60.0
        ps = np.zeros((24, 3))
        ps[0] = np.array([spin * t, 0.0, 0.0])
        ps[16] = [0, 0, -0.9]; ps[17] = [0, 0, 0.9]
        ps[4] = ps[5] = [-0.7, 0, 0]
        tr = np.zeros(3)
        tr[up_axis_idx] = 1.7 * t - 0.5 * 9.81 * t * t + 1.1
        tr[0] = 1.4 * t
        n6.pose_input.v = ps; n6.trans_input.v = tr
        if f == 60:
            n6._release()
        n6.execute()
        pos.append(np.asarray(n6.trans_output.last).copy())
        com.append(n6.sim.com.copy())
    return np.array(pos), np.array(com)

def accel(a):
    return (np.diff(np.diff(a[80:], axis=0), axis=0) * 3600.0).mean(axis=0)

# No spin: root and centre of mass move together, so the emitted trans is a
# clean parabola -- this checks the physics and the frame conversion at once.
for label, perm, up, idx in (('Y-up, no permutation', '', 'Y', 1),
                             ('Z-up via permutation', 'x, z, -y', 'Y', 2)):
    # Root free, joints still driven and static: root and centre of mass move
    # together, so the emitted trans is a clean parabola.
    pos, _ = leap('root', perm, up, idx, spin=0.0)
    a = accel(pos)
    print('  %-22s no spin: root acceleration %s m/s^2' % (label, np.round(a, 4)))
    assert abs(a[idx] + 9.81) < 2e-2, a
    assert np.all(np.abs(np.delete(a, idx)) < 2e-2), a
    assert np.all(np.isfinite(pos))

# Somersaulting: the root orbits the centre of mass, so only the centre of
# mass is ballistic. Both are correct; they are different points.
pos, com = leap('everything', '', 'Y', 1, spin=1.8)   # full ragdoll, spinning
ar, ac = accel(pos), accel(com)
print('  somersaulting: centre of mass %s  vs root %s m/s^2'
      % (np.round(ac, 4), np.round(ar, 3)))
print('    (the root is not ballistic while the body spins about its centre of mass -- expected)')
assert abs(ac[1] + 9.81) < 5e-2 and np.all(np.abs(np.delete(ac, 1)) < 5e-2), ac
print('  gravity survives the frame round trip on the correct axis, and only that axis.')

print('\nnode path OK')
