import os, sys
from _env import ROOT, HERE
"""Active (Shadow, 20 quaternion) input through the node."""
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from dpg_system.body_defs import JointTranslator as JT
from dpg_system.smpl_ragdoll import (SMPLRagdollNode, RagdollParams,
                                     ACTIVE_JOINT_COUNT, ACTIVE_NAME_TO_SMPL_INDEX)

class P:
    def __init__(self, v): self.v = v
    def __call__(self): return self.v
class In(P):
    def __init__(self, v, fresh=True): super().__init__(v); self.fresh_input = fresh
class Out:
    def __init__(self): self.last = None
    def send(self, v): self.last = v

class Stub(SMPLRagdollNode):
    def __init__(self, pose, free='left_arm', weight=0.0):
        self.processor=None; self.sim=None; self.framerate=60.0; self.gender='neutral'
        self.betas=np.zeros(10); self.total_mass=75.0
        self.free_indices=[]; self.root_free=False
        self.weights=np.zeros(22); self.weight_targets=np.zeros(22)
        self.params=RagdollParams()
        self.pose_input=In(pose); self.trans_input=In(np.zeros(3))
        self.config_input=In(None, fresh=False); self.weights_input=In(None, fresh=False)
        self.pose_output=Out(); self.smpl_pose_output=Out(); self.trans_output=Out()
        self.weights_output=Out(); self.torque_output=Out()
        self.contact_force_output=Out(); self.energy_output=Out()
        self.support_output = Out()
        self.free_joints_prop=P(free); self.weight_prop=P(weight); self.ramp_prop=P(120.0)
        self.gravity_prop=P(1.0); self.transport_prop=P(1.0)
        self.damping_prop=P(1.5); self.limit_stiffness_prop=P(1.0); self.stop_softness_prop=P(0.087)
        self.drag_prop=P(0.15); self.contact_damp_boost_prop=P(0.0)
        self.passivity_prop=P(True); self.passivity_rate_prop=P(0.0); self.passivity_rate_contact_prop=P(0.25)
        self.passivity_deadband_prop=P(0.01)
        self.auto_release_prop=P(False); self.auto_release_delay_prop=P(0.15)
        self._unsupported_time=0.0
        self.floor_enable_prop=P(True); self.self_collision_prop = P(True)
        self.self_depth_prop = P(0.04); self.self_max_g_prop = P(2.0); self.floor_height_prop=P(0.0); self.floor_auto_prop=P(True)
        self.friction_prop=P(0.8); self.contact_depth_prop=P(0.01)
        self.contact_damping_prop=P(4.0); self.partial_damping_prop=P(0.5); self.spring_rate_prop=P(60.0); self.gravity_comp_prop=P(1.0); self.blend_soft_prop=P(180.0); self.blend_firm_prop=P(1.0); self.slip_velocity_prop=P(0.02)
        self.max_penetration_prop=P(0.05); self.max_contact_g_prop=P(50.0)
        self.max_point_g_prop=P(10.0); self.max_point_accel_prop=P(300.0); self.recovery_speed_prop=P(0.1)
        self.damping_speed_prop=P(2.0)
        self.kp_prop=P(120.0); self.kd_prop=P(12.0)
        self.substeps_prop=P(4); self.engine_prop=P('bullet'); self.motor_strength_prop=P(1.0); self.motor_kp_prop=P(0.6); self.motor_kd_prop=P(0.3); self.root_spring_prop=P(40.0); self.substep_rate_prop=P(240.0); self.locked_stiffness_prop=P(1.0)
        self.pivot_smoothing_prop=P(0.25); self.max_ang_vel_prop=P(40.0)
        self.root_seed_smoothing_prop=P(0.3); self.total_mass_prop=P(75.0)
        self.up_axis_prop=P('Y'); self.axis_perm_prop=P(''); self.quat_format_prop=P('wxyz')
        self._parse_free_joints(free)

print('1. format detection')
for arr, expect in ((np.zeros(80), (1,20,4)), (np.zeros(60), (1,20,3)),
                    (np.zeros((20,4)), (1,20,4)), (np.zeros(72), (1,24,3)),
                    (np.zeros(156), (1,52,3)), (np.zeros((24,4)), (1,24,4))):
    r = SMPLRagdollNode._split_pose(arr)
    got = (r[1], r[2], r[3])
    kind = 'active' if r[2]==ACTIVE_JOINT_COUNT else 'smpl'
    print('   %-12s -> %s  %s  %s' % (str(np.shape(arr)), got, kind, 'ok' if got==expect else 'MISMATCH'))
    assert got == expect

print('\n2. active quaternions in, active out + smpl out')
ident = np.zeros((20,4)); ident[:,0]=1.0
n = Stub(ident.copy(), free='left_arm', weight=0.0)
for f in range(120):
    n.pose_input.v = ident.copy(); n.execute()
a_out = np.asarray(n.pose_output.last); s_out = np.asarray(n.smpl_pose_output.last)
print('   pose out %s (active layout preserved), smpl_pose out %s' % (a_out.shape, s_out.shape))
assert a_out.shape == (20,4) and s_out.shape == (24,3)
print('   quaternion norms all unit: %s' % bool(np.allclose(np.linalg.norm(a_out,axis=1),1.0,atol=1e-6)))
moved_active = [i for i in range(20) if np.linalg.norm(a_out[i]-ident[i])>1e-6]
moved_smpl = [j for j in range(24) if np.linalg.norm(s_out[j])>1e-6]
names = {v:k for k,v in JT.bmolab_active_joints.items()}
print('   active joints changed: %s' % [names[i] for i in moved_active])
print('   smpl joints changed:   %s' % moved_smpl)
assert moved_smpl == [16,18,20], moved_smpl

print('\n3. the two outputs agree')
back = JT.translate_from_bmolab_active_to_smpl(a_out)
q = np.roll(back[:22], -1, axis=-1)
q = q/np.linalg.norm(q,axis=1,keepdims=True)
err = np.abs(R.from_quat(q).as_rotvec() - s_out[:22]).max()
print('   max disagreement between pose and smpl_pose: %.2e rad' % err)
assert err < 1e-5

print('\n4. free joints named in the active convention')
n2 = Stub(ident.copy(), free='left_shoulder_blade, base_of_skull, mid_vertebrae')
print('   "left_shoulder_blade, base_of_skull, mid_vertebrae" -> smpl %s' % n2.free_indices)
assert n2.free_indices == [9, 13, 15], n2.free_indices

print('\n5. smpl input still round-trips unchanged')
sp = np.zeros(72)
n3 = Stub(sp.copy(), free='left_arm')
for f in range(30):
    n3.pose_input.v = sp.copy(); n3.execute()
print('   72-float smpl in -> pose out %s, smpl_pose out %s'
      % (np.shape(n3.pose_output.last), np.shape(n3.smpl_pose_output.last)))
assert np.shape(n3.pose_output.last) == (72,)
print('\nactive integration OK')
