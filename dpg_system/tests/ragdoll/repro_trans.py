import os, sys
from _env import ROOT, HERE
import sys
import numpy as np
from dpg_system.smpl_ragdoll import SMPLRagdollNode, RagdollParams, JOINT_GROUPS

class P:
    def __init__(self,v): self.v=v
    def __call__(self): return self.v
class In(P):
    def __init__(self,v,fresh=True): super().__init__(v); self.fresh_input=fresh
class Out:
    def __init__(self): self.last=None
    def send(self,v): self.last=v

class Stub(SMPLRagdollNode):
    def __init__(self, free, weight=1.0):
        self.processor=None; self.sim=None; self.framerate=60.0; self.gender='neutral'
        self.betas=np.zeros(10); self.total_mass=75.0
        self.free_indices=[]; self.root_free=False; self._unsupported_time=0.0
        self.weights=np.zeros(22); self.weight_targets=np.zeros(22)
        self.params=RagdollParams()
        self.pose_input=In(np.zeros((24,3))); self.trans_input=In(np.zeros(3))
        self.config_input=In(None,fresh=False); self.weights_input=In(None,fresh=False)
        for n in ('pose_output','smpl_pose_output','trans_output','weights_output',
                  'torque_output','contact_force_output','energy_output','support_output','contact_output'):
            setattr(self,n,Out())
        self.free_joints_prop=P(free); self.weight_prop=P(weight); self.ramp_prop=P(120.0)
        self.auto_release_prop=P(False); self.auto_release_delay_prop=P(0.15)
        self.gravity_prop=P(1.0); self.transport_prop=P(1.0)
        self.damping_prop=P(1.5); self.limit_stiffness_prop=P(1.0); self.stop_softness_prop=P(0.087)
        self.drag_prop=P(0.15); self.contact_damp_boost_prop=P(0.0)
        self.passivity_prop=P(True); self.passivity_rate_prop=P(0.0); self.passivity_rate_contact_prop=P(0.25)
        self.passivity_deadband_prop=P(0.01)
        self.floor_enable_prop=P(True); self.self_collision_prop = P(True)
        self.self_depth_prop = P(0.04); self.self_max_g_prop = P(2.0); self.floor_height_prop=P(0.0); self.floor_auto_prop=P(True); self.floor_tau_prop=P(2.0); self.floor_rate_prop=P(0.05)
        self.friction_prop=P(0.8); self.contact_sense_prop=P(True); self.contact_depth_prop=P(0.01)
        self.contact_damping_prop=P(4.0); self.slip_velocity_prop=P(0.02)
        self.max_penetration_prop=P(0.05); self.max_contact_g_prop=P(50.0)
        self.max_point_g_prop=P(10.0); self.max_point_accel_prop=P(300.0); self.recovery_speed_prop=P(0.1)
        self.damping_speed_prop=P(2.0); self.kp_prop=P(120.0); self.kd_prop=P(12.0)
        self.substeps_prop=P(4); self.engine_prop=P('bullet'); self.motor_strength_prop=P(1.0); self.motor_kp_prop=P(0.6); self.motor_kd_prop=P(0.3); self.partial_damping_prop=P(0.5); self.spring_rate_prop=P(60.0); self.gravity_comp_prop=P(1.0); self.blend_soft_prop=P(180.0); self.blend_firm_prop=P(1.0); self.root_spring_prop=P(40.0); self.substep_rate_prop=P(240.0); self.locked_stiffness_prop=P(1.0)
        self.pivot_smoothing_prop=P(0.25); self.max_ang_vel_prop=P(40.0)
        self.root_seed_smoothing_prop=P(0.3); self.total_mass_prop=P(75.0)
        # node defaults, as they appear in the patch
        self.up_axis_prop=P('Y'); self.axis_perm_prop=P('x, z, -y')
        self.quat_format_prop=P('wxyz')
        self._parse_free_joints(free)

print("what each group actually frees:")
for g in ('all', 'everything', 'joints', 'root', 'arms'):
    idx = sorted(set(JOINT_GROUPS[g]))
    print('   %-11s -> %-28s root included: %s'
          % (g, str(idx[:6]) + ('...' if len(idx) > 6 else ''), 0 in idx))

print("\ndriving the node: free_joints = <group>, weight 1.0, release at f60")
for group in ('all', 'joints'):
    n = Stub(group, weight=1.0)
    n.weight_prop = P(1.0); n._apply_weight_immediately()
    rows=[]
    for f in range(200):
        t=f/60.0
        ps=np.zeros((24,3)); ps[16]=[0,0,-1.0]; ps[17]=[0,0,1.0]
        cap=np.array([0.4*t, 0.0, 1.30])         # z-up input: level, never falls
        n.pose_input.v=ps; n.trans_input.v=cap
        if f==60: n._release()
        n.execute()
        rows.append((cap.copy(), np.asarray(n.trans_output.last, dtype=float).copy()))
    print('\n  free_joints = %r   (root_free=%s)' % (group, n.root_free))
    print('     f      captured trans        output trans        differs')
    for f in (59, 61, 80, 120, 199):
        c,o = rows[f]
        print('   %4d   %s   %s   %s'
              % (f, np.round(c,3), np.round(o,3),
                 'yes' if np.abs(c-o).max() > 1e-6 else 'NO -- mirrors the input'))
