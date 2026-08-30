import os, sys
from _env import ROOT, HERE
"""A handstand whose arms are released: the body must stop hanging in the air."""
import sys
import numpy as np
from dpg_system.smpl_ragdoll import SMPLRagdollNode, RagdollParams

class P:
    def __init__(self,v): self.v=v
    def __call__(self): return self.v
class In(P):
    def __init__(self,v,fresh=True): super().__init__(v); self.fresh_input=fresh
class Out:
    def __init__(self): self.last=None
    def send(self,v): self.last=v

class Stub(SMPLRagdollNode):
    def __init__(self, free, weight, auto):
        self.processor=None; self.sim=None; self.framerate=120.0; self.gender='neutral'
        self.betas=np.zeros(10); self.total_mass=75.0
        self.free_indices=[]; self.root_free=False
        self.weights=np.zeros(22); self.weight_targets=np.zeros(22)
        self.params=RagdollParams()
        self.pose_input=In(np.zeros((24,3))); self.trans_input=In(np.zeros(3))
        self.config_input=In(None,fresh=False); self.weights_input=In(None,fresh=False)
        for n in ('pose_output','smpl_pose_output','trans_output','weights_output',
                  'torque_output','contact_force_output','energy_output','support_output'):
            setattr(self,n,Out())
        self.free_joints_prop=P(free); self.weight_prop=P(weight); self.ramp_prop=P(200.0)
        self.auto_release_prop=P(auto); self.auto_release_delay_prop=P(0.15)
        self._unsupported_time=0.0
        self.gravity_prop=P(1.0); self.transport_prop=P(1.0)
        self.damping_prop=P(1.5); self.limit_stiffness_prop=P(1.0); self.stop_softness_prop=P(0.087)
        self.drag_prop=P(0.15); self.contact_damp_boost_prop=P(0.0)
        self.passivity_prop=P(True); self.passivity_rate_prop=P(0.0); self.passivity_rate_contact_prop=P(0.25)
        self.passivity_deadband_prop=P(0.01)
        self.floor_enable_prop=P(True); self.self_collision_prop = P(True)
        self.self_depth_prop = P(0.04); self.self_max_g_prop = P(2.0); self.floor_height_prop=P(0.0); self.floor_auto_prop=P(True); self.floor_tau_prop=P(2.0); self.floor_rate_prop=P(0.05)
        self.friction_prop=P(0.8); self.contact_depth_prop=P(0.01)
        self.contact_damping_prop=P(4.0); self.partial_damping_prop=P(0.5); self.spring_rate_prop=P(60.0); self.gravity_comp_prop=P(1.0); self.blend_soft_prop=P(180.0); self.blend_firm_prop=P(1.0); self.slip_velocity_prop=P(0.02)
        self.max_penetration_prop=P(0.05); self.max_contact_g_prop=P(50.0)
        self.max_point_g_prop=P(10.0); self.max_point_accel_prop=P(300.0); self.recovery_speed_prop=P(0.1)
        self.damping_speed_prop=P(2.0); self.kp_prop=P(120.0); self.kd_prop=P(12.0)
        self.substeps_prop=P(4); self.engine_prop=P('bullet'); self.motor_strength_prop=P(1.0); self.motor_kp_prop=P(0.6); self.motor_kd_prop=P(0.3); self.root_spring_prop=P(40.0); self.substep_rate_prop=P(240.0); self.locked_stiffness_prop=P(1.0)
        self.pivot_smoothing_prop=P(0.25); self.max_ang_vel_prop=P(40.0)
        self.root_seed_smoothing_prop=P(0.3); self.total_mass_prop=P(75.0)
        self.up_axis_prop=P('Y'); self.axis_perm_prop=P(''); self.quat_format_prop=P('wxyz')
        self._parse_free_joints(free)

FR=120.0
def handstand_pose():
    """Inverted: root rotated 180 about X so the body is upside down, arms
    overhead (i.e. toward the floor)."""
    ps=np.zeros((24,3))
    ps[0]=np.array([np.pi,0,0])
    ps[16]=[0,0,1.45]; ps[17]=[0,0,-1.45]     # arms straight along the body
    return ps

def run(free, auto, label, release_at=1.0, secs=3.5):
    n=Stub(free, 1.0, auto)
    n.weight_prop=P(1.0); n._apply_weight_immediately()
    rows=[]
    for f in range(int(secs*FR)):
        t=f/FR
        n.pose_input.v=handstand_pose()
        n.trans_input.v=np.array([0.0, 1.30, 0.0])   # capture holds it up, level
        if f==int(release_at*FR):
            n._release()          # let the arms go
        n.execute()
        tr=np.asarray(n.trans_output.last)
        rows.append((tr[1], float(n.support_output.last), n.weights[0]))
    r=np.array(rows)
    print('  %-34s' % label)
    for a in (0.9, 1.05, 1.3, 1.8, 2.5, 3.4):
        i=int(a*FR)
        print('     t=%.2f  root height %6.3f   support %.2f   root weight %.2f'
              % (a, r[i,0], r[i,1], r[i,2]))
    return r

print('handstand held by the capture; arms released at t=1.0 s\n')
a = run('arms', False, 'arms free, auto_release OFF')
print()
b = run('arms, root', True, 'arms free + root, auto_release ON')
print('\n  root fell by %.3f m with auto_release on, %.3f m with it off'
      % (b[0,0]-b[-1,0], a[0,0]-a[-1,0]))
assert a[-1,0] > 1.2, 'without auto_release the root should stay up (that is the complaint)'
assert b[-1,0] < 0.9, 'with auto_release the body should fall'
print('\nsupport-loss release works')
