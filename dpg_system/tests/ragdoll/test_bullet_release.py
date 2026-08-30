import os, sys
from _env import ROOT, HERE
"""The handoff: a body released in mid-leap must keep its momentum."""
import sys, numpy as np
from scipy.spatial.transform import Rotation as R
from dpg_system.smpl_processor import SMPLProcessor
from dpg_system.smpl_ragdoll import RagdollParams
from dpg_system.smpl_bullet import BulletRagdollSim
HERE=os.path.join(ROOT, 'dpg_system'); FR=120.0
proc=SMPLProcessor(framerate=FR, betas=np.zeros(10), gender='neutral', total_mass_kg=75.0, model_path=HERE)
p_=RagdollParams(); p_.dt=1/FR; p_.floor_enable=False
V0=np.array([1.4,3.2,0.35]); H0=6.0; SPIN=2.4
def cap_trans(t): return np.array([V0[0]*t, V0[1]*t-0.5*9.81*t*t+H0, V0[2]*t])
SWING=False
def cap_pose(t):
    ps=np.zeros((24,3)); ps[0]=np.array([SPIN*t,0,0]); ps[16]=[0,0,-1.0]; ps[17]=[0,0,1.0]; ps[4]=ps[5]=[-0.8,0,0]
    if SWING: ps[16]=[0,0,-1.0+0.8*np.sin(2*np.pi*1.5*t)]; ps[17]=[0,0,1.0-0.8*np.sin(2*np.pi*1.5*t)]
    return ps
print('captured leap, exactly ballistic with a %.1f rad/s somersault; released at 0.55 s' % SPIN)
for label, joints, swing in (('root only released', [], False), ('root released, arms swinging (motored)', [], True), ('everything released', list(range(1,22)), False)):
    SWING=swing
    sim=BulletRagdollSim(proc); sim.set_free_joints(joints); sim.set_root_free(True)
    w=np.ones(22); rel=int(0.55*FR); errs=[]; spins=[]
    for f in range(int(1.2*FR)):
        t=f/FR
        if f>=rel:
            w[0]=0.0
            for j in joints: w[j]=0.0
        res,rr,tr=sim.advance(cap_pose(t), cap_trans(t), w, p_)
        if f==rel: v_rel=sim.com_vel.copy(); s_rel=np.linalg.norm(sim.root_ang_vel)
        if f>rel: errs.append(np.linalg.norm(tr-cap_trans(t))); spins.append(np.linalg.norm(sim.root_ang_vel))
    true_v=V0.copy(); true_v[1]=V0[1]-9.81*(rel/FR)
    print('  %-22s velocity at release %s (capture %s)  spin %.3f (capture %.3f)'
          % (label, np.round(v_rel,2), np.round(true_v,2), s_rel, SPIN))
    print('  %-22s pelvis vs captured parabola after %.2f s of flight: %.3f m ; spin after: %.2f'
          % ('', len(errs)/FR, errs[-1], spins[-1]))
    if not joints:
        assert errs[-1] < 0.15, errs[-1]
        assert abs(s_rel-SPIN) < 0.15
print('handoff carries momentum')
