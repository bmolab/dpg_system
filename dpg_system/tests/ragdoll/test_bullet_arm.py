import os, sys
from _env import ROOT, HERE
import sys, numpy as np
from dpg_system.smpl_processor import SMPLProcessor
from dpg_system.smpl_ragdoll import RagdollParams
from dpg_system.smpl_bullet import BulletRagdollSim
HERE=os.path.join(ROOT, 'dpg_system'); FR=60.0
proc=SMPLProcessor(framerate=FR, betas=np.zeros(10), gender='neutral', total_mass_kg=75.0, model_path=HERE)
p_=RagdollParams(); p_.dt=1/FR; p_.floor_enable=False
def run(trans_fn, secs=4.0):
    sim=BulletRagdollSim(proc); sim.set_free_joints([16,18,20]); sim.set_root_free(False)
    w=np.ones(22); w[[16,18,20]]=0.0; pose=np.zeros((24,3)); hs=[]
    for f in range(int(secs*FR)):
        res,_,_=sim.advance(pose, trans_fn(f/FR), w, p_)
        aa=pose.copy()
        for j,v in res.items(): aa[j]=v
        hs.append((sim.body.joint_positions()[22][1], np.linalg.norm(res[16])))
        if f in (0,5,15,30,60,120,239): print('      f%-3d hand y %.3f  shoulder rotvec %s  elbow %s' % (f, hs[-1][0], np.round(res[16],2), np.round(res[18],2)))
    return np.array(hs), res[16]
print('left arm free from a T-pose, pelvis driven and still (Bullet):')
h,_=run(lambda t: np.array([0,1.0,0]))
print('   hand height: start %.3f -> final %.3f m (shoulder at ~%.2f; a hanging arm reaches ~%.2f)' % (h[0,0], h[-1,0], 1.48, 1.48-0.64))
print('   settled: last second hand height range %.4f m' % (h[-60:,0].max()-h[-60:,0].min()))
assert h[-1,0] < h[0,0]-0.4 and (h[-60:,0].max()-h[-60:,0].min()) < 0.03
print('pelvis shaken sideways at 2 Hz, +/-15 cm: the arm must swing with it')
h2,r2=run(lambda t: np.array([0.15*np.sin(2*np.pi*2*t),1.0,0]))
print('   shoulder rotation still %s vs shaken %s ; hand height range while shaken %.3f m' % (np.round(run(lambda t: np.array([0,1.0,0]))[1],2), np.round(r2,2), h2[-60:,0].max()-h2[-60:,0].min()))
print('limp arm behaves')
