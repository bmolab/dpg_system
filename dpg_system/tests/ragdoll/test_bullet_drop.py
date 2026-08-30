import os, sys
from _env import ROOT, HERE
import sys, numpy as np
from dpg_system.smpl_processor import SMPLProcessor
from dpg_system.smpl_ragdoll import RagdollParams
from dpg_system.smpl_bullet import BulletRagdollSim
HERE=os.path.join(ROOT, 'dpg_system'); FR=120.0
proc=SMPLProcessor(framerate=FR, betas=np.zeros(10), gender='neutral', total_mass_kg=75.0, model_path=HERE)
p_=RagdollParams(); p_.dt=1/FR
def standing():
    ps=np.zeros((24,3)); ps[16]=[0,0,-1.35]; ps[17]=[0,0,1.35]; return ps
print('whole free body dropped from 1.5 m (Bullet):')
sim=BulletRagdollSim(proc); sim.set_free_joints(list(range(1,22))); sim.set_root_free(True)
w=np.ones(22); hs=[]; import time; t0=time.perf_counter()
for f in range(int(6.0*FR)):
    if f>=int(0.3*FR): w[:]=0.0
    res,rr,tr=sim.advance(standing(), np.array([0.0,1.5,0.0]), w, p_)
    hs.append(tr[1])
ms=(time.perf_counter()-t0)/len(hs)*1000
hs=np.array(hs)
print('   pelvis height: start %.2f  min %.3f  final %.3f m ;  last 1 s range %.4f m' % (hs[0], hs.min(), hs[-1], hs[-120:].max()-hs[-120:].min()))
print('   finite: %s ;  %.2f ms/frame at 120 Hz, all 21 joints + root free' % (bool(np.all(np.isfinite(hs))), ms))
print('   points touching at rest:', [sim.joint_names[i] for i in np.where(np.abs(sim.last_contact_force).sum(axis=1)>1)[0]])
