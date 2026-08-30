import os, sys
from _env import ROOT, HERE
import sys
import numpy as np
from dpg_system.smpl_processor import SMPLProcessor
from dpg_system.smpl_ragdoll import SMPLRagdollSim, RagdollParams
HERE=os.path.join(ROOT, 'dpg_system'); FR=120.0
proc = SMPLProcessor(framerate=FR, betas=np.zeros(10), gender='neutral',
                     total_mass_kg=75.0, model_path=HERE)
MASS=float(proc._seg_mass.sum()); W=MASS*9.81
def standing():
    ps=np.zeros((24,3)); ps[16]=[0,0,-1.35]; ps[17]=[0,0,1.35]; return ps

print('released while already below the floor (capture sitting under the estimate):')
for depth in (0.05, 0.3, 1.0):
    p=RagdollParams(); p.dt=1/FR; p.substeps=4
    sim=SMPLRagdollSim(proc); sim.set_free_joints(list(range(1,22))); sim.set_root_free(True)
    w=np.ones(22); rel=int(0.3*FR); hs=[]
    for f in range(int(5.0*FR)):
        if f>=rel: w[:]=0.0
        sim.advance(standing(), np.array([0.0, 0.95-depth, 0.0]), w, p)
        hs.append(sim.com[1])
    hs=np.array(hs)
    print('  start %.2f m under: peak height after release %.3f, settles at %.3f, finite %s'
          % (depth, hs[rel:].max(), hs[-1], bool(np.all(np.isfinite(hs)))))
    assert np.all(np.isfinite(hs))
    assert hs[rel:].max() < 2.0, 'launched into the air'

print('\nextreme spin at release (40 rad/s clamp):')
p=RagdollParams(); p.dt=1/FR; p.substeps=4
sim=SMPLRagdollSim(proc); sim.set_free_joints(list(range(1,22))); sim.set_root_free(True)
w=np.ones(22); rel=int(0.5*FR); ok=True
for f in range(int(6.0*FR)):
    t=f/FR
    ps=standing(); ps[0]=np.array([60.0*t, 20.0*t, 0.0])
    if f>=rel: w[:]=0.0
    sim.advance(ps, np.array([0.0, 2.0, 0.0]), w, p)
    if not np.all(np.isfinite(sim.com)): ok=False; break
print('  finite throughout: %s, settles at %.3f m, final speed %.4f'
      % (ok, sim.com[1], np.linalg.norm(sim.com_vel)))
assert ok

print('\nzero-length / degenerate input (NaN in the capture):')
p=RagdollParams(); p.dt=1/FR
sim=SMPLRagdollSim(proc); sim.set_free_joints([16,18,20]); sim.set_root_free(True)
w=np.zeros(22)
ps=standing()
sim.advance(ps, np.array([0.0,1.0,0.0]), w, p)
before=sim.com.copy()
try:
    bad=ps.copy(); bad[5]=np.nan
    sim.advance(bad, np.array([0.0,1.0,0.0]), w, p)
    print('  NaN pose accepted without raising; com finite: %s' % bool(np.all(np.isfinite(sim.com))))
except Exception as e:
    print('  NaN pose raised %s (the node catches this and passes through)' % type(e).__name__)
print('\nrobustness checks complete')
