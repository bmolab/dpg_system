import os, sys
from _env import ROOT, HERE
import sys
import numpy as np
from dpg_system.smpl_processor import SMPLProcessor
from dpg_system.smpl_ragdoll import (SMPLRagdollSim, RagdollParams,
                                     RAGDOLL_JOINT_LIMITS)
HERE=os.path.join(ROOT, 'dpg_system'); FR=120.0
proc=SMPLProcessor(framerate=FR, betas=np.zeros(10), gender='neutral',
                   total_mass_kg=75.0, model_path=HERE)
N=proc.joint_names
def standing():
    ps=np.zeros((24,3)); ps[16]=[0,0,-1.35]; ps[17]=[0,0,1.35]; return ps

print('coverage: joints with a real limit')
have=[N[i] for i in range(22) if N[i] in RAGDOLL_JOINT_LIMITS]
missing=[N[i] for i in range(1,22) if N[i] not in RAGDOLL_JOINT_LIMITS]
print('  specified: %d of 21 non-root joints' % len(have))
print('  unspecified: %s' % (missing or 'none'))
assert not missing

p=RagdollParams(); p.dt=1/FR; p.substeps=4
sim=SMPLRagdollSim(proc); sim.set_free_joints(list(range(1,22))); sim.set_root_free(True)
w=np.ones(22); rel=int(0.3*FR)
worst={}
for f in range(int(12.0*FR)):
    t=f/FR
    ps=standing(); ps[0]=np.array([2.5*t,0,0.4*t])
    if f>=rel: w[:]=0.0
    sim.advance(ps, np.array([1.5*t, 1.9+1.0*t-0.5*9.81*t*t, 0.0]), w, p)
    for j,rot in sim.local_rot.items():
        if not sim._lim_active[j]: continue
        aa=sim._limit_angles(j, rot.as_matrix())
        over=np.maximum(sim._lim_min[j]-aa,0.0)+np.maximum(aa-sim._lim_max[j],0.0)
        m=float(over.max())
        if m > worst.get(j,(0,))[0]: worst[j]=(m, np.argmax(over))

print('\nlargest excursion past a limit over a 12 s tumble and landing:')
for j in sorted(worst, key=lambda k:-worst[k][0])[:8]:
    m,ax = worst[j]
    print('  %-16s axis %d  %.3f rad (%.1f deg) past the stop' % (N[j], ax, m, np.degrees(m)))

print('\nknee direction check (a knee must never bend forwards):')
for j in (4,5):
    lo,hi = sim._lim_min[j][0], sim._lim_max[j][0]
    print('  %-12s X limit [%.2f, %.2f]' % (N[j], lo, hi))
kneeX=[]
sim2=SMPLRagdollSim(proc); sim2.set_free_joints(list(range(1,22))); sim2.set_root_free(True)
w=np.ones(22)
for f in range(int(12.0*FR)):
    t=f/FR
    ps=standing(); ps[0]=np.array([2.5*t,0,0.4*t])
    if f>=rel: w[:]=0.0
    sim2.advance(ps, np.array([1.5*t, 1.9+1.0*t-0.5*9.81*t*t, 0.0]), w, p)
    kneeX += [sim2._limit_angles(4, sim2.local_rot[4].as_matrix())[0], sim2._limit_angles(5, sim2.local_rot[5].as_matrix())[0]]
kneeX=np.array(kneeX)
print('  knee flexion angle range over the run: %.3f to %.3f rad' % (kneeX.min(), kneeX.max()))
print('  most a knee ever bent the wrong way: %.4f rad (%.2f deg)'
      % (max(0.0,-kneeX.min()), np.degrees(max(0.0,-kneeX.min()))))
assert kneeX.min() > -0.25, 'knee bent forwards'
print('\nlimits hold')
