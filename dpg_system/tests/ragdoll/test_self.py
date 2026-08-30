import os, sys
from _env import ROOT, HERE
import sys
import numpy as np
from dpg_system.smpl_processor import SMPLProcessor
from dpg_system.smpl_ragdoll import (SMPLRagdollSim, RagdollParams,
                                     SELF_CAPSULES, _closest_points_on_segments)
HERE=os.path.join(ROOT, 'dpg_system'); FR=120.0
proc=SMPLProcessor(framerate=FR, betas=np.zeros(10), gender='neutral',
                   total_mass_kg=75.0, model_path=HERE)
MASS=float(proc._seg_mass.sum())
NAMES=[c[0] for c in SELF_CAPSULES]

sim0=SMPLRagdollSim(proc); sim0.set_free_joints(list(range(1,22))); sim0.set_root_free(True)
print('%d capsules, %d pairs tested' % (len(SELF_CAPSULES), len(sim0._self_pairs)))
print('pairs:', ', '.join('%s/%s' % (NAMES[i],NAMES[k]) for i,k in sim0._self_pairs[:6]), '...')

def worst_overlap(sim, world_pos):
    worst = (0.0, '')
    for i,k in sim._self_pairs:
        _,ai,bi,ri = SELF_CAPSULES[i]; _,ak,bk,rk = SELF_CAPSULES[k]
        _,_,d = _closest_points_on_segments(world_pos[ai],world_pos[bi],
                                            world_pos[ak],world_pos[bk])
        ov = (ri+rk) - d
        if ov > worst[0]: worst = (ov, '%s/%s' % (NAMES[i],NAMES[k]))
    return worst

# ---------------------------------------------------------------- rest
p=RagdollParams(); p.dt=1/FR
for label, ps in (('rest (T-pose)', np.zeros((24,3))),
                  ('arms at sides', None), ('seated', None)):
    ps2 = np.zeros((24,3))
    if label=='arms at sides': ps2[16]=[0,0,-1.45]; ps2[17]=[0,0,1.45]
    if label=='seated':
        ps2[1]=ps2[2]=[-1.4,0,0]; ps2[4]=ps2[5]=[1.6,0,0]
        ps2[16]=[0,0,-1.3]; ps2[17]=[0,0,1.3]
    wp,_ = sim0._full_fk(ps2 if label!='rest (T-pose)' else np.zeros((24,3)), np.array([0.,1.,0.]))
    ov,nm = worst_overlap(sim0, wp)
    print('  %-16s worst overlap %+.4f m  %s' % (label, ov, nm if ov>0 else '(all clear)'))
    assert ov <= 0.0, 'colliders overlap in a normal pose: ' + nm

# ---------------------------------------------------------------- ragdoll
print('\nfull ragdoll drops: worst self-overlap reached during 10 s')
def drop(seed, self_collision):
    p=RagdollParams(); p.dt=1/FR; p.substeps=4; p.self_collision=self_collision
    sim=SMPLRagdollSim(proc); sim.set_free_joints(list(range(1,22))); sim.set_root_free(True)
    rng=np.random.default_rng(seed); w=np.ones(22); rel=int(0.3*FR)
    spin=rng.normal(0,1.5,3); v0=rng.normal(0,1.0,3); v0[1]=abs(v0[1])
    worst=(0.0,''); hs=[]
    for f in range(int(10.0*FR)):
        t=f/FR
        ps=np.zeros((24,3)); ps[16]=[0,0,-1.2]; ps[17]=[0,0,1.2]; ps[0]=spin*t
        if f>=rel: w[:]=0.0
        sim.advance(ps, np.array([v0[0]*t, 1.6+v0[1]*t-0.5*9.81*t*t, v0[2]*t]), w, p)
        o = worst_overlap(sim, sim._world_pos_cache)
        if o[0] > worst[0]: worst = o
        hs.append(sim.com[1])
    return worst, hs[-1], np.all(np.isfinite(hs))

print('  seed   OFF: worst overlap        ON: worst overlap       ON final com')
for s in range(5):
    (o0,n0), h0, ok0 = drop(s, False)
    (o1,n1), h1, ok1 = drop(s, True)
    print('  %4d    %+.3f m %-12s   %+.3f m %-12s  %.3f m  %s'
          % (s, o0, n0, o1, n1, h1, 'finite' if ok1 else 'BLEW UP'))
    assert ok1
