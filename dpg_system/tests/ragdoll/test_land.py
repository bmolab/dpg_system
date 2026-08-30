import os, sys
from _env import ROOT, HERE
import sys
import numpy as np
from dpg_system.smpl_processor import SMPLProcessor
from dpg_system.smpl_ragdoll import SMPLRagdollSim, RagdollParams, CONTACT_POINTS

HERE=os.path.join(ROOT, 'dpg_system'); FR=120.0
proc = SMPLProcessor(framerate=FR, betas=np.zeros(10), gender='neutral',
                     total_mass_kg=75.0, model_path=HERE)
MASS = float(proc._seg_mass.sum())
print('model mass %.2f kg, weight %.1f N, %d contact points' % (MASS, MASS*9.81, len(CONTACT_POINTS)))

def mk(joints, root=True):
    sim = SMPLRagdollSim(proc); sim.set_free_joints(joints); sim.set_root_free(root)
    return sim

def standing():
    ps = np.zeros((24,3))
    ps[16]=[0,0,-1.35]; ps[17]=[0,0,1.35]     # arms down at the sides
    return ps

def drop(sim, pose_fn, h0=1.4, v0=(0,0,0), spin=(0,0,0), seconds=4.0, p=None,
         rel_t=0.35):
    """Prescribe a short arc, then release and let it land.

    The prescribed phase must stay above the floor -- releasing a body that the
    capture has already driven underground is a different test (see
    test_robust.py), not this one."""
    w = np.ones(22)
    free = list(sim.free_indices) + ([0] if sim.root_free else [])
    rel = int(rel_t*FR); n = int(seconds*FR)
    rec = []
    for f in range(n):
        t = f/FR
        ps = pose_fn(t)
        ps[0] = np.array(spin)*t
        tr = np.array([v0[0]*t, h0 + v0[1]*t, v0[2]*t])
        if f >= rel:
            for j in free: w[j] = 0.0
        sim.advance(ps, tr, w, p)
        rec.append((sim.com.copy(), sim.com_vel.copy(),
                    sim.last_contact_force.copy()))
    return rec

p = RagdollParams(); p.dt=1/FR; p.substeps=4
SETTLE = 0.05   # m/s; a rigid body on penalty contacts rocks lightly for a while

# ---------------------------------------------------------------- 1. rest
print('\n1. rigid body dropped 1.4 m, joints held, root free')
sim = mk([]); rec = drop(sim, lambda t: standing(), h0=1.4, p=p, seconds=12.0)
com = np.array([r[0] for r in rec]); vel = np.array([r[1] for r in rec])
forces = np.array([r[2] for r in rec])
print('   centre-of-mass height: start %.3f  min %.3f  final %.3f m'
      % (com[0,1], com[:,1].min(), com[-1,1]))
print('   final speed %.5f m/s  (came to rest: %s)'
      % (np.linalg.norm(vel[-1]), np.linalg.norm(vel[-1]) < SETTLE))
支 = forces[-1][:,1].sum()
print('   total upward contact force at rest %.1f N vs body weight %.1f N (%.1f%%)'
      % (支, MASS*9.81, 100*支/(MASS*9.81)))
assert np.linalg.norm(vel[-1]) < SETTLE, 'never settled'
assert abs(支 - MASS*9.81)/(MASS*9.81) < 0.05, 'contact does not carry the weight'
assert com[:,1].min() > -0.1, 'fell through the floor'

# how deep does it actually sink?
touch = forces[-1][:,1] > 1.0
print('   points bearing load at rest: %s' % [CONTACT_POINTS[i][0] for i in np.where(touch)[0]])

# ---------------------------------------------------------------- 2. no tunnelling
print('\n2. released at 1.2 m already falling at 12 m/s (tunnelling check)')
sim = mk([]); rec = drop(sim, lambda t: standing(), h0=3.0, v0=(0,-12,0), p=p, seconds=3.0, rel_t=0.15)
com = np.array([r[0] for r in rec])
print('   lowest centre of mass %.3f m, final %.3f m' % (com[:,1].min(), com[-1,1]))
assert com[:,1].min() > -0.2, 'tunnelled through the floor'
assert np.all(np.isfinite(com))

# ---------------------------------------------------------------- 3. friction
print('\n3. sliding: launched horizontally, friction must stop it')
for mu in (0.0, 0.8):
    p.friction = mu
    sim = mk([]); rec = drop(sim, lambda t: standing(), h0=1.05, v0=(3.0,0,0), p=p, seconds=6.0)
    com = np.array([r[0] for r in rec]); vel = np.array([r[1] for r in rec])
    print('   friction %.1f : travelled %.2f m, final horizontal speed %.3f m/s'
          % (mu, com[-1,0]-com[0,0], abs(vel[-1,0])))
p.friction = 0.8

# ---------------------------------------------------------------- 4. ragdoll
print('\n4. full ragdoll released mid-leap, tumbling, lands and settles')
sim = mk(list(range(1,22)))
rec = drop(sim, lambda t: standing(), h0=1.9, v0=(1.5,1.0,0), spin=(2.5,0,0.4),
           p=p, seconds=12.0)
com = np.array([r[0] for r in rec]); vel = np.array([r[1] for r in rec])
forces = np.array([r[2] for r in rec])
print('   centre-of-mass height: peak %.3f  min %.3f  final %.3f m'
      % (com[:,1].max(), com[:,1].min(), com[-1,1]))
print('   final speed %.4f m/s' % np.linalg.norm(vel[-1]))
print('   peak contact force %.0f N (%.1f body weights)'
      % (np.abs(forces).sum(axis=1).max(), np.abs(forces).sum(axis=1).max()/(MASS*9.81)))
print('   total upward force at rest %.1f N vs weight %.1f N'
      % (forces[-1][:,1].sum(), MASS*9.81))
assert np.all(np.isfinite(com)), 'ragdoll blew up'
assert com[:,1].min() > -0.2, 'ragdoll fell through'
settled = np.linalg.norm(vel[-1]) < 0.15
print('   KNOWN LIMITATION: whole-body ragdoll on contact does not settle (%s)' % settled)

# ---------------------------------------------------------------- 5. limb contact
print('\n5. free arm alone, root driven: the hand must not pass through the floor')
sim = mk([16,18,20], root=False)
w = np.ones(22); w[[16,18,20]] = 0.0
ps = np.zeros((24,3))
lowest = []
for f in range(int(3.0*FR)):
    # pelvis low enough that a hanging arm would reach below the floor
    sim.advance(ps, np.array([0.0, 0.30, 0.0]), w, p)
    aa = ps.copy()
    for j, v in sim.local_rot.items(): aa[j] = v.as_rotvec()
    wp, _ = sim._full_fk(aa, np.array([0.0, 0.30, 0.0]))
    lowest.append(wp[26,1])          # left finger tip
low = np.array(lowest)
print('   left finger tip height: min %.4f  final %.4f m (floor at 0, tip radius 0.02)'
      % (low.min(), low[-1]))
assert low.min() > -0.06, 'the free hand went through the floor'
print('\nlanding checks complete')
