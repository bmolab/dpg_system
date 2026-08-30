import os, sys
from _env import ROOT, HERE
import sys
import numpy as np
from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions
from dpg_system.smpl_ragdoll import SMPLRagdollSim, RagdollParams

HERE=os.path.join(ROOT, 'dpg_system'); FR=120.0
proc = SMPLProcessor(framerate=FR, betas=np.zeros(10), gender='neutral',
                     total_mass_kg=75.0, model_path=HERE)
p = RagdollParams(); p.dt=1/FR; p.substeps=4
p.floor_enable = False   # this checks flight seeding, not landing

# A captured leap that is itself exactly ballistic, with a somersault.
V0 = np.array([1.4, 3.2, 0.35]); H0 = 1.05; SPIN = 2.4
def cap_trans(t): return np.array([V0[0]*t, V0[1]*t - 0.5*9.81*t*t + H0, V0[2]*t])
def cap_pose(t):
    ps = np.zeros((24,3))
    ps[0] = np.array([SPIN*t, 0.0, 0.0])
    ps[16]=[0,0,-1.0]; ps[17]=[0,0,1.0]; ps[4]=ps[5]=[-0.8,0,0]
    return ps

for smooth in (1.0, 0.3):
    sim = SMPLRagdollSim(proc)
    sim.set_free_joints([])          # joints keep following the capture
    sim.set_root_free(True)
    p.root_seed_smoothing = smooth
    w = np.ones(22)
    rel = int(0.55*FR)
    err_t, err_r = [], []
    for f in range(int(1.3*FR)):
        t = f/FR
        w[0] = 1.0 if f < rel else 0.0
        sim.advance(cap_pose(t), cap_trans(t), w, p)
        if f > rel:
            # The capture continued ballistically; so should the simulation.
            err_t.append(np.linalg.norm(sim.trans - cap_trans(t)))
            ang = np.linalg.norm(sim.root_rot.as_rotvec() - cap_pose(t)[0])
            err_r.append(ang)
    print('seed smoothing %.2f : after %.2f s of flight, position error %.4f m, '
          'orientation error %.4f rad (%.2f deg)'
          % (smooth, (len(err_t))/FR, err_t[-1], err_r[-1], np.degrees(err_r[-1])))


# The spin seed, which is directly comparable.
sim = SMPLRagdollSim(proc); sim.set_free_joints([]); sim.set_root_free(True)
p.root_seed_smoothing = 0.3
w = np.ones(22); rel = int(0.55*FR)
for f in range(rel+1):
    t = f/FR
    w[0] = 1.0 if f < rel else 0.0
    sim.advance(cap_pose(t), cap_trans(t), w, p)
print('\nat release: root spin %.4f rad/s (captured %.4f)'
      % (np.linalg.norm(sim.root_ang_vel), SPIN))
assert abs(np.linalg.norm(sim.root_ang_vel) - SPIN) < 0.02
print('release inherits the captured momentum')
