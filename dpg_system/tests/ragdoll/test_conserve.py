import os, sys
from _env import ROOT, HERE
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions
from dpg_system.smpl_ragdoll import SMPLRagdollSim, RagdollParams

HERE=os.path.join(ROOT, 'dpg_system')
spin_rate = np.array([1.7, 0.9, 0.4])

def pose(t):
    ps = np.zeros((24,3))
    ps[16]=[0,0,-1.3]; ps[17]=[0,0,0.4]; ps[4]=[-0.9,0,0]
    ps[0]=spin_rate*t
    return ps

def run(FR, seconds=10.0):
    proc = SMPLProcessor(framerate=FR, betas=np.zeros(10), gender='neutral',
                         total_mass_kg=75.0, model_path=HERE)
    sim = SMPLRagdollSim(proc); sim.set_free_joints([]); sim.set_root_free(True)
    p = RagdollParams(); p.dt=1/FR; p.substeps=4
    p.floor_enable = False   # these check pure flight
    w = np.ones(22)
    seed = int(0.4*FR)
    n = int(seconds*FR)
    ke, Lnorm = [], []
    L0 = None
    for f in range(n):
        t = f/FR
        w[0] = 1.0 if f < seed else 0.0
        tr = np.array([1.2*t, 2.0*t - 0.5*9.81*t*t + 1.2, 0.3*t])
        sim.advance(pose(t), tr, w, p)
        if f == seed: L0 = sim.ang_momentum.copy()
        if f > seed:
            aa = pose(t).copy(); aa[0] = sim.root_rot.as_rotvec()
            wp, rm = sim._full_fk(aa, sim.trans)
            _, _, It, _ = sim._body_dynamics(wp, rm, np.zeros((24,3)))
            # The true torque-free invariant, from geometry alone.
            ke.append(0.5 * float(L0 @ np.linalg.solve(It, L0)))
            Lnorm.append(np.linalg.norm(sim.ang_momentum - L0))
    return np.array(ke), np.array(Lnorm), sim.root_rot, sim.com.copy()

ke, dL, rot120, com120 = run(120.0)
print('flight, 10 s, off-principal-axis spin, joints frozen:')
print('  stored angular momentum drift : %.3e  (exactly conserved by construction)' % dL.max())
print('  rotational energy (true invariant): %.4f J, drift over 10 s: %.3e J (%.4f%%)'
      % (ke[0], abs(ke-ke[0]).max(), 100*abs(ke-ke[0]).max()/ke[0]))

ke2, _, rot1200, com1200 = run(1200.0)
print('  same at 1200 Hz: energy drift %.3e J (%.4f%%)'
      % (abs(ke2-ke2[0]).max(), 100*abs(ke2-ke2[0]).max()/ke2[0]))
ang = float((rot120 * rot1200.inv()).magnitude())
print('  orientation after 10 s: 120 Hz vs 1200 Hz differ by %.4f rad (%.2f deg)' % (ang, np.degrees(ang)))
print('  centre of mass after 10 s: 120 Hz vs 1200 Hz differ by %.3e m' % np.linalg.norm(com120-com1200))
assert dL.max() < 1e-12
assert 100*abs(ke-ke[0]).max()/ke[0] < 2.0, 'energy drifting at 120 Hz'
print('  conserved; the spin variation seen is free precession, not integration drift.')
