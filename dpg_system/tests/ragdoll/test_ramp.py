import os, sys
from _env import ROOT, HERE
import sys
import numpy as np
from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions
from dpg_system.smpl_ragdoll import SMPLRagdollSim, RagdollParams

HERE = os.path.join(ROOT, 'dpg_system'); FR = 60.0
proc = SMPLProcessor(framerate=FR, betas=np.zeros(10), gender='neutral',
                     total_mass_kg=75.0, model_path=HERE)
sim = SMPLRagdollSim(proc); sim.set_free_joints([16, 18, 20])
opts = SMPLProcessingOptions(input_type='axis_angle', input_up_axis='Y',
                             axis_permutation=None, quat_format='xyzw', dt=1.0/FR)
p = RagdollParams(); p.dt = 1.0/FR; p.substeps = 4

def kin(pose, trans=np.zeros(3)):
    t, aa, q = proc._prepare_trans_and_pose(pose.reshape(1,24,3).copy(),
                                            np.asarray(trans).reshape(1,3), opts)
    wp, gr, _ = proc._compute_forward_kinematics(t, q)
    rm = np.zeros((30,3,3))
    for i in range(30): rm[i] = np.asarray(gr[i].as_matrix()).reshape(-1,3,3)[0]
    return aa[0], wp[0], rm

def target(t):
    ps = np.zeros((24,3)); ps[16] = np.array([0., 0., 0.6*np.sin(2*np.pi*0.5*t)])
    return ps

print('steady-state tracking error vs weight (0.5 Hz shoulder sine):')
for weight in (0.0, 0.2, 0.5, 0.9, 0.99, 0.999):
    ww = np.ones(22); ww[[16,18,20]] = weight
    sim.reset(); errs = []
    for f in range(240):
        ps = target(f/FR); aa, wp, rm = kin(ps)
        res = sim.step(aa, wp, rm, ww, p)
        if f > 120: errs.append(np.linalg.norm(res[16] - ps[16]))
    print('  weight %-6.3f  mean error %.4f rad (%.1f deg)'
          % (weight, np.mean(errs), np.degrees(np.mean(errs))))

# Continuity: ramp 1.0 -> 0.0 over 120 ms and watch the per-frame pose delta.
print('\nrelease ramp 1.0 -> 0.0 over 120 ms; per-frame shoulder pose jump:')
ww = np.ones(22); sim.reset()
prev = None; jumps = []
ramp_frames = int(0.120 * FR)
for f in range(300):
    ps = target(f/FR); aa, wp, rm = kin(ps)
    if f >= 120:
        k = min(1.0, (f - 120) / ramp_frames)
        ww[[16,18,20]] = 1.0 - k
    res = sim.step(aa, wp, rm, ww, p)
    cur = res[16]
    if prev is not None and 115 < f < 200:
        jumps.append((f, float(np.linalg.norm(cur - prev)), float(ww[16])))
    prev = cur
worst = max(jumps, key=lambda x: x[1])
for f, d, wv in jumps:
    if f in (118, 119, 120, 121, 123, 127, 133, 140, 160, 190):
        print('   f%-4d weight %.3f  pose delta %.5f rad' % (f, wv, d))
print('   worst per-frame delta in the ramp: %.5f rad at f%d (weight %.3f)'
      % (worst[1], worst[0], worst[2]))

# Compare against the motion the capture itself makes per frame, as a scale.
mo = [np.linalg.norm(target((f+1)/FR)[16] - target(f/FR)[16]) for f in range(120, 200)]
print('   captured motion per frame over the same span: max %.5f rad' % max(mo))
assert worst[1] < 10 * max(mo), 'release ramp introduces a visible jump'
print('\nramp is continuous')
