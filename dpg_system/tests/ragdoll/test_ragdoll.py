import os, sys
from _env import ROOT, HERE
"""Headless check of the ragdoll free-chain simulation."""
import sys, os
import numpy as np

from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions
from dpg_system.smpl_ragdoll import SMPLRagdollSim, RagdollParams

HERE = os.path.join(ROOT, 'dpg_system')
FR = 60.0

proc = SMPLProcessor(framerate=FR, betas=np.zeros(10), gender='neutral',
                     total_mass_kg=75.0, model_path=HERE)
print('limb masses:', {k: round(v, 2) for k, v in proc.limb_data['masses'].items()})

sim = SMPLRagdollSim(proc)
sim.set_free_joints([16, 18, 20])          # left arm from the shoulder down
print('free:', sim.free_indices, 'roots:', sim.roots, 'sim nodes:', sim.sim_nodes)
print('subtree of 16:', sim.subtree_of[16])
print('seg mass 16/18/20/22:', proc._seg_mass[[16, 18, 20, 22]])

opts = SMPLProcessingOptions(input_type='axis_angle', input_up_axis='Y',
                             axis_permutation=None, quat_format='xyzw', dt=1.0/FR)
p = RagdollParams(); p.dt = 1.0 / FR; p.substeps = 4

def kinematics(pose_aa, trans):
    frame = pose_aa.reshape(1, 24, 3).copy()
    t, aa, q = proc._prepare_trans_and_pose(frame, np.asarray(trans).reshape(1, 3), opts)
    wp, gr, _ = proc._compute_forward_kinematics(t, q)
    rm = np.zeros((30, 3, 3))
    for i in range(30):
        rm[i] = np.asarray(gr[i].as_matrix()).reshape(-1, 3, 3)[0]
    return aa[0], wp[0], rm

def hand_height(pose_aa, trans, result):
    pose = pose_aa.reshape(24, 3).copy()
    for j, v in result.items():
        pose[j] = v
    _, wp, _ = kinematics(pose, trans)
    return wp[22, 1], wp[20, 1]      # left hand, left wrist, world Y


# --- Test 1: T-pose arm held horizontally, then released -------------------
# SMPL rest pose has the arms out to the sides; a free arm must fall to hang.
pose = np.zeros((24, 3))
trans = np.array([0.0, 0.0, 0.0])
w = np.ones(22); w[[16, 18, 20]] = 0.0

sim.reset()
aa, wp, rm = kinematics(pose, trans)
print('\nrest wrist Y = %.3f  hand Y = %.3f' % (wp[20, 1], wp[22, 1]))

heights = []
for f in range(240):
    aa, wp, rm = kinematics(pose, trans)
    res = sim.step(aa, wp, rm, w, p)
    hy, wy = hand_height(pose, trans, res)
    heights.append(hy)
    if f in (0, 5, 15, 30, 60, 120, 239):
        print('  f%-4d hand Y = %+.4f   wrist Y = %+.4f   |w_shoulder| = %.3f'
              % (f, hy, wy, np.linalg.norm(sim.ang_vel[16])))

assert np.all(np.isfinite(heights)), 'non-finite arm height'
print('drop: %.3f -> %.3f m  (settled |w| = %.4f rad/s)'
      % (heights[0], heights[-1], np.linalg.norm(sim.ang_vel[16])))


# --- Test 2: driven body accelerating sideways -----------------------------
# Same free arm, but the pelvis translates with a sharp acceleration. The
# transport term should deflect the arm relative to the static case.
def run_with_trans(trans_fn, transport):
    p.transport = transport
    sim.reset()
    for f in range(180):
        tr = trans_fn(f / FR)
        aa, wp, rm = kinematics(pose, tr)
        res = sim.step(aa, wp, rm, w, p)
    return sim.local_rot[16].as_rotvec().copy()

still = lambda t: np.array([0.0, 0.0, 0.0])
# 2 Hz sideways shake, +/- 15 cm
shake = lambda t: np.array([0.15 * np.sin(2 * np.pi * 2.0 * t), 0.0, 0.0])

a_still = run_with_trans(still, 1.0)
a_shake_on = run_with_trans(shake, 1.0)
a_shake_off = run_with_trans(shake, 0.0)
print('\nshoulder rotvec, still           :', np.round(a_still, 4))
print('shoulder rotvec, shaken (transport on ):', np.round(a_shake_on, 4))
print('shoulder rotvec, shaken (transport off):', np.round(a_shake_off, 4))
print('transport deflection = %.4f rad ; without transport = %.4f rad'
      % (np.linalg.norm(a_shake_on - a_still), np.linalg.norm(a_shake_off - a_still)))
p.transport = 1.0


# --- Test 3: partial weight should track the capture ------------------------
# Animate the shoulder and check a half-driven joint follows better than a free one.
def tracking_error(weight):
    ww = np.ones(22); ww[[16, 18, 20]] = weight
    sim.reset()
    errs = []
    for f in range(180):
        t = f / FR
        ps = np.zeros((24, 3))
        ps[16] = np.array([0.0, 0.0, 0.6 * np.sin(2 * np.pi * 0.5 * t)])
        aa, wp, rm = kinematics(ps, trans)
        res = sim.step(aa, wp, rm, ww, p)
        if f > 60:
            errs.append(np.linalg.norm(res[16] - ps[16]))
    return float(np.mean(errs))

for weight in (0.0, 0.05, 0.2, 0.6):
    print('weight %.2f -> mean shoulder tracking error %.4f rad' % (weight, tracking_error(weight)))


# --- Test 4: release carries momentum --------------------------------------
# Prescribed (weight 1) up to the release, then free. The arm must keep moving.
sim.reset()
ww = np.ones(22)
speeds = []
for f in range(180):
    t = f / FR
    ps = np.zeros((24, 3))
    ps[16] = np.array([0.0, 0.0, 1.2 * np.sin(2 * np.pi * 0.8 * t)])
    aa, wp, rm = kinematics(ps, trans)
    if f == 90:
        ww[[16, 18, 20]] = 0.0      # let go at maximum swing speed
    res = sim.step(aa, wp, rm, ww, p)
    if f in (89, 90, 91, 95, 105):
        speeds.append((f, float(np.linalg.norm(sim.ang_vel[16]))))
print('\nshoulder angular speed around release:', [(f, round(s, 3)) for f, s in speeds])
assert speeds[1][1] > 0.5, 'released arm lost its momentum'

print('\nall checks finite and complete')
