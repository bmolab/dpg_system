import os, sys
from _env import ROOT, HERE
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions
from dpg_system.smpl_ragdoll import SMPLRagdollSim, RagdollParams, SMPLRagdollNode

HERE = os.path.join(ROOT, 'dpg_system'); FR = 60.0
proc = SMPLProcessor(framerate=FR, betas=np.zeros(10), gender='neutral',
                     total_mass_kg=75.0, model_path=HERE)

# ---------------------------------------------------------------- 1. frames
class FrameStub(SMPLRagdollNode):
    def __init__(self, proc, up, perm):
        self.processor = proc
        self.up_axis_prop = lambda: up
        self.axis_perm_prop = lambda: perm

rng = np.random.default_rng(0)
print('frame round trip (internal -> input):')
worst_r = worst_t = 0.0
for perm in ('', 'x, z, -y', 'x, y, z', 'y, -x, z', '-z, y, x'):
    for up in ('Y', 'Z'):
        for _ in range(20):
            pose = rng.normal(0, 0.6, (24, 3))
            trans = rng.normal(0, 1.5, 3)
            opts = SMPLProcessingOptions(input_type='axis_angle', input_up_axis=up,
                                         axis_permutation=perm, quat_format='xyzw', dt=1/FR)
            t_int, aa_int, _ = proc._prepare_trans_and_pose(
                pose.reshape(1, 24, 3).copy(), trans.reshape(1, 3).copy(), opts)
            node = FrameStub(proc, up, perm)
            rot_b, trans_b = node._to_input_frame(R.from_rotvec(aa_int[0][0]), t_int[0])
            ang = (rot_b * R.from_rotvec(pose[0]).inv()).magnitude()
            worst_r = max(worst_r, float(ang))
            worst_t = max(worst_t, float(np.linalg.norm(trans_b - trans)))
        print('  perm %-12r up=%s  worst rot err %.2e  worst trans err %.2e'
              % (perm, up, worst_r, worst_t))
assert worst_r < 1e-9 and worst_t < 1e-9, (worst_r, worst_t)

# ---------------------------------------------------------------- 2. flight
opts = SMPLProcessingOptions(input_type='axis_angle', input_up_axis='Y',
                             axis_permutation=None, quat_format='xyzw', dt=1/FR)
sim = SMPLRagdollSim(proc)
sim.set_free_joints([])          # joints driven; only the root flies
sim.set_root_free(True)
p = RagdollParams(); p.dt = 1/FR; p.substeps = 4
p.floor_enable = False   # these check pure flight
w = np.ones(22)

def leap_pose(t):
    ps = np.zeros((24, 3))
    ps[16] = [0, 0, -1.0]; ps[17] = [0, 0, 1.0]
    # a forward somersault: the root tips about X while airborne
    ps[0] = np.array([2.0 * t, 0.0, 0.0])
    return ps

def leap_trans(t):
    return np.array([1.5 * t, 1.6 * t - 0.5 * 9.81 * t * t + 1.0, 0.0])

print('\nprescribed leap for 30 frames, then release the root:')
coms, vys, oms = [], [], []
for f in range(150):
    t = f / FR
    if f < 30:
        ps, tr = leap_pose(t), leap_trans(t)
        w[0] = 1.0
    else:
        # capture continues, but the root is released
        ps, tr = leap_pose(t), leap_trans(t)
        w[0] = 0.0
    res, rrot, rtr = sim.advance(ps, tr, w, p)
    coms.append(sim.com.copy()); vys.append(sim.com_vel[1])
    oms.append(np.linalg.norm(sim.root_ang_vel))
coms = np.array(coms)
acc = np.diff(np.array(vys)[35:]) * FR
print('  vertical acceleration after release: mean %.4f m/s^2 (expected -9.8100), spread %.2e'
      % (acc.mean(), acc.max() - acc.min()))
assert abs(acc.mean() + 9.81) < 1e-6

# the root must be placed so the body's real centre of mass sits on the parabola
aa = np.zeros((24, 3)); aa[:] = leap_pose(149/FR)
aa[0] = sim.root_rot.as_rotvec()
wp, rm = sim._full_fk(aa, sim.trans)
_, com_fk, _, _ = sim._body_dynamics(wp, rm, np.zeros((24, 3)))
print('  centre of mass placement error at f149: %.3e m' % np.linalg.norm(com_fk - sim.com))
assert np.linalg.norm(com_fk - sim.com) < 1e-4

print('  root spin: at release %.3f rad/s, at f149 %.3f rad/s (captured spin was 2.000)'
      % (oms[31], oms[-1]))

# ---------------------------------------------------------------- 3. tuck
print('\ntuck test: root free, joints still driven, dancer eases into a tuck')
print('  (tuck runs f40-f85 with a cosine ease, so joint speed starts and ends at zero;')
print('   spin is compared before the tuck and well after it, both with limbs static)')

def tuck(t, amount):
    ps = np.zeros((24, 3))
    ps[0] = np.array([2.0 * t, 0.0, 0.0])
    ps[4] = ps[5] = np.array([-2.4 * amount, 0, 0])       # knees
    ps[1] = np.array([-1.2 * amount, 0, 0.1])             # hips
    ps[2] = np.array([-1.2 * amount, 0, -0.1])
    ps[16] = [0, 0, -1.0 + 0.6 * amount]; ps[17] = [0, 0, 1.0 - 0.6 * amount]
    return ps

def ease(f, start, length):
    k = min(1.0, max(0.0, (f - start) / float(length)))
    return 0.5 * (1.0 - np.cos(np.pi * k))

for label, tuck_it in (('stays extended', False), ('tucks', True)):
    sim2 = SMPLRagdollSim(proc); sim2.set_free_joints([]); sim2.set_root_free(True)
    ww = np.ones(22)
    spin, inertia_axis = [], []
    for f in range(200):
        t = f / FR
        amount = ease(f, 40, 45) if tuck_it else 0.0
        ww[0] = 1.0 if f < 30 else 0.0
        sim2.advance(tuck(t, amount), leap_trans(t), ww, p)
        spin.append(np.linalg.norm(sim2.root_ang_vel))
        if f in (39, 199):
            aa = tuck(t, amount).copy(); aa[0] = sim2.root_rot.as_rotvec()
            wp, rm = sim2._full_fk(aa, sim2.trans)
            _, _, it, _ = sim2._body_dynamics(wp, rm, np.zeros((24, 3)))
            axis = sim2.root_ang_vel / max(np.linalg.norm(sim2.root_ang_vel), 1e-9)
            inertia_axis.append(float(axis @ it @ axis))
    print('  %-15s spin %.3f -> %.3f rad/s (x%.2f) ; inertia about the spin axis %.3f -> %.3f kg m^2 (x%.2f)'
          % (label, spin[39], spin[199], spin[199] / spin[39],
             inertia_axis[0], inertia_axis[1], inertia_axis[1] / inertia_axis[0]))
    if tuck_it:
        assert spin[199] > spin[39] * 1.3, 'tuck did not speed up the spin'
        assert inertia_axis[1] < inertia_axis[0] * 0.8, 'tuck did not reduce the inertia'

print('\nroot six-DoF checks complete')
