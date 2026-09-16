"""smpl_resist on the walk and the punching take: does resisting the
passive share of motion take out the arm swing and the head lag while
leaving deliberate action alone?

Reports, at resist 0 / 1 / 2 (brake 2): the hands' forward-back travel
relative to the pelvis (the arm swing), the head's rotation range against
the spine, the mean passive share per joint, the largest deviation, and the
largest frame-to-frame jump in the output (an instability would show here).
On the punching take: elbow flexion range across the punches at each gain.
"""
import os, sys
from _env import ROOT, HERE
import numpy as np
from scipy.spatial.transform import Rotation as R
from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions
from dpg_system.smpl_resist import ResistFilter, ResistParams

DIR = os.path.join(ROOT, 'assets/motion_capture_files')
NAMES = ['pelvis', 'L_hip', 'R_hip', 'spine1', 'L_knee', 'R_knee', 'spine2', 'L_ankle', 'R_ankle', 'spine3',
         'L_foot', 'R_foot', 'neck', 'L_collar', 'R_collar', 'head', 'L_shoulder', 'R_shoulder',
         'L_elbow', 'R_elbow', 'L_wrist', 'R_wrist']


def load(name):
    d = np.load(os.path.join(DIR, name), allow_pickle=True)
    return d['poses'], d['trans'], float(d['mocap_framerate']), d['betas'][:10], str(d['gender'])


def run(name, start, n, resist, brake=0.0, leak=0.05, hold=10.0, legs=1.0):
    poses, trans, fr, betas, gender = load(name)
    proc = SMPLProcessor(framerate=fr, betas=betas, gender=gender, total_mass_kg=75.0,
                         model_path=os.path.join(ROOT, 'dpg_system'))
    filt = ResistFilter(proc)
    p_ = ResistParams(); p_.dt = 1.0 / fr; p_.resist = resist; p_.brake = brake; p_.leak_s = leak; p_.hold_s = hold
    opts = SMPLProcessingOptions(input_type='axis_angle', input_up_axis='Y', axis_permutation='x, z, -y',
                                 quat_format='wxyz', dt=p_.dt)
    gains = np.ones(24); gains[[1, 2, 4, 5, 7, 8, 10, 11]] = legs
    out, inp, share, dev = [], [], [], []
    for f in range(start, start + n):
        frame = poses[f][:72].reshape(1, 24, 3)
        t_int, aa_int, _ = proc._prepare_trans_and_pose(frame, trans[f].reshape(1, 3), opts)
        o = filt.step(aa_int[0], t_int[0], gains, p_)
        out.append(o); inp.append(aa_int[0].copy()); share.append(filt.passivity.copy())
        dev.append(np.linalg.norm(filt.dev, axis=1))
    return np.array(inp), np.array(out), np.array(share), np.array(dev), filt, fr


def hands_in_pelvis(filt, seq):
    """hand positions in the pelvis frame, (F, 2, 3)"""
    res = []
    for aa in seq:
        G, p, _ = filt.fk(aa, np.zeros(3))
        Rp = G[0].T
        res.append([Rp @ (p[20] - p[0]), Rp @ (p[21] - p[0])])
    return np.array(res)


def head_vs_spine(seq):
    return np.degrees([np.linalg.norm((R.from_rotvec(a[12]) * R.from_rotvec(a[15])).as_rotvec()) for a in seq])


def jumps(seq):
    d = [max(np.linalg.norm((R.from_rotvec(a[j]).inv() * R.from_rotvec(b[j])).as_rotvec()) for j in range(1, 22))
         for a, b in zip(seq[:-1], seq[1:])]
    return float(np.max(d))


print('WALK (Walk B17), 6 s from frame 120, leak 0.05 s, legs ON, brake 0')
for resist, brake, hold in ((0.0, 0.0, 10.0), (1.0, 0.0, 10.0), (3.0, 0.0, 10.0), (1.0, 1.0, 10.0)):
    inp, out, share, dev, filt, fr = run('Walk B17 - Walk 2 hop 2 walk_poses.npz', 120, 720, resist, brake, hold=hold)
    s = slice(int(fr), None)
    hi, ho = hands_in_pelvis(filt, inp[s]), hands_in_pelvis(filt, out[s])
    swing_in = [np.ptp(hi[:, k, 2]) for k in range(2)]; swing_out = [np.ptp(ho[:, k, 2]) for k in range(2)]
    print('resist %.0f brake %.0f hold %.1f:' % (resist, brake, hold))
    print('   hand forward-back travel vs pelvis: left %.2f -> %.2f m, right %.2f -> %.2f m'
          % (swing_in[0], swing_out[0], swing_in[1], swing_out[1]))
    print('   head rotation vs spine range: %.1f -> %.1f deg' % (np.ptp(head_vs_spine(inp[s])), np.ptp(head_vs_spine(out[s]))))
    print('   largest deviation %.2f rad (%s); largest frame jump in output %.3f rad (input %.3f)'
          % (dev[s].max(), NAMES[int(np.argmax(dev[s].max(0)[:22]))], jumps(out[s]), jumps(inp[s])))
    ef = lambda seq, j: np.degrees(np.linalg.norm(seq[:, j], axis=1))
    print('   legs: L_hip angle range %.0f -> %.0f deg, L_knee flexion range %.0f -> %.0f deg'
          % (np.ptp(ef(inp[s], 1)), np.ptp(ef(out[s], 1)), np.ptp(ef(inp[s], 4)), np.ptp(ef(out[s], 4))))
    if resist == 0.0 and brake == 0.0:
        ms = share[s].mean(0)
        print('   mean passivity: ' + ', '.join('%s %.2f' % (NAMES[j], ms[j]) for j in (1, 4, 7, 3, 12, 15, 16, 17, 18, 19)))

print('\nPUNCHING (INF S3), 8 s from frame 300')
for resist, brake, hold in ((0.0, 0.0, 10.0), (1.0, 0.0, 10.0), (3.0, 0.0, 10.0)):
    inp, out, share, dev, filt, fr = run('INF_PunchingKicking_S3_01_poses.npz', 300, 960, resist, brake, hold=hold)
    s = slice(int(fr), None)
    ef = lambda seq, j: np.degrees(np.linalg.norm(seq[:, j], axis=1))
    hi, ho = hands_in_pelvis(filt, inp[s]), hands_in_pelvis(filt, out[s])
    print('resist %.0f brake %.0f hold %.1f: elbow flexion range left %.0f -> %.0f deg, right %.0f -> %.0f deg; hand reach forward (95th pct) left %.2f -> %.2f m'
          % (resist, brake, hold, np.ptp(ef(inp[s], 18)), np.ptp(ef(out[s], 18)), np.ptp(ef(inp[s], 19)), np.ptp(ef(out[s], 19)),
             np.percentile(hi[:, 0, 2], 95), np.percentile(ho[:, 0, 2], 95)))
    print('   largest deviation %.2f rad (%s); largest frame jump %.3f rad (input %.3f)'
          % (dev[s].max(), NAMES[int(np.argmax(dev[s].max(0)[:22]))], jumps(out[s]), jumps(inp[s])))
