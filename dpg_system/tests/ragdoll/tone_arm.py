"""Tone on one arm at weight 1: does contraction pull and tremor read?

Runs the walk capture through the node with everything driven (weight 1)
and the left arm at tone 0, 0.5 and 1.  Reports, against the raw capture:
the left hand's mean displacement relative to the shoulder (the pull), the
elbow's mean flexion, the movement range about the clench (shrinkage), and
the tremor -- RMS and spectral peak of the fast residual at the elbow.
Right arm is the control: it must match the capture at every tone.
"""
import os, sys
from _env import ROOT, HERE
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, sosfiltfilt, welch
from repro_trans import Stub, P
from dpg_system.smpl_bullet import BulletBody

F = os.path.join(ROOT, 'assets/motion_capture_files/Walk B17 - Walk 2 hop 2 walk_poses.npz')
d = np.load(F, allow_pickle=True)
POSES = d['poses']; TRANS = d['trans']; FR = float(d['mocap_framerate'])
BETAS = d['betas'][:10]; GENDER = str(d['gender'])
START, SECS = 120, 6.0
N = int(SECS * FR)
LEFT = [13, 16, 18, 20]; RIGHT = [14, 17, 19, 21]


def run(tone):
    n = Stub('all', weight=1.0); n.framerate = FR; n.betas = BETAS; n.gender = GENDER
    n.weight_prop = P(1.0); n._apply_weight_immediately(); n.ramp_prop = P(120.0)
    n.clench_pull_prop = P(0.4); n.tremor_deg_prop = P(2.0); n.tremor_hz_prop = P(10.0)
    n.tone_targets[LEFT] = tone
    out = []
    for f in range(START, START + N):
        n.pose_input.v = POSES[f].copy(); n.trans_input.v = TRANS[f].copy(); n.execute()
        out.append(np.array(n.smpl_pose_output.last))
    return np.array(out), n


def fk(body, aa_seq):
    """hand and elbow relative to the shoulder, per frame, left and right"""
    rel = []
    for aa in aa_seq:
        body.set_pose(aa, np.array([0.0, 1.0, 0.0]))
        jp = body.joint_positions()
        rel.append([jp[20] - jp[16], jp[18] - jp[16], jp[21] - jp[17]])
    return np.array(rel)


def elbow_flex(aa_seq, j):
    return np.degrees(np.linalg.norm(aa_seq[:, j], axis=1))


def tremor(aa_seq, raw_seq, j):
    """the joint's rotation about the capture, band-passed 5-20 Hz (the
    pull's share of the arm swing lives below that): RMS in degrees and the
    spectral peak in Hz of the dominant axis"""
    rel = np.array([(R.from_rotvec(b).inv() * R.from_rotvec(a)).as_rotvec()
                    for a, b in zip(aa_seq[:, j], raw_seq[:, j])])
    sos = butter(4, [5.0, 20.0], btype='band', fs=FR, output='sos')
    fast = np.degrees(sosfiltfilt(sos, rel, axis=0))[int(FR):]
    rms = float(np.sqrt(np.mean(np.sum(fast ** 2, axis=1))))
    ax = int(np.argmax(np.var(fast, axis=0)))
    f_, P_ = welch(fast[:, ax], fs=FR, nperseg=256)
    return rms, float(f_[np.argmax(P_)])


raw = np.array([POSES[f][:72].reshape(24, 3) for f in range(START, START + N)])
res = {t: run(t) for t in (0.0, 0.5, 1.0)}
probe = BulletBody(res[0.0][1].processor, floor=False)
raw_rel = fk(probe, raw)
print('walk capture, %.0f Hz, %d frames; everything driven at weight 1; left arm toned' % (FR, N))
print('   capture: left elbow flexion mean %.1f deg, range %.1f deg; hand rel shoulder mean %s'
      % (elbow_flex(raw, 18).mean(), np.ptp(elbow_flex(raw, 18)), np.round(raw_rel[:, 0].mean(0), 3)))
for t, (out, n) in res.items():
    rel = fk(probe, out)
    dh = rel[:, 0] - raw_rel[:, 0]
    lf = elbow_flex(out, 18); rf = elbow_flex(out, 19); rf0 = elbow_flex(raw, 19)
    rms, hz = tremor(out, raw, 18)
    r_rms, _ = tremor(out, raw, 19)
    settle = slice(int(0.5 * FR), None)          # after the tone ramp
    print('tone %.1f:' % t)
    print('   left hand moved by %s m (mean over the take; +y up, +z forward), |mean| %.3f'
          % (np.round(dh[settle].mean(0), 3), np.linalg.norm(dh[settle].mean(0))))
    print('   left elbow flexion mean %.1f deg (capture %.1f), range %.1f (capture %.1f)'
          % (lf[settle].mean(), elbow_flex(raw, 18)[settle].mean(), np.ptp(lf[settle]), np.ptp(elbow_flex(raw, 18)[settle])))
    print('   left elbow tremor (5-20 Hz band): RMS %.2f deg, spectral peak %.1f Hz' % (rms, hz))
    print('   right arm (control): elbow mean %.1f vs capture %.1f deg, 5-20 Hz RMS %.2f deg, hand offset %.4f m'
          % (rf[settle].mean(), rf0[settle].mean(), r_rms, np.linalg.norm((rel[:, 2] - raw_rel[:, 2])[settle].mean(0))))
