"""Probe: how fast is torque, and does batch match streaming?

Answers two questions before the sweep gets sized:
  1. seconds per 1000 frames, batch vs streaming
  2. whether batch torque agrees with streaming torque closely enough that
     the sweep can use batch (the noise detector used streaming for
     state-continuity with the live node; characterization may not need it)
"""
import os
import sys
import time

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))  # repo root

from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions

MODEL_PATH = os.path.dirname(os.path.abspath(__file__)) + '/..'
N_JOINTS = 24
STREAM_KEYS = ('torques_vec', 'torques_dyn_vec',
               'torques_grav_vec', 'torques_passive_vec')


def make_options(dt):
    """Mirrors the smpl_torque node defaults, as estimate_noise_torque.py does."""
    return SMPLProcessingOptions(
        input_type='axis_angle',
        input_up_axis='Y',
        axis_permutation='x, z, -y',
        quat_format='wxyz',
        return_quats=False,
        dt=dt,
        add_gravity=True,
        enable_passive_limits=True,
        enable_apparent_gravity=True,
        use_s_curve_spine=True,
        world_frame_dynamics=True,
        floor_enable=True,
        floor_height=0.0,
        floor_tolerance=0.15,
        contact_method='logodds_valved',
        enable_body_contacts=True,
        logodds_enable_height=True,
        logodds_enable_kinematic=True,
        logodds_enable_structural=True,
        logodds_enable_divergence=True,
        logodds_weight_height=1.0,
        logodds_weight_kinematic=0.5,
        logodds_weight_structural=1.0,
        logodds_weight_divergence=1.0,
        logodds_decay_rate=0.90,
        logodds_struct_force_ema_alpha=1.0,
        logodds_struct_relief_logodds=0.3,
        enable_rate_limiting=False,
        enable_jitter_damping=False,
        enable_kf_smoothing=False,
        enable_velocity_gate=False,
        enable_one_euro_filter=False,
        smooth_input_window=0,
        smooth_contact_forces=False,
        acc_smooth_window=7,
        torque_smooth_window=0,
        com_pos_min_cutoff=999.0,
        com_pos_beta=1.0,
        com_vel_min_cutoff=20.0,
        com_vel_beta=0.1,
        com_acc_min_cutoff=5.0,
        com_acc_beta=0.8,
    )


def load(path, max_frames):
    d = np.load(path, allow_pickle=True)
    poses = d['poses'][:max_frames]
    trans = d['trans'][:max_frames]
    fps = 60.0
    for key in ('mocap_framerate', 'motioncapture_framerate', 'framerate'):
        if key in d:
            fps = float(d[key])
            break
    betas = None
    if 'betas' in d:
        betas = np.array(d['betas'], dtype=np.float64).flatten()[:10]
    gender = 'neutral'
    if 'gender' in d:
        g = d['gender']
        g = g.item() if hasattr(g, 'item') else g
        g = g.decode('utf-8') if isinstance(g, bytes) else str(g)
        if g in ('male', 'female', 'neutral'):
            gender = g
    return poses, trans, fps, betas, gender


def new_processor(fps, betas, gender):
    p = SMPLProcessor(framerate=fps, betas=betas, gender=gender,
                      total_mass_kg=75.0, model_path=MODEL_PATH)
    p.set_axis_permutation('x, z, -y')
    return p


def collect(res, out, t0, n):
    for key in STREAM_KEYS:
        v = res.get(key, None)
        if v is None:
            continue
        v = np.asarray(v)
        if v.ndim == 2:
            v = v[np.newaxis]
        m = min(N_JOINTS, v.shape[1])
        out[key][t0:t0 + n, :m] = v[:n, :m]


def run_streaming(poses, trans, fps, betas, gender):
    T = poses.shape[0]
    p = new_processor(fps, betas, gender)
    opts = make_options(1.0 / fps)
    out = {k: np.zeros((T, N_JOINTS, 3)) for k in STREAM_KEYS}
    start = time.perf_counter()
    for t in range(T):
        try:
            res = p.process_frame(poses[t:t + 1], trans[t:t + 1], opts)
        except Exception:
            continue
        collect(res, out, t, 1)
    return out, time.perf_counter() - start


def run_batch(poses, trans, fps, betas, gender):
    T = poses.shape[0]
    p = new_processor(fps, betas, gender)
    opts = make_options(1.0 / fps)
    out = {k: np.zeros((T, N_JOINTS, 3)) for k in STREAM_KEYS}
    start = time.perf_counter()
    res = p.process_frame(poses, trans, opts)
    collect(res, out, 0, T)
    return out, time.perf_counter() - start


def main():
    path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 600

    poses, trans, fps, betas, gender = load(path, max_frames)
    T = poses.shape[0]
    print(f"file   : {os.path.basename(path)}")
    print(f"frames : {T} @ {fps:.0f} fps   gender={gender}")

    s_out, s_sec = run_streaming(poses, trans, fps, betas, gender)
    print(f"streaming: {s_sec:6.2f} s  ->  {s_sec / T * 1000:7.2f} s/1000 frames")
    print(f"speedup  : {s_sec / max(b_sec, 1e-9):.1f}x")

    print("\nagreement (batch vs streaming), per stream:")
    for key in STREAM_KEYS:
        a, b = b_out[key], s_out[key]
        scale = np.percentile(np.abs(b), 99)
        if scale < 1e-9:
            print(f"  {key:22s} both ~zero")
            continue
        diff = np.abs(a - b)
        # correlation on the flattened channel-time field
        af, bf = a.ravel(), b.ravel()
        if af.std() > 1e-12 and bf.std() > 1e-12:
            r = float(np.corrcoef(af, bf)[0, 1])
        else:
            r = float('nan')
        print(f"  {key:22s} p99|stream|={scale:8.3f}  "
              f"median|diff|={np.median(diff):8.4f}  "
              f"p99|diff|={np.percentile(diff, 99):8.4f}  r={r:.4f}")


if __name__ == '__main__':
    main()
