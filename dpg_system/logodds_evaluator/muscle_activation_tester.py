#!/usr/bin/env python3
"""
Diagnostic test harness for the SMPL processor pipeline.

Covers:
  - Torque decomposition (gravity, dynamic, net — scalar and vector)
  - Contact pressure and force distribution
  - ZMP computation and jitter analysis
  - Body contact gating (CoM height + structural deficit)
  - Per-stream log-odds inspection
  - Surface distance verification

Usage:
    python muscle_activation_tester.py                          # defaults
    python muscle_activation_tester.py --file /path/to/data.npz # specific file
    python muscle_activation_tester.py --start 2920 --end 2940  # frame range
    python muscle_activation_tester.py --test torque_jitter      # specific test

Available tests:
    torque_jitter       Frame-to-frame torque vector magnitude jitter
    body_contact        Body contact gating diagnosis (heights, surface dists, intensities)
    zmp_pipeline        ZMP computation and noise decomposition
    pressure            Contact pressure distribution per joint
    logodds_streams     Per-stream log-odds increments for each group
    all                 Run all tests
"""

import sys
import argparse
import numpy as np

sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions

# ─────────────────────────────────────────────────────────────────────
# Default test data
# ─────────────────────────────────────────────────────────────────────
DEFAULT_FILE = '/Users/drokeby/dpg_system/dpg_system/Subject_81_F_19_poses.npz'
ALT_FILE = '/Users/drokeby/Projects/BMO Lab/GRANTS/NFRF 2023/smpl_data/HS_Nov_6/HS_take6_smpl_poses_b.npz'


def load_data(path):
    """Load NPZ mocap data, return (poses, trans, fps, betas, gender)."""
    d = np.load(path, allow_pickle=True)
    fps = float(d.get('mocap_framerate', 120.0))
    p, t = d['poses'], d['trans']
    betas = d['betas'] if 'betas' in d else None
    gender = str(d['gender']) if 'gender' in d else 'male'
    return p, t, fps, betas, gender


def reshape_pose(p):
    """Reshape a single-frame pose to (1, 24, 3) axis-angle."""
    return p[:72].reshape(1, 24, 3) if p.ndim == 1 and p.size >= 72 else p


def make_processor(fps, betas, gender):
    """Create an SMPLProcessor with standard settings."""
    return SMPLProcessor(
        framerate=fps,
        total_mass_kg=75.0,
        betas=betas,
        gender=gender,
        model_path='/Users/drokeby/dpg_system/dpg_system'
    )


def make_options(fps, **overrides):
    """Create SMPLProcessingOptions with the standard logodds_valved config.

    Pass keyword arguments to override any default option, e.g.:
        make_options(fps, enable_body_contacts=False, logodds_decay_rate=0.85)
    """
    defaults = dict(
        input_type='axis_angle',
        input_up_axis='Y',
        axis_permutation='x,z,-y',
        quat_format='wxyz',
        dt=1.0 / fps,
        add_gravity=True,
        enable_passive_limits=True,
        enable_apparent_gravity=True,
        floor_enable=True,
        floor_height=0.0,
        floor_tolerance=0.15,
        contact_method='logodds_valved',
        world_frame_dynamics=True,
        # CoM filtering
        com_pos_min_cutoff=999.0,   # effectively disables CoM position OEF
        com_pos_beta=1.0,
        com_vel_min_cutoff=20.0,
        com_vel_beta=0.1,
        com_acc_min_cutoff=5.0,
        com_acc_beta=0.8,
        # Smoothing
        smooth_input_window=5,
        enable_one_euro_filter=False,
        acc_smooth_window=7,
        # Contact
        enable_body_contacts=True,
        logodds_decay_rate=0.90,
        logodds_struct_force_ema_alpha=1.0,
        logodds_fe_relief_enable=True,
    )
    defaults.update(overrides)
    return SMPLProcessingOptions(**defaults)


def run_frames(processor, poses, trans, opts, start=0, end=None):
    """Run the processor over a frame range, returning per-frame results.

    Returns a list of dicts, one per frame in [start, end), each containing:
        'frame':          int
        'res':            dict (raw process_frame output)
        'world_pos':      (J, 3) world joint positions
        'floor_h':        float
        'com_h':          float (CoM height above floor)
        'logodds_result': LogOddsContactResult or None
        'contact_pressure': (J,) array or None
        'frame_eval':     EvalResult or None
    """
    if end is None:
        end = len(poses)
    end = min(end, len(poses))

    results = []
    for f in range(end):
        res = processor.process_frame(reshape_pose(poses[f]), trans[f:f + 1], opts)
        if f < start:
            continue

        wp = processor.last_world_pos
        if wp.ndim == 3:
            wp = wp[-1]
        floor_h = getattr(processor, '_inferred_floor_height', 0.0) or 0.0
        com = getattr(processor, '_prev_com_for_stability', None)
        com_h = com[0, 1] - floor_h if com is not None and com.ndim == 2 else 0.0

        lo = getattr(processor, '_logodds_result', None)
        cp = getattr(processor, 'contact_pressure', None)
        if cp is not None:
            cp = cp[0].copy() if cp.ndim > 1 else cp.copy()
        fe = getattr(processor, '_frame_eval_result', None)

        results.append({
            'frame': f,
            'res': res,
            'world_pos': wp.copy(),
            'floor_h': floor_h,
            'com_h': com_h,
            'logodds_result': lo,
            'contact_pressure': cp,
            'frame_eval': fe,
        })
    return results


# ─────────────────────────────────────────────────────────────────────
# Test functions
# ─────────────────────────────────────────────────────────────────────

def test_torque_jitter(path, start, end):
    """Measure frame-to-frame gravity/dynamic torque vector jitter."""
    poses, trans, fps, betas, gender = load_data(path)
    pr = make_processor(fps, betas, gender)
    opts = make_options(fps)

    grav_vecs, dyn_vecs = [], []
    for f in range(min(end, len(poses))):
        res = pr.process_frame(reshape_pose(poses[f]), trans[f:f + 1], opts)
        if f >= start and res and 'torques_grav_vec' in res:
            grav_vecs.append(res['torques_grav_vec'][0].copy())
            dyn_vecs.append(res['torques_dyn_vec'][0].copy())

    gv = np.array(grav_vecs)
    dv = np.array(dyn_vecs)
    gv_mag = np.linalg.norm(gv, axis=-1)
    dv_mag = np.linalg.norm(dv, axis=-1)
    dgv = np.diff(gv_mag, axis=0)
    ddv = np.diff(dv_mag, axis=0)

    joints = {'pelvis': 0, 'L_hip': 1, 'R_hip': 2, 'spine1': 3,
              'L_knee': 4, 'R_knee': 5, 'L_ankle': 7, 'R_ankle': 8}

    print('\n=== Torque Vector Magnitude Jitter (Nm, frame-to-frame) ===')
    print(f'{"Joint":>10s} | {"Gravity std":>10s} {"max":>6s} | {"Dynamic std":>10s} {"max":>6s}')
    print('-' * 60)
    for name, j in joints.items():
        print(f'{name:>10s} | {np.std(dgv[:, j]):10.3f} {np.max(np.abs(dgv[:, j])):6.2f}'
              f' | {np.std(ddv[:, j]):10.3f} {np.max(np.abs(ddv[:, j])):6.2f}')


def test_body_contact(path, start, end):
    """Diagnose body contact gating: heights, surface distances, intensities."""
    poses, trans, fps, betas, gender = load_data(path)
    pr = make_processor(fps, betas, gender)
    opts = make_options(fps)

    print(f'\n=== Body Contact Diagnosis (frames {start}-{end}) ===')
    print(f'{"f":>5s} | {"com_h":>6s} {"body":>5s} | {"pelv_raw":>8s} {"pelv_corr":>9s}'
          f' | {"Lhip_corr":>9s} {"Rhip_corr":>9s} | intensities')
    print('-' * 95)

    for f in range(min(end, len(poses))):
        res = pr.process_frame(reshape_pose(poses[f]), trans[f:f + 1], opts)
        if f < start:
            continue

        wp = pr.last_world_pos
        if wp.ndim == 3:
            wp = wp[-1]
        floor_h = getattr(pr, '_inferred_floor_height', 0.0) or 0.0
        com = getattr(pr, '_prev_com_for_stability', None)
        com_h = com[0, 1] - floor_h if com is not None and com.ndim == 2 else 0.0

        # Surface distances
        extents = getattr(pr, '_joint_surface_extents', None)
        min_dists = getattr(pr, '_joint_surface_min_dists', None)
        grots = getattr(pr, '_prev_global_rots', None)
        h_p = wp[0, 1] - floor_h
        h_p_corr = h_lh_corr = h_rh_corr = 0
        if extents is not None and grots is not None:
            from dpg_system.dynamic_frame_evaluator import DynamicFrameEvaluator
            fn = np.array([0, 1, 0])
            g = grots[0] if grots.ndim == 4 else grots
            sd_max = DynamicFrameEvaluator.compute_effective_surface_distances(
                extents, g, fn, num_joints=24)
            sd_min = (DynamicFrameEvaluator.compute_min_surface_distances(
                min_dists, g, fn, num_joints=24)
                if min_dists is not None else np.full(24, 0.03))
            # Pelvis uses sd_min, hips use sd_max
            h_p_corr = h_p - sd_min[0]
            h_lh_corr = wp[1, 1] - floor_h - sd_max[1]
            h_rh_corr = wp[2, 1] - floor_h - sd_max[2]

        est = getattr(pr, '_logodds_estimator', None)
        ba = getattr(est, '_body_contacts_active', False) if est else False

        lo = getattr(pr, '_logodds_result', None)
        int_str = ''
        if lo:
            for gn in ['LH2', 'RH2', 'PV', 'LK', 'RK']:
                v = lo.intensity.get(gn, 0)
                if v > 0.01:
                    int_str += f'{gn}={v:.2f} '

        print(f'{f:5d} | {com_h:6.3f} {str(ba):>5s} | {h_p:8.3f} {h_p_corr:+9.3f}'
              f' | {h_lh_corr:+9.3f} {h_rh_corr:+9.3f} | {int_str}')


def test_zmp_pipeline(path, start, end):
    """ZMP noise decomposition: position vs acceleration contributions."""
    poses, trans, fps, betas, gender = load_data(path)
    pr = make_processor(fps, betas, gender)
    opts = make_options(fps)

    zmps, coms, accs = [], [], []
    for f in range(min(end, len(poses))):
        res = pr.process_frame(reshape_pose(poses[f]), trans[f:f + 1], opts)
        if f >= start:
            zmp = pr.current_zmp.copy()
            if zmp.ndim > 1:
                zmp = zmp[-1]
            zmps.append(zmp)
            com = getattr(pr, '_prev_com_for_stability', None)
            if com is not None:
                c = com[0] if com.ndim == 2 else com
                coms.append(c.copy())
            accs.append(getattr(pr, 'prob_prev_com_acc', np.zeros(3)).copy())

    zmps = np.array(zmps)
    coms = np.array(coms)
    accs = np.array(accs)

    dzmp = np.diff(zmps, axis=0)
    dcom = np.diff(coms, axis=0)
    dacc = np.diff(accs, axis=0)

    print('\n=== ZMP Pipeline Noise (frame-to-frame std, mm) ===')
    print(f'{"Signal":>15s} | {"X":>8s} | {"Z":>8s}')
    print('-' * 38)
    print(f'{"ZMP":>15s} | {np.std(dzmp[:, 0]) * 1000:8.2f} | {np.std(dzmp[:, 2]) * 1000:8.2f}')
    print(f'{"CoM position":>15s} | {np.std(dcom[:, 0]) * 1000:8.2f} | {np.std(dcom[:, 2]) * 1000:8.2f}')
    print(f'{"CoM accel":>15s} | {np.std(dacc[:, 0]):8.4f} | {np.std(dacc[:, 2]):8.4f} (m/s²)')

    # Contribution analysis
    h_g = 0.85 / 9.81
    pos_contrib = np.std(dcom[:, [0, 2]], axis=0)
    acc_contrib = np.std(dacc[:, [0, 2]], axis=0) * h_g
    total = pos_contrib + acc_contrib
    print(f'\nAcceleration contribution to ZMP jitter: '
          f'X={acc_contrib[0] / total[0] * 100:.0f}% Z={acc_contrib[1] / total[1] * 100:.0f}%')


def test_pressure(path, start, end):
    """Contact pressure distribution per joint."""
    poses, trans, fps, betas, gender = load_data(path)
    pr = make_processor(fps, betas, gender)
    opts = make_options(fps)

    print(f'\n=== Contact Pressure (frames {start}-{end}) ===')
    joint_names = {0: 'pelvis', 1: 'L_hip', 2: 'R_hip', 4: 'L_knee', 5: 'R_knee',
                   7: 'L_ankle', 8: 'R_ankle', 10: 'L_foot', 11: 'R_foot',
                   20: 'L_wrist', 21: 'R_wrist', 22: 'L_hand', 23: 'R_hand',
                   28: 'L_heel', 29: 'R_heel'}

    for f in range(min(end, len(poses))):
        res = pr.process_frame(reshape_pose(poses[f]), trans[f:f + 1], opts)
        if f < start:
            continue
        cp = getattr(pr, 'contact_pressure', None)
        if cp is None:
            continue
        cp0 = cp[0] if cp.ndim > 1 else cp
        active = [(j, cp0[j]) for j in range(len(cp0)) if cp0[j] > 0.1]
        if active:
            parts = [f'{joint_names.get(j, f"j{j}")}={v:.1f}kg' for j, v in active]
            print(f'f{f}: {" ".join(parts)}  (total={sum(v for _, v in active):.1f}kg)')


def test_upper_limb_filtering(path, start, end):
    """Raw vs filtered torque attenuation for the upper-limb chain.

    Runs two passes over [0, end):
      A: canonical filter stack (smooth_input_window=5, acc_smooth_window=7,
         adaptive_effort_smooth=True)
      B: tunable filters disabled (smooth_input_window=0, acc_smooth_window=0,
         adaptive_effort_smooth=False). The per-joint One Euro filter on
         angular velocity inside _compute_angular_kinematics is unconditional
         and runs in both passes — that's intentional, we're isolating the
         effect of the user-tunable filters that the user is worried about.

    Reports per-joint (L/R shoulder, elbow, wrist):
      raw_rms / filt_rms         — torque magnitude RMS (Nm)
      retained_pct               — energy retained
      raw_djitter / filt_djitter — std of frame-to-frame magnitude diff (Nm)
      jit_retained_pct           — jitter retained (how much fast variation survives)
      frac_below_lo              — fraction of frames where raw mag < adaptive_effort_lo
                                   (== fraction stuck in heavy-smoothing regime)
      alpha_est_mean             — mean alpha that the adaptive EMA *would*
                                   choose given the raw-pass magnitude

    Then prints the K frames with the largest |raw - filt| gap so we can
    eyeball whether the gap is real motion or jitter.
    """
    from scipy.spatial.transform import Rotation as R_sp

    poses, trans, fps, betas, gender = load_data(path)
    end = min(end, len(poses))

    # Two processors, two option sets — separate state.
    pr_A = make_processor(fps, betas, gender)
    pr_B = make_processor(fps, betas, gender)
    opts_A = make_options(fps)  # canonical
    opts_B = make_options(
        fps,
        smooth_input_window=0,
        acc_smooth_window=0,
        adaptive_effort_smooth=False,
    )

    # SMPL upper-limb joint indices
    JOINTS = {
        'L_shoulder': 16, 'R_shoulder': 17,
        'L_elbow':    18, 'R_elbow':    19,
        'L_wrist':    20, 'R_wrist':    21,
    }
    # Per-joint alpha_max from smpl_processor.py (matching global_alpha_max=0.5)
    ALPHA_MAX = {16: 0.5, 17: 0.5, 18: 0.6, 19: 0.6, 20: 0.8, 21: 0.8}
    # Effort-based thresholds (||efforts_net|| = ||τ/τ_max||)
    LO = 0.1
    HI = 0.5
    ALPHA_MIN = 0.05

    tv_A = {n: [] for n in JOINTS}   # post-filter torque magnitude per joint
    tv_B = {n: [] for n in JOINTS}   # pre-filter (raw) torque magnitude per joint
    eff_B = {n: [] for n in JOINTS}  # raw effort magnitude per joint (||τ/τ_max||)
    tv_A_vec = {n: [] for n in JOINTS}
    tv_B_vec = {n: [] for n in JOINTS}
    kin_deg = {n: [] for n in JOINTS}  # raw per-frame angular displacement (deg)
    frame_idx = []
    max_torque = pr_A.max_torque_array  # (24, 3) Nm per axis

    prev_pose_aa = None
    for f in range(end):
        pose_f = reshape_pose(poses[f])
        trans_f = trans[f:f + 1]
        res_A = pr_A.process_frame(pose_f, trans_f, opts_A)
        res_B = pr_B.process_frame(pose_f, trans_f, opts_B)
        if f < start:
            prev_pose_aa = pose_f[0].copy() if pose_f.ndim == 3 else pose_f.copy()
            continue

        tv_A_arr = res_A.get('torques_vec')   # (1, J, 3)
        tv_B_arr = res_B.get('torques_vec')
        if tv_A_arr is None or tv_B_arr is None:
            continue
        ta = tv_A_arr[0]
        tb = tv_B_arr[0]

        # Kinematic ground truth: angular displacement of joint local rotation
        # frame-to-frame, from the raw input axis-angle.
        curr_aa = pose_f[0] if pose_f.ndim == 3 else pose_f
        for name, j in JOINTS.items():
            tv_A_vec[name].append(ta[j].copy())
            tv_B_vec[name].append(tb[j].copy())
            tv_A[name].append(float(np.linalg.norm(ta[j])))
            tv_B[name].append(float(np.linalg.norm(tb[j])))
            # Raw effort = ||τ / τ_max|| per axis. Use pass B torque so we
            # get an estimate of the pre-EMA effort the code would see.
            eff_B[name].append(float(np.linalg.norm(tb[j] / (max_torque[j] + 1e-6))))
            if prev_pose_aa is not None and j < curr_aa.shape[0]:
                r_curr = R_sp.from_rotvec(curr_aa[j])
                r_prev = R_sp.from_rotvec(prev_pose_aa[j])
                ang = np.linalg.norm((r_curr * r_prev.inv()).as_rotvec())
                kin_deg[name].append(float(np.degrees(ang)))
            else:
                kin_deg[name].append(0.0)
        frame_idx.append(f)
        prev_pose_aa = curr_aa.copy()

    # ─── Per-joint summary ──────────────────────────────────────────────
    print(f'\n=== Upper-Limb Filter Attenuation (frames {start}-{end}) ===')
    print(f'fps={fps:.1f}  effort_lo={LO}  effort_hi={HI}  alpha_min={ALPHA_MIN}')
    print(f'{"joint":>10s} | {"raw_rms":>7s} {"filt_rms":>8s} {"ret%":>5s}'
          f' | {"raw_djit":>8s} {"filt_djit":>8s} {"jit%":>5s}'
          f' | {"<lo%":>5s} {"alpha_est":>9s} | {"raw_max":>7s} {"eff_max":>7s} {"kin_max°":>8s}')
    print('-' * 120)
    summary = {}
    for name, j in JOINTS.items():
        a = np.array(tv_A[name])
        b = np.array(tv_B[name])
        e = np.array(eff_B[name])
        k = np.array(kin_deg[name])
        if len(a) < 2:
            continue
        raw_rms = float(np.sqrt(np.mean(b ** 2)))
        filt_rms = float(np.sqrt(np.mean(a ** 2)))
        retained = 100.0 * filt_rms / (raw_rms + 1e-9)
        raw_djit = float(np.std(np.diff(b)))
        filt_djit = float(np.std(np.diff(a)))
        jit_retained = 100.0 * filt_djit / (raw_djit + 1e-9)
        frac_below_lo = 100.0 * np.mean(e < LO)
        blend = np.clip((e - LO) / (HI - LO), 0.0, 1.0)
        alpha_est = ALPHA_MIN + blend * (ALPHA_MAX[j] - ALPHA_MIN)
        print(f'{name:>10s} | {raw_rms:7.2f} {filt_rms:8.2f} {retained:5.0f}%'
              f' | {raw_djit:8.3f} {filt_djit:8.3f} {jit_retained:5.0f}%'
              f' | {frac_below_lo:4.0f}% {float(np.mean(alpha_est)):9.3f}'
              f' | {float(np.max(b)):7.2f} {float(np.max(e)):7.2f} {float(np.max(k)):8.2f}')
        summary[name] = dict(a=a, b=b, e=e, k=k, alpha_est=alpha_est)

    # ─── Top-K most divergent frames per joint ──────────────────────────
    K = 8
    print(f'\n=== Top {K} frames with largest raw-vs-filtered torque gap ===')
    for name, s in summary.items():
        gap = np.abs(s['b'] - s['a'])
        order = np.argsort(-gap)[:K]
        print(f'\n  {name}:')
        print(f'    {"frame":>6s}  {"raw":>6s}  {"filt":>6s}  {"gap":>6s}'
              f'  {"eff":>5s}  {"alpha":>6s}  {"kin°":>6s}')
        for idx in order:
            f = frame_idx[idx]
            print(f'    {f:>6d}  {s["b"][idx]:6.2f}  {s["a"][idx]:6.2f}'
                  f'  {gap[idx]:6.2f}  {s["e"][idx]:5.2f}  {s["alpha_est"][idx]:6.3f}'
                  f'  {s["k"][idx]:6.2f}')


def test_logodds_streams(path, start, end):
    """Per-stream log-odds increments for each group."""
    poses, trans, fps, betas, gender = load_data(path)
    pr = make_processor(fps, betas, gender)
    opts = make_options(fps)

    print(f'\n=== Log-Odds Per-Stream Increments (frames {start}-{end}) ===')

    for f in range(min(end, len(poses))):
        res = pr.process_frame(reshape_pose(poses[f]), trans[f:f + 1], opts)
        if f < start:
            continue
        lo = getattr(pr, '_logodds_result', None)
        if lo is None:
            continue

        # Show per-stream for foot groups
        for gn in ['LF', 'RF']:
            streams = lo.per_stream.get(gn, {})
            parts = [f'{k}={v:+.2f}' for k, v in streams.items()
                     if k != 'total_increment' and abs(v) > 0.001]
            total = streams.get('total_increment', 0)
            intensity = lo.intensity.get(gn, 0)
            lo_state = lo.log_odds_state.get(gn, 0)
            print(f'f{f} {gn}: LO={lo_state:+.2f} I={intensity:.2f} | '
                  f'{" ".join(parts)} → Δ={total:+.2f}')


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

TESTS = {
    'torque_jitter': test_torque_jitter,
    'body_contact': test_body_contact,
    'zmp_pipeline': test_zmp_pipeline,
    'pressure': test_pressure,
    'logodds_streams': test_logodds_streams,
    'upper_limb_filtering': test_upper_limb_filtering,
}


def main():
    parser = argparse.ArgumentParser(
        description='SMPL processor pipeline diagnostic tester.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--file', '-f', default=DEFAULT_FILE,
                        help='Path to NPZ mocap data')
    parser.add_argument('--start', '-s', type=int, default=100,
                        help='Start frame (default: 100, skipping init transient)')
    parser.add_argument('--end', '-e', type=int, default=400,
                        help='End frame (default: 400)')
    parser.add_argument('--test', '-t', default='all',
                        choices=list(TESTS.keys()) + ['all'],
                        help='Which test to run (default: all)')
    args = parser.parse_args()

    print(f'File: {args.file}')
    print(f'Frames: {args.start} - {args.end}')

    if args.test == 'all':
        for name, fn in TESTS.items():
            print(f'\n{"=" * 70}')
            print(f'  {name}')
            print(f'{"=" * 70}')
            fn(args.file, args.start, args.end)
    else:
        TESTS[args.test](args.file, args.start, args.end)


if __name__ == '__main__':
    main()
