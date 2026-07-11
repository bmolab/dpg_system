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


# Per-joint alpha_max from smpl_processor.py (matching global_alpha_max=0.5).
# Keep in sync with _joint_alpha dict in adaptive_effort smoothing block.
_PER_JOINT_ALPHA_MAX = {
    0: 0.3,  1: 0.3,  2: 0.3,  3: 0.3,
    4: 0.4,  5: 0.4,  6: 0.3,  7: 0.6,
    8: 0.6,  9: 0.35, 10: 0.8, 11: 0.8,
    12: 0.4, 13: 0.4, 14: 0.4, 15: 0.8,
    16: 0.5, 17: 0.5, 18: 0.6, 19: 0.6,
    20: 0.8, 21: 0.8,
}


def _joint_filtering_diagnostic(path, start, end, joints, title, input_smooth=5):
    """Shared raw-vs-filtered torque diagnostic for a chosen joint set.

    Runs two passes over [0, end):
      A: canonical filter stack (smooth_input_window=5, acc_smooth_window=7,
         adaptive_effort_smooth=True)
      B: tunable filters disabled (smooth_input_window=0, acc_smooth_window=0,
         adaptive_effort_smooth=False). The per-joint One Euro filter on
         angular velocity inside _compute_angular_kinematics is unconditional
         and runs in both passes — we're isolating the user-tunable filters.

    Reports per-joint:
      raw_rms / filt_rms   — torque magnitude RMS (Nm)
      retained_pct         — energy retained
      raw/filt jit (std)   — std of frame-to-frame magnitude diff (Nm)
      jit_retained_pct     — how much fast variation survives
      <lo%                 — fraction of frames where raw effort < adaptive_effort_lo
      alpha_est_mean       — mean alpha the EMA would choose for those frames
      raw_max / eff_max    — peak raw torque (Nm) and effort fraction
      kin_max°             — peak per-frame local angular displacement (deg)

    Then prints the K frames with the largest |raw - filt| gap.
    """
    from scipy.spatial.transform import Rotation as R_sp

    poses, trans, fps, betas, gender = load_data(path)
    end = min(end, len(poses))

    pr_A = make_processor(fps, betas, gender)
    pr_B = make_processor(fps, betas, gender)
    opts_A = make_options(fps, smooth_input_window=input_smooth)
    opts_B = make_options(
        fps,
        smooth_input_window=0,
        acc_smooth_window=0,
        adaptive_effort_smooth=False,
    )

    LO, HI, ALPHA_MIN = 0.1, 0.5, 0.05

    tv_A = {n: [] for n in joints}
    tv_B = {n: [] for n in joints}
    eff_B = {n: [] for n in joints}
    kin_deg = {n: [] for n in joints}
    frame_idx = []
    max_torque = pr_A.max_torque_array

    prev_pose_aa = None
    for f in range(end):
        pose_f = reshape_pose(poses[f])
        trans_f = trans[f:f + 1]
        res_A = pr_A.process_frame(pose_f, trans_f, opts_A)
        res_B = pr_B.process_frame(pose_f, trans_f, opts_B)
        if f < start:
            prev_pose_aa = pose_f[0].copy() if pose_f.ndim == 3 else pose_f.copy()
            continue

        tv_A_arr = res_A.get('torques_vec')
        tv_B_arr = res_B.get('torques_vec')
        if tv_A_arr is None or tv_B_arr is None:
            continue
        ta = tv_A_arr[0]
        tb = tv_B_arr[0]

        curr_aa = pose_f[0] if pose_f.ndim == 3 else pose_f
        for name, j in joints.items():
            tv_A[name].append(float(np.linalg.norm(ta[j])))
            tv_B[name].append(float(np.linalg.norm(tb[j])))
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

    print(f'\n=== {title} Filter Attenuation (frames {start}-{end}) ===')
    print(f'fps={fps:.1f}  effort_lo={LO}  effort_hi={HI}  alpha_min={ALPHA_MIN}')
    print(f'{"joint":>10s} | {"raw_rms":>7s} {"filt_rms":>8s} {"ret%":>5s}'
          f' | {"raw_djit":>8s} {"filt_djit":>8s} {"jit%":>5s}'
          f' | {"<lo%":>5s} {"alpha_est":>9s} | {"raw_max":>7s} {"eff_max":>7s} {"kin_max°":>8s}')
    print('-' * 120)
    summary = {}
    for name, j in joints.items():
        a = np.array(tv_A[name]); b = np.array(tv_B[name])
        e = np.array(eff_B[name]); k = np.array(kin_deg[name])
        if len(a) < 2:
            continue
        raw_rms = float(np.sqrt(np.mean(b ** 2)))
        filt_rms = float(np.sqrt(np.mean(a ** 2)))
        retained = 100.0 * filt_rms / (raw_rms + 1e-9)
        raw_djit = float(np.std(np.diff(b)))
        filt_djit = float(np.std(np.diff(a)))
        jit_retained = 100.0 * filt_djit / (raw_djit + 1e-9)
        frac_below_lo = 100.0 * np.mean(e < LO)
        amax = _PER_JOINT_ALPHA_MAX.get(j, 0.5)
        blend = np.clip((e - LO) / (HI - LO), 0.0, 1.0)
        alpha_est = ALPHA_MIN + blend * (amax - ALPHA_MIN)
        print(f'{name:>10s} | {raw_rms:7.2f} {filt_rms:8.2f} {retained:5.0f}%'
              f' | {raw_djit:8.3f} {filt_djit:8.3f} {jit_retained:5.0f}%'
              f' | {frac_below_lo:4.0f}% {float(np.mean(alpha_est)):9.3f}'
              f' | {float(np.max(b)):7.2f} {float(np.max(e)):7.2f} {float(np.max(k)):8.2f}')
        summary[name] = dict(a=a, b=b, e=e, k=k, alpha_est=alpha_est)

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


def test_upper_limb_filtering(path, start, end):
    """Raw vs filtered torque attenuation for L/R shoulder, elbow, wrist."""
    _joint_filtering_diagnostic(path, start, end, {
        'L_shoulder': 16, 'R_shoulder': 17,
        'L_elbow':    18, 'R_elbow':    19,
        'L_wrist':    20, 'R_wrist':    21,
    }, title='Upper-Limb')


def test_lower_limb_filtering(path, start, end):
    """Raw vs filtered torque attenuation for L/R hip, knee, ankle."""
    # 60-fps files in this project use a 3-frame input smoother (5 is for
    # higher-fps mocap). If we ever drive this from data, read it from
    # the file. For now, default to 3 — safe for the LR take 3 file.
    _joint_filtering_diagnostic(path, start, end, {
        'L_hip':   1, 'R_hip':   2,
        'L_knee':  4, 'R_knee':  5,
        'L_ankle': 7, 'R_ankle': 8,
    }, title='Lower-Limb', input_smooth=3)


# ─────────────────────────────────────────────────────────────────────
# Coherence-gated adaptive-effort prototype (2026-07-06 popping work)
# ─────────────────────────────────────────────────────────────────────
#
# Hypothesis: the per-joint alpha_max cap (0.5 on shoulders) over-
# attenuates genuine popping accents (Subject10: R_shoulder f1137
# 106 Nm → 0.33 Nm filtered). Raising the cap globally would re-admit
# the noise we deliberately filter (soft-tissue ringing, magnetometer
# jitter, single-frame spikes, cadence steps). Instead, lift the cap
# per-frame-per-joint ONLY when two NON-LOCAL features agree:
#   coherence — the joint's fast motion is corroborated by simultaneous
#               fast motion in its kinematic neighbours. Real popping is a
#               coordinated whole-chain hit; ringing / per-IMU jitter are
#               spatially LOCAL, so a min-across-the-neighbourhood stays
#               low for them.
#   envelope  — the motion is a multi-frame ballistic ramp, not a 1-frame
#               spike. Sensor spikes and 2,2,1 cadence steps are temporally
#               ISOLATED (high instantaneous speed, low windowed mean).
# Both are soft valves in [0,1] (per soft-valving-over-gating); the gate
# is their product. gate=0 leaves the canonical cap untouched; gate=1
# lifts it to 1.0. Forward net-displacement / "commitment" was tried and
# REJECTED as a feature — it scores strike-and-recoil pops as low and
# collides with the reversal-at-speed glitch signature.
#
# NOTE ON CAUSALITY: the envelope window here is symmetric (±_ENV_HALF)
# for characterisation. A live/streaming port needs either a trailing
# window or a small (~2-3 frame / 20-30 ms) look-ahead latency.

# Kinematic neighbourhoods (SMPL 24-joint layout): {proximal, self, distal}
_ARM_NEIGH = {
    16: [13, 16, 18], 17: [14, 17, 19],   # shoulders: collar, self, elbow
    18: [16, 18, 20], 19: [17, 19, 21],   # elbows:    shoulder, self, wrist
    20: [18, 20],     21: [19, 21],        # wrists:    elbow, self
}
# Soft-valve knee points (tuned from the 2026-07-06 Subject10 popping scan:
# quiet-frame coherence baseline ~0.30, pop accents 1.6-2.7).
_COH_LO, _COH_HI = 0.7, 1.8      # min neighbour normalised speed
_ENV_LO, _ENV_HI = 0.30, 0.65    # windowed-mean / windowed-peak speed
_ENV_HALF = 2                    # symmetric ±frames for the envelope window


def _arm_kinematic_gate(poses, target_joints):
    """Per-frame coherence*envelope gate in [0,1] for each target joint.

    Features come from raw pose kinematics (deg/frame), independent of the
    torque pipeline. Returns (gates, coh_raw, env_raw), each a dict
    joint -> (N,) array.
    """
    from scipy.spatial.transform import Rotation as R_sp
    N = poses.shape[0]
    aa = poses[:, :72].reshape(N, 24, 3)
    r = R_sp.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(N, 24, 3, 3)

    def _ang(a, b):
        m = np.matmul(a, np.swapaxes(b, -1, -2))
        tr = np.clip((m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2] - 1) / 2, -1, 1)
        return np.degrees(np.arccos(tr))

    speed = np.zeros((N, 24))
    speed[1:] = _ang(r[1:], r[:-1])            # (N, 24) deg/frame
    mean_sp = speed[1:].mean(axis=0) + 1e-6    # per-joint normalisation scale
    norm_sp = speed / mean_sp

    gates, coh_out, env_out = {}, {}, {}
    for j in target_joints:
        nb = _ARM_NEIGH[j]
        coh_raw = norm_sp[:, nb].min(axis=1)   # whole neighbourhood co-active
        coh_v = np.clip((coh_raw - _COH_LO) / (_COH_HI - _COH_LO), 0.0, 1.0)
        s = speed[:, j]
        env_raw = np.zeros(N)
        for f in range(N):
            w = s[max(0, f - _ENV_HALF):f + _ENV_HALF + 1]
            env_raw[f] = w.mean() / (w.max() + 1e-6)
        env_v = np.clip((env_raw - _ENV_LO) / (_ENV_HI - _ENV_LO), 0.0, 1.0)
        gates[j] = coh_v * env_v
        coh_out[j] = coh_raw
        env_out[j] = env_raw
    return gates, coh_out, env_out


def test_coherence_gated_filtering(path, start, end):
    """Prototype: coherence+envelope-gated lift of the adaptive-effort
    alpha_max cap, vs the canonical EMA, referenced to the pre-EMA torque.

    Runs the processor once with the EMA OFF (front-end smoothing still on)
    to get the pre-EMA torque, then re-implements the exact adaptive-effort
    EMA in-tester two ways: canonical cap vs gated cap. The canonical
    reconstruction is validated against the processor's own full-pipeline
    output (sanity line — should be ~0).
    """
    joints = {'L_shoulder': 16, 'R_shoulder': 17,
              'L_elbow': 18, 'R_elbow': 19,
              'L_wrist': 20, 'R_wrist': 21}
    poses, trans, fps, betas, gender = load_data(path)
    end = min(end, len(poses))
    n_j = 22  # processor emits 22 body joints (target_joint_count)

    # Pass PRE: front-end smoothing ON, adaptive-effort EMA OFF.
    pr_pre = make_processor(fps, betas, gender)
    opts_pre = make_options(fps, adaptive_effort_smooth=False)
    # Pass A: full canonical pipeline, for reconstruction sanity check.
    pr_A = make_processor(fps, betas, gender)
    opts_A = make_options(fps)

    tv_pre = np.zeros((end, n_j, 3))
    en_pre = np.zeros((end, n_j, 3))
    tvA_mag = {n: np.zeros(end) for n in joints}
    for f in range(end):
        pose_f = reshape_pose(poses[f]); trans_f = trans[f:f + 1]
        rp = pr_pre.process_frame(pose_f, trans_f, opts_pre)
        ra = pr_A.process_frame(pose_f, trans_f, opts_A)
        tv_pre[f] = rp['torques_vec'][0]
        en_pre[f] = rp['efforts_net'][0]
        for n, j in joints.items():
            tvA_mag[n][f] = float(np.linalg.norm(ra['torques_vec'][0][j]))

    gates, coh_raw, env_raw = _arm_kinematic_gate(poses[:end], list(joints.values()))
    gate_arr = np.zeros((end, n_j))
    for j, g in gates.items():
        gate_arr[:, j] = g

    # Canonical EMA constants (match make_options / processor defaults).
    LO, HI, AMIN = 0.1, 0.5, 0.05
    K, CLO, CHI = 5, 0.02, 0.10
    Amax = np.array([_PER_JOINT_ALPHA_MAX.get(j, 0.5) for j in range(n_j)])

    def run_ema(gate_lift):
        """Replicate the adaptive-effort EMA. gate_lift: (end, n_j) alpha_max
        lift factor in [0,1], or None for canonical."""
        tv_sm = np.zeros((end, n_j, 3))
        en_sm = np.zeros((end, n_j, 3))
        ring = np.zeros((K, n_j)); rptr = 0; rcnt = 0
        for f in range(end):
            if f == 0:
                tv_sm[f] = tv_pre[f]; en_sm[f] = en_pre[f]
                continue
            eff_mags = np.linalg.norm(en_pre[f], axis=-1)
            eff_blend = np.clip((eff_mags - LO) / (HI - LO), 0.0, 1.0)
            deff = np.linalg.norm(en_pre[f] - en_sm[f - 1], axis=-1)
            ring[rptr] = deff; rptr = (rptr + 1) % K; rcnt = min(rcnt + 1, K)
            deff_med = np.median(ring[:rcnt], axis=0) if rcnt >= 3 else np.zeros(n_j)
            chg_blend = np.clip((deff_med - CLO) / (CHI - CLO), 0.0, 1.0)
            blend = np.maximum(eff_blend, chg_blend)
            amax = Amax.copy()
            if gate_lift is not None:
                amax = amax + gate_lift[f] * (1.0 - amax)
            alpha = (AMIN + blend * (amax - AMIN))[:, None]
            tv_sm[f] = alpha * tv_pre[f] + (1.0 - alpha) * tv_sm[f - 1]
            en_sm[f] = alpha * en_pre[f] + (1.0 - alpha) * en_sm[f - 1]
        return tv_sm

    tv_canon = run_ema(None)
    tv_gated = run_ema(gate_arr)

    # Sanity: canonical reconstruction vs processor over the reported range.
    max_dev = 0.0
    for n, j in joints.items():
        rec = np.linalg.norm(tv_canon[start:end, j], axis=-1)
        max_dev = max(max_dev, float(np.max(np.abs(rec - tvA_mag[n][start:end]))))

    print(f'\n=== Coherence-Gated Adaptive-Effort Prototype (frames {start}-{end}) ===')
    print(f'fps={fps:.1f}  coh_knee=[{_COH_LO},{_COH_HI}]  env_knee=[{_ENV_LO},{_ENV_HI}]')
    print(f'sanity: max|canon_recon - processor| = {max_dev:.2e} Nm  (should be ~0)')
    print(f'{"joint":>10s} | {"preEMA":>7s} {"canon":>6s} {"gated":>6s}'
          f' | {"can_ret%":>8s} {"gat_ret%":>8s} | {"can_jit%":>8s} {"gat_jit%":>8s}'
          f' | {"gate_mn":>7s} {"gate>.1":>7s}')
    print('-' * 118)
    summ = {}
    for n, j in joints.items():
        pre = np.linalg.norm(tv_pre[start:end, j], axis=-1)
        can = np.linalg.norm(tv_canon[start:end, j], axis=-1)
        gat = np.linalg.norm(tv_gated[start:end, j], axis=-1)
        g = gate_arr[start:end, j]
        pre_rms = float(np.sqrt(np.mean(pre ** 2)))
        can_rms = float(np.sqrt(np.mean(can ** 2)))
        gat_rms = float(np.sqrt(np.mean(gat ** 2)))
        can_djit = float(np.std(np.diff(can))); gat_djit = float(np.std(np.diff(gat)))
        pre_djit = float(np.std(np.diff(pre))) + 1e-9
        print(f'{n:>10s} | {pre_rms:7.2f} {can_rms:6.2f} {gat_rms:6.2f}'
              f' | {100 * can_rms / (pre_rms + 1e-9):8.0f} {100 * gat_rms / (pre_rms + 1e-9):8.0f}'
              f' | {100 * can_djit / pre_djit:8.0f} {100 * gat_djit / pre_djit:8.0f}'
              f' | {float(np.mean(g)):7.3f} {100 * float(np.mean(g > 0.1)):6.0f}%')
        summ[n] = dict(j=j, pre=pre, can=can, gat=gat, g=g)

    K_top = 6
    print(f'\n=== Top {K_top} pre-vs-canon gap frames: canon vs gated recovery ===')
    for n, s in summ.items():
        gap = np.abs(s['pre'] - s['can'])
        order = np.argsort(-gap)[:K_top]
        print(f'\n  {n}:')
        print(f'    {"frame":>6s}  {"preEMA":>6s}  {"canon":>6s}  {"gated":>6s}'
              f'  {"gate":>5s}  {"coh":>5s}  {"env":>5s}')
        for idx in order:
            f = start + idx
            print(f'    {f:>6d}  {s["pre"][idx]:6.2f}  {s["can"][idx]:6.2f}'
                  f'  {s["gat"][idx]:6.2f}  {s["g"][idx]:5.2f}'
                  f'  {coh_raw[s["j"]][f]:5.2f}  {env_raw[s["j"]][f]:5.2f}')


# Candidate adaptive-front-end config (2026-07-06). Kept here so the compare
# test and any tuning share one definition. Legacy = plain make_options(fps).
ADAPTIVE_FRONTEND_CFG = dict(
    adaptive_frontend=True,
    coherence_gate_enable=True,
    smooth_input_window=0,   # pose smoothing replaced by the gated OEF
    acc_smooth_window=3,     # light fixed accel cleanup (keeps quiet clean)
)


def test_frontend_compare(path, start, end):
    """Legacy fixed-window front-end vs the gated adaptive OEF front-end.

    Reports per-joint torque RMS, peak, and frame-to-frame jitter for both,
    plus the mean gate. Adaptive should lift peak/jitter on limbs where the
    gate opens (real accents) while leaving low-gate joints ~unchanged.
    """
    joints = {'L_shoulder': 16, 'R_shoulder': 17,
              'L_elbow': 18, 'R_elbow': 19,
              'L_wrist': 20, 'R_wrist': 21}
    poses, trans, fps, betas, gender = load_data(path)
    end = min(end, len(poses))

    def run(overrides):
        pr = make_processor(fps, betas, gender)
        opts = make_options(fps, **overrides)
        mags = {n: np.zeros(end) for n in joints}
        gate = {n: np.zeros(end) for n in joints}
        for f in range(end):
            r = pr.process_frame(reshape_pose(poses[f]), trans[f:f + 1], opts)
            tv = r['torques_vec'][0]
            g = getattr(pr, '_cg_gate', None)
            for n, j in joints.items():
                mags[n][f] = np.linalg.norm(tv[j])
                if g is not None:
                    gate[n][f] = g[j]
        return mags, gate

    leg, _ = run({})
    adp, gt = run(ADAPTIVE_FRONTEND_CFG)

    print(f'\n=== Front-end compare: legacy vs adaptive (frames {start}-{end}) ===')
    print(f'cfg: {ADAPTIVE_FRONTEND_CFG}')
    print(f'{"joint":>10s} | {"leg_rms":>7s} {"adp_rms":>7s} | {"leg_pk":>7s} {"adp_pk":>7s}'
          f' | {"leg_jit":>7s} {"adp_jit":>7s} | {"gate_mn":>7s}')
    print('-' * 90)
    for n in joints:
        l = leg[n][start:end]; a = adp[n][start:end]; g = gt[n][start:end]
        print(f'{n:>10s} | {np.sqrt((l**2).mean()):7.2f} {np.sqrt((a**2).mean()):7.2f}'
              f' | {l.max():7.1f} {a.max():7.1f}'
              f' | {np.std(np.diff(l)):7.3f} {np.std(np.diff(a)):7.3f} | {g.mean():7.3f}')


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
    'lower_limb_filtering': test_lower_limb_filtering,
    'coherence_gated': test_coherence_gated_filtering,
    'frontend_compare': test_frontend_compare,
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
