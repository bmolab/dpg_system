#!/usr/bin/env python3
"""
Find an optimal yaw rotation of the world space that minimizes
planted-foot skating in an SMPL motion file.

The IMU-derived body orientation (poses) and the 3D-sensor-derived
translation (trans) may have a residual yaw misalignment. When a foot is
planted on the floor it should not slide horizontally, so the yaw angle
that minimizes the total horizontal motion of planted feet across the
recording is the alignment correction.

The rotation is applied only to `trans` (the root translation) — leaving
the IMU-derived root orientation unchanged. This is the only choice that
reduces skating; rotating both leaves it unchanged.

Floor contact is detected with smpl_processor (the same pipeline used by
the smpl_torque node). The closed-form 2-D Procrustes solver recovers
~90% of the misalignment per pass (the remaining 10% comes from non-linear
filter interactions inside the processor that don't commute exactly with
rotation); the script iterates by default until either the per-pass angle
drops below `--converge-deg` or `--iterations` is reached.
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions

MODEL_PATH = str(Path(__file__).resolve().parent)

# Contact joints (from smpl_processor): toes and heel virtual joints.
# 10 = L_foot (toe), 11 = R_foot (toe), 28 = L_heel, 29 = R_heel
CONTACT_JOINTS = [10, 11, 28, 29]


def reshape_pose(p):
    return p[:72].reshape(1, 24, 3) if p.ndim == 1 and p.size >= 72 else p


def make_options(fps):
    return SMPLProcessingOptions(
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
        com_pos_min_cutoff=999.0,
        com_pos_beta=1.0,
        com_vel_min_cutoff=20.0,
        com_vel_beta=0.1,
        com_acc_min_cutoff=5.0,
        com_acc_beta=0.8,
        smooth_input_window=5,
        enable_one_euro_filter=False,
        acc_smooth_window=7,
        enable_body_contacts=True,
        logodds_decay_rate=0.90,
        logodds_struct_force_ema_alpha=1.0,
        logodds_fe_relief_enable=True,
    )


def collect_processor_data(poses, trans, fps, betas, gender, total_mass, verbose):
    """Run the processor over every frame; capture contact pressure and world joint
    positions per frame.

    Returns:
        contact_pressure: (T, J) array in kg
        wp_proc:          (T, J, 3) world joint positions in processor's Y-up frame
    """
    pr = SMPLProcessor(framerate=fps, betas=betas, gender=gender,
                       model_path=MODEL_PATH, total_mass_kg=total_mass)
    opts = make_options(fps)

    T = len(poses)
    # Probe shapes on first frame
    pr.process_frame(reshape_pose(poses[0]), trans[0:1], opts)
    cp0 = pr.contact_pressure
    if cp0 is None:
        raise RuntimeError("Processor did not produce contact_pressure on frame 0")
    cp0 = cp0[0] if cp0.ndim > 1 else cp0
    wp0 = pr.last_world_pos
    if wp0.ndim == 3:
        wp0 = wp0[-1]
    n_j = min(len(cp0), len(wp0))

    contact_pressure = np.zeros((T, n_j))
    wp_proc = np.zeros((T, n_j, 3))
    contact_pressure[0] = cp0[:n_j]
    wp_proc[0] = wp0[:n_j]

    for f in range(1, T):
        pr.process_frame(reshape_pose(poses[f]), trans[f:f + 1], opts)
        cp = pr.contact_pressure
        if cp is not None:
            cp = cp[0] if cp.ndim > 1 else cp
            contact_pressure[f] = cp[:n_j]
        wp = pr.last_world_pos
        if wp.ndim == 3:
            wp = wp[-1]
        wp_proc[f] = wp[:n_j]

        if verbose and (f % 500 == 0 or f == T - 1):
            print(f"    processed {f + 1}/{T} frames", end='\r')
    if verbose:
        print()

    return contact_pressure, wp_proc


def proc_to_file_frame(wp_proc):
    """Inverse of the processor's `axis_permutation='x,z,-y'`:
    proc (x_p, y_p, z_p) -> file (x_p, -z_p, y_p)  [Z-up]."""
    wp_file = np.empty_like(wp_proc)
    wp_file[..., 0] = wp_proc[..., 0]
    wp_file[..., 1] = -wp_proc[..., 2]
    wp_file[..., 2] = wp_proc[..., 1]
    return wp_file


def skating_total(theta, wp_file, contact_pair_mask, joint_indices):
    """Sum of horizontal skating magnitude over all (t, j) where both endpoints planted.

    Everything is computed in the processor's reference: wp[:, j] - wp[:, 0]
    gives R_root * J_local, and wp[:, 0] gives the (smoothed) root translation.
    Using wp[:, 0] for b keeps a and b temporally aligned; since rotation
    commutes with linear smoothing, the recovered θ still applies cleanly
    to the raw `trans` in the file.
    """
    c, s = np.cos(theta), np.sin(theta)
    db = np.diff(wp_file[:, 0, :2], axis=0)
    db_rot = np.column_stack([c * db[:, 0] - s * db[:, 1],
                              s * db[:, 0] + c * db[:, 1]])
    total = 0.0
    n = 0
    for j in joint_indices:
        off = wp_file[:, j, :2] - wp_file[:, 0, :2]
        a = np.diff(off, axis=0)
        v = a + db_rot
        mag = np.linalg.norm(v, axis=1)
        mask = contact_pair_mask[:, j]
        total += mag[mask].sum()
        n += int(mask.sum())
    return total, n


def apply_yaw(trans, theta):
    c, s = np.cos(theta), np.sin(theta)
    out = trans.copy()
    out[:, 0] = c * trans[:, 0] - s * trans[:, 1]
    out[:, 1] = s * trans[:, 0] + c * trans[:, 1]
    return out


def solve_yaw_once(poses, trans, fps, betas, gender, total_mass,
                   warmup_frames, contact_threshold_kg, verbose):
    """One processor pass + closed-form yaw on planted-foot pairs.

    Returns (theta, n_pairs, skating_before, skating_after, pelvis_err, joint_indices).
    """
    contact_pressure, wp_proc = collect_processor_data(
        poses, trans, fps, betas, gender, total_mass, verbose=verbose
    )
    wp_file = proc_to_file_frame(wp_proc)
    pelvis_err = float(np.linalg.norm(wp_file[warmup_frames, 0] - trans[warmup_frames]))

    n_j = wp_file.shape[1]
    joint_indices = [j for j in CONTACT_JOINTS if j < n_j]
    if not joint_indices:
        raise RuntimeError(f"No contact joints in range; world_pos has {n_j} joints, "
                           f"need at least {min(CONTACT_JOINTS) + 1}")

    cmask = contact_pressure[:, :n_j] > contact_threshold_kg
    cmask[:warmup_frames] = False
    pair_mask = cmask[1:] & cmask[:-1]

    # Closed-form 2-D Procrustes:
    #   minimize Σ ||a + Rz(θ) b||²  →  θ* = atan2(-B, -A)
    #   where A = Σ(a·b), B = Σ(a_y b_x - a_x b_y) = -Σ(a × b)
    A_sum = 0.0
    B_sum = 0.0
    n_pairs = 0
    db = np.diff(wp_file[:, 0, :2], axis=0)
    for j in joint_indices:
        off = wp_file[:, j, :2] - wp_file[:, 0, :2]
        a = np.diff(off, axis=0)
        mask = pair_mask[:, j]
        a_m = a[mask]
        b_m = db[mask]
        A_sum += np.sum(a_m[:, 0] * b_m[:, 0] + a_m[:, 1] * b_m[:, 1])
        B_sum += np.sum(a_m[:, 1] * b_m[:, 0] - a_m[:, 0] * b_m[:, 1])
        n_pairs += int(mask.sum())

    if n_pairs == 0:
        raise RuntimeError(
            "No planted-foot contact pairs found "
            f"(contact_threshold={contact_threshold_kg} kg, warmup={warmup_frames}). "
            "Lower the threshold or shorten the warmup."
        )

    theta = float(np.arctan2(-B_sum, -A_sum))
    before, _ = skating_total(0.0, wp_file, pair_mask, joint_indices)
    after, _ = skating_total(theta, wp_file, pair_mask, joint_indices)
    return theta, n_pairs, before, after, pelvis_err, joint_indices


def yaw_align_smpl(input_path, output_path, total_mass=75.0, warmup_frames=100,
                   contact_threshold_kg=2.0, iterations=3, converge_deg=0.05,
                   verbose=True):
    """Iterate the closed-form yaw solver until the per-pass angle drops below
    `converge_deg` or `iterations` is reached. Each pass corrects ~90% of the
    remaining misalignment (the residual comes from non-linear filter
    interactions inside the processor that don't commute exactly with rotation).
    """
    if verbose:
        print(f"  {os.path.basename(input_path)}")

    d = np.load(input_path, allow_pickle=True)
    poses = d['poses']
    trans_orig = d['trans'].astype(np.float64)
    fps = float(d['mocap_framerate']) if 'mocap_framerate' in d.files else 120.0
    betas = d['betas'] if 'betas' in d.files else None
    gender = str(d['gender']) if 'gender' in d.files else 'neutral'
    T = len(poses)

    if verbose:
        print(f"    frames={T} fps={fps} gender={gender}")

    trans_cur = trans_orig.copy()
    total_theta = 0.0
    for it in range(1, iterations + 1):
        if verbose:
            print(f"  ── iteration {it}/{iterations} ──")
            print(f"    running smpl_processor (contacts + world positions)…")
        theta, n_pairs, before, after, pelvis_err, joints_used = solve_yaw_once(
            poses, trans_cur, fps, betas, gender, total_mass,
            warmup_frames, contact_threshold_kg, verbose
        )
        if verbose:
            print(f"    sanity ||wp_pelvis - trans|| at f{warmup_frames}: {pelvis_err:.4f} m")
            print(f"    planted pairs: {n_pairs}  joints used: {joints_used}")
            print(f"    yaw this pass: {np.degrees(theta):+.4f}°  "
                  f"skating {before:.3f} → {after:.3f} m "
                  f"({(1.0 - after / max(before, 1e-9)) * 100:+.1f}%)")
        trans_cur = apply_yaw(trans_cur, theta)
        total_theta += theta
        if abs(np.degrees(theta)) < converge_deg:
            if verbose:
                print(f"    converged (|θ|<{converge_deg}°)")
            break

    if verbose:
        print(f"  cumulative yaw correction: {np.degrees(total_theta):+.4f}°")

    out = {k: d[k] for k in d.files}
    out['trans'] = trans_cur
    out['yaw_correction_deg'] = np.float64(np.degrees(total_theta))
    np.savez(output_path, **out)
    if verbose:
        print(f"  → Saved: {os.path.basename(output_path)}")

    return total_theta


def main():
    p = argparse.ArgumentParser(
        description='Optimal yaw alignment of trans to minimize planted-foot skating.')
    p.add_argument('files', nargs='*', help='SMPL .npz files to align')
    p.add_argument('--dir', help='Directory of SMPL .npz files')
    p.add_argument('--output-dir', help='Output directory (default: same as input)')
    p.add_argument('--mass', type=float, default=75.0,
                   help='Subject total mass in kg (default: 75.0)')
    p.add_argument('--warmup', type=int, default=100,
                   help='Skip this many initial frames during contact eval (default: 100)')
    p.add_argument('--contact-threshold', type=float, default=2.0,
                   help='Min contact pressure (kg) to count as planted (default: 2.0)')
    p.add_argument('--iterations', type=int, default=3,
                   help='Max passes; ~90%% recovery per pass, so 3 → ~99.9%% (default: 3)')
    p.add_argument('--converge-deg', type=float, default=0.05,
                   help='Early-stop when |yaw|<this many degrees (default: 0.05)')
    args = p.parse_args()

    files = list(args.files)
    if args.dir:
        files += [os.path.join(args.dir, f) for f in sorted(os.listdir(args.dir))
                  if f.endswith('.npz')]
    if not files:
        p.print_help()
        return

    for f in files:
        if not os.path.exists(f):
            print(f"  ⚠️  Not found: {f}")
            continue

        if args.output_dir:
            base = os.path.splitext(os.path.basename(f))[0]
            out = os.path.join(args.output_dir, base + '_yaw_aligned.npz')
        else:
            base = os.path.splitext(f)[0]
            out = base + '_yaw_aligned.npz'

        try:
            yaw_align_smpl(f, out,
                           total_mass=args.mass,
                           warmup_frames=args.warmup,
                           contact_threshold_kg=args.contact_threshold,
                           iterations=args.iterations,
                           converge_deg=args.converge_deg)
        except Exception as e:
            print(f"  ❌ Error: {f}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == '__main__':
    main()
