"""
Align mocap quaternion world frame with 3D tracker world frame.
Solves two calibration problems:
1. Yaw alignment: The rotation world (quats) and translation world (trans)
   may have different heading orientations. This script infers the yaw offset
   by correlating pelvis forward direction with walking velocity.
2. Sensor-to-root offset: The 3D tracker is on the back surface; the SMPL
   root (pelvis) is interior. A configurable local-frame offset is rotated
   by pelvis orientation each frame.
Usage:
    python align_mocap_worlds.py input.npz [--output corrected.npz]
    python align_mocap_worlds.py input.npz --walk-start 14320 --offset-z 0.12
"""
import argparse
import numpy as np
from scipy.optimize import minimize_scalar
def quat_rotate(q_wxyz, v):
    """Rotate vector v by quaternion q in (w, x, y, z) format."""
    w, x, y, z = q_wxyz
    t = 2.0 * np.array([
        y * v[2] - z * v[1],
        z * v[0] - x * v[2],
        x * v[1] - y * v[0]
    ])
    return v + w * t + np.cross(np.array([x, y, z]), t)
def quat_rotate_batch(quats_wxyz, v):
    """Rotate a single vector by an array of quaternions. quats: (N, 4), v: (3,) -> (N, 3)."""
    w = quats_wxyz[:, 0]
    xyz = quats_wxyz[:, 1:4]
    t = 2.0 * np.cross(xyz, np.broadcast_to(v, xyz.shape))
    return np.broadcast_to(v, xyz.shape) + w[:, None] * t + np.cross(xyz, t)
def rotate_y_matrix(angle_rad):
    """3x3 rotation matrix around Y axis by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rotate_y_points(points, angle_rad):
    """Rotate (N, 3) points around Y axis."""
    R = rotate_y_matrix(angle_rad)
    return points @ R.T
def infer_yaw_offset(quats, trans, pelvis_idx=5, walk_start=None, walk_end=None,
                     fps=100, speed_threshold=0.5, verbose=True):
    """
    Infer the yaw rotation needed to align translation world with quaternion world.
    Uses walking segments where pelvis forward direction should align with
    ground-plane velocity.
    Args:
        quats: (N, 20, 4) quaternions in wxyz format
        trans: (N, 3) tracker positions
        pelvis_idx: index of pelvis in the active joints (default 5)
        walk_start: first frame of walking segment (None = auto-detect)
        walk_end: last frame of walking segment (None = end of file)
        fps: frame rate
        speed_threshold: minimum speed (m/s) to consider a frame as walking
        verbose: print diagnostics
    Returns:
        yaw_rad: optimal yaw rotation in radians
        diagnostics: dict with alignment statistics
    """
    N = quats.shape[0]
    fwd_rest = np.array([0.0, 0.0, 1.0])  # forward in Y-up
    if walk_start is None:
        walk_start = 0
    if walk_end is None:
        walk_end = N
    # Compute pelvis forward direction for the walking segment
    segment_quats = quats[walk_start:walk_end, pelvis_idx]  # (M, 4)
    fwd_dirs = quat_rotate_batch(segment_quats, fwd_rest)  # (M, 3)
    fwd_dirs[:, 1] = 0  # project to ground plane
    fwd_norms = np.linalg.norm(fwd_dirs, axis=1, keepdims=True)
    fwd_dirs_norm = fwd_dirs / np.where(fwd_norms > 1e-6, fwd_norms, 1)
    # Compute ground-plane velocity
    vel = np.diff(trans[walk_start:walk_end], axis=0) * fps
    vel[:, 1] = 0  # project to ground
    speed = np.linalg.norm(vel, axis=1)
    # Trim forward dirs to match velocity (one fewer frame)
    fwd_walk = fwd_dirs_norm[:-1]
    # Walking mask
    walk_mask = speed > speed_threshold
    n_walking = walk_mask.sum()
    if verbose:
        print(f"Walking frames (speed > {speed_threshold} m/s): {n_walking} of {len(speed)}")
    if n_walking < 10:
        print("WARNING: Too few walking frames for reliable alignment!")
        if speed_threshold > 0.2:
            # Try lower threshold
            return infer_yaw_offset(quats, trans, pelvis_idx, walk_start, walk_end,
                                    fps, speed_threshold=0.2, verbose=verbose)
        return 0.0, {'n_walking': n_walking, 'warning': 'insufficient_data'}
    vel_norm = vel[walk_mask] / speed[walk_mask, None]
    fwd_sel = fwd_walk[walk_mask]
    def compute_residuals(angle_rad):
        """Compute alignment residuals for a given yaw rotation applied to trans."""
        R = rotate_y_matrix(angle_rad)
        vel_rot = (vel[walk_mask]) @ R.T
        vel_rot_h = vel_rot.copy()
        vel_rot_h[:, 1] = 0
        speed_rot = np.linalg.norm(vel_rot_h, axis=1)
        vel_rot_n = vel_rot_h / speed_rot[:, None]
        cross_y = vel_rot_n[:, 0] * fwd_sel[:, 2] - vel_rot_n[:, 2] * fwd_sel[:, 0]
        dot = vel_rot_n[:, 0] * fwd_sel[:, 0] + vel_rot_n[:, 2] * fwd_sel[:, 2]
        return np.arctan2(cross_y, dot)
    def objective(angle_rad):
        residuals = compute_residuals(angle_rad)
        return np.median(np.abs(residuals))
    # Search over full range, but use two-stage: coarse then fine
    # Coarse: test every 5 degrees
    coarse_angles = np.radians(np.arange(-180, 180, 5))
    coarse_scores = [objective(a) for a in coarse_angles]
    best_coarse = coarse_angles[np.argmin(coarse_scores)]
    # Fine: optimize around the best coarse candidate
    result = minimize_scalar(
        objective,
        bounds=(best_coarse - np.radians(10), best_coarse + np.radians(10)),
        method='bounded'
    )
    optimal_yaw = result.x
    # Compute final residuals for diagnostics
    final_residuals = compute_residuals(optimal_yaw)
    diagnostics = {
        'yaw_deg': np.degrees(optimal_yaw),
        'yaw_rad': optimal_yaw,
        'n_walking': n_walking,
        'residual_median_deg': np.degrees(np.median(final_residuals)),
        'residual_std_deg': np.degrees(np.std(final_residuals)),
        'residual_mad_deg': np.degrees(np.median(np.abs(final_residuals - np.median(final_residuals)))),
    }
    if verbose:
        print(f"Optimal yaw: {diagnostics['yaw_deg']:.2f}°")
        print(f"Residual: median={diagnostics['residual_median_deg']:.2f}°, "
              f"std={diagnostics['residual_std_deg']:.2f}°, "
              f"MAD={diagnostics['residual_mad_deg']:.2f}°")
    return optimal_yaw, diagnostics
def apply_sensor_to_root_offset(trans_aligned, quats, offset_local,
                                pelvis_idx=5):
    """
    Compute SMPL root position from aligned sensor position.
    Args:
        trans_aligned: (N, 3) yaw-corrected sensor positions
        quats: (N, 20, 4) quaternions in wxyz format
        offset_local: (3,) offset from sensor to root in pelvis-local frame
        pelvis_idx: index of pelvis in active joints
    Returns:
        root_positions: (N, 3) corrected root positions
    """
    pelvis_quats = quats[:, pelvis_idx]  # (N, 4)
    offsets_world = quat_rotate_batch(pelvis_quats, offset_local)  # (N, 3)
    return trans_aligned + offsets_world
def correct_mocap_file(input_path, output_path=None, walk_start=None, walk_end=None,
                       offset_xyz=(0.0, 0.0, 0.12), pelvis_idx=5, fps=100,
                       yaw_override=None, verbose=True):
    """
    Full correction pipeline: yaw alignment + sensor-to-root offset.
    Args:
        input_path: path to .npz file with 'quats' and 'trans' keys
        output_path: path to save corrected file (None = auto-name)
        walk_start: first frame of walking segment for yaw inference
        walk_end: last frame of walking segment
        offset_xyz: (x, y, z) sensor-to-root offset in pelvis-local meters
        pelvis_idx: pelvis index in active joints
        fps: frame rate
        yaw_override: if set, skip inference and use this yaw (degrees)
        verbose: print diagnostics
    """
    data = np.load(input_path)
    quats = data['quats']  # (N, 20, 4)
    trans = data['trans']  # (N, 3)
    if verbose:
        print(f"Loaded {input_path}")
        print(f"  quats: {quats.shape}, trans: {trans.shape}")
        print(f"  frames: {quats.shape[0]}")
    # Step 1: Yaw alignment
    if yaw_override is not None:
        yaw_rad = np.radians(yaw_override)
        if verbose:
            print(f"Using yaw override: {yaw_override:.2f}°")
        diagnostics = {'yaw_deg': yaw_override, 'yaw_rad': yaw_rad, 'override': True}
    else:
        yaw_rad, diagnostics = infer_yaw_offset(
            quats, trans, pelvis_idx=pelvis_idx,
            walk_start=walk_start, walk_end=walk_end,
            fps=fps, verbose=verbose
        )
    trans_aligned = rotate_y_points(trans, yaw_rad)
    # Step 2: Sensor-to-root offset
    offset_local = np.array(offset_xyz, dtype=np.float64)
    root_positions = apply_sensor_to_root_offset(
        trans_aligned, quats, offset_local, pelvis_idx=pelvis_idx
    )
    if verbose:
        print(f"\nSensor-to-root offset (local): {offset_local}")
        diff = root_positions - trans_aligned
        print(f"Offset in world frame (mean): "
              f"X={diff[:, 0].mean():.4f}, Y={diff[:, 1].mean():.4f}, Z={diff[:, 2].mean():.4f}")
        print(f"Offset magnitude (mean): {np.linalg.norm(diff, axis=1).mean():.4f} m")
    # Build output
    if output_path is None:
        output_path = input_path.replace('.npz', '_aligned.npz')
    # Preserve any extra keys from the original file
    out_dict = {k: data[k] for k in data.files}
    out_dict['trans'] = root_positions
    out_dict['trans_sensor_aligned'] = trans_aligned  # keep the sensor position too
    out_dict['trans_raw'] = trans  # keep original
    out_dict['alignment_yaw_deg'] = np.array(diagnostics['yaw_deg'])
    np.savez(output_path, **out_dict)
    if verbose:
        print(f"\nSaved corrected file to {output_path}")
        print(f"  Keys: {list(out_dict.keys())}")
    return output_path, diagnostics
def main():
    parser = argparse.ArgumentParser(
        description='Align mocap rotation and translation world frames')
    parser.add_argument('input', help='Input .npz file with quats and trans')
    parser.add_argument('--output', '-o', help='Output .npz path')
    parser.add_argument('--walk-start', type=int, default=None,
                        help='First frame of walking segment for yaw inference')
    parser.add_argument('--walk-end', type=int, default=None,
                        help='Last frame of walking segment')
    parser.add_argument('--offset-x', type=float, default=0.0,
                        help='Sensor-to-root offset X (right) in meters')
    parser.add_argument('--offset-y', type=float, default=0.0,
                        help='Sensor-to-root offset Y (up) in meters')
    parser.add_argument('--offset-z', type=float, default=0.12,
                        help='Sensor-to-root offset Z (forward) in meters')
    parser.add_argument('--pelvis-idx', type=int, default=5,
                        help='Pelvis index in active joints array')
    parser.add_argument('--fps', type=int, default=100,
                        help='Frame rate of the recording')
    parser.add_argument('--yaw', type=float, default=None,
                        help='Override yaw angle in degrees (skip inference)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze alignment, do not save')
    args = parser.parse_args()
    if args.analyze_only:
        data = np.load(args.input)
        quats = data['quats']
        trans = data['trans']
        yaw_rad, diag = infer_yaw_offset(
            quats, trans, pelvis_idx=args.pelvis_idx,
            walk_start=args.walk_start, walk_end=args.walk_end,
            fps=args.fps, verbose=True
        )
        print(f"\nDiagnostics: {diag}")
    else:
        correct_mocap_file(
            args.input,
            output_path=args.output,
            walk_start=args.walk_start,
            walk_end=args.walk_end,
            offset_xyz=(args.offset_x, args.offset_y, args.offset_z),
            pelvis_idx=args.pelvis_idx,
            fps=args.fps,
            yaw_override=args.yaw,
            verbose=True
        )
if __name__ == '__main__':
    main()
