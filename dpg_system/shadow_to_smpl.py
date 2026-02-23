#!/usr/bin/env python3
"""
Convert Shadow motion capture files to AMASS-structured SMPL npz files.

Shadow files contain:
  - quats: (T, 37, 4) LOCAL quaternions (y-up, wxyz)
  - positions: (T, 37, 3) joint world positions (y-up)

The Shadow system outputs parent-relative (local) quaternions.
We reindex them from Shadow→bmolab active→SMPL joint ordering,
convert the root orientation and translation from y-up to z-up,
and output axis-angle rotations in AMASS format.

Usage:
    python shadow_to_smpl.py input.npz [output.npz]
    python shadow_to_smpl.py --dir /path/to/shadow_files
"""

import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import os


# ── Joint index mappings ──────────────────────────────────────────────────

# active_joint_index → shadow_file_index (from motion_cap_nodes.py)
# 20 active joints extracted from 37 Shadow joints
ACTIVE_TO_SHADOW = [
    2,   # active 0  = shadow 2  = BaseOfSkull       → SMPL head (15)
    17,  # active 1  = shadow 17 = UpperVertebrae    → SMPL neck (12)
    1,   # active 2  = shadow 1  = MidVertebrae      → SMPL spine3 (9)
    32,  # active 3  = shadow 32 = LowerVertebrae    → SMPL spine2 (6)
    31,  # active 4  = shadow 31 = SpinePelvis       → SMPL spine1 (3)
    4,   # active 5  = shadow 4  = PelvisAnchor      → SMPL pelvis (0)
    14,  # active 6  = shadow 14 = LeftHip           → SMPL left_hip (1)
    12,  # active 7  = shadow 12 = LeftKnee          → SMPL left_knee (4)
    8,   # active 8  = shadow 8  = LeftAnkle         → SMPL left_ankle (7)
    28,  # active 9  = shadow 28 = RightHip          → SMPL right_hip (2)
    26,  # active 10 = shadow 26 = RightKnee         → SMPL right_knee (5)
    22,  # active 11 = shadow 22 = RightAnkle        → SMPL right_ankle (8)
    13,  # active 12 = shadow 13 = LeftShoulderBlade → SMPL left_collar (13)
    5,   # active 13 = shadow 5  = LeftShoulder      → SMPL left_shoulder (16)
    9,   # active 14 = shadow 9  = LeftElbow         → SMPL left_elbow (18)
    10,  # active 15 = shadow 10 = LeftWrist         → SMPL left_wrist (20)
    27,  # active 16 = shadow 27 = RightShoulderBlade→ SMPL right_collar (14)
    19,  # active 17 = shadow 19 = RightShoulder     → SMPL right_shoulder (17)
    23,  # active 18 = shadow 23 = RightElbow        → SMPL right_elbow (19)
    24,  # active 19 = shadow 24 = RightWrist        → SMPL right_wrist (21)
]

# bmolab active index → SMPL index (for the 20 active joints that have Shadow data)
ACTIVE_TO_SMPL = {
    0: 15,   # BaseOfSkull       → head
    1: 12,   # UpperVertebrae    → neck
    2: 9,    # MidVertebrae      → spine3
    3: 6,    # LowerVertebrae    → spine2
    4: 3,    # SpinePelvis       → spine1
    5: 0,    # PelvisAnchor      → pelvis
    6: 1,    # LeftHip           → left_hip
    7: 4,    # LeftKnee          → left_knee
    8: 7,    # LeftAnkle         → left_ankle
    9: 2,    # RightHip          → right_hip
    10: 5,   # RightKnee         → right_knee
    11: 8,   # RightAnkle        → right_ankle
    12: 13,  # LeftShoulderBlade → left_collar
    13: 16,  # LeftShoulder      → left_shoulder
    14: 18,  # LeftElbow         → left_elbow
    15: 20,  # LeftWrist         → left_wrist
    16: 14,  # RightShoulderBlade→ right_collar
    17: 17,  # RightShoulder     → right_shoulder
    18: 19,  # RightElbow        → right_elbow
    19: 21,  # RightWrist        → right_wrist
}

# Full chain: shadow_file_index → SMPL_index
SHADOW_TO_SMPL = {}
for active_i, shadow_i in enumerate(ACTIVE_TO_SHADOW):
    if active_i in ACTIVE_TO_SMPL:
        SHADOW_TO_SMPL[shadow_i] = ACTIVE_TO_SMPL[active_i]

SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand'
]

# Rotation to convert y-up global orientation to z-up: -90° about X
# This is applied ONLY to the root (pelvis) global rotation
_R_YUP_TO_ZUP = Rotation.from_euler('X', 90, degrees=True)


def yup_to_zup_position(pos):
    """Convert y-up positions to z-up: (x, y, z) → (x, z, -y)."""
    result = np.empty_like(pos)
    result[..., 0] = pos[..., 0]    # x → x
    result[..., 1] = pos[..., 2]    # z → new y
    result[..., 2] = -pos[..., 1]   # -y → new z
    return result


def convert_shadow_to_smpl(shadow_path, output_path=None, fps=100.0, verbose=True):
    """
    Convert a Shadow mocap .npz file to AMASS-structured SMPL .npz.

    Shadow quaternions are LOCAL (parent-relative), wxyz format, y-up.
    Output is AMASS format with z-up axis-angle local rotations.
    """
    if output_path is None:
        base = os.path.splitext(shadow_path)[0]
        output_path = base + '_smpl_poses.npz'

    data = np.load(shadow_path, allow_pickle=True)

    if 'quats' not in data or 'positions' not in data:
        raise ValueError(f"Not a Shadow file — missing 'quats' or 'positions': {shadow_path}")

    quats = data['quats']          # (T, 37, 4) local wxyz y-up
    positions = data['positions']  # (T, 37, 3) world positions y-up
    T = quats.shape[0]

    if verbose:
        print(f"  {os.path.basename(shadow_path)}: {T} frames ({T/fps:.1f}s)")

    # ── Step 1: Extract root translation ────────────────────────────────
    # Apply inverse of the SMPL processor's (x, z, -y) permutation,
    # which is (x, -z, y), so the processor restores original y-up.
    trans_raw = positions[:, 4, :].copy()  # PelvisAnchor at shadow index 4
    trans = np.empty_like(trans_raw)
    trans[:, 0] = trans_raw[:, 0]    # x
    trans[:, 1] = -trans_raw[:, 2]   # -z
    trans[:, 2] = trans_raw[:, 1]    # y

    # ── Step 2: Map Shadow local quats → SMPL joint ordering ──────────
    n_smpl = 24
    smpl_quats_wxyz = np.zeros((T, n_smpl, 4))
    smpl_quats_wxyz[:, :, 0] = 1.0  # identity default (w=1)

    mapped = 0
    for shadow_idx, smpl_idx in SHADOW_TO_SMPL.items():
        smpl_quats_wxyz[:, smpl_idx] = quats[:, shadow_idx]
        mapped += 1

    if verbose:
        print(f"  Mapped {mapped}/{n_smpl} SMPL joints from Shadow")

    # ── Step 3: Convert local quats to axis-angle ─────────────────────
    # The SMPL processor applies r_perm (≈ -90°X) to the root. Pre-apply
    # +90°X so the processor restores the correct y-up orientation.
    local_aa = np.zeros((T, n_smpl, 3))

    for j in range(n_smpl):
        # Convert wxyz → xyzw for scipy
        q_wxyz = smpl_quats_wxyz[:, j]
        q_xyzw = np.concatenate([q_wxyz[:, 1:], q_wxyz[:, :1]], axis=-1)
        r = Rotation.from_quat(q_xyzw)

        if j == 0:
            # Root: pre-apply +90°X to counteract processor's -90°X
            r = _R_YUP_TO_ZUP * r

        local_aa[:, j] = r.as_rotvec()

    # ── Step 4: Unwrap axis-angle discontinuities ─────────────────────
    # as_rotvec() flips the axis direction when angle crosses π.
    # When rotvec jumps by ~2π between frames, replace with the equivalent
    # rotation: rv' = rv * (1 - 2π/|rv|), which has the same rotation
    # but wrapped to the other side of π.
    for j in range(n_smpl):
        rv = local_aa[:, j]  # (T, 3)
        for t in range(1, T):
            diff = rv[t] - rv[t - 1]
            if np.dot(diff, diff) > 9.0:  # ~170° jump threshold
                # Flip: equivalent rotation on the other side of π
                angle = np.linalg.norm(rv[t])
                if angle > 1e-6:
                    rv[t] = rv[t] * (angle - 2 * np.pi) / angle

    # ── Step 4: Pack into AMASS format (156-dim for SMPL+H) ───────────
    n_smplh = 52
    poses_full = np.zeros((T, n_smplh, 3))
    poses_full[:, :n_smpl] = local_aa
    poses = poses_full.reshape(T, -1)  # (T, 156)

    # ── Step 5: Save ──────────────────────────────────────────────────
    np.savez(
        output_path,
        poses=poses,
        trans=trans,
        mocap_framerate=np.float64(fps),
        gender='neutral',
        betas=np.zeros(16, dtype=np.float64),
        dmpls=np.zeros((T, 8), dtype=np.float64),
    )

    if verbose:
        print(f"  → Saved: {os.path.basename(output_path)}")
        print(f"    poses: {poses.shape}, trans: {trans.shape}")
        print(f"    trans[0] = [{trans[0,0]:.3f}, {trans[0,1]:.3f}, {trans[0,2]:.3f}] (z-up)")

    return output_path


def main():
    p = argparse.ArgumentParser(description='Convert Shadow mocap to AMASS SMPL format')
    p.add_argument('files', nargs='*', help='Shadow .npz files to convert')
    p.add_argument('--dir', help='Directory of Shadow .npz files')
    p.add_argument('--fps', type=float, default=100.0, help='Framerate (default: 100)')
    p.add_argument('--output-dir', help='Output directory (default: same as input)')
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

        # Skip files that are already SMPL format
        d = np.load(f, allow_pickle=True)
        if 'quats' not in d.files:
            print(f"  Skipping (not Shadow format): {os.path.basename(f)}")
            continue

        if args.output_dir:
            base = os.path.splitext(os.path.basename(f))[0]
            out = os.path.join(args.output_dir, base + '_smpl_poses.npz')
        else:
            out = None

        try:
            convert_shadow_to_smpl(f, out, fps=args.fps)
        except Exception as e:
            print(f"  ❌ Error: {f}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == '__main__':
    main()
