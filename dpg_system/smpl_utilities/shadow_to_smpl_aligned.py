#!/usr/bin/env python3
"""
Convert Shadow motion capture files to AMASS-structured SMPL npz files.

Both Shadow and SMPL have rest poses with all-identity local quaternions, so
the world orientation of every joint at rest is identity in both skeletons.
For most joints the topological mapping between Shadow's active joints and
SMPL's body joints is one-to-one along each chain, and the correct retargeting
is a direct per-joint copy of the local rotation.

An earlier version of this script applied a per-joint change-of-basis
``q_smpl = D * q_shadow * D⁻¹`` to "re-express axes". That conjugation rotated
collar / neck rotations off-axis by 21-28° and made the shoulders visibly
mistranslated — the basic copy is mathematically equivalent to proper
global-orientation retargeting here and gives the correct result.

Exceptions: the Shadow T-pose has slightly bent elbows and a small amount of
shoulder roll where the SMPL T-pose is perfectly flat. A direct copy therefore
over-extends the SMPL elbows backwards. Per-joint bind-pose offsets at the
shoulders and elbows (see ``_BIND_OFFSETS_WXYZ`` below) are pre-multiplied in
the parent frame to correct this. The values were measured by matching T-pose
renderings via the ``pose_adjust`` node.

The Shadow calibration pose is T-pose: arms straight out, palms down,
feet together.  All local quaternions are identity in this pose.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import os

# ── Joint index mappings (same as shadow_to_smpl.py) ─────────────────────

ACTIVE_TO_SHADOW = [
    2, 17, 1, 32, 31, 4, 14, 12, 8, 28, 26, 22,
    13, 5, 9, 10, 27, 19, 23, 24,
]

ACTIVE_TO_SMPL = {
    0: 15, 1: 12, 2: 9, 3: 6, 4: 3, 5: 0,
    6: 1, 7: 4, 8: 7, 9: 2, 10: 5, 11: 8,
    12: 13, 13: 16, 14: 18, 15: 20,
    16: 14, 17: 17, 18: 19, 19: 21,
}

SHADOW_TO_SMPL = {}
for active_i, shadow_i in enumerate(ACTIVE_TO_SHADOW):
    if active_i in ACTIVE_TO_SMPL:
        SHADOW_TO_SMPL[shadow_i] = ACTIVE_TO_SMPL[active_i]

_R_YUP_TO_ZUP = Rotation.from_euler('X', 90, degrees=True)

# Bind-pose offsets between Shadow's rest pose and SMPL's rest pose.
# Shadow's T-pose has slightly bent elbows (and a hair of shoulder roll),
# so a direct copy of the local quat at rest produces over-extended SMPL
# elbows. These were measured by matching T-pose renderings via the
# pose_adjust node and are pre-multiplied in the parent's local frame
# (wxyz). Keys are SMPL joint indices.
_BIND_OFFSETS_WXYZ = {
    13: [1.0, 0.0,  0.025, 0.0],  # left_collar
    14: [1.0, 0.0, -0.025, 0.0],  # right_collar
    16: [1.0, 0.0, -0.050, 0.0],  # left_shoulder
    17: [1.0, 0.0,  0.050, 0.0],  # right_shoulder
    18: [1.0, 0.0, -0.100, 0.0],  # left_elbow
    19: [1.0, 0.0,  0.100, 0.0],  # right_elbow
    20: [1.0, 0.0,  0.050, 0.0],  # left_wrist
    21: [1.0, 0.0, -0.050, 0.0],  # right_wrist
}


def _wxyz_to_rotation(q_wxyz):
    q = np.asarray(q_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


_BIND_OFFSETS = {j: _wxyz_to_rotation(q) for j, q in _BIND_OFFSETS_WXYZ.items()}


def _resolve_meta(data, fps, gender, betas, verbose):
    """Fill missing fps/gender/betas from the loaded npz's own fields.

    Explicit (non-None) arguments win.  File values are used as fallback.
    Final fallbacks: 100 fps, 'neutral', zeros-equivalent (None).
    """
    auto = []
    if fps is None:
        if 'mocap_framerate' in data.files:
            fps = float(data['mocap_framerate'])
            auto.append(f'fps={fps:g}')
        else:
            fps = 100.0
    if gender is None:
        if 'gender' in data.files:
            g = data['gender']
            if isinstance(g, np.ndarray):
                g = g.item() if g.ndim == 0 else g.flat[0]
            if isinstance(g, bytes):
                g = g.decode('utf-8', errors='replace')
            g = str(g).lower().strip().strip("'\"")
            gender = g if g in ('male', 'female', 'neutral') else 'neutral'
            auto.append(f'gender={gender}')
        else:
            gender = 'neutral'
    if betas is None and 'betas' in data.files:
        b = np.asarray(data['betas'])
        if b.size > 0:
            betas = b
            auto.append(f'betas[{b.size}]')
    if verbose and auto:
        print(f"  Picked up from file: {', '.join(auto)}")
    return fps, gender, betas


def convert_shadow_to_smpl_aligned(shadow_path, output_path=None, fps=None,
                                    gender=None, betas=None, floor=None,
                                    verbose=True):
    """Convert Shadow mocap to AMASS SMPL format.

    Per-joint local quaternions are copied directly from Shadow to the matching
    SMPL joint; only the root receives a Y-up→Z-up rotation so the rest of the
    AMASS pipeline lands in Z-up.

    `fps`, `gender`, and `betas` are auto-detected from the input file's
    `mocap_framerate` / `gender` / `betas` fields when not explicitly given;
    an explicit argument always wins.  Final fallbacks are 100 fps, neutral,
    and zeros.
    """
    if output_path is None:
        base = os.path.splitext(shadow_path)[0]
        output_path = base + '_smpl_poses_aligned.npz'

    data = np.load(shadow_path, allow_pickle=True)
    if 'quats' not in data or ('positions' not in data and 'trans' not in data):
        raise ValueError(f"Not a Shadow file: {shadow_path}")

    fps, gender, betas = _resolve_meta(data, fps, gender, betas, verbose)

    quats = data['quats']          # (T, N, 4) local wxyz y-up
    T = quats.shape[0]
    n_quats = quats.shape[1]

    if n_quats == 37:
        fmt = 'shadow'
        pelvis_idx = 4   # shadow index for PelvisAnchor
        joint_map = SHADOW_TO_SMPL
    elif n_quats == 20:
        fmt = 'active'
        pelvis_idx = 5   # bmolab active index for pelvis_anchor
        joint_map = ACTIVE_TO_SMPL
    else:
        raise ValueError(
            f"Unexpected quat count {n_quats} in {shadow_path}; "
            f"expected 37 (shadow format) or 20 (active format)"
        )

    if verbose:
        print(f"  {os.path.basename(shadow_path)}: {T} frames ({T/fps:.1f}s) [{fmt}]")

    # Root translation: Y-up → Z-up = (x, -z, y).  Aligned active files store
    # the root translation directly as 'trans' (same Y-up frame as quats);
    # shadow files store full joint 'positions' from which we pull the
    # pelvis_anchor row.
    if 'positions' in data:
        trans_raw = data['positions'][:, pelvis_idx, :].copy()
    else:
        trans_raw = data['trans'].copy()
    trans = np.empty_like(trans_raw)
    trans[:, 0] = trans_raw[:, 0]
    trans[:, 1] = -trans_raw[:, 2]
    trans[:, 2] = trans_raw[:, 1]
    if floor is not None:
        trans[:, 2] -= floor

    n_smpl = 24
    local_aa = np.zeros((T, n_smpl, 3))

    # Map Shadow quats → SMPL joint order (wxyz, identity default).
    smpl_quats_wxyz = np.zeros((T, n_smpl, 4))
    smpl_quats_wxyz[:, :, 0] = 1.0
    mapped = 0
    for src_idx, smpl_idx in joint_map.items():
        smpl_quats_wxyz[:, smpl_idx] = quats[:, src_idx]
        mapped += 1
    if verbose:
        print(f"  Mapped {mapped}/{n_smpl} SMPL joints from {fmt}")

    # Per-joint conversion to axis-angle.  Root gets Y-up → Z-up pre-applied.
    for j in range(n_smpl):
        q_wxyz = smpl_quats_wxyz[:, j]
        q_xyzw = np.concatenate([q_wxyz[:, 1:], q_wxyz[:, :1]], axis=-1)
        r = Rotation.from_quat(q_xyzw)
        if j == 0:
            r = _R_YUP_TO_ZUP * r
        elif j in _BIND_OFFSETS:
            r = _BIND_OFFSETS[j] * r
        local_aa[:, j] = r.as_rotvec()

    # ── Step 6: Unwrap axis-angle discontinuities ─────────────────────
    for j in range(n_smpl):
        rv = local_aa[:, j]
        for t in range(1, T):
            diff = rv[t] - rv[t - 1]
            if np.dot(diff, diff) > 9.0:
                angle = np.linalg.norm(rv[t])
                if angle > 1e-6:
                    rv[t] = rv[t] * (angle - 2 * np.pi) / angle

    # ── Step 7: Pack into AMASS format ────────────────────────────────
    n_smplh = 52
    poses_full = np.zeros((T, n_smplh, 3))
    poses_full[:, :n_smpl] = local_aa
    poses = poses_full.reshape(T, -1)

    # ── Step 8: Save ──────────────────────────────────────────────────
    betas_out = np.zeros(16, dtype=np.float64)
    if betas is not None:
        b = np.array(betas, dtype=np.float64).flatten()
        betas_out[:min(len(b), 16)] = b[:16]

    np.savez(
        output_path,
        poses=poses,
        trans=trans,
        mocap_framerate=np.float64(fps),
        gender=gender,
        betas=betas_out,
        dmpls=np.zeros((T, 8), dtype=np.float64),
    )

    if verbose:
        print(f"  → Saved: {os.path.basename(output_path)}")
        print(f"    poses: {poses.shape}, trans: {trans.shape}")

    return output_path


def main():
    p = argparse.ArgumentParser(
        description='Convert Shadow mocap to AMASS SMPL format with T-pose alignment')
    p.add_argument('files', nargs='*', help='Shadow .npz files to convert')
    p.add_argument('--dir', help='Directory of Shadow .npz files')
    p.add_argument('--fps', type=float, default=None,
                   help='Framerate.  If omitted, taken from the file\'s '
                        '`mocap_framerate` field (else 100).')
    p.add_argument('--output-dir', help='Output directory (default: same as input)')
    p.add_argument('--gender', type=str, default=None,
                   choices=['male', 'female', 'neutral'],
                   help='Subject gender.  If omitted, taken from the file\'s '
                        '`gender` field (else neutral).')
    p.add_argument('--betas', type=str, default=None,
                   help='Path to .npy file with SMPL beta parameters.  If '
                        'omitted, taken from the file\'s `betas` field '
                        '(else zeros).')
    p.add_argument('--floor', type=float, default=None,
                   help='Floor height to subtract from the trans Z (height) component per frame')
    args = p.parse_args()

    betas = None
    if args.betas:
        if not os.path.isfile(args.betas):
            print(f'Error: betas file not found: {args.betas}')
            return
        betas = np.load(args.betas, allow_pickle=True)
        # Handle dict-wrapped betas (e.g. from robust mean estimation)
        if isinstance(betas, np.ndarray) and betas.ndim == 0:
            betas = betas.item()  # unwrap 0-d object array
        if isinstance(betas, dict):
            for key in ('betas', 'mean', 'robust_mean'):
                if key in betas:
                    betas = betas[key]
                    break
            else:
                # fallback: use first array-like value
                for v in betas.values():
                    betas = np.asarray(v)
                    break
        betas = np.array(betas, dtype=np.float64).flatten()
        print(f'Loaded betas: {betas.shape} values')

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

        d = np.load(f, allow_pickle=True)
        if 'quats' not in d.files:
            print(f"  Skipping (not Shadow format): {os.path.basename(f)}")
            continue

        if args.output_dir:
            base = os.path.splitext(os.path.basename(f))[0]
            out = os.path.join(args.output_dir, base + '_smpl_poses_aligned.npz')
        else:
            out = None

        try:
            convert_shadow_to_smpl_aligned(f, out, fps=args.fps, gender=args.gender,
                                           betas=betas, floor=args.floor)
        except Exception as e:
            print(f"  ❌ Error: {f}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == '__main__':
    main()
