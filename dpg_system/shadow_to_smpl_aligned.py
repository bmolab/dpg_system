#!/usr/bin/env python3
"""
Convert Shadow motion capture files to AMASS-structured SMPL npz files.

This version applies T-pose alignment corrections using global-orientation
retargeting: for each frame, we compute the global orientation of each bone
in the Shadow skeleton, extract the deviation from Shadow's rest pose, and
apply that same deviation to SMPL's rest pose. This correctly handles the
structural differences between the two skeletons.

The Shadow calibration pose is T-pose: arms straight out, palms down,
feet together. All local quaternions are identity in this pose.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import smplx

from dpg_system.body_defs import JointTranslator

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

# SMPL parent hierarchy
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
    18, 19, 20, 21,
]

# Shadow parent hierarchy (in bmolab active joint space, indices 0-19)
# Built from the definition.xml nesting and joint.py set_immediate_children
# pelvis_anchor(5) -> spine_pelvis(4) -> lower_vert(3) -> mid_vert(2) -> upper_vert(1) -> base_of_skull(0)
#                                                          mid_vert(2) -> l_shoulder_blade(12) -> l_shoulder(13) -> l_elbow(14) -> l_wrist(15)
#                                                          mid_vert(2) -> r_shoulder_blade(16) -> r_shoulder(17) -> r_elbow(18) -> r_wrist(19)
# pelvis_anchor(5) -> l_hip(6) -> l_knee(7) -> l_ankle(8)
# pelvis_anchor(5) -> r_hip(9) -> r_knee(10) -> r_ankle(11)
SHADOW_ACTIVE_PARENTS = {
    0: 1,    # base_of_skull -> upper_vertebrae
    1: 2,    # upper_vertebrae -> mid_vertebrae
    2: 3,    # mid_vertebrae -> lower_vertebrae
    3: 4,    # lower_vertebrae -> spine_pelvis
    4: 5,    # spine_pelvis -> pelvis_anchor
    5: -1,   # pelvis_anchor (root)
    6: 5,    # left_hip -> pelvis_anchor
    7: 6,    # left_knee -> left_hip
    8: 7,    # left_ankle -> left_knee
    9: 5,    # right_hip -> pelvis_anchor
    10: 9,   # right_knee -> right_hip
    11: 10,  # right_ankle -> right_knee
    12: 2,   # left_shoulder_blade -> mid_vertebrae
    13: 12,  # left_shoulder -> left_shoulder_blade
    14: 13,  # left_elbow -> left_shoulder
    15: 14,  # left_wrist -> left_elbow
    16: 2,   # right_shoulder_blade -> mid_vertebrae
    17: 16,  # right_shoulder -> right_shoulder_blade
    18: 17,  # right_elbow -> right_shoulder
    19: 18,  # right_wrist -> right_elbow
}

# bmolab active index -> SMPL index
BMOLAB_TO_SMPL = {}
for smpl_name, bmolab_name in JointTranslator.smpl_to_bmolab_active_joint_map.items():
    if smpl_name in JointTranslator.smpl_joints and bmolab_name in JointTranslator.bmolab_active_joints:
        si = JointTranslator.smpl_joints[smpl_name]
        bi = JointTranslator.bmolab_active_joints[bmolab_name]
        if si < 24 and bi < 20:
            BMOLAB_TO_SMPL[bi] = si

# Inverse: SMPL index -> bmolab active index
SMPL_TO_BMOLAB = {v: k for k, v in BMOLAB_TO_SMPL.items()}

_R_YUP_TO_ZUP = Rotation.from_euler('X', 90, degrees=True)


def load_shadow_offsets():
    """Load Shadow bone offset vectors from definition.xml, keyed by bmolab active index.
    These are the bone translation vectors in Shadow's T-pose (Y-up, in meters)."""
    base_dir = Path(__file__).resolve().parent
    def_path = base_dir / 'definition.xml'
    if not def_path.exists():
        raise FileNotFoundError(f"definition.xml not found at {def_path}")

    tree = ET.parse(str(def_path))
    root = tree.getroot()
    offsets = {}
    for node in root.iter('node'):
        if 'translate' in node.attrib:
            limb_name = node.attrib['id']
            ji = JointTranslator.shadow_limb_name_to_bmolab_index(limb_name)
            if ji != -1 and ji < 20:  # only active joints
                vals = list(map(float, node.attrib['translate'].split(' ')))
                offsets[ji] = np.array(vals) / 100.0  # cm -> meters
    return offsets


def load_smpl_offsets(gender='neutral', betas=None):
    """Load SMPL rest-pose bone offsets (child_pos - parent_pos) for 24 joints."""
    model_path = os.path.dirname(os.path.abspath(__file__))
    gender_map = {'male': 'MALE', 'female': 'FEMALE', 'neutral': 'MALE'}
    g_tag = gender_map.get(gender, 'MALE')

    model = smplx.create(model_path=model_path, model_type='smplh',
                         gender=g_tag, num_betas=10, ext='pkl')
    betas_tensor = torch.zeros(1, 10)
    if betas is not None:
        b = torch.tensor(betas, dtype=torch.float32).flatten()
        n = min(len(b), 10)
        betas_tensor[0, :n] = b[:n]

    output = model(betas=betas_tensor)
    joints = output.joints[0].detach().cpu().numpy()
    offsets = np.zeros((24, 3))
    for i in range(1, 24):
        offsets[i] = joints[i] - joints[SMPL_PARENTS[i]]
    return offsets


def compute_rest_global_orientations(offsets, parents, n_joints):
    """Compute rest-pose global orientation for each joint from bone offset directions.

    The orientation at each joint is defined by the direction its bone points
    (from the joint toward its child). For joints with no children, we inherit
    the parent orientation.

    We use a minimal-rotation frame: the Z-axis aligns with the bone direction,
    and X/Y are constructed via cross products with a reference vector.

    Returns: list of Rotation objects (global orientations in rest pose).
    """
    # Build children map
    children = {i: [] for i in range(n_joints)}
    for i in range(n_joints):
        p = parents.get(i, parents[i]) if isinstance(parents, dict) else (parents[i] if i < len(parents) else -1)
        if p >= 0:
            children[p].append(i)

    # Compute bone directions (primary child, or first child)
    bone_dirs = {}
    for i in range(n_joints):
        if isinstance(offsets, dict):
            if i in offsets:
                d = offsets[i].copy()
                n = np.linalg.norm(d)
                if n > 1e-6:
                    bone_dirs[i] = d / n
        else:
            d = offsets[i].copy()
            n = np.linalg.norm(d)
            if n > 1e-6:
                bone_dirs[i] = d / n

    return bone_dirs


def build_frame_from_direction(bone_dir, ref=np.array([0., 0., 1.])):
    """Build a rotation that aligns +Z with bone_dir, using ref to resolve twist."""
    z = bone_dir / np.linalg.norm(bone_dir)
    # Pick a reference that isn't parallel to z
    if abs(np.dot(z, ref)) > 0.95:
        ref = np.array([0., 1., 0.])
    x = np.cross(ref, z)
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-6:
        return Rotation.identity()
    x = x / x_norm
    y = np.cross(z, x)
    mat = np.column_stack([x, y, z])
    return Rotation.from_matrix(mat)


def convert_shadow_to_smpl_aligned(shadow_path, output_path=None, fps=100.0,
                                    gender='neutral', betas=None, floor=None,
                                    verbose=True):
    """Convert Shadow mocap to AMASS SMPL format with T-pose retargeting.

    Strategy (global-orientation retargeting):
      1. For each frame, compute global orientation of each Shadow bone
         by chaining local quats along the Shadow hierarchy.
      2. Compute the deviation of each bone from its Shadow rest-pose orientation.
      3. Apply that same deviation to the SMPL rest-pose orientation.
      4. Convert SMPL global orientations back to local quaternions.
    """
    if output_path is None:
        base = os.path.splitext(shadow_path)[0]
        output_path = base + '_smpl_poses_aligned.npz'

    data = np.load(shadow_path, allow_pickle=True)
    if 'quats' not in data or ('positions' not in data and 'trans' not in data):
        raise ValueError(f"Not a Shadow file: {shadow_path}")

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

    # Load skeleton data
    shadow_offsets = load_shadow_offsets()    # bmolab_active_idx -> (3,) vector
    smpl_offsets = load_smpl_offsets(gender, betas)  # (24, 3) array

    # ── Step 1: Extract root translation ────────────────────────────────
    # Aligned active files store the root translation directly as 'trans'
    # (same Y-up coord system as quats); shadow files store full joint
    # 'positions' from which we pull the pelvis_anchor row.
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

    n_active = 20

    # ── Step 3: Compute Shadow rest-pose bone directions ───────────────
    # In the Shadow rendering pipeline (body_base.py draw_to):
    #   translate_along_bone() then glMultMatrixf(transform)
    # The bone direction from joint j is defined by its bone_translation vector.
    # The local quaternion rotates this direction from rest pose.

    # Build topological sort for Shadow hierarchy (root first)
    shadow_topo_order = []
    visited = set()
    def topo_visit_shadow(j):
        if j in visited or j < 0:
            return
        p = SHADOW_ACTIVE_PARENTS.get(j, -1)
        if p >= 0:
            topo_visit_shadow(p)
        visited.add(j)
        shadow_topo_order.append(j)
    for j in range(n_active):
        topo_visit_shadow(j)

    # Build topological sort for SMPL hierarchy
    smpl_topo_order = []
    visited_smpl = set()
    def topo_visit_smpl(j):
        if j in visited_smpl or j < 0:
            return
        p = SMPL_PARENTS[j]
        if p >= 0:
            topo_visit_smpl(p)
        visited_smpl.add(j)
        smpl_topo_order.append(j)
    for j in range(24):
        topo_visit_smpl(j)

    # Compute rest-pose global frames for both skeletons.
    # The bone direction at joint j (in the parent's frame) is the offset vector.
    # In rest pose (identity local rotation), the global orientation at j
    # equals the global orientation of its parent.
    # The bone direction defines how the child bone LOOKS, but the local frame
    # at j for applying rotations is just inherited from the parent in rest pose.
    #
    # So: G_rest[j] = G_rest[parent[j]] (for both systems, since identity = rest).
    # In rest pose, all global orientations are identity.
    #
    # This means the raw local quaternion from Shadow already represents
    # a rotation in the parent's frame. The parent's frame in rest pose is identity
    # (recursively). So the global orientation is simply the chain of local rotations.
    #
    # The DIFFERENCE between skeletons is that the bones point in different
    # directions. When SMPL applies a local rotation to joint j, it rotates
    # all of j's children and their bone offsets. The bone offset of child c
    # from j, in the parent frame, is smpl_offsets[c]. After rotation q_j,
    # the child bone direction becomes q_j * smpl_offsets[c].
    #
    # For the two skeletons to show the same pose, the child bone directions
    # after rotation must match (in some common reference frame).

    # ── Step 4: Global-orientation retargeting ────────────────────────
    n_smpl = 24
    local_aa = np.zeros((T, n_smpl, 3))

    # Precompute rest-pose bone direction rotations for matched joints.
    # D_j = rotation from Shadow's child bone direction to SMPL's child bone direction.
    # This is computed per-joint, looking at the immediate children.
    #
    # For joint j: if Shadow's child bone goes in direction d_s and SMPL's goes in d_m,
    # then D_j = rotation(d_s -> d_m).
    #
    # The retargeted local rotation is: q_smpl_j = D_parent * q_shadow_j * D_j^{-1}
    # where D_parent accounts for the parent's frame difference and D_j^{-1}
    # accounts for this joint's frame difference.
    #
    # IMPORTANT: This formula is only valid when D values are computed correctly
    # AND applied consistently (no skipping). The previous version's bug was
    # skipping some joints while still using their D values for children.
    #
    # We compute D for every matched joint. For joints that don't have a Shadow
    # equivalent, D = identity (the rotation is passed through unchanged).

    # Build Shadow children map in bmolab active space
    shadow_children = {i: [] for i in range(n_active)}
    for i in range(n_active):
        p = SHADOW_ACTIVE_PARENTS.get(i, -1)
        if p >= 0:
            shadow_children[p].append(i)

    # Build SMPL children map
    smpl_children = {i: [] for i in range(n_smpl)}
    for i in range(1, n_smpl):
        smpl_children[SMPL_PARENTS[i]].append(i)

    def rotation_between(v_from, v_to):
        """Minimum rotation from v_from to v_to."""
        v_from = v_from / np.linalg.norm(v_from)
        v_to = v_to / np.linalg.norm(v_to)
        cross = np.cross(v_from, v_to)
        dot = np.dot(v_from, v_to)
        if dot > 0.9999:
            return Rotation.identity()
        if dot < -0.9999:
            perp = np.array([1., 0., 0.])
            if abs(np.dot(v_from, perp)) > 0.9:
                perp = np.array([0., 1., 0.])
            axis = np.cross(v_from, perp)
            axis /= np.linalg.norm(axis)
            return Rotation.from_rotvec(axis * np.pi)
        sn = np.linalg.norm(cross)
        axis = cross / sn
        angle = np.arctan2(sn, dot)
        return Rotation.from_rotvec(axis * angle)

    # Compute D[smpl_j] for each SMPL joint that has a Shadow equivalent.
    # D[j] represents how the child bone directions differ between the two
    # skeletons at joint j's local frame.
    D = [Rotation.identity()] * n_smpl
    D_inv = [Rotation.identity()] * n_smpl

    for smpl_j in range(n_smpl):
        bmolab_j = SMPL_TO_BMOLAB.get(smpl_j, -1)
        if bmolab_j < 0:
            continue

        # Gather child direction pairs
        shadow_dirs = []
        smpl_dirs = []
        for smpl_child in smpl_children[smpl_j]:
            bmolab_child = SMPL_TO_BMOLAB.get(smpl_child, -1)
            if bmolab_child < 0 or bmolab_child not in shadow_offsets:
                continue
            sv = shadow_offsets[bmolab_child]
            mv = smpl_offsets[smpl_child]
            sn = np.linalg.norm(sv)
            mn = np.linalg.norm(mv)
            if sn < 1e-6 or mn < 1e-6:
                continue
            shadow_dirs.append(sv / sn)
            smpl_dirs.append(mv / mn)

        if not shadow_dirs:
            continue

        if len(shadow_dirs) == 1:
            D[smpl_j] = rotation_between(shadow_dirs[0], smpl_dirs[0])
        else:
            try:
                r, _ = Rotation.align_vectors(
                    np.array(smpl_dirs),
                    np.array(shadow_dirs)
                )
                D[smpl_j] = r
            except Exception:
                D[smpl_j] = rotation_between(shadow_dirs[0], smpl_dirs[0])

        D_inv[smpl_j] = D[smpl_j].inv()

    if verbose:
        smpl_names = list(JointTranslator.smpl_joints.keys())
        print("  Rest-pose corrections (D_j):")
        for i in range(n_smpl):
            angle = D[i].magnitude() * 180 / np.pi
            name = smpl_names[i] if i < len(smpl_names) else f'j{i}'
            if angle > 0.1:
                print(f"    {name:20s}: {angle:.1f}°")

    # ── Step 5: Apply retargeting per frame ──────────────────────────
    # Map Shadow quats into SMPL joint order first (as wxyz quaternions)
    smpl_quats_wxyz = np.zeros((T, n_smpl, 4))
    smpl_quats_wxyz[:, :, 0] = 1.0  # identity
    mapped = 0
    for src_idx, smpl_idx in joint_map.items():
        smpl_quats_wxyz[:, smpl_idx] = quats[:, src_idx]
        mapped += 1
    if verbose:
        print(f"  Mapped {mapped}/{n_smpl} SMPL joints from {fmt}")

    # Apply retargeting: q_smpl_j = D[j] * q_shadow_j * D_inv[j]
    #
    # Both Shadow and SMPL have the same visual T-pose (arms out, palms down),
    # so identity must map to identity (D[j]*I*D_inv[j] = I). The correction
    # only re-interprets the rotation axes: a "bend elbow" rotation gets its
    # axis re-expressed in SMPL's local frame where the bone direction may
    # differ slightly from Shadow's.
    #
    # This is a per-joint change-of-basis that does NOT alter the rest pose.
    for t in range(T):
        for j in smpl_topo_order:
            # wxyz -> xyzw for scipy
            q_wxyz = smpl_quats_wxyz[t, j]
            q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
            r_local = Rotation.from_quat(q_xyzw)

            # Per-joint change of basis (identity-preserving)
            r_retargeted = D[j] * r_local * D_inv[j]

            # Root: apply Y-up to Z-up conversion
            if j == 0:
                r_retargeted = _R_YUP_TO_ZUP * r_retargeted

            local_aa[t, j] = r_retargeted.as_rotvec()

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
    p.add_argument('--fps', type=float, default=100.0, help='Framerate (default: 100)')
    p.add_argument('--output-dir', help='Output directory (default: same as input)')
    p.add_argument('--gender', type=str, default='neutral',
                   choices=['male', 'female', 'neutral'],
                   help='Subject gender (default: neutral)')
    p.add_argument('--betas', type=str, default=None,
                   help='Path to .npy file containing SMPL beta parameters')
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
