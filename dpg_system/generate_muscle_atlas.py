#!/usr/bin/env python3
"""
Generate anatomically-driven muscle weight atlas for SMPL-H model.

Usage:
    python generate_muscle_atlas.py --model_path /path/to/smpl/models

Produces:
    muscle_atlas_v3.npy  — shape (6890, N_muscles) float32 weight atlas
    muscle_atlas_v3_meta.npy — metadata dict with muscle names, joints, flex axes

All muscle regions are defined RELATIVE to joint positions so the atlas
adapts to any SMPL model instance.
"""

import argparse
import numpy as np
import torch
import smplx
import os


# ===========================================================================
# SMPL-H joint indices
# 0: Pelvis      1: L_Hip       2: R_Hip       3: Spine1
# 4: L_Knee      5: R_Knee      6: Spine2      7: L_Ankle
# 8: R_Ankle     9: Spine3     10: L_Foot     11: R_Foot
# 12: Neck       13: L_Collar   14: R_Collar   15: Head
# 16: L_Shoulder  17: R_Shoulder  18: L_Elbow   19: R_Elbow
# 20: L_Wrist     21: R_Wrist
# ===========================================================================

# Parent map for kinematic tree
PARENT = {
    1: 0, 2: 0, 3: 0,
    4: 1, 5: 2, 6: 3,
    7: 4, 8: 5, 9: 6,
    10: 7, 11: 8, 12: 9,
    13: 9, 14: 9, 15: 12,
    16: 13, 17: 14,
    18: 16, 19: 17,
    20: 18, 21: 19,
}


def compute_vertex_normals(verts, faces):
    """Compute per-vertex normals from mesh faces."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-10)
    vn = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    vn = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-10)
    return vn


def smooth_step(values, vmin, vmax, falloff):
    """Smooth [0,1] weight: 1.0 inside [vmin,vmax], Gaussian decay outside."""
    w = np.ones(len(values), dtype=np.float32)
    s = max(falloff, 0.001)
    if vmin is not None:
        mask = values < vmin
        w[mask] *= np.exp(-0.5 * ((vmin - values[mask]) / s) ** 2)
    if vmax is not None:
        mask = values > vmax
        w[mask] *= np.exp(-0.5 * ((values[mask] - vmax) / s) ** 2)
    return w


def compute_bone_local_coords(verts, jpos, j_from, j_to):
    """Compute per-vertex axial position (t) along bone and perpendicular distance."""
    a, b = jpos[j_from], jpos[j_to]
    ab = b - a
    bone_len = np.linalg.norm(ab)
    if bone_len < 1e-6:
        return np.zeros(len(verts)), np.zeros(len(verts)), bone_len
    bone_unit = ab / bone_len
    ap = verts - a
    t = (ap @ bone_unit) / bone_len  # 0 at j_from, 1 at j_to
    proj = a + (t[:, np.newaxis] * bone_len) * bone_unit
    perp_dist = np.linalg.norm(verts - proj, axis=1)
    return t, perp_dist, bone_len


# ===========================================================================
# Muscle definitions — joint-relative
#
# Each muscle specifies:
#   name, joint (torque source), bone (j_from, j_to for axial reference),
#   segments (skinning weight joints), direction ('front'/'back'/'left'/'right'),
#   t_range [t_min, t_max] along bone (0=proximal, 1=distal),
#   optional extra_bones for muscles spanning multiple bones (e.g., calf to heel),
#   falloff (meters), max_perp_dist (max perpendicular distance from bone axis)
# ===========================================================================

MUSCLE_DEFS = [
    # ===== LOWER LEG (Ankle torque) =====
    # Calf: posterior lower leg from knee to heel (via Achilles tendon)
    # Includes the main belly (upper 60%) and narrowing to Achilles + heel
    {'name': 'L_Calf', 'joint': 7, 'bone': [4, 7], 'segments': [4, 7],
     'direction': 'back', 't_range': [0.05, 1.0],
     'falloff': 0.05},

    # Shin: anterior lower leg, below knee to just above ankle
    {'name': 'L_TibAnt', 'joint': 7, 'bone': [4, 7], 'segments': [4],
     'direction': 'front', 't_range': [0.15, 0.85],
     'falloff': 0.05},

    {'name': 'R_Calf', 'joint': 8, 'bone': [5, 8], 'segments': [5, 8],
     'direction': 'back', 't_range': [0.05, 1.0],
     'falloff': 0.05},

    {'name': 'R_TibAnt', 'joint': 8, 'bone': [5, 8], 'segments': [5],
     'direction': 'front', 't_range': [0.15, 0.85],
     'falloff': 0.05},

    # Peroneal: lateral lower leg — eversion (turning foot outward)
    {'name': 'L_Peroneal', 'joint': 7, 'bone': [4, 7], 'segments': [4, 7],
     'direction': 'left', 't_range': [0.1, 0.9],
     'falloff': 0.05},

    {'name': 'R_Peroneal', 'joint': 8, 'bone': [5, 8], 'segments': [5, 8],
     'direction': 'right', 't_range': [0.1, 0.9],
     'falloff': 0.05},

    # Tibialis Posterior: medial lower leg — inversion (turning foot inward)
    {'name': 'L_TibPost', 'joint': 7, 'bone': [4, 7], 'segments': [4, 7],
     'direction': 'right', 't_range': [0.1, 0.9],
     'falloff': 0.05},

    {'name': 'R_TibPost', 'joint': 8, 'bone': [5, 8], 'segments': [5, 8],
     'direction': 'left', 't_range': [0.1, 0.9],
     'falloff': 0.05},

    # ===== UPPER LEG (Knee torque) =====
    {'name': 'L_Quad', 'joint': 4, 'bone': [1, 4], 'segments': [1, 4],
     'direction': 'front', 't_range': [0.35, 0.8],
     'falloff': 0.08},

    {'name': 'L_Hamstr', 'joint': 4, 'bone': [1, 4], 'segments': [1, 4],
     'direction': 'back', 't_range': [0.35, 0.8],
     'falloff': 0.08},

    {'name': 'R_Quad', 'joint': 5, 'bone': [2, 5], 'segments': [2, 5],
     'direction': 'front', 't_range': [0.35, 0.8],
     'falloff': 0.08},

    {'name': 'R_Hamstr', 'joint': 5, 'bone': [2, 5], 'segments': [2, 5],
     'direction': 'back', 't_range': [0.35, 0.8],
     'falloff': 0.08},

    # ===== HIP (Hip torque) — Sagittal =====
    # Reference bone: Spine1→Hip gives a tall vertical axis
    # t=[0.5,1.0] maps from pelvis height down to hip joint = one buttock
    # x_side prevents cross-side bleed on shared pelvis vertices
    {'name': 'L_Glute', 'joint': 1, 'bone': [3, 1], 'segments': [0, 1],
     'direction': 'back', 't_range': [0.5, 1.0],
     'falloff': 0.06, 'x_side': 'left'},

    {'name': 'L_HipFlex', 'joint': 1, 'bone': [1, 4], 'segments': [1, 4],
     'direction': 'front', 't_range': [0.0, 0.3],
     'falloff': 0.06, 'x_side': 'left'},

    {'name': 'R_Glute', 'joint': 2, 'bone': [3, 2], 'segments': [0, 2],
     'direction': 'back', 't_range': [0.5, 1.0],
     'falloff': 0.06, 'x_side': 'right'},

    {'name': 'R_HipFlex', 'joint': 2, 'bone': [2, 5], 'segments': [2, 5],
     'direction': 'front', 't_range': [0.0, 0.3],
     'falloff': 0.06, 'x_side': 'right'},

    # ===== HIP — Frontal (abduction/adduction) =====
    # Abductor: lateral upper thigh (gluteus medius surface projection)
    {'name': 'L_HipAbd', 'joint': 1, 'bone': [1, 4], 'segments': [1, 4],
     'direction': 'left', 't_range': [0.0, 0.2],
     'falloff': 0.06, 'x_side': 'left'},

    # Adductor: inner upper thigh
    {'name': 'L_HipAdd', 'joint': 1, 'bone': [1, 4], 'segments': [1, 4],
     'direction': 'right', 't_range': [0.0, 0.3],
     'falloff': 0.06, 'x_side': 'left'},

    {'name': 'R_HipAbd', 'joint': 2, 'bone': [2, 5], 'segments': [2, 5],
     'direction': 'right', 't_range': [0.0, 0.2],
     'falloff': 0.06, 'x_side': 'right'},

    {'name': 'R_HipAdd', 'joint': 2, 'bone': [2, 5], 'segments': [2, 5],
     'direction': 'left', 't_range': [0.0, 0.3],
     'falloff': 0.06, 'x_side': 'right'},

    # ===== HIP — Transverse (internal/external rotation) =====
    # External rotators (deep six): posterior lower buttock, horizontal fan
    # flex_axis is Y-aligned since rotation is around the bone (Y) axis
    {'name': 'L_HipExtRot', 'joint': 1, 'bone': [1, 4], 'segments': [1],
     'direction': 'back', 't_range': [0.0, 0.15],
     'falloff': 0.06, 'x_side': 'left', 'flex_axis': [0, 1, 0]},

    {'name': 'R_HipExtRot', 'joint': 2, 'bone': [2, 5], 'segments': [2],
     'direction': 'back', 't_range': [0.0, 0.15],
     'falloff': 0.06, 'x_side': 'right', 'flex_axis': [0, 1, 0]},

    # Internal rotators (glut med anterior, TFL): lateral upper thigh
    {'name': 'L_HipIntRot', 'joint': 1, 'bone': [1, 4], 'segments': [1],
     'direction': 'left', 't_range': [0.0, 0.15],
     'falloff': 0.06, 'x_side': 'left', 'flex_axis': [0, -1, 0]},

    {'name': 'R_HipIntRot', 'joint': 2, 'bone': [2, 5], 'segments': [2],
     'direction': 'right', 't_range': [0.0, 0.15],
     'falloff': 0.06, 'x_side': 'right', 'flex_axis': [0, -1, 0]},

    # ===== SPINE — Segmented by joint level =====
    # Each spine joint has its own segment of the continuous muscles.
    # Bones: [0,3]=Pelvis→Spine1, [3,6]=Spine1→Spine2, [6,9]=Spine2→Spine3, [9,12]=Spine3→Neck

    # --- Spine1 (joint 3) — Lower torso (waist/belly) ---
    {'name': 'LowerAbs', 'joint': 3, 'bone': [0, 3], 'segments': [0, 3],
     'direction': 'front', 't_range': [-0.1, 0.8],
     'falloff': 0.04, 'max_x_dist': 0.06},

    {'name': 'L_LowerErec', 'joint': 3, 'bone': [0, 3], 'segments': [0, 3],
     'direction': 'back', 't_range': [-0.1, 0.8],
     'falloff': 0.04, 'x_side': 'left', 'max_x_dist': 0.06, 'min_x_dist': 0.015},

    {'name': 'R_LowerErec', 'joint': 3, 'bone': [0, 3], 'segments': [0, 3],
     'direction': 'back', 't_range': [-0.1, 0.8],
     'falloff': 0.04, 'x_side': 'right', 'max_x_dist': 0.06, 'min_x_dist': 0.015},

    {'name': 'L_Obliques', 'joint': 3, 'bone': [0, 3], 'segments': [0, 3],
     'direction': 'left', 't_range': [-0.1, 0.8],
     'falloff': 0.05},

    {'name': 'R_Obliques', 'joint': 3, 'bone': [0, 3], 'segments': [0, 3],
     'direction': 'right', 't_range': [-0.1, 0.8],
     'falloff': 0.05},

    # --- Spine2 (joint 6) — Mid torso (rib cage) ---
    {'name': 'UpperAbs', 'joint': 6, 'bone': [3, 6], 'segments': [3, 6],
     'direction': 'front', 't_range': [0.0, 1.0],
     'falloff': 0.04, 'max_x_dist': 0.06},

    {'name': 'L_MidErec', 'joint': 6, 'bone': [3, 6], 'segments': [3, 6],
     'direction': 'back', 't_range': [0.0, 1.0],
     'falloff': 0.04, 'x_side': 'left', 'max_x_dist': 0.06, 'min_x_dist': 0.015},

    {'name': 'R_MidErec', 'joint': 6, 'bone': [3, 6], 'segments': [3, 6],
     'direction': 'back', 't_range': [0.0, 1.0],
     'falloff': 0.04, 'x_side': 'right', 'max_x_dist': 0.06, 'min_x_dist': 0.015},

    # Pecs and Lats — distinct chest muscles, also at Spine2
    {'name': 'Pecs', 'joint': 6, 'bone': [3, 6], 'segments': [3, 6, 9],
     'direction': 'front', 't_range': [0.5, 1.5],
     'falloff': 0.04},

    {'name': 'Lats', 'joint': 6, 'bone': [3, 6], 'segments': [3, 6, 9],
     'direction': 'back', 't_range': [0.0, 1.5],
     'falloff': 0.04},

    # --- Spine3 (joint 9) — Upper chest/back ---
    {'name': 'UpperChest', 'joint': 9, 'bone': [6, 9], 'segments': [6, 9],
     'direction': 'front', 't_range': [0.0, 1.2],
     'falloff': 0.04, 'max_x_dist': 0.06},

    {'name': 'L_UpperErec', 'joint': 9, 'bone': [6, 9], 'segments': [9],
     'direction': 'back', 't_range': [0.0, 1.2],
     'falloff': 0.04, 'x_side': 'left', 'max_x_dist': 0.06, 'min_x_dist': 0.015},

    {'name': 'R_UpperErec', 'joint': 9, 'bone': [6, 9], 'segments': [9],
     'direction': 'back', 't_range': [0.0, 1.2],
     'falloff': 0.04, 'x_side': 'right', 'max_x_dist': 0.06, 'min_x_dist': 0.015},

    # Sternocleidomastoid (SCM): two narrow bands on front/sides of neck
    {'name': 'L_SCM', 'joint': 12, 'bone': [9, 12], 'segments': [9, 12],
     'direction': 'front', 't_range': [0.0, 1.0],
     'falloff': 0.04, 'x_side': 'left', 'max_x_dist': 0.04, 'min_x_dist': 0.01},

    {'name': 'R_SCM', 'joint': 12, 'bone': [9, 12], 'segments': [9, 12],
     'direction': 'front', 't_range': [0.0, 1.0],
     'falloff': 0.04, 'x_side': 'right', 'max_x_dist': 0.04, 'min_x_dist': 0.01},

    {'name': 'L_NeckErec', 'joint': 12, 'bone': [9, 12], 'segments': [9, 12],
     'direction': 'back', 't_range': [0.0, 1.0],
     'falloff': 0.04, 'x_side': 'left', 'max_x_dist': 0.04, 'min_x_dist': 0.01},

    {'name': 'R_NeckErec', 'joint': 12, 'bone': [9, 12], 'segments': [9, 12],
     'direction': 'back', 't_range': [0.0, 1.0],
     'falloff': 0.04, 'x_side': 'right', 'max_x_dist': 0.04, 'min_x_dist': 0.01},

    # ===== SHOULDER (Deltoids) =====
    {'name': 'L_DeltAnt', 'joint': 16, 'bone': [13, 16], 'segments': [13, 16],
     'direction': 'front', 't_range': [0.3, 1.3],
     'falloff': 0.18, 'max_perp': 0.10},

    {'name': 'L_DeltLat', 'joint': 16, 'bone': [13, 16], 'segments': [13, 16],
     'direction': 'left', 't_range': [0.3, 1.3],
     'falloff': 0.18, 'max_perp': 0.10},

    {'name': 'L_DeltPost', 'joint': 16, 'bone': [13, 16], 'segments': [13, 16],
     'direction': 'back', 't_range': [0.3, 1.3],
     'falloff': 0.18, 'max_perp': 0.10},

    {'name': 'R_DeltAnt', 'joint': 17, 'bone': [14, 17], 'segments': [14, 17],
     'direction': 'front', 't_range': [0.3, 1.3],
     'falloff': 0.18, 'max_perp': 0.10},

    {'name': 'R_DeltLat', 'joint': 17, 'bone': [14, 17], 'segments': [14, 17],
     'direction': 'right', 't_range': [0.3, 1.3],
     'falloff': 0.18, 'max_perp': 0.10},

    {'name': 'R_DeltPost', 'joint': 17, 'bone': [14, 17], 'segments': [14, 17],
     'direction': 'back', 't_range': [0.3, 1.3],
     'falloff': 0.18, 'max_perp': 0.10},

    # ===== UPPER ARM (Elbow torque) =====
    {'name': 'L_Bicep', 'joint': 18, 'bone': [16, 18], 'segments': [16, 18],
     'direction': 'front', 't_range': [0.1, 0.95],
     'falloff': 0.18, 'max_perp': 0.06},

    {'name': 'L_Tricep', 'joint': 18, 'bone': [16, 18], 'segments': [16, 18],
     'direction': 'back', 't_range': [0.1, 0.95],
     'falloff': 0.18, 'max_perp': 0.06},

    {'name': 'R_Bicep', 'joint': 19, 'bone': [17, 19], 'segments': [17, 19],
     'direction': 'front', 't_range': [0.1, 0.95],
     'falloff': 0.18, 'max_perp': 0.06},

    {'name': 'R_Tricep', 'joint': 19, 'bone': [17, 19], 'segments': [17, 19],
     'direction': 'back', 't_range': [0.1, 0.95],
     'falloff': 0.18, 'max_perp': 0.06},

    # ===== FOREARM (Wrist torque) =====
    {'name': 'L_WristFlex', 'joint': 20, 'bone': [18, 20], 'segments': [18, 20],
     'direction': 'front', 't_range': [0.05, 0.9],
     'falloff': 0.245, 'max_perp': 0.05},

    {'name': 'L_WristExt', 'joint': 20, 'bone': [18, 20], 'segments': [18, 20],
     'direction': 'back', 't_range': [0.05, 0.9],
     'falloff': 0.245, 'max_perp': 0.05},

    {'name': 'R_WristFlex', 'joint': 21, 'bone': [19, 21], 'segments': [19, 21],
     'direction': 'front', 't_range': [0.05, 0.9],
     'falloff': 0.245, 'max_perp': 0.05},

    {'name': 'R_WristExt', 'joint': 21, 'bone': [19, 21], 'segments': [19, 21],
     'direction': 'back', 't_range': [0.05, 0.9],
     'falloff': 0.245, 'max_perp': 0.05},
]


def get_direction_vector(direction, bone_dir):
    """Get world-space direction vector for 'front'/'back'/'left'/'right'.

    For vertical bones: front=-Z, back=+Z, left=+X, right=-X
    For horizontal bones (arms): front=-Z, back=+Z, but 'front' also
    includes the upward-facing side (bicep) since the arm is horizontal in T-pose.
    """
    # In SMPL: +Z = forward (toes, belly), -Z = backward (heels, spine)
    world_dirs = {
        'front': np.array([0, 0,  1.0]),   # +Z = anterior/front
        'back':  np.array([0, 0, -1.0]),   # -Z = posterior/back
        'left':  np.array([1, 0,  0.0]),
        'right': np.array([-1, 0, 0.0]),
    }
    ref = world_dirs.get(direction, np.array([0, 0, -1.0]))

    # Project ref perpendicular to bone axis
    bd = bone_dir / max(np.linalg.norm(bone_dir), 1e-6)
    perp = ref - np.dot(ref, bd) * bd
    pl = np.linalg.norm(perp)
    if pl < 1e-6:
        # Reference is parallel to bone — use world up as fallback
        perp = np.array([0, 1, 0]) - np.dot(np.array([0, 1, 0]), bd) * bd
        pl = np.linalg.norm(perp)
    return perp / max(pl, 1e-6)


def compute_flex_axis(direction, bone_dir):
    """Compute the flex axis = cross(bone_dir, off_perp_dir)."""
    bd = bone_dir / max(np.linalg.norm(bone_dir), 1e-6)
    off_perp = get_direction_vector(direction, bone_dir)
    flex = np.cross(bd, off_perp)
    fl = np.linalg.norm(flex)
    if fl > 1e-6:
        return (flex / fl).astype(np.float32)
    return np.array([1, 0, 0], dtype=np.float32)


def compute_muscle_weight(verts, normals, skinning, jpos, mdef, spread_scale=1.0):
    """Compute per-vertex weight for a single muscle using joint-relative coords."""
    n_v = len(verts)
    falloff = mdef.get('falloff', 0.02) * spread_scale
    j_from, j_to = mdef['bone']
    t_min, t_max = mdef['t_range']

    # Segment mask
    seg_w = np.zeros(n_v, dtype=np.float32)
    for sj in mdef['segments']:
        if sj < skinning.shape[1]:
            seg_w += skinning[:, sj]
    seg_mask = np.clip((seg_w - 0.05) / 0.15, 0.0, 1.0)

    # Bone-local coordinates
    bone_dir = jpos[j_to] - jpos[j_from]
    t, perp_dist, bone_len = compute_bone_local_coords(verts, jpos, j_from, j_to)

    # Axial weight (with smooth falloff)
    w_axial = smooth_step(t, t_min, t_max, falloff / max(bone_len, 0.01))

    # Direction filter using surface normals
    dir_vec = get_direction_vector(mdef['direction'], bone_dir)
    dots = normals @ dir_vec
    # Sigmoid threshold — vertices whose normals face the muscle direction
    w_dir = 1.0 / (1.0 + np.exp(-4.0 * dots))

    # Combine (no perpendicular distance decay — the belly of the muscle
    # is the furthest from the bone and should be fully lit, not dimmed)
    w = seg_mask * w_axial * w_dir

    # X-side filter (for left/right split muscles like abs, erectors)
    x_side = mdef.get('x_side')
    if x_side == 'left':
        w *= smooth_step(verts[:, 0], 0.0, None, 0.015)
    elif x_side == 'right':
        w *= smooth_step(-verts[:, 0], 0.0, None, 0.015)

    # Max lateral distance from midline (for narrow columns like erectors)
    max_x_dist = mdef.get('max_x_dist')
    if max_x_dist is not None:
        x_dist = np.abs(verts[:, 0])
        w *= smooth_step(-x_dist, -max_x_dist, None, 0.01)

    # Min lateral distance from midline (creates gap at spine for erectors)
    min_x_dist = mdef.get('min_x_dist')
    if min_x_dist is not None:
        x_dist = np.abs(verts[:, 0])
        w *= smooth_step(x_dist, min_x_dist, None, 0.01)

    # Handle extra_bones (e.g., calf extending to heel via ankle-foot bone)
    if 'extra_bones' in mdef:
        for eb in mdef['extra_bones']:
            eb_from, eb_to = eb['bone']
            eb_tmin, eb_tmax = eb['t_range']
            # Additional segment mask for extra bone
            eb_seg = np.zeros(n_v, dtype=np.float32)
            for sj in eb.get('segments', eb['bone']):
                if sj < skinning.shape[1]:
                    eb_seg += skinning[:, sj]
            eb_seg_mask = np.clip((eb_seg - 0.05) / 0.15, 0.0, 1.0)

            eb_bone_dir = jpos[eb_to] - jpos[eb_from]
            eb_t, eb_perp, eb_blen = compute_bone_local_coords(
                verts, jpos, eb_from, eb_to)
            eb_axial = smooth_step(eb_t, eb_tmin, eb_tmax,
                                    falloff / max(eb_blen, 0.01))
            eb_dir_vec = get_direction_vector(mdef['direction'], eb_bone_dir)
            eb_dots = normals @ eb_dir_vec
            eb_w_dir = 1.0 / (1.0 + np.exp(-15.0 * eb_dots))

            eb_w = eb_seg_mask * eb_axial * eb_w_dir
            w = np.maximum(w, eb_w)  # Union of regions

    return w


def build_vertex_adjacency(faces, n_verts):
    """Build vertex adjacency list from faces."""
    adj = [set() for _ in range(n_verts)]
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[f[i]].add(f[j])
    return adj


def laplacian_smooth(weights, adj, iterations=1, alpha=0.5):
    """Smooth per-vertex weights along mesh topology.
    
    alpha: blend factor (0=no smoothing, 1=full neighbor average)
    """
    w = weights.copy()
    for _ in range(iterations):
        w_new = w.copy()
        for v in range(len(w)):
            neighbors = adj[v]
            if len(neighbors) > 0:
                neighbor_avg = np.mean([w[n] for n in neighbors])
                w_new[v] = (1.0 - alpha) * w[v] + alpha * neighbor_avg
        w = w_new
    return w


def generate_atlas(model_path, gender='male', smooth_iters=0, edge_threshold=0.0):
    """Generate the muscle weight atlas.
    
    Pipeline:
        1. Compute base muscle weights at original sigma values
        2. Laplacian-smooth along mesh topology (smooth_iters iterations)
        3. Subtract edge_threshold baseline
        4. Renormalize so peak = 1.0
    """
    print(f"Loading SMPL-H model from {model_path} ({gender})...")
    model = smplx.create(
        model_path=model_path, model_type='smplh',
        gender=gender.upper(), num_betas=10, ext='pkl'
    )
    model.eval()

    with torch.no_grad():
        output = model()
        verts = output.vertices[0].cpu().numpy()
        jpos = output.joints[0, :24].cpu().numpy()

    faces = np.array(model.faces, dtype=np.int32)
    skinning = model.lbs_weights.cpu().numpy()[:, :24]
    normals = compute_vertex_normals(verts, faces)

    # Build vertex adjacency for mesh smoothing
    adj = None
    if smooth_iters > 0:
        print(f"Building vertex adjacency for {len(verts)} vertices...")
        adj = build_vertex_adjacency(faces, len(verts))

    # Print joint positions
    names = ['Pelvis','L_Hip','R_Hip','Spine1','L_Knee','R_Knee','Spine2',
             'L_Ankle','R_Ankle','Spine3','L_Foot','R_Foot','Neck',
             'L_Collar','R_Collar','Head','L_Shoulder','R_Shoulder',
             'L_Elbow','R_Elbow','L_Wrist','R_Wrist']
    print("\nJoint positions:")
    for i, n in enumerate(names):
        if i < len(jpos):
            p = jpos[i]
            print(f"  {i:2d} {n:13s}: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]")

    # Generate atlas
    n_muscles = len(MUSCLE_DEFS)
    atlas = np.zeros((len(verts), n_muscles), dtype=np.float32)
    muscle_joints = np.zeros(n_muscles, dtype=np.int32)
    flex_axes = np.zeros((n_muscles, 3), dtype=np.float32)
    muscle_names = []

    print(f"\nGenerating atlas for {n_muscles} muscles "
          f"(smooth_iters={smooth_iters}, edge_threshold={edge_threshold}):")
    for i, mdef in enumerate(MUSCLE_DEFS):
        # Step 1: Base muscle form at original sigma
        w = compute_muscle_weight(verts, normals, skinning, jpos, mdef)

        # Normalize base form to max = 1.0
        wmax = w.max()
        if wmax > 1e-6:
            w = w / wmax

        # Step 2: Mesh-based Laplacian smoothing (respects topology)
        if smooth_iters > 0 and adj is not None:
            w = laplacian_smooth(w, adj, iterations=smooth_iters, alpha=0.5)

        # Step 3: Subtract baseline to pull edges back
        if edge_threshold > 0.001:
            w = w - edge_threshold
            w = np.clip(w, 0, None)

        # Step 4: Renormalize so peak = 1.0
        wmax = w.max()
        if wmax > 1e-6:
            w = w / wmax

        atlas[:, i] = w
        muscle_joints[i] = mdef['joint']

        # Compute flex axis (use explicit override if provided)
        if 'flex_axis' in mdef:
            flex_axes[i] = np.array(mdef['flex_axis'], dtype=np.float32)
        else:
            bone_dir = jpos[mdef['bone'][1]] - jpos[mdef['bone'][0]]
            flex_axes[i] = compute_flex_axis(mdef['direction'], bone_dir)
        muscle_names.append(mdef['name'])

        n_active = np.sum(w > 0.01)
        print(f"  {i:2d} {mdef['name']:16s}  joint={mdef['joint']:2d}  "
              f"dir={mdef['direction']:6s}  verts={n_active:5d}  "
              f"flex=[{flex_axes[i][0]:+.2f},{flex_axes[i][1]:+.2f},{flex_axes[i][2]:+.2f}]")

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    atlas_path = os.path.join(script_dir, 'muscle_atlas_v3.npy')
    meta_path = os.path.join(script_dir, 'muscle_atlas_v3_meta.npy')

    np.save(atlas_path, atlas)
    np.save(meta_path, {
        'muscle_names': muscle_names,
        'muscle_joints': muscle_joints,
        'flex_axes': flex_axes,
        'n_muscles': n_muscles,
    })

    print(f"\nSaved: {atlas_path}  shape={atlas.shape}")
    print(f"Coverage: {(atlas > 0.01).any(axis=1).sum()} / {len(verts)} vertices")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--gender', type=str, default='male',
                        choices=['male', 'female'])
    parser.add_argument('--smooth_iters', type=int, default=0,
                        help='Number of Laplacian smoothing iterations')
    parser.add_argument('--edge_threshold', type=float, default=0.0,
                        help='Baseline subtraction after smoothing')
    args = parser.parse_args()
    generate_atlas(args.model_path, args.gender, args.smooth_iters, args.edge_threshold)

