"""Diagnose pelvis-Z artifact vs right-hip flexion.

Confirms (or refutes) whether the residual "whole-body lift on right-leg lift"
artifact in a corrected take correlates with right-hip flexion angle — which
would indicate the planted-foot LS fit is extrapolating poorly to high-flexion
poses (mechanism #1 in the conversation thread).

Outputs `<stem>_diag.png` with three panels:
  1. Time series of pelvis-Z, L-foot-Z, R-foot-Z, with L-foot-planted bands.
  2. Right-hip flexion angle vs time.
  3. Scatter of pelvis-Z vs flexion (colored by L-foot-planted state).

If pelvis-Z spikes cluster at high flexion (i.e., correlate with right-leg
lift), the correction is failing to extrapolate.  Stationary L-foot bands
where pelvis-Z drifts indicate the corrected trans isn't anchored to the
planted foot.

Usage:
    python diag_pelvis_vs_flexion.py <corrected.npz>
"""
from __future__ import annotations

import sys
from pathlib import Path
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correct_thigh_offset import (
    shape_joints, fk_batch,
    R_HIP, L_FOOT, R_FOOT, N_BODY,
    DEFAULT_MODEL_FEMALE, DEFAULT_MODEL_MALE,
)

R_KNEE = 5  # SMPL right knee


def load_model(gender):
    g = str(gender).lower().strip().strip("'\"")
    p = DEFAULT_MODEL_FEMALE if g == 'female' else DEFAULT_MODEL_MALE
    with open(p, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    in_path = Path(sys.argv[1])
    data = np.load(in_path, allow_pickle=True)
    poses = data['poses']
    trans = data['trans']
    betas = np.asarray(data['betas']).flatten()
    gender = str(data['gender'])
    fps = float(data['mocap_framerate']) if 'mocap_framerate' in data.files else 100.0
    T = poses.shape[0]
    print(f'{in_path.name}: {T} frames @ {fps} fps, gender={gender}, betas[:5]={np.round(betas[:5],3)}')

    model = load_model(gender)
    parents = model['kintree_table'][0].astype(np.int64).copy()
    parents[0] = -1
    parents = parents[:N_BODY]
    J_rest = shape_joints(model, betas)[:N_BODY]
    body_pose = poses[:, :N_BODY * 3].reshape(T, N_BODY, 3)

    # FK in chunks → full joint world positions in model frame
    chunk = 4000
    t_world = np.empty((T, N_BODY, 3))
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        tw, _ = fk_batch(J_rest, body_pose[s:e], parents)
        t_world[s:e] = tw

    # World positions = model-frame positions + trans
    pelvis_z_world = trans[:, 2]                          # pelvis sits at model origin
    L_foot_z = t_world[:, L_FOOT, 2] + trans[:, 2]
    R_foot_z = t_world[:, R_FOOT, 2] + trans[:, 2]

    # Right-hip flexion: angle between (knee - hip) world axis and -world_z
    thigh_vec = t_world[:, R_KNEE] - t_world[:, R_HIP]
    thigh_dir = thigh_vec / np.linalg.norm(thigh_vec, axis=1, keepdims=True)
    cos_lift = -thigh_dir[:, 2]
    flexion_rad = np.arccos(np.clip(cos_lift, -1, 1))
    # Subtract rest-pose value so flexion ≈ 0 in T-pose stance
    rest_vec = J_rest[R_KNEE] - J_rest[R_HIP]
    rest_dir = rest_vec / np.linalg.norm(rest_vec)
    rest_lift = np.arccos(np.clip(-rest_dir[2], -1, 1))
    flexion_deg = np.rad2deg(flexion_rad - rest_lift)

    # Left-foot planted heuristic (same flavor as correct_thigh_offset)
    L_vel_z = np.zeros(T)
    if T >= 3:
        L_vel_z[1:-1] = np.abs(L_foot_z[2:] - L_foot_z[:-2]) * 0.5 * fps
    L_planted = (L_foot_z < np.quantile(L_foot_z, 0.10)) & (L_vel_z < 0.20)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(15, 11))
    t = np.arange(T) / fps

    ax = axes[0]
    ax.plot(t, pelvis_z_world, 'k-', lw=0.9, label='pelvis Z (corrected)')
    ax.plot(t, L_foot_z, 'b-', lw=0.5, alpha=0.6, label='L foot Z')
    ax.plot(t, R_foot_z, 'r-', lw=0.5, alpha=0.6, label='R foot Z')
    y0, y1 = ax.get_ylim()
    ax.fill_between(t, y0, y1, where=L_planted, color='blue', alpha=0.08,
                    transform=ax.get_xaxis_transform(), label='L foot planted')
    ax.set_ylabel('Z (m, world)')
    ax.set_xlabel('time (s)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title('Time series')

    ax = axes[1]
    ax.plot(t, flexion_deg, 'g-', lw=0.7)
    ax.set_ylabel('R-hip flexion (deg, vs rest)')
    ax.set_xlabel('time (s)')
    ax.grid(alpha=0.3)
    ax.set_title('Right-hip flexion angle')

    ax = axes[2]
    ax.scatter(flexion_deg[~L_planted], pelvis_z_world[~L_planted],
               c='gray', s=2, alpha=0.25, label='other')
    ax.scatter(flexion_deg[L_planted], pelvis_z_world[L_planted],
               c='blue', s=5, alpha=0.6, label='L foot planted')
    ax.set_xlabel('R-hip flexion (deg)')
    ax.set_ylabel('corrected pelvis Z (m)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('Pelvis-Z vs flexion — if spikes cluster at high flexion, '
                 'the LS fit is extrapolating')

    fig.suptitle(in_path.name, y=0.995)
    fig.tight_layout()
    out = in_path.parent / (in_path.stem + '_diag.png')
    fig.savefig(out, dpi=110)
    print(f'Saved {out}')

    # Quick numerical correlation summary
    if L_planted.sum() > 30:
        corr = np.corrcoef(flexion_deg[L_planted], pelvis_z_world[L_planted])[0, 1]
        print(f'Correlation (pelvis_z, flexion) on L-foot-planted frames: {corr:+.3f}')
    # High-flexion summary
    hi_mask = flexion_deg > np.quantile(flexion_deg, 0.90)
    print(f'Top-10% flexion frames: pelvis_z mean={pelvis_z_world[hi_mask].mean():+.3f}, '
          f'std={pelvis_z_world[hi_mask].std():.3f}')
    lo_mask = flexion_deg < np.quantile(flexion_deg, 0.10)
    print(f'Bottom-10% flexion frames: pelvis_z mean={pelvis_z_world[lo_mask].mean():+.3f}, '
          f'std={pelvis_z_world[lo_mask].std():.3f}')


if __name__ == '__main__':
    main()