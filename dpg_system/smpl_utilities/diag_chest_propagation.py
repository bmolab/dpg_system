#!/usr/bin/env python3
"""
Diagnostic: does a corrupted chest sensor propagate as a local-only artifact?

Hypothesis (see chest-IMU pop-out investigation):
  SMPL poses store *local* rotations.  A dangling chest sensor injects garbage
  rotation +δ into the chest's own local angle (spine3) and the inverse −δ into
  every child measured against it (neck, both collars).  In forward kinematics
  the two errors cancel:

      neck_world = chest_world(bad) · neck_local(bad⁻¹) ≈ neck_world_true

  So the corruption should show up as:
    - spine3 (chest): wild in BOTH local and world angular speed.
    - neck / collars (its children): wild in LOCAL, but QUIET in WORLD.

This script reconstructs per-joint world orientations by FK and reports local
vs world angular speed, both over the whole file and in a focused window around
the worst chest event.  If world speed stays low for the children while their
local speed spikes, the cancellation holds and the principled fix is to repair
the chest *world* orientation and re-derive the dependent locals.

Usage:
    python diag_chest_propagation.py file_smpl_poses.npz
    python diag_chest_propagation.py file.npz --window 60
"""

import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation

# SMPL 24-joint kinematic tree (parent index per joint; -1 = root).
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

SMPL_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand',
]

# Joints to report: the chest, its parents, and its children/grandchildren.
REPORT = ['spine1', 'spine2', 'spine3', 'neck', 'head',
          'left_collar', 'right_collar', 'left_shoulder', 'right_shoulder',
          'left_elbow', 'right_elbow']


def load_local_rotations(path):
    d = np.load(path, allow_pickle=True)
    if 'poses' not in d:
        raise ValueError(f"Not a SMPL poses file (no 'poses'): {path}")
    poses = np.asarray(d['poses'], dtype=np.float64)
    T = poses.shape[0]
    aa = poses.reshape(T, -1, 3)[:, :24]          # (T, 24, 3) axis-angle
    fps = 100.0
    for k in ('mocap_framerate', 'motioncapture_framerate', 'framerate'):
        if k in d:
            fps = float(d[k]); break
    return aa, fps, T


def forward_kinematics(aa):
    """Return list of length-T Rotation objects: world orientation per joint."""
    T = aa.shape[0]
    local = [Rotation.from_rotvec(aa[:, j]) for j in range(24)]
    world = [None] * 24
    for j in range(24):                            # SMPL parents always precede
        p = SMPL_PARENTS[j]
        world[j] = local[j] if p < 0 else world[p] * local[j]
    return local, world


def ang_speed(rot, fps):
    """Per-frame angular speed (rad/s) of a length-T Rotation series."""
    rel = rot[:-1].inv() * rot[1:]
    return np.linalg.norm(rel.as_rotvec(), axis=1) * fps


def p95(x):
    s = np.sort(x)
    return s[int(0.95 * (len(s) - 1))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file', help='SMPL _smpl_poses*.npz file')
    ap.add_argument('--window', type=int, default=40,
                    help='Half-width (frames) of focused window around worst chest event')
    args = ap.parse_args()

    aa, fps, T = load_local_rotations(args.file)
    local, world = forward_kinematics(aa)

    idx = {n: i for i, n in enumerate(SMPL_NAMES)}
    loc_sp = {n: ang_speed(local[idx[n]], fps) for n in REPORT}
    wld_sp = {n: ang_speed(world[idx[n]], fps) for n in REPORT}

    print(f"\n{os.path.basename(args.file)}: {T} frames @ {fps:.0f} fps\n")
    print("Angular speed (rad/s), local rotation vs FK world orientation:")
    print(f"{'joint':<16}{'loc p95':>9}{'loc max':>9}   "
          f"{'wld p95':>9}{'wld max':>9}   {'wld/loc max':>12}")
    print('-' * 70)
    for n in REPORT:
        lp, lm = p95(loc_sp[n]), loc_sp[n].max()
        wp, wm = p95(wld_sp[n]), wld_sp[n].max()
        ratio = wm / lm if lm > 1e-9 else 0.0
        flag = ''
        if n != 'spine3' and lm > 5 and ratio < 0.5:
            flag = '  ← local-only (cancels in world)'
        elif n == 'spine3' and ratio > 0.5:
            flag = '  ← wild in world too (true corrupted segment)'
        print(f"{n:<16}{lp:9.2f}{lm:9.2f}   {wp:9.2f}{wm:9.2f}   {ratio:12.2f}{flag}")

    # ── Focused window around the worst chest (spine3) local event ──────
    s3 = loc_sp['spine3']
    peak = int(np.argmax(s3))
    a = max(0, peak - args.window)
    b = min(T - 1, peak + args.window)
    print(f"\nWorst spine3 local event at frame {peak} "
          f"({peak / fps:.1f}s), speed {s3[peak]:.1f} rad/s")
    print(f"Window [{a}–{b}]  —  local (L) vs world (W) angular speed:\n")
    cols = ['spine2', 'spine3', 'neck', 'left_collar', 'right_collar']
    hdr = 'frame  ' + ''.join(f'{c[:9]:>11}' for c in cols)
    print(hdr)
    print('       ' + ''.join(f'{"L / W":>11}' for _ in cols))
    print('-' * len(hdr))
    for f in range(a, b + 1):
        if f >= T - 1:
            break
        row = f'{f:5d}  '
        for c in cols:
            row += f'{loc_sp[c][f]:5.0f}/{wld_sp[c][f]:<5.0f}'
        print(row)


if __name__ == '__main__':
    main()