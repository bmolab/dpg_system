#!/usr/bin/env python3
"""
Repair a corrupted chest-sensor span in an SMPL poses file.

Context (Shadow IMU capture, see chest-IMU pop-out investigation):
  The chest sensor can pop off the body (e.g. pec pulsing in krumping),
  feeding garbage orientation into the solved spine.  Because SMPL stores
  *local* rotations, the bad chest world orientation appears as:
    - spine2 / spine3 wild in WORLD,
    - neck partly wild (no neck sensor; interpolated chest↔head),
    - both collars wild in LOCAL but quiet in WORLD (their world is driven by
      the still-attached shoulder-back sensors, so it is actually correct).

  Surviving sensors bracket the chest: lower-back (→ spine1), the two
  shoulder-back sensors (→ collars/shoulders), and head.  This repair re-
  estimates the chest block's WORLD orientation from those survivors and re-
  derives the dependent local rotations, preserving every good-sensor world
  pose.  It does NOT recover true chest motion (the sensor was off the body) —
  it makes the gap physiologically consistent and non-propagating.

Method:
  1. FK current locals → per-joint world orientations.
  2. Detect bad spans where the chest block's world speed is non-physiological.
  3. Learn, on clean frames, the rigid offset of spine3 from the shoulder
     girdle (mean of the two collar worlds) and the geodesic blend fractions
     placing spine2 between spine1↔spine3 and neck between spine3↔head.
  4. In bad spans, set spine3_world = girdle·offset (tracks real torso motion
     via the surviving sensors), spine2/neck by the learned blends; taper at
     span edges for continuity.
  5. Recompute locals for spine2/spine3/neck (changed world) and for the
     collars + head (preserve their good world under the new parent).
  6. Write <name>_chestfix.npz.

Usage:
    python repair_chest_orientation.py file_smpl_poses.npz
    python repair_chest_orientation.py file.npz --json noise_report.json --thresh 14
"""

import argparse
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation

from diag_chest_propagation import (
    SMPL_PARENTS, SMPL_NAMES, load_local_rotations, forward_kinematics, ang_speed,
)

J = {n: i for i, n in enumerate(SMPL_NAMES)}


def quat_mean_pair(r_a, r_b):
    """Per-frame geodesic midpoint of two length-T Rotation series."""
    qa = r_a.as_quat()                      # (T,4) xyzw
    qb = r_b.as_quat().copy()
    flip = np.sum(qa * qb, axis=1) < 0      # sign-align before averaging
    qb[flip] *= -1.0
    qm = qa + qb
    qm /= np.linalg.norm(qm, axis=1, keepdims=True)
    return Rotation.from_quat(qm)


def slerp(r_a, r_b, f):
    """Geodesic interpolation a→b by fraction f (scalar or (T,))."""
    rv = (r_a.inv() * r_b).as_rotvec()
    f = np.asarray(f).reshape(-1, 1) if np.ndim(f) else f
    return r_a * Rotation.from_rotvec(f * rv)


def slerp_fraction(r_a, r_b, r_actual):
    """Per-frame scalar f minimising |slerp(a,b,f) − actual| along the geodesic."""
    v_full = (r_a.inv() * r_b).as_rotvec()
    v_act = (r_a.inv() * r_actual).as_rotvec()
    denom = np.sum(v_full * v_full, axis=1)
    denom[denom < 1e-9] = 1e-9
    return np.sum(v_act * v_full, axis=1) / denom


def detect_spans(badness, margin, min_len=1):
    """Frames flagged True → list of (start,end) spans, dilated by `margin`."""
    b = badness.copy()
    idx = np.where(b)[0]
    if len(idx) == 0:
        return []
    spans = []
    s = idx[0]
    prev = idx[0]
    for f in idx[1:]:
        if f - prev <= 2 * margin:           # merge near-adjacent flags
            prev = f
            continue
        spans.append((s, prev))
        s = prev = f
    spans.append((s, prev))
    T = len(badness)
    out = []
    for a, b_ in spans:
        if b_ - a + 1 < min_len:
            pass
        out.append((max(0, a - margin), min(T - 1, b_ + margin)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file', help='SMPL _smpl_poses*.npz file')
    ap.add_argument('--out', default=None, help='Output npz (default <name>_chestfix.npz)')
    ap.add_argument('--json', default=None, help='Noise report JSON; union its corruption zones into the bad spans')
    ap.add_argument('--thresh', type=float, default=14.0, help='Chest-block world speed (rad/s) flagging a bad frame')
    ap.add_argument('--clean-thresh', type=float, default=8.0, help='Chest-block world speed below which a frame is "clean" for learning')
    ap.add_argument('--margin', type=int, default=6, help='Frames to dilate each bad span')
    ap.add_argument('--taper', type=int, default=5, help='Frames to blend estimate↔original at span edges')
    args = ap.parse_args()

    aa, fps, T = load_local_rotations(args.file)
    local, world = forward_kinematics(aa)

    s2, s3, nk = J['spine2'], J['spine3'], J['neck']
    s1, hd, lc, rc = J['spine1'], J['head'], J['left_collar'], J['right_collar']

    # ── 2. detect bad spans (chest block world speed) ──────────────────
    sp2 = ang_speed(world[s2], fps)
    sp3 = ang_speed(world[s3], fps)
    block = np.zeros(T)
    block[1:] = np.maximum(sp2, sp3)          # assign motion to the later frame
    block[:-1] = np.maximum(block[:-1], np.maximum(sp2, sp3))
    badness = block > args.thresh

    if args.json and os.path.isfile(args.json):
        data = json.load(open(args.json))
        reports = (list(data['files'].values()) if isinstance(data, dict)
                   and isinstance(data.get('files'), dict)
                   else data if isinstance(data, list) else [data])
        for r in reports:
            for z in ((r.get('surgery') or {}).get('excision') or {}).get('zones', []):
                badness[z.get('start', 0):z.get('end', 0) + 1] = True

    spans = detect_spans(badness, args.margin)
    if not spans:
        print(f"No chest-block corruption above {args.thresh} rad/s — nothing to repair.")
        return
    span_mask = np.zeros(T, bool)
    for a, b in spans:
        span_mask[a:b + 1] = True

    # ── 3. learn anchor model on clean frames ──────────────────────────
    clean = (block <= args.clean_thresh) & ~span_mask
    if clean.sum() < 50:
        print(f"Only {clean.sum()} clean frames — anchor model would be unreliable; aborting.")
        return
    girdle = quat_mean_pair(world[lc], world[rc])
    offset3 = (girdle.inv() * world[s3])[clean].mean()           # spine3 rel girdle
    f2 = float(np.clip(np.median(slerp_fraction(world[s1], world[s3], world[s2])[clean]), 0, 1))
    fnk = float(np.clip(np.median(slerp_fraction(world[s3], world[hd], world[nk])[clean]), 0, 1))
    print(f"Learned anchor model on {clean.sum()} clean frames:")
    print(f"  spine3 offset from girdle: {np.degrees(offset3.magnitude()):.1f}°")
    print(f"  spine2 blend spine1→spine3: f={f2:.2f}   neck blend spine3→head: f={fnk:.2f}")

    # ── 4. estimate corrected world in spans (with edge taper) ─────────
    w3_new = girdle * offset3                                    # full series
    w2_new = slerp(world[s1], w3_new, f2)
    wn_new = slerp(w3_new, world[hd], fnk)

    def blended(orig, est):
        q = orig.as_quat().copy()
        for a, b in spans:
            for f in range(a, b + 1):
                # taper weight: 0 at span edge → 1 in core
                d = min(f - a, b - f)
                t = min(1.0, (d + 1) / (args.taper + 1)) if args.taper > 0 else 1.0
                qf = slerp(orig[f:f + 1], est[f:f + 1], t).as_quat()[0]
                q[f] = qf
        return Rotation.from_quat(q)

    w3 = blended(world[s3], w3_new)
    w2 = blended(world[s2], w2_new)
    wn = blended(world[nk], wn_new)

    # ── 5. recompute affected locals (preserve good-sensor worlds) ─────
    aa_new = aa.copy()
    aa_new[:, s2] = (world[s1].inv() * w2).as_rotvec()           # parent spine1 unchanged
    aa_new[:, s3] = (w2.inv() * w3).as_rotvec()
    aa_new[:, nk] = (w3.inv() * wn).as_rotvec()
    aa_new[:, lc] = (w3.inv() * world[lc]).as_rotvec()           # preserve collar world
    aa_new[:, rc] = (w3.inv() * world[rc]).as_rotvec()
    aa_new[:, hd] = (wn.inv() * world[hd]).as_rotvec()           # preserve head world
    # outside spans, keep the originals untouched
    keep = ~span_mask
    for j in (s2, s3, nk, lc, rc, hd):
        aa_new[keep, j] = aa[keep, j]

    # ── 6. write repaired poses ────────────────────────────────────────
    d = dict(np.load(args.file, allow_pickle=True))
    poses = np.asarray(d['poses'], dtype=np.float64)
    pj = poses.reshape(T, -1, 3)
    pj[:, [s2, s3, nk, lc, rc, hd]] = aa_new[:, [s2, s3, nk, lc, rc, hd]]
    d['poses'] = pj.reshape(T, -1)

    out = args.out or (os.path.splitext(args.file)[0] + '_chestfix.npz')
    np.savez(out, **d)

    repaired = int(span_mask.sum())
    print(f"\nRepaired {len(spans)} span(s), {repaired} frames "
          f"({100 * repaired / T:.1f}% of file):")
    for a, b in spans:
        print(f"  [{a:5d}–{b:5d}]  {(b - a + 1) / fps:.2f}s")
    print(f"→ Saved: {os.path.basename(out)}")


if __name__ == '__main__':
    main()
