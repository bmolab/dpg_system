"""Impossibility detector: left-forearm-through-torso self-intersection (capsule proxy).

Style-independent (unlike VPoser plausibility): a real pose, however unusual, never puts the
forearm inside the torso. Builds orientation-consistent FK positions from the bone offsets in
definition.xml (NOT the accel-integrated stored positions), approximates the torso as a capsule
(pelvis->upper-spine) and the forearm as a capsule (elbow->wrist+hand), and reports the
penetration depth per frame. Penetration = (r_torso + r_forearm) - segment_distance.

Used to (a) validate the geometry and (b) confirm the ORIGINAL take penetrates at some frames
(the thing to fix) while legal-but-unusual ranges do not -- so a minimal-correction solver has a
real, localized target and leaves good frames alone.
"""
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from diag_magnetometer_deviation import load_skeleton, fk_world, qrot_np, PELV, LEL, LWR, UPPV

LKNUCKLE = 6
R_TORSO, R_FOREARM = 13.0, 5.5            # cm; capsule radii (definition.xml units are cm)


def parse_offsets(def_path):
    """Local bone offset (cm) of each joint from its parent, indexed by Shadow joint index."""
    from diag_magnetometer_deviation import LIMB_TO_IDX
    root = ET.parse(def_path).getroot()
    off = np.zeros((37, 3))

    def walk(node):
        idx = LIMB_TO_IDX.get(node.get('id'))
        t = node.get('translate')
        if idx is not None and t:
            off[idx] = [float(v) for v in t.split()]
        for ch in node:
            walk(ch)
    walk(root)
    return off


def fk_positions(Gw, parent, order, offsets):
    """World joint positions from world orientations + local bone offsets (root pelvis at 0)."""
    T = Gw.shape[0]
    pos = np.zeros((T, 37, 3))
    for j in order:
        p = parent[j]
        if p >= 0:
            pos[:, j] = pos[:, p] + qrot_np(Gw[:, p], offsets[j])
    return pos


def seg_seg_dist(p1, q1, p2, q2):
    """Closest distance between segments p1q1 and p2q2 (vectorized over frames)."""
    d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2
    a = (d1 * d1).sum(-1); e = (d2 * d2).sum(-1); f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1); b = (d1 * d2).sum(-1)
    denom = a * e - b * b
    s = np.where(denom > 1e-9, np.clip((b * f - c * e) / np.where(denom > 1e-9, denom, 1), 0, 1), 0.0)
    t = (b * s + f) / np.where(e > 1e-9, e, 1)
    t2 = np.clip(t, 0, 1)
    s = np.clip((b * t2 - c) / np.where(a > 1e-9, a, 1), 0, 1)
    c1 = p1 + d1 * s[..., None]; c2 = p2 + d2 * t2[..., None]
    return np.linalg.norm(c1 - c2, axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("--clip", help="report mean penetration over a frame range 'a:b'")
    args = ap.parse_args()

    def_path = Path(__file__).resolve().parent.parent / 'definition.xml'
    parent, order, _ = load_skeleton(def_path)
    offsets = parse_offsets(def_path)

    d = np.load(args.infile, allow_pickle=True)
    q = d['quats'].astype(np.float64)
    fps = float(d['mocap_framerate']) if 'mocap_framerate' in d.files else 100.0
    Gw = fk_world(q, parent, order)
    pos = fk_positions(Gw, parent, order, offsets)

    torso_a, torso_b = pos[:, PELV], pos[:, UPPV]
    fore_a, fore_b = pos[:, LEL], pos[:, LWR]
    hand = pos[:, LKNUCKLE]
    dist_fa = seg_seg_dist(torso_a, torso_b, fore_a, fore_b)
    dist_hand = seg_seg_dist(torso_a, torso_b, hand, hand)
    pen = np.maximum((R_TORSO + R_FOREARM) - dist_fa, (R_TORSO + R_FOREARM) - dist_hand)
    pen = np.maximum(pen, 0.0)

    px = qrot_np(Gw[:, PELV], np.array([1.0, 0.0, 0.0]))
    heading = np.degrees(np.arctan2(px[:, 2], px[:, 0]))

    if args.clip:
        a, b = (int(v) for v in args.clip.split(":"))
        print(f"{Path(args.infile).name} f{a}-{b}: mean penetration {pen[a:b].mean():.1f} cm, "
              f"max {pen[a:b].max():.1f} cm, penetrating frames {100*(pen[a:b]>0).mean():.0f}%")
        return

    T = len(pen)
    print(f"{Path(args.infile).name}: {T} frames. left forearm/hand vs torso (capsule r={R_TORSO}+{R_FOREARM} cm)")
    print(f"penetrating frames: {100*(pen>0).mean():.1f}%   max depth {pen.max():.1f} cm")
    # contiguous penetrating runs
    mask = pen > 1.0
    runs, i = [], 0
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            if j - i >= 5:
                runs.append((i, j))
            i = j
        else:
            i += 1
    print(f"\n{len(runs)} penetration runs >= 5 frames:")
    print(f"  {'frames':>15} {'dur_s':>6} {'maxdepth':>8} {'head':>6}")
    for a, b in sorted(runs, key=lambda r: -(pen[r[0]:r[1]].max()))[:15]:
        print(f"  {a:7d}:{b:<7d} {(b-a)/fps:6.2f} {pen[a:b].max():8.1f} {np.median(heading[a:b]):6.0f}")
    print("\n(detector only) penetration runs are the impossibility targets for minimal correction.")


if __name__ == "__main__":
    main()
