"""Elbow-hinge impossibility constraint: the forearm direction must stay on a fixed circle.

The elbow is a hinge. Pronation/supination is a free TWIST about the forearm long axis, but the
forearm's DIRECTION can only sweep within one plane that is rigidly fixed to the upper arm. So,
expressed in the UPPER-ARM sensor's frame, the unit forearm long-axis direction must lie on a fixed
circle for every frame of every take -- a hard anatomical constraint no choreography can violate
(real flexion moves along the circle; free pronation doesn't move the direction at all).

We express the forearm direction in the upper-arm frame, fit the hinge plane (PCA: the smallest
principal axis = hinge normal), and measure the per-frame OUT-OF-PLANE angle = the hinge violation.
A frame off the plane is geometrically impossible => a data error (a frame-varying upper-arm axial
roll OR a forearm swing error both push the direction off-plane; real motion stays on it). This is
constant-free and choreography-free, and it tests whether a frame like f4318 is real or corrupt
WITHOUT external video.
"""
import argparse
from pathlib import Path

import numpy as np

from diag_magnetometer_deviation import (load_skeleton, fk_world, qmul_np, qrot_np,
                                         RSH, REL, LSH, LEL)


def qconj(q): return q * np.array([1., -1., -1., -1.])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("--frame", type=int, default=4318)
    ap.add_argument("--win", type=int, default=8)
    ap.add_argument("--robust", type=int, default=2, help="MAD-trim iterations for the plane fit")
    args = ap.parse_args()

    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    d = np.load(args.infile, allow_pickle=True)
    q = d['quats'].astype(np.float64); T = q.shape[0]
    G = fk_world(q, parent, order)

    for side, SH, EL in [('R', RSH, REL), ('L', LSH, LEL)]:
        rel = qmul_np(qconj(G[:, SH]), G[:, EL])          # forearm orientation in upper-arm frame
        dirn = qrot_np(rel, bax[EL])                      # forearm long-axis direction, upper-arm frame
        dirn = dirn / (np.linalg.norm(dirn, axis=-1, keepdims=True) + 1e-12)

        # robust hinge-plane fit: PCA, trim outliers by out-of-plane residual, refit
        keep = np.ones(T, bool)
        for _ in range(args.robust + 1):
            c = dirn[keep].mean(0)
            U, S, Vt = np.linalg.svd(dirn[keep] - c, full_matrices=False)
            normal = Vt[-1]                                # smallest-variance dir = hinge axis
            oop = (dirn - c) @ normal                      # signed out-of-plane component
            oop_ang = np.degrees(np.arcsin(np.clip(oop, -1, 1)))
            mad = np.median(np.abs(oop_ang[keep] - np.median(oop_ang[keep]))) + 1e-9
            keep = np.abs(oop_ang - np.median(oop_ang[keep])) < 4 * 1.4826 * mad
        var_explained = (S[:2] ** 2).sum() / (S ** 2).sum()
        flex = np.degrees(np.arccos(np.clip(dirn @ c / (np.linalg.norm(c) + 1e-9), -1, 1)))  # along-arc proxy

        print(f"\n{side} elbow hinge ({keep.mean()*100:.0f}% frames inlier): plane fit captures "
              f"{var_explained*100:.1f}% of direction variance (>~99% => clean hinge)")
        print(f"  out-of-plane |violation|: median {np.median(np.abs(oop_ang)):.1f}  "
              f"p95 {np.percentile(np.abs(oop_ang),95):.1f}  max {np.abs(oop_ang).max():.1f} deg  "
              f"(frac >15deg: {np.mean(np.abs(oop_ang)>15):.0%})")
        a, b = max(0, args.frame - args.win), min(T, args.frame + args.win + 1)
        print(f"  around f{args.frame}: out-of-plane = " + " ".join(f"{oop_ang[i]:+.0f}" for i in range(a, b, 2)))
        print(f"             @f{args.frame}: violation {oop_ang[args.frame]:+.1f} deg   "
              f"(median {np.median(np.abs(oop_ang)):.1f} => {abs(oop_ang[args.frame])/(np.median(np.abs(oop_ang))+1e-9):.0f}x typical)")
    print("\nLarge out-of-plane at a frame = the forearm is off its only possible hinge arc = the")
    print("reconstruction is impossible there (data error), independent of any external video.")


if __name__ == "__main__":
    main()
