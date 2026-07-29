"""Auto-detect bilateral upper-arm symmetry crossings across an entire take.

We can't pick symmetric frames by 'small L/R mirror residual' -- the heading-dependent error inflates
that residual exactly at the headings we care about, so a threshold would select for headings where
the error cancels (biased). Instead we find LOCAL MINIMA of the mirror residual over time: instants
where the arms PASS THROUGH their most-symmetric configuration. At such a crossing the true pose is
mirror-symmetric, so the residual VALUE there is the error itself -- sampled at whatever heading the
crossing happens at. Collected over the take, these give symmetric anchors spanning many headings,
the data a heading-DEPENDENT correction needs (a constant C can't fit a heading-dependent asymmetry).

Reports: number of crossings, their heading coverage (pelvis facing + each upper-arm's own heading),
and the crossing asymmetry vs heading -- i.e. the upper-arm error as a function of heading, read off
genuinely symmetric poses.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from diag_magnetometer_deviation import (load_skeleton, fk_world, qrot_np, PELV,
                                         LSH, RSH, LEL, REL, MIDV)

NB = 12
UP = np.array([0., 1., 0.])
XAX = np.array([1., 0., 0.])


def nrows(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
def yaw(v): return np.degrees(np.arctan2(v[:, 2], v[:, 0]))
def ybin(deg): return ((deg + 180) // 30).astype(int) % NB


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("--prominence", type=float, default=0.12)
    ap.add_argument("--distance", type=int, default=8, help="min frames between crossings")
    args = ap.parse_args()

    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    d = np.load(args.infile, allow_pickle=True)
    q = d['quats'].astype(np.float64); T = q.shape[0]
    G = fk_world(q, parent, order)
    P = d['positions'].astype(np.float64) if 'positions' in d.files else None
    have = (P is not None and np.any(P[:, LSH]) and np.any(P[:, LEL]))

    # per-frame lateral axis + upper-arm distal & secondary directions
    latv = (P[:, LSH] - P[:, RSH]) if have else qrot_np(G[:, MIDV], XAX)
    lat = nrows(latv - UP * (latv @ UP)[:, None])
    M = np.eye(3)[None] - 2 * lat[:, :, None] * lat[:, None, :]
    if have:
        nL = nrows(P[:, LEL] - P[:, LSH]); nR = nrows(P[:, REL] - P[:, RSH])
    else:
        nL = nrows(qrot_np(G[:, LSH], XAX)); nR = nrows(qrot_np(G[:, RSH], -XAX))
    mL = nrows(qrot_np(G[:, LSH], np.array([0, 0, 1.]))); mR = nrows(qrot_np(G[:, RSH], np.array([0, 0, 1.])))

    mnL = np.einsum('nij,nj->ni', M, nL); mmL = np.einsum('nij,nj->ni', M, mL)
    s = np.linalg.norm(mnL - nR, axis=1) + np.linalg.norm(mmL - mR, axis=1)   # full-orientation mirror residual
    cross, _ = find_peaks(-s, prominence=args.prominence, distance=args.distance)

    asym = np.degrees(np.arccos(np.clip((mnL * nR).sum(1), -1, 1)))           # distal mirror angle (deg)
    pelh = yaw(qrot_np(G[:, PELV], XAX))
    Lh = yaw(qrot_np(G[:, LSH], XAX)); Rh = yaw(qrot_np(G[:, RSH], XAX))

    print(f"{Path(args.infile).name}: {len(cross)} symmetric crossings over {T} frames "
          f"(residual at crossings: median {np.median(s[cross]):.2f}, p10 {np.percentile(s[cross],10):.2f})")
    print(f"  crossing upper-arm asymmetry (distal mirror angle): median {np.median(asym[cross]):.1f} deg, "
          f"p90 {np.percentile(asym[cross],90):.1f}")

    def hist(name, vals):
        b = ybin(vals[cross]); cnt = [int((b == k).sum()) for k in range(NB)]
        print(f"  {name:16s}" + " ".join(f"{c:4d}" for c in cnt))
    print(f"  {'heading bin':16s}" + " ".join(f"{-180+30*k:+4d}" for k in range(NB)))
    hist('crossings/pelvis', pelh); hist('crossings/L-arm', Lh); hist('crossings/R-arm', Rh)

    # asymmetry vs pelvis facing -- the upper-arm error as a function of heading, from symmetric poses
    b = ybin(pelh[cross])
    print(f"  {'asym by pelvis':16s}" + " ".join(
        (f"{asym[cross][b==k].mean():4.0f}" if (b == k).any() else "   .") for k in range(NB)))
    print("\nGood heading spread + asymmetry that varies across bins => heading-dependent error, and")
    print("enough symmetric samples across headings to fit a heading-dependent correction.")

    np.savez(args.infile.replace('.npz', '_symcross.npz'),
             frames=cross, asym=asym[cross], pelvis_head=pelh[cross], L_head=Lh[cross], R_head=Rh[cross])
    print(f"  saved crossing frames -> {Path(args.infile.replace('.npz','_symcross.npz')).name}")


if __name__ == "__main__":
    main()
