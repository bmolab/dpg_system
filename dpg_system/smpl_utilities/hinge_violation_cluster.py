"""Cluster the frames that VIOLATE the elbow-hinge constraint, and check the wrist axial twist.

Builds on elbow_hinge_violation.py: the forearm direction in the upper-arm frame must sit on a
fixed hinge circle, so the out-of-plane angle is a hard, choreography-free error signal. Here we
pull the high-violation frames and ask WHERE the error lives:
  - temporally  : isolated spikes (glitches) vs sustained runs (systematic)?
  - by heading  : do violations concentrate at particular forearm-sensor headings (magnetization)?
  - vs wrist    : the hand cannot axially spin on the forearm (that DOF is forearm pronation, already
                  in the forearm sensor) -- so large hand-vs-forearm twist is itself near-impossible;
                  do elbow violations co-occur with extreme wrist twist?

One take at a time (pass a file); the pattern (spikes/heading/wrist) tells us the error mechanism.
"""
import argparse
from pathlib import Path

import numpy as np

from diag_magnetometer_deviation import (load_skeleton, fk_world, qmul_np, qrot_np,
                                         RSH, REL, RWR, LSH, LEL, LWR)

NB = 12
XAX = np.array([1., 0., 0.])


def qconj(q): return q * np.array([1., -1., -1., -1.])
def twist(loc, ax): return np.degrees(2 * np.arctan2((loc[:, 1:] * ax).sum(-1), loc[:, 0]))
def sensor_yaw(Gj): v = qrot_np(Gj, XAX); return np.degrees(np.arctan2(v[:, 2], v[:, 0]))
def yaw_bin(deg): return ((deg + 180) // 30).astype(int) % NB


def hinge_oop(G, SH, EL, bax, robust=2):
    rel = qmul_np(qconj(G[:, SH]), G[:, EL])
    dirn = qrot_np(rel, bax[EL]); dirn /= np.linalg.norm(dirn, axis=-1, keepdims=True) + 1e-12
    keep = np.ones(len(dirn), bool)
    for _ in range(robust + 1):
        c = dirn[keep].mean(0)
        U, S, Vt = np.linalg.svd(dirn[keep] - c, full_matrices=False)
        oop = np.degrees(np.arcsin(np.clip((dirn - c) @ Vt[-1], -1, 1)))
        mad = np.median(np.abs(oop[keep] - np.median(oop[keep]))) + 1e-9
        keep = np.abs(oop - np.median(oop[keep])) < 4 * 1.4826 * mad
    return oop


def runs(mask):
    idx = np.flatnonzero(mask);
    if len(idx) == 0: return []
    splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [(s[0], s[-1]) for s in splits]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("--thr", type=float, default=15.0, help="out-of-plane violation threshold deg")
    args = ap.parse_args()

    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    d = np.load(args.infile, allow_pickle=True)
    q = d['quats'].astype(np.float64); T = q.shape[0]
    G = fk_world(q, parent, order)

    for side, SH, EL, WR in [('R', RSH, REL, RWR), ('L', LSH, LEL, LWR)]:
        oop = hinge_oop(G, SH, EL, bax)
        wtw = twist(qmul_np(qconj(G[:, EL]), G[:, WR]), bax[WR])    # hand-vs-forearm axial
        hi = np.abs(oop) > args.thr
        fhead = sensor_yaw(G[:, EL])
        print(f"\n=== {side} arm ===  {hi.sum()} frames over {args.thr:.0f}deg hinge violation ({100*hi.mean():.0f}%)")
        rr = runs(hi)
        lens = [b - a + 1 for a, b in rr]
        print(f"  temporal: {len(rr)} runs; lengths median {int(np.median(lens)) if lens else 0}, "
              f"max {max(lens) if lens else 0}; isolated(<=2f) {sum(l<=2 for l in lens)} / sustained(>=10f) {sum(l>=10 for l in lens)}")
        # by forearm-sensor heading: violation RATE per bin
        hb = yaw_bin(fhead)
        print(f"  {'heading':9s}" + " ".join(f"{-180+30*b:+4d}" for b in range(NB)))
        rate = [100*hi[hb == b].mean() if (hb == b).any() else np.nan for b in range(NB)]
        print(f"  {'viol %':9s}" + " ".join("  . " if np.isnan(r) else f"{r:4.0f}" for r in rate))
        nfr = [int((hb == b).sum()) for b in range(NB)]
        print(f"  {'n frames':9s}" + " ".join(f"{n:4d}" for n in nfr))
        # wrist twist: overall vs at high-violation frames
        print(f"  WRIST axial twist (hand vs forearm): median {np.median(wtw):+.0f}  p95 {np.percentile(np.abs(wtw),95):.0f}  "
              f"max|{np.abs(wtw).max():.0f}|  frac|>45 {np.mean(np.abs(wtw)>45):.0%}")
        if hi.any():
            print(f"     at hinge-violation frames: median |wrist tw| {np.median(np.abs(wtw[hi])):.0f}  "
                  f"vs {np.median(np.abs(wtw[~hi])):.0f} elsewhere; corr(|oop|,|wtw|) {np.corrcoef(np.abs(oop),np.abs(wtw))[0,1]:+.2f}")
        # top violation frames
        topf = np.argsort(-np.abs(oop))[:8]
        print(f"  worst frames: " + " ".join(f"f{int(i)}({oop[i]:+.0f}/w{wtw[i]:+.0f})" for i in topf))
    print("\nRun-length: many isolated spikes => fusion glitches; few long runs => systematic poses/headings.")
    print("(oop=elbow out-of-plane deg, w=wrist axial twist deg). Heading concentration => magnetization-linked.")


if __name__ == "__main__":
    main()
