"""Detect solver twist-toggling in STATIC poses: position frozen but orientation flips between states.

Refined per user: the artifact appears when there's almost no movement (joint positions nearly
frozen), yet the mesh visibly shifts — the sensor orientation (bone twist) toggling between two states
the solver can't disambiguate without motion. So we find STATIC windows (low endpoint position
velocity, sustained) and, inside them, measure each sensor's TWIST range/variation and count toggles
(midpoint crossings). Frozen position + wandering/bimodal twist = the glitch. Decoupling, not speed,
is the tell.
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
from correct_upper_arm_offset import forward_kinematics, qmul
from diag_magnetometer_deviation import load_skeleton, RSH, REL, RWR, LSH, LEL, LWR, RBLADE, LBLADE

def qc(q): return q * np.array([1., -1, -1, -1.])
def tw(loc, ax): return np.degrees(2 * np.arctan2((loc[:, 1:] * ax).sum(-1), loc[:, 0]))
def wrapu(a):
    a = np.asarray(a); return a  # twist already continuous-ish; use raw range within short window


def static_runs(vel, thr, minlen=15):
    still = vel < thr; runs = []
    i = 0
    while i < len(still):
        if still[i]:
            j = i
            while j < len(still) and still[j]: j += 1
            if j - i >= minlen: runs.append((i, j))
            i = j
        else: i += 1
    return runs


def toggles(x, amp=8.0):
    """count large excursions back and forth across the window mean (bimodal toggling)."""
    m = np.median(x); s = np.sign(x - m)
    crossings = np.flatnonzero(np.diff(s) != 0)
    big = 0
    for c in crossings:
        if np.abs(x[c + 1] - x[max(0, c - 2):c + 3].mean()) > amp: big += 1
    return big


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); args = ap.parse_args()
    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    files = sorted(glob.glob(os.path.join(args.dir, "*beta.npz")))
    chains = [('R-shldr', RSH, RBLADE, RWR), ('L-shldr', LSH, LBLADE, LWR),
              ('R-elbow', REL, RSH, RWR), ('L-elbow', LEL, LSH, LWR)]
    print("STATIC-pose twist behavior (position frozen). twist-range & toggles inside static windows:")
    print(f"  {'take':14s} {'#stat':>5} " + " ".join(f"{c[0]:>16s}" for c in chains))
    print(f"  {'':14s} {'wins':>5} " + " ".join(f"{'maxRng/medRng/tog':>16s}" for c in chains))
    for f in files:
        d = np.load(f, allow_pickle=True); G = forward_kinematics(d['quats'].astype(np.float64)); P = d['positions'].astype(np.float64)
        # stillness: both wrists' position velocity low (sustained)
        vR = np.linalg.norm(np.diff(P[:, RWR], axis=0), axis=1); vL = np.linalg.norm(np.diff(P[:, LWR], axis=0), axis=1)
        vel = np.maximum(vR, vL); vel = np.append(vel, vel[-1])
        thr = np.percentile(vel, 20)                                  # bottom-20% motion = "static"
        runs = static_runs(vel, thr, minlen=15)
        cells = []
        for nm, j, par, ep in chains:
            twj = tw(qmul(qc(G[:, par]), G[:, j]), bax[j])
            rngs = []; togs = 0
            for a, b in runs:
                w = twj[a:b]; w = (w - w[0] + 180) % 360 - 180 + w[0]   # local unwrap
                rngs.append(w.max() - w.min()); togs += toggles(w)
            rngs = np.array(rngs) if rngs else np.array([0.])
            cells.append(f"{rngs.max():4.0f}/{np.median(rngs):3.0f}/{togs:3d}")
        print(f"  {os.path.basename(f).replace('Subject7_take_','').replace('_beta.npz','')[:14]:14s} {len(runs):5d} " + " ".join(f"{c:>16s}" for c in cells))
    print("\nmaxRng=largest twist range in any static window; medRng=typical; tog=toggle count.")
    print("Large twist-range / toggles while position is frozen = solver twist-flip glitch.")


if __name__ == "__main__":
    main()
