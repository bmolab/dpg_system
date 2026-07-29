"""Does magnetization bleed into the TWIST DOF? Stratify the heading-harmonic twist by elevation.

Under a clean AHRS the magnetometer corrupts only world-YAW, so its projection onto bone-axis twist
is delta(psi)*sin(elevation): the session-consistent heading-dependent twist amplitude must scale as
sin(elev) -- ~0 when the bone is horizontal, growing as it tilts toward vertical. If instead the
amplitude stays large and ~flat at horizontal elevations, the fusion is letting magnetization corrupt
roll/twist directly (imperfect yaw/tilt decoupling) -- i.e. magnetization genuinely causes twist.

We fit the session-shared harmonic (per-take constant + shared B,C,D,E) to the twist binned by the
sensor's own heading, separately within elevation bands, and report the 1st-harmonic amplitude vs the
sin(elev) that clean-yaw theory predicts it should track.
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np

from diag_magnetometer_deviation import (load_skeleton, fk_world, qmul_np, qrot_np,
                                         RSH, REL, RWR, LSH, LEL, LWR)
from fit_session_deviation import fit, NB

XAX = np.array([1., 0., 0.])
BANDS = [(0, 20), (20, 40), (40, 65)]


def qconj(q): return q * np.array([1., -1., -1., -1.])
def twist(loc, ax): return np.degrees(2 * np.arctan2((loc[:, 1:] * ax).sum(-1), loc[:, 0]))
def sensor_yaw(Gj): v = qrot_np(Gj, XAX); return np.degrees(np.arctan2(v[:, 2], v[:, 0]))
def elevd(Gj, ax): d = qrot_np(Gj, ax); return np.degrees(np.arcsin(np.clip(d[:, 1], -1, 1)))
def yaw_bin(deg): return ((deg + 180) // 30).astype(int) % NB


def collect_band(files, parent, order, bax, pj, sj, lo, hi, stride):
    takes = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        q = d['quats'].astype(np.float64); T = q.shape[0]
        G = fk_world(q, parent, order)
        fr = np.arange(0, T, stride)
        vals = twist(qmul_np(qconj(G[fr][:, pj]), G[fr][:, sj]), bax[sj])
        ae = np.abs(elevd(G[fr][:, sj], bax[sj]))
        keep = (ae >= lo) & (ae < hi)
        hb = yaw_bin(sensor_yaw(G[fr][:, sj]))
        ym = np.full(NB, np.nan); nm = np.zeros(NB)
        for h in range(NB):
            m = keep & (hb == h)
            if m.sum() >= 5:
                ym[h] = vals[m].mean(); nm[h] = m.sum()
        takes.append((ym, nm))
    return takes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*beta.npz")))
    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    centers = np.radians(np.array([-180 + 30 * b for b in range(NB)], float))

    SENSORS = {'R-elbow (vs uparm)': (RSH, REL), 'L-elbow (vs uparm)': (LSH, LEL),
               'R-wrist (vs forearm)': (REL, RWR), 'L-wrist (vs forearm)': (LEL, LWR)}
    print("1st-harmonic twist amplitude (deg) by elevation band; clean-yaw predicts ~proportional to sin(elev_mid).")
    print(f"  {'sensor':22s}" + "".join(f"|elev {lo:2d}-{hi:2d} (sin={np.sin(np.radians((lo+hi)/2)):.2f})" for lo, hi in BANDS))
    for name, (pj, sj) in SENSORS.items():
        cells = []
        amps = []
        for lo, hi in BANDS:
            takes = collect_band(files, parent, order, bax, pj, sj, lo, hi, args.stride)
            occ = sum(np.sum(~np.isnan(t[0])) for t in takes)
            if occ < 24:
                cells.append("   (sparse)   "); amps.append(np.nan); continue
            A, BCDE, r2, rms = fit(takes, centers)
            H1 = np.hypot(BCDE[0], BCDE[1])
            cells.append(f"  {H1:5.1f} ({100*r2:2.0f}%)  ")
            amps.append(H1)
        print(f"  {name:22s}" + "".join(cells))
        # ratio to the steepest band's sin, to see if it tracks sin(elev)
        sins = [np.sin(np.radians((lo + hi) / 2)) for lo, hi in BANDS]
        if not np.isnan(amps[0]) and not np.isnan(amps[-1]) and amps[-1] > 1:
            pred0 = amps[-1] * sins[0] / sins[-1]
            print(f"  {'  -> horiz band:':22s}  observed {amps[0]:.1f} vs sin-scaled prediction {pred0:.1f}  "
                  f"({'FLAT=roll-bleed' if amps[0] > 2*pred0 + 2 else 'tracks sin => clean-yaw'})")
    print("\nAmp ~0 at horizontal, growing with elev => clean yaw-only (twist is a projection, not the error).")
    print("Amp large & flat at horizontal => magnetization bleeding into roll/twist (Mechanism B).")


if __name__ == "__main__":
    main()