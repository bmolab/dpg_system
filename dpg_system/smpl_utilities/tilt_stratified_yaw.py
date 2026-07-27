"""Is the recovered deviation curve delta(psi) tilt-stable, or really delta(psi, tilt)?

A hard/soft-iron distortion is fixed in the SENSOR frame and is added BEFORE the fusion tilt-
compensates the magnetometer, so the heading error it produces depends on the sensor's tilt as
well as its heading. The heading-only landscape is therefore the TILT-AVERAGED slice of a 2-D
surface. Here we stratify each sensor's de-meaned twist (and the constant-free headlock) by the
sensor's own TILT -- the elevation of its bone axis above horizontal -- into low/mid/steep bands,
and overlay the heading curves per band. If the bands overlay, a plain delta(psi) fit is justified;
if they separate, magnetization needs the tilt dimension (or we must restrict to a common tilt band).

De-meaning is done per take GLOBALLY (over all frames), so a constant LEVEL difference between tilt
bands is real tilt-dependence, not an artifact.
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np

from diag_magnetometer_deviation import (load_skeleton, fk_world, qrot_np, qmul_np,
                                         PELV, LSH, LEL, RSH, REL, LBLADE, RBLADE)

NB = 12
XAX = np.array([1., 0., 0.])
TILT_EDGES = [0., 20., 45., 90.]   # |bone-axis elevation| bands: near-horizontal / mid / steep
BAND_NAMES = ["lo(<20)", "mid(20-45)", "hi(>45)"]


def qconj(q): return q * np.array([1., -1., -1., -1.])
def twist(loc, ax): return np.degrees(2 * np.arctan2((loc[:, 1:] * ax).sum(-1), loc[:, 0]))
def wrapdeg(a): return (a + 180) % 360 - 180
def sensor_yaw(Gj): v = qrot_np(Gj, XAX); return np.degrees(np.arctan2(v[:, 2], v[:, 0]))
def elev(Gj, ax):                                    # bone-axis elevation above horizontal, deg
    d = qrot_np(Gj, ax); return np.degrees(np.arcsin(np.clip(d[:, 1], -1, 1)))
def yaw_bin(deg): return ((deg + 180) // 30).astype(int) % NB
def tilt_band(e):
    a = np.abs(e)
    return np.clip(np.digitize(a, TILT_EDGES) - 1, 0, len(BAND_NAMES) - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--nframes", type=int, default=4000)
    ap.add_argument("--turn-rate", type=float, default=0.5)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*beta.npz")))
    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')

    # twist signal -> (parent joint, sensor joint) ; sensor heading & tilt taken from sensor joint
    TW = {'twist L-elbow': (LSH, LEL), 'twist R-elbow': (RSH, REL),
          'twist L-shldr': (LBLADE, LSH), 'twist R-shldr': (RBLADE, RSH)}

    # per (signal, band): list of per-take de-meaned heading curves
    resid = {(s, b): [] for s in TW for b in range(len(BAND_NAMES))}
    hl_resid = {b: [] for b in range(len(BAND_NAMES))}      # headlock-L, constant-free

    for f in files:
        d = np.load(f, allow_pickle=True)
        q = d['quats'].astype(np.float64); T = q.shape[0]
        G = fk_world(q, parent, order)
        fr = np.arange(0, T, max(1, T // args.nframes))

        for name, (pj, sj) in TW.items():
            vals = twist(qmul_np(qconj(G[fr][:, pj]), G[fr][:, sj]), bax[sj])
            vals = vals - vals.mean()                       # de-mean per take, globally
            hb = yaw_bin(sensor_yaw(G[fr][:, sj]))
            tb = tilt_band(elev(G[fr][:, sj], bax[sj]))
            for b in range(len(BAND_NAMES)):
                curve = np.full(NB, np.nan)
                for h in range(NB):
                    m = (hb == h) & (tb == b)
                    if m.sum() >= 5:
                        curve[h] = vals[m].mean()
                resid[(name, b)].append(curve)

        # headlock-L slope, stratified by forearm tilt band
        pel = np.unwrap(np.radians(sensor_yaw(G[:, PELV])))
        psidot = wrapdeg(np.degrees(np.roll(pel, -1) - pel))
        turn = np.abs(psidot) > args.turn_rate
        relL = np.unwrap(np.radians(sensor_yaw(G[:, LEL]))) - pel
        drL = wrapdeg(np.degrees(np.roll(relL, -1) - relL))
        hbf = yaw_bin(sensor_yaw(G[:, LEL])); tbf = tilt_band(elev(G[:, LEL], bax[LEL]))
        for b in range(len(BAND_NAMES)):
            slope = np.full(NB, np.nan)
            for h in range(NB):
                m = turn & (hbf == h) & (tbf == b)
                if m.sum() >= 8 and (psidot[m] ** 2).sum() > 1e-6:
                    slope[h] = (psidot[m] * drL[m]).sum() / (psidot[m] ** 2).sum()
            occ = ~np.isnan(slope)
            hl_resid[b].append(slope - slope[occ].mean() if occ.any() else slope)
        print(f"  {os.path.basename(f).replace('Subject7_take','').replace('_a_beta.npz','')}")

    hdr = "  ".join(f"{-180+30*h:+4d}" for h in range(NB))

    def show(title, perband):
        print(f"\n{title}  (cols = sensor heading bins deg; rows = sensor tilt band)")
        print(f"  {'':12s}{hdr}")
        means = {}
        for b in range(len(BAND_NAMES)):
            arr = np.array(perband[b])
            mean = np.nanmean(arr, axis=0); means[b] = mean
            ntk = np.sum(~np.isnan(arr), axis=0)
            cells = ["   .  " if np.isnan(mean[h]) else f"{mean[h]:+4.1f}" for h in range(NB)]
            print(f"  {BAND_NAMES[b]:12s}" + " ".join(cells))
            ncells = ["  . " if ntk[h] == 0 else f"{ntk[h]:3d} " for h in range(NB)]
            print(f"  {'  n takes':12s}" + " ".join(ncells))
        # between-band spread where >=2 bands have data, vs typical within-band SEM
        ok = np.vstack([means[b] for b in means])
        col_ok = np.sum(~np.isnan(ok), axis=0) >= 2
        if col_ok.any():
            band_spread = np.nanmean(np.nanstd(ok[:, col_ok], axis=0))
            print(f"  -> mean spread BETWEEN tilt bands at shared headings: {band_spread:.1f} deg")
        return means

    for name in TW:
        show(name, {b: resid[(name, b)] for b in range(len(BAND_NAMES))})
    show("headlock L (constant-free)", hl_resid)
    print("\nBands OVERLAY (small between-band spread) => delta(psi) is tilt-stable; plain fit OK.")
    print("Bands SEPARATE (large spread) => delta(psi,tilt): condition on tilt or fit a common band.")


if __name__ == "__main__":
    main()
