"""Fit the session-shared compass-deviation series to each arm sensor's heading landscape.

Model (one joint weighted least-squares over ALL 11 takes simultaneously):

    twist_{t,h} = A_t  +  B sin psi_h + C cos psi_h + D sin 2psi_h + E cos 2psi_h

  - A_t : per-TAKE constant (recalibration drift + baked-in calibration offset, confounded -- a
          static yaw offset that does not distort motion). One per take. Fitting these jointly IS
          the de-meaning, done with correct per-bin weighting.
  - B,C,D,E : SESSION-SHARED harmonics = the magnetization, which physics says cannot drift within
          a session. (B,C) hard-iron 1st harmonic, (D,E) soft-iron 2nd. This is the yaw-VARYING
          error that actually distorts the motion -- the thing we want to correct.

Observations are per (take, 30deg sensor-heading bin), restricted to the well-conditioned tilt band
(|bone-axis elevation| <= TILT_MAX, where sensor heading is observable and twist isn't gimbal-
degenerate), weighted by frame count. We report the recovered harmonics, 1st/2nd amplitudes with
bootstrap-over-takes CIs, the fraction of twist variance the shared shape explains beyond the per-
take constants, and the fitted curve vs the pooled data. Nothing is written; this is the curve the
corrector will use.
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np

from diag_magnetometer_deviation import (load_skeleton, fk_world, qrot_np, qmul_np,
                                         LSH, LEL, RSH, REL, LBLADE, RBLADE)

NB = 12
XAX = np.array([1., 0., 0.])
TILT_MAX = 45.0


def qconj(q): return q * np.array([1., -1., -1., -1.])
def twist(loc, ax): return np.degrees(2 * np.arctan2((loc[:, 1:] * ax).sum(-1), loc[:, 0]))
def sensor_yaw(Gj): v = qrot_np(Gj, XAX); return np.degrees(np.arctan2(v[:, 2], v[:, 0]))
def elev(Gj, ax): d = qrot_np(Gj, ax); return np.degrees(np.arcsin(np.clip(d[:, 1], -1, 1)))
def yaw_bin(deg): return ((deg + 180) // 30).astype(int) % NB


def harm(psi_rad):
    return np.stack([np.sin(psi_rad), np.cos(psi_rad), np.sin(2 * psi_rad), np.cos(2 * psi_rad)], 1)


def collect(files, parent, order, bax, pj_sj_ax, nframes):
    """Per take: (heading-bin mean twist[NB], frame count[NB]) in the well-conditioned tilt band."""
    pj, sj, ax = pj_sj_ax
    out = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        q = d['quats'].astype(np.float64); T = q.shape[0]
        G = fk_world(q, parent, order)
        fr = np.arange(0, T, max(1, T // nframes))
        vals = twist(qmul_np(qconj(G[fr][:, pj]), G[fr][:, sj]), bax[sj])
        keep = np.abs(elev(G[fr][:, sj], bax[sj])) <= TILT_MAX
        hb = yaw_bin(sensor_yaw(G[fr][:, sj]))
        ym = np.full(NB, np.nan); nm = np.zeros(NB)
        for h in range(NB):
            m = keep & (hb == h)
            if m.sum() >= 5:
                ym[h] = vals[m].mean(); nm[h] = m.sum()
        out.append((ym, nm))
    return out


def fit(takes, bin_centers_rad):
    """Weighted LS for {A_t per take, shared B,C,D,E}. Returns coeffs, fitted-curve fn, R^2, resid."""
    H = harm(bin_centers_rad)                       # (NB,4)
    rows_X, rows_y, rows_w, take_of = [], [], [], []
    ntk = len(takes)
    for t, (ym, nm) in enumerate(takes):
        for h in range(NB):
            if not np.isnan(ym[h]) and nm[h] > 0:
                x = np.zeros(ntk + 4); x[t] = 1.0; x[ntk:] = H[h]
                rows_X.append(x); rows_y.append(ym[h]); rows_w.append(nm[h]); take_of.append(t)
    X = np.array(rows_X); y = np.array(rows_y); w = np.array(rows_w)
    W = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
    A = beta[:ntk]; BCDE = beta[ntk:]
    # variance explained by harmonics beyond per-take constants
    yhat_full = X @ beta
    Xc = X.copy(); Xc[:, ntk:] = 0.0
    bc, *_ = np.linalg.lstsq(Xc * W[:, None], y * W, rcond=None)
    yhat_const = Xc @ bc
    ss_const = (w * (y - yhat_const) ** 2).sum()
    ss_full = (w * (y - yhat_full) ** 2).sum()
    r2 = 1 - ss_full / (ss_const + 1e-12)
    rms = np.sqrt(ss_full / w.sum())
    return A, BCDE, r2, rms


def curve(BCDE, psi_rad): return harm(psi_rad) @ BCDE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--nframes", type=int, default=4000)
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*beta.npz")))
    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    centers = np.radians(np.array([-180 + 30 * b for b in range(NB)], float))

    SENSORS = {'L-elbow (forearm)': (LSH, LEL, None), 'R-elbow (forearm)': (RSH, REL, None),
               'L-shldr (upper-arm)': (LBLADE, LSH, None), 'R-shldr (upper-arm)': (RBLADE, RSH, None)}

    rng = np.random.default_rng(0)
    fine = np.radians(np.linspace(-180, 180, 25))
    for name, psax in SENSORS.items():
        takes = collect(files, parent, order, bax, psax, args.nframes)
        A, BCDE, r2, rms = fit(takes, centers)
        B, C, D, E = BCDE
        H1 = np.hypot(B, C); ph1 = np.degrees(np.arctan2(B, C))       # heading of 1st-harmonic max
        H2 = np.hypot(D, E); ph2 = np.degrees(np.arctan2(D, E)) / 2.0
        # bootstrap over takes
        H1s, H2s = [], []
        for _ in range(args.boot):
            samp = [takes[i] for i in rng.integers(0, len(takes), len(takes))]
            _, bb, _, _ = fit(samp, centers)
            H1s.append(np.hypot(bb[0], bb[1])); H2s.append(np.hypot(bb[2], bb[3]))
        H1lo, H1hi = np.percentile(H1s, [16, 84]); H2lo, H2hi = np.percentile(H2s, [16, 84])

        print(f"\n=== {name} ===")
        print(f"  shared harmonics:  B={B:+.2f} C={C:+.2f} D={D:+.2f} E={E:+.2f}")
        print(f"  1st (hard-iron):  amp {H1:5.2f} deg [{H1lo:.1f},{H1hi:.1f}]  peak heading {ph1:+.0f} deg")
        print(f"  2nd (soft-iron):  amp {H2:5.2f} deg [{H2lo:.1f},{H2hi:.1f}]")
        print(f"  harmonics explain {100*r2:4.1f}% of twist variance beyond per-take constants; fit RMS {rms:.1f} deg")
        # pooled data (de-mean each take by its own fitted A, then average) vs fitted curve
        pooled = np.full(NB, np.nan); cnt = np.zeros(NB)
        for t, (ym, nm) in enumerate(takes):
            for h in range(NB):
                if not np.isnan(ym[h]):
                    v = ym[h] - A[t]
                    pooled[h] = v * nm[h] if np.isnan(pooled[h]) else pooled[h] + v * nm[h]
                    cnt[h] += nm[h]
        pooled = np.where(cnt > 0, pooled / np.maximum(cnt, 1), np.nan)
        fit_at_bins = curve(BCDE, centers)
        print(f"  {'heading':9s}" + " ".join(f"{-180+30*b:+5d}" for b in range(NB)))
        print(f"  {'data':9s}" + " ".join("   . " if np.isnan(pooled[b]) else f"{pooled[b]:+5.1f}" for b in range(NB)))
        print(f"  {'fit':9s}" + " ".join(f"{fit_at_bins[b]:+5.1f}" for b in range(NB)))

    print("\nThe 'fit' row (shared B..E) is the session magnetization curve delta(psi) for that sensor,")
    print("with the per-take constant removed. Tight CI + high %% explained = a trustworthy correction.")


if __name__ == "__main__":
    main()
