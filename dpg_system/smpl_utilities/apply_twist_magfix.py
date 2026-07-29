"""Child-frame axial-twist corrector for roll-bleed magnetization (writes *_twistfix.npz).

The world-yaw magfix corrected the wrong DOF for the (left, more-magnetized) arm: there the
magnetometer error bleeds into the fusion's ROLL estimate, so it appears as a heading-dependent
TWIST about the sensor's own bone axis -- present even when the bone is horizontal (where a pure
world-yaw error would project to zero twist). The right correction is therefore a CHILD-frame
axial twist, postmultiplied onto the sensor's world orientation:

    G_corr[j] = G_meas[j] (x) Twist(-alpha_j(psi_j), bone_axis_local[j])

A twist about the bone's own axis leaves the bone's heading psi unchanged, so no iteration is needed.
alpha_j(psi) (the session-shared roll-bleed curve) is fit from the HORIZONTAL band (|elev|<20) where
the world-yaw projection delta*sin(elev) ~ 0, so it isolates the roll-bleed; only the heading-varying
harmonic (zero-mean) is applied -- the constant offset is left alone (ambiguous: real pronation vs
calibration). Fitting is DOWN-CHAIN: forearm first, then the hand against the corrected forearm, so
each sensor's curve is read against a cleaned parent. Forearms + hands only (sensors we've validated);
shoulders left for later. Self-check re-fits the horizontal twist amplitude after correction.
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np

from diag_magnetometer_deviation import (load_skeleton, fk_world, qmul_np, qrot_np,
                                         RSH, REL, RWR, LSH, LEL, LWR)
from fit_session_deviation import fit, curve, NB
from apply_forearm_magfix import relocal_from_world

XAX = np.array([1., 0., 0.])
ELEV_MAX = 20.0


def qconj(q): return q * np.array([1., -1., -1., -1.])
def twist(loc, ax): return np.degrees(2 * np.arctan2((loc[:, 1:] * ax).sum(-1), loc[:, 0]))
def sensor_yaw(Gj): v = qrot_np(Gj, XAX); return np.arctan2(v[:, 2], v[:, 0])     # radians
def elevd(Gj, ax): d = qrot_np(Gj, ax); return np.degrees(np.arcsin(np.clip(d[:, 1], -1, 1)))
def yaw_bin(rad): return ((np.degrees(rad) + 180) // 30).astype(int) % NB


def twist_quat(angle_rad, axis):
    """quaternion (w-first) for a twist about a (local) unit axis, for postmultiply."""
    h = angle_rad / 2.0
    q = np.zeros(angle_rad.shape + (4,))
    q[..., 0] = np.cos(h); q[..., 1:] = np.sin(h)[..., None] * axis
    return q


def apply_twist(G, j, coeffs, bax):
    """Postmultiply sensor j's world orientation by Twist(-alpha(psi_j)) about its bone axis."""
    G = G.copy()
    psi = sensor_yaw(G[:, j])
    alpha = np.radians(curve(coeffs, psi))           # heading-varying roll-bleed, deg->rad
    G[:, j] = qmul_np(G[:, j], twist_quat(-alpha, bax[j]))
    return G


def gather_horiz_twist(files, parent, order, bax, pj, sj, corrections, stride):
    """Per-take (binned-mean twist[NB], count[NB]) in the horizontal band, after prior corrections."""
    centers = None
    takes = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        q = d['quats'].astype(np.float64); T = q.shape[0]
        G = fk_world(q, parent, order)
        for cj, cc in corrections:
            G = apply_twist(G, cj, cc, bax)
        fr = np.arange(0, T, stride)
        vals = twist(qmul_np(qconj(G[fr][:, pj]), G[fr][:, sj]), bax[sj])
        keep = np.abs(elevd(G[fr][:, sj], bax[sj])) < ELEV_MAX
        hb = yaw_bin(sensor_yaw(G[fr][:, sj]))
        ym = np.full(NB, np.nan); nm = np.zeros(NB)
        for h in range(NB):
            m = keep & (hb == h)
            if m.sum() >= 5:
                ym[h] = vals[m].mean(); nm[h] = m.sum()
        takes.append((ym, nm))
    return takes


def fit_alpha(files, parent, order, bax, pj, sj, corrections, centers, stride):
    takes = gather_horiz_twist(files, parent, order, bax, pj, sj, corrections, stride)
    A, BCDE, r2, rms = fit(takes, centers)
    return BCDE, r2, np.hypot(BCDE[0], BCDE[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir")
    ap.add_argument("targets", nargs="+", help="file(s) to correct, or 'all'")
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    def_path = Path(__file__).resolve().parent.parent / 'definition.xml'
    parent, order, bax = load_skeleton(def_path)
    files = sorted(glob.glob(os.path.join(args.session_dir, "*beta.npz")))
    centers = np.radians(np.array([-180 + 30 * b for b in range(NB)], float))

    # --- down-chain session fit of the roll-bleed curve alpha(psi) ---
    print("Fitting session-shared roll-bleed alpha(psi) (horizontal band, heading-varying):")
    coeffs = {}
    for tag, P, S in [('L-forearm', LSH, LEL), ('R-forearm', RSH, REL)]:
        BCDE, r2, amp = fit_alpha(files, parent, order, bax, P, S, [], centers, args.stride)
        coeffs[S] = BCDE
        print(f"  {tag:10s}: 1st-harm {amp:5.1f} deg, explains {100*r2:2.0f}%  (corrected vs upper-arm)")
    for tag, P, S, par in [('L-hand', LEL, LWR, LEL), ('R-hand', REL, RWR, REL)]:
        BCDE, r2, amp = fit_alpha(files, parent, order, bax, P, S, [(par, coeffs[par])], centers, args.stride)
        coeffs[S] = BCDE
        print(f"  {tag:10s}: 1st-harm {amp:5.1f} deg, explains {100*r2:2.0f}%  (vs CORRECTED forearm)")

    order_apply = [LEL, REL, LWR, RWR]
    targets = files if args.targets == ["all"] else args.targets
    for f in targets:
        d = np.load(f, allow_pickle=True)
        G = fk_world(d['quats'].astype(np.float64), parent, order)
        for j in order_apply:
            G = apply_twist(G, j, coeffs[j], bax)
        q_new = relocal_from_world(G, parent, order)
        out = {k: d[k] for k in d.files}; out['quats'] = q_new
        outpath = f.replace('_beta.npz', '_twistfix.npz')
        if outpath == f:
            outpath = f.replace('.npz', '_twistfix.npz')
        np.savez(outpath, **out)
        print(f"  wrote {os.path.basename(outpath)}")

    # --- self-check: horizontal-band twist amplitude before vs after, on the session ---
    print("\nself-check (horizontal-band 1st-harm twist amplitude, should drop toward 0):")
    for tag, P, S, par in [('L-forearm', LSH, LEL, None), ('R-forearm', RSH, REL, None),
                           ('L-hand', LEL, LWR, LEL), ('R-hand', REL, RWR, REL)]:
        corr0, corrA = [], [(S, coeffs[S])]
        if par is not None:
            corr0 = [(par, coeffs[par])]; corrA = [(par, coeffs[par]), (S, coeffs[S])]
        _, _, a0 = fit_alpha(files, parent, order, bax, P, S, corr0, centers, args.stride)
        _, _, a1 = fit_alpha(files, parent, order, bax, P, S, corrA, centers, args.stride)
        print(f"  {tag:10s}: {a0:5.1f} -> {a1:5.1f} deg")


if __name__ == "__main__":
    main()
