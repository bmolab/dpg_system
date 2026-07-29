"""Apply the validated forearm magnetization correction and write *_magfix.npz.

Model: each Shadow IMU reports its own world orientation; the magnetometer error is a yaw about
world-vertical, delta_j(psi), a function of that sensor's own measured heading psi. We correct each
FOREARM sensor's world orientation independently:

    G_corr[:, j] = Yaw(-delta_j(psi_j)) (x) G_meas[:, j]      psi_j = measured heading of sensor j

then re-derive ALL parent-relative quats from the corrected world set. Consequences (exactly the
right semantics): the forearm world moves; the wrist/hand keep their OWN measured world orientation
(only their local-to-forearm changes); the upper arm is untouched. delta is the SESSION-SHARED,
world-yaw curve from the geometry-free headlock fit (phase cross-validated by the twist landscape):

    delta(psi) = b sin psi + c cos psi + d sin 2psi + e cos 2psi     (degrees)

fit once over the whole session here, so the correction can't go stale. Only the two forearm sensors
are corrected; shoulders are left alone (their curve isn't trustworthy yet). Positions are regenerated
from the corrected orientations so quats and positions stay consistent.
"""
import argparse
import glob
import os
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from diag_magnetometer_deviation import (load_skeleton, fk_world, qmul_np, qrot_np,
                                         LIMB_TO_IDX, LEL, REL)
from headlock_deviation import gather, solve

XAX = np.array([1., 0., 0.])
YUP = np.array([0., 1., 0.])


def parse_translate(def_path):
    """idx -> local bone-offset vector (child's offset in parent frame), for position FK."""
    root = ET.parse(def_path).getroot()
    tr = {j: np.zeros(3) for j in range(37)}

    def walk(node):
        idx = LIMB_TO_IDX.get(node.get('id'))
        if idx is not None:
            t = node.get('translate')
            if t:
                tr[idx] = np.array([float(v) for v in t.split()])
        for ch in node:
            walk(ch)

    walk(root)
    return tr


def heading(Gj):
    v = qrot_np(Gj, XAX)
    return np.arctan2(v[:, 2], v[:, 0])           # radians


def yaw_quat(theta):
    """quaternion (w-first) for rotation theta about world-vertical (+Y)."""
    h = theta / 2.0
    q = np.zeros(theta.shape + (4,))
    q[..., 0] = np.cos(h); q[..., 2] = np.sin(h)
    return q


def delta_deg(coeffs, psi):
    b, c, d, e = coeffs
    return b * np.sin(psi) + c * np.cos(psi) + d * np.sin(2 * psi) + e * np.cos(2 * psi)


def fk_positions(G, parent, order, tr, stored):
    T = G.shape[0]
    P = np.zeros((T, 37, 3))
    for j in order:
        p = parent[j]
        if p < 0:
            P[:, j] = stored[:, j]                # roots: keep measured world position
        else:
            P[:, j] = P[:, p] + qrot_np(G[:, p], tr[j])
    return P


def relocal_from_world(G, parent, order):
    """Inverse of fk_world: parent-relative quats from a world-orientation set."""
    Q = G.copy()
    for j in order:
        p = parent[j]
        if p >= 0:
            Q[:, j] = qmul_np(G[:, p] * np.array([1., -1., -1., -1.]), G[:, j])
    return Q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", help="folder of *beta.npz takes to FIT delta on")
    ap.add_argument("targets", nargs="+", help="file(s) to correct (e.g. one take, or 'all')")
    ap.add_argument("--turn-rate", type=float, default=0.3)
    # +1 (default) removes the measured +delta error: a world-+Y yaw by theta shifts the
    # atan2(z,x) heading by -theta, so theta=+delta subtracts delta. Verified by re-fitting the
    # headlock on the corrected orientations: +1 -> residual ~0.8 (R) / 5.7 (L) deg; -1 doubles it.
    ap.add_argument("--sign", type=float, default=1.0, help="correction sign (+1 removes +delta)")
    args = ap.parse_args()

    def_path = Path(__file__).resolve().parent.parent / 'definition.xml'
    parent, order, bax = load_skeleton(def_path)
    tr = parse_translate(def_path)
    files = sorted(glob.glob(os.path.join(args.session_dir, "*beta.npz")))

    # --- fit session-shared world-yaw delta for each forearm (geometry-free headlock) ---
    coeffs = {}
    for tag, sj in [('L', LEL), ('R', REL)]:
        X, y, _ = gather(files, parent, order, bax, sj, args.turn_rate)
        b, c, d, e = solve(X, y)
        coeffs[sj] = (b, c, d, e)
        amp1 = np.hypot(b, c); ph1 = np.degrees(np.arctan2(b, c)); amp2 = np.hypot(d, e)
        print(f"  {tag}-forearm delta(psi): b={b:+.2f} c={c:+.2f} d={d:+.2f} e={e:+.2f}"
              f"  | 1st {amp1:.1f}deg @ {ph1:+.0f}, 2nd {amp2:.1f}deg")

    targets = files if args.targets == ["all"] else args.targets
    for f in targets:
        d = np.load(f, allow_pickle=True)
        q = d['quats'].astype(np.float64)
        G = fk_world(q, parent, order)

        # verify position FK convention reproduces stored positions BEFORE trusting regeneration
        P0 = fk_positions(G, parent, order, tr, d['positions'])
        perr = np.linalg.norm(P0 - d['positions'], axis=-1).max()

        # apply Yaw(sign*delta) to each forearm sensor's world orientation
        for sj, cf in coeffs.items():
            psi = heading(G[:, sj])
            dd = np.radians(args.sign * delta_deg(cf, psi))
            G[:, sj] = qmul_np(yaw_quat(dd), G[:, sj])

        q_new = relocal_from_world(G, parent, order)
        P_new = fk_positions(G, parent, order, tr, d['positions'])

        out = {k: d[k] for k in d.files}
        out['quats'] = q_new
        out['positions'] = P_new if perr < 1e-3 else d['positions']
        outpath = f.replace('_beta.npz', '_magfix.npz')
        if outpath == f:
            outpath = f.replace('.npz', '_magfix.npz')
        np.savez(outpath, **out)
        wrist_shift = np.linalg.norm(P_new - P0, axis=-1)
        print(f"  wrote {os.path.basename(outpath)}  (pos-FK check {perr:.2e}m"
              f"{' OK' if perr < 1e-3 else ' >TOL: positions left unchanged'}; "
              f"max joint move {wrist_shift.max()*100:.1f}cm)")


if __name__ == "__main__":
    main()
