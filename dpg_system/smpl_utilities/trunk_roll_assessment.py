"""Trunk LATERAL-roll (abduction-relevant) tilt-bleed assessment -- axes identified, not assumed.

Correcting the earlier error (local-x is the LATERAL axis, not forward, so the prior 'roll' was
actually pitch). Here, per sensor, we IDENTIFY the local up axis (most vertical) and the local
lateral axis (most aligned with the body shoulder-line), then measure ROLL = the lateral lean of the
sensor's up-axis (the DOF that tilts the arm reference left/right -> asymmetric abduction/hang).
Session-pooled, per-take constant + shared heading shape. Only a session-stable lateral roll could be
the asymmetric-hang culprit; a per-take constant is a calibration/re-strap (squarable); a small/low-
expl shape means the core roll is not the cause.
"""
import glob
import os
from pathlib import Path

import numpy as np

from correct_upper_arm_offset import forward_kinematics, qrot, norm_rows, LSH, RSH
from diag_magnetometer_deviation import IDX_TO_NAME
from fit_session_deviation import fit, NB

BETA = "/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/Subject7_Bharathanatyam/beta"
UP = np.array([0., 1., 0.])
TRUNK = [4, 31, 32, 1, 17, 13, 27]   # pelvis, spinePelvis, lowerVert, midVert(chest), upperVert, L/R blade
SIGNED = [(s, np.array(v, float) * s) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1]) for s in (1, -1)]
centers = np.radians(np.array([-180 + 30 * b for b in range(NB)], float))


def identify_axes(G, j, shoulderline):
    """Return (up_local, lateral_local) signed unit vectors for sensor j's local frame."""
    means = {tuple(v): qrot(G[:, j], v).mean(0) for _, v in SIGNED}
    up_local = max(means, key=lambda v: np.dot(means[v], UP))
    lat_local = max(means, key=lambda v: np.dot(norm_rows(qrot(G[:, j], np.array(v))).mean(0), shoulderline))
    return np.array(up_local), np.array(lat_local)


def roll_head(G, j, up_local, lat_local):
    up_w = norm_rows(qrot(G[:, j], up_local))
    lat_w = qrot(G[:, j], lat_local); lat_h = norm_rows(lat_w - UP * (lat_w * UP).sum(1, keepdims=True))
    roll = np.degrees(np.arcsin(np.clip((up_w * lat_h).sum(1), -1, 1)))     # lateral lean of up-axis
    fwd = np.cross(up_w, lat_h)                                             # forward (for heading)
    head = np.degrees(np.arctan2(fwd[:, 2], fwd[:, 0]))
    return roll, head, up_w


def main():
    files = sorted(glob.glob(os.path.join(BETA, "*beta.npz")))
    # fix per-sensor local axes from the first take (local frame is constant across takes)
    G0 = forward_kinematics(np.load(files[0], allow_pickle=True)['quats'].astype(np.float64))
    P0 = np.load(files[0], allow_pickle=True)['positions'].astype(np.float64)
    shoulderline = norm_rows((P0[:, LSH] - P0[:, RSH]).mean(0)[None])[0]
    axes = {j: identify_axes(G0, j, shoulderline) for j in TRUNK}

    print("Trunk LATERAL-roll bleed (axes identified). amp=session-shared heading shape (deg),")
    print("expl%=variance beyond per-take const, const-std=per-take roll scatter.\n")
    print(f"  {'sensor':22s} {'up=':>5} {'lat=':>5} {'roll-amp':>9} {'expl%':>6} {'const-std':>9}")
    for j in TRUNK:
        up_l, lat_l = axes[j]
        takes = []
        for f in files:
            G = forward_kinematics(np.load(f, allow_pickle=True)['quats'].astype(np.float64))
            roll, head, up_w = roll_head(G, j, up_l, lat_l)
            ok = (up_w[:, 1] > 0.85) & (np.abs(roll) < 35); hb = ((head + 180) // 30).astype(int) % NB
            ym = np.full(NB, np.nan); cnt = np.zeros(NB)
            for k in range(NB):
                m = ok & (hb == k)
                if m.sum() >= 20: ym[k] = roll[m].mean(); cnt[k] = m.sum()
            takes.append((ym, cnt))
        if sum(np.sum(~np.isnan(t[0])) for t in takes) < 30:
            print(f"  {IDX_TO_NAME.get(j,j):22s}  (insufficient)"); continue
        A, B, r2, _ = fit(takes, centers)
        amp = np.hypot(B[0], B[1]) + np.hypot(B[2], B[3])
        ax = lambda v: 'xyz'[int(np.argmax(np.abs(v)))] if np.sum(v) >= 0 else '-' + 'xyz'[int(np.argmax(np.abs(v)))]
        print(f"  {IDX_TO_NAME.get(j,j):22s} {ax(up_l):>5} {ax(lat_l):>5} {amp:9.1f} {100*r2:6.0f} {np.std(A):9.1f}")
    print("\nSession-stable lateral roll (high amp + high expl%) => candidate hang culprit.")
    print("High const-std + low expl% => per-take calibration/re-strap roll (squarable, not bleed).")


if __name__ == "__main__":
    main()