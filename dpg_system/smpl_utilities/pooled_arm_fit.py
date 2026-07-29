"""Session-pooled heading-dependent upper-arm correction, fit from symmetry crossings of ALL takes.

No single take works: jathiswaram has lots of symmetry but only front-facing; padam has heading
spread but little symmetry. The magnetization is session-stable, so we POOL the auto-detected
symmetric crossings across every take -- jathiswaram supplies clean front symmetry in volume, the
others supply side/back headings -- giving symmetric ground truth across the heading circle. We then
fit, in ONE nonlinear least-squares over the pooled crossings, a session-shared per-arm:

    constant C (rotvec)  +  heading-dependent yaw delta(psi)=b sin psi + c cos psi

by minimizing the L/R upper-arm mirror residual (distal + secondary axes), with a ridge pinning the
unobservable common-mode gauge. Validated on padam's known-clean hand-picked sym windows (the test
that exposed the single-take fit's failure). Directions are orientation-derived (qrot(shoulder,X)),
matching apply_correction; the sagittal mirror axis uses shoulder positions (chest fallback).
"""
import glob
import os
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

import correct_upper_arm_offset as cuo
from correct_upper_arm_offset import (forward_kinematics, qmul, qrot, rotvec_to_q, head_yaw,
                                      norm_rows, LSH, RSH, MIDV)

UP = np.array([0., 1., 0.]); X = np.array([1., 0., 0.]); Z = np.array([0., 0., 1.])
BETA_DIR = "/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/Subject7_Bharathanatyam/beta"


def dirs(G, P):
    have = P is not None and np.any(P[:, LSH])
    latv = (P[:, LSH] - P[:, RSH]) if have else qrot(G[:, MIDV], X)
    lat = norm_rows(latv - UP * (latv @ UP)[:, None])
    M = np.eye(3)[None] - 2 * lat[:, :, None] * lat[:, None, :]
    nL = norm_rows(qrot(G[:, LSH], X)); nR = norm_rows(qrot(G[:, RSH], -X))
    mL = norm_rows(qrot(G[:, LSH], Z)); mR = norm_rows(qrot(G[:, RSH], Z))
    return M, nL, nR, mL, mR


def collect(files):
    gL, gR, MM = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        G = forward_kinematics(d['quats'].astype(np.float64))
        P = d['positions'].astype(np.float64) if 'positions' in d.files else None
        M, nL, nR, mL, mR = dirs(G, P)
        s = np.linalg.norm(np.einsum('nij,nj->ni', M, nL) - nR, axis=1) + \
            np.linalg.norm(np.einsum('nij,nj->ni', M, mL) - mR, axis=1)
        c, _ = find_peaks(-s, prominence=0.12, distance=8)
        gL.append(G[c][:, LSH]); gR.append(G[c][:, RSH]); MM.append(M[c])
    return np.concatenate(gL), np.concatenate(gR), np.concatenate(MM)


def mirror_resid(gL, gR, M, x):
    cl, cr, hyl, hyr = x[:3], x[3:6], x[6:8], x[8:10]
    gLc = head_yaw(qmul(np.broadcast_to(rotvec_to_q(cl), gL.shape), gL), 1.0, hyl)
    gRc = head_yaw(qmul(np.broadcast_to(rotvec_to_q(cr), gR.shape), gR), -1.0, hyr)
    nLc = norm_rows(qrot(gLc, X)); nRc = norm_rows(qrot(gRc, -X))
    mLc = norm_rows(qrot(gLc, Z)); mRc = norm_rows(qrot(gRc, Z))
    r1 = (np.einsum('nij,nj->ni', M, nLc) - nRc).ravel()
    r2 = (np.einsum('nij,nj->ni', M, mLc) - mRc).ravel()
    return r1, r2, nLc, nRc


def distal_asym(gL, gR, M, x):
    _, _, nLc, nRc = mirror_resid(gL, gR, M, x)
    return np.degrees(np.arccos(np.clip((np.einsum('nij,nj->ni', M, nLc) * nRc).sum(1), -1, 1)))


def main():
    files = sorted(glob.glob(os.path.join(BETA_DIR, "*beta.npz")))
    gL, gR, M = collect(files)
    print(f"pooled {len(gL)} symmetric crossings across {len(files)} takes")

    def resid(x):
        r1, r2, _, _ = mirror_resid(gL, gR, M, x)
        return np.concatenate([r1, r2, 0.05 * x])

    a0 = distal_asym(gL, gR, M, np.zeros(10))
    sol = least_squares(resid, np.zeros(10), method='lm')
    a1 = distal_asym(gL, gR, M, sol.x)
    cl, cr, hyl, hyr = sol.x[:3], sol.x[3:6], sol.x[6:8], sol.x[8:10]
    print(f"pooled crossing asym: {a0.mean():.1f} -> {a1.mean():.1f} deg (median {np.median(a0):.1f} -> {np.median(a1):.1f})")
    print(f"  C_L {np.round(np.degrees(cl),1)}  C_R {np.round(np.degrees(cr),1)} deg")
    print(f"  heading-yaw |L|={np.degrees(np.hypot(*hyl)):.0f}@{np.degrees(np.arctan2(*hyl)):+.0f}  "
          f"|R|={np.degrees(np.hypot(*hyr)):.0f}@{np.degrees(np.arctan2(*hyr)):+.0f} deg")

    # --- validate on padam's known-clean hand-picked sym windows ---
    d = np.load(os.path.join(BETA_DIR, "Subject7_take_padam_a_beta.npz"), allow_pickle=True)
    G = forward_kinematics(d['quats'].astype(np.float64)); P = d['positions'].astype(np.float64)
    Mp, nLp, nRp, mLp, mRp = dirs(G, P)
    print("\nVALIDATION on padam hand-picked clean sym windows (distal asym before -> after):")
    for a, b, lbl in [(7880, 8500, 'fwd'), (20660, 20953, 'back'), (10560, 10692, 'back')]:
        sl = slice(a, b)
        b0 = distal_asym(G[sl][:, LSH], G[sl][:, RSH], Mp[sl], np.zeros(10)).mean()
        b1 = distal_asym(G[sl][:, LSH], G[sl][:, RSH], Mp[sl], sol.x).mean()
        print(f"  {a}:{b} ({lbl}): {b0:5.1f} -> {b1:5.1f}")

    np.savez(os.path.join(BETA_DIR, "session_armfit.npz"), cl=cl, cr=cr, hy_l=hyl, hy_r=hyr)
    print(f"\nwrote session-shared fit -> session_armfit.npz (load in shadow_arm_correct node)")


if __name__ == "__main__":
    main()
