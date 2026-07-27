"""Correct the chest roll-bleed and test whether it improves arm symmetry in the chest frame.

The chest sensor carries a session-stable, heading-dependent ROLL (tilt-bleed; 6.8 deg, 68% explained).
Because the SMPL arm pose is the arm RELATIVE to the chest, this roll tilts the arm reference frame
per facing -> a rolled chest makes world-symmetric arms look asymmetric (one abducted, one adducted) =
the asymmetric hang. We fit the session-shared roll shape alpha(psi_chest) (lateral-lean, per-take
constant dropped), apply a world rotation of -alpha about the chest's forward axis (un-rolling it),
and measure the L/R upper-arm mirror asymmetry IN THE CHEST FRAME (mirror across the chest's actual,
non-horizontalized sagittal plane = what the render sees) at the symmetric crossings, before vs after.

Control: the horizontalized WORLD mirror residual (roll projects out of it) should barely change --
confirming the effect is specifically the chest-relative (rendered) symmetry, not the world metric.
"""
import glob
import os
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from correct_upper_arm_offset import forward_kinematics, qmul, qrot, norm_rows, LSH, RSH
from diag_magnetometer_deviation import MIDV
from fit_session_deviation import fit, NB

BETA = "/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/Subject7_Bharathanatyam/beta"
UP = np.array([0., 1., 0.]); X = np.array([1., 0, 0.]); Zc = np.array([0., 0, 1.])
centers = np.radians(np.array([-180 + 30 * b for b in range(NB)], float))


def chest_axes(G):
    uw = qrot(G[:, MIDV], UP)                                   # local-y is chest up (identified earlier)
    fwd = qrot(G[:, MIDV], X); fwd = fwd - UP * fwd[:, 1:2]; fwd = norm_rows(fwd)
    return uw, fwd


def chest_roll(G):
    uw, fwd = chest_axes(G)
    lat = np.cross(np.broadcast_to(UP, fwd.shape), fwd)
    roll = np.degrees(np.arcsin(np.clip((uw * lat).sum(1), -1, 1)))
    head = np.degrees(np.arctan2(fwd[:, 2], fwd[:, 0]))
    return roll, head, uw


def curve(B, psi):
    return B[0] * np.sin(psi) + B[1] * np.cos(psi) + B[2] * np.sin(2 * psi) + B[3] * np.cos(2 * psi)


def fit_chest_shape(files):
    takes = []
    for f in files:
        G = forward_kinematics(np.load(f, allow_pickle=True)['quats'].astype(np.float64))
        roll, head, uw = chest_roll(G)
        ok = (uw[:, 1] > 0.85) & (np.abs(roll) < 35); hb = ((head + 180) // 30).astype(int) % NB
        ym = np.full(NB, np.nan); cnt = np.zeros(NB)
        for k in range(NB):
            m = ok & (hb == k)
            if m.sum() >= 20: ym[k] = roll[m].mean(); cnt[k] = m.sum()
        takes.append((ym, cnt))
    _, B, r2, _ = fit(takes, centers)
    return B, r2


def apply_chest_roll_corr(G, B):
    """Un-roll the chest: world rotation of -alpha(psi) about the chest's horizontal forward axis."""
    G = G.copy()
    _, fwd = chest_axes(G)
    head = np.degrees(np.arctan2(fwd[:, 2], fwd[:, 0]))
    a = np.radians(-curve(B, np.radians(head)))                 # corrective roll angle
    q = np.zeros((len(a), 4)); q[:, 0] = np.cos(a / 2); q[:, 1:] = np.sin(a / 2)[:, None] * fwd
    G[:, MIDV] = qmul(q, G[:, MIDV])
    return G


def chest_frame_asym(G, frames):
    """L/R upper-arm mirror asym across the chest's ACTUAL sagittal plane (non-horizontalized)."""
    lat = norm_rows(qrot(G[frames][:, MIDV], Zc))              # chest lateral axis (carries the roll)
    M = np.eye(3)[None] - 2 * lat[:, :, None] * lat[:, None, :]
    nL = norm_rows(qrot(G[frames][:, LSH], X)); nR = norm_rows(qrot(G[frames][:, RSH], -X))
    return np.degrees(np.arccos(np.clip((np.einsum('nij,nj->ni', M, nL) * nR).sum(1), -1, 1)))


def world_asym(G, frames):
    """Control: horizontalized world mirror (roll projects out)."""
    sh = qrot(G[frames][:, MIDV], Zc); sh = norm_rows(sh - UP * (sh * UP).sum(1, keepdims=True))
    M = np.eye(3)[None] - 2 * sh[:, :, None] * sh[:, None, :]
    nL = norm_rows(qrot(G[frames][:, LSH], X)); nR = norm_rows(qrot(G[frames][:, RSH], -X))
    return np.degrees(np.arccos(np.clip((np.einsum('nij,nj->ni', M, nL) * nR).sum(1), -1, 1)))


def crossings(G):
    sh = norm_rows(qrot(G[:, MIDV], Zc)); sh = norm_rows(sh - UP * (sh * UP).sum(1, keepdims=True))
    M = np.eye(3)[None] - 2 * sh[:, :, None] * sh[:, None, :]
    nL = norm_rows(qrot(G[:, LSH], X)); nR = norm_rows(qrot(G[:, RSH], -X))
    mL = norm_rows(qrot(G[:, LSH], Zc)); mR = norm_rows(qrot(G[:, RSH], Zc))
    s = np.linalg.norm(np.einsum('nij,nj->ni', M, nL) - nR, axis=1) + \
        np.linalg.norm(np.einsum('nij,nj->ni', M, mL) - mR, axis=1)
    c, _ = find_peaks(-s, prominence=0.12, distance=8); return c


def main():
    files = sorted(glob.glob(os.path.join(BETA, "*beta.npz")))
    B, r2 = fit_chest_shape(files)
    print(f"chest roll shape: amp {np.hypot(B[0],B[1])+np.hypot(B[2],B[3]):.1f} deg, explains {100*r2:.0f}%\n")

    cf_b, cf_a, w_b, w_a, rc_b, rc_a = [], [], [], [], [], []
    for f in files:
        G = forward_kinematics(np.load(f, allow_pickle=True)['quats'].astype(np.float64))
        c = crossings(G)
        if len(c) == 0: continue
        Gc = apply_chest_roll_corr(G, B)
        cf_b += list(chest_frame_asym(G, c)); cf_a += list(chest_frame_asym(Gc, c))
        w_b += list(world_asym(G, c)); w_a += list(world_asym(Gc, c))
        r0, _, _ = chest_roll(G); r1, _, _ = chest_roll(Gc)
        rc_b.append(np.abs(r0).mean()); rc_a.append(np.abs(r1).mean())
    cf_b, cf_a, w_b, w_a = map(np.array, (cf_b, cf_a, w_b, w_a))
    print(f"self-check: |chest roll| mean {np.mean(rc_b):.1f} -> {np.mean(rc_a):.1f} deg (should drop)")
    print(f"\narm L/R asym at {len(cf_b)} crossings, before -> after chest un-roll:")
    print(f"  CHEST-FRAME (rendered)  : mean {cf_b.mean():5.1f} -> {cf_a.mean():5.1f}  median {np.median(cf_b):5.1f} -> {np.median(cf_a):5.1f}")
    print(f"  world horiz (control)   : mean {w_b.mean():5.1f} -> {w_a.mean():5.1f}  median {np.median(w_b):5.1f} -> {np.median(w_a):5.1f}")
    print("\nIf chest-frame asym DROPS while world control barely moves => the chest roll-bleed was")
    print("corrupting the rendered (chest-relative) arm symmetry, and un-rolling the chest fixes it.")


if __name__ == "__main__":
    main()