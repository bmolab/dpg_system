"""Per-sensor tilt-bleed (roll) magnetization assessment across the whole session.

Tilt is gravity-referenced, so each sensor's ROLL about its bone axis (relative to the gravity-up
reference) is measurable ABSOLUTELY, with no other sensor needed. Magnetization that bleeds into the
fusion's tilt estimate shows up as a heading-dependent roll. We bin each sensor's gravity-roll by its
own heading across all 11 takes and fit per-take-constant + session-shared harmonic shape: the
session-consistent heading-dependent shape = the tilt-bleed (real posture/motion varies per dance and
is absorbed by the per-take constant + averaged out). Reports, per sensor: shared roll-bleed shape
amplitude, %% of variance it explains beyond per-take constants (the SNR -- low for highly mobile
limbs where pose dominates), the per-take-constant spread (re-mount/calibration scatter), and usable
frames. Gimbal frames (bone within 30 deg of vertical, where roll is undefined) are gated out.

YAW deviation is NOT covered here: it has no absolute reference (gauge), so it's only measurable
relative to body/parent (headlock / twist-by-heading) -- a separate, gauge-limited assessment.
"""
import glob
import os
from pathlib import Path

import numpy as np

from diag_magnetometer_deviation import load_skeleton, fk_world, qrot_np, IDX_TO_NAME
from fit_session_deviation import fit, NB

UP = np.array([0., 1., 0.])
BETA = "/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/Subject7_Bharathanatyam/beta"
SENSORS = [4, 31, 32, 1, 17, 2, 13, 5, 9, 10, 27, 19, 23, 24, 14, 12, 8, 28, 26, 22]


def main():
    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    files = sorted(glob.glob(os.path.join(BETA, "*beta.npz")))
    centers = np.radians(np.array([-180 + 30 * b for b in range(NB)], float))

    # per sensor: list over takes of (binned roll mean[NB], count[NB])
    pertake = {j: [] for j in SENSORS}
    for f in files:
        G = fk_world(np.load(f, allow_pickle=True)['quats'].astype(np.float64), parent, order)
        for j in SENSORS:
            d = qrot_np(G[:, j], bax[j]); d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9
            # perpendicular sensor axis (orthogonalize a non-bone local axis against the bone)
            perp_local = np.array([0, 0, 1.]) if abs(bax[j][2]) < 0.9 else np.array([0, 1., 0])
            u = qrot_np(G[:, j], perp_local); u = u - (u * d).sum(-1, keepdims=True) * d
            u /= np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9
            r = UP - (UP * d).sum(-1, keepdims=True) * d                    # gravity-up made perp to bone
            r /= np.linalg.norm(r, axis=-1, keepdims=True) + 1e-9
            roll = np.degrees(np.arctan2((np.cross(r, u) * d).sum(-1), (r * u).sum(-1)))
            head = np.degrees(np.arctan2(d[:, 2], d[:, 0]))
            ok = np.abs(d[:, 1]) < 0.87                                     # bone >30deg from vertical
            hb = ((head + 180) // 30).astype(int) % NB
            ym = np.full(NB, np.nan); cnt = np.zeros(NB)
            for k in range(NB):
                m = ok & (hb == k)
                if m.sum() >= 20:
                    ym[k] = roll[m].mean(); cnt[k] = m.sum()
            pertake[j].append((ym, cnt))

    rows = []
    for j in SENSORS:
        takes = pertake[j]
        if sum(np.sum(~np.isnan(t[0])) for t in takes) < 30:
            rows.append((j, np.nan, np.nan, np.nan, 0)); continue
        A, BCDE, r2, rms = fit(takes, centers)
        amp = np.hypot(BCDE[0], BCDE[1]) + np.hypot(BCDE[2], BCDE[3])       # 1st+2nd harmonic magnitude
        nfr = int(sum(np.nansum(t[1]) for t in takes))
        rows.append((j, amp, 100 * r2, np.std(A), nfr))

    print("PER-SENSOR TILT-BLEED (gravity-roll), session-pooled. amp=shared heading-shape (deg);")
    print("expl%=variance explained beyond per-take const (SNR); const-std=per-take roll scatter (deg).\n")
    print(f"  {'sensor':22s} {'rollbleed-amp':>13} {'expl%':>6} {'const-std':>9} {'frames':>8}")
    for j, amp, ex, cs, nfr in sorted(rows, key=lambda r: (-(r[1] * (r[2] / 100)) if not np.isnan(r[1]) else 1)):
        nm = IDX_TO_NAME.get(j, str(j))
        if np.isnan(amp):
            print(f"  {nm:22s} {'(insufficient)':>13}")
        else:
            flag = '  <-- tilt-bleed' if (ex > 40 and amp > 2) else ('  (per-take roll)' if cs > 4 and ex < 25 else '')
            print(f"  {nm:22s} {amp:13.1f} {ex:6.0f} {cs:9.1f} {nfr:8d}{flag}")
    print("\nHigh amp + high expl% = session-stable heading-dependent tilt-bleed (magnetization).")
    print("Low expl% + high const-std = per-take mounting/calibration roll, not a heading bleed.")
    print("Low everything = clean in roll (says nothing about yaw, which needs a relative assessment).")


if __name__ == "__main__":
    main()
