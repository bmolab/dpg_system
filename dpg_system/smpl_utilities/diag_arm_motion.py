"""Motion-based magnetization detector (not static pose).

The distortion lives in MOTION, not in any single pose: magnetization injects apparent limb
motion locked to the rate of heading change. Writing measured orientation as true + a
heading-dependent yaw, theta_meas = theta_true + delta(psi), differentiating gives

    omega_meas = omega_true + delta'(psi) * psi_dot

So when the dancer HOLDS the arm in her body frame while turning, a body-fixed arm must
co-rotate with the torso exactly; any BODY-RELATIVE limb yaw motion during a turn is spurious =
delta'(psi)*psi_dot. Real gestures are uncorrelated with psi_dot; magnetization is perfectly
correlated with it -- a discriminator that static pose-plausibility/impossibility cannot give.

This regresses body-relative forearm yaw-rate on body yaw-rate (binned by forearm heading) to
estimate delta'(psi), then integrates to delta(psi) -- a deviation curve derived from MOTION
alone, independent of symmetry or VPoser. Also reports forearm angular-speed spikes and any
wrist trajectory that sweeps THROUGH the torso (the impossible front->back transition). Diagnostic
only. Compare ORIGINAL vs a candidate correction: a good correction REMOVES the heading-locked
motion; a bad one (e.g. the VPoser fafix) ADDS it.
"""
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from diag_magnetometer_deviation import (load_skeleton, fk_world, qrot_np, PELV, LEL, LWR, UPPV)
from diag_arm_selfintersect import parse_offsets, fk_positions, seg_seg_dist, R_TORSO, R_FOREARM


def world_yaw(Gw, joint):
    dir = qrot_np(Gw[:, joint], np.array([1.0, 0.0, 0.0]))
    return np.unwrap(np.arctan2(dir[:, 2], dir[:, 0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("--turn-rate", type=float, default=20.0,
                    help="only use frames where the body turns faster than this (deg/s) for the fit")
    args = ap.parse_args()

    def_path = Path(__file__).resolve().parent.parent / 'definition.xml'
    parent, order, _ = load_skeleton(def_path)
    offsets = parse_offsets(def_path)
    d = np.load(args.infile, allow_pickle=True)
    q = d['quats'].astype(np.float64)
    fps = float(d['mocap_framerate']) if 'mocap_framerate' in d.files else 100.0
    Gw = fk_world(q, parent, order)

    psi_fa = world_yaw(Gw, LEL)
    psi_pel = world_yaw(Gw, PELV)
    decoupled = psi_fa - psi_pel                          # body-relative forearm yaw
    dt = 1.0 / fps
    psidot_pel = np.gradient(psi_pel) / dt                # body yaw rate (rad/s)
    ddecoupled = np.gradient(decoupled) / dt              # body-relative forearm yaw rate
    fa_speed = np.abs(np.gradient(psi_fa) / dt)           # forearm world yaw speed

    # --- heading-locked spurious motion: regress ddecoupled on psidot_pel, binned by forearm heading ---
    turn = np.abs(np.degrees(psidot_pel)) > args.turn_rate
    psi_deg = np.degrees((psi_fa + np.pi) % (2 * np.pi) - np.pi)
    NB = 12
    hb = ((psi_deg + 180) // 30).astype(int) % NB
    kprime = np.full(NB, np.nan)
    for b in range(NB):
        sel = turn & (hb == b)
        if sel.sum() > 30:
            x = psidot_pel[sel]; y = ddecoupled[sel]
            kprime[b] = (x @ y) / (x @ x + 1e-9)          # slope through origin = delta'(psi)
    # integrate delta'(psi) over psi -> delta(psi) (bin width 30deg), zero-mean
    valid = ~np.isnan(kprime)
    delta = np.zeros(NB)
    if valid.sum() >= 3:
        kf = np.where(valid, kprime, 0.0)
        delta = np.cumsum(kf) * np.radians(30)
        delta = delta - np.nanmean(np.where(valid, delta, np.nan))

    # how much body-relative motion (during turns) is heading-locked vs real gesture
    sel = turn
    expl = (np.nan_to_num(kprime)[hb[sel]] * psidot_pel[sel])
    resid = ddecoupled[sel] - expl
    frac = 1 - resid.var() / (ddecoupled[sel].var() + 1e-9)

    # --- swept self-intersection: does the wrist path between frames pass through the torso? ---
    pos = fk_positions(Gw, parent, order, offsets)
    w = pos[:, LWR]
    swept = seg_seg_dist(pos[:-1, PELV], pos[:-1, UPPV], w[:-1], w[1:])   # torso vs wrist step
    swept_pen = np.maximum((R_TORSO + R_FOREARM) - swept, 0.0)

    print(f"{Path(args.infile).name}: {len(q)} frames @ {fps:g}Hz")
    print(f"\nHEADING-LOCKED spurious motion (during body turns > {args.turn_rate:g} deg/s):")
    print(f"  fraction of body-relative forearm motion explained by heading rate: {100*frac:.0f}%")
    print(f"  (high = magnetization-driven; low = real independent gesture)")
    print(f"\n  forearm heading bin: " + " ".join(f"{-180+30*b:+4d}" for b in range(NB)))
    print(f"  delta'(psi) [slope]: " + " ".join(f"{kprime[b]:+4.1f}" if valid[b] else "   ." for b in range(NB)))
    print(f"  delta(psi)  [deg]  : " + " ".join(f"{np.degrees(delta[b]):+4.0f}" if valid[b] else "   ." for b in range(NB)))
    print(f"\nforearm yaw-speed: median {np.degrees(np.median(fa_speed)):.0f} deg/s, "
          f"95th {np.degrees(np.percentile(fa_speed,95)):.0f}, max {np.degrees(fa_speed.max()):.0f} "
          f"(human reach ~ 500-900 deg/s)")
    print(f"wrist-path-through-torso frames: {100*(swept_pen>2).mean():.1f}% "
          f"(swept impossibility -- forearm transiting the body)")


if __name__ == "__main__":
    main()
