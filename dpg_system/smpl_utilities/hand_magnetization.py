"""Is the HAND sensor magnetized? Direct, no-body-turn test from the world orientation.

Magnetization is a yaw error that's a fixed function of the sensor's OWN heading, present in every
frame. So bin a relative-orientation signal by the HAND sensor's own heading, pool across the 11
takes (choreography averages out; the session-consistent heading-dependent part = magnetization),
fit session-shared harmonics + a per-take constant -- exactly the fit used on the forearm, no turns.

Reference = the forearm: the wrist has no axial DOF, so hand-vs-forearm axial twist should be ~fixed;
its heading-dependent variation is (delta_hand - delta_forearm) projected. We already know
delta_forearm, so a clear session-consistent hand curve = the hand sensor carries its own error.
Forearms (vs upper-arm) shown alongside as the validated reference scale. Bootstrap-over-takes CIs.
Caveat: if the forearm IMU is proximally mounted, some hand-vs-forearm twist is real pronation; the
session-CONSISTENT heading-locked part is still the magnetization signature.
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np

from diag_magnetometer_deviation import load_skeleton, REL, RWR, LEL, LWR, RSH, LSH
from fit_session_deviation import collect, fit, curve, NB


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--nframes", type=int, default=4000)
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*beta.npz")))
    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    centers = np.radians(np.array([-180 + 30 * b for b in range(NB)], float))
    rng = np.random.default_rng(0)

    # (parent/reference joint, sensor joint) ; sensor heading & tilt taken from sensor joint
    SENSORS = {'R-hand (vs forearm)': (REL, RWR), 'L-hand (vs forearm)': (LEL, LWR),
               'R-forearm (vs upper-arm, ref)': (RSH, REL), 'L-forearm (vs upper-arm, ref)': (LSH, LEL)}
    for name, (pj, sj) in SENSORS.items():
        takes = collect(files, parent, order, bax, (pj, sj, None), args.nframes)
        A, BCDE, r2, rms = fit(takes, centers)
        H1 = np.hypot(BCDE[0], BCDE[1]); ph1 = np.degrees(np.arctan2(BCDE[0], BCDE[1]))
        H2 = np.hypot(BCDE[2], BCDE[3])
        H1s = []
        for _ in range(args.boot):
            samp = [takes[i] for i in rng.integers(0, len(takes), len(takes))]
            _, bb, _, _ = fit(samp, centers); H1s.append(np.hypot(bb[0], bb[1]))
        lo, hi = np.percentile(H1s, [16, 84])
        # pooled data (de-mean each take by its fitted constant) vs fitted curve
        pooled = np.full(NB, np.nan); cnt = np.zeros(NB)
        for t, (ym, nm) in enumerate(takes):
            for h in range(NB):
                if not np.isnan(ym[h]):
                    v = (ym[h] - A[t]) * nm[h]
                    pooled[h] = v if np.isnan(pooled[h]) else pooled[h] + v
                    cnt[h] += nm[h]
        pooled = np.where(cnt > 0, pooled / np.maximum(cnt, 1), np.nan)
        fitc = curve(BCDE, centers)
        print(f"\n=== {name} ===")
        print(f"  1st-harm {H1:5.1f} deg [{lo:.1f},{hi:.1f}]  peak heading {ph1:+.0f}   2nd {H2:.1f}   "
              f"explains {100*r2:.0f}% beyond per-take constants; RMS {rms:.1f}")
        print(f"  {'heading':8s}" + " ".join(f"{-180+30*b:+5d}" for b in range(NB)))
        print(f"  {'data':8s}" + " ".join("   . " if np.isnan(pooled[b]) else f"{pooled[b]:+5.1f}" for b in range(NB)))
        print(f"  {'fit':8s}" + " ".join(f"{fitc[b]:+5.1f}" for b in range(NB)))
    print("\nHand 1st-harm comparable to forearm + tight CI + decent %% explained => hand IS magnetized.")
    print("Hand ~flat / wide CI / low %% => wrist twist is calibration offset or pronation, not magnetometer.")


if __name__ == "__main__":
    main()
