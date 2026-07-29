"""Geometry-free cross-check of the magnetization curve via the heading-locked motion signal.

The twist fit (fit_session_deviation.py) recovers the SHAPE and PHASE of delta(psi) but in TWIST
degrees, which under-state the world-yaw deviation (world-vertical yaw projects onto bone-axis twist
~sin(elevation), and we fit the low-tilt band). This recovers delta in TRUE WORLD-YAW degrees from
an independent signal that needs no mesh and no bone geometry.

Principle: during a body turn with the limb held fixed relative to the body, the limb sensor's
TRUE world yaw-rate equals the body's. The fusion's magnetometer error makes the MEASURED limb yaw
read psi_meas = psi_true + delta(psi), so d(psi_meas_limb)/dt = psi_dot_body * (1 + delta'(psi)).
Therefore the body-RELATIVE limb yaw-rate is

    dr = d(psi_limb - psi_body)/dt = psi_dot_body * delta'(psi_limb).

Genuine limb-vs-body motion is uncorrelated with psi_dot_body and averages out across many turning
frames, so a regression of dr on psi_dot_body (with a harmonic basis in psi) isolates delta'(psi):

    dr = psi_dot * ( b cos psi - c sin psi + 2 d cos 2psi - 2 e sin 2psi )

whose integral is delta(psi) = b sin psi + c cos psi + d sin 2psi + e cos 2psi -- the SAME basis and
gauge (mean-zero) as the twist fit, now in world-yaw degrees. We pool turning frames over all takes
(magnetization is session-shared), gate to the well-conditioned tilt band, and bootstrap over takes.
Compare 1st-harmonic PEAK HEADING to the twist fit (validation) and AMPLITUDE (magnitude calibration).
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np

from diag_magnetometer_deviation import load_skeleton, fk_world, qrot_np, PELV, LEL, REL, LSH, RSH

XAX = np.array([1., 0., 0.])
TILT_MAX = 45.0


def sensor_yaw(Gj): v = qrot_np(Gj, XAX); return np.arctan2(v[:, 2], v[:, 0])     # radians
def elev(Gj, ax): d = qrot_np(Gj, ax); return np.degrees(np.arcsin(np.clip(d[:, 1], -1, 1)))
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def design(psi, psidot):
    """rows of [psidot*cos, -psidot*sin, 2psidot*cos2, -2psidot*sin2] -> coeffs b,c,d,e (deg)."""
    return np.stack([psidot * np.cos(psi), -psidot * np.sin(psi),
                     2 * psidot * np.cos(2 * psi), -2 * psidot * np.sin(2 * psi)], 1)


def gather(files, parent, order, bax, sj, turn_thr):
    X, y, tk = [], [], []
    for ti, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        q = d['quats'].astype(np.float64)
        G = fk_world(q, parent, order)
        pel = np.unwrap(sensor_yaw(G[:, PELV]))
        limb = np.unwrap(sensor_yaw(G[:, sj]))
        psi = sensor_yaw(G[:, sj])                                  # heading at the frame (rad, wrapped)
        psidot = wrap(np.roll(pel, -1) - pel)                       # body yaw-rate, rad/frame
        rel = limb - pel
        dr = wrap(np.roll(rel, -1) - rel)                           # body-relative limb yaw-rate
        keep = (np.abs(psidot) > np.radians(turn_thr)) & (np.abs(elev(G[:, sj], bax[sj])) <= TILT_MAX)
        keep[-1] = False
        Xf = design(psi[keep], np.degrees(psidot[keep]))            # psidot in deg so coeffs come out in deg
        X.append(Xf); y.append(np.degrees(dr[keep])); tk.append(np.full(keep.sum(), ti))
    return np.vstack(X), np.concatenate(y), np.concatenate(tk)


def solve(X, y):
    # dr_deg = psidot_deg * delta'(psi); with psi in radians the LS coeffs come out in RADIANS,
    # so convert to degrees. (delta(psi)=b sin psi + c cos psi + d sin2psi + e cos2psi, world-yaw.)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.degrees(beta)                                         # b,c,d,e in world-yaw deg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--turn-rate", type=float, default=0.3, help="min |body yaw-rate| deg/frame")
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*beta.npz")))
    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    rng = np.random.default_rng(0)
    NB = 12
    centers_deg = np.array([-180 + 30 * b for b in range(NB)])
    centers = np.radians(centers_deg)

    # twist-fit reference (from fit_session_deviation.py) for side-by-side phase/amp comparison
    TWIST_REF = {'L-elbow (forearm)': (19.1, +123, 6.4), 'R-elbow (forearm)': (9.2, +90, 0.7)}
    SENSORS = {'L-elbow (forearm)': LEL, 'R-elbow (forearm)': REL,
               'L-shldr (upper-arm)': LSH, 'R-shldr (upper-arm)': RSH}

    for name, sj in SENSORS.items():
        X, y, tk = gather(files, parent, order, bax, sj, args.turn_rate)
        b, c, d, e = solve(X, y)
        H1 = np.hypot(b, c); ph1 = np.degrees(np.arctan2(b, c))
        H2 = np.hypot(d, e); ph2 = np.degrees(np.arctan2(d, e)) / 2
        H1s, ph1s, H2s = [], [], []
        ntk = len(files)
        for _ in range(args.boot):
            sel = rng.integers(0, ntk, ntk)
            mask = np.isin(tk, sel)            # note: approx (unique takes); fine for CI
            bb = solve(X[mask], y[mask])
            H1s.append(np.hypot(bb[0], bb[1])); ph1s.append(np.degrees(np.arctan2(bb[0], bb[1])))
            H2s.append(np.hypot(bb[2], bb[3]))
        H1lo, H1hi = np.percentile(H1s, [16, 84])
        phlo, phhi = np.percentile(ph1s, [16, 84])
        H2lo, H2hi = np.percentile(H2s, [16, 84])
        nfr = len(y)
        print(f"\n=== {name} ===   ({nfr} turning frames)")
        print(f"  world-yaw delta:  b={b:+.2f} c={c:+.2f} d={d:+.2f} e={e:+.2f}  (deg)")
        print(f"  1st (hard-iron):  amp {H1:5.2f} deg [{H1lo:.1f},{H1hi:.1f}]  peak heading {ph1:+.0f} deg [{phlo:+.0f},{phhi:+.0f}]")
        print(f"  2nd (soft-iron):  amp {H2:5.2f} deg [{H2lo:.1f},{H2hi:.1f}]")
        if name in TWIST_REF:
            ta, tp, t2 = TWIST_REF[name]
            print(f"  twist-fit ref :   amp {ta:.1f} (twist-deg)  peak {tp:+d}   2nd {t2:.1f}")
            print(f"  -> phase match: {abs((ph1-tp+180)%360-180):.0f} deg apart;  world-yaw/twist amp ratio {H1/ta:.2f}")
        dcurve = b*np.sin(centers) + c*np.cos(centers) + d*np.sin(2*centers) + e*np.cos(2*centers)
        print(f"  {'heading':9s}" + " ".join(f"{cd:+5d}" for cd in centers_deg))
        print(f"  {'delta(deg)':9s}" + " ".join(f"{v:+5.1f}" for v in dcurve))

    print("\nIf headlock peak-heading matches the twist fit, the curve is cross-validated by an")
    print("independent signal; the world-yaw amplitude (and amp ratio>1) gives the true magnitude.")


if __name__ == "__main__":
    main()
