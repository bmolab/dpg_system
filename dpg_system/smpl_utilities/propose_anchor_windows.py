"""Propose candidate anchor windows (symmetric poses + relaxed arms-down) for a Shadow take,
to feed the magnetometer-deviation fit's --sym / --relax. Held (low-motion) poses are where
those anchors live; this finds them, reports each window's heading + current L/R asymmetry +
arm elevation, and proposes a heading-spread shortlist. YOU confirm against the video --
this only points at candidates.
"""
import argparse
import math
from pathlib import Path

import numpy as np

from diag_magnetometer_deviation import (load_skeleton, fk_world, qrot_np,
                                         LSH, LEL, RSH, REL, PELV)

UP = np.array([0.0, 1.0, 0.0])
DOWN = np.array([0.0, -1.0, 0.0])


def angle_between_q(a, b):
    """Geodesic angle (rad) between consecutive world quats, per (T-1, J)."""
    dot = np.abs((a * b).sum(-1)).clip(-1, 1)
    return 2 * np.arccos(dot)


def runs(mask, min_len):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("infile")
    ap.add_argument("--min-hold", type=float, default=0.15, help="min held-pose duration (s)")
    ap.add_argument("--speed-pct", type=float, default=20, help="held = below this speed percentile")
    ap.add_argument("--n-sym", type=int, default=12, help="symmetric candidates to print")
    ap.add_argument("--head-sep", type=float, default=18, help="min heading separation in shortlist (deg)")
    ap.add_argument("--relax-elev", type=float, default=40, help="arms-down if both arms within this of DOWN (deg)")
    args = ap.parse_args()

    def_path = Path(__file__).resolve().parent.parent / 'definition.xml'
    parent, order, bone_axis = load_skeleton(def_path)

    d = np.load(args.infile, allow_pickle=True)
    Q = d['quats'].astype(np.float64)
    fps = float(d['mocap_framerate'])
    T = Q.shape[0]
    G = fk_world(Q, parent, order)

    # per-frame motion speed (deg/frame), mean over the moving joints (arms+legs+spine)
    movers = [LSH, LEL, RSH, REL, 14, 12, 8, 28, 26, 22, 1, 17, 32]
    spd = np.degrees(angle_between_q(G[:-1, movers], G[1:, movers])).mean(-1)
    spd = np.concatenate([spd, spd[-1:]])
    thr = np.percentile(spd, args.speed_pct)
    held = spd < thr
    min_len = max(5, int(args.min_hold * fps))
    windows = runs(held, min_len)

    # pelvis heading (deg)
    px = qrot_np(G[:, PELV], np.array([1.0, 0.0, 0.0]))
    heading = np.degrees(np.arctan2(px[:, 2], px[:, 0]))

    # mirror plane from pelvis lateral (horizontal)
    lat = qrot_np(G[:, PELV], np.array([1.0, 0.0, 0.0]))
    lat = lat - UP * (lat @ UP)[:, None]
    lat = lat / (np.linalg.norm(lat, axis=-1, keepdims=True) + 1e-9)
    M = np.eye(3)[None] - 2.0 * lat[:, :, None] * lat[:, None, :]

    dUA_L = qrot_np(G[:, LSH], bone_axis[LSH]); dUA_R = qrot_np(G[:, RSH], bone_axis[RSH])
    dFA_L = qrot_np(G[:, LEL], bone_axis[LEL]); dFA_R = qrot_np(G[:, REL], bone_axis[REL])

    def chord(a, b):
        return np.linalg.norm(a - b, axis=-1)

    asym_ua = chord(np.einsum('nij,nj->ni', M, dUA_L), dUA_R)
    asym_fa = chord(np.einsum('nij,nj->ni', M, dFA_L), dFA_R)
    asym = asym_ua + asym_fa                                    # 0 = perfectly mirror-symmetric

    elevL = np.degrees(np.arccos((dUA_L @ DOWN).clip(-1, 1)))   # 0 = arm hangs straight down
    elevR = np.degrees(np.arccos((dUA_R @ DOWN).clip(-1, 1)))

    rows = []
    for a, b in windows:
        sl = slice(a, b)
        rows.append(dict(a=a, b=b, dur=(b - a) / fps,
                         head=float(np.mean(heading[sl])),
                         asym=float(np.mean(asym[sl])),
                         elL=float(np.mean(elevL[sl])), elR=float(np.mean(elevR[sl]))))

    print(f"{Path(args.infile).name}: {T} frames @ {fps:g}Hz, held-speed thr={thr:.2f} deg/frame, "
          f"{len(windows)} held windows >= {min_len} frames\n")

    # --- symmetric shortlist: lowest asymmetry, spread across headings ---
    sym_sorted = sorted(rows, key=lambda r: r['asym'])
    picked, heads = [], []
    for r in sym_sorted:
        if all(abs((r['head'] - h + 180) % 360 - 180) > args.head_sep for h in heads):
            picked.append(r); heads.append(r['head'])
        if len(picked) >= args.n_sym:
            break
    print("SYMMETRIC candidates (low L/R asymmetry, heading-spread).  asym: 0=mirror-perfect")
    print(f"  {'frames':>15} {'dur_s':>6} {'head':>6} {'asym':>6}   --sym")
    syms = []
    for r in sorted(picked, key=lambda r: r['a']):
        print(f"  {r['a']:7d}:{r['b']:<7d} {r['dur']:6.2f} {r['head']:6.0f} {r['asym']:6.2f}")
        syms.append(f"{r['a']}:{r['b']}")
    print(f"\n  --sym {','.join(syms)}\n")

    # --- relaxed arms-down candidates ---
    relax = [r for r in rows if r['elL'] < args.relax_elev and r['elR'] < args.relax_elev]
    relax.sort(key=lambda r: -r['dur'])
    print(f"RELAXED arms-down candidates (both arms within {args.relax_elev:.0f} deg of vertical)")
    if not relax:
        print("  none found -- this take may have no arms-down rest; --relax can be omitted")
    else:
        print(f"  {'frames':>15} {'dur_s':>6} {'head':>6} {'elL':>5} {'elR':>5} {'asym':>6}")
        for r in relax[:6]:
            print(f"  {r['a']:7d}:{r['b']:<7d} {r['dur']:6.2f} {r['head']:6.0f} "
                  f"{r['elL']:5.0f} {r['elR']:5.0f} {r['asym']:6.2f}")
        best = relax[0]
        print(f"\n  --relax {best['a']}:{best['b']}")
    print("\nConfirm these against the video before using them as anchors.")


if __name__ == "__main__":
    main()
