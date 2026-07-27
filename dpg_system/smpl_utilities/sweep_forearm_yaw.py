"""Sweep candidate forearm magnetization-yaw per heading bin and look for clear advantages.

Instead of fitting a parametric delta(psi) anchored only forward/back (which extrapolates badly
to the side facings), sweep a constant yaw correction for the LEFT FOREARM sensor and, INSIDE
EACH heading bin, find the yaw that minimizes the per-frame IMPOSSIBILITY signal -- elbow axial
twist (the elbow has ~no pronation DOF) + elbow out-of-plane. That signal exists at every heading,
so it reconstructs delta(psi) across the whole circle, including the side facings that have no
symmetry anchor. A bin with a clear minimum and a real twist reduction = a trustworthy correction
there; a flat landscape or no advantage = under-determined / no magnetization at that heading.

Forearm-only (upper arm held at measured) to isolate the forearm sensor. Diagnostic; writes nothing.
"""
import argparse
import math
from pathlib import Path

import numpy as np

from diag_magnetometer_deviation import load_skeleton, fk_world, qrot_np, qmul_np, qconj_np if False else None  # noqa
from diag_magnetometer_deviation import load_skeleton, fk_world, qrot_np, qmul_np, PELV, LSH, LEL


def qconj_np(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def twist_angle(qloc, axis):
    proj = (qloc[..., 1:] * axis).sum(-1)
    return np.degrees(2.0 * np.arctan2(proj, qloc[..., 0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("--range", type=float, default=50.0, help="sweep +/- this many deg")
    ap.add_argument("--step", type=float, default=2.5)
    ap.add_argument("--sym", default="7880:8500,20660:20953,10560:10692",
                    help="symmetric anchor ranges, for the cross-check column")
    args = ap.parse_args()

    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    d = np.load(args.infile, allow_pickle=True)
    q = d['quats'].astype(np.float64)
    T = q.shape[0]
    Gw = fk_world(q, parent, order)

    # elbow hinge axis (upper-arm local axis the forearm stays perpendicular to)
    bone = bax[LSH]
    cands = [e for e in np.eye(3) if abs(float(e @ bone)) < 0.9]
    fdir0 = qrot_np(Gw[:, LEL], bax[LEL])
    hinge_n = min(cands, key=lambda c: np.abs((fdir0 * qrot_np(Gw[:, LSH], c)).sum(-1)).mean())

    upL = Gw[:, LSH]; faL = Gw[:, LEL]
    fa_head = np.degrees(np.arctan2(*(lambda x: (x[:, 2], x[:, 0]))(qrot_np(faL, np.array([1., 0, 0])))))
    NB = 12
    hb = ((fa_head + 180) // 30).astype(int) % NB

    yaws = np.arange(-args.range, args.range + args.step / 2, args.step)
    # impossibility per (yaw, frame): |elbow twist| + out-of-plane angle
    hinge_w = qrot_np(upL, hinge_n)
    cost = np.zeros((len(yaws), T))
    for k, y in enumerate(yaws):
        h = math.radians(-y) / 2.0
        qy = np.array([math.cos(h), 0.0, math.sin(h), 0.0])
        fc = qmul_np(np.broadcast_to(qy, faL.shape), faL)
        etw = np.abs(twist_angle(qmul_np(qconj_np(upL), fc), bax[LEL]))
        oop = np.degrees(np.arcsin(np.clip(np.abs((qrot_np(fc, bax[LEL]) * hinge_w).sum(-1)), 0, 1)))
        cost[k] = etw + oop

    print(f"{Path(args.infile).name}: forearm-yaw sweep +/-{args.range:g} deg, per heading bin")
    print(f"  impossibility = |elbow twist| + |elbow out-of-plane|, in degrees\n")
    print(f"  {'headbin':>8} {'n':>6} {'best_yaw':>8} {'cost@0':>7} {'cost@best':>9} {'gain':>6}")
    best_yaw = np.full(NB, np.nan)
    zero_k = int(np.argmin(np.abs(yaws)))
    for b in range(NB):
        sel = hb == b
        if sel.sum() < 50:
            print(f"  {-180+30*b:+8d} {sel.sum():6d}   (too few frames)")
            continue
        mc = cost[:, sel].mean(1)
        k = int(np.argmin(mc))
        best_yaw[b] = yaws[k]
        gain = mc[zero_k] - mc[k]
        flag = "  <-- clear" if gain > 5 and (mc[zero_k] - mc.min()) > 5 else ""
        print(f"  {-180+30*b:+8d} {sel.sum():6d} {yaws[k]:+8.1f} {mc[zero_k]:7.1f} {mc[k]:9.1f} {gain:6.1f}{flag}")

    print(f"\nreconstructed delta(psi) from twist-minimization (deg), per heading bin:")
    print("  " + " ".join(f"{-180+30*b:+4d}" for b in range(NB)))
    print("  " + " ".join(f"{best_yaw[b]:+4.0f}" if not np.isnan(best_yaw[b]) else "   ." for b in range(NB)))
    print("\nA smooth best_yaw curve with positive 'gain' across bins = a real, anchor-free")
    print("magnetization estimate (incl. side facings). Noisy/zero-gain bins = under-determined.")


if __name__ == "__main__":
    main()
