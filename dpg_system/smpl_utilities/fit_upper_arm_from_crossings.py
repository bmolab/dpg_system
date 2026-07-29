"""Fit a HEADING-DEPENDENT upper-arm correction from auto-detected symmetry crossings.

The constant-C fit fails because the upper-arm asymmetry is heading-dependent (a single C that fixes
the forward-facing pose breaks the backward ones). detect_sym_crossings.py gives ~100+ genuinely
symmetric poses spanning many arm headings; here we use them to fit:

  constant per-arm correction C (the mean asymmetric offset, incl. the ~6 deg elevation tilt), then
  a per-arm heading-dependent yaw delta(psi) = b sin psi + c cos psi (the hard-iron deviation curve)

both by minimizing the L/R upper-arm mirror residual over the crossings. With the crossings spanning
headings, delta(psi) is well-constrained (vs the 106 deg blow-up when only ~2 sym headings existed).
The gauge mode symmetry can't see (both arms equal) is pinned to identity by the ridge in
fit_corrections. Writes a fit npz (cl, cr, hy_l, hy_r) the shadow_arm_correct node / apply path loads.
This corrects upper-arm POINTING (yaw/swing) -- the visible 'arm raised / not thrust back' defect;
the bone-axis roll-bleed is a separate correction.
"""
import argparse
import os
from pathlib import Path

import numpy as np

from diag_magnetometer_deviation import qrot_np  # noqa  (parity import; uses cuo's own qrot below)
import correct_upper_arm_offset as cuo
from correct_upper_arm_offset import (forward_kinematics, fit_corrections, fit_heading_yaw,
                                      _frame_dirs, LSH, RSH, rotvec_to_q, qmul, qrot, norm_rows)
from scipy.spatial.transform import Rotation as Rot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take")
    ap.add_argument("--crossings", help="_symcross.npz (default: derived from take name)")
    ap.add_argument("--apply", action="store_true", help="also write a corrected *_armfix.npz")
    args = ap.parse_args()

    d = np.load(args.take, allow_pickle=True)
    Q = d['quats'].astype(np.float64)
    P = d['positions'].astype(np.float64) if 'positions' in d.files else None
    G = forward_kinematics(Q)
    T = Q.shape[0]

    cpath = args.crossings or args.take.replace('.npz', '_symcross.npz')
    frames = np.load(cpath)['frames'].astype(int)
    print(f"loaded {len(frames)} symmetric crossings from {os.path.basename(cpath)}")

    # build a symmetry anchor at every crossing frame (per-frame dirs over the whole take, indexed)
    lat, nL, nR, mL, mR = _frame_dirs(G, P, 0, T)
    M = np.eye(3)[None] - 2 * lat[:, :, None] * lat[:, None, :]
    pre = [(M[f], nL[f], nR[f], mL[f], mR[f]) for f in frames]

    def mirror_asym(cl_rv, cr_rv):
        CL = Rot.from_rotvec(cl_rv).as_matrix(); CR = Rot.from_rotvec(cr_rv).as_matrix()
        a = []
        for (Mi, nL0, nR0, mL0, mR0) in pre:
            a.append(np.degrees(np.arccos(np.clip(np.dot(Mi @ (CL @ nL0), CR @ nR0), -1, 1))))
        return np.array(a)

    before = mirror_asym(np.zeros(3), np.zeros(3))
    cl, cr = fit_corrections(pre)                      # constant per-arm C (gauge pinned by ridge)
    after_C = mirror_asym(cl, cr)
    hy_l, hy_r = fit_heading_yaw(Q, P, cl, cr, frames=frames)   # heading-dependent yaw on top

    # residual after C + delta(psi): apply head_yaw to the crossing-frame upper-arm globals
    gL = qmul(np.broadcast_to(rotvec_to_q(cl), G[frames][:, LSH].shape), G[frames][:, LSH])
    gR = qmul(np.broadcast_to(rotvec_to_q(cr), G[frames][:, RSH].shape), G[frames][:, RSH])
    gL = cuo.head_yaw(gL, 1.0, hy_l); gR = cuo.head_yaw(gR, -1.0, hy_r)
    nLf = norm_rows(qrot(gL, np.array([1.0, 0, 0]))); nRf = norm_rows(qrot(gR, np.array([-1.0, 0, 0])))
    mnLf = np.einsum('nij,nj->ni', M[frames], nLf)
    after_full = np.degrees(np.arccos(np.clip((mnLf * nRf).sum(1), -1, 1)))

    print(f"\nupper-arm L/R mirror asymmetry over {len(frames)} crossings (deg):")
    print(f"  before fit           : mean {before.mean():5.1f}  median {np.median(before):5.1f}  p90 {np.percentile(before,90):5.1f}")
    print(f"  + constant C         : mean {after_C.mean():5.1f}  median {np.median(after_C):5.1f}  p90 {np.percentile(after_C,90):5.1f}")
    print(f"  + heading-yaw delta  : mean {after_full.mean():5.1f}  median {np.median(after_full):5.1f}  p90 {np.percentile(after_full,90):5.1f}")
    print(f"\nfit:  C_L rotvec {np.round(np.degrees(cl),1)} deg   C_R {np.round(np.degrees(cr),1)} deg")
    print(f"      heading-yaw amp |L|={np.degrees(np.hypot(*hy_l)):.0f} @ {np.degrees(np.arctan2(*hy_l)):+.0f}"
          f"   |R|={np.degrees(np.hypot(*hy_r)):.0f} @ {np.degrees(np.arctan2(*hy_r)):+.0f}  deg")

    fitpath = args.take.replace('.npz', '_armfit.npz')
    np.savez(fitpath, cl=cl, cr=cr, hy_l=hy_l, hy_r=hy_r)
    print(f"\nwrote fit -> {os.path.basename(fitpath)}  (load in shadow_arm_correct node)")

    if args.apply:
        Qc = cuo.apply_correction(Q, G, P, cl, cr, hy_l=hy_l, hy_r=hy_r)
        save = {k: d[k] for k in d.files}; save['quats'] = Qc.astype(np.float32)
        outp = args.take.replace('.npz', '_armfix.npz')
        np.savez(outp, **save); print(f"wrote corrected take -> {os.path.basename(outp)}")


if __name__ == "__main__":
    main()
