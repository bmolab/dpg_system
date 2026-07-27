"""Scoped Stage-2: correct ONLY the left-forearm (LeftElbow) sensor's magnetometer yaw, using
VPoser implausibility as the objective and a HARD elbow-hinge barrier as the floor.

Session detection (batch_pose_plausibility) localized the dominant distortion to the left
forearm. This searches a smooth per-heading yaw curve delta(psi) for that one sensor that
minimizes VPoser pose-implausibility ||z||^2, subject to:
  * HARD hinge: the corrected forearm cannot leave the elbow's flexion plane (no medial swing
    into the chest) -- the constraint whose absence let the earlier free-yaw fit fold the elbow;
  * minimality + smoothness (small, low-order curve) so one sensor can't fake arbitrary poses.
psi is the FOREARM sensor's own heading, which sweeps widely as the arm gestures even within a
forward-facing take, so the curve is well-sampled from a single take. Correcting one sensor and
watching whole-body NLL drop is also the causal test that the left forearm was the driver.

Applies the fit to the take (forearm world yawed by -delta; LeftElbow + LeftWrist locals
recomputed) and writes *_fafix.npz for rendering. Originals (*beta.npz) are never modified.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from diag_pose_plausibility import shadow_to_body_pose, EXPR_DIR
from diag_magnetometer_deviation import (load_skeleton, fk_world, qrot_np, qmul_np,
                                         qmul_t, qrot_t, LSH, LEL, LWR)
from dpg_system.vae_nodes import VPoser, load_model

# SMPL body-pose row indices (BODY_JOINTS = SMPL 1..21, so row = smpl_idx - 1)
ROW_LELBOW = 18 - 1
ROW_LWRIST = 20 - 1


def discover_hinge_normal(Gmeas, prox, dist, bone_axis):
    bone = bone_axis[prox]
    cands = [e for e in np.eye(3) if abs(float(e @ bone)) < 0.9]
    dd = qrot_np(Gmeas[:, dist], bone_axis[dist])
    return min(cands, key=lambda c: np.abs((dd * qrot_np(Gmeas[:, prox], c)).sum(-1)).mean())


def qconj_t(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def quat_to_rotvec_t(q):
    """w-first quat -> axis-angle (shortest, w>=0), matching scipy as_rotvec convention."""
    q = q * torch.sign(q[..., :1] + 1e-12)                  # canonicalize w >= 0
    w = q[..., 0].clamp(-1, 1)
    v = q[..., 1:]
    vn = v.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(vn[..., 0], w)
    scale = torch.where(vn[..., 0] > 1e-8, angle / vn[..., 0].clamp_min(1e-8),
                        torch.full_like(angle, 2.0))         # small-angle -> ~2
    return v * scale.unsqueeze(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("-o", "--out")
    ap.add_argument("--expr-dir", default=EXPR_DIR)
    ap.add_argument("--stride", type=int, default=0, help="fit frame stride (0 = auto ~3000)")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--hinge-margin", type=float, default=12.0, help="allowed elbow out-of-plane (deg)")
    ap.add_argument("--w-hinge", type=float, default=300.0)
    ap.add_argument("--l2", type=float, default=2.0, help="smoothness (L2 on harmonics)")
    ap.add_argument("--l1", type=float, default=0.5, help="minimality (L1 on all params)")
    ap.add_argument("--no-apply", action="store_true")
    args = ap.parse_args()

    parent, order, bone_axis = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    d = np.load(args.infile, allow_pickle=True)
    quats = d['quats'].astype(np.float64)
    T = quats.shape[0]
    Gw = fk_world(quats, parent, order)
    hinge_n = discover_hinge_normal(Gw, LSH, LEL, bone_axis)

    stride = args.stride or max(1, T // 3000)
    idx = np.arange(0, T, stride)
    body = shadow_to_body_pose(quats)[idx].reshape(len(idx), 21, 3).astype(np.float32)
    psi = np.arctan2(*(lambda x: (x[..., 2], x[..., 0]))(qrot_np(Gw[idx, LEL], np.array([1.0, 0, 0]))))

    # measured world quats for the left arm chain (fit subset)
    up_w = torch.tensor(Gw[idx, LSH]); fa_w = torch.tensor(Gw[idx, LEL]); ha_w = torch.tensor(Gw[idx, LWR])
    psit = torch.tensor(psi)
    body_t = torch.tensor(body, dtype=torch.float64)
    fa_axis = torch.tensor(bone_axis[LEL]); hinge_t = torch.tensor(hinge_n)

    model = load_model(args.expr_dir, model_code=VPoser, remove_words_in_model_weights='vp_model.',
                       disable_grad=True, comp_device='cpu')[0].to('cpu').eval()

    P = torch.zeros(5, dtype=torch.float64, requires_grad=True)   # A,B,C,D,E
    margin = math.sin(math.radians(args.hinge_margin))

    def corrected_chain(p):
        delta = (p[0] + p[1] * torch.sin(psit) + p[2] * torch.cos(psit)
                 + p[3] * torch.sin(2 * psit) + p[4] * torch.cos(2 * psit))
        half = -delta / 2.0
        z = torch.zeros_like(half)
        qy = torch.stack([torch.cos(half), z, torch.sin(half), z], -1)     # yaw about world Y
        fc = qmul_t(qy, fa_w)                                              # corrected forearm world
        elbow_loc = qmul_t(qconj_t(up_w), fc)
        wrist_loc = qmul_t(qconj_t(fc), ha_w)
        return fc, elbow_loc, wrist_loc

    def nll_and_hinge(p):
        fc, elbow_loc, wrist_loc = corrected_chain(p)
        b = body_t.clone()
        b[:, ROW_LELBOW] = quat_to_rotvec_t(elbow_loc).to(b.dtype)
        b[:, ROW_LWRIST] = quat_to_rotvec_t(wrist_loc).to(b.dtype)
        nll = (model.encode(b.reshape(-1, 63).float()).mean ** 2).sum(-1).double()
        oop = (qrot_t(fc, fa_axis) * qrot_t(up_w, hinge_t)).sum(-1).abs()
        hinge = (torch.relu(oop - margin) ** 2)
        return nll, hinge

    with torch.no_grad():
        nll0, hinge0 = nll_and_hinge(torch.zeros(5, dtype=torch.float64))
    opt = torch.optim.Adam([P], lr=args.lr)
    for it in range(args.iters):
        opt.zero_grad()
        nll, hinge = nll_and_hinge(P)
        loss = nll.mean() + args.w_hinge * hinge.mean() + args.l2 * (P[1:] ** 2).sum() + args.l1 * P.abs().sum()
        loss.backward(); opt.step()
        if it % 100 == 0 or it == args.iters - 1:
            print(f"  it{it:4d}  nll={float(nll.mean()):7.1f}  hinge_viol={float((hinge>0).float().mean()):.3f}")

    with torch.no_grad():
        nll1, hinge1 = nll_and_hinge(P)
    Pn = P.detach().numpy()
    print(f"\nVPoser ||z||^2 (left forearm corrected):  median {np.median(nll0.numpy()):.0f}->"
          f"{np.median(nll1.numpy()):.0f}   mean {nll0.mean():.0f}->{nll1.mean():.0f}")
    print(f"frames implausible (||z||^2>96): {100*(nll0.numpy()>96).mean():.0f}% -> {100*(nll1.numpy()>96).mean():.0f}%")
    print(f"elbow out-of-plane viol frames: {100*float((hinge0>0).float().mean()):.0f}% -> {100*float((hinge1>0).float().mean()):.0f}%")
    print(f"delta curve (deg):  const {math.degrees(Pn[0]):+.1f}  "
          f"1stH {math.degrees(math.hypot(Pn[1],Pn[2])):.1f}  2ndH {math.degrees(math.hypot(Pn[3],Pn[4])):.1f}")

    if args.no_apply:
        return
    # apply over the FULL take: yaw the forearm world by -delta(psi), recompute LeftElbow+LeftWrist locals
    psif = np.arctan2(*(lambda x: (x[..., 2], x[..., 0]))(qrot_np(Gw[:, LEL], np.array([1.0, 0, 0]))))
    delta = (Pn[0] + Pn[1]*np.sin(psif) + Pn[2]*np.cos(psif) + Pn[3]*np.sin(2*psif) + Pn[4]*np.cos(2*psif))
    half = -delta / 2.0
    qy = np.stack([np.cos(half), np.zeros_like(half), np.sin(half), np.zeros_like(half)], -1)
    fc = qmul_np(qy, Gw[:, LEL])
    Qc = quats.copy()
    Qc[:, LEL] = qmul_np(Gw[:, LSH] * np.array([1, -1, -1, -1.0]), fc)         # LeftElbow local
    Qc[:, LWR] = qmul_np(fc * np.array([1, -1, -1, -1.0]), Gw[:, LWR])         # LeftWrist local
    out = args.out or args.infile.replace(".npz", "_fafix.npz")
    save = {k: d[k] for k in d.files}; save['quats'] = Qc
    np.savez(out, **save)
    print(f"\nwrote corrected take -> {out}")


if __name__ == "__main__":
    main()
