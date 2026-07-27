"""Sweep a sensor-frame bone-axis ROLL correction and score it by the MESH distortion it produces.

A shoulder roll error doesn't reach the independent forearm sensor, so it surfaces as spurious
elbow twist and pinches the forearm mesh band; correcting the shoulder roll therefore fixes the
DOWNSTREAM band (ripple). This sweeps a constant roll about a chosen sensor's bone axis (sensor-
frame, postmultiplied), rebuilds the pose, deforms the SMPL mesh, and reports the candy-wrapper
collapse of the affected bands vs roll angle -- looking for a clear minimum (the 'obvious
advantage'), now judged by the faithful mesh metric rather than an abstract twist angle.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import smplx

from diag_magnetometer_deviation import load_skeleton, fk_world, qmul_np, RSH, REL, RWR, RBLADE, LSH, LEL, LWR, LBLADE
from diag_pose_plausibility import shadow_to_body_pose
from mesh_joint_distortion import BANDS, band_verts, perp_spread, MODEL_PATH


def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("--arm", choices=["L", "R"], default="R")
    ap.add_argument("--range", type=float, default=45.0)
    ap.add_argument("--step", type=float, default=5.0)
    ap.add_argument("--nframes", type=int, default=600)
    args = ap.parse_args()

    SH, EL, WR, BL = (LSH, LEL, LWR, LBLADE) if args.arm == "L" else (RSH, REL, RWR, RBLADE)
    bands = [k for k in BANDS if k.startswith(args.arm)]

    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    d = np.load(args.infile, allow_pickle=True)
    quats = d['quats'].astype(np.float64); T = quats.shape[0]
    Gw = fk_world(quats, parent, order)

    gender = str(d['gender']); betas = torch.tensor(np.asarray(d['betas'])[None, :10], dtype=torch.float32)
    model = smplx.create(model_path=MODEL_PATH, model_type='smplh', gender=gender, num_betas=10, ext='pkl')
    lbs = model.lbs_weights.detach().numpy()
    idx_band = {k: band_verts(lbs, *BANDS[k]) for k in bands}
    with torch.no_grad():
        rest = model(betas=betas)
    rv, rj = rest.vertices[0].numpy(), rest.joints[0].numpy()
    rest_spread = {k: perp_spread(rv[None], rj[None, BANDS[k][0]], rj[None, BANDS[k][1]], idx_band[k])[0] for k in bands}

    fr = np.arange(0, T, max(1, T // args.nframes))
    axis = bax[SH]
    yaws = np.arange(-args.range, args.range + args.step / 2, args.step)

    def collapse_for(Qc):
        body = shadow_to_body_pose(Qc[fr]).reshape(len(fr), 63).astype(np.float32)
        bt = torch.tensor(body)
        out_v = np.zeros((len(fr), 6890, 3)); out_j = None
        with torch.no_grad():
            for s in range(0, len(fr), 512):
                sl = slice(s, s + 512); B = bt[sl].shape[0]
                nh = model.num_pca_comps if model.use_pca else 45
                o = model(betas=betas.expand(B, -1), body_pose=bt[sl], global_orient=torch.zeros(B, 3),
                          left_hand_pose=torch.zeros(B, nh), right_hand_pose=torch.zeros(B, nh))
                if out_j is None:
                    out_j = np.zeros((len(fr), o.joints.shape[1], 3))
                out_v[sl] = o.vertices.numpy(); out_j[sl] = o.joints.numpy()
        return {k: (perp_spread(out_v, out_j[:, BANDS[k][0]], out_j[:, BANDS[k][1]], idx_band[k]) / rest_spread[k])
                for k in bands}

    print(f"{Path(args.infile).name}: sweep {args.arm}-shoulder bone-axis roll, mesh band collapse (1=ok,<1=pinch)")
    print(f"  {'roll':>5} " + " ".join(f"{k:>13}" for k in bands))
    best = {}
    for y in yaws:
        h = math.radians(y) / 2.0
        rq = np.array([math.cos(h), *(math.sin(h) * axis)])
        SHw = qmul_np(Gw[:, SH], np.broadcast_to(rq, Gw[:, SH].shape))     # sensor-frame roll (postmul)
        Qc = quats.copy()
        Qc[:, SH] = qmul_np(qconj(Gw[:, BL]), SHw)
        Qc[:, EL] = qmul_np(qconj(SHw), Gw[:, EL])
        c = collapse_for(Qc)
        means = {k: c[k].mean() for k in bands}
        for k in bands:
            best.setdefault(k, (y, means[k]))
            if means[k] > best[k][1]:
                best[k] = (y, means[k])
        print(f"  {y:+5.0f} " + " ".join(f"{means[k]:13.2f}" for k in bands))
    print("\n  best roll (max mean collapse-ratio = least pinch):")
    for k in bands:
        print(f"    {k:13s} roll {best[k][0]:+.0f} deg  ratio {best[k][1]:.2f}")
    print("\nA clear interior peak = an 'obvious advantage' roll the mesh agrees with; flat = no gain.")


if __name__ == "__main__":
    main()