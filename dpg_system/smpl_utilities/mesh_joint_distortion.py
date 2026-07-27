"""Mesh-distortion-at-joints detector: the style-independent 'hint that relative rotations are wrong'.

A wrong relative rotation -- especially excess axial twist -- makes LBS collapse the skinned mesh
toward the bone axis at the joint (the 'candy-wrapper' pinch). We measure that directly: deform the
SMPL(-H) mesh per frame, and for the vertex band straddling each arm joint, compute the mean
perpendicular spread of those vertices about the bone axis, relative to the rest pose. Ratio < 1 =
collapse. This captures ripple automatically (a bad shoulder roll pinches the UPPER-ARM band; a bad
forearm twist pinches the FOREARM band), and unlike an abstract twist angle it's grounded in the
actual mesh. Diagnostic only.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import smplx

from diag_pose_plausibility import shadow_to_body_pose

MODEL_PATH = '/Users/drokeby/dpg_system/dpg_system'
# SMPL joint indices and the two bones whose band we watch for each arm segment
BANDS = {  # name: (proximal joint, distal joint)
    'L upper-arm': (16, 18), 'R upper-arm': (17, 19),
    'L forearm':   (18, 20), 'R forearm':   (19, 21),
}


def band_verts(lbs, pj, dj, thr=0.15):
    return np.where((lbs[:, pj] > thr) & (lbs[:, dj] > thr))[0]


def perp_spread(verts, j_prox, j_dist, idx):
    """Mean distance of band vertices from the bone axis (line through the two joints)."""
    a = j_prox; d = j_dist - j_prox
    d = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9)
    v = verts[:, idx] - a[:, None]
    along = (v * d[:, None]).sum(-1, keepdims=True)
    perp = v - along * d[:, None]
    return np.linalg.norm(perp, axis=-1).mean(-1)            # (frames,)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--clip", help="report a frame range 'a:b'")
    args = ap.parse_args()

    d = np.load(args.infile, allow_pickle=True)
    gender = str(d['gender']); betas = torch.tensor(np.asarray(d['betas'])[None, :10], dtype=torch.float32)
    model = smplx.create(model_path=MODEL_PATH, model_type='smplh', gender=gender, num_betas=10, ext='pkl')
    lbs = model.lbs_weights.detach().numpy()
    idx_band = {k: band_verts(lbs, *v) for k, v in BANDS.items()}

    quats = d['quats'].astype(np.float64)
    T = quats.shape[0]
    body = shadow_to_body_pose(quats).reshape(T, 63).astype(np.float32)

    # rest pose
    with torch.no_grad():
        rest = model(betas=betas)
    rv = rest.vertices[0].numpy(); rj = rest.joints[0].numpy()
    rest_spread = {k: perp_spread(rv[None], rj[None, BANDS[k][0]], rj[None, BANDS[k][1]], idx_band[k])[0]
                   for k in BANDS}

    stride = args.stride or max(1, T // 1500)
    fr = np.arange(0, T, stride)
    ratios = {k: np.zeros(len(fr)) for k in BANDS}
    bt = torch.tensor(body[fr])
    with torch.no_grad():
        for s in range(0, len(fr), 512):
            sl = slice(s, s + 512)
            B = bt[sl].shape[0]
            nh = model.num_pca_comps if model.use_pca else 45
            out = model(betas=betas.expand(B, -1), body_pose=bt[sl],
                        global_orient=torch.zeros(B, 3),
                        left_hand_pose=torch.zeros(B, nh), right_hand_pose=torch.zeros(B, nh))
            v = out.vertices.numpy(); j = out.joints.numpy()
            for k in BANDS:
                sp = perp_spread(v, j[:, BANDS[k][0]], j[:, BANDS[k][1]], idx_band[k])
                ratios[k][sl] = sp / rest_spread[k]

    if args.clip:
        a, b = (int(x) for x in args.clip.split(":"))
        m = (fr >= a) & (fr < b)
        print(f"{Path(args.infile).name} f{a}-{b} band collapse ratio (1=ok, <1=pinch):")
        for k in BANDS:
            print(f"  {k:12s} mean {ratios[k][m].mean():.2f}  min {ratios[k][m].min():.2f}")
        return

    print(f"{Path(args.infile).name}: {T} frames, {len(fr)} sampled. band collapse ratio (1=ok, <1=pinch)")
    for k in BANDS:
        r = ratios[k]
        worst = fr[np.argmin(r)]
        print(f"  {k:12s} band_verts {len(idx_band[k]):4d}  median {np.median(r):.2f}  "
              f"p05 {np.percentile(r,5):.2f}  min {r.min():.2f} @f{worst}  (<0.85 in {100*(r<0.85).mean():.0f}% frames)")


if __name__ == "__main__":
    main()
