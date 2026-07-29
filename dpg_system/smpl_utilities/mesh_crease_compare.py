"""Compare SMPL mesh-area crease (roll-sensitive) between two takes (orig vs a *fix.npz)."""
import argparse
from pathlib import Path

import numpy as np
import torch
import smplx

from diag_pose_plausibility import shadow_to_body_pose
from mesh_joint_distortion import BANDS, band_verts, MODEL_PATH


def band_faces(faces, vset): return faces[np.isin(faces, list(vset)).all(1)]
def areas(v, f):
    v0, v1, v2 = v[:, f[:, 0]], v[:, f[:, 1]], v[:, f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)


def collapse(infile, model, bf, rest_area, fr):
    d = np.load(infile, allow_pickle=True)
    betas = torch.tensor(np.asarray(d['betas'])[None, :10], dtype=torch.float32)
    body = torch.tensor(shadow_to_body_pose(d['quats'].astype(np.float64)[fr]).reshape(len(fr), 63).astype(np.float32))
    out = {k: np.zeros(len(fr)) for k in BANDS}
    with torch.no_grad():
        for s in range(0, len(fr), 512):
            sl = slice(s, s + 512); B = body[sl].shape[0]
            nh = model.num_pca_comps if model.use_pca else 45
            o = model(betas=betas.expand(B, -1), body_pose=body[sl], global_orient=torch.zeros(B, 3),
                      left_hand_pose=torch.zeros(B, nh), right_hand_pose=torch.zeros(B, nh))
            v = o.vertices.numpy()
            for k in BANDS:
                out[k][sl] = (areas(v, bf[k]) / rest_area[k][None]).mean(1)   # mean area ratio (1=ok,<1=crease)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orig"); ap.add_argument("fix")
    ap.add_argument("--nframes", type=int, default=1200)
    ap.add_argument("--frames", help="comma list of specific frames to also report")
    args = ap.parse_args()

    d = np.load(args.orig, allow_pickle=True); T = d['quats'].shape[0]; gender = str(d['gender'])
    model = smplx.create(model_path=MODEL_PATH, model_type='smplh', gender=gender, num_betas=10, ext='pkl')
    lbs = model.lbs_weights.detach().numpy(); faces = model.faces.astype(np.int64)
    bf = {k: band_faces(faces, set(band_verts(lbs, *BANDS[k]).tolist())) for k in BANDS}
    with torch.no_grad():
        rv = model(betas=torch.tensor(np.asarray(d['betas'])[None, :10], dtype=torch.float32)).vertices.numpy()
    rest_area = {k: areas(rv, bf[k])[0] for k in BANDS}

    fr = np.arange(0, T, max(1, T // args.nframes))
    co = collapse(args.orig, model, bf, rest_area, fr)
    cf = collapse(args.fix, model, bf, rest_area, fr)
    print(f"mesh area-collapse ratio (1=ok, <1=crease), {len(fr)} frames.  orig -> fix")
    for k in BANDS:
        print(f"  {k:12s} median {np.median(co[k]):.3f}->{np.median(cf[k]):.3f}  "
              f"p05 {np.percentile(co[k],5):.3f}->{np.percentile(cf[k],5):.3f}  "
              f"min {co[k].min():.3f}->{cf[k].min():.3f}  frac<0.85 {100*(co[k]<.85).mean():.0f}%->{100*(cf[k]<.85).mean():.0f}%")

    if args.frames:
        want = [int(x) for x in args.frames.split(",")]
        fr2 = np.array(want)
        co2 = collapse(args.orig, model, bf, rest_area, fr2); cf2 = collapse(args.fix, model, bf, rest_area, fr2)
        print("  specific frames:")
        for i, fnum in enumerate(want):
            print(f"   f{fnum}: " + "  ".join(f"{k.split()[0]}{k.split()[1][0]} {co2[k][i]:.2f}->{cf2[k][i]:.2f}" for k in BANDS))


if __name__ == "__main__":
    main()
