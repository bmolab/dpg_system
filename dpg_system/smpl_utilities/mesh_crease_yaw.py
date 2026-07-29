"""Roll-sensitive mesh-crease signal at the arm joints, studied as a LANDSCAPE vs global yaw.

Another soft instrument for the fusion (no single signal corrects anything). Unlike perpendicular
spread (~invariant to roll about the bone axis), a candy-wrapper crease COLLAPSES triangle areas,
so we measure per-face area loss in the band straddling each arm joint: crease = fraction of the
band's rest surface area that has collapsed this frame. Then we bin it by GLOBAL pelvis yaw and by
the arm sensor's OWN yaw -- because magnetization is a function of heading, so a crease signal that
peaks at particular yaws (aggregated over many poses) is the magnetization fingerprint, with
choreography averaged out. Diagnostic; pairs with the other signals' yaw-landscapes.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import smplx

from diag_magnetometer_deviation import load_skeleton, fk_world, qrot_np, PELV
from diag_pose_plausibility import shadow_to_body_pose
from mesh_joint_distortion import BANDS, band_verts, MODEL_PATH

NB = 12


def band_faces(faces, vset):
    m = np.isin(faces, list(vset)).all(1)
    return faces[m]


def areas(verts, faces):
    v0, v1, v2 = verts[:, faces[:, 0]], verts[:, faces[:, 1]], verts[:, faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)   # (frames, nfaces)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("--nframes", type=int, default=1500)
    ap.add_argument("--by", choices=["pelvis", "sensor"], default="pelvis")
    args = ap.parse_args()

    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    d = np.load(args.infile, allow_pickle=True)
    quats = d['quats'].astype(np.float64); T = quats.shape[0]
    Gw = fk_world(quats, parent, order)
    pel_yaw = np.degrees(np.arctan2(*(lambda x: (x[:, 2], x[:, 0]))(qrot_np(Gw[:, PELV], np.array([1., 0, 0])))))

    gender = str(d['gender']); betas = torch.tensor(np.asarray(d['betas'])[None, :10], dtype=torch.float32)
    model = smplx.create(model_path=MODEL_PATH, model_type='smplh', gender=gender, num_betas=10, ext='pkl')
    lbs = model.lbs_weights.detach().numpy(); faces = model.faces.astype(np.int64)
    bf = {k: band_faces(faces, set(band_verts(lbs, *BANDS[k]).tolist())) for k in BANDS}

    with torch.no_grad():
        rest = model(betas=betas)
    rv = rest.vertices.numpy()
    rest_area = {k: areas(rv, bf[k])[0] for k in BANDS}     # per-face rest area

    fr = np.arange(0, T, max(1, T // args.nframes))
    body = shadow_to_body_pose(quats[fr]).reshape(len(fr), 63).astype(np.float32)
    bt = torch.tensor(body)
    crease = {k: np.zeros(len(fr)) for k in BANDS}
    with torch.no_grad():
        for s in range(0, len(fr), 512):
            sl = slice(s, s + 512); B = bt[sl].shape[0]
            nh = model.num_pca_comps if model.use_pca else 45
            o = model(betas=betas.expand(B, -1), body_pose=bt[sl], global_orient=torch.zeros(B, 3),
                      left_hand_pose=torch.zeros(B, nh), right_hand_pose=torch.zeros(B, nh))
            v = o.vertices.numpy()
            for k in BANDS:
                a = areas(v, bf[k])                          # (B, nfaces)
                lost = np.clip(rest_area[k][None] - a, 0, None).sum(1) / (rest_area[k].sum() + 1e-9)
                crease[k][sl] = lost

    key = pel_yaw[fr]
    hb = ((key + 180) // 30).astype(int) % NB
    print(f"{Path(args.infile).name}: mesh-crease (fraction of band area collapsed) by global pelvis yaw")
    print(f"  {'band':12s} " + " ".join(f"{-180+30*b:+4d}" for b in range(NB)) + "   overall")
    for k in BANDS:
        row = [crease[k][hb == b].mean() if (hb == b).any() else np.nan for b in range(NB)]
        cells = " ".join(f"{v:4.2f}" if not np.isnan(v) else "   ." for v in row)
        print(f"  {k:12s} {cells}   {crease[k].mean():.3f}")
    cnt = [int((hb == b).sum()) for b in range(NB)]
    print(f"  {'n frames':12s} " + " ".join(f"{c:4d}" for c in cnt))
    print("\nA band whose crease PEAKS at particular global yaws (across varied poses) = magnetization")
    print("fingerprint at those headings. Compare across the other signals' yaw-landscapes + takes.")


if __name__ == "__main__":
    main()