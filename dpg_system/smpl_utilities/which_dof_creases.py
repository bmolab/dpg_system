"""Causal test: which axial DOF actually produces the visible candy-wrapper (at f4318)?

Kinematic-tree reasoning kept contradicting the data, so test it directly: at a frame window,
neutralize one axial DOF at a time in the Shadow world orientations, rebuild the SMPL pose, deform
the mesh, and measure each arm band's crease with the ROLL-SENSITIVE area-collapse metric (perp-
spread is ~roll-invariant -- the wrong instrument). The variant that RESTORES the band area is the
DOF that caused the crease.

Variants per frame:
  base        : original
  elbow0      : remove elbow axial twist (forearm rolled so forearm-vs-upperarm twist = 0)
  shldr0      : remove upper-arm axial roll vs blade, CARRYING the forearm rigidly (elbow-relative
                orientation preserved) -- isolates the shoulder roll's own effect
  both0       : both of the above
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import smplx

from diag_magnetometer_deviation import (load_skeleton, fk_world, qmul_np, qrot_np,
                                         RSH, REL, RBLADE, LSH, LEL, LBLADE)
from diag_pose_plausibility import shadow_to_body_pose
from mesh_joint_distortion import BANDS, band_verts, MODEL_PATH
from apply_forearm_magfix import relocal_from_world


def qconj(q): return q * np.array([1., -1., -1., -1.])


def swing_twist(q, axis):
    """Decompose q (w-first) into (swing, twist) about unit axis. twist is the rotation about axis."""
    v = q[..., 1:]
    proj = (v * axis).sum(-1, keepdims=True) * axis
    tw = np.concatenate([q[..., :1], proj], -1)
    tw = tw / (np.linalg.norm(tw, axis=-1, keepdims=True) + 1e-12)
    sw = qmul_np(q, qconj(tw))
    return sw, tw


def band_faces(faces, vset): return faces[np.isin(faces, list(vset)).all(1)]
def areas(v, f):
    v0, v1, v2 = v[:, f[:, 0]], v[:, f[:, 1]], v[:, f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("--frame", type=int, default=4318)
    ap.add_argument("--win", type=int, default=8)
    args = ap.parse_args()

    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    d = np.load(args.infile, allow_pickle=True)
    q = d['quats'].astype(np.float64); T = q.shape[0]
    G0 = fk_world(q, parent, order)
    fr = np.arange(max(0, args.frame - args.win), min(T, args.frame + args.win + 1))

    def neutral_elbow(G, SH, EL):
        G = G.copy()
        qe = qmul_np(qconj(G[:, SH]), G[:, EL])
        sw, _ = swing_twist(qe, bax[EL])
        G[:, EL] = qmul_np(G[:, SH], sw)        # twist about forearm axis removed
        return G

    def neutral_shoulder(G, BL, SH, EL):
        G = G.copy()
        qs = qmul_np(qconj(G[:, BL]), G[:, SH])
        sw, _ = swing_twist(qs, bax[SH])
        SHnew = qmul_np(G[:, BL], sw)
        rel = qmul_np(qconj(G[:, SH]), G[:, EL])  # preserve elbow-relative (carry forearm rigidly)
        G[:, EL] = qmul_np(SHnew, rel)
        G[:, SH] = SHnew
        return G

    variants = {'base': G0,
                'elbow0': neutral_elbow(neutral_elbow(G0, RSH, REL), LSH, LEL),
                'shldr0': neutral_shoulder(neutral_shoulder(G0, RBLADE, RSH, REL), LBLADE, LSH, LEL),
                }
    variants['both0'] = neutral_elbow(neutral_elbow(variants['shldr0'], RSH, REL), LSH, LEL)

    gender = str(d['gender']); betas = torch.tensor(np.asarray(d['betas'])[None, :10], dtype=torch.float32)
    model = smplx.create(model_path=MODEL_PATH, model_type='smplh', gender=gender, num_betas=10, ext='pkl')
    lbs = model.lbs_weights.detach().numpy(); faces = model.faces.astype(np.int64)
    bf = {k: band_faces(faces, set(band_verts(lbs, *BANDS[k]).tolist())) for k in BANDS}
    with torch.no_grad():
        rv = model(betas=betas).vertices.numpy()
    rest_area = {k: areas(rv, bf[k])[0] for k in BANDS}

    print(f"{Path(args.infile).name}  f{args.frame}+/-{args.win}: mean band area-collapse (1=ok, <1=crease)")
    print(f"  {'variant':8s}" + " ".join(f"{k:>12}" for k in BANDS))
    for name, G in variants.items():
        Qloc = relocal_from_world(G, parent, order)
        body = torch.tensor(shadow_to_body_pose(Qloc[fr]).reshape(len(fr), 63).astype(np.float32))
        with torch.no_grad():
            B = body.shape[0]; nh = model.num_pca_comps if model.use_pca else 45
            o = model(betas=betas.expand(B, -1), body_pose=body, global_orient=torch.zeros(B, 3),
                      left_hand_pose=torch.zeros(B, nh), right_hand_pose=torch.zeros(B, nh))
            v = o.vertices.numpy()
        row = {}
        for k in BANDS:
            a = areas(v, bf[k])
            row[k] = (a / rest_area[k][None]).mean()       # mean area ratio over band+frames
        print(f"  {name:8s}" + " ".join(f"{row[k]:12.2f}" for k in BANDS))
    print("\nThe variant that RAISES a band toward 1.0 identifies the DOF causing that band's crease.")


if __name__ == "__main__":
    main()