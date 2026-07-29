"""Mesh-crease incidence map: WHERE/WHEN the candy-wrapper fires, by sensor heading, pooled session.

One soft lens for the fusion landscape. Uses the ROLL-SENSITIVE area-collapse crease (not perp_spread).
For each arm band we bin per-frame crease by the heading of the sensor that drives it (upper-arm band
by upper-arm heading, forearm band by forearm heading), pooled across takes. A band whose crease PEAKS
at particular headings = a magnetization-linked fingerprint; heading-flat crease = pose/skinning. Also
prints per-take mean crease (watch for bimodal clustering = possible two-suit split, since which suit
per take is unrecorded).
"""
import argparse, glob, os
from pathlib import Path
import numpy as np, torch, smplx

from diag_magnetometer_deviation import load_skeleton, fk_world, qrot_np, LSH, RSH, LEL, REL
from diag_pose_plausibility import shadow_to_body_pose
from mesh_joint_distortion import BANDS, band_verts, MODEL_PATH

NB = 12
X = np.array([1., 0, 0.])
HEAD_SENSOR = {'L upper-arm': LSH, 'R upper-arm': RSH, 'L forearm': LEL, 'R forearm': REL}


def band_faces(faces, vset): return faces[np.isin(faces, list(vset)).all(1)]
def areas(v, f):
    v0, v1, v2 = v[:, f[:, 0]], v[:, f[:, 1]], v[:, f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)
def hbin(deg): return ((deg + 180) // 30).astype(int) % NB


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--nframes", type=int, default=1000)
    args = ap.parse_args()
    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    files = sorted(glob.glob(os.path.join(args.dir, "*beta.npz")))
    models, cache = {}, {}
    S = {k: np.zeros(NB) for k in BANDS}; C = {k: np.zeros(NB) for k in BANDS}; Cr = {k: np.zeros(NB) for k in BANDS}
    pertake = {k: [] for k in BANDS}

    for f in files:
        d = np.load(f, allow_pickle=True); q = d['quats'].astype(np.float64); T = q.shape[0]
        G = fk_world(q, parent, order); gender = str(d['gender'])
        if gender not in models:
            m = smplx.create(model_path=MODEL_PATH, model_type='smplh', gender=gender, num_betas=10, ext='pkl')
            lbs = m.lbs_weights.detach().numpy(); faces = m.faces.astype(np.int64)
            bf = {k: band_faces(faces, set(band_verts(lbs, *BANDS[k]).tolist())) for k in BANDS}
            with torch.no_grad(): rv = m(betas=torch.zeros(1, 10)).vertices.numpy()
            models[gender] = m; cache[gender] = (bf, {k: areas(rv, bf[k])[0] for k in BANDS})
        model = models[gender]; bf, rest = cache[gender]
        betas = torch.tensor(np.asarray(d['betas'])[None, :10], dtype=torch.float32)
        fr = np.arange(0, T, max(1, T // args.nframes))
        body = torch.tensor(shadow_to_body_pose(q[fr]).reshape(len(fr), 63).astype(np.float32))
        cr = {k: np.zeros(len(fr)) for k in BANDS}
        with torch.no_grad():
            for s in range(0, len(fr), 512):
                sl = slice(s, s + 512); B = body[sl].shape[0]; nh = model.num_pca_comps if model.use_pca else 45
                o = model(betas=betas.expand(B, -1), body_pose=body[sl], global_orient=torch.zeros(B, 3),
                          left_hand_pose=torch.zeros(B, nh), right_hand_pose=torch.zeros(B, nh))
                v = o.vertices.numpy()
                for k in BANDS: cr[k][sl] = (areas(v, bf[k]) / rest[k][None]).mean(1)
        for k in BANDS:
            hb = hbin(np.degrees(np.arctan2(*(lambda u: (u[:, 2], u[:, 0]))(qrot_np(G[fr][:, HEAD_SENSOR[k]], X)))))
            for b in range(NB):
                m_ = hb == b
                if m_.any(): S[k][b] += cr[k][m_].sum(); C[k][b] += m_.sum(); Cr[k][b] += (cr[k][m_] < 0.85).sum()
            pertake[k].append(cr[k].mean())
        print(f"  {os.path.basename(f).replace('Subject7_take_','').replace('_beta.npz','')[:13]}")

    print(f"\nMESH-CREASE by SENSOR heading (mean collapse ratio; 1=ok <1=crease). pooled {len(files)} takes")
    print(f"  {'band':12s}" + " ".join(f"{-180+30*b:+4d}" for b in range(NB)))
    for k in BANDS:
        vals = [S[k][b] / C[k][b] if C[k][b] else np.nan for b in range(NB)]
        print(f"  {k:12s}" + " ".join("  . " if np.isnan(v) else f"{v:4.2f}" for v in vals))
    print(f"  {'(frac<.85)':12s}")
    for k in BANDS:
        fr_ = [100 * Cr[k][b] / C[k][b] if C[k][b] else np.nan for b in range(NB)]
        print(f"  {k:12s}" + " ".join("  . " if np.isnan(v) else f"{int(v):3d}%" for v in fr_))
    print("\nper-take mean crease (watch for bimodal split = possible two-suit groups):")
    for k in BANDS:
        print(f"  {k:12s} " + " ".join(f"{v:.2f}" for v in pertake[k]))


if __name__ == "__main__":
    main()
