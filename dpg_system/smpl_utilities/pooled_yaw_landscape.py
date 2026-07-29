"""Combined, session-pooled yaw-landscape: every soft instrument as a function of GLOBAL yaw.

No single signal corrects anything; each paints a landscape, and the useful view is how it varies
with global yaw (magnetization is a function of yaw; aggregating over many poses per yaw bin and
over many takes averages out choreography). This pools the 11 *beta.npz takes and reports, per
30-deg global-pelvis-yaw bin, four instruments per arm:

  crease    : mesh band-area collapse (upper-arm, forearm)        [mesh, roll-sensitive]
  twist     : signed axial twist of the joint vs parent (deg)     [kinematic]
  asym      : L/R full-orientation mirror residual (combined)     [kinematic]
  headlock  : body-relative limb yaw-rate / body yaw-rate slope   [motion; = delta'(psi)]

Reading: a row ~FLAT vs yaw => constant/mount error; a row STRUCTURED vs yaw => magnetization. The
combination at each yaw, per sensor, is the substrate the correction is fit to.
"""
import argparse
import glob
import math
import os
from pathlib import Path

import numpy as np
import torch
import smplx

from diag_magnetometer_deviation import (load_skeleton, fk_world, qrot_np, qmul_np, PELV,
                                         LSH, LEL, RSH, REL, LBLADE, RBLADE)
from diag_pose_plausibility import shadow_to_body_pose
from mesh_joint_distortion import BANDS, band_verts, MODEL_PATH

NB = 12
UP = np.array([0.0, 1.0, 0.0])
SEC = [np.array([0., 0., 1.]), np.array([0., 1., 0.])]


def qconj(q): return q * np.array([1., -1., -1., -1.])
def twist(loc, ax): return np.degrees(2 * np.arctan2((loc[:, 1:] * ax).sum(-1), loc[:, 0]))
def wrapdeg(a): return (a + 180) % 360 - 180
def band_faces(faces, vset): return faces[np.isin(faces, list(vset)).all(1)]
def areas(v, f):
    v0, v1, v2 = v[:, f[:, 0]], v[:, f[:, 1]], v[:, f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--nframes", type=int, default=2500, help="mesh frames per take")
    ap.add_argument("--turn-rate", type=float, default=20.0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*beta.npz")))
    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    models, mesh_cache = {}, {}

    # accumulators: signal -> (sum[NB], count[NB]); headlock -> (sum_xy[NB], sum_xx[NB])
    rows = ['crease L-uarm', 'crease R-uarm', 'crease L-fore', 'crease R-fore',
            'twist L-shldr', 'twist R-shldr', 'twist L-elbow', 'twist R-elbow',
            'asym  uarm', 'asym  fore']
    S = {r: np.zeros(NB) for r in rows}; C = {r: np.zeros(NB) for r in rows}
    HLxy = {a: np.zeros(NB) for a in ('headlock L', 'headlock R')}
    HLxx = {a: np.zeros(NB) for a in ('headlock L', 'headlock R')}
    binN = np.zeros(NB, int)

    def addbin(name, vals, hb):
        for b in range(NB):
            m = hb == b
            if m.any():
                S[name][b] += vals[m].sum(); C[name][b] += m.sum()

    for f in files:
        d = np.load(f, allow_pickle=True)
        q = d['quats'].astype(np.float64); T = q.shape[0]
        G = fk_world(q, parent, order)
        pel = np.degrees(np.arctan2(*(lambda x: (x[:, 2], x[:, 0]))(qrot_np(G[:, PELV], np.array([1., 0, 0])))))

        # --- kinematic signals on a common subsample ---
        fr = np.arange(0, T, max(1, T // args.nframes))
        hb = ((pel[fr] + 180) // 30).astype(int) % NB
        binN += np.bincount(hb, minlength=NB)
        addbin('twist L-shldr', twist(qmul_np(qconj(G[fr][:, LBLADE]), G[fr][:, LSH]), bax[LSH]), hb)
        addbin('twist R-shldr', twist(qmul_np(qconj(G[fr][:, RBLADE]), G[fr][:, RSH]), bax[RSH]), hb)
        addbin('twist L-elbow', twist(qmul_np(qconj(G[fr][:, LSH]), G[fr][:, LEL]), bax[LEL]), hb)
        addbin('twist R-elbow', twist(qmul_np(qconj(G[fr][:, RSH]), G[fr][:, REL]), bax[REL]), hb)
        # symmetry residual (full-orientation mirror, pelvis plane), per band
        lat = qrot_np(G[fr][:, PELV], np.array([1., 0, 0])); lat -= UP * (lat @ UP)[:, None]
        lat /= np.linalg.norm(lat, axis=-1, keepdims=True) + 1e-9
        Mf = np.eye(3)[None] - 2 * lat[:, :, None] * lat[:, None, :]
        for nm, Ls, Rs in [('asym  uarm', LSH, RSH), ('asym  fore', LEL, REL)]:
            res = np.zeros(len(fr))
            for la, ra in [(bax[Ls], bax[Rs]), (SEC[0], SEC[0]), (SEC[1], SEC[1])]:
                res += np.linalg.norm(np.einsum('nij,nj->ni', Mf, qrot_np(G[fr][:, Ls], la)) - qrot_np(G[fr][:, Rs], ra), axis=-1)
            addbin(nm, res / 3.0, hb)
        # --- heading-locked slope on full frames ---
        def syaw(j): v = qrot_np(G[:, j], np.array([1., 0, 0])); return np.unwrap(np.arctan2(v[:, 2], v[:, 0]))
        psidot = wrapdeg(np.degrees(np.roll(np.unwrap(np.radians(pel)), -1) - np.unwrap(np.radians(pel))))
        turn = np.abs(psidot) > args.turn_rate
        hbf = ((pel + 180) // 30).astype(int) % NB
        for a, sj in [('headlock L', LEL), ('headlock R', REL)]:
            dr = wrapdeg(np.degrees(np.roll(syaw(sj) - np.unwrap(np.radians(pel)), -1) - (syaw(sj) - np.unwrap(np.radians(pel)))))
            for b in range(NB):
                m = turn & (hbf == b)
                if m.any():
                    HLxy[a][b] += (psidot[m] * dr[m]).sum(); HLxx[a][b] += (psidot[m] ** 2).sum()

        # --- crease (mesh) ---
        gender = str(d['gender'])
        if gender not in models:
            models[gender] = smplx.create(model_path=MODEL_PATH, model_type='smplh', gender=gender, num_betas=10, ext='pkl')
            m0 = models[gender]; lbs = m0.lbs_weights.detach().numpy(); faces = m0.faces.astype(np.int64)
            with torch.no_grad():
                rv = m0(betas=torch.zeros(1, 10)).vertices.numpy()
            bf = {k: band_faces(faces, set(band_verts(lbs, *BANDS[k]).tolist())) for k in BANDS}
            mesh_cache[gender] = (bf, {k: areas(rv, bf[k])[0] for k in BANDS})
        model = models[gender]; bf, rest_area = mesh_cache[gender]
        betas = torch.tensor(np.asarray(d['betas'])[None, :10], dtype=torch.float32)
        body = torch.tensor(shadow_to_body_pose(q[fr]).reshape(len(fr), 63).astype(np.float32))
        creasevals = {k: np.zeros(len(fr)) for k in BANDS}
        with torch.no_grad():
            for s in range(0, len(fr), 512):
                sl = slice(s, s + 512); B = body[sl].shape[0]
                nh = model.num_pca_comps if model.use_pca else 45
                o = model(betas=betas.expand(B, -1), body_pose=body[sl], global_orient=torch.zeros(B, 3),
                          left_hand_pose=torch.zeros(B, nh), right_hand_pose=torch.zeros(B, nh))
                v = o.vertices.numpy()
                for k in BANDS:
                    a = areas(v, bf[k]); creasevals[k][sl] = np.clip(rest_area[k][None] - a, 0, None).sum(1) / (rest_area[k].sum() + 1e-9)
        addbin('crease L-uarm', creasevals['L upper-arm'], hb); addbin('crease R-uarm', creasevals['R upper-arm'], hb)
        addbin('crease L-fore', creasevals['L forearm'], hb); addbin('crease R-fore', creasevals['R forearm'], hb)
        print(f"  pooled {os.path.basename(f).replace('Subject7_take','').replace('_a_beta.npz','')}")

    print(f"\nSESSION-POOLED yaw-landscape ({len(files)} takes). columns = global pelvis yaw bins:")
    hdr = "  " + " ".join(f"{-180+30*b:+5d}" for b in range(NB))
    print(f"  {'':14s}{hdr}")
    for r in rows:
        vals = [S[r][b] / C[r][b] if C[r][b] > 0 else np.nan for b in range(NB)]
        print(f"  {r:14s} " + " ".join(f"{v:5.2f}" if abs(v) < 100 and not np.isnan(v) else (f"{v:5.0f}" if not np.isnan(v) else "    .") for v in vals))
    for a in HLxy:
        vals = [HLxy[a][b] / HLxx[a][b] if HLxx[a][b] > 1e-6 else np.nan for b in range(NB)]
        print(f"  {a:14s} " + " ".join(f"{v:+5.2f}" if not np.isnan(v) else "    ." for v in vals))
    print(f"  {'n frames':14s} " + " ".join(f"{binN[b]:5d}" for b in range(NB)))
    print("\nFLAT row vs yaw => constant/mount (sensor-roll); STRUCTURED row => magnetization (yaw-curve).")


if __name__ == "__main__":
    main()
