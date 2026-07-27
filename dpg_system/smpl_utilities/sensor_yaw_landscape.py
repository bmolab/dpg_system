"""Session-pooled yaw-landscape binned by EACH SENSOR'S OWN world heading (not pelvis yaw).

Physical premise the earlier pooled_yaw_landscape.py violated: a magnetometer's deviation
delta_i(psi) is a function of THAT sensor's own heading psi in Earth's field -- not of the
pelvis/global yaw. Binning the forearm signal by pelvis yaw smears the real curve across
choreography. Here every signal is binned by the heading of the sensor that actually carries it:

  twist L/R-shldr  ->  upper-arm sensor heading      crease L/R upper-arm -> upper-arm sensor
  twist L/R-elbow  ->  forearm sensor heading        crease L/R forearm   -> forearm sensor
  headlock L/R     ->  forearm sensor heading

Magnetization cannot drift within a session, so a CONSISTENT yaw-shape must exist. To expose it
we DE-MEAN each take's binned curve (subtract that take's own mean) -- this removes the per-take
constant offset (recalibration drift) and leaves only the session-stable SHAPE. We then average
the de-meaned shapes across the 11 takes and report, per bin, the cross-take MEAN and the STD
across takes. A low STD with a structured mean = the magnetization fingerprint the physics
requires; a flat mean = constant/mount error only.
"""
import argparse
import glob
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
XAX = np.array([1., 0., 0.])


def qconj(q): return q * np.array([1., -1., -1., -1.])
def twist(loc, ax): return np.degrees(2 * np.arctan2((loc[:, 1:] * ax).sum(-1), loc[:, 0]))
def wrapdeg(a): return (a + 180) % 360 - 180
def sensor_yaw(Gj): v = qrot_np(Gj, XAX); return np.degrees(np.arctan2(v[:, 2], v[:, 0]))
def yaw_bin(deg): return ((deg + 180) // 30).astype(int) % NB
def band_faces(faces, vset): return faces[np.isin(faces, list(vset)).all(1)]
def areas(v, f):
    v0, v1, v2 = v[:, f[:, 0]], v[:, f[:, 1]], v[:, f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)


def binned_curve(vals, hb):
    """Per-bin mean and count for one take."""
    s = np.zeros(NB); c = np.zeros(NB)
    for b in range(NB):
        m = hb == b
        if m.any():
            s[b] = vals[m].sum(); c[b] = m.sum()
    return s, c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--nframes", type=int, default=2500, help="mesh frames per take")
    ap.add_argument("--turn-rate", type=float, default=0.5, help="min |body yaw-rate| deg/frame to count as a turn")
    ap.add_argument("--no-mesh", action="store_true", help="skip the (heavy) mesh-crease signal")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*beta.npz")))
    parent, order, bax = load_skeleton(Path(__file__).resolve().parent.parent / 'definition.xml')
    models, mesh_cache = {}, {}

    # signal name -> (binning sensor joint). de-meaned per-take residual curves are collected here.
    SIG_SENSOR = {
        'twist L-shldr': LSH, 'twist R-shldr': RSH,
        'twist L-elbow': LEL, 'twist R-elbow': REL,
    }
    CREASE_SENSOR = {'crease L-uarm': (LSH, 'L upper-arm'), 'crease R-uarm': (RSH, 'R upper-arm'),
                     'crease L-fore': (LEL, 'L forearm'),   'crease R-fore': (REL, 'R forearm')}
    HEADLOCK_SENSOR = {'headlock L': LEL, 'headlock R': REL}

    # twist/headlock: de-meaned per-take curves (per-take constant removed). crease: raw p90 per
    # bin, NOT de-meaned (crease=0 is the meaningful clean baseline; de-meaning would erase it).
    resid = {k: [] for k in list(SIG_SENSOR) + list(HEADLOCK_SENSOR)}
    crease_raw = {k: [] for k in (CREASE_SENSOR if not args.no_mesh else {})}

    def add_resid(name, sumv, cnt):
        with np.errstate(invalid='ignore', divide='ignore'):
            curve = np.where(cnt > 0, sumv / np.maximum(cnt, 1), np.nan)
        tot = cnt.sum()
        if tot < 1:
            resid[name].append(np.full(NB, np.nan)); return
        take_mean = sumv.sum() / tot                       # count-weighted take mean
        resid[name].append(curve - take_mean)              # de-meaned shape

    for f in files:
        d = np.load(f, allow_pickle=True)
        q = d['quats'].astype(np.float64); T = q.shape[0]
        G = fk_world(q, parent, order)
        pel = sensor_yaw(G[:, PELV])

        fr = np.arange(0, T, max(1, T // args.nframes))

        # --- twist signals, each binned by its OWN sensor heading ---
        twists = {
            'twist L-shldr': (twist(qmul_np(qconj(G[fr][:, LBLADE]), G[fr][:, LSH]), bax[LSH]), LSH),
            'twist R-shldr': (twist(qmul_np(qconj(G[fr][:, RBLADE]), G[fr][:, RSH]), bax[RSH]), RSH),
            'twist L-elbow': (twist(qmul_np(qconj(G[fr][:, LSH]), G[fr][:, LEL]), bax[LEL]), LEL),
            'twist R-elbow': (twist(qmul_np(qconj(G[fr][:, RSH]), G[fr][:, REL]), bax[REL]), REL),
        }
        for name, (vals, sj) in twists.items():
            hb = yaw_bin(sensor_yaw(G[fr][:, sj]))
            add_resid(name, *binned_curve(vals, hb))

        # --- heading-locked slope: body-relative limb yaw-rate vs body yaw-rate, by sensor heading ---
        def syaw_unwrap(j): v = qrot_np(G[:, j], XAX); return np.unwrap(np.arctan2(v[:, 2], v[:, 0]))
        pel_un = np.unwrap(np.radians(pel))
        psidot = wrapdeg(np.degrees(np.roll(pel_un, -1) - pel_un))
        turn = np.abs(psidot) > args.turn_rate
        for name, sj in HEADLOCK_SENSOR.items():
            rel = syaw_unwrap(sj) - pel_un
            dr = wrapdeg(np.degrees(np.roll(rel, -1) - rel))
            hbf = yaw_bin(sensor_yaw(G[:, sj]))
            slope = np.full(NB, np.nan)
            for b in range(NB):
                m = turn & (hbf == b)
                if m.sum() >= 8 and (psidot[m] ** 2).sum() > 1e-6:
                    slope[b] = (psidot[m] * dr[m]).sum() / (psidot[m] ** 2).sum()
            # de-mean the slope shape over occupied bins
            occ = ~np.isnan(slope)
            resid[name].append(slope - slope[occ].mean() if occ.any() else np.full(NB, np.nan))

        # --- mesh crease, each band binned by its segment's sensor heading ---
        if not args.no_mesh:
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
            cvals = {k: np.zeros(len(fr)) for k in BANDS}
            with torch.no_grad():
                for s in range(0, len(fr), 512):
                    sl = slice(s, s + 512); B = body[sl].shape[0]
                    nh = model.num_pca_comps if model.use_pca else 45
                    o = model(betas=betas.expand(B, -1), body_pose=body[sl], global_orient=torch.zeros(B, 3),
                              left_hand_pose=torch.zeros(B, nh), right_hand_pose=torch.zeros(B, nh))
                    v = o.vertices.numpy()
                    for k in BANDS:
                        a = areas(v, bf[k]); cvals[k][sl] = np.clip(rest_area[k][None] - a, 0, None).sum(1) / (rest_area[k].sum() + 1e-9)
            for name, (sj, band) in CREASE_SENSOR.items():
                hb = yaw_bin(sensor_yaw(G[fr][:, sj]))
                p90 = np.full(NB, np.nan)                     # crease is sparse+peaky: high quantile, not mean
                for b in range(NB):
                    m = hb == b
                    if m.sum() >= 8:
                        p90[b] = np.percentile(cvals[band][m], 90)
                crease_raw[name].append(p90)
        print(f"  pooled {os.path.basename(f).replace('Subject7_take','').replace('_a_beta.npz','')}")

    # --- aggregate across takes. Report the POOLED MEAN shape and its standard ERROR (std/sqrt(n
    # takes in bin)) -- per-take scatter is large (choreography), but the question is whether the
    # 11-take mean curve is resolved above its own uncertainty. SNR_mean = var(mean)/mean(SEM^2).
    hdr = "  ".join(f"{-180+30*b:+4d}" for b in range(NB))

    def report(name, arr, de_meaned):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        ntk = np.sum(~np.isnan(arr), axis=0)
        sem = np.where(ntk > 1, std / np.sqrt(np.maximum(ntk, 1)), np.nan)
        cells = ["   .  " if np.isnan(mean[b]) else f"{mean[b]:+4.1f}" for b in range(NB)]
        print(f"  {name:14s}" + " ".join(cells))
        scells = ["   .  " if np.isnan(sem[b]) else f"({sem[b]:3.1f})" for b in range(NB)]
        print(f"  {'  sem':14s}" + " ".join(scells))
        occ = ~np.isnan(mean) & ~np.isnan(sem)
        if occ.sum() >= 3:
            # for crease (not de-meaned) the structure is the spread of the curve about its own min/mean;
            # use variance about the mean in both cases for a like-for-like SNR.
            shape_var = np.nanvar(mean[occ])
            sem_var = np.nanmean(sem[occ] ** 2)
            snr = shape_var / (sem_var + 1e-9)
            print(f"  {'  SNR_mean':14s}{snr:5.2f}   (shape {np.sqrt(shape_var):.2f} vs SEM {np.sqrt(sem_var):.2f})")

    print(f"\nDE-MEANED twist/headlock yaw-shape, binned by EACH SENSOR'S OWN heading ({len(files)} takes).")
    print("Each cell = pooled mean (standard error). columns = sensor world-heading bins (deg).")
    print(f"  {'':14s}{hdr}")
    for name in list(SIG_SENSOR) + list(HEADLOCK_SENSOR):
        report(name, np.array(resid[name]), True)

    if not args.no_mesh:
        print(f"\nMESH-CREASE (band-area collapse, p90 per bin -- NOT de-meaned; 0 = clean) by sensor heading.")
        print(f"  {'':14s}{hdr}")
        for name in CREASE_SENSOR:
            report(name, np.array(crease_raw[name]), False)

    print("\nSNR_mean>1 => the 11-take pooled curve is resolved above its standard error = a")
    print("session-consistent yaw-shape (magnetization). Flat mean => constant/mount error only.")


if __name__ == "__main__":
    main()
