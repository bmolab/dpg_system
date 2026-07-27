"""Prototype: trans correction with planted-foot stationarity constraint.

Adds finite-difference equations to the LS fit: for any pair of consecutive
frames where a foot is planted in both, the corrected world position of that
foot should be approximately the same.  This anchors the corrected `trans`
to the planted foot directly, and (crucially for the right-leg-lift case)
samples the segment rotation `R_segment` in poses that the per-frame
planted-z-quantile fit underweights.

Formulation
-----------
For the rigid 3-coef model, the per-frame shift is `shift(t) = R(t) @ delta`
(3-D).  Left-foot world position is

    L_world(t) = trans_raw(t) - shift(t) + L_local(t)

Stationarity between consecutive planted frames (t, t+1) gives

    (R(t+1) - R(t)) @ delta  ≈  trans_raw(t+1) - trans_raw(t)
                              + L_local(t+1) - L_local(t)

which is 3 equations in `delta`; the floor offset `c` cancels.  The same
form holds for the right foot.

For the full 9-coef model the shift only affects Z, so the stationarity
constraint reduces to its Z component (1 equation per pair).

The new equations are stacked into the existing iterative LS with an
adjustable weight `lambda_stat`.

This is a prototype: output filename gets a `_stat` infix so it doesn't
clash with the production `_<seg>fix` files.

Usage:
    python correct_thigh_offset_stationary.py <input.npz> [<output.npz>]
        [--segment lhip|rhip|pelvis|auto] [--full]
        [--stationarity-weight 1.0] [--no-stationarity]
        [--smooth 0.5] [--fit-range LO:HI]
"""
from __future__ import annotations

import sys
from pathlib import Path
import pickle
import argparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correct_thigh_offset import (
    rodrigues, shape_joints, fk_batch,
    PELVIS, L_HIP, R_HIP, L_FOOT, R_FOOT, N_BODY,
    DEFAULT_MODEL_FEMALE, DEFAULT_MODEL_MALE, DEFAULT_SEGMENT,
    INLIER_PELVIS_BAND, PLANTED_VEL_MAX, PLANTED_BAND, PLANTED_QUANTILE,
    N_ITERS, FLOOR_QUANTILE, OUTLIER_K, OUTLIER_INNER_ITERS,
    RIDGE_LAMBDA, SMOOTH_WINDOW_S,
    smooth_shift, _ridge_fit,
    pick_best_segment,
)


def fit_segment_offset_stationary(R_segment, feet_z, feet_local, pelvis_z,
                                  trans_raw, fps,
                                  full=False, lambda_stat=1.0,
                                  enable_stationarity=True,
                                  verbose=True):
    """Iterative LS fit with planted-foot stationarity equations.

    R_segment: (T, 3, 3)  world rotation of the sensor segment per frame
    feet_z:    (T, 2)     world Z of (L_foot, R_foot)
    feet_local:(T, 2, 3)  model-frame foot positions relative to pelvis,
                          for both (L_foot, R_foot)
    pelvis_z:  (T,)       world Z of pelvis
    trans_raw: (T, 3)     uncorrected pelvis world position
    """
    T = feet_z.shape[0]
    if full:
        u = R_segment.reshape(T, 9)
    else:
        u = R_segment[:, 2, :]
    K = u.shape[1]
    lower = feet_z.min(axis=1)

    pz_med = np.median(pelvis_z)
    inlier = np.abs(pelvis_z - pz_med) < INLIER_PELVIS_BAND
    if verbose:
        print(f'  inliers: {inlier.sum()}/{T} ({100*inlier.mean():.2f}%)  '
              f'pelvis_z med={pz_med:+.3f}')

    # Initial LS on lower-foot z over inliers
    X0 = np.concatenate([u, np.ones((T, 1))], axis=1)
    beta, *_ = np.linalg.lstsq(X0[inlier], lower[inlier], rcond=None)
    delta = beta[:K]

    # Per-foot vertical velocity (central differences)
    vel = np.zeros_like(feet_z)
    if T >= 3:
        vel[1:-1] = np.abs(feet_z[2:] - feet_z[:-2]) * 0.5 * fps
        vel[0] = vel[1]; vel[-1] = vel[-2]

    n_pairs_used = 0
    for it in range(N_ITERS):
        rot_z = u @ delta if full else R_segment[:, 2, :] @ delta  # Z-shift
        # Construct shift_z used for planted-foot mask:
        if full:
            shift_z = rot_z
        else:
            shift_z = (R_segment @ delta)[:, 2]
        foot_corr = feet_z - shift_z[:, None]
        planted = np.zeros((T, 2), dtype=bool)
        for f in range(2):
            thresh = np.quantile(foot_corr[inlier, f], PLANTED_QUANTILE)
            planted[:, f] = (foot_corr[:, f] < thresh + PLANTED_BAND) \
                            & (vel[:, f] < PLANTED_VEL_MAX) & inlier

        # ── Block A: per-frame planted-foot Z equations (the original) ──
        rows_X, rows_y = [], []
        for f in range(2):
            m = planted[:, f]
            if m.sum() == 0:
                continue
            rows_X.append(np.concatenate([u[m], np.ones((m.sum(), 1))], axis=1))
            rows_y.append(feet_z[m, f])

        # ── Block B: consecutive-frame stationarity (the new part) ──
        n_pairs_used = 0
        if enable_stationarity:
            for f in range(2):
                pair = planted[:-1, f] & planted[1:, f]
                idx = np.where(pair)[0]
                if idx.size == 0:
                    continue
                # Δlocal[i] = L_local(t+1) - L_local(t),  Δtrans[i] = trans(t+1) - trans(t)
                d_local = feet_local[idx + 1, f] - feet_local[idx, f]   # (P, 3)
                d_trans = trans_raw[idx + 1] - trans_raw[idx]           # (P, 3)
                rhs_3d = d_trans + d_local                              # (P, 3)
                if full:
                    # Z component only;  Δu = u(t+1) - u(t)  in 9-dim
                    dU = u[idx + 1] - u[idx]                            # (P, 9)
                    X_pair = np.concatenate([dU, np.zeros((idx.size, 1))], axis=1)
                    y_pair = rhs_3d[:, 2]
                    rows_X.append(lambda_stat * X_pair)
                    rows_y.append(lambda_stat * y_pair)
                else:
                    # 3D stationarity, rigid model
                    dR_full = R_segment[idx + 1] - R_segment[idx]      # (P, 3, 3)
                    for i in range(3):
                        A_i = dR_full[:, i, :]                          # (P, 3)
                        X_pair = np.concatenate([A_i, np.zeros((idx.size, 1))], axis=1)
                        y_pair = rhs_3d[:, i]
                        rows_X.append(lambda_stat * X_pair)
                        rows_y.append(lambda_stat * y_pair)
                n_pairs_used += idx.size

        Xp = np.concatenate(rows_X, axis=0)
        yp = np.concatenate(rows_y, axis=0)

        # Ridge LS + MAD outlier trim (same as the production routine)
        n_in_iter = Xp.shape[0]
        n_kept = n_in_iter
        beta = _ridge_fit(Xp, yp, K, RIDGE_LAMBDA)
        for _ in range(OUTLIER_INNER_ITERS):
            r = Xp @ beta - yp
            r_med = np.median(r)
            mad = 1.4826 * np.median(np.abs(r - r_med))
            if mad < 1e-6:
                break
            keep = np.abs(r - r_med) < OUTLIER_K * mad
            if keep.sum() < max(K + 5, 0.5 * n_kept) or keep.sum() == n_kept:
                break
            Xp = Xp[keep]; yp = yp[keep]
            n_kept = Xp.shape[0]
            beta = _ridge_fit(Xp, yp, K, RIDGE_LAMBDA)

        delta_new, c_new = beta[:K], float(beta[K])
        diff = float(np.linalg.norm(delta_new - delta))
        delta = delta_new
        if verbose:
            print(f'  iter {it:2d}: stationarity_pairs={n_pairs_used:5d}  '
                  f'coefs={delta}  c={c_new:+.4f}  diff={diff:.5f}  '
                  f'(kept {n_kept}/{n_in_iter})')
        if diff < 1e-4:
            break

    # Final c chosen by lower-foot floor quantile (unchanged from production)
    rot_z_final = (R_segment @ delta)[:, 2] if not full else u @ delta
    pre_corr = lower - rot_z_final
    c = float(np.quantile(pre_corr[inlier], FLOOR_QUANTILE))
    info = {
        'inlier_fraction': float(inlier.mean()),
        'iterations': it + 1,
        'stationarity_pairs': int(n_pairs_used),
        'stationarity_enabled': bool(enable_stationarity),
        'lambda_stat': float(lambda_stat),
    }
    return delta, c, info


def correct_take_stationary(in_path, out_path,
                            segment=DEFAULT_SEGMENT,
                            smooth_window_s=SMOOTH_WINDOW_S,
                            full=False,
                            lambda_stat=1.0,
                            enable_stationarity=True,
                            fit_range=None,
                            verbose=True):
    data = np.load(in_path, allow_pickle=True)
    poses = data['poses']
    trans = data['trans']
    betas = np.asarray(data['betas']).flatten()
    gender = str(data['gender']).lower().strip().strip("'\"")
    fps = float(data['mocap_framerate']) if 'mocap_framerate' in data.files else 100.0
    T = poses.shape[0]

    model_path = DEFAULT_MODEL_FEMALE if gender == 'female' else DEFAULT_MODEL_MALE
    with open(model_path, 'rb') as f:
        model = pickle.load(f, encoding='latin1')
    parents = model['kintree_table'][0].astype(np.int64).copy()
    parents[0] = -1
    parents = parents[:N_BODY]
    J_rest = shape_joints(model, betas)[:N_BODY]
    body_pose = poses[:, :N_BODY * 3].reshape(T, N_BODY, 3)

    if verbose:
        seg_name = {PELVIS: 'pelvis', L_HIP: 'l_hip', R_HIP: 'r_hip'}.get(
            segment, f'joint_{segment}')
        print(f'Take: {in_path.name}  T={T}  fps={fps}  gender={gender}  '
              f'segment={seg_name}  model={"full(9)" if full else "rigid(3)"}  '
              f'stationarity={"on" if enable_stationarity else "off"} '
              f'(lambda={lambda_stat})')

    # FK in chunks → full feet positions (not just Z), R_segment, pelvis Z
    chunk = 4000
    feet_local = np.empty((T, 2, 3))      # (T, foot, xyz) in model frame
    R_segment = np.empty((T, 3, 3))
    pelvis_local_z = np.empty(T)
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        tw, Rw = fk_batch(J_rest, body_pose[s:e], parents)
        feet_local[s:e, 0] = tw[:, L_FOOT]
        feet_local[s:e, 1] = tw[:, R_FOOT]
        R_segment[s:e] = Rw[:, segment]
        pelvis_local_z[s:e] = tw[:, 0, 2]

    feet_z = feet_local[:, :, 2] + trans[:, 2:3]
    pelvis_z = pelvis_local_z + trans[:, 2]

    if fit_range is not None:
        lo, hi = fit_range
        lo = max(0, lo); hi = min(T, hi)
        sl = slice(lo, hi)
        if verbose:
            print(f'  fit_range: [{lo}, {hi}) — fitting on {hi-lo}/{T} frames')
    else:
        sl = slice(0, T)

    delta, c, info = fit_segment_offset_stationary(
        R_segment[sl], feet_z[sl], feet_local[sl], pelvis_z[sl], trans[sl],
        fps, full=full, lambda_stat=lambda_stat,
        enable_stationarity=enable_stationarity, verbose=verbose,
    )

    # Apply correction (same path as the production routine)
    if full:
        shift_z = R_segment.reshape(T, 9) @ delta
        shift_raw = np.zeros((T, 3))
        shift_raw[:, 2] = shift_z
    else:
        shift_raw = R_segment @ delta
    shift = smooth_shift(shift_raw, fps, window_s=smooth_window_s)
    trans_corr = trans - shift
    trans_corr[:, 2] -= c

    feet_corr_z = feet_local[:, :, 2] + trans_corr[:, 2:3]
    lower_corr = feet_corr_z.min(axis=1)
    inlier = np.abs(pelvis_z - np.median(pelvis_z)) < INLIER_PELVIS_BAND
    if verbose:
        print(f'\n  delta = {delta}  ||delta||={np.linalg.norm(delta):.4f} m')
        print(f'  c     = {c:+.4f}')
        print(f'  stationarity_pairs={info["stationarity_pairs"]}  '
              f'iters={info["iterations"]}')
        print(f'  lower_z (inliers) min/med/max: '
              f'{lower_corr[inlier].min():+.3f} / {np.median(lower_corr[inlier]):+.3f} / '
              f'{lower_corr[inlier].max():+.3f}')
        print(f'  fraction lower_z < -0.02 m: {(lower_corr[inlier]<-0.02).mean():.3%}')
        print(f'  fraction lower_z < -0.10 m: {(lower_corr[inlier]<-0.10).mean():.3%}')

    out = {k: data[k] for k in data.files}
    out['trans'] = trans_corr
    out['_belt_offset_delta'] = delta
    out['_belt_offset_floor_c'] = np.array(c)
    out['_belt_offset_segment'] = np.array(segment)
    out['_belt_offset_full_model'] = np.array(bool(full))
    out['_belt_offset_inlier_fraction'] = np.array(info['inlier_fraction'])
    out['_belt_offset_smooth_window_s'] = np.array(smooth_window_s)
    out['_belt_offset_stationarity_pairs'] = np.array(info['stationarity_pairs'])
    out['_belt_offset_stationarity_lambda'] = np.array(info['lambda_stat'])
    np.savez(out_path, **out)
    if verbose:
        print(f'\nSaved: {out_path}')
    return {'delta': delta, 'c': c, **info}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input')
    p.add_argument('output', nargs='?', default=None)
    p.add_argument('--segment', default='rhip',
                   help='pelvis | lhip | rhip | auto[:s1,s2,...]')
    p.add_argument('--full', action='store_true',
                   help='9-coef Z-only model (axial slip).  Default: rigid 3-coef.')
    p.add_argument('--stationarity-weight', type=float, default=1.0,
                   dest='lambda_stat',
                   help='Relative weight of stationarity equations vs '
                        'planted-foot-Z equations (default: 1.0).')
    p.add_argument('--no-stationarity', action='store_true',
                   help='Disable the new constraint (run the rigid/full fit '
                        'on planted-foot Z only, to compare against production).')
    p.add_argument('--smooth', type=float, default=SMOOTH_WINDOW_S)
    p.add_argument('--fit-range', default=None,
                   help='LO:HI frame range to use for fitting.')
    args = p.parse_args()

    fit_range = None
    if args.fit_range:
        lo, hi = args.fit_range.split(':')
        fit_range = (int(lo), int(hi))

    seg_table = {'lhip': L_HIP, 'rhip': R_HIP, 'pelvis': PELVIS}
    if args.segment.startswith('auto'):
        cand_str = args.segment.split(':', 1)[1] if ':' in args.segment else 'pelvis,lhip,rhip'
        cands = [seg_table[s.strip()] for s in cand_str.split(',')]
        seg = pick_best_segment(Path(args.input), cands)
        seg_label = {L_HIP: 'lhip', R_HIP: 'rhip', PELVIS: 'pelvis'}[seg]
    else:
        seg = seg_table[args.segment]
        seg_label = args.segment

    suffix = f'_{seg_label}{"full" if args.full else ""}stat'
    if args.no_stationarity:
        suffix = f'_{seg_label}{"full" if args.full else ""}nostat'
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else \
               in_path.with_name(in_path.stem + suffix + '.npz')

    correct_take_stationary(
        in_path, out_path, segment=seg,
        smooth_window_s=args.smooth, full=args.full,
        lambda_stat=args.lambda_stat,
        enable_stationarity=not args.no_stationarity,
        fit_range=fit_range,
    )


if __name__ == '__main__':
    main()
