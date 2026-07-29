"""Prototype: flexion-conditional trans correction.

Extends the rigid 3-coef sensor-offset model so `delta` is a linear function
of right-hip flexion:

    shift(t) = R_segment(t) @ (delta_0 + α · f(t))

where `f(t)` is the right-hip flexion angle in radians (centered at the
mean planted-foot value, for numerical conditioning).  `delta_0` captures
the rest-pose mount offset (constant component, including the radial
"thigh-thickness" offset).  `α` captures how the effective mount migrates
as the thigh flexes — the soft-tissue / strap-slip term that a constant-
delta model can't represent.

By default, also adds the planted-foot stationarity constraint from the
sister script `correct_thigh_offset_stationary.py`.

Output filename suffix: `_<seg>flexstat.npz` (or `_<seg>flexnostat.npz`
with `--no-stationarity`).

Usage:
    python correct_thigh_offset_flexion.py <input.npz> [<output.npz>]
        [--segment rhip|lhip|pelvis|auto]
        [--no-stationarity] [--stationarity-weight 1.0]
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

R_KNEE = 5  # SMPL right knee


def compute_flexion(t_world, J_rest):
    """Right-hip flexion angle (rad), centered at rest.

    Returns the angle between the (knee - hip) direction and -world_z,
    minus the same angle in rest pose.  Positive = thigh tilted away
    from vertical-down (forward, sideways, or backward — magnitude only).
    """
    thigh = t_world[:, R_KNEE] - t_world[:, R_HIP]
    thigh_dir = thigh / np.linalg.norm(thigh, axis=1, keepdims=True)
    cos_lift = -thigh_dir[:, 2]
    flex_rad = np.arccos(np.clip(cos_lift, -1, 1))

    rest_vec = J_rest[R_KNEE] - J_rest[R_HIP]
    rest_dir = rest_vec / np.linalg.norm(rest_vec)
    rest_lift = np.arccos(np.clip(-rest_dir[2], -1, 1))
    return flex_rad - rest_lift


def fit_flexion_conditional(R_segment, feet_z, feet_local, pelvis_z,
                            trans_raw, flexion, fps,
                            lambda_stat=1.0, enable_stationarity=True,
                            verbose=True):
    """Iterative LS with flexion-conditional delta + optional stationarity.

    Solves for [delta_0 (3), α (3), c (1)] such that:
        shift_z(t) = R(t)[2,:] @ (delta_0 + α * f_c(t))
    where f_c(t) = flexion(t) - mean(flexion[planted]).

    Returns delta_0, alpha, c, mean_flex_used, info.
    """
    T = feet_z.shape[0]
    K = 6                               # delta_0 (3) + α (3); intercept c added separately
    lower = feet_z.min(axis=1)

    pz_med = np.median(pelvis_z)
    inlier = np.abs(pelvis_z - pz_med) < INLIER_PELVIS_BAND
    if verbose:
        print(f'  inliers: {inlier.sum()}/{T} ({100*inlier.mean():.2f}%)  '
              f'pelvis_z med={pz_med:+.3f}  flexion range '
              f'[{np.degrees(flexion[inlier].min()):.1f}, '
              f'{np.degrees(flexion[inlier].max()):.1f}] deg')

    # Center flexion at planted-foot mean for conditioning
    # (we approximate "planted" pre-fit by inlier ∩ bottom 20% foot_z)
    init_planted = inlier & (lower < np.quantile(lower[inlier], 0.20))
    if init_planted.sum() < 10:
        init_planted = inlier
    mean_flex = float(flexion[init_planted].mean())
    f_c = flexion - mean_flex
    if verbose:
        print(f'  flexion centered at mean={np.degrees(mean_flex):.1f} deg over '
              f'{init_planted.sum()} init-planted frames')

    # u = R_segment(t)[2, :]  (3 cols), augmented with u * f_c (3 more)
    u3 = R_segment[:, 2, :]                          # (T, 3)

    def design_row_planted(idx):
        """Rows for per-frame planted-foot Z eq.  idx: (M,) frame indices."""
        return np.concatenate([
            u3[idx],                                 # delta_0 cols
            u3[idx] * f_c[idx, None],                # α cols
            np.ones((idx.size, 1)),                  # c col
        ], axis=1)

    # Initial fit on lower-foot Z over inliers
    idx_in = np.where(inlier)[0]
    X_init = design_row_planted(idx_in)
    y_init = lower[inlier]
    beta, *_ = np.linalg.lstsq(X_init, y_init, rcond=None)
    delta_0 = beta[:3]; alpha = beta[3:6]

    vel = np.zeros_like(feet_z)
    if T >= 3:
        vel[1:-1] = np.abs(feet_z[2:] - feet_z[:-2]) * 0.5 * fps
        vel[0] = vel[1]; vel[-1] = vel[-2]

    n_pairs_used = 0
    for it in range(N_ITERS):
        # Predicted shift_z(t) under current params, for planted-foot mask
        shift_z = u3 @ delta_0 + (u3 @ alpha) * f_c
        foot_corr = feet_z - shift_z[:, None]
        planted = np.zeros((T, 2), dtype=bool)
        for f in range(2):
            thresh = np.quantile(foot_corr[inlier, f], PLANTED_QUANTILE)
            planted[:, f] = (foot_corr[:, f] < thresh + PLANTED_BAND) \
                            & (vel[:, f] < PLANTED_VEL_MAX) & inlier

        rows_X, rows_y = [], []

        # Block A: planted-foot Z equations
        for f in range(2):
            m = np.where(planted[:, f])[0]
            if m.size == 0:
                continue
            rows_X.append(design_row_planted(m))
            rows_y.append(feet_z[m, f])

        # Block B: consecutive-frame stationarity (3D, rigid model)
        n_pairs_used = 0
        if enable_stationarity:
            for f in range(2):
                pair = planted[:-1, f] & planted[1:, f]
                idx = np.where(pair)[0]
                if idx.size == 0:
                    continue
                d_local = feet_local[idx + 1, f] - feet_local[idx, f]    # (P, 3)
                d_trans = trans_raw[idx + 1] - trans_raw[idx]             # (P, 3)
                rhs_3d = d_trans + d_local                                # (P, 3)

                # Δ(R_segment @ delta_0): shape (P, 3), per spatial component i.
                # Δ(R_segment * f_c @ alpha): same shape.
                R2 = R_segment[idx + 1]                                   # (P, 3, 3)
                R1 = R_segment[idx]
                dR0 = R2 - R1                                             # (P, 3, 3) for delta_0
                dRa = R2 * f_c[idx + 1, None, None] - R1 * f_c[idx, None, None]   # for α

                for i in range(3):
                    row = np.concatenate([
                        dR0[:, i, :],                                     # delta_0 cols
                        dRa[:, i, :],                                     # α cols
                        np.zeros((idx.size, 1)),                          # c col
                    ], axis=1)
                    rows_X.append(lambda_stat * row)
                    rows_y.append(lambda_stat * rhs_3d[:, i])
                n_pairs_used += idx.size

        Xp = np.concatenate(rows_X, axis=0)
        yp = np.concatenate(rows_y, axis=0)

        # Ridge + MAD outlier trim
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

        delta_0_new = beta[:3]
        alpha_new = beta[3:6]
        c_new = float(beta[6])
        diff = float(np.linalg.norm(np.concatenate([delta_0_new - delta_0,
                                                     alpha_new - alpha])))
        delta_0, alpha = delta_0_new, alpha_new
        if verbose:
            print(f'  iter {it:2d}: stat_pairs={n_pairs_used:5d}  '
                  f'δ0={np.round(delta_0,3)}  α={np.round(alpha,3)} m/rad  '
                  f'c={c_new:+.3f}  diff={diff:.5f}  '
                  f'(kept {n_kept}/{n_in_iter})')
        if diff < 1e-4:
            break

    # Final c chosen on lower-foot floor quantile (matching production routine)
    shift_z_final = u3 @ delta_0 + (u3 @ alpha) * f_c
    pre_corr = lower - shift_z_final
    c = float(np.quantile(pre_corr[inlier], FLOOR_QUANTILE))
    info = {
        'inlier_fraction': float(inlier.mean()),
        'iterations': it + 1,
        'stationarity_pairs': int(n_pairs_used),
        'stationarity_enabled': bool(enable_stationarity),
        'mean_flexion_centered_rad': float(mean_flex),
    }
    return delta_0, alpha, c, mean_flex, info


def correct_take_flexion(in_path, out_path, segment=R_HIP,
                         smooth_window_s=SMOOTH_WINDOW_S,
                         lambda_stat=1.0, enable_stationarity=True,
                         fit_range=None, verbose=True):
    data = np.load(in_path, allow_pickle=True)
    poses = data['poses']; trans = data['trans']
    betas = np.asarray(data['betas']).flatten()
    gender = str(data['gender']).lower().strip().strip("'\"")
    fps = float(data['mocap_framerate']) if 'mocap_framerate' in data.files else 100.0
    T = poses.shape[0]

    mp = DEFAULT_MODEL_FEMALE if gender == 'female' else DEFAULT_MODEL_MALE
    with open(mp, 'rb') as f:
        model = pickle.load(f, encoding='latin1')
    parents = model['kintree_table'][0].astype(np.int64).copy()
    parents[0] = -1; parents = parents[:N_BODY]
    J_rest = shape_joints(model, betas)[:N_BODY]
    body_pose = poses[:, :N_BODY * 3].reshape(T, N_BODY, 3)

    if verbose:
        seg_name = {PELVIS: 'pelvis', L_HIP: 'l_hip', R_HIP: 'r_hip'}.get(
            segment, f'joint_{segment}')
        print(f'Take: {in_path.name}  T={T}  fps={fps}  gender={gender}  '
              f'segment={seg_name}  model=flexion-conditional rigid (6 coef)  '
              f'stationarity={"on" if enable_stationarity else "off"}')

    # FK in chunks
    chunk = 4000
    feet_local = np.empty((T, 2, 3))
    R_segment = np.empty((T, 3, 3))
    pelvis_local_z = np.empty(T)
    t_world_chunks = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        tw, Rw = fk_batch(J_rest, body_pose[s:e], parents)
        feet_local[s:e, 0] = tw[:, L_FOOT]
        feet_local[s:e, 1] = tw[:, R_FOOT]
        R_segment[s:e] = Rw[:, segment]
        pelvis_local_z[s:e] = tw[:, 0, 2]
        t_world_chunks.append(tw)
    t_world = np.concatenate(t_world_chunks, axis=0)

    feet_z = feet_local[:, :, 2] + trans[:, 2:3]
    pelvis_z = pelvis_local_z + trans[:, 2]
    flexion = compute_flexion(t_world, J_rest)

    if fit_range is not None:
        lo, hi = fit_range
        lo = max(0, lo); hi = min(T, hi)
        sl = slice(lo, hi)
    else:
        sl = slice(0, T)

    delta_0, alpha, c, mean_flex, info = fit_flexion_conditional(
        R_segment[sl], feet_z[sl], feet_local[sl], pelvis_z[sl], trans[sl],
        flexion[sl], fps, lambda_stat=lambda_stat,
        enable_stationarity=enable_stationarity, verbose=verbose,
    )

    # Apply correction to ALL frames using the fit's centered flexion
    f_c = flexion - mean_flex
    shift = (R_segment * (delta_0 + alpha * f_c[:, None])[:, None, :]).sum(axis=2)
    # equivalent: shift[t] = R_segment[t] @ (delta_0 + alpha * f_c[t])
    shift = smooth_shift(shift, fps, window_s=smooth_window_s)
    trans_corr = trans - shift
    trans_corr[:, 2] -= c

    feet_corr_z = feet_local[:, :, 2] + trans_corr[:, 2:3]
    lower_corr = feet_corr_z.min(axis=1)
    inlier = np.abs(pelvis_z - np.median(pelvis_z)) < INLIER_PELVIS_BAND
    if verbose:
        print(f'\n  delta_0 = {delta_0}  ||δ_0||={np.linalg.norm(delta_0):.4f} m')
        print(f'  alpha   = {alpha}  ||α||={np.linalg.norm(alpha):.4f} m/rad')
        print(f'  c       = {c:+.4f}')
        print(f'  mean flexion centered at: {np.degrees(mean_flex):.2f} deg')
        print(f'  stationarity_pairs={info["stationarity_pairs"]}  '
              f'iters={info["iterations"]}')
        print(f'  lower_z (inliers) min/med/max: '
              f'{lower_corr[inlier].min():+.3f} / {np.median(lower_corr[inlier]):+.3f} / '
              f'{lower_corr[inlier].max():+.3f}')

    out = {k: data[k] for k in data.files}
    out['trans'] = trans_corr
    out['_belt_offset_delta'] = delta_0           # legacy key holds δ_0
    out['_belt_offset_alpha'] = alpha
    out['_belt_offset_mean_flexion_rad'] = np.array(mean_flex)
    out['_belt_offset_floor_c'] = np.array(c)
    out['_belt_offset_segment'] = np.array(segment)
    out['_belt_offset_full_model'] = np.array(False)
    out['_belt_offset_flexion_conditional'] = np.array(True)
    out['_belt_offset_stationarity_pairs'] = np.array(info['stationarity_pairs'])
    out['_belt_offset_smooth_window_s'] = np.array(smooth_window_s)
    np.savez(out_path, **out)
    if verbose:
        print(f'\nSaved: {out_path}')
    return {'delta_0': delta_0, 'alpha': alpha, 'c': c, **info}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input')
    p.add_argument('output', nargs='?', default=None)
    p.add_argument('--segment', default='rhip',
                   help='pelvis | lhip | rhip | auto[:s1,s2,...]')
    p.add_argument('--no-stationarity', action='store_true')
    p.add_argument('--stationarity-weight', type=float, default=1.0,
                   dest='lambda_stat')
    p.add_argument('--smooth', type=float, default=SMOOTH_WINDOW_S)
    p.add_argument('--fit-range', default=None)
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

    suffix = f'_{seg_label}flex{"stat" if not args.no_stationarity else "nostat"}'
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else \
               in_path.with_name(in_path.stem + suffix + '.npz')

    correct_take_flexion(
        in_path, out_path, segment=seg,
        smooth_window_s=args.smooth,
        lambda_stat=args.lambda_stat,
        enable_stationarity=not args.no_stationarity,
        fit_range=fit_range,
    )


if __name__ == '__main__':
    main()
