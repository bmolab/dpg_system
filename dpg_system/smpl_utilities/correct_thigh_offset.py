"""Correct SMPL `trans` for an offset error in the (waist-belt) IMU mount.

Background
----------
The IMU mocap solver back-propagates from the belt-mounted sensor's world
pose to the pelvis.  If the assumed sensor offset (in the local frame of the
body segment the sensor sits on) differs from the actual offset by a fixed
3-vector `delta`, then for every frame

    trans_recorded[t] = trans_true[t] + R_segment_world[t] @ delta

So the fix is

    trans_corrected[t] = trans_recorded[t] - R_segment_world[t] @ delta - [0,0,c]

where the extra constant `c` re-zeros the floor (the recorded data carries an
arbitrary world-Z origin; we want the lower foot to sit near z=0 when planted).

For a waist belt the relevant segment is the *pelvis* (default).  For a
thigh strap, switch to the left-hip joint.  The correct choice is the one
whose rotation explains the foot-height drift most cleanly — empirically a
pelvis (waist-belt) sensor produces a much tighter floor than a thigh fit.

Algorithm
---------
1. SMPL forward kinematics with the file's betas → world rotation of the left
   hip and the model-frame Z of l_foot, r_foot, pelvis (per frame).
2. Build an inlier mask that drops sensor dropouts (frames with wildly out-of-
   range pelvis_z relative to the median).
3. Iterate, alternating between
     a) selecting "planted" foot frames — feet whose CORRECTED z falls in the
        bottom band AND whose vertical velocity is small (classic foot-on-
        floor heuristic) — and
     b) least-squares re-fitting `lower-foot-z = R_lhip[2,:] @ delta + c` on
        those planted (frame, foot) pairs.
   The loop stabilises within a few iterations and converges to a delta that
   captures only the rotation-coupled artefact (no contamination from frames
   where the foot is genuinely lifted).
4. Pick the floor offset c so the lowest 0.5% of corrected lower-foot z on
   inliers sits at z=0.
5. Apply  trans_corrected = trans - R_lhip_world @ delta - [0,0,c]  to ALL
   frames (the dropouts still get the same per-frame correction; their
   residual artefacts come from the recorded sensor itself).

The corrected `.npz` keeps every original field (poses, betas, dmpls, fps,
gender, …) and overwrites `trans`.  The fitted parameters are stashed under
keys prefixed with `_thigh_offset_*` for auditability.

Usage:
    python correct_thigh_offset.py <input.npz> [<output.npz>]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pickle

DEFAULT_MODEL_FEMALE = '/Users/drokeby/dpg_system/dpg_system/smplh/SMPLH_FEMALE.pkl'
DEFAULT_MODEL_MALE = '/Users/drokeby/dpg_system/dpg_system/smplh/SMPLH_MALE.pkl'

PELVIS, L_HIP, R_HIP, L_FOOT, R_FOOT = 0, 1, 2, 10, 11
N_BODY = 22

# Which segment's world rotation is used as the regressor for the trans
# error.  Physically the IMU is on the LEFT THIGH (L_HIP), but the dominant
# error in this dataset's `trans` correlates with PELVIS rotation — that's
# probably because the mocap solver's back-propagation chain from the thigh
# sensor to the pelvis crosses a hip-joint offset whose location in the
# pelvis frame was assumed (and slightly wrong).  See the README block for
# the discussion.  Use --segment lhip to fit the thigh model instead.
DEFAULT_SEGMENT = L_HIP

# --- knobs (only touch if needed) ---
INLIER_PELVIS_BAND = 0.6   # m;  reject frames whose pelvis_z deviates more
PLANTED_VEL_MAX = 0.20     # m/s; foot vertical velocity cutoff
PLANTED_BAND = 0.03        # m;   tolerance above per-foot lower-band quantile
PLANTED_QUANTILE = 0.05    # bottom 5% of per-foot corrected z is "planted"
N_ITERS = 12
FLOOR_QUANTILE = 0.005     # 0.5%-ile of corrected lower-foot z -> z=0
OUTLIER_K = 3.0            # drop planted pairs whose |residual - median| > K * MAD
OUTLIER_INNER_ITERS = 4    # rounds of inner trim+refit per outer iteration
RIDGE_LAMBDA = 1e-3        # tiny L2 penalty on coefs (units of m^2 per coef^2)
                            # — keeps the 9-coef fit from running away on takes
                            # with poor planted-foot diversity

# The shift vector R_segment @ delta inherits frame-to-frame jitter from the
# recorded segment rotation.  Smooth it temporally before subtracting from
# trans.  Savitzky-Golay preserves slow trends.  Window length is in seconds.
# 95% of the artifact's spectral energy is typically below ~0.6 Hz; a 0.5 s
# window with cubic polynomial gives an effective cutoff well above the real
# signal while killing per-frame jitter.
SMOOTH_WINDOW_S = 0.50     # 0.50 s smoothing window (~50 frames at 100 fps)
SMOOTH_POLYORDER = 3       # cubic polynomial inside the window


def rodrigues(rotvecs: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvecs, axis=-1, keepdims=True)
    safe = np.where(theta < 1e-12, 1.0, theta)
    k = rotvecs / safe
    K = np.zeros(rotvecs.shape[:-1] + (3, 3))
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]; K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    s = np.sin(theta)[..., None]; c = np.cos(theta)[..., None]
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    return np.where(theta[..., None] < 1e-12, np.eye(3), R)


def shape_joints(model: dict, betas: np.ndarray) -> np.ndarray:
    sd = model['shapedirs'][:, :, :len(betas)]
    return model['J_regressor'] @ (model['v_template'] + sd @ betas)


def fk_batch(J_rest: np.ndarray, body_pose: np.ndarray, parents: np.ndarray):
    """Vectorised SMPL forward kinematics.  body_pose: (T, J, 3) axis-angle.

    Returns (t_world, R_world): joint positions and world rotations in the
    *model* frame (pelvis stays at J_rest[0]; add `trans` for world space).
    """
    T, J, _ = body_pose.shape
    R_local = rodrigues(body_pose)
    R_world = np.empty_like(R_local)
    t_world = np.empty((T, J, 3))
    R_world[:, 0] = R_local[:, 0]
    t_world[:, 0] = J_rest[0]
    for j in range(1, J):
        p = parents[j]
        R_world[:, j] = R_world[:, p] @ R_local[:, j]
        t_world[:, j] = t_world[:, p] + (R_world[:, p] @ (J_rest[j] - J_rest[p]))
    return t_world, R_world


def segment_diversity(R_segment: np.ndarray) -> float:
    """Measure how much the segment tilts across the take.

    Returns the std of bone_z = (−R_segment @ +ŷ_local)·ẑ_world.
    Standing posture has bone_z = −1 (bone points world −Z); a horizontal or
    raised segment moves bone_z toward 0 or +1.  High std = lots of tilt
    variation across frames = good leverage for the LS fit.
    """
    bone_z = (-R_segment @ np.array([0.0, 1.0, 0.0]))[:, 2]
    return float(bone_z.std())


def pick_best_segment(npz_path, candidates,
                      model_path_by_gender: dict | None = None,
                      preference_factor: float = 1.5,
                      verbose: bool = True) -> int:
    """Pick the segment from `candidates` with the most rotational diversity.

    Honours candidate order as user preference: the first candidate is used
    unless another candidate has at least `preference_factor`× its diversity.
    This avoids flipping segments across takes when the choice is close, while
    still falling back when the preferred segment is genuinely weak (e.g.,
    pelvis during a spin where the torso barely tilts).

    Does a mini SMPL FK on the file to compute world rotations of all
    candidate joints, then ranks by `segment_diversity`.
    """
    if model_path_by_gender is None:
        model_path_by_gender = {
            'female': DEFAULT_MODEL_FEMALE,
            'male': DEFAULT_MODEL_MALE,
        }
    data = np.load(npz_path, allow_pickle=True)
    poses = data['poses']
    betas = data['betas']
    gender = str(data['gender']).lower().strip()
    T = poses.shape[0]

    model_path = model_path_by_gender.get(gender, DEFAULT_MODEL_FEMALE)
    with open(model_path, 'rb') as f:
        model = pickle.load(f, encoding='latin1')
    parents = model['kintree_table'][0].astype(np.int64).copy()
    parents[0] = -1
    parents = parents[:N_BODY]
    J_rest = shape_joints(model, betas)[:N_BODY]
    body_pose = poses[:, :N_BODY * 3].reshape(T, N_BODY, 3)

    # FK in chunks; only need world rotations of candidate joints
    chunk = 4000
    R_cand = {c: np.empty((T, 3, 3)) for c in candidates}
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        _, Rw = fk_batch(J_rest, body_pose[s:e], parents)
        for c in candidates:
            R_cand[c][s:e] = Rw[:, c]
    diversities = {c: segment_diversity(R_cand[c]) for c in candidates}
    preferred = candidates[0]
    best = preferred
    pref_div = diversities[preferred]
    for c in candidates[1:]:
        if diversities[c] > preference_factor * pref_div and diversities[c] > diversities[best]:
            best = c
    if verbose:
        names = {PELVIS: 'pelvis', L_HIP: 'lhip', R_HIP: 'rhip'}
        rep = ', '.join(f'{names.get(c,str(c))}={diversities[c]:.3f}' for c in candidates)
        print(f'  diversity:  {rep}  →  picked {names.get(best,str(best))}')
    return best


def _ridge_fit(X: np.ndarray, y: np.ndarray, K: int, lam: float) -> np.ndarray:
    """Solve  min || X @ beta - y ||^2 + lam * || beta[:K] ||^2.
    The intercept (last column of X) is NOT penalised."""
    n, p = X.shape
    A = X.T @ X
    b = X.T @ y
    if lam > 0:
        I = np.eye(p)
        I[-1, -1] = 0.0           # don't penalise the intercept
        A = A + lam * I
    return np.linalg.solve(A, b)


def fit_segment_offset(R_segment: np.ndarray,
                       feet_z: np.ndarray,
                       pelvis_z: np.ndarray,
                       fps: float,
                       full: bool = False,
                       verbose: bool = True) -> tuple[np.ndarray, float, dict]:
    """Run the iterative planted-foot fit.

    R_segment: (T, 3, 3) world rotation of the sensor-bearing segment per frame
    feet_z:    (T, 2)    world Z of (l_foot, r_foot) per frame, recorded
    pelvis_z:  (T,)      world Z of pelvis per frame, recorded
    full:      if True, regress shift_z against ALL 9 entries of R_segment
               (captures pose-dependent strap slip / axial sensor rotation).
               If False, the rigid-strap model: shift_z = R_segment[2,:] @ delta.
    Returns: coefs (3 or 9,), c (float), info dict.

    The returned `coefs` is interpreted as:
       shift_z[t] = coefs @ predictors[t]
       where predictors[t] = R_segment[t][2,:]   (full=False, 3 coefs)
                          or R_segment[t].flatten() (full=True, 9 coefs)
    """
    T = feet_z.shape[0]
    if full:
        u = R_segment.reshape(T, 9)                     # (T, 9)
    else:
        u = R_segment[:, 2, :]                          # (T, 3)
    lower = feet_z.min(axis=1)

    # Inlier mask: drop sensor dropouts
    pz_med = np.median(pelvis_z)
    inlier = np.abs(pelvis_z - pz_med) < INLIER_PELVIS_BAND
    if verbose:
        print(f'  inliers: {inlier.sum()}/{T}  ({100*inlier.sum()/T:.2f}%)  '
              f'pelvis_z med={pz_med:+.3f}')

    # Initial LS on lower-foot z over inliers
    K = u.shape[1]                                       # 3 or 9
    X = np.concatenate([u, np.ones((T, 1))], axis=1)
    beta, *_ = np.linalg.lstsq(X[inlier], lower[inlier], rcond=None)
    delta = beta[:K]
    if verbose:
        print(f'  init coefs (k={K}) = {delta}')

    # Per-foot vertical velocity from recorded feet_z (central differences)
    vel = np.zeros_like(feet_z)
    if T >= 3:
        vel[1:-1] = np.abs(feet_z[2:] - feet_z[:-2]) * 0.5 * fps
        vel[0] = vel[1]; vel[-1] = vel[-2]

    # Iterate: pick planted (frame, foot) pairs, refit
    for it in range(N_ITERS):
        rot = u @ delta
        foot_corr = feet_z - rot[:, None]   # (T, 2) — c not yet applied
        planted = np.zeros((T, 2), dtype=bool)
        for f in range(2):
            thresh = np.quantile(foot_corr[inlier, f], PLANTED_QUANTILE)
            planted[:, f] = (foot_corr[:, f] < thresh + PLANTED_BAND) \
                            & (vel[:, f] < PLANTED_VEL_MAX) & inlier
        rows_X, rows_y = [], []
        for f in range(2):
            m = planted[:, f]
            if m.sum() == 0:
                continue
            rows_X.append(np.concatenate([u[m], np.ones((m.sum(), 1))], axis=1))
            rows_y.append(feet_z[m, f])
        if not rows_X:
            raise RuntimeError('No planted-foot frames found; cannot fit.')
        Xp = np.concatenate(rows_X, axis=0)
        yp = np.concatenate(rows_y, axis=0)

        # Ridge-regularised LS that also iteratively trims outliers.  Each
        # inner round drops planted pairs whose residual is far from the
        # median.  This rescues takes where a few false-planted frames pull
        # the fit toward absurd coefficients.
        n_in_iter = Xp.shape[0]
        n_kept = n_in_iter
        beta = _ridge_fit(Xp, yp, K, RIDGE_LAMBDA)
        for _inner in range(OUTLIER_INNER_ITERS):
            r = Xp @ beta - yp
            r_med = np.median(r)
            mad = 1.4826 * np.median(np.abs(r - r_med))
            if mad < 1e-6:
                break
            keep = np.abs(r - r_med) < OUTLIER_K * mad
            if keep.sum() < max(K + 5, 0.5 * n_kept):
                break
            if keep.sum() == n_kept:
                break
            Xp = Xp[keep]; yp = yp[keep]
            n_kept = Xp.shape[0]
            beta = _ridge_fit(Xp, yp, K, RIDGE_LAMBDA)

        delta_new, c_new = beta[:K], float(beta[K])
        diff = float(np.linalg.norm(delta_new - delta))
        delta = delta_new
        if verbose:
            print(f'  iter {it:2d}: planted_pairs={Xp.shape[0]:5d}  '
                  f'coefs={delta}  c={c_new:+.4f}  diff={diff:.5f}')
        if verbose:
            print(f'    (kept {n_kept}/{n_in_iter} after outlier trim)')
        if diff < 1e-4:
            break

    # Final c chosen on lower-foot floor quantile over inliers
    rot = u @ delta
    pre_corr = lower - rot
    c = float(np.quantile(pre_corr[inlier], FLOOR_QUANTILE))
    info = {
        'inlier_fraction': float(inlier.mean()),
        'iterations': it + 1,
        'planted_pairs_final': int(Xp.shape[0]),
    }
    return delta, c, info


def smooth_shift(shift: np.ndarray, fps: float,
                 window_s: float = SMOOTH_WINDOW_S,
                 polyorder: int = SMOOTH_POLYORDER) -> np.ndarray:
    """Temporally smooth the per-frame correction vector with Savitzky-Golay.

    shift: (T, 3).  Returns a smoothed (T, 3) array of the same shape.
    Set window_s = 0 to disable smoothing.
    """
    if window_s <= 0:
        return shift
    win = max(int(round(window_s * fps)), polyorder + 2)
    if win % 2 == 0:
        win += 1                   # savgol needs odd window
    if win >= shift.shape[0]:
        return shift               # too short to smooth, leave alone
    from scipy.signal import savgol_filter
    return savgol_filter(shift, win, polyorder, axis=0, mode='nearest')


def correct_take(in_path: Path, out_path: Path,
                 segment: int = DEFAULT_SEGMENT,
                 smooth_window_s: float = SMOOTH_WINDOW_S,
                 full: bool = False,
                 fit_range: tuple[int, int] | None = None,
                 model_path_by_gender: dict | None = None,
                 verbose: bool = True) -> dict:
    if model_path_by_gender is None:
        model_path_by_gender = {
            'female': DEFAULT_MODEL_FEMALE,
            'male': DEFAULT_MODEL_MALE,
        }
    data = np.load(in_path, allow_pickle=True)
    poses = data['poses']
    trans = data['trans']
    betas = data['betas']
    gender = str(data['gender']).lower().strip()
    fps = float(data['mocap_framerate']) if 'mocap_framerate' in data.files else 100.0
    T = poses.shape[0]
    assert poses.shape[1] >= N_BODY * 3, f'expected >= {N_BODY*3} pose params, got {poses.shape[1]}'

    model_path = model_path_by_gender.get(gender, DEFAULT_MODEL_FEMALE)
    with open(model_path, 'rb') as f:
        model = pickle.load(f, encoding='latin1')
    parents = model['kintree_table'][0].astype(np.int64).copy()
    parents[0] = -1
    parents = parents[:N_BODY]
    J_rest = shape_joints(model, betas)[:N_BODY]
    body_pose = poses[:, :N_BODY * 3].reshape(T, N_BODY, 3)

    if verbose:
        print(f'Take: {in_path.name}  T={T}  fps={fps}  gender={gender}')
        print(f'  betas={betas}')

    # FK in chunks
    chunk = 4000
    feet_local_z = np.empty((T, 2))
    R_segment = np.empty((T, 3, 3))
    pelvis_local_z = np.empty(T)
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        tw, Rw = fk_batch(J_rest, body_pose[s:e], parents)
        feet_local_z[s:e, 0] = tw[:, L_FOOT, 2]
        feet_local_z[s:e, 1] = tw[:, R_FOOT, 2]
        R_segment[s:e] = Rw[:, segment]
        pelvis_local_z[s:e] = tw[:, 0, 2]

    feet_z = feet_local_z + trans[:, 2:3]
    pelvis_z = pelvis_local_z + trans[:, 2]
    lower = feet_z.min(axis=1)

    if verbose:
        seg_name = {PELVIS: 'pelvis', L_HIP: 'l_hip', R_HIP: 'r_hip'}.get(segment, f'joint_{segment}')
        print(f'  sensor segment: {seg_name}   model: {"full (9 coefs)" if full else "rigid (3 coefs)"}')

    # Optionally restrict the FIT to a subrange of frames (the correction is
    # still applied to all frames).  Use this to exclude bad sensor sections.
    if fit_range is not None:
        lo_fit, hi_fit = fit_range
        lo_fit = max(0, lo_fit); hi_fit = min(T, hi_fit)
        if verbose:
            print(f'  fit_range: frames [{lo_fit}, {hi_fit}) — fitting on '
                  f'{hi_fit - lo_fit}/{T} frames')
        R_fit = R_segment[lo_fit:hi_fit]
        feet_z_fit = feet_z[lo_fit:hi_fit]
        pelvis_z_fit = pelvis_z[lo_fit:hi_fit]
    else:
        R_fit = R_segment
        feet_z_fit = feet_z
        pelvis_z_fit = pelvis_z

    delta, c, info = fit_segment_offset(R_fit, feet_z_fit, pelvis_z_fit, fps,
                                        full=full, verbose=verbose)

    # Apply correction.  Smooth the shift vector to suppress per-frame jitter
    # in the recorded segment rotation; the underlying artifact is slow.
    if full:
        # shift_z is a linear combination of all 9 entries of R_segment.
        # We don't fit shift_x/shift_y; leave trans_x, trans_y unchanged.
        shift_z_raw = R_segment.reshape(T, 9) @ delta       # (T,)
        shift_raw = np.zeros((T, 3))
        shift_raw[:, 2] = shift_z_raw
    else:
        shift_raw = R_segment @ delta
    shift = smooth_shift(shift_raw, fps, window_s=smooth_window_s)
    if verbose and smooth_window_s > 0:
        win_frames = int(round(smooth_window_s * fps))
        if win_frames % 2 == 0:
            win_frames += 1
        rms_jitter = float(np.sqrt(np.mean((shift_raw - shift) ** 2)))
        print(f'  smooth window: {smooth_window_s:.2f} s ({win_frames} frames), '
              f'savgol order {SMOOTH_POLYORDER};  rms jitter removed = {rms_jitter*1000:.1f} mm')
    trans_corr = trans - shift
    trans_corr[:, 2] -= c

    feet_corr = feet_local_z + trans_corr[:, 2:3]
    pelvis_corr = pelvis_local_z + trans_corr[:, 2]
    lower_corr = feet_corr.min(axis=1)

    inlier = np.abs(pelvis_z - np.median(pelvis_z)) < INLIER_PELVIS_BAND
    if verbose:
        if full:
            print(f'\n  coefs (k=9) = {delta}')
        else:
            print(f'\n  delta = {delta}  ||delta||={np.linalg.norm(delta):.4f} m')
        print(f'  c     = {c:+.4f}')
        print(f'  inlier_fraction={info["inlier_fraction"]:.4f}  iters={info["iterations"]}')
        print('\n  --- BEFORE (all frames) ---')
        print(f'  pelvis_z  min/med/max: {pelvis_z.min():+.3f} / {np.median(pelvis_z):+.3f} / {pelvis_z.max():+.3f}')
        print(f'  l_foot_z  min/med/max: {feet_z[:,0].min():+.3f} / {np.median(feet_z[:,0]):+.3f} / {feet_z[:,0].max():+.3f}')
        print(f'  r_foot_z  min/med/max: {feet_z[:,1].min():+.3f} / {np.median(feet_z[:,1]):+.3f} / {feet_z[:,1].max():+.3f}')
        print(f'  lower_z   min/med/max: {lower.min():+.3f} / {np.median(lower):+.3f} / {lower.max():+.3f}')
        print('\n  --- AFTER  (inliers) ---')
        print(f'  pelvis_z  min/med/max: {pelvis_corr[inlier].min():+.3f} / {np.median(pelvis_corr[inlier]):+.3f} / {pelvis_corr[inlier].max():+.3f}')
        print(f'  l_foot_z  min/med/max: {feet_corr[inlier,0].min():+.3f} / {np.median(feet_corr[inlier,0]):+.3f} / {feet_corr[inlier,0].max():+.3f}')
        print(f'  r_foot_z  min/med/max: {feet_corr[inlier,1].min():+.3f} / {np.median(feet_corr[inlier,1]):+.3f} / {feet_corr[inlier,1].max():+.3f}')
        print(f'  lower_z   min/med/max: {lower_corr[inlier].min():+.3f} / {np.median(lower_corr[inlier]):+.3f} / {lower_corr[inlier].max():+.3f}')
        print(f'  fraction lower_z < -0.02 m: {(lower_corr[inlier]<-0.02).mean():.3%}')
        print(f'  fraction lower_z < -0.10 m: {(lower_corr[inlier]<-0.10).mean():.3%}')
        print('\n  --- AFTER  (all frames, incl. dropouts) ---')
        print(f'  lower_z   min/med/max: {lower_corr.min():+.3f} / {np.median(lower_corr):+.3f} / {lower_corr.max():+.3f}')

    out = {k: data[k] for k in data.files}
    out['trans'] = trans_corr
    out['_belt_offset_delta'] = delta
    out['_belt_offset_floor_c'] = np.array(c)
    out['_belt_offset_segment'] = np.array(segment)
    out['_belt_offset_full_model'] = np.array(bool(full))
    out['_belt_offset_inlier_fraction'] = np.array(info['inlier_fraction'])
    out['_belt_offset_smooth_window_s'] = np.array(smooth_window_s)
    np.savez(out_path, **out)
    if verbose:
        print(f'\nSaved corrected file: {out_path}')

    return {'delta': delta, 'c': c, **info}


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input')
    p.add_argument('output', nargs='?', default=None)
    p.add_argument('--segment', default='lhip',
                   help='Which segment\'s world rotation to use as the '
                        'regressor: pelvis | lhip | rhip | auto[:s1,s2,...].  '
                        'auto picks the segment with most rotational '
                        'diversity in this file (default candidates: '
                        'pelvis,lhip,rhip).  Constrain candidates with e.g. '
                        '--segment auto:pelvis,rhip.')
    p.add_argument('--smooth', type=float, default=SMOOTH_WINDOW_S,
                   help=f'Savitzky-Golay smoothing window in seconds for the '
                        f'shift vector before applying.  Set to 0 to disable. '
                        f'Default: {SMOOTH_WINDOW_S}.')
    p.add_argument('--full', action='store_true',
                   help='Use the full 9-coefficient model on the segment '
                        'rotation matrix (captures axial-strap-slip / pose-'
                        'dependent sensor mounting).  Otherwise the rigid-'
                        'strap 3-coefficient model is used.')
    p.add_argument('--fit-range', type=str, default=None,
                   help='Frame range LO:HI to use for fitting (correction '
                        'still applies to all frames).  Use this to exclude '
                        'bad sensor sections, e.g. --fit-range 0:10000.')
    args = p.parse_args()
    fit_range = None
    if args.fit_range:
        lo, hi = args.fit_range.split(':')
        fit_range = (int(lo), int(hi))
    in_path = Path(args.input)
    seg_table = {'lhip': L_HIP, 'rhip': R_HIP, 'pelvis': PELVIS}
    if args.segment.startswith('auto'):
        cand_str = (args.segment.split(':', 1)[1]
                    if ':' in args.segment else 'pelvis,lhip,rhip')
        cands = [seg_table[s.strip()] for s in cand_str.split(',')]
        print(f'Auto-picking segment from {cand_str}')
        seg = pick_best_segment(in_path, cands)
        seg_label = {L_HIP: 'lhip', R_HIP: 'rhip', PELVIS: 'pelvis'}[seg]
    else:
        seg = seg_table[args.segment]
        seg_label = args.segment
    suffix = f'_{seg_label}{"full" if args.full else ""}fix'
    out_path = Path(args.output) if args.output else \
        in_path.with_name(in_path.stem + suffix + '.npz')
    correct_take(in_path, out_path, segment=seg,
                 smooth_window_s=args.smooth, full=args.full,
                 fit_range=fit_range)


if __name__ == '__main__':
    main()
