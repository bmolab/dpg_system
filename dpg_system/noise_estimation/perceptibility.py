"""Off-trajectory contextual residual — the perceptibility/relevance measure.

For a flagged event (attributed joint + frame), how far the rendered surface of
that limb sits OFF its smooth trajectory: per-vertex deviation from a quadratic
fit of each vertex's position over neighbouring frames (which ABSORBS smooth
acceleration/curvature — the direction fit is baked into the prediction),
localized to the attributed joint's kinematic subtree, p90 over those vertices,
in cm.  Validated full-range: real glitches 2-16 cm, smooth-fast motion / clean
/ imperceptible 0.0-0.7 cm; ~1 cm separates them (0.7-1.5 = fuzzy margin).

ABSOLUTE (not activity-normalised — normalising over-corrects, false-flagging
low-activity smooth motion).  The contextual fit is already in the quadratic
prediction.

Needs the SMPL-H model for LBS; only called on the sparse set of flagged frames,
so it's a light pass.  Returns None if the model is unavailable (graceful
fallback — callers leave off_traj_cm unset).
"""

import os
import numpy as np

_MODEL_PATHS = [
    '/Users/drokeby/Dev/amass/support_data/body_models/smplh/SMPLH_NEUTRAL.npz',
    os.path.join(os.path.dirname(__file__), 'SMPLH_NEUTRAL.npz'),
]
_M = None          # cached model arrays
_SUBTREE = None    # cached per-joint subtree vertex masks
NJ = 52


def _load():
    global _M, _SUBTREE
    if _M is not None:
        return _M
    path = next((p for p in _MODEL_PATHS if os.path.exists(p)), None)
    if path is None:
        _M = False
        return _M
    m = np.load(path, allow_pickle=True)
    vt = m['v_template'].astype(np.float64)
    par = m['kintree_table'][0].astype(int).copy(); par[0] = -1
    W = np.asarray(m['weights'])
    _M = {
        'vt': vt,
        'sd': m['shapedirs'],
        'pd': m['posedirs'].reshape(m['posedirs'].shape[0], 3, -1),
        'Jreg': np.asarray(m['J_regressor']),
        'W': W,
        'par': par,
        'V': vt.shape[0],
        'domj': W.argmax(1),
    }
    # Per-joint subtree vertex INDICES + pre-sliced posedirs/weights, so LBS
    # skins only the limb's ~hundreds of vertices, not all 6890 (~20x faster —
    # the full-body skinning was the cause of the stalled corpus run).
    def subtree(j):
        s = {j}; ch = True
        while ch:
            ch = False
            for k in range(NJ):
                if par[k] in s and k not in s:
                    s.add(k); ch = True
        return s
    # Store only vertex INDICES (tiny); slice posedirs/weights on the fly per
    # call.  (Pre-slicing all 52 joints' posedir blocks blew worker memory.)
    _SUBTREE = {j: np.where(np.isin(_M['domj'], list(subtree(j))))[0]
                for j in range(NJ)}
    # Per-joint VERTEX subtree reach (m): max rest distance from the joint to
    # any vertex it skins (incl. hand/foot mesh).  Leverage arm for the cheap
    # off-traj UPPER-BOUND proxy: a glitch of dev rad swings the surface at most
    # dev*reach.  (Captures elbow->hand etc. that bone length misses.)
    Jrest = _M['Jreg'] @ vt
    _M['reach'] = np.array([
        float(np.linalg.norm(vt[_SUBTREE[j]] - Jrest[j], axis=1).max())
        if _SUBTREE[j].size else 0.0 for j in range(NJ)])
    return _M


from scipy.spatial.transform import Rotation
_EYE = np.eye(3)
# quadratic value-at-centre weights for neighbour offsets [-2,-1,1,2]
_QW = np.linalg.pinv(np.vstack([np.array([4., 1, 1, 4]),
                                np.array([-2., -1, 1, 2]), np.ones(4)]).T)[2]


def _global_transforms(pose, J, par):
    R = Rotation.from_rotvec(pose).as_matrix()
    G = np.zeros((NJ, 4, 4))
    for j in range(NJ):
        L = np.eye(4); L[:3, :3] = R[j]
        L[:3, 3] = J[j] if par[j] < 0 else (J[j] - J[par[j]])
        G[j] = L if par[j] < 0 else G[par[j]] @ L
    T = G.copy()
    for j in range(NJ):
        jr = np.array([*J[j], 0.0]); T[j, :, 3] = G[j, :, 3] - G[j] @ jr
    return R, T


def _lbs_sub(pose, J, vshaped_sub, pd_sub, W_sub, par):
    """Skin ONLY the subtree vertices (sliced posedirs/weights)."""
    pose = pose.copy(); pose[0] = 0.0       # body-local
    R, T = _global_transforms(pose, J, par)
    vp = vshaped_sub + (pd_sub @ (R[1:] - _EYE).reshape(-1))      # (n,3)
    Tv = np.einsum('vk,kij->vij', W_sub, T)                       # (n,4,4)
    n = vshaped_sub.shape[0]
    return np.einsum('vij,vj->vi', Tv,
                     np.concatenate([vp, np.ones((n, 1))], 1))[:, :3]


def proxy_cm(poses, frame):
    """Cheap leverage-aware UPPER BOUND on off_traj_cm at `frame` (no LBS).

    For each body joint, the geodesic deviation of its local rotation from the
    quadratic prediction (the same direction-fit residual off_traj measures),
    times that joint's vertex reach (the surface lever arm) — the most the
    rendered surface could be swung off-trajectory by that single joint.  The
    max over joints bounds the true mesh deviation from ABOVE (mesh <= proxy:
    skinning blend and child compensation only reduce it).  So `proxy < tol`
    guarantees `off_traj_cm < tol` — a SAFE short-circuit that skips the LBS
    with no risk of missing a perceptible glitch.  Returns None if model
    unavailable / frame too near the ends."""
    M = _load()
    if not M:
        return None
    P = np.asarray(poses, dtype=np.float64)
    if P.ndim == 2:
        P = P.reshape(P.shape[0], -1, 3)
    t = int(frame)
    if not (2 <= t < P.shape[0] - 2):
        return None
    P = P[:, :NJ, :]
    reach = M['reach']
    pred = (_QW[0] * P[t - 2] + _QW[1] * P[t - 1]
            + _QW[2] * P[t + 1] + _QW[3] * P[t + 2])       # (NJ,3) axis-angle
    # body joints only (1..21); root has no parent rotation to deviate
    dev = (Rotation.from_rotvec(P[t, 1:22])
           * Rotation.from_rotvec(pred[1:22]).inv()).magnitude()   # rad
    return round(float((dev * reach[1:22]).max()) * 100.0, 3)


def off_traj_cm(poses, betas, frame, joint_idx):
    """Off-trajectory contextual residual (cm) for a flagged event at `frame`
    on `joint_idx` (the attributed peak).  None if model unavailable / frame too
    near the ends.  Skins only the attributed limb's subtree (fast)."""
    M = _load()
    if not M:
        return None
    P = np.asarray(poses, dtype=np.float64)
    if P.ndim == 2:
        P = P.reshape(P.shape[0], -1, 3)
    P = P[:, :NJ, :]
    t = int(frame)
    if not (2 <= t < P.shape[0] - 2):
        return None
    idx = _SUBTREE.get(int(joint_idx))
    if idx is None or idx.size == 0:
        return None
    b = np.zeros(16); bb = np.asarray(betas, dtype=np.float64).ravel()[:16]
    b[:len(bb)] = bb
    vshaped = M['vt'] + (M['sd'] @ b)
    J = M['Jreg'] @ vshaped
    vshaped_sub = vshaped[idx]; pd_sub = M['pd'][idx]; W_sub = M['W'][idx]
    par = M['par']
    Vs = np.stack([_lbs_sub(P[k], J, vshaped_sub, pd_sub, W_sub, par) for k in range(t - 2, t + 3)])
    pred = _QW[0] * Vs[0] + _QW[1] * Vs[1] + _QW[2] * Vs[3] + _QW[3] * Vs[4]
    resid = np.linalg.norm(Vs[2] - pred, axis=1) * 100.0    # cm per vertex
    return round(float(np.percentile(resid, 90)), 3)
