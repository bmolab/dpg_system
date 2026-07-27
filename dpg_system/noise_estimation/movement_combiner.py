"""Movement-vs-glitch combiner — the complementary-specialist classifier.

A single trajectory feature can't separate a real complex movement (hard throw,
floor contact, jumping-jack) from a mocap glitch: exhaustive testing showed every
single-event kinematic feature caps at ~0.79 AUC, because the two classes are
heterogeneous and overlap on any one axis.  What works is fusing complementary
*physical* specialists, each tuned to a different movement/glitch sub-type
(validated against a hand-adjudicated set, LOO-CV AUC 0.87):

  accom   multiscale fittability — the scale at which a filter bank accommodates
          the frame (sustained-complex motion vs spike).  The dominant signal.
  aspread 3D acceleration spread — physical force applied over frames vs impulse.
  rough   region disorder (reversal density) — catches stream/oscillation
          glitches that corrupt the trajectory baseline and are INVISIBLE to
          every deviation-from-trajectory metric (off_traj, iso read ~0 on them).
  step/off/iso/late/rev/act/burst  magnitude + shape + activity context.

Used as a MOVEMENT-RESCUE valve: a spike/excursion cluster whose off-trajectory
residual would fragment a clean segment is ABSORBED instead if the combiner is
confident it is real movement (movement_prob >= threshold).  The default
threshold is conservative (rescue only the clearest movements, ~0 glitch leak);
lower it to rescue more aggressively.  Rationale for erring toward leak on the
hard cases: the glitches the combiner can't catch are the subtle ones that take
many viewing angles to even see — i.e. imperceptible, hence safe to leave in.

Frozen model params live in movement_combiner_params.json (StandardScaler mean/
std + logistic coef/intercept).  Re-fit by re-running the freeze step as the
labeled set grows.  Returns None if the SMPL model or params are unavailable.
"""

import os
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import gaussian_filter1d

try:
    import perceptibility as _pc
except ImportError:                                  # package-qualified fallback
    from dpg_system.noise_estimation import perceptibility as _pc

# CASCADE tier-1 decisive thresholds: a single detector past these is louder
# than any real movement in the labeled set (calibrated above the most-extreme of
# 16 movements → 0 movement FP).  Any violation short-circuits the soft combiner:
# the event is a glitch, full stop — no weighted arbitration to water it down.
# Conservative (set by 16 movements); lower as the movement set grows.
# (sf / single-frame relevance is decisive too, but handled by the sfglitch HARD
# lens, so it isn't repeated here.)
_DECISIVE = {'off': 3.5, 'rough': 6.0, 'rev': 135.0}

_PARAMS = None
_M = None
_SIG = [0.6, 1.0, 1.5, 2.2, 3.2, 4.5, 6.5]


def _load():
    """Cache the frozen params and the SMPL kinematic data (parents, rest joints)."""
    global _PARAMS, _M
    if _PARAMS is None:
        p = os.path.join(os.path.dirname(__file__), 'movement_combiner_params.json')
        _PARAMS = json.load(open(p)) if os.path.exists(p) else False
    if _M is None:
        m = _pc._load()
        if not m:
            _M = False
        else:
            par = m['par']
            _M = {'par': par, 'Jrest': m['Jreg'] @ m['vt'],
                  'child': {j: [k for k in range(22) if par[k] == j] for j in range(22)}}
    return _PARAMS, _M


# ── per-frame kinematics ────────────────────────────────────────────────
def _av(P, t, j):
    return (Rotation.from_rotvec(P[t, j]) * Rotation.from_rotvec(P[t - 1, j]).inv()).as_rotvec()


def _event_joint(P, t):
    """The body joint with the largest single-frame angular acceleration near t
    (robust to upstream attribution picking the wrong joint/side)."""
    best = (0.0, 18)
    for j in range(1, 22):
        for tt in range(max(2, t - 2), min(P.shape[0] - 1, t + 3)):
            a = np.linalg.norm(_av(P, tt + 1, j) - _av(P, tt, j))
            if a > best[0]:
                best = (a, j)
    return best[1]


def _signals(P, tc, j, W=10):
    a = max(1, tc - W); b = min(P.shape[0] - 1, tc + W)
    avs = np.array([np.linalg.norm(_av(P, k, j)) for k in range(a, b + 1)])
    pk = avs.max() if avs.size else 1e-9
    return np.degrees(np.median(avs)), float(np.sum(avs >= 0.4 * pk)), np.degrees(pk)


def _iso(P, t, j, half=5):
    def loo(k):
        key = Rotation.concatenate([Rotation.from_rotvec(P[k - 1, j]), Rotation.from_rotvec(P[k + 1, j])])
        return float((Rotation.from_rotvec(P[k, j]) * Slerp([0, 2], key)([1])[0].inv()).magnitude())
    tb = max(range(max(2, t - 2), min(P.shape[0] - 2, t + 3)), key=loo)
    vals = {k: loo(k) for k in range(max(2, tb - half), min(P.shape[0] - 2, tb + half + 1))}
    o = sorted(v for k, v in vals.items() if k != tb); med = np.median(o) if o else 1e-9
    return vals[tb] / med if med > 1e-9 else 99.0


def _revscore(P, tc, half=3):
    best = 0.0
    for j in range(1, 22):
        for tt in range(max(2, tc - half), min(P.shape[0] - 1, tc + half + 1)):
            w1, w2 = _av(P, tt, j), _av(P, tt + 1, j); n1, n2 = np.linalg.norm(w1), np.linalg.norm(w2)
            if n1 > 1e-6 and n2 > 1e-6 and w1 @ w2 / (n1 * n2) < 0:
                best = max(best, np.degrees(n1) * np.degrees(n2) * (-(w1 @ w2 / (n1 * n2))))
    return best


def _roughness(P, tc, j, half=12):
    """Region disorder: density of significant direction-reversals in the window.
    Baseline-free — catches stream/oscillation glitches that corrupt the local
    trajectory and so read ~0 on every deviation metric."""
    n = 0
    for tt in range(max(2, tc - half), min(P.shape[0] - 1, tc + half)):
        w1, w2 = _av(P, tt, j), _av(P, tt + 1, j); a, b = np.linalg.norm(w1), np.linalg.norm(w2)
        if a > 1e-6 and b > 1e-6 and np.degrees(min(a, b)) > 1.0 and w1 @ w2 / (a * b) < -0.2:
            n += 1
    return float(n)


def _accom(P, tc, j, half=16):
    q = Rotation.from_rotvec(P[tc - half:tc + half + 1, j]).as_quat()
    for i in range(1, len(q)):
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    c = half; r = []
    for s in _SIG:
        sm = gaussian_filter1d(q, s, axis=0, mode='nearest'); sm /= np.linalg.norm(sm, axis=1, keepdims=True)
        r.append(np.degrees(2 * np.arccos(np.clip(abs(q[c] @ sm[c]), -1, 1))))
    r = np.array(r)
    return r[0] / (r.max() + 1e-6), (r[-1] - r[3]) / (r[3] + 1e-6)


def _gpos(pose, trans, par, Jrest):
    R = Rotation.from_rotvec(pose[:52]).as_matrix(); pos = np.zeros((52, 3)); G = [None] * 52
    for j in range(52):
        if par[j] < 0:
            pos[j] = Jrest[j]; G[j] = R[j]
        else:
            pos[j] = pos[par[j]] + G[par[j]] @ (Jrest[j] - Jrest[par[j]]); G[j] = G[par[j]] @ R[j]
    return pos - pos[0] + trans


_EXTREM = [20, 21, 7, 8, 10, 11, 15, 4, 5]


def _envelope(P, trans, tc, j, par, Jrest, child):
    d = j
    while child[d]:
        d = child[d][0]
    GP = np.array([_gpos(P[k], trans[k], par, Jrest) for k in range(tc - 8, tc + 9)])
    spd = np.linalg.norm(np.diff(GP[:, d, :], axis=0), axis=1); acc = np.abs(np.diff(spd))
    aspread = acc.sum() / (acc.max() + 1e-9)
    others = [x for x in _EXTREM if x != d]
    cmin = min(np.linalg.norm(GP[k, d, :] - GP[k, o, :]) for k in range(len(GP)) for o in others)
    return aspread, cmin


def movement_prob(poses, trans, frame, off_traj_cm):
    """Probability (0..1) that a flagged spike/excursion event at `frame` is real
    movement rather than a glitch.  `off_traj_cm` is the already-computed
    perceptibility residual for the cluster.  Returns None if the model/params
    are unavailable or the frame is too near the file ends."""
    params, MM = _load()
    if not params or not MM:
        return None
    P = np.asarray(poses, dtype=np.float64)
    if P.ndim == 2:
        P = P.reshape(P.shape[0], -1, 3)
    P = P[:, :52, :]
    t = int(frame)
    if not (26 <= t < P.shape[0] - 26):
        return None
    if trans is None:
        trans = np.zeros((P.shape[0], 3))
    par, Jrest, child = MM['par'], MM['Jrest'], MM['child']
    j = _event_joint(P, t)
    accs = {tt: np.linalg.norm(_av(P, tt + 1, j) - _av(P, tt, j)) for tt in range(t - 3, t + 4)}
    tc = max(accs, key=accs.get)
    acc, late = _accom(P, tc, j)
    aspread, cmin = _envelope(P, trans, tc, j, par, Jrest, child)
    actv, burst, step = _signals(P, tc, j)
    feats = {'off': float(off_traj_cm if off_traj_cm is not None and off_traj_cm >= 0 else 0.0),
             'accom': acc, 'late': late, 'aspread': aspread, 'cmin': cmin,
             'iso': _iso(P, t, j), 'rev': _revscore(P, tc),
             'act': actv, 'burst': burst, 'step': step, 'rough': _roughness(P, tc, j)}
    # CASCADE tier-1: a decisive violation on any one detector overrides the soft
    # combiner — definite glitch, never rescued.
    if (feats['off'] > _DECISIVE['off'] or feats['rough'] > _DECISIVE['rough']
            or feats['rev'] > _DECISIVE['rev']):
        return 0.0
    x = np.array([feats[f] for f in params['feature_names']])
    z = (x - np.array(params['mean'])) / np.array(params['std'])
    logit = float(z @ np.array(params['coef']) + params['intercept'])
    prob_glitch = 1.0 / (1.0 + np.exp(-logit))
    return round(1.0 - prob_glitch, 4)


def movement_threshold():
    params, _ = _load()
    return float(params['movement_threshold']) if params else 0.89
