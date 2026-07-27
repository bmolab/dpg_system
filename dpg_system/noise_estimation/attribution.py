"""Re-attribute a flagged event to the (region, joint, frame) where it originates.

Lenses flag a candidate joint/frame, but the *perceptible* event is often on a
different joint (a distal rotation glitch swings a limb that reads as proximal)
and a frame or two away (the flag and the visible motion don't always coincide).
This refines a flag to the (joint, frame) that maximises `rotation_deviation ×
bone_length` — the joint whose local rotation glitch swings the most limb — and
reports the LIMB REGION (which sidesteps the elbow-vs-shoulder ambiguity, which
is genuinely ill-posed) with the peak joint as a sub-detail.

Pure rotation math (geodesic deviation from a quadratic-predicted clean pose);
no SMPL model / LBS dependency, so it runs on the `--skip-torque` fast path.
"""

import numpy as np
from scipy.spatial.transform import Rotation

JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
]

# Limb regions (joint index -> region).  Collars go with the arm they attach to.
REGIONS = {
    'left_leg':  {1, 4, 7, 10},
    'right_leg': {2, 5, 8, 11},
    'spine':     {0, 3, 6, 9},
    'head':      {12, 15},
    'left_arm':  {13, 16, 18, 20},
    'right_arm': {14, 17, 19, 21},
}
JOINT_REGION = {j: r for r, js in REGIONS.items() for j in js}

# Per-joint leverage = bone length to the primary child (from SMPL-H rest
# skeleton); a rotation glitch swings limb proportional to this.  Tips floored.
BONELEN = [0.115, 0.377, 0.385, 0.135, 0.401, 0.401, 0.08, 0.134, 0.135, 0.218,
           0.08, 0.08, 0.083, 0.096, 0.102, 0.08, 0.261, 0.255, 0.249, 0.255,
           0.08, 0.08]

# quadratic-fit value-at-centre weights for neighbour offsets x=[-2,-1,1,2]
_QW = np.linalg.pinv(np.vstack([np.array([4., 1, 1, 4]), np.array([-2., -1, 1, 2]),
                                np.ones(4)]).T)[2]


def _poses22(poses):
    p = np.asarray(poses, dtype=np.float64)
    if p.ndim == 2:
        p = p.reshape(p.shape[0], -1, 3)
    return p[:, :22, :]


def attribute(poses, start, end=None, window=3):
    """Attribute a flagged event (frame `start`, or span `start..end`) to the
    (region, joint, frame) where its `rotation_deviation × bone_length` peaks.

    Returns {region, joint, joint_name, frame, score} (score in deg·m), or None
    if the span is too close to the file ends to evaluate.
    """
    P = _poses22(poses)
    T = P.shape[0]
    end = start if end is None else end
    best = None
    for t in range(start - window, end + window + 1):
        if not (2 <= t < T - 2):
            continue
        pred = (_QW[0] * P[t - 2] + _QW[1] * P[t - 1]
                + _QW[2] * P[t + 1] + _QW[3] * P[t + 2])      # (22,3) clean est.
        Ra = Rotation.from_rotvec(P[t]); Rp = Rotation.from_rotvec(pred)
        dev = (Ra * Rp.inv()).magnitude()                    # (22,) geodesic rad
        for j in range(1, 22):
            score = np.degrees(dev[j]) * BONELEN[j]
            if best is None or score > best[0]:
                best = (score, t, j)
    if best is None:
        return None
    score, t, j = best
    return {'region': JOINT_REGION.get(j, 'spine'), 'joint': j,
            'joint_name': JOINT_NAMES[j], 'frame': int(t),
            'score': round(float(score), 3)}
