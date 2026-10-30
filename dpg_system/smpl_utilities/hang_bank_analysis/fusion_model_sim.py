"""Can magnetization in a sensor, through a fusion that lets the field pull on tilt (the
pre-2025-12-18 Shadow algorithm), produce the hang bank as measured: zero at the T-pose,
present with the arm lowered, not dependent on facing?
Sensor reads gravity (exact) and field = true field + constant offset b (sensor frame).
Fusion: the orientation that best aligns both with their references, field weighted k
against gravity (Wahba / SVD). k = 0 is a fusion whose tilt is gravity only.
Calibration: at a T-pose facing 0 the reported orientation is set to the true one.
Then the arm hangs (lowered 75 deg about the forward axis) at every facing; report the
sideways tilt error of the bone (the bank) and how much it varies with facing.
World: Y up, field 54.6 uT, dip 70 deg, north = +Z."""
import numpy as np
from scipy.spatial.transform import Rotation as R

E = 54.6 * np.array([0, -np.sin(np.radians(70)), np.cos(np.radians(70))])
UP = np.array([0, 1.0, 0])
BONE = np.array([1.0, 0, 0])          # left upper arm points +X in the T-pose


def fuse(Rtrue, b, k):
    a = Rtrue.inv().apply(UP)
    m = Rtrue.inv().apply(E) + b
    # Wahba: minimise sum w |R v_body - v_ref|^2 over unit vectors
    B = (1 - k) * np.outer(UP, a / np.linalg.norm(a)) + k * np.outer(E / np.linalg.norm(E), m / np.linalg.norm(m))
    U, _, Vt = np.linalg.svd(B)
    M = np.diag([1, 1, np.linalg.det(U) * np.linalg.det(Vt)])
    return R.from_matrix(U @ M @ Vt)


def bank_vs_facing(b, k, lower=-75.0):
    T = R.identity()
    offset = fuse(T, b, k).inv() * T
    out = []
    for face in np.arange(0, 360, 30):
        F = R.from_euler('y', face, degrees=True)
        true = F * R.from_euler('z', lower, degrees=True)
        rep = fuse(true, b, k) * offset
        # bone direction in the body (facing) frame
        vt = (F.inv() * true).apply(BONE); vr = (F.inv() * rep).apply(BONE)
        side = lambda v: np.degrees(np.arctan2(v[0], -v[1]))
        out.append(side(vr) - side(vt))
    tp = []
    for face in (0, 90, 180, 270):
        F = R.from_euler('y', face, degrees=True)
        vr = (F.inv() * (fuse(F, b, k) * offset)).apply(BONE)
        tp.append(np.degrees(np.arctan2(vr[1], vr[0])))
    return np.array(out), np.array(tp)


rng = np.random.default_rng(1)
print('per row: offset size, coupling k -> over 300 random offset directions:')
print('  median |bank| hanging, median facing swing (max-min over facings), median |T-pose error| at other facings')
for mag in (2, 5, 10, 20):
    for k in (0.0, 0.1, 0.3, 0.5):
        rows = []
        for _ in range(300):
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            o, tp = bank_vs_facing(mag * d, k)
            rows.append((abs(o.mean()), o.max() - o.min(), np.abs(tp).max()))
        r = np.array(rows)
        print(f'  {mag:3d} uT  k {k:.1f}:  bank {np.median(r[:, 0]):5.1f}  (90th pct {np.percentile(r[:, 0], 90):5.1f})   facing swing {np.median(r[:, 1]):5.1f}   T-pose {np.median(r[:, 2]):5.1f}')

print('\nwhich offset direction (sensor frame) gives the largest steady bank, 10 uT, k 0.3:')
best = []
for d in np.vstack([np.eye(3), -np.eye(3)]):
    o, tp = bank_vs_facing(10 * d, 0.3)
    best.append((d, o.mean(), o.max() - o.min(), np.abs(tp).max()))
for d, m, sw, tp in best:
    print(f'  offset along {d.astype(int)}: bank {m:+6.1f}, facing swing {sw:5.1f}, T-pose at other facings {tp:4.1f}')
