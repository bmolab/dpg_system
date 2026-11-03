"""Which description of the upper arms' orientation predicts the hang bank? Per frame, each
upper arm's rotation away from its calibration orientation (relative to the chest), as a
rotation vector (axis x angle, chest frame: x the dancer's left, y up, z forward); the right
arm's is mirrored (x, -y, -z) so mirror-image arms match, and the two are averaged. Mirrored
fore/aft frames only (as bank_shape_unbiased.py). Candidate models, fit on even takes, scored
on odd takes, by height band:
  elevation ramp  - the hang term's ramp (25 to 55 deg below level) x a fitted size
  total angle     - a ramp of the total rotation angle (0 at 15 deg, full at 60) x a size
  rotation vector - linear in the three components (degrees)
  rotation vector, saturating - the same with each component limited by a smooth cap"""
import sys, glob, hashlib, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST, as_pose_tensor
from dpg_system.quaternion_nodes import quaternion_multiply_wxyz_t
A = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/'
_, _, bax = skeleton(); bax = bax.double()
SESSION = sys.argv[1] if len(sys.argv) > 1 else 'Subject7_Bharathanatyam/raw'


def rotvec(R):
    R = R * np.sign(R[:, :1] + 1e-12)
    s = np.linalg.norm(R[:, 1:], axis=1)
    ang = 2 * np.arctan2(s, R[:, 0])
    return R[:, 1:] / (s[:, None] + 1e-12) * np.degrees(ang)[:, None]


rows, seen = [], set()
for ti, f in enumerate(sorted(glob.glob(A + SESSION + '/*.npz'))):
    z = np.load(f)
    k = next((k for k in z.files if z[k].ndim == 3 and z[k].shape[1:] in ((20, 4), (37, 4))), None)
    if k is None:
        continue
    h = hashlib.md5(z[k][::50].tobytes()).hexdigest()
    if h in seen:
        continue
    seen.add(h)
    q, _ = as_pose_tensor(z[k])
    G = world_orientations(q, [CHEST, 5, 19])
    C = conj(G[CHEST])
    d = [rotate_vector_wxyz_t(C, rotate_vector_wxyz_t(G[j], bax[j])).numpy() for j in (5, 19)]
    rv = [rotvec(quaternion_multiply_wxyz_t(C, G[j]).numpy()) for j in (5, 19)]
    rv[1] = rv[1] * np.array([1, -1, -1])
    fwd = [np.degrees(np.arcsin(np.clip(a[:, 2], -1, 1))) for a in d]
    m = (np.abs(fwd[0] - fwd[1]) < 15) & (np.abs(fwd[0]) < 45) & (np.abs(fwd[1]) < 45)
    bank = (np.degrees(np.arctan2(d[0][:, 0], -d[0][:, 1])) + np.degrees(np.arctan2(d[1][:, 0], -d[1][:, 1]))) / 2
    height = np.degrees(np.arcsin(np.clip((d[0][:, 1] + d[1][:, 1]) / 2, -1, 1)))
    below = -np.stack([a[:, 1] for a in d], 1)                      # per arm, for the elevation ramp
    lo, hi = np.sin(np.radians(25)), np.sin(np.radians(55))
    x = np.clip((below - lo) / (hi - lo), 0, 1); elev_ramp = (x * x * (3 - 2 * x)).mean(1)
    tot = np.stack([np.linalg.norm(r, axis=1) for r in rv], 1)
    x = np.clip((tot - 15) / 45, 0, 1); tot_ramp = (x * x * (3 - 2 * x)).mean(1)
    v = (rv[0] + rv[1]) / 2
    rows.append(dict(bank=bank[m], height=height[m], elev=elev_ramp[m], tot=tot_ramp[m], v=v[m], ang=tot.mean(1)[m],
                     take=np.full(m.sum(), ti)))
D = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
even = D['take'] % 2 == 0
odd = ~even
y = D['bank']


def fit(X):
    # no constant: at the calibration orientation the bank is 0
    X1 = np.asarray(X)
    # robust: two passes of reweighting (pseudo-Huber, 5 deg)
    w = np.ones(even.sum())
    for _ in range(5):
        c, *_ = np.linalg.lstsq(X1[even] * w[:, None], y[even] * w, rcond=None)
        r = y[even] - X1[even] @ c
        w = 1 / np.sqrt(1 + (r / 5) ** 2) ** 0.5
    return X1 @ c, c


cap = lambda v, s: s * np.tanh(v / s)
models = {
    'elevation ramp': fit(D['elev'][:, None]),
    'total angle': fit(D['tot'][:, None]),
    'rotation vector': fit(D['v']),
    'rotation vector, saturating 45': fit(cap(D['v'], 45.0)),
    'rotation vector, saturating 30': fit(cap(D['v'], 30.0)),
}
bands = [(-90, -50), (-50, -20), (-20, 20), (20, 60), (60, 90)]
print(SESSION, ' frames', len(y), ' (fit on even takes, scored on odd)')
print('median bank, measured and predicted, odd takes, by mean arm height:')
print(f'{"":32s}' + ' '.join(f'{lo:+d}..{hi:+d}'.rjust(10) for lo, hi in bands) + '   median |error|')
print(f'{"measured":32s}' + ' '.join(f'{np.median(y[odd & (D["height"] >= lo) & (D["height"] < hi)]):+10.1f}'
                                     if (odd & (D["height"] >= lo) & (D["height"] < hi)).sum() > 50 else '.'.rjust(10)
                                     for lo, hi in bands))
for name, (p, c) in models.items():
    cells = []
    for lo, hi in bands:
        k = odd & (D['height'] >= lo) & (D['height'] < hi)
        cells.append(f'{np.median(p[k]):+10.1f}' if k.sum() > 50 else '.'.rjust(10))
    print(f'{name:32s}' + ' '.join(cells) + f'   {np.median(np.abs(y[odd] - p[odd])):6.2f}   coef {np.round(c, 3)}')

tb = [(0, 30), (30, 45), (45, 60), (60, 90), (90, 180)]
for hname, hk in (('below the shoulders (mean height under +20)', D['height'] < 20), ('raised (mean height +20 or more)', D['height'] >= 20)):
    print('\n' + hname + ', odd takes, by total rotation from calibration:')
    print(f'{"":32s}' + ' '.join(f'{lo}..{hi}'.rjust(10) for lo, hi in tb))
    cnt = [(odd & hk & (D['ang'] >= lo) & (D['ang'] < hi)).sum() for lo, hi in tb]
    print(f'{"frames":32s}' + ' '.join(f'{n:10d}' for n in cnt))
    for name, p in [('measured', y)] + [(n, m[0]) for n, m in models.items()]:
        cells = []
        for lo, hi in tb:
            k = odd & hk & (D['ang'] >= lo) & (D['ang'] < hi)
            cells.append(f'{np.median(p[k]):+10.1f}' if k.sum() > 40 else '.'.rjust(10))
        print(f'{name:32s}' + ' '.join(cells))
