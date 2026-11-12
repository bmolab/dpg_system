"""Which estimate of a session's fully-lowered bank: mean, median, np.robust_mean (mean
of samples within 3 MADs of the median, as the node computes it), the pseudo-Huber
estimate hang_symmetry's loss settles on (width 4 deg), or the half-sample mode (the
densest point).  Frames: both upper arms >60 deg below level at similar heights.
Stability: resample whole TAKES with replacement (frames within a take are not
independent) and report each estimate's spread; also its distance between the
session's odd-numbered and even-numbered takes."""
import sys, glob, hashlib, numpy as np, torch
from scipy.stats import median_abs_deviation
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST, as_pose_tensor
A = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/'
_, _, bax = skeleton(); bax = bax.double()
OLD = {'S1': 'Subject1_Waacking/pre/raw', 'S2': 'Subject2_Contemporary/pre/raw', 'S3': 'Subject3_Vogue/pre/raw',
       'S4': 'Subject4_Krumping/pre/raw', 'S5': 'Subject5_Cueca/pre/shadow', 'S6': 'Subject6_Contemporary/pre/raw',
       'S7': 'Subject7_Bharathanatyam/raw'}
rng = np.random.default_rng(0)


def robust_mean(x, t=3.0):
    med = np.median(x); mad = median_abs_deviation(x)
    return x[np.abs(x - med) < t * mad].mean() if mad > 0 else med


def huber(x, w=4.0):
    m = np.median(x)
    for _ in range(50):           # iteratively reweighted: pseudo-Huber weights 1/sqrt(1+(r/w)^2)
        r = x - m
        wt = 1 / np.sqrt(1 + (r / w) ** 2)
        m = np.sum(wt * x) / np.sum(wt)
    return m


def hsm(x):
    x = np.sort(x)
    while len(x) > 3:
        h = (len(x) + 1) // 2
        i = np.argmin(x[h - 1:] - x[:len(x) - h + 1])
        x = x[i:i + h]
    return x.mean()


EST = {'mean': np.mean, 'median': np.median, 'robust_mean': robust_mean, 'huber 4': huber, 'densest': hsm}


def takes(d):
    out, seen = [], set()
    for f in sorted(glob.glob(A + d + '/*.npz')):
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
        v = [rotate_vector_wxyz_t(conj(G[CHEST]), rotate_vector_wxyz_t(G[j], bax[j])).numpy() for j in (5, 19)]
        b = (np.degrees(np.arctan2(v[0][:, 0], -v[0][:, 1])) + np.degrees(np.arctan2(v[1][:, 0], -v[1][:, 1]))) / 2
        m = (v[0][:, 1] < -0.87) & (v[1][:, 1] < -0.87) & (np.abs(v[0][:, 1] - v[1][:, 1]) < 0.2)
        if m.sum() >= 60:
            out.append(b[m])
    return out


print('session estimates (deg) | spread when takes are resampled (sd) | odd-vs-even takes difference')
print(f'{"":4s} {"takes":>5s}  ' + ''.join(f'{n:>12s}' for n in EST) + '   skew')
tot = {n: [] for n in EST}
for s, d in OLD.items():
    T = takes(d)
    if len(T) < 3:
        continue
    allf = np.concatenate(T)
    est = {n: f(allf) for n, f in EST.items()}
    boot = {n: [] for n in EST}
    for _ in range(200):
        pick = np.concatenate([T[i] for i in rng.integers(0, len(T), len(T))])
        for n, f in EST.items():
            boot[n].append(f(pick))
    odd, even = np.concatenate(T[0::2]), np.concatenate(T[1::2])
    split = {n: abs(f(odd) - f(even)) for n, f in EST.items()}
    q = np.percentile(allf, [10, 50, 90])
    skew = (q[2] - q[1]) - (q[1] - q[0])
    print(f'{s:4s} {len(T):5d}  ' + ''.join(f'{est[n]:12.1f}' for n in EST) + f'   {skew:+5.1f}')
    print(f'{"":4s} {"sd":>5s}  ' + ''.join(f'{np.std(boot[n]):12.2f}' for n in EST))
    print(f'{"":4s} {"split":>5s}  ' + ''.join(f'{split[n]:12.2f}' for n in EST))
    for n in EST:
        tot[n].append((np.std(boot[n]), split[n]))
print('\nmedian over sessions:  ' + ''.join(f'{n:>12s}' for n in EST))
print(f'{"resample sd":>22s}' + ''.join(f'{np.median([t[0] for t in tot[n]]):12.2f}' for n in EST))
print(f'{"odd/even split":>22s}' + ''.join(f'{np.median([t[1] for t in tot[n]]):12.2f}' for n in EST))
