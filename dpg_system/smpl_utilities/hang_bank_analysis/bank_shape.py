"""Shape of the hang bank against arm elevation, per session: is it one curve times a
per-session size?  Frames with the two upper arms at similar heights (within 0.2 in
height, as hang_symmetry); bank median per elevation band (degrees below level, chest
frame), then divided by the session's bank with the arms fully lowered (60+ below)."""
import sys, glob, hashlib, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST, as_pose_tensor
A = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/'
_, _, bax = skeleton(); bax = bax.double()
OLD = {'S1': 'Subject1_Waacking/pre/raw', 'S2': 'Subject2_Contemporary/pre/raw', 'S3': 'Subject3_Vogue/pre/raw',
       'S4': 'Subject4_Krumping/pre/raw', 'S5': 'Subject5_Cueca/pre/shadow', 'S6': 'Subject6_Contemporary/pre/raw',
       'S7': 'Subject7_Bharathanatyam/raw'}
BANDS = [(-30, -10), (-10, 5), (5, 20), (20, 35), (35, 50), (50, 60), (60, 90)]   # degrees below level
MIN = 150


def collect(d):
    B, H, seen = [], [], set()
    for f in glob.glob(A + d + '/*.npz'):
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
        side = [np.degrees(np.arctan2(a[:, 0], -a[:, 1])) for a in v]
        m = np.abs(v[0][:, 1] - v[1][:, 1]) < 0.2
        below = -np.degrees(np.arcsin(np.clip((v[0][:, 1] + v[1][:, 1]) / 2, -1, 1)))
        # side angle from straight down is not a bank for arms near level; use the angle in the
        # front plane measured from each arm's own outward direction instead
        B.append(((side[0] + side[1]) / 2)[m]); H.append(below[m])
    return np.concatenate(B), np.concatenate(H)


print('bank by elevation band (deg below level); then as a fraction of the 60+ band')
print(f'{"":4s} ' + ' '.join(f'{lo:+d}..{hi:+d}'.rjust(8) for lo, hi in BANDS))
fr = {}
for s, d in OLD.items():
    b, h = collect(d)
    med = []
    for lo, hi in BANDS:
        k = (h >= lo) & (h < hi)
        med.append(np.median(b[k]) if k.sum() >= MIN else np.nan)
    med = np.array(med)
    fr[s] = med / med[-1]
    print(f'{s:4s} ' + ' '.join(f'{x:8.1f}' if np.isfinite(x) else '       .' for x in med))
print('fraction of the fully-lowered bank:')
for s, f in fr.items():
    print(f'{s:4s} ' + ' '.join(f'{x:8.2f}' if np.isfinite(x) else '       .' for x in f))
