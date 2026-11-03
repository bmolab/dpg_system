"""Arms overhead: like hanging arms, the upper arm (and the sensor's long axis) is vertical.
Shared sideways shift of both upper arms in the chest's front plane, measured from
straight UP (+ toward the dancer's left), for frames with both arms >50 deg above level at
similar heights; next to the same session's hanging bank."""
import sys, glob, hashlib, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST, as_pose_tensor
A = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/'
_, _, bax = skeleton(); bax = bax.double()
SESS = {'S1 (old)': 'Subject1_Waacking/pre/raw', 'S2 (old)': 'Subject2_Contemporary/pre/raw', 'S3 (old)': 'Subject3_Vogue/pre/raw',
        'S4 (old)': 'Subject4_Krumping/pre/raw', 'S5 (old)': 'Subject5_Cueca/pre/shadow', 'S6 (old)': 'Subject6_Contemporary/pre/raw',
        'S7 (old)': 'Subject7_Bharathanatyam/raw', 'S11 (new)': 'Subject11_Shaolin_Tai_Chi_60', 'S15 (new)': 'Subject15_Contortion',
        'S21 (new)': 'Subject21'}
print(f'{"session":10s} {"hanging bank":>14s} {"overhead shift":>16s}  frames   (overhead by height: 50-65 / 65-90 above)')
for s, d in SESS.items():
    hb, ob, oh, seen = [], [], [], set()
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
        sim = np.abs(v[0][:, 1] - v[1][:, 1]) < 0.2
        down = (v[0][:, 1] < -0.77) & (v[1][:, 1] < -0.77) & sim
        up = (v[0][:, 1] > 0.77) & (v[1][:, 1] > 0.77) & sim
        hb.append(((np.degrees(np.arctan2(v[0][:, 0], -v[0][:, 1])) + np.degrees(np.arctan2(v[1][:, 0], -v[1][:, 1]))) / 2)[down])
        # from straight up: a mirror-symmetric pair reads 0 here too
        ob.append(((np.degrees(np.arctan2(v[0][:, 0], v[0][:, 1])) + np.degrees(np.arctan2(v[1][:, 0], v[1][:, 1]))) / 2)[up])
        oh.append(np.degrees(np.arcsin(np.clip((v[0][:, 1] + v[1][:, 1]) / 2, -1, 1)))[up])
    hb, ob, oh = map(np.concatenate, (hb, ob, oh))
    split = ' / '.join(f'{np.median(ob[(oh >= lo) & (oh < hi)]):+5.1f}' if ((oh >= lo) & (oh < hi)).sum() >= 100 else '   . '
                       for lo, hi in ((50, 65), (65, 90)))
    print(f'{s:10s} {np.median(hb):+10.1f} ({len(hb)})  ' + (f'{np.median(ob):+10.1f}  {len(ob):7d}   ({split})' if len(ob) >= 100 else f'{"too few":>10s}  {len(ob):7d}'))
