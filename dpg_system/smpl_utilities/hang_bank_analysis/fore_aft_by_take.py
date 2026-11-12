"""Right arm further forward/back than the left (degrees, + = right further forward), and
reading higher (+ = right higher), with the two upper arms at similar heights, per take and
per height band: above level / level / below. A suit error should hold across choreographies."""
import sys, glob, os, hashlib, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST, as_pose_tensor
A = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/'
_, _, bax = skeleton(); bax = bax.double()
SESS = {'S7': 'Subject7_Bharathanatyam/raw', 'S2': 'Subject2_Contemporary/pre/raw', 'S5': 'Subject5_Cueca/pre/shadow',
        'S1': 'Subject1_Waacking/pre/raw', 'S6': 'Subject6_Contemporary/pre/raw', 'S15 new': 'Subject15_Contortion',
        'S21 new': 'Subject21'}
BANDS = [(-90, -10), (-10, 20), (20, 90)]
print('fwd R-L | elev R-L, per band: above level (-90..-10) / near level (-10..20) / below (20..90); session pooled first')
for s, d in SESS.items():
    rows, seen, pool = [], set(), {b: ([], []) for b in BANDS}
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
        below = [-np.degrees(np.arcsin(np.clip(a[:, 1], -1, 1))) for a in v]
        fwd = [np.degrees(np.arctan2(a[:, 2], np.hypot(a[:, 0], a[:, 1]))) for a in v]
        sim = np.abs(below[0] - below[1]) < 12
        mb = (below[0] + below[1]) / 2
        cells = []
        for b in BANDS:
            m = sim & (mb >= b[0]) & (mb < b[1])
            if m.sum() >= 100:
                pool[b][0].append((fwd[1] - fwd[0])[m]); pool[b][1].append((below[0] - below[1])[m])
                cells.append(f'{np.median((fwd[1] - fwd[0])[m]):+6.1f} |{np.median((below[0] - below[1])[m]):+5.1f}')
            else:
                cells.append('      . |    .')
        rows.append(f'   {os.path.basename(f)[:30]:30s} ' + '   '.join(cells))
    cells = []
    for b in BANDS:
        if pool[b][0]:
            fr, el = np.concatenate(pool[b][0]), np.concatenate(pool[b][1])
            cells.append(f'{np.median(fr):+6.1f} |{np.median(el):+5.1f}')
        else:
            cells.append('      . |    .')
    print(f'{s:8s} {"(pooled)":25s} ' + '   '.join(cells))
    if s == 'S7':
        print('\n'.join(rows))
