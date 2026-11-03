"""Bank of hanging upper arms per take against recording time, every session.
Recording time = earliest of the file's creation and modification times (raw folders
keep the recording time as creation; later copies keep it as modification)."""
import sys, glob, os, hashlib, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST, as_pose_tensor
A = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/'
parent, order, bax = skeleton(); bax = bax.double()
S = {'S1': 'Subject1_Waacking/pre/raw', 'S2': 'Subject2_Contemporary/pre/raw', 'S3': 'Subject3_Vogue/pre/raw',
     'S4': 'Subject4_Krumping/pre/raw', 'S5': 'Subject5_Cueca/pre/shadow', 'S6': 'Subject6_Contemporary/pre/raw',
     'S7': 'Subject7_Bharathanatyam/raw', 'S11': 'Subject11_Shaolin_Tai_Chi_60', 'S12': 'Subject12_Cai_Li_Fo_Kung_Fu',
     'S13': 'Subject13_Caribbean_Stilts_Afrodiasporic_Forms', 'S14': 'Subject14_Capoeira', 'S15': 'Subject15_Contortion',
     'S16': 'Subject16_Manual_Wheelchair_Dancing', 'S17': 'Subject17_Wing_Chun_Kung_Fu', 'S18': 'Subject18',
     'S19': 'Subject19', 'S20': 'Subject20', 'S21': 'Subject21', 'S9r': 'Subject9_retakes', 'S22': 'Subject22'}
def bank(qq):
    q, _ = as_pose_tensor(qq)
    G = world_orientations(q, [CHEST, 5, 19])
    out = []
    for j in (5, 19):
        v = rotate_vector_wxyz_t(conj(G[CHEST]), rotate_vector_wxyz_t(G[j], bax[j])).numpy()
        out.append((np.degrees(np.arctan2(v[:, 0], -v[:, 1])), v[:, 1]))
    m = (out[0][1] < -0.77) & (out[1][1] < -0.77)
    return (out[0][0] + out[1][0])[m] / 2
for s, d in S.items():
    rows, seen = [], set()
    for f in glob.glob(A + d + '/*.npz'):
        z = np.load(f)
        k = next((k for k in z.files if z[k].ndim == 3 and z[k].shape[1:] in ((20, 4), (37, 4))), None)
        if k is None:
            continue
        h = hashlib.md5(z[k][::50].tobytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        st = os.stat(f); t = min(st.st_birthtime, st.st_mtime)
        b = bank(z[k])
        if len(b) >= 120:
            rows.append((t, np.median(b), len(b)))
    rows.sort()
    if len(rows) < 3:
        continue
    t = np.array([r[0] for r in rows]); b = np.array([r[1] for r in rows]); w = np.sqrt([r[2] for r in rows])
    hrs = (t - t[0]) / 3600
    slope = np.polyfit(hrs, b, 1, w=w)[0] if hrs[-1] > 0.2 else np.nan
    day = os.popen(f"date -r {int(t[0])} '+%Y-%m-%d %H:%M'").read().strip()
    print(f'{s:4s} {day}  span {hrs[-1]:5.1f} h  n {len(rows):2d}  slope {slope:+6.1f} deg/h  first->last {b[0]:+6.1f} .. {b[-1]:+6.1f}   ' + ' '.join(f'{x:+.0f}' for x in b))
