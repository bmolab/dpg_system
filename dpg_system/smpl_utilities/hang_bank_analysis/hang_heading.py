"""Is the hang bank a heading error (new fusion) or a tilt error (old fusion)?
A shared heading error of both upper arms - a turn about the vertical relative to the
chest - moves arms that hang partly FORWARD sideways, by about the heading error times
the sine of the forward angle, and leaves arms hanging straight down unmoved. A tilt
error moves them sideways whatever the forward angle.  Per session: frames with both
upper arms >50 deg below level and hanging forward by similar amounts (within 10 deg),
binned by mean forward angle; fit  bank = a + h * sin(forward)  over the bin medians.
a = bank with the arms straight down (tilt part); h = heading error, degrees."""
import sys, glob, os, hashlib, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST, as_pose_tensor
A = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/'
_, _, bax = skeleton(); bax = bax.double()
OLD = {'S1': 'Subject1_Waacking/pre/raw', 'S2': 'Subject2_Contemporary/pre/raw', 'S3': 'Subject3_Vogue/pre/raw',
       'S4': 'Subject4_Krumping/pre/raw', 'S5': 'Subject5_Cueca/pre/shadow', 'S6': 'Subject6_Contemporary/pre/raw',
       'S7': 'Subject7_Bharathanatyam/raw'}
NEW = {'S11': 'Subject11_Shaolin_Tai_Chi_60', 'S12': 'Subject12_Cai_Li_Fo_Kung_Fu',
       'S13': 'Subject13_Caribbean_Stilts_Afrodiasporic_Forms', 'S14': 'Subject14_Capoeira', 'S15': 'Subject15_Contortion',
       'S17': 'Subject17_Wing_Chun_Kung_Fu', 'S18': 'Subject18', 'S19': 'Subject19', 'S20': 'Subject20',
       'S21': 'Subject21', 'S9r': 'Subject9_retakes', 'S22': 'Subject22', 'Gaia': 'Gaia_suit_testing'}
BINS = [(-15, -5), (-5, 5), (5, 15), (15, 25), (25, 40)]
MIN_BIN = 200


def frames(d):
    B, F, seen = [], [], set()
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
        fwd = [np.degrees(np.arctan2(a[:, 2], -a[:, 1])) for a in v]
        m = (v[0][:, 1] < -0.77) & (v[1][:, 1] < -0.77) & (np.abs(fwd[0] - fwd[1]) < 10)
        B.append(((side[0] + side[1]) / 2)[m]); F.append(((fwd[0] + fwd[1]) / 2)[m])
    return (np.concatenate(B), np.concatenate(F)) if B else (np.zeros(0), np.zeros(0))


print('bank (deg, + toward dancer-left) by how far forward both arms hang; fit bank = a + h * sin(forward)')
print(f'{"":5s} ' + ' '.join(f'{lo:+d}..{hi:+d}'.rjust(9) for lo, hi in BINS) + '   |      a       h')
for label, group in (('OLD fusion (2025)', OLD), ('NEW fusion (2026)', NEW)):
    print('--', label)
    for s, d in group.items():
        b, f = frames(d)
        meds, xs, ws, cells = [], [], [], []
        for lo, hi in BINS:
            k = (f >= lo) & (f < hi)
            if k.sum() >= MIN_BIN:
                meds.append(np.median(b[k])); xs.append(np.sin(np.radians(np.median(f[k])))); ws.append(np.sqrt(k.sum()))
                cells.append(f'{meds[-1]:+6.1f}({k.sum() // 1000:d}k)' if k.sum() >= 1000 else f'{meds[-1]:+6.1f}( {k.sum() // 100:d}h)')
            else:
                cells.append('        .')
        fit = '' if len(meds) < 3 else '   | {:+6.1f} {:+7.1f}'.format(*np.polyfit(xs, meds, 1, w=ws)[::-1])
        print(f'{s:5s} ' + ' '.join(c.rjust(9) for c in cells) + fit)
