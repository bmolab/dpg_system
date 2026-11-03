"""The hang bank against arm height, selected without looking at height. bank_shape.py kept
frames with the two upper arms at similar heights - but near level the bank is what makes the
heights differ, so it threw away the banked frames and the bank seemed to vanish at level.
Here frames are kept when the arms are mirrored fore/aft (forward angles within 15 deg, both
within 45 deg of the front plane) and the bank is the mean of the two front-plane angles, as in
bank_shape.py, banded by the pair's mean height. Old-fusion sessions, and two new ones."""
import sys, glob, hashlib, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST, as_pose_tensor
A = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/'
_, _, bax = skeleton(); bax = bax.double()
SESSIONS = {'S1': 'Subject1_Waacking/pre/raw', 'S2': 'Subject2_Contemporary/pre/raw', 'S3': 'Subject3_Vogue/pre/raw',
            'S5': 'Subject5_Cueca/pre/shadow', 'S6': 'Subject6_Contemporary/pre/raw', 'S7': 'Subject7_Bharathanatyam/raw',
            'S12 new': 'Subject12_Cai_Li_Fo_Kung_Fu', 'S17 new': 'Subject17_Wing_Chun_Kung_Fu'}
BANDS = [(-90, -60), (-60, -40), (-40, -20), (-20, 0), (0, 20), (20, 40), (40, 60)]   # mean height above level
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
        fwd = [np.degrees(np.arcsin(np.clip(a[:, 2], -1, 1))) for a in v]
        side = [np.degrees(np.arctan2(a[:, 0], -a[:, 1])) for a in v]
        m = (np.abs(fwd[0] - fwd[1]) < 15) & (np.abs(fwd[0]) < 45) & (np.abs(fwd[1]) < 45)
        height = np.degrees(np.arcsin(np.clip((v[0][:, 1] + v[1][:, 1]) / 2, -1, 1)))
        B.append(((side[0] + side[1]) / 2)[m]); H.append(height[m])
    return np.concatenate(B), np.concatenate(H)


print('bank (deg, + toward the dancer\'s left) by the pair\'s mean height above level; frames in brackets')
print(f'{"":8s} ' + ' '.join(f'{lo:+d}..{hi:+d}'.rjust(14) for lo, hi in BANDS))
for s, d in SESSIONS.items():
    b, h = collect(d)
    cells = []
    for lo, hi in BANDS:
        k = (h >= lo) & (h < hi)
        cells.append(f'{np.median(b[k]):+6.1f} ({k.sum():5d})' if k.sum() >= MIN else '.'.rjust(14))
    print(f'{s:8s} ' + ' '.join(c.rjust(14) for c in cells))
