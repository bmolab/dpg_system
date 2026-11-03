"""Bank (shared sideways shift, + toward dancer's left) of hanging upper arms and of
forearms, plus a level-arms control, for the 20-joint 2026 takes and Gaia suit tests."""
import sys, glob, os, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import (skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST, as_pose_tensor)
A = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/'
parent, order, bax = skeleton(); bax = bax.double()
def ang(G, j):
    v = rotate_vector_wxyz_t(conj(G[CHEST]), rotate_vector_wxyz_t(G[j], bax[j])).numpy()
    return np.degrees(np.arctan2(v[:, 0], -v[:, 1])), np.degrees(np.arctan2(v[:, 2], -v[:, 1])), v[:, 1]
groups = [(os.path.basename(d), sorted(glob.glob(d + '/*.npz'))) for d in sorted(glob.glob(A + 'Subject1[1-7]*'))]
groups.append(('Gaia_suit_testing', sorted(glob.glob(A + 'Gaia_suit_testing/*.npz'))))
print(f'{"take":44s} {"date":10s} {"n":>6s} {"bank":>6s} {"L fwd":>6s} {"R fwd":>6s} {"fa bank":>7s} | level-arms bank (n)')
for g, files in groups:
    print('--', g)
    for f in files:
        z = np.load(f)
        if 'quats' not in z.files:
            continue
        q, _ = as_pose_tensor(z['quats'])
        if q is None:
            continue
        G = world_orientations(q, [CHEST, 5, 9, 19, 23])
        sL, fL, hL = ang(G, 5); sR, fR, hR = ang(G, 19)
        a, _, _ = ang(G, 9); c, _, _ = ang(G, 23)
        m = (hL < -0.77) & (hR < -0.77)
        lv = (np.abs(hL) < 0.26) & (np.abs(hR) < 0.26)
        date = os.popen(f"stat -f '%SB' -t '%Y-%m-%d' '{f}'").read().strip()
        lvs = f'{np.median(((sL + sR) / 2)[lv]):6.1f} ({lv.sum()})' if lv.sum() > 60 else '     .'
        if m.sum() < 120:
            print(f'{os.path.basename(f)[:44]:44s} {date} {m.sum():6d}   too few hanging frames | {lvs}')
            continue
        med = lambda x: np.median(x[m])
        print(f'{os.path.basename(f)[:44]:44s} {date} {m.sum():6d} {med((sL + sR) / 2):6.1f} {med(fL):6.1f} {med(fR):6.1f} {med((a + c) / 2):7.1f} | {lvs}')
