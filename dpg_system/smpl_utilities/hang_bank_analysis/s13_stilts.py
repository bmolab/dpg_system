"""Subject 13: bank of hanging upper arms vs root height (stilts), per take in recording order."""
import sys, glob, os, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST, as_pose_tensor
A = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/Subject13_Caribbean_Stilts_Afrodiasporic_Forms/'
parent, order, bax = skeleton(); bax = bax.double()
def ang(G, j):
    v = rotate_vector_wxyz_t(conj(G[CHEST]), rotate_vector_wxyz_t(G[j], bax[j])).numpy()
    return np.degrees(np.arctan2(v[:, 0], -v[:, 1])), v[:, 1]
files = sorted(glob.glob(A + '*.npz'), key=os.path.getmtime)
print(f'{"take":42s} {"mtime":11s} {"root height":>11s} {"n":>6s} {"bank":>6s}   bank by height third (low/mid/high)')
for f in files:
    z = np.load(f)
    q, _ = as_pose_tensor(z['quats'])
    G = world_orientations(q, [CHEST, 5, 19])
    sL, hL = ang(G, 5); sR, hR = ang(G, 19)
    y = z['trans'][:, 1]
    m = (hL < -0.77) & (hR < -0.77)
    b = (sL + sR) / 2
    t = os.popen(f"stat -f '%Sm' -t '%m-%d %H:%M' '{f}'").read().strip()
    if m.sum() < 120:
        print(f'{os.path.basename(f)[:42]:42s} {t} {np.median(y):11.2f}   too few'); continue
    cuts = np.quantile(y[m], [1/3, 2/3])
    parts = [b[m & (y < cuts[0])], b[m & (y >= cuts[0]) & (y < cuts[1])], b[m & (y >= cuts[1])]]
    print(f'{os.path.basename(f)[:42]:42s} {t} {np.median(y):11.2f} {m.sum():6d} {np.median(b[m]):6.1f}   ' + ' '.join(f'{np.median(p):6.1f}' for p in parts))
