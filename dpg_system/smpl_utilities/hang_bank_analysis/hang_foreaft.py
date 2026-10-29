"""Hanging arms (both upper arms >50 deg below level, chest frame): sideways and
forward angles of each upper arm, split into shared and opposite parts.
sideways: + toward dancer's left; forward: + toward the front."""
import sys, glob, os, re, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST
A = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/'
parent, order, bax = skeleton(); bax = bax.double()
sets = [(s, sorted(glob.glob(A + s + '/pre/raw/*.npz') + glob.glob(A + s + '/pre/shadow/*.npz')))
        for s in ['Subject1_Waacking', 'Subject2_Contemporary', 'Subject3_Vogue', 'Subject4_Krumping', 'Subject5_Cueca', 'Subject6_Contemporary']]
sets.append(('Subject7_Bharathanatyam', sorted(glob.glob(A + 'Subject7_Bharathanatyam/raw/*.npz'))))
def ang(G, j):
    v = rotate_vector_wxyz_t(conj(G[CHEST]), rotate_vector_wxyz_t(G[j], bax[j])).numpy()
    return np.degrees(np.arctan2(v[:, 0], -v[:, 1])), np.degrees(np.arctan2(v[:, 2], -v[:, 1])), v[:, 1]
print(f'{"take":34s} {"n":>6s} | upper arm: {"bank":>5s} {"fwd both":>8s} {"L fwd":>6s} {"R fwd":>6s} | forearm: {"bank":>5s} {"fwd both":>8s} {"L-R fwd/2":>9s}')
for subj, files in sets:
    print('--', subj)
    for f in files:
        z = np.load(f)
        if 'quats' not in z.files or z['quats'].shape[1:] != (37, 4):
            continue
        q = torch.as_tensor(z['quats'].astype(np.float64))
        G = world_orientations(q, [CHEST, 5, 9, 19, 23])
        sL, fL, hL = ang(G, 5); sR, fR, hR = ang(G, 19)
        m = (hL < -0.77) & (hR < -0.77)
        if m.sum() < 120:
            continue
        a, b, _ = ang(G, 9); c, d, _ = ang(G, 23)
        med = lambda x: np.median(x[m])
        print(f'{os.path.basename(f)[:34]:34s} {m.sum():6d} | {"":10s} {med((sL+sR)/2):5.1f} {med((fL+fR)/2):8.1f} {med(fL):6.1f} {med(fR):6.1f} |'
              f' {"":8s} {med((a+c)/2):5.1f} {med((b+d)/2):8.1f} {med((b-d)/2):9.1f}')
