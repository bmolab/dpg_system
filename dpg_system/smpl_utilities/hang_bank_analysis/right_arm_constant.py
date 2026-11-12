"""Jathiswaram (S7): is there a height-independent part on top of the height-ramped bank?
Frames with the two upper arms at similar heights. Per height band (mean of the two arms,
degrees below level; negative = above level):
  bank      shared sideways angle in the chest's front plane, from straight down (+ = left)
  elev R-L  how much higher the right arm reads than the left (degrees)
  fwd R-L   how much further forward the right arm sits than the left (degrees)
A constant offset on one arm shows in 'bank' as half its size at every height, and in
'elev R-L' with the arms near level."""
import sys, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST, parse_ranges
_, _, bax = skeleton(); bax = bax.double()
F = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/Subject7_Bharathanatyam/beta/Subject7_take_jathiswaram_a_beta.npz'
q = torch.as_tensor(np.load(F)['quats'].astype(np.float64))
G = world_orientations(q, [CHEST, 5, 19])
v = [rotate_vector_wxyz_t(conj(G[CHEST]), rotate_vector_wxyz_t(G[j], bax[j])).numpy() for j in (5, 19)]
below = [-np.degrees(np.arcsin(np.clip(a[:, 1], -1, 1))) for a in v]
side = [np.degrees(np.arctan2(a[:, 0], -a[:, 1])) for a in v]
fwd = [np.degrees(np.arctan2(a[:, 2], np.hypot(a[:, 0], a[:, 1]))) for a in v]
sim = np.abs(below[0] - below[1]) < 12
mean_below = (below[0] + below[1]) / 2
bank = (side[0] + side[1]) / 2
BANDS = [(-60, -30), (-30, -10), (-10, 5), (5, 20), (20, 35), (35, 50), (50, 65), (65, 90)]
print(f'{"band below level":>17s} {"frames":>7s} {"bank":>7s} {"elev R-L":>9s} {"fwd R-L":>8s}')
for lo, hi in BANDS:
    m = sim & (mean_below >= lo) & (mean_below < hi)
    if m.sum() < 100:
        print(f'{lo:+4d}..{hi:+4d}          .'); continue
    print(f'{lo:+4d}..{hi:+4d}     {m.sum():7d} {np.median(bank[m]):+7.1f} {np.median((below[0] - below[1])[m]):+9.1f} {np.median((fwd[1] - fwd[0])[m]):+8.1f}')
# David's flagged T-pose stretch at the start
tp = np.zeros(len(q), bool); tp[0:1721] = True
lv = tp & sim & (np.abs(mean_below) < 15)
print(f'\nopening stretch 0-1721, arms within 15 deg of level: {lv.sum()} frames, bank {np.median(bank[lv]):+.1f}, '
      f'elev R-L {np.median((below[0] - below[1])[lv]):+.1f}, fwd R-L {np.median((fwd[1] - fwd[0])[lv]):+.1f}')
