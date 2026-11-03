"""Starting values for shadow_heading_adjust: per arm, the swing and phase that best remove
the once-per-turn part of that upper arm's forward/back angle (chest frame, arm within 20 deg
of level) against the dancer's facing. Grid search; reports the swing of that once-per-turn
part before and after."""
import sys, math, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import (skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST,
                                         heading_weights, adjust_heading)
_, _, bax = skeleton(); bax = bax.double()
F = sys.argv[1] if len(sys.argv) > 1 else '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/Subject7_Bharathanatyam/beta/Subject7_take_jathiswaram_a_beta.npz'
q = torch.as_tensor(np.load(F)['quats'].astype(np.float64)[::4])


def turn_swing(Q, j):
    G = world_orientations(Q, [CHEST, j])
    a = rotate_vector_wxyz_t(conj(G[CHEST]), rotate_vector_wxyz_t(G[j], bax[j])).numpy()
    below = -np.degrees(np.arcsin(np.clip(a[:, 1], -1, 1)))
    fwd = np.degrees(np.arctan2(a[:, 2], np.hypot(a[:, 0], a[:, 1])))
    cf = rotate_vector_wxyz_t(G[CHEST], torch.tensor([0., 0., 1.], dtype=torch.float64)).numpy()
    r = np.arctan2(cf[:, 0], cf[:, 2])
    m = np.abs(below) < 20
    X = np.column_stack([np.ones(m.sum()), np.cos(r[m]), np.sin(r[m])])
    c, *_ = np.linalg.lstsq(X, fwd[m], rcond=None)
    return np.hypot(c[1], c[2]), m.sum()


zero = torch.zeros(2, dtype=torch.float64)
for a, (side, j) in enumerate((('left', 5), ('right', 19))):
    base, n = turn_swing(q, j)
    best = (base, 0.0, 0.0)
    for sw in range(-40, 41, 4):
        for ph in range(-180, 180, 15):
            yaw = torch.zeros(2, 6, dtype=torch.float64)
            yaw[a] = torch.tensor(heading_weights(sw, ph), dtype=torch.float64)
            s, _ = turn_swing(adjust_heading(q, yaw, zero)[0], j)
            if s < best[0]:
                best = (s, sw, ph)
    # refine
    s0, sw0, ph0 = best
    for sw in np.arange(sw0 - 4, sw0 + 4.1, 1):
        for ph in np.arange(ph0 - 15, ph0 + 15.1, 3):
            yaw = torch.zeros(2, 6, dtype=torch.float64)
            yaw[a] = torch.tensor(heading_weights(sw, ph), dtype=torch.float64)
            s, _ = turn_swing(adjust_heading(q, yaw, zero)[0], j)
            if s < best[0]:
                best = (s, sw, ph)
    print(f'{side:5s} arm ({n} near-level frames): once-per-turn swing {base:4.1f} deg -> {best[0]:4.1f} with swing {best[1]:+.0f}, phase {((best[2] + 180) % 360) - 180:+.0f}')
