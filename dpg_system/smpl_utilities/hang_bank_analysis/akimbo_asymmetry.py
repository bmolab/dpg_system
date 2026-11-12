"""Hands-on-hips frames (both elbows bent, upper arms partly lowered, forearms pointing in
toward the body): how the left arm differs from the mirrored right, as found and after a
fixed bank. Chest frame, +X dancer's left, +Y up, +Z forward; the right arm mirrored (X
negated) so a symmetric pose reads 0 everywhere. Differences are LEFT minus mirrored RIGHT:
  side   + = left arm further out to the left than the right is to the right
  shared the bank (both arms shifted the same way, + = toward the dancer's left)
  height + = left arm higher
  fwd    + = left arm further forward
for the upper arm and the forearm; and the shared part against facing."""
import sys, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import (skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST,
                                         correct_pose, PER_ARM, quaternion_multiply_wxyz_t)
_, _, bax = skeleton(); bax = bax.double()
F = sys.argv[1] if len(sys.argv) > 1 else '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/Subject7_Bharathanatyam/beta/Subject7_take_jathiswaram_a_beta.npz'
BANK = float(sys.argv[2]) if len(sys.argv) > 2 else -17.9
q = torch.as_tensor(np.load(F)['quats'].astype(np.float64))


def arms(Q):
    G = world_orientations(Q, [CHEST, 5, 9, 19, 23])
    C = conj(G[CHEST])
    d = {}
    for nm, j in (('uaL', 5), ('faL', 9), ('uaR', 19), ('faR', 23)):
        d[nm] = rotate_vector_wxyz_t(C, rotate_vector_wxyz_t(G[j], bax[j])).numpy()
    for s in 'LR':
        d['bend' + s] = np.degrees(np.arccos(np.clip((d['ua' + s] * d['fa' + s]).sum(-1), -1, 1)))
    cf = rotate_vector_wxyz_t(G[CHEST], torch.tensor([0., 0., 1.], dtype=torch.float64)).numpy()
    d['face'] = np.degrees(np.arctan2(cf[:, 0], cf[:, 2]))
    return d


def angles(v, mirror):
    x = -v[:, 0] if mirror else v[:, 0]
    side = np.degrees(np.arctan2(x, -v[:, 1]))            # out from straight down
    height = np.degrees(np.arcsin(np.clip(v[:, 1], -1, 1)))
    fwd = np.degrees(np.arctan2(v[:, 2], np.hypot(x, v[:, 1])))
    return side, height, fwd


d = arms(q)
pick = ((d['bendL'] > 60) & (d['bendL'] < 140) & (d['bendR'] > 60) & (d['bendR'] < 140)
        & (d['uaL'][:, 1] < -0.25) & (d['uaL'][:, 1] > -0.92) & (d['uaR'][:, 1] < -0.25) & (d['uaR'][:, 1] > -0.92)
        & (d['faL'][:, 0] < -0.3) & (d['faR'][:, 0] > 0.3))
idx = np.where(pick)[0]
runs = np.split(idx, np.where(np.diff(idx) > 30)[0] + 1) if len(idx) else []
runs = [r for r in runs if len(r) >= 60]
print(f'hands-on-hips frames: {sum(len(r) for r in runs)} in {len(runs)} holds of 1 s or more')
print('   holds (start-end):', ', '.join(f'{r[0]}-{r[-1]}' for r in runs[:30]))
sel = np.concatenate(runs) if runs else idx


def report(D, label):
    sL, hL, fL = angles(D['uaL'], False); sR, hR, fR = angles(D['uaR'], True)
    tL, gL, wL = angles(D['faL'], False); tR, gR, wR = angles(D['faR'], True)
    m = sel
    shared_ua = ((sL - sR) / 2)  # placeholder, replaced below
    # shared sideways shift (bank) in raw chest X: + toward dancer's left
    rawL = np.degrees(np.arctan2(D['uaL'][:, 0], -D['uaL'][:, 1])); rawR = np.degrees(np.arctan2(D['uaR'][:, 0], -D['uaR'][:, 1]))
    bank = (rawL + rawR) / 2
    print(f'-- {label}')
    print(f'   upper arm: side L-R {np.median((sL - sR)[m]):+6.1f}  height L-R {np.median((hL - hR)[m]):+6.1f}  fwd L-R {np.median((fL - fR)[m]):+6.1f}   shared bank {np.median(bank[m]):+6.1f}')
    print(f'   forearm:   side L-R {np.median((tL - tR)[m]):+6.1f}  height L-R {np.median((gL - gR)[m]):+6.1f}  fwd L-R {np.median((wL - wR)[m]):+6.1f}')
    print(f'   elbow bend L-R {np.median((D["bendL"] - D["bendR"])[m]):+6.1f}   (left {np.median(D["bendL"][m]):.0f}, right {np.median(D["bendR"][m]):.0f}; upper arms {np.median(-hL[m]):.0f} / {np.median(-hR[m]):.0f} below level)')
    r = np.radians(D['face'][m]); X = np.column_stack([np.ones(len(m)), np.cos(r), np.sin(r)])
    for nm, val in (('upper arm fwd L-R', (fL - fR)[m]), ('shared bank', bank[m]), ('upper arm side L-R', (sL - sR)[m])):
        c, *_ = np.linalg.lstsq(X, val, rcond=None)
        print(f'   {nm:20s} vs facing: steady {c[0]:+6.1f}, once-per-turn swing {np.hypot(c[1], c[2]):5.1f}')


report(d, 'as recorded')
w = torch.zeros(2, PER_ARM, dtype=torch.float64); w[:, -1] = np.radians(-BANK)
report(arms(correct_pose(q, w, terms=('hang',))), f'after fixed bank {BANK}')
faces = d['face'][sel]
print('facings of the holds (deg, 8 sectors from -180):', np.bincount(((faces + 180) // 45).astype(int) % 8, minlength=8))
