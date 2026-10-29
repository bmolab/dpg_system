"""Does the bank keep changing after the arms come down and hold still (the ~5 deg / 2 s
creep once read from three drops in Subject 2 take 3)?  For every arms-down hold: both
upper arms reach 50 deg below level after >= 1 s not down, then stay down >= 4 s with the
arms roughly still (splay and forward angles vary < 5 deg sd from 0.5 s to 4 s).  Reports
the absolute bank against time after the drop, and its change from 0.5 s to 4 s.  60 Hz.
Sessions before the 2025-12-18 Shadow update use the old fusion algorithm.
Result 2026-10-29: old fusion +1.3 (relaxes toward 0, 78% of 23 holds), new fusion -0.3
(46% of 116 holds, interval -1.0..+0.4): no growing creep under either."""
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
       'S16': 'Subject16_Manual_Wheelchair_Dancing', 'S17': 'Subject17_Wing_Chun_Kung_Fu', 'S18': 'Subject18',
       'S19': 'Subject19', 'S20': 'Subject20', 'S21': 'Subject21', 'S9r': 'Subject9_retakes', 'S22': 'Subject22',
       'Gaia': 'Gaia_suit_testing'}
LAGS = [0, 15, 30, 60, 90, 120, 180, 239]               # frames after the drop
BEFORE, HOLD, STILL = 60, 240, 5.0
REF = 30                                                # 0.5 s


def holds(qq):
    q, _ = as_pose_tensor(qq)
    G = world_orientations(q, [CHEST, 5, 19])
    v = [rotate_vector_wxyz_t(conj(G[CHEST]), rotate_vector_wxyz_t(G[j], bax[j])).numpy() for j in (5, 19)]
    side = [np.degrees(np.arctan2(a[:, 0], -a[:, 1])) for a in v]
    fwd = [np.degrees(np.arctan2(a[:, 2], -a[:, 1])) for a in v]
    bank, splay = (side[0] + side[1]) / 2, (side[0] - side[1]) / 2
    down = (v[0][:, 1] < -0.77) & (v[1][:, 1] < -0.77)
    out, i = [], BEFORE
    while i < len(down) - HOLD - 10:
        if down[i] and not down[i - BEFORE:i].any() and down[i:i + HOLD].all():
            w = slice(i + REF, i + HOLD)
            if max(np.std(splay[w]), np.std(fwd[0][w]), np.std(fwd[1][w])) < STILL:
                out.append([bank[i + l] for l in LAGS])
            i += HOLD
        else:
            i += 1
    return out


def session(d):
    rows, seen = [], set()
    for f in glob.glob(A + d + '/*.npz'):
        z = np.load(f)
        k = next((k for k in z.files if z[k].ndim == 3 and z[k].shape[1:] in ((20, 4), (37, 4))), None)
        if k is None:
            continue
        h = hashlib.md5(z[k][::50].tobytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            rows += holds(z[k])
    return np.array(rows)


def line(name, r):
    ch = r[:, -1] - r[:, LAGS.index(REF)]
    return (f'{name:6s} {len(r):4d}  ' + ' '.join(f'{np.median(r[:, k]):+5.1f}' for k in range(len(LAGS))) +
            f'   | {np.median(ch):+5.1f}  {np.mean(ch > 0) * 100:3.0f}%')


print('bank (deg) after the arms come down and stay down >= 4 s, roughly still; median over holds')
print(f'{"":6s} {"holds":>5s}  ' + ' '.join(f'{l / 60:4.2f}s'[:5].rjust(5) for l in LAGS) + '   | change 0.5 s -> 4 s, holds toward left')
for label, group in (('OLD fusion (2025)', OLD), ('NEW fusion (2026)', NEW)):
    print('--', label)
    pooled = []
    for s, d in group.items():
        r = session(d)
        if len(r):
            print(line(s, r))
            pooled.append(r)
    P = np.concatenate(pooled)
    print(line('ALL', P))
    ch = P[:, -1] - P[:, LAGS.index(REF)]
    rng = np.random.default_rng(0)
    boot = [np.median(ch[rng.integers(0, len(ch), len(ch))]) for _ in range(2000)]
    print(f'       change 0.5 s -> 4 s: 95% interval {np.percentile(boot, 2.5):+.1f} to {np.percentile(boot, 97.5):+.1f}')
