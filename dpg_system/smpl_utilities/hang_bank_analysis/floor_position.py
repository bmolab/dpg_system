"""Does the hanging-arm bank vary with where the dancer stands, or which way they face?
Frames with both upper arms >50 deg below level.  Each take's own median bank is removed
first, so time drift and choreography differences between takes can't pose as position.
Position: trans_raw where present (new space, room-fixed tracker), otherwise trans.
Statistic: spread (frame-weighted sd) of the median residual per 0.5 m floor cell, or per
45 deg facing sector.  Null: slide the position/facing track in time against the bank
within each take (keeps both tracks' smoothness), 300 times."""
import sys, glob, os, hashlib, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST, as_pose_tensor
A = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/'
_, _, bax = skeleton(); bax = bax.double()
SESSIONS = {'S12': 'Subject12_Cai_Li_Fo_Kung_Fu', 'S13': 'Subject13_Caribbean_Stilts_Afrodiasporic_Forms',
            'S14': 'Subject14_Capoeira', 'S15': 'Subject15_Contortion', 'S17': 'Subject17_Wing_Chun_Kung_Fu',
            'S18': 'Subject18', 'S19': 'Subject19', 'S20': 'Subject20', 'S21': 'Subject21',
            'S9r': 'Subject9_retakes', 'S22': 'Subject22'}
NEW_SPACE = {'S19', 'S20', 'S21', 'S9r', 'S22'}
CELL, MIN_CELL = 0.5, 120
rng = np.random.default_rng(0)


def take_tracks(f):
    z = np.load(f)
    k = next((k for k in z.files if z[k].ndim == 3 and z[k].shape[1:] in ((20, 4), (37, 4))), None)
    if k is None or 'trans' not in z.files:
        return None
    q, _ = as_pose_tensor(z[k])
    G = world_orientations(q, [CHEST, 5, 19])
    v = [rotate_vector_wxyz_t(conj(G[CHEST]), rotate_vector_wxyz_t(G[j], bax[j])).numpy() for j in (5, 19)]
    bank = (np.degrees(np.arctan2(v[0][:, 0], -v[0][:, 1])) + np.degrees(np.arctan2(v[1][:, 0], -v[1][:, 1]))) / 2
    down = (v[0][:, 1] < -0.77) & (v[1][:, 1] < -0.77)
    fw = rotate_vector_wxyz_t(G[CHEST], torch.tensor([0., 0., 1.], dtype=torch.float64)).numpy()
    facing = np.degrees(np.arctan2(fw[:, 0], fw[:, 2]))
    pos = (z['trans_raw'] if 'trans_raw' in z.files else z['trans'])[:, [0, 2]]
    h = hashlib.md5(z[k][::50].tobytes()).hexdigest()
    return h, bank, down, pos, facing


def spread(res, labels):
    keys, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    meds = np.array([np.median(res[inv == i]) for i in range(len(keys))])
    ok = cnt >= MIN_CELL
    if ok.sum() < 2:
        return np.nan, keys[ok], meds[ok], cnt[ok]
    w = cnt[ok]
    m = np.average(meds[ok], weights=w)
    return np.sqrt(np.average((meds[ok] - m) ** 2, weights=w)), keys[ok], meds[ok], cnt[ok]


def label_pos(p):
    c = np.floor(p / CELL).astype(int)
    return c[:, 0] * 1000 + c[:, 1]


def label_face(a):
    return (np.floor((a + 180) / 45).astype(int)) % 8


def test(takes, which):
    def build(shift):
        R, L = [], []
        for bank, down, pos, facing in takes:
            n = len(bank)
            s = rng.integers(n // 5, n - n // 5) if shift else 0
            lab = label_pos(np.roll(pos, s, 0)) if which == 'pos' else label_face(np.roll(facing, s))
            m = down
            if m.sum() < 60:
                continue
            R.append(bank[m] - np.median(bank[m])); L.append(lab[m])
        return np.concatenate(R), np.concatenate(L)
    r, l = build(False)
    obs, keys, meds, cnt = spread(r, l)
    null = np.array([spread(*build(True))[0] for _ in range(300)])
    null = null[np.isfinite(null)]
    p = (np.sum(null >= obs) + 1) / (len(null) + 1)
    return obs, np.percentile(null, 95), p, keys, meds, cnt


print(f'{"session":7s} {"frames":>7s} | position: spread  null95   p     cells | facing: spread  null95   p')
store = {}
for s, d in SESSIONS.items():
    takes, seen = [], set()
    for f in sorted(glob.glob(A + d + '/*.npz')):
        t = take_tracks(f)
        if t is None or t[0] in seen:
            continue
        seen.add(t[0]); takes.append(t[1:])
    nfr = sum(int(t[1].sum()) for t in takes)
    po = test(takes, 'pos'); fa = test(takes, 'face')
    store[s] = po
    print(f'{s:7s} {nfr:7d} | {po[0]:14.1f} {po[1]:7.1f} {po[2]:6.3f} {len(po[3]):5d} | {fa[0]:13.1f} {fa[1]:7.1f} {fa[2]:6.3f}'
          + ('   new space' if s in NEW_SPACE else ''))

print('\nnew space, per 0.5 m cell (trans_raw x across, z down): median residual bank, per session; . = too few frames')
xs = sorted({int(k // 1000) if k >= 0 else -((-k) // 1000) for s in NEW_SPACE for k in store[s][3]})
for s in sorted(NEW_SPACE):
    keys, meds = store[s][3], store[s][4]
    grid = {}
    for k, m in zip(keys, meds):
        cx = int(np.round(k / 1000)); cz = int(k - cx * 1000)
        grid[(cx, cz)] = m
    if not grid:
        continue
    cxs = range(min(c[0] for c in grid), max(c[0] for c in grid) + 1)
    czs = range(min(c[1] for c in grid), max(c[1] for c in grid) + 1)
    print(f'-- {s}   x cells {cxs.start * CELL:+.1f}..{(cxs.stop) * CELL:+.1f} m')
    for cz in czs:
        print(f'  z {cz * CELL:+5.1f} ' + ' '.join(f'{grid[(cx, cz)]:+5.1f}' if (cx, cz) in grid else '    .' for cx in cxs))
