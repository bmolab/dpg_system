"""Does the hang bank depend on the upper arm's axial rotation, not just its height?
Hanging frames (both upper arms >50 deg below level, similar heights). Each upper arm's
axial rotation: its local +Z (forward in the T-pose) seen from the chest, as an angle in
the horizontal plane (0 = pointing forward; + = turned toward the dancer's left). Bank per
band of the two arms' mean axial rotation, and per arm (each arm's own sideways angle,
minus the session median, against its own rotation)."""
import sys, glob, hashlib, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST, as_pose_tensor
A = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/'
_, _, bax = skeleton(); bax = bax.double()
SESS = {'S7 (old)': 'Subject7_Bharathanatyam/raw', 'S2 (old)': 'Subject2_Contemporary/pre/raw',
        'S5 (old)': 'Subject5_Cueca/pre/shadow', 'S1 (old)': 'Subject1_Waacking/pre/raw',
        'S15 (new)': 'Subject15_Contortion', 'S11 (new)': 'Subject11_Shaolin_Tai_Chi_60'}
BANDS = [(-180, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 180)]


def collect(d):
    side, rot, seen = [[], []], [[], []], set()
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
        v, r = [], []
        for j in (5, 19):
            Gc = conj(G[CHEST])
            v.append(rotate_vector_wxyz_t(Gc, rotate_vector_wxyz_t(G[j], bax[j])).numpy())
            fz = rotate_vector_wxyz_t(Gc, rotate_vector_wxyz_t(G[j], torch.tensor([0., 0., 1.], dtype=torch.float64))).numpy()
            r.append(np.degrees(np.arctan2(fz[:, 0], fz[:, 2])))
        m = (v[0][:, 1] < -0.77) & (v[1][:, 1] < -0.77) & (np.abs(v[0][:, 1] - v[1][:, 1]) < 0.2)
        for a in range(2):
            side[a].append(np.degrees(np.arctan2(v[a][m, 0], -v[a][m, 1]))); rot[a].append(r[a][m])
    return [np.concatenate(x) for x in side], [np.concatenate(x) for x in rot]


def circmean(a):
    r = np.radians(a); return np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))


for s, d in SESS.items():
    side, rot = collect(d)
    bank = (side[0] + side[1]) / 2
    # each arm's rotation relative to its own session typical, mirrored so + = turned outward for both
    rel = [((rot[0] - circmean(rot[0]) + 180) % 360) - 180, -(((rot[1] - circmean(rot[1]) + 180) % 360) - 180)]
    mrot = (rel[0] + rel[1]) / 2
    print(f'-- {s}: {len(bank)} frames, typical rotation L {circmean(rot[0]):+.0f} R {circmean(rot[1]):+.0f}, bank median {np.median(bank):+.1f}')
    print('   both arms turned outward (+) / inward (-) from typical, deg:  ' + '  '.join(f'{lo:+d}..{hi:+d}' for lo, hi in BANDS))
    cells = []
    for lo, hi in BANDS:
        k = (mrot >= lo) & (mrot < hi)
        cells.append(f'{np.median(bank[k]):+6.1f}({k.sum()})' if k.sum() >= 150 else '      .')
    print('   bank by mean rotation:                                         ' + ' '.join(cells))
    for a, nm in ((0, 'left'), (1, 'right')):
        own = side[a] - np.median(side[a])
        cells = []
        for lo, hi in BANDS:
            k = (rel[a] >= lo) & (rel[a] < hi)
            cells.append(f'{np.median(own[k]):+6.1f}({k.sum()})' if k.sum() >= 150 else '      .')
        print(f'   {nm:5s} sideways vs own rotation:                                ' + ' '.join(cells))
