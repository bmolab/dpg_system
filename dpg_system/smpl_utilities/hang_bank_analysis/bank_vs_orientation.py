"""Does the hang bank follow each upper arm's whole orientation away from the calibration
T-pose, rather than its elevation? A T-pose is level and reads no bank, while other level
poses in S7 jathiswaram read about -16. Each upper arm's orientation relative to the chest is
split into swing (how far the bone direction has moved from its T-pose direction) and twist
(axial rotation about the bone, 0 at the T-pose; mirrored so that mirror-image arms read the
same). Frames mirrored fore/aft as in bank_shape_unbiased.py; bank = mean front-plane angle."""
import sys, glob, hashlib, numpy as np, torch
sys.path.insert(0, '/Users/drokeby/dpg_system')
from dpg_system.shadow_mag_nodes import skeleton, world_orientations, rotate_vector_wxyz_t, conj, CHEST, as_pose_tensor
from dpg_system.quaternion_nodes import quaternion_multiply_wxyz_t
A = '/Users/drokeby/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/'
_, _, bax = skeleton(); bax = bax.double()
SESSION = sys.argv[1] if len(sys.argv) > 1 else 'Subject7_Bharathanatyam/raw'


def twist_angle(R, b):
    """Rotation of R (..., 4) about axis b (3,), degrees (swing-twist)."""
    w, v = R[..., 0], R[..., 1:]
    p = (v * b).sum(-1)
    return np.degrees(2 * np.arctan2(p, w))


def collect():
    out, seen = [], set()
    for f in sorted(glob.glob(A + SESSION + '/*.npz')):
        z = np.load(f)
        k = next((k for k in z.files if z[k].ndim == 3 and z[k].shape[1:] in ((20, 4), (37, 4))), None)
        if k is None:
            continue
        h = hashlib.md5(z[k][::50].tobytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        q, _ = as_pose_tensor(z[k])
        G = world_orientations(q, [CHEST, 5, 19, 9, 23])
        C = conj(G[CHEST])
        cols = {}
        for s, j, e in (('L', 5, 9), ('R', 19, 23)):
            R = quaternion_multiply_wxyz_t(C, G[j])
            R = R * torch.sign(R[..., :1] + 1e-12)
            d = rotate_vector_wxyz_t(R, bax[j]).numpy()
            fe = rotate_vector_wxyz_t(C, rotate_vector_wxyz_t(G[e], bax[e])).numpy()
            b = bax[j].numpy()
            cols[s + 'swing'] = np.degrees(np.arccos(np.clip(d @ b, -1, 1)))
            tw = twist_angle(R.numpy(), b)
            cols[s + 'twist'] = tw if s == 'L' else -tw
            cols[s + 'height'] = np.degrees(np.arcsin(np.clip(d[:, 1], -1, 1)))
            cols[s + 'fwd'] = np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1)))
            cols[s + 'side'] = np.degrees(np.arctan2(d[:, 0], -d[:, 1]))
            cols[s + 'bend'] = np.degrees(np.arccos(np.clip((d * fe).sum(1), -1, 1)))
        cols['take'] = np.full(len(q), f.split('/')[-1])
        out.append(cols)
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


c = collect()
mir = (np.abs(c['Lfwd'] - c['Rfwd']) < 15) & (np.abs(c['Lfwd']) < 45) & (np.abs(c['Rfwd']) < 45)
bank = (c['Lside'] + c['Rside']) / 2
height = (c['Lheight'] + c['Rheight']) / 2
swing = (c['Lswing'] + c['Rswing']) / 2
twist = (c['Ltwist'] + c['Rtwist']) / 2
print(SESSION, ' mirrored frames', mir.sum(), ' twist L vs R correlation in them %.2f' %
      np.corrcoef(c['Ltwist'][mir], c['Rtwist'][mir])[0, 1])

tpose = mir & (np.abs(c['Lheight']) < 12) & (np.abs(c['Rheight']) < 12) & (c['Lbend'] < 20) & (c['Rbend'] < 20) \
        & (np.abs(c['Lfwd']) < 15)
print('\nT-pose-like frames (both arms within 12 deg of level, within 15 of the side, elbows straight): %d' % tpose.sum())
if tpose.sum():
    print('   bank %+.1f, twist L %+.0f R %+.0f, swing %.0f' % (np.median(bank[tpose]), np.median(c['Ltwist'][tpose]),
                                                           np.median(c['Rtwist'][tpose]), np.median(swing[tpose])))

level = mir & (np.abs(height) < 20)
print('\narms near level (mean height within 20 deg), by twist (mirrored, mean of the two arms):')
for lo, hi in [(-180, -90), (-90, -45), (-45, -15), (-15, 15), (15, 45), (45, 90), (90, 180)]:
    m = level & (twist >= lo) & (twist < hi)
    if m.sum() >= 100:
        print(f'   twist {lo:+4d}..{hi:+4d}: {m.sum():6d} frames, bank {np.median(bank[m]):+6.1f}, '
              f'elbow bend {np.median((c["Lbend"]+c["Rbend"])[m]/2):4.0f}, swing {np.median(swing[m]):4.0f}')
print('\narms near level, by elbow bend:')
eb = (c['Lbend'] + c['Rbend']) / 2
for lo, hi in [(0, 20), (20, 50), (50, 90), (90, 180)]:
    m = level & (eb >= lo) & (eb < hi)
    if m.sum() >= 100:
        print(f'   bend {lo:3d}..{hi:3d}: {m.sum():6d} frames, bank {np.median(bank[m]):+6.1f}, twist {np.median(twist[m]):+5.0f}')
print('\nall mirrored frames, by twist and height (bank, frames):')
H = [(-90, -50), (-50, -20), (-20, 20), (20, 60)]
T = [(-180, -60), (-60, -20), (-20, 20), (20, 60), (60, 180)]
print('   height \\ twist ' + ' '.join(f'{lo:+d}..{hi:+d}'.rjust(15) for lo, hi in T))
for hlo, hhi in H:
    row = []
    for tlo, thi in T:
        m = mir & (height >= hlo) & (height < hhi) & (twist >= tlo) & (twist < thi)
        row.append(f'{np.median(bank[m]):+6.1f} ({m.sum():5d})' if m.sum() >= 100 else '.')
    print(f'   {hlo:+4d}..{hhi:+4d}      ' + ' '.join(r.rjust(15) for r in row))

print('\nelbow bend against twist (bank, frames), arms near level and hanging:')
B = [(0, 30), (30, 70), (70, 110), (110, 180)]
T2 = [(-90, -15), (-15, 15), (15, 45), (45, 90), (90, 180)]
for name, hm in (('near level (within 20)', np.abs(height) < 20), ('hanging (50+ below)', height < -50),
                 ('in between (20-50 below)', (height >= -50) & (height < -20))):
    print('  ' + name)
    print('   bend \\ twist ' + ' '.join(f'{lo:+d}..{hi:+d}'.rjust(15) for lo, hi in T2))
    for blo, bhi in B:
        row = []
        for tlo, thi in T2:
            m = mir & hm & (eb >= blo) & (eb < bhi) & (twist >= tlo) & (twist < thi)
            row.append(f'{np.median(bank[m]):+6.1f} ({m.sum():5d})' if m.sum() >= 80 else '.')
        print(f'   {blo:3d}..{bhi:3d}     ' + ' '.join(r.rjust(15) for r in row))

# total rotation of each upper arm away from its calibration orientation (relative to the chest)
def total_angles():
    out, seen = [], set()
    for f in sorted(glob.glob(A + SESSION + '/*.npz')):
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
        C = conj(G[CHEST])
        out.append(np.stack([np.degrees(2 * np.arccos(np.clip(quaternion_multiply_wxyz_t(C, G[j])[..., 0].abs().numpy(), 0, 1)))
                             for j in (5, 19)], 1))
    return np.concatenate(out)


tot = total_angles().mean(1)
print('\nbank by the arms\' total rotation from calibration (mean of the two), at each height:')
TB = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 90), (90, 120), (120, 180)]
print('   height \\ total ' + ' '.join(f'{lo}..{hi}'.rjust(15) for lo, hi in TB))
for hlo, hhi in [(-90, -50), (-50, -20), (-20, 20), (20, 60), (60, 90)]:
    row = []
    for tlo, thi in TB:
        m = mir & (height >= hlo) & (height < hhi) & (tot >= tlo) & (tot < thi)
        row.append(f'{np.median(bank[m]):+6.1f} ({m.sum():5d})' if m.sum() >= 80 else '.')
    print(f'   {hlo:+4d}..{hhi:+4d}       ' + ' '.join(r.rjust(15) for r in row))
row = []
for tlo, thi in TB:
    m = mir & (tot >= tlo) & (tot < thi)
    row.append(f'{np.median(bank[m]):+6.1f} ({m.sum():5d})' if m.sum() >= 80 else '.')
print('   all heights      ' + ' '.join(r.rjust(15) for r in row))
