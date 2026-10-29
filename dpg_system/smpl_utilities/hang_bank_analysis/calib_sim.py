"""What does a calibration-time error do to the hanging arms' bank?
Model: each sensor gets a fixed offset at the T-pose so it reads the assumed T-pose
(identity global, Shadow's rest). Reported = sensor * offset.  Mounting M cancels
exactly; what survives is the body's actual pose at calibration vs the assumed one.
Bank = hang_symmetry's measure: mean angle of the two upper arms in the chest's
front plane, + toward the dancer's left. Frame: Y up, +Z forward, +X dancer's left."""
import numpy as np
from scipy.spatial.transform import Rotation as R
D = np.radians
ax = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
def rot(a, deg): return R.from_rotvec(np.array(ax[a]) * D(deg))
I = R.identity()
bone = {'L': np.array([1.0, 0, 0]), 'R': np.array([-1.0, 0, 0])}

def bank(G):
    s = []
    for side in 'LR':
        v = (G['chest'].inv() * G[side]).apply(bone[side])
        s.append(np.degrees(np.arctan2(v[0], -v[1])))
    return (s[0] + s[1]) / 2, s

def run(name, cal_true, mount=None, heading_from_root=False):
    mount = mount or {}
    seg = ['root', 'chest', 'L', 'R']
    M = {k: mount.get(k, I) for k in seg}
    sensor_cal = {k: cal_true[k] * M[k] for k in seg}
    off = {k: sensor_cal[k].inv() for k in seg}                 # makes each read identity
    if heading_from_root:
        # alternative solver: each sensor levels itself from gravity, but takes its
        # heading from the root sensor's reading (yaw part only)
        yaw_root = R.from_euler('YXZ', sensor_cal['root'].as_euler('YXZ') * [1, 0, 0])
        for k in seg:
            yaw_k = R.from_euler('YXZ', sensor_cal[k].as_euler('YXZ') * [1, 0, 0])
            off[k] = (yaw_k.inv() * sensor_cal[k]).inv() * (yaw_k.inv() * yaw_root).inv() * I
            off[k] = sensor_cal[k].inv() * yaw_k * yaw_root.inv()
    # now stand upright, arms hanging straight down (true bank 0), 15 deg splay
    hang = {'root': I, 'chest': I, 'L': rot('z', -75), 'R': rot('z', 75)}
    rep = {k: hang[k] * M[k] * off[k] for k in seg}
    b, s = bank(rep)
    tp = {k: I * M[k] * off[k] for k in seg}   # a correct T-pose afterwards
    print(f'{name:58s} hang bank {b:+6.1f}  (L {s[0]-15:+5.1f}, R {s[1]+15:+5.1f})  correct T-pose reads {bank(tp)[0]:+5.1f}')

up = {k: I for k in ['root', 'chest', 'L', 'R']}
print('-- sensor mounting errors, ordinary per-sensor calibration')
for a in 'xyz':
    run(f'root sensor rotated 10 deg about {a}', up, {'root': rot(a, 10)})
run('chest sensor rotated 10 deg about z', up, {'chest': rot('z', 10)})
print('-- same, solver taking every heading from the root sensor')
for a in 'xyz':
    run(f'root sensor rotated 10 deg about {a}', up, {'root': rot(a, 10)}, heading_from_root=True)
print('-- the body at calibration differs from the assumed T-pose')
run('whole body tilted 10 deg sideways (about forward axis)', {k: rot('z', 10) for k in up})
run('pelvis/root tilted 10 deg, rest upright', dict(up, root=rot('z', 10)))
run('torso side-bent 10 deg (left shoulder up), arms level to floor', dict(up, chest=rot('z', 10)))
run('torso side-bent 10 deg (left shoulder down), arms level', dict(up, chest=rot('z', -10)))
run('upright torso, left arm 5 up, right arm 5 down', dict(up, L=rot('z', 5), R=rot('z', 5)))
run('upright torso, arms both 5 up (mirror)', dict(up, L=rot('z', 5), R=rot('z', -5)))
run('torso turned 10 deg about vertical, arms square to room', dict(up, chest=rot('y', 10)))
