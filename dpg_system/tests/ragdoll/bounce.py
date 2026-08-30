import os, sys
from _env import ROOT, HERE
exec(open(os.path.join(HERE, 'wpart.py')).read().split("for wv in")[0])
import pybullet as p
for wv in (0.9, 0.75, 0.5):
    n=mk('all'); n.auto_release_prop=P(False); vy=[]; dxz=[]
    for f in range(100, 500):
        n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy()
        if f==150: n.weight_targets[:]=wv
        n.execute()
        if f>=200:
            b=n.sim.body; bv=p.getBaseVelocity(b.body, physicsClientId=b.cid)[0]; vy.append(bv[1])
            jp=b.joint_positions(); d=jp[0]-np.asarray(n.sim.prev_trans); dxz.append(np.hypot(d[0], d[2]))
    vy=np.array(vy); hf=vy-np.convolve(vy, np.ones(9)/9, mode='same')
    print('w %.2f: pelvis horizontal off capture mean %.3f max %.3f m   vertical bobbing %.3f m/s rms' % (wv, np.mean(dxz), np.max(dxz), np.sqrt(np.mean(hf**2))))
