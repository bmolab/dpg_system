import os, sys
from _env import ROOT, HERE
exec(open(os.path.join(HERE, 'wpart.py')).read().split("for wv in")[0])
import pybullet as p
rng=np.random.default_rng(3)
# IMU-like root height error: slow wander +-2 cm, plus 5 cm dips for ~10 frames now and then
F=600; wander=np.convolve(rng.normal(0,0.02,F+40), np.ones(40)/40, mode='same')[:F]
dips=np.zeros(F)
for s in rng.integers(150, F-20, 6): dips[s:s+10]-=0.05
jit=wander+dips
for wv, label in ((1.0,'root driven'), (0.9,'root 0.9'), (0.7,'root 0.7')):
    n=mk('all'); n.auto_release_prop=P(False); fl=[]; py=[]; pref=[]
    for f in range(100, 100+F):
        tr=TRANS[f].copy(); tr[2]+=jit[f-100]          # source file is Z-up: z is height
        n.pose_input.v=POSES[f].copy(); n.trans_input.v=tr
        if f==150: n.weight_targets[:]=wv
        n.execute()
        if f>=200:
            fl.append(n.sim.floor_level); jp=n.sim.body.joint_positions(); py.append(jp[0,1]); pref.append(TRANS[f][2])
    fl=np.array(fl); py=np.array(py); pref=np.array(pref)
    err=py-pref; err-=err.mean()
    print('%-12s floor plane moved: range %.3f m, std %.3f   pelvis height vs clean capture: rms %.3f m, worst %.3f' % (label, fl.max()-fl.min(), fl.std(), np.sqrt(np.mean(err**2)), np.abs(err).max()))
