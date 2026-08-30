import os, sys
from _env import ROOT, HERE
exec(open(os.path.join(HERE, 'wpart.py')).read().split("for wv in")[0])
import pybullet as p
for gc in (0.0, 1.0):
    for wv in (0.7, 0.5):
        n=mk('joints'); n.auto_release_prop=P(False); n.gravity_comp_prop=P(gc); acc={}
        for f in range(100, 400):
            n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy()
            if f==150: n.weight_targets[:]=wv
            n.execute()
            if f>=200:
                b=n.sim.body; js=p.getJointStatesMultiDof(b.body, list(range(21)), physicsClientId=b.cid); aa=np.asarray(POSES[f]).reshape(-1,3)
                for j in range(1,22): acc.setdefault(j,[]).append(np.degrees(np.linalg.norm((R.from_quat(js[j-1][0]).inv()*R.from_rotvec(aa[j])).as_rotvec())))
        m={j:np.mean(v) for j,v in acc.items()}
        print('gravity_comp %.0f w %.1f: mean error  hips %.1f knees %.1f  spine1 %.1f spine2 %.1f spine3 %.1f neck %.1f head %.1f  shoulders %.1f elbows %.1f' % (gc, wv, (m[1]+m[2])/2, (m[4]+m[5])/2, m[3], m[6], m[9], m[12], m[15], (m[16]+m[17])/2, (m[18]+m[19])/2))
