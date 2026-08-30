import os, sys
from _env import ROOT, HERE
exec(open(os.path.join(HERE, 'leg_release.py')).read().split('def run(')[0].split('# which foot')[0])
import numpy as np, pybullet as p
from scipy.spatial.transform import Rotation as R
for wv in (0.95, 0.9, 0.7, 0.5, 0.3):
    for free in ('joints', 'all'):
        n=mk(free); n.auto_release_prop=P(False); errs=[]; jerrs=[]
        for f in range(100, 400):
            n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy()
            if f==150: n.weight_targets[:]=wv
            n.execute()
            if f>=200:
                b=n.sim.body; jp=b.joint_positions(); errs.append(np.linalg.norm(jp[0]-n.sim.prev_trans))
                js=p.getJointStatesMultiDof(b.body, list(range(21)), physicsClientId=b.cid)
                aa=np.asarray(POSES[f]).reshape(-1,3)
                jerrs.append(max(np.degrees(np.linalg.norm((R.from_quat(js[j-1][0]).inv()*R.from_rotvec(aa[j])).as_rotvec())) for j in range(1,22)))
        print('weight %.2f free=%-6s  pelvis off capture: mean %.3f max %.3f m   worst joint off capture: mean %5.1f max %5.1f deg' % (wv, free, np.mean(errs), np.max(errs), np.mean(jerrs), np.max(jerrs)))
