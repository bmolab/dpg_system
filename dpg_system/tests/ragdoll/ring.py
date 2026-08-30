import os, sys
from _env import ROOT, HERE
import sys; sys.argv=['x']
exec(open(os.path.join(HERE, 'test_cycle.py')).read().split('r = cycle()')[0])
import numpy as np, pybullet as p
from scipy.spatial.transform import Rotation as R
def ring(wv, g, kd=0.3, kp=0.6, j=18, link=18, torque=(0,0,60), fr=120.0):
    n = Stub('all', weight=1.0); n.weight_prop = P(1.0); n._apply_weight_immediately(); n.ramp_prop = P(120.0); n.auto_release_prop=P(False); n.framerate=fr
    n.motor_kd_prop=P(kd); n.motor_kp_prop=P(kp)
    ps=np.zeros((24,3)); ps[16]=[0,0,-1.0]; ps[17]=[0,0,1.0]
    errs=[]
    for f in range(int(fr*3)):
        n.pose_input.v=ps; n.trans_input.v=np.array([0.0,0,1.30]); n.partial_damping_prop=P(g)
        if f==int(fr*0.5): n.weight_targets[:]=wv
        if int(fr*1.0) <= f < int(fr*1.0)+6:   # a 50 ms shove on the forearm
            b=n.sim.body; p.applyExternalTorque(b.body, link-1, list(torque), p.LINK_FRAME, physicsClientId=b.cid)
        n.execute()
        if f>=int(fr*1.0):
            b=n.sim.body; st=p.getJointStateMultiDof(b.body, j-1, physicsClientId=b.cid)
            errs.append(np.degrees(np.linalg.norm((R.from_quat(st[0]).inv()*R.from_rotvec(ps[j])).as_rotvec())))
    e=np.array(errs)
    settle=next((i for i in range(len(e)) if np.all(e[i:]<2.0)), len(e))/fr
    peaks=sum(1 for i in range(1,len(e)-1) if e[i]>e[i-1] and e[i]>e[i+1] and e[i]>2.0)
    return e.max(), settle, peaks
print('elbow at a partial weight, forearm shoved for 50 ms: peak error (deg), time to settle under 2 deg (s), number of swings above 2 deg')
for wv in (0.75, 0.5):
    for g, kd, kp in ((0,0.3,0.6), (0,0.9,0.6), (0,0.9,0.3), (1,0.3,0.6), (3,0.3,0.6), (10,0.3,0.6), (3,0.9,0.3)):
        r=ring(wv, g, kd, kp)
        print('  w %.2f  partial_damping %2d  kd %.1f kp %.1f   peak %5.1f  settle %.2f s  swings %d' % (wv, g, kd, kp, *r))
