import os, sys
from _env import ROOT, HERE
exec(open(os.path.join(HERE, 'rev_sweep.py')).read().split('def jvel(')[0].split('# find the upside-down')[0])
import pybullet as p
def mk2(free, w):
    n=Stub(free, weight=1.0); n.framerate=FR; n.betas=BETAS; n.gender=GENDER
    n.weight_prop=P(1.0); n._apply_weight_immediately(); n.ramp_prop=P(120.0)
    n.up_axis_prop=P('Y'); n.axis_perm_prop=P('x, z, -y'); n.auto_release_prop=P(False); return n
for free, wv in (('arms',0.9), ('arms',0.7), ('all',0.9), ('all',0.7)):
    n=mk2(free, wv); print('free=%s weight %.1f' % (free, wv))
    for f in range(0, 900):
        n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy()
        if f==100: n.weight_targets[:]=wv
        n.execute()
        if f>=150 and f%75==0:
            b=n.sim.body; js=p.getJointStatesMultiDof(b.body, list(range(21)), physicsClientId=b.cid)
            aa=np.asarray(POSES[f]).reshape(-1,3)
            sp=[np.linalg.norm(s[1]) for s in js]; err=[np.degrees(np.linalg.norm((R.from_quat(js[j-1][0]).inv()*R.from_rotvec(aa[j])).as_rotvec())) for j in range(1,22)]
            i=int(np.argmax(sp)); k=int(np.argmax(err))
            print('   f%3d  fastest joint %-6s %5.1f rad/s   worst error %-6s %5.0f deg   pelvis off %.2f m' % (f, N[i+1], sp[i], N[k+1], err[k], np.linalg.norm(b.joint_positions()[0]-n.sim.prev_trans)))
