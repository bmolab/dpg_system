import os, sys
from _env import ROOT, HERE
exec(open(os.path.join(HERE, 'wpart.py')).read().split("for wv in")[0])
import pybullet as p
A, B = 150, 400   # loop this span
def run(wv, free='all', loops=3):
    n=mk(free); n.auto_release_prop=P(False)
    rows=[]
    for lp in range(loops):
        for f in range(A, B):
            n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy()
            if lp==0 and f==A+40: n.weight_targets[:]=wv
            n.execute()
            if lp>0 and f<A+60:
                b=n.sim.body; js=p.getJointStatesMultiDof(b.body, list(range(21)), physicsClientId=b.cid)
                sp=max(np.linalg.norm(s[1]) for s in js)
                aa=np.asarray(POSES[f]).reshape(-1,3)
                err=max(np.degrees(np.linalg.norm((R.from_quat(js[j-1][0]).inv()*R.from_rotvec(aa[j])).as_rotvec())) for j in range(1,22))
                jp=b.joint_positions(); rows.append((lp, f-A, sp, err, np.linalg.norm(jp[0]-n.sim.prev_trans)))
    r=np.array(rows)
    for lp in range(1, loops):
        m=r[r[:,0]==lp]
        settle=next((int(m[i,1]) for i in range(len(m)) if np.all(m[i:,3]<15.0)), 60)
        print('   loop %d: peak joint speed %5.1f rad/s   worst error %5.0f deg   pelvis off %5.2f m   error <15 deg after %d frames' % (lp, m[:,2].max(), m[:,3].max(), m[:,4].max(), settle))
print('capture loops back %d->%d (trans jumps %.2f m); everything at the weight below:' % (B, A, np.linalg.norm(TRANS[B]-TRANS[A])))
for wv, free in ((1.0,'all'), (0.8,'all'), (0.0,'arms')):
    print(' weight %.1f free=%s' % (wv, free)); run(wv, free)
