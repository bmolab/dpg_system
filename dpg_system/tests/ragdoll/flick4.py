import os, sys
from _env import ROOT, HERE
import sys, numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from repro_trans import Stub, P
F=os.path.join(ROOT, 'assets/motion_capture_files/Walk B17 - Walk 2 hop 2 walk_poses.npz')
d=np.load(F, allow_pickle=True); POSES=d['poses']; TRANS=d['trans']; FR=float(d['mocap_framerate']); BETAS=d['betas'][:10]; GENDER=str(d['gender'])
N=['pelvis','L_hip','R_hip','spine1','L_knee','R_knee','spine2','L_ank','R_ank','spine3','L_foot','R_foot','neck','L_col','R_col','head','L_sh','R_sh','L_elb','R_elb','L_wr','R_wr','L_hand','R_hand']
dt=1/FR
def mk(floor, selfc):
    n=Stub('all', weight=1.0); n.framerate=FR; n.betas=BETAS; n.gender=GENDER
    n.weight_prop=P(1.0); n._apply_weight_immediately(); n.ramp_prop=P(60.0)
    n.up_axis_prop=P('Y'); n.axis_perm_prop=P('x, z, -y'); n.self_collision_prop=P(selfc); n.floor_enable_prop=P(floor); return n
print('driven body at the release frames: lowest link point vs floor, and floor contact force:')
for rel in (500, 540, 600):
    n=mk(True,True)
    for f in range(rel-120, rel+1):
        n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy(); n.execute()
    b=n.sim.body; jp=b.joint_positions(); lo=int(np.argmin(jp[:,1]))
    cs=p.getContactPoints(bodyA=b.body, bodyB=b.plane, physicsClientId=b.cid)
    per={}
    for c in cs: per[N[c[3]+1]]=per.get(N[c[3]+1],0)+c[9]
    print('   f%d: lowest joint %s at y=%+.3f ; floor contacts: %s' % (rel, N[lo], jp[lo,1], ', '.join('%s %.0f N'%kv for kv in sorted(per.items(), key=lambda kv:-kv[1])[:4]) or 'none'))
def peak(rel, floor, selfc):
    n=mk(floor, selfc); pk=0; who=-1; prev=None
    for f in range(rel-120, rel+30):
        n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy()
        if f==rel: n._release()
        n.execute()
        if f>=rel:
            out=np.asarray(n.smpl_pose_output.last)
            if prev is not None:
                for j in range(1,22):
                    sp=np.linalg.norm((R.from_rotvec(prev[j]).inv()*R.from_rotvec(out[j])).as_rotvec())/dt
                    if sp>pk: pk,who=sp,j
            prev=out
    return pk, N[who]
print('peak joint speed after release:')
print('   frame    floor+self      no self         no floor        neither')
for rel in (500, 540, 600):
    r=[peak(rel,True,True), peak(rel,True,False), peak(rel,False,True), peak(rel,False,False)]
    print('   %4d  ' % rel + '  '.join('%5.1f (%-6s)' % x for x in r))
