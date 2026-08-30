import os, sys
from _env import ROOT, HERE
import sys, numpy as np
from scipy.spatial.transform import Rotation as R
from repro_trans import Stub, P
F=os.path.join(ROOT, 'assets/motion_capture_files/0007_Cartwheel001_poses.npz')
d=np.load(F, allow_pickle=True); POSES=d['poses']; TRANS=d['trans']; FR=float(d['mocap_framerate']); BETAS=d['betas'][:10]; GENDER=str(d['gender'])
N=['pelvis','L_hip','R_hip','spine1','L_knee','R_knee','spine2','L_ank','R_ank','spine3','L_foot','R_foot','neck','L_col','R_col','head','L_sh','R_sh','L_elb','R_elb','L_wr','R_wr','L_hand','R_hand']
dt=1/FR
def mk(limits=1.0, selfc=True, floor=True):
    n=Stub('all', weight=1.0); n.framerate=FR; n.betas=BETAS; n.gender=GENDER
    n.weight_prop=P(1.0); n._apply_weight_immediately(); n.ramp_prop=P(60.0)
    n.up_axis_prop=P('Y'); n.axis_perm_prop=P('x, z, -y'); n.self_collision_prop=P(selfc); n.floor_enable_prop=P(floor)
    n.limit_stiffness_prop=P(limits); return n
# find the upside-down section: head below pelvis in the driven body
n=mk(); ups=[]
for f in range(0, len(POSES)):
    n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy(); n.execute()
    jp=n.sim.body.joint_positions()
    if jp[15,1] < jp[0,1]: ups.append(f)
print('upside-down frames: %d..%d (fps %.0f)' % (ups[0], ups[-1], FR))
def jvel(a, b):
    return np.array([(R.from_rotvec(a[j]).inv()*R.from_rotvec(b[j])).as_rotvec()/dt for j in range(22)])
def run(rel, **kw):
    n=mk(**kw); outs=[]
    for f in range(rel-150, rel+20):
        n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy()
        if f==rel: n._release()
        n.execute()
        if f>=rel-2: outs.append(np.asarray(n.smpl_pose_output.last).reshape(-1,3)[:22].copy())
    v0=jvel(outs[0], outs[2])/2      # capture velocity into the release (still driven)
    worst=(0.0, None, 0)
    for t in range(3, len(outs)):
        v=jvel(outs[t-1], outs[t])
        for j in range(1,22):
            n0=np.linalg.norm(v0[j])
            if n0<1.0: continue
            proj=np.dot(v[j], v0[j])/n0   # component along the incoming velocity
            if proj < 0 and -proj > worst[0]:
                worst=(-proj, N[j], t-2)
    return worst
print('worst reversal after release: speed against incoming direction (rad/s), joint, frames after release')
print('   frame   baseline               no limits              no self-coll           no floor')
for rel in range(ups[0], ups[-1]+1, max(1,(ups[-1]-ups[0])//8)):
    rs=[run(rel), run(rel, limits=0.0), run(rel, selfc=False), run(rel, floor=False)]
    print('   %4d   ' % rel + '  '.join('%5.1f %-7s @%2d   ' % r for r in rs))
