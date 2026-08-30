import os, sys
from _env import ROOT, HERE
import sys, numpy as np
from repro_trans import Stub, P
F=os.path.join(ROOT, 'assets/motion_capture_files/Walk B17 - Walk 2 hop 2 walk_poses.npz')
d=np.load(F, allow_pickle=True); POSES=d['poses']; TRANS=d['trans']; FR=float(d['mocap_framerate']); BETAS=d['betas'][:10]; GENDER=str(d['gender'])
def mk(free):
    n=Stub(free, weight=1.0); n.framerate=FR; n.betas=BETAS; n.gender=GENDER
    n.weight_prop=P(1.0); n._apply_weight_immediately(); n.ramp_prop=P(120.0)
    n.up_axis_prop=P('Y'); n.axis_perm_prop=P('x, z, -y'); n.auto_release_prop=P(True); return n
# which foot is the capture standing on, per frame (lower foot, near the floor)
n=mk('left_leg, root'); lows={}
for f in range(100, 420):
    n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy(); n.execute()
    jp=n.sim.body.joint_positions(); lows[f]=(jp[10,1]-n.sim.floor_level, jp[11,1]-n.sim.floor_level)
left_stance=[f for f in range(200,400) if lows[f][1]-lows[f][0]>0.06]
right_stance=[f for f in range(200,400) if lows[f][0]-lows[f][1]>0.06]
print('left-foot stance frames: %s...  right-foot stance: %s...' % (left_stance[:3], right_stance[:3]))
def run(rel, label):
    n=mk('left_leg, root')
    rows=[]
    for f in range(rel-120, rel+150):
        n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy()
        if f==rel: n.check_for_messages('release left_leg') if hasattr(n,'message_handlers') else None
        if f==rel:
            for j in (1,4,7,10): n.weight_targets[j]=0.0
        n.execute()
        rows.append((f-rel, n.support_output.last, n.weights[0], np.asarray(n.trans_output.last,float)[2]-TRANS[f][2]))
    r=np.array(rows)
    print('  %s' % label)
    print('     t(frames)  support   root weight   pelvis height vs capture')
    for k in (0,10,20,40,80,120,149):
        print('     %+4d       %.2f       %.2f         %+.3f m' % tuple(r[120+k]))
run(right_stance[len(right_stance)//2], 'release LEFT leg while standing on the RIGHT foot (should stay up):')
run(left_stance[len(left_stance)//2],  'release LEFT leg while standing on the LEFT foot (should give way):')
# sanity: the capture FK used by the support test must agree with the driven body
n=mk('left_leg, root'); n.pose_input.v=POSES[300].copy(); n.trans_input.v=TRANS[300].copy(); n.execute()
jp=n.sim.body.joint_positions()
import numpy as _np
aa=_np.asarray(n.sim._last_aa) if hasattr(n.sim,'_last_aa') else None
