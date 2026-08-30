import os, sys
from _env import ROOT, HERE
exec(open(os.path.join(HERE, 'leg_release.py')).read().split('def run(')[0].split('# which foot')[0])
import numpy as np
for label, auto in (('manual release root+left_leg (weights 0)', False), ('auto: release left_leg, auto_release lets root go', True)):
    n=mk('left_leg, root'); n.auto_release_prop=P(auto); rel=250
    print(label)
    for f in range(rel-100, rel+300):
        n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy()
        if f==rel:
            for j in ((0,1,4,7,10) if not auto else (1,4,7,10)): n.weight_targets[j]=0.0
        n.execute()
        if f>=rel and (f-rel)%30==0:
            jp=n.sim.body.joint_positions(); fl=n.sim.floor_level
            print('  +%3d  root w %.2f  pelvis y-floor %+.2f  L_foot %+.2f  R_foot %+.2f  head %+.2f  support %.2f' % (f-rel, n.weights[0], jp[0,1]-fl, jp[10,1]-fl, jp[11,1]-fl, jp[15,1]-fl, n.support_output.last))
