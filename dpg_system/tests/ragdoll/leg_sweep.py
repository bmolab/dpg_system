import os, sys
from _env import ROOT, HERE
exec(open(os.path.join(HERE, 'leg_release.py')).read().split('def run(')[0].split('# which foot')[0])
import numpy as np
print('release left_leg at frame f: support at release, frames with support<0.5, root weight after 60 frames, pelvis drop after 90')
for rel in range(200, 400, 8):
    n=mk('left_leg, root'); sup=[]
    for f in range(rel-100, rel+90):
        n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy()
        if f==rel:
            for j in (1,4,7,10): n.weight_targets[j]=0.0
        n.execute()
        if f>=rel: sup.append(n.support_output.last)
        if f==rel+60: w60=n.weights[0]
    drop=np.asarray(n.trans_output.last,float)[2]-TRANS[rel+89][2]
    low=[i for i,v in enumerate(sup[:40]) if v<0.5]
    print('  f%3d  support %.2f  low frames %s  root w %.2f  drop %+.2f m' % (rel, sup[0], ('%d..%d (%d)' % (low[0],low[-1],len(low))) if low else 'none', w60, drop))
