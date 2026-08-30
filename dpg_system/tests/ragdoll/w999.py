import os, sys
from _env import ROOT, HERE
exec(open(os.path.join(HERE, 'leg_release.py')).read().split('def run(')[0].split('# which foot')[0])
import numpy as np
for wv, free in ((0.999,'all'), (0.999,'joints'), (0.9,'all'), (0.5,'all')):
    n=mk(free); print('weights %.3f free=%s' % (wv, free))
    for f in range(100, 400):
        n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy()
        if f==150: n.weight_targets[:]=wv
        n.execute()
        if f in (149, 155, 170, 200, 300, 399):
            tr=np.asarray(n.trans_output.last,float); jp=n.sim.body.joint_positions()
            print('   f%d w0 %.3f trans %s  err %.3f m  head y %.2f  support %.2f' % (f, n.weights[0], np.round(tr,2), np.linalg.norm(jp[0]-n.sim.prev_trans), jp[15,1], n.support_output.last))
