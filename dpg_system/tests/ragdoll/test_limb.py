import os, sys
from _env import ROOT, HERE
import sys
import numpy as np
from dpg_system.smpl_processor import SMPLProcessor
from dpg_system.smpl_ragdoll import SMPLRagdollSim, RagdollParams
HERE=os.path.join(ROOT, 'dpg_system'); FR=120.0
proc = SMPLProcessor(framerate=FR, betas=np.zeros(10), gender='neutral',
                     total_mass_kg=75.0, model_path=HERE)
p=RagdollParams(); p.dt=1/FR; p.substeps=4
print('free left arm, root driven, pelvis lowered so the arm must reach the floor:')
for pelvis_y in (0.25, 0.15, 0.05):
    for floor in (False, True):
        p.floor_enable = floor
        sim=SMPLRagdollSim(proc); sim.set_free_joints([16,18,20]); sim.set_root_free(False)
        w=np.ones(22); w[[16,18,20]]=0.0
        ps=np.zeros((24,3)); tr=np.array([0.0,pelvis_y,0.0])
        lows=[]
        for f in range(int(4.0*FR)):
            sim.advance(ps, tr, w, p)
            aa=ps.copy()
            for j,v in sim.local_rot.items(): aa[j]=v.as_rotvec()
            wp,_=sim._full_fk(aa, tr)
            lows.append(min(wp[26,1], wp[20,1]))     # finger tip / wrist
        lows=np.array(lows)
        tag = 'floor ON ' if floor else 'floor OFF'
        print('  pelvis %.2f m  %s : lowest arm point min %+.4f  final %+.4f m'
              % (pelvis_y, tag, lows.min(), lows[-1]))
print('\n(with the floor on, the arm must rest at about its 0.02 m tip radius')
print(' rather than sinking to where it hangs with the floor off)')
