import os, sys
from _env import ROOT, HERE
"""Stage 1: the Bullet body's joints must sit where the processor's FK puts them."""
import sys, numpy as np
from scipy.spatial.transform import Rotation as R
from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions
from dpg_system.smpl_bullet import BulletBody, build_urdf
HERE=os.path.join(ROOT, 'dpg_system')
proc=SMPLProcessor(framerate=120, betas=np.zeros(10), gender='neutral', total_mass_kg=75.0, model_path=HERE)
opts=SMPLProcessingOptions(input_type='axis_angle', input_up_axis='Y', axis_permutation=None, quat_format='xyzw', dt=1/120)
def fk(pose, trans):
    t,aa,q=proc._prepare_trans_and_pose(pose.reshape(1,24,3).copy(), np.asarray(trans).reshape(1,3), opts)
    wp,_,_=proc._compute_forward_kinematics(t,q); return wp[0,:24]
body=BulletBody(proc)
print('URDF: %d lines, %d links' % (build_urdf(proc).count('\n'), len(body.link_of)+1))
rng=np.random.default_rng(0); worst=0.0; worst_rt=0.0
for k in range(25):
    pose=np.zeros((24,3)) if k==0 else rng.normal(0,0.7,(24,3)); trans=rng.normal(0,1,3)
    body.set_pose(pose, trans)
    err=np.abs(body.joint_positions()-fk(pose,trans)).max(); worst=max(worst,err)
    aa,tr=body.read_pose()
    rt=max(max(float((R.from_rotvec(aa[j])*R.from_rotvec(pose[j]).inv()).magnitude()) for j in range(24)), float(np.abs(tr-trans).max()))
    worst_rt=max(worst_rt,rt)
print('joint world positions vs processor FK, rest + 24 random poses: worst %.2e m' % worst)
print('pose round trip through the engine (set_pose -> read_pose): worst %.2e' % worst_rt)
assert worst < 1e-5 and worst_rt < 1e-6
print('frames agree')
