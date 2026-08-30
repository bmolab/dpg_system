import os, sys
from _env import ROOT, HERE
exec(open(os.path.join(HERE, 'heli.py')).read().split("for free, wv in")[0])
FW=os.path.join(ROOT, 'assets/motion_capture_files/Walk B17 - Walk 2 hop 2 walk_poses.npz')
dw=np.load(FW, allow_pickle=True); WP=dw['poses']; WT=dw['trans']
def run(src, wv, damp, kp=None, frames=(100,500)):
    P_, T_ = src
    n=mk2('all', wv); err_hist=[]; sp_hist=[]
    for f in range(0, frames[1]):
        n.pose_input.v=P_[f].copy(); n.trans_input.v=T_[f].copy()
        if kp is not None: n.params.motor_kp=kp
        if f==frames[0]: n.weight_targets[:]=wv
        n.execute()
        if f==frames[0]+20 and damp>0:
            b=n.sim.body
            for j in range(1,22): p.changeDynamics(b.body, j-1, jointDamping=damp, physicsClientId=b.cid)
        if f>=frames[0]+40:
            b=n.sim.body; js=p.getJointStatesMultiDof(b.body, list(range(21)), physicsClientId=b.cid)
            aa=np.asarray(P_[f]).reshape(-1,3)
            err_hist.append([np.linalg.norm((R.from_quat(js[j-1][0]).inv()*R.from_rotvec(aa[j])).as_rotvec()) for j in range(1,22)])
            sp_hist.append([np.linalg.norm(js[j-1][1]) for j in range(1,22)])
    e=np.array(err_hist); s=np.array(sp_hist)
    # oscillation: sign flips of the error derivative per second, on the joint with the largest error
    j=int(np.argmax(e.mean(0))); de=np.diff(e[:,j]); flips=np.sum(np.sign(de[1:])!=np.sign(de[:-1]))/(len(de)/FR)
    return np.degrees(e.mean()), np.degrees(e.max()), s.mean(), flips, N[j+1]
print('everything at a partial weight: mean/max joint error (deg), mean joint speed (rad/s), error zig-zags per second on the worst joint')
for label, src in (('walk', (WP, WT)), ('cartwheel', (POSES, TRANS))):
    for wv in (0.75, 0.6):
        for damp, kp in ((0,None), (2,None), (5,None), (15,None), (0,0.2), (5,0.2)):
            r=run(src, wv, damp, kp)
            print('  %-9s w %.2f  damping %2d  kp %s   err %5.1f / %5.1f   speed %5.2f   zigzag %5.1f/s (%s)' % (label, wv, damp, kp if kp else '.6 ', *r))
