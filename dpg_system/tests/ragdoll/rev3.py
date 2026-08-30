import os, sys
from _env import ROOT, HERE
exec(open(os.path.join(HERE, 'rev_sweep.py')).read().split('def jvel(')[0].split('# find the upside-down')[0])
n=mk(); H=[]
for f in range(len(POSES)):
    n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy(); n.execute()
    jp=n.sim.body.joint_positions(); H.append((jp[0,1], jp[15,1], jp[20,1], jp[21,1]))
H=np.array(H)
hs=[f for f in range(len(H)) if H[f,1] < H[f,0]-0.3 and max(H[f,2],H[f,3]) < H[f,0]-0.3]
# group into runs
runs=[]; 
for f in hs:
    if runs and f==runs[-1][1]+1: runs[-1][1]=f
    else: runs.append([f,f])
print('file %d frames; handstand runs (head and hands well below pelvis): %s' % (len(H), runs))
def jvel(a, b):
    return np.array([(R.from_rotvec(a[j]).inv()*R.from_rotvec(b[j])).as_rotvec()/dt for j in range(22)])
def run(rel):
    n=mk(); outs=[]
    for f in range(rel-150, rel+12):
        n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy()
        if f==rel: n._release()
        n.execute()
        if f>=rel-3: outs.append(np.asarray(n.smpl_pose_output.last).reshape(-1,3)[:22].copy())
    vin=jvel(outs[0], outs[3])/3
    worst=(0,None,0,0)
    for t in range(4, len(outs)):
        v=jvel(outs[t-1], outs[t])
        for j in range(1,22):
            sp=np.linalg.norm(v[j]); n0=np.linalg.norm(vin[j])
            proj=np.dot(v[j], vin[j])/max(n0,1e-6)
            if proj < -worst[0]: worst=(-proj, N[j], t-3, n0)
    return worst
print('releases through the handstands: strongest early (<=8 frames) local-joint reversal: rad/s against incoming, joint, frame, incoming speed')
for a,b in runs[:3]:
    for rel in range(a-8, b+9, 4):
        w=run(rel)
        print('  f%4d  %5.1f rad/s  %-7s @%d  (incoming %4.1f)%s' % (rel, w[0], w[1], w[2], w[3], '  <--' if w[0]>8 else ''))
