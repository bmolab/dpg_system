import os, sys
from _env import ROOT, HERE
exec(open(os.path.join(HERE, 'osc.py')).read().split("def run(")[0])
def run(src, wv, frames=(100,500)):
    P_, T_ = src
    n=mk2('all', wv); lim=np.zeros(22); flips=np.zeros(22); prevd=np.zeros(22); cnt=0; exc=np.zeros(22)
    for f in range(0, frames[1]):
        n.pose_input.v=P_[f].copy(); n.trans_input.v=T_[f].copy()
        if f==frames[0]: n.weight_targets[:]=wv
        n.execute()
        if f>=frames[0]+40:
            cnt+=1; lt=np.asarray(n.sim.last_torque, dtype=float); lt=lt.reshape(lt.shape[0],-1).max(axis=1)[:22]; lim += (lt > 0.5)
            b=n.sim.body; js=p.getJointStatesMultiDof(b.body, list(range(21)), physicsClientId=b.cid)
            aa=np.asarray(P_[f]).reshape(-1,3); pa=np.asarray(P_[f-1]).reshape(-1,3)
            for j in range(1,22):
                capw=(R.from_rotvec(pa[j]).inv()*R.from_rotvec(aa[j])).as_rotvec()*FR
                d=np.linalg.norm(js[j-1][1])-np.linalg.norm(capw); exc[j]+=abs(d)
                if np.sign(d)!=np.sign(prevd[j]): flips[j]+=1
                prevd[j]=d
    return lim/cnt, flips/(cnt/FR), exc/cnt
for label, src in (('walk', (WP, WT)), ('cartwheel', (POSES, TRANS))):
    lim, flips, exc = run(src, 0.75)
    order=np.argsort(-exc)[:6]
    print('%s at 0.75 -- worst joints: excess motion rad/s, speed-error sign flips per second, fraction of frames in the limit branch' % label)
    for j in order: print('   %-7s excess %.2f  flips %5.1f/s  limit %3.0f%%' % (N[j], exc[j], flips[j], 100*lim[j]))
