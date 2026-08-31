import os, sys
from _env import ROOT, HERE
exec(open(os.path.join(HERE, 'wpart.py')).read().split("for wv in")[0])
import pybullet as p
for sense in (False, True):
    n=mk('all'); n.auto_release_prop=P(False); n.contact_sense_prop=P(sense)
    sup=[]; swing_push=0; frames=0
    for f in range(100, 500):
        n.pose_input.v=POSES[f].copy(); n.trans_input.v=TRANS[f].copy()
        if f==150: n.weight_targets[:]=0.8
        n.execute()
        if 260<=f<300: sup.append(n.support_output.last)     # the hop: both feet airborne
        if f>=200 and n.sim.contact_intensity is not None:
            for g, links in (('LF',(6,9)),('RF',(7,10))):
                if n.sim.contact_intensity[g] < 0.2:
                    b=n.sim.body
                    fN=sum(c[9] for li in links for c in p.getContactPoints(bodyA=b.body, linkIndexA=li, bodyB=b.plane, physicsClientId=b.cid))
                    frames+=1; swing_push += (fN > 50.0)
    tag = ('swing-foot frames pushing >50 N on the floor: %d of %d' % (swing_push, frames)) if frames else 'no intensity available'
    print('contact_sense %-5s: support during the airborne hop  mean %.2f  min %.2f   %s' % (sense, np.mean(sup), np.min(sup), tag))
