import os, sys
from _env import ROOT, HERE
"""The operating cycle through the node: arm, release, catch, release again.

Written because three defects in a row were in the control surface -- group
names, the catch default, state carried across a re-engagement -- none of which
the physics tests could see.
"""
import sys
import numpy as np
from repro_trans import Stub, P

FR = 60.0
CAP_SPEED = 0.4

def cycle(free='all', blend=1.0, ramp=200.0, events=((60,'release'),(200,'catch'),(250,'release')), auto=False):
    n = Stub(free, weight=blend)
    n.weight_prop = P(blend); n._apply_weight_immediately(); n.ramp_prop = P(ramp)
    n.auto_release_prop = P(auto)
    ev = dict(events)
    rows = []
    for f in range(420):
        t = f/FR
        ps = np.zeros((24,3)); ps[16]=[0,0,-1.0]; ps[17]=[0,0,1.0]
        cap = np.array([CAP_SPEED*t, 0.0, 1.30])
        n.pose_input.v = ps; n.trans_input.v = cap
        if ev.get(f) == 'release': n._release()
        if ev.get(f) == 'catch':   n._catch()
        n.execute()
        rows.append((cap.copy(), np.asarray(n.trans_output.last, dtype=float).copy(),
                     n.weights[0], np.linalg.norm(n.sim.com_vel)))
    return rows

r = cycle()
cap = np.array([a for a,b,c,d in r]); out = np.array([b for a,b,c,d in r])
w   = np.array([c for a,b,c,d in r]); vel = np.array([d for a,b,c,d in r])
dist = np.linalg.norm(out-cap, axis=1)

print('arm -> release f60 -> catch f200 -> release again f214\n')
print('  armed (f0-59): follows the capture exactly      max deviation %.2e m'
      % dist[:60].max())
assert dist[:60].max() < 1e-9

print('  released (f60-199): departs from the capture    final deviation %.3f m'
      % dist[199])
assert dist[199] > 0.5

print('  caught (f199-249): back on the capture          deviation at f249 %.2e m'
      % dist[249])
assert dist[249] < 1e-9, dist[249]

print('  released again at f250, once the catch has completed:')
print('     release speed %.3f m/s   (capture is %.3f; a clean release starts there)'
      % (vel[250], CAP_SPEED))
assert abs(vel[250] - CAP_SPEED) < 0.25, vel[250]
print('     departs again, deviation at f419 %.3f m' % dist[419])
assert dist[419] > 0.5

print('\n  root weight through the cycle: %s'
      % np.round(w[[0, 61, 120, 205, 213, 215, 260]], 2))

print('\nblend_weight 0 leaves catch inert (and says so):')
r2 = cycle(blend=0.0)
w2 = np.array([c for a,b,c,d in r2])
print('  root weight after catch: %.2f (0 = still released, as documented)' % w2[260])

print('\n"joints" keeps the root driven, so translation follows the capture:')
r3 = cycle(free='joints')
cap3 = np.array([a for a,b,c,d in r3]); out3 = np.array([b for a,b,c,d in r3])
print('  max translation deviation over the whole cycle: %.2e m'
      % np.abs(out3-cap3).max())
assert np.abs(out3-cap3).max() < 1e-9

print("\nauto_release_unsupported must not fire before anything is released,")
print("and must not undo a catch (this body never touches the floor at all):")
ra = cycle(auto=True)
capa = np.array([a for a,b,c,d in ra]); outa = np.array([b for a,b,c,d in ra])
wa = np.array([c for a,b,c,d in ra]); dista = np.linalg.norm(outa-capa, axis=1)
print('  armed, root weight at f59: %.2f   deviation %.2e m' % (wa[59], dista[59]))
assert wa[59] == 1.0 and dista[59] < 1e-9, 'auto-release fired before any release'
print('  after catch, root weight at f249: %.2f   deviation %.2e m' % (wa[249], dista[249]))
assert wa[249] == 1.0 and dista[249] < 1e-9, 'auto-release undid the catch'

print('\nrelease is smooth: the root carries its speed on, and no more')
rj = cycle(free='root', events=((30,'release'),))
outj = np.array([b for a,b,c,d in rj])
step_before = np.linalg.norm(outj[30]-outj[29])
step_after  = np.linalg.norm(outj[31]-outj[30])
# Inherited motion plus one frame of gravity -- nothing else may appear.
expect = CAP_SPEED/60.0 + 0.5*9.81*(1/60.0)**2
print('  frame before release %.4f m, frame after %.4f m, expected at most %.4f m'
      % (step_before, step_after, expect))
assert step_after <= expect * 1.5, 'release jumps'
assert abs(step_after - step_before) < 0.002, 'release is not continuous'

print('\ncycle behaves')
