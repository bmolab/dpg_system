import os, sys
from _env import ROOT, HERE
"""Per-joint weight control through the node's message path and weights input."""
import sys
import numpy as np
from repro_trans import Stub, P, In

def mk(free='all'):
    n = Stub(free, weight=1.0); n.weight_prop = P(1.0); n._apply_weight_immediately()
    n.ramp_prop = P(60.0)
    n.message_handlers = {'weight': n._weight_message, 'release': n._release_message,
                          'catch': n._catch_message}
    return n
def step(n, k=1):
    for _ in range(k):
        n.pose_input.v = np.zeros((24, 3)); n.trans_input.v = np.array([0.0, 1.3, 0.0]); n.execute()
LARM, RARM, LEGS = [16, 18, 20], [17, 19, 21], [1, 4, 7, 10, 2, 5, 8, 11]

print('1. string message "weight left_arm 0" through check_for_messages')
n = mk(); step(n, 5)
assert n.check_for_messages('weight left_arm 0')
assert all(n.weight_targets[j] == 0.0 for j in LARM) and all(n.weight_targets[j] == 1.0 for j in RARM + LEGS + [0])
step(n, 30)
print('   left arm weights %s, right arm %s' % (np.round(n.weights[LARM], 2), np.round(n.weights[RARM], 2)))
assert np.all(n.weights[LARM] == 0.0) and np.all(n.weights[RARM] == 1.0)

print('2. a second message mid-swing does not reset the simulation')
sim_before = n.sim; rot_before = np.asarray(n.pose_output.last)[16].copy()
n.check_for_messages(['weight', 'right_arm', 0.0]); step(n, 1)
jump = np.linalg.norm(np.asarray(n.pose_output.last)[16] - rot_before)
print('   sim object identical: %s ; left shoulder moved %.4f rad in that frame' % (n.sim is sim_before, jump))
assert n.sim is sim_before and jump < 0.2

print('3. catch left_arm returns only the left arm')
n.check_for_messages('catch left_arm'); step(n, 30)
assert np.all(n.weights[LARM] == 1.0) and np.all(n.weights[RARM] == 0.0)
print('   left %s right %s' % (n.weights[LARM], n.weights[RARM]))

print('4. bare release / catch act on everything')
n.check_for_messages('release'); step(n, 30); assert np.all(n.weights[:22] == 0.0)
n.check_for_messages('catch'); step(n, 30); assert np.all(n.weights[:22] == 1.0)
print('   ok')

print('5. weights input: 22-array, then a 20-array in active order')
n = mk(); step(n, 3)
w22 = np.ones(22); w22[LEGS] = 0.3; w22[0] = 1.0
n.weights_input = In(w22, fresh=True); step(n, 1); n.weights_input.fresh_input = False
assert np.allclose(n.weight_targets[LEGS], 0.3) and n.weight_targets[0] == 1.0 and n.weight_targets[16] == 1.0
w20 = np.ones(20); w20[6] = 0.2          # active index 6 = left_hip -> SMPL 1
n.weights_input = In(w20, fresh=True); step(n, 1); n.weights_input.fresh_input = False
print('   active[6]=0.2 -> SMPL left_hip target %.2f, right_hip %.2f' % (n.weight_targets[1], n.weight_targets[2]))
assert n.weight_targets[1] == 0.2 and n.weight_targets[2] == 1.0

print('6. a joint outside the free set is refused, not added (no reset)')
n = mk('left_arm'); step(n, 3); sim_before = n.sim; t = n.weight_targets.copy()
n.check_for_messages('weight legs 0')
assert n.sim is sim_before and np.array_equal(t, n.weight_targets) and n.free_indices == LARM
print('   free set still %s, targets unchanged' % n.free_indices)

print('7. bad input is a message, not a crash')
n = mk(); n.check_for_messages('weight left_arm abc'); n.check_for_messages('weight'); n.check_for_messages('weight nosuch 0')
print('   ok')

print('8. default free set is everything, root included')
n = mk(); print('   root_free = %s, %d joints free' % (n.root_free, len(n.free_indices)))
assert n.root_free and len(n.free_indices) == 22
print('\nper-joint weights OK')
