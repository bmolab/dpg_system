"""Drive FingerZonesNode.handle_touch with stubbed widgets: no device, no GUI."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np
from dpg_system.erae_nodes import FingerZonesNode, ACTION_DOWN, ACTION_MOVE, ACTION_UP
import threading


class P:
    def __init__(self, v): self.v = v
    def __call__(self): return self.v
    def set(self, v): self.v = v

class Out:
    def __init__(self): self.last = None; self.log = []
    def send(self, v): self.last = v; self.log.append(v)


class StubZones(FingerZonesNode):
    def __init__(self, hands='right'):
        self.lock = threading.Lock()
        self.centres = np.zeros((0, 2), dtype=np.float32); self.names = []
        self.assigned = {}; self.positions = {}; self.lifted = {}; self.held_drawn = None
        self.in_loading_process = False
        self.touch_out = Out(); self.voices_out = Out(); self.count_out = Out(); self.image_out = Out()
        self.hands = P(hands); self.source = P('erae')
        self.width = P(43); self.height = P(25)
        self.centre_x = P(21.0); self.centre_y = P(12.0); self.spread = P(11.0)
        self.separation = P(24.0); self.rotation = P(0.0); self.radius = P(3.0)
        self.catch_radius = P(0.0); self.reacquire_radius = P(3.0); self.reacquire_time = P(0.3)
        self.named = P(False); self.show_held = P(True)
        self.zone_color = P([0.1, 0.3, 0.6, 1.0]); self.held_color = P([0.2, 0.7, 1.0, 1.0]); self.back_color = P([0, 0, 0, 1])
        self.compute_layout(); self.send_image()


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        sys.exit(1)


n = StubZones('right')
print('right hand centres:')
for name, (x, y) in zip(n.names, n.centres):
    print('  %-7s %5.1f %5.1f' % (name, x, y))
check(all(0 <= x < 43 and 0 <= y < 25 for x, y in n.centres), 'all home circles inside a full-pad zone')
img = n.image_out.last
check(img is not None and img.shape == (25, 43, 3) and img.dtype == np.uint8, 'zones image is 25 x 43 x 3 uint8')

# land on the middle finger's circle first, then the thumb: ids follow the zone, not the order
mx, my = n.centres[2]; tx, ty = n.centres[0]
n.handle_touch(0, ACTION_DOWN, mx + 0.5, my, 0.5)
check(n.touch_out.last[0] == 2, 'first touch on the middle circle is finger 2')
n.handle_touch(1, ACTION_DOWN, tx, ty - 1.0, 0.4)
check(n.touch_out.last[0] == 0, 'second touch on the thumb circle is finger 0')
check(n.count_out.last == 2, 'count 2')
# wander: the middle finger drags over the ring circle and keeps its name
rx, ry = n.centres[3]
n.handle_touch(0, ACTION_MOVE, rx, ry, 0.6)
check(n.touch_out.last[0] == 2 and n.touch_out.last[1] == ACTION_MOVE, 'a held finger keeps its name while wandering')
v = n.voices_out.last
check(v.shape == (5, 4) and v[2, 0] == 1 and abs(v[2, 1] - rx) < 1e-5 and v[3, 0] == 0, 'voices row 2 follows the wandering finger, row 3 stays free')
# a third touch on the ring circle, now occupied by finger 2 spatially but not by name: nearest FREE circle is ring
n.handle_touch(2, ACTION_DOWN, rx + 0.2, ry, 0.3)
check(n.touch_out.last[0] == 3, 'landing on the ring circle gives the ring finger, still free by name')
# same circle again: ring is taken, so the next nearest free (pinky or index) wins
n.handle_touch(3, ACTION_DOWN, rx + 0.2, ry, 0.3)
check(n.touch_out.last[0] in (1, 4), 'a second landing on a claimed circle takes the nearest free one')
# lift everything
for t in (0, 1, 2, 3):
    n.handle_touch(t, ACTION_UP, 0, 0, 0)
check(n.count_out.last == 0 and n.voices_out.last[:, 0].sum() == 0, 'all lifted')

# re-acquisition: the middle finger lifts far from home (over the ring circle) and retouches there quickly
n.handle_touch(5, ACTION_DOWN, mx, my, 0.5)
n.handle_touch(5, ACTION_MOVE, rx, ry, 0.5)
n.handle_touch(5, ACTION_UP, rx, ry, 0.0)
n.handle_touch(6, ACTION_DOWN, rx + 0.5, ry, 0.5)
check(n.touch_out.last[0] == 2, 'a quick retouch where the middle finger lifted is the middle finger again, not the ring')
n.handle_touch(6, ACTION_UP, rx, ry, 0.0)
# ...but after the reacquire time it is the ring finger by geometry
n.reacquire_time.set(0.05); time.sleep(0.1)
n.handle_touch(7, ACTION_DOWN, rx + 0.5, ry, 0.5)
check(n.touch_out.last[0] == 3, 'a late retouch goes by the circle again')
n.handle_touch(7, ACTION_UP, rx, ry, 0.0)

# catch radius: a landing far from every circle is unassigned (-1) and still comes out
n.catch_radius.set(2.0)
n.handle_touch(8, ACTION_DOWN, 1.0, 24.0, 0.5)
check(n.touch_out.last[0] == -1, 'beyond the catch radius the touch is -1')
check(n.count_out.last == 0, 'an unassigned touch does not count')
n.handle_touch(8, ACTION_UP, 1.0, 24.0, 0.0)
n.catch_radius.set(0.0)

# names
n.named.set(True)
n.handle_touch(9, ACTION_DOWN, tx, ty, 0.5)
check(n.touch_out.last[0] == 'thumb', "'finger as name' gives 'thumb'")
n.handle_touch(9, ACTION_UP, tx, ty, 0.0)
n.named.set(False)

# both hands: ten circles, left mirrored, image highlights held ones
b = StubZones('both')
check(len(b.names) == 10 and b.names[0] == 'left thumb' and b.names[9] == 'right pinky', 'both hands: ten named fingers')
lt = b.centres[0]; rt = b.centres[5]
check(lt[0] > b.centres[4][0] and rt[0] < b.centres[9][0], 'left thumb is right of the left pinky; right thumb left of the right pinky')
print('both-hands centres:')
for name, (x, y) in zip(b.names, b.centres):
    print('  %-12s %5.1f %5.1f' % (name, x, y))
before = b.image_out.last.copy()
b.handle_touch(0, ACTION_DOWN, rt[0] + 0.6, rt[1], 0.5)
after = b.image_out.last
check(b.touch_out.last[0] == 5, 'right thumb is finger 5 in both-hands mode')
check(not np.array_equal(before, after), 'the image changes when a finger is held')
check(b.voices_out.last.shape == (10, 4), 'voices is 10 x 4 for both hands')

# the string action form from erae's 'action as name'
vals = FingerZonesNode.touch_values([0, 0, 'down', 3.0, 4.0, 0.5])
check(vals == [0, 0, 0, 3.0, 4.0, 0.5], "'down' as a word parses to 0")
print('all passed')
