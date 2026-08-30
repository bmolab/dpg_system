import os, sys
from _env import ROOT, HERE
exec(open(os.path.join(HERE, 'ring.py')).read().split("print('elbow")[0])
print('shove ring-down: peak deg / settle s / swings')
for g in (0.0, 0.5, 1.0, 2.0):
    row=[]
    for wv in (0.9, 0.5):
        row.append(ring(wv, g, j=12, link=15, torque=(3,0,0))); row.append(ring(wv, g))
    print('  partial_damping %.1f   neck@.9 %5.1f/%.2f/%2d   elbow@.9 %5.1f/%.2f/%2d   neck@.5 %5.1f/%.2f/%2d   elbow@.5 %5.1f/%.2f/%2d' % (g, *row[0], *row[1], *row[2], *row[3]))
