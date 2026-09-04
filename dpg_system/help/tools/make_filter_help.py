import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# --------------------------------------------------------------------- filter
body = """The filter node smooths a jittery stream by letting it change only gradually.

Each time a value arrives, the node moves its own held value part of the way 
towards it, rather than jumping straight there. The degree decides how far: 
at 0 it goes all the way, at 0.99 it barely budges. What comes out is a version 
of the input with the sudden movements worn off.

smooth is the same node under a friendlier name.

This is the cheapest useful filter there is - one multiply and one add per value - 
and it is usually the right first thing to reach for on a noisy sensor. 
Its cost is lag: the smoother you make it, the further behind the real signal 
it runs. That trade is the whole story of this node.

It works on single numbers, NumPy arrays and PyTorch tensors. 
Arrays are smoothed element by element, each with its own history, 
and the node resizes itself if the array shape changes.

SYNTAX:
filter <degree: float>
smooth <degree: float>

EXAMPLE:
filter 0.9

INPUTS and PARAMETERS:

in:
The data to be smoothed. Receiving data here triggers the node.

degree:
How much of the old value to keep, from 0.0 to 1.0. The default is 0.9. 

  0.0    no smoothing at all - the input passes straight through
  0.5    light smoothing, barely any lag
  0.9    noticeable smoothing, noticeable lag
  0.99   heavy smoothing; slow drifts survive, everything else is flattened

Values outside 0 to 1 are clamped, so the filter cannot be made to run away.

OUTPUTS: 

out:
The smoothed value.

CHOOSING A DEGREE:
There is no correct value - it depends on how fast the thing you care about 
moves, and how fast the noise moves. If the two are similar, no setting of this 
node will separate them, and you need a filter that knows something more about 
the signal. See one_euro_filter, which varies its own smoothing with speed, 
and band_pass, which selects by frequency."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 4.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 4.0, 0.7)},
    {'key': 'tog', 'init': 'toggle', 'pos': (185, 62), 'w': 45, 'h': 42,
     'props': {'': True}},
    {'key': 'met', 'init': 'metro 16', 'pos': (185, 110), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 16.0, 'units': 'milliseconds'}},
    {'key': 'rnd', 'init': 'random.gauss 0.0 0.25', 'pos': (185, 190), 'w': 175, 'h': 100},
    {'key': 'add', 'init': '+ 0.0', 'pos': (30, 232), 'w': 120, 'h': 70,
     'props': {'operand': 0.0}},
    {'key': 'c0', 'comment': True, 'text': 'a clean wave plus noise', 'pos': (30, 310)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 348), 'w': 208, 'h': 176,
     'props': PLOT(-1.5, 1.5)},
    {'key': 'flt', 'init': 'filter 0.9', 'pos': (30, 545), 'w': 130, 'h': 70,
     'props': {'degree': 0.9}},
    {'key': 'p2', 'init': 'plot', 'pos': (30, 630), 'w': 208, 'h': 176,
     'props': PLOT(-1.5, 1.5)},
    {'key': 'c1', 'comment': True, 'text': 'drag degree: watch noise fall and lag grow',
     'pos': (255, 565)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('tog', '', 'met', 'on'), ('met', '', 'rnd', 'trigger'),
         ('sig', '', 'add', 'in'), ('rnd', 'out', 'add', 'operand'),
         ('add', 'result', 'p1', 'y'),
         ('add', 'result', 'flt', 'in'), ('flt', 'out', 'p2', 'y')]
print(build('filter', 'filter - smooth a jittery stream', body, demo, links,
            demo_width=560, text_width=800, text_height=680))

# ----------------------------------------------------------------------- diff
body = """The diff node reports how much the input CHANGED since the last value, 
rather than what it is.

Each time a value arrives, the node subtracts the previous one and sends the 
difference. A steady input gives zero. A rising input gives a positive number, 
a falling one a negative number, and the size tells you how fast.

This turns a position into a movement, a count into a rate, an angle into a 
turning speed. It is the counterpart of accumulate, which goes the other way.

It works on single numbers, NumPy arrays and PyTorch tensors, 
differencing element by element.

SYNTAX:
diff

EXAMPLE:
diff

INPUTS and PARAMETERS:

in:
The value to difference. Receiving data here triggers the node.

absolute:
When checked, the sign is discarded and you get the SIZE of the change 
regardless of direction. Useful when you care that something moved, 
not which way. Unchecked by default.

reset:
A button that forgets the previous value. 
The next value to arrive is treated as a fresh start.

OUTPUTS: 

out:
The change since the previous value.

TWO THINGS TO WATCH:
The result is a change PER VALUE RECEIVED, not per second. If your source does 
not arrive at a steady rate, the numbers do not mean a speed. 

Differencing amplifies noise - it is the opposite of smoothing, and small 
jitter in the input becomes large jitter in the output. It is common to put a 
filter node after diff, or to use diff_filter, which does both at once.

If you are differencing an ANGLE, put continuous_rotation before this node, 
or the wrap from 359 to 0 will read as an enormous jump backwards."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 3.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 3.0)},
    {'key': 'p1', 'init': 'plot', 'pos': (200, 108), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c0', 'comment': True, 'text': 'position', 'pos': (200, 288)},
    {'key': 'df', 'init': 'diff', 'pos': (30, 235), 'w': 110, 'h': 90},
    {'key': 'p2', 'init': 'plot', 'pos': (200, 325), 'w': 208, 'h': 176,
     'props': PLOT(-0.1, 0.1)},
    {'key': 'c1', 'comment': True, 'text': 'change: fastest where the wave is steepest\nzero at the peaks, where it turns around',
     'pos': (30, 345)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'p1', 'y'), ('sig', '', 'df', ''), ('df', '', 'p2', 'y')]
print(build('diff', 'diff - how much did it change?', body, demo, links,
            demo_width=430, text_width=790, text_height=620))
