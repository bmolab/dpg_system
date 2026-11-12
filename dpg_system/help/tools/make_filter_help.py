import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# filter_help.json is laid out by hand now (David's version) and is no longer
# generated here; regenerating it would overwrite that layout.

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
