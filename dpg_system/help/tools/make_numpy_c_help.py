"""np cumulative, target geometry, utilities."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ------------------------------------------------------------- np_cumulative
body = """These nodes look along an array and report how it develops, rather than 
summarising it in one number.

A sum tells you the total. A cumulative sum tells you the total SO FAR at every 
point - the running story rather than the ending. That difference is what makes 
these useful for anything with an order to it: a trajectory, a signal over time, 
a profile across a body.

THE NODES:

np.cumsum          the running total at each position
np.cumprod         the running product
np.diff            the difference between neighbouring elements
np.rolling_buffer  keep the last N things that arrived, as one array

cumsum and diff undo each other. diff turns positions into steps; cumsum turns 
steps back into positions. Reach for diff when an array holds where something 
WAS and you want how far it MOVED between samples, and cumsum when you have 
the movements and want the path.

np.rolling_buffer is the one that works across time rather than within an 
array. Feed it a value or a small array on every frame and it hands back the 
last N of them stacked together - which is how a stream becomes something you 
can take a mean of, plot as a history, or run a filter across. 
Almost anything you want to know about "the recent past" starts here.

SYNTAX:
np.cumsum
np.diff <order: int>
np.rolling_buffer <length> <width> ...

EXAMPLE:
np.rolling_buffer 60

INPUTS and PARAMETERS:

in / input:
The array, or on rolling_buffer the latest sample. Receiving it triggers 
the node.

axis:
Which direction to accumulate or difference along.

order (np.diff):
How many times to difference. 1 gives you the change, 2 the change in the 
change - so from positions, order 1 is speed and order 2 is acceleration.

reset (np.rolling_buffer):
Empties the buffer and starts collecting again.

OUTPUTS: 

out:
The running result, or the buffer's contents.

TWO THINGS TO WATCH:
np.diff returns an array one element SHORTER than its input - with five 
positions there are only four gaps between them. Anything expecting the 
original length will complain.

A rolling buffer only reports what it has actually collected, so a freshly 
reset one is short until it fills. That is deliberate - it avoids feeding you 
a stretch of zeros that never happened - but it does mean the buffer's length 
changes for the first N frames."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'met', 'init': 'metro 30', 'pos': (30, 112), 'w': 129, 'h': 70,
     'props': {'on': False, 'period': 30.0, 'units': 'milliseconds'}},
    {'key': 'rnd', 'init': 'random 1.0', 'pos': (30, 192), 'w': 140, 'h': 80,
     'props': {'range': 1.0, 'bipolar': False}},
    {'key': 'c0', 'comment': True, 'text': 'switch on: a stream of single values',
     'pos': (30, 280)},
    {'key': 'rb', 'init': 'np.rolling_buffer 60', 'pos': (30, 320), 'w': 200, 'h': 80},
    {'key': 'c1', 'comment': True, 'text': 'the last 60 of them, as one array',
     'pos': (30, 410)},
    {'key': 'mn', 'init': 'np.mean', 'pos': (30, 450), 'w': 140, 'h': 70},
    {'key': 'f1', 'init': 'float', 'pos': (30, 535), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': 'now the stream can be averaged',
     'pos': (30, 585)},
    {'key': 'cs', 'init': 'np.cumsum', 'pos': (280, 450), 'w': 150, 'h': 70},
    {'key': 'p1', 'init': 'plot', 'pos': (280, 535), 'w': 208, 'h': 176,
     'props': {'color': 'none', 'width': 200, 'height': 128, 'style': 'line',
               'update style': 'input is multi-channel sample', 'sample count': 60,
               'min x': 0.0, 'max x': 60.0, 'min y': 0.0, 'max y': 40.0}},
    {'key': 'c3', 'comment': True, 'text': 'the running total across the buffer',
     'pos': (280, 720)},
]
links = [('tog', '', 'met', 'on'), ('met', '', 'rnd', 'trigger'),
         ('rnd', 'out', 'rb', 'input'),
         ('rb', 'out', 'mn', 'in'), ('mn', '', 'f1', '', 0),
         ('rb', 'out', 'cs', 'in'), ('cs', '', 'p1', 'y', 0)]
print(build('np_cumulative', 'np cumulative - the running story, not the total',
            body, demo, links, demo_width=520, text_width=800, text_height=700))

# ----------------------------------------------------------------- np_target
body = """These nodes measure a position against a reference, and report when it gets close.

They exist because "how near is this to that" is a question you ask constantly 
of tracked bodies, and answering it with separate subtract, norm and compare 
nodes is both tedious and easy to get wrong.

THE NODES:

np.distance_from_target  how far the input is from a stored target
np.proximity_to_target   the same measurement inverted, so near is HIGH - 
                         the shape you want for driving something
proximity_trigger        the same, plus a trigger that fires on arrival
np.line_intersection     where two line segments cross, if they do
rotate_position          rotate a position about an axis

The target is captured rather than typed. Put the thing where you want the 
target to be, click "set target", and the node remembers that position. 
From then on it reports against it.

proximity_trigger is the one that turns this into an event. It has two 
thresholds, near and far, and fires when the input comes closer than the near 
one - then will not fire again until it has retreated past the far one. 
That is the same hysteresis the trigger node uses, and for the same reason: 
without the gap, a hand hovering at the boundary fires continuously.

SYNTAX:
np.distance_from_target
proximity_trigger

EXAMPLE:
proximity_trigger

INPUTS and PARAMETERS:

input:
The current position, as an array. Receiving it triggers the node.

set target:
A button. Captures whatever is at the input right now as the reference.

axis:
For arrays of several positions at once, which direction the coordinates run 
along - so you can measure a whole set of points against a whole set of 
targets in one go.

threshold / release_threshold (proximity_trigger):
How close counts as arrived, and how far away it must go before it can arrive 
again. Keep the release larger than the threshold.

arm / reset_count (proximity_trigger):
Enable the trigger, and set its count back to zero.

line 1 start / line 1 end / line 2 start / line 2 end (np.line_intersection):
The two segments.

direction matters / intersect line 2 segment (np.line_intersection):
Whether to treat the lines as directed, and whether the crossing has to fall 
within the second segment rather than on its infinite extension.

position / angle / axis (rotate_position):
The point to rotate, by how much, and about which axis.

OUTPUTS: 

norm:
The distance, or the proximity.

state / count / proximity (proximity_trigger):
Whether it is currently near, how many times it has arrived, and the 
underlying measurement.

intersection is valid / point of intersection / fraction of segment 2:
Whether the segments actually cross, where, and how far along the second one - 
check the valid outlet before using the point, since parallel lines have no 
crossing to report."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 4.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 4.0)},
    {'key': 'sig2', 'init': 'signal 5.0 cos', 'pos': (200, 132), 'w': 129, 'h': 78,
     'props': SIG('cos', 5.0)},
    {'key': 'pk', 'init': 'pack 2', 'pos': (30, 232), 'w': 130, 'h': 80},
    {'key': 'c0', 'comment': True, 'text': 'a position wandering about', 'pos': (30, 322)},
    {'key': 'dt', 'init': 'np.distance_from_target', 'pos': (30, 360), 'w': 240, 'h': 120},
    {'key': 'c1', 'comment': True, 'text': 'click "set target" to capture a reference',
     'pos': (30, 490)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 530), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 2.5)},
    {'key': 'c2', 'comment': True, 'text': 'distance from wherever you captured',
     'pos': (30, 715)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'), ('tt', '1', 'sig2', 'on'),
         ('sig', '', 'pk', 'in 1'), ('sig2', '', 'pk', 'in 2'),
         ('pk', 'out', 'dt', 'input'), ('dt', 'norm', 'p1', 'y')]
print(build('np_target', 'np targets - how near is this to that', body, demo, links,
            demo_width=460, text_width=810, text_height=760))

# ------------------------------------------------------------------- np_util
body = """Four odds and ends for working with arrays.

THE NODES:

np.astype      convert an array to a different number type
np.add_alpha   add an opacity channel to an image
np.edit        show an array's values and let you change them by hand
np.load        read a saved .npz file from disk

np.astype is the one that comes up. An array of float64 and one of float32 hold 
the same numbers and are not interchangeable everywhere - image work usually 
wants uint8, torch usually wants float32, and indices have to be integers. 
When a node refuses an array that looks correct, the type is a likely culprit, 
and info will tell you what it actually is.

np.edit is for when you want to see and adjust the numbers themselves, 
the way table does for a grid. Useful for calibration values, small weight 
vectors, anything you are tuning by hand.

np.add_alpha turns a three-channel image into a four-channel one, so it can be 
composited over something else.

np.load reads back an .npz file - the format np.sequence writes, and the one 
most numpy data arrives in.

SYNTAX:
np.astype
np.load <path>

EXAMPLE:
np.astype

INPUTS and PARAMETERS:

input array / array in:
The array. Receiving it triggers the node.

type (np.astype):
What to convert to, chosen from a menu: 
bool, uint8, int8, int64, float, float32, double. 
Note that there is no int32 here, and that "float" means the platform double 
while "float32" is the single-precision one torch usually wants.

indices / values (np.edit):
Change particular positions from the patch rather than by hand.

widget width (np.edit):
How wide to draw the fields.

load / send / path (np.load):
Read a file, send what was read, and where to read it from.

OUTPUTS: 

converted array / output / image with alpha:
The result.

dict out (np.load):
The file's contents. An .npz holds several named arrays, so what comes out is 
a dictionary - use the dict nodes to take it apart, and dict_keys to find out 
what is in a file you did not write.

A NOTE ON CONVERTING TO INTEGERS:
Converting a float array to an int type TRUNCATES rather than rounding, so 0.9 
becomes 0. If you want the nearest whole number, round first with np.round and 
then convert."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 'np.rand 4', 'pos': (30, 120), 'w': 140, 'h': 110,
     'props': {'min': 0.0, 'max': 10.0, 'dim 0': 4, 'dtype': 'float32'}},
    {'key': 'l1', 'init': 'list', 'pos': (30, 245), 'w': 280, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'floats between 0 and 10', 'pos': (30, 295)},
    {'key': 'at', 'init': 'np.astype', 'pos': (30, 335), 'w': 180, 'h': 80,
     'props': {'type': 'int64'}},
    {'key': 'l2', 'init': 'list', 'pos': (30, 425), 'w': 280, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'truncated, not rounded: 9.7 becomes 9',
     'pos': (30, 475)},
    {'key': 'inf', 'init': 'info', 'pos': (30, 515), 'w': 240, 'h': 80},
    {'key': 'c2', 'comment': True, 'text': 'info reports the type it really is',
     'pos': (30, 605)},
]
links = [('btn', '', 'rnd', ''), ('rnd', '', 'l1', '', 0),
         ('rnd', '', 'at', 'input array', 0),
         ('at', 'converted array', 'l2', ''),
         ('at', 'converted array', 'inf', 'in')]
print(build('np_util', 'np utilities - types, editing, loading', body, demo, links,
            demo_width=440, text_width=790, text_height=680))
