"""np cumulative, rolling buffer, target geometry, utilities."""
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

np.cumsum   the running total at each position
np.cumprod  the running product
np.diff     the difference between each element and the one before it

cumsum and diff are nearly each other's opposites. diff turns positions into 
steps; cumsum turns steps back into positions. Reach for diff when an array 
holds where something WAS and you want how far it MOVED between samples, and 
cumsum when you have the movements and want the path.

"Nearly", because the starting point is lost on the way. In the demo, 
3 1 4 1 5 9 summed and then differenced comes back as 1 4 1 5 9: the first 
value has nothing before it to differ from. To rebuild a path from its 
steps exactly, you need to know where it started.

SYNTAX:
np.cumsum <axis>
np.cumprod <axis>
np.diff <order> <axis>

EXAMPLE:
np.diff 2

INPUTS and PARAMETERS:

in:
The array (a list of numbers works too). Receiving it triggers the node.

axis:
Which direction to accumulate or difference along. Default -1, the last 
axis - along each row. An axis the array does not have sends nothing.

order (np.diff):
How many times to difference. 1 gives you the change, 2 the change in the 
change - so from positions, order 1 is speed and order 2 is acceleration. 
Anything below 1 is treated as 1.

OUTPUTS: 

cumsum / cumprod:
The running result, the same shape as the input.

differences (np.diff):
The differences.

np.diff IS SHORTER:
np.diff returns an array SHORTER than its input - with six positions there 
are only five gaps between them, and each further order loses one more. 
Anything expecting the original length will complain. An array too short 
to difference at the chosen order sends nothing at all.

RELATED:
np.rolling_buffer collects a stream into an array these can run along.
t.energy sums the size of the differences into one number."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 200, 'h': 42,
     'props': {'text in': '3 1 4 1 5 9', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'click: six numbers', 'pos': (250, 125)},
    {'key': 'cs', 'init': 'np.cumsum', 'pos': (30, 180), 'w': 150, 'h': 70},
    {'key': 'l1', 'init': 'list', 'pos': (30, 265), 'w': 280, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'the running total', 'pos': (330, 272)},
    {'key': 'df', 'init': 'np.diff', 'pos': (30, 330), 'w': 150, 'h': 90},
    {'key': 'l2', 'init': 'list', 'pos': (30, 435), 'w': 280, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'differenced: back, minus the first',
     'pos': (330, 442)},
    {'key': 'cp', 'init': 'np.cumprod', 'pos': (30, 500), 'w': 150, 'h': 70},
    {'key': 'l3', 'init': 'list', 'pos': (30, 585), 'w': 280, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'the running product', 'pos': (330, 592)},
]
links = [('btn', '', 'm1', ''),
         ('m1', 'message out', 'cs', 'in'), ('cs', '', 'l1', '', 0),
         ('cs', '', 'df', 'in', 0), ('df', 'differences', 'l2', ''),
         ('m1', 'message out', 'cp', 'in'), ('cp', '', 'l3', '', 0)]
print(build('np_cumulative', 'np.cumsum, np.cumprod, np.diff - the running story',
            body, demo, links, demo_width=560, text_width=800, text_height=760))

# ---------------------------------------------------------- np.rolling_buffer
body = """np.rolling_buffer keeps the last N things that arrived, as one array.

It works across TIME rather than within an array. Feed it a value or a small 
array on every frame and it hands back the most recent N of them stacked 
together - which is how a stream becomes something you can take a mean of, 
plot as a history, or run a filter across. Almost anything you want to know 
about "the recent past" starts here.

NEWEST FIRST:
Row 0 of the output is the sample that just arrived; the last row is the 
oldest still kept. A plot of it scrolls from left to right, with now at the 
left edge.

THE SHAPE:
The arguments are the buffer's shape. The first is how many samples to keep; 
any after it are the shape of each sample. np.rolling_buffer 60 keeps 60 
single numbers; np.rolling_buffer 60 3 keeps 60 three-value samples, as a 60 
by 3 array. With no arguments it keeps 16 single numbers.

A sample of the wrong shape is ignored without complaint - if nothing comes 
out, check what is going in. A sample with one extra leading axis of size 1 
(a 1 by 3 array into a buffer of threes) is accepted.

SYNTAX:
np.rolling_buffer <length> <sample shape...>

EXAMPLE:
np.rolling_buffer 60 3

INPUTS and PARAMETERS:

input:
The latest sample. Receiving it adds it to the buffer and sends the buffer.

reset:
Empties the buffer and sends the empty array, so displays clear at once.

OUTPUTS: 

out:
The buffer's contents, newest first. Each send is a fresh array, so 
anything downstream that keeps one sees it stay as it was.

IT GROWS BEFORE IT ROLLS:
A rolling buffer only reports what it has actually collected, so a freshly 
reset one is short until it fills: 1 row, then 2, up to N. That is 
deliberate - it avoids feeding you a stretch of zeros that never happened - 
but it does mean the output's length changes for the first N samples.

RELATED:
np.cumsum and np.diff run along the buffer once you have one."""

demo = [
    {'key': 'met', 'init': 'metro 30', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 30.0, 'units': 'milliseconds'}},
    {'key': 'rnd', 'init': 'random 1.0', 'pos': (30, 142), 'w': 140, 'h': 80,
     'props': {'range': 1.0, 'bipolar': False}},
    {'key': 'c0', 'comment': True, 'text': 'a stream of single values',
     'pos': (30, 230)},
    {'key': 'rb', 'init': 'np.rolling_buffer 60', 'pos': (30, 270), 'w': 200, 'h': 80},
    {'key': 'c1', 'comment': True, 'text': 'the last 60 of them, as one array',
     'pos': (30, 360)},
    {'key': 'mn', 'init': 'np.mean', 'pos': (30, 400), 'w': 140, 'h': 70},
    {'key': 'f1', 'init': 'float', 'pos': (30, 485), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': 'now the stream can be averaged',
     'pos': (30, 535)},
    {'key': 'p1', 'init': 'plot', 'pos': (280, 400), 'w': 208, 'h': 176,
     'props': {'color': 'none', 'width': 200, 'height': 128, 'style': 'line',
               'update style': 'input is multi-channel sample', 'sample count': 60,
               'min x': 0.0, 'max x': 60.0, 'min y': 0.0, 'max y': 1.0}},
    {'key': 'c3', 'comment': True, 'text': 'the history: newest at the left',
     'pos': (280, 585)},
]
links = [('met', '', 'rnd', 'trigger'),
         ('rnd', 'out', 'rb', 'input'),
         ('rb', 'out', 'mn', 'in'), ('mn', '', 'f1', '', 0),
         ('rb', 'out', 'p1', 'y')]
print(build('np.rolling_buffer', 'np.rolling_buffer - the recent past, as an array',
            body, demo, links, demo_width=520, text_width=790, text_height=700))

# ----------------------------------------------------------------- np_target
body = """These nodes measure a position against a reference, and report when it gets close.

They exist because "how near is this to that" is a question you ask constantly
of tracked bodies, and answering it with separate subtract, norm and compare
nodes is both tedious and easy to get wrong.

THE NODES:

np.distance_from_target  how far the input is from one stored target
np.proximity_to_target   closeness to any number of targets, 1 at a target
                         falling to 0 at the edge of its zone - the shape you
                         want for driving sounds
proximity_trigger        an event when the input arrives at one target

The target is captured rather than typed. Put the thing where you want the
target to be, click "set target" (or "capture"), and the node remembers that
position. From then on it reports against it.

np.distance_from_target:
Sends the straight-line distance from the input to the target every time an
input arrives. Nothing comes out until a target has been set. Its target is
saved with the patch.

"axis" matters when the input holds several positions at once. Left at None,
the whole difference is measured as one distance. For an array of positions,
one per row, choose axis 1 and you get one distance per row. The target is whatever
was at the input when you captured it, so a set of positions captures a set,
and each is measured against its own.

np.proximity_to_target:
Holds as many targets as you like. Set "target index" to a slot, put the
tracker where that target should be, click "capture". Choose another index
and capture again. Skipping ahead leaves the slots in between unset, and they
report 0. The "targets" inlet replaces the whole set with a list of
positions; "clear" empties it. The targets are saved with the patch, and the
"targets" outlet sends the whole set whenever it changes.

Each target has a responsive zone of size "radius": proximity is 1 at the
target and falls to 0 at the radius. "exponent" shapes the fall - above 1 the
response concentrates near the centre, below 1 it stays high across the zone
and drops off at the edge.

The "fade" outlet plugs straight into the fade inlet of polyphonic_sampler~:
it sends [[sound_id, fade], ...], target i driving sound "first sound" + i.
Like effort_fader it is sparse - a sound is sent while its target is in range,
gets a single 0 when the tracker leaves the zone (so the sample stops), and
is otherwise left alone. With several trackers at once (one per row) the
closest one sets each target's fade. "nearest" gives the index of the closest
target whose zone you are in, or -1.

proximity_trigger:
Turns arrival into an event. It has two distances, "threshold" and
"release_threshold". When the input comes closer than the threshold it sends
true from "state" and adds one to "count"; it will not fire again until the
input has gone further away than the release threshold, when it sends false. That is
the same hysteresis the trigger node uses, and for the same reason: without
the gap, a hand hovering at the boundary fires continuously. Keep the release
larger than the threshold.

"arm" must be ticked. It starts unticked, and while it is off the node does
nothing at all - no distance, no events. Ticking or unticking it (or
clicking "reset_count") sets the count back to zero.

Its target is saved with the patch, whatever its size. With "axis" set and
several positions at the input (one per row), the nearest row decides: any
row inside the threshold fires it, and it releases only when every row is past
the release threshold.

SYNTAX:
np.distance_from_target
np.proximity_to_target
proximity_trigger

EXAMPLE:
proximity_trigger

INPUTS and PARAMETERS:

input:
The current position, as a list or array. Receiving it triggers the node.

set target (np.distance_from_target, proximity_trigger):
A button. Captures whatever is at the input right now as the reference.

axis (np.distance_from_target, proximity_trigger):
Which direction to measure along when the input holds several positions.
Leave it at None for a single position.

target index / capture / targets / clear (np.proximity_to_target):
Choose a slot, capture into it, replace all targets at once, or forget them.

radius / exponent (np.proximity_to_target):
The size of each target's zone, and the shape of the fall-off across it.

first sound (np.proximity_to_target):
The sound number target 0 drives.

arm / reset_count (proximity_trigger):
Switch the trigger on (it is off until you do), and set the count to zero.

threshold / release_threshold (proximity_trigger):
How close counts as arrived, and how far away it must go before it can
arrive again.

OUTPUTS:

norm (np.distance_from_target):
The distance.

fade / nearest / targets (np.proximity_to_target):
The [[sound_id, fade], ...] list, the index of the nearest target in range
(or -1), and the current set of targets.

state / count / proximity (proximity_trigger):
true on arrival and false on release (sent only when it changes), how many times it
has arrived, and the distance to the target on every input (one per row when
"axis" is set). Despite its name, the "proximity" outlet sends the DISTANCE,
not a closeness: it gets SMALLER as you approach, and is 0 at the target.

RELATED:
np.line_intersection and rotate_position used to share this page; each has
its own now."""

demo = [
    {'key': 'sig', 'init': 'signal 4.0 sin', 'pos': (30, 62), 'w': 129, 'h': 78,
     'props': SIG('sin', 4.0)},
    {'key': 'sig2', 'init': 'signal 5.0 cos', 'pos': (180, 62), 'w': 129, 'h': 78,
     'props': SIG('cos', 5.0)},
    {'key': 'sig3', 'init': 'signal 7.0 sin', 'pos': (330, 62), 'w': 129, 'h': 78,
     'props': SIG('sin', 7.0, rng=0.3)},
    {'key': 'pk', 'init': 'pack 3', 'pos': (30, 160), 'w': 130, 'h': 100},
    {'key': 'c0', 'comment': True, 'text': 'an x y z position wandering about',
     'pos': (30, 270)},
    {'key': 'dt', 'init': 'np.distance_from_target', 'pos': (30, 310), 'w': 240, 'h': 120},
    {'key': 'c1', 'comment': True, 'text': 'click "set target" on both to capture\na reference',
     'pos': (30, 440)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 490), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 3.0)},
    {'key': 'c2', 'comment': True, 'text': 'distance from wherever you captured',
     'pos': (30, 675)},
    {'key': 'pt', 'init': 'proximity_trigger', 'pos': (300, 310), 'w': 240, 'h': 200,
     'props': {'arm': True, 'threshold': 0.5, 'release_threshold': 0.8}},
    {'key': 'tg', 'init': 'toggle', 'pos': (300, 530), 'w': 45, 'h': 42},
    {'key': 'i1', 'init': 'int', 'pos': (380, 530), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c3', 'comment': True, 'text': 'on while within 0.5, off again past 0.8;\nthe count goes up once per arrival',
     'pos': (300, 585)},
]
links = [('sig', '', 'pk', 'in 1'), ('sig2', '', 'pk', 'in 2'), ('sig3', '', 'pk', 'in 3'),
         ('pk', 'out', 'dt', 'input'), ('dt', 'norm', 'p1', 'y'),
         ('pk', 'out', 'pt', 'input'), ('pt', 'state', 'tg', ''),
         ('pt', 'count', 'i1', '')]
print(build('np_target', 'np targets - how near is this to that', body, demo, links,
            demo_width=600, text_width=810, text_height=760))

# ------------------------------------------------------- np.line_intersection
body = """np.line_intersection finds where two lines cross, in two dimensions.

Each line is given by two points. The node works out where the two infinite
lines through them meet, then checks whether that crossing counts - whether it
falls on the second segment, and whether it lies ahead along the first line -
and tells you separately.

It is the core of a pointing test. Line 1 is a ray: from the shoulder through
the hand, say. Line 2 is a target: the edge of a screen or a wall. The
crossing is where the pointing lands, and the fraction says how far along the
target that is.

TWO DIMENSIONS ONLY:
Every point is an x y pair. Points with a third number are refused: a
message is printed once and nothing is sent. To point in 3D, project onto a
plane first (take the x and z of each point for a floor plan, for example).

ONLY "line 1 start" TRIGGERS:
The other three inlets just store their points. Send them first, then line 1
start; for a moving pointer, make the moving end line 1 start.

WHEN IS THE CROSSING VALID:
"direction matters" (on by default): the crossing must be closer to line 1's
end than to its start - so line 1 behaves like a ray fired from its start
through its end, and a crossing behind the start does not count.

"intersect line 2 segment" (on by default): the crossing must lie between line
2's two points, not out on the line's extension beyond them.

Parallel lines never meet: the point comes out as inf inf, the fraction as
inf, and the crossing is always reported as not valid, whatever the checks.
The same goes for a line whose two points are the same - it has no direction,
so there is no crossing to find.

SYNTAX:
np.line_intersection

EXAMPLE:
np.line_intersection

INPUTS and PARAMETERS:

line 1 start:
A point, as x y. Receiving it triggers the calculation.

line 1 end / line 2 start / line 2 end:
The other three points. They are stored until line 1 start arrives.

direction matters:
Only count a crossing ahead of line 1 (nearer its end than its start).

intersect line 2 segment:
Only count a crossing between line 2's two points.

OUTPUTS:

intersection is valid:
true if the crossing passes the checks that are on, false if not.

point of intersection:
Where the lines cross, as x y. It is sent even when the crossing is not
valid, so check the valid outlet before using it.

fraction of segment 2:
How far the crossing is from line 2 start, as a fraction of line 2's length:
0 at the start, 1 at the end. It is a distance, never negative, so a crossing
a little way BEFORE line 2 start gives the same small number as one the same
distance after it - another reason to read the valid outlet."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 120), 'w': 100, 'h': 42,
     'props': {'text in': '0 0.5', 'font size': '24'}},
    {'key': 'm2', 'init': 'message', 'pos': (180, 120), 'w': 130, 'h': 42,
     'props': {'text in': '-1 1', 'font size': '24'}},
    {'key': 'm3', 'init': 'message', 'pos': (330, 120), 'w': 130, 'h': 42,
     'props': {'text in': '1 1', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'line 1 ends at 0 0.5; line 2 runs from\n-1 1 to 1 1',
     'pos': (30, 175)},
    {'key': 'sig', 'init': 'signal 6.0 sin', 'pos': (30, 230), 'w': 129, 'h': 78,
     'props': SIG('sin', 6.0, rng=2.0)},
    {'key': 'pk', 'init': 'pack 2', 'pos': (30, 325), 'w': 130, 'h': 80},
    {'key': 'c1', 'comment': True, 'text': 'line 1 starts at x 0, sliding\nfrom -2 to 2',
     'pos': (30, 415)},
    {'key': 'li', 'init': 'np.line_intersection', 'pos': (30, 470), 'w': 260, 'h': 200},
    {'key': 'tg', 'init': 'toggle', 'pos': (30, 690), 'w': 45, 'h': 42},
    {'key': 'c2', 'comment': True, 'text': 'valid only while the crossing lands\nbetween -1 and 1',
     'pos': (30, 745)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 800), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'where line 1 meets y = 1',
     'pos': (30, 850)},
    {'key': 'p1', 'init': 'plot', 'pos': (330, 470), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 1.6)},
    {'key': 'c4', 'comment': True, 'text': 'fraction along line 2: above 1 when it\nmisses past the end',
     'pos': (330, 660)},
]
links = [('lb', 'out', 'm1', ''), ('lb', 'out', 'm2', ''), ('lb', 'out', 'm3', ''),
         ('m1', 'message out', 'li', 'line 1 end'),
         ('m2', 'message out', 'li', 'line 2 start'),
         ('m3', 'message out', 'li', 'line 2 end'),
         ('sig', '', 'pk', 'in 1'), ('pk', 'out', 'li', 'line 1 start'),
         ('li', 'intersection is valid', 'tg', ''),
         ('li', 'point of intersection', 'l1', ''),
         ('li', 'fraction of segment 2', 'p1', 'y')]
print(build('np.line_intersection', 'np.line_intersection - where two lines cross',
            body, demo, links, demo_width=580, text_width=790, text_height=700))

# ------------------------------------------------------------ rotate_position
body = """rotate_position turns a position about one of the three axes.

Give it an x y z position and an angle in degrees; it sends the position
rotated by that angle about the X, Y or Z axis, around the origin. It is the
quick way to turn a tracked point into a different frame - to correct a sensor
mounted at an angle, or to face a body forward before measuring it.

A ROTATION FOLLOWS THE RIGHT-HAND RULE:
Point your right thumb along the chosen axis; positive angles turn the way
your fingers curl. About Y, a point at 1 0 0 turned by 90 degrees ends at
0 0 -1.

ONE POSITION OR MANY:
A single position of three numbers comes out as three numbers. An array of
positions, one x y z per row, comes out with every row rotated. Anything else
- a two-number point, say - is passed through UNCHANGED, and the node prints
a warning the first time it happens.

ONLY THE POSITION TRIGGERS:
Changing "angle" does not send anything by itself - the new angle is used on
the next position that arrives. Changing "axis" does re-send the last position
straight away. To sweep the angle over a fixed point, keep sending the point,
as the demo does with tick.

The result is always float32.

SYNTAX:
rotate_position

EXAMPLE:
rotate_position

INPUTS and PARAMETERS:

position:
An x y z position, or an array of them. Receiving it triggers the node.

angle:
The rotation, in degrees.

axis:
X, Y or Z. Default Y - the vertical axis in most body data, so the default
turns things about the vertical.

OUTPUTS:

rotated:
The rotated position or positions.

RELATED:
For rotations about an arbitrary axis, or by a quaternion, use the
quaternion nodes."""

demo = [
    {'key': 'met', 'init': 'tick', 'pos': (30, 62), 'w': 33, 'h': 42, 'props': {'': True}},
    {'key': 'm1', 'init': 'message', 'pos': (30, 120), 'w': 100, 'h': 42,
     'props': {'text in': '1 0 0', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'a point on the x axis, sent every frame',
     'pos': (30, 170)},
    {'key': 'sig', 'init': 'signal 4.0 saw', 'pos': (230, 62), 'w': 129, 'h': 78,
     'props': SIG('saw', 4.0, rng=180.0)},
    {'key': 'c1', 'comment': True, 'text': 'an angle sweeping -180 to 180', 'pos': (230, 150)},
    {'key': 'rp', 'init': 'rotate_position', 'pos': (30, 210), 'w': 220, 'h': 120},
    {'key': 'l1', 'init': 'list', 'pos': (30, 350), 'w': 280, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 410), 'w': 208, 'h': 176,
     'props': dict(PLOT(-1.0, 1.0), **{'update style': 'input is multi-channel sample'})},
    {'key': 'c2', 'comment': True, 'text': 'about Y: x and z trace a circle\nwhile y stays at 0',
     'pos': (30, 600)},
]
links = [('met', '', 'm1', ''), ('m1', 'message out', 'rp', 'position'),
         ('sig', '', 'rp', 'angle'),
         ('rp', 'rotated', 'l1', ''), ('rp', 'rotated', 'p1', 'y')]
print(build('rotate_position', 'rotate_position - turn a point about an axis', body,
            demo, links, demo_width=420, text_width=790, text_height=640))

# ----------------------------------------------------------------- np.astype
body = """np.astype converts an array to a different number type.

An array of float64 and one of float32 hold the same numbers and are not
interchangeable everywhere - image work usually wants uint8, torch usually
wants float32, and indices have to be integers. When a node refuses an array
that looks correct, the type is a likely culprit, and info will tell you what
it actually is.

THE TYPES:
Chosen from the "type" menu: bool, uint8, int8, int64, float, float32, double.
There is no int32 here. "float" and "double" are the same thing, the
64-bit float; "float32" is the single-precision one torch usually wants.

CONVERTING TO INTEGERS TRUNCATES:
Converting a float array to an integer type cuts off the fraction rather than
rounding, so 9.7 becomes 9 and -0.9 becomes 0. If you want the nearest whole
number, round first with np.round and then convert.

Out-of-range values do not stop at the limit: converted to uint8, 300 comes
out as 44 and -1 as 255. Clip first (np.clip) if the data can leave the 0 to
255 range.

Lists and single numbers are accepted too, and come out as arrays.

SYNTAX:
np.astype

EXAMPLE:
np.astype

INPUTS and PARAMETERS:

input array:
The array. Receiving it triggers the node.

type:
What to convert to.

OUTPUTS:

converted array:
The same values, in the new type."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 'np.rand 4', 'pos': (30, 120), 'w': 140, 'h': 110,
     'props': {'min': 0.0, 'max': 10.0, 'dim 0': 4, 'dtype': 'float32'}},
    {'key': 'l1', 'init': 'list', 'pos': (30, 245), 'w': 280, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'click: floats between 0 and 10', 'pos': (30, 295)},
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
print(build('np.astype', 'np.astype - change an array\'s number type', body, demo, links,
            demo_width=440, text_width=790, text_height=560))

# ------------------------------------------------------- shared helpers
# Helpers for the one-node-per-page sections below (copied from
# make_numpy_b_help.py rather than imported across generators).


def HM(width, height, count, miny=0.0, maxy=1.0, fmt='%.2f', color='viridis'):
    """Property block for a heat_map that draws a whole array as one picture."""
    return {'color': color, 'width': width, 'height': height, 'sample count': count,
            'min y': miny, 'max y': maxy, 'update_mode': 'heat_map',
            'number format': fmt}


LIST = {'text in': '', 'font size': '24'}


def see_also(y, stems, x=30):
    """A SEE ALSO row: one help_link per related page, left to right."""
    row = []
    for s in stems:
        w = int(len(s) * 7.7) + 30
        row.append({'key': 'sa_' + s, 'init': 'help_link ' + s, 'pos': (x, y),
                    'w': w, 'h': 30})
        x += w + 14
    row.append({'key': 'sa_c', 'comment': True, 'text': 'SEE ALSO - click to open',
                'pos': (x, y)})
    return row


# np.add_alpha_help.json is laid out by hand (David rebuilt it with an
# mgl_sphere behind each image) and is no longer generated here -- a
# rebuild would overwrite it.


# -------------------------------------------------------------------- np.edit
body = """Overwrites chosen positions in an array with a value you type.

Name the positions in "indices", put the new value in "values", and every
array that passes through comes out with those positions replaced. It is for
patching a few numbers by hand: zeroing a channel you do not trust, pinning a
calibration value, forcing one element of a weight vector while you tune.

IT DOES NOT DISPLAY THE ARRAY:
np.edit shows only the two fields. Put a display after it - a heat_map, as
here, for a grid - to see the result.

WRITING INDICES:
The same slice notation as numpy, one entry per dimension, separated by
commas:

  2        just position 2
  1:3      positions 1 and 2 (the end is not included)
  3:       from position 3 to the end
  :        everything
  0, 1:3   in row 0, columns 1 and 2

Dimensions you leave out are taken whole, so on a grid "2" means all of row 2.
-1 means the last position.

NOTHING HAPPENS UNTIL YOU TYPE A VALUE:
A fresh np.edit has indices ":" and an empty "values" field, and passes arrays
through unchanged until you put something in "values". Clearing the field
again turns the edit off.

WHAT THE VALUE CAN BE:
A single number fills every chosen position. A list of numbers fills them in
order: "1 2" into "3:5" puts 1 at position 3 and 2 at position 4. A list longer
than the selection is cut short. For a grid, a list as long as a row is
copied into every chosen row; a list of any other length is ignored on a
grid (nothing changes, without a message).

The value takes the array's own type, so 2.7 written into an integer array
becomes 2.

EDITS WITHOUT A NEW ARRAY:
Changing either field re-sends straight away, applied to a fresh copy of the
last array that arrived - so only the current edit shows (set 1:3 to 9, then
change indices to 4, and only position 4 is 9).

SYNTAX:
np.edit <indices>

EXAMPLE:
np.edit 1:3, 2:6

INPUTS and PARAMETERS:

array in:
The array. Receiving it triggers the node.

indices:
Which positions to replace.

values:
What to put there.

widget width (option):
How wide to draw the two fields. They can also be dragged wider.

OUTPUTS:

output:
A copy of the array with the edit applied. The incoming array itself is not
changed.

READING THE DEMO:
np.rand makes a 5 x 8 grid of numbers between 0 and 0.5 - all dark - and
the left heat_map shows it. np.edit writes 1 into rows 1 and 2, columns 2
to 5, and the right heat_map shows the result: the same grid with a bright
block in it. Every other cell is untouched.

Type into the fields and the right picture changes at once: indices "0" for
all of row 0, ":, 7" for the last column, "4, 0" for one cell; values
"0.25" for a middle shade, or a whole row of eight such as
"0 1 0 1 0 1 0 1" to stripe the chosen rows. Click np.rand for a fresh
grid - the same edit is applied to it.

RELATED:
np.[] reads positions out instead of writing them. np.where replaces values
by a condition rather than by position. np.clip limits values; np.zeros and
np.ones make an array to edit into."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 42},
    {'key': 'rnd', 'init': 'np.rand 5 8', 'pos': (30, 120), 'w': 140, 'h': 140,
     'props': {'min': 0.0, 'max': 0.5, 'dtype': 'float32'}},
    {'key': 'c0', 'comment': True, 'text': 'click for a fresh grid of\nnumbers between 0 and 0.5',
     'pos': (30, 130)},
    {'key': 'ed', 'init': 'np.edit 1:3, 2:6', 'pos': (380, 300), 'w': 180, 'h': 70,
     'props': {'indices': '1:3, 2:6', 'values': '1'}},
    {'key': 'c1', 'comment': True, 'text': 'try indices 0  or  :, 7  or  4, 0\nand values 0.25  or  0 1 0 1 0 1 0 1',
     'pos': (30, 300)},
    {'key': 'h1', 'init': 'heat_map', 'pos': (30, 400), 'w': 340, 'h': 150,
     'props': HM(320, 120, 5, 0.0, 1.0, '%.1f')},
    {'key': 'h2', 'init': 'heat_map', 'pos': (380, 400), 'w': 340, 'h': 150,
     'props': HM(320, 120, 5, 0.0, 1.0, '%.1f')},
    {'key': 'c2', 'comment': True, 'text': 'left: the array in. right: after the edit',
     'pos': (30, 410)},
] + see_also(590, ['np.[]', 'np.where', 'np.clip', 'np.zeros'])
links = [('lb', 'out', 'rnd', ''),
         ('rnd', '', 'h1', 'y'),
         ('rnd', '', 'ed', 'array in'),
         ('ed', 'output', 'h2', 'y')]
print(build('np.edit', 'np.edit - overwrite chosen positions in an array', body, demo,
            links, demo_width=760, text_width=790, text_height=1000))

# -------------------------------------------------------------------- np.load
body = """Reads a numpy .npz file and sends the arrays in it, as a dictionary.

An .npz is several named arrays saved together in one file - the usual way
numpy data is shipped, and the format of most motion-capture archives (AMASS,
for one). Because it holds several arrays, what comes out is a DICTIONARY: one
entry per array, under its name. Use dict_keys to find out what is in a file
you did not write, and the dict nodes (dict_extract and friends) to take out
the arrays you want.

LOADING:
Click "load" and choose a file. It is read straight away and sent. "send"
sends the same contents again without re-reading the file.

To load a file by name instead, send the node the message "path" followed
by the file's path, as the message box in the demo does. That sets the path
and reads the file at once - from a patch, with no dialog.

The path is saved with the patch (it is the "path" option), and when the
patch is opened again the file is read and sent automatically. A path typed
after the name (np.load /path/to/file.npz) fills in that option and is read
as soon as the node is made; so is a path typed into the option. A path that
is not absolute is looked for relative to the folder the app was started
from - normally the top folder of the installation, the one holding
dpg_system_main.py, which is why the demo's path starts with
dpg_system/help.

Only .npz files. A plain .npy file (a single array) is not read by this node.

If the file cannot be read, the error is printed to the console and nothing
is sent.

A WORD OF CAUTION:
The file is read with numpy's pickle support switched on, so that arrays of
mixed objects load. A pickled file can run code as it loads - only open .npz
files you trust.

SYNTAX:
np.load <path>

EXAMPLE:
np.load dpg_system/help/data/np_load_example.npz

INPUTS and PARAMETERS:

load:
A button. Choose a file, read it and send it. Any inlet also accepts the
message "path <file>".

send:
A button. Send the last file read, again.

path (option):
The file to read. Set by "load" or by a "path" message; read automatically
when the patch opens.

OUTPUTS:

dict out:
A dictionary of the file's arrays, keyed by name.

READING THE DEMO:
The help comes with a small file, np_load_example.npz, holding two arrays:
"rings", a 16 x 16 target of concentric rings, and "checker", an 8 x 12
checkerboard of 2 x 2 squares, both with values from 0 to 1. Click the
message box to load it: dict_keys lists the two names, and dict_extract
takes each array out by name for a heat_map.

RELATED:
dict_keys and dict_extract unpack the dictionary. np.sequence records and
plays back a series of arrays."""

demo = [
    {'key': 'm1', 'init': 'message', 'pos': (30, 62), 'w': 470, 'h': 42,
     'props': {'text in': 'path dpg_system/help/data/np_load_example.npz', 'font size': '24',
               'width': 540}},
    {'key': 'c0', 'comment': True, 'text': 'click to load the example file',
     'pos': (30, 72)},
    {'key': 'ld', 'init': 'np.load', 'pos': (30, 130), 'w': 180, 'h': 100},
    {'key': 'dk', 'init': 'dict_keys', 'pos': (30, 260), 'w': 140, 'h': 42},
    {'key': 'l1', 'init': 'list', 'pos': (30, 320), 'w': 300, 'h': 42, 'props': dict(LIST)},
    {'key': 'c1', 'comment': True, 'text': 'the names of the arrays inside it',
     'pos': (30, 330)},
    {'key': 'dx', 'init': 'dict_extract rings checker', 'pos': (30, 390), 'w': 220, 'h': 70,
     'props': {'key count': 2, 'key 0': 'rings', 'key 1': 'checker',
               'include key in output': False}},
    {'key': 'h1', 'init': 'heat_map', 'pos': (30, 490), 'w': 260, 'h': 270,
     'props': HM(240, 240, 16, 0.0, 1.0, 'none', 'greys')},
    {'key': 'h2', 'init': 'heat_map', 'pos': (310, 490), 'w': 380, 'h': 270,
     'props': HM(360, 240, 8, 0.0, 1.0, 'none', 'greys')},
    {'key': 'c2', 'comment': True, 'text': 'left: "rings" (16 x 16). right: "checker" (8 x 12)',
     'pos': (30, 500)},
] + see_also(800, ['dict_keys', 'dict_extract', 'np.sequence'])
links = [('m1', '', 'ld', 'load'),
         ('ld', 'dict out', 'dk', 'dict in'), ('dk', 'keys out', 'l1', ''),
         ('ld', 'dict out', 'dx', 'dict in'),
         ('dx', '', 'h1', 'y', 0), ('dx', '', 'h2', 'y', 1)]
print(build('np.load', 'np.load - read an .npz file', body, demo, links,
            demo_width=720, text_width=790, text_height=1050))
