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
    {'key': 'c0', 'comment': True, 'text': "check the metro's on: a stream of single values",
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
NOT saved with the patch - set it again after opening.

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

It measures one position against one target. Its target is saved with the
patch, but only as x, y and z: use three-number positions with it.

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
has arrived, and the distance to the target on every input. Despite its name,
the "proximity" outlet sends the distance, so it gets SMALLER as you approach.

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
Every point is an x y pair. Points with a third number are refused with an
error; to point in 3D, project onto a plane first (take the x and z of each
point for a floor plan, for example).

ONLY "line 1 start" TRIGGERS:
The other three inlets just store their points. Send them first, then line 1
start; for a moving pointer, make the moving end line 1 start.

WHEN IS THE CROSSING VALID:
"direction matters" (on by default): the crossing must be closer to line 1's
end than to its start - so line 1 behaves like a ray fired from its start
through its end, and a crossing behind the start does not count.

"intersect line 2 segment" (on by default): the crossing must lie between line
2's two points, not out on the line's extension beyond them.

Parallel lines never meet: the point comes out as inf inf and, with either
check on, the crossing is reported as not valid. Turn both checks off and a
parallel pair reports valid, with that infinite point.

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
- a two-number point, say - is passed through UNCHANGED, with no warning, so
if nothing seems to turn, check the shape of what you are sending.

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

# --------------------------------------------------------------- np.add_alpha
body = """np.add_alpha adds a fourth channel to a three-channel image, made from its brightness.

The new channel is the AVERAGE of each pixel's three channels. So it is not a
plain "fully opaque" alpha: used as opacity, bright pixels come out solid and
dark ones transparent, and black disappears altogether. That is what you want
for laying light material - text, glows, a camera image of a lit figure on
black - over something else, so that the black background drops out.

If you want a fully opaque alpha instead, this is not the node; build the
channel yourself.

WHAT IT ACCEPTS:
An image as an array of height x width x channels. Anything that is not three
dimensional is ignored. It does not check the channel count, so an image that
already has four channels gets a fifth.

AN INTEGER IMAGE COMES OUT AS FLOAT:
Averaging produces fractions. A float image keeps its type and its range. But
a uint8 image (0 to 255) comes out as 64-bit float, still with values from 0
to 255 - put np.astype uint8 after it if the next node expects bytes.

SYNTAX:
np.add_alpha

EXAMPLE:
np.add_alpha

INPUTS and PARAMETERS:

input:
The image. Receiving it triggers the node.

OUTPUTS:

image with alpha:
The image with the brightness channel added as its last channel."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 'np.rand 1 2 3', 'pos': (30, 120), 'w': 140, 'h': 150,
     'props': {'min': 0.0, 'max': 1.0, 'dtype': 'float32'}},
    {'key': 'c0', 'comment': True, 'text': 'click: a tiny image, one row of two\npixels, three channels each',
     'pos': (30, 285)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 330), 'w': 400, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'aa', 'init': 'np.add_alpha', 'pos': (30, 390), 'w': 180, 'h': 70},
    {'key': 'l2', 'init': 'list', 'pos': (30, 480), 'w': 400, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'each pixel gains a fourth number:\nthe average of its three',
     'pos': (30, 530)},
]
links = [('btn', '', 'rnd', ''), ('rnd', '', 'l1', '', 0),
         ('rnd', '', 'aa', 'input', 0),
         ('aa', 'image with alpha', 'l2', '')]
print(build('np.add_alpha', 'np.add_alpha - brightness as a fourth channel', body, demo,
            links, demo_width=480, text_width=790, text_height=520))

# -------------------------------------------------------------------- np.edit
body = """np.edit overwrites chosen positions in an array with a value you type.

Name the positions in "indices", put the new value in "values", and every
array that passes through comes out with those positions replaced. It is for
patching a few numbers by hand: zeroing a channel you do not trust, pinning a
calibration value, forcing one element of a weight vector while you tune.

IT DOES NOT DISPLAY THE ARRAY:
np.edit shows only the two fields. Put a list (or a table, for a grid) after
it to see the result.

WRITING INDICES:
The same slice notation as numpy, one entry per dimension, separated by
commas:

  2        just position 2
  1:3      positions 1 and 2 (the end is not included)
  3:       from position 3 to the end
  :        everything
  0, 1:3   in row 0, columns 1 and 2

Dimensions you leave out are taken whole, so on a grid "2" means all of row 2.
Use positive numbers: a single negative index such as -1 selects nothing.

THE DEFAULT CHANGES EVERYTHING:
A fresh np.edit has indices ":" and value 0 - so until you change them, every
array comes out entirely zeros.

WHAT THE VALUE CAN BE:
A single number fills every chosen position. A list of numbers fills them in
order: "1 2" into "3:5" puts 1 at position 3 and 2 at position 4. A list longer
than the selection is cut short. For a grid, a list as long as a row is
copied into every chosen row.

The value takes the array's own type, so 2.7 written into an integer array
becomes 2.

EDITS WITHOUT A NEW ARRAY:
Changing either field re-sends straight away, applied to the copy the node
already holds - so successive changes build up on that copy (set 1:3 to 9,
then change indices to 4 and both edits show). The next array that arrives
starts again from a clean copy, with only the current edit applied.

SYNTAX:
np.edit <indices>

EXAMPLE:
np.edit 1:3

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
changed."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 'np.rand 6', 'pos': (30, 120), 'w': 140, 'h': 110,
     'props': {'min': 0.0, 'max': 1.0, 'dim 0': 6, 'dtype': 'float32'}},
    {'key': 'l1', 'init': 'list', 'pos': (30, 245), 'w': 400, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'click: six numbers between 0 and 1', 'pos': (30, 295)},
    {'key': 'ed', 'init': 'np.edit 1:3', 'pos': (30, 335), 'w': 180, 'h': 100,
     'props': {'indices': '1:3', 'values': '9'}},
    {'key': 'c1', 'comment': True, 'text': 'try 4: or 0 or a list such as 5 6',
     'pos': (30, 445)},
    {'key': 'l2', 'init': 'list', 'pos': (30, 485), 'w': 400, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'positions 1 and 2 replaced by 9',
     'pos': (30, 535)},
]
links = [('btn', '', 'rnd', ''), ('rnd', '', 'l1', '', 0),
         ('rnd', '', 'ed', 'array in', 0),
         ('ed', 'output', 'l2', '')]
print(build('np.edit', 'np.edit - overwrite chosen positions in an array', body, demo,
            links, demo_width=480, text_width=790, text_height=820))

# -------------------------------------------------------------------- np.load
body = """np.load reads a numpy .npz file and sends what is in it.

An .npz is several named arrays saved together in one file - the usual way
numpy data is shipped, and the format of most motion-capture archives (AMASS,
for one). Because it holds several arrays, what comes out is a DICTIONARY: one
entry per array, under its name. Use dict_keys to find out what is in a file
you did not write, and the dict nodes (dict_extract and friends) to take out
the arrays you want.

LOADING:
Click "load" and choose a file. It is read straight away and sent. "send"
sends the same contents again without re-reading the file.

The chosen path is saved with the patch (it is the "path" option), and when
the patch is opened again the file is read and sent automatically. A path
typed after the name (np.load /path/to/file.npz) fills in that option, but
the file is not read until the patch is saved and opened again - "send" does
nothing before then. A path that is not absolute is looked for relative to
the folder the app was started from.

Only .npz files. A plain .npy file (a single array) is not read by this node.

A WORD OF CAUTION:
The file is read with numpy's pickle support switched on, so that arrays of
mixed objects load. A pickled file can run code as it loads - only open .npz
files you trust.

SYNTAX:
np.load <path>

EXAMPLE:
np.load

INPUTS and PARAMETERS:

load:
A button. Choose a file, read it and send it.

send:
A button. Send the last file read, again.

path (option):
The file to read. Set by "load"; read automatically when the patch opens.

OUTPUTS:

dict out:
A dictionary of the file's arrays, keyed by name."""

demo = [
    {'key': 'ld', 'init': 'np.load', 'pos': (30, 62), 'w': 180, 'h': 100},
    {'key': 'c0', 'comment': True, 'text': 'click "load" and choose an .npz file',
     'pos': (30, 175)},
    {'key': 'dk', 'init': 'dict_keys', 'pos': (30, 215), 'w': 140, 'h': 60},
    {'key': 'l1', 'init': 'list', 'pos': (30, 295), 'w': 400, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'the names of the arrays inside it',
     'pos': (30, 345)},
]
links = [('ld', 'dict out', 'dk', 'dict in'), ('dk', 'keys out', 'l1', '')]
print(build('np.load', 'np.load - read an .npz file', body, demo, links,
            demo_width=480, text_width=790, text_height=600))
