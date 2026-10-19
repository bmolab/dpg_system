"""np limits, running totals and targets: one page per node.

np.clip, np.min, np.max (one class, NumpyClipNode) were the np_clip family
page; their pages are hand-made now (see tools/README.md), so this script no
longer builds them. np.cumsum, np.cumprod, np.diff were np_cumulative, and
are hand-made now too. np.distance_from_target, np.proximity_to_target,
proximity_trigger were np_target. Each now has its own page whose demo shows
the input and the output, with a SEE ALSO row under it; the
np.distance_from_target page is hand-made now, so this script builds only
proximity_trigger and np.proximity_to_target.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT


LIST = {'text in': '', 'font size': '24'}


def see_also(y, stems, x=30, right=580):
    """A SEE ALSO row: one help_link per related page, left to right, wrapping
    onto a second row rather than running under the prose."""
    row = []
    x0 = x
    for s in stems:
        w = int(len(s) * 7.7) + 30
        if x > x0 and x + w > right:
            x, y = x0, y + 44
        row.append({'key': 'sa_' + s, 'init': 'help_link ' + s, 'pos': (x, y),
                    'w': w, 'h': 30})
        x += w + 14
    if x + 200 > right:
        x, y = x0, y + 44
    row.append({'key': 'sa_c', 'comment': True, 'text': 'SEE ALSO - click to open',
                'pos': (x, y)})
    return row


def MULTI_PLOT(miny=-1.0, maxy=1.0, count=200, width=300):
    """A plot that draws each element of a list (or each row) as its own line."""
    return dict(PLOT(miny, maxy, count),
                **{'width': width, 'update style': 'input is multi-channel sample'})


def text_h(body):
    """Height of the prose block: 15px per line at text size 30, plus a margin."""
    return (body.count('\n') + 1) * 15 + 30


def set_state(path, node_name, state):
    """Add saved node state (what a node's save_custom writes) to a built page.

    build() has no slot for it; the target nodes keep their targets this way,
    so the demo works the moment the page opens."""
    doc = json.load(open(path))
    patches = list(doc['patches'].values()) if 'patches' in doc else [doc]
    for p in patches:
        for n in p['nodes'].values():
            if n.get('name') == node_name:
                n.update(state)
    json.dump(doc, open(path, 'w'), indent=4)


# --------------------------------------------------------- demo subpatchers
# Inside make_point: two sines at different speeds make an x y point that
# wanders over the square from -1 to 1, coming back near any spot every few
# seconds (the path repeats every 12 seconds).
POINT_SUB = {
    'name': 'make_point',
    'host': 'mp',
    'demo': [
        {'key': 'sx', 'init': 'signal 3.0 sin', 'pos': (40, 40), 'w': 129, 'h': 78,
         'props': SIG('sin', 3.0)},
        {'key': 'sy', 'init': 'signal 4.0 cos', 'pos': (200, 40), 'w': 129, 'h': 78,
         'props': SIG('cos', 4.0)},
        {'key': 'cx', 'init': 'comment x: once every 3 s, y: once every 4 s, both -1 to 1',
         'pos': (40, 135), 'w': 460, 'h': 30},
        {'key': 'pk', 'init': 'pack 2', 'pos': (40, 180), 'w': 130, 'h': 100},
        {'key': 'cp', 'init': 'comment the point as a list: x y',
         'pos': (200, 200), 'w': 260, 'h': 30},
        {'key': 'out', 'init': 'out point', 'pos': (40, 310), 'w': 150, 'h': 80},
    ],
    'links': [('sx', '', 'pk', 'in 1'), ('sy', '', 'pk', 'in 2'),
              ('pk', 'out', 'out', 'point')],
}

TARGET = [0.5, 0.7]
TARGETS = [[0.5, 0.7], [-0.5, -0.7], [0.0, 0.0]]


# np.clip, np.min, np.max: their pages are hand-made now (see tools/README.md).

# np.cumsum, np.cumprod, np.diff: their pages are hand-made now (see tools/README.md).

# np.distance_from_target: its page is hand-made now (see tools/README.md).

# ------------------------------------------------------------- proximity_trigger
body = """Fires an event when its input arrives near a stored target, and
counts the arrivals.

It turns "how far is it" into "it got there": when the distance from the
input to the target drops below threshold, state sends true and count
goes up by one. It cannot fire again until the input has gone further
away than release_threshold, when state sends false. Use it to trigger a
cue when a dancer's hand reaches a spot, or to count how often a tracked
point visits a place.

TWO THRESHOLDS:
The gap between threshold and release_threshold is hysteresis, the same
idea as the hysteresis node: without it, a hand hovering at the boundary
would fire over and over as tracking noise carries it back and forth.
Keep release_threshold larger than threshold; if it is smaller, the
trigger releases on the very next input and re-fires while you stay
near.

SYNTAX:
proximity_trigger

INPUTS and PARAMETERS:

input:
The current position, as a list or array of any length. Every input is
measured against the target. The node does nothing at all until a
target is set and arm is ticked. An input whose shape does not fit the
target sends nothing.

arm (checkbox):
Switches the trigger on. It starts OFF, and while it is off nothing is
measured or sent. Ticking or unticking it also sets the count back to
0 and sends that 0 from count.

reset_count (button):
Sets the count back to 0 and sends the 0 from count at once.

set target (button):
Captures whatever was last received at input as the target. The target
is saved with the patch - this page opens with 0.5 0.7.

threshold (drag float, default 0.3):
How close counts as arrived. Strictly closer than this fires.

release_threshold (drag float, default 0.5):
How far it must go (this far or further) before it can fire again.

axis (menu, also an inlet):
Only matters when the input holds several positions, one per row. With
None (the default) the whole difference is one distance. With 1 each row
has its own distance, and the NEAREST decides: any row inside threshold
fires, and it releases only when every row is past release_threshold.

OUTPUTS:

state:
true on arrival, false on release - sent only when it changes.

count:
How many arrivals since the count was last reset, sent on each arrival
and on each reset.

proximity:
Despite its name, the DISTANCE to the target, sent on every input while
armed (one per row with axis 1). It gets smaller as you approach, and is
0 at the target.

READING THE DEMO:
make_point moves an x y point around the square from -1 to 1. The plot
shows its distance from the target 0.5 0.7, with two flat lines at 0.3
and 0.5 - the thresholds as saved. Each time the curve dips below the
lower line the toggle turns on and the count goes up; the toggle turns
off only when the curve climbs back past the upper line. The point
arrives about every three seconds. Drag the thresholds on the node (the
lines stay at the saved values), or click set target to move the target.

RELATED:
np.distance_from_target sends the same distance without the event.
np.proximity_to_target grades closeness to several targets from 0 to 1.
trigger and hysteresis do the same two-threshold firing on a single
number."""

demo = [
    {'key': 'mp', 'init': 'p make_point', 'pos': (30, 62), 'w': 170, 'h': 70},
    {'key': 'c0', 'comment': True, 'text': 'an x y point wandering from -1 to 1',
     'pos': (30, 150)},
    {'key': 'pt', 'init': 'proximity_trigger', 'pos': (30, 190), 'w': 240, 'h': 220,
     'props': {'arm': True, 'threshold': 0.3, 'release_threshold': 0.5, 'axis': 'None'}},
    {'key': 'c1', 'comment': True, 'text': 'armed; target 0.5 0.7', 'pos': (30, 195)},
    {'key': 'tg', 'init': 'toggle', 'pos': (30, 440), 'w': 45, 'h': 42},
    {'key': 'i1', 'init': 'int', 'pos': (110, 440), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'state: on while arrived.  count of arrivals',
     'pos': (30, 445)},
    {'key': 'pk', 'init': 'pack f 0.3 0.5', 'pos': (30, 510), 'w': 150, 'h': 100},
    {'key': 'c3', 'comment': True, 'text': 'the distance, with lines at 0.3 and 0.5',
     'pos': (30, 515)},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 630), 'w': 308, 'h': 178,
     'props': MULTI_PLOT(0.0, 2.5)},
    {'key': 'c4', 'comment': True,
     'text': 'fires when the curve dips under 0.3,\nre-arms when it climbs back past 0.5',
     'pos': (30, 820)},
] + see_also(890, ['np.distance_from_target', 'np.proximity_to_target', 'trigger', 'hysteresis'])
links = [('mp', 'point', 'pt', 'input'), ('pt', 'state', 'tg', ''),
         ('pt', 'count', 'i1', ''), ('pt', 'proximity', 'pk', 'float'),
         ('pk', 'out', 'pl', 'y')]
p = build('proximity_trigger', 'proximity_trigger - an event on arrival at a target',
          body, demo, links, demo_width=600, text_width=760, text_height=text_h(body),
          subpatch=POINT_SUB)
set_state(p, 'proximity_trigger', {'target': TARGET})
print(p)

# -------------------------------------------------------- np.proximity_to_target
body = """Measures how close its input is to each of a set of targets, as a
level from 1 (at the target) falling to 0 at the edge of the target's
zone - the shape you want for fading sounds in and out as a tracked point
moves about.

Each target is a spot with a round zone around it, radius wide. Inside
the zone the level rises as you approach; outside it the level is 0.
The output is made to plug straight into the fade inlet of
polyphonic_sampler~: target 0 drives one sound, target 1 the next, and
so on, so walking from spot to spot crossfades between sounds.

SETTING THE TARGETS:
Choose a slot with target index, put the tracker where that target
should be, and click capture. Choose another index and capture again.
Skipping ahead (index 5 when there are 2) leaves the slots in between
unset; they report 0. The targets are saved with the patch - this page
opens with three: 0.5 0.7, -0.5 -0.7 and 0 0. Capturing a point of a
different length (x y z into a set of x y targets) throws the whole set
away and starts a new one.

SYNTAX:
np.proximity_to_target

INPUTS and PARAMETERS:

input:
The tracker position, as a list or array (x y, x y z...), or several
trackers, one per row. Every input is measured and sent. Nothing is sent
until there are targets, or when the input's length does not match
theirs.

target index (int box) / capture (button):
The slot to capture into, and the button that copies the last input into
it. capture sends the new set from targets and measures at once.

targets:
Replaces the whole set: a list of positions, one per row ([[x, y], ...]).
A flat list is taken as ONE target.

clear (button):
Forgets every target, fades out whatever is sounding (one last 0 for
each), and sends an empty list from targets.

radius (drag float, default 0.5):
The size of each zone, in the input's units. The level is 1 minus
(distance / radius), so it falls in a straight line from 1 to 0.

exponent (drag float, default 1.0):
Shapes that fall. Above 1 the level stays low until you are close and
then rises sharply - a tighter spot. Below 1 it rises quickly at the
edge and stays high across the zone - a broad plateau.

first sound (int box, default 0):
The sound number target 0 drives; target i drives first sound + i.
Changing it fades out the sounds that were on.

OUTPUTS:

fade:
[[sound, level], ...] for every target whose zone you are in. It is
sparse: a sound is listed while its level is above 0, gets a single 0
the moment you leave its zone (so the sampler stops it), and is then
left out. With nothing in range and nothing just left, nothing is sent.
With several trackers the closest one sets each target's level. Levels
are rounded to 4 decimals.

nearest:
The index of the target with the highest level, or -1 when the tracker
is in no zone (one per tracker with several).

targets:
The whole set of targets, sent when it changes (capture, the targets
inlet, clear). Unset slots are nan.

READING THE DEMO:
make_point moves an x y point around the square from -1 to 1. As it
passes through a zone, fade lists that target's sound with a level that
rises to near 1 at the centre and falls away, then sends one last 0;
nearest switches between 0, 1 and 2, and is -1 between zones, as the
plot shows. Where two zones overlap (0.5 0.7 and 0 0 just do), fade
lists two sounds at once; drag radius up and the zones grow until they
overlap widely. Raise exponent and the levels stay
lower except near the centres.

RELATED:
np.distance_from_target gives the plain distance to one target.
proximity_trigger fires an event on arrival instead of a level.
polyphonic_sampler~ plays the sounds this fades. effort_fader sends
fades in the same sparse format."""

demo = [
    {'key': 'mp', 'init': 'p make_point', 'pos': (30, 62), 'w': 170, 'h': 70},
    {'key': 'c0', 'comment': True, 'text': 'an x y point wandering from -1 to 1',
     'pos': (30, 150)},
    {'key': 'px', 'init': 'np.proximity_to_target', 'pos': (30, 190), 'w': 240, 'h': 280,
     'props': {'target index': 0, 'radius': 0.5, 'exponent': 1.0, 'first sound': 0}},
    {'key': 'c1', 'comment': True, 'text': 'three targets saved; drag radius, exponent',
     'pos': (30, 195)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 490), 'w': 360, 'h': 42, 'props': LIST},
    {'key': 'c2', 'comment': True, 'text': 'fade: [[sound, level], ...]', 'pos': (30, 495)},
    {'key': 'i1', 'init': 'int', 'pos': (30, 550), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c3', 'comment': True, 'text': 'nearest target, -1 for none', 'pos': (30, 555)},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 610), 'w': 308, 'h': 178,
     'props': dict(PLOT(-1.5, 2.5, style='stair'), width=300)},
    {'key': 'c4', 'comment': True, 'text': 'nearest over time: 0, 1, 2, or -1 between zones',
     'pos': (30, 800)},
    {'key': 'l2', 'init': 'list', 'pos': (30, 840), 'w': 360, 'h': 42, 'props': LIST},
    {'key': 'c5', 'comment': True, 'text': 'targets: sent when the set changes (capture)',
     'pos': (30, 845)},
] + see_also(920, ['np.distance_from_target', 'proximity_trigger',
                   'polyphonic_sampler~', 'effort_fader'])
links = [('mp', 'point', 'px', 'input'), ('px', 'fade', 'l1', ''),
         ('px', 'nearest', 'i1', ''), ('px', 'nearest', 'pl', 'y'),
         ('px', 'targets', 'l2', '')]
p = build('np.proximity_to_target', 'np.proximity_to_target - closeness to several targets, 1 to 0',
          body, demo, links, demo_width=600, text_width=760, text_height=text_h(body),
          subpatch=POINT_SUB)
set_state(p, 'np.proximity_to_target', {'targets': TARGETS})
print(p)
