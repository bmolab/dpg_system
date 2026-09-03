"""Vive trackers and base stations."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """Position and orientation from a Vive tracker, in room coordinates.

THESE NODES ARE LINUX ONLY:
They need OpenVR and a running SteamVR, together with the triad_openvr wrapper
alongside this file. On any other platform the module does not load at all -
the console says so at startup and the nodes simply do not exist, so a patch
built around them will open with those nodes missing rather than half working.

That is worth knowing before you go looking for a fault: if 'vive_tracker' is
not in the node list, the machine is the reason.

THE NODES:

vive_tracker         where a tracker is, and which way up
vive_base_stations   whether the base stations are steady enough to believe

RAW TRACKER SPACE IS NOT ROOM SPACE:
The tracking system has its own origin and its own idea of which way is forward,
decided by where the base stations happen to be and how SteamVR was set up. What
you almost always want instead is coordinates in the ROOM - metres from the
middle of the floor, with forward pointing where the audience is.

The play-area settings are that transform. You can type them if you know them,
but the corner capture is the way to get them right.

THE CORNER CAPTURE, WHICH IS THE PART WORTH LEARNING:
Put the tracker on the floor at each corner of the space in turn and press the
matching button - front-left, front-right, back-right, back-left. Then press
'compute_from_corners'.

From those four points it works out all of it at once: the centre of the space,
its width and depth, the yaw needed to square it to the room, and the height of
the floor. It prints what it found, and fills the play-area fields in for you.

The corners do not have to be exact and the space does not have to be square -
it averages the opposite edges, so a slightly trapezoidal capture still gives a
sensible answer. What matters is that the four points are at floor level and in
the right order.

'apply_chaperone' pushes the result into SteamVR itself, so its own boundary
matches the space you just measured.

'clear_corners' starts again.

'output_format' - QUATERNION, EULER OR MATRIX:
Quaternion is the one to prefer for anything that gets combined, interpolated or
filtered. Euler angles are easier to read on screen and are the right choice for
a display or a single angle you want to threshold, but they wrap and they gimbal
and they do not average sensibly.

'which_tracker' selects between up to four.

vive_base_stations IS A DIAGNOSTIC, NOT A SOURCE:
It does not track anything. It watches the base stations and reports whether
they are still where they were, which is what you check when tracking has become
unreliable and you do not yet know why.

It separates two quite different problems:

jitter_mm    how much the reading wobbles from moment to moment, as a spread
             about its own recent average. This is noise - reflections, a
             partly blocked view, sunlight.
drift_mm     how far that average has MOVED from a baseline you captured. This
             is the mounting shifting - a bumped stand, a flexing truss, a
             building warming up over an afternoon.

They need different remedies, which is why they are separate outlets. Jitter is
an environment problem; drift is a screwdriver problem.

Press 'set_baseline' once when everything is right - after a warm-up, before the
audience arrives. From then on 'drift_mm' answers "has anything moved since
then", and 'all_stable' is a single yes-or-no against your thresholds, which is
the one to wire to a warning light.

'window_size' is how many samples the statistics cover, so it sets how quickly
they respond and how much they smooth.

SYNTAX:
vive_tracker
vive_base_stations

EXAMPLE:
vive_tracker

INPUTS and PARAMETERS:

enable_in:
Start reading.

which_tracker / output_format:
Which of up to four, and quaternion, euler or matrix.

capture_FL_corner ... capture_BL_corner:
Record the tracker's position at each corner, in order.

compute_from_corners:
Work out the play area from those four points.

apply_chaperone / clear_corners:
Push the result into SteamVR; start the capture again.

play_area_x_m / play_area_z_m / play_area_yaw_deg / play_area_center_x_m /
play_area_center_z_m / floor_height_m:
The transform itself, if you would rather type it.

window_size / jitter_threshold_mm / drift_threshold_mm (base stations):
How much history the statistics cover, and what counts as too much.

set_baseline / print_report / reset:
Capture the reference, write a report, start over.

OUTPUTS: 

position / orientation:
Where the tracker is, in room coordinates, and which way it is facing.

connected:
Whether it is being seen.

num_stations / serials / positions_m / orientations:
What base stations are present and where.

jitter_mm / orient_jitter_deg / drift_mm:
Wobble, and movement since the baseline.

all_stable:
One number: everything within thresholds, or not.

report:
The written version.

RELATED:
shadow nodes for a whole body rather than one point.
quaternion nodes for working with the orientation once you have it."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'c0', 'comment': True, 'text': 'LINUX ONLY - needs OpenVR and SteamVR.',
     'pos': (90, 62)},
    {'key': 'c1', 'comment': True, 'text': 'On other platforms these nodes do not',
     'pos': (90, 92)},
    {'key': 'c2', 'comment': True, 'text': 'exist at all, and the console says so',
     'pos': (90, 122)},

    {'key': 'vt', 'init': 'vive_tracker', 'pos': (30, 170), 'w': 340, 'h': 560},
    {'key': 'c3', 'comment': True, 'text': 'walk the tracker to each corner of the',
     'pos': (30, 745)},
    {'key': 'c4', 'comment': True, 'text': 'floor, capture in order FL FR BR BL,',
     'pos': (30, 775)},
    {'key': 'c5', 'comment': True, 'text': 'then compute_from_corners - it works out',
     'pos': (30, 805)},
    {'key': 'c6', 'comment': True, 'text': 'centre, size, yaw and floor height at',
     'pos': (30, 835)},
    {'key': 'c7', 'comment': True, 'text': 'once, and fills the fields in',
     'pos': (30, 865)},

    {'key': 'inf', 'init': 'info', 'pos': (420, 170), 'w': 260, 'h': 80},
    {'key': 'c8', 'comment': True, 'text': 'position, in metres from the middle of',
     'pos': (420, 265)},
    {'key': 'c9', 'comment': True, 'text': 'the floor - not raw tracker space',
     'pos': (420, 295)},
    {'key': 'l1', 'init': 'list', 'pos': (420, 340), 'w': 300, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c10', 'comment': True, 'text': 'orientation - quaternion by default,',
     'pos': (420, 390)},
    {'key': 'c11', 'comment': True, 'text': 'which is what to prefer for anything',
     'pos': (420, 420)},
    {'key': 'c12', 'comment': True, 'text': 'combined, filtered or interpolated',
     'pos': (420, 450)},

    {'key': 'bs', 'init': 'vive_base_stations', 'pos': (420, 500), 'w': 320, 'h': 400},
    {'key': 'pl', 'init': 'plot', 'pos': (780, 500), 'w': 300, 'h': 180,
     'props': PLOT(0.0, 5.0, 200)},
    {'key': 'c13', 'comment': True, 'text': 'JITTER: wobble about its own average -',
     'pos': (780, 690)},
    {'key': 'c14', 'comment': True, 'text': 'reflections, a blocked view, sunlight',
     'pos': (780, 720)},
    {'key': 'pl2', 'init': 'plot', 'pos': (780, 765), 'w': 300, 'h': 180,
     'props': PLOT(0.0, 20.0, 200)},
    {'key': 'c15', 'comment': True, 'text': 'DRIFT: how far that average has moved',
     'pos': (780, 955)},
    {'key': 'c16', 'comment': True, 'text': 'since you set the baseline - a bumped',
     'pos': (780, 985)},
    {'key': 'c17', 'comment': True, 'text': 'stand, a flexing truss, a warm building',
     'pos': (780, 1015)},
    {'key': 'c18', 'comment': True, 'text': 'jitter is an environment problem,',
     'pos': (780, 1055)},
    {'key': 'c19', 'comment': True, 'text': 'drift is a screwdriver problem',
     'pos': (780, 1085)},

    {'key': 'tg2', 'init': 'toggle', 'pos': (420, 930), 'w': 45, 'h': 42},
    {'key': 'c20', 'comment': True, 'text': 'all_stable - wire this to a warning',
     'pos': (480, 930)},
    {'key': 'c21', 'comment': True, 'text': 'light and set the baseline once when',
     'pos': (480, 960)},
    {'key': 'c22', 'comment': True, 'text': 'everything is right', 'pos': (480, 990)},
]
links = [('tog', '', 'vt', 'enable_in'),
         ('vt', 'position', 'inf', 'in'),
         ('vt', 'orientation', 'l1', ''),
         ('bs', 'jitter_mm', 'pl', 'y'),
         ('bs', 'drift_mm', 'pl2', 'y'),
         ('bs', 'all_stable', 'tg2', '')]
print(build('vive_tracker', 'vive_tracker - a point in the room (Linux only)',
            body, demo, links, demo_width=1120, text_width=810, text_height=790))
