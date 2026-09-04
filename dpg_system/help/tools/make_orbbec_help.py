"""Orbbec Femto depth cameras."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """This is the depth camera - the front of the point cloud pipeline.

THE NODE:

femto        an Orbbec Femto. 'femto_bolt' is the same node
femto_bolt   the other spelling

It gives you two things: 'depth', a picture where each pixel is a distance, and
'point_cloud', those distances turned into points in space. The point cloud is
what the pc_ nodes want.

The Bolt uses the same time-of-flight module as the Azure Kinect, so if you know
that camera the geometry here will be familiar.

THE DEPTH MODES ARE A REAL CHOICE:

NFOV unbinned   640 x 576     30 fps    narrow, most detail
NFOV binned     320 x 288     30 fps    narrow, less detail, less data
WFOV binned     512 x 512     30 fps    wide, the usual choice for a room
WFOV unbinned   1024 x 1024   15 fps    wide and detailed, at HALF the rate

Narrow sees further into a smaller cone; wide sees a room but less far. Binning
combines neighbouring pixels - fewer points, and each one quieter, because
averaging removes noise.

The one to notice is WFOV unbinned: it is the only mode that cannot do 30 fps.
If a patch that felt responsive suddenly does not, check whether that mode got
selected - the camera is doing what it was asked, at 15.

'level to gravity' IS THE ONE THAT SAVES THE MOST TIME:
Tick it and the camera works out which way down is, from its own accelerometer,
and rotates the cloud so the floor is flat and vertical is vertical - whatever
angle the camera is actually bolted at.

That matters because a depth camera is almost never level. It is on a bracket,
angled down at a space, and without this every measurement you make is in the
camera's tilted frame rather than the room's. Levelling once at the top makes
every height, every floor test and every crop box downstream mean what it says.

It calibrates for about a second and then STOPS reading the accelerometer,
deliberately: leaving the IMU streaming makes the Bolt deliver depth in clumps
rather than evenly. Toggle it off and on to recalibrate if the camera is moved.

'yaw (deg)' turns the cloud about vertical, for squaring it to the room after
levelling has sorted out the tilt.

'remove background' LEARNS THE EMPTY ROOM:
With 'background frames' frames of an empty space it builds a model and then
drops anything at that distance, so what comes out is only what is new. It is
the same idea as pc_background, done in the camera node before the points are
even built - which is cheaper.

'background guard (mm)' is the tolerance: how much nearer than the learned
surface something must be to count as foreground. Too small and sensor noise
comes through as a crust; too large and someone standing near a wall disappears
into it.

WHEN IT GOES BURSTY - THE USB PROBLEM:
These cameras can drop into a state where frames arrive in clumps every hundred
milliseconds or so instead of steadily. The camera is not broken and the frame
rate is nominally unchanged; the USB session has degraded, and it will stay that
way until it is reset.

'report frame gaps' shows it happening. 'auto usb reset on stutter' recovers
from it without you noticing, and 'reset usb device' does it by hand.

If a camera has gone lumpy mid-rehearsal, that is this - reach for the reset
before suspecting the patch.

THE REST OF THE CLEANING:
'median filter' removes speckle, 'fill holes' patches small gaps where the
sensor got no return - both cost a little time and both make the cloud easier to
work with. 'undistort' corrects the lens. 'flip x/y/z' fix a mirrored or
inverted mounting.

'units' chooses metres or millimetres. Metres, unless something downstream
insists otherwise - the pc_ nodes' crop boxes are written in metres.

WITHOUT THE SDK THE NODE STILL EXISTS:
If pyorbbecsdk is not installed the node is still created and reports the
problem when you enable capture, rather than taking the whole node library down
at import. So a patch built around a camera still opens on a machine that has
none.

SYNTAX:
femto
femto_bolt

EXAMPLE:
femto_bolt

INPUTS and PARAMETERS:

enable:
Start and stop capture.

resolution / fps:
The depth mode, and the rate. WFOV unbinned tops out at 15.

level to gravity / yaw (deg):
Level the cloud using the accelerometer; then square it to the room.

remove background / background frames / background guard (mm):
Learn the empty space and drop it.

units:
Metres or millimetres.

median filter / fill holes / undistort:
Cleaning.

flip x / flip y / flip z:
For a mirrored or inverted mounting.

report frame gaps / auto usb reset on stutter / reset usb device:
Detect and recover from the bursty-USB state.

OUTPUTS: 

depth:
A picture where each pixel is a distance.

point_cloud:
The points, ready for pc_crop.

RELATED:
pc_crop, which should be the very next node - it both reduces the data and tells
everything downstream what volume it is working in.
pc_background and pc_denoise if you want that work done after the camera rather
than inside it."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'c0', 'comment': True, 'text': 'tick to start capture', 'pos': (90, 62)},

    {'key': 'cam', 'init': 'femto_bolt', 'pos': (30, 120), 'w': 340, 'h': 420},
    {'key': 'c1', 'comment': True, 'text': "'level to gravity' is the one that saves\nthe most time - the floor becomes flat\nwhatever angle the camera is bolted at,\nso every height downstream means what\nit says",
     'pos': (30, 555)},

    {'key': 'inf', 'init': 'info', 'pos': (420, 120), 'w': 260, 'h': 80},
    {'key': 'c6', 'comment': True, 'text': 'the depth picture - each pixel a distance',
     'pos': (420, 215)},

    {'key': 'crop', 'init': 'pc_crop', 'pos': (420, 270), 'w': 300, 'h': 200},
    {'key': 'c7', 'comment': True, 'text': 'pc_crop should be the very NEXT node:\nit throws away most of the room and\ntells the rest of the chain what volume\nit is working in',
     'pos': (420, 485)},

    {'key': 'pci', 'init': 'pc_info', 'pos': (420, 620), 'w': 240, 'h': 180},
    {'key': 'i1', 'init': 'int', 'pos': (420, 815), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c11', 'comment': True, 'text': 'how many points survive the crop',
     'pos': (420, 865)},

    {'key': 'c12', 'comment': True, 'text': "IF IT GOES LUMPY mid-rehearsal - frames\nin clumps rather than steadily - that is\nthe USB session degrading, not the patch.\nTurn on 'auto usb reset on stutter'",
     'pos': (30, 725)},
]
links = [('tog', '', 'cam', 'enable'),
         ('cam', 'depth', 'inf', 'in'),
         ('cam', 'point_cloud', 'crop', 'point cloud'),
         ('crop', 'cropped', 'pci', 'point cloud'),
         ('pci', 'count', 'i1', '')]
print(build('femto', 'femto - the depth camera', body,
            demo, links, demo_width=780, text_width=810, text_height=780))
