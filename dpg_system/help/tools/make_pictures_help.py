"""Other ways to get a picture."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """Two other ways to get a picture into a patch.

THE NODES:

ndi_receiver     video over the network, from another machine
depth_anything   depth from an ordinary photograph

ndi_receiver TAKES VIDEO OFF THE NETWORK:
NDI is how video moves between machines on a local network - one computer sends,
others receive, with no capture card and no cable beyond the ethernet already
there. Anything producing NDI on the network appears in the 'source' list.

'refresh sources' looks again, which you need after something is switched on.
'bandwidth' trades quality for network load: the low setting sends a much
smaller proxy stream, which is enough for analysis and not enough to project.

That is the useful distinction. If the picture is going to a vision model or a
filter, take the proxy and leave the network alone. If it is going on a screen,
take the full stream and expect it to cost you.

'output_type' sets the form the frames arrive in.

depth_anything GUESSES DEPTH FROM ONE IMAGE:
Give it an ordinary picture and it returns a depth image - an estimate, per
pixel, of how far away things are. No depth camera, no stereo pair, no
calibration. It works from the same cues a person uses: perspective, occlusion,
texture getting finer with distance, the way light falls.

The catch is that it is RELATIVE. It tells you what is nearer and further, not
how many metres away anything is, and the scale can shift between frames. So it
is excellent for separating foreground from background, for masking, for
driving something by depth-order - and it is not a substitute for femto when the
question has a unit in it.

Fed a video it works frame by frame with no memory, so the depth of a static
object can shimmer slightly between frames even when nothing moved. Smooth it if
that matters.

WHEN TO USE WHICH:
femto gives you real metres and a point cloud, and needs a depth camera in the
room. depth_anything gives you relative depth from any picture at all -
including one arriving over NDI from another building, or a film from 1974.

SYNTAX:
ndi_receiver
depth_anything

EXAMPLE:
ndi_receiver

INPUTS and PARAMETERS:

on/off:
Start receiving.

source / refresh sources:
Which NDI sender, and look for more.

bandwidth:
Full stream, or the small proxy. Proxy for analysis, full for projection.

output_type:
The form the frames take.

input_image:
A picture. Receiving it estimates depth.

OUTPUTS: 

image:
The received video.

depth_image:
Relative depth, per pixel.

RELATED:
femto for real depth in metres, from a depth camera.
cv_camera for a camera attached to this machine.
The k. and tv. nodes for filtering whatever arrives."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'ndi', 'init': 'ndi_receiver', 'pos': (30, 120), 'w': 320, 'h': 220},
    {'key': 'c0', 'comment': True, 'text': "'refresh sources' after switching a\nsender on. Take the PROXY for analysis,\nthe full stream only for projection",
     'pos': (30, 355)},

    {'key': 'inf', 'init': 'info', 'pos': (400, 120), 'w': 260, 'h': 80},
    {'key': 'c3', 'comment': True, 'text': 'frames arrive as arrays, like a camera',
     'pos': (400, 215)},

    {'key': 'da', 'init': 'depth_anything', 'pos': (30, 470), 'w': 280, 'h': 80},
    {'key': 'inf2', 'init': 'info', 'pos': (400, 470), 'w': 260, 'h': 80},
    {'key': 'c4', 'comment': True, 'text': 'depth from an ordinary picture - no\ndepth camera, no stereo, no calibration',
     'pos': (400, 565)},

    {'key': 'c6', 'comment': True, 'text': 'but it is RELATIVE: nearer and further,\nnot metres, and the scale can shift\nbetween frames. Good for masking and\ndepth-order, not for measuring\nno memory between frames, so a still\nobject can shimmer slightly - smooth it\nif that matters',
     'pos': (30, 570)},
]
links = [('tog', '', 'ndi', 'on/off'),
         ('ndi', 'image', 'inf', 'in'),
         ('ndi', 'image', 'da', 'input_image'),
         ('da', 'depth_image', 'inf2', 'in')]
print(build('ndi_receiver', 'ndi_receiver and depth_anything - other pictures',
            body, demo, links, demo_width=700, text_width=800, text_height=720))
