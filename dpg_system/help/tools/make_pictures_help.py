"""Other ways to get a picture: ndi_receiver and depth_anything, one page each."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """Video over the network, from another machine.

ndi_receiver TAKES VIDEO OFF THE NETWORK:
NDI is how video moves between machines on a local network - one computer sends,
others receive, with no capture card and no cable beyond the ethernet already
there. Anything producing NDI on the network can appear in the 'source' list.

FIND A SOURCE FIRST:
The list starts empty. 'refresh sources' looks for senders - it listens for
about two seconds, and the patch waits while it does - and fills the 'source'
menu. Press it again after a sender is switched on. Choosing a source connects
to it; switching 'on/off' on also connects to whatever the menu shows, if
nothing is connected yet.

BANDWIDTH IS THE TRADE:
'highest' (the default) takes the full stream. 'lowest' takes a much smaller
proxy stream the sender makes alongside it, which is enough for analysis and
not enough to project. 'audio only' takes no picture at all. 'active' is listed
but currently changes nothing.

That is the useful distinction. If the picture is going to a vision model or a
filter, take the proxy and leave the network alone. If it is going on a screen,
take the full stream and expect it to cost you.

WHAT ARRIVES:
Frames come out of 'image' once per frame of the patch while it is on,
repeating the last picture until a new one arrives. Each is four channels -
red, green, blue and alpha, as bytes from 0 to 255. 'output_type' sets the
form: numpy gives an array of height by width by 4, torch a tensor of 4 by
height by width.

Four channels matters: a node that expects an ordinary three-channel colour
picture needs the alpha channel dropped first.

SYNTAX:
ndi_receiver

EXAMPLE:
ndi_receiver

INPUTS and PARAMETERS:

on/off:
Start and stop receiving.

source / refresh sources:
Which NDI sender, and look again for more.

output_type:
numpy (height by width by 4) or torch (4 by height by width).

bandwidth:
highest for the full stream, lowest for the small proxy, audio only for no
picture. Proxy for analysis, full for projection.

OUTPUTS:

image:
The received video, four channels.

RELATED:
cv_camera for a camera attached to this machine.
depth_anything to estimate depth from what arrives.
femto for a depth camera.
The k. and tv. nodes for filtering whatever arrives."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'c1', 'comment': True, 'text': 'on/off', 'pos': (380, 62)},
    {'key': 'ndi', 'init': 'ndi_receiver', 'pos': (30, 120), 'w': 160, 'h': 98},
    {'key': 'c0', 'comment': True, 'text': "'refresh sources' first, and after\nswitching a sender on. Take the PROXY\nfor analysis, the full stream only\nfor projection",
     'pos': (380, 120)},

    {'key': 'inf', 'init': 'info', 'pos': (30, 270), 'w': 235, 'h': 42},
    {'key': 'c3', 'comment': True, 'text': 'frames arrive as arrays of four\nchannels: red, green, blue, alpha',
     'pos': (380, 270)},
]
links = [('tog', '', 'ndi', 'on/off'),
         ('ndi', 'image', 'inf', 'in')]
print(build('ndi_receiver', 'ndi_receiver - video over the network',
            body, demo, links, demo_width=660, text_width=800, text_height=720))


body = """Depth from an ordinary photograph.

depth_anything GUESSES DEPTH FROM ONE IMAGE:
Give it an ordinary picture and it returns a depth image - an estimate, per
pixel, of how far away things are. No depth camera, no stereo pair, no
calibration. It works from the same cues a person uses: perspective, occlusion,
texture getting finer with distance, the way light falls.

THE NUMBERS ARE METRES, BUT THEY ARE GUESSES:
It uses the metric version of the Depth Anything V2 model, trained on indoor
scenes, so each pixel is an estimate in metres, from 0 up to a ceiling of 20.
Trust the ordering more than the numbers. What is nearer and what is further is
reliable; how many metres is plausible in a room and drifts outdoors, where
anything beyond 20 metres is simply 20. It is excellent for separating
foreground from background, for masking, for driving something by depth-order -
and it is not a substitute for femto when the measurement itself matters.

Fed a video it works frame by frame with no memory, so the depth of a static
object can shimmer slightly between frames even when nothing moved. Smooth it if
that matters.

WHAT TO GIVE IT:
A colour picture as a numpy array of height by width by 3. It reads the
channels in OpenCV's blue-green-red order; a picture in red-green-blue order
works too, with little change to the depth. A four-channel picture, such as
ndi_receiver sends, needs its alpha channel dropped first.

The depth image comes back at the size of the picture that went in, as
floating point numbers.

IT IS HEAVY:
The model is loaded the first time a picture arrives, which takes some seconds,
and each frame is a full pass of a large network - on the graphics processor if
there is one. 'small' loads a much lighter model that is faster and coarser.

It needs the Depth Anything V2 code and its checkpoint, in
dpg_system/depth_anything_v2 (the checkpoint goes in its 'checkpoints' folder).
If either is missing it says so in the console the first time it is asked, and
sends nothing.

WHEN TO USE IT:
femto gives you measured metres and a point cloud, and needs a depth camera in
the room. depth_anything gives you estimated depth from any picture at all -
including one arriving over NDI from another building, or a film from 1974.

SYNTAX:
depth_anything [small]

With no argument it uses the large model
(depth_anything_v2_metric_hypersim_vitl.pth); 'small' uses the small one
(depth_anything_v2_metric_hypersim_vits.pth).

EXAMPLE:
depth_anything small

INPUTS:

input_image:
A colour picture, height by width by 3. Receiving it estimates depth.

OUTPUTS:

depth_image:
Estimated depth in metres, one number per pixel, the size of the input.

RELATED:
femto for real depth in metres, from a depth camera.
cv_camera for a picture from a camera on this machine.
ndi_receiver for a picture from another machine."""

demo = [
    {'key': 'c0', 'comment': True, 'text': 'patch a colour picture (height by width\nby 3) into input_image - from cv_camera,\nfor instance',
     'pos': (380, 62)},
    {'key': 'da', 'init': 'depth_anything', 'pos': (30, 62), 'w': 280, 'h': 80},
    {'key': 'inf', 'init': 'info', 'pos': (30, 200), 'w': 235, 'h': 42},
    {'key': 'c4', 'comment': True, 'text': 'depth_image: estimated metres per pixel,\nthe same size as the picture',
     'pos': (380, 200)},
    {'key': 'c6', 'comment': True, 'text': 'trust what is nearer and further more\nthan the numbers. No memory between\nframes, so a still object can shimmer\nslightly - smooth it if that matters',
     'pos': (380, 270)},
]
links = [('da', 'depth_image', 'inf', 'in')]
print(build('depth_anything', 'depth_anything - depth from one picture',
            body, demo, links, demo_width=660, text_width=800, text_height=720))
