"""Camera and image input via OpenCV."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """These are where pictures come into a patch.

THE NODES:

cv_camera    a live camera. 'cv_capture' is the same node
cv_capture   the other spelling
cv_image     a still image from a file

cv_camera IS A TAP YOU TURN ON:
Tick 'on/off' and frames start arriving, one per rendered frame, as RGB arrays.
Untick it and the camera is released - which matters, because a camera held open
by one patch is not available to anything else.

'source' lists the cameras it found. On a laptop that is usually just the built
in one; plug in another and press 'refresh' to look again.

REFRESHING PROBES EVERY CAMERA, ONE BY ONE:
'refresh' finds cameras by trying to OPEN each index in turn, up to ten. There
is no polite way to ask; that is how it has to be done.

Two consequences. It takes a moment, and it briefly opens every camera on the
machine - so an indicator light may flicker on a camera you are not using, and
another application holding one may see it disturbed. The node refreshes once
when it is created, and after that only when you press the button, so this is
not happening continuously.

WHAT COMES OUT IS RGB, ALREADY CONVERTED:
OpenCV works in BGR - blue, green, red - which is a long-standing quirk of the
library and a reliable source of pictures that look wrong in an unmistakable
way, with skin tones gone blue. These nodes convert to RGB before sending, so
what leaves them is the ordinary order and everything downstream is right.

Worth knowing anyway, because if you ever hand an array to an OpenCV function
yourself, it will expect BGR and you will have to convert back.

THE ARRAY IS HEIGHT, WIDTH, CHANNELS:
Frames arrive as (height, width, 3) - rows first. Most of the image nodes here
expect channels FIRST, and guess when they are handed the other order; the guess
is right for anything camera-sized. See the k. and tv. pages for when it is not.

cv_image IS A FILE, LOADED ON DEMAND:
Put a path in 'path in' and send anything to 'show image' to load and send it.
It is not watching the file - it reads when triggered, so the same node can walk
through a directory if something else is supplying the paths.

WHAT TO PUT AFTER IT:
A camera at thirty frames a second is a lot of data, and most of what you might
do with it does not need every frame or the full resolution. Anything expensive
downstream - a vision model, a large filter - is better fed from a subsample,
and the vision_describe nodes will drop frames by themselves rather than fall
behind.

SYNTAX:
cv_camera
cv_capture
cv_image <path>

EXAMPLE:
cv_camera

INPUTS and PARAMETERS:

on/off:
Whether the camera is running. Off releases it.

source:
Which camera, by index.

refresh:
Look for cameras again. Opens each one to find out.

show image (cv_image):
Anything at all - it triggers the load.

path in (cv_image):
Where the file is.

OUTPUTS: 

The single outlet carries the picture, as an RGB array of height by width by 3.

RELATED:
vision_describe to get words out of what the camera sees.
k. and tv. nodes for filtering and adjusting the image.
mgl_texture and the gl nodes to put it on screen."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'cam', 'init': 'cv_camera', 'pos': (30, 115), 'w': 240, 'h': 160},
    {'key': 'c0', 'comment': True, 'text': "tick to start. Untick to release the\ncamera for something else to use\n'refresh' opens every camera index in\nturn to find them - it is not instant",
     'pos': (30, 290)},

    {'key': 'inf', 'init': 'info', 'pos': (320, 115), 'w': 260, 'h': 80},
    {'key': 'c4', 'comment': True, 'text': 'height, width, 3 - rows first, and\nalready RGB, not OpenCV BGR',
     'pos': (320, 210)},

    {'key': 'sub', 'init': 'subsample 10', 'pos': (30, 440), 'w': 170, 'h': 80,
     'props': {'rate': 10}},
    {'key': 'c6', 'comment': True, 'text': 'thin the stream before anything\nexpensive downstream',
     'pos': (30, 530)},

    {'key': 'gs', 'init': 'tv.Grayscale', 'pos': (30, 605), 'w': 200, 'h': 80},
    {'key': 'sq', 'init': 't.squeeze', 'pos': (30, 700), 'w': 160, 'h': 80},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 795), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 64,
               'min y': 0.0, 'max y': 255.0, 'update_mode': 'heat_map',
               'number format': '%.0f'}},
    {'key': 'c8', 'comment': True, 'text': 'the camera, as one brightness channel',
     'pos': (30, 955)},

    {'key': 'img', 'init': 'cv_image', 'pos': (320, 440), 'w': 320, 'h': 120},
    {'key': 'btn', 'init': 'button', 'pos': (320, 380), 'w': 88, 'h': 46},
    {'key': 'inf2', 'init': 'info', 'pos': (320, 580), 'w': 260, 'h': 80},
    {'key': 'c9', 'comment': True, 'text': 'a still, loaded when triggered - so\nit can walk a directory if something\nelse supplies the paths',
     'pos': (320, 675)},
]
links = [('tog', '', 'cam', 'on/off'),
         ('cam', '', 'inf', 'in'),
         ('cam', '', 'sub', 'input'),
         ('sub', 'out', 'gs', 'tensor in'),
         ('gs', 'output', 'sq', 'tensor in'), ('sq', 'output', 'hm', 'y'),
         ('btn', '', 'img', 'show image'),
         ('img', '', 'inf2', 'in')]
print(build('cv_camera', 'cv_camera and cv_image - pictures in', body,
            demo, links, demo_width=680, text_width=800, text_height=720))
