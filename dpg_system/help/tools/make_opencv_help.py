"""Camera and image input via OpenCV."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ------------------------------------------------------------------ cv_camera
body = """cv_camera brings a live camera into the patch, as a stream of pictures.

THE NAMES:
cv_camera and cv_capture are the same node.

IT IS A TAP YOU TURN ON:
Tick 'on/off' and frames start arriving, one per frame of the patch as far as
the camera keeps up, as RGB arrays. Untick it and the frames stop.

Unticking also lets go of the camera: its light goes off and other
applications can use it. Ticking again reopens it, which takes a moment.
Choosing a different 'source' while on switches cameras at once; while off it
just sets which camera the next tick opens.

'source' lists the cameras it found. On a laptop that is usually just the built
in one; plug in another and press 'refresh' to look again.

REFRESHING PROBES EVERY CAMERA, ONE BY ONE:
'refresh' finds cameras by trying to OPEN each index in turn, from 0 to 9.
There is no polite way to ask; that is how it has to be done.

Two consequences. It takes a moment, and it briefly opens every camera on the
machine - so an indicator light may flicker on a camera you are not using, and
another application holding one may see it disturbed. The node refreshes once
when it is created, and after that only when you press the button.

WHAT COMES OUT IS RGB, ALREADY CONVERTED:
OpenCV works in BGR - blue, green, red - a long-standing quirk of the library
and a reliable source of pictures with skin tones gone blue. cv_camera converts
to RGB before sending, so everything downstream gets the ordinary order.

If you ever hand an array to an OpenCV function yourself, it will expect BGR
and you will have to convert back.

THE ARRAY IS HEIGHT, WIDTH, CHANNELS:
Frames arrive as (height, width, 3) bytes - rows first. Most of the image
nodes here expect channels FIRST, and guess when they are handed the other
order; the guess is right for anything camera-sized. See the k. and tv. pages
for when it is not.

WHAT TO PUT AFTER IT:
A camera at thirty frames a second is a lot of data, and most of what you might
do with it does not need every frame or the full resolution. Anything expensive
downstream - a vision model, a large filter - is better fed from a subsample,
and the vision_describe nodes will drop frames by themselves rather than fall
behind.

SYNTAX:
cv_camera
cv_capture

EXAMPLE:
cv_camera

INPUTS and PARAMETERS:

on/off:
Whether frames are sent. On opens the camera, off releases it.

source:
Which camera, by index. Choosing one while on switches to it.

refresh:
Look for cameras again. Opens each one to find out.

OUTPUTS:

The single outlet carries the picture, as an RGB array of height by width by 3.

RELATED:
cv_image for a still picture from a file; it used to share this page.
movie_player for recorded video.
vision_describe to get words out of what the camera sees.
k. and tv. nodes for filtering and adjusting the image.
mgl_texture and the gl nodes to put it on screen."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'cam', 'init': 'cv_camera', 'pos': (30, 115), 'w': 240, 'h': 160},
    {'key': 'c0', 'comment': True, 'text': "tick to start, untick to stop and\nrelease the camera\n'refresh' opens every camera index in\nturn to find them - it is not instant",
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
]
links = [('tog', '', 'cam', 'on/off'),
         ('cam', '', 'inf', 'in'),
         ('cam', '', 'sub', 'input'),
         ('sub', 'out', 'gs', 'tensor in'),
         ('gs', 'output', 'sq', 'tensor in'), ('sq', 'output', 'hm', 'y')]
print(build('cv_camera', 'cv_camera - a live camera', body,
            demo, links, demo_width=680, text_width=800, text_height=800))

# ------------------------------------------------------------------- cv_image
body = """cv_image reads a picture from a file and sends it.

IT READS WHEN THE PATH ARRIVES:
Send a path to 'path in' - or type one into it - and the file is read and sent
straight away. A path given as the node's argument (cv_image <path>) is read
when the node is made. 'show image' sends the picture already read again; it
does not go back to the file.

So the node is not watching the file. To pick up a changed file, send the path
again. To walk through a directory, have something else supply one path after
another.

A path that cannot be read prints a message in the console, and the picture
read before it is dropped: 'show image' then sends nothing until a good path
arrives.

THE ARRAY IS HEIGHT, WIDTH, CHANNELS - IN RGB:
The picture comes out as (height, width, 3) bytes, rows first, whatever the
file held - an alpha channel is dropped and a grey image is made three-channel.

OpenCV reads files in blue, green, red order; cv_image converts to RGB before
sending, like cv_camera and movie_player, so the pictures from all three can
go to the same places. If you hand the array to an OpenCV function yourself,
it will expect BGR and you will have to convert back.

SYNTAX:
cv_image <path>

EXAMPLE:
cv_image

INPUTS and PARAMETERS:

show image:
Anything at all sends the picture again.

path in:
The file. Receiving it reads the file and sends the picture.

OUTPUTS:

The single outlet carries the picture, as an RGB array of height by width by 3.

RELATED:
cv_camera for live pictures; cv_image used to share its page.
movie_player for video files.
k. and tv. nodes for filtering and adjusting the image."""

demo = [
    {'key': 'path', 'init': 'string', 'pos': (30, 62), 'w': 520, 'h': 42,
     'props': {'text in': '/path/to/a/picture.jpg', 'font size': '24', 'width': 480}},
    {'key': 'c0', 'comment': True, 'text': 'put a real path here and press return',
     'pos': (30, 112)},
    {'key': 'btn', 'init': 'button', 'pos': (380, 150), 'w': 88, 'h': 46},
    {'key': 'c1', 'comment': True, 'text': 'sends the same picture again',
     'pos': (380, 205)},
    {'key': 'img', 'init': 'cv_image', 'pos': (30, 250), 'w': 320, 'h': 120},
    {'key': 'inf', 'init': 'info', 'pos': (30, 390), 'w': 260, 'h': 80},
    {'key': 'c2', 'comment': True, 'text': 'height, width, 3 - red, green, blue,\nthe same order as the camera',
     'pos': (30, 485)},
]
links = [('path', 'string out', 'img', 'path in'),
         ('btn', '', 'img', 'show image'),
         ('img', '', 'inf', 'in')]
print(build('cv_image', 'cv_image - a picture from a file', body,
            demo, links, demo_width=560, text_width=800, text_height=640))
