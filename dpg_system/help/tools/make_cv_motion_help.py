"""Dense optical flow, motion features, and flow visualization."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_help import build


body = """These three nodes turn movement between video frames into data.

THE NODES:

cv_optical_flow      compare consecutive pictures and produce a motion field
cv_flow_features     reduce that field to five useful numbers
cv_flow_visualize    turn the field into an RGB direction-and-speed picture

START WITH CONSECUTIVE FRAMES:
cv_optical_flow remembers the preceding picture. The first picture after the
node is created, reset, or given a new image size only fills that memory, so it
produces no output. Each later picture is compared with the one before it.

The input may be a grayscale array or an RGB/RGBA image. cv_camera supplies the
expected uint8 RGB images directly. Floating-point images are treated as 0..1
and converted to bytes before processing.

THE FLOW IS A VECTOR AT EVERY PIXEL:
The result is a float32 array shaped (height, width, 2). The last dimension is
(dx, dy): horizontal and vertical displacement in pixels per PROCESSED frame.
Positive dx points right and positive dy points down in image coordinates.

'process width' makes the calculation cheaper by shrinking wide pictures while
keeping their aspect ratio. Zero uses the original size. Flow distances are in
pixels of that processed image, so changing the width also changes the numeric
scale of the result.

THE FARNEBACK CONTROLS TRADE SPEED FOR STABILITY:
'pyramid scale' controls the size of each successive image-pyramid layer.
'pyramid levels' controls how many scales are examined. More levels can follow
larger movement but cost more. 'window size' averages over a wider region;
larger windows are smoother but lose detail. 'iterations' spends more work
refining the estimate. The defaults are a useful starting point for a camera.

Press 'reset' after a discontinuity such as seeking a movie or switching
sources, otherwise the next field describes the cut rather than real motion.

cv_flow_features MAKES FIVE NUMBERS:
'motion energy' is the mean vector magnitude across every pixel. 'horizontal
motion' and 'vertical motion' are signed means of dx and dy. Opposite movement
can cancel in those signed outputs even while motion energy stays high.

'active fraction' is the fraction of pixels whose magnitude is greater than
'motion threshold'. 'max motion' is the largest magnitude in the frame. All
distance-like values are pixels per processed frame, not pixels per second.

cv_flow_visualize MAKES DIRECTION VISIBLE:
Hue represents direction and brightness represents magnitude. 'full brightness
at' says which magnitude reaches maximum brightness; smaller values reveal
subtle movement and larger values preserve detail in fast movement. The output
is an ordinary uint8 RGB image shaped (height, width, 3).

THE EXAMPLE:
Turn the toggle on to start the camera, then move in front of it. The flow goes
both to the feature node (motion energy is printed) and to the visualization,
which is drawn by mgl_image. Turn the toggle off when finished to release the
camera.

SYNTAX:
cv_optical_flow
cv_flow_features
cv_flow_visualize

INPUTS and PARAMETERS:

RGB image (cv_optical_flow):
Grayscale, RGB, or RGBA NumPy image. A new frame triggers the calculation.

reset:
Forget the previous frame. The next frame becomes the new starting point.

process width (0=original):
Downsample wider inputs to this width before calculating flow.

pyramid scale / pyramid levels / window size / iterations:
Farneback quality, motion-range, smoothing, and refinement controls.

flow (cv_flow_features and cv_flow_visualize):
A float32 (height, width, 2) array carrying dx and dy.

motion threshold (px/frame):
Minimum magnitude counted by active fraction.

full brightness at (px/frame):
Magnitude mapped to full brightness in the RGB visualization.

OUTPUTS:

flow (dx,dy):
Dense float32 motion vectors shaped (height, width, 2).

motion energy / horizontal motion / vertical motion / active fraction / max motion:
Five scalar summaries of the field.

RGB image:
Direction as hue and magnitude as brightness, shaped (height, width, 3).

RELATED:
cv_camera and cv_capture for live pictures.
cv_image for still images (send changing images to measure motion).
mgl_image and the other image nodes for displaying or processing the result."""


demo = [
    {'key': 'on', 'init': 'toggle', 'pos': (30, 62), 'w': 50, 'h': 42},
    {'key': 'cam', 'init': 'cv_camera', 'pos': (30, 120), 'w': 220, 'h': 120,
     'props': {'on/off': False}},
    {'key': 'flow', 'init': 'cv_optical_flow', 'pos': (30, 285), 'w': 245, 'h': 190},
    {'key': 'features', 'init': 'cv_flow_features', 'pos': (30, 530), 'w': 245, 'h': 150},
    {'key': 'printer', 'init': 'print', 'pos': (30, 725), 'w': 180, 'h': 90,
     'props': {'identifier': 'motion energy', 'precision': 3}},
    {'key': 'visualize', 'init': 'cv_flow_visualize', 'pos': (330, 530),
     'w': 245, 'h': 100},
    {'key': 'ctx', 'init': 'mgl_context', 'pos': (330, 62), 'w': 300, 'h': 300,
     'props': {'auto_render': True, 'width': 640, 'height': 480,
               'node_display_width': 300, 'node_display_height': 225,
               'match_texture_to_window': False, 'display_mode': 'node'}},
    {'key': 'image', 'init': 'mgl_image', 'pos': (330, 405), 'w': 245, 'h': 80},
    {'key': 'c0', 'comment': True,
     'text': 'turn on to claim the camera; turn off\nwhen finished so other programs can use it',
     'pos': (30, 835)},
    {'key': 'c1', 'comment': True,
     'text': 'the first frame stores a baseline;\nflow begins with the second frame',
     'pos': (330, 660)},
]

links = [
    ('on', '', 'cam', 'on/off'),
    ('cam', '', 'flow', 'RGB image'),
    ('flow', 'flow (dx,dy)', 'features', 'flow'),
    ('flow', 'flow (dx,dy)', 'visualize', 'flow'),
    ('features', 'motion energy', 'printer', 'in'),
    ('ctx', 'mgl_chain', 'image', 'mgl chain in'),
    ('visualize', 'RGB image', 'image', 'texture'),
]

help_dir = os.path.dirname(HERE)
print(build('cv_optical_flow',
            'cv optical flow - movement between frames',
            body, demo, links, demo_width=680, text_width=800,
            text_height=1110, out_dir=help_dir))
