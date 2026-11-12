"""pc_accumulate: where the cloud has been, as a cloud."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, INT

body = """pc_accumulate records where a cloud has been, and sends it as a cloud of voxels.

Tick 'record' and every voxel the cloud touches is counted, frame after
frame. The output is one point per voxel that was touched at least 'min hits'
times, with a weight saying how much went on there. It is the C++ app's
ACCUMULATE VOXELS: walk through the space, stop, and you have a picture of
everywhere you went - ready for pc_regions' 'capture cloud in'.

The input passes straight through on 'cloud out', so it can sit in the middle
of a chain.

THE WEIGHTS ADD UP:
Each voxel's weight is the SUM, over the recorded frames, of the weight the
incoming cloud gave it. From pc_voxel that is its per-voxel weight; a cloud
with no weights counts 1 for each frame a voxel is occupied. So a voxel you
stood in for a second reads thirty times what one frame gave it, and an
accumulation is always at least as strong as any single frame.

That makes the raw numbers grow without limit, and mgl_point_cloud can only
show 0 to 1 - it clamps anything above. Two controls choose which part of
the accumulation lands in that visible range.

'normalize' rescales the sums first:

  none        the raw sums. Grows the longer you record.
  per frame   divided by the frames recorded: the average weight per frame.
              Stays the same scale however long you record.
  max         divided by the largest voxel shown, so the busiest one reads 1
              and everything else is a fraction of it.

'weight scale' then multiplies the result. With 'max', lower values than 1
are pointless and higher ones push the quieter voxels up into view while the
busiest saturate - the way to read detail in the places you passed through
rather than the places you stood.

mgl_point_cloud has a 'weight scale' of its own, applied when it draws. It
changes only the picture; this one changes the numbers that go on to
pc_regions too.

A FADING TRAIL:
'half-life (s)' makes the accumulation fade while it records: everything
gathered so far loses half its weight every half-life, so recent presence
reads brightest and the past falls away behind it. It goes by the clock,
not by frames, so a frame rate change does not change how fast it fades. 0
(the default) is no fading.

A voxel nobody has been in for a while is forgotten outright, not just
dimmed - once what it had has faded to a twentieth of a single frame's,
about four half-lives after a brief visit and longer after a long stay. It
then stops being sent, so a capture never takes a faded trail along with
it.

Nothing fades while 'record' is off. Untick it and the accumulation holds
as it is, ready to capture. With 'per frame', the frame count it divides by
fades too, so it stays an average over the recent past.

MIN HITS:
'min hits' drops voxels touched in fewer than that many frames - raise it to
lose what was only walked through and keep where you stayed. It counts
frames, not weight (with a half-life, frames since the voxel was last
forgotten), and it is applied before 'max' finds its maximum.

RECORDING ALONE:
'delay (s)' and 'duration (s)' (options) are for working without a second
person at the computer: tick 'record', walk into the volume during the delay,
and it stops itself after the duration. A duration of 0 records until you
untick it. 'clear' empties the accumulation; so does changing the volume or
the voxel size, since the old counts no longer map onto the new grid.

THE GRID:
Like every point cloud node it works on a voxel grid. When the cloud comes
from pc_voxel or pc_crop, the volume and voxel size ride in on the frame and
pc_accumulate uses those; 'voxel size (cm)' and 'min (x,y,z)' / 'max (x,y,z)'
(options) are only for a raw cloud with nothing attached.

SYNTAX:
pc_accumulate

EXAMPLE:
pc_accumulate

INPUTS and PARAMETERS:

point cloud:
The cloud: an (N, 3) array or a cloud frame. Each one is a frame of the
recording while 'record' is on.

record:
Accumulate while ticked.

clear:
Empty the accumulation.

min hits:
Frames a voxel must be touched in to be sent. Default 1.

normalize:
none, per frame or max, as above.

weight scale:
Multiplies the weights after normalize. Default 1.

half-life (s):
How fast the accumulation fades while recording. 0 is no fading.

delay (s) / duration (s) / voxel size (cm) / min (x,y,z) / max (x,y,z)
(options):
As above.

OUTPUTS:

accumulated:
One point per voxel at the voxel centre, as a cloud frame carrying the voxel
size and the weights. Sent every recorded frame, and whenever a setting
changes.

frames:
How many frames have been recorded.

cloud out:
The input, unchanged.

RELATED:
pc_voxel is the usual source. mgl_point_cloud draws the result. pc_regions
captures it into a region."""

demo = [
    {'key': 'ctx', 'init': 'mgl_context', 'pos': (30, 62), 'w': 340, 'h': 340,
     'props': {'auto_render': True, 'display_width': 320,
               'display_height': 240, 'node_mouse_events': True}},
    {'key': 'c0', 'comment': True, 'text': 'drag on the picture to orbit',
     'pos': (400, 62)},
    {'key': 'cam', 'init': 'mgl_orbit_camera', 'pos': (30, 430), 'w': 240, 'h': 160},
    {'key': 'pcl', 'init': 'mgl_point_cloud', 'pos': (30, 620), 'w': 240, 'h': 300,
     'props': {'mode': 'points', 'cull': False}},
    {'key': 'c5', 'comment': True,
     'text': "its 'weight scale' changes only\nthe picture",
     'pos': (30, 935)},

    {'key': 'met', 'init': 'metro 33', 'pos': (400, 130), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 33.0, 'units': 'milliseconds'}},
    {'key': 'rnd', 'init': 'np.rand 2000 3', 'pos': (400, 220), 'w': 140, 'h': 130,
     'props': {'min': -0.15, 'max': 0.15, 'dtype': 'float32'}},
    {'key': 'c1', 'comment': True, 'text': 'a small blob of points...',
     'pos': (400, 360)},
    {'key': 's1', 'init': 'signal 5.0 sin', 'pos': (600, 130), 'w': 129, 'h': 78,
     'props': SIG('sin', 5.0, rng=0.7)},
    {'key': 's2', 'init': 'signal 7.0 cos', 'pos': (750, 130), 'w': 129, 'h': 78,
     'props': SIG('cos', 7.0, rng=0.7)},
    {'key': 'pk', 'init': 'pack 3', 'pos': (600, 230), 'w': 130, 'h': 100},
    {'key': 'add', 'init': '+', 'pos': (400, 400), 'w': 140, 'h': 70},
    {'key': 'c2', 'comment': True, 'text': '...wandering about the volume',
     'pos': (400, 480)},
    {'key': 'vox', 'init': 'pc_voxel', 'pos': (400, 530), 'w': 300, 'h': 260,
     'props': {'voxel size (cm)': 10.0, 'distance compensation': 'none',
               'min (x,y,z)': [-1.0, -1.0, -1.0], 'max (x,y,z)': [1.0, 1.0, 1.0]}},
    {'key': 'acc', 'init': 'pc_accumulate', 'pos': (400, 810), 'w': 300, 'h': 290,
     'props': {'record': True, 'normalize': 'max', 'weight scale': 1.0,
               'half-life (s)': 3.0,
               'min (x,y,z)': [-1.0, -1.0, -1.0], 'max (x,y,z)': [1.0, 1.0, 1.0]}},
    {'key': 'c3', 'comment': True,
     'text': "recording from the start, fading with a 3 s\nhalf-life: a trail behind the blob. Set\nhalf-life to 0 and it builds up for good.\n'max' makes the busiest voxel 1; raise\n'weight scale' to bring up where it only\npassed through. Try 'per frame' and 'none'.",
     'pos': (400, 1115)},
    {'key': 'i1', 'init': 'int', 'pos': (730, 810), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c4', 'comment': True, 'text': 'frames recorded', 'pos': (730, 860)},
]
links = [('ctx', 'mgl_chain', 'cam', 'mgl chain in'), ('ctx', 'ui', 'cam', 'ui'),
         ('cam', 'mgl chain out', 'pcl', 'mgl chain in'),
         ('met', '', 'rnd', ''),
         ('s1', '', 'pk', 'in 1'), ('s2', '', 'pk', 'in 2'),
         ('pk', 'out', 'add', 'operand'),
         ('rnd', '', 'add', 'in', 0),
         ('add', 'result', 'vox', 'point cloud'),
         ('vox', 'voxel cloud', 'acc', 'point cloud'),
         ('acc', 'frames', 'i1', ''),
         ('acc', 'accumulated', 'pcl', 'points')]
print(build('pc_accumulate', 'pc_accumulate - where the cloud has been', body,
            demo, links, demo_width=900, text_width=810, text_height=790))
