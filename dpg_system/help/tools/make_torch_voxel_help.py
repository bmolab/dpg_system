"""Torch point cloud crop and voxelise."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """Cropping and voxelising a point cloud, in torch.

THE NODES:

t.point_cloud_crop     keep the points inside a box
t.point_cloud_voxels   put them on a grid - as points, or as a solid volume

THESE ARE THE TORCH TWINS OF pc_crop AND pc_voxel:
The pc_ nodes do the same jobs in numpy, and for a live depth camera they are
usually faster - at thirty frames a second with a few hundred thousand points,
moving data in and out of torch costs more than the arithmetic saves.

So reach for these when the cloud is ALREADY a tensor - when it came out of a
model, or is on its way into one - and you would otherwise be paying to convert
it twice. And reach for them when you want the thing the pc_ nodes cannot give
you at all, which is the next section.

'voxels out' IS A SOLID VOLUME, AND IT IS THE REASON THESE EXIST:
Most point-cloud work gives you back fewer points. This gives you back a
three-dimensional ARRAY - a grid of cells covering the box, each holding how
many points fell in it. Not a cloud that happens to be tidy: an actual volume,
shaped depth by height by width.

That is what a three-dimensional convolution wants, and what a learned model
wants, and what you can slice, threshold or filter like an image with an extra
axis. A list of points cannot be any of those things without being turned into
this first.

Measured on a 5000-point scene, cropped to a 2 metre box at 0.2 metre voxels:

    point cloud out    (2996, 3)      the surviving points
    voxels out         (10, 10, 10)   the volume - 2 m / 0.2 m = 10 a side
    voxels cloud out   (300, 3)       the 300 occupied cells, as points

The grid's contents sum to 2996 - every surviving point counted once - and 300
of its 1000 cells are occupied. The three outputs are three views of the same
result, and they agree.

TURN ON ONLY WHAT YOU NEED:
Each output has its own switch, and only 'output voxels cloud' starts on. That
is deliberate: building the dense grid and building the reduced cloud are
separate pieces of work, and there is no reason to pay for a volume you are not
going to look at.

THE BOUNDS ARE NAMED, NOT PAIRED:
Where pc_crop takes a minimum and maximum triple, these take six separate
numbers - left, right, top, bottom, front, back - which is easier to adjust one
at a time while watching the result.

Note that 'top' is the SMALLER number and 'bottom' the larger, following the
screen convention where down is positive rather than the room convention where
up is. If a crop box seems inverted vertically, that is why.

'front' and 'back' are distance from the camera, so the front plane is the near
one - and its default is 0.1 m rather than zero, because a depth sensor returns
noise at very short range that you almost never want.

THEY DO NOT CARRY A CLOUD FRAME:
The pc_ nodes pass a small dictionary between them, so a crop set at the top of
a chain tells every grid node downstream what volume to work in. These send
plain tensors, with nothing attached.

So the bounds have to be set on each node that needs them. If you are mixing the
two families, that is the thing to watch: a pc_ node fed from one of these has
no volume to inherit and falls back to its own widgets.

SYNTAX:
t.point_cloud_crop
t.point_cloud_voxels

EXAMPLE:
t.point_cloud_voxels

INPUTS and PARAMETERS:

point cloud in:
An (N, 3) tensor. Receiving it does the work.

left / right / top / bottom / front / back (m):
The box. Top is the smaller of the vertical pair.

voxel size (m):
The size of a cell. The box divided by this is the shape of the volume, so
halving it makes the grid eight times bigger.

output point cloud / output voxels / output voxels cloud:
Which of the three to produce.

OUTPUTS: 

point cloud out:
The surviving points.

voxels out:
The dense volume, depth by height by width.

voxels cloud out:
One point per occupied cell.

RELATED:
pc_crop and pc_voxel for the numpy versions, which are the better choice
straight off a camera.
femto, which is where the cloud comes from.
t.conv3d and the torch nodes, for what the dense volume is actually for."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 3000 3', 'pos': (30, 120), 'w': 200, 'h': 180},
    {'key': 'c0', 'comment': True, 'text': 'a cloud of 3000 points, 0 to 1 in each',
     'pos': (30, 310)},
    {'key': 'c1', 'comment': True, 'text': 'axis - a cube to carve up',
     'pos': (30, 340)},

    {'key': 'vx', 'init': 't.point_cloud_voxels', 'pos': (30, 390), 'w': 340, 'h': 460,
     'props': {'left (m)': 0.0, 'right (m)': 1.0, 'top (m)': 0.0, 'bottom (m)': 1.0,
               'front (m)': 0.0, 'back (m)': 1.0, 'voxel size (m)': 0.2,
               'output voxels': True, 'output voxels cloud': True}},
    {'key': 'c2', 'comment': True, 'text': 'a 1 m box at 0.2 m voxels is 5 cells a',
     'pos': (30, 865)},
    {'key': 'c3', 'comment': True, 'text': 'side. Halve the voxel size and the grid',
     'pos': (30, 895)},
    {'key': 'c4', 'comment': True, 'text': 'gets EIGHT times bigger',
     'pos': (30, 925)},

    {'key': 'inf', 'init': 'info', 'pos': (420, 390), 'w': 260, 'h': 80},
    {'key': 'c5', 'comment': True, 'text': 'the dense volume: depth by height by',
     'pos': (420, 485)},
    {'key': 'c6', 'comment': True, 'text': 'width. Not points - an actual array,',
     'pos': (420, 515)},
    {'key': 'c7', 'comment': True, 'text': 'which is what a 3D convolution wants',
     'pos': (420, 545)},

    {'key': 'inf2', 'init': 'info', 'pos': (420, 600), 'w': 260, 'h': 80},
    {'key': 'c8', 'comment': True, 'text': 'and the same result as points - one per',
     'pos': (420, 695)},
    {'key': 'c9', 'comment': True, 'text': 'occupied cell. Three views of one thing',
     'pos': (420, 725)},

    {'key': 'c10', 'comment': True, 'text': 'only the cloud output starts on. Each is',
     'pos': (420, 775)},
    {'key': 'c11', 'comment': True, 'text': 'separate work - do not pay for a volume',
     'pos': (420, 805)},
    {'key': 'c12', 'comment': True, 'text': 'you are not going to look at',
     'pos': (420, 835)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'vx', 'point cloud in'),
         ('vx', 'voxels out', 'inf', 'in'),
         ('vx', 'voxels cloud out', 'inf2', 'in')]
print(build('t.point_cloud_voxels', 't point cloud - crop, and a solid volume',
            body, demo, links, demo_width=740, text_width=810, text_height=770))
