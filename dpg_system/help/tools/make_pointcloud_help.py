"""Point cloud processing: the depth-sensor pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

FRAME = """
THE CLOUD FRAME, AND WHY pc_crop GOES FIRST:
A cloud travelling between these nodes is either a plain (N, 3) array of points,
or a small dict carrying the points AND the volume they were cropped to.

pc_crop is what turns the first into the second. Every grid-based node
downstream then uses the volume carried on the frame to build its voxel grid,
and their own min/max widgets are only a fallback for when a raw array arrives
with nothing attached.

So pc_crop belongs at the head of the chain, and not only to throw away points.
It is how the rest of the chain learns what volume it is working in. Set the
box once, at the top, and everything after it agrees.

The metadata is passed along by every node, and the renderers unwrap it, so
either form draws directly - you never have to unpack it by hand.
"""

ENGINE = """
ONE ENGINE UNDERNEATH: A DENSE VOXEL GRID:
Every one of these nodes works the same way. The volume is divided into a grid
of cubes, each point is turned into an integer cube number, and all the actual
work - counting, occupancy, background models, persistence - is arithmetic on
flat arrays of those numbers. No kd-tree, no nearest-neighbour search, no
per-point Python.

That is why they keep up with a live sensor, and it is also why 'voxel size (m)'
appears on nearly all of them: it is the resolution of that grid, and it is the
one setting that changes everything else's meaning.

They are numpy rather than torch on purpose. At thirty frames a second with a
few hundred thousand points, moving data in and out of torch costs more than the
arithmetic saves. torch_voxel_nodes has the torch equivalents if you need them
in a tensor chain.
"""

# ------------------------------------------------------------------- pc_crop
body = """These reduce a cloud to the part you care about, and tell you what is in it.

THE NODES:

pc_crop   keep only the points inside a box
pc_voxel  collapse the cloud onto a grid - fewer, tidier points
pc_info   count, bounds and centroid, without changing anything
""" + FRAME + ENGINE + """
pc_crop IS THE BIGGEST SPEED WIN YOU HAVE:
A depth camera returns everything it can see - walls, floor, ceiling, the far
side of the room. Usually you want a few cubic metres of it. Cropping first
means every node after it is working on a fraction of the data, and the
difference is not subtle: in the measured example on this page, a scene of 5,560
points crops to 1,506.

'invert' keeps the OUTSIDE of the box instead, which is how you remove a known
obstruction rather than isolate a region.

pc_voxel TRADES DETAIL FOR EVENNESS:
It divides the volume into cubes and replaces all the points in each cube with
one point - the cube's centre, or the centroid of what was in it.

The real gain is not just fewer points, it is UNIFORM density. A depth camera
gives you far more points close up than far away, purely because a near surface
subtends more pixels; voxelising removes that bias, so a measurement over the
cloud is not dominated by whatever happens to be nearest.

Measured on the 1,506-point crop above:

voxel size 0.05 m   1,328 points
voxel size 0.1 m      746 points
voxel size 0.2 m      247 points

'reduce' chooses centre or centroid. Centre gives you a tidy lattice; centroid
keeps the points where the data actually was, which looks less mechanical.

'min points' drops cubes holding fewer than that many points, which is a cheap
way to lose speckle in the same pass - but read the warning about it on the
pc_denoise page first, because it is sharper than it looks.

WEIGHTS, AND WHY DISTANCE COMPENSATION EXISTS:
pc_voxel also sends 'counts', and puts a per-voxel weight on the frame so a
renderer can show how many points each voxel stood for.

'distance compensation' corrects those counts for the same near/far bias: a
voxel twice as far away catches about a quarter as many depth pixels, so the
default squares the distance before weighting. It uses radial distance rather
than depth, so it survives levelling and yaw.

pc_info IS FOR SETTING THE OTHERS UP:
It reports count, the bounding box, and the centroid, and passes the cloud
through untouched. Put it after the sensor while you are choosing crop bounds:
the min and max it reports ARE the numbers to type into pc_crop.

SYNTAX:
pc_crop
pc_voxel
pc_info

EXAMPLE:
pc_crop

INPUTS and PARAMETERS:

point cloud:
The cloud. Receiving it does the work.

min (x,y,z) / max (x,y,z):
The box, in metres. On pc_crop these are the real setting; elsewhere they are
only a fallback for a raw cloud with no volume attached.

invert (pc_crop):
Keep the outside instead.

voxel size (m):
The grid resolution. The most consequential number here.

reduce / min points (pc_voxel):
Cube centre or centroid, and the density floor.

OUTPUTS: 

cropped / voxel cloud:
The reduced cloud, with the metadata carried forward.

counts:
Points per voxel.

count / min / max / centroid:
What pc_info found.

cloud out:
pc_info's passthrough - the cloud, unchanged.

RELATED:
pc_background and pc_denoise remove things rather than reduce them.
femto and femto_bolt are the usual sources; mgl_point_cloud draws the result."""

demo = [
    {'key': 'src', 'init': 'femto', 'pos': (30, 62), 'w': 280, 'h': 300},
    {'key': 'c0', 'comment': True, 'text': 'the sensor returns the whole room',
     'pos': (30, 375)},
    {'key': 'crop', 'init': 'pc_crop', 'pos': (30, 420), 'w': 300, 'h': 200},
    {'key': 'c1', 'comment': True, 'text': 'crop FIRST: it discards most of the',
     'pos': (30, 635)},
    {'key': 'c2', 'comment': True, 'text': 'data and tells everything downstream',
     'pos': (30, 665)},
    {'key': 'c3', 'comment': True, 'text': 'what volume to build its grid over',
     'pos': (30, 695)},

    {'key': 'info', 'init': 'pc_info', 'pos': (400, 420), 'w': 240, 'h': 180},
    {'key': 'i1', 'init': 'int', 'pos': (400, 615), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c4', 'comment': True, 'text': 'how many points are left', 'pos': (400, 665)},
    {'key': 'l1', 'init': 'list', 'pos': (400, 710), 'w': 280, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'l2', 'init': 'list', 'pos': (400, 765), 'w': 280, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c5', 'comment': True, 'text': 'min and max - these ARE the numbers',
     'pos': (400, 815)},
    {'key': 'c6', 'comment': True, 'text': 'to type into pc_crop above',
     'pos': (400, 845)},

    {'key': 'vox', 'init': 'pc_voxel', 'pos': (30, 745), 'w': 300, 'h': 240},
    {'key': 'info2', 'init': 'pc_info', 'pos': (30, 1005), 'w': 240, 'h': 180},
    {'key': 'i2', 'init': 'int', 'pos': (30, 1200), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c7', 'comment': True, 'text': 'fewer points, and evenly spread -',
     'pos': (30, 1250)},
    {'key': 'c8', 'comment': True, 'text': 'no longer denser wherever it is nearer',
     'pos': (30, 1280)},
]
links = [('src', 'point_cloud', 'crop', 'point cloud'),
         ('crop', 'cropped', 'info', 'point cloud'),
         ('info', 'count', 'i1', ''), ('info', 'min', 'l1', ''), ('info', 'max', 'l2', ''),
         ('info', 'cloud out', 'vox', 'point cloud'),
         ('vox', 'voxel cloud', 'info2', 'point cloud'),
         ('info2', 'count', 'i2', '')]
print(build('pc_crop', 'pc_crop, pc_voxel, pc_info - reduce and inspect', body,
            demo, links, demo_width=720, text_width=810, text_height=790))

# -------------------------------------------------------------- pc_background
body = """These remove what you do not want: the room itself, and the sensor's noise.

THE NODES:

pc_background   learn the empty room, then subtract it
pc_denoise      drop speckle and flicker
""" + FRAME + """
pc_background: SHOW IT THE EMPTY ROOM:
Press 'learn' with nobody in the volume. It watches for 'frames' frames, marks
every voxel that was occupied in at least 'min hits' of them, and from then on
throws away any point landing in one of those voxels.

What is left is whatever is NEW - which for most installations is the person.
It is far more robust than trying to crop a person out geometrically, because it
does not care what shape the room is.

PRESS learn WITH THE CLOUD ALREADY RUNNING:
This is the one procedural trap. The learned model is tied to the exact voxel
grid it was built on, so if the volume changes - because the first frame arrives
carrying a crop, or because you adjust the crop or the voxel size - the model no
longer maps to anything and the run is thrown away.

Start the sensor, get the crop the way you want it, and only then press learn.
Doing it the other way round, the node now tells you it has abandoned the run;
it used to announce that it had started learning and then stop in silence.

'dilate' IS USUALLY WORTH 1:
Sensor noise makes points on a learned surface jitter into neighbouring voxels,
where the model does not expect them, and they survive as a thin crust of false
foreground. dilate grows the background by that many voxels to catch them.

Measured on a 3,000-point wall with realistic jitter:

dilate 0    229 points of wall survive
dilate 1      1 point survives
dilate 2      0 points survive

and in the same test the person in front was untouched at every setting -
1,500 points in, 1,500 out from dilate 1 upward. The cost of dilating is that
the background grows slightly, so something pressed right up against a learned
surface starts to disappear into it. 1 is the sensible default; go higher only
if the crust persists.

'clear' throws the model away and passes everything through again.

pc_denoise: TWO FILTERS, EITHER OR BOTH:
'min points' is the spatial one - drop any voxel holding fewer than that many
points this frame. Speckle is by definition isolated, so it goes first.

'persistence' is the temporal one - keep a running average of how often each
voxel is occupied, and drop the ones that only flicker on briefly. 0 turns it
off. 'decay' sets how long that memory is. This is the one for a sensor that
sparkles in mid-air, which no amount of spatial filtering will fix, because at
any single instant the sparkle looks like a real point.

THE TRAP: min points AND voxel size ARE ONE CONTROL, NOT TWO:
'min points' means nothing on its own - it is a density, and density depends
entirely on how big the voxel is. The same setting can be harmless or
catastrophic. Measured on a solid 1,500-point object:

                 min points 1    2       4
voxel 0.04 m         1500      194       5
voxel 0.08 m         1500      861     215
voxel 0.15 m         1500     1345    1099

At the default voxel size, asking for just two points per voxel destroys seven
eighths of a perfectly real object. If denoising is eating your subject, this is
why - raise the voxel size or lower min points, and change them together.

SYNTAX:
pc_background
pc_denoise

EXAMPLE:
pc_denoise

INPUTS and PARAMETERS:

point cloud:
The cloud. Receiving it does the work.

learn / frames / min hits:
Start learning, over how many frames, requiring how many hits per voxel.

dilate (voxels):
Grow the learned background to catch surface jitter. 1 is usually right.

clear:
Forget the background.

min points / persistence / decay:
The spatial filter, the temporal one, and its memory.

OUTPUTS: 

foreground:
What was not in the learned background.

denoised:
What survived the filters.

Points outside the volume pass through both nodes untouched - the filtering only
happens inside the box.

RELATED:
pc_crop, which should come before either of these.
pc_info to see how many points you are actually losing."""

demo = [
    {'key': 'src', 'init': 'femto', 'pos': (30, 62), 'w': 280, 'h': 300},
    {'key': 'crop', 'init': 'pc_crop', 'pos': (30, 375), 'w': 300, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'crop first, always', 'pos': (30, 590)},

    {'key': 'bg', 'init': 'pc_background', 'pos': (30, 635), 'w': 320, 'h': 300,
     'props': {'dilate (voxels)': 1}},
    {'key': 'c1', 'comment': True, 'text': 'get the crop right, THEN press learn',
     'pos': (30, 950)},
    {'key': 'c2', 'comment': True, 'text': 'with nobody in the volume - changing',
     'pos': (30, 980)},
    {'key': 'c3', 'comment': True, 'text': 'the volume abandons the run',
     'pos': (30, 1010)},
    {'key': 'c4', 'comment': True, 'text': 'dilate 1 catches the jitter crust on',
     'pos': (30, 1055)},
    {'key': 'c5', 'comment': True, 'text': 'learned surfaces - almost always worth it',
     'pos': (30, 1085)},

    {'key': 'den', 'init': 'pc_denoise', 'pos': (420, 635), 'w': 300, 'h': 260},
    {'key': 'c6', 'comment': True, 'text': 'min points is a DENSITY: it means',
     'pos': (420, 910)},
    {'key': 'c7', 'comment': True, 'text': 'nothing without the voxel size. At 0.04',
     'pos': (420, 940)},
    {'key': 'c8', 'comment': True, 'text': 'min points 2 removes 7/8 of a real object',
     'pos': (420, 970)},

    {'key': 'info', 'init': 'pc_info', 'pos': (420, 1020), 'w': 240, 'h': 180},
    {'key': 'i1', 'init': 'int', 'pos': (420, 1215), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c9', 'comment': True, 'text': 'watch the count while you tune - it is',
     'pos': (420, 1265)},
    {'key': 'c10', 'comment': True, 'text': 'how you catch a filter eating the subject',
     'pos': (420, 1295)},
]
links = [('src', 'point_cloud', 'crop', 'point cloud'),
         ('crop', 'cropped', 'bg', 'point cloud'),
         ('bg', 'foreground', 'den', 'point cloud'),
         ('den', 'denoised', 'info', 'point cloud'),
         ('info', 'count', 'i1', '')]
print(build('pc_background', 'pc_background and pc_denoise - removing things', body,
            demo, links, demo_width=780, text_width=810, text_height=790))
