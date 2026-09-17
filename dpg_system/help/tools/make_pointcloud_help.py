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

GROUPING VOXELS INTO BOXES:
'boxes (x,y,z)' divides the working volume into a coarser lattice - 8,8,8 for
512 boxes - and sends the sum of the voxel weights in each one out of 'box
values' as an (x,y,z) array. Leave any axis at 0 and nothing is subdivided.

Boxes are not a layer on top of the voxels, they are a constraint on them. The
crop divides into EXACTLY the number of boxes you asked for, and the voxel size
then bends to the nearest one that fits a whole number of voxels into each box:

    box size   = crop / boxes
    n          = round(box size / your voxel size)
    voxel size = box size / n

So 'voxel size (cm)' becomes a TARGET once boxes are on. Ask for 7 cm voxels in
a 2 m box divided 4 ways and you get 7.1429 cm - 7 voxels per box, on the nose,
every box holding the same 343 voxels with nothing left over. There is nothing
to tune and nothing to feed back; it is a calculation, not a search.

The one thing you give up is exactly cubic voxels: for an arbitrary crop the
three axes snap by different amounts. The C++ app makes the same trade, and for
the same reason - you cannot have both.

The frame carries the boxes on to whatever comes next as a 'clusters' entry -
per-voxel labels, per-box values, and the lattice geometry. Boxes are only the
first way of grouping voxels; blobs, k-means and hand-painted regions would put
the same thing on the frame, so anything reading it works with all of them.

CLEANING THE BOX VALUES UP:
pc_cluster_filter sits between pc_voxel and whatever reads the boxes, and
conditions the per-box numbers. It changes nothing else on the frame, so you
can put one in, or two, or none.

'threshold' with 'gate' set to 'squeeze' subtracts the threshold from every box
and floors at zero, so a box just over the line starts from zero instead of
jumping; 'gate' zeroes what is under and passes the rest untouched. This is
where you kill the floor of sensor noise.

The filter is adaptive either way - smooth hard while a box is idle, get out
of the way when it moves. 'filter' (option) picks which:

'one euro' is the default and the better of the two. It low-passes each box at
a cutoff that rises with that box's speed: 'min cutoff' sets how still an idle
box looks, 'beta' how little a moving one lags. Tune it the way its authors
say - beta to 0, drop 'min cutoff' until an idle box stops shimmering, then
raise beta until a real arrival stops lagging.

'adaptive' is the C++ cBins filter, kept for fidelity: 'smooth' is the floor on
the smoothing and 'knee' the change at which it is released.

The reason one euro is the default is not subtlety, it is TIME. Its numbers are
in Hz against the measured frame interval, so its behaviour does not move when
the frame rate does. Measured over 10 to 90 fps, the C++ filter's step response
ran from 0.044 s to 0.400 s - a factor of nine - while one euro held 0.400 s
throughout. Tune the C++ one at 30 fps and it is a different filter at 15. It
also took about 10% less lag at every jitter budget tested.

'decay' bleeds a constant off per second so a box does not rest on a residue.

'motion' switches from the level to the CHANGE in the level - the C++
dynamicBins - so a box lights up when something moves through it rather than
when something is in it. Differencing amplifies noise, so it comes after the
filter, not before; on a depth sensor do not turn it on with 'filter' set to
none. Measured on a still but noisy box, the spurious motion reported was 7.7
unfiltered, 0.9 through the adaptive filter and 0.34 through one euro.

SHOWING WHAT IS IN EACH BOX:
mgl_cluster_boxes draws the result: one translucent cube per box, its colour
carrying that box's sum. Feed it the voxel cloud and put it on an mgl chain.

The sum drives brightness, so an empty box vanishes and a busy one glows;
'sensitivity' is the gain on that, and is the control you actually use - the
raw sums depend on your voxel size, your crop and your sense setting, so wind
it until the range of the room reads well and leave it. 1.0 is roughly right
for a room; the widget is scaled so that it is.

THE FRAME IS WHERE THE DYNAMIC RANGE COMES FROM:
Each box is drawn twice - a filled cube and its wireframe outline - off the
same sum but at two gains. 'frame sensitivity' defaults to 4x 'sensitivity',
so the outline saturates while the fill is still a quarter of the way up. The
bottom of the range is carried by the frame lighting up, the top by the fill
blooming behind it, and you read far more of the range than either could show
alone. The C++ app does the same thing with boxGain and boxFrameGain.

Wind 'frame sensitivity' up for a room where almost nothing is happening and
you want the faintest presence to register; wind it down toward 'sensitivity'
when everything is already bright and the outlines are just glare. 'show fill'
and 'show frames' turn off either half.

'color mode' decides the hue. 'uniform' gives every box the same hue and
saturation, so the picture is purely intensity - what you want when you are
reading activity. 'per box' walks the hue by 'hue spread' per box, the way the
C++ app colours its regions, so neighbouring boxes stay distinguishable - what
you want when you are working out WHICH box is which.

'threshold' hides boxes below a level rather than drawing them nearly black.
'fill' shrinks each cube inside its box so the lattice reads as separate cells.
'blend' is additive by default: these cubes are translucent and unsorted, so
alpha blending would make the picture depend on the order they happen to be
drawn in, while additive does not.

SEEING THE VOLUME:
pc_voxel also sits on an mgl chain. Patch 'mgl chain in' from the chain that
draws the cloud and it draws the working volume as a wireframe, in the chain's
current colour, then passes the draw on. With boxes on you get the box lattice;
with them off, the bare outline of the volume. Either way it is the volume it
is really using - the crop carried in on the frame, or its own min/max options
when the cloud arrived raw - so it is the fastest way to see your crop against
the live cloud instead of reading coordinates off pc_info.

Four options (plus a colour each) decide what that wireframe looks like: 'show
lines' draws the cell edges, 'show points' draws a dot at every lattice node -
every corner of every box - at 'point size' pixels, and 'line color' /
'point color' set them independently so the two read apart. Turn both off and
the overlay goes away without unpatching. Points alone, at a decent size, give
you the box corners as a constellation without the cage of lines over the
cloud; that is usually the more readable of the two once the subdivision gets
fine.

pc_info IS FOR SETTING THE OTHERS UP:
It reports count, the bounding box, and the centroid, and passes the cloud
through untouched. Put it after the sensor while you are choosing crop bounds:
the min and max it reports ARE the numbers to type into pc_crop.

SYNTAX:
pc_crop
pc_voxel
pc_info
pc_cluster_filter
mgl_cluster_boxes

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

min points (pc_voxel):
The density floor - voxels holding fewer points than this are dropped.

boxes (x,y,z) (pc_voxel):
Divide the volume into this many boxes per axis. 0 for none. Snaps the voxel
size so a whole number of voxels fits each box.

mgl chain in (pc_voxel):
A draw from an mgl chain. Draws the volume as a wireframe - the box lattice
when subdivided, the outline when not.

reduce (pc_voxel, option):
Cube centre or centroid.

OUTPUTS: 

cropped / voxel cloud:
The reduced cloud, with the metadata carried forward.

counts:
Points per voxel.

box values (pc_voxel):
Sum of the voxel weights in each box, as an (x,y,z) array.

voxel cloud (mgl_cluster_boxes):
The frame from pc_voxel. It reads the boxes off the frame, so nothing else
needs patching.

sensitivity / frame sensitivity (mgl_cluster_boxes):
Gain on the sum for the fill and for the outline. The outline wants the higher
of the two.

hue / saturation / alpha (mgl_cluster_boxes):
The colour it is drawn in.

cloud / threshold / min cutoff / beta / motion (pc_cluster_filter):
The frame, and the conditioning applied to the values on it. 'min cutoff' and
'beta' are the one euro filter's two controls.

mgl chain out (pc_voxel):
The draw, passed on to the rest of the chain.

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
    {'key': 'c1', 'comment': True, 'text': 'crop FIRST: it discards most of the\ndata and tells everything downstream\nwhat volume to build its grid over',
     'pos': (30, 635)},

    {'key': 'info', 'init': 'pc_info', 'pos': (400, 420), 'w': 240, 'h': 180},
    {'key': 'i1', 'init': 'int', 'pos': (400, 615), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c4', 'comment': True, 'text': 'how many points are left', 'pos': (400, 665)},
    {'key': 'l1', 'init': 'list', 'pos': (400, 710), 'w': 280, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'l2', 'init': 'list', 'pos': (400, 765), 'w': 280, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c5', 'comment': True, 'text': 'min and max - these ARE the numbers\nto type into pc_crop above',
     'pos': (400, 815)},

    {'key': 'vox', 'init': 'pc_voxel', 'pos': (30, 745), 'w': 300, 'h': 240},
    {'key': 'info2', 'init': 'pc_info', 'pos': (30, 1005), 'w': 240, 'h': 180},
    {'key': 'i2', 'init': 'int', 'pos': (30, 1200), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c7', 'comment': True, 'text': 'fewer points, and evenly spread -\nno longer denser wherever it is nearer',
     'pos': (30, 1250)},
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
    {'key': 'c1', 'comment': True, 'text': 'get the crop right, THEN press learn\nwith nobody in the volume - changing\nthe volume abandons the run\ndilate 1 catches the jitter crust on\nlearned surfaces - almost always worth it',
     'pos': (30, 950)},

    {'key': 'den', 'init': 'pc_denoise', 'pos': (420, 635), 'w': 300, 'h': 260},
    {'key': 'c6', 'comment': True, 'text': 'min points is a DENSITY: it means\nnothing without the voxel size. At 0.04\nmin points 2 removes 7/8 of a real object',
     'pos': (420, 910)},

    {'key': 'info', 'init': 'pc_info', 'pos': (420, 1020), 'w': 240, 'h': 180},
    {'key': 'i1', 'init': 'int', 'pos': (420, 1215), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c9', 'comment': True, 'text': 'watch the count while you tune - it is\nhow you catch a filter eating the subject',
     'pos': (420, 1265)},
]
links = [('src', 'point_cloud', 'crop', 'point cloud'),
         ('crop', 'cropped', 'bg', 'point cloud'),
         ('bg', 'foreground', 'den', 'point cloud'),
         ('den', 'denoised', 'info', 'point cloud'),
         ('info', 'count', 'i1', '')]
print(build('pc_background', 'pc_background and pc_denoise - removing things', body,
            demo, links, demo_width=780, text_width=810, text_height=790))
