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

That is why they keep up with a live sensor, and it is also why 'voxel size (cm)'
appears on nearly all of them: it is the resolution of that grid, and it is the
one setting that changes everything else's meaning.

They are numpy rather than torch on purpose. At thirty frames a second with a
few hundred thousand points, moving data in and out of torch costs more than the
arithmetic saves. t.point_cloud_voxels is the torch equivalent if you need one
in a tensor chain.
"""


def cloud_source(n, lo, hi, period=100.0):
    """metro -> np.rand: a synthetic cloud of n random points, so the demo
    runs without a depth sensor attached."""
    return [
        {'key': 'met', 'init': 'metro %d' % int(period), 'pos': (30, 62), 'w': 129, 'h': 70,
         'props': {'on': True, 'period': period, 'units': 'milliseconds'}},
        {'key': 'rnd', 'init': 'np.rand %d 3' % n, 'pos': (30, 150), 'w': 140, 'h': 130,
         'props': {'min': lo, 'max': hi, 'dtype': 'float32'}},
    ]

CLOUD_LINKS = [('met', '', 'rnd', '')]

# ------------------------------------------------------------------- pc_crop
body = """pc_crop keeps only the points inside a box, and tells everything after it which box that was.

A depth camera returns everything it can see - walls, floor, ceiling, the far
side of the room. Usually you want a few cubic metres of it.
""" + FRAME + """
pc_crop IS THE BIGGEST SPEED WIN YOU HAVE:
Cropping first means every node after it is working on a fraction of the
data, and the difference is not subtle: in one measured room, a scene of
5,560 points cropped to 1,506.

THE BOX:
'min (x,y,z)' and 'max (x,y,z)' are its corners, in metres, in the sensor's
own coordinates. If you get them the wrong way round on an axis the node
swaps them for you. A point exactly on a face counts as inside. To find the
numbers, put pc_info after the sensor: the min and max it reports are the
room's extent, and you can shrink in from there.

'invert' keeps the OUTSIDE of the box instead, which is how you remove a known
obstruction rather than isolate a region. An inverted crop is not bounded by
the box, so it does NOT put the box on the frame - the nodes after it fall
back on their own min/max settings.

SYNTAX:
pc_crop

EXAMPLE:
pc_crop

INPUTS and PARAMETERS:

point cloud:
The cloud: an (N, 3) array of points, or a cloud frame. Receiving it does
the work.

min (x,y,z) / max (x,y,z):
The box, in metres. Default -3 -3 0 to 3 3 6.

invert:
Keep the outside instead.

OUTPUTS:

cropped:
The points that survived, as a cloud frame carrying the box (or, inverted,
whatever metadata came in).

RELATED:
pc_info to find the box. pc_voxel, pc_background and pc_denoise usually come
next. femto and femto_bolt are the usual sources; mgl_point_cloud draws the
result."""

demo = cloud_source(5000, -5.0, 5.0) + [
    {'key': 'c0', 'comment': True, 'text': 'stand-in for a sensor: 5000 random\npoints filling -5 to 5 on every axis',
     'pos': (30, 290)},
    {'key': 'crop', 'init': 'pc_crop', 'pos': (30, 340), 'w': 300, 'h': 160},
    {'key': 'c1', 'comment': True, 'text': 'the default box: -3 to 3 across,\n0 to 6 deep',
     'pos': (30, 510)},
    {'key': 'info', 'init': 'pc_info', 'pos': (30, 560), 'w': 240, 'h': 160},
    {'key': 'i1', 'init': 'int', 'pos': (30, 735), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'about 900 left inside; tick invert\nand about 4100 are left outside',
     'pos': (30, 785)},
]
links = CLOUD_LINKS + [('rnd', '', 'crop', 'point cloud', 0),
                       ('crop', 'cropped', 'info', 'point cloud'),
                       ('info', 'count', 'i1', '')]
print(build('pc_crop', 'pc_crop - keep only the points inside a box', body,
            demo, links, demo_width=400, text_width=810, text_height=760))

# ------------------------------------------------------------------ pc_voxel
body = """pc_voxel collapses a cloud onto a grid of cubes: fewer, tidier, evenly spread points.

It divides the volume into cubes ('voxels') and replaces all the points in
each occupied cube with one point - the cube's centre, or the centroid of what
was in it. Along the way it can drop sparse and isolated cubes, smooth their
flicker, group them into boxes, and draw the grid it is using.
""" + ENGINE + """
THE VOLUME:
A cloud that comes from pc_crop carries its box, and pc_voxel builds its grid
over exactly that box. A raw cloud uses the node's own 'min (x,y,z)' and
'max (x,y,z)' options instead (default -3 -3 0 to 3 3 6), and pc_voxel then
puts THAT volume on the frame, so the nodes after it agree with it. Points
outside the volume are dropped.

pc_voxel TRADES DETAIL FOR EVENNESS:
The real gain is not just fewer points, it is UNIFORM density. A depth camera
gives you far more points close up than far away, purely because a near
surface subtends more pixels; voxelising removes that bias, so a measurement
over the cloud is not dominated by whatever happens to be nearest.

Measured on a cropped scene of 1,506 points:

voxel size 5 cm    1,328 points
voxel size 10 cm     746 points
voxel size 20 cm     247 points

'voxel size (cm)' is in centimetres, though the cloud itself is in metres.
Untick the 'cubic voxels' option to give width, height and depth separately
in 'voxel size x,y,z (cm)'.

'reduce' (option) chooses center or centroid. Centre gives you a tidy lattice;
centroid keeps the points where the data actually was, which looks less
mechanical.

'min points' drops cubes holding fewer than that many points, which is a cheap
way to lose speckle in the same pass - but read the warning about it on the
pc_background and pc_denoise page first, because it is sharper than it looks.

WEIGHTS, AND WHY DISTANCE COMPENSATION EXISTS:
pc_voxel sends each voxel's point count from 'counts', and puts a weight per
voxel on the frame (0 to 1) so mgl_point_cloud can show how many points each
voxel stood for.

'distance compensation' corrects those weights for the near/far bias: a
voxel twice as far away catches about a quarter as many depth pixels, so the
default, 'squared', multiplies by the square of the distance from the sensor.
It uses straight-line distance rather than depth, so it survives levelling
and yaw. 'sense' is the overall gain on the weights - in squared mode,
doubling it brightens four times.

STOPPING VOXELS FLICKERING:
'count cutoff (Hz)' filters each voxel's point count before 'min points' is
applied to it. 0 is off, which is the default; 0.2 is a good starting point,
with 'count beta' (option) at 0.2 beside it.

Thresholding a raw count makes a voxel blink whenever the count crosses the
line, and within a point or two of the line the count is mostly sensor noise.
What you want is a filter that ignores a stray reading but does not hesitate
over a real one - and the two are told apart by how FAST the count is moving,
not by how big it is. This is a One Euro filter: it smooths hard at 'count
cutoff' while a voxel is quiet, and opens the cutoff up by 'count beta' times
the rate of change when it is not. Lower the cutoff for less flicker at rest,
raise beta if a real arrival lags.

Measured on one voxel at 30 fps, min points 3:

  a 2-frame stray of 4 then 3 points     lit 2 frames raw, 0 filtered
  a real arrival of 25 points            lit on frame 1 either way
  a body sweeping past for 6 frames      6 frames raw, 7 filtered
  a marginal voxel wobbling 1-5 points   19 flips raw, 8 filtered

The two simpler filters both fail, in opposite directions, and it is worth
knowing why. A plain low-pass sits below the true count the whole time it is
rising, so it dims and deletes the moving content it is lagging behind.
Clamping it so it can never fall below the raw count cures that but amplifies
noise instead - a stray reading is believed at once and then held, and that
same sweeping body trails 13 frames instead of 7.

The numbers are in Hz against the measured frame interval, so none of this
changes meaning when the capture rate does.

This is not pc_denoise's 'persistence', which smooths the on/off flag rather
than the count and has already thrown the useful information away.

THE LONELY VOXEL FILTER - USUALLY WHAT YOU ACTUALLY WANT:
'min neighbours' drops any voxel with fewer than that many of its 26 neighbours
occupied. 0 is off. 1 is usually the setting.

This is the only one of these filters that does not confuse noise with content,
and the reason is that it asks a different question. Every threshold on the
count - flat, smoothed, or hysteretic - judges a voxel by itself, and a real
object edge genuinely does flicker, so all of them shave the edges off real
things along with the speckle. A real surface is a SHEET: even a jittery voxel
at the edge of an object has the object beside it. A speckle voxel has nothing.

Measured on a slab with 60 isolated speckle clusters scattered around it, 'min
neighbours' 1 removed all 40 speckle voxels and kept 98% of the slab. On a
simulated room it removed 5.9% of the false voxels while keeping 100.0% of the
true ones, edges included - where raising 'min points' to 8 for a comparable
cut cost 27% of them.

A one-voxel-wide strip survives at 1 (each voxel has two neighbours along it)
and starts to lose its ends at 2, so 1 is the setting that keeps thin real
structure. It is cheap: the neighbour count runs only over the region that is
occupied, not over the whole grid.

GROUPING VOXELS INTO BOXES:
'boxes (x,y,z)' divides the working volume into a coarser lattice - 8,8,8 for
512 boxes - and sends the sum of the voxel weights in each one out of 'box
values' as an (x,y,z) array. Leave any axis at 0 and nothing is subdivided.

Boxes are not a layer on top of the voxels, they are a constraint on them. The
volume divides into EXACTLY the number of boxes you asked for, and the voxel
size then bends to the nearest one that fits a whole number of voxels into
each box:

    box size   = volume / boxes
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
per-voxel labels, per-box values, and the lattice geometry. pc_cluster_filter
cleans the values up, and mgl_cluster_boxes draws them.

SEEING THE VOLUME:
pc_voxel also sits on an mgl chain. Patch 'mgl chain in' from the chain that
draws the cloud and it draws the working volume as a wireframe, then passes
the draw on. With boxes on you get the box lattice; with them off, the bare
outline of the volume. Either way it is the volume it is really using - the
crop carried in on the frame, or its own min/max options when the cloud
arrived raw - so it is the fastest way to see your crop against the live
cloud instead of reading coordinates off pc_info.

Options decide what that wireframe looks like: 'show lines' draws the cell
edges, 'show points' draws a dot at every lattice corner at 'point size'
pixels, and 'show bounds' draws the outer box on its own so the extent stays
readable under a fine lattice. 'line color', 'point color' and 'bounds color'
set each independently, and they are used as set rather than taking the
chain's colour. Turn all three off and the overlay goes away without
unpatching.

SYNTAX:
pc_voxel

EXAMPLE:
pc_voxel

INPUTS and PARAMETERS:

point cloud:
The cloud. Receiving it does the work.

voxel size (cm):
The grid resolution. The most consequential number here. Default 5.

distance compensation / sense:
How the weights are corrected for distance, and their overall gain.

min points:
The density floor - voxels holding fewer points than this are dropped.

count cutoff (Hz):
One Euro filter on each voxel's count before the threshold. 0 is off.

min neighbours:
Drop voxels with fewer than this many of their 26 neighbours occupied. 0 is
off, 1 kills isolated speckle without touching object edges.

boxes (x,y,z):
Group the voxels into this many boxes per axis. 0 is off.

mgl chain in:
A draw from an mgl chain. Draws the volume as a wireframe.

reduce / min (x,y,z) / max (x,y,z) / cubic voxels / voxel size x,y,z (cm) /
count beta (options):
Centre or centroid, the volume for a raw cloud, separate voxel sides, and how
far a fast change opens the count filter.

OUTPUTS:

voxel cloud:
One point per surviving voxel, as a frame carrying the volume, the voxel size,
the weights and (when boxes are on) the clusters.

counts:
Points per surviving voxel.

box values:
Sum of the voxel weights in each box, as an (x,y,z) array. Sent every frame
when boxes are on, all zeros when nothing is there.

mgl chain out:
The draw, passed on to the rest of the chain.

RELATED:
pc_crop should come before it. pc_background and pc_denoise remove things
rather than reduce them."""

demo = cloud_source(4000, 0.0, 1.0) + [
    {'key': 'c0', 'comment': True, 'text': '4000 random points in a one metre cube',
     'pos': (30, 290)},
    {'key': 'vox', 'init': 'pc_voxel', 'pos': (30, 330), 'w': 300, 'h': 260,
     'props': {'voxel size (cm)': 25.0,
               'min (x,y,z)': [0.0, 0.0, 0.0], 'max (x,y,z)': [1.0, 1.0, 1.0]}},
    {'key': 'c1', 'comment': True, 'text': 'voxel size 25 cm; the volume is set to\nthe cube in the options, since this cloud\ndid not come through pc_crop',
     'pos': (30, 600)},
    {'key': 'info', 'init': 'pc_info', 'pos': (30, 670), 'w': 240, 'h': 160},
    {'key': 'i1', 'init': 'int', 'pos': (30, 845), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': '64 points: one per voxel', 'pos': (30, 895)},
    {'key': 'mn', 'init': 'np.mean', 'pos': (330, 670), 'w': 140, 'h': 70},
    {'key': 'f1', 'init': 'float', 'pos': (330, 755), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'about 62 points went\ninto each voxel',
     'pos': (330, 805)},
]
links = CLOUD_LINKS + [('rnd', '', 'vox', 'point cloud', 0),
                       ('vox', 'voxel cloud', 'info', 'point cloud'),
                       ('info', 'count', 'i1', ''),
                       ('vox', 'counts', 'mn', 'in'), ('mn', '', 'f1', '', 0)]
print(build('pc_voxel', 'pc_voxel - fewer, tidier, evenly spread points', body,
            demo, links, demo_width=520, text_width=810, text_height=790))

# ------------------------------------------------------------------- pc_info
body = """pc_info reports how many points a cloud holds, where it reaches, and where its middle is.

It changes nothing - the cloud passes straight through from 'cloud out' - so
it can sit anywhere in a chain while you look.

IT IS FOR SETTING THE OTHERS UP:
Put it after the sensor while you are choosing crop bounds: the min and max
it reports ARE the numbers to type into pc_crop, and you shrink in from
there. Put it after a filter and watch the count while you tune - it is how
you catch a filter eating the subject.

AN EMPTY CLOUD:
When a cloud arrives with no points in it, pc_info sends 0 from 'count' and
nothing from min, max or centroid, but still passes the empty frame on from
'cloud out' - so a renderer or voxel node after it clears when the room
empties.

SYNTAX:
pc_info

EXAMPLE:
pc_info

INPUTS and PARAMETERS:

point cloud:
The cloud, as an (N, 3) array or a cloud frame. Receiving it triggers the node.

OUTPUTS:

count:
How many points.

min / max:
The smallest and largest x, y and z found - the corners of the box that
just holds the cloud.

centroid:
The average position of all the points.

cloud out:
The cloud, unchanged - frame, metadata and all."""

demo = cloud_source(1000, -1.0, 2.0) + [
    {'key': 'c0', 'comment': True, 'text': '1000 random points between -1 and 2',
     'pos': (30, 290)},
    {'key': 'info', 'init': 'pc_info', 'pos': (30, 330), 'w': 240, 'h': 160},
    {'key': 'i1', 'init': 'int', 'pos': (30, 510), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 'count', 'pos': (180, 510)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 565), 'w': 300, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'min: close to -1 -1 -1', 'pos': (30, 615)},
    {'key': 'l2', 'init': 'list', 'pos': (30, 655), 'w': 300, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'max: close to 2 2 2', 'pos': (30, 705)},
    {'key': 'l3', 'init': 'list', 'pos': (30, 745), 'w': 300, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c4', 'comment': True, 'text': 'centroid: about 0.5 0.5 0.5', 'pos': (30, 795)},
]
links = CLOUD_LINKS + [('rnd', '', 'info', 'point cloud', 0),
                       ('info', 'count', 'i1', ''), ('info', 'min', 'l1', ''),
                       ('info', 'max', 'l2', ''), ('info', 'centroid', 'l3', '')]
print(build('pc_info', 'pc_info - count, bounds and centre of a cloud', body,
            demo, links, demo_width=420, text_width=790, text_height=560))

# --------------------------------------------------------- pc_cluster_filter
body = """pc_cluster_filter cleans up the per-box values that pc_voxel puts on the frame.

When pc_voxel groups its voxels into boxes, each box gets a value every frame -
the sum of the voxel weights in it. Raw, those values shimmer with sensor
noise. This node sits between pc_voxel and whatever reads the boxes, and
conditions those numbers. It changes nothing else on the frame, so you can put
one in, or two, or none. A frame without boxes on it passes through untouched.

THE CHAIN, IN ORDER:

1. THRESHOLD:
'threshold' with the 'gate' option set to 'squeeze' (the default) subtracts
the threshold from every box and floors at zero, so a box just over the line
starts from zero instead of jumping; 'gate' zeroes what is at or under it and
passes the rest untouched; 'none' skips this step. This is where you kill the
floor of sensor noise. 'threshold' starts at 0, which is off.

2. FILTER:
Adaptive either way - smooth hard while a box is idle, get out of the way when
it moves. The 'filter' option picks which:

'one euro' is the default and the better of the two. It low-passes each box at
a cutoff that rises with that box's speed: 'min cutoff' sets how still an idle
box looks, 'beta' how little a moving one lags. Tune it the way its authors
say - beta to 0, drop 'min cutoff' until an idle box stops shimmering, then
raise beta until a real arrival stops lagging. 'd cutoff' (option) is the
filter's own smoothing of the speed estimate; 1 rarely wants changing.

'adaptive' is the C++ cBins filter, kept for fidelity: 'smooth' (option) is
the floor on the smoothing and 'knee' (option) the change at which it is
released.

The reason one euro is the default is not subtlety, it is TIME. Its numbers are
in Hz against the measured frame interval, so its behaviour does not move when
the frame rate does. Measured over 10 to 90 fps, the C++ filter's step response
ran from 0.044 s to 0.400 s - a factor of nine - while one euro held 0.400 s
throughout. Tune the C++ one at 30 fps and it is a different filter at 15. It
also took about 10% less lag at every jitter budget tested.

'decay' (option) bleeds that much off every box per second, so a box does not
rest on a residue. It only acts when a filter is on.

3. MOTION:
'motion' switches from the level to the CHANGE in the level - the C++
dynamicBins - so a box lights up when something moves through it rather than
when something is in it. Differencing amplifies noise, so it comes after the
filter, not before; on a depth sensor do not turn it on with 'filter' set to
none. Measured on a still but noisy box, the spurious motion reported was 7.7
unfiltered, 0.9 through the adaptive filter and 0.34 through one euro.

Changing the number of boxes resets the filter's memory, since box 7 is then
a different piece of the room.

SYNTAX:
pc_cluster_filter

EXAMPLE:
pc_cluster_filter

INPUTS and PARAMETERS:

cloud:
The frame from pc_voxel (or another pc_cluster_filter). Receiving it does the
work.

threshold:
The noise floor to gate or squeeze away. 0 is off.

min cutoff / beta:
The one euro filter's two controls.

motion:
Send the change in each box rather than its level.

gate / filter / d cutoff / smooth / knee / decay (options):
As above.

OUTPUTS:

cloud out:
The same frame with the box values replaced by the conditioned ones -
mgl_cluster_boxes draws them from here.

values:
The conditioned values, as an (x,y,z) array like pc_voxel's 'box values'."""

demo = cloud_source(8000, 0.0, 1.0, period=33.0) + [
    {'key': 'c0', 'comment': True, 'text': '8000 random points in a one metre cube,\n30 times a second',
     'pos': (30, 290)},
    {'key': 'vox', 'init': 'pc_voxel', 'pos': (30, 340), 'w': 300, 'h': 260,
     'props': {'voxel size (cm)': 10.0, 'distance compensation': 'none',
               'boxes (x,y,z)': [2.0, 2.0, 2.0],
               'min (x,y,z)': [0.0, 0.0, 0.0], 'max (x,y,z)': [1.0, 1.0, 1.0]}},
    {'key': 'c1', 'comment': True, 'text': 'divided into 2 x 2 x 2 boxes',
     'pos': (30, 610)},
    {'key': 'cf', 'init': 'pc_cluster_filter', 'pos': (30, 650), 'w': 260, 'h': 180},
    {'key': 'rv2', 'init': 'np.ravel', 'pos': (30, 850), 'w': 140, 'h': 70},
    {'key': 'p2', 'init': 'plot', 'pos': (30, 935), 'w': 308, 'h': 176,
     'props': dict(PLOT(9.0, 11.0), **{'width': 300, 'update style': 'input is multi-channel sample'})},
    {'key': 'c3', 'comment': True, 'text': 'filtered: the same eight boxes, steadied',
     'pos': (30, 1120)},
    {'key': 'rv1', 'init': 'np.ravel', 'pos': (380, 650), 'w': 140, 'h': 70},
    {'key': 'p1', 'init': 'plot', 'pos': (380, 735), 'w': 308, 'h': 176,
     'props': dict(PLOT(9.0, 11.0), **{'width': 300, 'update style': 'input is multi-channel sample'})},
    {'key': 'c2', 'comment': True, 'text': "raw 'box values': eight boxes, each\nabout 10, shimmering with noise",
     'pos': (380, 920)},
]
links = CLOUD_LINKS + [('rnd', '', 'vox', 'point cloud', 0),
                       ('vox', 'voxel cloud', 'cf', 'cloud'),
                       ('cf', 'values', 'rv2', 'input'), ('rv2', 'flattened array', 'p2', 'y'),
                       ('vox', 'box values', 'rv1', 'input'), ('rv1', 'flattened array', 'p1', 'y')]
print(build('pc_cluster_filter', 'pc_cluster_filter - steady the box values', body,
            demo, links, demo_width=720, text_width=810, text_height=790))

# --------------------------------------------------------- mgl_cluster_boxes
body = """mgl_cluster_boxes draws pc_voxel's boxes: one translucent cube per box, lit by what is in it.

Feed it the voxel cloud from a pc_voxel with 'boxes (x,y,z)' set - directly,
or through pc_cluster_filter - and put it on an mgl chain. It reads the boxes
off the frame, so nothing else needs patching. A frame with no boxes on it
draws nothing.

BRIGHTNESS IS THE BOX'S SUM:
An empty box vanishes and a busy one glows. 'sensitivity' is the gain on
that, and is the control you actually use - the raw sums depend on your voxel
size, your crop and pc_voxel's 'sense', so wind it until the range of the room
reads well and leave it. 1.0 is roughly right for a room; the widget is scaled
so that it is.

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
and 'show frames' (options) turn off either half.

COLOUR:
'hue', 'saturation' and 'alpha' set the colour. The 'color mode' option
decides the hue. 'uniform' gives every box the same hue, so the picture is
purely intensity - what you want when you are reading activity. 'per box'
walks the hue by 'hue spread' per box, the way the C++ app colours its
regions, so neighbouring boxes stay distinguishable - what you want when you
are working out WHICH box is which.

THE OTHER OPTIONS:
'threshold' hides boxes below a brightness (0 to 1) rather than drawing them
nearly black. 'fill' shrinks each cube inside its box so the lattice reads as
separate cells. 'blend' is additive by default: these cubes are translucent
and unsorted, so alpha blending would make the picture depend on the order
they happen to be drawn in, while additive does not.

Cubes are drawn from the outside: a camera placed inside a box does not see
that box.

SYNTAX:
mgl_cluster_boxes

EXAMPLE:
mgl_cluster_boxes

INPUTS and PARAMETERS:

mgl chain in:
The draw, from the chain.

voxel cloud:
The frame from pc_voxel or pc_cluster_filter.

sensitivity / frame sensitivity:
Gain on the sum for the fill and for the outline. The outline wants the higher
of the two.

hue / saturation / alpha:
The colour it is drawn in.

show fill / show frames / color mode / hue spread / threshold / fill / blend
(options):
As above.

OUTPUTS:

mgl chain out:
The draw, passed on to the rest of the chain."""

demo = [
    {'key': 'ctx', 'init': 'mgl_context', 'pos': (30, 62), 'w': 340, 'h': 340,
     'props': {'auto_render': True, 'node_display_width': 320,
               'node_display_height': 240, 'node_mouse_events': True}},
    {'key': 'c0', 'comment': True, 'text': 'drag on the picture to orbit',
     'pos': (400, 62)},
    {'key': 'cam', 'init': 'mgl_orbit_camera', 'pos': (30, 430), 'w': 240, 'h': 160},
    {'key': 'met', 'init': 'metro 33', 'pos': (400, 430), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 33.0, 'units': 'milliseconds'}},
    {'key': 'rnd', 'init': 'np.rand 3000 3', 'pos': (400, 520), 'w': 140, 'h': 130,
     'props': {'min': -0.25, 'max': 0.25, 'dtype': 'float32'}},
    {'key': 'c1', 'comment': True, 'text': 'a half-metre blob of points...',
     'pos': (400, 660)},
    {'key': 's1', 'init': 'signal 5.0 sin', 'pos': (600, 430), 'w': 129, 'h': 78,
     'props': SIG('sin', 5.0, rng=0.7)},
    {'key': 's2', 'init': 'signal 7.0 cos', 'pos': (750, 430), 'w': 129, 'h': 78,
     'props': SIG('cos', 7.0, rng=0.7)},
    {'key': 'pk', 'init': 'pack 3', 'pos': (600, 530), 'w': 130, 'h': 100},
    {'key': 'add', 'init': '+', 'pos': (400, 700), 'w': 140, 'h': 70},
    {'key': 'c2', 'comment': True, 'text': '...wandering about the volume',
     'pos': (400, 780)},
    {'key': 'vox', 'init': 'pc_voxel', 'pos': (30, 620), 'w': 300, 'h': 300,
     'props': {'voxel size (cm)': 10.0, 'distance compensation': 'none',
               'boxes (x,y,z)': [4.0, 4.0, 4.0],
               'min (x,y,z)': [-1.0, -1.0, -1.0], 'max (x,y,z)': [1.0, 1.0, 1.0]}},
    {'key': 'c3', 'comment': True, 'text': 'pc_voxel divides -1 to 1 into 4 x 4 x 4\nboxes, and draws their lattice',
     'pos': (30, 935)},
    {'key': 'cb', 'init': 'mgl_cluster_boxes', 'pos': (30, 990), 'w': 260, 'h': 200,
     'props': {'sensitivity': 10.0, 'frame sensitivity': 40.0}},
    {'key': 'c4', 'comment': True, 'text': 'the boxes the blob is in light up;\nsensitivity 10, because one small blob\nsums to far less than a room',
     'pos': (30, 1200)},
]
links = [('ctx', 'mgl_chain', 'cam', 'mgl chain in'), ('ctx', 'ui', 'cam', 'ui'),
         ('cam', 'mgl chain out', 'vox', 'mgl chain in'),
         ('vox', 'mgl chain out', 'cb', 'mgl chain in'),
         ('met', '', 'rnd', ''),
         ('s1', '', 'pk', 'in 1'), ('s2', '', 'pk', 'in 2'),
         ('pk', 'out', 'add', 'operand'),
         ('rnd', '', 'add', 'in', 0),
         ('add', 'result', 'vox', 'point cloud'),
         ('vox', 'voxel cloud', 'cb', 'voxel cloud')]
print(build('mgl_cluster_boxes', 'mgl_cluster_boxes - draw what is in each box', body,
            demo, links, demo_width=900, text_width=810, text_height=790))

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
