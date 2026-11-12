"""pc_regions, pc_region_moves and mgl_regions: hand-authored regions over the voxel volume.

Both pages share one scene: a blob wandering through a two-metre volume,
voxelised by pc_voxel, summed into regions by pc_regions and drawn by
mgl_regions. pc_regions starts with a small map already in it (four regions
round the blob's path), written into the patch the way the node saves its
own, so the values move as soon as the page opens."""
import base64
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from help_common import SIG, PLOT, INT

# --------------------------------------------------------------- the demo map
# The grid the map is drawn on: -1 to 1 on every axis in 10 cm cells, which is
# exactly the grid pc_voxel hands on with the settings below.
LO, SIZE, DIMS = -1.0, 0.1, 20
REGIONS = [  # id, name, colour, test on the cell centre (x, y, z)
    (0, '-x side', (1.0, 0.55, 0.15), lambda x, y, z: x < -0.3),
    (1, '+x side', (0.2, 0.6, 1.0), lambda x, y, z: x > 0.3),
    (2, '-y side', (0.3, 1.0, 0.3), lambda x, y, z: (abs(x) < 0.3) & (y < -0.3)),
    (3, '+y side', (1.0, 0.3, 0.75), lambda x, y, z: (abs(x) < 0.3) & (y > 0.3)),
]


def demo_map():
    """The region map as pc_regions saves it in a patch: an .npz, base64."""
    c = LO + (np.arange(DIMS) + 0.5) * SIZE
    z, y, x = np.meshgrid(c, c, c, indexing='ij')       # lin = x + y*nx + z*nx*ny
    x, y, z = x.ravel(), y.ravel(), z.ravel()
    slab = np.abs(z) < 0.3
    data = {
        'format_version': np.int32(1),
        'grid_lo': np.full(3, LO, dtype=np.float32),
        'grid_voxel_size': np.full(3, SIZE, dtype=np.float32),
        'grid_dims': np.full(3, DIMS, dtype=np.int64),
        'region_ids': np.asarray([r[0] for r in REGIONS], dtype=np.int32),
        'region_names': np.asarray([r[1] for r in REGIONS], dtype=str),
        'region_colors': np.asarray([r[2] for r in REGIONS], dtype=np.float32),
    }
    for rid, _, _, test in REGIONS:
        data[f'cells_{rid}'] = np.flatnonzero(slab & test(x, y, z)).astype(np.int64)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **data)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def embed_map(path):
    doc = json.load(open(path))
    patches = doc['patches'].values() if 'patches' in doc else [doc]
    hits = 0
    for patch in patches:
        for node in patch['nodes'].values():
            if node.get('name') == 'pc_regions':
                node['region_map_npz'] = demo_map()
                hits += 1
    assert hits == 1, path
    json.dump(doc, open(path, 'w'), indent=4)


# ------------------------------------------------------------- the shared scene
BLOB_SUB = {
    'name': 'make_blob',
    'host': 'mb',
    'demo': [
        {'key': 'ct', 'init': 'comment a stand-in for a depth sensor', 'pos': (40, 10),
         'w': 300, 'h': 30},
        {'key': 'met', 'init': 'metro 33', 'pos': (40, 60), 'w': 129, 'h': 70,
         'props': {'on': True, 'period': 33.0, 'units': 'milliseconds'}},
        {'key': 'rnd', 'init': 'np.rand 2000 3', 'pos': (40, 150), 'w': 140, 'h': 130,
         'props': {'min': -0.15, 'max': 0.15, 'dtype': 'float32'}},
        {'key': 'cr', 'init': 'comment 2000 points in a 30 cm cube', 'pos': (40, 290),
         'w': 280, 'h': 30},
        {'key': 's1', 'init': 'signal 5.0 sin', 'pos': (260, 60), 'w': 129, 'h': 78,
         'props': SIG('sin', 5.0, rng=0.7)},
        {'key': 's2', 'init': 'signal 7.0 cos', 'pos': (410, 60), 'w': 129, 'h': 78,
         'props': SIG('cos', 7.0, rng=0.7)},
        {'key': 'pk', 'init': 'pack 3', 'pos': (260, 160), 'w': 130, 'h': 100},
        {'key': 'cp', 'init': 'comment wandering in x and y', 'pos': (410, 170),
         'w': 240, 'h': 30},
        {'key': 'add', 'init': '+', 'pos': (40, 340), 'w': 140, 'h': 70},
        {'key': 'out', 'init': 'out cloud', 'pos': (40, 440), 'w': 150, 'h': 80},
    ],
    'links': [('met', '', 'rnd', ''),
              ('s1', '', 'pk', 'in 1'), ('s2', '', 'pk', 'in 2'),
              ('pk', 'out', 'add', 'operand'),
              ('rnd', '', 'add', 'in', 0),
              ('add', 'result', 'out', 'cloud')],
}


def scene():
    demo = [
        {'key': 'ctx', 'init': 'mgl_context', 'pos': (30, 62), 'w': 340, 'h': 340,
         'props': {'auto_render': True, 'display_width': 320,
                   'display_height': 240, 'node_mouse_events': True}},
        {'key': 'c0', 'comment': True,
         'text': 'drag to orbit; click a region to make\nit current. shift-click the blob: add it.\nctrl-click a region: cut that bit out',
         'pos': (30, 410)},
        {'key': 'cam', 'init': 'mgl_orbit_camera', 'pos': (30, 500), 'w': 240, 'h': 160},
        {'key': 'mr', 'init': 'mgl_regions', 'pos': (30, 690), 'w': 260, 'h': 260,
         'props': {'sensitivity': 0.1, 'frame sensitivity': 0.4}},
        {'key': 'mb', 'init': 'p make_blob', 'pos': (420, 62), 'w': 170, 'h': 70},
        {'key': 'c1', 'comment': True, 'text': 'a blob wandering through the volume',
         'pos': (420, 140)},
        {'key': 'vox', 'init': 'pc_voxel', 'pos': (420, 190), 'w': 300, 'h': 260,
         'props': {'voxel size (cm)': 10.0, 'distance compensation': 'none',
                   'min (x,y,z)': [-1.0, -1.0, -1.0], 'max (x,y,z)': [1.0, 1.0, 1.0]}},
        {'key': 'reg', 'init': 'pc_regions', 'pos': (420, 480), 'w': 300, 'h': 560},
    ]
    links = [('ctx', 'mgl_chain', 'cam', 'mgl chain in'), ('ctx', 'ui', 'cam', 'ui'),
             ('ctx', 'ui', 'mr', 'ui'),
             ('cam', 'mgl chain out', 'mr', 'mgl chain in'),
             ('mb', 'cloud', 'vox', 'point cloud'),
             ('vox', 'voxel cloud', 'reg', 'point cloud'),
             ('reg', 'cloud', 'mr', 'region cloud'),
             ('mr', 'picked region', 'reg', 'region'),
             ('mr', 'click command', 'reg', 'command')]
    return demo, links


# ------------------------------------------------------------------ pc_regions
body = """pc_regions holds a map of named regions in the volume, and sums the cloud into each of them, every frame.

A region is any set of voxels you choose to name: 'the chair', 'stage left',
'where she stood'. pc_voxel's boxes divide the room into a regular lattice;
regions are whatever shapes the room actually has. Put pc_regions after
pc_voxel.

WHAT COMES OUT:
'region values' is one number per region, indexed by region number: the sum
of the weights of the points inside it (their count, for a cloud without
weights). From pc_voxel that is how much of the cloud is in each region. A
region number is a slot and is never renumbered - emptying region 2 leaves
region 3 as region 3, so whatever you mapped to it keeps working.

'cloud' passes the frame on with the regions attached, in the same form
pc_voxel's boxes take, so pc_cluster_filter steadies region values exactly
as it does box values, and mgl_regions draws them. 'region cells' sends the
cells of the region you are working on (or all of them - 'show', option) as
a plain cloud whenever the map changes. 'info' describes the current region.

THE MAP'S VOLUME:
The map takes its grid from the first frame - the volume and the voxel size
carried in from pc_voxel or pc_crop - and keeps it from then on, for as long
as any region holds a cell. Points are looked up by position, so changing
pc_voxel's voxel size or the crop later does not wipe the map. The options
show the volume actually in use. 'clear all' lets it follow the cloud again.

MAKING REGIONS:
The node reads top to bottom in the order the work goes, and every button
acts on the CURRENT region at once. 'undo' takes back the last change.

'region' and 'name' choose the region you are working on and name it. An
unused number is a new region. 'pick region' makes current whichever region
holds most of the cloud: stand in it and press. 'clear region' empties the
current one; 'clear all' removes every region.

CAPTURE INTO REGION takes the cells under a cloud - the one on 'capture cloud
in', or the live cloud when nothing is connected there: 'where I am standing
now'. 'add' puts them in the region, 'remove' takes them out, 'replace' makes
them the whole of it. When a cell already belongs to another region, 'assign
mode' (option) decides: 'steal' it, or 'yield' to the other region.

Wire pc_accumulate into 'capture cloud in' to capture everywhere someone has
been over a stretch of time, rather than one moment.

EDIT REGION reshapes the region itself, by 'steps' cells:
  grow / shrink  out or in by 'steps' cells.
  fill           close gaps up to 'steps' wide, then fill what that encloses.
                 Mends a speckled capture.
  hull           the convex hull. A captured body is only the shell the
                 camera saw; hull makes it solid.
  box            the bounding box.
  snap           the pc_voxel boxes at least 'snap coverage' (option) full of
                 the region, whole.
  cut            remove the pieces of the region (its separate blobs - the
                 count is shown under the name) that the capture cloud
                 touches: stand in the stray bit and press.

VOXELS OR BOXES:
'work in' sets the grain all of that works in. 'voxels' is the map's own
cells. 'boxes' is pc_voxel's boxes (set 'boxes (x,y,z)' there): a capture
takes every box the cloud touches, whole; an edit treats the region as the
boxes it touches and gives it back as those boxes; 'steps' counts boxes.
'min points' (option) is how many points must land in a cell or box for it
to count as cloud - in boxes it stops a box that a stray voxel grazes from
being taken.

BUILDING ONE PIECE AT A TIME:
'absorb' (option, from region 'absorb from', or the message 'absorb n')
moves every cell of another region into the current one. To sculpt a piece
before it joins a region that already has cells - where a hull would bridge
the two - capture it into a spare number, shape it there, and absorb it.

CLICKING IN THE PICTURE:
mgl_regions draws the map and lets you author it with the mouse. Wire
mgl_context's 'ui' to it, its 'picked region' to 'region' and its 'click
command' to 'command', as here. A click makes the clicked region current; a
shift-click adds the clicked piece of the cloud to the current region; a
ctrl-click cuts the clicked piece out of whichever region it hit; an
alt-click makes the clicked piece a new region.

MESSAGES:
Every button is also a message of the same name, accepted on any input -
'command' is there to give a pedal, OSC or a button_set somewhere to plug
in. Some actions are messages only:

  left / right / down / up / back / forward
                 move the region by 'steps' cells along -x / +x, -y / +y,
                 -z / +z. pc_region_moves is a ready-made set of these.
  move x y z     move by whole cells.
  extend x|y|z [+|-]
                 stretch the region to the edge of the volume along an axis,
                 both ways or one: 'extend y' makes a footprint a pillar.
  add / replace / new / cut / pick  ox oy oz dx dy dz
                 the action aimed along a ray (an origin and a direction, in
                 metres): it takes only the piece the ray first meets. This
                 is what mgl_regions' clicks send.
  absorb n       as above.

KEEPING THE MAP:
The map is saved in the patch. It can also live in an .npz file of its own:
'save' and 'load' (options). With 'path' set, saving the patch rewrites that
file too, and a patch with no map in it loads the file.

SYNTAX:
pc_regions

EXAMPLE:
pc_regions

INPUTS and PARAMETERS:

point cloud:
The live cloud, usually from pc_voxel. Each one is summed into the regions.

capture cloud in:
The cloud that capture, 'pick region' and 'cut' read. With nothing
connected they read the live cloud.

region / name:
The current region's number and name.

work in:
voxels or boxes.

steps:
How far grow, shrink, fill and the moves go. Default 1.

command:
Messages, as above.

assign mode / snap coverage / min points / absorb from / absorb / show /
surface only / save / load / path / voxel size (cm) / min (x,y,z) /
max (x,y,z) (options):
As above. 'show' and 'surface only' choose what 'region cells' sends; the
volume options are only used before the map has a grid.

OUTPUTS:

cloud:
The input frame with the regions attached, for mgl_regions or
pc_cluster_filter.

region values:
The sum in each region, indexed by region number.

region cells:
The current region's cells (or all regions') as a cloud of cell centres.

info:
The current region's name, size, and how many regions there are.

RELATED:
pc_voxel comes before it. mgl_regions draws the map and takes clicks.
pc_accumulate makes a capture cloud of where someone has been.
pc_region_moves moves regions about. pc_cluster_filter steadies the values."""

demo, links = scene()
demo += [
    {'key': 'c2', 'comment': True,
     'text': 'four regions already in the map,\nround the edges of the blob\'s path',
     'pos': (420, 1050)},
    {'key': 'pl', 'init': 'plot', 'pos': (760, 480), 'w': 308, 'h': 176,
     'props': dict(PLOT(0.0, 25.0), **{'width': 300,
                                       'update style': 'input is multi-channel sample'})},
    {'key': 'c3', 'comment': True,
     'text': "'region values': one line per region,\nrising as the blob passes through it",
     'pos': (760, 670)},
]
links += [('reg', 'region values', 'pl', 'y')]
path = build('pc_regions', 'pc_regions - named regions of the volume', body, demo, links,
             demo_width=1100, text_width=810, text_height=1000, subpatch=BLOB_SUB,
             out_dir=os.environ.get('HELP_OUT', HELP_DIR))
embed_map(path)
print(path)

# ------------------------------------------------------------- pc_region_moves
body = """pc_region_moves is a set of buttons that move and stretch pc_regions' current region.

Wire its outlet to pc_regions' 'command'. Each button sends the message on
its label:

  left / right   move the region along x, by pc_regions' 'steps' cells
  down / up      along y
  back / forward along z
  extend x       stretch the region to both edges of the volume along x
  extend y       the same along y: a footprint becomes a floor-to-ceiling
                 pillar
  extend z       the same along z
  extend y -     stretch it along y one way only
  extend y +     the other way
  undo           take back the last change

The directions are the axes of pc_regions' grid. With a levelled camera y is
up and z is away from the camera.

It is an ordinary button_set otherwise: relabel the buttons, change the
messages they send or colour them in its options, and the patch keeps what
you set. 'extend x -' or 'move 0 2 0' are as good as any of these.

SYNTAX:
pc_region_moves

EXAMPLE:
pc_region_moves

INPUTS and PARAMETERS:

One inlet per button: anything arriving presses it.

message / width / uniform_width / height / flash_duration / label 1 ...
(options):
As for button_set.

OUTPUTS:

out:
The pressed button's message.

RELATED:
pc_regions, which these messages drive. button_set is the general node."""

demo, links = scene()
demo += [
    {'key': 'mv', 'init': 'pc_region_moves', 'pos': (760, 480), 'w': 140, 'h': 380},
    {'key': 'c2', 'comment': True,
     'text': 'click a region in the picture,\nthen move it about',
     'pos': (760, 430)},
]
links += [('mv', 'out', 'reg', 'command')]
path = build('pc_region_moves', 'pc_region_moves - push a region about', body, demo, links,
             demo_width=960, text_width=810, text_height=860, subpatch=BLOB_SUB,
             out_dir=os.environ.get('HELP_OUT', HELP_DIR))
embed_map(path)
print(path)

# ----------------------------------------------------------------- mgl_regions
body = """mgl_regions draws pc_regions' map: each region as a translucent skin and an outline, in its own colour, lit by how much is in it.

Put it on an mgl chain and wire pc_regions' 'cloud' to 'region cloud'. It is
mgl_cluster_boxes' picture for regions of any shape, and it is also where
regions are authored with the mouse.

BRIGHTNESS:
Each region is lit by its value - what pc_regions summed into it this frame.
'sensitivity' is the gain on that for the skin, 'frame sensitivity' for the
outline. As with mgl_cluster_boxes, the outline wants the higher gain: it
lights up at the faintest presence while the skin carries the top of the
range. The values depend on your voxel size and pc_voxel's 'sense', so wind
the two until a person in a region reads well.

SEEING THE MAP WHILE YOU MAKE IT:
Most regions are empty most of the time, and an empty region drawn at its
value would be invisible. Two floors keep them visible: 'frame floor'
(option) is the least an outline is ever drawn at, 'fill floor' (option) the
same for the skin.

The region pc_regions is working on is drawn at 'current boost' (option)
times those floors, and with 'current outline' (option) its outline pulses
at 'outline pulse (Hz)' (option; 0 is steady), whatever its value. So one
picture shows what exists, what is active, and which one you are on. 'solo'
hides every region but the current one.

CLICKING:
Wire mgl_context's 'ui' to 'ui', 'picked region' to pc_regions' 'region' and
'click command' to pc_regions' 'command', as here.

A click (not a drag) makes current the first region under the cursor;
clicking it again takes the next one along the line of sight, which is how a
region hidden behind another is reached. 'picked region' sends its number and
'picked point' where the line entered it, in metres.

A click with a modifier picks nothing. It sends a command aimed along the
line of sight out of 'click command', for pc_regions to act on:

  shift-click   'add': the piece of the live cloud you clicked goes into the
                current region
  ctrl-click    'cut': the piece of whichever region you clicked is removed,
                and that region becomes current
  alt-click     'new': the piece you clicked becomes a region of its own

'shift-click', 'ctrl-click' and 'alt-click' (options) change what each sends:
add, replace, new, cut or none. Ctrl wins over alt, alt over shift. On a Mac,
command counts as ctrl and option as alt. A 'piece' is a connected bit, in
pc_regions' 'work in' grain - voxels, or boxes taken whole.

BOXES:
When pc_regions' 'work in' is boxes, the boxes the cloud occupies - exactly
what a capture or a click would take - are outlined at 'box level' (option),
whether or not a region holds them, so a box map can be made from this view
alone. 'show boxes' (option) turns that off.

THE OTHER OPTIONS:
'show fill' and 'show frames' turn off either half of the picture. 'color
mode' is 'per region' (each region's own colour, from pc_regions) or
'uniform', every region in 'hue' - for reading activity rather than telling
regions apart. 'blend' is additive by default, so the picture does not depend
on the order the translucent skins happen to be drawn in.

A skin is only the region's outer faces, with the far side culled, so a
region's brightness is its value, not how deep it is along the view.

SYNTAX:
mgl_regions

EXAMPLE:
mgl_regions

INPUTS and PARAMETERS:

mgl chain in:
The draw, from the chain.

region cloud:
pc_regions' 'cloud'.

sensitivity / frame sensitivity:
Gain on the value for the skin and the outline.

alpha:
Overall opacity.

solo:
Draw only the current region.

ui:
mgl_context's 'ui' out, for clicks.

show fill / show frames / frame floor / fill floor / current boost / current
outline / outline pulse (Hz) / color mode / hue / blend / show boxes / box
level / shift-click / ctrl-click / alt-click (options):
As above.

OUTPUTS:

mgl chain out:
The draw, passed on.

picked region:
The number of the region a plain click picked.

picked point:
Where the line of sight entered it, x y z in metres.

click command:
A modifier-click as a command and a ray, for pc_regions' 'command'.

RELATED:
pc_regions makes the map. mgl_cluster_boxes draws pc_voxel's boxes the same
way."""

demo, links = scene()
demo += [
    {'key': 'i1', 'init': 'int', 'pos': (245, 700), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'the region last clicked', 'pos': (245, 750)},
]
links += [('mr', 'picked region', 'i1', '')]
path = build('mgl_regions', 'mgl_regions - draw the regions, click to make them', body,
             demo, links, demo_width=960, text_width=810, text_height=1000, subpatch=BLOB_SUB,
             out_dir=os.environ.get('HELP_OUT', HELP_DIR))
embed_map(path)
print(path)
