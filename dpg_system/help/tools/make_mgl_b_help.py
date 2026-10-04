"""mgl primitives, geometry from data, and the body nodes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

def chain(x=30, y=62, auto=True):
    return [
        {'key': 'ctx', 'init': 'mgl_context', 'pos': (x, y), 'w': 220, 'h': 120,
         'props': {'auto_render': auto}},
        {'key': 'cam', 'init': 'mgl_orbit_camera', 'pos': (x, y + 138), 'w': 240, 'h': 220},
        {'key': 'lgt', 'init': 'mgl_light', 'pos': (x, y + 373), 'w': 220, 'h': 200},
        {'key': 'dsp', 'init': 'mgl_display', 'pos': (x + 300, y + 138), 'w': 220, 'h': 160},
    ]

CHAIN_LINKS = [('ctx', 'mgl_chain', 'cam', 'mgl chain in'),
               ('ctx', 'ui', 'cam', 'ui'),
               ('cam', 'mgl chain out', 'lgt', 'mgl chain in'),
               ('ctx', 'texture_tag', 'dsp', 'texture_tag')]

# ----------------------------------------------------------------- primitives
body = """The built-in shapes: things you can draw without supplying any geometry.

THE NODES:

mgl_box           a cube or a rectangular block
mgl_sphere        a sphere, divided into latitude and longitude
mgl_geo_sphere    a geodesic sphere - triangles of nearly equal size, with no 
                  crowding at the poles
mgl_cylinder      a cylinder
mgl_plane         a flat rectangle, optionally subdivided
mgl_disk          a filled circle, or a ring if you give it a hole
mgl_partial_disk  a wedge - a disk with a start angle and a sweep
mgl_line          a line along a vector

WHY TWO SPHERES:
The ordinary sphere is built like a globe, so its triangles bunch together at 
the poles and stretch at the equator. That is fine for a plain surface and 
visible the moment you texture it, or use it to show a distribution over 
directions. The geodesic one has near-uniform triangles everywhere, which is 
what you want when the sphere is standing for a set of directions rather than 
being an object.

mgl_partial_disk IS THE ONE TO REMEMBER:
A wedge with a start angle and a sweep is how you draw a value as an arc - a 
dial, a proportion, a range of orientations. Patch a stream into 'sweep angle' 
and it becomes a readout that lives in the scene rather than in the patch.

COMMON PARAMETERS:
Every shape has the same first few inlets: 'mode' for whether it is drawn as 
filled triangles, lines or points; 'cull' for whether back faces are skipped; 
'point_size' for point mode; and 'texture' for an image to map onto it.

'mode' set to lines is the quickest way to see what a shape is actually made 
of, and to see whether a subdivision setting is doing what you expect.

SYNTAX:
mgl_sphere
mgl_partial_disk

EXAMPLE:
mgl_cylinder

INPUTS and PARAMETERS:

mgl chain in:
The chain. This triggers the node.

radius / height / size / width / depth:
The dimensions, depending on the shape.

slices / segments / rings / subdivisions:
How finely the shape is divided. More is smoother and slower.

hole_ratio (mgl_disk):
Turns a disk into a ring.

start angle / sweep angle (mgl_partial_disk):
Where the wedge begins and how far it goes round.

vector (mgl_line):
The direction and length of the line.

OUTPUTS: 

mgl chain out:
The chain, continuing - shapes do not change state for what follows.

WHERE A SHAPE APPEARS:
At the origin, until a transform moves it. Two shapes with no transform between 
them are in the same place, one inside the other. That is the usual reason a 
scene seems to be missing something."""

demo = chain() + [
    {'key': 'sph', 'init': 'mgl_geo_sphere', 'pos': (30, 660), 'w': 240, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'set mode to lines to see the triangles',
     'pos': (30, 875)},
    {'key': 'tr', 'init': 'mgl_translate 2 0 0', 'pos': (30, 915), 'w': 220, 'h': 120},
    {'key': 'pd', 'init': 'mgl_partial_disk', 'pos': (30, 1050), 'w': 240, 'h': 240},
    {'key': 'c1', 'comment': True, 'text': 'drag sweep angle: a value as an arc',
     'pos': (30, 1305)},
]
links = CHAIN_LINKS + [
    ('lgt', 'mgl chain out', 'sph', 'mgl chain in'),
    ('sph', 'mgl chain out', 'tr', 'mgl chain in'),
    ('tr', 'mgl chain out', 'pd', 'mgl chain in')]
print(build('mgl_sphere', 'mgl shapes - the built-in geometry', body, demo, links,
            demo_width=560, text_width=810, text_height=760))

# ------------------------------------------------------------ geometry from data
body = """These draw geometry you supply, rather than a shape they already know.

THE NODES:

mgl_mesh         a mesh handed to it down a cord: corners and triangles
mgl_model        a mesh loaded from a file
mgl_point_cloud  a set of points
mgl_surface      a grid of points drawn as a continuous surface
mgl_line_array   many lines at once, built for trails of motion

All but mgl_line_array are mgl shapes: they are lit, take the current colour 
and material, and share the shape inputs 'mode' (solid, wireframe or 
points), 'cull', 'point_size', 'round' and 'texture'. mgl_line_array draws 
in the current colour without lighting.

mgl_mesh TAKES A MESH FROM THE PATCH:
On 'mesh' it takes a dictionary with 'vertices' (n by 3), 'faces' (m by 3 
corner indices) and optionally 'normals' - what shape_modes sends - or a 
list of [vertices, faces], or a bare n by 3 array of corners with no faces. 
Normals are worked out from the faces when none come with it. Faces that 
point past the last corner are thrown away, leaving just the corners. 
Corners without faces are drawn as points whatever 'mode' says, 
'point_size' pixels across. They all face the viewer along z, so with 
'cull' ticked they vanish when seen from behind. The 'holding' 
line says what it last received - 'nothing yet', so many points, or so many 
corners and triangles - which tells a mesh that never arrived from one drawn 
somewhere you are not looking.

mgl_model LOADS A FILE:
Name the file in 'file_path' (anything trimesh reads: obj, stl, ply, glb...). 
A file holding several meshes is merged into one. 'uv_mode' can replace the 
file's own texture coordinates with a projection - sphere, cylinder, 
plane_xy, plane_xz ('box' currently falls back to sphere) - applied the 
next time the file loads; click 'generate_uv' to reload it now. A file 
that will not load is reported once in the console, and tried again when 
'file_path' changes or 'generate_uv' is clicked.

mgl_point_cloud FOR POINTS:
'points' takes an n by 3 array (or a flat list of x y z triples), or a cloud 
frame from the point cloud nodes - a dictionary whose 'point_cloud' entry 
holds the points and which may carry per-point 'weights' (0 to 1) and a 
'voxel_size'. 'weights' chooses whether the weights set each point's size, 
brightness, both, or neither; 'blend' additive (the default) lets dense 
regions glow rather than the nearest point hiding the rest; 'draw at voxel size' draws 
each point at the frame's voxel size in the scene rather than at 
'point_size' pixels. Points that are not finite numbers are dropped. 
A new node starts in points mode with culling off; a saved one keeps the 
'mode' and 'cull' it was saved with.

mgl_surface FOR A GRID:
'points' takes a rows by columns by 3 array of positions, and joins 
neighbours into a smooth surface, lit on both sides. It is a grid of 3D 
points, not a list of heights: to draw a height field, build the x and z 
of each grid point and put the height in y.

mgl_line_array IS THE ONE BUILT FOR MOTION:
'array' takes points by lines by 3 - each line is one column, newest point 
first - or a points by 3 array for a single line. It draws the whole array 
at once, and its options are about making motion legible: 'alpha_fade' and 
'fade_rate' let older parts of a trail die away; 'accent_motion' measures 
how far each point moved since the last array and lets that speed brighten 
the lines ('accent_brightness'), colour them along a dark-to-yellow scale 
('accent_colour'), and thicken them ('accent_width'), each with its own 
scale. A trail drawn at constant width and brightness tells you where 
something went; one that brightens where it moved fast tells you how.

SYNTAX:
mgl_mesh
mgl_model
mgl_point_cloud
mgl_surface
mgl_line_array

None of them takes arguments.

EXAMPLE:
mgl_point_cloud

INPUTS and PARAMETERS:

mgl chain in:
The chain. This triggers the drawing.

mesh / file_path / points / array:
The geometry. Data can arrive as often as you like and on any thread: the 
nodes store it, and build what they draw on the next render.

scale / center (mgl_mesh, mgl_model):
Scale multiplies the size. 'center' moves the middle of the mesh to the 
origin - the average corner for mgl_mesh, the centre of mass for mgl_model.

fit (mgl_mesh):
Scales the mesh so its farthest corner is one unit out, which saves guessing 
at the units it was made in.

uv_mode / uv_scale / generate_uv (mgl_model):
Texture coordinates, as above.

weights / blend / draw at voxel size (mgl_point_cloud):
As above.

line_width:
The width in pixels. 'perspective_width' makes it shrink with distance.

alpha_fade / fade_rate (mgl_line_array):
Fade along each line from its newest point; 'fade_rate' above 1 shortens 
the bright part, below 1 holds it longer.

accent_motion / accent_brightness / accent_scale / accent_colour / 
accent_colour_scale / white_hot / accent_width / accent_width_scale / 
accent_width_curve / accent_width_max (mgl_line_array):
Whether speed shows, and how: each accent has a switch and a scale, 
'white_hot' carries the top of the colour scale on into white, and the 
width accent has a curve and a ceiling.

selected_lines (mgl_line_array):
Line numbers to draw, separated by spaces. Empty draws them all.

use_line_colors / color index / color_control (mgl_line_array):
With 'use_line_colors' ticked each line has its own colour: set 'color 
index' to a line number and pick its colour; index -1 recolours every line.

OUTPUTS:

mgl chain out:
The chain, continuing.

RELATED:
mgl_sphere and the other built-in shapes. shape_modes and the point cloud 
nodes produce data for these. mgl_text draws text in the scene and used to 
share this page. gl_line_array is the older equivalent of mgl_line_array."""

demo = [
    {'key': 'ctx', 'init': 'mgl_context', 'pos': (30, 62), 'w': 340, 'h': 340,
     'props': {'auto_render': True, 'node_display_width': 320,
               'node_display_height': 240, 'node_mouse_events': True}},
    {'key': 'c0', 'comment': True, 'text': 'drag on the picture to orbit',
     'pos': (30, 120)},
    {'key': 'cam', 'init': 'mgl_orbit_camera', 'pos': (30, 440), 'w': 240, 'h': 160},
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 630), 'w': 88, 'h': 42},
    {'key': 'nr1', 'init': 'np.rand 40 3 3', 'pos': (30, 700), 'w': 140, 'h': 160,
     'props': {'min': -1.0, 'max': 1.0}},
    {'key': 'c1', 'comment': True, 'text': '40 points on each of 3 lines',
     'pos': (30, 700)},
    {'key': 'la', 'init': 'mgl_line_array', 'pos': (30, 890), 'w': 220, 'h': 400,
     'props': {'alpha_fade': True}},
    {'key': 'c2', 'comment': True,
     'text': 'alpha_fade: each line fades from\nits first point to its last',
     'pos': (30, 890)},
    {'key': 'nr2', 'init': 'np.rand 500 3', 'pos': (30, 1320), 'w': 140, 'h': 140,
     'props': {'min': -0.6, 'max': 0.6}},
    {'key': 'c3', 'comment': True, 'text': '500 points in a cube', 'pos': (30, 1320)},
    # mode/cull saved explicitly: custom_create only switches a NEW node to
    # points, and a node loaded from file would otherwise come up solid
    {'key': 'pc', 'init': 'mgl_point_cloud', 'pos': (30, 1490), 'w': 240, 'h': 280,
     'props': {'mode': 'points', 'cull': False}},
]
links = [('ctx', 'mgl_chain', 'cam', 'mgl chain in'),
         ('ctx', 'ui', 'cam', 'ui'),
         ('cam', 'mgl chain out', 'la', 'mgl chain in'),
         ('lb', 'out', 'nr1', ''),
         ('nr1', 'random array', 'la', 'array'),
         ('la', 'mgl chain out', 'pc', 'mgl chain in'),
         ('lb', 'out', 'nr2', ''),
         ('nr2', 'random array', 'pc', 'points')]
print(build('mgl_mesh', 'mgl geometry - drawing data you supply', body, demo, links,
            demo_width=560, text_width=810, text_height=1100))

# -------------------------------------------------------------------- mgl_text
body = """mgl_text draws a line of text into the 3D scene.

WHERE THE TEXT GOES:
The text starts at the origin of the chain's current coordinates and runs 
along plus x, sitting on that line, flat in the x-y plane. Put a transform 
before it to place, turn or size it, like any other shape. One unit of 
'scale' makes letters about half a unit tall at the default font size.

It draws in the current colour (set with mgl_color), without lighting, and 
with the depth test off - so it is never shaded and is drawn over everything 
before it in the chain, even shapes standing in front of it. Shapes drawn 
after it can still cover it.

mgl_text NEEDS A BILLBOARD TO STAY READABLE:
Flat text turns edge-on and disappears as the camera moves. Put 
mgl_billboard before it and it keeps facing the viewer whatever the orbit 
is doing.

WHAT CAN BE SENT:
A string is drawn as it is. Anything else is turned into text first, so a 
number shows as that number - but a list shows with its brackets, so join a 
list into a string before sending it. Only the plain Latin characters (space 
to tilde) are drawn; anything else is skipped. There is one line - a return 
does not start a new one.

SYNTAX:
mgl_text
mgl_text <font size>
mgl_text <font file>

A number argument is the font size (48 by default), a word is the font file. 
Either order.

EXAMPLE:
mgl_text 64

INPUTS and PARAMETERS:

mgl chain in:
The chain. This triggers the drawing.

text:
What to draw. It stays until something else is sent.

scale:
The size of the letters in the scene. 1 by default.

font (option):
The font file, Inconsolata-g.otf by default, found from the folder the 
program runs in. Changing it reloads the letters.

size (option):
The size the font is rendered at. A larger size gives sharper letters and 
also draws them larger, since the drawn size is this times 'scale'.

OUTPUTS:

mgl chain out:
The chain, continuing. The text does not move anything downstream.

RELATED:
mgl_billboard to keep it facing the camera; mgl_color for its colour; 
mgl_translate to place it. mgl_mesh covers the geometry nodes and used to 
share this page. gl_text is the older equivalent."""

demo = [
    {'key': 'ctx', 'init': 'mgl_context', 'pos': (30, 62), 'w': 340, 'h': 340,
     'props': {'auto_render': True, 'node_display_width': 320,
               'node_display_height': 240, 'node_mouse_events': True}},
    {'key': 'c0', 'comment': True, 'text': 'drag on the picture to orbit',
     'pos': (30, 120)},
    {'key': 'cam', 'init': 'mgl_orbit_camera', 'pos': (30, 440), 'w': 240, 'h': 160},
    {'key': 'bb', 'init': 'mgl_billboard', 'pos': (30, 630), 'w': 120, 'h': 60},
    {'key': 'c1', 'comment': True, 'text': 'keeps what follows facing the camera',
     'pos': (30, 630)},
    {'key': 'tr', 'init': 'mgl_translate -0.6 0 0', 'pos': (30, 720), 'w': 160, 'h': 110,
     'props': {'x': -0.6, 'y': 0.0, 'z': 0.0}},
    {'key': 'c2', 'comment': True, 'text': 'the text starts at the origin:\nmove it left to centre it',
     'pos': (30, 720)},
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 860), 'w': 88, 'h': 42},
    {'key': 'st', 'init': 'string', 'pos': (30, 920), 'w': 160, 'h': 42,
     'props': {'text in': 'hello', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'type new text and press return',
     'pos': (30, 920)},
    # font pinned to its default: restore_properties skips a no-op restore,
    # so font_changed never fires on load
    {'key': 'tx', 'init': 'mgl_text', 'pos': (30, 990), 'w': 160, 'h': 110,
     'props': {'font': 'Inconsolata-g.otf'}},
]
links = [('ctx', 'mgl_chain', 'cam', 'mgl chain in'),
         ('ctx', 'ui', 'cam', 'ui'),
         ('cam', 'mgl chain out', 'bb', 'mgl chain in'),
         ('bb', 'mgl chain out', 'tr', 'mgl chain in'),
         ('tr', 'mgl chain out', 'tx', 'mgl chain in'),
         ('lb', 'out', 'st', ''),
         ('st', 'string out', 'tx', 'text')]
print(build('mgl_text', 'mgl_text - text in the scene', body, demo, links,
            demo_width=560, text_width=810, text_height=900))

# ------------------------------------------------------------------- mgl_body
body = """These draw a moving body, and the things you want to see about how it moves.

THE NODES:

mgl_body              a skeleton, driven by a pose
body_proportions      per-segment length, width and depth factors for it
mgl_body_orientation  the same, with per-joint orientation disks
mgl_orientation_disks the disks on their own
mgl_contact_disks     where the body is touching the ground, sized by area
mgl_torque_arc        a torque drawn as an arc about its own axis

mgl_body IS THE MAIN ONE:
Patch a stream of quaternions into 'pose' and it draws the skeleton. It accepts 
data as fast as it arrives and draws the latest on each render, so the frame 
rate of the source and of the display are independent.

'skeleton_mode' and 'display_mode' change what it draws - spheres at the joints, 
limbs between them, or both.

CHANGING THE PROPORTIONS:
There are two layers, and they stack.

'limb_lengths' takes the dict smpl_beta_editor sends and sets the BASE skeleton: 
absolute segment lengths in metres, for a body that is not the default size.

'limb_scale' multiplies whatever the base is. Each segment has three factors - 
length, width, depth - with 1.0 meaning unchanged. Length moves the joint at the 
far end and stretches the drawn limb; width and depth fatten it. The pelvis bowl 
and shoulder blades keep their drawn shape under a length factor and only move 
what hangs off them. The factors stay put when new base lengths arrive or the 
skeleton mode changes.

Send it a dict, or a message:

limb_scale left_upper_arm 3.0 0.5 0.5    length, width, depth
limb_scale upper_arm 1.2                 one value: length only, both sides
limb_scale left_hand 2.0 2.0             two values: width and depth only
limb_scale left 0.7                      a whole side
limb_scale all 1.0 2.0 2.0               everything
limb_scale reset

The same messages work on the 'mgl chain in' inlet. Segment names are the 
segment that ENDS at a joint: left_upper_leg is hip to knee. The names are 
spine_lower, spine_mid, spine_upper, spine_to_neck, neck, head, and per side 
hip, upper_leg, lower_leg, foot, toes, heel, shoulder_blade, collar, upper_arm, 
lower_arm, hand, fingers.

body_proportions IS THE HAND-EDITING FRONT END FOR THAT:
One row per segment, three drag floats each. It sends the whole dict on every 
edit and once when the patch loads, so mgl_body needs no priming - wire 
'limb_scale' to 'limb_scale' and drag. mgl_smpl_mesh and mgl_smpl_heatmap take 
the same inlet, so the one node can drive the skeleton and the skin together. 'symmetric' mirrors a left edit onto the 
right row; untick it and the sides go their own way, which is how you get a 
lopsided body on purpose. 'scales in' takes the same dict or message forms 
listed above and sets the rows, so a preset or a patch can drive it. A change 
costs nothing beyond the normal draw - the dims ride in each limb's bone matrix 
rather than in its geometry - so animate the proportions freely, from the node 
or with messages at frame rate.

SEEING WHAT A NUMBER MEANS:
The other four nodes exist because a value about a body is much easier to 
understand drawn ON the body than plotted beside it.

mgl_contact_disks puts a disk at each contact, scaled by the area - so the 
weight shifting between the feet is something you watch rather than read. 
mgl_torque_arc draws a torque as an arc around the axis it acts about, which is 
what a torque actually is and what a number cannot show. 
mgl_orientation_disks shows each joint's orientation as a ring, so a twist is 
visible as a twist.

THE CALLBACK OUTLETS:
mgl_body reports which joint was clicked on - 'joint_id' and 'joint_callback' - 
so the scene can be an interface. Tick 'enable_callbacks' and clicking a joint 
tells the patch which one, which is how you select a joint to inspect without 
building a list of them somewhere else.

SYNTAX:
mgl_body
mgl_contact_disks

EXAMPLE:
mgl_body

INPUTS and PARAMETERS:

pose:
The joint rotations, as quaternions. This is the data inlet.

skeleton_mode / display_mode / draw_spheres:
What to draw.

scale / joint_radius / limb_lengths:
The size of the body and its parts.

limb_scale:
Per-segment length, width and depth factors over the base. See above.

scales in / symmetric / reset (body_proportions):
Set every row from a dict, mirror edits across the body, or put everything 
back to 1.0.

joint_data / color:
Per-joint values and colours - this is how you colour joints by a measurement.

s_curve_spine:
Whether the spine is drawn as a curve rather than straight segments.

instanced_mode:
Draw the joints in one call rather than separately. Faster for a full skeleton.

contacts / area_scale / min_radius / max_radius (mgl_contact_disks):
The contact data and how its area maps to a disk size.

show_disks / disk_scale / disk_orientations (mgl_body_orientation):
The orientation rings.

OUTPUTS: 

mgl chain out:
The chain, continuing.

joint_id / joint_callback / joint_data:
Which joint was clicked, and what it carries.

RELATED:
See the smpl nodes for where pose data comes from, and the quaternion nodes for 
working on it before it is drawn."""

demo = chain() + [
    {'key': 'bd', 'init': 'mgl_body', 'pos': (30, 660), 'w': 260, 'h': 400},
    {'key': 'c0', 'comment': True, 'text': 'patch a stream of quaternions into pose\ntick enable_callbacks, then click a joint',
     'pos': (30, 1075)},
    {'key': 'i1', 'init': 'int', 'pos': (340, 660), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'which joint was clicked', 'pos': (340, 710)},
    {'key': 'bp', 'init': 'body_proportions', 'pos': (340, 760), 'w': 300, 'h': 980},
    {'key': 'c4', 'comment': True, 'text': 'each row: length, width, depth\nuntick symmetric for a lopsided body',
     'pos': (340, 1755)},
    {'key': 'cd', 'init': 'mgl_contact_disks', 'pos': (30, 1150), 'w': 260, 'h': 300},
    {'key': 'c3', 'comment': True, 'text': 'contacts drawn where they happen,\neach disk sized by its area',
     'pos': (30, 1465)},
]
links = CHAIN_LINKS + [
    ('lgt', 'mgl chain out', 'bd', 'mgl chain in'),
    ('bd', 'joint_id', 'i1', ''),
    ('bp', 'limb_scale', 'bd', 'limb_scale'),
    ('bd', 'mgl chain out', 'cd', 'mgl chain in')]
print(build('mgl_body', 'mgl_body - drawing a body, and what it is doing', body,
            demo, links, demo_width=580, text_width=820, text_height=820))
