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
plane_xy, plane_xz, or box, which puts the image once on each side of the 
bounding box, each face taking the side it faces most - applied the 
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


# ------------------------------------------------------------- body scene
def scene(y=62, light=False):
    """A context showing its picture in the node, an orbit camera and,
    for lit drawing, a light. The chain continues from the last key."""
    nodes = [
        {'key': 'ctx', 'init': 'mgl_context', 'pos': (30, y), 'w': 340, 'h': 340,
         'props': {'auto_render': True, 'node_display_width': 320,
                   'node_display_height': 240, 'node_mouse_events': True}},
        {'key': 'c0', 'comment': True, 'text': 'drag on the picture to orbit',
         'pos': (30, y + 58)},
        {'key': 'cam', 'init': 'mgl_orbit_camera', 'pos': (30, y + 378), 'w': 240, 'h': 160},
    ]
    links = [('ctx', 'mgl_chain', 'cam', 'mgl chain in'),
             ('ctx', 'ui', 'cam', 'ui')]
    last = 'cam'
    if light:
        nodes.append({'key': 'lgt', 'init': 'mgl_light', 'pos': (30, y + 568), 'w': 220, 'h': 200})
        links.append(('cam', 'mgl chain out', 'lgt', 'mgl chain in'))
        last = 'lgt'
    return nodes, links, last


# ------------------------------------------------------------------- mgl_body
body = """mgl_body draws a skeleton in the scene and moves it with a stream of
joint rotations.

WHAT IT DRAWS:
With nothing patched into 'pose' it stands in its rest pose, arms out,
about 1.8 units tall with the pelvis at the origin of the chain's current
coordinates (one unit is a metre, y is up). Each limb is a block running
from one joint to the next.

'display_mode' lines draws the edges of those blocks as plain lines in
the current colour (set it with mgl_color), with no lighting. solid draws
them filled and lit, so it needs an mgl_light earlier in the chain.

THE POSE:
'pose' takes one quaternion per joint, w first, as an array - what the
shadow node's body quaternions outlet sends, or smpl_to_active makes from
an SMPL pose. A plain list works too: nested, one [w x y z] per joint, or
flat, read four numbers at a time. The
node keeps the latest pose and draws it on each render, so the source and
the display can run at different rates.

With 'input_order' set to smpl it takes SMPL joint order instead, as
quaternions or as rotation vectors (axis times angle), and rearranges them
onto its own joints; joints SMPL does not drive stay at rest. SMPL poses
are z-up - tick 'y is up' to stand such a body upright. 'joint_data' is
rearranged the same way.

SPHERES SIZED BY A MEASUREMENT:
'joint_data' takes one value per joint, in the same joint order as the pose.
With 'draw_spheres' ticked a sphere is drawn at each joint with a radius of
that value times 'joint_radius' - so a per-joint measurement (effort, speed,
noise) is seen where it happens. A row of several numbers per joint is
reduced to its length. No spheres are drawn until joint_data has arrived.
'color' is the colour of the spheres, and 'instanced_mode' draws them
solid, as wireframe or as points.

DRAWING AT EVERY JOINT:
With 'enable_callbacks' ticked (it is by default), on every render the node
visits each joint in turn: it moves the chain's coordinates to that joint,
sends the joint's number from 'joint_id', that joint's entry of joint_data
from 'joint_data', and then 'draw' from 'joint_callback'. Anything patched to
joint_callback is drawn once per joint, at the joint - a marker, a label,
a shape sized by joint_data. The coordinates there keep the orientation
of the limb leading into the joint. This is not mouse picking: nothing is
clicked, and the outlets fire for every joint on every frame.

PROPORTIONS:
'limb_lengths' takes the dict smpl_beta_editor sends from its limb_lengths
outlet, and resizes the base skeleton to those segment lengths. 'limb_scale'
multiplies whatever the base is, segment by segment - see body_proportions,
which is the hand-editing front end for it. The two stack: new base lengths
leave the scale factors in place.

limb_scale takes a dict {segment: factors} or a message:

limb_scale left_upper_arm 3.0 0.5 0.5    length, width, depth
limb_scale upper_arm 1.2                 one value: length only, both sides
limb_scale left_hand 2.0 2.0             two values: width and depth only
limb_scale left 0.7                      a whole side
limb_scale all 1.0 2.0 2.0               everything
limb_scale reset

Send these to the 'limb_scale' inlet, or as messages into 'mgl chain in'.
A message on the chain is used here and not passed on, so it reshapes this
body and not any body further down the chain.

'skeleton_mode' smpl rebuilds the skeleton from SMPL's own bone offsets -
from limb_lengths if it carried them, otherwise from the neutral SMPL body
(this needs the smplx package). 's_curve_spine' bends that SMPL spine into
a natural S curve. shadow, the default, is the motion capture suit's
skeleton.

SYNTAX:
mgl_body

It takes no arguments.

EXAMPLE:
mgl_body

INPUTS and PARAMETERS:

mgl chain in:
The chain. 'draw' draws the body.

pose:
Joint rotations, as above.

enable_callbacks:
Whether the per-joint outlets fire. On by default.

draw_spheres:
Whether joint_data is shown as spheres.

display_mode:
lines or solid.

scale:
Size of the whole body, about the chain's origin. 1 by default.

joint_radius:
Sphere radius per unit of joint_data. 0.1 by default.

joint_data:
One value (or one row) per joint.

limb_lengths / limb_scale:
The base skeleton and the factors over it, as above.

skeleton_mode:
shadow or smpl.

input_order:
active (the body's own joint order) or smpl.

y is up:
Turns an SMPL pose upright. Only used with input_order smpl.

s_curve_spine:
Shapes the SMPL skeleton's spine. Only used in smpl skeleton mode.

color:
The colour of the spheres.

instanced_mode:
solid, wireframe or points, for the spheres.

OUTPUTS:

mgl chain out:
The chain, continuing.

joint_callback / joint_data / joint_id:
Once per joint per render, as above.

RELATED:
body_proportions edits limb_scale by hand. mgl_body_orientation is this
node with orientation disks at the joints. mgl_contact_disks and
mgl_torque_arc draw other body measurements in the scene. mgl_smpl_mesh
draws a skin rather than a skeleton."""

demo, links, last = scene(light=True)
demo += [
    {'key': 'bd', 'init': 'mgl_body', 'pos': (30, 860), 'w': 260, 'h': 420},
    {'key': 'c1', 'comment': True,
     'text': 'nothing in pose, so it stands at rest\npatch body quaternions into pose',
     'pos': (30, 860)},
    {'key': 'met', 'init': 'metro 500', 'pos': (340, 640), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 500.0, 'units': 'milliseconds'}},
    {'key': 'nr', 'init': 'np.rand 31', 'pos': (340, 730), 'w': 140, 'h': 110,
     'props': {'min': 0.0, 'max': 0.6}},
    {'key': 'c2', 'comment': True, 'text': 'one value per joint\nsizes the spheres',
     'pos': (340, 730)},
    {'key': 'bx', 'init': 'mgl_box', 'pos': (30, 1310), 'w': 220, 'h': 160,
     'props': {'size': [0.04, 0.04, 0.04]}},
    {'key': 'c3', 'comment': True,
     'text': 'joint_callback draws this small box\nonce at every joint',
     'pos': (30, 1310)},
]
links += [
    (last, 'mgl chain out', 'bd', 'mgl chain in'),
    ('met', '', 'nr', ''),
    ('nr', 'random array', 'bd', 'joint_data'),
    ('bd', 'joint_callback', 'bx', 'mgl chain in')]
print(build('mgl_body', 'mgl_body - drawing a moving body', body,
            demo, links, demo_width=580, text_width=820, text_height=820))

# ----------------------------------------------------------- body_proportions
body = """body_proportions is a panel of sliders for reshaping a body: one row per
segment, each with a length, a width and a depth factor.

HOW IT IS USED:
Patch its 'limb_scale' outlet to the 'limb_scale' inlet of mgl_body,
mgl_body_orientation, mgl_smpl_mesh or mgl_smpl_heatmap - they all speak the
same vocabulary, so one panel can drive the skeleton and the skin together.

It sends the whole set of factors, as a dict, on every edit and once when
the patch opens, so the body needs no priming: open the patch and the
saved proportions are already applied.

THE FACTORS:
1.0 means unchanged. Length moves the joint at the far end of the segment
and stretches the drawn limb; width and depth fatten or thin it. Each
segment is named for the joint it ENDS at: left_upper_leg is hip to knee,
left_hand is wrist to knuckles. The pelvis bowl and the shoulder blades
keep their drawn shape under a length factor and only move what hangs off
them.

The rows are spine_lower, spine_mid, spine_upper, spine_to_neck, neck, head,
and for each side hip, upper_leg, lower_leg, foot, toes, heel,
shoulder_blade, collar, upper_arm, lower_arm, hand, fingers.

Reshaping costs nothing beyond the normal draw on mgl_body, so the factors
can be animated freely.

SYMMETRY:
With 'symmetric' ticked (the default) an edit to a left row is copied to
the matching right row and the other way round. Untick it and each side
keeps its own values - how you make a lopsided body on purpose. Values
arriving at 'scales in' are not mirrored.

SETTING IT FROM THE PATCH:
'scales in' takes a dict {segment: factors}. A segment name can be a row
name, a name without its side (upper_arm sets both sides), left, right or
all. One number sets the length only, two set width and depth, three set
all three. The rows update and the whole set is sent on. 'reset' puts
every row back to 1.0 and sends.

A message naming one row, such as 'left_upper_arm 1.2 1 1', sets that row as
an edit would: mirrored to the other side if 'symmetric' is ticked, then sent.

Its rows are also saved by the presets node.

SYNTAX:
body_proportions

It takes no arguments.

EXAMPLE:
body_proportions

INPUTS and PARAMETERS:

scales in:
A dict of factors, as above.

symmetric:
Mirror edits between left and right rows.

reset:
Back to 1.0 everywhere.

spine_lower ... right_fingers:
One row per segment: length, width, depth.

OUTPUTS:

limb_scale:
A dict of every segment's three factors.

RELATED:
mgl_body, whose page lists the limb_scale messages those nodes also accept
directly. smpl_beta_editor sets a body's base proportions from SMPL shape
values; this scales on top of that."""

demo, links, last = scene(light=True)
demo += [
    {'key': 'bd', 'init': 'mgl_body', 'pos': (30, 860), 'w': 260, 'h': 420,
     'props': {'display_mode': 'solid'}},
    {'key': 'bp', 'init': 'body_proportions', 'pos': (400, 62), 'w': 300, 'h': 980},
    {'key': 'c1', 'comment': True,
     'text': 'drag a row: length, width, depth\nuntick symmetric for a lopsided body',
     'pos': (400, 1060)},
]
links += [
    (last, 'mgl chain out', 'bd', 'mgl chain in'),
    ('bp', 'limb_scale', 'bd', 'limb_scale')]
print(build('body_proportions', 'body_proportions - reshaping a body by hand', body,
            demo, links, demo_width=720, text_width=800, text_height=820))

# ------------------------------------------------------- mgl_body_orientation
body = """mgl_body_orientation is mgl_body with a set of orientation disks drawn at
each joint.

Everything mgl_body does, it does - the same pose, joint_data, proportions,
callbacks and inputs (see the mgl_body page). On top of that, each joint
can carry several rings, each turned to face along a rotation axis and
sized by the rotation's angle. A disk lying in the plane of a turn, as large
as the turn, is a rotation you can see: which way it faces is the axis, how
big it is is how far.

THE DISK DATA:
'disk_orientations' takes, for each joint, one row per disk:

  joints by disks by 3       rotation vectors: the direction is the axis,
                             the length the angle in radians
  joints by disks by 4       an axis followed by a size
  joints by disks by 3 by 3  rotation matrices

Joints are in the body's own order - 20 of them - or in SMPL order with
'input_order' smpl. The rows are read in the frame of the limb leading
into each joint. A disk with no rotation is not drawn, and the data stays
until new data arrives.

SIZES AND COLOURS:
A disk's radius is its angle (or size) times 'disk_scale' times that
disk's own factor. The factors grow by half again from disk to disk
(2.25, 2.25, 3.375, 5.06 and so on), so for the same angle each disk
is drawn larger than the one before.
'disk_scales' replaces them with a list. 'disk color 0', 'disk color 1'
and so on set each disk's colour as red green blue alpha from 0 to 1; the
same colour is used for that disk at every joint. The colours glow
regardless of lighting and are translucent by default.

Each disk is a ring: 'disk_ring_width' is the ring's width as a fraction
of its radius (1 fills it in). 'disk_slices' sets how round it is.

SYNTAX:
mgl_body_orientation
mgl_body_orientation <disks per joint>

7 disks per joint by default.

EXAMPLE:
mgl_body_orientation 3

INPUTS and PARAMETERS:

All of mgl_body's, and:

show_disks:
Whether the disks are drawn.

disk_scale:
Radius per radian (or per unit of size). 0.5 by default.

disk_slices / disk_ring_width:
Roundness, and ring width as a fraction of the radius.

disk_scales:
A list of per-disk factors.

disk color 0, disk color 1 ...:
One inlet per disk, red green blue alpha.

disk_orientations:
The rotations, as above.

OUTPUTS:

The same as mgl_body.

RELATED:
mgl_body without the disks. mgl_orientation_disks draws a set of disks at
one place, wherever the chain puts it."""

demo, links, last = scene()
demo += [
    {'key': 'met', 'init': 'metro 1000', 'pos': (340, 440), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 1000.0, 'units': 'milliseconds'}},
    {'key': 'nr', 'init': 'np.rand 20 3 3', 'pos': (340, 530), 'w': 140, 'h': 160,
     'props': {'min': -0.4, 'max': 0.4}},
    {'key': 'c1', 'comment': True,
     'text': '20 joints, 3 disks each,\na rotation vector per disk',
     'pos': (340, 530)},
    {'key': 'bo', 'init': 'mgl_body_orientation 3', 'pos': (30, 640), 'w': 280, 'h': 600,
     'props': {'disk_scale': 0.2}},
    {'key': 'c2', 'comment': True,
     'text': 'each ring faces along its axis,\nits size is the angle',
     'pos': (30, 1260)},
]
links += [
    (last, 'mgl chain out', 'bo', 'mgl chain in'),
    ('met', '', 'nr', ''),
    ('nr', 'random array', 'bo', 'disk_orientations')]
print(build('mgl_body_orientation', 'mgl_body_orientation - a body with rotation disks',
            body, demo, links, demo_width=580, text_width=800, text_height=820))

# ------------------------------------------------------ mgl_orientation_disks
body = """mgl_orientation_disks draws rotations as rings, so an orientation can be
seen rather than read.

A rotation is three or four numbers and you cannot read it. A ring lying in
the plane of the rotation, as large as the rotation is, you can: the
direction it faces is the axis, and its size is how far it has turned.

THE DATA:
'axis-angle' takes one row per disk. A row of three is a rotation vector -
its direction is the axis and its length the angle in radians - and the
disk is drawn facing along that axis with a radius of the angle times
'scale'. A row of four is an axis followed by a size, and the radius is
that size times 'scale'. A disk with no rotation is not drawn. Rows beyond
the node's disk count are ignored.

A single disk can be sent as a flat row, [x y z] or [x y z size]: a flat
list is read as one row.

WHERE THEY APPEAR:
All the disks are drawn at the same place - the origin of the chain's
current coordinates - one over another. To put a set at a joint, use a
transform before it, or patch it to mgl_body's joint_callback; or use
mgl_body_orientation, which draws a set at every joint at once.

COLOURS:
Each disk has its own colour, sent to 'color 0', 'color 1' and so on as
red green blue alpha from 0 to 1. The colours glow regardless of lighting,
and the defaults are translucent so overlapping rings show through each
other.

'ring_width' is the ring's width as a fraction of its radius - 1 fills the
disk in. 'slices' sets how round it is.

SYNTAX:
mgl_orientation_disks
mgl_orientation_disks <count> <scale> <slices>

All optional: 6 disks, scale 0.5, 32 slices.

EXAMPLE:
mgl_orientation_disks 3 0.25

INPUTS and PARAMETERS:

mgl chain in:
The chain. 'draw' draws the disks.

scale:
Radius per radian (or per unit of size, for four-number rows).

slices / ring_width:
Roundness, and ring width as a fraction of the radius. 0.1 by default.

axis-angle:
One row per disk, as above. It is kept until new rows arrive.

color 0, color 1 ...:
One inlet per disk, red green blue alpha.

OUTPUTS:

mgl chain out:
The chain, continuing.

RELATED:
mgl_body_orientation draws these at every joint of a body.
gl_orientation_disks is the older equivalent."""

demo, links, last = scene()
demo += [
    {'key': 'met', 'init': 'metro 1000', 'pos': (340, 440), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 1000.0, 'units': 'milliseconds'}},
    {'key': 'nr', 'init': 'np.rand 3 3', 'pos': (340, 530), 'w': 140, 'h': 110,
     'props': {'min': -1.5, 'max': 1.5}},
    {'key': 'c1', 'comment': True, 'text': 'three random rotation vectors,\none row per disk',
     'pos': (340, 530)},
    {'key': 'od', 'init': 'mgl_orientation_disks 3 0.25', 'pos': (30, 640), 'w': 220, 'h': 260},
    {'key': 'c2', 'comment': True,
     'text': 'each disk faces along its axis,\nits radius is the angle times scale',
     'pos': (30, 920)},
]
links += [
    (last, 'mgl chain out', 'od', 'mgl chain in'),
    ('met', '', 'nr', ''),
    ('nr', 'random array', 'od', 'axis-angle')]
print(build('mgl_orientation_disks', 'mgl_orientation_disks - seeing rotations',
            body, demo, links, demo_width=560, text_width=800, text_height=760))

# ---------------------------------------------------------- mgl_contact_disks
body = """mgl_contact_disks draws a flat disk on the floor at each place a body is
touching the ground, with the disk's area standing for the force there.

Weight moving from one foot to the other becomes something you watch rather
than read.

THE DATA:
'contacts' takes one row per contact: x, y, z and a force - what
smpl_torque's contact_points outlet sends, with the force in newtons. A
flat list is read four numbers at a time, and an empty array clears the
disks. A contact with no force (or a negative one) is not drawn.

THE SIZE:
The force times 'area_scale' is the disk's AREA, so the radius grows as
the square root of the force - twice the force is twice the area, not
twice the width. With the default 0.001, 500 newtons (a foot carrying
most of a person's weight) is a disk of radius about 0.4.
'min_radius' and 'max_radius' clamp the radius; a max_radius of 0 or less
means no limit.

THE DISKS:
Each disk lies flat in the x-z plane - y is up - at the contact's height
plus 'height_offset', which lifts it just clear of a floor drawn at the
same height. For z-up data, turn the chain before this node. The disks
are lit shapes in the current colour, so they need an mgl_light earlier in
the chain, and they take the usual shape inputs: 'mode' (solid, wireframe
or points), 'cull', 'point_size', 'round' and 'texture'. 'segments' sets how
round each disk is.

SYNTAX:
mgl_contact_disks

It takes no arguments.

EXAMPLE:
mgl_contact_disks

INPUTS and PARAMETERS:

mgl chain in:
The chain. 'draw' draws the disks.

contacts:
x y z force, one row per contact. Kept until new contacts arrive.

area_scale:
Disk area per unit of force. 0.001 by default.

min_radius / max_radius:
Limits on the radius. 0 and 0.5 by default.

height_offset:
Lift above the contact height. 0.001 by default.

segments:
Roundness. 24 by default.

mode / cull / point_size / round / texture:
As for the other mgl shapes.

OUTPUTS:

mgl chain out:
The chain, continuing.

RELATED:
smpl_torque produces the contacts. mgl_body draws the body standing on
them; mgl_torque_arc draws the joint torques."""

demo, links, last = scene(light=True)
demo += [
    {'key': 'lb', 'init': 'load_bang', 'pos': (340, 640), 'w': 88, 'h': 42},
    {'key': 'm1', 'init': 'message', 'pos': (340, 700), 'w': 300, 'h': 42,
     'props': {'text in': '-0.3 0 0 450 0.3 0 0.1 150', 'font size': '24'}},
    {'key': 'm2', 'init': 'message', 'pos': (340, 760), 'w': 300, 'h': 42,
     'props': {'text in': '-0.3 0 0 100 0.3 0 0.1 500', 'font size': '24'}},
    {'key': 'c1', 'comment': True,
     'text': 'two feet: x y z force, x y z force\nclick each to shift the weight',
     'pos': (340, 820)},
    {'key': 'cd', 'init': 'mgl_contact_disks', 'pos': (30, 860), 'w': 260, 'h': 420},
    {'key': 'c2', 'comment': True,
     'text': 'disk area follows the force',
     'pos': (30, 1300)},
]
links += [
    (last, 'mgl chain out', 'cd', 'mgl chain in'),
    ('lb', 'out', 'm1', ''),
    ('m1', '', 'cd', 'contacts'),
    ('m2', '', 'cd', 'contacts')]
print(build('mgl_contact_disks', 'mgl_contact_disks - where the weight is', body,
            demo, links, demo_width=660, text_width=800, text_height=800))

# ------------------------------------------------------------- mgl_torque_arc
body = """mgl_torque_arc draws each of a set of torques as an arc with an arrowhead,
turning about the torque's own axis at the place it acts.

A torque is a turning effort about an axis - which is exactly what a number
cannot show and an arc can. Its size is in the arc's size; its axis is the
axis the arc goes around.

THE DATA:
Two inputs, both one row of three per joint: 'torques' (each a torque
vector) and 'positions' (where each is drawn). Nothing is drawn until both
have arrived; then each keeps its last value, so positions can be sent
once and torques streamed. Only as many arcs are drawn as there are rows
in the shorter of the two. Rows must be n by 3 arrays - a flat list is
ignored. The positions are in the chain's current coordinates.

THE ARC:
Its radius is the torque's size times 'scale', so 'scale' converts torque
units into scene units - for net joint torques in newton metres, the
default 0.01 makes 100 newton metres a radius of 1. It sweeps 'sweep'
degrees around the axis, 270 by default, and ends in an arrowhead 'arrow
size' times the radius long. The arc runs AGAINST the turn the torque
vector describes (by the right-hand rule) - it shows the resisting torque,
the way the joint is being loaded. Torques smaller than 'threshold' are
not drawn.

The arcs are plain lines in the current colour - set it with mgl_color -
with no lighting.

SYNTAX:
mgl_torque_arc

It takes no arguments.

EXAMPLE:
mgl_torque_arc

INPUTS and PARAMETERS:

mgl chain in:
The chain. 'draw' draws the arcs.

torques:
One torque vector per row. Receiving it does not draw; the next render does.

positions:
One position per row.

scale:
Radius per unit of torque. 0.01 by default.

threshold:
Smallest torque drawn. 0.1 by default.

sweep:
How far round each arc goes, in degrees.

segments:
How smooth each arc is.

arrow size:
Arrowhead length as a fraction of the radius. 0.3 by default.

OUTPUTS:

mgl chain out:
The chain, continuing.

RELATED:
smpl_torque computes joint torques; mgl_body draws the body they act on.
mgl_contact_disks shows the forces at the floor."""

demo, links, last = scene()
demo += [
    {'key': 'lb', 'init': 'load_bang', 'pos': (340, 440), 'w': 88, 'h': 42},
    {'key': 'np1', 'init': 'np.rand 3 3', 'pos': (340, 500), 'w': 140, 'h': 110,
     'props': {'min': -0.8, 'max': 0.8}},
    {'key': 'c1', 'comment': True, 'text': 'three positions, once',
     'pos': (340, 500)},
    {'key': 'met', 'init': 'metro 1000', 'pos': (340, 640), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 1000.0, 'units': 'milliseconds'}},
    {'key': 'np2', 'init': 'np.rand 3 3', 'pos': (340, 730), 'w': 140, 'h': 110,
     'props': {'min': -1.0, 'max': 1.0}},
    {'key': 'c2', 'comment': True, 'text': 'three torques, each second',
     'pos': (340, 730)},
    {'key': 'ta', 'init': 'mgl_torque_arc', 'pos': (30, 640), 'w': 240, 'h': 280,
     'props': {'scale': 0.3}},
    {'key': 'c3', 'comment': True,
     'text': 'scale 0.3: radius is the torque size\ntimes 0.3',
     'pos': (30, 940)},
]
links += [
    (last, 'mgl chain out', 'ta', 'mgl chain in'),
    ('lb', 'out', 'np1', ''),
    ('np1', 'random array', 'ta', 'positions'),
    ('met', '', 'np2', ''),
    ('np2', 'random array', 'ta', 'torques')]
print(build('mgl_torque_arc', 'mgl_torque_arc - a torque drawn as a turn', body,
            demo, links, demo_width=560, text_width=800, text_height=800))
