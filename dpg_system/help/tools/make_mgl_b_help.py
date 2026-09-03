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

mgl_mesh        vertices and faces you hand it
mgl_model       geometry loaded from a file
mgl_point_cloud a set of points
mgl_surface     a height field drawn as a surface
mgl_line_array  many lines at once, with per-line colour and width
mgl_text        text in the scene

mgl_line_array IS THE ONE BUILT FOR MOTION:
Drawing a trail or a set of trajectories as separate line nodes does not scale. 
This takes the whole array at once and draws it in one go, and its options are 
about making motion legible rather than about geometry: 'alpha_fade' and 
'fade_rate' let older parts of a trail die away; 'accent_motion' brightens and 
thickens the lines where they are moving fastest, so speed reads directly; 
'perspective_width' keeps distant lines from vanishing.

Those accent controls are the difference between a plot of a movement and a 
picture of it. A trail drawn at constant width and brightness tells you where 
something went; one that thickens where it accelerated tells you how.

mgl_surface FOR A FIELD:
Give it a 2D array of heights and it draws the surface. That is the way to see 
a spectrum over time, a distance field, a correlation matrix - anything where 
the interesting thing is the shape of a 2D function rather than its numbers.

mgl_text NEEDS A BILLBOARD:
Text drawn in 3D turns edge-on and disappears as the camera moves. 
Put mgl_billboard before it and it stays facing the viewer.

SYNTAX:
mgl_point_cloud
mgl_line_array
mgl_model <path>

EXAMPLE:
mgl_line_array

INPUTS and PARAMETERS:

mgl chain in:
The chain. This triggers the drawing.

mesh / array / file_path:
The geometry. These inlets accept streaming data - a point cloud or a line 
array can be replaced every frame.

scale / center / fit (mgl_mesh, mgl_model):
How to place the loaded geometry. 'fit' scales it to a sensible size, which 
saves guessing at the units a model file was saved in.

line_width / alpha_fade / fade_rate (mgl_line_array):
The basic appearance of the lines and how they die away.

accent_motion / accent_brightness / accent_scale / accent_width:
How much the lines respond to speed, and in what way.

use_line_colors / color index / color_control:
Whether each line carries its own colour, and where that comes from.

OUTPUTS: 

mgl chain out:
The chain, continuing.

A NOTE ON DATA ARRIVING FAST:
These inlets take streaming data, and the data can arrive on another thread. 
The nodes store what arrives and draw it on the next render from the main 
thread, so a fast source cannot outrun the renderer or corrupt a frame - it 
simply means some frames are skipped, which is the right behaviour."""

demo = chain() + [
    {'key': 'la', 'init': 'mgl_line_array', 'pos': (30, 660), 'w': 260, 'h': 380},
    {'key': 'c0', 'comment': True, 'text': 'patch an array of line points in',
     'pos': (30, 1055)},
    {'key': 'c1', 'comment': True, 'text': 'raise accent_motion to make speed visible',
     'pos': (30, 1085)},
    {'key': 'bb', 'init': 'mgl_billboard', 'pos': (30, 1130), 'w': 220, 'h': 90},
    # font pinned to its default, as on gl_text: restore_properties skips a
    # no-op, so font_changed never fires and no file dialog is raised
    {'key': 'tx', 'init': 'mgl_text', 'pos': (30, 1235), 'w': 220, 'h': 160,
     'props': {'font': 'Inconsolata-g.otf'}},
    {'key': 'c2', 'comment': True, 'text': 'the billboard keeps the text readable',
     'pos': (30, 1410)},
]
links = CHAIN_LINKS + [
    ('lgt', 'mgl chain out', 'la', 'mgl chain in'),
    ('la', 'mgl chain out', 'bb', 'mgl chain in'),
    ('bb', 'mgl chain out', 'tx', 'mgl chain in')]
print(build('mgl_mesh', 'mgl geometry - drawing data you supply', body, demo, links,
            demo_width=560, text_width=810, text_height=780))

# ------------------------------------------------------------------- mgl_body
body = """These draw a moving body, and the things you want to see about how it moves.

THE NODES:

mgl_body              a skeleton, driven by a pose
mgl_body_orientation  the same, with per-joint orientation disks
mgl_orientation_disks the disks on their own
mgl_contact_disks     where the body is touching the ground, sized by area
mgl_torque_arc        a torque drawn as an arc about its own axis

mgl_body IS THE MAIN ONE:
Patch a stream of quaternions into 'pose' and it draws the skeleton. It accepts 
data as fast as it arrives and draws the latest on each render, so the frame 
rate of the source and of the display are independent.

'skeleton_mode' and 'display_mode' change what it draws - spheres at the joints, 
limbs between them, or both. 'limb_lengths' lets you set the proportions rather 
than taking the defaults, which matters when the pose came from a body that is 
not the default size.

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
    {'key': 'c0', 'comment': True, 'text': 'patch a stream of quaternions into pose',
     'pos': (30, 1075)},
    {'key': 'c1', 'comment': True, 'text': 'tick enable_callbacks, then click a joint',
     'pos': (30, 1105)},
    {'key': 'i1', 'init': 'int', 'pos': (340, 660), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'which joint was clicked', 'pos': (340, 710)},
    {'key': 'cd', 'init': 'mgl_contact_disks', 'pos': (30, 1150), 'w': 260, 'h': 300},
    {'key': 'c3', 'comment': True, 'text': 'contacts drawn where they happen,',
     'pos': (30, 1465)},
    {'key': 'c4', 'comment': True, 'text': 'each disk sized by its area', 'pos': (30, 1495)},
]
links = CHAIN_LINKS + [
    ('lgt', 'mgl chain out', 'bd', 'mgl chain in'),
    ('bd', 'joint_id', 'i1', ''),
    ('bd', 'mgl chain out', 'cd', 'mgl chain in')]
print(build('mgl_body', 'mgl_body - drawing a body, and what it is doing', body,
            demo, links, demo_width=580, text_width=820, text_height=820))
