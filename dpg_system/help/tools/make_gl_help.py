"""the older GL system: context, transforms, appearance, shapes, data, text."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

OLDER = """
THIS IS THE OLDER SYSTEM:
mgl_ nodes do the same job in the newer scene graph, with real materials, 
shaders and a main-thread guard on the drawing. For anything new, use those.

These remain because patches are built on them, because gl_body and its 
relatives live in this world, and because the fixed-function pipeline they use 
is sometimes simpler to reason about when all you want is a few shapes.
"""

def chain(x=30, y=62):
    return [
        {'key': 'ctx', 'init': 'gl_context', 'pos': (x, y), 'w': 240, 'h': 120},
        {'key': 'lgt', 'init': 'gl_light', 'pos': (x, y + 140), 'w': 240, 'h': 240},
    ]

CHAIN_LINKS = [('ctx', 'gl_chain', 'lgt', 'gl chain in')]

# ------------------------------------------------------------------ gl_context
body = """gl_context is the root of a scene in the older GL system.

HOW THE CHAIN WORKS:

gl_context has a "gl_chain" outlet. Every other gl node has a "gl chain in" 
inlet and a "gl chain out" outlet, wired in a line. Each frame the context 
sends the word "draw" along it, and each node in turn draws itself and passes 
the message on.

The line looks flat and is not. Each node does:

  save the current state
  draw itself
  send "draw" onward - so everything downstream draws now
  restore the state

Because the downstream draw happens BETWEEN the save and the restore, whatever 
a node changes applies to everything after it and is undone at the end. 
That is what makes a chain of transforms nest like a scene graph without any 
branching syntax. To branch, split the chain cord - each branch is drawn inside 
whatever preceded the split.

THE NODES:

gl_context  the root: holds the GL context and drives the chain
gl_enable   turn a GL flag on or off for the rest of the chain

gl_enable IS MORE USEFUL HERE THAN IT LOOKS:
The fixed-function pipeline is a pile of state, and most of the visual 
behaviour you might want is a flag rather than a setting - depth testing, 
blending, face culling, lighting itself. Turning depth testing off for one 
branch is how you draw an overlay that is always on top; turning lighting off 
is how you draw something at a flat known colour among lit objects.

Because it obeys the same save-and-restore rule as everything else, the change 
lasts only for the rest of that branch.
""" + OLDER + """
SYNTAX:
gl_context
gl_enable

EXAMPLE:
gl_context

INPUTS and PARAMETERS:

commands (gl_context):
Messages to the context.

enabled / flag (gl_enable):
Which GL state, and whether on.

OUTPUTS: 

gl_chain:
The chain. Patch it into the first thing to be drawn.

ui:
Interaction events from the window.

RELATED:
mgl_context is the current equivalent, and its help patch explains the same 
chain idea. gl_body draws a skeleton in this chain."""

demo = chain() + [
    {'key': 'rot', 'init': 'gl_rotate', 'pos': (30, 516), 'w': 240, 'h': 160},
    {'key': 'sph', 'init': 'gl_sphere', 'pos': (30, 696), 'w': 240, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'the rotate applies to everything after it',
     'pos': (30, 911)},
    {'key': 'en', 'init': 'gl_enable', 'pos': (310, 516), 'w': 240, 'h': 120},
    {'key': 'c1', 'comment': True, 'text': 'a flag, for the rest of this branch',
     'pos': (310, 651)},
]
links = CHAIN_LINKS + [('lgt', 'gl chain out', 'rot', 'gl chain in'),
                       ('rot', 'gl chain out', 'sph', 'gl chain in')]
print(build('gl_context', 'gl_context - the older scene chain', body, demo, links,
            demo_width=580, text_width=810, text_height=740))

# ---------------------------------------------------------------- gl_translate
body = """These move, turn and resize everything that comes AFTER them in the chain.

A transform draws nothing. It changes where the drawing happens for every node 
downstream, then puts things back - so its position in the chain is its scope, 
and transforms accumulate down a branch. An arm on a body, a hand on the arm: 
the hierarchy falls out of the chain order without any extra syntax.

THE NODES:

gl_translate          move
gl_rotate             turn, as three angles
gl_quaternion_rotate  turn, given a quaternion
gl_axis_angle_rotate  turn, given an axis and an angle
gl_scale              resize
gl_align              turn so that a given direction points a given way

gl_align IS THE ONE WORTH KNOWING:
Give it a direction and it orients whatever follows to point along it. That is 
what you want for drawing a vector quantity in place - a velocity, a force, a 
bone direction, a surface normal. Working out the rotation that achieves that 
by hand is fiddly and easy to get wrong near the poles; this does it.

WHICH ROTATION NODE:
Three angles are readable and gimbal-lock. For anything driven by real 
orientation data use gl_quaternion_rotate - the sensors produce quaternions, 
and converting them to angles to feed gl_rotate throws away exactly the 
property that makes them safe. See the rotation conversions help patch.

gl_axis_angle_rotate is for rotations with a natural axis - a joint's own axis, 
a spin, a torque direction.

ORDER MATTERS, AND NOT THE WAY IT READS:
Rotate then translate turns the object on the spot and then moves it. 
Translate then rotate moves it away and then swings it about the origin, 
which is an orbit. If something orbits when you wanted it to spin, that is the 
two the wrong way round.
""" + OLDER + """
SYNTAX:
gl_translate <x> <y> <z>
gl_rotate <x> <y> <z>

EXAMPLE:
gl_translate 0 1 0

INPUTS and PARAMETERS:

gl chain in:
The chain. This triggers the node.

x / y / z:
The amounts. All accept a stream, so a transform can be driven continuously.

quaternion / rotation vector:
The rotation, for those nodes.


OUTPUTS: 

gl chain out:
The chain, with the transform in force downstream."""

demo = chain() + [
    {'key': 'tr1', 'init': 'gl_translate', 'pos': (30, 516), 'w': 240, 'h': 160},
    {'key': 'sph', 'init': 'gl_sphere', 'pos': (30, 696), 'w': 240, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'moved by the first translate', 'pos': (30, 911)},
    {'key': 'tr2', 'init': 'gl_translate', 'pos': (30, 951), 'w': 240, 'h': 160},
    {'key': 'cyl', 'init': 'gl_cylinder', 'pos': (30, 1131), 'w': 240, 'h': 220},
    {'key': 'c1', 'comment': True, 'text': 'by BOTH: they accumulate down the chain',
     'pos': (30, 1366)},
    {'key': 'al', 'init': 'gl_align', 'pos': (310, 951), 'w': 240, 'h': 160},
    {'key': 'c2', 'comment': True, 'text': 'points what follows along a direction',
     'pos': (310, 1126)},
]
links = CHAIN_LINKS + [
    ('lgt', 'gl chain out', 'tr1', 'gl chain in'),
    ('tr1', 'gl chain out', 'sph', 'gl chain in'),
    ('sph', 'gl chain out', 'tr2', 'gl chain in'),
    ('tr2', 'gl chain out', 'cyl', 'gl chain in')]
print(build('gl_translate', 'gl transforms - moving what comes after', body, demo,
            links, demo_width=580, text_width=810, text_height=760))

# ------------------------------------------------------------------- gl_light
body = """These decide how surfaces are lit and what colour they are.

THE NODES:

gl_light     a light source
gl_material  how a surface responds to light
gl_color     a flat colour, ignoring lighting

A SCENE NEEDS A LIGHT:
Without one, anything using a material is black. That is the usual reason a 
chain that looks correct renders a dark window - the shapes are being drawn and 
there is nothing to see them by.

gl_color IS NOT gl_material:
A colour is flat, so the shape comes out an even patch with no shading and no 
sense of form. A material responds to lights, so the same shape shows its 
curvature. Use colour for markers, lines and anything schematic; material for 
anything meant to look like an object.

THE COMPONENTS:
'ambient' is the colour in shadow, 'diffuse' the colour in plain light, 
'specular' the colour of the highlight, and 'shininess' how tight that highlight 
is - high is a small hard glint, low a broad sheen. The light has the matching 
three, and what you see is the product of the two, which is why a red light on 
a green object gives almost nothing.

gl_material also has 'emission', which the light does not: a surface that 
appears to give out light without illuminating anything else. That is how you 
make something read as glowing or as self-lit among lit objects.

MULTIPLE LIGHTS:
gl_light has an 'id' so several can exist at once, and 'positional' decides 
whether it is a point in the scene or a direction from infinitely far away. 
A directional light is the sun; a positional one falls off with distance and 
shows where it is.
""" + OLDER + """
SYNTAX:
gl_light
gl_material

EXAMPLE:
gl_material

INPUTS and PARAMETERS:

enabled / id / positional (gl_light):
Whether it is on, which light it is, and what kind.

position / ambient / diffuse / specular:
Where the light is and what it contributes.

ambient / diffuse / specular / emission / shininess / alpha (gl_material):
How the surface answers, and its transparency.

red / green / blue / alpha (gl_color):
The flat colour.

OUTPUTS: 

gl chain out:
The chain, with the setting in force downstream.

A NOTE ON ALPHA:
Transparency needs blending enabled, which is a gl_enable flag, and it is 
order-dependent - transparent things have to be drawn after what shows through 
them. If something transparent is disappearing behind what it should reveal, 
that is the chain order rather than the alpha value."""

demo = chain() + [
    {'key': 'mat', 'init': 'gl_material', 'pos': (30, 516), 'w': 240, 'h': 240},
    {'key': 'sph', 'init': 'gl_sphere', 'pos': (30, 776), 'w': 240, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'material: it shows its curvature',
     'pos': (30, 991)},
    {'key': 'col', 'init': 'gl_color', 'pos': (310, 516), 'w': 240, 'h': 200},
    {'key': 'cyl', 'init': 'gl_cylinder', 'pos': (310, 736), 'w': 240, 'h': 220},
    {'key': 'c1', 'comment': True, 'text': 'colour: flat, no shading', 'pos': (310, 971)},
]
links = CHAIN_LINKS + [
    ('lgt', 'gl chain out', 'mat', 'gl chain in'),
    ('mat', 'gl chain out', 'sph', 'gl chain in'),
    ('sph', 'gl chain out', 'col', 'gl chain in'),
    ('col', 'gl chain out', 'cyl', 'gl chain in')]
print(build('gl_light', 'gl light, material and colour', body, demo, links,
            demo_width=590, text_width=800, text_height=740))

# ------------------------------------------------------------------ gl_sphere
body = """The built-in shapes: geometry you can draw without supplying any.

THE NODES:

gl_sphere         a sphere
gl_cylinder       a cylinder, or a cone if the two radii differ
gl_disk           a filled circle, or a ring if you give it an inner radius
gl_partial_disk   a wedge - a disk with a start angle and a sweep
gl_line           a line between two points
gl_nested_spheres several spheres at a set of sizes, drawn together
gl_billboard      a flat rectangle, plain or showing a picture

gl_cylinder MAKES CONES:
Its base and top radii are separate, so setting the top to zero gives a cone 
and setting it small gives a tapered shaft. That is the shape you want for 
arrows and for anything that should read as pointing.

gl_partial_disk IS THE READOUT:
A wedge with a start angle and a sweep is how you draw a value as an arc - a 
dial, a proportion, a range. Patch a stream into 'sweep angle' and it becomes a 
gauge that lives in the scene rather than in the patch, next to the thing it 
describes.

gl_nested_spheres:
Takes a list of sizes and draws a sphere at each. With transparency enabled 
that is a set of shells - a way of showing several radii at once, a 
distribution over distance, or a falloff.

gl_billboard IS A SCREEN IN THE SCENE:
It draws a flat rectangle, centred on the origin and lying flat in the x-y 
plane, filled with whatever picture arrives at its 'texture' inlet - a camera, 
a movie, an image - or plain, if none has. Despite its name it does not turn 
to face the viewer; to aim it, put a transform before it. (The newer 
mgl_billboard is a different thing: a transform that really does turn what 
follows to face the camera.)

WHERE A SHAPE APPEARS:
At the origin, until a transform moves it. Two shapes with nothing between them 
are in the same place, one inside the other - which is the usual reason a scene 
seems to be missing something.

'slices' and 'stacks' are how finely a curved shape is divided. More is 
smoother and slower. On a sphere they are longitude and latitude, so the 
triangles bunch at the poles.
""" + OLDER + """
SYNTAX:
gl_sphere
gl_partial_disk

EXAMPLE:
gl_cylinder

INPUTS and PARAMETERS:

gl chain in:
The chain. This triggers the node.

size / base radius / top radius / height:
The dimensions.

outer radius / inner radius:
For the disks - an inner radius makes a ring.

start angle / sweep angle (gl_partial_disk):
Where the wedge begins and how far it goes.

slices / stacks / rings:
How finely it is divided.

start_vertex / end_vertex (gl_line):
The two ends.

sizes (gl_nested_spheres):
The list of radii.

width / height (gl_billboard):
The rectangle's size - 1.6 by 0.9 unless set.

texture:
An image mapped onto the surface.

OUTPUTS: 

gl chain out:
The chain, continuing - shapes change no state for what follows."""

demo = chain() + [
    {'key': 'cyl', 'init': 'gl_cylinder', 'pos': (30, 516), 'w': 240, 'h': 240},
    {'key': 'c0', 'comment': True, 'text': 'set top radius to 0 for a cone',
     'pos': (30, 771)},
    {'key': 'tr', 'init': 'gl_translate', 'pos': (30, 811), 'w': 240, 'h': 160},
    {'key': 'pd', 'init': 'gl_partial_disk', 'pos': (30, 991), 'w': 240, 'h': 280},
    {'key': 'c1', 'comment': True, 'text': 'drag sweep angle: a value as an arc',
     'pos': (30, 1286)},
    {'key': 'ns', 'init': 'gl_nested_spheres', 'pos': (310, 811), 'w': 240, 'h': 220},
    {'key': 'c2', 'comment': True, 'text': 'a sphere at each size in the list',
     'pos': (310, 1046)},
]
links = CHAIN_LINKS + [
    ('lgt', 'gl chain out', 'cyl', 'gl chain in'),
    ('cyl', 'gl chain out', 'tr', 'gl chain in'),
    ('tr', 'gl chain out', 'pd', 'gl chain in')]
print(build('gl_sphere', 'gl shapes - the built-in geometry', body, demo, links,
            demo_width=580, text_width=800, text_height=740))

# -------------------------------------------------------------- gl_line_array
body = """These draw geometry that comes from data rather than a built-in shape.

THE NODES:

gl_line_array     many lines at once, from an array - built for trails of motion
gl_vertex_buffer  raw vertex data, drawn as points, lines or triangles

gl_line_array IS BUILT FOR MOTION:
Drawing a trail or a set of trajectories as separate line nodes does not scale. 
This takes the whole array and draws it in one go. 'array' is points by lines 
by coordinates: each line is one column, its first point the head of the 
trail, so a rolling buffer of joint positions draws as one trail per joint.

Its options are about making motion legible rather than about geometry. 
'alpha_fade' (on by default) fades each line from its first point to its last, 
so a trail dies away behind its head. 'accent_motion' measures how far each 
point moved since the previous array and makes the lines only as bright as 
that movement times 'accent_scale', so a still trail disappears and a fast one 
shines; 'accent_colour' shows the same speed as colour instead, along a 
dark-to-yellow scale. 'selected_joints' restricts it to the lines you care 
about, which matters as soon as a full body's worth of trails becomes a 
thicket.

That accenting is the difference between a plot of a movement and a picture of 
one. A trail drawn at constant brightness tells you where something went; one 
that lights up where it moved fast tells you how.

COLOURS:
Each of the first 20 lines has its own colour, white to begin with: set 
'color index' to a line number and pick its colour with 'color_control'. 
Index -1 recolours them all. Lines past the twentieth share the twentieth's 
colour.

LIGHTING AND DEPTH:
gl_line_array switches lighting and the depth test off while it draws its 
lines, so they are never shaded or hidden, and switches them back on before 
the chain continues: shapes chained after it are lit and depth-tested as 
usual.

gl_vertex_buffer:
Vertex data drawn directly, with 'draw_mode' choosing how the vertices are 
joined. The escape hatch for geometry none of the other nodes produces. It 
draws in the chain's current colour and material, and until data arrives it 
shows 64 random points in the unit cube. Points shrink with distance from 
the viewer.
""" + OLDER + """
SYNTAX:
gl_line_array
gl_vertex_buffer

Neither takes arguments.

EXAMPLE:
gl_line_array

INPUTS and PARAMETERS:

gl chain in:
The chain. This triggers the drawing.

array (gl_line_array):
Points by lines by coordinates (2, 3 or 4 of them). Anything not three 
dimensional is not drawn: a single line still needs the middle dimension, 
points by 1 by 3. With 'accent_motion' on, an array of a new shape starts 
the movement measure afresh.

alpha_fade / line_width (gl_line_array):
The fade along each line, and the width of the lines.

accent_motion / accent_colour / accent_scale (gl_line_array):
Whether speed shows, whether as brightness or colour, and the gain on it 
(50 by default) - the movement since the last array is multiplied by this 
and capped at full brightness.

selected_joints (gl_line_array):
Line numbers to draw, separated by spaces. Empty draws them all.

color index / color_control (gl_line_array):
Per-line colours, as above.

vertex_data (gl_vertex_buffer):
An array of vertices, one per row, with 2, 3 or 4 coordinates - or a cloud 
frame from the point cloud nodes, whose 'point_cloud' entry is used. It can 
be replaced every frame.

draw_mode (gl_vertex_buffer):
GL_POINTS (the default), GL_LINES, GL_LINE_STRIP, GL_LINE_LOOP, 
GL_TRIANGLES, GL_TRIANGLE_STRIP or GL_TRIANGLE_FAN.

size (gl_vertex_buffer):
The point size, 1 by default.

OUTPUTS:

gl chain out:
The chain, continuing.

RELATED:
mgl_line_array is the newer equivalent of gl_line_array, with more control 
over the accenting. np.rolling_buffer builds trails from a stream. 
gl_orientation_disks and gl_rotation_disk, which used to share this page, 
draw orientations."""

demo = chain() + [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 540), 'w': 88, 'h': 42},
    {'key': 'nr', 'init': 'np.rand 30 4 3', 'pos': (30, 610), 'w': 140, 'h': 160,
     'props': {'min': -0.3, 'max': 0.3}},
    {'key': 'c0', 'comment': True, 'text': '30 points on each of 4 lines',
     'pos': (30, 610)},
    {'key': 'la', 'init': 'gl_line_array', 'pos': (30, 800), 'w': 220, 'h': 280},
    {'key': 'c1', 'comment': True,
     'text': 'alpha_fade: each line fades from\nits first point to its last',
     'pos': (30, 800)},
    {'key': 'vb', 'init': 'gl_vertex_buffer', 'pos': (30, 1110), 'w': 200, 'h': 120,
     'props': {'draw_mode': 'GL_POINTS', 'size': 4.0}},
    {'key': 'c2', 'comment': True,
     'text': 'on its own branch of the chain:\n64 random points until vertex_data arrives',
     'pos': (30, 1110)},
]
links = CHAIN_LINKS + [
    ('lgt', 'gl chain out', 'la', 'gl chain in'),
    ('lgt', 'gl chain out', 'vb', 'gl chain in'),
    ('lb', 'out', 'nr', ''),
    ('nr', 'random array', 'la', 'array')]
print(build('gl_line_array', 'gl_line_array - drawing data as lines and points', body,
            demo, links, demo_width=600, text_width=800, text_height=1000))

# ------------------------------------------------------- gl_orientation_disks
body = """These draw rotations as disks, so an orientation can be seen rather than read.

THE NODES:

gl_orientation_disks  a set of disks, each turned to face along its own 
                      rotation axis and sized by its angle
gl_rotation_disk      three coloured disks, one in each axis plane, sized by 
                      the parts of one quaternion

A rotation is three or four numbers and you cannot read it. A disk lying in 
the plane of the rotation, as large as the rotation is, you can: the 
direction it faces is the axis, and its size is how far it has turned.

gl_orientation_disks:
'axis-angle' takes one row per disk. A row of three is a rotation vector - 
its direction is the axis and its length the angle in radians - and the disk 
is drawn facing along that axis with a radius of the angle times 'scale'. 
A row of four is an axis followed by a size, and the radius is that size 
times 'scale'. A disk with no rotation is not drawn.

Each disk is a ring: 'ring_width' is taken off the inside of it, either as 
a distance or, with 'width is fraction' ticked, as a fraction of the radius. 
A disk smaller than the ring width is drawn solid.

All the disks are drawn at the same place - the origin of the chain's 
current coordinates - one over another. They do not place themselves at 
joints: to show a body's joints, draw one set per joint after a transform 
that moves it there, or overlay several rotations at one spot to compare 
them.

Each disk has its own colour, sent to 'color 0', 'color 1' and so on as red 
green blue alpha from 0 to 1. The colours are drawn as glowing (emissive), 
so they show the same whatever the lights do. The six default colours are 
translucent, so overlapping disks show through each other where blending 
is on.

gl_rotation_disk:
Takes a quaternion, w first, on 'quaternion in'. It draws a red disk in the 
x-y plane whose radius is the quaternion's x part, a blue one in the y-z 
plane sized by its y part, and a green one in the x-z plane sized by its z 
part, each times the scale. For a turn about a single axis only one disk 
shows, and its size is the sine of half the angle; no turn at all draws 
nothing. Its colours are set on the material, so they show only with 
lighting on; the material is put back afterwards, so the colours do not 
carry on to whatever is chained after it.
""" + OLDER + """
SYNTAX:
gl_orientation_disks <count> <scale> <slices> <rings>
gl_rotation_disk

All four arguments are optional: 6 disks, scale 0.5, 32 slices, 1 ring.

EXAMPLE:
gl_orientation_disks 3 0.25

INPUTS and PARAMETERS:

gl chain in:
The chain. This triggers the drawing.

axis-angle (gl_orientation_disks):
One row per disk, as above. One disk is drawn per row, up to the node's count; 
extra rows are ignored. A single row of three or four numbers draws one disk. 
The 'orient_scale' message, which aims a single shape, does not apply here and 
is ignored with a note in the console.

scale (gl_orientation_disks):
Radius per radian (or per unit of size, for four-number rows).

slices / rings (gl_orientation_disks):
How finely each disk is divided around and across. More slices makes a 
rounder disk.

ring_width / width is fraction (gl_orientation_disks):
How much of each disk is cut away from the middle, as above. 0.1 by default.

color 0, color 1 ... (gl_orientation_disks):
One inlet per disk, red green blue alpha.

quaternion in (gl_rotation_disk):
The orientation, w x y z.

scale (gl_rotation_disk):
Multiplies the disk sizes, 1 by default. It is read when a quaternion 
arrives, so a change shows with the next quaternion. (Older versions 
labelled this inlet 'gl chain in'.)

shading / style (options):
Smooth, flat or no normals; and fill, outline or points.

OUTPUTS:

gl chain out:
The chain, continuing.

RELATED:
mgl_orientation_disks is the newer equivalent. gl_quaternion_rotate and 
gl_axis_angle_rotate turn the chain by such rotations instead of showing 
them. gl_line_array, which used to share this page, draws motion trails."""

demo = chain() + [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 540), 'w': 88, 'h': 42},
    {'key': 'nr', 'init': 'np.rand 3 3', 'pos': (30, 610), 'w': 140, 'h': 140,
     'props': {'min': -1.5, 'max': 1.5}},
    {'key': 'c0', 'comment': True, 'text': 'three random rotation vectors,\none row per disk',
     'pos': (30, 610)},
    {'key': 'od', 'init': 'gl_orientation_disks 3 0.25', 'pos': (30, 780), 'w': 220, 'h': 280},
    {'key': 'c1', 'comment': True,
     'text': 'each disk faces along its axis,\nits radius is the angle times scale',
     'pos': (30, 780)},
    {'key': 'tr', 'init': 'gl_translate 0.5 0 0', 'pos': (30, 1090), 'w': 160, 'h': 110,
     'props': {'x': 0.5, 'y': 0.0, 'z': 0.0}},
    {'key': 'm1', 'init': 'message', 'pos': (30, 1230), 'w': 200, 'h': 42,
     'props': {'text in': '0.85 0.3 0.25 0.35', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'a quaternion, w x y z', 'pos': (30, 1230)},
    {'key': 'rd', 'init': 'gl_rotation_disk', 'pos': (30, 1300), 'w': 200, 'h': 180},
    {'key': 'c3', 'comment': True,
     'text': 'red, blue and green disks\nsized by x, y and z',
     'pos': (30, 1300)},
]
links = CHAIN_LINKS + [
    ('lgt', 'gl chain out', 'od', 'gl chain in'),
    ('lgt', 'gl chain out', 'tr', 'gl chain in'),
    ('tr', 'gl chain out', 'rd', 'gl chain in'),
    ('lb', 'out', 'nr', ''),
    ('nr', 'random array', 'od', 'axis-angle'),
    ('lb', 'out', 'm1', ''),
    ('m1', 'message out', 'rd', 'quaternion in')]
print(build('gl_orientation_disks', 'gl_orientation_disks - seeing rotations', body,
            demo, links, demo_width=600, text_width=800, text_height=1100))

# -------------------------------------------------------------------- gl_text
body = """These draw text into the scene.

THE NODES:

gl_text         text in the scene, from the Latin character set 
gl_korean_text  the same, drawing Hangul syllables

WHERE THE TEXT GOES:
The text is drawn flat, facing along the chain's current z axis, at 
'position_x' and 'position_y' in the chain's current coordinates and two units 
further away than whatever came before it. Lighting and depth testing are 
switched off while it draws, so it is never shaded and never hidden behind 
anything drawn before it.

Because the older system has no camera, text placed straight after gl_context 
(or after nodes that do not transform, like gl_light) stays fixed on the 
window: right for a title, a readout, or a state indicator. Put a 
gl_translate, gl_rotate or gl_scale before it and the text moves with them, 
like any other shape - which places a label beside the thing it names, but 
also means a rotation can turn it edge-on until it vanishes.

WHAT CAN BE SENT:
A string is drawn as it is. A list is drawn piece by piece, with 'separator' 
between pieces. A piece can also be a pair - some text and a weight - and is 
then drawn with its transparency multiplied by that weight raised to 'alpha 
power', and left out entirely when the weight is zero or less. That makes a 
list of words with confidences, say, readable at a glance.

gl_korean_text:
Builds its glyphs from the Hangul syllable block only. Letters, digits and 
spaces are not in it and are skipped, so words run together. Its default font 
has no Hangul, so it needs a Korean font named as its argument or in 'font', 
and building over eleven thousand glyphs takes a long time. Its glyph atlas 
looks to be laid out for 256 characters rather than for the Hangul range, so 
treat it as unfinished.
""" + OLDER + """
SYNTAX:
gl_text
gl_text <font size>
gl_text <font file>
gl_korean_text <font file> <font size>

A number argument is the font size, a word is the font file. Either order.

EXAMPLE:
gl_text 36

INPUTS and PARAMETERS:

gl chain in:
The chain. This triggers the drawing.

text:
What to draw: a string, or a list as above.

position_x / position_y:
Where the text starts, in the chain's current coordinates - its left end, 
with the letters hanging below that height.

alpha:
Overall transparency, 0 to 1.

scale:
The size of the letters in the scene. 1 is the default.

font:
The font file. Inconsolata-g.otf by default.

size (option):
The size the font is rendered at, 24 by default for gl_text and 6 for 
gl_korean_text. A larger size gives sharper letters and also draws them 
larger, since the drawn size is this times 'scale'.

colour (option):
The text colour. (Older versions labelled it 'alpha', the same as the 
transparency inlet; patches saved then still load their colour.)

alpha power / separator (options):
For lists, as above.

OUTPUTS:

gl chain out:
The chain, continuing. The text's own offset does not carry downstream.

RELATED:
mgl_text is the newer equivalent. gl_button_grid, which used to share this 
page, draws a grid of highlighted squares in the same way."""

demo = chain() + [
    {'key': 'la1', 'init': 'load_action fixed on the window', 'pos': (30, 516),
     'w': 300, 'h': 90},
    # 'font' saved at its exact default: restore_properties skips a no-op, so
    # font_changed never fires. (mgl_text's font callback opens a native file
    # dialog; gl_text's only reloads the font, but keep the restore a no-op.)
    {'key': 'tx', 'init': 'gl_text', 'pos': (30, 626), 'w': 260, 'h': 260,
     'props': {'font': 'Inconsolata-g.otf'}},
    {'key': 'c0', 'comment': True, 'text': 'straight after the context: no transform is\nin force, so it stays put on the window',
     'pos': (30, 901)},
    {'key': 'tr', 'init': 'gl_translate 0 -0.5 0', 'pos': (30, 951), 'w': 240, 'h': 160,
     'props': {'x': 0.0, 'y': -0.5, 'z': 0.0}},
    {'key': 'la2', 'init': 'load_action moved by the translate', 'pos': (30, 1131),
     'w': 300, 'h': 90},
    {'key': 'tx2', 'init': 'gl_text', 'pos': (30, 1241), 'w': 260, 'h': 260,
     'props': {'font': 'Inconsolata-g.otf'}},
    {'key': 'c2', 'comment': True, 'text': 'after a transform: it moves, and would\nturn, with everything else downstream',
     'pos': (30, 1516)},
]
links = CHAIN_LINKS + [
    ('lgt', 'gl chain out', 'tx', 'gl chain in'),
    ('la1', 'out', 'tx', 'text'),
    ('tx', 'gl chain out', 'tr', 'gl chain in'),
    ('tr', 'gl chain out', 'tx2', 'gl chain in'),
    ('la2', 'out', 'tx2', 'text')]
print(build('gl_text', 'gl_text - text in the scene', body, demo, links,
            demo_width=610, text_width=800, text_height=760))

# ------------------------------------------------------------- gl_button_grid
body = """gl_button_grid draws a four by four grid of squares in the scene, with one 
of them highlighted.

WHAT IT IS, AND IS NOT:
It is an indicator, not a control. It does not respond to the mouse, and it 
has no outlet but the chain.
'selection' is an INPUT - send it a number from 0 to 15 and that square is 
outlined in red while the rest stay grey. Any other number highlights none.

That makes it a way to show, inside a fullscreen render where no patch is 
visible, which of sixteen states, presets or sections is active.

THE LAYOUT:
Square 0 is at the bottom left, at 'position_x' and 'position_y'. The 
numbers run left to right along a row, then up a row, so square 15 is at the 
top right. 'scale' is the distance from one square to the next, and 'spacing' 
is the part of that distance left as a gap - 0.2 leaves a fifth. The squares 
are drawn as outlines, 'thickness' pixels wide.

WHERE IT GOES:
Like gl_text, it is drawn in the chain's current coordinates and two units 
further away. Straight after gl_context it is fixed on the window; after a 
transform it moves with it.

IT PUTS THINGS BACK:
Its two-unit offset, the lighting it switches off, and its outline and 
blending settings all apply only to the grid: what is chained after it is 
drawn as if the grid were not there.
""" + OLDER + """
SYNTAX:
gl_button_grid

EXAMPLE:
gl_button_grid

INPUTS and PARAMETERS:

gl chain in:
The chain. This triggers the drawing.

selection:
Which square to highlight, 0 to 15.

position_x / position_y:
Where the bottom left corner of the grid sits.

spacing:
The gap between squares, as a part of 'scale'. Default 0.2.

thickness:
The outline width in pixels. Default 4.

alpha:
Transparency, 0 to 1. A low alpha gives an indicator that is there without 
dominating what it sits over.

scale:
The distance from one square to the next. Default 0.1.

OUTPUTS:

gl chain out:
The chain, continuing.

RELATED:
gl_text, which used to share this page, draws text in the scene in the same 
way, and suits a label beside the grid."""

demo = chain() + [
    {'key': 'met', 'init': 'metro 500', 'pos': (30, 516), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 500.0, 'units': 'milliseconds'}},
    {'key': 'cnt', 'init': 'counter 16 1', 'pos': (30, 606), 'w': 123, 'h': 84,
     'props': {'count': 16, 'step': 1}},
    {'key': 'i1', 'init': 'int', 'pos': (30, 705), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c0', 'comment': True, 'text': 'counts 0 to 15, then wraps', 'pos': (30, 755)},
    {'key': 'bg', 'init': 'gl_button_grid', 'pos': (30, 795), 'w': 260, 'h': 280},
    {'key': 'c1', 'comment': True, 'text': 'the counted square lights up red,\nbottom left to top right',
     'pos': (30, 1090)},
]
links = CHAIN_LINKS + [
    ('lgt', 'gl chain out', 'bg', 'gl chain in'),
    ('met', '', 'cnt', 'input'),
    ('cnt', 'count out', 'i1', ''),
    ('i1', 'int out', 'bg', 'selection')]
print(build('gl_button_grid', 'gl_button_grid - a sixteen-square indicator', body, demo,
            links, demo_width=610, text_width=800, text_height=760))
