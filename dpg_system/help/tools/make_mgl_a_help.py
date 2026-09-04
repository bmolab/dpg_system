"""mgl_context (the framework), transforms, camera and appearance."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ---------------------------------------------------------------- mgl_context
body = """mgl_context is the root of a 3D scene. Everything drawn hangs off its chain.

HOW THE CHAIN WORKS:

mgl_context has an "mgl_chain" outlet. Every other mgl node has an 
"mgl chain in" inlet and an "mgl chain out" outlet, and you wire them in a line. 
When a frame is rendered the context sends the word "draw" along that line, 
and each node in turn draws itself and passes the message on.

The line LOOKS flat and is not. What each node actually does is:

  save the current state
  draw itself
  send "draw" onward - so everything downstream draws now
  restore the state

Because the downstream draw happens BETWEEN the save and the restore, anything 
a node changes applies to everything after it in the chain, and is undone at 
the end. That is what makes a chain of transforms nest the way a scene graph 
does, without any branching syntax: a translate followed by three shapes moves 
all three, and a translate after those three does not move them.

To make a branch, split the chain cord to two destinations. Each branch is 
drawn inside whatever transforms preceded the split, and neither affects the 
other.

GETTING SOMETHING ON SCREEN:

    mgl_context  ->  mgl_camera  ->  mgl_light  ->  mgl_box
         |
    texture_tag  ->  mgl_display

The context renders into a texture; mgl_display puts that texture in a window. 
They are separate so that the scene can be rendered once and shown in several 
places, or shown at a size unrelated to the resolution it was drawn at.

THE NODES:

mgl_context  the root: holds the GL context, renders the chain
mgl_display  shows the rendered texture in a window
mgl_enable   turn a GL flag on or off for the rest of the chain

SYNTAX:
mgl_context
mgl_display

EXAMPLE:
mgl_context

INPUTS and PARAMETERS:

auto_render (mgl_context):
Render every frame, continuously. Leave it off and nothing is drawn until 
something bangs "render".

render (mgl_context):
Draw one frame now.

texture_tag (mgl_display):
The texture to show - patch it from the context's texture_tag outlet.

width / height / fullscreen (mgl_display):
The window.

enabled / flag (mgl_enable):
Which GL state to change and whether to switch it on. Like everything else in 
the chain, it applies from that point onward and is restored afterwards.

OUTPUTS: 

mgl_chain (mgl_context):
The chain. Patch it into the first thing you want drawn.

texture_tag:
The rendered image, for mgl_display.

ui:
Interaction events from the window, for the orbit camera.

WHY GL WORK IS MAIN-THREAD ONLY:
A pose or a point cloud arriving on a streaming thread can trigger a node's 
execute at the same moment the chain does. Running GL from that thread has no 
context current and crashes inside the driver, so every mgl node refuses to 
draw off the main thread and leaves the message for the chain's own trigger a 
moment later. Nothing is lost; it is worth knowing because it means data can 
arrive as fast as it likes without any risk to the render."""

demo = [
    {'key': 'ctx', 'init': 'mgl_context', 'pos': (30, 62), 'w': 220, 'h': 120,
     'props': {'auto_render': True}},
    {'key': 'c0', 'comment': True, 'text': 'tick auto_render to draw every frame',
     'pos': (30, 195)},
    {'key': 'cam', 'init': 'mgl_camera', 'pos': (30, 240), 'w': 220, 'h': 180},
    {'key': 'lgt', 'init': 'mgl_light', 'pos': (30, 435), 'w': 220, 'h': 200},
    {'key': 'rot', 'init': 'mgl_rotate', 'pos': (30, 675), 'w': 220, 'h': 120},
    {'key': 'box', 'init': 'mgl_box', 'pos': (30, 810), 'w': 220, 'h': 200},
    {'key': 'c1', 'comment': True, 'text': 'the rotate applies to everything after it',
     'pos': (30, 1025)},
    {'key': 'dsp', 'init': 'mgl_display', 'pos': (300, 240), 'w': 220, 'h': 160},
    {'key': 'c2', 'comment': True, 'text': 'the context renders to a texture;\ndisplay puts it in a window',
     'pos': (300, 415)},
]
links = [('ctx', 'mgl_chain', 'cam', 'mgl chain in'),
         ('cam', 'mgl chain out', 'lgt', 'mgl chain in'),
         ('lgt', 'mgl chain out', 'rot', 'mgl chain in'),
         ('rot', 'mgl chain out', 'box', 'mgl chain in'),
         ('ctx', 'texture_tag', 'dsp', 'texture_tag')]
print(build('mgl_context', 'mgl_context - the root of a 3D scene', body, demo, links,
            demo_width=550, text_width=820, text_height=820))

# -------------------------------------------------------------- mgl_transform
body = """These move, turn and resize everything that comes AFTER them in the chain.

A transform node does not draw anything. It changes where the drawing happens, 
for every node downstream of it, and then puts things back as they were. 
So the position of a transform in the chain is what decides its scope - 
everything between it and the end of that branch is affected.

Two shapes drawn in different places is one chain with two transforms in it:

    mgl_translate -> mgl_box -> mgl_translate -> mgl_sphere

The sphere's position is the sum of both translates, because it is downstream 
of both. That is how a scene graph accumulates, and it is why hierarchies - 
an arm on a body, a hand on the arm - fall out of the chain order without any 
extra syntax.

THE NODES:

mgl_transform          translate, rotate and scale in one node
mgl_translate          move
mgl_rotate             turn, as three angles
mgl_quaternion_rotate  turn, given a quaternion
mgl_axis_angle_rotate  turn, given an axis and an angle around it
mgl_scale              resize
mgl_billboard          turn whatever follows to face the camera, whatever the 
                       camera does

WHICH ROTATION NODE:
Three angles are easy to read and to set by hand, and they gimbal-lock - two 
axes line up at certain orientations and you lose a degree of freedom. 
For anything driven by real orientation data, use mgl_quaternion_rotate: the 
sensors produce quaternions, and converting them to angles to feed mgl_rotate 
throws away exactly the property that makes them well behaved.

mgl_axis_angle_rotate is the one to use when the rotation has a natural axis - 
a joint's own axis, a torque direction, a spin.

mgl_billboard IS FOR THINGS THAT MUST STAY READABLE:
Text and flat markers turn edge-on and disappear as the camera moves. 
Putting a billboard before them keeps them facing the viewer whatever the 
orbit is doing.

SYNTAX:
mgl_translate <x> <y> <z>
mgl_rotate <x> <y> <z>
mgl_scale <s>

EXAMPLE:
mgl_translate 0 1 0

INPUTS and PARAMETERS:

mgl chain in:
The chain. This is what triggers the node.

translate / rotate / scale:
The amounts. All of them accept a stream, so a transform can be driven 
continuously from anywhere in the patch.

rotation vector (mgl_axis_angle_rotate):
The axis, with the angle carried in its length or given alongside.

OUTPUTS: 

mgl chain out:
The chain, continuing - with the transform in force for everything downstream.

ORDER MATTERS, AND NOT THE WAY IT READS:
Rotate then translate is not the same as translate then rotate. The first turns 
the object on the spot and then moves it; the second moves it away and then 
swings it around the origin, which is an orbit. If something is orbiting when 
you wanted it to spin, that is the two the wrong way round."""

demo = [
    {'key': 'ctx', 'init': 'mgl_context', 'pos': (30, 62), 'w': 220, 'h': 120,
     'props': {'auto_render': True}},
    {'key': 'cam', 'init': 'mgl_camera', 'pos': (30, 200), 'w': 220, 'h': 180},
    {'key': 'lgt', 'init': 'mgl_light', 'pos': (30, 395), 'w': 220, 'h': 200},
    {'key': 'tr1', 'init': 'mgl_translate -1 0 0', 'pos': (30, 635), 'w': 220, 'h': 120},
    {'key': 'box', 'init': 'mgl_box', 'pos': (30, 770), 'w': 220, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'the box is moved by the first translate',
     'pos': (30, 985)},
    {'key': 'tr2', 'init': 'mgl_translate 2 0 0', 'pos': (30, 1025), 'w': 220, 'h': 120},
    {'key': 'sph', 'init': 'mgl_sphere', 'pos': (30, 1160), 'w': 220, 'h': 200},
    {'key': 'c1', 'comment': True, 'text': 'the sphere by BOTH: they accumulate',
     'pos': (30, 1375)},
    {'key': 'dsp', 'init': 'mgl_display', 'pos': (300, 200), 'w': 220, 'h': 160},
]
links = [('ctx', 'mgl_chain', 'cam', 'mgl chain in'),
         ('cam', 'mgl chain out', 'lgt', 'mgl chain in'),
         ('lgt', 'mgl chain out', 'tr1', 'mgl chain in'),
         ('tr1', 'mgl chain out', 'box', 'mgl chain in'),
         ('box', 'mgl chain out', 'tr2', 'mgl chain in'),
         ('tr2', 'mgl chain out', 'sph', 'mgl chain in'),
         ('ctx', 'texture_tag', 'dsp', 'texture_tag')]
print(build('mgl_transform', 'mgl transforms - moving what comes after', body, demo,
            links, demo_width=550, text_width=820, text_height=820))

# ----------------------------------------------------------------- mgl_camera
body = """These decide where the scene is seen from, how it is lit, and what it is made of.

Like everything in the chain they apply from where they sit onward - so a 
material placed before three shapes gives all three that material, and a second 
material after them changes only what follows.

THE NODES:

mgl_camera        a fixed viewpoint: a position, a target and a field of view
mgl_orbit_camera  a viewpoint you can drag - it takes the window's interaction 
                  events and turns them into yaw, elevation and distance
mgl_light         a light source
mgl_material      how surfaces respond to light
mgl_color         a flat colour, ignoring lighting
mgl_texture       an image mapped onto surfaces
mgl_image         an image drawn directly

A SCENE NEEDS A CAMERA AND A LIGHT:
Without a camera there is no viewpoint and nothing appears. Without a light, 
anything using a material is black. Those two are the usual reason a chain that 
looks correct renders an empty window - the shapes are being drawn, and there 
is nothing to see them by.

mgl_color IS NOT mgl_material:
A colour is applied flat, so the shape comes out an even patch with no shading 
and no sense of form. A material responds to lights, so the same shape shows its 
curvature. Use colour for markers, lines and anything schematic; use material 
for anything meant to look like an object.

THE MATERIAL COMPONENTS:
'ambient' is the colour in shadow, 'diffuse' the colour in plain light, 
'specular' the colour of the highlight, and 'shininess' how tight that highlight 
is. High shininess is a small hard glint; low is a broad sheen. The light node 
has the matching three, and what you see is the product of the two.

SYNTAX:
mgl_camera
mgl_orbit_camera

EXAMPLE:
mgl_orbit_camera

INPUTS and PARAMETERS:

pos / target / up (mgl_camera):
Where the camera is, what it looks at, and which way is up.

fov:
The field of view. Wide values exaggerate depth; narrow ones flatten it.

target / distance / yaw / elevation (mgl_orbit_camera):
What to orbit around, and where on the orbit to be.

ui (mgl_orbit_camera):
Patch this from the context's 'ui' outlet and dragging in the window moves 
the camera.

position / ambient / diffuse / specular / intensity (mgl_light):
Where the light is and what it contributes.

ambient / diffuse / specular / shininess (mgl_material):
How the surface answers.

texture (mgl_texture, and the texture inlet on every shape):
The image to map.

OUTPUTS: 

mgl chain out:
The chain, with the setting in force downstream."""

demo = [
    {'key': 'ctx', 'init': 'mgl_context', 'pos': (30, 62), 'w': 220, 'h': 120,
     'props': {'auto_render': True}},
    {'key': 'cam', 'init': 'mgl_orbit_camera', 'pos': (30, 200), 'w': 240, 'h': 220},
    {'key': 'c0', 'comment': True, 'text': 'drag in the window to orbit', 'pos': (30, 435)},
    {'key': 'lgt', 'init': 'mgl_light', 'pos': (30, 475), 'w': 220, 'h': 200},
    {'key': 'mat', 'init': 'mgl_material', 'pos': (30, 715), 'w': 220, 'h': 180},
    {'key': 'sph', 'init': 'mgl_sphere', 'pos': (30, 941), 'w': 220, 'h': 200},
    {'key': 'c1', 'comment': True, 'text': 'material: it shows its curvature',
     'pos': (30, 1156)},
    {'key': 'col', 'init': 'mgl_color', 'pos': (300, 715), 'w': 220, 'h': 120},
    {'key': 'bx', 'init': 'mgl_box', 'pos': (300, 850), 'w': 220, 'h': 200},
    {'key': 'c2', 'comment': True, 'text': 'colour: flat, no shading', 'pos': (300, 1096)},
    {'key': 'dsp', 'init': 'mgl_display', 'pos': (320, 200), 'w': 220, 'h': 160},
]
links = [('ctx', 'mgl_chain', 'cam', 'mgl chain in'),
         ('ctx', 'ui', 'cam', 'ui'),
         ('cam', 'mgl chain out', 'lgt', 'mgl chain in'),
         ('lgt', 'mgl chain out', 'mat', 'mgl chain in'),
         ('mat', 'mgl chain out', 'sph', 'mgl chain in'),
         ('sph', 'mgl chain out', 'col', 'mgl chain in'),
         ('col', 'mgl chain out', 'bx', 'mgl chain in'),
         ('ctx', 'texture_tag', 'dsp', 'texture_tag')]
print(build('mgl_camera', 'mgl camera, light and material - seeing the scene', body,
            demo, links, demo_width=560, text_width=810, text_height=800))
