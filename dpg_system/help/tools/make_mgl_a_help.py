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
body = """These decide where the scene is seen from, how it is lit, and what colour 
and surface it has.

THE NODES:

mgl_camera        a fixed viewpoint: a position, a target and a field of view
mgl_orbit_camera  a viewpoint you can drag - it takes the context's interaction 
                  events and turns them into yaw, elevation and distance
mgl_light         a light source
mgl_material      how surfaces respond to light
mgl_color         a colour that tints whatever follows

YOU DO NOT NEED EITHER TO SEE SOMETHING:
mgl_context supplies a default camera (at 0 0 3, looking at the origin) and 
a default light, both switchable in its options ('default_camera', 
'default_light'). A camera node replaces the default viewpoint for 
everything drawn after it, and the first mgl_light in a frame removes the 
default light, so the lights you place are the only ones.

WHAT IS UNDONE AT THE END OF A BRANCH, AND WHAT IS NOT:
mgl_material and mgl_color behave like transforms: they apply from where they 
sit onward and are put back afterwards, so a material placed before three 
shapes gives all three that material, and a second material after them 
changes only what follows.

Cameras and lights are not put back. A camera sets the view for the rest of 
the frame, including branches drawn after its own. Each mgl_light adds a 
light that stays on for the rest of the frame - up to eight are used - 
so shapes drawn before a light do not see it, and everything drawn after it 
does. A light's position is moved by any transforms in force where it sits.

COLOUR AND MATERIAL MULTIPLY:
Shapes are always lit. What you see is roughly the light that reaches the 
surface (its ambient plus diffuse, scaled by the material's ambient and 
diffuse) multiplied by the current colour, plus a highlight from the 
material's specular, which the colour does not tint. So mgl_color sets the 
hue and keeps the shading; mgl_material sets how the surface answers light. 
With no light at all, a shape shows a dim fifth of its colour.
mgl_line_array and mgl_text are the exceptions: they draw in the current 
colour without any lighting.

THE MATERIAL COMPONENTS:
'ambient' is the colour in shadow, 'diffuse' the colour in plain light, 
'specular' the colour of the highlight, and 'shininess' how tight that 
highlight is. High shininess is a small hard glint; low is a broad sheen. 
The light node has the matching three, and each pair multiplies.

SYNTAX:
mgl_camera
mgl_orbit_camera
mgl_light
mgl_material
mgl_color

EXAMPLE:
mgl_orbit_camera

INPUTS and PARAMETERS:

mgl chain in (all five):
The chain. This is what triggers the node.

fov (both cameras):
The vertical field of view in degrees, 60 by default. Wide values exaggerate 
depth; narrow ones flatten it.

pos / target / up (mgl_camera):
Where the camera is (0 0 3), what it looks at (the origin), and which way 
is up (0 1 0).

near / far (options, both cameras):
The nearest and farthest distances drawn, 0.1 and 100.

target / distance / yaw / elevation (mgl_orbit_camera):
What to orbit around, how far away (3), and where on the orbit: yaw is the 
angle around the vertical in degrees - 0 puts the camera on the plus z side - 
and elevation the angle above the horizontal (20).

ui (mgl_orbit_camera):
Patch this from the context's 'ui' outlet. Dragging with the left button 
orbits, the scroll wheel zooms. The events come from the context's own 
picture - its window or fullscreen view, or the picture inside the node 
when the context's 'node_mouse_events' option is ticked, as here.

top / bottom / front / back / left / right (messages, mgl_orbit_camera):
Snap to that view, keeping target and distance. 'front' looks at a subject 
facing minus z from in front of it.

projection / drag_speed / zoom_speed (options, mgl_orbit_camera):
Perspective or orthographic, and how fast dragging and scrolling move it.

position / ambient / diffuse / specular / intensity (mgl_light):
Where the light is (0 5 5) and what it contributes; intensity scales its 
diffuse and highlight, not its ambient.

ambient / diffuse / specular / shininess (mgl_material):
How the surface answers. Shininess runs from 1 to 256, 32 by default.

color (mgl_color):
The colour, as red green blue alpha, either from 0 to 1 or from 0 to 255 - 
a colour with any value above 1 is read as 0 to 255. Three numbers take 
alpha as full.

OUTPUTS:

mgl chain out:
The chain, with the setting in force downstream.

RELATED:
mgl_context for the default camera and light. mgl_texture puts an image on 
a surface; it used to share this page. The gl_ equivalents are gl_light, 
gl_material and gl_color."""

demo = [
    {'key': 'ctx', 'init': 'mgl_context', 'pos': (30, 62), 'w': 340, 'h': 340,
     'props': {'auto_render': True, 'node_display_width': 320,
               'node_display_height': 240, 'node_mouse_events': True}},
    {'key': 'c0', 'comment': True,
     'text': 'drag on the picture to orbit,\nscroll to zoom', 'pos': (30, 120)},
    {'key': 'cam', 'init': 'mgl_orbit_camera', 'pos': (30, 440), 'w': 240, 'h': 160},
    {'key': 'c1', 'comment': True, 'text': "fed from the context's ui outlet",
     'pos': (30, 440)},
    {'key': 'lgt', 'init': 'mgl_light', 'pos': (30, 630), 'w': 220, 'h': 240},
    {'key': 'c2', 'comment': True, 'text': "replaces the context's default light",
     'pos': (30, 630)},
    {'key': 'mat', 'init': 'mgl_material', 'pos': (30, 900), 'w': 220, 'h': 230},
    {'key': 'tr1', 'init': 'mgl_translate -0.8 0 0', 'pos': (30, 1160), 'w': 160, 'h': 110,
     'props': {'x': -0.8, 'y': 0.0, 'z': 0.0}},
    {'key': 'sph', 'init': 'mgl_sphere', 'pos': (30, 1300), 'w': 200, 'h': 180},
    {'key': 'c3', 'comment': True, 'text': 'material only: white, lit', 'pos': (30, 1300)},
    {'key': 'tr2', 'init': 'mgl_translate 1.6 0 0', 'pos': (30, 1510), 'w': 160, 'h': 110,
     'props': {'x': 1.6, 'y': 0.0, 'z': 0.0}},
    {'key': 'col', 'init': 'mgl_color', 'pos': (30, 1650), 'w': 220, 'h': 260,
     'props': {'color': [255.0, 140.0, 0.0, 255.0]}},
    {'key': 'c4', 'comment': True,
     'text': 'colour tints what follows,\nand the shading stays', 'pos': (30, 1650)},
    {'key': 'bx', 'init': 'mgl_box', 'pos': (30, 1940), 'w': 200, 'h': 150},
]
links = [('ctx', 'mgl_chain', 'cam', 'mgl chain in'),
         ('ctx', 'ui', 'cam', 'ui'),
         ('cam', 'mgl chain out', 'lgt', 'mgl chain in'),
         ('lgt', 'mgl chain out', 'mat', 'mgl chain in'),
         ('mat', 'mgl chain out', 'tr1', 'mgl chain in'),
         ('tr1', 'mgl chain out', 'sph', 'mgl chain in'),
         ('sph', 'mgl chain out', 'tr2', 'mgl chain in'),
         ('tr2', 'mgl chain out', 'col', 'mgl chain in'),
         ('col', 'mgl chain out', 'bx', 'mgl chain in')]
print(build('mgl_camera', 'mgl camera, light, material and colour - seeing the scene',
            body, demo, links, demo_width=560, text_width=810, text_height=1100))

# ---------------------------------------------------------------- mgl_texture
body = """These put an image into the scene: onto the surface of a shape, or across 
the whole picture.

THE NODES:

mgl_texture  turns an array into a texture on the graphics card, and passes 
             that texture on to whatever shapes it is patched into
mgl_image    draws an image filling the whole picture, as a backdrop

AN IMAGE IS AN ARRAY:
Both take an image as an array of height by width by channels - 3 channels 
for red green blue, 4 to add alpha - such as a frame from a camera node, a 
picture loaded from a file, or anything computed in numpy or torch. 

Both also take a single-channel (height by width) array, and accept 
floats from 0 to 1 as well as bytes from 0 to 255. A torch tensor is copied 
to the cpu either way.

WHY mgl_texture, WHEN EVERY SHAPE HAS A TEXTURE INLET:
Every mgl shape has a 'texture' inlet, and an array patched straight into it 
works. But the shape then uploads that array to the graphics card again on 
every frame it draws, and two shapes showing the same image upload it twice. 
mgl_texture uploads once per new array, and hands out the 
finished texture - which also goes straight into mgl_image's 'texture' 
inlet.

mgl_texture does not draw anything and changes nothing in the chain: it 
passes 'draw' straight on, and each time it runs it sends its texture out of 
the 'texture' outlet. Its place in the chain decides when that happens - 
put it before the shapes that use it.

WHEN THE ARRAY REACHES THE GRAPHICS CARD:
mgl_texture only stores an array when it arrives, and uploads it when 
'draw' next passes through, so the array can come from anywhere - a 
load_bang, a camera thread, or, as here, the context's own chain banging the 
generator for a fresh image every frame. The texture first appears at the 
first draw after the first array.

mgl_image IS A BACKDROP:
It covers the whole picture whatever the camera is doing, and draws with the 
depth test off, so it hides everything drawn before it and nothing drawn 
after it. Put it first in the chain to put a picture behind the scene.

SYNTAX:
mgl_texture
mgl_image

EXAMPLE:
mgl_texture

INPUTS and PARAMETERS:

mgl chain in:
The chain. mgl_image draws when 'draw' arrives; mgl_texture passes it on.

source (mgl_texture):
The array to turn into a texture: height by width (by 1, 3 or 4), bytes 
or floats from 0 to 1. A list is converted to bytes. A new size or channel 
count makes a new texture.

texture (mgl_image):
The image to draw: a texture from mgl_texture, or an array as above.

OUTPUTS:

mgl chain out:
The chain, continuing.

texture (mgl_texture):
The texture, for a shape's 'texture' inlet or mgl_image's. Sent each time 
the node runs, once there is one.

RELATED:
mgl_box, mgl_sphere and the other shapes take the texture. mgl_model can 
lay a projection of the image around a loaded model ('uv_mode'). 
mgl_camera covers camera, light, material and colour, and used to share 
this page."""

demo = [
    {'key': 'ctx', 'init': 'mgl_context', 'pos': (30, 62), 'w': 340, 'h': 340,
     'props': {'auto_render': True, 'node_display_width': 320,
               'node_display_height': 240}},
    {'key': 'img', 'init': 'mgl_image', 'pos': (30, 440), 'w': 130, 'h': 80},
    {'key': 'c0', 'comment': True, 'text': 'the backdrop: an array drawn across\nthe whole picture',
     'pos': (30, 440)},
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 550), 'w': 88, 'h': 42},
    {'key': 'nr1', 'init': 'np.rand 24 32 3', 'pos': (30, 620), 'w': 140, 'h': 140,
     'props': {'min': 0.0, 'max': 0.4}},
    {'key': 'c1', 'comment': True, 'text': 'float noise, 0 to 0.4: mgl_image\nscales floats itself',
     'pos': (30, 620)},
    {'key': 'tex', 'init': 'mgl_texture', 'pos': (30, 800), 'w': 130, 'h': 80},
    {'key': 'c2', 'comment': True, 'text': 'uploads each new array once and\nsends the texture to the box',
     'pos': (30, 800)},
    {'key': 'nr2', 'init': 'np.rand 4 4 3', 'pos': (30, 910), 'w': 140, 'h': 160,
     'props': {'dtype': 'uint8', 'min': 0.0, 'max': 255.0}},
    {'key': 'c3', 'comment': True, 'text': "bytes, banged by the chain's own draw:\na new image every frame",
     'pos': (30, 910)},
    {'key': 'rot', 'init': 'mgl_rotate 30 40 0', 'pos': (30, 1100), 'w': 160, 'h': 110,
     'props': {'x': 30.0, 'y': 40.0, 'z': 0.0}},
    {'key': 'bx', 'init': 'mgl_box', 'pos': (30, 1240), 'w': 200, 'h': 150},
]
links = [('ctx', 'mgl_chain', 'img', 'mgl chain in'),
         ('ctx', 'mgl_chain', 'nr2', ''),
         ('lb', 'out', 'nr1', ''),
         ('nr1', 'random array', 'img', 'texture'),
         ('img', 'mgl chain out', 'tex', 'mgl chain in'),
         ('nr2', 'random array', 'tex', 'source'),
         ('tex', 'mgl chain out', 'rot', 'mgl chain in'),
         ('tex', 'texture', 'bx', 'texture'),
         ('rot', 'mgl chain out', 'bx', 'mgl chain in')]
print(build('mgl_texture', 'mgl_texture - an image on a surface', body, demo, links,
            demo_width=560, text_width=810, text_height=1000))
