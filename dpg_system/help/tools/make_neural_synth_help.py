"""neural_synth: a picture grown from what one feature of a network responds to."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import INT

body = """neural_synth grows a picture of what one feature of an image-recognition network responds to, and sends the picture out as it grows, step by step.

It is David Rokeby's 2018 NeuralSynth, rebuilt. The network is Google's GoogLeNet ("Inception5h", the network behind DeepDream), or VGG19 as an alternative.

HOW THE PICTURE GROWS:
It starts as noise at a very small size. Each step nudges every pixel in the direction that makes the chosen feature respond more strongly. After a number of steps the picture is enlarged (by about 1.4 times) and the work continues at the larger size, then again, octave by octave, until it reaches full size. The small octaves settle the overall layout; the large ones add the detail.

Before each step's change is applied it is split into bands from coarse to fine, each band is brought to the same strength, and the bands are weighted and recombined. Colour is reduced as the picture grows, and the steps get smaller. This shaping is most of what gives the pictures their look.

WHAT COMES OUT:
'frame' sends the picture as an RGB array (height by width by 3 bytes, like cv_camera) every 'frame every' steps and at the end of each octave, always at the full output size, so the stream can go straight to mgl_image, movie_writer, or anything else that takes pictures. The development is the point: a whole image is several hundred frames.

Each frame is contrast-stretched for display (block by block, so dim regions are lifted). The very first frames of an image are faint; this is the 2018 behaviour, and 'norm min range' sets it.

Each time the picture grows to the next octave it is resampled onto a finer grid, which softens its finest detail all at once. 'octave fade' (12 steps by default, 0 to switch off) eases that change into the stream: the first frame after the change matches the last one before it, and the difference fades out over that many steps while the growth carries on. It changes only what is shown; the picture being grown, and every frame after the fade, are the same either way.

CHOOSING WHAT TO GROW:
'layer' picks a layer of the network, and 'feature' one channel in it (the count differs per layer; numbers past the end wrap round). Layer names follow the original DeepDream names: mixed4d_3x3_bottleneck_pre_relu is a 3x3 branch of block 4d, taken before its ReLU. Early layers (conv2d0, mixed3a) respond to textures and edges, middle ones (mixed4a to 4e) to patterns and parts of things, the last ones (mixed5a, 5b) to whole creatures and objects.

'octave schedule' takes a list and sets the target octave by octave. A list of numbers is features on the current layer: '139 139 139 0 0 0 0 0 0' lays the picture out with feature 139 and fills it in with feature 0. Layer names in the list switch layer from that point. The last entry carries on to the remaining octaves. 'clear' (or an empty list) goes back to the single layer and feature.

'direction' grows a mix of all the channels of a block instead of one channel: a direction from neural_atlas, chosen from a map of what the layer does with real photographs ('find volcano'), or a plain list with one weight per channel of the current layer. Touching 'layer' or 'feature' goes back to single channels.

THE STARTING POINT:
Every picture starts from noise: a random value for every pixel and colour, made at full size and shrunk to the first octave. The network sees pixels on a 0 to 255 scale, and this noise only runs from 0 to 1, so it is a nearly black start that does little more than give the growth something to break the symmetry with. 'noise amplitude' scales it (255 for noise across the whole range). 'noise spectrum' sets its grain: 0 is fine, independent grain (the 2018 noise), 1 is softer, 2 is cloudy. 'noise colour' is 1 for independent noise in each colour and 0 for the same noise in all three (grey). 'noise tint' (three numbers, red green blue) multiplies it per colour, and 'noise base' (three numbers, on the 0 to 255 scale) adds a level per colour. At their defaults the noise is exactly the 2018 noise.

'seed' picks the noise. With 'seed mode' on fixed, the same seed and target grow exactly the same picture every time (the seed also fixes the 'roll' jitter); 'at end' on repeat shows this. With 'seed mode' on random each picture gets a new seed. Each picture's seed is the last item of 'status', and a patch can send one to 'seed' to bring a picture back. 'deterministic' matters only on cuda (NVIDIA): the graphics card's arithmetic there is not exactly repeatable by default, and this makes it so, at some cost in speed (on a Mac it changes nothing; pictures repeat without it). python -m dpg_system.neural_synth.determinism_test checks whether pictures repeat on a machine.

A picture can start from an image instead. Send one to 'init' (an RGB array, such as a cv_camera frame or a loaded picture; 'clear' goes back to noise). It is shrunk to the first octave's size, which keeps only its large shapes and colours: that is the layout the growth then builds on. 'init blur' blurs it further (in pixels of that small size). 'init noise' mixes the noise in: 0 is the image alone, 1 the noise alone. 'mask' (a grey image, 0 to 1 or 0 to 255) says where the noise goes: white takes the full 'init noise', black keeps the image. With the default noise, mixing toward it means fading toward black, which leaves those areas to the network; raise 'noise amplitude' to mix in visible noise instead. 'init centre' (on by default) subtracts the image's average brightness, since the grown pictures sit around 0 rather than mid-grey; 'init level' scales it (1 gives about the same contrast as a grown picture).

'start octave' begins the picture at a later octave, skipping the smallest ones. With an 'init' image this keeps more of the image (a larger size means less blurring); with noise, it starts the growth at a finer scale. Octaves are counted from 0, and a schedule's entries still belong to the same octaves whether or not they are skipped.

WHEN A CHANGE TAKES EFFECT:
A new layer, feature or schedule is held until 'apply' lets it in: 'next image' (the default) waits for the current picture to finish, 'next octave' switches when the picture next grows, and 'next step' switches at once, in the middle of an octave. Changing mid-image grows one feature's picture out of another's. 'restart' starts a new picture now. While the synth is holding a finished picture, a new target starts the next one on the following frame, so several messages sent together (a layer, a feature and a seed) all belong to the same picture.

WHAT HAPPENS AT THE END OF A PICTURE:
'at end' chooses: 'hold' keeps the finished picture (a new target then starts a new one), 'repeat' grows the same target again, and 'next feature' moves on to the next feature of the layer and keeps going - an endless walk through the layer, which is how the 2018 movies were made. 'done' bangs as each picture finishes.

To choose each next picture with a patch of your own, set 'at end' to hold and link 'done' to the patch, which then sends the next target (and seed, or init image). Just before 'done', 'target' sends what each octave of the finished picture grew, in the form 'octave schedule' takes: a list of features, one per octave, with a layer name wherever the layer changes (and at the start, if it is not the current 'layer'). The patch can change any entry and send the list back to 'octave schedule'.

SPEED:
The work runs on the graphics processor ('device': mps on a Mac, cuda on an NVIDIA machine) in its own thread, so the patch stays responsive. On an M1 Max a 1920 by 1080 picture with the 2018 settings (13 octaves, 200 steps in the first) takes about 35 seconds and about 1500 steps. Rendering a frame every step adds about a quarter to that; every 5 steps adds about 4 percent, every 25 nothing measurable. With 'drop frames' off (the default) the synth waits for each frame to be sent, so a recording gets every step; it then cannot run faster than the patch's frame rate. Turn 'drop frames' on to let it run free and send only the latest frame.

SIZE AND LENGTH:
'width' and 'height' are the output size, 'octaves' how many times the picture grows on the way there, 'iterations' the steps in the first octave, and 'iteration factor' how much fewer each later octave takes (1.1: about 10 percent fewer each time). These take effect with the next picture. This page uses 640 by 360, 9 octaves and 100 steps so a picture takes about 12 seconds.

THE SHAPING, FOR EXPERIMENT (options):
'objective': 'sum' grows the feature's total response (2018 default), 'square' favours its strongest spots, 'all' grows the whole layer at once. 'direction cossim' sets how strongly a direction's picture is pushed to match the direction rather than just respond to it (1 by default, 0 for not at all). 'roll' jitters the picture by up to this fraction of its height each step, which avoids grid artefacts. 'octave scaler' is how much the step shrinks per octave. 'sat elbow' and 'base sat' set how colour is reduced as the picture grows. 'band weights start' and 'band weights end' (five numbers, coarse to fine) weight the bands, moving from start to end over the octaves when 'lerp band weights' is on. 'desaturate': 'row mean' is what the 2018 code actually did (it averaged along image rows, not across colour), 'luma' is a true grey. 'dual path' builds the displayed picture with a second set of 'out' shaping settings while the first keeps driving the growth. 'norm scale', 'norm smoothing', 'norm gamma' and 'norm min range' control the display contrast, and 'octave fade' the easing across octave changes.

COLOUR:
'norm saturation' sets how colourful the frames are: 0.5 by default, 1 for the full colour of the 2018 pictures, 0 for black and white. It changes only what is shown and recorded, not the picture being grown, and takes effect at once. The display's contrast stretch pushes the weakest colour in each region toward black, which on its own makes the colours purer; this is the control for that. 'sat elbow', 'base sat' and 'desaturate' do something different: they set how coloured each step's change is, octave by octave. The first octaves, which lay down the large areas of colour, run at full colour whatever they say, so they change the texture and layout of the picture more than how colourful it looks. 'lrn size' is a setting inside the network (the 2018 synth used 11).

THE NETWORK'S WEIGHTS:
Inception5h's weights live in dpg_system/neural_synth/inception5h.pt, converted from the 2018 NeuralSynth weights folder by dpg_system/neural_synth/convert_inception5h.py. VGG19's are downloaded by torchvision (548 MB) the first time it is chosen.

SYNTAX:
neural_synth [inception5h | vgg19]

EXAMPLE:
neural_synth

INPUTS and PARAMETERS:

run:
Start or pause.

restart:
Start a new picture now, with the latest target.

layer:
The layer to grow from.

feature:
The channel in that layer.

octave schedule:
A list of features (and layer names) for successive octaves; 'clear' to stop.

direction:
A direction to grow instead of one feature, from neural_atlas.

apply:
When a new target takes effect: next image, next octave, next step.

at end:
hold, repeat or next feature.

seed:
The seed of the starting noise (with 'seed mode' fixed).

init:
An image to start from instead of noise; 'clear' to go back.

mask:
Where the noise is mixed into the 'init' image.

OUTPUTS:

frame:
The picture as it grows, as an RGB array.

status:
[image, octave, octaves, step in octave, steps in octave, step in image, steps in image, layer, feature, width, height, seed] with each frame.

target:
What each octave of the finished picture grew, as a schedule, just before done.

done:
A bang as each picture finishes.

RELATED:
neural_atlas to choose directions by what they look like. movie_writer to record the stream. mgl_image or mgl_texture to show it."""

demo = [
    # choose the target
    {'key': 'c0', 'comment': True, 'text': 'pick a feature: it starts with\nthe next picture, or press restart',
     'pos': (30, 62)},
    {'key': 'feat', 'init': 'int', 'pos': (30, 115), 'w': 127, 'h': 42,
     'props': dict(INT, **{'': 139})},
    {'key': 'c1', 'comment': True, 'text': 'a schedule: lay out with 139,\nfill in with 0',
     'pos': (30, 185)},
    {'key': 'm1', 'init': 'message', 'pos': (30, 238), 'w': 260, 'h': 42,
     'props': {'text in': '139 139 139 0 0 0 0 0 0', 'font size': '24'}},
    {'key': 'm2', 'init': 'message', 'pos': (30, 290), 'w': 80, 'h': 42,
     'props': {'text in': 'clear', 'font size': '24'}},

    # the synth
    {'key': 'c2', 'comment': True, 'text': 'the classic DeepDream layer;\nfeature 139 grows flowers',
     'pos': (330, 62)},
    {'key': 'ns', 'init': 'neural_synth', 'pos': (330, 115), 'w': 330, 'h': 260,
     'props': {'run': True, 'layer': 'mixed4d_3x3_bottleneck_pre_relu', 'feature': 139,
               'apply': 'next image', 'at end': 'next feature',
               'width': 640, 'height': 360, 'octaves': 9, 'iterations': 100,
               'frame every': 2}},

    # watching it
    {'key': 'c3', 'comment': True, 'text': 'the picture as it grows', 'pos': (700, 62)},
    {'key': 'ctx', 'init': 'mgl_context', 'pos': (700, 115), 'w': 500, 'h': 290,
     'props': {'auto_render': True, 'width': 640, 'height': 360,
               'display_width': 480, 'display_height': 270}},
    {'key': 'img', 'init': 'mgl_image', 'pos': (700, 440), 'w': 130, 'h': 80},
    {'key': 'c4', 'comment': True, 'text': 'status: image, octave, step ... layer, feature',
     'pos': (700, 540)},
    {'key': 'st', 'init': 'list', 'pos': (700, 585), 'w': 500, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c5', 'comment': True, 'text': 'done: each finished picture', 'pos': (700, 650)},
    {'key': 'b1', 'init': 'button', 'pos': (700, 695), 'w': 46, 'h': 42},
]
links = [('feat', '', 'ns', 'feature'),
         ('m1', '', 'ns', 'octave schedule'),
         ('m2', '', 'ns', 'octave schedule'),
         ('ctx', 'mgl_chain', 'img', 'mgl chain in'),
         ('ns', 'frame', 'img', 'texture'),
         ('ns', 'status', 'st', ''),
         ('ns', 'done', 'b1', '')]
print(build('neural_synth', 'neural_synth - a picture grown from inside a network', body,
            demo, links, demo_width=1240, text_width=800, text_height=1600))
