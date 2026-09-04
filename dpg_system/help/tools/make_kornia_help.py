"""kornia image operations: colour conversion and filters."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

HM = lambda n=32, lo=0.0, hi=1.0, fmt='%.2f': {
    'color': 'viridis', 'width': 200, 'height': 100, 'sample count': n,
    'min y': lo, 'max y': hi, 'update_mode': 'heat_map', 'number format': fmt}

LAYOUT = """
THE SHAPE THEY EXPECT - CHANNELS FIRST:
kornia works in CHANNELS-FIRST order: (channels, height, width). A 3-channel
colour image is (3, H, W), a grayscale one is (1, H, W).

Images from cameras and image files usually arrive the other way round,
(H, W, 3) - height, width, channels - so these nodes guess. If the last
dimension is small (3 or 4, the size of a pixel) and the third-from-last is
large, they take it as height-width-channels and transpose it for you.

That guess is right nearly always and wrong in one case: a very small image,
where 'small last dimension' and 'large third-from-last' stop being different.
Put a t.permute in front if you are working at those sizes and the result looks
transposed.

FLOATS, IN THE RANGE 0 TO 1:
Give these nodes float images scaled 0 to 1, not 0 to 255 integers. They convert
integers for you rather than failing, but the range is yours to get right - an
image still scaled 0..255 will come out saturated, because every filter and
every colour formula here assumes 1.0 means full brightness.
"""

# --------------------------------------------------------- k.rgb_to_grayscale
body = """These convert an image from one colour representation to another.

THE NODES:

k.rgb_to_grayscale   colour to a single brightness channel
k.rgb_to_hls         red/green/blue to hue, lightness, saturation
k.apply_colormap     a single channel back to false colour
""" + LAYOUT + """
WATCH THE CHANNEL COUNT, IT IS THE WHOLE STORY:
Each of these changes how many channels the image has, and that is the clearest
way to see what they do:

k.rgb_to_grayscale   3 channels in, 1 out    - colour discarded
k.rgb_to_hls         3 channels in, 3 out    - same information, different axes
k.apply_colormap     1 channel in,  3 out    - colour invented for display

Feeding an image that is ALREADY grayscale to k.rgb_to_grayscale is an error,
not a no-op: it wants three channels and says so. If a chain might be handed
either, convert once at the top and keep track.

WHY HLS RATHER THAN RGB:
In red/green/blue, "how bright" and "what colour" are mixed through all three
numbers - you cannot change one without disturbing the other. HLS separates
them: hue is which colour, lightness is how bright, saturation is how strong.

So "find everything reddish regardless of lighting" is awkward in RGB and easy
in HLS - it is a range on one channel. Hue is an ANGLE, though, and wraps: red
sits at both ends of the scale, so a plain range test cuts red in half.

k.apply_colormap IS FOR LOOKING, NOT FOR MEASURING:
It takes a one-channel image and paints it in false colour so your eye can read
it - the same idea as the heat_map node. Handed a 3-channel image it converts to
grayscale first, so it always has one channel to work from.

The palette is fixed, and it is a display step: the numbers it produces are
colours, not measurements, and nothing downstream should treat them as data.

SYNTAX:
k.rgb_to_grayscale
k.rgb_to_hls
k.apply_colormap

EXAMPLE:
k.rgb_to_grayscale

INPUTS and PARAMETERS:

tensor in:
The image. Receiving it does the conversion.

OUTPUTS: 

output:
The converted image, channels first.

RELATED:
color_convert does the same conversions one colour at a time rather than over a
whole image.
k.sobel and the other filters expect the kind of single-channel image
k.rgb_to_grayscale produces."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 3 32 32', 'pos': (30, 120), 'w': 190, 'h': 180},
    {'key': 'c0', 'comment': True, 'text': 'a 3 channel image, 32 by 32',
     'pos': (30, 310)},
    {'key': 'i0', 'init': 't.info', 'pos': (250, 120), 'w': 200, 'h': 150},
    {'key': 'l0', 'init': 'list', 'pos': (250, 285), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},

    {'key': 'gs', 'init': 'k.rgb_to_grayscale', 'pos': (30, 355), 'w': 230, 'h': 80},
    {'key': 'i1', 'init': 't.info', 'pos': (290, 355), 'w': 200, 'h': 150},
    {'key': 'l1', 'init': 'list', 'pos': (290, 520), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': '3 channels in, 1 out - colour gone',
     'pos': (30, 450)},

    {'key': 'sq', 'init': 't.squeeze', 'pos': (30, 495), 'w': 160, 'h': 80},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 590), 'w': 208, 'h': 148,
     'props': HM(32)},
    {'key': 'c2', 'comment': True, 'text': 'squeeze the 1 away to look at it',
     'pos': (30, 750)},

    {'key': 'cm', 'init': 'k.apply_colormap', 'pos': (520, 400), 'w': 220, 'h': 80},
    {'key': 'i2', 'init': 't.info', 'pos': (520, 505), 'w': 200, 'h': 150},
    {'key': 'l2', 'init': 'list', 'pos': (520, 670), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': '1 channel back out to 3 - false colour',
     'pos': (520, 720)},

    {'key': 'hls', 'init': 'k.rgb_to_hls', 'pos': (520, 120), 'w': 200, 'h': 80},
    {'key': 'i3', 'init': 't.info', 'pos': (520, 215), 'w': 200, 'h': 150},
    {'key': 'c4', 'comment': True, 'text': '3 in, 3 out - same information,\nlaid out on different axes',
     'pos': (760, 215)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'i0', 'in'), ('i0', 'shape', 'l0', ''),
         ('rnd', 'random tensor', 'gs', 'tensor in'),
         ('gs', 'output', 'i1', 'in'), ('i1', 'shape', 'l1', ''),
         ('gs', 'output', 'sq', 'tensor in'), ('sq', 'output', 'hm', 'y'),
         ('gs', 'output', 'cm', 'tensor in'),
         ('cm', 'output', 'i2', 'in'), ('i2', 'shape', 'l2', ''),
         ('rnd', 'random tensor', 'hls', 'tensor in'),
         ('hls', 'output', 'i3', 'in')]
print(build('k.rgb_to_grayscale', 'k colour - grayscale, HLS, false colour', body,
            demo, links, demo_width=1000, text_width=800, text_height=740))

# ------------------------------------------------------------------- k.sobel
body = """These are the image filters: smoothing, and finding where things change.

THE NODES:

k.gaussian_blur         smooth the image
k.sobel                 how fast brightness changes, and which way
k.canny                 thinned, thresholded edges
k.dog_response_single   blob response at one scale
""" + LAYOUT + """
BLUR FIRST, THEN LOOK FOR EDGES:
This is the order that matters. Every edge filter is asking "how different is
this pixel from its neighbours", and noise is exactly that - a pixel different
from its neighbours. Run an edge filter on a noisy image and you get an answer
dominated by the noise.

k.gaussian_blur first, then k.sobel, is the standard arrangement, and the amount
of blur is really a choice about SCALE: it sets how big a thing has to be before
it counts as an edge rather than as texture.

'sigma' is that choice - how far the smoothing reaches. 'kernel size' is only
the window the maths is done in, and it has to be comfortably wider than sigma
or the blur is cut off at the edges of its own window. Sigma is the dial you
want; leave kernel size alone until sigma gets large.

SOBEL AND CANNY ANSWER DIFFERENT QUESTIONS:
k.sobel gives a CONTINUOUS answer: at every pixel, how steeply brightness is
changing. Strong edges give big numbers, gentle shading gives small ones, and
nothing is thrown away. Multi-channel images are filtered channel by channel.

k.canny gives a DECISION: it takes the same gradients, thins them to lines a
single pixel wide, and applies a threshold, so what comes out is essentially a
map of edge or not-edge. That is what you want to count or trace shapes; it is
the wrong thing to feed anything that wants a magnitude, because the strength
has been thresholded away.

Reach for sobel when the answer is a quantity, canny when the answer is a set of
lines.

A threshold cuts both ways, though. Because canny decides, it can decide there
is nothing there: blur an image hard enough and the gradients all fall under the
threshold, and canny returns an empty map while sobel still reports small
non-zero numbers everywhere. An empty canny result usually means too much blur
upstream, not an image without edges.

k.dog_response_single IS A BLOB DETECTOR:
It subtracts one blurred copy of the image from another blurred slightly more.
What survives is detail at the scale BETWEEN the two blurs - finer detail is in
both copies and cancels, coarser detail is in both copies and cancels.

So the two sigmas are a band: they pick a size of thing to respond to. The
defaults are close together (1.0 and 1.6), which is the classic ratio and a
narrow band. Push them apart and it answers to a wider range of sizes.

SYNTAX:
k.gaussian_blur
k.sobel
k.canny
k.dog_response_single <sigma_1> <sigma_2>

EXAMPLE:
k.dog_response_single 1.0 1.6

INPUTS and PARAMETERS:

tensor in:
The image. Receiving it runs the filter.

kernel size / sigma (k.gaussian_blur):
How far the smoothing reaches, and the window it is computed in. Sigma is
clamped above zero - a blur of nothing is not a blur.

sigma_1 / sigma_2 (k.dog_response_single):
The two scales whose difference is the answer.

OUTPUTS: 

output:
The filtered image, channels first, as floats.

RELATED:
k.rgb_to_grayscale first if you want one answer per pixel rather than one per
colour channel."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 1 32 32', 'pos': (30, 120), 'w': 190, 'h': 180},
    {'key': 'sq0', 'init': 't.squeeze', 'pos': (250, 120), 'w': 160, 'h': 80},
    {'key': 'hm0', 'init': 'heat_map', 'pos': (250, 215), 'w': 208, 'h': 148,
     'props': HM(32)},
    {'key': 'c0', 'comment': True, 'text': 'pure noise: every pixel unlike its\nneighbours - the worst case for an\nedge filter',
     'pos': (30, 310)},

    {'key': 'gb', 'init': 'k.gaussian_blur', 'pos': (30, 415), 'w': 220, 'h': 150,
     'props': {'sigma': 1.5, 'kernel size': '15'}},
    {'key': 'sq1', 'init': 't.squeeze', 'pos': (280, 415), 'w': 160, 'h': 80},
    {'key': 'hm1', 'init': 'heat_map', 'pos': (280, 510), 'w': 208, 'h': 148,
     'props': HM(32, 0.2, 0.8)},
    {'key': 'c3', 'comment': True, 'text': 'blur first - now there are blobs\nbig enough to have edges',
     'pos': (30, 580)},

    {'key': 'sb', 'init': 'k.sobel', 'pos': (30, 655), 'w': 190, 'h': 80},
    {'key': 'sq2', 'init': 't.squeeze', 'pos': (250, 700), 'w': 160, 'h': 80},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (250, 795), 'w': 208, 'h': 148,
     'props': HM(32, 0.0, 0.10)},
    {'key': 'c5', 'comment': True, 'text': 'sobel: a continuous steepness',
     'pos': (30, 750)},

    {'key': 'cn', 'init': 'k.canny', 'pos': (520, 415), 'w': 190, 'h': 80},
    {'key': 'sq3', 'init': 't.squeeze', 'pos': (520, 510), 'w': 160, 'h': 80},
    {'key': 'hm3', 'init': 'heat_map', 'pos': (520, 605), 'w': 208, 'h': 148,
     'props': HM(32, 0.0, 1.0, '%.0f')},
    {'key': 'c6', 'comment': True, 'text': 'canny: a decision - edge or not',
     'pos': (520, 765)},

    {'key': 'dg', 'init': 'k.dog_response_single 1.0 1.6', 'pos': (770, 120),
     'w': 260, 'h': 150},
    {'key': 'sq4', 'init': 't.squeeze', 'pos': (770, 285), 'w': 160, 'h': 80},
    {'key': 'hm4', 'init': 'heat_map', 'pos': (770, 380), 'w': 208, 'h': 148,
     'props': HM(32, -0.05, 0.05)},
    {'key': 'c7', 'comment': True, 'text': 'the two sigmas are a band: detail\nbetween them survives, the rest cancels',
     'pos': (770, 540)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'sq0', 'tensor in'), ('sq0', 'output', 'hm0', 'y'),
         ('rnd', 'random tensor', 'gb', 'tensor in'),
         ('gb', 'output', 'sq1', 'tensor in'), ('sq1', 'output', 'hm1', 'y'),
         ('gb', 'output', 'sb', 'tensor in'),
         ('sb', 'output', 'sq2', 'tensor in'), ('sq2', 'output', 'hm2', 'y'),
         ('gb', 'output', 'cn', 'tensor in'),
         ('cn', 'output', 'sq3', 'tensor in'), ('sq3', 'output', 'hm3', 'y'),
         ('gb', 'output', 'dg', 'tensor in'),
         ('dg', 'output', 'sq4', 'tensor in'), ('sq4', 'output', 'hm4', 'y')]
print(build('k.sobel', 'k filters - blur, edges, blobs', body,
            demo, links, demo_width=1060, text_width=800, text_height=760))
