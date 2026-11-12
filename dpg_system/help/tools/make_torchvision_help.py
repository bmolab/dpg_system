"""torchvision image adjustments and transforms."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

HM = lambda n=32, lo=0.0, hi=1.0, fmt='%.2f': {
    'color': 'viridis', 'width': 200, 'height': 100, 'sample count': n,
    'min y': lo, 'max y': hi, 'update_mode': 'heat_map', 'number format': fmt}

SHARED = """
CHANNELS FIRST, AND 0 TO 1:
These want images as (channels, height, width) - (3, H, W) for colour, (1, H, W)
for grayscale. Images from cameras and files usually arrive the other way round,
(H, W, 3), so the nodes guess: a small last dimension and a large third-from-last
means height-width-channels, and they transpose it for you. The guess is right
except on very small images, where those two tests stop telling you anything.

Unlike the k. nodes, these are happy with whole-number images: hand them 0..255
integers and you get 0..255 integers back, correctly scaled. Floats are taken to
run 0 to 1.
"""

# ------------------------------------------------------- tv.adjust_brightness
body = """These are the photographic adjustments - the sliders of an image editor.

THE NODES:

tv.adjust_brightness   how light the image is
tv.adjust_contrast     how far apart light and dark are
tv.adjust_saturation   how strong the colour is
tv.adjust_sharpness    how crisp the detail is
tv.adjust_hue          which colours they are

ONE NUMBER, AND 1.0 MEANS LEAVE IT ALONE:
Four of the five take a FACTOR, and at 1.0 the image comes back untouched - not
approximately, exactly. Below 1.0 the quality is reduced, above 1.0 it is
exaggerated past where it started. There is no upper limit that means "maximum";
2.0 is twice as much as the image already had.

WHAT FACTOR 0 GIVES YOU IS THE REAL DEFINITION:
Each of these scales the image towards something, and setting the factor to 0
shows you what that something is. It is the clearest way to understand what you
are actually adjusting:

brightness 0   pure black
contrast 0     one flat grey - the image's own average, all detail gone
saturation 0   grey, but the detail intact - this is a grayscale conversion
sharpness 0    a slightly softened image

So contrast scales away from the average, saturation scales away from grey, and
brightness scales away from black. A factor above 1.0 simply keeps going in the
opposite direction, which is why it can push values off the end of the scale.

HUE IS THE ODD ONE - AN ANGLE, NOT A FACTOR:
tv.adjust_hue takes an OFFSET, and 0.0 is the one that changes nothing. It
rotates every colour around the colour wheel, so it runs -0.5 to +0.5, where a
whole turn is 1.0 and lands you back where you started.

That means -0.5 and +0.5 give exactly the same image: half a turn is half a turn
whichever way you go round. Hue does not make anything brighter or duller, it
only swaps which colour is which.

THE TRAP: THEY CLIP, AND EVEN A NO-OP CLIPS:
These assume a float image runs 0 to 1, and they enforce it. Send one that runs
0 to 3 and it comes back 0 to 1 - clipped, permanently.

This bites hardest because it happens even at factor 1.0. An adjustment set to
change nothing will still flatten everything above 1.0 in an image that was not
scaled properly, and there is no warning. If a chain mysteriously loses its
highlights, look for an adjustment node sitting at its default.

tv.adjust_hue is the exception: being a rotation, it leaves the range alone.

SYNTAX:
tv.adjust_brightness
tv.adjust_contrast
tv.adjust_saturation
tv.adjust_sharpness
tv.adjust_hue

EXAMPLE:
tv.adjust_saturation

INPUTS and PARAMETERS:

tensor in:
The image. Receiving it applies the adjustment.

brightness / contrast / saturation / sharpness:
The factor. 1.0 changes nothing, 0.0 removes the quality entirely, above 1.0
exaggerates it.

hue offset:
The rotation, -0.5 to +0.5. 0.0 changes nothing.

OUTPUTS: 

output:
The adjusted image, channels first, in the dtype you sent.
""" + SHARED + """
RELATED:
tv.Grayscale is what tv.adjust_saturation at 0 amounts to, said directly.
k.rgb_to_hls gives you hue as a channel you can measure, rather than an
adjustment you apply."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 3 32 32', 'pos': (30, 120), 'w': 190, 'h': 180},
    {'key': 's0', 'init': 't.select', 'pos': (250, 120), 'w': 170, 'h': 120},
    {'key': 'hm0', 'init': 'heat_map', 'pos': (250, 255), 'w': 208, 'h': 148,
     'props': HM(32)},
    {'key': 'c0', 'comment': True, 'text': 'the original, channel 0', 'pos': (250, 415)},

    {'key': 'br', 'init': 'tv.adjust_brightness', 'pos': (30, 460), 'w': 240, 'h': 110},
    {'key': 'mx1', 'init': 't.max', 'pos': (30, 585), 'w': 160, 'h': 110},
    {'key': 'f1', 'init': 'float', 'pos': (30, 710), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'drag brightness to 0: the brightest\npixel becomes 0 - pure black',
     'pos': (30, 760)},

    {'key': 'co', 'init': 'tv.adjust_contrast', 'pos': (330, 460), 'w': 230, 'h': 110},
    {'key': 'mn2', 'init': 't.min', 'pos': (330, 585), 'w': 160, 'h': 110},
    {'key': 'f2', 'init': 'float', 'pos': (330, 710), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'mx2', 'init': 't.max', 'pos': (500, 585), 'w': 160, 'h': 110},
    {'key': 'f3', 'init': 'float', 'pos': (500, 710), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'drag contrast to 0 and these two meet:\none flat grey, the image average',
     'pos': (330, 760)},

    {'key': 'sa', 'init': 'tv.adjust_saturation', 'pos': (700, 120), 'w': 240, 'h': 110},
    {'key': 's1', 'init': 't.select', 'pos': (700, 245), 'w': 170, 'h': 120},
    {'key': 'hm1', 'init': 'heat_map', 'pos': (700, 380), 'w': 208, 'h': 148,
     'props': HM(32)},
    {'key': 'c5', 'comment': True, 'text': 'saturation 0: still all the detail,\nbut every channel identical - grey',
     'pos': (700, 540)},

    {'key': 'hu', 'init': 'tv.adjust_hue', 'pos': (700, 615), 'w': 230, 'h': 110},
    {'key': 's2', 'init': 't.select', 'pos': (700, 740), 'w': 170, 'h': 120},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (700, 875), 'w': 208, 'h': 148,
     'props': HM(32)},
    {'key': 'c7', 'comment': True, 'text': 'hue: 0 changes nothing, and -0.5 and\n+0.5 are the same half turn',
     'pos': (700, 1035)},

    {'key': 'sh', 'init': 'tv.adjust_sharpness', 'pos': (980, 120), 'w': 240, 'h': 110},
    {'key': 's3', 'init': 't.select', 'pos': (980, 245), 'w': 170, 'h': 120},
    {'key': 'hm3', 'init': 'heat_map', 'pos': (980, 380), 'w': 208, 'h': 148,
     'props': HM(32)},
    {'key': 'c9', 'comment': True, 'text': 'sharpness 0 softens; above 1 it\nexaggerates past the original',
     'pos': (980, 540)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 's0', 'tensor in'), ('s0', 'output', 'hm0', 'y'),
         ('rnd', 'random tensor', 'br', 'tensor in'),
         ('br', 'output', 'mx1', 'tensor in'), ('mx1', '', 'f1', '', 0),
         ('rnd', 'random tensor', 'co', 'tensor in'),
         ('co', 'output', 'mn2', 'tensor in'), ('mn2', '', 'f2', '', 0),
         ('co', 'output', 'mx2', 'tensor in'), ('mx2', '', 'f3', '', 0),
         ('rnd', 'random tensor', 'sa', 'tensor in'),
         ('sa', 'output', 's1', 'tensor in'), ('s1', 'output', 'hm1', 'y'),
         ('rnd', 'random tensor', 'hu', 'tensor in'),
         ('hu', 'output', 's2', 'tensor in'), ('s2', 'output', 'hm2', 'y'),
         ('rnd', 'random tensor', 'sh', 'tensor in'),
         ('sh', 'output', 's3', 'tensor in'), ('s3', 'output', 'hm3', 'y')]
print(build('tv.adjust_brightness', 'tv adjustments - the image editor sliders', body,
            demo, links, demo_width=1240, text_width=800, text_height=780))

# ------------------------------------------------------------- tv.Grayscale
body = """Turns a colour image into a single brightness channel.

GRAYSCALE IS NOT THE AVERAGE OF THE CHANNELS:
It is a weighted sum - about 0.30 of red, 0.59 of green and 0.11 of blue -
because that is how human vision works rather than because of anything about
the numbers. Two colours that look equally bright to you end up equally bright;
averaging the three channels instead would make blues too light and greens too
dark.

Which matters whenever brightness is standing in for something perceptual. If
you want a plain average, that is a mean over the channel axis, not this.

ALREADY-GRAYSCALE INPUT IS FINE:
A one-channel image passes through unchanged. Anything else that is not three
channels - a four-channel image with alpha, for instance - is refused with a
message in the console.

IT HAS A k. TWIN:
k.rgb_to_grayscale does the same job with the same weights. Two differences are
worth knowing. tv.Grayscale keeps whole-number images as whole numbers (0..255
in, 0..255 out); k.rgb_to_grayscale converts to floats first. And
k.rgb_to_grayscale refuses a one-channel image, where tv.Grayscale passes it
through - so if a chain might be handed either kind, this is the forgiving one.

SYNTAX:
tv.Grayscale

INPUTS and PARAMETERS:

tensor in:
The image. Receiving it does the work.

OUTPUTS:

output:
The grayscale image, (1, H, W) - one channel, channels first, in the dtype you
sent.
""" + SHARED + """
RELATED:
tv.adjust_saturation at 0 gives a grey image that still has three channels;
tv.Grayscale gives one channel. Which you want depends on what comes next."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 3 32 32', 'pos': (30, 120), 'w': 190, 'h': 180},
    {'key': 'i0', 'init': 't.info', 'pos': (250, 120), 'w': 200, 'h': 150},
    {'key': 'l0', 'init': 'list', 'pos': (250, 285), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': '3 channels in', 'pos': (250, 335)},

    {'key': 'gs', 'init': 'tv.Grayscale', 'pos': (30, 380), 'w': 200, 'h': 80},
    {'key': 'i1', 'init': 't.info', 'pos': (250, 380), 'w': 200, 'h': 150},
    {'key': 'l1', 'init': 'list', 'pos': (250, 545), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': '1 channel out - a weighted sum,\nnot the average of the three',
     'pos': (250, 595)},
    {'key': 'sq', 'init': 't.squeeze', 'pos': (30, 480), 'w': 160, 'h': 80},
    {'key': 'hm0', 'init': 'heat_map', 'pos': (30, 575), 'w': 208, 'h': 148,
     'props': HM(32)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'i0', 'in'), ('i0', 'shape', 'l0', ''),
         ('rnd', 'random tensor', 'gs', 'tensor in'),
         ('gs', 'output', 'i1', 'in'), ('i1', 'shape', 'l1', ''),
         ('gs', 'output', 'sq', 'tensor in'), ('sq', 'output', 'hm0', 'y')]
print(build('tv.Grayscale', 'tv.Grayscale - colour to one brightness channel', body,
            demo, links, demo_width=500, text_width=800, text_height=640))

# --------------------------------------------------------- tv.gaussian_blur
body = """Smooths an image with a Gaussian blur.

BLUR IS A CHOICE ABOUT SCALE:
'sigma' is how far the smoothing reaches, and that is really a decision about
what counts as detail and what counts as noise. 'kernel size' is just the window
the arithmetic happens in; it needs to be comfortably wider than sigma - about
six times sigma covers the whole curve - or the blur is cut off at the edge of
its own window. Sigma is the dial you want.

Sigma is clamped above zero: set it to 0 or below and it becomes 0.1, and the
widget is corrected to show that - a blur of nothing is not a blur, and it will
not pretend otherwise.

The defaults, sigma 0.5 in a 9-wide window, are a very light blur.

IT HAS A k. TWIN:
k.gaussian_blur does the same job and offers larger windows (up to 29 against
19 here). The difference that matters: tv.gaussian_blur keeps whole-number
images as whole numbers (0..255 in, 0..255 out), while k.gaussian_blur converts
to floats first. If an image arrives from a camera or a file as integers and you
want it to stay that way, use this one. If the next step is an edge or blob
filter, you are in kornia's territory anyway, and staying in one family keeps
the assumptions consistent.

SYNTAX:
tv.gaussian_blur

INPUTS and PARAMETERS:

tensor in:
The image. Receiving it does the work. Each channel is blurred separately.

sigma:
How far the smoothing reaches, in pixels. Default 0.5.

kernel size:
The width of the window, 3 to 19, odd numbers only. Default 9.

OUTPUTS:

output:
The blurred image, the same shape, channels first, in the dtype you sent.
""" + SHARED

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 1 32 32', 'pos': (30, 120), 'w': 190, 'h': 180},
    {'key': 'sq0', 'init': 't.squeeze', 'pos': (250, 120), 'w': 160, 'h': 80},
    {'key': 'hm0', 'init': 'heat_map', 'pos': (250, 215), 'w': 208, 'h': 148,
     'props': HM(32)},
    {'key': 'c0', 'comment': True, 'text': 'noise - every pixel on its own', 'pos': (250, 375)},
    {'key': 'gb', 'init': 'tv.gaussian_blur', 'pos': (30, 420), 'w': 230, 'h': 150,
     'props': {'sigma': 2.0, 'kernel size': '13'}},
    {'key': 'sq1', 'init': 't.squeeze', 'pos': (30, 585), 'w': 160, 'h': 80},
    {'key': 'hm1', 'init': 'heat_map', 'pos': (30, 680), 'w': 208, 'h': 148,
     'props': HM(32, 0.3, 0.7)},
    {'key': 'c1', 'comment': True, 'text': 'the same picture, smoothed - sigma\ndecides what counts as detail',
     'pos': (30, 840)},
]
links = [('btn', '', 'rnd', '###input'),
         ('rnd', 'random tensor', 'sq0', 'tensor in'), ('sq0', 'output', 'hm0', 'y'),
         ('rnd', 'random tensor', 'gb', 'tensor in'),
         ('gb', 'output', 'sq1', 'tensor in'), ('sq1', 'output', 'hm1', 'y')]
print(build('tv.gaussian_blur', 'tv.gaussian_blur - smoothing an image', body,
            demo, links, demo_width=500, text_width=800, text_height=680))
