"""Rendering text as an image."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """These set text as an image, for putting on a screen rather than reading in the patch.

THE NODES:

cairo_layout   general typesetting - font, size, leading, wrapping
llm_layout     the same, purpose-built for a language model's output

WHY NOT text_display:
text_display is for you, while you work. These are for an audience. They render
through Cairo into an image, so the result goes to a projector, a texture, or
anything else that takes a picture - with a font you chose, at a size you chose,
laid out how you want it.

The 'layout' outlet is that image: 1920 by 1080, three channels, as floats -
a full HD frame, laid out height-first like a camera frame rather than
channels-first. So it goes straight into the same places a camera picture does,
and the k. and tv. filters will guess its orientation correctly.

cairo_layout IS THE PLAIN ONE:
Send it text and it sets it. 'font path' takes a font file; leave it empty and
it falls back to a bundled monospace, and then to Cairo's own default, so a
missing font never stops the node working - it just looks wrong, which is easier
to diagnose than a crash.

'leading' is the distance between lines, in the typographer's sense - line
spacing, not letter spacing. It is the setting that does most for how the text
feels, and the one worth adjusting first.

'brightness' and 'alpha power' control how the text sits against what is behind
it. Alpha power shapes the falloff of the transparency rather than scaling it,
which is what you want when text is being composited over an image and a plain
opacity setting makes it either invisible or a solid block.

'wrap text' folds long lines to the width instead of letting them run off.

llm_layout IS BUILT FOR TEXT ARRIVING A TOKEN AT A TIME:
Wire gemma_4's 'layout_out' to its 'input'. That outlet does not send text - it
sends COMMANDS, and this node is what understands them:

add                    another token
prompt / streaming_prompt   the prompt, shown differently from the answer
choose / choice_list   which alternative was picked, and what the others were
step_back              undo a token
temperature / show_probs    display settings changing
save                   write the text out

So the two nodes together give you a screen that shows the model writing,
including the parts a plain text display cannot show: which alternatives it was
considering, and which one was chosen.

'active_line' IS WHY IT READS WELL:
New text always appears on the same line of the screen - line 17 by default -
and everything above scrolls up as it fills. The writing happens at a fixed
height rather than creeping down the page, which is far easier to read and to
point a camera at.

The node moves it up to line 5 while a streaming prompt is being entered, so
there is room below for the answer, and back again afterwards. That is worth
knowing so it does not look like a fault when the text jumps.

'colour_mode' SHOWS THE MODEL'S UNCERTAINTY:
Each token can be coloured by temperature, entropy or probability - so the text
carries how sure the model was as it wrote. Confident passages and uncertain
ones look different, which is a reading of the generation you cannot get from
the words alone.

'include prompt' decides whether what you asked is shown along with the answer.

SYNTAX:
cairo_layout
llm_layout

EXAMPLE:
cairo_layout

INPUTS and PARAMETERS:

input:
Text, for cairo_layout. Layout commands, for llm_layout.

clear:
Empty it.

font path / font size:
The typeface and its size. Empty falls back to a bundled font.

leading:
Line spacing. The setting that matters most.

brightness / alpha power:
How the text sits over what is behind it.

wrap text:
Fold long lines to the width.

active_line:
Which line of the screen new text appears on.

colour_mode (llm_layout):
Colour each token by temperature, entropy or probability.

include prompt:
Show the prompt as well as the answer.

OUTPUTS: 

layout:
The rendered image.

RELATED:
gemma_4, whose 'layout_out' this is built to receive.
text_display to read text inside the patch instead.
mgl_texture or gl nodes to get the image onto a screen."""

demo = [
    {'key': 'src', 'init': 'string', 'pos': (30, 62), 'w': 520, 'h': 42,
     'props': {'text in': 'the room was colder than it had been',
               'font size': '24', 'width': 480}},
    {'key': 'cl', 'init': 'cairo_layout', 'pos': (30, 120), 'w': 320, 'h': 340},
    {'key': 'c0', 'comment': True, 'text': 'leading is line spacing - the setting\nthat does most for how it feels\nempty font path falls back to a bundled\none - a missing file never stops it',
     'pos': (30, 475)},

    {'key': 'inf', 'init': 'info', 'pos': (400, 120), 'w': 260, 'h': 80},
    {'key': 'c4', 'comment': True, 'text': 'the layout is an IMAGE - send it to a\ntexture, a projector, anything that\ntakes a picture',
     'pos': (400, 215)},

    {'key': 'gm', 'init': 'gemma_4', 'pos': (30, 630), 'w': 340, 'h': 500},
    {'key': 'll', 'init': 'llm_layout', 'pos': (420, 630), 'w': 320, 'h': 260},
    {'key': 'c7', 'comment': True, 'text': 'layout_out does not send TEXT - it sends\ncommands, and this node understands them\nso the screen can show which alternatives\nthe model weighed, not just what it said\ncolour_mode paints each token by how\nsure the model was as it wrote it',
     'pos': (420, 905)},
    {'key': 'c13', 'comment': True, 'text': 'new text stays on active_line, and the\nrest scrolls up past it - the writing\nhappens at a fixed height on screen',
     'pos': (30, 1145)},
]
links = [('src', 'string out', 'cl', 'input'),
         ('cl', 'layout', 'inf', 'in'),
         ('gm', 'layout_out', 'll', 'input')]
print(build('cairo_layout', 'cairo_layout - text as a picture', body,
            demo, links, demo_width=800, text_width=800, text_height=760))
