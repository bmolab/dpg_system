"""Rendering text as an image."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# --------------------------------------------------------------- cairo_layout
body = """cairo_layout sets text as a picture, for a screen or projector rather than the patch.

WHY NOT text_display:
text_display is for you, while you work. This is for an audience. It renders
through Cairo into an image, so the result goes to a projector, a texture, or
anything else that takes a picture - in a font you chose, at a size you chose.

The 'layout' outlet is that image: 1920 by 1080 unless the node is given
another size (cairo_layout <width> <height>), three channels, as floats from
0 to 1 - laid out height-first like a camera frame rather than channels-first.
So it goes straight into the same places a camera picture does. The text is
white on black.

TWO INLETS, TWO JOBS:
'input' REPLACES: each text that arrives clears the picture and sets the new
text. 'command' builds up and edits:

add <words>        append words to what is there
add_char <c>       append one character, with no space before it
delete_line        remove back to the last line break
clear / reset      empty it
scroll_up / scroll_down   move active_line by one
save               write the text to llama_output_<date>.txt in the
                   folder the app was started from

Inside the text, \\n is a line break, a word \\c clears the picture and a word
\\d deletes back to the last line break.

'active_line' IS WHERE THE TEXT ENDS:
The last line of the text sits on line 'active_line' of the picture (counting
lines of 'leading' from the top), and everything earlier is above it - lines
pushed off the top are not drawn. Set it low and you see only the end of a
long text; set it near the bottom and the text fills the picture as it grows.

A new node uses line 1, whatever the box shows, until active_line is changed
- so a long text shows only its last line, at the top. The demo sets it to 8.
Changing active_line also moves what is already shown too far; the next text
lays out correctly.

THE TYPE:
'font path' takes a font file. Empty uses the Inconsolata font that comes with
the app; a path that cannot be loaded falls back to Cairo's own default, so a
missing font never stops the node working - it just looks wrong, which is
easier to diagnose than a crash.

'leading' is the distance between lines in pixels - line spacing, not letter
spacing. Changing 'font size' resets it to one and a half times the size, so
set the size first.

'wrap text' folds long lines to the width instead of letting them run off.

BRIGHTNESS PER WORD:
A word can arrive with a weight from 0 to 1, as a [word, weight] pair in a
list. Its grey level is the weight raised to 'alpha power', times
'brightness'. Plain words have weight 1, so for them only 'brightness'
matters. 'alpha power' shapes the falloff: at 1 a weight of 0.1 is a dark
grey, at the default 0.1 it is nearly as bright as the rest. A word of
weight 0 is left out.

SYNTAX:
cairo_layout <width> <height>

EXAMPLE:
cairo_layout

INPUTS and PARAMETERS:

input:
Text to show, replacing what was there.

command:
The commands above.

clear:
Empty it.

font path / font size:
The typeface and its size (40).

brightness / alpha power:
The grey level of the text, and how word weights map to it.

leading:
Line spacing, 60.

active_line:
The line the text ends on.

wrap text:
Fold long lines to the width.

OUTPUTS:

layout:
The rendered image.

RELATED:
llm_layout, the same idea built for a language model's output; it used to
share this page. text_display to read text inside the patch instead.
mgl_texture or gl nodes to get the image onto a screen."""

demo = [
    {'key': 'src', 'init': 'string', 'pos': (30, 62), 'w': 520, 'h': 42,
     'props': {'text in': 'the room was colder than it had been',
               'font size': '24', 'width': 480}},
    {'key': 'c0', 'comment': True, 'text': 'edit and press return: replaces the text',
     'pos': (30, 112)},
    {'key': 'm1', 'init': 'message', 'pos': (30, 150), 'w': 320, 'h': 42,
     'props': {'text in': 'add and the light had gone', 'font size': '24'}},
    {'key': 'm2', 'init': 'message', 'pos': (370, 150), 'w': 80, 'h': 42,
     'props': {'text in': 'clear', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'commands build up the text',
     'pos': (30, 200)},
    {'key': 'cl', 'init': 'cairo_layout', 'pos': (30, 250), 'w': 320, 'h': 340,
     'props': {'active_line': 8}},
    {'key': 'c2', 'comment': True, 'text': 'active_line 8: the text ends on the\neighth line of the picture',
     'pos': (30, 605)},

    {'key': 'inf', 'init': 'info', 'pos': (420, 250), 'w': 260, 'h': 80},
    {'key': 'c4', 'comment': True, 'text': 'the layout is an IMAGE - send it to a\ntexture, a projector, anything that\ntakes a picture',
     'pos': (420, 345)},
]
links = [('src', 'string out', 'cl', 'input'),
         ('m1', '', 'cl', 'command'), ('m2', '', 'cl', 'command'),
         ('cl', 'layout', 'inf', 'in')]
print(build('cairo_layout', 'cairo_layout - text as a picture', body,
            demo, links, demo_width=800, text_width=800, text_height=900))

# ----------------------------------------------------------------- llm_layout
body = """llm_layout shows a language model writing, as a picture, token by token.

IT TAKES COMMANDS, NOT TEXT:
Wire gemma_4's 'layout_out' to its 'input' (qwen_moe and bonsai_2 have the
same outlet). That outlet does not send text - it sends COMMANDS, and this
node is what understands them:

add <id> <token> <value>    another token, coloured by the value
prompt <id> <token>         a token of the prompt, in white
streaming_prompt / accept_streamed_prompt / backspace_streaming_prompt
                            a prompt being typed in live
choice_list / choose        the alternatives the model weighed, and which
                            one was picked
step_back <n>               take back the last n tokens
temperature / show_probs    display settings changing
scroll_up / scroll_down     move active_line by one
clear / reset               empty it
save                        write the text to llama_output_<date>.txt in
                            the folder the app was started from

So the two nodes together give you a screen that shows the model writing,
including the parts a plain text display cannot show: which alternatives it
was considering, and which one was chosen.

The picture is 1920 by 1080, three channels, as floats from 0 to 1, laid out
height-first like a camera frame, in the Inconsolata font.

'active_line' IS WHY IT READS WELL:
New text always appears on the same line of the screen - line 17 by default -
and everything above scrolls up as it fills. The writing happens at a fixed
height rather than creeping down the page, which is far easier to read and to
point a camera at.

When the model's 'show_probs' is switched on, the node moves it up to line 5,
so there is room below for the list of alternatives, and back to 17 when it is
switched off. That is worth knowing so it does not look like a fault when the
text jumps.

'colour_mode' SHOWS THE MODEL'S UNCERTAINTY:
Each 'add' carries a value, and the token is coloured by it - so the text
carries how sure the model was as it wrote. Which value is sent is set on the
model ('display_mode' on gemma_4): its temperature, the entropy, or the
probability of the token. 'colour_mode' here picks the colour scale to read it
with, and should match.

'include prompt' decides whether 'prompt' tokens - what you asked - are shown
along with the answer.

SYNTAX:
llm_layout

EXAMPLE:
llm_layout

INPUTS and PARAMETERS:

input:
The layout commands.

clear:
Empty it.

active_line:
Which line of the screen new text appears on, 17.

colour_mode:
temperature, entropy or probability - the scale for each token's value.

include prompt:
Show the prompt as well as the answer.

OUTPUTS:

layout:
The rendered image.

RELATED:
gemma_4, qwen_moe and bonsai_2, whose 'layout_out' this is built to receive.
cairo_layout for plain text as a picture; llm_layout used to share its page.
mgl_texture or gl nodes to get the image onto a screen."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'c0', 'comment': True, 'text': 'switch on and WAIT - the model loads',
     'pos': (90, 70)},
    {'key': 'pr', 'init': 'string', 'pos': (30, 120), 'w': 520, 'h': 42,
     'props': {'text in': 'Describe a room nobody has entered for a year.',
               'font size': '24', 'width': 480}},
    {'key': 'c1', 'comment': True, 'text': 'then edit the prompt and press return',
     'pos': (30, 170)},
    {'key': 'gm', 'init': 'gemma_4', 'pos': (30, 215), 'w': 340, 'h': 500,
     'props': {'n_ctx': '8192'}},
    {'key': 'm1', 'init': 'message', 'pos': (420, 215), 'w': 200, 'h': 42,
     'props': {'text in': 'add 0 hello 1.0', 'font size': '24'}},
    {'key': 'm2', 'init': 'message', 'pos': (640, 215), 'w': 80, 'h': 42,
     'props': {'text in': 'clear', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'the same commands by hand -\nno model needed',
     'pos': (420, 265)},
    {'key': 'll', 'init': 'llm_layout', 'pos': (420, 330), 'w': 320, 'h': 260},
    {'key': 'c7', 'comment': True, 'text': 'layout_out sends COMMANDS, and this node\nunderstands them: tokens, the alternatives\nweighed, and how sure the model was',
     'pos': (420, 605)},
    {'key': 'inf', 'init': 'info', 'pos': (420, 690), 'w': 260, 'h': 80},
    {'key': 'c8', 'comment': True, 'text': 'the layout is an image', 'pos': (420, 785)},
]
links = [('tog', '', 'gm', 'on / off'),
         ('pr', 'string out', 'gm', 'prompt'),
         ('gm', 'layout_out', 'll', 'input'),
         ('m1', '', 'll', 'input'), ('m2', '', 'll', 'input'),
         ('ll', 'layout', 'inf', 'in')]
print(build('llm_layout', 'llm_layout - a language model writing, as a picture', body,
            demo, links, demo_width=800, text_width=800, text_height=1000))
