"""print, text/text_display, table, color, mouse/keys, pan_view, load_bang."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ---------------------------------------------------------------------- print
body = """The print node writes whatever it receives to the console.

It is the first thing to reach for when a patch is not doing what you expect. 
Wire it in beside whatever you are unsure of and you can see the actual values 
going past, rather than inferring them from what happens downstream.

The output goes to the terminal the patch was launched from, not into the patch 
itself. If you started it from an application icon rather than a terminal, 
you will not see anything - which is the usual reason a print appears to do 
nothing.

SYNTAX:
print <identifier>

EXAMPLE:
print incoming

INPUTS and PARAMETERS:

in:
Anything at all. It is printed as it arrives.

identifier:
A label printed before the value. Give every print node one as soon as you have 
more than a couple - unlabelled output from three prints at once is unreadable, 
and naming them costs nothing.

precision:
How many decimal places to show for floating point numbers. 
Raise it when you are trying to see whether something is really zero, 
or really constant.

end:
What is printed after each value. A newline by default. 
Set it to a space and successive values run along one line, which is much 
easier to scan when you are watching a fast stream.

OUTPUTS: 

None. This node is a dead end by design.

RELATED:
info and type report what KIND of thing something is, rather than its value - 
which is often the actual question when a node is refusing data. 
start_trace and end_trace report which nodes execute, and in what order."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'met', 'init': 'metro 500', 'pos': (30, 112), 'w': 129, 'h': 70,
     'props': {'on': False, 'period': 500.0, 'units': 'milliseconds'}},
    {'key': 'cnt', 'init': 'counter', 'pos': (30, 192), 'w': 123, 'h': 84,
     'props': {'step': 1}},
    {'key': 'pr', 'init': 'print count', 'pos': (30, 290), 'w': 170, 'h': 120,
     'props': {'identifier': 'count', 'precision': 3, 'end': '\\n'}},
    {'key': 'c0', 'comment': True, 'text': 'switch it on, then watch the console',
     'pos': (30, 425)},
    {'key': 'c1', 'comment': True, 'text': 'nothing appears in the patch itself',
     'pos': (30, 455)},
    {'key': 'c2', 'comment': True, 'text': 'always give it an identifier',
     'pos': (30, 485)},
]
links = [('tog', '', 'met', 'on'), ('met', '', 'cnt', 'input'),
         ('cnt', 'count out', 'pr', 'in')]
print(build('print', 'print - see what is actually going past', body, demo, links,
            demo_width=400, text_width=780, text_height=560))

# ----------------------------------------------------------------------- text
body = """text and text_display hold many lines of text rather than one value.

text is editable - type into it, and what you type is available to the patch. 
text_display is for output: text arriving is added to it, and it scrolls as it 
fills, which makes it the place to put a running log, a transcript, or 
anything you want to read inside the patch rather than in a terminal.

SYNTAX:
text
text_display

INPUTS and PARAMETERS:

text in:
The text. On text_display each arrival is appended; on text it replaces what 
is there.

wrap:
Whether long lines fold to fit the width or run off the edge.

height / width:
The size of the box.

max_lines (text_display):
How many lines to keep. Older ones fall off the top once it is full, 
so a long-running log cannot grow without limit.

autoscroll (text_display):
Whether the view follows new text as it arrives. On is what you want while 
watching; turn it off to read back through what has already gone by without 
being dragged to the bottom.

copy to clipboard (text_display):
Takes the whole contents, for pasting somewhere else.

font size / bind to / parameter name:
As on the other text widgets.

OUTPUTS: 

string out / list out / message out:
The contents, as a single string, as a list of words, or as a message.

A NOTE ON WRAPPING:
Only these two can wrap. The single-line string and message widgets cannot 
fold text or scroll it, so anything longer than the box simply disappears off 
the end - if you are losing text, this is usually why, and text_display is 
the fix."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'met', 'init': 'metro 700', 'pos': (30, 112), 'w': 129, 'h': 70,
     'props': {'on': False, 'period': 700.0, 'units': 'milliseconds'}},
    {'key': 'cnt', 'init': 'counter', 'pos': (30, 192), 'w': 123, 'h': 84,
     'props': {'step': 1}},
    {'key': 'pp', 'init': 'prepend line', 'pos': (30, 290), 'w': 150, 'h': 60},
    {'key': 'td', 'init': 'text_display', 'pos': (30, 370), 'w': 320, 'h': 220,
     'props': {'width': 300, 'height': 180, 'wrap': True, 'max_lines': 100,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'switch on: lines pile up and scroll',
     'pos': (30, 605)},
    {'key': 'c1', 'comment': True, 'text': 'untick autoscroll to read back',
     'pos': (30, 635)},
]
links = [('tog', '', 'met', 'on'), ('met', '', 'cnt', 'input'),
         ('cnt', 'count out', 'pp', 'in'), ('pp', 'out', 'td', '###text in')]
print(build('text', 'text and text_display - many lines, not one value', body,
            demo, links, demo_width=450, text_width=780, text_height=620))

# ---------------------------------------------------------------------- table
body = """The table node is a grid of numbers you can see and edit.

Where an array is a block of data you have to display with something else, 
a table shows the numbers themselves, laid out in rows and columns, and lets 
you change any of them by hand. It is both a readout and an input.

Use it for anything small and structured enough to want to look at directly - 
a matrix, a set of weights, a short lookup, calibration figures.

SYNTAX:
table <rows: int> <columns: int>

EXAMPLE:
table 4 4

INPUTS and PARAMETERS:

array in:
An array or list to load into the grid. Receiving one triggers the node and 
redraws the table.

set:
Change one cell without replacing everything. Send the position and the value.

get:
Ask for a cell's value.

OUTPUTS: 

out:
The table's contents.

RELATED:
array holds the same kind of data without showing it. 
heat_map displays a grid as colour rather than as numbers, which is the better 
choice once the grid is too large to read."""

demo = [
    {'key': 'tb', 'init': 'table 4 4', 'pos': (30, 62), 'w': 300, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'click a cell and type into it',
     'pos': (30, 275)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 320), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 4,
               'min y': 0.0, 'max y': 16.0, 'update_mode': 'heat_map',
               'number format': '%.0f'}},
    {'key': 'c1', 'comment': True, 'text': 'the same numbers as colour', 'pos': (30, 480)},
]
links = [('tb', 'out', 'hm', 'y')]
print(build('table', 'table - numbers you can see and edit', body, demo, links,
            demo_width=400, text_width=780, text_height=520))

# ---------------------------------------------------------------------- color
body = """These nodes pick a colour by eye rather than by number.

color opens the usual colour picker - a wheel or a square, with sliders beside 
it - and sends the result as a list. color_cmy is the subtractive equivalent, 
built from cyan, magenta and yellow for when you are thinking in inks rather 
than in light.

Picking a colour is one of the things that genuinely cannot be done well with 
three number boxes. Use these to choose, then color_convert to get it into 
whatever space and range the rest of your patch wants.

SYNTAX:
color
color_cmy

INPUTS and PARAMETERS:

color in:
A colour to set the picker to.

cyan / magenta / yellow (color_cmy):
The three components, settable individually.

hue_wheel (color):
Whether to show a wheel or a square. A matter of preference.

alpha (color):
Whether to include a transparency component, making it four numbers rather 
than three.

inputs (color):
Whether the numeric fields are shown beside the picker, for typing an exact 
value.

OUTPUTS: 

out / cmy:
The colour, as a list of components.

RELATED:
color_convert turns the result into any other space, and rescales it - 
0 to 1 for a patch, 0 to 255 for a file or a device."""

demo = [
    {'key': 'co', 'init': 'color', 'pos': (30, 62), 'w': 280, 'h': 300,
     'props': {'hue_wheel': False, 'alpha': False, 'inputs': True}},
    {'key': 'c0', 'comment': True, 'text': 'pick a colour', 'pos': (30, 375)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 415), 'w': 260, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'cc', 'init': 'rgb_to_hsl', 'pos': (30, 475), 'w': 170, 'h': 140,
     'props': {'from': 'rgb', 'to': 'hsl', 'in scale': '0-1', 'out scale': '0-1'}},
    {'key': 'l2', 'init': 'list', 'pos': (30, 630), 'w': 260, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'the same colour as hue, saturation, lightness',
     'pos': (30, 680)},
]
links = [('co', '', 'l1', ''), ('co', '', 'cc', 'in'), ('cc', 'out', 'l2', '')]
print(build('color', 'color - pick a colour by eye', body, demo, links,
            demo_width=420, text_width=780, text_height=620))

# --------------------------------------------------------------- mouse, keys
body = """These two nodes let the patch respond to the mouse and the keyboard directly.

mouse reports where the pointer is, continuously. 
keys reports which keys are down, and what was typed.

They are how a patch becomes playable without building an interface for it - 
useful for performance, for testing something quickly, and for anything where 
a widget would be in the way of what you are doing with your hands.

SYNTAX:
mouse
keys <key name> <key name> ...

EXAMPLE:
keys space a s d f

INPUTS and PARAMETERS:

in (mouse):
Anything here asks for the current position.

list keys (keys):
A button that prints every key name the node recognises, to the console. 
Click it when you are not sure what a key is called - the names are what you 
give as arguments.

Arguments to keys name the keys you want their own outlets for. 
Without arguments you still get the modifier outlets and the general 
character and code outlets.

OUTPUTS - mouse:

x / y:
The pointer position, on separate outlets.

OUTPUTS - keys:

shift / control / command / alt:
The modifier keys, each reporting 1 while held and 0 when released. 
Having them separately is what lets you use a modifier to change what an 
ordinary key does.

character out:
The character typed.

key code out:
Its numeric code, for keys that produce no character.

<one outlet per named key>:
1 while that key is held, 0 when released.

A NOTE ON FOCUS:
These read the keyboard and mouse for the application, so they pick up 
everything - including keys you press while editing a patch. Bear that in mind 
when a key is wired to something destructive, and consider gating it behind a 
toggle you can switch off while working."""

demo = [
    {'key': 'mo', 'init': 'mouse', 'pos': (30, 62), 'w': 130, 'h': 70},
    {'key': 'f1', 'init': 'float', 'pos': (30, 150), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'f2', 'init': 'float', 'pos': (190, 150), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c0', 'comment': True, 'text': 'move the mouse over the patch',
     'pos': (30, 200)},
    {'key': 'ky', 'init': 'keys space', 'pos': (30, 245), 'w': 200, 'h': 200},
    {'key': 'c1', 'comment': True, 'text': 'hold shift, or press space', 'pos': (30, 460)},
    {'key': 'i1', 'init': 'int', 'pos': (30, 500), 'w': 127, 'h': 42, 'props': INT},
    {'key': 's1', 'init': 'string', 'pos': (30, 555), 'w': 180, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'click "list keys" for every name',
     'pos': (30, 605)},
]
links = [('mo', 'x', 'f1', ''), ('mo', 'y', 'f2', ''),
         ('ky', 'shift', 'i1', ''), ('ky', 'character out', 's1', '')]
print(build('mouse', 'mouse and keys - play the patch directly', body, demo, links,
            demo_width=420, text_width=790, text_height=620))

# ------------------------------------------------------------ pan_view, home
body = """These two put navigation buttons into the patch itself.

A patch larger than the window is normally moved around by dragging the 
background. That is fine while building, and no use at all once the patch is 
presented and someone else is using it - the working canvas is not something 
you want them dragging.

These nodes give you the movement as buttons you can place and label: 
pan_view shifts the view by a fixed amount, home_view returns it to the origin. 
Set them to hide their title bars and they become plain arrows on the surface 
of a finished interface.

THE NODES:

pan_view    move the view by a set offset
home_view   return to the origin

SYNTAX:
pan_view <horizontal> <vertical>
home_view

EXAMPLE:
pan_view 400 0

INPUTS and PARAMETERS:

in:
The button. Clicking it, or sending anything here, moves the view - 
so the patch can navigate itself as well as offering the click.

h_offset / v_offset (pan_view):
How far to move, in pixels. Positive and negative give you the two directions, 
so a set of four buttons is four pan_views with different offsets.

title:
The button's label - "left", "next page", whatever the move means.

hide_title_bar:
Removes the node's frame, leaving just the button. 
This is what makes it look like part of an interface rather than part of a patch.

OUTPUTS: 

None - these act on the view rather than passing data on.

RELATED:
present switches the patch into presentation mode, where nodes set to hidden 
disappear. These two are meant to be used together: build the pages, add the 
navigation, then present it."""

demo = [
    {'key': 'pv1', 'init': 'pan_view -400 0', 'pos': (30, 62), 'w': 170, 'h': 120,
     'props': {'title': 'left', 'h_offset': -400, 'v_offset': 0}},
    {'key': 'pv2', 'init': 'pan_view 400 0', 'pos': (230, 62), 'w': 170, 'h': 120,
     'props': {'title': 'right', 'h_offset': 400, 'v_offset': 0}},
    {'key': 'hv', 'init': 'home_view', 'pos': (30, 200), 'w': 170, 'h': 90,
     'props': {'title': 'home'}},
    {'key': 'c0', 'comment': True, 'text': 'click these to move the view', 'pos': (30, 305)},
    {'key': 'c1', 'comment': True, 'text': 'home brings it back to the origin',
     'pos': (30, 335)},
    {'key': 'c2', 'comment': True, 'text': 'tick hide_title_bar for a plain button',
     'pos': (30, 365)},
]
print(build('pan_view', 'pan_view - navigation the patch can offer', body, demo, [],
            demo_width=440, text_width=780, text_height=560))

# --------------------------------------------------------- load_bang, action
body = """These fire when the patch finishes opening, so it can set itself up.

A patch that needs a metro running, a mode chosen, or a file loaded should not 
need you to click anything to get there. load_bang sends a bang once, as soon 
as the patch is ready, and whatever you hang off it happens automatically.

load_action does the same but sends a message of your choosing rather than a 
bang - so one node can put a value in place rather than merely triggering 
something that knows the value already.

THE NODES:

load_bang     send a bang when the patch opens
load_action   send a message of your choosing when the patch opens

SYNTAX:
load_bang
load_action <message>

EXAMPLE:
load_action open recordings/take_3.wav

INPUTS and PARAMETERS:

trigger:
A button, so you can fire it again by hand without reopening the patch. 
This is worth using while building: it lets you test the startup path 
immediately rather than saving and reopening each time.

loadActionString (load_action):
The message to send. Editable after the fact.

OUTPUTS: 

out:
The bang, or the message.

ORDER MATTERS AT STARTUP:
Several load_bangs in one patch fire in no order you can rely on. When one 
thing must happen before another - a file loaded before it is read, a mode set 
before something acts on it - use ONE load_bang into a repeat_in_order or a t 
node, and take the ordering from that instead."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'this fired when the patch opened',
     'pos': (30, 115)},
    {'key': 'rp', 'init': 'repeat_in_order 2', 'pos': (30, 155), 'w': 190, 'h': 70},
    {'key': 'c1', 'comment': True, 'text': 'one load_bang, ordered by repeat',
     'pos': (30, 235)},
    {'key': 'la', 'init': 'load_action hello there', 'pos': (30, 280), 'w': 220, 'h': 90,
     'props': {'loadActionString': 'hello there'}},
    {'key': 's1', 'init': 'string', 'pos': (30, 390), 'w': 220, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'a message, not just a trigger',
     'pos': (30, 440)},
    {'key': 'cnt', 'init': 'counter', 'pos': (300, 280), 'w': 123, 'h': 84,
     'props': {'step': 1}},
    {'key': 'i1', 'init': 'int', 'pos': (300, 375), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c3', 'comment': True, 'text': 'click trigger to fire it again',
     'pos': (300, 425)},
]
links = [('lb', 'out', 'rp', ''),
         ('rp', 'first', 'la', 'trigger'), ('rp', 'second', 'cnt', 'input'),
         ('la', 'out', 's1', ''), ('cnt', 'count out', 'i1', '')]
print(build('load_bang', 'load_bang - set the patch up as it opens', body, demo, links,
            demo_width=470, text_width=780, text_height=600))
