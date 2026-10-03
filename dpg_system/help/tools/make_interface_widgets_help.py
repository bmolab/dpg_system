"""float, int, slider, knob, the param_ widgets, button, toggle."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

WIDGET_OPTIONS = """
COMMON OPTIONS:

min / max:
The limits. A number typed or dragged beyond them is held at the limit. 
Leaving both at 0 means no limit at all.

format:
How the number is displayed, as a printf pattern - "%.3f" for three decimal 
places, "%d" for a whole number. This changes only what you SEE; 
the value itself keeps its full precision.

speed_property:
How far the value moves per pixel of drag. Lower is finer. 
Worth turning down whenever you find yourself unable to land on a value.

width / font size:
The size of the widget and its text.

bind to:
The name of a variable. Once bound, the widget and the variable are the same 
thing - move the widget and the variable changes, set the variable and the 
widget moves. This is how one control drives several distant parts of a patch 
without a cord. See the var help patch.

hide_title_bar:
Draws the widget alone, without the node's title bar and frame. Every 
interface node has it - for a panel of controls that is meant to be looked 
at rather than patched.
"""

# ---------------------------------------------------------------------- float
body = """float is a number box: you can read the value, and you can change it by hand.

It is the plainest interface in the system, and it does three jobs at once. 
A number arriving at the inlet is displayed, so it works as a readout. 
Dragging or typing in it sends a number, so it works as a control. 
And it holds the value between times, so the patch can ask for it later.

float keeps decimals. For a count, an index or a channel - anything where a 
fraction would be meaningless - use int instead.

TO USE IT:
Drag left and right on the number to change it. Double-click, or click and type, 
to enter one exactly. A bang at the inlet re-sends the current value without 
changing it - which is how you ask a number box what it is holding.

SYNTAX:
float
float <value>
float +

EXAMPLE:
float 0.5

A number argument is the value the box starts with. '+' draws it as a typing 
box with step buttons, instead of a box you drag.

INPUTS and PARAMETERS:

in:
The value to display and store. Receiving a number here sets the box and sends 
it on. Receiving a bang re-sends whatever is already there.
""" + WIDGET_OPTIONS + """
OUTPUTS: 

float out:
The value, sent whenever it changes - whether that was you dragging it or a 
number arriving at the inlet.

RELATED:
int is the whole-number version. 
slider and knob are the same value with a different way of setting it. 
message and list hold text rather than numbers."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 4.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 4.0)},
    {'key': 'f1', 'init': 'float', 'pos': (30, 232), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c0', 'comment': True, 'text': 'as a readout: it shows what arrives',
     'pos': (30, 282)},
    {'key': 'f2', 'init': 'float 0.5', 'pos': (30, 330), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'mul', 'init': '* 100.0', 'pos': (30, 390), 'w': 130, 'h': 70,
     'props': {'operand': 100.0}},
    {'key': 'f3', 'init': 'float', 'pos': (30, 480), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'as a control: drag the 0.5 box',
     'pos': (30, 530)},
    {'key': 'btn', 'init': 'button', 'pos': (200, 330), 'w': 88, 'h': 46},
    {'key': 'c3', 'comment': True, 'text': 'bang it to re-send without changing',
     'pos': (200, 385)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'f1', ''), ('btn', '', 'f2', ''),
         ('f2', 'float out', 'mul', 'in'), ('mul', 'result', 'f3', '')]
print(build('float', 'float - read a number, or set one', body, demo, links,
            demo_width=430, text_width=800, text_height=700))

# ------------------------------------------------------------------------ int
body = """int is a number box for whole numbers.

It does what float does - it shows what arrives, sends what you drag, and keeps 
the value for later - but it only ever holds a whole number. Use it wherever a 
fraction would be meaningless: a count, an index, a channel, a step in a list. 
It stops nonsense arriving downstream rather than leaving the next node to tidy 
it up.

A FRACTION IS DROPPED, NOT ROUNDED:
2.7 arriving at an int becomes 2, and -2.7 becomes -2. If you want the nearest 
whole number, send the value through round first.

TO USE IT:
Drag left and right on the number to change it. Double-click, or click and type, 
to enter one exactly. A bang at the inlet re-sends the current value.

SYNTAX:
int
int <max>
int +

EXAMPLE:
int 10

NOTE: the argument is not the same as float's. A number given to int is the 
LARGEST value the box will hold, not the value it starts with - 'int 10' cannot 
be dragged past 10. '+' draws it as a typing box with step buttons.

INPUTS and PARAMETERS:

in:
The value to display and store. Receiving a number here sets the box (dropping 
any fraction) and sends it on. Receiving a bang re-sends what is already there.
""" + WIDGET_OPTIONS + """
OUTPUTS: 

int out:
The value, sent whenever it changes.

RELATED:
float keeps the decimals. 
slider and knob given a whole-number argument also send whole numbers. 
round gives the nearest whole number rather than dropping the fraction."""

demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'f1', 'init': 'float 2.7', 'pos': (30, 130), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'i1', 'init': 'int', 'pos': (30, 200), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c0', 'comment': True, 'text': '2.7 becomes 2: the fraction is dropped\ndrag the float to see it',
     'pos': (30, 250)},
    {'key': 'i2', 'init': 'int 10', 'pos': (30, 330), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': "'int 10' stops at 10 -\nthe argument is its maximum",
     'pos': (30, 380)},
]
links = [('lb', 'out', 'f1', ''), ('f1', 'float out', 'i1', '')]
print(build('int', 'int - a whole number, read or set', body, demo, links,
            demo_width=430, text_width=800, text_height=660))

# --------------------------------------------------------------------- slider
body = """slider sets a number by dragging, within limits you decide.

It holds the same kind of value a number box does, and sends it the same way. 
What it adds is a sense of WHERE the value sits in its range - you can see at a 
glance that something is near the top of its travel, which a number alone does 
not tell you.

Use it wherever the range matters as much as the number: levels, mixes, 
thresholds, anything a person will adjust by feel rather than by typing.

TO USE IT:
Drag to change. Double-click to type a value exactly.

SYNTAX:
slider
slider <max>

EXAMPLE:
slider 100

The argument is the top of the range, not the starting value. A decimal 
('slider 2.0') makes a slider of decimals; a whole number ('slider 100') makes 
one that only sends whole numbers. Without one the slider runs from 0 to 1.

INPUTS and PARAMETERS:

in:
The value to show and store. A number sets the slider; a bang re-sends the 
current value.
""" + WIDGET_OPTIONS + """
power (decimal sliders only):
Bends the scale, so that the travel is not evenly distributed across the range. 
At 1 the slider is linear. Above 1 the low end gets more of the travel, 
which is what you want for anything perceptual - loudness, brightness, 
frequency - where the interesting detail is all down at the bottom and a linear 
slider spends most of its length on values you do not care about.

OUTPUTS: 

float out / int out:
The value, sent whenever it changes. Which one depends on the argument.

RELATED:
knob is the same control in less room. 
float and int are the same value without the travel. 
gain is a slider that multiplies what passes through it, rather than sending 
its own value. 
slider_bank is a row of named sliders that each send a message."""

demo = [
    {'key': 'sl', 'init': 'slider', 'pos': (30, 62), 'w': 220, 'h': 60,
     'props': {'min': 0.0, 'max': 1.0, 'format': '%.3f', 'width': 200, 'power': 1.0}},
    {'key': 'c0', 'comment': True, 'text': 'drag it; double-click to type', 'pos': (30, 130)},
    {'key': 'f1', 'init': 'float', 'pos': (30, 170), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'sl2', 'init': 'slider 100', 'pos': (30, 250), 'w': 220, 'h': 60,
     'props': {'width': 200}},
    {'key': 'i1', 'init': 'int', 'pos': (30, 330), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': "'slider 100': 0 to 100, whole numbers",
     'pos': (30, 380)},
]
links = [('sl', 'float out', 'f1', ''), ('sl2', 'int out', 'i1', '')]
print(build('slider', 'slider - set a number by dragging', body, demo, links,
            demo_width=420, text_width=800, text_height=720))

# ----------------------------------------------------------------------- knob
body = """knob is a slider wound into a circle: the same value, in less room.

Drag on it to turn it. Like a slider, it shows where the value sits in its range at a glance, and it is the 
better choice when a panel needs many of them side by side.

TO USE IT:
Drag to turn it. Double-click to type a value exactly.

SYNTAX:
knob
knob <max>

EXAMPLE:
knob 10.0

The argument is the top of the range, not the starting value. A decimal makes 
a knob of decimals; a whole number ('knob 100') makes one that sends whole 
numbers. Without one the knob runs from 0 to 1.

INPUTS and PARAMETERS:

in:
The value to show and store. A number turns the knob; a bang re-sends the 
current value.
""" + WIDGET_OPTIONS + """
OUTPUTS: 

float out / int out:
The value, sent whenever it changes. Which one depends on the argument.

RELATED:
slider is the same control laid out straight, and has a power option to bend 
its scale; the knob does not. 
float and int are the same value without the travel."""

demo = [
    {'key': 'kn', 'init': 'knob', 'pos': (30, 62), 'w': 100, 'h': 110,
     'props': {'min': 0.0, 'max': 1.0, 'format': '%.3f'}},
    {'key': 'c0', 'comment': True, 'text': 'drag up or right to turn it', 'pos': (30, 185)},
    {'key': 'f1', 'init': 'float', 'pos': (30, 225), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'kn2', 'init': 'knob 100', 'pos': (220, 62), 'w': 100, 'h': 110},
    {'key': 'i1', 'init': 'int', 'pos': (220, 225), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': "'knob 100': whole numbers", 'pos': (220, 275)},
]
links = [('kn', 'float out', 'f1', ''), ('kn2', 'int out', 'i1', '')]
print(build('knob', 'knob - a slider in less room', body, demo, links,
            demo_width=420, text_width=800, text_height=560))

# ------------------------------------------------------------ param_ widgets
body = """The param_ widgets are ordinary widgets that also carry a NAME.

Every one of them behaves exactly like the widget it is named after - 
param_float is a float box, param_slider is a slider. The difference is that 
the first argument is a parameter name, and the widget carries it. 

That matters when a value has to travel somewhere that needs to know what it IS, 
not just what it equals - an OSC address, a preset file, a control surface, a 
list of settings being gathered up. A bare 0.75 tells the far end nothing. 
"gain 0.75" tells it everything.

THE NODES:

param_float     a float box with a name
param_int       an int box with a name
param_slider    a slider with a name
param_knob      a knob with a name
param_string    a text box with a name
param_message   a message with a name
param_list      a list box with a name

SYNTAX:
param_<widget> <parameter name> <value>

EXAMPLE:
param_slider gain 0.5

INPUTS and PARAMETERS:

in:
The value, exactly as on the plain widget.

parameter name:
The name this widget carries. It is set by the first argument and can be 
changed here afterwards.

Everything else - min, max, format, speed_property, width, font size, bind to - 
works as it does on the plain widget. See the float, slider and string help 
patches for those.

OUTPUTS: 

out:
The value. The parameter name travels with it wherever the receiving end knows 
to look for it.

CHOOSING BETWEEN THIS AND bind to:
Both attach a name to a widget, and they solve different problems. 
"bind to" ties the widget to a variable INSIDE this patch, so several places 
share one live value. A parameter name labels the value for something OUTSIDE 
the patch - a console, a file, a device. You can use both on the same widget."""

demo = starter() + [
    {'key': 'ps', 'init': 'param_slider gain 0.5', 'pos': (30, 132), 'w': 220, 'h': 80,
     'props': {'parameter name': 'gain', 'min': 0.0, 'max': 1.0,
               'format': '%.3f', 'width': 200}},
    {'key': 'c0', 'comment': True, 'text': 'a slider that knows it is called gain',
     'pos': (30, 220)},
    {'key': 'f1', 'init': 'float', 'pos': (30, 260), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'pf', 'init': 'param_float threshold 0.25', 'pos': (30, 320), 'w': 160, 'h': 42,
     'props': {'parameter name': 'threshold', 'format': '%.3f', 'width': 120}},
    {'key': 'f2', 'init': 'float', 'pos': (30, 380), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'open the options to see the names\notherwise they are ordinary widgets',
     'pos': (30, 430)},
]
links = [('lb', 'out', 'tt', ''), ('ps', 'float out', 'f1', ''),
         ('pf', 'float out', 'f2', '')]
print(build('param_widgets', 'param_ widgets - a value that carries its name', body,
            demo, links, demo_width=430, text_width=790, text_height=620))

# --------------------------------------------------------------------- button
body = """A button is a moment. Click it and it sends, then it is done - nothing is 
remembered. Use it to start something.

If you want something that stays on until you switch it off, that is a toggle - 
see the toggle help patch. Choosing the wrong one is a common source of patches 
that almost work: a button cannot tell you whether something is currently 
running, and a toggle cannot tell you that it just started.

THE NODES:

button      click to send; b is a shorter name for it
button_set  a column of buttons, each one labelled with what it sends; 
            buttons is another name for it

button_set is the button for a choice rather than a moment. Its arguments are 
the labels, and the label is the message - 'button_set red green blue' is three 
buttons, and clicking the middle one sends 'green'. That is the whole node: 
nothing to fill in, and what a button says is what comes out of it. Where a 
radio shows which one is current, a button_set does not remember - it is a row 
of separate moments that happen to be named.

SYNTAX:
button
button_set <label> <label> ...
button_set <count>

INPUTS and PARAMETERS:

in (button):
Anything arriving here acts as a click.

the buttons (button_set):
Each button is an inlet of its own, and anything arriving there presses it - 
so the patch can press a button the way a person does. A label sent as a 
message to any of them presses THAT button whichever inlet it arrives at, 
which is how a patch presses one by name without knowing where it sits.

message (button):
What the button actually sends. The default is the word "bang". 
Change it and the button sends that instead - which turns a button into a 
one-click way of firing any fixed value or command.

flash_duration / color (button):
How long the button lights up when clicked, and what colour it is. 
Worth setting when several buttons sit together and you want them told apart.

width / height (button):
Its size.

message (button_set):
A template, when the bare label is not the message you want. '{name}' is the 
label and '{index}' its position: 'preset {name}' sends 'preset red'. The 
buttons go on reading as their names.

label 1, label 2, ... (button_set):
The labels, editable after the fact. Renaming a button renames the message it 
sends, the inlet, and the message that presses it, all at once.

width / height / uniform_width / flash_duration (button_set):
Buttons fit their labels unless a width is set here, and stand at the height of 
their text unless a height is; uniform_width squares the column off to the 
widest one. Height is worth raising for anything meant to be hit in a hurry, or 
on a touchscreen. flash_duration is how long a button lights up when it is 
pressed - which is what tells you a button the PATCH pressed went off at all.

colours (button_set):
Wire a color node straight into a button and it colours that button. A colour - 
3 or 4 numbers, 0-1 as a color node sends them, or 0-255 - arriving at a 
button's inlet sets its colour instead of pressing it; anything else still 
presses. By message, 'color <button> r g b a' colours a button named by its 
label or its number counting from 1, and 'color <button>' on its own puts it 
back to the default look. The colours are saved with the patch, and a pressed 
button still flashes and then returns to its own colour.

hide_title_bar:
Draws the widget alone, without the node's title bar and frame. For a panel 
of controls that is meant to be looked at rather than patched.

OUTPUTS: 

out:
button sends its message, once per click. 
button_set sends the label of whichever button was pressed.

A NOTE ON WHAT A BUTTON SENDS:
By default it is the WORD "bang", not a number. Anything expecting a number 
will read it as 0 - so a button wired into accumulate adds nothing at all. 
Send a bang to a counter to count clicks, or set the message option to a 
number if you want arithmetic."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'a moment: click and it is over',
     'pos': (30, 115)},
    {'key': 'cnt', 'init': 'counter', 'pos': (30, 155), 'w': 123, 'h': 84,
     'props': {'step': 1}},
    {'key': 'i1', 'init': 'int', 'pos': (30, 250), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 'counter counts bangs; accumulate would not',
     'pos': (30, 300)},
    {'key': 'bs', 'init': 'button_set red green blue', 'pos': (250, 62),
     'w': 100, 'h': 120},
    {'key': 's1', 'init': 'string', 'pos': (250, 195), 'w': 160, 'h': 42},
    {'key': 'c4', 'comment': True, 'text': 'the label IS the message',
     'pos': (250, 245)},
]
links = [('btn', '', 'cnt', 'input'), ('cnt', 'count out', 'i1', ''),
         ('bs', 'out', 's1', '')]
print(build('button', 'button - a moment', body, demo, links,
            demo_width=440, text_width=790, text_height=700))

# --------------------------------------------------------------------- toggle
body = """A toggle is a state. Click it and it stays on until you click again, sending 1 
and 0 as it changes. Use it to enable something.

If you want something that fires once and is done, that is a button - see the 
button help patch. The distinction is the same one togedge draws between an 
event and a state: a toggle can tell you whether something is running, but not 
that it just started.

THE NODES:

toggle      click to switch between on and off
set_reset   a toggle driven by two inlets instead of by clicking

set_reset is the toggle for when the patch, rather than a person, decides. 
Anything arriving at "set" turns it on, anything at "reset" turns it off, 
and it holds that state in between - which is how you latch a condition that 
begins in one place and ends in another.

SYNTAX:
toggle
set_reset

INPUTS and PARAMETERS:

in (toggle):
The box itself. Click it, or send it something: 
  bang          flips it, on to off or off to on 
  a number      on if its whole-number part is not zero, off if it is - so 0.5 
                turns it OFF, the fraction being dropped first 
  set <value>   changes it the same way WITHOUT sending anything, for putting 
                a toggle in step with something it should not then trigger

set / reset (set_reset):
Turn the state on and off. Anything sent works; only the arrival matters.

bind to:
A variable name. A bound toggle and its variable are the same thing.

prefix / prefix_as_label:
Word(s) sent in front of the value. A prefix of "record" makes the toggle send 
"record 1" and "record 0" - a message rather than a bare number, ready for a 
node that reads the first word as a command. With prefix_as_label the prefix 
is also drawn as a name in front of the toggle, so a chromeless toggle still 
says what it is for.

font size:
The size of the box and its name - 24, 30, 36 or 48.

hide_title_bar:
Draws the widget alone, without the node's title bar and frame. For a panel 
of controls that is meant to be looked at rather than patched.

OUTPUTS: 

out:
1 when it turns on and 0 when it turns off (after the prefix, if there is one)."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42,
     'props': {'prefix': 'run', 'font size': '30', 'hide_title_bar': True}},
    {'key': 'c2', 'comment': True, 'text': 'a state: click it, and it stays where you put it',
     'pos': (30, 62)},
    {'key': 'met', 'init': 'metro 200', 'pos': (30, 132), 'w': 129, 'h': 70,
     'props': {'on': False, 'period': 200.0, 'units': 'milliseconds'}},
    {'key': 'cnt2', 'init': 'counter', 'pos': (30, 217), 'w': 123, 'h': 84,
     'props': {'step': 1}},
    {'key': 'i2', 'init': 'int', 'pos': (30, 312), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c4', 'comment': True, 'text': 'counts while the toggle is on',
     'pos': (30, 312)},
    {'key': 'bset', 'init': 'button', 'pos': (30, 410), 'w': 88, 'h': 46},
    {'key': 'bres', 'init': 'button', 'pos': (140, 410), 'w': 88, 'h': 46},
    {'key': 'c3', 'comment': True, 'text': 'left button sets, right button resets',
     'pos': (30, 410)},
    {'key': 'sr', 'init': 'set_reset', 'pos': (30, 480), 'w': 130, 'h': 90},
    {'key': 'i3', 'init': 'int', 'pos': (30, 585), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c5', 'comment': True, 'text': 'set_reset: the same state, decided by the patch',
     'pos': (30, 585)},
]
links = [('tog', '', 'met', 'on'), ('met', '', 'cnt2', 'input'),
         ('cnt2', 'count out', 'i2', ''),
         ('bset', '', 'sr', 'set'), ('bres', '', 'sr', 'reset'), ('sr', '', 'i3', '')]
print(build('toggle', 'toggle - a state', body, demo, links,
            demo_width=300, text_width=790, text_height=640))
