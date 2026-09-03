"""float/int, slider/knob, the param_ widgets, button/toggle."""
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
"""

# ---------------------------------------------------------------- float / int
body = """float and int are number boxes: you can read the value, and you can change it by hand.

They are the plainest interface in the system, and they do three jobs at once. 
A number arriving at the inlet is displayed, so they work as a readout. 
Dragging or typing in them sends a number, so they work as a control. 
And they hold the value between times, so the patch can ask for it later.

float keeps decimals; int rounds to whole numbers. 
Choose int when a fraction would be meaningless - a count, an index, a channel - 
because it stops nonsense arriving downstream rather than tidying it up later.

TO USE THEM:
Drag left and right on the number to change it. Double-click, or click and type, 
to enter one exactly. A bang at the inlet re-sends the current value without 
changing it - which is how you ask a number box what it is holding.

SYNTAX:
float <value>
int <value>

EXAMPLE:
float 0.5

INPUTS and PARAMETERS:

in:
The value to display and store. Receiving a number here sets the box and sends 
it on. Receiving a bang re-sends whatever is already there.
""" + WIDGET_OPTIONS + """
OUTPUTS: 

float out / int out:
The value, sent whenever it changes - whether that was you dragging it or a 
number arriving at the inlet.

RELATED:
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
    {'key': 'i1', 'init': 'int', 'pos': (30, 480), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 'as a control: drag the 0.5 box',
     'pos': (30, 530)},
    {'key': 'c2', 'comment': True, 'text': 'int rounds whatever it is given',
     'pos': (30, 560)},
    {'key': 'btn', 'init': 'button', 'pos': (200, 330), 'w': 88, 'h': 46},
    {'key': 'c3', 'comment': True, 'text': 'bang it to re-send without changing',
     'pos': (200, 385)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'f1', ''), ('btn', '', 'f2', ''),
         ('f2', 'float out', 'mul', 'in'), ('mul', 'result', 'i1', '')]
print(build('float', 'float and int - read a number, or set one', body, demo, links,
            demo_width=430, text_width=800, text_height=700))

# ---------------------------------------------------------------- slider/knob
body = """slider and knob set a number by dragging, within limits you decide.

They hold the same kind of value a number box does, and send it the same way. 
What they add is a sense of WHERE the value sits in its range - you can see at a 
glance that something is near the top of its travel, which a number alone does 
not tell you.

Use them wherever the range matters as much as the number: levels, mixes, 
thresholds, anything a person will adjust by feel rather than by typing.

TO USE THEM:
Drag to change. Double-click to type a value exactly.

SYNTAX:
slider <value>
knob <value>

EXAMPLE:
slider 0.5

INPUTS and PARAMETERS:

in:
The value to show and store. A number sets the slider; a bang re-sends the 
current value.
""" + WIDGET_OPTIONS + """
power:
Bends the scale, so that the travel is not evenly distributed across the range. 
At 1 the slider is linear. Above 1 the low end gets more of the travel, 
which is what you want for anything perceptual - loudness, brightness, 
frequency - where the interesting detail is all down at the bottom and a linear 
slider spends most of its length on values you do not care about.

OUTPUTS: 

float out / int out:
The value, sent whenever it changes.

RELATED:
float and int are the same value without the travel. 
gain is a slider that multiplies a signal passing through it, rather than 
sending its own value. 
slider_bank is a row of named sliders that each send a message."""

demo = starter() + [
    {'key': 'sl', 'init': 'slider 0.5', 'pos': (30, 132), 'w': 220, 'h': 60,
     'props': {'min': 0.0, 'max': 1.0, 'format': '%.3f', 'width': 200, 'power': 1.0}},
    {'key': 'c0', 'comment': True, 'text': 'drag it; double-click to type', 'pos': (30, 200)},
    {'key': 'f1', 'init': 'float', 'pos': (30, 240), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'kn', 'init': 'knob 0.5', 'pos': (30, 300), 'w': 100, 'h': 110,
     'props': {'min': 0.0, 'max': 1.0, 'format': '%.3f'}},
    {'key': 'f2', 'init': 'float', 'pos': (30, 430), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'a knob is a slider that takes less room',
     'pos': (30, 485)},
    {'key': 'c2', 'comment': True, 'text': 'raise power to stretch the low end',
     'pos': (30, 515)},
]
links = [('lb', 'out', 'tt', ''), ('sl', 'float out', 'f1', ''),
         ('kn', 'float out', 'f2', '')]
print(build('slider', 'slider and knob - set a number by dragging', body, demo, links,
            demo_width=420, text_width=800, text_height=680))

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
    {'key': 'c1', 'comment': True, 'text': 'open the options to see the names',
     'pos': (30, 430)},
    {'key': 'c2', 'comment': True, 'text': 'otherwise they are ordinary widgets',
     'pos': (30, 460)},
]
links = [('lb', 'out', 'tt', ''), ('ps', 'float out', 'f1', ''),
         ('pf', 'float out', 'f2', '')]
print(build('param_widgets', 'param_ widgets - a value that carries its name', body,
            demo, links, demo_width=430, text_width=790, text_height=620))

# ------------------------------------------------------------- button, toggle
body = """These are the two ways of clicking something: one that fires, and one that stays.

A button is a moment. Click it and it sends, then it is done - nothing is 
remembered. Use it to start something.

A toggle is a state. Click it and it stays on until you click again, sending 1 
and 0 as it changes. Use it to enable something.

The distinction is the same one togedge draws between an event and a state, 
and choosing the wrong one is a common source of patches that almost work: 
a button cannot tell you whether something is currently running, and a toggle 
cannot tell you that it just started.

THE NODES:

button      click to send; b is a shorter name for it
toggle      click to switch between on and off
set_reset   a toggle driven by two inlets instead of by clicking

set_reset is the toggle for when the patch, rather than a person, decides. 
Anything arriving at "set" turns it on, anything at "reset" turns it off, 
and it holds that state in between - which is how you latch a condition that 
begins in one place and ends in another.

SYNTAX:
button
toggle
set_reset

INPUTS and PARAMETERS:

in (button, toggle):
Anything arriving here acts as a click.

set / reset (set_reset):
Turn the state on and off. Anything sent works; only the arrival matters.

message (button):
What the button actually sends. The default is the word "bang". 
Change it and the button sends that instead - which turns a button into a 
one-click way of firing any fixed value or command.

flash_duration / color (button):
How long the button lights up when clicked, and what colour it is. 
Worth setting when several buttons sit together and you want them told apart.

width / height (button):
Its size.

bind to:
A variable name. A bound toggle and its variable are the same thing.

OUTPUTS: 

out:
button sends its message, once per click. 
toggle and set_reset send 1 when they turn on and 0 when they turn off.

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
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 345), 'w': 45, 'h': 42},
    {'key': 'c2', 'comment': True, 'text': 'a state: it stays where you put it',
     'pos': (30, 395)},
    {'key': 'met', 'init': 'metro 200', 'pos': (30, 435), 'w': 129, 'h': 70,
     'props': {'on': False, 'period': 200.0, 'units': 'milliseconds'}},
    {'key': 'cnt2', 'init': 'counter', 'pos': (30, 520), 'w': 123, 'h': 84,
     'props': {'step': 1}},
    {'key': 'i2', 'init': 'int', 'pos': (30, 615), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'sr', 'init': 'set_reset', 'pos': (250, 345), 'w': 130, 'h': 90},
    {'key': 'c3', 'comment': True, 'text': 'the same state, decided by the patch',
     'pos': (250, 445)},
]
links = [('btn', '', 'cnt', 'input'), ('cnt', 'count out', 'i1', ''),
         ('tog', '', 'met', 'on'), ('met', '', 'cnt2', 'input'),
         ('cnt2', 'count out', 'i2', '')]
print(build('button', 'button and toggle - a moment, or a state', body, demo, links,
            demo_width=440, text_width=790, text_height=700))
