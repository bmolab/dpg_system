"""momentary widgets, presets, shape sequencers, envelope, slider_bank/gain."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ------------------------------------------------------------------ momentary
body = """These controls spring back to the middle the moment you let go.

An ordinary slider stays where you leave it. A momentary one does not - release 
it and it returns to zero on its own. That makes it a control for RATE rather 
than position: while you hold it away from centre you are saying "keep going, 
this fast, this way", and letting go means stop.

It is the difference between a throttle and a dial. For nudging a value, 
steering a view, jogging through time, or anything where you want to push and 
then have things settle, this is the shape of control you want - and it cannot 
be left switched on by accident, which matters when the thing at the far end 
keeps moving as long as the control is off centre.

THE NODES:

momentary               one float slider, -1 to 1
momentary_slider        the same thing
momentary_int           whole numbers instead, -20 to 20
momentary_slider_int    the same thing
momentary_xy            a two-dimensional pad, springing back to the centre
joy_stick               the same pad, with the spring optional

SYNTAX:
momentary                     one slider
momentary <count: int>        that many sliders, if between 1 and 10
momentary <range: int>        one slider with that range
momentary <name> <name> ...   one named slider per name

EXAMPLE:
momentary pan tilt

Giving names is worth doing whenever there is more than one - a row of unlabelled 
sliders is unreadable a week later.

INPUTS and PARAMETERS:

<one inlet per slider>:
Sets that slider from the patch. It still springs back when released.

range:
How far the slider travels either side of centre. 
Defaults to 1.0, or 20 for the int versions.

width / height / marker size:
The size of the control, and of the dot on the xy pad.

momentary (joy_stick):
Whether the pad springs back at all. Unchecked, joy_stick keeps its position 
like an ordinary xy pad - which is the only thing separating it from 
momentary_xy.

OUTPUTS: 

<one outlet per slider>:
The current value, sent as you move it and again as it springs back - 
so the last thing you receive is always the zero.

x out / y out (the pads):
The two axes, on separate outlets.

A NOTE ON WHAT THE SPRING MEANS:
Because releasing sends a zero, whatever you drive with this must treat zero as 
"stop" rather than as a position. Send it to accumulate to turn the push into 
movement; send it straight to a position and the thing will snap back to the 
origin every time you let go."""

demo = [
    {'key': 'mo', 'init': 'momentary', 'pos': (30, 62), 'w': 200, 'h': 70,
     'props': {'range': 1.0, 'width': 120}},
    {'key': 'c0', 'comment': True, 'text': 'drag it, then let go', 'pos': (30, 145)},
    {'key': 'f1', 'init': 'float', 'pos': (30, 185), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'it comes back to zero by itself',
     'pos': (30, 235)},
    {'key': 'acc', 'init': 'accumulate', 'pos': (30, 280), 'w': 140, 'h': 100},
    {'key': 'f2', 'init': 'float', 'pos': (30, 395), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': 'through accumulate it becomes a throttle:',
     'pos': (30, 445)},
    {'key': 'c3', 'comment': True, 'text': 'hold to travel, release to stop',
     'pos': (30, 475)},
    {'key': 'js', 'init': 'joy_stick', 'pos': (280, 62), 'w': 200, 'h': 220,
     'props': {'momentary': True, 'range': 1.0, 'width': 160, 'height': 160,
               'marker size': 6}},
    {'key': 'c4', 'comment': True, 'text': 'two axes at once', 'pos': (280, 295)},
]
links = [('mo', '', 'f1', ''), ('mo', '', 'acc', 'in'), ('acc', 'sum', 'f2', '')]
print(build('momentary', 'momentary - a control that springs back', body, demo, links,
            demo_width=500, text_width=800, text_height=700))

# -------------------------------------------------------------------- presets
body = """These nodes remember the state of a patch, so you can put it back later.

Click a numbered button to recall a stored state; hold and click, or use the 
remember option, to store the current one into it. What gets remembered depends 
on which of these you use, and that is the whole distinction between them.

THE NODES:

presets      remember where the WIDGETS are - every slider, knob, toggle 
             and number box in the patch
snapshots    remember the state of the NODES themselves
states       the same as snapshots
versions     the same as presets
archive      the same as presets

Widget presets are the everyday case: a set of positions you can flip between 
while working, and the thing you want when someone is performing with the patch. 
Node snapshots go deeper, capturing state that is not on the surface.

SYNTAX:
presets <count: int>

EXAMPLE:
presets 12

The argument sets how many slots there are. The default is 8.

INPUTS and PARAMETERS:

in:
Recall a preset by number. Sending 3 here is the same as clicking the third 
button - which is how a patch recalls its own presets, from a sequencer, 
a key, or an incoming message.

remember:
The store mode. With this on, clicking a slot WRITES the current state into it 
rather than recalling it. Turn it off again once you have stored what you 
wanted, or you will overwrite a preset the next time you try to recall one.

OUTPUTS: 

out:
The number of the preset that was just recalled, so the rest of the patch can 
follow along - to change a label, or to trigger something that belongs with 
that state.

WHAT IS ACTUALLY SAVED:
Presets are stored with the patch, so they survive being saved and reopened. 
Store deliberately: because "remember" is a mode rather than a separate 
gesture, the commonest way to lose a preset is to leave it switched on and then 
click a slot expecting to recall."""

demo = [
    {'key': 'pr', 'init': 'presets 8', 'pos': (30, 62), 'w': 130, 'h': 240},
    {'key': 'c0', 'comment': True, 'text': 'move the sliders, tick remember,',
     'pos': (30, 315)},
    {'key': 'c1', 'comment': True, 'text': 'click a slot to store the positions',
     'pos': (30, 345)},
    {'key': 'c2', 'comment': True, 'text': 'untick, then click slots to recall',
     'pos': (30, 375)},
    {'key': 'sl1', 'init': 'slider 0.5', 'pos': (220, 62), 'w': 220, 'h': 60,
     'props': {'min': 0.0, 'max': 1.0, 'format': '%.2f', 'width': 200}},
    {'key': 'sl2', 'init': 'slider 0.5', 'pos': (220, 135), 'w': 220, 'h': 60,
     'props': {'min': 0.0, 'max': 1.0, 'format': '%.2f', 'width': 200}},
    {'key': 'tg', 'init': 'toggle', 'pos': (220, 210), 'w': 45, 'h': 42},
    {'key': 'i1', 'init': 'int', 'pos': (30, 420), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c3', 'comment': True, 'text': 'which preset was recalled', 'pos': (30, 470)},
]
links = [('pr', '', 'i1', '')]
print(build('presets', 'presets - store a state, and get it back', body, demo, links,
            demo_width=470, text_width=790, text_height=640))

# ------------------------------------------------- envelope, shape_sequencer
body = """envelope is a curve you draw with the mouse and then read values from.

Drag a point to move it. Right-click on empty space to add one, or near an 
existing point to remove it. Shift and left-drag a segment to bend it into a 
curve rather than a straight line.

Once drawn, there are two ways to get values out. Send an x position and it 
reports the height of the curve there - the curve acting as a lookup table, 
a mapping from one range to another that you shaped by hand instead of 
calculating. Or bang "trigger" and a playhead sweeps across the whole curve 
over the duration you set, sending values as it goes - the curve acting as an 
envelope in the usual sense, a shape unfolding in time.

The first use is the more interesting one in a patch. Any relationship you can 
describe better by drawing than by writing - a response curve, a fade law, 
a mapping from effort to brightness - can be drawn here and read continuously.

SYNTAX:
envelope

INPUTS and PARAMETERS:

x:
A position along the curve. The height there is reported immediately. 
This is the lookup use.

trigger:
Starts a sweep from the beginning of the curve to the end, taking "duration" 
to do it and sending values as it travels.

duration:
How long a triggered sweep takes, in seconds.

x max / y min / y max:
The ranges the curve spans, which set what the values coming out actually mean.

width / height:
The size of the editor.

OUTPUTS: 

value out:
The height of the curve - at the x you asked for, or at the playhead during 
a sweep.

points out:
The control points themselves, so a curve can be stored, sent elsewhere, 
or restored later."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 4.0 saw', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('saw', 4.0, 1.0, False)},
    {'key': 'c0', 'comment': True, 'text': 'a ramp sweeping across the curve',
     'pos': (30, 215)},
    {'key': 'env', 'init': 'envelope', 'pos': (30, 255), 'w': 320, 'h': 260,
     'props': {'x max': 1.0, 'y min': 0.0, 'y max': 1.0,
               'width': 280, 'height': 200}},
    {'key': 'c1', 'comment': True, 'text': 'drag the points; right-click to add one',
     'pos': (30, 530)},
    {'key': 'c2', 'comment': True, 'text': 'shift-drag a segment to bend it',
     'pos': (30, 560)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 600), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 1.0)},
    {'key': 'c3', 'comment': True, 'text': 'the shape you drew, read out over time',
     'pos': (30, 785)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'env', 'x'), ('env', 'value out', 'p1', 'y')]
print(build('envelope', 'envelope - draw a curve, read values from it', body,
            demo, links, demo_width=440, text_width=790, text_height=640))

# ------------------------------------------------------------ shape_sequencer
body = """A step sequencer whose steps hold CURVES rather than single values.

An ordinary sequencer steps through a list of numbers: beat one gives you this, 
beat two gives you that. This one steps through a list of functions. 
On each beat it advances a step, looks at the x inlet, reads THAT step's curve 
at that x, and sends the result.

A plain value sequencer is the flat case - every curve a horizontal line, so x 
makes no difference and each step is just a number. The interesting case is a 
continuous input running through it: a fader, an lfo, a stream of effort data. 
Then each step is not a value but an INTERPRETATION - this beat, map the input 
gently; next beat, map it steeply; the beat after, invert it.

Each step's curve is edited the way the envelope node's is: drag the points, 
right-click to add or remove one, shift and left-drag a segment to curve it.

THE NODES:

shape_seq            
shape_sequencer      the same node
function_sequencer   the same node

SYNTAX:
shape_seq <steps: int>

EXAMPLE:
shape_seq 8

INPUTS and PARAMETERS:

beat:
Advance to the next step and send its value. This is the clock inlet - 
drive it from a metro, or from whatever else marks time in your patch.

x:
Where to read the current step's curve. Feed a continuous signal here and the 
sequencer becomes a bank of mappings rather than a bank of values.

reset:
Return to the first step.

step / steps:
The step now playing, and how many there are altogether.

direction:
Which way to run through the steps.

edit step / follow play / show other steps:
Which step the editor is showing, whether it follows the one playing, and 
whether the others are drawn faintly behind it for comparison.

copy shape / paste shape / copy to all steps:
Move a curve between steps. "copy to all steps" is how you start from one 
shape everywhere and then vary it.

x max / y min / y max:
The ranges, which set what the numbers coming out mean.

show profile / profile height:
Draw the whole sequence's shape as one strip, so you can see the arc across all 
the steps rather than one at a time.

OUTPUTS: 

value out:
The current step's curve, read at x.

step out:
Which step is playing, so the rest of the patch can follow.

cycle:
Fires when the sequence wraps back to the beginning - use it to chain 
sequencers, or to count times through."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'met', 'init': 'metro 500', 'pos': (30, 112), 'w': 129, 'h': 70,
     'props': {'on': False, 'period': 500.0, 'units': 'milliseconds'}},
    {'key': 'c0', 'comment': True, 'text': 'two beats a second', 'pos': (30, 190)},
    {'key': 'sig', 'init': 'signal 3.0 saw', 'pos': (250, 62), 'w': 129, 'h': 78,
     'props': SIG('saw', 3.0, 1.0, False)},
    {'key': 'c1', 'comment': True, 'text': 'a continuous x to read each shape at',
     'pos': (250, 150)},
    {'key': 'lb2', 'init': 'load_bang', 'pos': (420, 62), 'w': 88, 'h': 46},
    {'key': 'tt2', 'init': 't 1', 'pos': (420, 120), 'w': 40, 'h': 46},
    {'key': 'ss', 'init': 'shape_seq 8', 'pos': (30, 230), 'w': 360, 'h': 320},
    {'key': 'c2', 'comment': True, 'text': 'each step holds its own curve',
     'pos': (30, 565)},
    {'key': 'c3', 'comment': True, 'text': 'tick follow play to watch it move',
     'pos': (30, 595)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 635), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 1.0)},
    {'key': 'i1', 'init': 'int', 'pos': (270, 635), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c4', 'comment': True, 'text': 'the step now playing', 'pos': (270, 685)},
]
links = [('tog', '', 'met', 'on'), ('met', '', 'ss', 'beat'),
         ('lb2', 'out', 'tt2', ''), ('tt2', '1', 'sig', 'on'),
         ('sig', '', 'ss', 'x'),
         ('ss', 'value out', 'p1', 'y'), ('ss', 'step out', 'i1', '')]
print(build('shape_sequencer', 'shape_sequencer - a sequence of curves, not values',
            body, demo, links, demo_width=540, text_width=800, text_height=760))

# ------------------------------------------------------- slider_bank and gain
body = """slider_bank is a row of NAMED sliders, each of which sends a message when moved.

An ordinary slider sends a bare number, and the patch has to know from the 
wiring what that number was for. A slider bank sends the name with the value, 
so one outlet can carry a whole control panel: move the "spine" slider and 
"spine 0.4" comes out, move "left_arm" and "left_arm 0.7" does.

That is what makes it scale. Twenty separate sliders means twenty cords and 
twenty places to be wrong; one bank means one cord and a name you can read.

The message is a template you set - by default "{name} {value}", but it can be 
anything, so "weight {name} {value}" produces messages ready for a node that 
expects that shape.

gain is a different thing that looks similar: a single slider that MULTIPLIES 
whatever passes through it, rather than sending its own value. Signal in, 
scaled signal out.

SYNTAX:
slider_bank <count: int>
slider_bank <name> <name> ...
gain <max: float>

EXAMPLE:
slider_bank root spine left_arm right_arm

INPUTS and PARAMETERS - slider_bank:

in:
Accepts messages: 
  set <name or index> <value>   move one slider, and send its message
  send                          send every slider's message, in order
A plain list of numbers sets the sliders in order.

message:
The template each slider fills in. "{name}" and "{value}" are replaced.

min / max:
The range every slider in the bank shares.

INPUTS and PARAMETERS - gain:

in:
The signal to scale. Numbers, NumPy arrays and PyTorch tensors all pass through.

max:
The top of the slider's range. Above 1.0 the node can amplify as well as 
attenuate.

OUTPUTS: 

messages (slider_bank):
The filled-in message for whichever slider moved, as a list.

out (gain):
The input multiplied by the slider position."""

demo = starter() + [
    {'key': 'sb', 'init': 'slider_bank root spine left_arm', 'pos': (30, 132),
     'w': 280, 'h': 200, 'props': {'message': '{name} {value}',
                                   'min': 0.0, 'max': 1.0}},
    {'key': 'c0', 'comment': True, 'text': 'move any slider', 'pos': (30, 345)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 385), 'w': 260, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'the name comes with the value',
     'pos': (30, 435)},
    {'key': 'sig', 'init': 'signal 3.0 sin', 'pos': (30, 480), 'w': 129, 'h': 78,
     'props': SIG('sin', 3.0)},
    # the gain slider's own property is unnamed; start it part-open so the
    # demo shows a scaled wave rather than a flat line at zero
    {'key': 'gn', 'init': 'gain 1.0', 'pos': (30, 575), 'w': 240, 'h': 70,
     'props': {'': 0.7, 'max': 1.0}},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 660), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c2', 'comment': True, 'text': 'gain scales what passes through it',
     'pos': (30, 845)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sb', 'messages', 'l1', ''),
         ('sig', '', 'gn', ''), ('gn', '', 'p1', 'y')]
print(build('slider_bank', 'slider_bank - many sliders, each with a name', body,
            demo, links, demo_width=440, text_width=800, text_height=700))
