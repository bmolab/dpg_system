"""momentary, joy_stick, presets, shape sequencers, envelope, slider_bank, gain."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ------------------------------------------------------------------ momentary
body = """These sliders spring back to the middle the moment you let go.

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

For the same idea in two dimensions, see joy_stick and momentary_xy.

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

width:
The length of the sliders.

OUTPUTS: 

<one outlet per slider>:
The current value, sent as you move it and again as it springs back - 
so the last thing you receive is always the zero.

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
    {'key': 'c2', 'comment': True, 'text': 'through accumulate it becomes a throttle:\nhold to travel, release to stop',
     'pos': (30, 445)},
]
links = [('mo', '', 'f1', ''), ('mo', '', 'acc', 'in'), ('acc', 'sum', 'f2', '')]
print(build('momentary', 'momentary - a slider that springs back', body, demo, links,
            demo_width=440, text_width=800, text_height=660))

# ------------------------------------------------------------------ joy_stick
body = """joy_stick is a two-dimensional pad: drag the dot and it sends x and y.

One control for two values that belong together - a position on a floor, a 
pan and tilt, a pair of mix amounts - so that one gesture moves both at once, 
which two separate sliders cannot do.

THE NODES:

joy_stick      the pad; the dot stays where you leave it
momentary_xy   the same pad, springing back to the centre when you let go

The name only sets where it starts: the momentary option switches either one 
between the two behaviours.

The springing version is a control for RATE, like the momentary sliders: hold 
it off centre to keep something moving, let go to stop. When it springs back it 
sends 0 and 0, so whatever it drives should read zero as "stop".

SYNTAX:
joy_stick
joy_stick <range>
momentary_xy <range>

EXAMPLE:
momentary_xy 0.5

Both axes run from -range to +range, with 0 at the centre. The range is 1.0 
unless given.

INPUTS and PARAMETERS:

momentary:
Whether the dot springs back to the centre when released.

range:
How far each axis runs either side of centre.

width / height / marker size:
The size of the pad, and of the dot.

OUTPUTS: 

x out / y out:
The two axes, on separate outlets, sent as you drag - and 0, 0 when a momentary 
pad springs back."""

demo = [
    {'key': 'js', 'init': 'joy_stick', 'pos': (30, 62), 'w': 200, 'h': 220,
     'props': {'momentary': False, 'range': 1.0, 'width': 160, 'height': 160,
               'marker size': 6}},
    {'key': 'c0', 'comment': True, 'text': 'joy_stick: the dot stays where you leave it',
     'pos': (30, 62)},
    {'key': 'fx', 'init': 'float', 'pos': (30, 300), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'fy', 'init': 'float', 'pos': (170, 300), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': 'x and y', 'pos': (30, 300)},
    {'key': 'mx', 'init': 'momentary_xy', 'pos': (30, 390), 'w': 200, 'h': 220,
     'props': {'momentary': True, 'range': 1.0, 'width': 160, 'height': 160,
               'marker size': 6}},
    {'key': 'c1', 'comment': True, 'text': 'momentary_xy: it springs back to the centre',
     'pos': (30, 390)},
    {'key': 'gx', 'init': 'float', 'pos': (30, 628), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'gy', 'init': 'float', 'pos': (170, 628), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'and sends 0, 0 when it does', 'pos': (30, 628)},
]
links = [('js', 'x out', 'fx', ''), ('js', 'y out', 'fy', ''),
         ('mx', 'x out', 'gx', ''), ('mx', 'y out', 'gy', '')]
print(build('joy_stick', 'joy_stick - two values in one gesture', body, demo, links,
            demo_width=340, text_width=760, text_height=640))

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
    {'key': 'c0', 'comment': True, 'text': 'move the sliders, tick remember,\nclick a slot to store the positions\nuntick, then click slots to recall',
     'pos': (30, 315)},
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
    {'key': 'c1', 'comment': True, 'text': 'drag the points; right-click to add one\nshift-drag a segment to bend it',
     'pos': (30, 530)},
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
    {'key': 'c2', 'comment': True, 'text': 'each step holds its own curve\ntick follow play to watch it move',
     'pos': (30, 565)},
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

# ---------------------------------------------------------------- slider_bank
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

SYNTAX:
slider_bank <count: int>
slider_bank <name> <name> ...

EXAMPLE:
slider_bank root spine left_arm right_arm

INPUTS and PARAMETERS:

in:
Accepts messages: 
  set <name or index> <value>   move one slider, and send its message
  send                          send every slider's message, in order
A plain list of numbers sets the sliders in order.

message:
The template each slider fills in. "{name}" and "{value}" are replaced.

min / max:
The range every slider in the bank shares.

OUTPUTS: 

messages:
The filled-in message for whichever slider moved, as a list.

RELATED:
slider is a single one, sending a bare number. 
gain is a single slider that scales what passes through it."""

demo = [
    {'key': 'sb', 'init': 'slider_bank root spine left_arm', 'pos': (30, 62),
     'w': 280, 'h': 200, 'props': {'message': '{name} {value}',
                                   'min': 0.0, 'max': 1.0}},
    {'key': 'c0', 'comment': True, 'text': 'move any slider', 'pos': (30, 275)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 315), 'w': 260, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'the name comes with the value',
     'pos': (30, 365)},
]
links = [('sb', 'messages', 'l1', '')]
print(build('slider_bank', 'slider_bank - many sliders, each with a name', body,
            demo, links, demo_width=440, text_width=800, text_height=600))

# ----------------------------------------------------------------------- gain
body = """gain scales whatever passes through it by the position of its slider.

It looks like a slider, but it does not send its own value. A number, NumPy 
array or PyTorch tensor arriving at the inlet comes out multiplied by where the 
slider sits - signal in, scaled signal out. It is a volume control for data.

Because only arriving data is sent on, moving the slider by itself sends 
nothing. The new setting shows in the next value that passes through - for a 
stream, that is the next frame.

SYNTAX:
gain
gain <max>

EXAMPLE:
gain 2.0

The slider runs from 0 to max, which is 1.0 unless given. With a max above 1 
the node amplifies as well as attenuates.

INPUTS and PARAMETERS:

in:
The data to scale: numbers, NumPy arrays and PyTorch tensors. 
Anything else - text, a bang - is ignored.

(the slider):
The multiplier.

max:
The top of the slider's range.

OUTPUTS: 

out:
The input multiplied by the slider position.

RELATED:
slider sends its own value rather than scaling another. 
* multiplies by a number you send it, rather than one you drag."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 3.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 3.0)},
    # the gain slider's own property is unnamed; start it part-open so the
    # demo shows a scaled wave rather than a flat line at zero
    {'key': 'gn', 'init': 'gain 1.0', 'pos': (30, 230), 'w': 240, 'h': 70,
     'props': {'': 0.7, 'max': 1.0}},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 315), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c2', 'comment': True, 'text': 'drag the gain: the wave grows and shrinks',
     'pos': (30, 500)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'gn', ''), ('gn', '', 'p1', 'y')]
print(build('gain', 'gain - a volume control for data', body, demo, links,
            demo_width=440, text_width=780, text_height=600))
