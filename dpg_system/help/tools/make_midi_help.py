"""MIDI: the connection, notes, controllers, expression, and two devices."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

PORT = """
port AND channel:
Every one of these has a 'port' inlet, which selects the physical device, and 
most have a 'channel' option. MIDI carries sixteen channels on one cable, so 
the port says which piece of hardware and the channel says which voice within 
it. A node listening on the wrong channel hears nothing at all and looks 
broken, so channel is the first thing to check when a device is plainly sending 
and nothing arrives.
"""

# ------------------------------------------------------------------ midi_device
body = """These are the MIDI connections and the raw message streams.

THE NODES:

midi_device  a device you both send to and receive from
midi_in      everything arriving on a port
midi_out     send raw MIDI to a port

WHAT MIDI IS:
Short messages, each a few bytes: this note started, this controller moved, 
this wheel bent. It is old, small and universal - almost any musical hardware 
speaks it, which is why it remains the easiest way to get physical controls 
into a patch.

USE THE TYPED NODES INSTEAD, MOSTLY:
midi_in gives you every message on the port, raw. That is useful for finding 
out what a device actually sends - patch it into a print and press things - but 
it is a poor way to build with, because you then have to decode the messages 
yourself.

The typed nodes - midi_note_in, midi_control_in and the rest - do the decoding 
and hand you the numbers directly. Start with midi_in to discover the device, 
then wire the typed nodes for the messages you found.

midi_device IS FOR TWO-WAY HARDWARE:
Controllers with lights, motorised faders or displays need to be sent to as 
well as heard from, and this holds both directions in one node - so the port 
setting is in one place rather than two that can disagree.
""" + PORT + """
SYNTAX:
midi_device
midi_in
midi_out

EXAMPLE:
midi_in

INPUTS and PARAMETERS:

port / in port / out port:
Which device.

midi to send:
Raw messages out.

channel:
Which MIDI channel.

OUTPUTS: 

midi out / midi received:
Every message arriving, undecoded.

FINDING OUT WHAT A DEVICE SENDS:
Patch midi_in into a print node and operate the device. Every knob and button 
reports its own controller or note number, and writing those down is the whole 
of getting a new controller working. It is the same technique as watching 
osc_route's unmatched outlet, and for the same reason: what a device actually 
sends and what its manual says are often different."""

demo = [
    {'key': 'mi', 'init': 'midi_in', 'pos': (30, 62), 'w': 240, 'h': 120},
    {'key': 'pr', 'init': 'print midi', 'pos': (30, 205), 'w': 200, 'h': 120,
     'props': {'identifier': 'midi', 'precision': 0}},
    {'key': 'c0', 'comment': True, 'text': 'press things on the device and watch\nthe console - that is how you learn it',
     'pos': (30, 340)},
    {'key': 'md', 'init': 'midi_device', 'pos': (320, 62), 'w': 260, 'h': 180},
    {'key': 'c2', 'comment': True, 'text': 'for hardware with lights or motors',
     'pos': (320, 255)},
    {'key': 'mo', 'init': 'midi_out', 'pos': (320, 300), 'w': 240, 'h': 120},
]
links = [('mi', 'midi out', 'pr', 'in')]
print(build('midi_device', 'midi_device - connections and raw messages', body, demo,
            links, demo_width=620, text_width=790, text_height=680))

# ---------------------------------------------------------------- midi_note_in
body = """These handle notes: something started, something stopped, and how hard.

THE NODES:

midi_note_in            notes arriving
midi_note_out           notes going out
midi_poly_pressure_in   per-note pressure arriving
midi_poly_pressure_out  per-note pressure going out

A NOTE IS TWO NUMBERS:
The note number - which key - and the velocity - how hard it was struck. 
midi_note_in gives you both on separate outlets, which is what you usually 
want: the number chooses something and the velocity scales it.

A note ENDING is a note message with velocity zero. That is a MIDI convention 
rather than a separate message, and it catches people out: if something is 
retriggering when it should be stopping, a velocity-zero note is being treated 
as a note-on.

POLY PRESSURE IS PER NOTE, AND RARE:
Ordinary aftertouch is one value for the whole keyboard - see the pitchbend and 
aftertouch help patch. Poly pressure is a separate value for EVERY held note, 
which is far more expressive and far less common. Most keyboards do not send 
it; the ones that do are usually worth the trouble.

If you are working with a controller that has it, it is the most direct 
continuous gesture MIDI offers - pressure on a key you are already holding, 
independent per finger.
""" + PORT + """
SYNTAX:
midi_note_in
midi_note_out

EXAMPLE:
midi_note_in

INPUTS and PARAMETERS:

port / channel:
Which device and which channel.

midi to send / velocity (midi_note_out):
The note number and how hard.

pressure (midi_poly_pressure_out):
The pressure for that note.

OUTPUTS: 

note out / velocity out:
Which note and how hard.

pressure out:
The per-note pressure.

RELATED:
midi_control_in handles knobs and faders, which is where most physical 
controllers put their continuous controls."""

demo = [
    {'key': 'ni', 'init': 'midi_note_in', 'pos': (30, 62), 'w': 240, 'h': 160},
    {'key': 'i1', 'init': 'int', 'pos': (30, 240), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'i2', 'init': 'int', 'pos': (180, 240), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c0', 'comment': True, 'text': 'note number and velocity\na note ENDING is velocity zero -\nnot a separate message', 'pos': (30, 295)},
    {'key': 'gt', 'init': '> 0', 'pos': (30, 395), 'w': 130, 'h': 70,
     'props': {'output_type': 'int'}},
    {'key': 'i3', 'init': 'int', 'pos': (30, 480), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c3', 'comment': True, 'text': 'so test velocity to tell them apart',
     'pos': (30, 530)},
    {'key': 'pp', 'init': 'midi_poly_pressure_in', 'pos': (330, 395), 'w': 280, 'h': 160},
    {'key': 'c4', 'comment': True, 'text': 'a separate value per held note -\nrare, and the most expressive MIDI has',
     'pos': (330, 570)},
]
links = [('ni', 'note out', 'i1', ''), ('ni', 'velocity out', 'i2', ''),
         ('ni', 'velocity out', 'gt', 'in'), ('gt', 'result', 'i3', '')]
print(build('midi_note_in', 'midi notes - what started, and how hard', body, demo,
            links, demo_width=640, text_width=790, text_height=700))

# ------------------------------------------------------------- midi_control_in
body = """These handle the continuous controls - knobs, faders, pedals - and program changes.

THE NODES:

midi_control_in    a controller's value arriving
midi_control_out   send a controller value
midi_program_in    a program change arriving
midi_program_out   send a program change

CONTROLLERS ARE NUMBERED, NOT NAMED:
Every knob and fader on a device sends a CONTROLLER NUMBER and a value from 0 
to 127. The number is how you tell one knob from another, and there is no 
naming - controller 74 is whatever the maker decided it should be on that 
device.

So the first job with any controller is finding the numbers. Patch midi_in into 
a print, move each control, write down what it reports. Then set 'controller #' 
on a midi_control_in per control you want.

Some numbers are conventional - 1 is the modulation wheel, 7 is volume, 64 is 
the sustain pedal - but conventions are not guarantees.

0 TO 127 IS THE WHOLE RANGE:
A MIDI controller value is a whole number in that range and nothing finer. 
That is 128 steps, which is audibly coarse on anything smoothly varying - a 
slow filter sweep driven directly from a MIDI knob steps.

The fix is downstream: put a filter or one_euro_filter after it and the steps 
become a smooth line. It costs a little lag and is almost always worth it for 
anything a listener will hear moving.

PROGRAM CHANGES SELECT, THEY DO NOT SET:
A program change says "use setting 12" rather than carrying any values. What 
setting 12 IS lives in the device. So a program change is the right thing for 
recalling a state a synth or effect already holds, and no use at all for 
telling it something new.
""" + PORT + """
SYNTAX:
midi_control_in <controller #>
midi_control_out <controller #>

EXAMPLE:
midi_control_in 1

INPUTS and PARAMETERS:

controller #:
Which knob or fader.

midi to send:
The value, 0 to 127.

program to send:
Which program.

port / channel:
Which device and channel.

OUTPUTS: 

out:
The controller's value, or the program number.

RELATED:
ranger maps 0 to 127 into whatever range you actually need, and can calibrate 
itself against the control's real travel."""

demo = [
    {'key': 'ci', 'init': 'midi_control_in 1', 'pos': (30, 62), 'w': 260, 'h': 180},
    {'key': 'i1', 'init': 'int', 'pos': (30, 260), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c0', 'comment': True, 'text': '0 to 127, in whole numbers', 'pos': (30, 315)},
    {'key': 'p0', 'init': 'plot', 'pos': (30, 355), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 127.0)},
    {'key': 'c1', 'comment': True, 'text': 'the steps are audible on a sweep',
     'pos': (30, 540)},
    {'key': 'flt', 'init': 'filter 0.9', 'pos': (30, 585), 'w': 160, 'h': 70,
     'props': {'degree': 0.9}},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 670), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 127.0)},
    {'key': 'c2', 'comment': True, 'text': 'smoothed: worth the small lag',
     'pos': (30, 855)},
    {'key': 'rg', 'init': 'ranger 0.0 127.0 0.0 1.0', 'pos': (300, 585), 'w': 240, 'h': 200,
     'props': {'input_min': 0.0, 'input_max': 127.0,
               'output_min': 0.0, 'output_max': 1.0, 'clamp': True}},
    {'key': 'c3', 'comment': True, 'text': 'and into the range you actually want',
     'pos': (300, 800)},
]
links = [('ci', '', 'i1', '', 0),
         ('ci', '', 'p0', 'y', 0),
         ('ci', '', 'flt', 'in', 0), ('flt', 'out', 'p1', 'y'),
         ('flt', 'out', 'rg', 'in')]
print(build('midi_control_in', 'midi controllers - knobs, faders, programs', body,
            demo, links, demo_width=580, text_width=790, text_height=720))

# ----------------------------------------------------------- midi_pitchbend_in
body = """These are the expression messages: bend and pressure applied to the whole instrument.

THE NODES:

midi_pitchbend_in    the bend wheel arriving
midi_pitchbend_out   send a bend
midi_aftertouch_in   channel pressure arriving
midi_aftertouch_out  send channel pressure

PITCH BEND IS THE ONE FINE CONTROL MIDI HAS:
Where a controller is 0 to 127, pitch bend is fourteen bits - over sixteen 
thousand steps - because a stepped pitch would be obviously wrong in a way a 
stepped filter is not. It is centred, so it runs from fully down through zero 
at rest to fully up.

That resolution makes it useful for things that are not pitch. If you need one 
smooth continuous value from a MIDI device and the controllers are too coarse, 
the bend wheel is the finest channel available.

CHANNEL AFTERTOUCH IS ONE VALUE FOR EVERYTHING:
Press harder on any key and the whole channel's aftertouch rises - it is not 
per note. Most keyboards send the maximum pressure across whatever is held, so 
it responds to your hardest finger rather than to any particular one.

That makes it a good gesture and a poor per-note control. When you want per 
note, that is poly pressure - see the notes help patch - and most keyboards do 
not have it.

Aftertouch also tends to have a dead zone at the bottom and to reach maximum 
well before the key does, so its useful range is narrower than 0 to 127. 
Watch the actual values before mapping it.
""" + PORT + """
SYNTAX:
midi_pitchbend_in
midi_aftertouch_in

EXAMPLE:
midi_pitchbend_in

INPUTS and PARAMETERS:

pitchbend to send / aftertouch to send:
The value going out.

port / channel:
Which device and channel.

OUTPUTS: 

out:
The bend or pressure value.

RELATED:
ranger will map the useful part of an aftertouch range onto a full one, and its 
calibrate switch will find that range for you - press as hard as you mean to 
while it is calibrating and it learns your actual maximum rather than the 
theoretical one."""

demo = [
    {'key': 'pb', 'init': 'midi_pitchbend_in', 'pos': (30, 62), 'w': 260, 'h': 160},
    {'key': 'p0', 'init': 'plot', 'pos': (30, 240), 'w': 208, 'h': 176,
     'props': PLOT(-8192.0, 8192.0)},
    {'key': 'c0', 'comment': True, 'text': 'fourteen bits, centred at rest\nthe finest continuous channel MIDI has',
     'pos': (30, 425)},
    {'key': 'at', 'init': 'midi_aftertouch_in', 'pos': (330, 62), 'w': 260, 'h': 160},
    {'key': 'i1', 'init': 'int', 'pos': (330, 240), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'one value for the whole channel,\nnot per note',
     'pos': (330, 295)},
    {'key': 'rg', 'init': 'ranger 20.0 100.0 0.0 1.0', 'pos': (330, 365), 'w': 240, 'h': 200,
     'props': {'input_min': 20.0, 'input_max': 100.0,
               'output_min': 0.0, 'output_max': 1.0, 'clamp': True,
               'calibrate': False}},
    {'key': 'c4', 'comment': True, 'text': 'tick calibrate, press as hard as you\nmean to, untick - it learns your range',
     'pos': (330, 580)},
]
links = [('pb', '', 'p0', 'y', 0),
         ('at', '', 'i1', '', 0), ('at', '', 'rg', 'in', 0)]
print(build('midi_pitchbend_in', 'midi expression - bend and pressure', body, demo,
            links, demo_width=620, text_width=790, text_height=700))

# ---------------------------------------------------------------------- mpd218
body = """A node that knows one particular piece of hardware: the Akai MPD218 pad
controller.

THE NODE:

mpd218   an Akai MPD218, with its pads working as a set of lit selectors

WHAT IT DOES:
It finds the MPD218 by name when it is created and connects to it in both
directions - listening to it, and talking back to it to light its pads.

The pads behave like radio buttons. Strike one and it lights, the pad lit
before it goes dark, and the pad comes out of 'pad'. Because only one pad is
ever lit, the device itself shows which choice is current - useful for picking
a scene, a preset or a section without looking at the screen.

'pad' sends the pad's NOTE NUMBER, exactly as the MPD218 sends it - which notes
the pads play is set in the device's own preset - not a pad count from 1 to 16.
Each strike sends once; releasing the lit pad only re-lights it.

The knobs come out of 'controller' as a pair: [controller number, value], with
the value 0 to 127 as the device sends it.

'select' does the same thing from the patch: send it a note number and the
node darkens the pads - note-offs for notes 36 to 83, the pad notes of the
factory presets' three banks, plus whichever note it lit last - then lights
that note's pad and sends the number out of 'pad', as though it had been struck.

WHEN THE DEVICE IS NOT THERE:
If no MIDI device whose name begins 'MPD218' is present, the node stays
unconnected and says so in the console - it never talks to some other device
in its place. Plug the MPD218 in and pick it in 'in port' and 'out port'. When
it is connected, creating the node darkens all its pads.

WHAT IT IGNORES:
Arguments - the device name is built in. The 'channel' option is shown but not
used: notes and knobs are accepted on any channel, and the lights are sent on
channel 1.

SYNTAX:
mpd218

EXAMPLE:
mpd218

INPUTS and PARAMETERS:

select:
A note number: light that pad, darken the rest, send it out of 'pad'.

in port / out port:
The MIDI ports, as menus. Both are set to the MPD218 automatically when it is
present.

channel (option):
Not used by this node.

OUTPUTS:

pad:
The note number of the pad struck (or selected), once per strike.

controller:
[controller number, value] for each knob movement.

RELATED:
blue_board, the other device-specific node, for the iRig BlueBoard foot
controller.
midi_note_in and midi_control_in, for any controller this node does not know."""

demo = [
    {'key': 'i0', 'init': 'int', 'pos': (30, 62), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c0', 'comment': True, 'text': 'select a pad from the patch',
     'pos': (330, 62)},
    {'key': 'mp', 'init': 'mpd218', 'pos': (30, 130), 'w': 280, 'h': 240},
    {'key': 'c1', 'comment': True,
     'text': 'strike a pad: it lights, the last one\ngoes dark - the device shows the choice',
     'pos': (330, 130)},
    {'key': 'i2', 'init': 'int', 'pos': (30, 400), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'the pad\'s note number, as the device sends it',
     'pos': (330, 400)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 470), 'w': 240, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': '[controller number, value] from the knobs',
     'pos': (330, 470)},
]
links = [('i0', 'int out', 'mp', 'select'),
         ('mp', 'pad', 'i2', ''),
         ('mp', 'controller', 'l1', '')]
print(build('mpd218', 'mpd218 - a pad controller that shows its choice', body,
            demo, links, demo_width=660, text_width=790, text_height=820))


# ------------------------------------------------------------------ blue_board
body = """A node that knows one particular piece of hardware: the iRig BlueBoard
foot controller.

THE NODE:

blue_board   an iRig BlueBoard - four foot buttons, four lights, over Bluetooth

blue_board IS FOR HANDS-FREE:
A foot controller matters when your hands are doing something else - which, for
a performer wearing a suit, is most of the time. Four buttons and four lights,
with the lights telling you the state you cannot see because you are not
looking at the patch.

It finds the BlueBoard by name when it is created and connects in both
directions: it hears the buttons and drives the lights.

A MODE PER BUTTON:
Each button has a menu beside its outlet, and setting it is the thing to get
right:

    momentary   1 while your foot is down, 0 when it lifts, and the light
                follows your foot. The default.
    toggle      each press flips between 1 and 0 and the light shows which;
                lifting your foot sends nothing. For something you turn on
                and walk away from.
    raw         1 down and 0 up, like momentary, but the node leaves the
                light alone.

THE LIGHTS FROM THE PATCH:
Beside each button is a checkbox inlet labelled LED - four of them, one per
button, in order A to D. Ticking one lights that button's lamp and unticking it
puts it out, so the patch can show a state the foot did not set. In momentary
and toggle modes it also sets the button's state, and if that changes, the new
state comes out of the button's outlet - so the light and the outlet never
disagree. In raw mode it only drives the light.

WHEN THE DEVICE IS NOT THERE:
If no MIDI device named 'iRig BlueBoard Bluetooth' (or beginning so) is
present, the node stays unconnected and says so in the console - it never
talks to some other device in its place; plug it in and pick it in 'in port'
and 'out port'. The buttons are heard as
controllers 20 to 23 on any channel, and the lights are sent as those
controllers on channel 1.

WHAT IT IGNORES:
Arguments - the device name is built in. The 'channel' option is shown but not
used.

SYNTAX:
blue_board

EXAMPLE:
blue_board

INPUTS and PARAMETERS:

LED (four of them, A to D):
Light or darken that button's lamp.

mode menus (one per button):
momentary, toggle or raw. momentary by default.

in port / out port:
The MIDI ports, as menus. Both are set to the BlueBoard automatically when it
is paired and present.

channel (option):
Not used by this node.

OUTPUTS:

A / B / C / D:
The four buttons, 1 or 0, as the mode decides.

RELATED:
mpd218, the other device-specific node, for the Akai MPD218 pad controller.
midi_control_in, for a foot controller this node does not know."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'c0', 'comment': True, 'text': 'lights button A\'s lamp from the patch',
     'pos': (330, 62)},
    {'key': 'bb', 'init': 'blue_board', 'pos': (30, 130), 'w': 280, 'h': 300},
    {'key': 'c1', 'comment': True,
     'text': 'set each button momentary, toggle or\nraw in the menu beside its outlet',
     'pos': (330, 130)},
    {'key': 'i1', 'init': 'int', 'pos': (30, 460), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'button A: 1 or 0',
     'pos': (330, 460)},
]
links = [('tog', '', 'bb', 'LED'), ('bb', 'A', 'i1', '')]
print(build('blue_board', 'blue_board - a foot controller with lights', body,
            demo, links, demo_width=660, text_width=790, text_height=820))
