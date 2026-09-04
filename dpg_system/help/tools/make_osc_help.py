"""OSC transport, messages, widgets, query, and the pipo sensors."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ----------------------------------------------------------------- osc_source
body = """These are the connections: the sockets OSC arrives on and leaves by.

OSC is how this patch talks to other machines and other software - a lighting 
desk, a sound console, a phone, another instance of this system. A message is 
an ADDRESS, which looks like a file path, and some values.

THE NODES:

osc_source        listen on a port
osc_source_async  the same, on its own thread
osc_target        send to an address and port
osc_device        a source and a target together, for something you both send 
                  to and hear from
osc_manager       the registry: what sources and targets exist

SOURCES AND TARGETS ARE NAMED:
You create a source or a target once, give it a name, and everything else in 
the patch refers to it by that name rather than by address and port. 
osc_receive and osc_send take a 'source name' or 'target name' - so moving a 
console to a different IP means changing one node, not twenty.

osc_device IS THE COMMON CASE:
Most things you talk to also talk back. A device is one node holding both 
directions, with the target port and the source port separate because they 
usually differ.

CHOOSING A SOURCE TYPE:
osc_source is the plain one. osc_source_async runs its own event loop, which 
matters when messages arrive faster than the patch's frame rate - a motion 
source at 200 Hz will otherwise be handled in bursts.

Both have a 'handle in main loop' or 'use_queue' option, and this is the 
setting to know about. An OSC message arriving on a network thread can trigger 
nodes on that thread, which is unsafe for anything doing GL work or touching 
widgets. Queueing hands them to the main loop instead. If a patch crashes or 
behaves strangely only when OSC is flowing, turn this on.

SYNTAX:
osc_source <name> <port>
osc_target <name> <ip> <port>
osc_device <name> <ip> <target port> <source port>

EXAMPLE:
osc_target console 192.168.1.50 8000

INPUTS and PARAMETERS:

name:
What this connection is called. Everything else refers to it by this.

ip / port / target port / source port:
Where to send, and what to listen on.

osc to send:
Messages out.

use_queue / handle in main loop:
Hand incoming messages to the main loop rather than acting on the network 
thread. See above.

sources (osc_manager):
The registry of what exists.

OUTPUTS: 

osc received:
Everything arriving on this connection, address and all.

RELATED:
osc_receive filters that stream down to one address. 
osc_send is the tidier way to send to a named target."""

demo = [
    {'key': 'tg', 'init': 'osc_target console 127.0.0.1 8000', 'pos': (30, 62),
     'w': 280, 'h': 220},
    {'key': 'c0', 'comment': True, 'text': 'named once; everything else refers to\nit by name, not by address',
     'pos': (30, 300)},
    {'key': 'sr', 'init': 'osc_source listener 8001', 'pos': (30, 375), 'w': 280, 'h': 220},
    {'key': 'c2', 'comment': True, 'text': 'tick use_queue if OSC drives anything\nthat draws or touches widgets',
     'pos': (30, 610)},
    {'key': 'pr', 'init': 'print osc', 'pos': (30, 685), 'w': 180, 'h': 120,
     'props': {'identifier': 'osc', 'precision': 3}},
    {'key': 'om', 'init': 'osc_manager', 'pos': (350, 375), 'w': 260, 'h': 200},
    {'key': 'c4', 'comment': True, 'text': 'what sources and targets exist',
     'pos': (350, 590)},
]
links = [('sr', 'osc received', 'pr', 'in')]
print(build('osc_source', 'osc_source and osc_target - the connections', body, demo,
            links, demo_width=630, text_width=800, text_height=720))

# ---------------------------------------------------------------- osc_receive
body = """These send and receive individual messages, by address.

An OSC address looks like a file path - /fader/1/level, /body/left_hand/x. 
A connection carries all of them mixed together; these three pick out the ones 
you want and send the ones you mean.

THE NODES:

osc_receive  everything arriving at one address
osc_send     send to one address on a named target
osc_route    split an incoming stream by address, one outlet each

osc_route IS THE USEFUL ONE:
Give it a list of addresses and it grows an outlet for each, sending each 
message out of its own. Anything that matches nothing goes to 'unmatched' - 
and wiring that up is worth doing, because it is how you discover what a device 
is actually sending as opposed to what its documentation says.

That is the normal way to handle a device: one source, one route, and a branch 
per thing you care about. The alternative - an osc_receive per address - works 
but scales badly and hides the shape of what is arriving.

THROTTLING:
osc_receive has a 'throttle (ms)' setting, which limits how often it passes a 
message on. Some devices send continuously at a rate far beyond what anything 
downstream needs; throttling there is cheaper than filtering afterwards, and it 
keeps the patch's frame from being spent on messages it will discard.

SYNTAX:
osc_receive <source name> <address>
osc_send <target name> <address>
osc_route <address> <address> ...

EXAMPLE:
osc_route /fader /button /encoder

INPUTS and PARAMETERS:

source name / target name:
Which connection, by the name you gave it.

address:
The OSC address. On osc_receive this is what to listen for; on osc_send it is 
where to send.

in (osc_route):
The stream to split - patch it from a source's 'osc received'.

throttle (ms) (osc_receive):
Minimum time between messages passed on.

type (osc_send):
How to type the outgoing values.

OUTPUTS: 

osc received:
Messages at that address.

one outlet per address (osc_route):
The matching messages.

unmatched:
Everything that matched nothing. Wire it up.

A NOTE ON ADDRESS MATCHING:
OSC addresses are hierarchical, and routing on a prefix catches everything 
below it - /fader catches /fader/1/level. That is usually what you want, and 
it means the order you list addresses in matters when one is a prefix of 
another."""

demo = [
    {'key': 'sr', 'init': 'osc_source listener 8001', 'pos': (30, 62), 'w': 280, 'h': 220},
    {'key': 'rt', 'init': 'osc_route /fader /button', 'pos': (30, 305), 'w': 280, 'h': 160},
    {'key': 'c0', 'comment': True, 'text': 'one outlet per address', 'pos': (30, 480)},
    {'key': 'f1', 'init': 'float', 'pos': (30, 520), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'i1', 'init': 'int', 'pos': (180, 520), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'pr', 'init': 'print unmatched', 'pos': (30, 585), 'w': 200, 'h': 120,
     'props': {'identifier': 'unmatched', 'precision': 3}},
    {'key': 'c1', 'comment': True, 'text': 'wire unmatched up: it tells you what\nthe device is really sending',
     'pos': (30, 720)},
    {'key': 'sl', 'init': 'slider 0.0', 'pos': (350, 305), 'w': 220, 'h': 60,
     'props': {'min': 0.0, 'max': 1.0, 'format': '%.2f', 'width': 200}},
    {'key': 'sn', 'init': 'osc_send console /fader/1', 'pos': (350, 380), 'w': 280, 'h': 180},
    {'key': 'c3', 'comment': True, 'text': 'and out to a named target', 'pos': (350, 575)},
]
links = [('sr', 'osc received', 'rt', 'in'),
         ('rt', 'unmatched', 'pr', 'in'),
         ('sl', 'float out', 'sn', 'osc to send')]
print(build('osc_receive', 'osc_receive, osc_send, osc_route - by address', body,
            demo, links, demo_width=660, text_width=800, text_height=720))

# ------------------------------------------------------------------ osc_float
body = """These are ordinary widgets that send and receive OSC by themselves.

A slider wired to an osc_send does the same job. What these save is the wiring, 
and what they add is the two-way behaviour: the widget both sends when you move 
it AND moves when a message arrives at its address. So the patch and the remote 
surface stay in step without you arranging it.

THE NODES:

osc_float    osc_int     osc_slider   osc_knob
osc_string   osc_message osc_button   osc_toggle
osc_menu     osc_radio   osc_vector

Each behaves like the plain widget of the same name - see the float, slider, 
button and menu help patches for what they do as controls. What follows is 
only what the osc_ prefix adds.

THE TWO SETTINGS THAT MATTER:
'path' is the OSC address this widget lives at. 'target name' is which 
connection it sends on - the name you gave an osc_target or osc_device.

Set those two and the widget is on the network. Nothing else is needed.

WHY TWO-WAY MATTERS:
A control surface that shows the wrong value is worse than none, because 
someone will act on it. If the patch changes a level and the remote fader stays 
where it was, the next touch of that fader jumps the level back. These widgets 
send and receive at the same address, so the surface follows whatever moved it - 
the patch, a person, or another controller.

osc_vector IS FOR THE MULTI-VALUE CASE:
A position, a colour, a set of weights - several numbers under one address. 
'rows' and 'columns' shape it, so a grid of values is one node and one address 
rather than a node each.

'mode' AND 'address':
The widgets with a 'mode' option can send their value in more than one form - 
as the address's argument, or with the selection encoded into the address 
itself, which is what some devices expect from a menu or a radio group. 
If a remote device ignores a menu that looks correct, this is the setting.

SYNTAX:
osc_slider <path>
osc_float <path>

EXAMPLE:
osc_slider /fader/1

INPUTS and PARAMETERS:

path:
The OSC address.

target name:
Which connection to send on.

mode / address:
How the value is encoded, on the widgets that offer a choice.

rows / columns (osc_vector):
The shape of the value.

Everything else - min, max, format, width, font size, bind to - works as it 
does on the plain widget.

OUTPUTS: 

out:
The value, as an ordinary patch message, whether it came from the widget or 
from the network.

RELATED:
The param_ widgets carry a name without the networking. 
osc_send and osc_receive do the same job explicitly when you want the messages 
visible in the patch."""

demo = [
    {'key': 'tg', 'init': 'osc_target console 127.0.0.1 8000', 'pos': (30, 62),
     'w': 280, 'h': 220},
    {'key': 'os', 'init': 'osc_slider /fader/1', 'pos': (30, 305), 'w': 280, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'set path and target name; that is all',
     'pos': (30, 520)},
    {'key': 'f1', 'init': 'float', 'pos': (30, 560), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'it moves when a message arrives too',
     'pos': (30, 610)},
    {'key': 'ot', 'init': 'osc_toggle /mute/1', 'pos': (350, 305), 'w': 280, 'h': 200},
    {'key': 'i1', 'init': 'int', 'pos': (350, 520), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'ov', 'init': 'osc_vector /colour', 'pos': (350, 585), 'w': 280, 'h': 220},
    {'key': 'c2', 'comment': True, 'text': 'several numbers, one address', 'pos': (350, 820)},
]
links = [('os', 'float out', 'f1', ''), ('ot', '', 'i1', '')]
print(build('osc_float', 'osc widgets - controls that are already on the network',
            body, demo, links, demo_width=670, text_width=800, text_height=760))

# ------------------------------------------------------------- osc_query_json
body = """Two nodes for finding out what a device offers, and for driving a cue list.

THE NODES:

osc_query_json  print a device's OSCQuery description
osc_cue         send a cue number to a target

osc_query_json AND THE DISCOVERY PROBLEM:
OSC on its own is write-only in a sense: you can send to an address, but 
nothing tells you what addresses exist. Every integration therefore begins with 
finding out what the far end responds to, and that is usually done by reading 
documentation that is out of date.

OSCQuery is the answer to that - a device publishes a description of its whole 
namespace, and this node prints it. What you get back is the actual, current 
list of addresses, their types and their ranges, from the device itself.

That turns "which address is the fader" into something you look up rather than 
guess, and it is the first thing to reach for with an unfamiliar device.

Note that not every device offers OSCQuery. When one does not, the way to learn 
its namespace is the reverse: patch an osc_route with nothing matched and watch 
'unmatched' while you touch things on the device.

osc_cue:
Sends a cue number to a named target. Cue lists are how lighting and sound 
consoles are actually operated, so the useful interface to one is usually not 
a fader per parameter but a cue number - the console already holds the states, 
and the patch just says which one.

SYNTAX:
osc_query_json
osc_cue

EXAMPLE:
osc_cue

INPUTS and PARAMETERS:

print osc query json:
Fetch and print the description.

cue # to send:
The cue number.

target name:
Which connection to send it on.

path:
The address, where it differs from the default.

OUTPUTS: 

None - both act rather than pass data on. The query prints to the console the 
patch was launched from.

RELATED:
There are OSCQuery proxies in this repo for consoles that do not offer one 
themselves - see eos_oscquery_proxy.py and digico_oscquery_proxy.py, which 
expose a Gio and an S21 respectively."""

demo = [
    {'key': 'dv', 'init': 'osc_device console 127.0.0.1 8000 8001', 'pos': (30, 62),
     'w': 300, 'h': 260},
    {'key': 'qj', 'init': 'osc_query_json', 'pos': (30, 345), 'w': 260, 'h': 120},
    {'key': 'c0', 'comment': True, 'text': 'prints the whole namespace of the device\nto the console, not into the patch',
     'pos': (30, 480)},
    {'key': 'i1', 'init': 'int', 'pos': (30, 555), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'cu', 'init': 'osc_cue', 'pos': (30, 615), 'w': 260, 'h': 140},
    {'key': 'c2', 'comment': True, 'text': 'the console holds the states;\nthe patch just says which one', 'pos': (30, 770)},
]
links = [('i1', 'int out', 'cu', 'cue # to send')]
print(build('osc_query_json', 'osc_query_json - finding out what a device offers',
            body, demo, links, demo_width=620, text_width=790, text_height=700))

# ------------------------------------------------------------------ pipo_motion
body = """Two nodes that read a phone's sensors over the network.

THE NODES:

pipo_motion  orientation and acceleration
pipo_range   a distance reading

These take data from a phone running a sensor-streaming app, over OSC, on a 
port you set. That makes a phone into a cheap and immediately available motion 
sensor - useful for testing a patch without setting up a suit, for a second 
performer, or for putting a sensor somewhere a suit will not go.

WHAT YOU GET:
pipo_motion gives yaw, pitch and roll as separate outlets, plus acceleration. 
Those are Euler angles rather than a quaternion, which means they gimbal-lock 
and are awkward to do arithmetic on - see the rotation conversions help patch 
for why, and convert with euler_to_quaternion before doing anything but 
displaying them.

The acceleration outlet is the more directly useful of the two for movement 
work, because it does not depend on a heading and so is not subject to 
magnetic error.

pipo_range gives a distance, from whatever ranging sensor the phone offers.

SYNTAX:
pipo_motion
pipo_range

EXAMPLE:
pipo_motion

INPUTS and PARAMETERS:

port:
The port to listen on. Set the phone to send there.

OUTPUTS: 

yaw / pitch / roll:
Orientation, as angles.

acc:
Acceleration.

dist (pipo_range):
The distance reading.

WHAT TO EXPECT OF THE DATA:
A phone's orientation comes from the same kind of sensor fusion a suit uses, so 
it has the same weakness: the heading depends on the magnetic field and is 
wrong near steel. The pitch and roll come from gravity and are dependable; 
the yaw is the one to distrust. That is the same story as the mag_offset help 
patch tells for the suit, and the same reasoning applies."""

demo = [
    {'key': 'pm', 'init': 'pipo_motion', 'pos': (30, 62), 'w': 260, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'set the port the phone sends to',
     'pos': (30, 275)},
    {'key': 'f1', 'init': 'float', 'pos': (30, 315), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'f2', 'init': 'float', 'pos': (180, 315), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'f3', 'init': 'float', 'pos': (330, 315), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'yaw, pitch, roll - pitch and roll are\ndependable; yaw is the magnetic one',
     'pos': (30, 370)},
    {'key': 'pk', 'init': 'pack 3', 'pos': (30, 445), 'w': 140, 'h': 100},
    {'key': 'eq', 'init': 'euler_to_quaternion', 'pos': (30, 560), 'w': 260, 'h': 120,
     'props': {'degrees': True}},
    {'key': 'c3', 'comment': True, 'text': 'convert before doing anything but showing',
     'pos': (30, 695)},
    {'key': 'p1', 'init': 'plot', 'pos': (330, 445), 'w': 208, 'h': 176,
     'props': PLOT(-2.0, 2.0)},
    {'key': 'c4', 'comment': True, 'text': 'acceleration needs no heading,\nso no magnetic error',
     'pos': (330, 630)},
]
links = [('pm', 'yaw', 'f1', ''), ('pm', 'pitch', 'f2', ''), ('pm', 'roll', 'f3', ''),
         ('pm', 'yaw', 'pk', 'in 1'), ('pm', 'pitch', 'pk', 'in 2'),
         ('pm', 'roll', 'pk', 'in 3'),
         ('pk', 'out', 'eq', 'xyz rotation'),
         ('pm', 'acc', 'p1', 'y')]
print(build('pipo_motion', 'pipo_motion - a phone as a sensor', body, demo, links,
            demo_width=580, text_width=790, text_height=680))
