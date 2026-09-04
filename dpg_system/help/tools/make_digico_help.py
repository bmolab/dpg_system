"""DiGiCo console faders over OSC."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """A bank of channel faders on a DiGiCo console, over OSC.

THE NODE:

digico.fader   as many channel faders as you ask for

Each slider is one channel's fader, in decibels, running -80 to +10 - the range
a console fader actually covers, with 0 as unity and the top ten decibels of
gain above it.

IT WORKS BOTH WAYS, AND THAT IS THE POINT:
Move a slider here and the console moves. Move the fader on the DESK and the
slider here follows. The node sends and receives the same addresses, so the
patch and the console stay in step without you arranging it.

That matters for anything performed. An operator can take a fader by hand
mid-show and the patch knows where it now is, instead of fighting them or
carrying on from a stale value.

THE ADDRESSES ARE PLAIN AND ONE-BASED:

    /channel/1/fader
    /channel/2/fader   ... and so on

Fader 1 is channel 1. Anything else that speaks OSC can send those addresses
too, so the node is not required - it is a convenient bank of them.

THE FIRST INLET TAKES A WHOLE LIST:
Send a list of numbers to 'fader 1' and it sets EVERY fader from it, in order,
and sends them all. Send a single number and it does just that one.

    [-6, -7, -8, -9, -10, -11]  ->  all six channels set, six messages sent

That is how to recall a state - a mix stored in a dict, a shape from a signal
chain, a line from a file - without wiring twenty separate cords. The other
inlets take one value each.

Send it from a message node, not a string node. A message splits what you type
into separate numbers, which is what this inlet wants; a string sends the whole
line as one piece and you get a single fader set to nothing useful.

GETTING THE CONSOLE TALKING:
DiGiCo has no way to be asked what it has - there is no query. It only tells you
about a control when that control moves, or when you press 'Resend All' on the
console, which makes it announce everything at once.

So the sequence when setting up is: connect, press Resend All on the desk, and
the patch learns the current state. Until you do, the sliders here are at
whatever they were and the desk is wherever it is, and neither knows about the
other.

'target name' is the OSC device to send through - an osc_device or eos-style
node named the same. The count of faders is the second argument.

THESE FADERS ARE DISCOVERABLE:
The node registers itself in the OSCQuery registry, so oscq_browse can find the
faders by name and anything else speaking OSCQuery can drive them. A patch built
this way is controllable from a phone without further work.

SYNTAX:
digico.fader <target name> <fader count>

EXAMPLE:
digico.fader desk 20

INPUTS and PARAMETERS:

fader 1 ... fader N:
One channel each, in decibels. Fader 1 also accepts a list for all of them.

target name:
Which OSC device to send through.

OUTPUTS: 

None - the node sends to the console and updates its own sliders when the
console sends back.

RELATED:
osc_device and the osc nodes, for the connection itself.
oscq_browse, which will find these faders once they are registered.
eos_console if the thing to control is lighting rather than sound."""

demo = [
    {'key': 'df', 'init': 'digico.fader desk 8', 'pos': (30, 62), 'w': 320, 'h': 420},
    {'key': 'c0', 'comment': True, 'text': 'move a slider and the console moves.\nMove the fader on the DESK and this\nfollows - it sends and receives the\nsame addresses',
     'pos': (30, 500)},

    {'key': 'm1', 'init': 'message', 'pos': (400, 62), 'w': 420, 'h': 42,
     'props': {'text in': '-6 -7 -8 -9 -10 -11 -12 -13', 'font size': '24'}},
    {'key': 'c4', 'comment': True, 'text': 'a whole list into fader 1 sets them all\nin order - how to recall a stored mix\nwithout twenty separate cords\na message node, NOT a string: this inlet\nwants a list, and string sends one piece',
     'pos': (400, 112)},

    {'key': 'dev', 'init': 'osc_device', 'pos': (400, 285), 'w': 320, 'h': 220},
    {'key': 'c7', 'comment': True, 'text': "'target name' must match this device",
     'pos': (400, 575)},

    {'key': 'c8', 'comment': True, 'text': "DiGiCo cannot be ASKED what it has -\nthere is no query. Press 'Resend All'\non the console and it announces\neverything at once - that is how the\npatch learns the current state\nthe faders register themselves, so\noscq_browse can find them and a phone\ncan drive them with no extra work",
     'pos': (400, 575)},
]
links = [('m1', 'message out', 'df', 'fader 1')]
print(build('digico.fader', 'digico.fader - a console fader bank', body,
            demo, links, demo_width=860, text_width=800, text_height=740))
