"""Controlling a projector over PJLink."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """This controls a projector over the network, using PJLink.

THE NODE:

pjlink_projector    one projector. 'projector_control' is the same node
projector_control   the other spelling

PJLINK IS A STANDARD, WHICH IS THE WHOLE POINT:
Nearly every projector made for installation speaks it, whatever the badge on
the front - Panasonic, Epson, Christie, Barco. So a patch written against this
node keeps working when the venue changes the projector, which is not something
you can say about a manufacturer's own protocol.

It always uses port 4352.

CONNECT FIRST, AND WATCH 'status':
'connect' is a switch rather than a button: ticking it opens the socket and
exchanges the handshake, and unticking it closes the connection again. It starts
off, so nothing reaches the network until you ask.

What happened is reported in 'status', and that is the thing to look at when a
command does nothing - a projector that is not connected simply ignores you
rather than complaining.

THE PASSWORD IS PART OF THE HANDSHAKE, NOT SENT SEPARATELY:
On connecting, the projector says whether it wants authentication. If it does,
it sends a one-time seed, and the node answers with a hash of that seed and the
password combined. The password itself never crosses the network.

Practically: get the password right in the options before connecting, and if the
handshake reports something unexpected the node refuses to mark itself connected
rather than letting every later command fail quietly.

THE COMMANDS:

power_on        on or off. This is the slow one - see below
shutter_mute    black the picture without turning anything off
freeze          hold the current frame
volume          if the projector has sound
input_code      which input to show

POWER IS SLOW, AND SHUTTER IS NOT:
A projector takes tens of seconds to warm up, and often will not turn off again
until it has cooled - during which it ignores commands, including being turned
back on. So power is not something to put under live control.

'shutter_mute' is what you want for anything performed. It blacks the output
instantly and reversibly, the lamp stays where it is, and there is no penalty
for doing it repeatedly. Black the shutter between cues; leave the power alone.

'freeze' holds the last frame rather than blacking it, which is the other useful
one - it lets you change what the computer is doing without the audience seeing
the change happen.

THE INPUT CODES ARE THE STANDARD'S, NOT THE PROJECTOR'S LABELS:

RGB 1  11     Video 1  21     HDMI 1  31     Storage  41
RGB 2  12     Video 2  22     HDMI 2  32     Network  51
                             HDMI 3  33

A projector will only have some of these, and its front-panel labels may not
match - what a menu calls "Computer 1" is RGB 1 here. If an input change appears
to do nothing, the projector most likely does not have that input rather than
the command having failed.

'custom_cmd' FOR ANYTHING NOT COVERED:
PJLink has more commands than this node exposes, and manufacturers add their
own. Type a raw command and it is sent as typed - the class prefix is part of
it, so a query looks like '%1POWR ?' and a set like '%1AVMT 31'.

'response' carries whatever comes back, so a query is how you ask the projector
about itself - lamp hours, error status, what model it is.

'print_debug' puts the handshake and the traffic in the console, which is the
first thing to turn on when a projector will not answer.

SYNTAX:
pjlink_projector <ip> <password>
projector_control <ip> <password>

EXAMPLE:
pjlink_projector 10.1.1.141

INPUTS and PARAMETERS:

connect:
A switch. On opens the connection, off closes it. Nothing works until it
succeeds.

power_on:
On or off. Slow, and ignores you while cooling.

shutter_mute:
Black the picture. Instant, and the one to use in performance.

freeze:
Hold the current frame.

volume / input_code:
Sound level, and which input.

ip / port / password:
Where the projector is, and how to authenticate. Port 4352 unless something
unusual is going on.

custom_cmd:
A raw PJLink command, sent as typed.

print_debug:
Put the traffic in the console.

OUTPUTS: 

response:
What the projector said back.

RELATED:
visca_camera for cameras, which is the same kind of job over a different
protocol.
eos_console if the thing to control is lighting.
nvx nodes for routing what the projector is showing."""

demo = [
    {'key': 'pj', 'init': 'pjlink_projector 10.1.1.141', 'pos': (30, 62),
     'w': 340, 'h': 320},
    {'key': 'c0', 'comment': True, 'text': 'set ip and password in the options,',
     'pos': (30, 400)},
    {'key': 'c1', 'comment': True, 'text': 'THEN connect. Watch status - an',
     'pos': (30, 430)},
    {'key': 'c2', 'comment': True, 'text': 'unconnected projector just ignores you',
     'pos': (30, 460)},

    {'key': 'td', 'init': 'text_display', 'pos': (420, 62), 'w': 340, 'h': 220,
     'props': {'width': 320, 'height': 180, 'wrap': True, 'max_lines': 100,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'what it says back - queries answer here',
     'pos': (420, 295)},

    {'key': 'tog', 'init': 'toggle', 'pos': (30, 510), 'w': 45, 'h': 42},
    {'key': 'c4', 'comment': True, 'text': 'shutter: instant, reversible, free.',
     'pos': (90, 510)},
    {'key': 'c5', 'comment': True, 'text': 'This is the one to use in performance',
     'pos': (90, 540)},

    {'key': 'met', 'init': 'metro 2000', 'pos': (30, 590), 'w': 129, 'h': 70,
     'props': {'on': False, 'period': 2000.0, 'units': 'milliseconds'}},
    {'key': 'q', 'init': 'string', 'pos': (30, 675), 'w': 260, 'h': 42,
     'props': {'text in': '%1POWR ?', 'font size': '24', 'width': 220}},
    {'key': 'c6', 'comment': True, 'text': 'a raw query, sent as typed - this is',
     'pos': (30, 725)},
    {'key': 'c7', 'comment': True, 'text': 'how you ask about lamp hours, errors,',
     'pos': (30, 755)},
    {'key': 'c8', 'comment': True, 'text': 'or which model it actually is',
     'pos': (30, 785)},
    {'key': 'c9', 'comment': True, 'text': 'leave POWER alone in performance - it',
     'pos': (30, 825)},
    {'key': 'c10', 'comment': True, 'text': 'takes tens of seconds and ignores you',
     'pos': (30, 855)},
    {'key': 'c11', 'comment': True, 'text': 'entirely while it cools',
     'pos': (30, 885)},
]
links = [('pj', 'response', 'td', '###text in'),
         ('tog', '', 'pj', 'shutter_mute'),
         ('met', '', 'q', '###text in')]
print(build('pjlink_projector', 'pjlink_projector - a projector on the network',
            body, demo, links, demo_width=800, text_width=810, text_height=760))
