"""ETC Eos lighting console over OSC."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """These drive an ETC Eos lighting console from a patch, over OSC.

THE NODES:

eos_console    the connection to the desk - one of these per console
color_source   intensity and colour for one channel, as sliders
eos_send       one named parameter of one channel

eos_console FIRST, AND ONLY ONE:
It is the connection, not a command. Everything else finds the desk by NAME, so
this node has to exist before the senders do anything - and there should be one
of it, however many senders you have.

Its defaults are the usual Eos arrangement: name 'eos', the desk at 10.1.3.11,
sending to port 1101 and listening on 1102. Change the address to your desk;
change the name only if you have more than one console and need to tell them
apart, in which case the senders' 'target name' has to match.

The listening port matters as much as the sending one. Eos replies - to say what
it did, and to report state - and those replies arrive on the source port and
come out of 'osc received'. A patch that sends fine but hears nothing has the
wrong source port.

WHAT AN EOS OSC ADDRESS LOOKS LIKE:
Commands are paths, and the shape of a channel parameter is:

    /eos/user/<user>/chan/<channel>/param/<name>

which is what color_source composes for you. The 'user' is the desk's user
number - 99 is the conventional choice for an external controller, because it
keeps what you send out of the way of whoever is sitting at the desk.

color_source IS THE CONVENIENT ONE:
Five sliders - intensity, red, green, blue, lime - all aimed at one channel,
each sending only when it moves. Set 'target channel' and it composes the rest.

It sends only the parameters that have actually changed, which matters on a
busy network: a patch nudging one colour should not be re-sending everything at
frame rate.

'lime' is there because many LED fixtures have a lime or lime-green emitter as
well as red, green and blue, and the extra emitter is what gives them a decent
white. A fixture without one simply ignores it.

eos_send IS THE GENERAL ONE:
One value, one named parameter, on one channel. 'parameter' is the parameter
name as Eos spells it - intens, red, pan, tilt, zoom, and so on - and 'min' and
'max' set the range of the input slider, so you can drive a 0-to-100 intensity
or a -270-to-270 pan with the same node.

Use it for anything color_source does not cover: position, beam, gobo, or a
parameter peculiar to one fixture.

CHANNEL NUMBERS ARE THE DESK'S, NOT THE FIXTURE'S:
Both senders address a CHANNEL as the console understands it - the number you
would type on the desk - not a DMX address and not a universe. If the desk has
been patched, those are different numbers, and the channel is the one that
matters here.

SYNTAX:
eos_console <name> <ip> <target port> <source port>
color_source <channel>
eos_send <parameter> <channel>

EXAMPLE:
color_source 7

INPUTS and PARAMETERS:

name / ip / target port / source port (eos_console):
Which desk, where, and the two ports. Defaults are name 'eos', 10.1.3.11,
1101 out and 1102 back.

intensity / red / green / blue / lime (color_source):
The parameters, 0 to 100. Each sends when it moves.

target channel:
Which channel on the desk.

address (color_source):
The command path before the channel. The default is the user-99 channel path.

osc to send (eos_send):
The value.

parameter (eos_send):
The Eos parameter name.

min / max (eos_send):
The range of the input.

target name:
Which eos_console to send through. Must match its name.

OUTPUTS: 

osc received (eos_console):
Whatever the desk sends back.

The senders have no outlets - they are ends of the chain.

RELATED:
osc_send and the osc nodes, if you want to talk to the desk in paths you compose
yourself rather than through these.
The eos_console node is an OSC device like any other, so anything that can
address an OSC target by name can use it."""

demo = [
    {'key': 'con', 'init': 'eos_console', 'pos': (30, 62), 'w': 300, 'h': 240},
    {'key': 'c0', 'comment': True, 'text': 'one of these, before anything else -\nthe senders find it by name\nset the ip to your desk. 1102 is where\nits replies come back',
     'pos': (30, 315)},
    {'key': 'td', 'init': 'text_display', 'pos': (380, 62), 'w': 320, 'h': 200,
     'props': {'width': 300, 'height': 160, 'wrap': True, 'max_lines': 60,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c4', 'comment': True, 'text': 'what the desk says back', 'pos': (380, 275)},

    {'key': 'cs', 'init': 'color_source 7', 'pos': (30, 470), 'w': 320, 'h': 320},
    {'key': 'c5', 'comment': True, 'text': 'five sliders at channel 7 - each sends\nonly when it moves\nlime is the fourth emitter on many LED\nfixtures - it is what makes a good white',
     'pos': (30, 805)},

    {'key': 'sig', 'init': 'signal', 'pos': (420, 470), 'w': 129, 'h': 78,
     'props': SIG('sin', 6.0, 135.0, True)},
    {'key': 'es', 'init': 'eos_send pan 7', 'pos': (420, 570), 'w': 300, 'h': 240,
     'props': {'min': -270, 'max': 270}},
    {'key': 'c9', 'comment': True, 'text': 'anything color_source does not cover:\npan, tilt, zoom, gobo - named as Eos\nspells it, with min/max to suit',
     'pos': (420, 825)},
]
links = [('con', 'osc received', 'td', '###text in'),
         ('sig', '', 'es', 'osc to send')]
print(build('eos_console', 'eos - driving a lighting desk', body,
            demo, links, demo_width=760, text_width=810, text_height=780))
