"""ETC Eos lighting console over OSC."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """These drive an ETC Eos lighting console from a patch, over OSC.

THE NODES:

eos_console    the connection to the desk - one of these per console
eos_send       one named parameter, sent to one or more channels
eos_int        the same, and eos_float / eos_slider / eos_knob / eos_toggle -
               one node per kind of input

eos_console FIRST, AND ONLY ONE:
It is the connection, not a command. Everything else finds the desk by NAME, so
this node has to exist before the senders do anything - and there should be one
of it, however many senders you have.

Its defaults are the usual Eos arrangement: name 'eos', the desk at 10.1.3.11,
sending to port 1101 and listening on 1102. They apply only when it is created
with no arguments at all - give any, and give all four. Change the address to
your desk; change the name only if you have more than one console and need to
tell them apart, in which case the senders' 'target name' has to match.

The listening port matters as much as the sending one. Eos replies - to say what
it did, and to report state - and those replies arrive on the source port. Any
that no osc_receive has claimed come out of 'osc received' as [address, [values]].
A patch that sends fine but hears nothing has the wrong source port.

It is an osc_device underneath, so 'osc to send' also takes a raw message - a
list, address first and then the values - for anything the senders do not
cover.

WHAT AN EOS OSC ADDRESS LOOKS LIKE:
Commands are paths, and the shape of a channel parameter is:

    /eos/user/<user>/chan/<channel>/param/<name>

which is what eos_send composes for you. The 'user' is the desk's user number -
99 is the conventional choice for an external controller, because it keeps
what you send out of the way of whoever is sitting at the desk.

eos_send:
One value, one named parameter. 'parameter' is the parameter name as Eos spells
it - intens, red, pan, tilt, zoom, and so on - and 'min' and 'max' set the range
of the input widget. The range starts at 0 to 100, or -360 to 360 when the
parameter is pan or tilt, so the same node drives an intensity or a pan.

Each value is sent when it changes - when the widget moves or a value arrives -
to every channel in the list.

The path it composes is shown in full under the inputs - user, channel and
parameter, exactly as it will go out - so a parameter that does nothing can be
checked against what the desk expects without a sniffer. 'user' is an option,
99 by default.

MANY CHANNELS FROM ONE NODE:
'target channel' is a channel LIST, not one number. Ten fixtures in channels 1
to 10 are '1-10' or '1 thru 10'; the odd ones are '1 3 5 7 9' or '1,3,5,7,9';
and the forms mix, so '1-4 7 9-10' is fine too. Each value goes out once per
channel, because the desk's parameter path takes exactly one channel - the
readout under the inputs shows the list compacted and says how many messages
that is.

ONE NODE PER KIND OF INPUT:
eos_send takes whole numbers in a drag box. The same node under other names
gives the other inputs, and Eos is happy with fractional values, so pick
whichever suits the hand:

    eos_int      a drag box of whole numbers (what eos_send is)
    eos_float    a drag box of fractional values
    eos_slider   a slider
    eos_knob     a knob
    eos_toggle   a checkbox - sends 'max' when on and 'min' when off, so with
                 the defaults it is full and out for an intensity

They differ only in the input widget; parameter, channel, user and the target
are the same on all of them.

CHANNEL NUMBERS ARE THE DESK'S, NOT THE FIXTURE'S:
The senders address a CHANNEL as the console understands it - the number you
would type on the desk - not a DMX address and not a universe. If the desk has
been patched, those are different numbers, and the channel is the one that
matters here.

SYNTAX:
eos_console <name> <ip> <target port> <source port>
eos_send <parameter> <channels>
eos_int / eos_float / eos_slider / eos_knob / eos_toggle <parameter> <channels>

EXAMPLE:
eos_slider intens 1-10

INPUTS and PARAMETERS:

osc to send (eos_console):
A raw OSC message: a list, address first, then the values.

name / ip / target port / source port (eos_console):
Which desk, where, and the two ports. Defaults are name 'eos', 10.1.3.11,
1101 out and 1102 back.

handle in main loop (eos_console option):
Pass incoming messages on from the app's main loop rather than the network
thread.

target channel (eos_send and family):
The channel list - '1-10', '1 3 5', '1,3,5', '1 thru 10' - and one message goes
out per channel. '1' by default.

osc to send (eos_send and family):
The value. On eos_toggle, on sends 'max' and off sends 'min'.

parameter (eos_send and family):
The Eos parameter name.

user (eos_send and family):
The desk user number in the path. 99 by default.

min / max (eos_send and family):
The range of the input.

The line under the inputs is the composed path, for reading only.

target name (eos_send and family, option):
Which eos_console to send through. 'eos' by default; must match its name.

OUTPUTS: 

osc received (eos_console):
What the desk sends back, as [address, [values]], for any message no
osc_receive has claimed.

The senders have no outlets - they are ends of the chain.

RELATED:
osc_send and the osc nodes, if you want to talk to the desk in paths you compose
yourself rather than through these.
The eos_console node is an OSC device like any other, so anything that can
address an OSC target by name can use it.
color_source, five sliders - intensity, red, green, blue, lime - for one
channel, which used to share this page."""

demo = [
    {'key': 'con', 'init': 'eos_console', 'pos': (30, 62), 'w': 300, 'h': 160},
    {'key': 'c0', 'comment': True, 'text': 'one of these, before anything else -\nthe senders find it by name\nset the ip to your desk. 1102 is where\nits replies come back',
     'pos': (400, 62)},
    {'key': 'td', 'init': 'text_display', 'pos': (30, 260), 'w': 320, 'h': 200,
     'props': {'width': 300, 'height': 160, 'wrap': True, 'max_lines': 60,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c4', 'comment': True, 'text': 'what the desk says back', 'pos': (400, 260)},

    {'key': 'sl', 'init': 'eos_slider intens 1-10', 'pos': (30, 500), 'w': 320, 'h': 100},
    {'key': 'c10', 'comment': True, 'text': 'one slider, ten channels - the readout\nshows the list compacted and how many\nmessages each move sends',
     'pos': (400, 500)},

    {'key': 'sig', 'init': 'signal', 'pos': (30, 640), 'w': 129, 'h': 78,
     'props': SIG('sin', 6.0, 135.0, True)},
    {'key': 'c8', 'comment': True, 'text': 'a slow sweep, plus and minus 135',
     'pos': (400, 640)},
    {'key': 'es', 'init': 'eos_send pan 7', 'pos': (30, 750), 'w': 300, 'h': 100,
     'props': {'min': -270, 'max': 270}},
    {'key': 'c9', 'comment': True, 'text': 'pan, tilt, zoom, gobo - named as Eos\nspells it, with min/max to suit',
     'pos': (400, 750)},
]
links = [('con', 'osc received', 'td', '###text in'),
         ('sig', '', 'es', 'osc to send')]
print(build('eos_console', 'eos - driving a lighting desk', body,
            demo, links, demo_width=760, text_width=810, text_height=800))


# ---------------------------------------------------------------- color_source
body = """Intensity and colour for one lighting channel, as five sliders, sent to an
ETC Eos desk over OSC.

THE NODE:

color_source   intensity, red, green, blue and lime for one Eos channel

WHAT IT DOES:
Five sliders - intensity, red, green, blue, lime - each 0 to 100, all aimed at
one channel. Move one and it sends that parameter to the desk:

    /eos/user/99/chan/<channel>/param/intens
    /eos/user/99/chan/<channel>/param/red      ... and so on

It sends only the parameters that have actually changed, at most once per
frame, which matters on a busy network: a patch nudging one colour should not
be re-sending everything at frame rate.

'lime' is there because many LED fixtures have a lime or lime-green emitter as
well as red, green and blue, and the extra emitter is what gives them a decent
white. A fixture without one simply ignores it.

IT NEEDS THE CONSOLE NODE:
It sends through the OSC target named in 'target name' - 'eos' by default,
which is eos_console's default name. With no such target in the patch, nothing
is sent; changes wait and go out once it appears.

ONE CHANNEL:
'target channel' is a single channel number, as the desk understands it - the
number you would type on the desk, not a DMX address. For one colour across
many channels, use eos_send's family with a channel list instead.

THE ADDRESS:
'address' is the part of the path before the channel, '/eos/user/99/chan' by
default. Change the 99 to send as another desk user.

SYNTAX:
color_source <channel>
color_source <target name> <channel>

EXAMPLE:
color_source 7

INPUTS and PARAMETERS:

intensity / red / green / blue / lime:
The five parameters, whole numbers 0 to 100, 0 at the start. Each sends when it
changes, whether moved by hand or set by a message.

target name:
Which eos_console (or other OSC target) to send through. 'eos' by default.

address:
The path before the channel. '/eos/user/99/chan' by default.

target channel:
The desk channel. The number argument, or 1.

OUTPUTS:

None - it sends to the desk.

RELATED:
eos_console, the connection this sends through, and eos_send, for any other
parameter and for channel lists. Both used to share this page."""

demo = [
    {'key': 'con', 'init': 'eos_console', 'pos': (30, 62), 'w': 300, 'h': 160},
    {'key': 'c0', 'comment': True, 'text': 'the desk connection, named eos -\nset its ip to your desk',
     'pos': (400, 62)},
    {'key': 'cs', 'init': 'color_source 7', 'pos': (30, 260), 'w': 320, 'h': 320},
    {'key': 'c1', 'comment': True, 'text': 'five sliders at channel 7 - each sends\nonly when it moves\nlime is the fourth emitter on many LED\nfixtures - it is what makes a good white',
     'pos': (400, 260)},
    {'key': 'c2', 'comment': True, 'text': 'for other parameters, or many channels: eos_send',
     'pos': (30, 610)},
]
links = []
print(build('color_source', 'color_source - colour for one lighting channel', body,
            demo, links, demo_width=760, text_width=810, text_height=800))
