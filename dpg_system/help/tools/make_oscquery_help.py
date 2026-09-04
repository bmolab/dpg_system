"""OSCQuery: OSC that describes itself."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """OSCQuery is OSC that can tell you what it accepts. These use that.

THE NODES:

oscq_service   connect to a named service, by name rather than by address
oscq_browse    look through what a service offers, and build controls from it
oscq_host      publish a subpatch as a service of your own

WHAT OSCQUERY ADDS TO PLAIN OSC:
With ordinary OSC you have to know everything in advance - the address of the
machine, the port, and every path it will listen on. That knowledge lives in a
manual, or in someone's memory, and nothing checks it.

An OSCQuery device publishes two things instead. It ANNOUNCES itself on the
network, so it can be found by name rather than by address; and it describes its
whole namespace - every path, what type each one takes, and often its range.

Which means the patch can ask rather than be told. That is what these nodes are
for, and it is the difference between typing addresses and picking them from a
list that is guaranteed to be right.

oscq_service IS THE SHORTCUT:
Give it a service name and it finds that service on the network and makes the
osc_device node for it, with the address and ports already correct. It is what
you want when you know which device you are after and just want to be talking
to it.

'available services' lists what has been found. Press 'refresh' if something was
turned on after the patch started.

oscq_browse IS THE ONE THAT SAVES REAL TIME:
It shows the namespace as a list you can drill into, and then BUILDS THE NODES
for whatever you have selected - a single parameter, a whole container, or an
entire service.

Because the description carries types and ranges, what it builds is already
right: a float from 0 to 1 gets a slider with those limits, a string gets a text
field. Nothing to look up and nothing to mistype.

'create' makes what is selected. 'create all' makes everything at the current
level. 'layout' arranges them horizontally, vertically, or as a tree.

'create as' DECIDES WHICH DIRECTION IT GOES:
widget    a control you operate, which sends when you move it
send      a node that transmits a value you give it
receive   a node that listens and reports what the device says

The same parameter can be any of the three. A fader on a mixing desk is a widget
if you want to drive it, a receive if you want to follow what someone else is
doing with it, and both if you want to do one and watch the other.

'subset' FOR DEVICES WITH MANY CHANNELS:
Consoles have namespaces like /ch/1/... through /ch/64/..., and you rarely want
all of them. Put a range in 'subset' - '1-8', or '1,3,5', or '1-3,7,9-11' - and
only those are built.

Without it, 'create all' on a large desk will do exactly what you asked and fill
the patch with hundreds of nodes.

oscq_host PUBLISHES A SUBPATCH:
Put one inside a subpatcher and that subpatch becomes an OSCQuery service on the
network, named after the patcher. Anything else that speaks OSCQuery - another
copy of dpg_system, or a phone - can then find it and build controls for it.

The name comes from the patcher, so name the patcher what you want the service
called. An argument sets the port; without one it picks a free one.

This is the reason to reach for a subpatcher even when the tidiness is not
needed: it gives the group of parameters a name, and the name becomes the
service.

SYNTAX:
oscq_service <service name>
oscq_browse
oscq_host <port>

EXAMPLE:
oscq_service lights

INPUTS and PARAMETERS:

service name / available services / refresh (oscq_service):
Which service to connect to, what was found, and look again.

search / browser (oscq_browse):
Type to search across services; the list drills down as you select.

add url:
Reach a service that does not announce itself, by address.

subset:
Which numbered channels to build. '1-8', '1,3,5', '1-3,7,9-11'.

create / create all / go to:
Build the selection, build everything here, or jump to what was made.

layout / create as:
How to arrange the new nodes, and whether they are widgets, senders or
receivers.

OUTPUTS: 

selected path / service info / param info:
What is selected, and what is known about it.

address space (oscq_service):
The whole namespace, as data.

RELATED:
osc_send and the osc nodes for talking to devices that do not describe
themselves.
p / patcher, since oscq_host takes its service name from the patcher it is in."""

demo = [
    {'key': 'svc', 'init': 'oscq_service', 'pos': (30, 62), 'w': 300, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'name a service and it finds it, and\nmakes the osc_device for you',
     'pos': (30, 275)},
    {'key': 'td', 'init': 'text_display', 'pos': (380, 62), 'w': 340, 'h': 200,
     'props': {'width': 320, 'height': 160, 'wrap': True, 'max_lines': 60,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'the whole namespace it published',
     'pos': (380, 275)},

    {'key': 'br', 'init': 'oscq_browse', 'pos': (30, 355), 'w': 360, 'h': 420},
    {'key': 'c3', 'comment': True, 'text': "search, drill in, then CREATE - it builds\nthe nodes, already the right type and\nrange, because the device said so\nput '1-8' in subset before 'create all'\non a desk with 64 channels, or you will\nget all 64",
     'pos': (30, 790)},

    {'key': 'l1', 'init': 'list', 'pos': (440, 355), 'w': 300, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c9', 'comment': True, 'text': 'the path currently selected',
     'pos': (440, 405)},

    {'key': 'pp', 'init': 'p my_controls', 'pos': (440, 470), 'w': 220, 'h': 70},
    {'key': 'c10', 'comment': True, 'text': 'an oscq_host inside a subpatcher\npublishes it as a service, named after\nthe patcher - so name it well',
     'pos': (440, 555)},
]
links = [('svc', 'address space', 'td', '###text in'),
         ('br', 'selected path', 'l1', '')]

sub = {
    'name': 'my_controls',
    'host': 'pp',
    'demo': [
        {'key': 'host', 'init': 'oscq_host', 'pos': (40, 40), 'w': 260, 'h': 160},
        {'key': 'sl', 'init': 'slider', 'pos': (40, 230), 'w': 200, 'h': 60},
        {'key': 'o1', 'init': 'out level', 'pos': (40, 320), 'w': 150, 'h': 80},
    ],
    'links': [('sl', '', 'o1', 'level')],
}
print(build('oscq_service', 'OSCQuery - OSC that describes itself', body,
            demo, links, demo_width=780, text_width=810, text_height=760,
            subpatch=sub))
