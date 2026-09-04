"""KVM over IP with Crestron DM-NVX."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """Switch one workstation between several computers, over the network.

THE NODE:

nvx_kvm   one button per target machine, and a way to switch from the patch

WHAT IT IS DOING:
Each computer you want to reach has a Crestron DM-NVX transmitter on its video
and USB. The workstation - the monitor, keyboard and mouse you are actually
sitting at - has a receiver. Switching means pointing that receiver at a
different transmitter.

Video and USB route INDEPENDENTLY, so a switch is really two changes made
together. That is worth knowing when only one of them follows: a picture that
changed but a mouse that did not is the USB pairing, not the video subscription.

There is no control processor in this. The node talks to the endpoints directly
over their REST API.

THE BUTTONS COME FROM A CONFIG FILE:
It reads ~/.nvx_kvm.json - or another path given as the argument - and makes one
button per entry in its 'targets' list, in that order, marking the live one.

Because a node's shape is fixed when it is created, CHANGING THE TARGET LIST
MEANS RE-CREATING THE NODE. Edit the file, then delete and retype the node;
reopening the patch is not enough.

DRIVING IT FROM THE PATCH:
'select' takes either a target's name or its number, counting from 1. 'active
target' sends whenever the live target changes - including when someone presses
a button by hand, so it is a report of what happened rather than an echo of what
you asked for.

Opening a patch does NOT fire a switch. The button callbacks check whether the
patch is loading, which they have to: without that, loading a saved patch would
replay the buttons and switch your KVM out from under you.

IT TAKES ABOUT A SECOND, SOMETIMES SEVERAL:
Measured over four switches: 0.8, 0.9, 3.5 and 7.7 seconds. Usually about a
second. The long tail is the receiver taking its time to report itself fully
started and paired, not the picture being slow to arrive.

Every call is network-bound, so switching and polling happen on worker threads
and the results are applied to the widgets on the main thread. The node stays
responsive while a switch is in progress.

THE THING THAT WILL BITE YOU: IGMP SNOOPING:
A 4K60 stream is about 700 megabits a second. On a switch WITHOUT IGMP snooping,
starting one floods every port - which in this system took all four NVX units off
the network and only recovered when the transmitter was physically unplugged.

So: snooping and a querier must be enabled on the switch these live on, before a
transmitter is started with a real source attached. If this setup ever moves to
different network hardware, check that first. It is a prerequisite, not a
tuning option.

WHY THE STATUS LINE IS WORTH READING:
The devices are not always honest, and the node's status line is where the truth
surfaces. In particular, an HTTP 200 from one of these does NOT mean the write
succeeded - a refused write returns 200 too, and the real outcome is buried in a
per-property status inside the reply. The underlying code checks that and raises;
if it did not, every failed write would look like it worked.

SYNTAX:
nvx_kvm
nvx_kvm <path to config>

EXAMPLE:
nvx_kvm ~/patches/studio_kvm.json

INPUTS and PARAMETERS:

select:
A target's name, or its number counting from 1.

the buttons:
One per target, in config order. The live one is marked.

OUTPUTS: 

active target:
The target that is now live, sent whenever it changes - including from a button
press or a switch made elsewhere.

RELATED:
The full recipe, including the topology, the config format and what was learned
against real hardware, is in dpg_system/NVX_KVM_README.md. Read that before
setting one up; this page is for using one that already works.
pjlink_projector and visca_camera for the other things in a rack that answer to
the network."""

demo = [
    {'key': 'kvm', 'init': 'nvx_kvm', 'pos': (30, 62), 'w': 320, 'h': 260},
    {'key': 'c0', 'comment': True, 'text': 'one button per target, from the config\nfile. The live one is marked.\nEDIT THE CONFIG and you must re-create\nthe node - its shape is fixed when it is\nmade, so reopening the patch is not enough',
     'pos': (30, 335)},

    {'key': 'm1', 'init': 'message', 'pos': (400, 62), 'w': 140, 'h': 42,
     'props': {'text in': '2', 'font size': '24'}},
    {'key': 'c5', 'comment': True, 'text': 'select takes a name or a number,\ncounting from 1',
     'pos': (400, 112)},

    {'key': 'l1', 'init': 'list', 'pos': (400, 190), 'w': 300, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c7', 'comment': True, 'text': 'what is live NOW - it reports, rather\nthan echoing what you asked for, so a\nbutton press by hand shows up here too\na switch takes about a second - measured\n0.8, 0.9, 3.5 and 7.7s over four. The\nlong ones are the receiver reporting\nitself paired, not the picture arriving\nvideo and USB route SEPARATELY. Picture\nchanged but mouse did not? That is the\nUSB pairing, not the video',
     'pos': (400, 240)},
]
links = [('m1', 'message out', 'kvm', 'select'),
         ('kvm', 'active target', 'l1', '')]
print(build('nvx_kvm', 'nvx_kvm - KVM over IP', body,
            demo, links, demo_width=760, text_width=800, text_height=740))
