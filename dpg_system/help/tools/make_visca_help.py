"""Controlling a PTZ camera over VISCA."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """This drives a pan-tilt-zoom camera over the network, using VISCA.

THE NODE:

visca_camera   one camera. 'ptz_camera' is the same node
ptz_camera     the other spelling

VISCA is Sony's camera protocol and is spoken by most PTZ cameras made for
broadcast and installation, whatever the badge. Over IP it goes to UDP port
52381.

THE SLIDERS ARE SPEED, NOT POSITION - THIS IS THE THING TO UNDERSTAND:
'pan' and 'tilt' run -20 to +20, 'zoom' -7 to +7, and none of them are places.
They are RATES. Set pan to 5 and the camera starts panning and KEEPS PANNING
until you set it back to zero or press stop.

That is how a camera operator's controller works - you push a stick and the head
moves while you hold it - and it is why the numbers are small and centred on
zero. It also means a patch that sends a value and forgets about it has left the
camera turning, so anything driving these should return to zero deliberately
rather than by ceasing to send.

'stop' halts everything at once, and is the thing to wire to a panic button.

There is no connect step on this node: it sends as soon as a value arrives. That
is why the demo on this page is deliberately left UNCONNECTED - opening a help
file should not start moving a camera. Set the address first, then wire it up.

FOR PLACES, USE PRESETS:
'preset to store' and 'store preset' remember where the camera is under a
number; 'preset_recall' sends it back there. The camera does the move itself, at
its own sensible speed, and gets to exactly the same place every time.

This is how to work in practice. Frame the shots by hand once, store them, and
then the patch only ever recalls numbers - which is both more reliable and far
easier to cue than driving the head with rates and hoping.

'pan_abs' / 'tilt_abs' / 'abs_speed' with 'drive_absolute' will go to a raw
position at a chosen speed, for when you need somewhere no preset covers.

FOCUS:
'focus' is also a rate - -1, 0 or +1, near or far - and it only does anything
with 'auto_focus' off. Leave auto focus on unless it is hunting, which it will
do on a dark or low-contrast stage; that is when to turn it off and set focus by
hand once.

'reset_sequence' IS THE RECOVERY BUTTON:
VISCA over IP numbers every packet, and the camera expects them in order. If the
count gets out of step - the camera was power-cycled, packets were lost, another
controller spoke to it - the camera silently ignores everything you send.

The symptom is distinctive: the node looks fine, no errors appear, and the
camera simply does nothing. Press 'reset_sequence'. It restarts the count and
sends a clear to the camera, and that fixes it far more often than anything
else. 'reconnect' rebuilds the socket, for when that is not enough.

If a camera has stopped responding, try those two before suspecting the network.

'custom_cmd_hex' FOR THE REST OF THE PROTOCOL:
VISCA has a great many commands and this node exposes the common ones. Type raw
bytes as hex - '81 01 06 04 FF' is the home command - and they are sent as
given. Manufacturers add their own commands to the same scheme, so this is how
you reach anything specific to your camera.

'print debug' puts the traffic in the console, which is the first thing to turn
on when something is not working.

SYNTAX:
visca_camera <ip>
ptz_camera <ip>

EXAMPLE:
visca_camera 10.1.1.160

INPUTS and PARAMETERS:

pan / tilt:
Speed and direction, -20 to +20. Zero is stationary.

zoom:
Speed, -7 to +7. Negative is wider.

focus / auto_focus:
Focus rate, and whether the camera does it itself.

stop:
Halt all movement.

home:
Return to the camera's own home position.

preset_recall / preset to store / store preset:
Remember and return to framings by number.

pan_abs / tilt_abs / abs_speed / drive_absolute:
Go to a raw position at a set speed.

reset_sequence / reconnect:
Restart the packet count; rebuild the socket.

power:
On or off.

ip / port:
Where the camera is. 52381 unless something unusual is going on.

custom_cmd_hex / send_custom:
Raw VISCA bytes.

OUTPUTS: 

None - this node only sends.

RELATED:
pjlink_projector for projectors, which is the same kind of job.
cv_camera if you want the PICTURE rather than control of the head - these are
separate things, and a PTZ camera usually delivers video by another route
entirely."""

demo = [
    {'key': 'cam', 'init': 'visca_camera 10.1.1.160', 'pos': (30, 62),
     'w': 360, 'h': 480},
    {'key': 'c0', 'comment': True, 'text': 'pan and tilt are SPEEDS. Set one and the\ncamera keeps moving until you set it\nback to zero or press stop',
     'pos': (30, 560)},

    {'key': 'sig', 'init': 'signal', 'pos': (450, 62), 'w': 129, 'h': 78,
     'props': SIG('sin', 12.0, 8.0)},
    {'key': 'c3', 'comment': True, 'text': 'NOTHING HERE IS CONNECTED, on purpose:\nthis node sends as soon as a value\narrives, and a help file should not\nstart moving your camera. Set the ip,\nthen connect these yourself\nsine -> pan sweeps back and forth, and\nworks because it returns to zero',
     'pos': (450, 155)},

    {'key': 'btn', 'init': 'button', 'pos': (450, 385), 'w': 88, 'h': 46},
    {'key': 'c6', 'comment': True, 'text': 'wire this to stop - something you can hit',
     'pos': (450, 655)},

    {'key': 'i1', 'init': 'int', 'pos': (450, 490), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c7', 'comment': True, 'text': 'presets are how to work: frame the shots\nby hand once, store them, then the patch\nonly ever recalls numbers',
     'pos': (450, 655)},

    {'key': 'c10', 'comment': True, 'text': 'IF THE CAMERA STOPS RESPONDING with no\nerror at all, press reset_sequence -\nthe packet count has got out of step,\nand it ignores you silently until reset',
     'pos': (450, 655)},
]
# Deliberately NOT wired to the camera. This node sends the moment a value
# arrives -- there is no connect gate -- so a live cord here would start
# panning a real camera simply because someone opened the help file.
links = []
print(build('visca_camera', 'visca_camera - driving a PTZ head', body,
            demo, links, demo_width=820, text_width=810, text_height=760))
