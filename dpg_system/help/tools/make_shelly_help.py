"""Switching a Shelly smart plug from the patch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import FLT

PLUG = '10.1.1.222'

body = """Switch a Shelly smart plug on and off, and read how much power it is passing.

THE NODE:

shelly   one plug (or one relay of a multi-relay Shelly)

IT TALKS TO THE PLUG DIRECTLY:
Every Shelly from Gen2 on answers JSON-RPC calls on its own web server, so the
node needs no Shelly cloud account and no app - only the plug on the same network.
Built against a Shelly Plug US Gen4; any Gen2/Gen3/Gen4 switch device should work.

FINDING THE PLUG:
The argument is the plug's IP address:

shelly 10.1.1.222

Give each plug a FIXED address, set on the plug itself (Wifi.SetConfig, ipv4mode
static) or reserved in the router, so the patch keeps finding it.

Its device id works too, and needs no fixed address: a bare id becomes
<id>.local and is found by name on the network.

shelly shellyplugusg4-98c37700dc94

PICK IT FROM THE LIST:
The 'device' list on the node holds every Shelly on the network, by address and
name (or id, until you name the plug). The node scans when it is created - that
takes five seconds - so the list fills in shortly after. Pick an entry and the
node switches to that plug. With no argument, it takes the first plug found.

'scan' rescans, and sends the list as [id, ip, model, generation, name] to
'reply'. Plugs that join the network later appear after a rescan.

OPENING A PATCH NEVER SWITCHES THE PLUG:
The 'switch' checkbox is saved with the patch, but restoring it does nothing - the
node ignores widget callbacks during a load. The checkbox then shows the plug's
REAL state at the first poll.

IT POLLS, SO THE PLUG'S OWN BUTTON SHOWS UP:
Once a second (the 'poll interval' option; 0 stops polling) the node reads the
plug's status. Press the button on the plug and 'switch' follows within a second.
'on' sends only when the state changes; 'power (W)' and 'status' send at every
poll, so they can drive a plot directly.

'status' is the whole reading as a dict: output, apower (W), voltage, current
(A), freq, aenergy.total (Wh since the counters were reset), temperature.

A TIMER THAT LIVES IN THE PLUG:
'on 3' switches on and has the PLUG switch itself off again 3 seconds later
('off 3' is the reverse). The timer runs on the plug, so it completes even if
the patch is closed or the computer crashes - safer than a delay in the patch for
anything that must not be left on.

THE LED RING:
The 'led' slider sets the ring's brightness, 0 to 100; 'led 20' as a message
does the same. It applies to both of the ring's modes - showing the power drawn
(how the plugs ship) or showing on/off colours - so it holds whichever the plug
is in. The slider shows the plug's own brightness once the node has read it;
opening a patch does not push the saved value to the plug.

Colours, the ring's mode and night dimming are reachable with 'rpc':
rpc PLUGS_UI.GetConfig

EVERYTHING ELSE: 'rpc':
'rpc <Method> [params]' makes any call the plug supports and sends the result to
'reply'. Parameters are key value pairs or one JSON object:

rpc Shelly.GetDeviceInfo
rpc Switch.GetConfig id 0

The full method list: rpc Shelly.ListMethods

A PASSWORD, IF YOU SET ONE:
Plugs ship with no password. If one has been set (in the Shelly app, or
Shelly.SetAuth), give it as the second argument or in the options. The status
line says 'password required' or 'wrong password' when it is the problem.

All network calls run on a worker thread one at a time, so a slow or missing
plug never stalls the patch. Errors show in the status line.

SYNTAX:
shelly <host> [password]

EXAMPLE:
shelly 10.1.1.222
shelly shellyplugusg4-98c37700dc94

INPUTS and PARAMETERS:

device:
Every Shelly on the network; pick one to switch to it.

switch:
Checkbox. Switches the plug, and follows its real state.

led:
LED ring brightness, 0 to 100. Also as a message: led 20.

in:
on / off / 1 / 0 / toggle; on <seconds> / off <seconds>; bang (read status now);
rpc <Method> [params]; scan (rescan); host <host>.

OPTIONS:

host, password, poll interval (seconds), switch id (for multi-relay devices),
print debug (every call and reply in the console).

OUTPUTS:

on:
1 or 0, sent when the state changes - including from the plug's own button.

power (W):
Active power, at every poll.

status:
The full Switch.GetStatus reading as a dict, at every poll.

reply:
Results of 'rpc' calls, and the device list from 'scan'.

RELATED:
pjlink_projector and nvx_kvm for the other things in the rack that answer to the
network."""

msg = lambda key, text, y: {'key': key, 'init': 'message', 'pos': (30, y), 'w': 140, 'h': 50,
                            'props': {'text in': text, 'font size': '24'}}

demo = [
    {'key': 'c0', 'comment': True, 'text': 'click a message to send it',
     'pos': (30, 44)},
    msg('m_on', 'on', 92),
    msg('m_off', 'off', 148),
    msg('m_tog', 'toggle', 204),
    msg('m_pulse', 'on 3', 260),
    msg('m_led', 'led 20', 316),
    {'key': 'c1', 'comment': True, 'text': "'on 3': on now, and the PLUG\nswitches itself off after 3 s\n'led 20': LED ring brightness",
     'pos': (30, 374)},

    {'key': 'plug', 'init': 'shelly ' + PLUG, 'pos': (250, 92), 'w': 125, 'h': 135},
    {'key': 'c2', 'comment': True, 'text': 'device: pick any Shelly on the\nnetwork. switch: follows the plug;\nopening this page never switches it',
     'pos': (250, 270)},

    {'key': 'on', 'init': 'toggle', 'pos': (600, 92), 'w': 45, 'h': 42},
    {'key': 'c3', 'comment': True, 'text': 'on / off, sent when it changes -\nalso from the button on the plug',
     'pos': (660, 100)},
    {'key': 'pw', 'init': 'float', 'pos': (600, 160), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c4', 'comment': True, 'text': 'watts, every poll', 'pos': (740, 172)},

    {'key': 'c5', 'comment': True, 'text': 'any other call, and finding plugs',
     'pos': (30, 444)},
    dict(msg('m_info', 'rpc Shelly.GetDeviceInfo', 492), w=250,
         props={'text in': 'rpc Shelly.GetDeviceInfo', 'font size': '24', 'width': 220}),
    msg('m_scan', 'scan', 548),
    {'key': 'rep', 'init': 'text_display', 'pos': (600, 492), 'w': 440, 'h': 160,
     'props': {'width': 420, 'height': 120, 'max_lines': 50,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c6', 'comment': True, 'text': "'reply': rpc results, and scan's list of\n[id, ip, model, generation, name].\nA scan listens for 5 seconds",
     'pos': (600, 670)},
]
links = [('m_on', 'message out', 'plug', 'in'),
         ('m_off', 'message out', 'plug', 'in'),
         ('m_tog', 'message out', 'plug', 'in'),
         ('m_pulse', 'message out', 'plug', 'in'),
         ('m_led', 'message out', 'plug', 'in'),
         ('m_info', 'message out', 'plug', 'in'),
         ('m_scan', 'message out', 'plug', 'in'),
         ('plug', 'on', 'on', ''),
         ('plug', 'power (W)', 'pw', ''),
         ('plug', 'reply', 'rep', '')]
print(build('shelly', 'shelly - smart plug', body,
            demo, links, demo_width=1060, text_width=800, text_height=900))
