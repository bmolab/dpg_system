"""value_router: each entry of a list to its own OSC address or the outlet."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from help_common import SIG, PLOT, FLT

# The channels the demo starts with, saved into the node the way it saves
# its own (see ValueRouterNode.save_custom).
CHANNELS = {
    '0': {'destination': 'osc', 'target': 'demo', 'address': '/level/a',
          'in min': 0.0, 'in max': 10.0, 'out min': 0.0, 'out max': 1.0,
          'clamp': True, 'threshold': 2.0},
    '1': {'destination': 'osc + outlet', 'target': 'demo', 'address': '/level/b',
          'in min': 0.0, 'in max': 10.0, 'out min': 0.0, 'out max': 127.0,
          'clamp': True, 'threshold': 0.0},
    '2': {'destination': 'outlet', 'target': 'demo', 'address': '/level/c',
          'in min': 0.0, 'in max': 10.0, 'out min': 1.0, 'out max': 0.0,
          'clamp': True, 'threshold': 0.0},
}

body = """value_router sends each entry of a list somewhere of its own - an OSC address, or the outlet - scaled and thresholded channel by channel.

It is made for pc_regions' 'region values': one number per region, every
frame, each of which usually wants to become a different OSC message with
its own range. It works the same on any list of numbers.

CHANNELS:
Each position in the list is a channel. 'index' chooses the channel the
editor shows, and every field below it belongs to that channel - change
'index' and they show that channel's settings instead. The line under
'index' gives the region's name, when the list came from pc_regions' 'cloud',
and 'now' shows the channel's incoming value and what it becomes, which is
the quickest way to choose a range and a threshold.

The list at the bottom shows every channel that is set up and where it
goes. 'clear channel' returns the one being edited to its defaults.

FOR EACH CHANNEL:

destination:
off, osc, outlet, or osc + outlet. A channel is off until you set it.

target:
The name of an osc_target (the name it shows, as osc_send takes it). When
the patch has exactly one osc_target, new channels start with it.

address:
The OSC address it is sent to - and the name it goes out under on 'out'. A
new channel starts as /region/<region name> when the names are known, or
/value/<index>.

in min / in max / out min / out max:
Linear scaling: in min becomes out min and in max becomes out max. Out min
can be larger than out max, to turn a channel upside down. 'clamp' keeps
the result between them.

threshold:
An incoming value below the threshold is sent as out min - a gate that
ignores a region with only a stray voxel or two in it. 0, the default, lets
everything through when the values are never negative.

So each value becomes: below threshold ? out min : scaled.

INSIDE THE PATCH INSTEAD OF OSC:
A channel whose destination includes the outlet sends 'address value' out of
'out'. route splits them by address, as here, so a channel can drive
something in the patch instead of, or as well as, going out over the
network.

'processed' sends the whole list each time one arrives, with every channel
that is set up processed by its settings (whatever its destination) and the
rest passed through unchanged - handy for a plot.

SENDING LESS:
With 'send on change' (option, on by default) a channel is only sent when
its value has changed. A region nobody is in stops sending after its first
zero, rather than repeating it thirty times a second.

The channel settings are saved with the patch.

SYNTAX:
value_router

EXAMPLE:
value_router

INPUTS and PARAMETERS:

values:
A list or array of numbers, or pc_regions' 'cloud' frame. Each one is
processed and sent.

index:
The channel the editor shows.

destination / target / address / in min / in max / out min / out max /
clamp / threshold:
The selected channel's settings, as above.

clear channel:
Return the selected channel to its defaults.

send on change (option):
Only send a channel when its value changes.

OUTPUTS:

out:
'address value' for each channel sending to the outlet.

processed:
The whole list, processed.

RELATED:
pc_regions makes 'region values'. osc_target is where OSC goes. osc_send
sends a single address. route splits 'out' by address."""

VALUES_SUB = {
    'name': 'make_values',
    'host': 'mv',
    'demo': [
        {'key': 'ct', 'init': 'comment four values, each wandering between 0 and 10',
         'pos': (40, 10), 'w': 420, 'h': 30},
        {'key': 's1', 'init': 'signal 3.0 sin', 'pos': (40, 60), 'w': 129, 'h': 78,
         'props': SIG('sin', 3.0, rng=10.0, bipolar=False)},
        {'key': 's2', 'init': 'signal 5.0 sin', 'pos': (190, 60), 'w': 129, 'h': 78,
         'props': SIG('sin', 5.0, rng=10.0, bipolar=False)},
        {'key': 's3', 'init': 'signal 7.0 sin', 'pos': (340, 60), 'w': 129, 'h': 78,
         'props': SIG('sin', 7.0, rng=10.0, bipolar=False)},
        {'key': 's4', 'init': 'signal 11.0 sin', 'pos': (490, 60), 'w': 129, 'h': 78,
         'props': SIG('sin', 11.0, rng=10.0, bipolar=False)},
        {'key': 'pk', 'init': 'pack 4', 'pos': (40, 170), 'w': 130, 'h': 120},
        {'key': 'out', 'init': 'out values', 'pos': (40, 320), 'w': 150, 'h': 80},
    ],
    # the cold inlets first, so the first signal's send carries all four
    'links': [('s4', '', 'pk', 'in 4'), ('s3', '', 'pk', 'in 3'),
              ('s2', '', 'pk', 'in 2'), ('s1', '', 'pk', 'in 1'),
              ('pk', 'out', 'out', 'values')],
}

demo = [
    {'key': 'mv', 'init': 'p make_values', 'pos': (30, 62), 'w': 170, 'h': 70},
    {'key': 'c0', 'comment': True,
     'text': "a list of four values - like pc_regions'\n'region values' for four regions",
     'pos': (230, 62)},
    {'key': 'tg', 'init': 'osc_target demo 127.0.0.1 9000', 'pos': (560, 62), 'w': 260, 'h': 130},
    {'key': 'c4', 'comment': True, 'text': "the target channels 0 and 1\nsend to, by its name 'demo'",
     'pos': (560, 200)},
    {'key': 'vr', 'init': 'value_router', 'pos': (30, 160), 'w': 300, 'h': 520},
    {'key': 'c1', 'comment': True,
     'text': "already set up: 0 to OSC, 1 to OSC and\nthe outlet, 2 turned upside down to the\noutlet only, 3 off. Drag 'index' to see\neach channel's settings.",
     'pos': (30, 700)},
    {'key': 'rt', 'init': 'route /level/b /level/c', 'pos': (420, 300), 'w': 200, 'h': 120},
    {'key': 'f1', 'init': 'float', 'pos': (420, 440), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'f2', 'init': 'float', 'pos': (560, 440), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': "'out', split by address: channel 1\n(0 to 127) and channel 2 (1 to 0)",
     'pos': (420, 490)},
    {'key': 'pl', 'init': 'plot', 'pos': (420, 560), 'w': 308, 'h': 176,
     'props': dict(PLOT(0.0, 127.0), **{'width': 300,
                                        'update style': 'input is multi-channel sample'})},
    {'key': 'c3', 'comment': True, 'text': "'processed': the whole list, processed",
     'pos': (420, 745)},
]
links = [('mv', 'values', 'vr', 'values'),
         ('vr', 'out', 'rt', 'in'),
         ('rt', '/level/b', 'f1', '', 0),
         ('rt', '/level/c', 'f2', '', 1),
         ('vr', 'processed', 'pl', 'y', 1)]
path = build('value_router', 'value_router - a list to OSC, channel by channel', body,
             demo, links, demo_width=860, text_width=810, text_height=1000,
             subpatch=VALUES_SUB, out_dir=os.environ.get('HELP_OUT', HELP_DIR))

doc = json.load(open(path))
hits = 0
for patch in doc['patches'].values():
    for node in patch['nodes'].values():
        if node.get('name') == 'value_router':
            node['value_router_channels'] = CHANNELS
            hits += 1
assert hits == 1
json.dump(doc, open(path, 'w'), indent=4)
print(path)
