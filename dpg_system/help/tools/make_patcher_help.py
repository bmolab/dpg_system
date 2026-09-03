"""Subpatchers: p / patcher, and their in / out ports."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """A subpatcher is a patch inside a patch, showing as one node.

THE NODES:

p <name>        make a subpatch. 'patcher' is the same node, spelled out
patcher <name>  the long spelling
in <name>       an inlet on the parent node, seen from inside
out <name>      an outlet on the parent node, seen from inside

WHAT THIS IS FOR:
Two quite different things, and it is worth knowing which one you are doing.

The first is TIDINESS. A dozen nodes that together do one job become one node
with a name on it, and the patch you are actually reading gets shorter. Nothing
changes about how it runs.

The second is REUSE. Once a piece of work has a name and clean edges, you can
have several of them, each with its own settings, and think about them as
things rather than as a tangle of cords.

HOW TO MAKE ONE:
Type 'p something'. You get a node with a button on it, and clicking the button
opens the subpatch in its own tab.

Inside, an 'in' node makes an INLET on the parent, and an 'out' node makes an
OUTLET. That is the whole mechanism: the ports on the outside are created by the
nodes you put on the inside, and they appear as soon as you make them.

Give them names - 'in signal', 'out scaled' - and those names label the ports on
the parent node, which is what makes the parent readable from outside. An
unnamed one still works and is just called 'in 0'.

FINDING YOUR WAY BACK:
Each 'in' has a 'source' button and each 'out' has a 'dest' button. Pressing one
jumps to the parent patch. It is easy to lose track of which tab you are in
once there are a few, and those buttons are the way back up.

THE ORDER OF THE PORTS IS THE ORDER YOU MADE THEM:
Ports are handed out as the in and out nodes ask for them, so the first 'in' you
create is the first inlet. Deleting one frees its slot for the next one made -
which can shuffle the ports, and any cords into the freed slot are dropped.

So if you are going to have several inlets, make them in the order you want to
read them, and expect to reconnect if you delete one later. Twenty inlets and
twenty outlets is the limit; past that it says so and refuses.

IT IS ALL ONE FILE:
The subpatch is not saved separately. It lives in the parent's file, and opening
the parent brings it back with its contents and connections intact. Copying the
file copies everything inside it.

Subpatches can hold subpatches, as deep as you like.

RENAMING:
'patcher name' on the parent renames it, and the button changes to match.
'input name' and 'output name' on the in and out nodes rename the ports, live -
the label on the parent updates as you type, and existing cords stay connected.

SYNTAX:
p <name>
patcher <name>
in <name>
out <name>

EXAMPLE:
p scale_and_offset

INPUTS and PARAMETERS:

patcher name:
What the subpatch is called. Shown on the button.

input name / output name:
What this port is called on the parent node.

source / dest:
Buttons that jump back up to the parent patch.

show options:
Whether the node's options are visible.

OUTPUTS: 

The inlets and outlets of a p node are whatever the in and out nodes inside it
have asked for - it has none of its own.

RELATED:
The demo here contains a real subpatch: click the button on the p node to open
it, and use the 'source' button inside to come back."""

demo = [
    {'key': 'sig', 'init': 'signal', 'pos': (30, 62), 'w': 129, 'h': 78,
     'props': SIG('sin', 3.0)},
    {'key': 'c0', 'comment': True, 'text': 'a plain sine, running -1 to 1',
     'pos': (30, 155)},

    {'key': 'pp', 'init': 'p scale_and_offset', 'pos': (30, 200), 'w': 220, 'h': 70},
    {'key': 'c1', 'comment': True, 'text': 'CLICK THE BUTTON to open the subpatch',
     'pos': (30, 285)},
    {'key': 'c2', 'comment': True, 'text': 'the two ports on this node were made by',
     'pos': (30, 315)},
    {'key': 'c3', 'comment': True, 'text': 'the in and out nodes inside it',
     'pos': (30, 345)},

    {'key': 'pl', 'init': 'plot', 'pos': (30, 400), 'w': 300, 'h': 180,
     'props': PLOT(-1.0, 2.0, 200)},
    {'key': 'c4', 'comment': True, 'text': 'half the size, shifted up by half -',
     'pos': (30, 590)},
    {'key': 'c5', 'comment': True, 'text': 'now running 0 to 1',
     'pos': (30, 620)},

    {'key': 'pl0', 'init': 'plot', 'pos': (380, 62), 'w': 300, 'h': 180,
     'props': PLOT(-1.0, 2.0, 200)},
    {'key': 'c6', 'comment': True, 'text': 'the same signal before it goes in,',
     'pos': (380, 252)},
    {'key': 'c7', 'comment': True, 'text': 'for comparison', 'pos': (380, 282)},
]
links = [('sig', '', 'pp', 'signal'),
         ('pp', 'scaled', 'pl', 'y'),
         ('sig', '', 'pl0', 'y')]

sub = {
    'name': 'scale_and_offset',
    'host': 'pp',
    'demo': [
        {'key': 'in1', 'init': 'in signal', 'pos': (40, 40), 'w': 150, 'h': 80},
        {'key': 'mul', 'init': '* 0.5', 'pos': (40, 150), 'w': 140, 'h': 60},
        {'key': 'add', 'init': '+ 0.5', 'pos': (40, 240), 'w': 140, 'h': 60},
        {'key': 'out1', 'init': 'out scaled', 'pos': (40, 330), 'w': 150, 'h': 80},
    ],
    'links': [('in1', 'signal', 'mul', 'in'),
              ('mul', '', 'add', 'in'),
              ('add', '', 'out1', 'scaled')],
}
print(build('patcher', 'p / patcher - a patch inside a patch', body,
            demo, links, demo_width=720, text_width=800, text_height=700,
            subpatch=sub))
