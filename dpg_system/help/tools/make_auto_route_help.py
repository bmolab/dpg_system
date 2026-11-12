"""auto_route: route that makes an outlet for each address it meets."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from help_common import FLT

body = """auto_route sends each value out of an outlet named after its address - and makes that outlet the first time the address arrives.

It is route without having to type the addresses: put it after something
that sends 'address value' messages - value_router's 'out', an OSC
receiver, a message box - and an outlet appears for each address as it
first arrives, labelled with it. Then wire the outlets where they need to go.

WHAT GOES OUT:
A message is an address followed by its value: '/chair 0.4', as a list or as
a string split at its spaces. What goes out of that address's outlet is the
rest of the message - a single value as itself (0.4), several as a list (1 2
3). With 'include address in output' (option) the address stays on the
front, as route's unmatched messages do.

Anything it does not route goes out of 'unmatched': a message with nothing
after the address, an address that arrives once all the outlets are taken,
or a new address while it is not learning.

LEARNING:
Typed with no addresses ('auto_route'), it learns: every new address takes
the next outlet, up to 32. Typed with addresses ('auto_route /a /b'), it
starts with those and, like route, learns nothing more. 'learn new
addresses' (option) turns learning on or off either way - turn it off once
the outlets are wired, so a stray message cannot add another.

The addresses it has are kept as its arguments. Save the patch and it comes
back with the same outlets in the same order, still wired.

EDITING THE ADDRESSES:
'address count' and 'address 0', 'address 1' ... (options) are the outlets'
addresses, in order. Change one to rename that outlet, or lower the count to
drop the last ones. As with unpack_dict, a link stays with its outlet's
position, not its name.

SYNTAX:
auto_route [address ...]

EXAMPLE:
auto_route
auto_route /level/a /level/b

INPUTS and PARAMETERS:

in:
'address value' messages.

learn new addresses (option):
Give each new address an outlet. On when it was typed with no addresses.

include address in output (option):
Keep the address on the front of what goes out.

address count / address 0 ... (options):
The outlets' addresses.

OUTPUTS:

one per address:
The value that came with that address.

unmatched:
Whatever was not routed.

RELATED:
route, which has only the addresses it is given. value_router sends
'address value' messages from a list. unpack_dict does the same for a
dictionary's keys."""

MESSAGES = [('m1', '/red 0.2'), ('m2', '/green 0.7'), ('m3', '/blue 1 2 3')]

demo = [
    {'key': 'c0', 'comment': True, 'text': 'click each message', 'pos': (30, 62)},
]
for i, (key, text) in enumerate(MESSAGES):
    demo.append({'key': key, 'init': 'message', 'pos': (30 + 170 * i, 100), 'w': 150, 'h': 42,
                 'props': {'text in': text, 'font size': '24'}})
demo += [
    {'key': 'ar', 'init': 'auto_route', 'pos': (30, 200), 'w': 150, 'h': 70},
    {'key': 'c1', 'comment': True,
     'text': 'learning: an outlet appears, named\nafter each address, the first time\nit arrives',
     'pos': (30, 290)},
    {'key': 'ak', 'init': 'auto_route /red /green /blue', 'pos': (380, 200), 'w': 290, 'h': 140,
     'props': {'learn new addresses': False}},
    {'key': 'c2', 'comment': True,
     'text': 'given its addresses: the outlets\nare there to wire from the start',
     'pos': (380, 360)},
    {'key': 'f1', 'init': 'float', 'pos': (380, 430), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'f2', 'init': 'float', 'pos': (520, 430), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'mo', 'init': 'message', 'pos': (380, 490), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': "/blue's three values arrive as a list",
     'pos': (380, 545)},
]
links = []
for key, _ in MESSAGES:
    links += [(key, '', 'ar', 'in'), (key, '', 'ak', 'in')]
links += [('ak', '/red', 'f1', '', 0),
          ('ak', '/green', 'f2', '', 1),
          ('ak', '/blue', 'mo', '###text in', 2)]
if os.environ.get('AUTOFIRE'):     # testing only: click the three messages on load
    demo.append({'key': 'lb', 'init': 'load_bang', 'pos': (700, 62), 'w': 88, 'h': 42})
    links += [('lb', 'out', key, '') for key, _ in MESSAGES]
print(build('auto_route', 'auto_route - an outlet for each address', body, demo, links,
            demo_width=720, text_width=810, text_height=860,
            out_dir=os.environ.get('HELP_OUT', HELP_DIR)))
