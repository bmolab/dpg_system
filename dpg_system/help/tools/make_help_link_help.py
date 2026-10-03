"""help_link - the button the node browser is made of."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build

body = """A button that opens a page of the node browser, or a help patch.

THE NODE:

help_link   one button, one destination

The node browser (Help > Node Browser) is built from these. You can also put
them in your own patches to point at documentation.

SYNTAX:
help_link <target> [label]

EXAMPLES:
help_link nodes_math Math
help_link trig

<target> is either a browser page - a file in dpg_system/help/browser - or a
help patch, named without its _help suffix ('trig' opens trig_help.json). The
label is optional; without one the button shows the target.

A PAGE LINK REPLACES THE PATCH IT IS IN:
A blue button opens its page and closes the patch holding the button. Walking
the browser therefore leaves one page open, however deep you go. Each page has
a link back up to its parent at the top.

A HELP LINK REPLACES THE LAST HELP IT OPENED:
Any other colour opens a help patch, and closes the help patch the browser
opened before it. The browser colours these by section, so the groups on a
page read apart at a glance. Browsing leaves at most one page and one help patch open.

NOTHING YOU HAVE EDITED IS CLOSED:
A patch you have changed - added a node, moved one, made a connection - is left
open rather than closed, so you never lose work or get asked to save.

INPUTS and PARAMETERS:

<label>:
The button. Click it to go.

width (option):
The button's width in pixels. 0 sizes it to its label. The browser pages set
one width per page so the buttons form a column.

colour (option):
auto, green, orange, violet, teal, pink or olive. auto is blue for a page and
green for a help patch.

The browser pages are generated: edit help/tools/make_browser.py and rerun it,
rather than editing the pages themselves."""

demo = [
    {'key': 'c1', 'comment': True, 'text': 'opens the top page of the browser, in place of this one:',
     'pos': (24, 70)},
    {'key': 'top', 'init': 'help_link nodes dpg_system nodes', 'pos': (24, 100), 'w': 180, 'h': 44},
    {'key': 'c2', 'comment': True, 'text': 'opens a help patch:', 'pos': (24, 170)},
    {'key': 'trig', 'init': 'help_link trig', 'pos': (24, 200), 'w': 100, 'h': 44},
]

print(build('help_link', 'help_link - a door to another page', body, demo, [],
            demo_width=440, text_width=640, text_height=620))
