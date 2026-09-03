"""color_convert, the dict family, dict_search/list_box, the replace family."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# -------------------------------------------------------------- color_convert
body = """These nodes convert a colour from one way of describing it to another.

A colour is three numbers, but what those three numbers MEAN depends on the 
space you are working in, and the spaces are good at different things. 
Converting between them is not decoration - it is what lets you say 
"the same colour but brighter" without also changing its hue.

THE SPACES:

rgb   red, green, blue - how a screen makes light. 
      Natural for output, awkward for adjustment: there is no component 
      that means "brightness" or "how vivid".
cmy   cyan, magenta, yellow - the inverse of rgb, how inks subtract light. 
      Each value is simply one minus the rgb one.
hsl   hue, saturation, lightness. Hue is the colour itself as an angle 
      round the wheel, saturation is how vivid, lightness runs black to white 
      with the pure colour at the middle.
hsv   hue, saturation, value. Like hsl, but value runs black to the pure 
      colour rather than through to white. Handy when you think of it as 
      "how much light is on this".

Reach for hsl or hsv whenever you want to CHANGE a colour - rotate the hue, 
drain the saturation, dim it - then convert back to rgb to display it.

THE NODES:

There is one node per pairing, named for what it does:

rgb_to_cmy   rgb_to_hsl   rgb_to_hsv
cmy_to_rgb   cmy_to_hsl   cmy_to_hsv
hsl_to_rgb   hsl_to_cmy   hsl_to_hsv
hsv_to_rgb   hsv_to_cmy   hsv_to_hsl

color_convert is the same node with nothing preset - choose both spaces 
yourself from its menus. The named versions are just conveniences; every one 
of them can be repointed after the fact, so a rgb_to_hsl node can be turned 
into a hsv_to_rgb node without replacing it.

SYNTAX:
<from>_to_<to>
color_convert

EXAMPLE:
rgb_to_hsl

INPUTS and PARAMETERS:

in:
Three numbers - a list, or a NumPy array. Receiving data here triggers 
the conversion.

from / to:
The spaces being converted between: rgb, cmy, hsl or hsv. 
The node name sets these; the menus let you change them.

in scale / out scale:
What numeric range the values use, INDEPENDENTLY at each end:

  0-1     0.0 to 1.0, the usual choice inside a patch
  0-100   percentages, as design tools tend to use
  0-255   bytes, as image files and many devices use

Because the two are separate, this node also does the rescaling for you: 
set in scale to 0-255 and out scale to 0-1 to bring an image's colours into 
patch range in the same move as the conversion.

HUE IS ALWAYS IN DEGREES:
The scale settings apply to every component EXCEPT hue, which is always 0 to 
360 whatever the scale says. Hue is an angle, not a proportion, and 360 of them 
go round the circle regardless. This catches people out: an hsl value coming 
out as [210, 0.5, 0.4] with a 0-1 scale is correct, not a bug.

OUTPUTS: 

out:
The converted colour, as a list of three numbers in the output space and scale."""

demo = starter() + [
    {'key': 't2', 'init': 't 0.8', 'pos': (200, 66), 'w': 40, 'h': 46},
    {'key': 't3', 'init': 't 0.5', 'pos': (260, 66), 'w': 40, 'h': 46},
    {'key': 'sig', 'init': 'signal 6.0 saw', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('saw', 6.0, 360.0, False)},
    {'key': 'c0', 'comment': True, 'text': 'a hue sweeping round the wheel', 'pos': (30, 215)},
    {'key': 'pk', 'init': 'pack 3', 'pos': (30, 255), 'w': 150, 'h': 100},
    {'key': 'c1', 'comment': True, 'text': 'hue, saturation 0.8, lightness 0.5',
     'pos': (30, 365)},
    {'key': 'cc', 'init': 'hsl_to_rgb', 'pos': (30, 405), 'w': 170, 'h': 140,
     'props': {'from': 'hsl', 'to': 'rgb', 'in scale': '0-1', 'out scale': '0-1'}},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 565), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 3,
               'min y': 0.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c2', 'comment': True, 'text': 'red, green and blue rise and fall in turn',
     'pos': (30, 725)},
    {'key': 'c3', 'comment': True, 'text': 'set out scale to 0-255 for byte values',
     'pos': (30, 755)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('lb', 'out', 't2', ''), ('t2', '0.8', 'pk', 'in 2'),
         ('lb', 'out', 't3', ''), ('t3', '0.5', 'pk', 'in 3'),
         ('sig', '', 'pk', 'in 1'), ('pk', 'out', 'cc', 'in'),
         ('cc', 'out', 'hm', 'y')]
print(build('color_convert', 'color_convert - the same colour, described differently',
            body, demo, links, demo_width=430, text_width=820, text_height=760))

# ----------------------------------------------------------------------- dict
body = """These nodes build dictionaries, take them apart, and look things up in them.

A dictionary is a bundle of named values travelling as one thing. 
Where a list says "three numbers, and you remember what they are", a dictionary 
says "x is this, y is that, confidence is the other". 
It is what you want when data has to survive being passed around a patch 
without every node needing to agree on the ordering.

THE NODES:

dict              a named store you can save to and load from disk
construct_dict    gather several inlets into one dictionary
gather_to_dict    the same as construct_dict
dict_extract      split a dictionary into one outlet per named key
unpack_dict       the same as dict_extract
dict_keys         report just the names in a dictionary
dict_retrieve     pull one value out, by a key you set
dict_stream       send every key and value in turn, as pairs

BUILDING ONE:
construct_dict takes its keys as arguments, so "construct_dict x y" gives you an 
x inlet and a y inlet. Fill them, then click "send dict" to emit the result. 
The dictionary is emptied after each send, so each one is a fresh snapshot. 
You can also add entries at run time through "labelled data in" by sending a 
list whose first element is the key.

TAKING ONE APART:
dict_extract is the mirror image: name the keys as arguments and each gets its 
own outlet, in order. dict_keys tells you what is in a dictionary you did not 
build yourself. dict_stream sends every entry one at a time as a two-element 
list - the way to loop over a dictionary's contents.

LOOKING SOMETHING UP:
dict_retrieve holds one key and reports the matching value whenever a dictionary 
arrives - or whenever you change the key, if it already holds a dictionary. 
That second behaviour matters: you can browse a dictionary you already have 
without re-sending it.

SYNTAX:
dict <name>
construct_dict <key> <key> ...
dict_extract <key> <key> ...
dict_retrieve

EXAMPLE:
construct_dict x y confidence

INPUTS and PARAMETERS:

dict / dict in / in:
The dictionary. Receiving one triggers the node.

<key inlets> (construct_dict):
One per argument. Values sit here until you send.

labelled data in (construct_dict):
Adds an entry at run time. Send a list whose first element is the key and 
whose remainder is the value.

send dict (construct_dict, dict):
Emits the dictionary built so far, then clears it.

key (dict_retrieve):
The name to look up. Changing it re-reports from the dictionary already held.

key count / key N (dict_extract):
How many outlets are in use, and the key each one carries. 
There are 32 outlets underneath; key count decides how many are visible.

include key in output (dict_extract):
When checked each outlet sends the key alongside the value, rather than the 
value alone.

store / retrieve by key / clear / load / save / name (dict):
The dict node is a persistent store. Send a key and a value to "store", 
send a key to "retrieve by key" to get it back, and use save and load to keep 
the whole collection on disk under the name you gave it.

MESSAGES (dict):
clear, dump, save, load, search, next, reset, keys, random - 
sent to the node as messages. "random" retrieves an arbitrary entry, 
"next" walks through them in turn, and "keys" reports the names.

OUTPUTS: 

dict out:
The assembled dictionary.

keys out:
A list of the names.

value out:
The one value that was looked up.

key value lists out:
Two-element lists, one per entry, sent in succession.

unmatched (dict):
The key, sent here instead when nothing in the store matches it. 
Wire this up rather than assuming a lookup succeeded."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click to build and send a dict', 'pos': (30, 115)},
    # repeat_in_order so one click fills both keys BEFORE asking for the send
    {'key': 'rp', 'init': 'repeat_in_order 3', 'pos': (30, 150), 'w': 190, 'h': 70},
    {'key': 'm1', 'init': 'message', 'pos': (30, 235), 'w': 140, 'h': 42,
     'props': {'text in': '0.25', 'font size': '24'}},
    {'key': 'm2', 'init': 'message', 'pos': (200, 235), 'w': 140, 'h': 42,
     'props': {'text in': '0.75', 'font size': '24'}},
    {'key': 'cd', 'init': 'construct_dict x y', 'pos': (30, 300), 'w': 180, 'h': 120},
    {'key': 'c1', 'comment': True, 'text': 'two named values in one bundle', 'pos': (30, 430)},
    {'key': 'dk', 'init': 'dict_keys', 'pos': (30, 470), 'w': 140, 'h': 60},
    {'key': 'l1', 'init': 'list', 'pos': (30, 545), 'w': 160, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'the names it contains', 'pos': (30, 595)},
    {'key': 'de', 'init': 'dict_extract x y', 'pos': (230, 470), 'w': 170, 'h': 100,
     'props': {'key count': 2, 'key 0': 'x', 'key 1': 'y',
               'include key in output': False}},
    {'key': 'f1', 'init': 'float', 'pos': (230, 585), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'f2', 'init': 'float', 'pos': (230, 635), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'one outlet per named key', 'pos': (230, 685)},
]
links = [('btn', '', 'rp', ''),
         ('rp', 'first', 'm1', ''), ('rp', 'second', 'm2', ''),
         ('rp', 'third', 'cd', 'send dict'),
         ('m1', 'message out', 'cd', 'x'), ('m2', 'message out', 'cd', 'y'),
         ('cd', 'dict out', 'dk', 'dict in'), ('dk', 'keys out', 'l1', ''),
         ('cd', 'dict out', 'de', 'dict in'),
         # dict_extract's outlets are all unnamed, so address them by index
         ('de', '', 'f1', '', 0), ('de', '', 'f2', '', 1)]
print(build('dict', 'dict - named values travelling together', body, demo, links,
            demo_width=430, text_width=830, text_height=780))

# ------------------------------------------------------- dict_search, list_box
body = """These two nodes let you FIND something in a large collection by typing at it, 
rather than by wiring up a lookup.

Both show a scrolling list with a search field above it. Type, and the list 
narrows to what matches. Click an entry, or press select, and it is sent out.

They exist because a dictionary with hundreds of keys, or a list of every file 
in a directory, is not something you navigate with patch cords. 
You need to look at it.

THE NODES:

dict_search   browse the keys of a dictionary. Understands nested keys 
              written with slashes, so a hierarchy can be searched as paths
list_box      the same browser over a plain list

Use dict_search when the collection arrives as a dictionary - an OSC namespace, 
a set of presets, a parameter tree. Use list_box for a list of names.

SYNTAX:
dict_search
list_box

INPUTS and PARAMETERS:

dict in / list in:
The collection to browse. Receiving it triggers the node and refills the list.

search_term:
The text field. Typing filters the list as you go; leave it empty to see 
everything.

options:
The list itself. Clicking an entry selects it.

select (dict_search):
A button that sends the current selection. Useful when you want the choice to 
take effect on a deliberate click rather than as you scroll.

OUTPUTS: 

results out:
What you selected.

RELATED:
dict_keys gives you the same names as a plain list, without the interface - 
use that when the patch is choosing, and these when a person is."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click to load the list', 'pos': (30, 115)},
    {'key': 'm1', 'init': 'message', 'pos': (30, 150), 'w': 300, 'h': 42,
     'props': {'text in': 'red orange yellow green blue indigo violet',
               'font size': '24'}},
    {'key': 'lb2', 'init': 'list_box', 'pos': (30, 215), 'w': 330, 'h': 200},
    {'key': 'c1', 'comment': True, 'text': 'type to narrow, click to choose',
     'pos': (30, 430)},
    {'key': 's1', 'init': 'string', 'pos': (30, 470), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
]
links = [('btn', '', 'm1', ''), ('m1', 'message out', 'lb2', 'list in'),
         ('lb2', 'results out', 's1', '')]
print(build('dict_search', 'dict_search and list_box - find it by typing', body,
            demo, links, demo_width=430, text_width=790, text_height=560))

# -------------------------------------------------------------------- replace
body = """These nodes substitute one value for another as data passes through.

They are search and replace for a patch: everything flows through unchanged 
except the thing you named, which comes out as something else. 
Use them to rename incoming labels to match what your patch expects, to swap a 
code number for a word, or to fix up a stream from a device that calls things 
by names you did not choose.

THE NODES:

replace       find a piece of text, put a different one in its place
int_replace   the same for whole numbers
dict_replace  many substitutions at once, held as a dictionary of pairs

replace and int_replace each hold ONE substitution and are the simple case. 
dict_replace holds as many as you like - build the table by sending it pairs, 
and it applies all of them to everything that passes.

SYNTAX:
replace <find> <replace>
int_replace <find: int> <replace: int>
dict_replace

EXAMPLE:
replace left_hand LeftHand

INPUTS and PARAMETERS:

in / int in:
The data to pass through. Receiving it triggers the node. 
A single value or a list; a list is checked element by element and only the 
matching elements are changed.

find / replace:
The value to look for and what to put in its place.

replace pairs (dict_replace):
Adds one substitution to the table. Send a two-element list - the thing to 
find, then what to replace it with. Send a ONE-element list to remove that 
substitution from the table again.

clear (dict_replace):
Empties the table, so nothing is substituted any more.

OUTPUTS: 

out:
The data, with any matches substituted. Everything else is untouched, 
and the shape of the input is preserved - a list in, a list out.

dict out (dict_replace):
The substitution table itself, so you can inspect what the node is currently 
doing or store it for later."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 280, 'h': 42,
     'props': {'text in': 'left_hand right_hand head', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'a list of names', 'pos': (30, 168)},
    {'key': 'rp', 'init': 'replace left_hand LeftHand', 'pos': (30, 205), 'w': 210, 'h': 100,
     'props': {'find': 'left_hand', 'replace': 'LeftHand'}},
    {'key': 'l1', 'init': 'list', 'pos': (30, 320), 'w': 280, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'only the matching name changes',
     'pos': (30, 370)},
    {'key': 'c2', 'comment': True, 'text': 'edit find and replace and click again',
     'pos': (30, 400)},
]
links = [('btn', '', 'm1', ''), ('m1', 'message out', 'rp', 'int in'),
         ('rp', 'out', 'l1', '')]
print(build('replace', 'replace - swap one value for another in passing', body,
            demo, links, demo_width=420, text_width=790, text_height=560))
