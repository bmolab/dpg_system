"""The dict family (one page per node), dict_search/list_box, the replace family."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# color_convert is no longer generated here: color_convert_help.json is laid out
# by hand, and derive_color_convert_help.py copies its layout into one page per
# conversion (rgb_to_hsl, ...). Regenerating it here would overwrite the hand
# layout on every rebuild.

# ----------------------------------------------------------------------- dict
body = """The dict node is a named store: a collection of entries, each a key and
the value filed under it. You put entries in, ask for them back by key, and can
save the whole collection to a file and load it again.

A dictionary is a bundle of named values travelling as one thing. Where a list
says "three numbers, and you remember what they are", a dictionary says "x is
this, y is that, confidence is the other".

The collection is also saved with the patch, so it is still there the next
time the patch is opened.

PUTTING ENTRIES IN:
Send a list to 'store': the first element is the key, the rest is the value.
"red 255 0 0" files the list 255 0 0 under red. A key with a single value
after it files just that value. Sending a whole dictionary to 'store' - or
text that starts with { and reads as JSON - replaces the entire collection.

GETTING THEM BACK:
Send a key to 'retrieve by key'. If it is there, its value goes out of 'out';
if not, the key goes out of 'unmatched' instead. Wire 'unmatched' up rather
than assuming a lookup succeeded.
A key containing / reaches into nested dictionaries: synth/freq looks up freq
inside the dictionary filed under synth.

'retrieve by key' can store too. A list of two or more elements there files
everything after the first under the first, always as a list. "delete red"
removes red, and "append" followed by a value files the value under the next
unused whole number.

SYNTAX:
dict <name>

The name is only the label the node shows. It does not load anything; after a
save or load the label shows the file's name.

EXAMPLE:
dict colours

INPUTS and PARAMETERS:

retrieve by key:
A key to look up, or a list to store, as above.

store:
A key followed by its value, a whole dictionary, or JSON text.

load:
Click to choose a JSON file; its contents replace the collection.

save:
Click to choose where to write the collection as a JSON file.

clear:
Empties the collection.

send dict:
Sends the whole collection out of 'dict out'. The collection is kept.

MESSAGES:
These can be sent to any inlet.

keys - send the list of keys out of 'out'
next - send the next entry, as a one-entry dictionary; it stops at the last
reset - make next start from the first entry again
random - send one entry, chosen at random, as a one-entry dictionary
dump - send every entry as a list: the key followed by its value. This only
works when every value is a list; a single number stored as a value stops it
with an error
search <beginning> - send, as one list, every entry whose value is itself a
dictionary and whose path has a level beginning with the text: each item is
the path and the dictionary found there
save <file> / load <file> - save or load without the file dialog
clear - empty the collection

A consequence: an entry whose key is one of these words cannot be stored or
looked up by sending a list that starts with it - the word is taken as the
message.

OUTPUTS:

out:
The value that was looked up, and the replies to the messages above.

unmatched:
The key, when nothing in the collection matches it.

dict out:
The whole collection, when 'send dict' is clicked.

RELATED:
construct_dict builds a dictionary from inlets, dict_extract and dict_retrieve
take one apart, dict_keys lists its keys and dict_stream sends its entries one by
one. dict_search finds a branch of a nested dictionary by typing."""

demo = [
    {'key': 'm1', 'init': 'message', 'pos': (30, 62), 'w': 180, 'h': 42,
     'props': {'text in': 'red 255 0 0', 'font size': '24'}},
    {'key': 'm2', 'init': 'message', 'pos': (30, 115), 'w': 180, 'h': 42,
     'props': {'text in': 'green 0 255 0', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'click to store: a key, then its value',
     'pos': (30, 165)},
    {'key': 'k1', 'init': 'message', 'pos': (260, 62), 'w': 120, 'h': 42,
     'props': {'text in': 'red', 'font size': '24'}},
    {'key': 'k2', 'init': 'message', 'pos': (260, 115), 'w': 120, 'h': 42,
     'props': {'text in': 'purple', 'font size': '24'}},
    {'key': 'k3', 'init': 'message', 'pos': (400, 115), 'w': 120, 'h': 42,
     'props': {'text in': 'keys', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'then click to look up, or ask for the keys',
     'pos': (260, 200)},
    {'key': 'dc', 'init': 'dict', 'pos': (30, 240), 'w': 180, 'h': 180},
    {'key': 'l1', 'init': 'list', 'pos': (30, 450), 'w': 220, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'what was found', 'pos': (30, 500)},
    {'key': 's1', 'init': 'string', 'pos': (300, 450), 'w': 160, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'a key that is not there', 'pos': (300, 500)},
]
links = [('m1', 'message out', 'dc', 'store'), ('m2', 'message out', 'dc', 'store'),
         ('k1', 'message out', 'dc', 'retrieve by key'),
         ('k2', 'message out', 'dc', 'retrieve by key'),
         ('k3', 'message out', 'dc', 'retrieve by key'),
         ('dc', 'out', 'l1', ''), ('dc', 'unmatched', 's1', '')]
print(build('dict', 'dict - a store of named values', body, demo, links,
            demo_width=560, text_width=830, text_height=900))


def dict_source(text, x=30, y=62):
    """button -> repeat_in_order 2 -> JSON text into a dict, then send it on.
    The demos below need a ready-made dictionary; this is the shortest way."""
    return [
        {'key': 'btn', 'init': 'button', 'pos': (x, y), 'w': 88, 'h': 46},
        {'key': 'c0', 'comment': True, 'text': 'click to send a dictionary', 'pos': (x, y + 53)},
        {'key': 'rp', 'init': 'repeat_in_order 2', 'pos': (x, y + 88), 'w': 190, 'h': 70},
        {'key': 'src', 'init': 'string', 'pos': (x, y + 173), 'w': 380, 'h': 42,
         'props': {'text in': text, 'font size': '24'}},
        {'key': 'dc', 'init': 'dict', 'pos': (x, y + 233), 'w': 160, 'h': 180},
    ], [('btn', '', 'rp', ''),
        ('rp', 'first', 'src', ''),
        ('rp', 'second', 'dc', 'send dict'),
        ('src', 'string out', 'dc', 'store')]


SRC_XY = '{"x": 0.25, "y": 0.75, "name": "left hand"}'

# ------------------------------------------------------------- construct_dict
body = """The construct_dict node gathers several inlets into one dictionary.

Its arguments are the keys: "construct_dict x y" has an x inlet and a y inlet.
Fill them in any order, then click 'send dict' (or send anything to it) and the
dictionary goes out - one bundle carrying every value under its name.

gather_to_dict is another name for the same node.

WHAT IS KEPT BETWEEN SENDS:
The inlets named by the arguments keep their last value, so each send includes
every one of them - a key whose inlet has never received anything is sent with
no value. Entries added through 'labelled data in' are cleared after each send.

ADDING ENTRIES AT RUN TIME:
'labelled data in' takes a list whose first element is the key. The rest of the
list becomes the value, and is always a list - "speed 3" files the list 3, not
the number 3. A key that is also an argument is overwritten by that argument's
inlet when the dictionary is sent.

SYNTAX:
construct_dict <key> <key> ...
gather_to_dict <key> <key> ...

EXAMPLE:
construct_dict x y confidence

INPUTS and PARAMETERS:

send dict:
Sends the dictionary.

<one inlet per key>:
The value for that key. Arriving here does not send anything.

labelled data in:
A key followed by its value, added to the next dictionary only.

OUTPUTS:

dict out:
The dictionary.

RELATED:
dict_extract takes a dictionary apart again, one outlet per key.
dict is a store you add to and look up from."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click to fill both keys and send', 'pos': (30, 115)},
    # repeat_in_order so one click fills both keys BEFORE asking for the send
    {'key': 'rp', 'init': 'repeat_in_order 3', 'pos': (30, 150), 'w': 190, 'h': 70},
    {'key': 'm1', 'init': 'float', 'pos': (30, 235), 'w': 127, 'h': 42,
     'props': dict(FLT, **{'': 0.25})},
    {'key': 'm2', 'init': 'float', 'pos': (200, 235), 'w': 127, 'h': 42,
     'props': dict(FLT, **{'': 0.75})},
    {'key': 'cd', 'init': 'construct_dict x y', 'pos': (30, 300), 'w': 180, 'h': 120},
    {'key': 'c1', 'comment': True, 'text': 'two named values in one bundle', 'pos': (30, 430)},
    {'key': 'tx', 'init': 'text', 'pos': (30, 470), 'w': 400, 'h': 160},
]
links = [('btn', '', 'rp', ''),
         ('rp', 'first', 'm1', ''), ('rp', 'second', 'm2', ''),
         ('rp', 'third', 'cd', 'send dict'),
         ('m1', '', 'cd', 'x'), ('m2', '', 'cd', 'y'),
         ('cd', 'dict out', 'tx', '###text in')]
print(build('construct_dict', 'construct_dict - gather inlets into a dictionary', body,
            demo, links, demo_width=460, text_width=800, text_height=700))

# --------------------------------------------------------------- dict_extract
body = """The dict_extract node takes a dictionary apart: one outlet per key you
name, each sending the value filed under that key.

"dict_extract x y" has an x outlet and a y outlet. When a dictionary arrives,
each key that is in it sends its value; a key that is missing sends nothing.
The outlets fire from right to left - the last key first - so the leftmost
outlet is the last to send.

unpack_dict is another name for the same node.

WITH NO ARGUMENTS:
dict_extract on its own takes its keys from the first dictionary it receives,
and grows an outlet for each. On that first dictionary each value is sent
twice. The keys it learned are kept from then on.

SYNTAX:
dict_extract <key> <key> ...
unpack_dict <key> <key> ...

EXAMPLE:
dict_extract x y confidence

INPUTS and PARAMETERS:

dict in:
The dictionary to take apart.

key count (option):
How many outlets are shown, up to 32.

key 0, key 1, ... (options):
The key each outlet carries. Changing one relabels its outlet.

include key in output (option):
When ticked, each outlet sends its key and value together as a two-element
list, rather than the value alone.

OUTPUTS:

<one outlet per key>:
The value filed under that key.

RELATED:
dict_retrieve takes out one value, by a key that can change while the patch
runs. construct_dict does the reverse of this node."""

demo, links = dict_source(SRC_XY)
demo += [
    {'key': 'de', 'init': 'dict_extract x y', 'pos': (30, 520), 'w': 170, 'h': 100,
     'props': {'key count': 2, 'key 0': 'x', 'key 1': 'y',
               'include key in output': False}},
    {'key': 'f1', 'init': 'float', 'pos': (30, 640), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'f2', 'init': 'float', 'pos': (200, 640), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'one outlet per named key -\nname is in the dictionary but not asked for',
     'pos': (30, 695)},
]
links += [('dc', 'dict out', 'de', 'dict in'),
          # dict_extract's outlets are all unnamed, so address them by index
          ('de', '', 'f1', '', 0), ('de', '', 'f2', '', 1)]
print(build('dict_extract', 'dict_extract - one outlet per key', body, demo, links,
            demo_width=460, text_width=800, text_height=760))

# ------------------------------------------------------------------ dict_keys
body = """The dict_keys node reports the keys of a dictionary, as a list.

Use it to see what is in a dictionary you did not build yourself - the output of
an OSC query, a JSON file, a language model - before deciding what to take out.

SYNTAX:
dict_keys

INPUTS and PARAMETERS:

dict in:
The dictionary. Anything that is not a dictionary is ignored.

OUTPUTS:

keys out:
The keys, in the order they were added. Only the top level is listed; keys of
dictionaries inside it are not.

RELATED:
dict_stream sends the keys together with their values, one entry at a time.
dict_search lets a person browse a nested dictionary by typing."""

demo, links = dict_source(SRC_XY)
demo += [
    {'key': 'dk', 'init': 'dict_keys', 'pos': (30, 520), 'w': 140, 'h': 60},
    {'key': 'l1', 'init': 'list', 'pos': (30, 600), 'w': 220, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'the keys it contains', 'pos': (30, 650)},
]
links += [('dc', 'dict out', 'dk', 'dict in'), ('dk', 'keys out', 'l1', '')]
print(build('dict_keys', 'dict_keys - the names in a dictionary', body, demo, links,
            demo_width=460, text_width=760, text_height=520))

# -------------------------------------------------------------- dict_retrieve
body = """The dict_retrieve node takes one value out of a dictionary, by a key you
send it.

It keeps a copy of the last dictionary it received. Whenever a dictionary
arrives, or the key changes, it looks the key up and sends the value it finds.
That second part matters: you can browse a dictionary you already have by
changing the key, without sending the dictionary again.

When the key is not in the dictionary nothing is sent.

SYNTAX:
dict_retrieve

INPUTS and PARAMETERS:

dict in:
The dictionary to look in. It is kept until the next one arrives.

key:
The key to look up. Send it as a single string (from a string node, say),
not as a list: a message box sends a list even for one word, and a list here
stops the node with an error.

OUTPUTS:

value out:
The value filed under the key.

RELATED:
dict_extract takes out several values at once, by keys fixed in advance.
dict looks up by key in a collection it stores itself."""

demo, links = dict_source(SRC_XY)
demo += [
    {'key': 'k1', 'init': 'string', 'pos': (270, 520), 'w': 120, 'h': 42,
     'props': {'text in': 'x', 'font size': '24'}},
    {'key': 'k2', 'init': 'string', 'pos': (270, 572), 'w': 120, 'h': 42,
     'props': {'text in': 'name', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'then click a key - the value\ncomes back without resending',
     'pos': (270, 625)},
    {'key': 'dr', 'init': 'dict_retrieve', 'pos': (30, 700), 'w': 160, 'h': 80},
    {'key': 's1', 'init': 'string', 'pos': (30, 800), 'w': 220, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
]
links += [('dc', 'dict out', 'dr', 'dict in'),
          ('k1', 'string out', 'dr', 'key'), ('k2', 'string out', 'dr', 'key'),
          ('dr', 'value out', 's1', '')]
print(build('dict_retrieve', 'dict_retrieve - one value, by a key you can change', body,
            demo, links, demo_width=460, text_width=780, text_height=620))

# ---------------------------------------------------------------- dict_stream
body = """The dict_stream node sends every entry of a dictionary, one after
another, each as a two-element list: the key, then its value.

It is the way to loop over a dictionary's contents - everything downstream
handles one entry at a time, in the order the entries were added.

SYNTAX:
dict_stream

INPUTS and PARAMETERS:

dict in:
The dictionary. Anything that is not a dictionary is ignored.

OUTPUTS:

key value lists out:
One list per entry, sent in quick succession. Something that only shows the
latest value - a list box, say - ends up showing the last entry.

RELATED:
dict_keys sends just the keys, all at once.
construct_dict's 'labelled data in' takes lists of the same shape, so a stream
from this node can rebuild a dictionary."""

demo, links = dict_source(SRC_XY)
demo += [
    {'key': 'ds', 'init': 'dict_stream', 'pos': (30, 520), 'w': 160, 'h': 60},
    {'key': 'l1', 'init': 'list', 'pos': (30, 600), 'w': 260, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'the last entry to arrive', 'pos': (30, 650)},
    {'key': 'cnt', 'init': 'counter', 'pos': (320, 600), 'w': 123, 'h': 84},
    {'key': 'i1', 'init': 'int', 'pos': (320, 700), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c4', 'comment': True, 'text': 'counts the entries -\nthree per click', 'pos': (320, 750)},
]
links += [('dc', 'dict out', 'ds', 'dict in'), ('ds', 'key value lists out', 'l1', ''),
          ('ds', 'key value lists out', 'cnt', 'input'), ('cnt', 'count out', 'i1', '')]
print(build('dict_stream', 'dict_stream - entries one at a time', body, demo, links,
            demo_width=480, text_width=760, text_height=560))


# ------------------------------------------------------- dict_search, list_box
body = """The dict_search node lets you FIND a branch of a nested dictionary by typing
part of its path, rather than by wiring up a lookup.

It shows a search field with a scrolling list under it. Type, and the list fills
with the paths that match. Click an entry, or press select, and that branch is
sent out.

It is built for OSC Query style trees - an OSC namespace, a parameter tree -
where every level is a dictionary and its children sit under a key called
CONTENTS. The list shows paths with the CONTENTS levels left out, so
synth/CONTENTS/freq appears as synth/freq.

WHAT IS LISTED:
Only entries that are themselves dictionaries appear - the nodes of the tree,
not the plain values stored inside them. A flat dictionary of numbers or words
therefore shows nothing; use dict_keys for that.

A search matches any level of a path, not only the first. Typing "f" finds
every path with a level beginning with f; typing "synth/f" finds a level
beginning with f directly under synth. A path's children match along with it.

SYNTAX:
dict_search

EXAMPLE:
dict_search

INPUTS and PARAMETERS:

dict in:
The tree to search. It is stored, not shown: the list stays as it was until
you next type in the search field.

search field:
Typing filters as you go, and highlights the first match. Pressing return,
or clicking away from the field, sends the highlighted entry. Text can also be
sent into this inlet, which sends the highlighted entry without searching again.

list:
The matching paths. Clicking an entry sends it.

select:
A button that sends the highlighted entry again.

OUTPUTS:

results out:
A list with one item per match - the chosen branch and everything under it.
Each item is itself a three-part list: the short path, the full path with its
CONTENTS levels, and the dictionary found there.

RELATED:
list_box is the same search-and-choose interface over a plain list of names.
dict_keys gives the names of a dictionary as a list, without the interface -
use that when the patch is choosing, and dict_search when a person is."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click to load the tree', 'pos': (30, 115)},
    # repeat_in_order: one click stores the text in the dict, THEN sends it on
    {'key': 'rp', 'init': 'repeat_in_order 2', 'pos': (30, 150), 'w': 190, 'h': 70},
    {'key': 's0', 'init': 'string', 'pos': (30, 235), 'w': 380, 'h': 42,
     'props': {'text in': '{"CONTENTS": {"synth": {"CONTENTS": {"freq": {"VALUE": 440}, '
                          '"gain": {"VALUE": 0.5}}}, "lights": {"CONTENTS": '
                          '{"level": {"VALUE": 1.0}}}}}',
               'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'a small OSC Query style tree, as text',
     'pos': (30, 285)},
    {'key': 'dc', 'init': 'dict', 'pos': (30, 325), 'w': 160, 'h': 180},
    {'key': 'c2', 'comment': True, 'text': 'turns the text into a dictionary',
     'pos': (30, 515)},
    {'key': 'ds', 'init': 'dict_search', 'pos': (30, 555), 'w': 330, 'h': 230},
    {'key': 'c3', 'comment': True,
     'text': 'type s, f or synth/f to fill the list\nthen click an entry to send it',
     'pos': (30, 795)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 850), 'w': 380, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
]
links = [('btn', '', 'rp', ''),
         ('rp', 'first', 's0', ''),
         ('rp', 'second', 'dc', 'send dict'),
         ('s0', 'string out', 'dc', 'store'),
         ('dc', 'dict out', 'ds', 'dict in'),
         ('ds', 'results out', 'l1', '')]
print(build('dict_search', 'dict_search - find a branch of a tree by typing', body,
            demo, links, demo_width=430, text_width=790, text_height=820))

# ------------------------------------------------------------------- list_box
body = """The list_box node lets you choose one item from a list by typing at it.

It shows a search field with a scrolling list under it. Send it a list, and
every item appears. Type, and the list narrows to the items that contain what
you typed. Click an entry, and it is sent out.

It exists because a list of every file in a directory, or every preset name,
is not something you navigate with patch cords. You need to look at it.

SYNTAX:
list_box

EXAMPLE:
list_box

INPUTS and PARAMETERS:

list in:
The items to choose from. A new list replaces the old one and is filtered
straight away by whatever is in the search field. Every item is shown, searched
and sent as text - a number arrives as its text form, so 2.5 reads 2.500 and
comes out as the text '2.500', not a number.

search field:
Typing filters as you go. The match is "contains", not "begins with",
and capitals count: "re" finds red and green, "Re" finds neither.
Leave it empty to see everything. Pressing return, or clicking away from the
field, sends the highlighted entry. Text can also be sent into this inlet;
it sends the highlighted entry, and filters the next list that arrives.

list:
The items that match. When the list is refilled, the entry that was
highlighted stays highlighted if it is still there; otherwise the first entry is.
Clicking an entry sends it.

OUTPUTS:

results out:
The chosen item, as a string. Nothing is sent just because the list changed -
only when a choice is made.

RELATED:
dict_search is the same interface for finding a branch of a nested dictionary
by its path."""

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
print(build('list_box', 'list_box - choose from a list by typing', body,
            demo, links, demo_width=430, text_width=790, text_height=640))

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
    {'key': 'c1', 'comment': True, 'text': 'only the matching name changes\nedit find and replace and click again',
     'pos': (30, 370)},
]
links = [('btn', '', 'm1', ''), ('m1', 'message out', 'rp', 'int in'),
         ('rp', 'out', 'l1', '')]
print(build('replace', 'replace - swap one value for another in passing', body,
            demo, links, demo_width=420, text_width=790, text_height=560))
