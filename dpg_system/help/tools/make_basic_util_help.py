"""send/receive, var, repeat, list ops, trace, patcher utilities."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ----------------------------------------------------------------------- send
body = """send and receive move data across a patch without a cord between them.

Give a send node a name and a receive node the same name, and whatever goes into 
the send comes out of the receive - however far apart they are, and however many 
receives share the name. One send can feed any number of receives.

This is not a convenience for tidiness alone. A patch that has grown past one 
screen becomes unreadable when long cords cross it, and a value that many places 
need - a master level, a frame clock, a mode - is genuinely better named than 
wired. The name IS the documentation.

The cost is that the connection is invisible. A cord you can trace with your 
eye; a conduit you have to search for. Use them for things that are genuinely 
global, and keep local plumbing on cords where you can see it.

THE NODES:

send      the sending end
s         a shorter name for send
receive   the receiving end
r         a shorter name for receive

SYNTAX:
send <name>
receive <name>

EXAMPLE:
send master_level

INPUTS and PARAMETERS:

<the conduit name> (send):
The inlet, labelled with the conduit's name so you can see where it goes 
without opening anything. Anything arriving here is passed to every receive 
sharing that name.

name:
The conduit. Changing it detaches from the old one and attaches to the new, 
while the patch runs - so you can repoint a send or a receive without 
rebuilding anything. A name that does not exist yet is created on the spot.

OUTPUTS: 

<the conduit name> (receive):
Whatever was sent, emitted as it arrives.

RELATED:
var is the other way to share a value by name. The difference is memory: 
a conduit passes data through and keeps nothing, so a receive created later 
hears nothing until the next send. A variable HOLDS its value, and can be 
asked for it at any time. Use send and receive for events and streams, 
and var for state."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42, 'props': {'': True}},
    {'key': 'met', 'init': 'metro 200', 'pos': (30, 112), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 200.0, 'units': 'milliseconds'}},
    {'key': 'cnt', 'init': 'counter', 'pos': (30, 192), 'w': 123, 'h': 84,
     'props': {'step': 1}},
    {'key': 'snd', 'init': 'send demo_count', 'pos': (30, 290), 'w': 180, 'h': 70},
    {'key': 'c0', 'comment': True, 'text': 'no cord from here down', 'pos': (30, 370)},
    {'key': 'r1', 'init': 'receive demo_count', 'pos': (30, 420), 'w': 180, 'h': 70},
    {'key': 'i1', 'init': 'int', 'pos': (30, 500), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'r2', 'init': 'receive demo_count', 'pos': (250, 420), 'w': 180, 'h': 70},
    {'key': 'i2', 'init': 'int', 'pos': (250, 500), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 'both receives hear the same send',
     'pos': (30, 555)},
]
links = [('tog', '', 'met', 'on'), ('met', '', 'cnt', 'input'),
         ('cnt', 'count out', 'snd', 'demo_count'),
         ('r1', 'demo_count', 'i1', ''), ('r2', 'demo_count', 'i2', '')]
print(build('send', 'send and receive - a cord without the cord', body, demo, links,
            demo_width=470, text_width=790, text_height=620))

# ------------------------------------------------------------------------ var
body = """The var node is a named value that any part of the patch can read or write.

Send something in and the variable takes that value, and every other var node 
with the same name updates too. The difference from send and receive is that a 
variable REMEMBERS: it holds its value, so a var node created afterwards already 
knows it, and you can ask for it at any moment rather than waiting for the next 
message.

That makes it the right tool for state - a mode, a threshold, a chosen file, 
a master level - anything the patch needs to know rather than to be told.

Variables also connect to widgets. Most number and text widgets have a 
"bind to" option; put a variable name there and the widget and the variable 
become the same thing. Move the slider and the variable changes; change the 
variable and the slider moves. That is how one control ends up driving several 
distant places without a single cord.

SYNTAX:
var <name>

EXAMPLE:
var master_level

INPUTS and PARAMETERS:

in:
Sets the variable. Every var node with this name, and every widget bound to it, 
updates at once. Sending a bang here reports the current value instead of 
changing it.

name:
Which variable this node refers to. Changing it detaches from the old one and 
attaches to the new while the patch runs. A name that does not exist yet is 
created, starting at 0.0.

OUTPUTS: 

out:
The value, sent when the variable changes or when you ask it for one.

RELATED:
send and receive pass data through and keep nothing - better for streams and 
events, where holding the last value would be meaningless. 
Use var when there is a current value worth asking about."""

demo = starter(x=260, y=62) + [
    {'key': 'tv', 'init': 't 0.6', 'pos': (400, 130), 'w': 45, 'h': 46},
    {'key': 'sl', 'init': 'slider', 'pos': (30, 62), 'w': 200, 'h': 60,
     'props': {'bind to': 'demo_level', 'min': 0.0, 'max': 1.0,
               'format': '%.3f', 'width': 180}},
    {'key': 'c0', 'comment': True, 'text': 'a slider bound to demo_level', 'pos': (30, 130)},
    {'key': 'v1', 'init': 'var demo_level', 'pos': (30, 175), 'w': 170, 'h': 70},
    {'key': 'f1', 'init': 'float', 'pos': (30, 258), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'btn', 'init': 'button', 'pos': (240, 200), 'w': 88, 'h': 46},
    {'key': 'c1', 'comment': True, 'text': 'bang a var to ask its value', 'pos': (240, 255)},
    {'key': 'v2', 'init': 'var demo_level', 'pos': (30, 320), 'w': 170, 'h': 70},
    {'key': 'f2', 'init': 'float', 'pos': (30, 403), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': 'a second var, same name, same value',
     'pos': (30, 455)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'tv', ''), ('tv', '0.6', 'v1', 'in'),
         ('btn', '', 'v1', 'in'), ('v1', 'out', 'f1', ''), ('v2', 'out', 'f2', '')]
print(build('var', 'var - a value the whole patch can reach', body, demo, links,
            demo_width=420, text_width=780, text_height=560))

# --------------------------------------------------------------------- repeat
body = """The repeat node sends one incoming value out of several outlets, in a defined order.

Every outlet gets the same value. What the node gives you is the ORDER: 
the rightmost outlet fires first, then the next, and so on leftward. 

That matters more than it sounds. In a patch, order of execution decides 
correctness whenever one branch depends on another having already run - 
set a parameter before triggering the thing that uses it, store a value before 
sending the bang that reads it. Fanning a cord out to several places gives you 
no control over which happens first. This node does.

THE NODES:

repeat            outlets labelled "out 0", "out 1", ...
repeat_in_order   outlets labelled "first", "second", "third", ... 
                  naming the order they actually fire in

They behave identically; repeat_in_order simply labels the outlets so the 
sequence is visible on the node rather than something you have to remember.

SYNTAX:
repeat <count: int>
repeat_in_order <count: int>

EXAMPLE:
repeat_in_order 3

INPUTS and PARAMETERS:

in:
The value to send on. Receiving data here triggers the node.

The count is given as an argument when you create the node and decides how many 
outlets there are. The default is 2.

OUTPUTS: 

One outlet per repeat, all carrying the same value. 
They fire RIGHT TO LEFT - the rightmost first, the leftmost last. 
On repeat_in_order the labels tell you this directly: "first" is the rightmost 
outlet, and it goes first.

RELATED:
The t node does the same job when you want to send different CONSTANTS in a 
defined order, rather than the same value several times."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click once', 'pos': (30, 115)},
    {'key': 'rp', 'init': 'repeat_in_order 3', 'pos': (30, 155), 'w': 190, 'h': 70},
    {'key': 'c1', 'comment': True, 'text': 'the rightmost outlet fires first',
     'pos': (30, 235)},
    {'key': 'a1', 'init': 'counter', 'pos': (30, 280), 'w': 123, 'h': 84,
     'props': {'step': 1}},
    {'key': 'i1', 'init': 'int', 'pos': (30, 380), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'the count rises by three per click',
     'pos': (30, 430)},
]
# counter, not accumulate: a button sends the word 'bang', which accumulate
# would read as the number 0 and add nothing.
links = [('btn', '', 'rp', ''),
         ('rp', '', 'a1', 'input', 0), ('rp', '', 'a1', 'input', 1),
         ('rp', '', 'a1', 'input', 2),
         ('a1', 'count out', 'i1', '')]
print(build('repeat', 'repeat - the same value, in a known order', body, demo, links,
            demo_width=400, text_width=780, text_height=560))

# ------------------------------------------------------------------- list ops
body = """These nodes cut pieces out of a list.

THE NODES:

slice_list    cut a list in two at a position you choose
sublist       take an element or a range, by index

slice_list divides once and gives you both halves on separate outlets -
the start of the list up to the cut, and everything after. Use it to peel a
header off a message, or to split a packed reading into the part you want and
the rest.

sublist uses the same index notation as Python. One number takes that single
element - the element itself, not a list holding it. Counting starts at 0, and
a negative number counts back from the end, so -1 is the last. Two numbers with
a colon take a range as a list: 1:3 is elements 1 and 2. Leave a side blank to
run to that end - 2: is everything from element 2 on - and add a third number
for a step: ::2 is every second element. The default, a bare colon, passes the
whole list.

A COMMA GOES DEEPER, IT DOES NOT PICK SEVERAL:
In sublist, commas do not make a selection of separate elements. Each part
after a comma indexes into the RESULT of the part before it, so '0, 2' is
element 2 of element 0 - for a list of lists, row 0 and column 2. After a
range, the next part indexes the range, not each element in it: ':, 0' is
simply element 0.

SYNTAX:
slice_list
sublist <indices>

slice_list takes no argument - set 'slice after' on the node.

EXAMPLE:
sublist 1:3

INPUTS and PARAMETERS:

list input (slice_list) / list in (sublist):
The list. Receiving it triggers the node.
slice_list splits a string at its spaces first; sublist converts whatever
arrives to a list, so a string of numbers or words works as one.

slice after (slice_list):
The position to cut at, counted from 0; default 0. The first outlet gets
everything up to and including this position; the second gets the rest.
If the list is too short to cut, the whole list goes to the first outlet and
the second sends an empty list.

output only if slice 2 (slice_list option):
When checked, the node stays silent unless there is genuinely something in the
second half - so a list too short to cut produces nothing at all rather than
passing through whole. Use it when a short list means "not ready".

Indices (sublist):
What to take, as above. Changing it re-sends from the last list received.
A position past the end of the list prints an error in the console and sends
None; an empty list sends nothing.

OUTPUTS:

slice 1 out / slice 2 out (slice_list):
The two halves. The second is sent first.

output (sublist):
The element, or the list of elements, taken.

RELATED:
stream_list sends a list's elements one at a time, as separate messages.
unpack sends each element of a list from its own outlet."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click to send the list',
     'pos': (300, 62)},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 250, 'h': 42,
     'props': {'text in': '10 20 30 40 50', 'font size': '24'}},
    {'key': 'sl', 'init': 'slice_list', 'pos': (30, 180), 'w': 180, 'h': 110,
     'props': {'slice after': 1, 'output only if slice 2': False}},
    {'key': 'c1', 'comment': True, 'text': "'slice after' is 1: the cut falls\nafter position 1",
     'pos': (300, 180)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 315), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'slice 1: the first two', 'pos': (300, 315)},
    {'key': 'l2', 'init': 'list', 'pos': (30, 370), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'slice 2: the rest', 'pos': (300, 370)},
    {'key': 'su', 'init': 'sublist 1:3', 'pos': (30, 440), 'w': 240, 'h': 100},
    {'key': 'c4', 'comment': True, 'text': 'a range: elements 1 and 2, as a list',
     'pos': (300, 440)},
    {'key': 'l3', 'init': 'list', 'pos': (30, 565), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'su2', 'init': 'sublist -1', 'pos': (30, 635), 'w': 240, 'h': 100},
    {'key': 'c5', 'comment': True, 'text': 'one index: the last element itself',
     'pos': (300, 635)},
    {'key': 'i1', 'init': 'int', 'pos': (30, 760), 'w': 127, 'h': 42, 'props': INT},
]
links = [('btn', '', 'm1', ''), ('m1', 'message out', 'sl', 'list input'),
         ('sl', 'slice 1 out', 'l1', ''), ('sl', 'slice 2 out', 'l2', ''),
         ('m1', 'message out', 'su', 'list in'), ('su', 'output', 'l3', ''),
         ('m1', 'message out', 'su2', 'list in'), ('su2', 'output', 'i1', '')]
print(build('slice_list', 'slice_list and sublist - cut pieces out of a list',
            body, demo, links, demo_width=600, text_width=790, text_height=660))

# ---------------------------------------------------------------- stream_list
body = """stream_list sends the elements of a list one at a time, as separate messages.

THE NODE:

stream_list   send every element in turn, one after another

It turns one list into a sequence of separate messages. This is how you make
something that expects single values - an accumulator, a counter, a node that
draws one point - process a whole list.

THE WHOLE SEQUENCE HAPPENS AT ONCE:
There is no timing between the elements. Each one is sent, and everything
downstream of it finishes, before the next is sent - so by the time anything
else in the patch runs, the whole list has gone through. If you want the
elements spaced out in time, drive a counter from a metro and pick each element
out with sublist instead.

WHAT COUNTS AS AN ELEMENT:
Whatever arrives is first made into a list: a string of numbers or words is
split at its spaces, and a single value becomes a list of one. Each element is
sent as it is, so an element that is itself a list goes out as a list, and an
array goes out one row at a time.

A bang sends the last list again.

SYNTAX:
stream_list

EXAMPLE:
stream_list

INPUTS and PARAMETERS:

list in:
The list. Each one received is streamed out at once.

OUTPUTS:

stream out:
Each element in turn, as separate messages.

RELATED:
slice_list cuts a list in two; sublist takes an element or a range.
unpack sends each element from its own outlet, all at once.
accumulate sums whatever arrives, so it totals a streamed list."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click to send the list', 'pos': (300, 62)},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 250, 'h': 42,
     'props': {'text in': '10 20 30 40 50', 'font size': '24'}},
    {'key': 'st', 'init': 'stream_list', 'pos': (30, 185), 'w': 150, 'h': 60},
    {'key': 'c1', 'comment': True, 'text': 'five separate messages: 10, 20, 30, 40, 50',
     'pos': (300, 185)},
    {'key': 'a1', 'init': 'accumulate', 'pos': (30, 270), 'w': 140, 'h': 110},
    {'key': 'c2', 'comment': True, 'text': "adds each one as it arrives - 'reset'\nsets it back to zero",
     'pos': (300, 270)},
    {'key': 'i1', 'init': 'int', 'pos': (30, 405), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c3', 'comment': True, 'text': '150 after one click; each click adds 150',
     'pos': (300, 405)},
]
links = [('btn', '', 'm1', ''), ('m1', 'message out', 'st', 'list in'),
         ('st', 'stream out', 'a1', 'in'), ('a1', 'sum', 'i1', '')]
print(build('stream_list', 'stream_list - a list as a sequence of messages',
            body, demo, links, demo_width=640, text_width=790, text_height=560))

# -------------------------------------------------------------------- tracing
body = """start_trace and end_trace print what the patch is doing, between the two of them.

Put start_trace where you want to begin watching and end_trace where you want to 
stop, wire your data through both, and the patch reports every node that 
executes in between, in the order it happens.

This is the tool for "why did that not fire?" and "which of these runs first?". 
Execution order in a patch is decided by the shape of the connections, and it is 
not always the order you assumed. A trace shows you what actually happened 
rather than what the layout suggests.

Both nodes pass their input straight through, so you can leave them in place and 
switch tracing off rather than rewiring to remove them.

THE NODES:

start_trace   begin reporting, and pass the input on
end_trace     stop reporting, and pass the input on

SYNTAX:
start_trace
end_trace

INPUTS and PARAMETERS:

start trace / end trace:
The data. It triggers the node, is passed through unchanged, and marks the 
point in the flow where tracing starts or stops.

enable (start_trace):
Turns tracing on and off without unwiring anything. When off, the node is a 
plain pass-through.

OUTPUTS: 

pass input:
Whatever arrived, unchanged.

WHERE THE OUTPUT GOES:
The trace is printed to the console the patch was launched from, not into the 
patch. Start it as narrowly as you can - a trace across a busy patch produces 
a great deal of text very quickly, and the thing you are looking for scrolls 
past."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click, then look at the console',
     'pos': (30, 115)},
    {'key': 'st', 'init': 'start_trace', 'pos': (30, 155), 'w': 150, 'h': 80,
     'props': {'enable': True}},
    {'key': 'rp', 'init': 'repeat_in_order 2', 'pos': (30, 255), 'w': 190, 'h': 70},
    {'key': 'c1', 'comment': True, 'text': 'everything between the two is reported',
     'pos': (30, 335)},
    {'key': 'cnt', 'init': 'counter', 'pos': (30, 375), 'w': 123, 'h': 84,
     'props': {'step': 1}},
    {'key': 'et', 'init': 'end_trace', 'pos': (30, 475), 'w': 150, 'h': 60},
    {'key': 'i1', 'init': 'int', 'pos': (30, 550), 'w': 127, 'h': 42, 'props': INT},
]
links = [('btn', '', 'st', 'start trace'),
         ('st', 'pass input', 'rp', ''),
         ('rp', '', 'cnt', 'input', 0),
         ('cnt', 'count out', 'et', 'end trace'),
         ('et', 'pass input', 'i1', '')]
print(build('start_trace', 'start_trace - watch what the patch actually does', body,
            demo, links, demo_width=400, text_width=780, text_height=560))

# --------------------------------------------------------- patcher utilities
body = """The present node switches its patch between editing and presentation mode.

It is for finishing a patch - turning something you built into something
someone can use, without them seeing the wiring.

Every node has a presentation state as well as its ordinary visibility.
In presentation mode each node shows only what its presentation state allows,
so nodes set to hidden disappear, and nothing can be dragged. You can lay out a
clean panel of just the controls, on top of the working patch, and switch
between the two.

Because the checkbox is saved with the patch, ticking it and saving is how you
make a patch OPEN presented - which is how you hand it to someone who should
not be looking at the machinery.

SYNTAX:
present

EXAMPLE:
present

INPUTS and PARAMETERS:

open as presentation:
Ticking it switches the patch into presentation mode straight away; unticking
it switches back to editing. The setting is saved with the patch, so a patch
saved with it ticked comes up presented when it is opened.

Leave the present node itself visible in presentation mode (its presentation
state at show all), or there is no checkbox left on screen to untick.

OUTPUTS:

None - present acts on the patch itself rather than passing data on.

RELATED:
active_widget reports which widget is being worked, and patch_window_position
sets the window's place and size - the other two nodes for arranging a patch
for someone else."""

demo = [
    {'key': 'pr', 'init': 'present', 'pos': (30, 62), 'w': 190, 'h': 60},
    {'key': 'c0', 'comment': True,
     'text': 'tick to present this patch, untick to edit again\nhidden nodes vanish while it is presented',
     'pos': (30, 130)},
]
print(build('present', 'present - arrange the patch for someone else', body,
            demo, [], demo_width=420, text_width=780, text_height=560))

# ------------------------------------------------------------- active_widget
body = """The active_widget node shows which widget is being worked right now.

While you hold down a slider, drag a number box or type into a text field,
that widget is "active", and active_widget shows its identifying number.
As soon as you let go, or click away, it goes back to -1.

The number is the widget's internal item number in the interface,
so it is mostly useful for finding your way around the program itself -
seeing whether a click is landing on the widget you think it is.
It is not a hover detector: merely pointing at a widget does not count,
and buttons and drop-down menus always report -1.

SYNTAX:
active_widget

EXAMPLE:
active_widget

INPUTS and PARAMETERS:

active_widget:
The display. It is refreshed every frame from the app; typing into it has no
lasting effect.

OUTPUTS:

None - the number is shown on the node, not sent on.

RELATED:
present switches the patch into presentation mode, and patch_window_position
sets the window's place and size."""

demo = [
    {'key': 'aw', 'init': 'active_widget', 'pos': (30, 62), 'w': 190, 'h': 60},
    {'key': 'c0', 'comment': True, 'text': '-1 when nothing is being worked',
     'pos': (30, 130)},
    {'key': 'sl', 'init': 'slider', 'pos': (30, 175), 'w': 160, 'h': 60},
    {'key': 'c1', 'comment': True, 'text': 'hold this slider down and watch the number',
     'pos': (30, 245)},
]
print(build('active_widget', 'active_widget - which widget is being worked', body,
            demo, [], demo_width=420, text_width=780, text_height=520))

# ----------------------------------------------------- patch_window_position
body = """The patch_window_position node moves and resizes the app's window.

Use it to bring a patch up at a known size on a particular screen, or to fit
a projector. It acts on the whole app window, not just this patch's tab.

When the node is made it fills in the window's current place and size,
so it starts out describing where the window already is. Change a number -
drag it, or send one in - and the window moves or resizes at once.
The four values are saved with the patch, and the node applies them again
whenever the patch is loaded, which is how a patch puts its window where it
belongs when it opens.

SYNTAX:
patch_window_position

EXAMPLE:
patch_window_position

INPUTS and PARAMETERS:

top:
The vertical position of the window, in pixels from the top of the screen.

left:
The horizontal position of the window, in pixels from the left edge of the
screen.

width / height:
The size of the window, in pixels.

Each of the four acts the moment it changes, and all four are applied
together, so the window is always set to the whole of what the node shows.

OUTPUTS:

None - the node acts on the window rather than passing data on.

RELATED:
present switches the patch into presentation mode, and display_info
reports the screens available to put the window on."""

demo = [
    {'key': 'pw', 'init': 'patch_window_position', 'pos': (30, 62), 'w': 210, 'h': 140},
    {'key': 'c0', 'comment': True,
     'text': 'starts at where the window is now\ndrag a number to move or resize it',
     'pos': (30, 215)},
]
print(build('patch_window_position', 'patch_window_position - place the app window',
            body, demo, [], demo_width=420, text_width=780, text_height=600))

# --------------------------------------------------------- directory_iterator
body = """The directory_iterator node walks through the files in a folder, one at a time.

Point it at a directory and each bang on "next file" sends the next path out. 
When it runs out, it says so on a separate outlet rather than going quiet, 
so a batch process can tell the difference between "still working" and "done".

Use it to run the same treatment over a whole dataset - load each file, process, 
save, ask for the next - without listing the files by hand.

SYNTAX:
directory_iterator

EXAMPLE:
directory_iterator

INPUTS and PARAMETERS:

next file:
Sends the next path. Bang it once per file; this is what drives the loop.

directory in:
The folder to walk. Send a path here to point the node somewhere new, 
which also starts it again from the beginning.

saving path:
Where results are being written. The node uses it to work out what has already 
been done, which is what makes resuming possible.

reset:
Go back to the first file.

resume from last run:
When checked, the node skips files that already have a result in the saving 
path, and carries on from where a previous run stopped. 
This is what makes a long batch survive being interrupted - the thing you want 
on a run of thousands of files that fails on file 800.

OUTPUTS: 

next path out:
The path of the next file, as a string.

done:
Fires when there are no files left. Wire this up - it is the only way the patch 
learns that the batch has finished, and without it a loop driven by "next file" 
will simply stop with no indication of why."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click for the next file', 'pos': (30, 115)},
    {'key': 'di', 'init': 'directory_iterator', 'pos': (30, 155), 'w': 210, 'h': 160},
    {'key': 's1', 'init': 'string', 'pos': (30, 335), 'w': 300, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'send a folder path to "directory in" first',
     'pos': (30, 385)},
    {'key': 'btn2', 'init': 'button', 'pos': (30, 425), 'w': 88, 'h': 46,
     'props': {'message': 'done'}},
    {'key': 'c2', 'comment': True, 'text': 'this flashes when the folder runs out',
     'pos': (30, 480)},
]
links = [('btn', '', 'di', 'next file'),
         ('di', 'next path out', 's1', ''),
         ('di', 'done', 'btn2', '')]
print(build('directory_iterator', 'directory_iterator - walk a folder, file by file',
            body, demo, links, demo_width=420, text_width=780, text_height=560))
