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
body = """These nodes cut lists into pieces and feed them out.

THE NODES:

slice_list    cut a list in two at a position you choose
sublist       pull out particular positions, by index
stream_list   send every element in turn, one after another

slice_list divides once and gives you both halves on separate outlets - 
the first N elements and everything after. Use it to peel a header off a 
message, or to split a packed reading into the part you want and the rest.

sublist picks elements out by index. Give it the indices and it returns just 
those, in the order you asked for - which also means it can reorder or repeat.

stream_list turns a list into a sequence of separate messages, sent one after 
another as fast as the patch will take them. This is how you make something 
that expects single values process a whole list.

SYNTAX:
slice_list <position: int>
sublist <indices>
stream_list

EXAMPLE:
slice_list 2

INPUTS and PARAMETERS:

list input / list in:
The list. Receiving it triggers the node. 
A string is split on spaces first, so a message works as a list.

slice after (slice_list):
The position to cut at. The first outlet gets everything up to and including 
this index; the second gets the rest. 
If the list is too short to cut, everything goes to the first outlet and the 
second sends nothing.

Indices (sublist):
Which positions to take, counted from 0.

output only if slice 2 (slice_list):
When checked, the node stays silent unless there is genuinely something in the 
second half - so a list too short to cut produces nothing at all rather than 
passing through whole. Use it when a short list means "not ready".

OUTPUTS: 

slice 1 out / slice 2 out:
The two halves.

output (sublist):
The chosen elements, as a list.

stream out (stream_list):
Each element in turn, as separate messages."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 250, 'h': 42,
     'props': {'text in': '10 20 30 40 50', 'font size': '24'}},
    {'key': 'sl', 'init': 'slice_list 1', 'pos': (30, 180), 'w': 160, 'h': 100,
     'props': {'slice after': 1, 'output only if slice 2': False}},
    {'key': 'l1', 'init': 'list', 'pos': (30, 300), 'w': 180, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'l2', 'init': 'list', 'pos': (240, 300), 'w': 180, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'first two, then the rest', 'pos': (30, 350)},
    {'key': 'st', 'init': 'stream_list', 'pos': (30, 395), 'w': 150, 'h': 60},
    {'key': 'a1', 'init': 'accumulate', 'pos': (30, 470), 'w': 140, 'h': 100},
    {'key': 'i1', 'init': 'int', 'pos': (30, 585), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 'every element arrives separately: 150',
     'pos': (30, 635)},
]
links = [('btn', '', 'm1', ''), ('m1', 'message out', 'sl', 'list input'),
         ('sl', 'slice 1 out', 'l1', ''), ('sl', 'slice 2 out', 'l2', ''),
         ('m1', 'message out', 'st', 'list in'),
         ('st', 'stream out', 'a1', 'in')]
print(build('slice_list', 'slice_list - cut a list into pieces', body, demo, links,
            demo_width=440, text_width=790, text_height=620))

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
body = """These three nodes let a patch arrange its own window and interface.

They are for finishing a patch - turning something you built into something 
someone can use, without them seeing the wiring.

THE NODES:

present                  switch the patch into presentation mode
active_widget            report which widget the mouse is over
patch_window_position    set the window's position and size

Presentation mode is the important one. Every node has a presentation state, 
and nodes set to hidden disappear when the patch is presented - so you can lay 
out a clean panel of just the controls, on top of the working patch, and switch 
between the two. The "open as presentation" option makes a patch open that way 
from the start, which is how you hand it to someone who should not be looking 
at the machinery.

SYNTAX:
present
active_widget
patch_window_position

INPUTS and PARAMETERS:

open as presentation (present):
When checked, this patch opens directly into presentation mode rather than 
showing the patch as built.

active_widget:
Reports the widget currently under the mouse. Use it when a patch needs to know 
what the person is pointing at - context help, or a control that shows what it 
does when you approach it.

top / left / width / height (patch_window_position):
The window's position and size, in pixels. Send values here to move or resize 
the patch window while it runs - to bring it up at a known size on a particular 
screen, or to fit a projector.

OUTPUTS: 

None of these three have outlets - they act on the patch itself rather than 
passing data on."""

demo = [
    {'key': 'pr', 'init': 'present', 'pos': (30, 62), 'w': 190, 'h': 60},
    {'key': 'c0', 'comment': True, 'text': 'tick to open this patch presented',
     'pos': (30, 130)},
    {'key': 'aw', 'init': 'active_widget', 'pos': (30, 175), 'w': 190, 'h': 60},
    {'key': 'c1', 'comment': True, 'text': 'shows what the mouse is over',
     'pos': (30, 245)},
    {'key': 'pw', 'init': 'patch_window_position', 'pos': (30, 290), 'w': 210, 'h': 140},
    {'key': 'c2', 'comment': True, 'text': 'send numbers here to move the window\nset a node to hidden to leave it out',
     'pos': (30, 440)},
]
print(build('present', 'present - arrange the patch for someone else', body,
            demo, [], demo_width=420, text_width=780, text_height=520))

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
