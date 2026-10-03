"""UDP, TCP, and distributed torch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

CHOOSE = """
UDP OR TCP - THE CHOICE THAT MATTERS:
UDP throws a packet at an address and does not look back. Nothing is
acknowledged, nothing is retried, and packets can be lost or arrive out of
order. TCP opens a connection and guarantees that everything arrives, in order.

TCP's guarantee is not free: to keep it, a slow or stalled receiver makes the
SENDER wait. For a continuous stream that is the wrong trade - the lag grows
and never recovers, and you end up watching a body that is several seconds
behind the person.

For streaming live data at frame rate, UDP is almost always right. A dropped
frame is invisible; a growing delay is not, and the next frame is along in
sixteen milliseconds anyway.

Use TCP when every byte must arrive and lateness is better than loss - a set of
weights, a configuration, a single large transfer, anything you would have to
send again if part of it went missing.
"""

# ------------------------------------------------------------- udp_numpy_send
body = """These send NumPy arrays between machines over UDP - fire and forget.

THE NODES:

udp_numpy_send     send an array to an address and port
udp_numpy_receive  listen on a port

This is the way to get pose data, effort values, point clouds or any other array
from one machine to another while it is happening. One machine runs the suit,
another does the rendering or the sound; the array crosses between them.
""" + CHOOSE + """
SIZE LIMITS ARE REAL:
A UDP packet has a practical size ceiling - beyond roughly 64 kilobytes it will
not go at all, and well before that a packet large enough to be split across
network fragments is lost if ANY fragment is lost, which makes losses much more
likely than the raw packet loss rate suggests.

A pose is tiny and no trouble. A point cloud or an image may not be. If a
stream works locally and fails across a real network, size is the first thing to
suspect - send less per packet, or use TCP.

SEND ONLY WHAT CHANGED:
Because there is no cost to a receiver falling behind, it is tempting to send
everything every frame. On a busy network that is what causes loss. Thinning the
stream with subsample, or gating on change, costs nothing when nothing is
happening and leaves headroom for when it is.

BOTH ENDS DEFAULT TO PORT 3500:
Dropped in with no arguments a send and a receive will find each other. That is
not true of their TCP counterparts, whose defaults differ - so do not carry the
habit across. Setting the port explicitly at both ends, as this patch does,
costs nothing and never surprises you.

SYNTAX:
udp_numpy_send <ip> <port>
udp_numpy_receive <port>

EXAMPLE:
udp_numpy_send 192.168.1.20 9000

INPUTS and PARAMETERS:

data:
The array to send. Receiving it sends it.

ip / port:
Where to send, and which port to listen on. Both ends must agree on the port.

OUTPUTS: 

received data:
The array, reconstructed.

WHAT ARRIVES IS WHAT WAS SENT, OR NOTHING:
There is no partial delivery - an array either arrives whole or does not arrive.
So the receiving side never has to check for corruption, only for silence. If
the stream stops, look at the sender, the port and the firewall, in that order.

RELATED:
ip_address tells you what address this machine is on, which is what the other
end needs.
osc_send carries small named messages rather than arrays - better for controls,
worse for data."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'aj', 'init': 'active_joints', 'pos': (30, 400), 'w': 220, 'h': 90},
    {'key': 'sub', 'init': 'subsample 2', 'pos': (30, 505), 'w': 160, 'h': 80,
     'props': {'rate': 2}},
    {'key': 'c0', 'comment': True, 'text': 'thin the stream: it costs nothing when\nstill, and leaves headroom when not',
     'pos': (30, 600)},
    {'key': 'us', 'init': 'udp_numpy_send 127.0.0.1 9000', 'pos': (30, 675),
     'w': 300, 'h': 160},
    {'key': 'ur', 'init': 'udp_numpy_receive 9000', 'pos': (380, 675), 'w': 280, 'h': 120},
    {'key': 'inf', 'init': 'info', 'pos': (380, 815), 'w': 240, 'h': 80},
    {'key': 'c2', 'comment': True, 'text': 'both ends must agree on the port',
     'pos': (380, 910)},
    {'key': 'ipb', 'init': 'button', 'pos': (380, 400), 'w': 88, 'h': 46},
    {'key': 'ip', 'init': 'ip_address', 'pos': (380, 455), 'w': 240, 'h': 120},
    {'key': 'l1', 'init': 'list', 'pos': (380, 590), 'w': 260, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'what the other end should send to',
     'pos': (380, 640)},
]
links = [('ipb', '', 'ip', 'get_ip'),
         ('sh', 'body 1 quaternions', 'aj', 'full pose quats in'),
         ('aj', 'active joint quats out', 'sub', 'input'),
         ('sub', 'out', 'us', 'data'),
         ('ur', 'received data', 'inf', 'in'),
         ('ip', 'ip_addresses_out', 'l1', '')]
print(build('udp_numpy_send', 'udp_numpy_send - arrays across the network', body,
            demo, links, demo_width=700, text_width=800, text_height=720))

# ------------------------------------------------------------- tcp_numpy_send
body = """These send arrays over TCP - a connection, with delivery guaranteed.

THE NODES:

tcp_numpy_send      send arrays over a connection
tcp_numpy_receive   accept a connection and receive them
tcp_latent_send     the same, carrying a position and a serial number alongside
tcp_latent_receive  the receiving end of that
""" + CHOOSE + """
ONE END SERVES, THE OTHER CONNECTS:
The receiving node has a 'serving ip' - it waits for a connection. The sending
node has an 'ip' - it goes and finds one. Both have a 'connected' outlet, and
watching it is how you tell a network problem from a data problem: if connected
never goes true, nothing about the data matters yet.

Start the receiver first. A sender with nothing to connect to will keep trying,
but the order saves confusion.

THE DEFAULT PORTS DO NOT MATCH:
Worth knowing before it costs you an afternoon: tcp_latent_send defaults to port
4501 and tcp_latent_receive defaults to 4500. Dropped in with no arguments they
will sit there, both reporting not connected, looking for all the world like a
network fault.

Set the port on both ends explicitly, as this patch does, and the question never
comes up.

AND THE RECEIVER DOES NOT DEFAULT TO LOOPBACK:
The other half of the same trap. tcp_latent_receive sets its 'serving ip' to
this machine's NETWORK address when it is created, not to 127.0.0.1 - which is
what you want between two machines, and wrong for two patches on one.

Left alone it binds to the network address while a sender aimed at 127.0.0.1
knocks on loopback and is refused. The symptom is a receiver that is plainly
running and a sender that cannot see it.

For two patches on one machine, set 127.0.0.1 at BOTH ends, as this patch does.

THE LATENT VARIANTS CARRY IDENTITY:
tcp_latent_send takes a 'position' and a 'serial' alongside the array, and the
receiving end hands all three back out.

That matters when the arrays are frames of something - latents from a model,
successive poses - and the receiver needs to know WHICH frame it just got rather
than only that it got one. Over TCP nothing is lost or reordered, so a serial is
not there to detect loss; it is there so both ends can agree what they are
talking about, which matters as soon as one side is doing work that takes
longer than a frame.

SYNTAX:
tcp_numpy_send <ip> <port>
tcp_numpy_receive <port>

EXAMPLE:
tcp_latent_send

INPUTS and PARAMETERS:

data / latents:
The array to send.

position / serial (the latent nodes):
What this array is - where it sits, and which one it is.

ip / serving ip / port:
Where to connect to, where to listen, and on which port.

OUTPUTS: 

connected:
Whether the connection is up. Check this first, always.

data / latents:
What arrived.

position / serial:
The identity that was sent with it.

A SLOW RECEIVER SLOWS THE SENDER:
This is the property to keep in mind. TCP will not drop data to keep up, so a
receiver that cannot process fast enough applies back-pressure all the way to
the sending patch - which can show up as the sender's frame rate falling for no
locally visible reason.

If a patch slows down when a network node is connected and speeds up when it is
not, that is this, and UDP is the answer."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 'np.rand 16', 'pos': (30, 120), 'w': 180, 'h': 180},
    {'key': 'ctr', 'init': 'counter', 'pos': (240, 120), 'w': 180, 'h': 110},
    {'key': 'ts', 'init': 'tcp_latent_send 127.0.0.1 4500', 'pos': (30, 325), 'w': 280, 'h': 200},
    {'key': 'i1', 'init': 'int', 'pos': (30, 545), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c8', 'comment': True, 'text': 'serial and position are set first -\nthe latents inlet is what sends',
     'pos': (240, 245)},
    {'key': 'c0', 'comment': True, 'text': 'check connected before anything else',
     'pos': (30, 595)},
    {'key': 'tr', 'init': 'tcp_latent_receive 127.0.0.1 4500', 'pos': (380, 325), 'w': 280, 'h': 160},
    {'key': 'i2', 'init': 'int', 'pos': (380, 505), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 'the serial says WHICH frame arrived\nstart the receiver first',
     'pos': (380, 555)},
    {'key': 'c3', 'comment': True, 'text': 'BOTH set to 4500 - the defaults differ\nand 127.0.0.1 so both ends are on the\nsame address on this one machine',
     'pos': (30, 625)},
    {'key': 'inf', 'init': 'info', 'pos': (380, 625), 'w': 240, 'h': 80},
]
# position and serial are cold inlets: they must be set BEFORE the array
# arrives, because it is the latents inlet that triggers the send. Drive the
# counter first, then let its output produce the array.
links = [('btn', '', 'ctr', 'input'),
         ('ctr', 'count out', 'ts', 'serial'),
         ('ctr', 'count out', 'ts', 'position'),
         ('ctr', 'count out', 'rnd', ''),
         ('rnd', 'random array', 'ts', 'latents'),
         ('ts', 'connected', 'i1', ''),
         ('tr', 'serial', 'i2', ''),
         ('tr', 'latents', 'inf', 'in')]
print(build('tcp_numpy_send', 'tcp_numpy_send - a connection, with guarantees', body,
            demo, links, demo_width=700, text_width=800, text_height=720))

# -------------------------------------------------------------- process_group
body = """process_group - joining a torch distributed process group.

THE NODE:
It makes this patch one participant in torch's own distributed machinery, with
every participant given a RANK and the group knowing its WORLD SIZE, and sends
tensors to another participant as part of that group.

process_group IS NOT THE SAME AS SENDING AN ARRAY:
The socket nodes move data between two patches that each do their own work.
This joins several processes into ONE computation - what you want when a model
is too large for one GPU, or when the same computation should run across several
machines with tensors moving between them as part of the calculation rather than
as messages.

SETTINGS COME FROM THE ARGUMENTS, IN ORDER:
ip, port, rank, world size, backend. Anything left out takes its default:
127.0.0.1, 29500, rank 0, world size 2, backend gloo.

The group is formed from these arguments the moment the node is created. The
ip, port, rank, backend and world_size fields on the node show those values,
but changing them afterwards does NOT change the group - to change a setting,
retype the node with new arguments.

Every participant must agree on the ip, the port and the world size, and each
must have a different rank. Rank 0 is the coordinator, and the ip and port are
where it can be found - on rank 0 itself that is its own address. 'backend'
selects how they talk: gloo works on any machine; nccl is for NVIDIA GPUs.

CREATING THE NODE STARTS THE RENDEZVOUS:
A process group is not something you set up and then connect. The moment the
node exists it begins looking for the other participants, and it waits until all
of them have arrived - which, if you are still building the other end, is a
while, and if you mistyped the world size, is forever. The waiting happens in
the background, so the patch keeps running meanwhile.

That is why there is no live process_group in this patch: opening a help file
should not go looking for machines. Build the other participants first, then
make this node last. One patch can only belong to one group, so there is no
point in making two of these in the same patch.

SENDING:
A tensor (or anything that converts to one) arriving at data_to_send is sent to
rank 1. The destination_rank field is there, but in the current version it is
not read - sends always go to rank 1. Only one send can be in flight at a time.

WHAT DOES NOT WORK YET:
Be warned that this node is unfinished. The node never checks whether a send
has finished, so sending_complete never sends anything - and since the node
also never clears its "already sending" flag, only the first tensor sent ever
goes out. Nothing starts a receive either, so received_data never sends
anything, and expected_tensor_example only prepares the space a receive would
use. Treat it as a starting point rather than a working link.

SYNTAX:
process_group [ip] [port] [rank] [world size] [backend]

EXAMPLE:
process_group 192.168.1.20 29500 1 2 gloo

INPUTS and PARAMETERS:

data_to_send:
A tensor to send to rank 1.

destination_rank:
Meant to choose who gets the tensor; currently ignored.

expected_tensor_example:
A tensor with the shape and dtype a receive should expect, because a
distributed receive has to allocate before it knows what is coming.

ip / port / rank / backend / world_size:
Show the settings the group was made with. Set them with the arguments.

OUTPUTS: 

sending_complete:
Meant to send a bang when a send finishes; currently never fires.

received_data:
Meant to send a received tensor; currently never fires.

WHEN NOT TO REACH FOR process_group:
If two patches are exchanging results, the socket nodes are simpler and more
robust - they tolerate one end restarting, which a process group does not.
Distributed torch is for one computation spread across participants, and it
expects all of them to be present and healthy.

RELATED:
ip_address tells you what address to give the other participants.
udp_numpy_send and tcp_numpy_send for passing arrays between patches."""

demo = [
    {'key': 'c0', 'comment': True, 'text': 'process_group is deliberately NOT in this\npatch: creating one starts a rendezvous\nthat waits for its other participants\n\nmake it with all its arguments, last,\nonce the other participants exist:\n\n    process_group 192.168.1.20 29500 1 2 gloo\n\nevery participant agrees on ip, port\nand world size; each has its own rank',
     'pos': (30, 62)},
    {'key': 'ip', 'init': 'ip_address', 'pos': (30, 250), 'w': 240, 'h': 120},
    {'key': 'c1', 'comment': True, 'text': 'ip_address: the address to give the\nother participants',
     'pos': (30, 385)},
]
links = []
print(build('process_group', 'process_group - one computation across machines', body,
            demo, links, demo_width=560, text_width=790, text_height=760))

# ----------------------------------------------------------------- ip_address
body = """ip_address - what addresses this machine has.

THE NODE:
It reports the addresses this machine is reachable at - every IPv4 address on
every network interface, shown in the node itself and sent as a list. That is
what you tell the other end to send to, and it saves going to look it up in a
system panel - which matters when the address changes, as it does on a network
you do not control.

WHICH ONE TO USE:
A machine usually has several. The one to use is on the same network as the
other machine; the loopback address 127.0.0.1 only ever reaches this machine
itself, which is the right choice for testing two patches on one computer and
useless for anything else.

WHEN IT LOOKS:
The node looks once when it is created, and again whenever anything arrives at
get_ip, or the get_ip button is clicked. It does not watch for changes on its
own - if you move to another network, ask again. Up to 20 addresses are shown.

Only IPv4 addresses are listed - the dotted four-number kind that the socket
nodes expect.

SYNTAX:
ip_address

EXAMPLE:
ip_address

INPUTS and PARAMETERS:

get_ip:
Anything arriving here, or a click on its button, looks the addresses up again
and sends them.

OUTPUTS: 

ip_addresses_out:
A list of this machine's IPv4 addresses.

RELATED:
udp_numpy_send / tcp_numpy_send take the address of the other end.
process_group needs the coordinator's address as its first argument."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'ip', 'init': 'ip_address', 'pos': (30, 120), 'w': 240, 'h': 120},
    {'key': 'l1', 'init': 'list', 'pos': (30, 255), 'w': 320, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'tell the other end this address\n127.0.0.1 only reaches this machine -\nright for testing, useless otherwise',
     'pos': (30, 305)},
]
links = [('btn', '', 'ip', 'get_ip'), ('ip', 'ip_addresses_out', 'l1', '')]
print(build('ip_address', 'ip_address - where this machine can be reached', body,
            demo, links, demo_width=560, text_width=790, text_height=560))
