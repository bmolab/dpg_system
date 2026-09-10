"""mpe_in: a multi-touch controller's MIDI read as fingers."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import INT, FLT

body = """A multi-touch controller's MIDI, read as fingers rather than as notes.

THE NODE:

mpe_in   one voice per MIDI channel, each finger out as a list

WHAT MPE IS:
Ordinary MIDI has one pitch bend and one pressure per channel, so two fingers on
one channel fight over them. MPE gives every finger its own channel: the note
says where it landed, pitch bend says how far it has slid sideways since,
controller 74 says how far up, and channel pressure says how hard. An Erae, a
Linnstrument or a Seaboard in MPE mode is sixteen channels of that.

This node keeps one voice per channel and hands each finger out as a list, so
the pad reads the way a touch surface should. A plain single-channel controller
works the same way - its channel is voice 0 - so it is also the node for an
Erae layout that is not in MPE mode yet.

THE TOUCH LIST:
'touch' sends voice, state, note, pitch, pressure, slide, velocity. The voice
is the channel, 0..15, which MPE keeps constant for the life of a finger. The
state is 0 down, 1 move, 2 up. The pitch is the note plus the bend in
semitones, a float, so a slide reads as a continuous position. Pressure, slide
and velocity come out 0..1 unless 'normalize' is off.

'voices' is the whole state at once - a 16 by 5 array of active, note, pitch,
pressure, slide - which is the handy form for drawing all the fingers or
feeding a bank of voices in one go. 'count' is how many are down.

100 HZ, COALESCED:
Expressive controllers stream bend and pressure on every finger about a hundred
times a second, as separate messages. With 'coalesce' on the node gathers those
and sends one 'move' per finger per frame; downs and ups are never dropped.
Turn it off to get every message the instant it arrives.

BEND RANGE IS THE THING TO SET:
The bend arrives as a fraction of a range the controller and you agreed on and
MIDI never carries. MPE's convention is 48 semitones per finger; if the pitch
you see slides too far or too little, this is the number to change.

MODE:
'all channels' treats every channel as a finger. 'mpe lower' makes channel 1
the master - its bend applies to every finger, its other messages come out of
'master' - and 'mpe upper' does the same with channel 16.

SYNTAX:
mpe_in
mpe_in <port name>

EXAMPLE:
mpe_in

INPUTS and PARAMETERS:

port:
The device.

mode:
all channels, mpe lower, or mpe upper.

bend range / master bend range:
Semitones at full bend, per finger and for the master channel.

slide cc:
Which controller carries the upward slide - 74 by convention.

normalize:
Pressure, slide and velocity as 0..1 rather than 0..127.

coalesce:
One move per finger per frame, or every message as it comes.

OUTPUTS:

touch:
voice, state, note, pitch, pressure, slide, velocity per event.

voices:
The 16 by 5 array of every voice: active, note, pitch, pressure, slide.

count:
How many fingers are down.

master:
The master channel's messages, raw, in the mpe modes."""

demo = [
    {'key': 'mp', 'init': 'mpe_in', 'pos': (30, 62), 'w': 300, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'one list per finger event:\nvoice state note pitch pressure slide velocity',
     'pos': (30, 280)},
    {'key': 'up', 'init': 'unpack 7', 'pos': (30, 340), 'w': 340, 'h': 110},
    {'key': 'i0', 'init': 'int', 'pos': (30, 500), 'w': 80, 'h': 42, 'props': INT},
    {'key': 'i1', 'init': 'int', 'pos': (170, 500), 'w': 80, 'h': 42, 'props': INT},
    {'key': 'f3', 'init': 'float', 'pos': (310, 500), 'w': 100, 'h': 42, 'props': FLT},
    {'key': 'f4', 'init': 'float', 'pos': (470, 500), 'w': 100, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'voice          state            pitch              pressure',
     'pos': (30, 560)},
    {'key': 'i5', 'init': 'int', 'pos': (380, 62), 'w': 80, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'fingers down', 'pos': (380, 115)},
]
links = [('mp', 'touch', 'up', ''),
         ('up', '', 'i0', ''), ('up', '', 'i1', '', 1), ('up', '', 'f3', '', 3), ('up', '', 'f4', '', 4),
         ('mp', 'count', 'i5', '')]
print(build('mpe_in', 'mpe_in - fingers from a multi-touch controller', body, demo, links,
            demo_width=640, text_width=790, text_height=760))
