"""erae: the Embodme Erae 2 through its API - finger stream in, drawing out."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import INT

body = """The Embodme Erae 2 (or Erae Touch) as a touch surface and a display,
through Embodme's own API rather than through MIDI notes.

THE NODE:

erae   finger stream in, pixels / rectangles / images out

WHY NOT JUST MIDI:
The pad already sends MIDI and MPE, and midi_note_in and the others read that
fine. But a note is a coarse thing to make of a finger: a number for where it
landed, a bent pitch for where it went, and no way to tell two fingers apart
once they cross. The API gives you each finger as it is - a float x, y and
pressure z, with an identity that lasts from the moment it lands to the moment
it lifts. And it lets you draw on the pad, so the surface can show the state
of the patch under your hands.

API ZONES:
The API only sees 'API Zone' elements, which you place on the pad in Embodme's
Erae Lab and give an index, 0 and up. The rest of a layout - keys, faders, pads
- goes on speaking MIDI as before, and that arrives at 'midi received'. A zone
index that is not in the current layout answers 'query' with a size of 0 0 and
never sends a finger, which is the first thing to check when nothing comes out.

THE FINGER STREAM:
'touch' sends one list per event: finger, zone, action, x, y, z. The finger
number is a small slot - 0 for the first finger down, 1 for the next while the
first is held - and it is freed when the finger lifts, so it is stable enough
to route on. The action is 0 for down, 1 for move, 2 for up, or the words
'down' 'move' 'up' if 'action as name' is ticked. x and y are in zone pixels
- 0..42 across and 0..24 up on a full-pad zone, bottom-left origin - and z is
the pressure, 0..1, and it is always 0 on the 'up' event so a stream of z ends
at rest. Events come at about 100 a second per finger.

DRAWING:
'pixel' takes x y r g b, 'rect' takes x y w h r g b, both with colours 0..255.
'image' takes an array of height x width x 3 (or a grey height x width), top
row first like any picture, and puts it in the zone at 'image x', 'image y'.
Long images are cut into runs of 32 pixels per message on the way out, as the
device asks. 'clear' wipes the zone. Every drawing command goes to the zone
named in 'zone'.

show touches DRAWS THE FINGERS FOR YOU:
Tick 'show touches' and the node paints the zone itself: a dot under each
finger with a ring round it that widens as you press, the way Erae Lab's key
element shows a slide. 'halo min radius' and 'halo max radius' are the ring at
no pressure and at full pressure, 'halo width' its thickness, and the two
colours are the ring and the empty zone. Only the pixels that changed since
the last frame are sent, at most 'display fps' times a second, so the pad keeps
up with four fingers moving at once.

api mode IS THE SWITCH:
The node turns the API on when it is made and off when it is deleted. Untick
'api mode' to hand the pad back to its ordinary behaviour without deleting the
node.

SYNTAX:
erae
erae <port name>

EXAMPLE:
erae

INPUTS and PARAMETERS:

api mode:
Whether the API is switched on in the device.

zone:
Which API Zone the queries and drawing commands address.

query:
Ask the size of the current zone; the answer comes out of 'zone size'.

clear:
Wipe the current zone's display.

pixel / rect / image:
Drawing, as above.

in port / out port:
The device. The API travels on 'Erae 2 MIDI', not the '(MPE)' port.

product:
Erae 2 or Erae Touch - they differ by one byte in the address.

action as name:
Send 'down' 'move' 'up' in place of 0 1 2.

image x / image y:
Where the bottom-left corner of an image lands in the zone.

show touches:
Paint a pressure halo under every finger in the zone.

halo color / background:
The ring, and the zone with no fingers on it.

halo min radius / halo max radius / halo width:
The ring's radius in pixels at zero and at full pressure, and its thickness.

pressure range:
The z value that counts as full pressure - 1 for the Erae 2.

display fps:
The most frames a second the halo display will send.

halo offset:
Nudges the halos onto the fingertips. The LED grid sits half a pixel left of
and below the touch grid, so 0.5 lines them up; larger moves them left and down.

OUTPUTS:

touch:
finger, zone, action, x, y, z per event.

zone size:
zone, width, height in answer to 'query' - 0 0 if the zone is not laid out.

api version:
The API version the device reports, sent once when the API is enabled.

midi received:
Everything the pad sends that is not API traffic - its ordinary notes and
controllers."""

demo = [
    {'key': 'er', 'init': 'erae', 'pos': (30, 62), 'w': 300, 'h': 420},
    {'key': 'c0', 'comment': True, 'text': 'one list per finger event:\nfinger zone action x y z',
     'pos': (30, 500)},
    {'key': 'up', 'init': 'unpack 6', 'pos': (30, 560), 'w': 300, 'h': 80},
    {'key': 'i0', 'init': 'int', 'pos': (30, 680), 'w': 80, 'h': 42, 'props': INT},
    {'key': 'f3', 'init': 'float', 'pos': (170, 680), 'w': 100, 'h': 42,
     'props': {'format': '%.2f', 'width': 100, 'font size': '24'}},
    {'key': 'f4', 'init': 'float', 'pos': (330, 680), 'w': 100, 'h': 42,
     'props': {'format': '%.2f', 'width': 100, 'font size': '24'}},
    {'key': 'f5', 'init': 'float', 'pos': (490, 680), 'w': 100, 'h': 42,
     'props': {'format': '%.2f', 'width': 100, 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'finger          x                 y                 z',
     'pos': (30, 740)},
    {'key': 'btn', 'init': 'button', 'pos': (380, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (380, 118), 'w': 260, 'h': 42,
     'props': {'text in': '2 2 6 4 255 40 0', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'x y w h r g b - a rectangle lit\non the pad, in the current zone',
     'pos': (380, 170)},
    {'key': 'p0', 'init': 'print', 'pos': (380, 260), 'w': 160, 'h': 80},
    {'key': 'c3', 'comment': True, 'text': "zone width height - 0 0 means\nthe zone is not in the layout",
     'pos': (380, 350)},
]
links = [('er', 'touch', 'up', ''),
         ('up', '', 'i0', ''), ('up', '', 'f3', '', 3), ('up', '', 'f4', '', 4), ('up', '', 'f5', '', 5),
         ('btn', '', 'm1', ''), ('m1', 'message out', 'er', 'rect'),
         ('er', 'zone size', 'p0', 'in')]
print(build('erae', 'erae - the Embodme Erae 2 through its API', body, demo, links,
            demo_width=720, text_width=790, text_height=900))
