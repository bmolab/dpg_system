"""finger_zones: which finger is this touch - home circles, sticky names, re-acquisition."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import INT

body = """Turns the erae node's fingers - numbered in the order they landed - into
named fingers: thumb, index, middle, ring, pinky, of one hand or both.

THE NODE:

finger_zones   touch in, the same touch out with a finger name, and a picture
               of the home circles to draw on the pad

HOW A TOUCH GETS ITS FINGER:
Five circles (ten for 'both') sit in the natural resting positions of a hand
laid palm down on the pad. When a touch lands it takes the finger of the
nearest circle that no other touch holds, and it KEEPS that finger for as
long as it stays down, however far it wanders - a middle finger that slides
over the ring finger's circle is still the middle finger. The names are freed
on 'up'. So the circles only matter at the moment of landing; after that the
finger is an identity, not a place.

RE-ACQUISITION:
A finger that lifts and lands again close to where it left - within 'reacquire
radius', inside 'reacquire time' - gets its old name back rather than being
judged afresh by the circles. That is what keeps a trill or a re-press on a
finger that has drifted from home from turning the index finger into the
middle one. Only a name that is still free can be re-acquired.

WHEN THE CIRCLE IS TAKEN:
Two touches landing in the same circle cannot both be the same finger, so the
second takes the nearest circle still free. Set 'catch radius' and a landing
farther than that from every free circle is left unassigned: it still comes
out of 'touch', with finger -1, so nothing is lost, but it holds no name and
no voice.

THE PICTURE:
'zones image' sends the circles as a height x width x 3 image whenever the
layout changes, on 'draw', and (with 'show held') whenever a finger lands or
lifts, held circles filled in 'held color'. Connect it to the erae node's
'backdrop' and the pad shows the home positions under the pressure halos, or
to 'image' to draw it plainly. Width and height are the zone's size - 43 x 25
for a full-pad zone - and the erae node's 'zone size' outlet can set them.

THE LAYOUT:
'hands' is right, left or both. 'centre x' and 'centre y' place the hand (the
midpoint between the hands for 'both'), 'spread' scales it - the thumb-to-
pinky span is about twice 'spread' - 'separation' is the distance between the
two hands' centres, and 'rotation' turns each hand, both splaying outward by
the same angle. 'radius' is the drawn circle, in pixels; it does not affect
which finger a touch gets.

OUTPUTS:
'touch' is finger, action, x, y, z per event, where finger is 0..4 (thumb to
pinky) or 0..9 (left thumb to right pinky), -1 if unassigned, or the name
itself with 'finger as name'. 'voices' is an array with one row per finger:
held, x, y, z - a fixed slot per finger, so a patch can read the ring finger's
pressure without routing on ids. 'count' is how many fingers hold a name.

Through 'source' the node also reads mpe_in's touch lists, taking pitch as x,
slide as y and pressure as z; the circles are then laid out in note and slide
units.

SYNTAX:
finger_zones

EXAMPLE:
finger_zones

INPUTS and PARAMETERS:

touch:
A finger event from erae (finger zone action x y z) or mpe_in.

zone size:
zone width height, as the erae node sends it, or width height: sets the size
of the picture.

draw:
Send the zones image now.

reset:
Forget every held finger and recent lift.

hands:
right, left or both.

source:
erae or mpe_in - which touch list shape to read.

width / height:
The picture's size in pixels.

centre x / centre y / spread / separation / rotation:
Where the hands sit, how big they are, how far apart, and how they are turned.

radius:
The drawn circle's radius in pixels.

catch radius:
Farther than this from every free circle, a landing is unassigned. 0 means no
limit.

reacquire radius / reacquire time:
How close to where a finger lifted, and how soon after, a landing counts as
the same finger again.

finger as name:
Send 'thumb' 'index' ... (or 'left thumb' ...) in place of 0 1 ....

show held:
Send the image again whenever a finger lands or lifts, with held circles
filled.

zone color / held color / background:
The circles, the circles a finger holds, and the rest of the picture.

OUTPUTS:

touch:
finger, action, x, y, z per event.

voices:
One row per finger: held, x, y, z.

count:
How many fingers hold a name.

zones image:
The home circles as a picture, top row first."""

demo = [
    {'key': 'er', 'init': 'erae', 'pos': (30, 62), 'w': 300, 'h': 420,
     'props': {'show touches': True}},
    {'key': 'fz', 'init': 'finger_zones', 'pos': (30, 520), 'w': 300, 'h': 120,
     'props': {'finger as name': True}},
    {'key': 'c0', 'comment': True, 'text': 'the erae fingers in, named fingers out;\nthe circles go back to the pad as the backdrop\nunder the halos, and the zone size sets the picture',
     'pos': (360, 520)},
    {'key': 'up', 'init': 'unpack 5', 'pos': (30, 680), 'w': 260, 'h': 80},
    {'key': 's0', 'init': 'string', 'pos': (30, 790), 'w': 160, 'h': 42, 'props': {'font size': '24'}},
    {'key': 'f3', 'init': 'float', 'pos': (220, 790), 'w': 100, 'h': 42,
     'props': {'format': '%.2f', 'width': 100, 'font size': '24'}},
    {'key': 'f4', 'init': 'float', 'pos': (380, 790), 'w': 100, 'h': 42,
     'props': {'format': '%.2f', 'width': 100, 'font size': '24'}},
    {'key': 'f5', 'init': 'float', 'pos': (540, 790), 'w': 100, 'h': 42,
     'props': {'format': '%.2f', 'width': 100, 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'finger              x                 y                 z',
     'pos': (30, 850)},
    {'key': 'i0', 'init': 'int', 'pos': (380, 680), 'w': 80, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'fingers holding a name', 'pos': (470, 690)},
]
links = [('er', 'touch', 'fz', 'touch'),
         ('er', 'zone size', 'fz', 'zone size'),
         ('fz', 'zones image', 'er', 'backdrop'),
         ('fz', 'touch', 'up', ''),
         ('up', '', 's0', ''), ('up', '', 'f3', '', 2), ('up', '', 'f4', '', 3), ('up', '', 'f5', '', 4),
         ('fz', 'count', 'i0', '')]
print(build('finger_zones', 'finger_zones - which finger is this touch', body, demo, links,
            demo_width=720, text_width=790, text_height=900))
