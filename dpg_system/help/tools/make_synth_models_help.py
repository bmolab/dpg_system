"""The physical models, grouped by how the energy gets in."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

COMMON = """
THE THREE WAYS TO PLAY ONE:

Most of these accept excitation three ways at once, and they are genuinely 
different gestures rather than three spellings of the same one.

  the button      click it, or bang it, for one event
  a trigger       patch a signal and it plays with sample accuracy, struck as 
                  hard as the trigger is tall
  'excite in'     patch any signal and it drives the model CONTINUOUSLY - 
                  enveloped noise bows it, and an effort stream played straight 
                  in makes the movement itself the excitation

The third is the one that matters for effort data. There is no triggering, no 
threshold and no note: the model simply sounds while the movement is happening 
and is silent when it is not, because that is what the physics does.
"""

# ---------------------------------------------------- struck and plucked models
body = """These models are sounded by a HIT: something strikes them and they ring.

They divide into what is struck and what does the striking.

WHAT RINGS:

modal~      a bank of resonant modes - bells, bars, bowls, membranes. 
            'material' picks the tuning table, 'frequency' places it, 
            'decay' and 'brightness' stretch it
resonator~  the same node
string~     a Karplus-Strong string
pluck~      the same node
drum~       the membrane bank plus the physics modal~ leaves out: a hard hit 
            lands pitched sharp and bends down through its ring, which is 
            'tension', where tabla and toms live. 'snares' are wires shaken 
            by the head itself

WHAT HITS:

strike~     a mallet SWUNG - one hit when you ask for one, in a choice of 
            characters. Patch its output into any resonator's excite inlet
bounce~     a mallet DROPPED. Everything after the first fall is gravity, 
            which is what a roll is made of
drop~       the same node
rattle~     loose things in a container, shaken and turned - actual particles, 
            with positions and velocities and walls that hit them
shaker~     shaken percussion by the STATISTICS of a gesture rather than by 
            simulating grains, which is cheap and very good for rain and 
            sleighbells
rain~       the same node

bounce~ IS THE INTERESTING ONE:
'drop' rising from zero drops the mallet from that height, and the bounces 
accelerate and weaken geometrically until they blur into a buzz - which is 
exactly what gravity does and exactly what a drum roll is. An LFO into 'drop' 
is one stroke per cycle. A hand's height patched in is a roll played by 
lowering your hand.

rattle~ VERSUS shaker~:
shaker~ is a collision RATE driven by an agitation - nobody wants to simulate 
a hundred thousand grains of rain. rattle~ has actual particles, which is what 
you want when there are few enough things in the container to hear 
individually, and when tipping and turning it should matter.
""" + COMMON + """
SYNTAX:
modal~
string~ <frequency>
bounce~

EXAMPLE:
modal~

INPUTS and PARAMETERS:

frequency / pitch:
Where the model is tuned. pitch is exponential, in octaves.

material (modal~, drum~):
Which mode table - the tuning that makes it a bell rather than a bar.

hardness / position:
What it is struck with and where it lands. Position changes which modes are 
excited, which is why the same bell struck at the rim and at the crown are 
different sounds.

decay / brightness:
How long it rings and how much high end survives.

excite in / sensitivity:
Continuous excitation, and how much of it is taken.

tension / snares (drum~):
The downward pitch bend of a hard hit, and the wires under the head.

drop / gravity / bounce (bounce~):
The height, the acceleration and how much is returned each time.

OUTPUTS: 

out:
The sound.

modes out:
The mode table, so it can be inspected, stored, or sent to another node.

RELATED:
shape_modes works a mode table out from a described SHAPE rather than taking a 
preset - patch its 'modes' outlet into modal~ or rub~ and the table becomes 
whatever you described."""

demo = [
    {'key': 'ck', 'init': 'clock~ 0.7', 'pos': (30, 62), 'w': 220, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'tick run to start striking', 'pos': (30, 272)},
    {'key': 'st', 'init': 'strike~', 'pos': (30, 315), 'w': 220, 'h': 200},
    {'key': 'c1', 'comment': True, 'text': 'a mallet swung: one hit per ask',
     'pos': (30, 525)},
    {'key': 'md', 'init': 'modal~', 'pos': (30, 565), 'w': 260, 'h': 300},
    {'key': 'c2', 'comment': True, 'text': 'change material, then hardness and position',
     'pos': (30, 875)},
    {'key': 'bn', 'init': 'bounce~', 'pos': (330, 315), 'w': 240, 'h': 220},
    {'key': 'c3', 'comment': True, 'text': 'raise drop: the roll is gravity, not a pattern',
     'pos': (330, 545)},
    {'key': 'sc', 'init': 'scope~', 'pos': (330, 585), 'w': 260, 'h': 220},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 971), 'w': 220, 'h': 220},
]
links = [('ck', 'trigger', 'st', 'hit'),
         ('st', 'out', 'md', 'excite in'),
         ('bn', 'out', 'md', 'excite in'),
         ('md', 'out', 'sc', 'in'),
         ('md', 'out', 'fo', 'left')]
print(build('modal~', 'modal~ and the struck models - hit it, hear it ring', body,
            demo, links, demo_width=620, text_width=830, text_height=820))

# --------------------------------------------------------------- blown models
body = """These models are sounded by BREATH: air moving through or past something.

There is no trigger on any of them, and that is the point. A blown instrument 
has none - everything about playing it lives in the pressure. Lean on the 
slider, patch an adsr~ for tongued notes, an lfo~ for vibrato, or an effort 
stream so that moving hard is blowing hard.

THE NODES:

wind~      a blown instrument. The reed speaks from about half pressure; the 
           flute wants nearly a full breath and cracks when overblown
reed~      the same node
flute~     the same node
brass~     one bore, the note chosen by the lip. 'frequency' is the 
           instrument's size - its pedal fundamental - and 'lip' is the 
           embouchure, mapped across the first sixteen harmonics
horn~      the same node
blow~      a blown MODE TABLE - the third hand, after the mallet and the bow. 
           modal~ strikes a table, rub~ bows it, this blows it
pipe~      the same node
bubbles~   liquid: each bubble a decaying sine at the pitch its size dictates, 
           rising as it dies
gurgle~    the same node
whoosh~    motion through air - patch a speed, hear the swish
swish~     the same node
vessel~    a vessel with water in it - modal~ with the water added

brass~ CLIMBS THE SERIES:
Sweeping 'lip' does not slide the pitch. It steps up the harmonic series, 
like a bugler, because the lip's own resonance locks to the nearest bore mode. 
That is the model behaving as the instrument does, not a quantiser.

whoosh~ IS PHYSICS, NOT A MAPPING:
Its pitch is speed over size - the physics of vortex shedding rather than a 
relationship anyone chose - and loudness rises steeply the way aeolian sound 
does. Slow motion whispers, fast motion roars, stillness is silent. 
Patch a limb speed straight in.

bubbles~ IS PLAYED BY FLOW:
'flow' is the whole interface - the rate rides it and stillness is silent. 
'size' runs from fizz to glug. The rising pitch as each bubble dies is the 
Minnaert inflection, and it is what makes water sound like water.

SYNTAX:
wind~
brass~ <frequency>
whoosh~

EXAMPLE:
brass~ 116

INPUTS and PARAMETERS:

pressure / breath:
How hard it is being blown, and the noise in the breath. 
This is the playing interface - there is nothing else to trigger.

frequency / pitch:
The instrument's size, and exponential transposition.

lip (brass~):
The embouchure. Sweep it to climb the harmonic series.

embouchure (wind~):
Where the air is directed - the difference between a note speaking and not.

mute / stem / wah (brass~):
What is in the bell.

flow / size / spread / chirp / gulp (bubbles~):
The rate of bubbles, how big, how varied, and the character of each.

speed / size / edge / wake (whoosh~):
How fast the thing is moving, how big it is, how sharp its edge, and the 
turbulence behind it.

fill / tip / turn / swirl (vessel~):
How much water, how far it is tipped, and how it moves.

OUTPUTS: 

out:
The sound."""

demo = [
    {'key': 'sl', 'init': 'slider 0.0', 'pos': (30, 62), 'w': 220, 'h': 60,
     'props': {'min': 0.0, 'max': 1.0, 'format': '%.2f', 'width': 200}},
    {'key': 'c0', 'comment': True, 'text': 'this is the whole playing interface',
     'pos': (30, 130)},
    {'key': 'sg', 'init': 'sig~', 'pos': (30, 170), 'w': 200, 'h': 140},
    {'key': 'br', 'init': 'brass~ 116', 'pos': (30, 330), 'w': 260, 'h': 300},
    {'key': 'c1', 'comment': True, 'text': 'lean on the slider to make it speak\nthen sweep lip: it climbs the series',
     'pos': (30, 640)},
    {'key': 'wh', 'init': 'whoosh~', 'pos': (330, 330), 'w': 240, 'h': 220},
    {'key': 'c3', 'comment': True, 'text': 'patch a limb speed into speed', 'pos': (330, 560)},
    {'key': 'sc', 'init': 'scope~', 'pos': (330, 600), 'w': 260, 'h': 220},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 710), 'w': 220, 'h': 220},
]
links = [('sl', 'float out', 'sg', 'value'),
         ('sg', 'signal', 'br', 'pressure'),
         ('sg', 'signal', 'wh', 'speed'),
         ('br', 'out', 'sc', 'in'),
         ('br', 'out', 'fo', 'left')]
print(build('wind~', 'wind~ and the blown models - no trigger, only breath', body,
            demo, links, demo_width=620, text_width=820, text_height=800))

# ------------------------------------------------------------ friction models
body = """These models are sounded by RUBBING: sustained contact, stick and slip.

Friction is what makes a bow sing rather than scrape, a hinge creak rather than 
squeak, a spinning coin rattle faster as it settles. In every one of these the 
sound comes from a surface catching and releasing many times a second, and the 
character comes from how it catches.

THE NODES:

bow~     a bowed string. Velocity and force are the whole bow arm
bowed~   the same node
rub~     bowed glass - modal~'s tables under bow~'s hands
glass~   the same node
stroke~  a bow ARM: coordinated velocity and force from one gesture
bowing~  the same node
strain~  solids under stress - bending made audible
creak~   the same node
motor~   a machine, with speed and load as two effort streams
engine~  the same node
spin~    a spinning disc settling: the rattle that runs away
coin~    the same node

bow~ MISBEHAVES CORRECTLY:
The two sliders are mapped so their middles bow cleanly at any pitch. 
The trouble at the edges - octave whistle from a fast light bow, subharmonic 
scratch from a slow heavy one - is the model's own, and it happens in the same 
directions a real instrument's does. Those are not bugs to avoid; they are how 
you get the sound of someone learning, or straining.

stroke~ IS THE ARM, NOT THE INSTRUMENT:
Patch its 'velocity' and 'force' outlets to the same inlets on bow~ or rub~, 
with the destination's own knobs at zero, and the two move together the way a 
player's arm does - velocity a cornered trapezoid that crosses the awkward 
low-speed region quickly, force leaning into the string.

strain~ TAKES EFFORT DIRECTLY:
It is the model whose input IS effort. Patch a joint angle, a stretch, a slow 
fader into 'strain' and the model runs on it. Motion releases events, stillness 
is silent by construction, and the material remembers where it has been bent - 
so repeating a movement quiets it, the way a hinge worked back and forth 
settles down.

spin~ RUNS AWAY:
A dropped coin, a plate set down spinning, a hubcap in the road. The contact 
point races round the rim at a rate that goes as one over the square root of 
the tilt, so as the lean bleeds away the rattle ACCELERATES without limit and 
then stops. It is not a bounce, and no envelope produces that shape.

motor~ IS TWO EFFORT STREAMS:
'speed' is rotation - pitch linear in it, loudness rising, stillness silent. 
'load' is torque - each firing punchier and less regular, the bearing grind 
rising underneath. Velocity into one and torque into the other and a joint 
becomes a machine.

SYNTAX:
bow~ <frequency>
strain~
spin~

EXAMPLE:
bow~ 196

INPUTS and PARAMETERS:

velocity / force (bow~, rub~):
How fast the bow moves and how hard it presses. The whole bow arm.

position:
Where on the string it is bowed.

strain / resist / stretch (strain~):
The bending, how much the material resists, and how far it gives.

squeal / grind / texture / pops (strain~):
The characters the release can take.

speed / load (motor~):
Rotation and torque.

spin / size / settle / rush / twist (spin~):
How fast it is spinning, how big it is, how quickly the lean bleeds away, 
and the acceleration into the finish.

OUTPUTS: 

out:
The sound.

velocity / force / tick (stroke~):
The bow arm's two coordinated signals, and a pulse at each stroke.

grind / landing / rate / face (spin~):
The separate parts of the settle, and where it ends up."""

demo = [
    {'key': 'ck', 'init': 'clock~ 0.4', 'pos': (30, 62), 'w': 220, 'h': 200},
    {'key': 'sk', 'init': 'stroke~', 'pos': (30, 280), 'w': 240, 'h': 260},
    {'key': 'c0', 'comment': True, 'text': 'the bow arm: two coordinated signals',
     'pos': (30, 550)},
    {'key': 'bw', 'init': 'bow~ 196', 'pos': (30, 590), 'w': 260, 'h': 260},
    {'key': 'c1', 'comment': True, 'text': 'set its own velocity and force knobs to zero\nthe triad sums, so the arm drives it',
     'pos': (30, 860)},
    {'key': 'sc', 'init': 'scope~', 'pos': (330, 590), 'w': 260, 'h': 220},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 930), 'w': 220, 'h': 220},
]
links = [('ck', 'trigger', 'sk', 'gate'),
         ('sk', 'velocity', 'bw', 'velocity'),
         ('sk', 'force', 'bw', 'force'),
         ('bw', 'out', 'sc', 'in'),
         ('bw', 'out', 'fo', 'left')]
print(build('bow~', 'bow~ and the friction models - stick, slip, sing', body,
            demo, links, demo_width=620, text_width=830, text_height=820))
