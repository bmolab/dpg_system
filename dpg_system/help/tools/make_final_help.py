"""noise_review and context_tracker."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# --------------------------------------------------------------- noise_review
body = """Walk through everything a noise report flagged, one issue at a time.

THE NODE:

noise_review   step through flagged sections across a whole batch of files

WHAT IT IS FOR:
estimate_noise_torque.py goes through a corpus of motion capture and writes a
JSON report of everything that looked wrong - stream breaks, corruption zones,
spike frames, glitch clusters, and the various lens flags. That report is a long
list of file-and-frame references, and reading it is not the same as SEEING what
was flagged.

This node turns the report into a review session. Load the JSON, point it at the
corpus root, and Prev/Next walk you through the flags. Each step sends the NPZ
path and the start frame, so a take player can load that file and jump straight
to the moment in question.

The point is to look at each one and decide whether the detector was right.

THE CHECKBOXES CHOOSE WHAT YOU ARE REVIEWING:
'stream breaks', 'corruption zones', 'spike frames', 'glitch clusters', 'lens
flags' and 'clean sections' each turn a category of flag on or off in the walk.

Reviewing one category at a time is far more useful than all of them mixed
together, because the judgement is different for each: a stream break is a fact,
a spike is a suspicion, and a lens flag is a statistical lean. Keeping them
separate stops one kind of decision contaminating another.

'clean sections' is worth using deliberately. Walking through what the detector
did NOT flag is how you find false negatives, which no amount of staring at the
positives will show you.

'prev file' and 'next file' jump a whole file rather than a flag, for when a file
is obviously bad and stepping through forty flags in it tells you nothing new.

RECORDING THE VERDICT:
'classification' and 'flag unrepresentative' are where the judgement goes -
whether this flag was right, and whether this file is odd enough that it should
not count towards a rate. 'folder yield' shows how much of a subset survives,
which is the number that tells you whether a filter is usable.

SYNTAX:
noise_review

EXAMPLE:
noise_review

INPUTS and PARAMETERS:

json path / load:
The report from estimate_noise_torque.py.

amass root:
Where the corpus actually is, so the paths in the report resolve.

prev / next:
Step through flags.

prev file / next file:
Step a whole file.

section / filter:
Which flag, and which categories to include.

OUTPUTS: 

npz path / frame / end frame:
The file and the moment, for a player to open.

bang:
Fires on each step, to trigger the load.

RELATED:
take to load the file this points at.
The validation notes in dpg_system/noise_estimation/ for what the detector is
actually claiming."""

demo = [
    {'key': 'nr', 'init': 'noise_review', 'pos': (30, 62), 'w': 340, 'h': 520},
    {'key': 'c0', 'comment': True, 'text': "review ONE category at a time - the\njudgement is different for each. A break\nis a fact, a spike is a suspicion, a lens\nflag is a statistical lean\nwalk 'clean sections' too - it is the\nonly way to find FALSE NEGATIVES",
     'pos': (30, 600)},

    {'key': 'l1', 'init': 'list', 'pos': (420, 62), 'w': 420, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c6', 'comment': True, 'text': 'the file this flag is in', 'pos': (420, 112)},
    {'key': 'i1', 'init': 'int', 'pos': (420, 160), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c7', 'comment': True, 'text': 'and the frame to jump to', 'pos': (420, 210)},

    {'key': 'ot', 'init': 'take', 'pos': (420, 260), 'w': 320, 'h': 200},
    {'key': 'c8', 'comment': True, 'text': 'the take player loads it and goes\nstraight to the moment in question\nso the loop is: next, look, decide',
     'pos': (420, 475)},
]
links = [('nr', 'npz path', 'l1', ''),
         ('nr', 'frame', 'i1', '')]
print(build('noise_review', 'noise_review - looking at what was flagged', body,
            demo, links, demo_width=880, text_width=800, text_height=740))

# ------------------------------------------------------------ context_tracker
body = """Keeps track of where, when and who, from a stream of text.

THE NODE:

context_tracker   pulls enduring context out of passing words

WHAT IT IS DOING:
Speech arrives a phrase at a time and most of it is transient. But some of it
establishes something that stays true: a place, a time of day, a season, who is
present, what they are carrying. This node watches the stream and keeps those.

The result is a CONTEXT - a small set of registers describing the world the words
have built - which is what you feed an image generator so that successive images
belong to the same scene rather than each starting fresh.

THE REGISTERS:
Place, time of day, time of year, era, weather, style, medium, actors and the
props they carry. Each has its own vocabulary inlet, so you can replace what
counts as a place-word or a style-word without touching the node.

PLACE AND TIME ARE LADDERS, NOT SINGLE VALUES:
A room is inside a building is inside a city. An hour is inside a part of the day
is inside a season is inside an era. The tracker holds these at several scales at
once, so saying "the kitchen" does not throw away "in winter" - it only replaces
the level it actually names.

That is why moving place ejects props: a thing being carried belongs to a scene,
and when the scene changes the tracker stops asserting that the cup is still
there. Actors persist across a move; what they were holding does not.

ACTORS CARRY ATTRIBUTES:
An actor is not just a name - the tracker sorts what is said about them into
slots, so "the woman in the red coat" attaches the coat to the woman rather than
adding it as a loose fact. Later mentions update the same actor.

THE WEIGHTS ARE FOR WHAT COMES NEXT:
Every register has a weight. They do not change what is tracked - they change how
strongly each part is asserted when the context is handed on, which is what
prompt_composer uses to balance the prompt.

Turn 'place weight' up and the images stay in the room; turn it down and the room
becomes a suggestion. That is a compositional control, not a detection one.

IT WANTS THE BIG SPACY MODEL:
It works better with en_core_web_lg than the small one, because deciding that a
word is a place rather than a name is exactly the kind of judgement the larger
model is better at.

'set context' seeds it directly, for starting somewhere rather than waiting for
the text to establish it. 'clear' empties everything.

SYNTAX:
context_tracker

EXAMPLE:
context_tracker

INPUTS and PARAMETERS:

text in:
Phrases, as they arrive.

set context / clear:
Seed it, or empty it.

time vocab / place vocab / style vocab / medium vocab / weather vocab:
Replace what counts as each kind of word.

era events / artist in:
Historical anchors, and an artist reference.

the weights:
How strongly each register is asserted downstream.

OUTPUTS: 

context out:
The context, weighted, ready for prompt_composer.

context dict:
The same thing structured, for anything that wants to read one register.

detected:
What it just recognised, which is how you see it working.

RELATED:
prompt_composer, which is what 'context out' is shaped for.
whisper to supply the text.
spacy_vector and the spacy nodes for the model underneath."""

demo = [
    {'key': 'wh', 'init': 'whisper', 'pos': (30, 62), 'w': 300, 'h': 260},
    {'key': 'c0', 'comment': True, 'text': 'phrases, as they settle', 'pos': (30, 340)},

    {'key': 'ct', 'init': 'context_tracker', 'pos': (30, 390), 'w': 340, 'h': 520},
    {'key': 'c1', 'comment': True, 'text': 'place and time are LADDERS - a room is\nin a building is in a city. Saying\n"the kitchen" does not throw away\n"in winter"\nmoving place EJECTS PROPS: what someone\nwas carrying belongs to the scene, and\nthe actors persist but the cup does not',
     'pos': (30, 925)},

    {'key': 'td', 'init': 'text_display', 'pos': (420, 390), 'w': 340, 'h': 200,
     'props': {'width': 320, 'height': 160, 'wrap': True, 'max_lines': 60,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c8', 'comment': True, 'text': "'detected' - watch this to see it\nworking, and to find out why it is not",
     'pos': (420, 605)},

    {'key': 'pc', 'init': 'prompt_composer', 'pos': (420, 690), 'w': 320, 'h': 340},
    {'key': 'c10', 'comment': True, 'text': "'context out' is shaped for this - the\nenduring half of the prompt, against the\nlive phrases as the other half\nthe weights do not change what is\ntracked - they change how strongly each\npart is asserted downstream",
     'pos': (420, 1045)},
]
links = [('wh', 'phrases', 'ct', 'text in'),
         ('wh', 'phrases', 'pc', 'phrases'),
         ('ct', 'detected', 'td', '###text in'),
         ('ct', 'context out', 'pc', 'context')]
print(build('context_tracker', 'context_tracker - the world the words build',
            body, demo, links, demo_width=800, text_width=800, text_height=770))
