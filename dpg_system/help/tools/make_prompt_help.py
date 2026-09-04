"""Building weighted prompts for an image generator."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """These assemble a prompt for an image generator, with weights.

THE NODES:

weighted_prompt   several phrases with weights, as a list
ambient_prompt    the same idea, written with brackets instead
prompt_composer   merges live speech with enduring context, within a budget

WHY WEIGHTS:
A prompt is rarely one thing. It is a place, a light, a mood and a subject, and
they do not all matter equally - nor does their balance stay still while a piece
runs. A weight per phrase lets the balance be something a patch controls rather
than something you retype.

TWO WAYS OF SAYING THE SAME THING:
weighted_prompt sends a LIST of phrase-and-weight pairs:

    [['a dark forest', 2.0], ['rain', 1.0], ['harsh light', -2.0]]

ambient_prompt sends a STRING, using the older bracket convention where nesting
is emphasis:

    ((a dark forest)), (rain), [[harsh light]],

Same intent, two conventions. Which you want depends entirely on what is
downstream: a generator that accepts weighted lists should get the list, because
the weights stay numbers and can be varied smoothly. The bracket string is for
anything that only takes text - the weight has to be a whole number of brackets,
so it moves in steps.

NEGATIVE MEANS PUSH AWAY:
A negative weight asks for LESS of something, not merely none of it. In the list
that is a negative number; in the bracket form it is square brackets rather than
round. It is how you say "not harshly lit" without the word "harsh" ending up in
the picture, which is the usual failure of writing it as a sentence.

TYPE THE WEIGHT WITH AN @:
Both take 'phrase@weight' in the text boxes - 'a dark forest@2', 'rain@1'. You
can also send a list of the phrase and its number.

The box then REWRITES ITSELF to show the weight it settled on, as
'a dark forest@2.000'. That is worth knowing, because a phrase typed with no @
keeps whatever weight that slot had before - which for a fresh slot is ZERO, and
a phrase weighted zero contributes nothing. If a phrase seems to be ignored,
look at the number the box is showing you.

'strength' IS THE MASTER FADER:
Every weight is multiplied by it before sending, so one number moves the whole
prompt's influence without disturbing the balance between its parts. That is the
one to automate.

prompt_composer IS FOR LIVE TEXT:
It merges two streams that behave quite differently. 'phrases' is what is being
said now - short-lived, arriving constantly. 'context' is what has been
established and endures - the place, the time of day, who is present.

'prefix' and 'suffix' bracket the whole thing, for the parts that never change:
a style, a medium, a camera.

'order' puts context or phrases first. 'char budget' and 'max chunks' cap the
result, because a generator will silently ignore whatever runs past its limit
and you would rather choose what gets cut than let it be chosen for you.

WHAT GETS DROPPED, AND WHAT NEVER DOES:
When the budget is exceeded it drops the OLDEST phrases first, then the
lowest-weight context. The newest phrase, the prefix and the suffix are never
dropped. So the thing just said always survives, and so does the style.

'newest phrase at' DOES NOT REORDER ANYTHING:
It tells the budget which END of the incoming list is the newest, so that drops
come off the old end. The phrases themselves pass through in exactly the order
they arrived. If you want the in-progress text to lead, set that on whatever is
feeding it, not here.

WEIGHT ZERO MEANS EXPIRED, SO NEGATIVES DO NOT SURVIVE THE COMPOSER:
prompt_composer treats a weight of zero or less as a phrase whose time is up and
drops it. That is how live speech fades: a phrase arrives at full weight, decays
as it ages, and disappears when it reaches zero without anything having to
remember to remove it.

The consequence is that the "push away" trick does not pass through here. A
phrase at -2 goes straight through weighted_prompt and ambient_prompt and is
discarded by prompt_composer. If you want something suppressed in a composed
prompt, put it in the prefix or suffix, which are never dropped.

REPEATS ARE FADED, NOT REMOVED:
A context item that is already present in the live phrases has its weight scaled
down by 'dedupe scale' rather than being dropped. Saying something out loud
should not delete it from the background - it should stop it being said twice at
full strength.

SYNTAX:
weighted_prompt <count>
ambient_prompt <count>
prompt_composer

EXAMPLE:
weighted_prompt 6

INPUTS and PARAMETERS:

the numbered text boxes:
A phrase each, as 'phrase@weight'.

strength:
Multiplies every weight. The master fader.

phrases / context (prompt_composer):
Live text, and enduring text.

prefix / suffix:
What goes before and after, never dropped.

char budget / max chunks:
How long the result may get.

dedupe scale:
How far to fade a context item that is already being said.

order / newest phrase at:
Context or phrases first; and which end of the phrase list is newest.

OUTPUTS: 

weighted prompt out:
The list of phrase-and-weight pairs. From ambient_prompt, the bracket string.

string out (prompt_composer):
The same thing as plain text, weights discarded - for anything that wants only
words.

RELATED:
context_tracker produces the enduring context this composes with.
fifo_string holds the recent live phrases.
gemma_4 or a vision_describe node can supply the text in the first place."""

demo = [
    {'key': 'q0', 'init': 'string', 'pos': (30, 62), 'w': 280, 'h': 42,
     'props': {'text in': 'a dark forest@2', 'font size': '24', 'width': 240}},
    {'key': 'q1', 'init': 'string', 'pos': (30, 115), 'w': 200, 'h': 42,
     'props': {'text in': 'rain@1', 'font size': '24', 'width': 160}},
    {'key': 'q2', 'init': 'string', 'pos': (30, 168), 'w': 280, 'h': 42,
     'props': {'text in': 'harsh light@-2', 'font size': '24', 'width': 240}},
    {'key': 'q3', 'init': 'string', 'pos': (30, 221), 'w': 200, 'h': 42,
     'props': {'text in': 'fog@0.5', 'font size': '24', 'width': 160}},
    {'key': 'cq', 'comment': True, 'text': 'click these four to fill the slots.',
     'pos': (30, 271)},
    {'key': 'cq2', 'comment': True, 'text': 'string, not message - a message would\nsplit the phrase into separate words',
     'pos': (30, 1330)},
    {'key': 'wp', 'init': 'weighted_prompt 4', 'pos': (30, 315), 'w': 320, 'h': 280,
     'props': {'width': 200}},
    {'key': 'c0', 'comment': True, 'text': "type 'a dark forest@2' - the box then\nrewrites itself as 'a dark forest@2.000'\na phrase with no @ keeps the slot old\nweight - zero on a fresh one, and zero\ncontributes nothing",
     'pos': (30, 610)},

    {'key': 'l1', 'init': 'list', 'pos': (400, 62), 'w': 420, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c5', 'comment': True, 'text': 'phrase and weight pairs - the weights\nstay numbers, so they can be moved\nsmoothly. Negative asks for LESS',
     'pos': (400, 112)},

    {'key': 'ap', 'init': 'ambient_prompt 4', 'pos': (400, 230), 'w': 320, 'h': 260},
    {'key': 'l2', 'init': 'list', 'pos': (400, 505), 'w': 420, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c8', 'comment': True, 'text': 'the same idea as brackets - ((more)),\n[[less]]. For anything that only takes\ntext, but the weight moves in steps',
     'pos': (400, 555)},

    {'key': 'px', 'init': 'string', 'pos': (400, 690), 'w': 260, 'h': 42,
     'props': {'text in': 'cinematic', 'font size': '24', 'width': 220}},
    {'key': 'cx', 'init': 'string', 'pos': (400, 745), 'w': 300, 'h': 42,
     'props': {'text in': 'a cold room', 'font size': '24', 'width': 260}},
    {'key': 'cx2', 'comment': True, 'text': 'prefix and context. The phrases come\nfrom weighted_prompt above - already\nphrase-and-weight pairs, which is what\nthis inlet wants',
     'pos': (400, 795)},
    {'key': 'pc', 'init': 'prompt_composer', 'pos': (30, 790), 'w': 320, 'h': 340},
    {'key': 'l3', 'init': 'list', 'pos': (30, 1150), 'w': 480, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c11', 'comment': True, 'text': 'prefix, then context, then what is being\nsaid, then suffix. Over budget it drops\noldest phrases first - never the newest,\nand never the prefix or suffix',
     'pos': (30, 1200)},
]
links = [('q0', 'string out', 'wp', '##0'),
         ('q1', 'string out', 'wp', '##1'),
         ('q2', 'string out', 'wp', '##2'),
         ('q3', 'string out', 'wp', '##3'),
         ('q0', 'string out', 'ap', 'in_0'),
         ('q1', 'string out', 'ap', 'in_1'),
         ('q2', 'string out', 'ap', 'in_2'),
         ('q3', 'string out', 'ap', 'in_3'),
         ('wp', 'weighted prompt out', 'l1', ''),
         ('ap', 'weighted prompt out', 'l2', ''),
         ('cx', 'string out', 'pc', 'context'),
         ('px', 'string out', 'pc', 'prefix'),
         ('wp', 'weighted prompt out', 'pc', 'phrases'),
         ('pc', 'weighted prompt out', 'l3', '')]
print(build('weighted_prompt', 'weighted prompts - balance you can move', body,
            demo, links, demo_width=860, text_width=800, text_height=780))
