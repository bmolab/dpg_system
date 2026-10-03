"""Building weighted prompts for an image generator."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# -------------------------------------------------------------- weighted_prompt
body = """These assemble a prompt for an image generator from several phrases, each
with a weight.

THE NODES:

weighted_prompt   several phrases with weights, sent as a list
ambient_prompt    the same idea, written as a string with brackets

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
anything that only takes text. There the weight becomes a whole number of
brackets - the fraction is simply cut off - so 1.9 gives one pair, and anything
between minus one and one gives no brackets at all: the phrase is still there,
just unweighted.

NEGATIVE MEANS PUSH AWAY:
A negative weight asks for LESS of something, not merely none of it. In the list
that is a negative number; in the bracket form it is square brackets rather than
round. It is how you say "not harshly lit" without the word "harsh" ending up in
the picture, which is the usual failure of writing it as a sentence.

TYPE THE WEIGHT WITH AN @:
Both take 'phrase@weight' in their text boxes - 'a dark forest@2', 'rain@1'.
Each box is also an inlet, so a string arriving there does the same thing.

What happens to a phrase typed WITHOUT an @ differs:

weighted_prompt keeps whatever weight that slot had before - which for a fresh
slot is ZERO. The phrase is still sent, but at weight zero, which a generator
treats as nothing. After each entry the box REWRITES ITSELF to show the weight
it settled on, as 'a dark forest@2.000', so if a phrase seems to be ignored,
look at the number the box is showing you.

ambient_prompt resets the weight to zero every time, and zero there means no
brackets - the phrase goes into the string plainly. Its boxes are not rewritten.

OTHER WAYS IN (weighted_prompt):
A number on its own changes the slot's weight and keeps its phrase - the way to
automate one part of the balance. A list sets both: its words become the phrase
and a decimal number in it becomes the weight (write 2.0, not 2 - a whole
number in a list is not taken as the weight).

ambient_prompt also takes a two-element list: the phrase, then its weight.

'strength' IS THE MASTER FADER (weighted_prompt):
Every weight is multiplied by it before sending, so one number moves the whole
prompt's influence without disturbing the balance between its parts. That is the
one to automate. Changing it resends the list. ambient_prompt has no strength -
brackets cannot be scaled smoothly.

'clear' (weighted_prompt) empties every slot and sends an empty list.

SYNTAX:
weighted_prompt <count: int>
ambient_prompt <count: int>

count is the number of phrase slots; both default to 6.

EXAMPLE:
weighted_prompt 4

INPUTS and PARAMETERS:

the numbered text boxes:
A phrase each, as 'phrase@weight'. weighted_prompt's inlets are named ##0, ##1
and so on; ambient_prompt's are in_0, in_1 and so on.

strength (weighted_prompt):
Multiplies every weight. The master fader.

clear (weighted_prompt):
Empties every slot.

width (weighted_prompt option):
The width of the text boxes.

OUTPUTS:

weighted prompt out:
From weighted_prompt, the list of phrase-and-weight pairs, one for every slot
that holds a phrase. From ambient_prompt, the bracket string, each phrase
followed by a comma.

RELATED:
prompt_composer merges a weighted list like this with live speech and enduring
context, within a length budget.
fifo_string holds recent live phrases, weighted by age.
gemma_4 or a vision_describe node can supply the text in the first place."""

demo = [
    {'key': 'q0', 'init': 'string', 'pos': (30, 62), 'w': 280, 'h': 42,
     'props': {'text in': 'a dark forest@2', 'font size': '24', 'width': 240}},
    {'key': 'cq', 'comment': True, 'text': 'click these four to fill the slots',
     'pos': (540, 62)},
    {'key': 'q1', 'init': 'string', 'pos': (30, 115), 'w': 200, 'h': 42,
     'props': {'text in': 'rain@1', 'font size': '24', 'width': 160}},
    {'key': 'cq2', 'comment': True, 'text': 'string, not message - a message would\nsplit the phrase into separate words',
     'pos': (540, 115)},
    {'key': 'q2', 'init': 'string', 'pos': (30, 168), 'w': 280, 'h': 42,
     'props': {'text in': 'harsh light@-2', 'font size': '24', 'width': 240}},
    {'key': 'q3', 'init': 'string', 'pos': (30, 221), 'w': 200, 'h': 42,
     'props': {'text in': 'fog@0.5', 'font size': '24', 'width': 160}},

    {'key': 'wp', 'init': 'weighted_prompt 4', 'pos': (30, 290), 'w': 320, 'h': 200,
     'props': {'width': 200}},
    {'key': 'c0', 'comment': True, 'text': "each box rewrites itself to show its\nweight: 'a dark forest@2.000'. A phrase\ntyped with no @ keeps the slot's old\nweight - zero on a fresh slot",
     'pos': (540, 290)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 520), 'w': 420, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c5', 'comment': True, 'text': 'phrase and weight pairs - the weights\nstay numbers, so they can be moved\nsmoothly. Negative asks for LESS',
     'pos': (540, 520)},

    {'key': 'ap', 'init': 'ambient_prompt 4', 'pos': (30, 610), 'w': 320, 'h': 160},
    {'key': 'c7', 'comment': True, 'text': 'the same four phrases, as brackets',
     'pos': (540, 610)},
    {'key': 'l2', 'init': 'list', 'pos': (30, 800), 'w': 420, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c8', 'comment': True, 'text': "((more)), [[less]]. The weight is cut to a\nwhole number of brackets, so fog@0.5\ngets none - it moves in steps",
     'pos': (540, 800)},
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
         ('ap', 'weighted prompt out', 'l2', '')]
print(build('weighted_prompt', 'weighted_prompt - balance you can move', body,
            demo, links, demo_width=880, text_width=800, text_height=780))

# -------------------------------------------------------------- prompt_composer
body = """prompt_composer builds one weighted prompt for an image generator out of
live text and enduring context, and keeps it within a length budget.

THE NODE:

prompt_composer   merges live phrases with enduring context, within a budget

TWO KINDS OF TEXT:
It merges two streams that behave quite differently. 'phrases' is what is being
said now - short-lived, arriving constantly. 'context' is what has been
established and endures - the place, the time of day, who is present.

Both take a list of phrase-and-weight pairs, such as fifo_string's 'weighted
out' or weighted_prompt's output. A plain string counts as one phrase at weight
one, and a list of strings as one phrase per element at weight one - so send a
whole phrase from a string node, not a message, which would split it into
words. Each new arrival REPLACES what that inlet held before.

'prefix' and 'suffix' bracket the whole thing, for the parts that never change:
a style, a medium, a camera. They carry their own weights, 'prefix weight' and
'suffix weight'.

THE ORDER OF THE RESULT:
prefix, then context, then phrases, then suffix - or with 'order' set to
phrases_first, the phrases before the context. Context is sorted, heaviest
first. Phrases stay in exactly the order they arrived.

WHAT GETS DROPPED, AND WHAT NEVER DOES:
'char budget' caps the total length in characters (counting a space between
phrases) and 'max chunks' caps the number of phrases. A generator will silently
ignore whatever runs past its limit, and you would rather choose what gets cut
than let it be chosen for you.

When either is exceeded it drops the OLDEST phrases first, then the
lowest-weight context. The newest phrase, the prefix and the suffix are never
dropped. So the thing just said always survives, and so does the style.

'newest phrase at' DOES NOT REORDER ANYTHING:
It tells the budget which END of the incoming phrase list is the newest, so that
drops come off the old end. Set it to match whatever feeds the inlet: fifo_string
sends newest at the end unless its 'order' says otherwise. If you want the
in-progress text to lead, set that on fifo_string, not here.

WEIGHT ZERO MEANS EXPIRED, SO NEGATIVES DO NOT SURVIVE:
A phrase or context item with a weight of zero or less is treated as one whose
time is up, and is discarded on arrival. That is how live speech fades: a phrase
arrives at full weight, fifo_string lowers its weight as it ages, and it
disappears when it reaches zero without anything having to remember to remove
it.

The consequence is that the "push away" trick of weighted_prompt does not pass
through here. A phrase at -2 is discarded. If you want something suppressed in a
composed prompt, put it in the prefix or suffix, which are never dropped.

REPEATS ARE FADED, NOT REMOVED:
A context item whose text already appears in the live phrases has its weight
multiplied by 'dedupe scale' rather than being dropped. Saying something out
loud should not delete it from the background - it should stop it being said
twice at full strength. It returns to full weight once the phrase has gone.

SYNTAX:
prompt_composer

EXAMPLE:
prompt_composer

INPUTS and PARAMETERS:

phrases:
Live text: phrase-and-weight pairs, newest at one end. Sends a new prompt.

context:
Enduring text, in the same form. Sends a new prompt.

prefix / suffix:
What goes before and after, never dropped. Sends a new prompt.

clear:
Forgets the phrases and the context (not the prefix or suffix) and sends what
is left.

order:
context_first (the default) or phrases_first.

newest phrase at:
end (the default) or start - which end of the phrase list is newest.

char budget / max chunks:
How long the result may get: 300 characters and 8 phrases by default.

dedupe scale:
How far to fade a context item that is already being said. Default 0.25.

prefix weight / suffix weight:
The weights given to the prefix and suffix. Default 1.

strength:
Multiplies every weight in the result, prefix and suffix included. Default 1.

OUTPUTS:

weighted prompt out:
The list of phrase-and-weight pairs, in order.

string out:
The same phrases as plain text joined by spaces, weights discarded - for
anything that wants only words.

RELATED:
weighted_prompt builds a balanced list of phrases by hand; ambient_prompt
writes the same thing as brackets.
fifo_string holds the recent live phrases, weighted by age.
context_tracker produces the enduring context this composes with."""

demo = [
    {'key': 's1', 'init': 'string', 'pos': (30, 62), 'w': 300, 'h': 42,
     'props': {'text in': 'rain on the window.', 'font size': '24', 'width': 260}},
    {'key': 'c1', 'comment': True, 'text': 'click one, then the other: two phrases\nof live speech',
     'pos': (540, 62)},
    {'key': 's2', 'init': 'string', 'pos': (30, 115), 'w': 300, 'h': 42,
     'props': {'text in': 'someone is singing.', 'font size': '24', 'width': 260}},
    {'key': 'fs', 'init': 'fifo_string', 'pos': (30, 180), 'w': 280, 'h': 180},
    {'key': 'c2', 'comment': True, 'text': 'fifo_string keeps them as\nphrase-and-weight pairs, newest last',
     'pos': (540, 180)},
    {'key': 'px', 'init': 'string', 'pos': (30, 390), 'w': 260, 'h': 42,
     'props': {'text in': 'cinematic', 'font size': '24', 'width': 220}},
    {'key': 'c3', 'comment': True, 'text': 'the prefix - never dropped',
     'pos': (540, 390)},
    {'key': 'cx', 'init': 'string', 'pos': (30, 443), 'w': 300, 'h': 42,
     'props': {'text in': 'a cold room', 'font size': '24', 'width': 260}},
    {'key': 'c4', 'comment': True, 'text': 'context - what endures',
     'pos': (540, 443)},
    {'key': 'pc', 'init': 'prompt_composer', 'pos': (30, 510), 'w': 320, 'h': 300,
     'props': {'max chunks': 3}},
    {'key': 'c5', 'comment': True, 'text': 'max chunks is 3 here. Prefix, context and\ntwo phrases make four, so the OLDER\nphrase is dropped - never the newest',
     'pos': (540, 510)},
    {'key': 'l3', 'init': 'list', 'pos': (30, 840), 'w': 480, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c6', 'comment': True, 'text': 'prefix, then context, then phrases',
     'pos': (540, 840)},
    {'key': 's3', 'init': 'string', 'pos': (30, 900), 'w': 480, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c7', 'comment': True, 'text': 'the same words, weights discarded',
     'pos': (540, 900)},
]
links = [('s1', 'string out', 'fs', 'in'),
         ('s2', 'string out', 'fs', 'in'),
         ('fs', 'weighted out', 'pc', 'phrases'),
         ('px', 'string out', 'pc', 'prefix'),
         ('cx', 'string out', 'pc', 'context'),
         ('pc', 'weighted prompt out', 'l3', ''),
         ('pc', 'string out', 's3', '')]
print(build('prompt_composer', 'prompt_composer - live text within a budget', body,
            demo, links, demo_width=880, text_width=800, text_height=780))
