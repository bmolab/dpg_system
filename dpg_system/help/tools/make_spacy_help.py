"""spacy: meaning as vectors, and the sentence builder."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

HM = lambda n=4, lo=0.0, hi=1.0, fmt='%.2f': {
    'color': 'viridis', 'width': 200, 'height': 100, 'sample count': n,
    'min y': lo, 'max y': hi, 'update_mode': 'heat_map', 'number format': fmt}
MSG = lambda t: {'text in': t, 'font size': '24'}

MODEL = """
THEY ALL SHARE ONE LANGUAGE MODEL:
The first of these nodes to be created loads en_core_web_lg, and every spacy
node in every patch then shares it. It is a few hundred megabytes and takes a
moment, so the first one costs something and the rest are free.

If the model is not installed the nodes still build, but they do nothing at all
- no output, no error each time. A spacy node that is silent when everything is
wired correctly is the thing to suspect.
"""

# --------------------------------------------------------------- spacy_vector
body = """These turn words into vectors, so that meaning can be measured.

THE NODES:

spacy_vector      a phrase as 300 numbers
spacy_similarity  how alike two phrases are
spacy_confusion   every phrase in one list against every phrase in another

WHAT A WORD VECTOR IS:
Each word has been given a position in a 300-dimensional space, arranged so that
words used in similar contexts sit near each other. Nothing was hand-labelled;
the positions come from how the words were actually used across a very large
amount of text.

This is what lets a patch treat meaning as a quantity. 'apple' and 'pear' are
close together, 'apple' and 'engine' are far apart, and the distance is a number
you can threshold, plot or send to a synth.

A PHRASE IS THE AVERAGE OF ITS WORDS, AND THAT MATTERS:
spacy_vector gives one 300-number vector for whatever you send it, however long.
It gets there by averaging the vectors of the individual words - and averaging
is a blunt instrument.

It means word order is invisible: 'the dog bit the man' and 'the man bit the
dog' come out identical. It also means every extra word pulls the result towards
the middle, because common words like 'a', 'the' and 'in' sit near the centre of
the space and drag everything towards each other.

READ THE SIMILARITY NUMBERS HONESTLY:
Similarity runs 0 to 1, but it does NOT use the bottom of that range. Measured
on this model, over a spread of deliberately unrelated phrases:

single words        0.01 to 0.61, averaging about 0.21
whole phrases       0.25 to 0.85, averaging about 0.44

So 0.85 is a strong match between phrases and 0.4 means nothing in particular -
it is roughly what any two unrelated phrases score. Judge a number against that
baseline rather than against zero, and calibrate on your own material before
choosing a threshold.

The compression gets worse the longer the phrases are. 'apple' against itself is
1.0; against 'a red apple' it is 0.78; against 'a red apple sitting on a wooden
table in the morning light' it is 0.52 - the same subject, scored lower purely
for having more words around it. Compare short things.

spacy_confusion IS THE ONE TO REACH FOR FIRST:
It takes two lists and gives back the whole grid of comparisons at once - one
row per phrase in the second list, one column per phrase in the first. Sent to a
heat_map you can see the structure immediately, which is far more use than
squinting at single numbers.

SYNTAX:
spacy_vector
spacy_similarity
spacy_confusion

EXAMPLE:
spacy_similarity

INPUTS and PARAMETERS:

phrase in / phrase 2 in:
The two phrases to compare. Either one arriving recomputes the answer.

input / input2 (spacy_confusion):
Two LISTS of words. input2 becomes the rows, input the columns.

OUTPUTS: 

phrase vector out:
300 numbers.

phrase similarity out:
One number, 0 to 1.

output (spacy_confusion):
A matrix, rows by columns.
""" + MODEL + """
RELATED:
lemma first, if you want 'running' and 'ran' to be the same word.
np.cosine_similarity does the same arithmetic on vectors you already have."""

demo = [
    {'key': 'm1', 'init': 'message', 'pos': (30, 62), 'w': 220, 'h': 42,
     'props': MSG('a red apple')},
    {'key': 'm2', 'init': 'message', 'pos': (30, 120), 'w': 220, 'h': 42,
     'props': MSG('a green pear')},
    {'key': 'sim', 'init': 'spacy_similarity', 'pos': (30, 180), 'w': 230, 'h': 110},
    {'key': 'f1', 'init': 'float', 'pos': (30, 305), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c0', 'comment': True, 'text': 'about 0.85 - a strong match',
     'pos': (30, 355)},
    {'key': 'm3', 'init': 'message', 'pos': (30, 400), 'w': 280, 'h': 42,
     'props': MSG('quantum field theory')},
    {'key': 'c1', 'comment': True, 'text': 'send this to the second inlet instead:\nabout 0.27 - and that is as low as\nunrelated phrases go. 0.4 means nothing.',
     'pos': (30, 450)},

    {'key': 'vec', 'init': 'spacy_vector', 'pos': (350, 62), 'w': 220, 'h': 80},
    {'key': 'inf', 'init': 'info', 'pos': (350, 160), 'w': 240, 'h': 80},
    {'key': 'c4', 'comment': True, 'text': '300 numbers, however long the phrase\nword order is invisible: it is an average',
     'pos': (350, 255)},

    {'key': 'm4', 'init': 'message', 'pos': (350, 400), 'w': 300, 'h': 42,
     'props': MSG('apple pear engine')},
    {'key': 'm5', 'init': 'message', 'pos': (350, 460), 'w': 360, 'h': 42,
     'props': MSG('fruit machine banana theory')},
    {'key': 'con', 'init': 'spacy_confusion', 'pos': (350, 520), 'w': 230, 'h': 110},
    {'key': 'hm', 'init': 'heat_map', 'pos': (350, 645), 'w': 208, 'h': 148,
     'props': HM(4)},
    {'key': 'c6', 'comment': True, 'text': 'rows apple/pear/engine, columns\nfruit/machine/banana/theory - the two\nfruits light up together, engine alone',
     'pos': (350, 805)},
]
links = [('m1', 'message out', 'sim', 'phrase in'),
         ('m2', 'message out', 'sim', 'phrase 2 in'),
         ('sim', 'phrase similarity out', 'f1', ''),
         ('m1', 'message out', 'vec', 'phrase in'),
         ('vec', 'phrase vector out', 'inf', 'in'),
         ('m4', 'message out', 'con', 'input2'),
         ('m5', 'message out', 'con', 'input'),
         ('con', 'output', 'hm', 'y')]
print(build('spacy_vector', 'spacy vectors - meaning as a measurable quantity', body,
            demo, links, demo_width=780, text_width=800, text_height=760))

# ------------------------------------------------------------------- rephrase
body = """Two nodes that use the grammar of a sentence rather than its vectors.

THE NODES:

lemma     every word reduced to its dictionary form
rephrase  fragments accumulated into a growing phrase

lemma IS THE SMALL USEFUL ONE:
It gives back the dictionary form of every word: 'geese' becomes 'goose', 'were'
becomes 'be', 'running' becomes 'run'. It is not chopping off endings - it knows
the irregular forms, because it has parsed the sentence and knows which word is
doing what.

The use is matching. Counting words, comparing phrases or looking for a term all
work better on lemmas, because 'run', 'running' and 'ran' stop being three
different things. Put it in front of spacy_similarity or a word count and the
results get noticeably steadier.

It outputs a LIST, one lemma per word, punctuation included.

rephrase IS NOT WHAT ITS NAME SUGGESTS:
It does not simplify a sentence. It ACCUMULATES: each fragment you send it is
folded into the phrase built so far, using the grammar of both to decide where
the new material belongs.

Send it these, one at a time:

    a woman             -> a woman
    in a red coat       -> a woman, in a red coat
    she is walking      -> a woman , in a red walking coat
    through the park    -> a woman , in a red walking coat, through the park

(the spacing around the commas really is uneven - it is stitching fragments
together, not writing them out)

Notice the third step. 'she' was resolved to the woman it has been tracking, and
the verb was folded in as a modifier rather than appended - which is how 'a red
walking coat' happens. That is the character of this node: it is a generative
instrument that produces compressed, slightly strange descriptions, not a
grammar engine that produces correct ones. Judge it by whether the results are
worth having.

Send 'no', 'wrong' or 'go back' and it steps back to the previous phrase.

THE THRESHOLDS, AND WHICH WAY THEY POINT:
'complexity replace threshold' is an upper limit on what it will TRY. Complexity
counts nouns, verbs and prepositions, plus a quarter of the word count - so
about 1.5 for 'a woman' and 14 for a long descriptive sentence. A fragment above
the threshold is passed through untouched rather than folded in, so the default
of 6 means it works with fragments, not whole sentences.

'clip score' is a feedback path. When the score arriving there is ABOVE 'clip
score threshold' the node stops rewriting and emits nothing at all - the reading
is that whatever is consuming the phrase is already doing well enough, so leave
it alone. A rephrase that has gone quiet is usually this, not a fault.

'clear input pause' is how long a phrase stays live. Fragments arriving within
that many seconds are folded into the phrase in progress; after the pause, the
next fragment starts fresh.

'output as list' sends the result word by word instead of as one string.

SYNTAX:
lemma
rephrase

EXAMPLE:
rephrase

INPUTS and PARAMETERS:

text in:
A fragment. Receiving it folds it in and sends the result.

clip score:
The feedback number. High enough, and the node goes quiet.

replace similarity:
How alike two pieces must be before one replaces the other.

OUTPUTS: 

lemmas out:
A list, one dictionary form per word.

results:
The accumulated phrase.
""" + MODEL + """
RELATED:
spacy_similarity works on meaning rather than grammar.
context_tracker keeps track of place, time and actors across a stream of text."""

demo = [
    {'key': 'm1', 'init': 'message', 'pos': (30, 62), 'w': 380, 'h': 42,
     'props': MSG('the geese were running quickly')},
    {'key': 'lm', 'init': 'lemma', 'pos': (30, 120), 'w': 180, 'h': 80},
    {'key': 'l1', 'init': 'list', 'pos': (30, 215), 'w': 380, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'geese -> goose, were -> be, running -> run\nit knows the irregular forms - it has\nparsed the sentence, not chopped endings',
     'pos': (30, 265)},

    {'key': 'm2', 'init': 'message', 'pos': (30, 400), 'w': 180, 'h': 42,
     'props': MSG('a woman')},
    {'key': 'm3', 'init': 'message', 'pos': (30, 455), 'w': 220, 'h': 42,
     'props': MSG('in a red coat')},
    {'key': 'm4', 'init': 'message', 'pos': (30, 510), 'w': 220, 'h': 42,
     'props': MSG('she is walking')},
    {'key': 'm5', 'init': 'message', 'pos': (30, 565), 'w': 240, 'h': 42,
     'props': MSG('through the park')},
    {'key': 'm6', 'init': 'message', 'pos': (30, 620), 'w': 140, 'h': 42,
     'props': MSG('no')},
    {'key': 'c3', 'comment': True, 'text': "click these in order, one at a time\n'no' steps back to the previous phrase",
     'pos': (30, 670)},

    {'key': 'rp', 'init': 'rephrase', 'pos': (450, 400), 'w': 300, 'h': 260},
    {'key': 'l2', 'init': 'list', 'pos': (450, 675), 'w': 420, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c5', 'comment': True, 'text': 'each fragment is folded into the phrase\nbuilt so far, not appended to it\ncomplexity above the threshold is passed\nthrough untouched - it works on fragments',
     'pos': (450, 725)},
]
links = [('m1', 'message out', 'lm', 'text in'), ('lm', 'lemmas out', 'l1', ''),
         ('m2', 'message out', 'rp', 'text in'),
         ('m3', 'message out', 'rp', 'text in'),
         ('m4', 'message out', 'rp', 'text in'),
         ('m5', 'message out', 'rp', 'text in'),
         ('m6', 'message out', 'rp', 'text in'),
         ('rp', 'results', 'l2', '')]
print(build('rephrase', 'lemma and rephrase - working from grammar', body,
            demo, links, demo_width=900, text_width=800, text_height=760))
