"""Feature steering: gemma, and neuronpedia_search to find what to steer."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ----------------------------------------------------------------------- gemma
body = """A language model you can reach inside and push.

THE NODE:

gemma   Gemma 2, with feature steering through a sparse autoencoder

STATUS - READ THIS FIRST:
In this build the node is a working skeleton, not a working model. The lines
that load Gemma 2 and its autoencoder, run the model and apply the steering are
all commented out in gemma_node.py, so pressing 'start' writes the placeholder
word TEST! once per step instead of real text. The controls, the pacing and the
steering bookkeeping all run, which is what this page describes. Its module is
also not in dpg_system's import list at present, so the node only exists when
that is added.

THIS IS NOT THE SAME NODE AS gemma_4:
gemma_4 is a chat model you prompt. This is Gemma 2 with a sparse autoencoder
attached, which lets you reach into the middle of the network and turn
individual CONCEPTS up or down while it writes.

Prompting asks the model to write about something. Steering changes what it is
thinking about, which is a different kind of control and produces a different
kind of text - the subject can stay put while the disposition underneath it
shifts.

HOW STEERING WORKS:
The sparse autoencoder decomposes what is happening inside one layer into
sixteen thousand features, most of which turn out to correspond to something a
person can name - "strong emotions", "dogs", a place, a register. Each has an
index number. Steering adds a feature's direction back into that layer while
the model writes.

'interventions' takes a list of features as text, in exactly the form
neuronpedia_search puts out:

    [[4253, "strong emotions", 1], [8398, "dogs", 2]]

Each entry is [index, description, weight]. Only the index and the third number
are used: a feature is pushed by 'intervention_strength' times its weight, and
an entry whose weight is 0 is skipped. Text that does not read as a list is
treated as an empty list.

'intervention_active' switches the whole thing on and off, and with
'intervention_strength' at 0 nothing is pushed either. Both, and the
interventions themselves, are read again at every step, so you can change them
while it writes and hear the difference against the same prompt.

Push gently. A small strength colours the writing; a large one makes the model
obsessive and eventually incoherent.

THE MODEL IS FIXED:
The model and its autoencoder are a set, chosen in the code rather than on the
node: Gemma 2 9B quantised to four bits, with the Gemma Scope autoencoder for
layer 20, sixteen thousand features wide. Feature indices only mean anything
for that model and layer, which is why neuronpedia_search defaults to the same
pair.

HOW IT WRITES:
'start' begins a run from the prompt. It writes up to fifty steps, then carries
on from what it has written for up to ten more rounds of fifty. 'delay' is the
pause between steps in seconds, read again at each step. 'pause' holds it where
it is and 'start' lets it go on; 'reset' stops it and clears the text. While a
run is going, 'start' does not begin a second one.

SYNTAX:
gemma

EXAMPLE:
gemma

INPUTS and PARAMETERS:

prompt:
The text it begins from.

delay:
Seconds between steps. 1.0 by default.

temperature / top_k / top_p:
The ordinary sampling controls: 0.7, 50 and 0.9 by default. In this skeleton
top_k and top_p are read, and temperature is not used at all.

intervention_active:
Steering on or off. Off by default.

intervention_strength:
How hard to push, a whole number. 0 by default.

interventions:
The features, as a text list of [index, description, weight]. '[]' by default.

start / pause / reset:
Run, hold, stop and clear.

OUTPUTS:

generated_text:
Everything written so far in this run, sent again after every step - not just
the newest word.

RELATED:
neuronpedia_search finds the feature numbers by description, and its output
goes straight into 'interventions'.
gemma_4 for ordinary chat, which is faster and better at answering."""

demo = [
    {'key': 'iv', 'init': 'string', 'pos': (30, 62), 'w': 380, 'h': 42,
     'props': {'text in': '[[4253, "strong emotions", 1]]', 'font size': '24',
               'width': 340}},
    {'key': 'c0', 'comment': True,
     'text': '[index, description, weight] - the same\nform neuronpedia_search sends out',
     'pos': (440, 62)},
    {'key': 'gm', 'init': 'gemma', 'pos': (30, 140), 'w': 340, 'h': 520},
    {'key': 'c1', 'comment': True,
     'text': 'switch intervention_active on and off\nagainst the same prompt. Push GENTLY -\na large strength makes it obsessive,\nthen incoherent',
     'pos': (440, 140)},
    {'key': 'c2', 'comment': True,
     'text': 'in this build it writes TEST! at each\nstep - the model lines are commented out',
     'pos': (440, 400)},
    {'key': 'td', 'init': 'text_display', 'pos': (30, 700), 'w': 340, 'h': 220,
     'props': {'width': 320, 'height': 180, 'wrap': True, 'max_lines': 200,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c3', 'comment': True,
     'text': 'the whole text so far, resent each step',
     'pos': (440, 700)},
    {'key': 'c4', 'comment': True,
     'text': 'neuronpedia_search finds the feature numbers',
     'pos': (30, 950)},
]
links = [('iv', 'string out', 'gm', 'interventions'),
         ('gm', 'generated_text', 'td', '###text in')]
print(build('gemma', 'gemma - steering a model from inside', body,
            demo, links, demo_width=760, text_width=800, text_height=900))


# ---------------------------------------------------------- neuronpedia_search
body = """Finds the sparse-autoencoder feature to steer, by describing it.

THE NODE:

neuronpedia_search   ask Neuronpedia for features matching a description

WHAT IT IS FOR:
gemma steers by feature NUMBER, and there are sixteen thousand of them. You do
not want to read them all. Type a description of what you want more or less of
and this node asks Neuronpedia - a public website holding explanations of these
features, contributed by people and by automatic labelling - which features fit.

It needs the internet. Nothing is sent until you press 'search' (or send it a
message); typing in the boxes does not search.

WHAT COMES BACK:
Each search takes the best three matches and adds them to a list the node keeps
for as long as it exists. What goes out is that whole list, as text:

    [[4253, "strong emotions", 1], [12514, "intense feelings", 2]]

Each entry is [index, description, count]. The description is the first
explanation Neuronpedia holds for the feature. The count is NOT a vote on the
website: it is how many of your searches so far have returned that feature.
Search for "anger", then "rage", then "fury", and a feature that turns up all
three times has a count of 3.

That matters downstream, because gemma reads the third number as a weight - a
feature that keeps coming back is pushed harder. Searching around a concept
from several directions is therefore a way of building a steering mix. The list
only grows; delete and recreate the node to start a new one.

MODEL AND LAYER MUST MATCH:
A feature index only means anything for the model and layer it was found in -
the same number in a different layer is a different concept, and nothing warns
you. The defaults, gemma-2-9b and 20-gemmascope-res-16k, are the pair the gemma
node uses.

Problems - no search text, a timeout after fifteen seconds, a network or server
error, no results - are printed to the console, and nothing is sent.

SYNTAX:
neuronpedia_search

EXAMPLE:
neuronpedia_search

INPUTS and PARAMETERS:

search text:
What to look for, in plain words.

model:
The Neuronpedia model id. gemma-2-9b by default.

layer:
The Neuronpedia layer and autoencoder id. 20-gemmascope-res-16k by default.

search:
Ask. Any message arriving here searches too.

OUTPUTS:

results:
Every feature found so far, as a text list of [index, description, count].

RELATED:
gemma, whose 'interventions' inlet takes this output as it is."""

demo = [
    {'key': 's1', 'init': 'string', 'pos': (30, 62), 'w': 380, 'h': 42,
     'props': {'text in': 'strong emotions', 'font size': '24', 'width': 340}},
    {'key': 'c0', 'comment': True, 'text': 'describe what you want more of',
     'pos': (440, 62)},
    {'key': 'np', 'init': 'neuronpedia_search', 'pos': (30, 130), 'w': 320, 'h': 200},
    {'key': 'c1', 'comment': True,
     'text': 'press search - it needs the internet\nmodel and layer must match gemma',
     'pos': (440, 130)},
    {'key': 'td', 'init': 'text_display', 'pos': (30, 370), 'w': 340, 'h': 220,
     'props': {'width': 320, 'height': 180, 'wrap': True, 'max_lines': 200,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c2', 'comment': True,
     'text': '[index, description, count] for every\nfeature found so far - count is how\nmany of YOUR searches returned it',
     'pos': (440, 370)},
    {'key': 'c3', 'comment': True,
     'text': 'this text goes straight into gemma\'s interventions',
     'pos': (30, 620)},
]
links = [('s1', 'string out', 'np', 'search text'),
         ('np', 'results', 'td', '###text in')]
print(build('neuronpedia_search', 'neuronpedia_search - finding the feature to steer',
            body, demo, links, demo_width=760, text_width=800, text_height=900))
