"""Feature steering: gemma + neuronpedia."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """A language model you can reach inside and push.

THE NODES:

gemma                Gemma 2, with feature steering
neuronpedia_search   find the feature to steer, by describing it

THIS IS NOT THE SAME NODE AS gemma_4:
gemma_4 is a chat model you prompt. This is Gemma 2 with a sparse autoencoder
attached, which lets you reach into the middle of the network and turn
individual CONCEPTS up or down while it writes.

Prompting asks the model to write about something. Steering changes what it is
thinking about, which is a different kind of control and produces a different
kind of text - the subject can stay put while the disposition underneath it
shifts.

HOW STEERING WORKS, IN PRACTICE:
The sparse autoencoder decomposes what is happening inside a layer into many
thousands of features, most of which turn out to correspond to something a
person can name - "strong emotions", "dogs", a place, a register. Each has a
NUMBER.

'interventions' takes those numbers. 'intervention_strength' says how hard to
push, and 'intervention_active' switches the whole thing on and off so you can
hear the difference against the same prompt and seed.

Push gently. A small strength colours the writing; a large one makes the model
obsessive and eventually incoherent, in a way that is interesting once and
tiresome after that.

neuronpedia_search IS HOW YOU FIND THE NUMBER:
You do not want to read sixteen thousand features. Type a description - what you
want more or less of - and it asks Neuronpedia, which holds explanations of
these features contributed by other people, and returns the best three as

    [index, description, votes]

The index is what gemma wants. The votes are how much agreement there is that
the explanation is right, which is worth reading: a feature with a confident
explanation behaves more like its label than one with a tentative one.

'model' and 'layer' must MATCH what gemma is running. A feature index only means
anything for the model and layer it was found in - the same number in a
different layer is a different concept entirely, and nothing will warn you.

THE MODEL AND ITS SAE ARE A SET:
Each entry pairs a model with the Gemma Scope autoencoder trained on it, and a
layer. The larger models are quantised to four bits to fit. Changing the model
means changing the autoencoder and re-finding your features.

'start', 'pause' and 'reset' control the generation; 'delay' paces it.
'temperature', 'top_k' and 'top_p' are the ordinary sampling controls.

SYNTAX:
gemma
neuronpedia_search

EXAMPLE:
neuronpedia_search

INPUTS and PARAMETERS:

prompt:
What to write about.

interventions / intervention_strength / intervention_active:
Which features to push, how hard, and whether to push at all.

temperature / top_k / top_p / delay:
Sampling, and pacing.

start / pause / reset:
Run, hold, begin again.

search text / model / layer / search:
What to look for, where, and go.

OUTPUTS: 

generated_text:
What it wrote.

results:
Up to three [index, description, votes] triples.

RELATED:
gemma_4 for ordinary chat, which is faster and better at answering.
The steering here is the interesting one; gemma_4 is the useful one."""

demo = [
    {'key': 's1', 'init': 'string', 'pos': (30, 62), 'w': 380, 'h': 42,
     'props': {'text in': 'strong emotions', 'font size': '24', 'width': 340}},
    {'key': 'np', 'init': 'neuronpedia_search', 'pos': (30, 120), 'w': 320, 'h': 200},
    {'key': 'l1', 'init': 'list', 'pos': (30, 335), 'w': 480, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'describe what you want more of, and it\nreturns [index, description, votes].\nThe index is what gemma wants\nmodel and layer MUST match gemma - the\nsame number in another layer is a\ndifferent concept, and nothing warns you',
     'pos': (30, 385)},

    {'key': 'gm', 'init': 'gemma', 'pos': (560, 62), 'w': 340, 'h': 520},
    {'key': 'td', 'init': 'text_display', 'pos': (560, 610), 'w': 340, 'h': 220,
     'props': {'width': 320, 'height': 180, 'wrap': True, 'max_lines': 200,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c6', 'comment': True, 'text': 'switch intervention_active on and off\nagainst the same prompt to hear what\nthe feature is actually doing\npush GENTLY - a small strength colours\nthe writing, a large one makes it\nobsessive and then incoherent',
     'pos': (560, 845)},

    {'key': 'c12', 'comment': True, 'text': 'prompting asks it to write ABOUT\nsomething. Steering changes what it is\nthinking about - the subject can stay\nput while the disposition shifts',
     'pos': (30, 600)},
]
links = [('s1', 'string out', 'np', 'search text'),
         ('np', 'results', 'l1', ''),
         ('gm', 'generated_text', 'td', '###text in')]
print(build('gemma', 'gemma - steering a model from inside', body,
            demo, links, demo_width=940, text_width=800, text_height=740))
