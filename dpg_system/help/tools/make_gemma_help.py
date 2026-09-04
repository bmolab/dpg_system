"""Gemma 4 chat, running locally."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """A language model running on this machine, with the generation opened up.

THE NODES:

gemma_4       Gemma 4 12B, quantised. 8192 tokens of context
gemma_4_31b   the 31B model. 2048 of context, because it barely fits

Same node, different weights. The 31B is better and much slower, and its context
is short for a reason - at 31 billion parameters and 4096 of context it does not
fit in 32GB of unified memory alongside anything else.

The model is downloaded on first use and cached; the first run of either will
sit there for a long time before it says anything. Nothing is loaded until you
switch the node on, so having one in a patch costs nothing until you do.

Several nodes using the same model SHARE one copy of it - the weights are held
once, not per node. So a patch can have three gemma_4 nodes with different
system prompts for the price of one model in memory. Mixing gemma_4 and
gemma_4_31b in one patch does load both, and that is what will not fit.

IT IS NOT A CHATBOT NODE, IT IS A GENERATION YOU CAN REACH INTO:
Most of what is unusual here is about interrupting, steering and measuring the
generation while it happens, rather than about asking a question and waiting.

THREE WAYS TO STOP, AND THEY ARE DIFFERENT:

polite_stop     finish the current sentence and stop. The text ends properly.
interrupt       stop generating now, mid-word if need be.
hard interrupt  abandon it entirely.

Use polite_stop for anything an audience is reading. The other two exist for
when it has gone somewhere you do not want and waiting is worse than a broken
sentence.

'target_length' NUDGES, IT DOES NOT CUT:
It does not truncate. It makes the end-of-turn token progressively more likely
as the text approaches the length you asked for, so the model finds its own way
to finish at about the right size. Set it to 50 and you get something that reads
as a finished short answer, not a long answer with the end cut off.

'max_tokens' is the hard ceiling, and that one does cut.

'step' AND 'choice' - WRITING WITH IT RATHER THAN RECEIVING FROM IT:
'step' generates ONE token, forward or back. 'choice' then walks through the
alternatives the model was considering at that point, in order of how likely it
thought they were, and substitutes the one you pick - it sends a backspace and
the replacement.

So you can take a sentence one token at a time and choose a different branch
wherever you like, from what the model itself was weighing. It is a different
relationship to the text: not asking for output, but choosing a path through
what it was already prepared to say.

'show_probs' has to be on for the alternatives to exist.

'slowdown' PACES IT FOR READING:
Tokens otherwise arrive as fast as the machine manages, which is too fast to
read and looks nothing like writing. This holds them back. It is a performance
control, not a technical one.

THE SAMPLING CONTROLS, IN THE ORDER WORTH TOUCHING:

temperature       flatness of the choice. Low is predictable and repetitive,
                  high is surprising and eventually incoherent.
min_p             discards anything much less likely than the best candidate.
                  The one to reach for after temperature.
top_k / top_p     older ways of narrowing the field - a fixed number of
                  candidates, or enough to cover a share of the probability.
repeat_penalty    pushes against saying the same thing again.
seed              the same seed and the same settings give the same text.

SCORING TEXT YOU PUT IN, RATHER THAN TEXT IT MAKES:
With 'score_incoming_text' on, text you send is run through the model and each
token reported with how likely the model thought it was -
'input_token_score' per token, 'input_cumulative_score' as a running average.

This is a measure of how PREDICTABLE something is to the model. Ordinary prose
scores high, and something strange or specific scores low, so it can be used to
find the unusual parts of a text without generating anything at all.

'display_mode' picks what the layout display shows per token: temperature,
entropy, probability, or unnormalised probability. 'sigmoid scaler' and
'sigmoid offset' shape that reading into a usable range.

THINKING IS A SEPARATE CHANNEL:
Gemma 4 can reason before answering, and that reasoning comes out of 'thinking'
rather than 'output'. Keep them apart - the thinking is often more interesting
than the answer, and it is almost never what you want an audience to read.

SYNTAX:
gemma_4
gemma_4_31b

EXAMPLE:
gemma_4

INPUTS and PARAMETERS:

on / off:
Load the model and be ready. This is what takes the time.

prompt / pre-prompt / streaming_prompt:
What to answer, what to put before it, and text arriving a piece at a time.

system_prompt:
Who it is being. Set before starting.

polite_stop / interrupt / hard interrupt:
The three stops, in increasing rudeness.

step (+-1) / choice (+-1):
One token at a time, and which of the alternatives.

temperature / top_k / top_p / min_p / repeat_penalty / seed:
The sampling.

target_length / max_tokens:
A length to aim at, and a length not to exceed.

slowdown:
Pace the output for reading.

score_incoming_text / reset_input_score:
Measure how predictable incoming text is.

n_ctx / n_gpu_layers:
How much context, and how much runs on the GPU.

OUTPUTS: 

output:
The text, token by token as it comes.

thinking:
The reasoning, if there was any.

token_out / end / active:
Each token, a signal at the end, and whether it is generating.

input_token_score / input_cumulative_score:
How predictable the text you sent was.

layout_out / actions_out:
Display information, and anything marked as an action.

RELATED:
text_display to read the output in the patch.
context_tracker and prompt_composer to turn what it says into something else.
vision_describe if the thing you want described is a picture."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'c0', 'comment': True, 'text': 'switch on and WAIT - the model is loaded\nhere, and downloaded the first time',
     'pos': (90, 62)},

    {'key': 'sys', 'init': 'string', 'pos': (30, 130), 'w': 560, 'h': 42,
     'props': {'text in': 'You are a terse, concrete writer.', 'font size': '24',
               'width': 520}},
    {'key': 'pr', 'init': 'string', 'pos': (30, 185), 'w': 560, 'h': 42,
     'props': {'text in': 'Describe a room nobody has entered for a year.',
               'font size': '24', 'width': 520}},

    {'key': 'gm', 'init': 'gemma_4', 'pos': (30, 250), 'w': 380, 'h': 620,
     'props': {'n_ctx': 8192}},
    {'key': 'c2', 'comment': True, 'text': 'target_length nudges it towards a size\nby making the ending more likely - it\ndoes not cut. max_tokens does cut\nslowdown paces it for reading',
     'pos': (30, 885)},

    {'key': 'td', 'init': 'text_display', 'pos': (460, 250), 'w': 360, 'h': 260,
     'props': {'width': 340, 'height': 220, 'wrap': True, 'max_lines': 200,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c6', 'comment': True, 'text': 'the answer, arriving token by token',
     'pos': (460, 525)},

    {'key': 'th', 'init': 'text_display', 'pos': (460, 570), 'w': 360, 'h': 220,
     'props': {'width': 340, 'height': 180, 'wrap': True, 'max_lines': 200,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c7', 'comment': True, 'text': 'thinking is a SEPARATE channel - often\nthe interesting part, and almost never\nwhat an audience should read',
     'pos': (460, 805)},

    {'key': 'act', 'init': 'toggle', 'pos': (460, 910), 'w': 45, 'h': 42},
    {'key': 'c10', 'comment': True, 'text': 'lit while it is generating',
     'pos': (520, 910)},

    {'key': 'pl', 'init': 'plot', 'pos': (460, 970), 'w': 300, 'h': 180,
     'props': PLOT(-8.0, 0.0, 200)},
    {'key': 'c11', 'comment': True, 'text': 'with score_incoming_text on, this is how\nPREDICTABLE your text was to it - low\nmeans unusual, and no generating needed',
     'pos': (460, 1160)},
]
links = [('tog', '', 'gm', 'on / off'),
         ('sys', 'string out', 'gm', 'system_prompt'),
         ('pr', 'string out', 'gm', 'prompt'),
         ('gm', 'output', 'td', '###text in'),
         ('gm', 'thinking', 'th', '###text in'),
         ('gm', 'active', 'act', ''),
         ('gm', 'input_token_score', 'pl', 'y')]
print(build('gemma_4', 'gemma_4 - a language model you can reach into', body,
            demo, links, demo_width=880, text_width=810, text_height=800))
