"""Qwen3.6 35B-A3B, a mixture of experts running in this process on MLX."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """A 35 billion parameter model that generates faster than a 12 billion one.

WHY IT IS FAST:
It is a mixture of experts: 35 billion parameters in total, but only about 3
billion are read for any one token. Generating is limited by how many bytes come
out of memory per token, so what matters is the 3 billion and not the 35. On this
machine, measured the same way:

qwen_moe (this node)   56 tok/s
gemma_4 12B            37 tok/s
bonsai_2 27B           21 tok/s
gemma_4 31B            16 tok/s

It is also the most capable of the four. It is not a trade-off - it wins on both
counts. What it costs is memory: 20 GB resident against bonsai_2's 7, which is
most of this machine, so the two do not sit in memory together comfortably. Use
one or the other rather than both.

IT RUNS IN THIS PROCESS, UNLIKE bonsai_2:
bonsai_2 has to drive a separate server, because its weight format needs a
llama.cpp fork. This model is ordinary enough to run through MLX inside
dpg_system, and that changes three things for the better.

THE MEASUREMENTS ARE EXACT, NOT SAMPLED:
'display_mode' entropy, probability and unnormalised probability are computed
over the WHOLE vocabulary - all 248320 tokens - because the logits are right
here. bonsai_2 can only see the forty candidates its server reports, so its
entropy is a floor rather than the figure. The candidate list carries RAW LOGITS,
as gemma_4's does, so 'sigmoid scaler' and 'sigmoid offset' behave the same way
in all three nodes.

'score_incoming_text' is exact for the same reason. bonsai_2 has to force each
token with a bias and read the value back; here the log-probability is simply
read off the distribution.

STEPPING BACK IS CHEAP, BUT NOT FREE:
30 of this model's 40 layers carry a recurrent state rather than a key-value
cache, and a recurrent state cannot be rewound a token at a time. So the node
keeps a snapshot of the state at the start of each turn and replays from there,
which puts a step back inside the current answer at about a twentieth of a
second. Stepping back further than the start of the turn rebuilds from the
beginning, at around 600 tokens a second.

'step' AND 'choice' WORK AS THEY DO ELSEWHERE:
'step' takes one token, forward or back. 'choice' walks the alternatives the
model was weighing and substitutes one. While free-running the node keeps the
model a step ahead of itself, which is worth about a fifth of the speed; the
moment you step, choose or slow it down it drops back to exact token-at-a-time
work. You do not have to do anything for that - it notices.

WHAT IT DOES NOT HAVE:
No 'reasoning_effort'. Bonsai 2's template takes one; this model's does not, and
injecting a sentence it was not trained on would be worse than leaving it out.

Everything else reads the same as bonsai_2: the three stops, target_length
nudging towards a length by making the ending likelier at a sentence end,
thinking on its own outlet, asides marked with a single asterisk going to
'actions_out', stale reasoning dropped between turns, and words read into the
model as they arrive.

SYNTAX:
qwen_moe
qwen_moe <model>    a hugging face id or a local path to MLX weights

EXAMPLE:
qwen_moe

INPUTS and PARAMETERS:

on / off:
Load the model and be ready. Twenty gigabytes, so this is what takes the time.

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

target_length / max_tokens / stop_bias:
A length to aim at, a length not to exceed, and how hard the aim pushes.

slowdown:
Pace the output for reading.

score_incoming_text / reset_input_score:
Measure how predictable incoming text is, exactly.

show_probs / display_mode / sigmoid scaler / sigmoid offset:
The candidate list, and what the layout colours each token by.

thinking / thinking in layout:
Whether it reasons first, and whether that shows in the layout.

keep thinking in context / read prompt as it arrives / candidates shown:
Keep earlier reasoning, read words as they arrive, and how many alternatives to
collect.

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
Display information, and anything marked as an unspoken aside.

RELATED:
bonsai_2 for a 27B model in seven gigabytes, when memory is short.
gemma_4 for a smaller model in this process.
text_display to read the output in the patch."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'c0', 'comment': True, 'text': 'switch on and WAIT - twenty gigabytes\nof weights load here',
     'pos': (90, 62)},

    {'key': 'sys', 'init': 'string', 'pos': (30, 130), 'w': 560, 'h': 42,
     'props': {'text in': 'You are a terse, concrete writer.', 'font size': '24',
               'width': 520}},
    {'key': 'pr', 'init': 'string', 'pos': (30, 185), 'w': 560, 'h': 42,
     'props': {'text in': 'Describe a room nobody has entered for a year.',
               'font size': '24', 'width': 520}},

    {'key': 'qw', 'init': 'qwen_moe', 'pos': (30, 250), 'w': 380, 'h': 620},
    {'key': 'c2', 'comment': True, 'text': 'three billion of thirty-five billion parameters\nare read per token, which is why it outruns\nevery smaller model here',
     'pos': (30, 885)},

    {'key': 'td', 'init': 'text_display', 'pos': (460, 250), 'w': 360, 'h': 260,
     'props': {'width': 340, 'height': 220, 'wrap': True, 'max_lines': 200,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c6', 'comment': True, 'text': 'the answer, arriving token by token',
     'pos': (460, 525)},

    {'key': 'th', 'init': 'text_display', 'pos': (460, 570), 'w': 360, 'h': 220,
     'props': {'width': 340, 'height': 180, 'wrap': True, 'max_lines': 200,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c7', 'comment': True, 'text': 'thinking on its own outlet, as in bonsai_2',
     'pos': (460, 805)},

    {'key': 'act', 'init': 'toggle', 'pos': (460, 860), 'w': 45, 'h': 42},
    {'key': 'c10', 'comment': True, 'text': 'lit while it is generating',
     'pos': (520, 860)},

    {'key': 'pl', 'init': 'plot', 'pos': (460, 920), 'w': 300, 'h': 180,
     'props': PLOT(-8.0, 0.0, 200)},
    {'key': 'c11', 'comment': True, 'text': 'with score_incoming_text on, how PREDICTABLE\nyour text was - exact here, read straight off\nthe whole distribution',
     'pos': (460, 1110)},
]
links = [('tog', '', 'qw', 'on / off'),
         ('sys', 'string out', 'qw', 'system_prompt'),
         ('pr', 'string out', 'qw', 'prompt'),
         ('qw', 'output', 'td', '###text in'),
         ('qw', 'thinking', 'th', '###text in'),
         ('qw', 'active', 'act', ''),
         ('qw', 'input_token_score', 'pl', 'y')]
print(build('qwen_moe', 'qwen_moe - a mixture of experts, in this process', body,
            demo, links, demo_width=880, text_width=810, text_height=800))
