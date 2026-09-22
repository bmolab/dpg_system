"""Bonsai 2 27B, running locally in a server process."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """A 27 billion parameter model on this machine, in 7.2 GB of weights.

WHAT IT IS:
Bonsai 2 is PrismML's compression of Qwen3.8 27B down to ternary weights - a
little over two bits each. A model this size normally needs 54 GB and a big
graphics card; this one fits in memory alongside everything else you are running
and generates at about 20 tokens a second on an M1 Max.

IT RUNS IN A SEPARATE PROCESS, AND NEEDS A PARTICULAR BINARY:
Unlike gemma_4, which loads the model inside dpg_system, this node starts
llama-server as a child process and talks to it over the local network. That is
not a design preference - Bonsai 2's weight format and its activation transform
are not in stock llama.cpp, so the version dpg_system links against cannot read
these files at all.

You need a binary from PrismML's fork, unpacked into

    ~/.cache/bonsai/llama.cpp-prism-<build>/

or pointed at by the BONSAI_LLAMA_SERVER environment variable. Releases are at
github.com/PrismML-Eng/llama.cpp. The node says so plainly if it cannot find
one.

A stock llama.cpp build is worse than useless here: it reads an older ternary
format without complaining and produces fluent nonsense.

The model itself is downloaded on first use and cached - 7.2 GB, so the first
run takes a while. After that the server starts in a second or two. Nothing
happens until you switch the node on.

Several nodes using the same model SHARE one server process. Deleting a node
does not stop it; quitting dpg_system does.

THINKING IS ON BY DEFAULT IN THIS MODEL, AND WORTH UNDERSTANDING:
Bonsai 2 reasons before answering. The reasoning comes out of 'thinking' rather
than 'output' - keep them apart, since the thinking is often more interesting
than the answer and almost never what an audience should read.

The 'thinking' option does not ask the model nicely. With it off, the node hands
the model a thought block that is already closed, so there is nothing for it to
reason into and it answers directly. That is faster and blunter.

'reasoning_effort' sets how hard it thinks when thinking is on: xhigh is the
model's own default, medium is shorter and quicker. low is offered because the
template accepts it, but this model largely ignores it.

IT IS NOT A CHATBOT NODE, IT IS A GENERATION YOU CAN REACH INTO:
Everything gemma_4 does for steering a generation while it happens is here and
works the same way.

THREE WAYS TO STOP, AND THEY ARE DIFFERENT:

polite_stop     finish the current sentence and stop. The text ends properly.
interrupt       stop generating now, mid-word if need be.
hard interrupt  abandon it entirely.

Use polite_stop for anything an audience is reading.

'target_length' NUDGES, IT DOES NOT CUT:
It makes the end of the turn progressively more likely as the text approaches
the length you asked for, and only at a sentence end, so the model finds its own
way to finish at about the right size. 'stop_bias' is how hard that push is.
'max_tokens' is the hard ceiling, and that one does cut.

It counts the ANSWER, not the reasoning, and it never pushes while the model is
still thinking. That matters here: at xhigh effort the thought alone runs past
any modest target, and a nudge applied inside it ends the turn before the answer
starts - the model reasons and then says nothing.

'separate_actions' - UNSPOKEN ASIDES, AND WHY IT IS OFF BY DEFAULT:
Asked to, this model marks anything unspoken with a single asterisk at the start
of the phrase - *Looks around the room.* - and 'separate_actions' sends those to
'actions_out' instead of 'output', so a speech synthesiser on 'output' never says
them.

Three things about how it reads the text, each of which was a bug first:

The closing asterisk usually arrives FUSED to punctuation, as '.*' or '*.'
rather than on its own, so any token carrying a single asterisk closes an aside.

An aside also ends at the END OF ITS LINE. Without that, an aside the model never
closed swallowed the speech that followed it and handed it back one phrase later
- the asides stayed silent until the next spoken phrase, then everything came out
at once, a whole phrase out of step.

DOUBLED asterisks never mark an aside. This model uses **bold** for emphasis on
words it IS speaking, and diverting those took words out of the speech. What it
cannot tell apart is single-asterisk *italic*, which looks exactly like a
one-word aside and so goes unspoken.

It is off by default because with it on a markdown bulleted list reads as asides,
one line at a time - fine when you are working with a performer, wrong when you
just want the text.

STALE REASONING IS DROPPED BETWEEN TURNS:
Each new turn removes the previous turns' thinking from the context, which is
what the model's own template does by default. Left in, several turns of visible
deliberation give the model its own earlier reasoning to imitate and it begins
restating the previous answer rather than writing a new one - and the context
fills up several times faster: five turns came to 248 tokens with it dropped
against 876 with it kept. 'keep thinking in context' turns that off if you want
the whole trace kept.

Taking something out of the middle of the context costs the server a re-read of
everything after the gap, because this model's attention cannot shift a cache.
That happens in the pause AFTER an answer rather than before the next one, so it
does not show up as a wait before the model starts speaking. With thinking off
there is nothing to drop and nothing to pay.

'read prompt as it arrives' - FOR SPEECH DRIVING THE PROMPT:
With this on, words arriving at 'streaming_prompt' are read into the model as
they come rather than all at once when you submit, so the model has already
processed the sentence by the time you finish it. A spoken prompt of about forty
tokens starts answering in 0.31s instead of 0.65s - a third of a second saved,
and it grows with the length of what was said.

Nothing happens until the node is switched on; typing a character is not what
pulls seven gigabytes of weights into memory. The reading happens off the main
thread, so the editor keeps drawing.

The real reason to use it is 'score_incoming_text': with both on, every word is
scored AS IT ARRIVES, so 'input_token_score' reports how expected each word was
while the person is still speaking, and 'input_cumulative_score' tracks the
running average. That signal does not exist if you hand over the whole sentence
at once.

Three things it handles that are easy to get wrong. The whole utterance is
re-tokenised on each read rather than the new words alone, because a byte-pair
merge can reach across a word boundary - the result is byte-identical to the
sentence arriving whole. A recogniser that revises what it already said is fine:
the tokens back to the first difference are dropped and re-read. And an utterance
abandoned before submitting is taken back out of the context entirely, system
block included.

Bear in mind that a spoken prompt commits the turn early, so the model is holding
your half-finished sentence. That is the point, but it means the one server slot
is busy reading while you speak - reading and generating do not overlap.

IT IS THE SMALLER SAVING, THOUGH:
If the wait before the model speaks matters, thinking costs far more than
prefill. With thinking on the model spends 100 to 130 tokens reasoning before the
first spoken word, which is five to seven seconds. Turning thinking off brings
that to about half a second. Do that first; this is worth a third of a second on
top.

'stream_chunk' - HOW MANY TOKENS IT ASKS FOR AT A TIME:
While it is free-running the node takes a short run of tokens per request rather
than one at a time, which is about 11% faster - asking per token spends around
5 ms of round trip on every one. The default of 8 recovers effectively all of
that while still re-reading the sampling controls often enough that moving the
temperature mid-generation is felt within about half a second. Set it to 1 to
have every token re-read everything, at that 11%.

It applies only while free-running. Stepping takes one token at a time by
definition, 'slowdown' is already spending far more than the round trip, and the
last stretch before target_length goes back to one at a time so the ending nudge
keeps working exactly as described above.

'step' AND 'choice' - WRITING WITH IT RATHER THAN RECEIVING FROM IT:
'step' generates ONE token, forward or back. 'choice' then walks the
alternatives the model was weighing at that point and substitutes the one you
pick. Stepping back here costs nothing: the server keeps everything before the
point you stepped back to, so only the new tail is recomputed.

'show_probs' has to be on for the alternatives to exist.

THE SAMPLING CONTROLS, IN THE ORDER WORTH TOUCHING:

temperature       flatness of the choice. Low is predictable and repetitive,
                  high is surprising and eventually incoherent.
min_p             discards anything much less likely than the best candidate.
top_k / top_p     older ways of narrowing the field.
repeat_penalty    pushes against saying the same thing again.
seed              the same seed and the same settings give the same text.

The model's own recommended settings are the defaults here: temperature 1.0,
top_p 0.95, top_k 20 for thinking, and temperature 0.7 with top_p 0.8 without.

SCORING TEXT YOU PUT IN, RATHER THAN TEXT IT MAKES:
With 'score_incoming_text' on, text you send is run through the model and each
token reported with how likely the model thought it was - 'input_token_score'
per token, 'input_cumulative_score' as a running average. Ordinary prose scores
high and something strange scores low, so it finds the unusual parts of a text
without generating anything.

'display_mode' picks what the layout display shows per token: temperature,
entropy, probability, or unnormalised probability. Because the model is across a
process boundary, those last three are worked out from the forty candidates the
server reports rather than the whole vocabulary - close to exact when the model
is confident, a slight underestimate of entropy when it is not.

SYNTAX:
bonsai_2
bonsai_2 27B_1bit      the smaller 5.95 GB pack
bonsai_2 <path.gguf>   a particular file

EXAMPLE:
bonsai_2

INPUTS and PARAMETERS:

on / off:
Start the server and be ready. This is what takes the time.

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

read prompt as it arrives:
Read words into the model as they arrive rather than on submit. On by default.

stream_chunk:
Tokens taken per request while free-running. 8 by default, 1 for per-token.

separate_actions:
Send unspoken asides - a single asterisk at the start of a phrase - to
'actions_out' rather than 'output'. Off by default.

keep thinking in context:
Keep earlier turns' reasoning in the context instead of dropping it.

score_incoming_text / reset_input_score:
Measure how predictable incoming text is.

n_ctx / n_gpu_layers:
How much context, and how much runs on the GPU. The model will take 262144 of
context; 8192 is a working default.

thinking / reasoning_effort:
Whether it reasons first, and how hard.

server_log:
Show what the server process is saying, once it is past loading.

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
gemma_4 for a smaller model loaded in this process rather than beside it.
text_display to read the output in the patch.
context_tracker and prompt_composer to turn what it says into something else."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'c0', 'comment': True, 'text': 'switch on and WAIT - the server starts\nhere, and the 7.2 GB of weights are\ndownloaded the first time',
     'pos': (90, 62)},

    {'key': 'sys', 'init': 'string', 'pos': (30, 140), 'w': 560, 'h': 42,
     'props': {'text in': 'You are a terse, concrete writer.', 'font size': '24',
               'width': 520}},
    {'key': 'pr', 'init': 'string', 'pos': (30, 195), 'w': 560, 'h': 42,
     'props': {'text in': 'Describe a room nobody has entered for a year.',
               'font size': '24', 'width': 520}},

    {'key': 'bn', 'init': 'bonsai_2', 'pos': (30, 260), 'w': 380, 'h': 620,
     'props': {'n_ctx': 8192}},
    {'key': 'c2', 'comment': True, 'text': 'target_length nudges it towards a size by\nmaking the ending more likely at a sentence\nend - it does not cut. max_tokens does cut',
     'pos': (30, 895)},

    {'key': 'td', 'init': 'text_display', 'pos': (460, 260), 'w': 360, 'h': 260,
     'props': {'width': 340, 'height': 220, 'wrap': True, 'max_lines': 200,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c6', 'comment': True, 'text': 'the answer, arriving token by token',
     'pos': (460, 535)},

    {'key': 'th', 'init': 'text_display', 'pos': (460, 580), 'w': 360, 'h': 220,
     'props': {'width': 340, 'height': 180, 'wrap': True, 'max_lines': 200,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c7', 'comment': True, 'text': 'thinking is a SEPARATE channel, and this\nmodel thinks by default - turn the thinking\noption off to get the answer straight',
     'pos': (460, 815)},

    {'key': 'act', 'init': 'toggle', 'pos': (460, 920), 'w': 45, 'h': 42},
    {'key': 'c10', 'comment': True, 'text': 'lit while it is generating',
     'pos': (520, 920)},

    {'key': 'pl', 'init': 'plot', 'pos': (460, 980), 'w': 300, 'h': 180,
     'props': PLOT(-8.0, 0.0, 200)},
    {'key': 'c11', 'comment': True, 'text': 'with score_incoming_text on, this is how\nPREDICTABLE your text was to it - low\nmeans unusual, and no generating needed',
     'pos': (460, 1170)},
]
links = [('tog', '', 'bn', 'on / off'),
         ('sys', 'string out', 'bn', 'system_prompt'),
         ('pr', 'string out', 'bn', 'prompt'),
         ('bn', 'output', 'td', '###text in'),
         ('bn', 'thinking', 'th', '###text in'),
         ('bn', 'active', 'act', ''),
         ('bn', 'input_token_score', 'pl', 'y')]
print(build('bonsai_2', 'bonsai_2 - a 27B model in seven gigabytes', body,
            demo, links, demo_width=880, text_width=810, text_height=800))
