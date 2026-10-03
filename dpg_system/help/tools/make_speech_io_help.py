"""Speech in (whisper~) and speech out (eleven_labs~) - one page each."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """Listens, and gives you what was said, as text.

whisper~ RUNS HERE:
whisper~ transcribes on this machine - nothing leaves it, there is no account,
and once its model has been downloaded it keeps working with the network
unplugged. Anything an audience says in confidence can go through it.

IT GIVES YOU TWO STREAMS, AND THE DIFFERENCE MATTERS:

in_progress   what it thinks is being said RIGHT NOW. It changes as more sound
              arrives - words appear, then get revised, then settle.
phrases       what it has decided. A phrase is emitted once and does not change.

Use in_progress for anything live and visual, where the guessing and revising is
part of what the audience sees. Use phrases for anything that acts on meaning -
a decision, a lookup, a prompt to a language model - because acting on
in_progress means acting on words it is about to withdraw.

The two are not alternatives. A common arrangement uses both: in_progress drives
the display, phrases drive the work.

HOW A GUESS BECOMES A PHRASE:
The sound is cut into segments, and each one has an AGE - how many analysis
rounds it has survived unchanged. 'confirmation_age' is how old one must be
before it is trusted. A trusted segment is emitted as a phrase when it ends in
a full stop, question mark or exclamation, or when two more segments have
already formed after it, so a speaker who never uses a full stop is still
heard.

The age required shrinks as the sentence gets longer, by one round for every
'length_factor' characters. The reasoning is that a long utterance has already
given the model plenty of context, so its early words are unlikely to be
revised - and waiting the full age on every one of them would leave the
transcript lagging badly behind the speaker.

Two other things push out whatever is pending, finished or not: a pause (see
silence below), and the listening buffer filling past
'buffer_overflow_fraction' during a long unbroken stretch of speech.

'min_trailing_confidence' trims words the model is unsure of off the end of
each segment, where guessing is worst. 'minimum_confidence' keeps doubtful
segments out of in_progress. 'minimum_lifespan' is kept for older patches but
has no effect at present - every emission overrides it.

THE 'noises' OUTLET:
When Whisper hears something that is not speech it often says so, in brackets
or asterisks - [Music], (coughing), *door closes*. Those segments are kept out
of 'phrases' and come out of 'noises' instead, with the brackets stripped.

Which makes 'noises' quite interesting in its own right: it is what the room
sounded like to something trying very hard to hear language in it.

Whisper can also hallucinate plain words out of near-silence. Those are not
bracketed, so they are not caught here - a sensible 'silence_threshold' is the
defence against them.

SILENCE:
'energy' reports how much is happening in the sound - the sum of the
sample-to-sample changes in each block it takes in. A block below
'silence_threshold' counts as quiet, and after 'silence_period' quiet blocks in
a row the utterance is over: what is pending is emitted and the buffer starts
afresh. Watch 'energy' while the room is quiet and while someone speaks, and
set the threshold between the two. 'gain' scales the sound before any of this.

THE MODEL IS A TRADE:
Larger models are more accurate and slower, and the delay is felt directly as
the transcript trailing the speaker. Start small and go larger only if the words
are actually wrong; a smaller model that keeps up is usually better in
performance than a better one that lags. The '.en' models know only English
and are better at it.

'language' can be set or left at 'auto' to detect, and 'translate' asks for
English out regardless of what went in. 'backend' chooses the engine that runs
the model - whisper.cpp (the default) or faster-whisper.

SYNTAX (the plain name, without ~, still works for older patches):
whisper~ [model] [language] [translate]

All optional, in any order. model is one of tiny, tiny.en, base, base.en,
small, small.en, medium, medium.en, large-v3-turbo, large-v3 (default
base.en). language is a name or a two-letter code - french or fr - or auto
(default english). 'translate' switches translation on.

EXAMPLE:
whisper~ small french translate

INPUTS and PARAMETERS:

on/off:
Start and stop listening. The model is loaded the first time it is switched
on, which takes a few seconds.

in:
A ~ signal to listen to (adc~, a voice, stream~ for audio held as an array).
Patched, it replaces 'audio device' - decided at the moment it is switched on.

audio device:
Which input to listen to when nothing is patched to 'in'.

model / language / translate / backend:
Which model, what language, whether to render English, and which engine.

OPTIONS:

gain:
Scales the incoming sound.

silence_threshold / silence_period:
What counts as nothing being said, and for how many blocks.

confirmation_age / length_factor:
How long a segment must stay unchanged before it is trusted.

minimum_confidence / min_trailing_confidence / minimum_lifespan:
Filters on the model's own certainty.

update_period:
Milliseconds between analysis rounds. Shorter is more responsive and costs more.

overlap / buffer_overflow_fraction:
How much audio is carried across between rounds, and how full the buffer may
get before it is emptied.

debug:
Prints what it is deciding, in the console.

OUTPUTS:

phrases:
Settled text. Act on this.

in_progress:
The current guess, revised as it goes. Display this.

noises:
What it marked as sounds rather than speech.

energy:
The level, sent every frame while it runs. Use it to set silence_threshold.

rate:
An estimate of how fast the person is speaking - characters against time.

language:
The language it thinks it is hearing.

RELATED:
eleven_labs~ to say text aloud - the other direction.
nemotron~ for a streaming alternative.
translate to move between languages in between.
gemma_4 to answer rather than repeat.
cairo_layout to put the words on a screen."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'c0', 'comment': True, 'text': 'on/off - the model loads the first time',
     'pos': (440, 62)},
    {'key': 'wh', 'init': 'whisper~', 'pos': (30, 120), 'w': 386, 'h': 196},
    {'key': 'c1', 'comment': True, 'text': 'runs on this machine - no account, and\nnothing leaves the room',
     'pos': (440, 120)},

    {'key': 'td', 'init': 'text_display', 'pos': (30, 350), 'w': 334, 'h': 170,
     'props': {'width': 320, 'height': 140, 'wrap': True, 'max_lines': 100,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'in_progress: the current guess, revised\nas it listens. Show this.',
     'pos': (440, 350)},

    {'key': 'td2', 'init': 'text_display', 'pos': (30, 550), 'w': 334, 'h': 170,
     'props': {'width': 320, 'height': 140, 'wrap': True, 'max_lines': 100,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c4', 'comment': True, 'text': 'phrases: settled, emitted once. ACT on\nthis - in_progress will be withdrawn',
     'pos': (440, 550)},

    {'key': 'td3', 'init': 'text_display', 'pos': (30, 750), 'w': 334, 'h': 130,
     'props': {'width': 320, 'height': 100, 'wrap': True, 'max_lines': 60,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c6', 'comment': True, 'text': 'noises: what it marked as sounds,\nnot speech - [Music], (coughing).\nKept out of phrases on purpose',
     'pos': (440, 750)},

    {'key': 'pl', 'init': 'plot', 'pos': (30, 910), 'w': 208, 'h': 178,
     'props': PLOT(0.0, 1.0, 200)},
    {'key': 'c9', 'comment': True, 'text': 'energy - set silence_threshold between\nthe quiet room and someone speaking',
     'pos': (440, 985)},
]
links = [('tog', '', 'wh', 'on/off'),
         ('wh', 'in_progress', 'td', '###text in'),
         ('wh', 'phrases', 'td2', '###text in'),
         ('wh', 'noises', 'td3', '###text in'),
         ('wh', 'energy', 'pl', 'y')]
print(build('whisper', 'whisper~ - speech into text', body,
            demo, links, demo_width=760, text_width=810, text_height=790))


body = """Takes text, and says it aloud - as a signal, like any other source.

eleven_labs~ DOES NOT RUN HERE:
It sends your text to the ElevenLabs service and needs an account. Put your API
key in dpg_system/elevenlabs_key.py, as a line reading api_key = '...'. Without
the network, or without a valid key, it is silent.

That decides where it belongs. Anything an audience says in confidence must not
be fed to it - whisper~, which runs on this machine, is the safe end of a
conversation; this is not.

IT SPEAKS THROUGH THE PATCH:
The speech comes out of 'left out' / 'right out' as a signal, so it sounds
through whatever fader_out~ or audio_out~ it is patched to and can go through
vocoder~, vst~ or a filter on the way, like any other source. Unpatched, it is
silent. To keep what it says, patch the signal into record~; to analyse it,
into a speech node or capture~.

IT QUEUES, AND TELLS YOU:
Sending a second line while the first is still going stacks them up rather than
interrupting; the line holds sixteen, and text arriving at a full line is
dropped.

'speaking' is true from the moment a line is sent off until the last of it has
played - that is what to gate new text on. 'sounding' is narrower: true only
while sound is actually coming out, not during the pause before a phrase
starts. That is the one for a face that should move with the voice, or to
switch whisper~ off (inverted, to its on/off) so it does not transcribe the
node talking. 'backlog' is true while more than one phrase is waiting in line,
and false again once the line has emptied.

'stop' and 'hard stop' do the same thing: they cut the voice off at once and
throw away everything waiting in the line. 'accept input' closes the door
without stopping what is already queued - unticked, new text is ignored.

SPEAKING A PHRASE THAT IS STILL BEING WRITTEN:
'text to speak' wants a finished phrase, which means waiting for the last word
of it before the first word can be spoken. When the text is coming from a
language model, that wait is most of the delay you hear.

'stream text' takes it as it is made instead: patch the model's output straight
in, and bang 'end of response' - the model's own 'end' outlet - when it has
finished. The words go up to the service over a socket it can answer on
continuously, so the voice starts once there is enough to start on, and keeps
going while the rest is still being written. It also sounds better than
chopping the answer into sentences and sending each one: the delivery carries
across the whole answer instead of restarting at every full stop.

'characters before speaking' is how much it gathers before beginning. Fifty is
the least the service accepts and the soonest it will start; more gives it more
to read ahead into, which it says better. A word is never split between two
messages, and a '<backspace>' from a model that is re-choosing a word takes back
a character that has not been sent yet.

Whole phrases and streamed ones share the one queue, so they are spoken in the
order they were given, and 'speaking' stays true from the first piece of a
streamed phrase until the last of it has played. The v3 models cannot be typed
into this way; given streamed text they wait for the end and speak it whole.

THE VOICE SETTINGS ARE PERFORMANCE DIRECTION:
'stability' low lets the delivery vary between renderings and sound more alive;
high makes it consistent and flatter. 'style exaggeration' pushes the character
of the voice. 'similarity_boost' holds it closer to the original recording.
All three run from 0 to 1. 'speed' is rate, from 0.7 to 1.2.

'latency' trades responsiveness against quality - worth raising only if the
delay before it starts is a problem, because the cost is audible. The v3
models do not take it at all, so it is left out of the request for them.

SYNTAX (the plain name, without ~, still works for older patches):
eleven_labs~ [voice name]

The voice is one of the names in your ElevenLabs account; the 'voice' menu
lists them once the node has reached the service. The model starts as Eleven
Turbo v2.5.

EXAMPLE:
eleven_labs~ David

INPUTS and PARAMETERS:

text to speak:
A finished phrase to say.

stream text / end of response:
A phrase as it is being written, and the bang that says that is all of it.

voice / model:
Who says it, and which of the service's models renders it.

speed / stability / similarity_boost / style exaggeration:
How it is said.

latency:
0 to 5. Higher starts sooner and sounds worse.

characters before speaking:
How much streamed text it gathers before it starts. Less is sooner, more is
better spoken.

stop / hard stop:
Cut off now, and empty the line.

accept input:
Unticked, new text is ignored.

level / enable:
How loud (0 to 2), and whether it is on at all.

OUTPUTS:

left out / right out:
The speech as a signal. Patch to fader_out~ or audio_out~ to hear it.

speaking:
True while it is busy with a line, until the last of it has played.

sounding:
True while sound is coming out this instant.

backlog:
True while more than one phrase is waiting.

RELATED:
whisper~ for the other direction - speech into text, on this machine.
gemma_4 or qwen_moe to write what it says.
fader_out~ and audio_out~ to hear it, record~ to keep it."""

demo = [
    {'key': 's1', 'init': 'string', 'pos': (30, 62), 'w': 360, 'h': 42,
     'props': {'text in': 'Hello. I am reading this aloud.', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'press the button beside the text to\nsend it', 'pos': (440, 62)},
    {'key': 'el', 'init': 'eleven_labs~', 'pos': (30, 130), 'w': 293, 'h': 308},
    {'key': 'c1', 'comment': True, 'text': 'this one SENDS YOUR TEXT to a service\nand needs an API key. Nothing private\nshould go through it',
     'pos': (440, 130)},

    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (30, 470), 'w': 70, 'h': 308},
    {'key': 'c2', 'comment': True, 'text': 'the speech is a signal: nothing\nsounds until it reaches a fader_out~\nor audio_out~',
     'pos': (440, 470)},

    {'key': 'tg1', 'init': 'toggle', 'pos': (30, 810), 'w': 45, 'h': 42},
    {'key': 'c3', 'comment': True, 'text': "speaking - gate on this. A second line\nwhile the first is going is queued,\nit does not interrupt",
     'pos': (440, 810)},
    {'key': 'tg2', 'init': 'toggle', 'pos': (30, 880), 'w': 45, 'h': 42},
    {'key': 'c4', 'comment': True, 'text': 'sounding - sound is coming out right now',
     'pos': (440, 880)},
    {'key': 'tg3', 'init': 'toggle', 'pos': (30, 950), 'w': 45, 'h': 42},
    {'key': 'c5', 'comment': True, 'text': 'backlog - more than one phrase waiting',
     'pos': (440, 950)},
]
links = [('s1', 'string out', 'el', 'text to speak'),
         ('el', 'speaking', 'tg1', ''),
         ('el', 'sounding', 'tg2', ''),
         ('el', 'backlog', 'tg3', ''),
         ('el', 'left out', 'fo', 'left', 0, 0),
         ('el', 'right out', 'fo', 'right', 1, 1)]
print(build('eleven_labs', 'eleven_labs~ - text into speech', body,
            demo, links, demo_width=760, text_width=810, text_height=790))
