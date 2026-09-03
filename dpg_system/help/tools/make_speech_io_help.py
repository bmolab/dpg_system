"""Speech in and speech out."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """Speech into text, and text back into speech.

THE NODES:

whisper       listens, and gives you what was said
eleven_labs   takes text, and says it aloud

whisper RUNS HERE; eleven_labs DOES NOT:
whisper transcribes on this machine - nothing leaves it, there is no account,
and it keeps working with the network unplugged. eleven_labs sends your text to
a service and needs an API key in dpg_system/elevenlabs_key.py.

That difference decides where each belongs. Anything an audience says in
confidence can go through whisper and must not go through eleven_labs.

whisper GIVES YOU TWO STREAMS, AND THE DIFFERENCE MATTERS:

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
Segments have an AGE - how many analysis rounds they have survived unchanged -
and 'confirmation_age' is how old one must be before it is trusted. A phrase is
then emitted when a confirmed segment ends in a full stop, question mark or
exclamation.

The age required shrinks as the sentence gets longer, scaled by 'length_factor'.
The reasoning is that a long utterance has already given the model plenty of
context, so its early words are unlikely to be revised - and waiting the full
age on every one of them would leave the transcript lagging badly behind the
speaker.

'minimum_confidence' and 'min_trailing_confidence' throw away segments the model
itself is unsure of, the second applying to the end of a phrase where guessing
is worst. 'minimum_lifespan' stops very short-lived fragments being emitted at
all.

THE 'noises' OUTLET IS NOT A MISTAKE:
Whisper hallucinates when there is no speech - it will confidently transcribe
breathing, a chair, a hum as words. Rather than let that into 'phrases', those
segments come out of 'noises' instead.

Which makes 'noises' quite interesting in its own right: it is what the room
sounded like to something trying very hard to hear language in it.

'silence_threshold' and 'silence_period' set what counts as nothing happening,
and 'energy' reports the level so you can see where the threshold should be.

THE MODEL IS A TRADE:
Larger models are more accurate and slower, and the delay is felt directly as
the transcript trailing the speaker. Start small and go larger only if the words
are actually wrong; a smaller model that keeps up is usually better in
performance than a better one that lags.

'language' can be set or left to detect, and 'translate' asks for English out
regardless of what went in.

eleven_labs SPEAKS, AND QUEUES:
Send text and it says it. 'backlog' reports how much is waiting, and 'speaking'
whether it is talking now - which is what to gate on, because sending a second
line while the first is still going stacks them up rather than interrupting.

'stop' finishes the current phrase and stops; 'hard stop' cuts immediately.
'accept input' closes the door without stopping what is already queued.

THE VOICE SETTINGS ARE PERFORMANCE DIRECTION:
'stability' low lets the delivery vary between renderings and sound more alive;
high makes it consistent and flatter. 'style exaggeration' pushes the character
of the voice. 'similarity_boost' holds it closer to the original recording.
'speed' is rate.

'latency' trades responsiveness against quality - worth raising only if the
delay before it starts is a problem, because the cost is audible.

SYNTAX:
whisper
eleven_labs

EXAMPLE:
whisper

INPUTS and PARAMETERS:

on/off / audio device / audio_in:
Start listening, and where from.

model / language / translate:
Which model, what language, and whether to render English.

silence_threshold / silence_period:
What counts as nothing being said.

confirmation_age / minimum_lifespan / minimum_confidence / length_factor:
How sure it must be before a phrase is emitted.

text to speak / voice / model:
What to say and who says it.

speed / stability / style exaggeration / similarity_boost:
How it is said.

stop / hard stop / accept input:
End politely, end now, or stop taking new text.

OUTPUTS: 

phrases:
Settled text. Act on this.

in_progress:
The current guess, revised as it goes. Display this.

noises:
What it heard as words when there was no speech.

energy / rate / language:
Level, speed, and what language it thinks it is.

speaking / backlog:
Whether it is talking, and how much is queued.

RELATED:
translate to move between languages in between.
gemma_4 to answer rather than repeat.
cairo_layout to put the words on a screen."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'wh', 'init': 'whisper', 'pos': (30, 120), 'w': 340, 'h': 480},
    {'key': 'c0', 'comment': True, 'text': 'runs on this machine - no account, and',
     'pos': (30, 615)},
    {'key': 'c1', 'comment': True, 'text': 'nothing leaves the room',
     'pos': (30, 645)},

    {'key': 'td', 'init': 'text_display', 'pos': (420, 120), 'w': 340, 'h': 180,
     'props': {'width': 320, 'height': 140, 'wrap': True, 'max_lines': 100,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'in_progress: the current guess, revised',
     'pos': (420, 315)},
    {'key': 'c3', 'comment': True, 'text': 'as it listens. Show this.',
     'pos': (420, 345)},

    {'key': 'td2', 'init': 'text_display', 'pos': (420, 390), 'w': 340, 'h': 180,
     'props': {'width': 320, 'height': 140, 'wrap': True, 'max_lines': 100,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c4', 'comment': True, 'text': 'phrases: settled, emitted once. ACT on',
     'pos': (420, 585)},
    {'key': 'c5', 'comment': True, 'text': 'this - in_progress will be withdrawn',
     'pos': (420, 615)},

    {'key': 'td3', 'init': 'text_display', 'pos': (420, 660), 'w': 340, 'h': 140,
     'props': {'width': 320, 'height': 100, 'wrap': True, 'max_lines': 60,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c6', 'comment': True, 'text': 'noises: what it heard as words when',
     'pos': (420, 815)},
    {'key': 'c7', 'comment': True, 'text': 'nobody was speaking. Kept out of',
     'pos': (420, 845)},
    {'key': 'c8', 'comment': True, 'text': 'phrases on purpose - and interesting',
     'pos': (420, 875)},

    {'key': 'pl', 'init': 'plot', 'pos': (30, 690), 'w': 300, 'h': 180,
     'props': PLOT(0.0, 1.0, 200)},
    {'key': 'c9', 'comment': True, 'text': 'energy - use it to set silence_threshold',
     'pos': (30, 880)},

    {'key': 'el', 'init': 'eleven_labs', 'pos': (30, 930), 'w': 340, 'h': 420},
    {'key': 'c10', 'comment': True, 'text': 'this one SENDS YOUR TEXT to a service',
     'pos': (30, 1365)},
    {'key': 'c11', 'comment': True, 'text': 'and needs an API key. Nothing private',
     'pos': (30, 1395)},
    {'key': 'c12', 'comment': True, 'text': 'should go through it',
     'pos': (30, 1425)},

    {'key': 'tg2', 'init': 'toggle', 'pos': (420, 930), 'w': 45, 'h': 42},
    {'key': 'c13', 'comment': True, 'text': "'speaking' - gate on this. Sending a",
     'pos': (480, 930)},
    {'key': 'c14', 'comment': True, 'text': 'second line while the first is going',
     'pos': (480, 960)},
    {'key': 'c15', 'comment': True, 'text': 'stacks them up, it does not interrupt',
     'pos': (480, 990)},
    {'key': 'i1', 'init': 'int', 'pos': (420, 1030), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c16', 'comment': True, 'text': 'backlog - how much is waiting',
     'pos': (420, 1080)},
]
links = [('tog', '', 'wh', 'on/off'),
         ('wh', 'in_progress', 'td', '###text in'),
         ('wh', 'phrases', 'td2', '###text in'),
         ('wh', 'noises', 'td3', '###text in'),
         ('wh', 'energy', 'pl', 'y'),
         ('wh', 'phrases', 'el', 'text to speak'),
         ('el', 'speaking', 'tg2', ''),
         ('el', 'backlog', 'i1', '')]
print(build('whisper', 'whisper and eleven_labs - speech in and out', body,
            demo, links, demo_width=840, text_width=810, text_height=790))
