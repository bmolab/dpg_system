"""Streaming speech to text with a transducer - words that are final the instant they appear."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """Speech into text, with no guessing stage.

THE NODE:

nemotron   listens, and gives you each word the moment it is decided

HOW IT DIFFERS FROM whisper:
whisper re-reads the last thirty seconds many times a second and works out what
has stopped changing; that is where its in_progress guesses and its confirmation
delay come from. nemotron is a streaming transducer: sound goes in once, and
every token it emits is final. Nothing is ever withdrawn. Words arrive about a
tenth of a second behind the speaker, and the cost is the same whether a phrase
has been open for one second or thirty.

The price is that there is no second look. If it hears a word wrong, that word
stays wrong. whisper's revisions were often corrections; here there are none.

It runs on this machine, on the GPU through MLX. The first switch-on downloads
the model (about 1.2 GB) and takes a few seconds to load; after that it starts
at once.

FOUR OUTLETS, TWO KINDS OF BOUNDARY:
The model does not say where a phrase or a sentence ends, so the node decides,
and it decides at two scales.

in_progress            the current phrase, growing a word at a time. It only
                       ever gets longer. Feed the newest fifo_string register.
phrase                 the phrase, once it closes: a pause of 'phrase_silence'
                       seconds, a sentence end, or 'max_phrase_words'. Shift
                       the fifo on this.
sentence_in_progress   the sentence so far, sent again each time a phrase
                       closes. This is for context_tracker's 'provisional in':
                       the tracker follows it at once without committing.
sentence               the sentence, once it closes: a full stop, question mark
                       or exclamation; a pause of 'sentence_silence' seconds;
                       or 'max_sentence_words'. Feed context_tracker's 'text in'.

A sentence is one or more phrases. A pause for thought closes a phrase but not
the sentence, so the tracker sees the half-sentence provisionally, and the
whole sentence once, later.

PUNCTUATION COMES LATE, ON PURPOSE:
The model emits the full stop together with the FIRST WORD OF THE NEXT SENTENCE,
not at the pause. So a pause usually closes the phrase before its full stop
arrives. When a bare full stop then turns up, the node attaches it to the
sentence rather than starting a new phrase with it - which is also what closes
the sentence. At the shortest look-ahead the model often omits sentence
punctuation altogether, and 'sentence_silence' does the work instead.

LOOK-AHEAD IS THE TRADE:
'look-ahead' is how much future sound the model hears before committing to a
word: 80 ms up to 1120 ms. Shorter is faster and slightly less accurate, and
punctuation thins out. 160 ms is a good default; 560 ms if the words matter
more than the beat.

'language' is a prompt key such as en-US or fr-FR; 'auto' lets it detect.
'model' offers the 8-bit build, which is smaller and about as good.

CONTINUOUS SPEECH RUNS ON, AND 'semantic_split' IS THE ANSWER:
Someone reading, or a podcast, leaves pauses too short for the silence rules,
and the model marks only some of the sentence ends it hears. Sentences then
grow until a length cap cuts them somewhere arbitrary. 'semantic_split' turns
on a second, small model on the CPU that reads the open text and says where
the sentences end - the ends the recogniser left unmarked. It runs when a
phrase closes and every 'split_check_every' new words, once the open text is
at least 'split_min_words' long, and never cuts inside the last
'split_guard_words', which have no right context yet. Where the recogniser
itself put a comma it is trusted and no cut is made there.

Some speakers never end a sentence at all. Past 'split_max_words' with no
sentence end in sight, the node breaks at the most plausible clause boundary
instead: a comma the recogniser heard, or the strongest clause mark the
splitter predicts, preferring the later of near-equal candidates so the pieces
stay as long as they plausibly can. Only when even that fails does
'max_sentence_words' cut blindly. No piece is ever shorter than a few words.

When a cut falls inside the current phrase, that much of it goes out as a
'phrase' first, so the fifo shifts, then the 'sentence', then the remainder
carries on as the new phrase. The first switch-on downloads about a gigabyte.
Leave it off for live conversation, where pauses and the model's own
punctuation already do the work.

'record_debug' writes the audio the model actually heard, and a log of every
token with its time, to ~/nemotron_debug. For when words go missing: if they
are in the recording but not the log, the model lost them; if they are not in
the recording, something before the node did.

SYNTAX:
nemotron
nemotron 8bit
nemotron 560 ms fr-FR

EXAMPLE:
nemotron

INPUTS and PARAMETERS:

on/off / audio device / audio_in / sample_rate_in:
Start listening, and where from. Patch a signal into audio_in to use that
instead of a device.

model / look-ahead / language:
Which build, how much future it hears, and what language to expect.

phrase_silence / sentence_silence:
How long a pause closes a phrase, and how long a pause closes a sentence.

max_phrase_words / max_sentence_words:
Length caps, for when nobody pauses. 0 is off.

gain:
Input level.

semantic_split / split_min_words / split_guard_words / split_check_every:
Find the sentence ends the recogniser leaves unmarked in continuous speech.

split_max_words:
Past this, with no sentence end found, break at the best clause boundary.

record_debug:
Save what the model heard, and what it made of it, for diagnosis.

OUTPUTS:

in_progress:
The current phrase, growing. Display this, or feed a fifo register.

phrase:
A closed phrase. Shift the fifo on this.

sentence_in_progress:
The sentence so far. context_tracker 'provisional in'.

sentence:
A closed sentence. context_tracker 'text in'.

RELATED:
whisper, which guesses and revises, and hears noise as words.
context_tracker, which this was shaped to feed.
fifo_string, for the last few phrases as a rolling window."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'nm', 'init': 'nemotron', 'pos': (30, 120), 'w': 340, 'h': 300},
    {'key': 'c0', 'comment': True, 'text': 'every word is final when it appears -\nnothing is revised, and nothing is\nre-read',
     'pos': (30, 435)},

    {'key': 'td', 'init': 'text_display', 'pos': (420, 120), 'w': 340, 'h': 180,
     'props': {'width': 320, 'height': 140, 'wrap': True, 'max_lines': 100,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'in_progress: the phrase, growing a word\nat a time. It only gets longer.',
     'pos': (420, 315)},

    {'key': 'td2', 'init': 'text_display', 'pos': (420, 390), 'w': 340, 'h': 180,
     'props': {'width': 320, 'height': 140, 'wrap': True, 'max_lines': 100,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c4', 'comment': True, 'text': 'phrase: closed by a pause or a sentence\nend. Shift a fifo on this.',
     'pos': (420, 585)},

    {'key': 'td3', 'init': 'text_display', 'pos': (420, 660), 'w': 340, 'h': 180,
     'props': {'width': 320, 'height': 140, 'wrap': True, 'max_lines': 100,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c6', 'comment': True, 'text': 'sentence_in_progress: the sentence so far,\nre-sent at each phrase. context_tracker\n"provisional in" follows it without\ncommitting',
     'pos': (420, 855)},

    {'key': 'td4', 'init': 'text_display', 'pos': (420, 960), 'w': 340, 'h': 180,
     'props': {'width': 320, 'height': 140, 'wrap': True, 'max_lines': 100,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c8', 'comment': True, 'text': 'sentence: closed by . ? ! or a long pause.\ncontext_tracker "text in" digests this once',
     'pos': (420, 1155)},

    {'key': 'c9', 'comment': True, 'text': 'the full stop arrives with the NEXT\nsentence\'s first word, so a pause\ncloses the phrase before its\npunctuation - the node attaches it\nto the sentence when it comes',
     'pos': (30, 520)},
]
links = [('tog', '', 'nm', 'on/off'),
         ('nm', 'in_progress', 'td', '###text in'),
         ('nm', 'phrase', 'td2', '###text in'),
         ('nm', 'sentence_in_progress', 'td3', '###text in'),
         ('nm', 'sentence', 'td4', '###text in')]
print(build('nemotron', 'nemotron - streaming speech to text, final as it arrives', body,
            demo, links, demo_width=840, text_width=810, text_height=1700))
