"""text: assembling, substituting, matching, accumulating, codes, files."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ---------------------------------------------------------------------- combine
body = """These put strings together and take them apart.

THE NODES:

combine   join several inlets into one string, with a separator
kombine   the same node
join      join a list into one string
split     cut a string into a list

combine VERSUS join:
combine takes several INLETS, one per thing you are assembling, and joins 
whatever is sitting in each. join takes one LIST and joins its elements. 
Use combine when the parts arrive from different places in the patch, and join 
when they already travel together.

combine holds each inlet's value between messages, so the parts do not have to 
arrive at the same moment - the leftmost inlet triggers the output and the rest 
contribute whatever they last received. That is what makes it usable for 
assembling a line out of several slow sources.

split IS THE INVERSE OF join:
Give it a string and something to split at, and you get a list. The default is 
a space, which turns a sentence into words - and once it is a list, the list 
nodes and the numpy nodes can work on it.

SYNTAX:
combine <separator>
join
split <split at>

EXAMPLE:
combine " "

INPUTS and PARAMETERS:

separator (combine):
What goes between the parts. A space by default.

<one inlet per part> (combine):
The pieces. The first triggers the output.

in / join with (join):
The list, and what to put between its elements.

in / split at (split):
The string, and what to cut on.

OUTPUTS: 

out / string out / substrings out:
The assembled string, or the list of pieces.

A NOTE ON SEPARATORS:
Splitting on a separator and joining with the same one is not always a round 
trip - consecutive separators produce empty pieces, and leading or trailing 
ones produce empties at the ends. If a rejoined string has doubled spaces in 
it, that is why."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 180, 'h': 42,
     'props': {'text in': 'the quick brown fox', 'font size': '24'}},
    {'key': 'sp', 'init': 'split', 'pos': (30, 180), 'w': 180, 'h': 90},
    {'key': 'l1', 'init': 'list', 'pos': (30, 285), 'w': 280, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'now a list of words', 'pos': (30, 335)},
    {'key': 'jn', 'init': 'join', 'pos': (30, 375), 'w': 180, 'h': 90},
    {'key': 's1', 'init': 'string', 'pos': (30, 480), 'w': 280, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'and back to one string', 'pos': (30, 530)},
    {'key': 'm2', 'init': 'message', 'pos': (340, 118), 'w': 140, 'h': 42,
     'props': {'text in': 'hello', 'font size': '24'}},
    {'key': 'm3', 'init': 'message', 'pos': (340, 175), 'w': 140, 'h': 42,
     'props': {'text in': 'there', 'font size': '24'}},
    {'key': 'cb', 'init': 'combine 2', 'pos': (340, 235), 'w': 200, 'h': 120},
    {'key': 's2', 'init': 'string', 'pos': (340, 375), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'parts from different places', 'pos': (340, 425)},
]
links = [('btn', '', 'm1', ''), ('btn', '', 'm2', ''), ('btn', '', 'm3', ''),
         ('m1', 'message out', 'sp', 'in'), ('sp', 'substrings out', 'l1', ''),
         ('sp', 'substrings out', 'jn', 'in'), ('jn', 'string out', 's1', ''),
         ('m3', 'message out', 'cb', 'in 2'),
         ('m2', 'message out', 'cb', 'in 1'), ('cb', 'out', 's2', '')]
print(build('combine', 'combine, join, split - assembling and taking apart', body,
            demo, links, demo_width=580, text_width=790, text_height=640))

# --------------------------------------------------------------- string_replace
body = """These substitute or clean up parts of a string.

THE NODES:

string_replace  find a piece of text anywhere and put another in its place
word_replace    the same, but only where the match is a WHOLE word
unescape_text   turn HTML escapes back into the characters they stand for
printable       drop anything that is not a printable character

word_replace VERSUS string_replace:
string_replace matches anywhere, so replacing "cat" also hits "concatenate". 
word_replace matches only whole words, so it does not. Which you want depends on 
whether you are editing text or editing markup - for anything language-shaped, 
whole words is almost always right, and the substring version's surprises are 
the classic source of mangled output.

unescape_text AND printable ARE FOR TEXT THAT ARRIVED FROM SOMEWHERE ELSE:
Text off a network, out of a web page, or from a speech service arrives with 
things in it that are not really part of the text - HTML escapes like &amp;, 
control characters, byte-order marks, invisible formatting. 

unescape_text handles the escapes; printable removes what cannot be shown. 
Running both on anything incoming costs nothing and saves the class of problem 
where a string looks correct and behaves oddly - a comparison that fails, a 
width that is wrong, a word that will not match.

SYNTAX:
string_replace <find> <replace>
word_replace <find> <replace>

EXAMPLE:
word_replace colour color

INPUTS and PARAMETERS:

string in:
The text. Receiving it triggers the node.

find / replace:
What to look for and what to put in its place.

OUTPUTS: 

string out / printable characters out / unescaped string out:
The result.

RELATED:
replace and int_replace do the same job on lists and numbers rather than 
within a string - see the replace help patch. 
dict_replace holds many substitutions at once."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 320, 'h': 42,
     'props': {'text in': 'the cat concatenated', 'font size': '24'}},
    {'key': 'sr', 'init': 'string_replace cat dog', 'pos': (30, 180), 'w': 220, 'h': 110},
    {'key': 's1', 'init': 'string', 'pos': (30, 305), 'w': 320, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'matches anywhere: concatenated changes too',
     'pos': (30, 355)},
    {'key': 'wr', 'init': 'word_replace cat dog', 'pos': (30, 395), 'w': 220, 'h': 110},
    {'key': 's2', 'init': 'string', 'pos': (30, 520), 'w': 320, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'whole words only: it does not', 'pos': (30, 570)},
    {'key': 'pt', 'init': 'printable', 'pos': (400, 395), 'w': 200, 'h': 90},
    {'key': 'c2', 'comment': True, 'text': 'run this on anything incoming',
     'pos': (400, 495)},
]
links = [('btn', '', 'm1', ''),
         ('m1', 'message out', 'sr', 'string in'), ('sr', 'string out', 's1', ''),
         ('m1', 'message out', 'wr', 'string in'), ('wr', 'string out', 's2', ''),
         ('m1', 'message out', 'pt', 'characters in')]
print(build('string_replace', 'string_replace - substituting and cleaning up', body,
            demo, links, demo_width=630, text_width=790, text_height=640))

# ----------------------------------------------------------------- word_trigger
body = """These watch text going past and react to particular words.

THE NODES:

word_trigger          an outlet per word you name; fires when that word appears
first_letter_trigger  an outlet per letter; fires on the word's first letter
word_gate             pass only words that are in a dictionary
fuzzy_match           find the closest match rather than an exact one

word_trigger IS route FOR LANGUAGE:
Name the words as arguments and each gets an outlet. A stream of text - from 
speech recognition, from a chat, from a file - then drives the patch directly: 
say a word, something happens. It is the same shape as osc_route or the select 
node, applied to language.

fuzzy_match EXISTS BECAUSE EXACT MATCHING FAILS ON SPEECH:
Recognised speech is approximate. It gives you "recognise" for "recognize", 
"there" for "their", and words that are simply wrong but close. Exact matching 
against a list therefore misses most of what was actually said.

fuzzy_match reports the closest entry and a SCORE for how close, so you can 
decide what counts. The 'threshold' sets how far off is acceptable. Watch the 
score outlet while tuning - a threshold set without looking at real scores is 
a guess, and the useful range is usually narrower than you would expect.

word_gate FILTERS BY VOCABULARY:
Give it a dictionary and only words in it get through. That is how you keep a 
text stream to a known vocabulary - useful when the words are going to drive 
something that only understands certain ones, and as a way of ignoring the 
noise in a recognition stream without having to enumerate what to reject.

SYNTAX:
word_trigger <word> <word> ...
first_letter_trigger
word_gate
fuzzy_match

EXAMPLE:
word_trigger stop start faster

INPUTS and PARAMETERS:

string in:
The text to watch. Receiving it triggers the node.

dictionary in / load dictionary / include word (word_gate):
The vocabulary, and a way to add to it.

threshold (fuzzy_match):
How close a match has to be.

OUTPUTS: 

<one outlet per word or letter>:
Fires when that word or letter appears.

gated string out:
Only the words that passed.

string out / score out / replacement out (fuzzy_match):
What was matched, how close it was, and what to substitute.

RELATED:
The spacy nodes do grammatical analysis where these do matching. 
select and route do the same job for messages rather than words."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 320, 'h': 42,
     'props': {'text in': 'please go faster now', 'font size': '24'}},
    {'key': 'wt', 'init': 'word_trigger stop faster slower', 'pos': (30, 180),
     'w': 280, 'h': 140},
    {'key': 'c0', 'comment': True, 'text': 'an outlet per word you name',
     'pos': (30, 335)},
    {'key': 'cnt', 'init': 'counter', 'pos': (30, 375), 'w': 123, 'h': 84},
    {'key': 'i1', 'init': 'int', 'pos': (30, 470), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c1', 'comment': True, 'text': 'counts how often faster was said',
     'pos': (30, 520)},
    {'key': 'fm', 'init': 'fuzzy_match', 'pos': (400, 375), 'w': 240, 'h': 180},
    {'key': 'f1', 'init': 'float', 'pos': (400, 570), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': 'watch the score before setting\na threshold - guessing it rarely works',
     'pos': (400, 620)},
]
links = [('btn', '', 'm1', ''),
         ('m1', 'message out', 'wt', 'string in'),
         ('wt', 'faster', 'cnt', 'input'), ('cnt', 'count out', 'i1', ''),
         ('m1', 'message out', 'fm', 'string in'), ('fm', 'score out', 'f1', '')]
print(build('word_trigger', 'word_trigger - reacting to what was said', body, demo,
            links, demo_width=670, text_width=800, text_height=680))

# --------------------------------------------------------------- gather_sentence
body = """These accumulate text over time, rather than acting on one message.

Speech recognition and language models emit text a fragment at a time. 
None of it is a sentence, and most of what you want to do needs one. 
These four assemble the stream into something whole, in different ways.

THE NODES:

gather_sentence  collect fragments until a sentence ends, then send it
string_builder   collect until you ask for it
fifo_string      keep a window of recent text, with older parts fading
text_change      report only the words that are NEW

gather_sentence DECIDES WHERE A SENTENCE ENDS:
'auto sentence end' looks for the punctuation that ends one. 'end on return' 
treats a line break as the end. 'force string end' lets the patch decide. 
'enforce spaces' fixes fragments that arrive without them, which recognition 
output often does.

'skip framed by' ignores anything between a pair of characters - so stage 
directions, annotations or markup in the stream do not end up in the sentence.

fifo_string IS THE ONE FOR A ROLLING CONTEXT:
It holds recent text as a window, and its 'weighted out' outlet carries the 
text with a weight per piece that DECAYS with age. That is what you want when 
feeding something that should be influenced more by what was just said than by 
what was said a minute ago - an image prompt, a mood, a running state.

'decay_rate' sets how fast the past fades. 'length_threshold' caps how much is 
kept. 'progress' advances the ages without adding anything, so time can pass 
without new text.

text_change REPORTS ONLY WHAT IS NEW:
Given a stream that keeps restating the same thing, it sends only the words 
that were not there before. 'persistence' is how many times a word must be 
absent before it counts as new again, and 'reset_period' clears the memory 
periodically so a word can recur.

That is how you drive something from speech without it re-firing on every 
repetition of the same phrase.

SYNTAX:
gather_sentence
fifo_string
text_change

EXAMPLE:
gather_sentence

INPUTS and PARAMETERS:

string in / text input:
The fragments.

force string end / issue text:
Emit what has been collected now.

progress (fifo_string):
Age the window without adding to it.

clear / dump_oldest:
Empty it, or drop the oldest piece.

order / decay_rate / length_threshold (fifo_string):
Which end is newest, how fast the past fades, and how much is kept.

persistence / reset_period (text_change):
How long a word is remembered, and how often the memory clears.

OUTPUTS: 

sentences out / text out:
The assembled text.

weighted out / string out (fifo_string):
The window with per-piece weights, and as plain text.

new words out:
Only the words not seen recently."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 300, 'h': 42,
     'props': {'text in': 'the light is changing.', 'font size': '24'}},
    {'key': 'gs', 'init': 'gather_sentence', 'pos': (30, 180), 'w': 260, 'h': 180},
    {'key': 's1', 'init': 'string', 'pos': (30, 375), 'w': 320, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'fragments in, whole sentences out',
     'pos': (30, 425)},
    {'key': 'fs', 'init': 'fifo_string', 'pos': (30, 465), 'w': 280, 'h': 220},
    {'key': 's2', 'init': 'string', 'pos': (30, 700), 'w': 320, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'a rolling window; older text fades',
     'pos': (30, 750)},
    {'key': 'tc', 'init': 'text_change', 'pos': (400, 465), 'w': 260, 'h': 180},
    {'key': 's3', 'init': 'string', 'pos': (400, 660), 'w': 260, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'click twice: the second time\nnothing is new',
     'pos': (400, 710)},
]
links = [('btn', '', 'm1', ''),
         ('m1', 'message out', 'gs', 'string in'),
         ('gs', 'sentences out', 's1', ''),
         ('m1', 'message out', 'fs', 'in'), ('fs', 'string out', 's2', ''),
         ('m1', 'message out', 'tc', 'text input'),
         ('tc', 'new words out', 's3', '')]
print(build('gather_sentence', 'gather_sentence - assembling text over time', body,
            demo, links, demo_width=690, text_width=800, text_height=740))

# ------------------------------------------------------------------------ ascii
body = """These convert between characters and their numeric codes.

THE NODES:

ascii      a character to its code
ord        the same node
char       a code back to its character
character  the same node

WHY YOU WOULD WANT THIS:
Because a code is a NUMBER, and numbers can be compared, offset, sorted and 
arithmetic'd where characters cannot. Testing whether a character is a digit is 
a range test on its code. Shifting a letter through the alphabet is addition. 
Sorting by character is sorting by code.

It is also how you reach characters that cannot be typed into a message - a 
tab, a newline, a control character. Send the code and get the character back.

THE CODES WORTH KNOWING:
32 is space, 48 to 57 are the digits, 65 to 90 are capital A to Z, and 97 to 
122 are lower case a to z. The gap of 32 between the cases is why adding or 
subtracting 32 changes case - which is a trick worth knowing and worth not 
relying on, because it only holds for unaccented Latin letters.

SYNTAX:
ascii
char

EXAMPLE:
ascii

INPUTS and PARAMETERS:

character in:
The character. Receiving it triggers the node.

char code in:
The code.

OUTPUTS: 

ascii out:
The numeric code.

character out:
The character.

A NOTE ON WHAT 'ASCII' MEANS HERE:
The name is historical. Codes above 127 are not ASCII at all, and a character 
outside the Latin alphabet may not be one code but several bytes. For plain 
English text these nodes behave exactly as expected; for anything else, expect 
the relationship between characters and codes to be less simple than the name 
suggests."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'm1', 'init': 'message', 'pos': (30, 118), 'w': 120, 'h': 42,
     'props': {'text in': 'A', 'font size': '24'}},
    {'key': 'as', 'init': 'ascii', 'pos': (30, 180), 'w': 160, 'h': 70},
    {'key': 'i1', 'init': 'int', 'pos': (30, 265), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c0', 'comment': True, 'text': 'capital A is 65', 'pos': (30, 315)},
    {'key': 'add', 'init': '+ 32', 'pos': (30, 355), 'w': 130, 'h': 70,
     'props': {'operand': 32}},
    {'key': 'ch', 'init': 'char', 'pos': (30, 440), 'w': 160, 'h': 70},
    {'key': 's1', 'init': 'string', 'pos': (30, 525), 'w': 160, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'add 32 and it becomes lower case\ntrue for plain Latin letters only',
     'pos': (30, 575)},
]
links = [('btn', '', 'm1', ''),
         ('m1', 'message out', 'as', 'character in'), ('as', 'ascii out', 'i1', ''),
         ('as', 'ascii out', 'add', 'in'),
         ('add', 'result', 'ch', 'char code in'), ('ch', 'character out', 's1', '')]
print(build('ascii', 'ascii and char - characters as numbers', body, demo, links,
            demo_width=420, text_width=780, text_height=600))

# -------------------------------------------------------------------- text_file
body = """These hold a body of text you can edit, load and save.

THE NODES:

text_editor  an editable text area in the patch
text_file    the same node

WHAT IT IS FOR:
Anything the patch needs that is longer than a message and more editable than a 
file on disk. A prompt, a script, a list of names, a set of instructions, notes 
that belong with the patch.

Because the contents are stored WITH the patch, text kept here travels with it - 
you do not end up with a patch that depends on a file somebody has to remember 
to copy. For text that genuinely belongs in its own file, 'load' and 'save' 
handle that.

THE TWO OUTLETS:
'out' sends the whole contents. 'messages' sends it line by line, each line as 
a separate message - which is how you use the editor as a script: write the 
lines, click send, and each one goes out in order.

That second one is worth knowing, because it turns a text area into a small 
sequencer of commands without any other machinery.

SYNTAX:
text_editor
text_file <name>

EXAMPLE:
text_editor

INPUTS and PARAMETERS:

text in:
Replace the contents.

append text in:
Add to the end rather than replacing - so the editor can accumulate a log.

send:
Emit the contents.

clear:
Empty it.

load / save:
Read from or write to a file. These open a file dialog.

name:
What this block of text is called.

editor width / editor height:
The size of the area.

OUTPUTS: 

out:
The whole contents.

messages:
The contents line by line, in order.

RELATED:
text_block is the node these help patches use for their own prose - a locked 
block for reading rather than editing. 
text_display is for output that scrolls."""

demo = [
    {'key': 'te', 'init': 'text_editor', 'pos': (30, 62), 'w': 340, 'h': 280},
    {'key': 'c0', 'comment': True, 'text': 'type into it; the text is saved with\nthe patch, so it travels with it',
     'pos': (30, 355)},
    {'key': 's1', 'init': 'string', 'pos': (30, 425), 'w': 340, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'click send: out gives the whole thing',
     'pos': (30, 475)},
    {'key': 'pr', 'init': 'print line', 'pos': (30, 515), 'w': 200, 'h': 120,
     'props': {'identifier': 'line', 'precision': 3}},
    {'key': 'c3', 'comment': True, 'text': 'messages gives it a line at a time -\nthe editor as a little script',
     'pos': (30, 650)},
]
links = [('te', 'out', 's1', ''), ('te', 'messages', 'pr', 'in')]
print(build('text_file', 'text_editor - a body of text in the patch', body, demo,
            links, demo_width=420, text_width=790, text_height=620))
