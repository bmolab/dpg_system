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
body = """These find a piece of text and put another in its place.

THE NODES:

string_replace  replace the text wherever it appears, even inside a word
word_replace    replace it only where it is a WHOLE word

word_replace VERSUS string_replace:
string_replace matches anywhere, so replacing "cat" also hits "concatenate".
word_replace matches only whole words, so it does not. For anything
language-shaped, whole words is almost always what you want, and the
everywhere version's surprises are the classic source of mangled output.

Both are case-sensitive: "cat" does not touch "Cat".

'find' IS A PATTERN, NOT PLAIN TEXT:
It is read as a regular expression, so some characters have special meanings.
A full stop matches any character, "colou?r" matches both "color" and "colour",
and "cat|dog" matches either word. To find one of . * + ? ( ) [ ] | \\ ^ $
literally, put a backslash in front of it. An unmatched bracket is an error,
and the node then sends nothing.

LISTS:
string_replace given a list replaces within each item that is text and leaves
numbers alone; a list goes out. word_replace turns a list of words into one
string, joined with spaces, before it starts; a string goes out.

A STRING THAT STARTS WITH 'find' OR 'replace':
Text arriving at 'string in' whose first word is 'find' or 'replace' is taken
as a setting, not as text: "replace dog" sets the replace field to "dog" and
nothing is sent. Handy from a message box, a surprise in running text.

If 'find' is empty the input is passed on unchanged.

SYNTAX:
string_replace <find> <replace>
word_replace <find> <replace>

EXAMPLE:
word_replace colour color

INPUTS and PARAMETERS:

string in:
The text. Receiving it triggers the node.

find:
What to look for.

replace:
What to put in its place. Leave it empty to delete what was found.

OUTPUTS:

string out:
The result.

RELATED:
unescape_text and printable clean up text that arrived from somewhere else.
replace and int_replace do the same job on lists and numbers rather than
within a string - see the replace help patch.
dict_replace holds many substitutions at once."""

demo = [
    {'key': 's0', 'init': 'string', 'pos': (30, 62), 'w': 320, 'h': 42,
     'props': {'text in': 'the cat concatenated', 'font size': '24', 'width': 280}},
    {'key': 'c0', 'comment': True, 'text': 'click to send it', 'pos': (30, 112)},
    {'key': 'sr', 'init': 'string_replace cat dog', 'pos': (30, 150), 'w': 220, 'h': 110},
    {'key': 's1', 'init': 'string', 'pos': (30, 275), 'w': 320, 'h': 42,
     'props': {'text in': '', 'font size': '24', 'width': 280}},
    {'key': 'c1', 'comment': True, 'text': 'matches anywhere: concatenated changes too',
     'pos': (30, 325)},
    {'key': 'wr', 'init': 'word_replace cat dog', 'pos': (30, 365), 'w': 220, 'h': 110},
    {'key': 's2', 'init': 'string', 'pos': (30, 490), 'w': 320, 'h': 42,
     'props': {'text in': '', 'font size': '24', 'width': 280}},
    {'key': 'c2', 'comment': True, 'text': 'whole words only: it does not', 'pos': (30, 540)},
]
links = [('s0', 'string out', 'sr', 'string in'), ('sr', 'string out', 's1', ''),
         ('s0', 'string out', 'wr', 'string in'), ('wr', 'string out', 's2', '')]
print(build('string_replace', 'string_replace - find and put something else in its place', body,
            demo, links, demo_width=400, text_width=790, text_height=640))

# ---------------------------------------------------------------- unescape_text
body = """unescape_text turns HTML escapes back into the characters they stand for.

Text out of a web page, a feed or some network services arrives with
characters written as codes: "&amp;" for an ampersand, "&lt;" and "&gt;" for
angle brackets, "&quot;" for a double quote, "&eacute;" for an accented e,
"&#8212;" or "&#x2014;" for a long dash. Until they are turned back, the string
looks wrong when shown and fails to match words it should match.

Every named and numbered escape HTML knows is handled. Anything else is left
as it is.

ONE TO WATCH:
"&nbsp;" becomes a non-breaking space. It looks like a space but is not one,
so splitting on spaces will not split there - and printable, which keeps only
plain characters, removes it entirely, joining the words on either side.

WHAT IT TAKES:
Anything; it is turned into a string first. A list of words is joined with
spaces, and line breaks are removed.

SYNTAX:
unescape_text

INPUTS:

escaped string in:
The text. Receiving it triggers the node.

OUTPUTS:

unescaped string out:
The same text with the escapes replaced.

RELATED:
printable removes characters rather than translating them.
string_replace for any other substitution."""

demo = [
    {'key': 's0', 'init': 'string', 'pos': (30, 62), 'w': 440, 'h': 42,
     'props': {'text in': 'fish &amp; chips &lt;b&gt; caf&eacute; &#8212; &quot;hi&quot;',
               'font size': '24', 'width': 400}},
    {'key': 'c0', 'comment': True, 'text': 'click to send it', 'pos': (30, 112)},
    {'key': 'ue', 'init': 'unescape_text', 'pos': (30, 150), 'w': 200, 'h': 80},
    {'key': 's1', 'init': 'string', 'pos': (30, 245), 'w': 440, 'h': 42,
     'props': {'text in': '', 'font size': '24', 'width': 400}},
    {'key': 'c1', 'comment': True, 'text': 'each code becomes the character it names',
     'pos': (30, 295)},
]
links = [('s0', 'string out', 'ue', 'escaped string in'),
         ('ue', 'unescaped string out', 's1', '')]
print(build('unescape_text', 'unescape_text - HTML escapes back to characters', body,
            demo, links, demo_width=500, text_width=790, text_height=560))

# -------------------------------------------------------------------- printable
body = """printable keeps only plain characters and drops everything else.

"Plain" here means the basic English keyboard set: the letters a to z and A to
Z, the digits, ordinary punctuation, spaces, tabs and carriage returns.
Everything outside that set is removed - other control characters and
invisible formatting, but also every
accented letter, curly quotes, long dashes, symbols, emoji and every
non-Latin script.

WHAT IT IS FOR:
Text that arrived from somewhere else - off a network, out of a file, from a
speech service - and has things in it that are not really part of the text.
Those are the strings that look correct and behave oddly: a comparison that
fails, a width that is wrong, a word that will not match.

WHAT IT COSTS:
It does not translate anything, it deletes. "café" becomes "caf", a curly
quoted word loses its quotes, and text in another language can vanish
completely. Use it where plain English is what you expect; anywhere else it
destroys the text.

A non-breaking space is removed too, not turned into a space, so the words on
either side run together.

WHAT IT TAKES:
Anything; it is turned into a string first. A list of words is joined with
spaces, a float is written with three decimals, and line breaks are removed. If nothing
is left, nothing is sent.

SYNTAX:
printable

INPUTS:

characters in:
The text. Receiving it triggers the node.

OUTPUTS:

printable characters out:
What survives.

RELATED:
unescape_text turns HTML codes into characters - run it first, since printable
would leave the codes behind as plain text.
string_replace to take out particular characters while keeping others."""

demo = [
    {'key': 's0', 'init': 'string', 'pos': (30, 62), 'w': 380, 'h': 42,
     'props': {'text in': 'café “quoted” — plain text',
               'font size': '24', 'width': 340}},
    {'key': 'c0', 'comment': True, 'text': 'click to send it', 'pos': (30, 112)},
    {'key': 'pt', 'init': 'printable', 'pos': (30, 150), 'w': 200, 'h': 80},
    {'key': 's1', 'init': 'string', 'pos': (30, 245), 'w': 380, 'h': 42,
     'props': {'text in': '', 'font size': '24', 'width': 340}},
    {'key': 'c1', 'comment': True, 'text': 'the accent, the curly quotes and the dash\nare gone - deleted, not translated',
     'pos': (30, 295)},
]
links = [('s0', 'string out', 'pt', 'characters in'),
         ('pt', 'printable characters out', 's1', '')]
print(build('printable', 'printable - keep only plain characters', body,
            demo, links, demo_width=440, text_width=790, text_height=640))

# ----------------------------------------------------------------- word_trigger
body = """These watch text going past and fire when particular words, or words
beginning with particular letters, appear in it.

THE NODES:

word_trigger          an outlet per word you name; fires when that word appears
first_letter_trigger  an outlet per letter; fires when a word begins with it

word_trigger IS route FOR LANGUAGE:
Name the words as arguments and each gets an outlet. A stream of text - from
speech recognition, from a chat, from a file - then drives the patch directly:
say a word, something happens. It is the same shape as route or select,
applied to language.

Each message is checked for every word. A word's outlet sends a bang when the
word appears anywhere in the message as a whole word - "faster" does not fire
on "fasterer" - and it fires once however many times the word appears.
Several words in one message each fire.

Capitals are ignored: the text and the arguments are both lowered to small
letters before matching, so "word_trigger Stop" fires on "stop" and on "STOP".

first_letter_trigger LISTENS FOR INITIALS:
Name single letters as arguments. A letter's outlet sends a bang when any word
in the message begins with it - once per message, however many words do.
Capitals are ignored, as for word_trigger.

SYNTAX:
word_trigger <word> <word> ...
first_letter_trigger <letter> <letter> ...

EXAMPLE:
word_trigger stop start faster

INPUTS and PARAMETERS:

string in:
The text to watch. Receiving it triggers the node. A list is read as its
words joined by spaces.

OUTPUTS:

<one outlet per word or letter>:
A bang when that word, or a word beginning with that letter, appears.

RELATED:
word_gate passes only the words in a vocabulary, instead of firing on them.
fuzzy_match corrects a name that was spelled or heard slightly wrong.
select and route do the same job for messages rather than words."""

demo = [
    {'key': 's0', 'init': 'string', 'pos': (30, 62), 'w': 320, 'h': 42,
     'props': {'text in': 'please go faster now', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'click to send the text', 'pos': (30, 112)},
    {'key': 'wt', 'init': 'word_trigger stop faster slower', 'pos': (30, 150),
     'w': 280, 'h': 140},
    {'key': 'c1', 'comment': True, 'text': 'an outlet per word you name',
     'pos': (30, 305)},
    {'key': 'cnt', 'init': 'counter', 'pos': (30, 345), 'w': 123, 'h': 84},
    {'key': 'i1', 'init': 'int', 'pos': (30, 440), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'counts how often faster was said',
     'pos': (30, 490)},
    {'key': 'fl', 'init': 'first_letter_trigger g s', 'pos': (380, 150),
     'w': 240, 'h': 110},
    {'key': 'c3', 'comment': True, 'text': 'an outlet per letter - go\nbegins with g, nothing with s',
     'pos': (380, 270)},
    {'key': 'cnt2', 'init': 'counter', 'pos': (380, 345), 'w': 123, 'h': 84},
    {'key': 'i2', 'init': 'int', 'pos': (380, 440), 'w': 127, 'h': 42, 'props': INT},
]
links = [('s0', 'string out', 'wt', 'string in'),
         ('wt', 'faster', 'cnt', 'input'), ('cnt', 'count out', 'i1', ''),
         ('s0', 'string out', 'fl', 'string in'),
         ('fl', 'g', 'cnt2', 'input'), ('cnt2', 'count out', 'i2', '')]
print(build('word_trigger', 'word_trigger - reacting to what was said', body, demo,
            links, demo_width=660, text_width=800, text_height=640))

# -------------------------------------------------------------------- word_gate
body = """The word_gate node passes only the words that are in its vocabulary,
and drops the rest.

That is how you keep a text stream to a known set of words - useful when the
words are going to drive something that only understands certain ones, and as
a way of ignoring the noise in a recognition stream without having to list
what to reject.

HOW IT READS THE TEXT:
The incoming text is broken into words at anything that is not a letter or an
apostrophe, so numbers and punctuation never get through. Each word is looked
up as it is, then in small letters. A word that is not in the vocabulary on
its own can still pass as the second half of a two-word entry: with "new york"
in the vocabulary, "New York" passes as one item.

The words that pass are sent as one string, joined by spaces. When nothing
passes, nothing is sent.

THE VOCABULARY IS A DICTIONARY:
Each entry is a word and what to send in its place. 'include word' files a
word under itself, so it passes unchanged. A dictionary arriving at
'dictionary in' replaces the vocabulary, and can map a word to something else:
{"crimson": "red"} lets crimson through as red.

SYNTAX:
word_gate

INPUTS and PARAMETERS:

string in:
The text to filter. Receiving it triggers the node.

dictionary in:
A dictionary that replaces the vocabulary.

load dictionary:
The path of a JSON file holding the vocabulary as a dictionary. Entering a
path here, or sending one, loads it and replaces the vocabulary.

include word:
Adds to the vocabulary. A list adds each of its words; a single string is
added whole, so a string node holding "new york" adds the two-word entry.

OUTPUTS:

gated string out:
The words that passed, or what the vocabulary maps them to.

RELATED:
word_trigger fires on particular words rather than passing them along.
string_replace and dict_replace substitute without filtering anything out."""

demo = [
    {'key': 'm1', 'init': 'message', 'pos': (30, 62), 'w': 260, 'h': 42,
     'props': {'text in': 'red green blue', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'click first: adds three words\nto the vocabulary',
     'pos': (330, 62)},
    {'key': 's0', 'init': 'string', 'pos': (30, 125), 'w': 420, 'h': 42,
     'props': {'text in': 'The red car stopped at the green light.', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'then click to send the sentence',
     'pos': (30, 175)},
    {'key': 'wg', 'init': 'word_gate', 'pos': (30, 215), 'w': 220, 'h': 130},
    {'key': 's1', 'init': 'string', 'pos': (30, 370), 'w': 260, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'only the words it knows', 'pos': (30, 420)},
]
links = [('m1', 'message out', 'wg', 'include word'),
         ('s0', 'string out', 'wg', 'string in'),
         ('wg', 'gated string out', 's1', '')]
print(build('word_gate', 'word_gate - pass only known words', body, demo,
            links, demo_width=560, text_width=800, text_height=660))

# ------------------------------------------------------------------ fuzzy_match
body = """The fuzzy_match node corrects a name in a phrase to the closest name in a
list you give it.

It was made for image-generation prompts built from recognised speech: "a
harbour painted by rembrant" should reach the image model as "a harbour painted
by Rembrandt". Recognition gives you names that are close but wrong, and exact
matching against a list misses most of them.

WHERE IT LOOKS FOR THE NAME:
It does not search the whole phrase. It looks for the name after "by " - up to
" of " or " from " if either follows - or, failing that, after "style of ".
"by a" and "by the" are not taken as names, and "by" has to be in small
letters and be a word of its own - "nearby " is not a "by ". When the phrase
has several "by "s, each is tried and the one whose name scores best is used.
With several "style of "s, the last one is used.

A phrase with neither is passed through unchanged. So is a phrase where the
name in front of " of " or " from " is longer than 32 characters or 5 spaces;
no score is sent for that one.

HOW CLOSE IS CLOSE ENOUGH:
Every name in the list is scored against the words found, from 0 to 100,
ignoring capitals; a short name inside a long stretch of words scores lower
than the same name on its own. If the best score is above 'threshold' (60 by
default), the name in the phrase is replaced by the best one from the list.
Otherwise the phrase goes out as it came in.

'score out' reports the best score each time - and reads 20 when nothing
scored higher than that, which includes a phrase with no name to look for.
Watch it while tuning: a threshold set without looking at real scores is a
guess.

THE LIST:
A JSON file holding a list of names. A JSON dictionary works too: its keys are
the names. Give the file as the node's argument, click 'load match file' to
choose one, or send the message "path <file>". With no list loaded nothing is
ever replaced.

SYNTAX:
fuzzy_match <file>

EXAMPLE:
fuzzy_match painters.json

INPUTS and PARAMETERS:

string in:
The phrase. Receiving it triggers the node. A list is read as its words joined
by spaces; anything else is ignored.

load match file:
Click to choose the list's file.

threshold:
The score a match must beat to be used.

path (option):
The list's file. Setting it loads the file.

OUTPUTS:

string out:
The phrase, with the name corrected when a match was good enough.

score out:
The best score found.

replacement out:
The words that were put in, with "by " or "style of " in front -
"by Rembrandt". Sent only when a replacement was made.

RELATED:
word_gate passes only the words in a vocabulary.
string_replace swaps exact text for other text."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click first: writes a small list of\nnames to a file and loads it',
     'pos': (30, 115)},
    # repeat_in_order: fill the dict, save it as a file, then point fuzzy_match at it
    {'key': 'rp', 'init': 'repeat_in_order 3', 'pos': (30, 170), 'w': 190, 'h': 70},
    {'key': 'src', 'init': 'string', 'pos': (30, 255), 'w': 420, 'h': 42,
     'props': {'text in': '{"Rembrandt": 1, "Vermeer": 1, "Turner": 1, "Hokusai": 1}',
               'font size': '24'}},
    {'key': 'sv', 'init': 'message', 'pos': (30, 310), 'w': 420, 'h': 42,
     'props': {'text in': 'save /tmp/fuzzy_match_names.json', 'font size': '24'}},
    {'key': 'dc', 'init': 'dict', 'pos': (30, 370), 'w': 160, 'h': 180},
    {'key': 'pth', 'init': 'message', 'pos': (30, 570), 'w': 420, 'h': 42,
     'props': {'text in': 'path /tmp/fuzzy_match_names.json', 'font size': '24'}},
    {'key': 'p1', 'init': 'string', 'pos': (30, 640), 'w': 420, 'h': 42,
     'props': {'text in': 'a harbour painted by rembrant', 'font size': '24'}},
    {'key': 'p2', 'init': 'string', 'pos': (30, 692), 'w': 420, 'h': 42,
     'props': {'text in': 'a garden in the style of hokusia', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'then click a phrase', 'pos': (30, 742)},
    {'key': 'fm', 'init': 'fuzzy_match', 'pos': (30, 780), 'w': 240, 'h': 180},
    {'key': 's1', 'init': 'string', 'pos': (30, 985), 'w': 420, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'the phrase, name corrected', 'pos': (30, 1035)},
    {'key': 'f1', 'init': 'float', 'pos': (30, 1075), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'how close the best name was', 'pos': (30, 1125)},
    {'key': 's2', 'init': 'string', 'pos': (240, 1075), 'w': 220, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
]
links = [('btn', '', 'rp', ''),
         ('rp', 'first', 'src', ''), ('rp', 'second', 'sv', ''), ('rp', 'third', 'pth', ''),
         ('src', 'string out', 'dc', 'store'),
         ('sv', 'message out', 'dc', 'retrieve by key'),
         ('pth', 'message out', 'fm', 'string in'),
         ('p1', 'string out', 'fm', 'string in'), ('p2', 'string out', 'fm', 'string in'),
         ('fm', 'string out', 's1', ''), ('fm', 'score out', 'f1', ''),
         ('fm', 'replacement out', 's2', '')]
print(build('fuzzy_match', 'fuzzy_match - correct a name to the closest in a list', body,
            demo, links, demo_width=500, text_width=800, text_height=900))

# --------------------------------------------------------------- gather_sentence
body = """The gather_sentence node collects fragments of text until a sentence
ends, then sends the whole sentence.

Speech recognition and language models emit text a fragment at a time.
None of it is a sentence, and most of what you want to do needs one.

WHERE A SENTENCE ENDS:
With 'auto sentence end' on (the default), a fragment that ENDS in . ? ! ; or :
completes the sentence and it is sent, that fragment included. So does a
fragment ending in two dashes, or in two line breaks, and one ending in a
closing bracket when the sentence began with an opening one. An ellipsis (...)
does not end a sentence. The punctuation has to be the fragment's last
character - 'changing. ' with a space after does not count.

'force string end' sends whatever has been collected, whenever the patch
decides - an empty string if nothing has been. A fragment that is exactly
'<backspace>' removes the last fragment collected (or does nothing when nothing
has been).

SPACES:
Fragments are joined exactly as they arrive - no space is added between them.
Recognition output usually carries its own spaces; if yours does not, tick
'enforce spaces', which puts a space between fragments that meet without one -
but not before a full stop, comma or other closing punctuation, nor after an
opening bracket. A '<backspace>' removes the fragment together with the space
added in front of it.

LINE BREAKS:
'end on return' ends the sentence at a line break: what came before it is sent,
and what follows starts the next sentence. Line breaks never appear in what is
sent.

LEAVING THINGS OUT:
'skip framed by' takes a single character and removes everything between a
pair of it - so with '*', 'it is *laughs* cold.' is sent as 'it is  cold.'.
Stage directions or annotations in the stream do not end up in the sentence.

SYNTAX:
gather_sentence

INPUTS and PARAMETERS:

string in:
The fragments.

force string end:
Send what has been collected now.

enforce spaces / auto sentence end / end on return / skip framed by:
As above.

OUTPUTS:

sentences out:
The assembled sentence.

RELATED:
string_builder collects until it is asked, rather than until a sentence ends.
fifo_string keeps a rolling window of recent sentences.
text_change reports only the words in a stream that are new."""

demo = [
    {'key': 'f1', 'init': 'string', 'pos': (30, 62), 'w': 260, 'h': 42,
     'props': {'text in': 'the light ', 'font size': '24', 'width': 200}},
    {'key': 'c0', 'comment': True, 'text': 'click these in turn - fragments, as\nrecognition sends them. The first\nends in a space',
     'pos': (420, 62)},
    {'key': 'f2', 'init': 'string', 'pos': (30, 115), 'w': 260, 'h': 42,
     'props': {'text in': 'is changing.', 'font size': '24', 'width': 200}},
    {'key': 'gs', 'init': 'gather_sentence', 'pos': (30, 180), 'w': 260, 'h': 160},
    {'key': 'c1', 'comment': True, 'text': 'held until a fragment ends a sentence',
     'pos': (420, 180)},
    {'key': 's1', 'init': 'string', 'pos': (30, 370), 'w': 360, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
]
links = [('f1', 'string out', 'gs', 'string in'),
         ('f2', 'string out', 'gs', 'string in'),
         ('gs', 'sentences out', 's1', '')]
print(build('gather_sentence', 'gather_sentence - collect text until a sentence ends', body,
            demo, links, demo_width=800, text_width=800, text_height=780))

# ---------------------------------------------------------------- string_builder
body = """The string_builder node collects text until you ask for it.

Every message to 'string in' is added on, with a space between pieces when
neither side already has one. Anything arriving at 'issue text' - a bang, a
click on its button - sends the collected text and starts again.

Where gather_sentence decides for itself when a sentence is finished,
string_builder leaves that to the patch: a button, a pause detector, the end
of a recognition turn.

SYNTAX:
string_builder

INPUTS and PARAMETERS:

issue text:
Send what has been collected, and start again. The word 'clear' here empties
it without sending.

string in:
The text to add.

OUTPUTS:

text out:
The collected text. Asking with nothing collected sends an empty string.

RELATED:
gather_sentence collects until a sentence ends.
combine joins several inlets into one string at once."""

demo = [
    {'key': 'f1', 'init': 'string', 'pos': (30, 62), 'w': 260, 'h': 42,
     'props': {'text in': 'the light', 'font size': '24', 'width': 200}},
    {'key': 'f2', 'init': 'string', 'pos': (30, 115), 'w': 260, 'h': 42,
     'props': {'text in': 'is changing', 'font size': '24', 'width': 200}},
    {'key': 'c0', 'comment': True, 'text': 'click these, in any order and as\noften as you like',
     'pos': (330, 62)},
    {'key': 'btn', 'init': 'button', 'pos': (30, 180), 'w': 88, 'h': 46},
    {'key': 'c1', 'comment': True, 'text': 'then click this to send it all',
     'pos': (330, 180)},
    {'key': 'sb', 'init': 'string_builder', 'pos': (30, 245), 'w': 220, 'h': 100},
    {'key': 's2', 'init': 'string', 'pos': (30, 370), 'w': 360, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'the pieces, with spaces between',
     'pos': (30, 420)},
]
links = [('f1', 'string out', 'sb', 'string in'),
         ('f2', 'string out', 'sb', 'string in'),
         ('btn', '', 'sb', 'issue text'),
         ('sb', 'text out', 's2', '')]
print(build('string_builder', 'string_builder - collect text until asked', body,
            demo, links, demo_width=640, text_width=760, text_height=560))

# ------------------------------------------------------------------- fifo_string
body = """The fifo_string node keeps a rolling window of the most recent phrases,
older ones fading.

It holds as many phrases as its argument says, 4 by default, and pushes the
oldest out as new ones arrive. That is what you want when feeding something
that should be influenced more by what was just said than by what was said a
minute ago - an image prompt, a mood, a running state.

HOW A MESSAGE BECOMES PHRASES:
A message to 'in' is split at its full stops into separate phrases (but not
after Mr, Mrs, Dr or St), each ending in a full stop of its own. A message
longer than 'length_threshold' characters is split at commas, colons,
semicolons, question and exclamation marks as well, and each piece again gets a
full stop.

WEIGHTS:
Each phrase carries a weight that starts at one and falls by 'decay_rate' per
second, reaching zero after 100 seconds at the default 0.01. 'weighted out'
sends every phrase with its weight; 'string out' sends the phrases whose weight
is still above zero, as text.

The weights are only brought up to date when something arrives, so the window
does not change while nothing is being said.

TEXT STILL BEING SPOKEN:
'progress' is for the partial text recognition sends while a phrase is still
being spoken: it shows in the newest place, and is replaced there when the
finished phrase arrives at 'in'.

SYNTAX:
fifo_string <count: int>

EXAMPLE:
fifo_string 6

INPUTS and PARAMETERS:

in:
Finished phrases.

progress:
Text still being spoken, held in the newest place.

clear:
Anything arriving here empties the window, and an empty window is sent.

dump_oldest:
Anything arriving here drops the oldest phrase and sends the window.

order:
newest_at_end (the default) or newest_at_start.

decay_rate / length_threshold:
How fast the weights fall, per second, and how long a message must be before it
is split at every kind of punctuation.

OUTPUTS:

weighted out:
A list of phrase-and-weight pairs.

string out:
The phrases still above zero weight, as plain text, each followed by a space.

RELATED:
prompt_composer turns fifo_string's weighted phrases into an image prompt.
gather_sentence assembles fragments into sentences to feed this node."""

demo = [
    {'key': 'p1', 'init': 'string', 'pos': (30, 62), 'w': 320, 'h': 42,
     'props': {'text in': 'The tide is out.', 'font size': '24'}},
    {'key': 'p2', 'init': 'string', 'pos': (30, 115), 'w': 320, 'h': 42,
     'props': {'text in': 'Gulls circle the pier.', 'font size': '24'}},
    {'key': 'p3', 'init': 'string', 'pos': (30, 168), 'w': 520, 'h': 42,
     'props': {'text in': 'Rain is coming. Mr. Hale closes the shutters.', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'click these in turn - the last holds two\nsentences, so the oldest falls out of three',
     'pos': (30, 220)},
    {'key': 'fs', 'init': 'fifo_string 3', 'pos': (30, 275), 'w': 280, 'h': 180},
    {'key': 's3', 'init': 'string', 'pos': (30, 480), 'w': 560, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'the window, as text', 'pos': (30, 530)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 570), 'w': 560, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'each phrase with its weight', 'pos': (30, 620)},
]
links = [('p1', 'string out', 'fs', 'in'), ('p2', 'string out', 'fs', 'in'),
         ('p3', 'string out', 'fs', 'in'),
         ('fs', 'string out', 's3', ''), ('fs', 'weighted out', 'l1', '')]
print(build('fifo_string', 'fifo_string - a rolling window of recent phrases', body,
            demo, links, demo_width=620, text_width=800, text_height=760))

# ------------------------------------------------------------------ text_change
body = """text_change reports only the words in a stream of text that are NEW.

THE NODE:

text_change   pass on only the words not seen recently

Given a stream that keeps restating the same thing - recognition revising a
sentence as it goes, a description of a scene that hardly changes - it sends
only the words that were not there before. That is how you drive something from
speech without it re-firing on every repetition of the same phrase.

HOW IT DECIDES:
Each message is split into words at spaces, lower-cased, and stripped of
punctuation. A word it has not got in memory is new: it is sent, and
remembered.

'persistence' is how many FURTHER messages a new word keeps being sent for. At
the default of 3, a new word appears in the output of the message that brought
it and of the next three, whether or not they contain it - so a word stays
'news' for a short while. At 0 a word is sent once only.

'reset_period' governs forgetting. Once a word has been in memory for more than
that many messages, it is forgotten the first time a message arrives WITHOUT
it. After that, if it comes back, it is new again. A word that keeps being said
is never forgotten. Both counts are in messages, not seconds.

'clear' forgets everything at once.

The output is the reported words joined by spaces, in the order they were first
seen. A message with nothing new sends an empty string.

STRINGS OR LISTS:
Send text as a string - from a string node, or from recognition - or as a list
of words, which is what a message node sends. A list is joined with spaces, so
both are read word by word.

SYNTAX:
text_change

EXAMPLE:
text_change

INPUTS and PARAMETERS:

text input:
The text. Each message is compared with what is remembered.

persistence:
How many further messages a new word keeps being reported for. Default 3.

reset_period:
How many messages a word must have been remembered before an absence makes it
forgettable. Default 10.

clear:
Forget every word.

OUTPUTS:

new words out:
The new words, as one string.

RELATED:
gather_sentence, string_builder and fifo_string assemble a text stream over
time.
word_trigger reacts to particular words."""

demo = [
    {'key': 'a', 'init': 'string', 'pos': (30, 62), 'w': 320, 'h': 42,
     'props': {'text in': 'the light is changing.', 'font size': '24', 'width': 280}},
    {'key': 'c0', 'comment': True, 'text': 'click this one first',
     'pos': (440, 62)},
    {'key': 'b', 'init': 'string', 'pos': (30, 115), 'w': 320, 'h': 42,
     'props': {'text in': 'the light is fading', 'font size': '24', 'width': 280}},
    {'key': 'c1', 'comment': True, 'text': "then this - only 'fading' is new",
     'pos': (440, 115)},
    {'key': 'tc', 'init': 'text_change', 'pos': (30, 180), 'w': 260, 'h': 150,
     'props': {'persistence': 0}},
    {'key': 'c2', 'comment': True, 'text': 'persistence 0 here, so each new word is\nreported once. At the default 3 it is\nreported for three more messages',
     'pos': (440, 180)},
    {'key': 's1', 'init': 'string', 'pos': (30, 360), 'w': 360, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'click either again: nothing is new',
     'pos': (440, 360)},
]
links = [('a', 'string out', 'tc', 'text input'),
         ('b', 'string out', 'tc', 'text input'),
         ('tc', 'new words out', 's1', '')]
print(build('text_change', 'text_change - only what is new', body,
            demo, links, demo_width=780, text_width=800, text_height=700))

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
