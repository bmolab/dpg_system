"""Translating text with Google."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """These translate text between languages, using Google.

THE NODES:

translate      the free route. No account, no key, works immediately
translate_api  the official Cloud API. Needs credentials, and is reliable

THEY ARE NOT TWO WAYS OF DOING THE SAME THING:
The difference matters more than the shared name suggests, and it decides which
one belongs in a piece.

translate asks the ordinary Google Translate web page for a translation, the way
a browser would. Nothing to sign up for, nothing to configure - drop it in and
it works. Measured here: a sentence of English into French came back in about
0.7 seconds, with no account of any kind.

translate_api uses Google's paid Cloud Translation service. It needs a Google
Cloud project, the SDK installed, and GOOGLE_APPLICATION_CREDENTIALS pointing at
a key file. In exchange it is supported, rate-limited generously, and will not
change under you.

IF translate_api IS NOT THERE, THAT IS WHY:
The node only appears when the Google Cloud SDK is installed. Without it,
'translate' still works and 'translate_api' is simply not registered - the
console says so when the patch starts.

If you have the SDK but the node reports no credentials, that is the environment
variable, not the node.

WHAT 'FREE' COSTS YOU:
translate is unofficial. It is not a service Google offers you, it is a page
Google serves to browsers, so:

It can stop working without warning, if the page changes.
It will refuse you if you ask too often - it is not built for a stream of
requests, and there is no quota to raise.
Text is limited to 5000 characters at a time.

So: excellent for making a piece, for rehearsal, and for anything where an
occasional failure is survivable. Not the thing to build an installation on that
has to run unattended for three months. Use translate while you work and swap in
translate_api if it matters.

YOUR TEXT GOES TO GOOGLE, EITHER WAY:
Both send what you give them over the network to be translated. That is
obvious once said, and worth saying: anything private, anything an audience
said in confidence, anything you would not put in a search box, should not go
through these.

'use queue' - BECAUSE A TRANSLATION TAKES TIME:
A round trip is most of a second, which is an eternity in a patch running at
frame rate. With the queue on, requests are handled one after another in the
background and the patch never waits. With it off, a request that arrives while
another is in flight is simply lost.

Leave it on unless you specifically want only the most recent request to matter.

'time out' is how long to wait before giving up on one, in seconds. Five is
sensible; raise it on a slow connection rather than lowering it to make things
feel responsive, because a timeout does not make the answer arrive sooner - it
just throws it away.

LANGUAGES ARE PICKED BY NAME:
'source language' and 'dest language' are lists of language names rather than
codes, so you choose English and French rather than en and fr. Setting the
source to automatic detection is the usual choice when the text is arriving from
speech and you do not know what is coming.

SYNTAX:
translate <source> <dest>
translate_api <source> <dest>

EXAMPLE:
translate English French

INPUTS and PARAMETERS:

text in:
The text. Receiving it starts a translation.

source language / dest language:
From and to, by name.

use queue:
Handle requests one after another in the background. Leave on.

time out:
Seconds to wait before abandoning a request.

OUTPUTS: 

translation out:
The translated text, when it arrives - which is not on the same frame it went
out.

RELATED:
The whisper nodes to get text out of speech in the first place.
text_display or cairo_layout to show the result.
gemma_4 will also translate, locally and without sending anything anywhere,
though less reliably for the languages it saw little of."""

demo = [
    {'key': 's1', 'init': 'string', 'pos': (30, 62), 'w': 520, 'h': 42,
     'props': {'text in': 'the room was colder than it had been',
               'font size': '24', 'width': 480}},
    {'key': 'c0', 'comment': True, 'text': 'click to translate', 'pos': (30, 112)},

    {'key': 'tr', 'init': 'translate', 'pos': (30, 160), 'w': 320, 'h': 260},
    {'key': 'c1', 'comment': True, 'text': 'no account, no key - it asks the same\npage a browser would. About 0.7s',
     'pos': (30, 435)},

    {'key': 'td', 'init': 'text_display', 'pos': (400, 160), 'w': 340, 'h': 200,
     'props': {'width': 320, 'height': 160, 'wrap': True, 'max_lines': 50,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'the answer arrives LATER, not on the\nsame frame it went out - drive what\ncomes next from this, not from the send',
     'pos': (400, 375)},

    {'key': 'c6', 'comment': True, 'text': 'leave use queue ON: with it off, a\nrequest arriving while another is in\nflight is simply lost\ntranslate_api is the paid, supported\nroute. It only appears if the Google\nCloud SDK is installed - and then needs\nGOOGLE_APPLICATION_CREDENTIALS set\nBOTH send your text to Google. Nothing\nprivate should go through either',
     'pos': (30, 510)},
]
links = [('s1', 'string out', 'tr', 'text in'),
         ('tr', 'translation out', 'td', '###text in')]
print(build('translate', 'translate - free and official routes', body,
            demo, links, demo_width=780, text_width=800, text_height=740))
