"""buffer / rolling_buffer, and the two matrix makers."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

HM = lambda n=8, lo=0.0, hi=1.0, fmt='%.2f': {
    'color': 'viridis', 'width': 200, 'height': 100, 'sample count': n,
    'min y': lo, 'max y': hi, 'update_mode': 'heat_map', 'number format': fmt}
MSG = lambda t: {'text in': t, 'font size': '24'}

STYLE = """
'update style' DECIDES WHAT AN INCOMING VALUE IS:
This is the setting to get right first, because it changes what the buffer even
holds. Sending the list [1, 2, 3] to a buffer of 8, the three choices give three
different things:

buffer holds one sample of input   the buffer BECOMES the input. Shape (3,) -
                                   the sample count is ignored entirely.
input is stream of samples         the three numbers are three successive
                                   samples, written in one after another.
                                   Shape (8,).
input is multi-channel sample      the three numbers are one moment of a
                                   3-channel signal. Shape (8, 3) - eight rows
                                   of three.

The last is the one to use for anything with several streams at once - the
channels of a sound, the joints of a body - because it keeps them aligned in
time instead of interleaving them.
"""

# --------------------------------------------------------------------- buffer
body = """These two keep a history of what has gone past.

THE NODES:

buffer          a fixed block you write into and read back from
rolling_buffer  a window on the recent past, always scrolling

WHICH OF THE TWO:
buffer is a place to PUT things. It has an index inlet, so you can write a
stream in and then ask for sample 40 whenever you like. Use it when the data is
a record you want to address.

rolling_buffer is a place to WATCH things. There is no index; the newest sample
goes in at one end, the oldest falls off the other, and what comes out is always
the last N moments in order. Use it when you want the shape of the recent past -
which is nearly always what you want for a display or an analysis window.
""" + STYLE + """
'output style' ON buffer:
output buffer on every input       the whole buffer comes out each time
                                   something goes in. This is the streaming
                                   arrangement - straight into a plot.
output samples on demand by index  nothing comes out until you ask. Send a
                                   number to 'sample to output' and you get
                                   that one sample.

The second is what makes buffer a lookup table: fill it once, then read it in
any order you like - forwards, backwards, at random, driven by another signal.
That is a wavetable, or a delay line, depending on what you do with the index.

WHAT COMES OUT IS IN ORDER, NOT IN STORAGE ORDER:
Internally the buffer is circular - the write position runs round and wraps. But
what it sends is rotated back into time order, oldest first, so you never have
to think about where the seam is. Wire it to a plot and the picture is stable
rather than lurching each time the write position wraps.

'scroll direction' ON rolling_buffer:
Whether new data arrives at the right and pushes left, or at the bottom and
pushes up. It changes the orientation of the output matrix, so pick whichever
matches how you are drawing it.

SYNTAX:
buffer <sample count>
rolling_buffer <sample count>

EXAMPLE:
buffer 256

INPUTS and PARAMETERS:

input:
The data. What it counts as depends on 'update style'.

sample to output (buffer):
An index. Send one and that sample comes out.

reset (rolling_buffer):
Clear it and start again, keeping the same size.

sample count:
How long the history is. In seconds, that is this divided by the rate things
are arriving - at 60 frames a second, 60 samples is one second.

OUTPUTS: 

output:
The buffer, or one sample of it.

RELATED:
t.buffer and t.rolling_buffer are the same two ideas for torch tensors.
plot and heat_map are the usual destinations."""

demo = [
    {'key': 'sig', 'init': 'signal', 'pos': (30, 62), 'w': 129, 'h': 78,
     'props': SIG('sin', 3.0)},
    {'key': 'c0', 'comment': True, 'text': 'a slow sine to fill them with',
     'pos': (30, 155)},

    {'key': 'buf', 'init': 'buffer 128', 'pos': (30, 200), 'w': 300, 'h': 200,
     'props': {'sample count': 128, 'update style': 'input is stream of samples',
               'output style': 'output buffer on every input'}},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 415), 'w': 300, 'h': 180,
     'props': PLOT(-1.0, 1.0, 128)},
    {'key': 'c1', 'comment': True, 'text': 'the whole buffer on every input -\nand it comes out in time order, so the\npicture does not lurch when it wraps',
     'pos': (30, 605)},

    {'key': 'buf2', 'init': 'buffer 128', 'pos': (400, 200), 'w': 300, 'h': 200,
     'props': {'sample count': 128, 'update style': 'input is stream of samples',
               'output style': 'output samples on demand by index'}},
    {'key': 'ix', 'init': 'signal', 'pos': (400, 62), 'w': 129, 'h': 78,
     'props': SIG('saw', 4.0, 127.0, False)},
    {'key': 'c4', 'comment': True, 'text': 'a ramp of indices reads the buffer',
     'pos': (400, 155)},
    {'key': 'f1', 'init': 'float', 'pos': (400, 415), 'w': 127, 'h': 42,
     'props': FLT},
    {'key': 'c5', 'comment': True, 'text': 'one sample at a time, on demand -\nwhich is what makes it a wavetable',
     'pos': (400, 465)},

    {'key': 'rb', 'init': 'rolling_buffer 64', 'pos': (30, 710), 'w': 300, 'h': 200,
     'props': {'sample count': 64, 'update style': 'input is stream of samples'}},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 925), 'w': 208, 'h': 148,
     'props': HM(64, -1.0, 1.0)},
    {'key': 'c7', 'comment': True, 'text': 'no index, no asking: always the last\n64 moments, oldest to newest',
     'pos': (30, 1085)},
]
links = [('sig', '', 'buf', 'input'), ('buf', 'output', 'pl', 'y'),
         ('sig', '', 'buf2', 'input'),
         ('ix', '', 'buf2', 'sample to output'),
         ('buf2', 'output', 'f1', ''),
         ('sig', '', 'rb', 'input'), ('rb', 'output', 'hm', 'y')]
print(build('buffer', 'buffer and rolling_buffer - keeping a history', body,
            demo, links, demo_width=760, text_width=800, text_height=770))

# ------------------------------------------------------------------------ cwt
body = """Two nodes that turn what you send them into a matrix to look at.

THE NODES:

cwt        a signal, spread out into time against frequency
confusion  two lists, compared item by item

They have nothing to do with each other mathematically. What they share is the
shape of the answer: both take something one-dimensional and give back a grid,
and both are meant for a heat_map, where a pattern you would never find in a
column of numbers is obvious at a glance.

cwt: WHERE THE FREQUENCIES ARE, AND WHEN:
A Fourier transform tells you which frequencies are present in a signal, and
throws away when they happened. A wavelet transform keeps both: the output is a
matrix, frequency down one axis and time along the other, so you can see a pitch
rise, a thump arrive, or a tremor start and stop.

That is the whole reason to reach for it over an FFT. If the question has the
word "when" in it, you want this.

The output is the MAGNITUDE of the coefficients - how much of each frequency was
present at each moment. The phase is discarded, which is what you want for
looking and not what you want for reconstructing.

'octaves' IS REALLY VOICES PER OCTAVE:
It sets how finely the frequency axis is sampled, not how many octaves are
covered. Higher gives more rows and a smoother picture, at proportionally more
work. Measured on a 512-sample input:

octaves 2   ->  17 rows by 512 columns
octaves 8   ->  65 rows by 512 columns

The columns always match the input length, so the time axis is simply the signal
you sent. Feed it from a buffer or rolling_buffer to give it a window to work on
- a single sample has no time in it to analyse.

'wavelet' chooses the shape being matched against the signal. They trade
sharpness in time against sharpness in frequency, and you cannot have both -
that is a property of the world, not of the software. morlet is the familiar
general-purpose choice; the others differ in where they sit on that trade.

confusion: WHERE TWO LISTS AGREE:
It takes two lists and marks every place an item in one EXACTLY equals an item
in the other - one row per item in the second list, one column per item in the
first, 1.0 where they match and 0 everywhere else.

Sending ['apple','pear','engine'] and ['fruit','apple','engine','apple']:

              fruit  apple  engine  apple
    apple       0      1       0      1
    pear        0      0       0      0
    engine      0      0       1      0

Note 'pear' is an empty row: it appears in neither. That is the useful part -
the gaps show you what one list has that the other does not, and repeats show up
as repeated marks along a row.

IT IS EQUALITY, NOT SIMILARITY:
Worth being clear about, because there is a node that looks almost identical and
does something quite different. This one asks "are these the same thing"; it has
no notion of two things being nearly alike, and 'apple' against 'apples' scores
zero.

spacy_confusion has the same shape of output and compares MEANING, so 'apple'
against 'fruit' scores well there and nothing at all here. Use this one for
tokens, labels and categories; use that one for words.

SYNTAX:
cwt
confusion

EXAMPLE:
cwt

INPUTS and PARAMETERS:

input (cwt):
A window of signal. Receiving it does the transform.

octaves / wavelet:
How finely to sample frequency, and which wavelet shape to use.

input / input2 (confusion):
The two lists. input2 becomes the rows, input the columns.

OUTPUTS: 

output (cwt):
A matrix, frequency by time.

output (confusion):
A matrix of ones and zeros, rows by columns.

RELATED:
rolling_buffer is the usual way to give cwt a window.
spacy_confusion for meaning rather than equality.
heat_map is where either of these wants to end up."""

demo = [
    {'key': 'sig', 'init': 'signal', 'pos': (30, 62), 'w': 129, 'h': 78,
     'props': SIG('sin', 2.0)},
    {'key': 'rb', 'init': 'rolling_buffer 256', 'pos': (30, 155), 'w': 300, 'h': 200,
     'props': {'sample count': 256, 'update style': 'input is stream of samples'}},
    {'key': 'c0', 'comment': True, 'text': 'cwt needs a WINDOW - a single sample\nhas no time in it to analyse',
     'pos': (30, 370)},
    {'key': 'cw', 'init': 'cwt', 'pos': (30, 445), 'w': 240, 'h': 160,
     'props': {'octaves': 8, 'wavelet': 'morlet'}},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 620), 'w': 208, 'h': 148,
     'props': HM(256, 0.0, 0.5)},
    {'key': 'c2', 'comment': True, 'text': 'frequency down, time across - an FFT\nwould tell you which frequencies but\nnot when they happened',
     'pos': (30, 780)},

    {'key': 'm1', 'init': 'message', 'pos': (400, 62), 'w': 300, 'h': 42,
     'props': MSG('apple pear engine')},
    {'key': 'm2', 'init': 'message', 'pos': (400, 120), 'w': 360, 'h': 42,
     'props': MSG('fruit apple engine apple')},
    {'key': 'cf', 'init': 'confusion', 'pos': (400, 180), 'w': 220, 'h': 110},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (400, 305), 'w': 208, 'h': 148,
     'props': HM(4, 0.0, 1.0, '%.0f')},
    {'key': 'c5', 'comment': True, 'text': 'EXACT matches only - apple hits two\ncolumns, pear is an empty row\nspacy_confusion looks the same and\ncompares meaning instead',
     'pos': (400, 465)},
]
links = [('sig', '', 'rb', 'input'), ('rb', 'output', 'cw', 'input'),
         ('cw', 'output', 'hm', 'y'),
         ('m1', 'message out', 'cf', 'input2'),
         ('m2', 'message out', 'cf', 'input'),
         ('cf', 'output', 'hm2', 'y')]
print(build('cwt', 'cwt and confusion - making a matrix to look at', body,
            demo, links, demo_width=800, text_width=800, text_height=770))
