"""Seed the help pages for the fifth batch of numpy nodes (2026-10-24): one page per node.

    python3 dpg_system/help/tools/seed_numpy_batch5_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

np.save, np.savetxt, np.loadtxt, np.random.normal, np.random.integers,
np.random.choice, np.random.permutation, np.sliding_window_view,
np.maximum.accumulate, np.minimum.accumulate, np.insert, np.delete, np.append,
np.nanargmax, np.nanargmin, np.bincount.

The np.save and np.savetxt demos leave the path empty and wire no button node to
"write": smoke.py --click bangs every button node, and a write without a path opens
a native save dialog that would stall the run. np.loadtxt reads
help/data/np_loadtxt_example.csv.

Like the earlier seeders it is not a make_* script, so rebuild.py never runs it,
and it will not write over an existing page without --force. Treat the pages as
hand-made once written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from seed_numpy_more_help import LIST, MSG, HM, BARS, C, N, see_also
from seed_numpy_batch2_help import METRO, METRO_SLOW, metro_row


PAGES = {}


def page(stem, title, body, demo, links, also, also_y, text_height=820):
    PAGES[stem] = dict(title=title, body=body.strip('\n'), demo=demo + see_also(also_y, also),
                       links=links, text_height=text_height)


SEED_TEXT = """seed:
An option, under the node. -1 (the default) draws different numbers every time the
patch opens. Any other number starts the same sequence every time - the same
"random" piece on every run, which is what you want for rehearsal, comparison and
debugging."""


# -------------------------------------------------------------- save / savetxt
SAVE = {
    'np.save': ('write an array to a numpy file', '.npy', """
Writes an array to a file in numpy's own format, exactly as it is: every value, the
shape, the number type. np.load reads it back.

A .npy file holds one array. A path ending in .npz writes the numpy archive format
instead, with the array under the name data - the format np.load and most motion-
capture archives use. Without an extension, .npy is added.""",
                """It is the way to keep something a patch has computed - a recorded take, a
calibration, a fitted model, a set of poses - and use it again later or elsewhere.""",
                'np.load', 'np.savetxt'),
    'np.savetxt': ('write an array as a text table', '.csv', """
Writes a list or 2-D array to a text file, one row per line, values separated by
commas (or tabs or spaces). A spreadsheet, a text editor, R, Excel or any other
program can open it; np.loadtxt reads it back.

Text is easy to share and to read by eye, but it is larger than np.save's format,
keeps only 1-D and 2-D arrays, and writes each value to 9 significant digits. Without
an extension, .csv is added.""",
                   """Use it to hand measurements to other software, or to look at numbers a patch has
produced.""",
                   'np.loadtxt', 'np.save'),
}

for stem, (gist, ext, intro, use, reader, other) in SAVE.items():
    text = stem == 'np.savetxt'
    page(stem, f'{stem} - {gist}',
f"""{intro.strip()}

{use}

WRITING ONLY WHEN ASKED:
An array arriving at in is only held, not written: a stream of arrays does not
rewrite the file sixty times a second. The file is written when write is clicked (or
sent a bang).

SYNTAX:
{stem} <path>
Default: no path - the first write asks where to save.

EXAMPLE:
{stem} recordings/take_1{ext}

INPUTS and PARAMETERS:

in:
The array to save (a list is turned into one). Held until written; each new one
replaces the last.

write (button, also an inlet):
Writes the held array to the path. With no path yet, it asks where, as save as does.

save as (button, also an inlet):
Asks for a file name and folder, writes there, and keeps that path for the next
write.

path:
An option, under the node: where write saves. A path that is not absolute is
relative to the folder the app was started from. It is saved with the patch.
""" + ("""
delimiter:
An option, under the node: comma (the default), tab or space between the values on
a line.
""" if text else '') + f"""
OUTPUTS:

saved path:
The full path of the file, sent each time it is written - ready to pass on, or to
show where the file went.

READING THE DEMO:
{'A 2 x 3 table' if text else 'Four values'} arrive and are held. Click write (on the node) and choose where to save; the
path appears below. Nothing is saved until you do, and nothing is saved in the help
folder.

RELATED:
{reader} reads the file back. {other} writes the other format.
""",
    [
        C('c1', 'once at load', 28, 53),
        N('lb', 'load_bang', 28, 92, 88, 46),
        C('c2', 'a 2 x 3 table' if text else 'four values', 140, 53),
        N('m1', 'message', 140, 92, 180, 42, MSG('1 2 3 4 5 6' if text else '1.5 2 3 4')),
    ] + ([N('rs', 'np.reshape 2 3', 140, 165, 150, 90)] if text else []) + [
        C('c3', 'click write, then choose where', 380, 53),
        N('sv', stem, 380, 92, 170, 110),
        C('c4', 'where it went', 380, 250),
        N('lp', 'list', 380, 285, 420, 42, LIST),
    ],
    ([('lb', 'out', 'm1', ''), ('m1', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'sv', 'in')]
     if text else [('lb', 'out', 'm1', ''), ('m1', 'message out', 'sv', 'in')]) +
    [('sv', 'saved path', 'lp', '')],
    [reader, other, 'np.load' if text else 'np.loadtxt'], 420, 940)


# ---------------------------------------------------------------- loadtxt
page('np.loadtxt', 'np.loadtxt - read a text table into an array',
"""
Reads a text file of numbers - one row per line, values separated by commas, tabs or
spaces - into a 2-D array. CSV files from a spreadsheet, a data logger or another
program; files written by np.savetxt.

A cell that is empty or is not a number becomes nan, rather than stopping the
whole load: np.isnan finds those, np.nan_to_num fills them, and the nan- nodes skip
them. (It reads with numpy's genfromtxt, which is the forgiving reader.)

LOADING:
Click load and choose a file; it is read and sent at once. send sends the same array
again without re-reading. A path in the path option, or sent as the message "path
<file>", is read straight away - from a patch, with no dialog - and is read again
when the patch opens.

SYNTAX:
np.loadtxt <path>

INPUTS and PARAMETERS:

load (button, also an inlet):
Asks for a file, then reads and sends it.

send (button, also an inlet):
Sends the array again.

path:
An option, under the node: the file to read. A path that is not absolute is relative
to the folder the app was started from, which is why the demo's path starts with
dpg_system/help.

delimiter:
An option, under the node: comma (the default), tab, or any space (any run of spaces
and tabs).

skip rows:
An option, under the node: how many lines to skip at the top - 1 for a header line of
column names.

OUTPUTS:

array:
The table: one row per line. A single column, or a single line, comes out as a 1-D
list.

READING THE DEMO:
At load the message sets the path to a small table in the help folder: a header line
(speed,height), then six rows. skip rows is 1, so the header is left out. The fourth
row has no height, which comes out as nan.

RELATED:
np.savetxt writes text tables. np.load reads numpy's own files; text_file reads any
text line by line.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', 'set the path and read', 140, 53),
    N('mp', 'message', 140, 92, 480, 42, MSG('path dpg_system/help/data/np_loadtxt_example.csv')),
    C('c3', 'a header line, then 6 rows', 140, 170),
    N('lt', 'np.loadtxt', 140, 205, 150, 110, {'skip rows': 1}),
    C('c4', 'speed and height: one gap', 340, 170),
    N('hm', 'heat_map', 340, 205, 168, 250, HM(150, 210, 1, 0.0, 1.6, '%.2f')),
],
[('lb', 'out', 'mp', ''), ('mp', 'message out', 'lt', 'load'), ('lt', 'array', 'hm', 'y')],
['np.savetxt', 'np.load', 'np.isnan', 'text_file'], 520, 960)


# ----------------------------------------------------------- random normal
page('np.random.normal', 'np.random.normal - random values on a bell curve',
"""
Makes an array of random values from the normal distribution - the bell curve. Most
values land near the mean; fewer and fewer further out. About two thirds fall within
one spread of the mean, and 95 in a hundred within two.

It is the shape of natural variation: measurement noise, small human imprecision,
the scatter of many small independent causes added together. Use it to add
believable jitter to a movement, a pitch or a timing; to simulate sensor noise for
testing a filter; or to scatter points around a centre.

np.rand gives evenly spread values instead, every value in its range equally likely.

SYNTAX:
np.random.normal <dims ...>
Default: 10 values.

EXAMPLES:
np.random.normal 100
np.random.normal 24 3

INPUTS and PARAMETERS:

(button):
Makes a new array and sends it.

shape (text box, also an inlet):
The size along each axis: 100 for a list of 100, 24 3 for 24 rows of 3.

mean (float box, also an inlet):
The centre of the curve.

spread (float box, also an inlet):
The standard deviation: how wide the curve is. 0 gives the mean every time.

""" + SEED_TEXT + """

OUTPUTS:

random array:
The values (32-bit floats).

READING THE DEMO:
2000 values five times a second, counted into 24 bins from -3 to 3: the bell shape,
tallest at 0, nearly empty beyond 2 either way. Change spread to 0.5 and it narrows;
change mean and it slides along.

RELATED:
np.rand gives evenly spread values. np.random.integers gives whole numbers.
random.gauss gives a single value; the t.dist nodes give PyTorch tensors from many
distributions.
""",
metro_row() + [
    C('c1', '2000 values on a bell curve', 180, 53),
    N('rn', 'np.random.normal 2000', 180, 92, 200, 150),
    C('c2', 'count them in 24 bins', 420, 53),
    N('hg', 'np.histogram 24 -3 3', 420, 92, 190, 170),
    C('c3', 'the bell shape', 28, 300),
    N('pl', 'plot', 28, 335, 528, 250, BARS(24, 250.0)),
],
[('mt', '', 'rn', ''), ('rn', 'random array', 'hg', 'in'), ('hg', 'counts', 'pl', 'y')],
['np.rand', 'np.random.integers', 'random.gauss', 't.dist.normal'], 640, 1100)


page('np.random.integers', 'np.random.integers - random whole numbers',
"""
Makes an array of random whole numbers from low up to, but not including, high -
each one equally likely. low 1 and high 7 rolls dice.

Use it to pick random indices (which joint, which sample, which preset), random
steps of a scale, random MIDI notes in a range, random delays in whole frames.

SYNTAX:
np.random.integers <dims ...>
Default: 10 values.

EXAMPLE:
np.random.integers 5

INPUTS and PARAMETERS:

(button):
Makes a new array and sends it.

shape (text box, also an inlet):
The size along each axis.

low (int box, also an inlet):
The smallest possible value.

high (int box, also an inlet):
One more than the largest possible value: high itself never comes up. It must be above
low.

""" + SEED_TEXT + """

OUTPUTS:

random array:
The whole numbers (64-bit integers).

READING THE DEMO:
Five dice, low 1 and high 7, rolled twice a second. Each is 1 to 6, never 7.

RELATED:
np.random.choice picks from any list, with weights. np.rand and np.random.normal give
decimals. random gives a single random number.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', 'five dice: 1 to 6', 180, 53),
    N('ri', 'np.random.integers 5', 180, 92, 200, 150, {'low': 1, 'high': 7}),
    C('c2', 'rolled', 420, 53),
    N('ls', 'list', 420, 92, 180, 42, LIST),
],
[('mt', '', 'ri', ''), ('ri', 'random array', 'ls', '')],
['np.random.choice', 'np.rand', 'np.random.normal', 'random'], 420, 900)


page('np.random.choice', 'np.random.choice - pick at random from a list',
"""
Picks at random from a list - numbers, words, rows of a matrix - one at a time or
several at once, every item equally likely or weighted.

Use it to choose the next note of a scale, a word to speak, a pose to move to, a
clip to play, a direction to turn; with weights, to make some choices common and
others rare; without repeats, to deal from a deck - every item once before any comes
again.

SYNTAX:
np.random.choice <n>
Default: nothing to choose from until a list arrives at from. With n, it chooses from
0 to n-1.

EXAMPLE:
np.random.choice 12

INPUTS and PARAMETERS:

(button):
Makes a choice and sends it.

from:
The list (or array) to choose from, held. For a 2-D array, whole rows are chosen.

count (int box, also an inlet):
How many to pick. 1 (the default) sends the item itself; more sends them as a list.

weights:
How likely each item is, held: one non-negative number per item. They need not add
up to 1; 3 1 makes the first item three times as likely as the second. Without
weights all are equally likely.

repeats:
An option, under the node. On (the default), an item can be picked again within one
count. Off, each item at most once per count - like dealing cards (count is then at
most the number of items).

""" + SEED_TEXT + """

OUTPUTS:

chosen:
The pick, or the list of picks.

READING THE DEMO:
Top: a word from do re mi fa sol, twice a second, each equally likely.

Bottom: 16 picks from 0 and 1, weighted 9 to 1: mostly 0s, a 1 now and then - about
one in ten.

RELATED:
np.random.permutation shuffles a whole list. np.random.integers picks whole numbers
in a range. np.take picks by position.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 210, 88, 46),
    C('c2', 'the words, held', 180, 175),
    N('mw', 'message', 180, 210, 200, 42, MSG('do re mi fa sol')),
    C('c3', 'one at a time', 420, 53),
    N('c1n', 'np.random.choice', 420, 92, 180, 150),
    N('l1', 'list', 640, 92, 100, 42, LIST),
    C('c4', 'two choices, weighted 9 to 1', 180, 300),
    N('mf', 'message', 180, 335, 80, 42, MSG('0 1')),
    N('mg', 'message', 280, 335, 80, 42, MSG('9 1')),
    C('c5', '16 picks: mostly 0', 420, 300),
    N('c2n', 'np.random.choice', 420, 335, 180, 150, {'count': 16}),
    N('l2', 'list', 420, 510, 380, 42, LIST),
],
[('lb', 'out', 'mw', ''), ('lb', 'out', 'mf', ''), ('lb', 'out', 'mg', ''),
 ('mw', 'message out', 'c1n', 'from'), ('mf', 'message out', 'c2n', 'from'),
 ('mg', 'message out', 'c2n', 'weights'),
 ('mt', '', 'c1n', ''), ('mt', '', 'c2n', ''), ('c1n', 'chosen', 'l1', ''), ('c2n', 'chosen', 'l2', '')],
['np.random.permutation', 'np.random.integers', 'np.take', 'random'], 640, 1100)


page('np.random.permutation', 'np.random.permutation - shuffle',
"""
Shuffles a list into a random order: every item kept, each exactly once, in a new
order every time. Given a whole number n instead, it shuffles 0 to n-1.

Use it for a random running order of scenes, a play order of clips that repeats none
until all have played, a random assignment of performers to parts, or a shuffled set
of positions to take values from (with np.take).

SYNTAX:
np.random.permutation

INPUTS and PARAMETERS:

in:
A list or array to shuffle - for a 2-D array, whole rows are shuffled - or a single
whole number n, for the numbers 0 to n-1 shuffled (a message box sends 8 as a list of
one; that counts as n too). Every input is shuffled and sent.

""" + SEED_TEXT + """

OUTPUTS:

shuffled:
The same items in a random order.

READING THE DEMO:
Top: the words do re mi fa sol in a new order every second. Bottom: 8, which makes the
numbers 0 to 7 in a random order - a set of positions for np.take.

RELATED:
np.random.choice picks some at random, possibly with repeats. np.sort and np.argsort
put things back in order. np.take reorders by a list of positions.
""",
metro_row(label='every second', props={'on': True, 'period': 1000.0, 'units': 'milliseconds'},
          init='metro 1000') + [
    C('c1', 'five words', 180, 53),
    N('m1', 'message', 180, 92, 200, 42, MSG('do re mi fa sol')),
    C('c2', 'shuffled', 420, 53),
    N('p1', 'np.random.permutation', 420, 92, 200, 70),
    N('l1', 'list', 420, 180, 300, 42, LIST),
    C('c3', 'a whole number: 0 to 7 shuffled', 180, 270),
    N('m2', 'message', 180, 305, 60, 42, MSG('8')),
    N('p2', 'np.random.permutation', 420, 305, 200, 70),
    N('l2', 'list', 420, 390, 300, 42, LIST),
],
[('mt', '', 'm1', ''), ('mt', '', 'm2', ''), ('m1', 'message out', 'p1', 'in'),
 ('m2', 'message out', 'p2', 'in'), ('p1', 'shuffled', 'l1', ''), ('p2', 'shuffled', 'l2', '')],
['np.random.choice', 'np.take', 'np.argsort', 'np.sort'], 490, 900)


# ----------------------------------------------------------- sliding windows
page('np.sliding_window_view', 'np.sliding_window_view - every run of neighbouring values',
"""
Cuts a signal into overlapping windows - every run of window neighbouring values, one
after another, each starting one step on - and stacks them as the rows of an array.

That turns any node that measures a whole array into a rolling one. np.mean of the
windows is a rolling mean; np.ptp, the range within each moment; np.std, how
jittery each stretch is; np.fft.rfft, a spectrogram. A whole recording is analysed
window by window in one step, with every result lined up in a row.

SYNTAX:
np.sliding_window_view <window: int> <step: int>
Defaults: window 4, step 1.

EXAMPLE:
np.sliding_window_view 30 10

INPUTS and PARAMETERS:

in:
The signal (a list or array). Every input is cut and sent.

window (int box, also an inlet):
How many values in each window. It cannot be longer than the signal.

step (int box, also an inlet):
How far each window starts after the last: 1 (the default) gives every window, 10 every
tenth - fewer windows, overlapping less.

axis (int box, also an inlet):
Which way the signal runs: -1 (the default) along each row, 0 down each column - for a
buffer of frames x joints, axis 0 windows the frames, keeping the joints.

OUTPUTS:

windows:
One row per window. For a list of n values and window w there are n - w + 1 windows
(fewer with a larger step); for more axes, the window becomes a new last axis.

READING THE DEMO:
1 to 6 in windows of 3: 1 2 3, 2 3 4, 3 4 5 and 4 5 6. np.mean along axis 1 - across each
row - gives a rolling mean: 2 3 4 5. A metro re-sends the list twice a second.

RELATED:
np.rolling_buffer keeps the latest values of a stream. np.convolve with equal weights is
a rolling mean in one step. np.split cuts into pieces that do not overlap.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', 'six values', 180, 53),
    N('m1', 'message', 180, 92, 160, 42, MSG('1 2 3 4 5 6')),
    C('c2', 'every run of 3', 380, 53),
    N('sw', 'np.sliding_window_view 3', 380, 92, 230, 130),
    N('lw', 'list', 380, 260, 400, 42, LIST),
    C('c3', 'the mean of each window', 380, 350),
    N('mn', 'np.mean 1', 380, 385, 150, 90, {'axis': '1'}),
    N('lm', 'list', 560, 385, 200, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'sw', 'in'), ('sw', 'windows', 'lw', ''),
 ('sw', 'windows', 'mn', 'in'), ('mn', 'mean', 'lm', '')],
['np.rolling_buffer', 'np.convolve', 'np.split', 'np.mean'], 540, 1000)


# ----------------------------------------------------------- running extremes
EXTREME = {
    'np.maximum.accumulate': ('the highest so far', 'max', 'largest', 'peak hold', 'rises',
                              'np.minimum.accumulate', 'np.amax', 't.cummax',
                              'the highest a hand has reached so far in a phrase, the loudest moment so far, a level that never falls back'),
    'np.minimum.accumulate': ('the lowest so far', 'min', 'smallest', 'trough hold', 'falls',
                              'np.maximum.accumulate', 'np.amin', 't.cummin',
                              'the lowest point reached so far, the best time so far, a floor that only ever drops'),
}

for stem, (gist, short, word, hold, moves, other, single, torch_name, uses) in EXTREME.items():
    page(stem, f'{stem} - {gist}',
f"""
Replaces each value with the {word} value seen up to that point: the running {short}. The
result only ever {moves} or stays level - a {hold}.

Use it for {uses}. Applied to a whole recording at once, it shows at every moment
the record so far.

nan values are skipped: a gap does not reset the run.

SYNTAX:
{stem} <axis: int>
Default: axis -1 (along each row).

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input is processed and sent.

axis (int box, also an inlet):
Which way to run: -1 along each row, 0 down each column - one running {short} per joint
for a buffer of frames x joints.

OUTPUTS:

running {short}:
The {word} value so far at each position, the same shape as the input.

READING THE DEMO:
32 random values (left) and their running {short} (right): a staircase that steps
{'up' if short == 'max' else 'down'} each time a new record is set and holds level in between. A new random array
comes every second.

RELATED:
{other} holds the other extreme. {single} gives just the final {word} value. {torch_name}
is the PyTorch version. accumulate keeps a running total of a stream.
""",
    metro_row(label='every second', props={'on': True, 'period': 1000.0, 'units': 'milliseconds'},
              init='metro 1000') + [
        C('c1', '32 random values', 180, 53),
        N('rn', 'np.rand 32', 180, 92, 150, 110),
        C('c2', f'the {word} so far', 380, 53),
        N('ac', stem, 380, 92, 230, 70),
        C('c3', 'in', 28, 250),
        N('p1', 'plot', 28, 285, 388, 180, BARS(32, 1.0, 380, 130, 'line')),
        C('c4', f'out: the running {short}', 440, 250),
        N('p2', 'plot', 440, 285, 388, 180, BARS(32, 1.0, 380, 130, 'stair')),
    ],
    [('mt', '', 'rn', ''), ('rn', 'random array', 'ac', 'in'), ('rn', 'random array', 'p1', 'y'),
     ('ac', f'running {short}', 'p2', 'y')],
    [other, single, torch_name, 'accumulate'], 520, 820)


# --------------------------------------------------------- insert/delete/append
page('np.insert', 'np.insert - put values in at a position',
"""
Makes an array longer by putting values in at a position, moving everything after it
along.

Use it to add a joint to a skeleton's list, put a marker or a pause into a sequence,
add a row to a table at a particular place. For adding at the end, np.append is
simpler.

SYNTAX:
np.insert <index: int>
Default: index 0 (at the start).

EXAMPLE:
np.insert -1

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input has the values put in and is sent -
the only inlet that makes the node send.

values:
What to put in, held: send it first. A single value, a list (put in together), or a
row for a 2-D array.

index (int box, also an inlet):
Where to put them: 0 before the first element, 1 after it, and so on. -1 puts them at
the very end, -2 before the last element. Beyond either end goes to that end.

axis (int box, also an inlet):
Which way: 0 (the default) between elements of a list, or between rows of a matrix; 1
between columns.

OUTPUTS:

inserted:
The longer array.

READING THE DEMO:
9 is put into 1 2 3 at index 1: 1 9 2 3. A metro re-sends the list twice a second; change
the index to move the 9.

RELATED:
np.append adds at the end. np.delete takes elements out. np.concatenate joins two
arrays end to end. np.pad adds around the edges.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', 'three values', 180, 53),
    N('m1', 'message', 180, 92, 120, 42, MSG('1 2 3')),
    C('c2', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 210, 88, 46),
    C('c3', 'the value to put in, held', 180, 175),
    N('mv', 'message', 180, 210, 60, 42, MSG('9')),
    C('c4', 'at index 1', 400, 53),
    N('ins', 'np.insert 1', 400, 92, 160, 130),
    N('l1', 'list', 600, 92, 180, 42, LIST),
],
[('lb', 'out', 'mv', ''), ('mv', 'message out', 'ins', 'values'), ('mt', '', 'm1', ''),
 ('m1', 'message out', 'ins', 'in'), ('ins', 'inserted', 'l1', '')],
['np.append', 'np.delete', 'np.concatenate', 'np.pad'], 420, 900)


page('np.delete', 'np.delete - take elements out by position',
"""
Makes an array shorter by taking out the elements at some positions, closing up the
gaps.

Use it to drop joints a model does not need, remove a bad frame or a glitched channel,
or take the first or last element off a list.

SYNTAX:
np.delete <indices: ints ...>
Default: none - the array passes through unchanged.

EXAMPLES:
np.delete 0
np.delete 0 -1

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input has the elements taken out and is
sent.

indices:
The positions to take out, replacing any given as arguments. Held. -1 is the last;
positions that do not exist are ignored.

axis (int box, also an inlet):
Which way: 0 (the default) elements of a list, or rows of a matrix; 1 columns.

OUTPUTS:

remaining:
The shorter array.

READING THE DEMO:
10 20 30 40 without its first and last elements (positions 0 and -1): 20 30. A metro
re-sends the list twice a second.

RELATED:
np.insert and np.append put elements in. np.take keeps the positions you name instead
of removing them. np.[] and np.crop cut out a run.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', 'four values', 180, 53),
    N('m1', 'message', 180, 92, 160, 42, MSG('10 20 30 40')),
    C('c2', 'without the first and last', 380, 53),
    N('dl', 'np.delete 0 -1', 380, 92, 170, 110),
    N('l1', 'list', 580, 92, 160, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'dl', 'in'), ('dl', 'remaining', 'l1', '')],
['np.insert', 'np.take', 'np.[]', 'np.crop'], 320, 820)


page('np.append', 'np.append - add values at the end',
"""
Makes an array longer by adding values at its end - or, for a matrix, a row along the
bottom.

Use it to add a closing value, extend a sequence, or add a row to a table. To keep a
growing record of a stream, np.rolling_buffer is usually better: it keeps the latest
values at a fixed size instead of growing for ever.

SYNTAX:
np.append <values: floats ...>
Default: no values (send them to the values inlet).

EXAMPLE:
np.append 0

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input has the values added and is sent -
the only inlet that makes the node send.

values:
What to add, replacing any given as arguments. Held. A list is added element by
element; a single row onto a matrix is added as a new row.

axis (int box, also an inlet):
Which way: 0 (the default) at the end of a list, or as new rows of a matrix; 1 as new
columns.

OUTPUTS:

appended:
The longer array.

READING THE DEMO:
4 5 added to the end of 1 2 3: 1 2 3 4 5. A metro re-sends the list twice a second.

RELATED:
np.insert puts values in at any position. np.concatenate joins two arrays.
np.rolling_buffer keeps the latest values of a stream.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', 'three values', 180, 53),
    N('m1', 'message', 180, 92, 120, 42, MSG('1 2 3')),
    C('c2', 'add 4 5 at the end', 380, 53),
    N('ap', 'np.append 4 5', 380, 92, 170, 110),
    N('l1', 'list', 580, 92, 180, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'ap', 'in'), ('ap', 'appended', 'l1', '')],
['np.insert', 'np.concatenate', 'np.rolling_buffer', 'np.delete'], 320, 820)


# --------------------------------------------------------- nanargmax/nanargmin
NANARG = {
    'np.nanargmax': ('largest', 'np.argmax', 'argmax', '2', 'np.nanmax', 'np.nanargmin'),
    'np.nanargmin': ('smallest', 'np.argmin', 'argmin', '3', 'np.nanmin', 'np.nanargmax'),
}

for stem, (word, plain, plain_out, result, value_node, other) in NANARG.items():
    out = stem.split('.')[-1]
    page(stem, f'{stem} - where the {word} value is, skipping gaps',
f"""
Gives the position of the {word} value in an array, ignoring any nan ("not a number")
in it.

{plain} treats nan as if it were the answer - it gives the position of the first
nan - so a single dropped reading points at the gap instead of the real {word}
value. {stem} looks only at the values that are there.

SYNTAX:
{stem} <axis: int>
Default: axis None (the whole array, counted as if flattened into one list).

INPUTS and PARAMETERS:

in:
The array (a list is turned into one). Every input is searched and sent.

axis (menu, also an inlet):
None searches every element. A number searches along that axis: for a 2-D array, 0
gives one position per column, 1 one per row.

OUTPUTS:

{out}:
The position: a single number, or one per column or row. If a whole row or column is
nan there is no position to give, and nothing is sent.

READING THE DEMO:
3 nan 7 1. {stem} gives {result}, the position of the {'7' if word == 'largest' else '1'}. {plain} gives 1 - the position of the nan. A
metro re-sends the list twice a second.

RELATED:
{value_node} gives the {word} value itself. {plain} does not skip gaps. {other} finds
the other extreme.
""",
    metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
        C('c1', 'a list with a gap', 180, 53),
        N('m1', 'message', 180, 92, 160, 42, MSG('3 nan 7 1')),
        C('c2', 'skipping the gap', 380, 53),
        N('n1', stem, 380, 92, 150, 90),
        N('l1', 'list', 560, 92, 80, 42, LIST),
        C('c3', 'without skipping: the nan', 380, 230),
        N('n2', plain, 380, 265, 150, 90),
        N('l2', 'list', 560, 265, 80, 42, LIST),
    ],
    [('mt', '', 'm1', ''), ('m1', 'message out', 'n1', 'in'), ('m1', 'message out', 'n2', 'in'),
     ('n1', out, 'l1', ''), ('n2', plain_out, 'l2', '')],
    [plain, value_node, other, 'np.isnan'], 420, 820)


# --------------------------------------------------------------- bincount
page('np.bincount', 'np.bincount - count, or total, by label',
"""
Counts how many times each whole-number label occurs: the count of 0s, of 1s, of 2s,
and so on up to the largest label. With weights it totals them instead: the sum of the
weights of every element labelled 0, of every element labelled 1, and so on.

Labels come from np.digitize (which zone, which range), np.unique's inverse, a
classifier, a list of joint or performer numbers. bincount then gives how often each
occurred - or, weighted by speed or energy, how much activity happened in each zone,
in one step.

SYNTAX:
np.bincount <length: int>
Default: length 0 (just long enough for the largest label).

INPUTS and PARAMETERS:

labels:
Whole numbers, 0 or more (a list is turned into one; any shape is flattened). Every
input is counted and sent - the only inlet that makes the node send.

weights:
One weight per label, held. Without them, or if their number does not match, each
label counts 1.

length (int box, also an inlet):
The smallest length of the result. Set it to the number of labels there could be, so
the result keeps the same length even when the highest labels do not occur.

OUTPUTS:

counts:
One count (or total) per label: position 0 for label 0, and so on.

READING THE DEMO:
Top: 0 2 2 1 2 holds one 0, one 1 and three 2s: 1 1 3.

Bottom: the same labels weighted by 0.5 1 0.25 2 0.25 - say, speeds in three zones. Zone
0 totals 0.5, zone 1 totals 2, zone 2 totals 1 + 0.25 + 0.25 = 1.5. A metro re-sends the
labels twice a second.

RELATED:
np.unique counts any values, not only small whole numbers. np.histogram counts values
falling in ranges. np.digitize turns values into labels.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', 'labels', 180, 53),
    N('m1', 'message', 180, 92, 160, 42, MSG('0 2 2 1 2')),
    C('c2', 'how many of each', 380, 53),
    N('b1', 'np.bincount', 380, 92, 150, 110),
    N('l1', 'list', 560, 92, 160, 42, LIST),
    C('c3', 'once at load', 28, 175),
    N('lb', 'load_bang', 28, 210, 88, 46),
    C('c4', 'a weight for each label, held', 180, 250),
    N('mw', 'message', 180, 285, 200, 42, MSG('0.5 1 0.25 2 0.25')),
    C('c5', 'the total weight of each', 400, 250),
    N('b2', 'np.bincount', 400, 285, 150, 110),
    N('l2', 'list', 580, 285, 160, 42, LIST),
],
[('mt', '', 'm1', ''), ('lb', 'out', 'mw', ''), ('mw', 'message out', 'b2', 'weights'),
 ('m1', 'message out', 'b1', 'labels'), ('m1', 'message out', 'b2', 'labels'),
 ('b1', 'counts', 'l1', ''), ('b2', 'counts', 'l2', '')],
['np.unique', 'np.histogram', 'np.digitize', 'np.sum'], 460, 900)


def main():
    args = sys.argv[1:]
    force = '--force' in args
    wanted = [a for a in args if not a.startswith('--')] or list(PAGES)
    for stem in wanted:
        p = PAGES[stem]
        path = os.path.join(HELP_DIR, stem + '_help.json')
        if os.path.exists(path) and not force:
            print('exists, skipped (use --force):', path)
            continue
        print(build(stem, p['title'], p['body'], p['demo'], p['links'], demo_width=840,
                    text_width=760, text_height=p['text_height']))


if __name__ == '__main__':
    main()
