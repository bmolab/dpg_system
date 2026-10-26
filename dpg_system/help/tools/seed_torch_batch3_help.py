"""Seed the help pages for the third batch of new torch nodes (2026-10-25): one page per node.

    python3 dpg_system/help/tools/seed_torch_batch3_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

t.multinomial, t.randperm, t.one_hot, t.max_pool, t.avg_pool.

The random pages follow np.random.choice and np.random.permutation. Demo data is
chosen so every value shown in a list is exact in float32 (the avg_pool means are
whole or halves); probabilities from t.nn.softmax are shown in a heat map, which
prints with a format. Not a make_* script: rebuild.py never runs it, and it will not
write over an existing page without --force. Treat the pages as hand-made once written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from seed_numpy_more_help import LIST, MSG, HM, C, N, see_also


PAGES = {}


def page(stem, title, body, demo, links, also, also_y, text_height=820):
    PAGES[stem] = dict(title=title, body=body.strip('\n'), demo=demo + see_also(also_y, also),
                       links=links, text_height=text_height)


METRO = {'on': True, 'period': 200.0, 'units': 'milliseconds'}
METRO_HALF = {'on': True, 'period': 500.0, 'units': 'milliseconds'}
METRO_1S = {'on': True, 'period': 1000.0, 'units': 'milliseconds'}


def metro(y=53, label='five times a second', props=METRO, init='metro 200'):
    return [C('cm', label, 28, y), N('mt', init, 28, y + 39, 129, 70, props)]


def W(text):
    """A message or list wide enough for this text."""
    return max(60, int(len(text) * 13.5) + 30)


SEED_TEXT = """seed:
An option, under the node. -1 (the default) draws different numbers every time the
patch opens. Any other number starts the same sequence every time - the same
"random" piece on every run, which is what you want for rehearsal, comparison and
debugging."""


# -------------------------------------------------------------- t.multinomial
PROBS = '0.125 0.125 0.25 0.5'
SCORES = '0 1 2 3'
page('t.multinomial', 't.multinomial - choose at random, by probability',
"""
Picks indices at random, each as likely as its probability: with 0.125 0.125 0.25 0.5,
index 3 comes up half the time and index 0 one time in eight.

It is how a trained network that predicts "what comes next" is played. Such a network
(t.model) ends with a score for every possible next note, word, gesture or pose;
t.nn.softmax turns the scores into probabilities, and t.multinomial picks one - the
likely choices most of the time, the unlikely ones now and then, so the result is
neither fixed nor arbitrary. Feed each choice back in, and the network writes a
sequence. Without a network it is weighted chance: a pose to move to, a clip, a
direction, each as often as you choose.

The probabilities need not add up to 1 - 1 1 2 4 means the same as the list above - but
none may be negative and not all may be 0. Give one row per distribution (a 2-D
tensor) and each row gets its own pick, in one step. The picks are made on the
probabilities' device.

SYNTAX:
t.multinomial <count>
Default: count 1.

INPUTS and PARAMETERS:

probabilities:
How likely each index is: a list or tensor, or one row per distribution. Every input
picks and sends.

count (int box, also an inlet):
How many to pick. 1 (the default) sends the index itself (one per row for rows); more
sends them as a list.

temperature (float box, also an inlet):
Reshapes the probabilities before the pick: each is raised to the power 1/temperature.
1 (the default) leaves them as they are. Below 1 the likely choices grow likelier -
near 0 the most likely is chosen almost every time. Above 1 the choices even out - the
music grows more surprising. It is the main control for playing a generative model.

repeats:
An option, under the node. On (the default), an index can be picked again within one
count. Off, each at most once per count - like dealing cards (count is then at most
the number of choices).

""" + SEED_TEXT + """

OUTPUTS:

chosen:
The index picked, or the list of indices.

READING THE DEMO:
Top: 16 picks from 0 1 2 3 with probabilities 0.125 0.125 0.25 0.5, twice a second:
about eight 3s, four 2s, and two each of 0 and 1 - different every time.

Bottom: a model's scores for four notes, 0 1 2 3, made into probabilities by
t.nn.softmax (0.03 0.09 0.24 0.64), and one note picked from them. Drag temperature
down towards 0.1 and the top note takes over; up towards 5 and all four come up
almost equally.

RELATED:
np.random.choice picks from a list of items. t.nn.softmax turns scores into
probabilities; t.model runs the network that makes the scores. t.argmax always takes
the most likely, with no chance at all.
""",
metro(label='twice a second', props=METRO_HALF, init='metro 500') + [
    C('c1', 'probabilities for 0 1 2 3', 180, 53),
    N('m1', 'message', 180, 92, W(PROBS), 42, MSG(PROBS)),
    C('c2', '16 picks', 520, 53),
    N('mn', 't.multinomial 16', 520, 92, 180, 150),
    N('l1', 'list', 520, 270, 360, 42, LIST),
    C('c3', "a model's scores for four notes", 180, 360),
    N('m2', 'message', 180, 395, W(SCORES), 42, MSG(SCORES)),
    N('sm', 't.nn.softmax', 180, 465, 150, 90),
    C('c4', 'their probabilities', 180, 580),
    N('hp', 'heat_map', 180, 615, 248, 80, HM(230, 40, 1, 0.0, 1.0, '%.2f')),
    C('c5', 'one note: drag temperature', 520, 360),
    N('m3', 't.multinomial', 520, 395, 180, 150),
    N('l2', 'list', 520, 575, 80, 42, LIST),
],
[('mt', '', 'm1', ''), ('mt', '', 'm2', ''), ('m1', 'message out', 'mn', 'probabilities'),
 ('mn', 'chosen', 'l1', ''), ('m2', 'message out', 'sm', 'tensor in'),
 ('sm', 'output', 'hp', 'y'), ('sm', 'output', 'm3', 'probabilities'), ('m3', 'chosen', 'l2', '')],
['np.random.choice', 't.nn.softmax', 't.model', 't.argmax'], 740, 1150)


# ----------------------------------------------------------------- t.randperm
NUMS = '10 20 30 40 50'
page('t.randperm', 't.randperm - shuffle',
"""
Shuffles a tensor into a random order: every item kept, each exactly once, in a new
order every time. Given a whole number n instead, it shuffles 0 to n-1.

Use it for a random running order of scenes, a play order of clips that repeats none
until all have played, a random assignment of performers to parts, or a shuffled set of
positions to take values from (with t.index_select or t.take). For a stack of poses or
recordings, one per row, it shuffles the rows. A shuffled tensor stays on its device.

SYNTAX:
t.randperm

INPUTS and PARAMETERS:

in:
A list or tensor to shuffle - for a 2-D tensor, whole rows are shuffled - or a single
whole number n, for the numbers 0 to n-1 shuffled (a message box sends 8 as a list of
one; that counts as n too). Every input is shuffled and sent.

""" + SEED_TEXT + """

OUTPUTS:

shuffled:
The same items in a random order.

READING THE DEMO:
Top: 10 20 30 40 50 in a new order every second. Bottom: 8, which makes the numbers 0 to
7 in a random order - a set of positions for t.index_select.

RELATED:
t.multinomial picks some at random, by probability. t.sort and t.argsort put things
back in order. np.random.permutation is the numpy version.
""",
metro(label='every second', props=METRO_1S, init='metro 1000') + [
    C('c1', 'five numbers', 180, 53),
    N('m1', 'message', 180, 92, W(NUMS), 42, MSG(NUMS)),
    C('c2', 'shuffled', 420, 53),
    N('p1', 't.randperm', 420, 92, 150, 70),
    N('l1', 'list', 420, 180, 300, 42, LIST),
    C('c3', 'a whole number: 0 to 7 shuffled', 180, 270),
    N('m2', 'message', 180, 305, 60, 42, MSG('8')),
    N('p2', 't.randperm', 420, 305, 150, 70),
    N('l2', 'list', 420, 390, 300, 42, LIST),
],
[('mt', '', 'm1', ''), ('mt', '', 'm2', ''), ('m1', 'message out', 'p1', 'in'),
 ('m2', 'message out', 'p2', 'in'), ('p1', 'shuffled', 'l1', ''), ('p2', 'shuffled', 'l2', '')],
['t.multinomial', 't.argsort', 't.index_select', 'np.random.permutation'], 490, 900)


# ------------------------------------------------------------------ t.one_hot
LABELS = '2 0 3 1'
page('t.one_hot', 't.one_hot - a class number as a row of 0s and one 1',
"""
Turns class numbers into one-hot vectors: class 2 of 4 becomes 0 0 1 0 - a 1 in the
class's place and 0 everywhere else.

It is the form most networks take a choice in, and the form their training targets
are written in: a note from a scale, a gesture from a vocabulary, a performer from a
cast, given to t.model as a vector rather than a number (a number would suggest that
class 3 is "more" than class 1). Fed back, the choices t.multinomial makes become the
network's next input this way. A list of class numbers gives one row each.

SYNTAX:
t.one_hot <classes>
Default: classes -1 - as many as the largest class number needs.

INPUTS and PARAMETERS:

indices:
A class number, or a list or tensor of them: whole numbers from 0. Every input is
converted and sent.

classes (int box, also an inlet):
How many classes there are - the length of each vector. Give it, so a vector has the
same length whatever classes happen to arrive; -1 takes the largest class number plus
one. A class number at or past it is refused.

output_type:
An option, under the node: float (the default, what networks take), int or bool.

OUTPUTS:

one hot:
One vector per class number: a single number gives one vector, a list gives one row
each.

READING THE DEMO:
Top: the class numbers 2 0 3 1, with 4 classes: four rows, each with its 1 in a
different place. Bottom: the single class 2 gives 0 0 1 0 - as one row, because a
message box sends 2 as a list of one. A plain number (from an int box) gives the
vector itself.

RELATED:
t.argmax goes the other way, from a vector of scores back to a class number.
t.multinomial picks a class number by probability. t.model takes the vectors.
""",
metro() + [
    C('c1', 'class numbers', 180, 53),
    N('m1', 'message', 180, 92, W(LABELS), 42, MSG(LABELS)),
    C('c2', 'of 4 classes', 400, 53),
    N('oh', 't.one_hot 4', 400, 92, 150, 90),
    C('c3', 'one row each', 600, 53),
    N('hm', 'heat_map', 600, 92, 168, 168, HM(150, 120, 1, 0.0, 1.0, '%.0f')),
    C('c4', 'a single class', 180, 300),
    N('m2', 'message', 180, 335, 60, 42, MSG('2')),
    N('o2', 't.one_hot 4', 400, 335, 150, 90),
    N('l2', 'list', 600, 335, 260, 42, LIST),
],
[('mt', '', 'm1', ''), ('mt', '', 'm2', ''), ('m1', 'message out', 'oh', 'indices'),
 ('oh', 'one hot', 'hm', 'y'), ('m2', 'message out', 'o2', 'indices'), ('o2', 'one hot', 'l2', '')],
['t.argmax', 't.multinomial', 't.model', 't.eye'], 490, 900)


# ------------------------------------------------------------- t.max_pool / avg
SIG = '1 5 3 8 2 2 9 0 6'
POOL_COMMON = """
SYNTAX:
{stem} <window> <step>
Defaults: window 2, step the same as the window (windows side by side).

EXAMPLE:
{stem} 4
{stem} 8 1

INPUTS and PARAMETERS:

tensor in:
The values (a list or tensor). Every input is pooled and sent.

window (int box, also an inlet):
How many values each window holds (each side of a tile, over two dims).

step (int box, also an inlet):
How far each window moves on from the last. The same as the window (the default)
takes windows side by side, with no overlap: the length shrinks by that factor. 1
slides the window one value at a time: one result per position{running}.

dim (int box, also an inlet):
Which dim to pool along: -1 (the default) along each row, 0 down each column - over
time, for a buffer of frames x joints.

pool over:
An option, under the node. one dim (the default), or last two dims - square tiles of
an image or grid, window x window, every image of a stack at once.

partial window:
An option, under the node. Off (the default), values left over at the end that do not
fill a window are dropped. On, they make one last, shorter window.

OUTPUTS:

pooled:
The {what} of each window, the shape of the input with the pooled dim shortened.
"""

page('t.max_pool', 't.max_pool - the largest value in each window',
"""
Takes the largest value in each window of values: windows side by side shrink a
sequence (or an image) while keeping its peaks; a window sliding one value at a time
holds every peak for the length of the window.

Shrinking while keeping peaks is what pooling layers do inside networks, and what you
want when reducing a fast stream - a hit, a spike or a sudden movement survives,
where averaging would blur it away. Sliding, it is a running maximum: the loudest of
the last eight frames, a simple envelope that rises at once and falls only when the
peak leaves the window. The answer stays on the input's device.
""" + POOL_COMMON.format(stem='t.max_pool', running=' - the running maximum', what='largest value') + """
READING THE DEMO:
Top: 1 5 3 8 2 2 9 0 6 in windows of 3 side by side - 1 5 3, 8 2 2, 9 0 6 - gives the
three peaks, 5 8 9. Bottom: the same window sliding one value at a time gives 5 8 8 8
9 9 9: each peak held while it stays in the window.

RELATED:
t.avg_pool takes the mean instead. t.unfold gives every window itself; t.max along a
dim, or t.cummax for the running maximum from the start.
""",
metro() + [
    C('c1', 'nine values', 180, 53),
    N('m1', 'message', 180, 92, W(SIG), 42, MSG(SIG)),
    C('c2', 'windows of 3, side by side', 440, 53),
    N('p1', 't.max_pool 3', 440, 92, 150, 130),
    N('l1', 'list', 640, 92, 200, 42, LIST),
    C('c3', 'windows of 3, sliding by 1', 440, 260),
    N('p2', 't.max_pool 3 1', 440, 295, 150, 130),
    N('l2', 'list', 640, 295, 220, 42, LIST),
],
[('mt', '', 'm1', ''), ('m1', 'message out', 'p1', 'tensor in'), ('m1', 'message out', 'p2', 'tensor in'),
 ('p1', 'pooled', 'l1', ''), ('p2', 'pooled', 'l2', '')],
['t.avg_pool', 't.unfold', 't.max', 't.cummax'], 470, 1150)


IMG = '0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15'
page('t.avg_pool', 't.avg_pool - the mean of each window',
"""
Takes the mean of each window of values: windows side by side shrink a sequence (or an
image) by that factor, each value standing for the average of the ones it replaces; a
window sliding one value at a time smooths a stream, a moving average.

Use it to bring a fast stream down to a slower rate without the aliasing of simply
dropping frames, to make a small, blurred version of an image or grid (pool over the
last two dims), or to smooth a buffer of joint data over time. It is also the
averaging layer inside many networks. The answer stays on the input's device.
""" + POOL_COMMON.format(stem='t.avg_pool', running=' - a moving average', what='mean') + """
READING THE DEMO:
Top: 1 5 3 8 2 2 9 0 6 in windows of 3 side by side - 1 5 3, 8 2 2, 9 0 6 - gives their
means, 3 4 5.

Bottom: the numbers 0 to 15 as a 4 x 4 image, pooled over the last two dims in 2 x 2
tiles: a 2 x 2 image, each value the mean of its tile - 2.5 4.5 10.5 12.5.

RELATED:
t.max_pool keeps the peaks instead. t.convolve smooths with any weighting;
t.interpolate resizes to any size. t.mean along a dim averages everything at once.
""",
metro() + [
    C('c1', 'nine values', 180, 53),
    N('m1', 'message', 180, 92, W(SIG), 42, MSG(SIG)),
    C('c2', 'windows of 3, side by side', 440, 53),
    N('p1', 't.avg_pool 3', 440, 92, 150, 130),
    N('l1', 'list', 640, 92, 200, 42, LIST),
    C('c3', '0 to 15 as a 4 x 4 image', 180, 260),
    N('m2', 'message', 180, 295, W(IMG), 42, MSG(IMG)),
    N('rs', 't.reshape 4 4', 180, 360, 150, 70),
    N('hi', 'heat_map', 180, 450, 168, 168, HM(150, 120, 1, 0.0, 15.0, '%.0f')),
    C('c4', 'means of 2 x 2 tiles', 440, 400),
    N('p2', 't.avg_pool 2', 440, 435, 150, 130, {'pool over': 'last two dims'}),
    N('h2', 'heat_map', 640, 435, 168, 130, HM(150, 80, 1, 0.0, 15.0, '%.1f')),
],
[('mt', '', 'm1', ''), ('mt', '', 'm2', ''), ('m1', 'message out', 'p1', 'tensor in'),
 ('p1', 'pooled', 'l1', ''), ('m2', 'message out', 'rs', 'tensor in'), ('rs', 'output', 'hi', 'y'),
 ('rs', 'output', 'p2', 'tensor in'), ('p2', 'pooled', 'h2', 'y')],
['t.max_pool', 't.convolve', 't.interpolate', 't.mean'], 660, 1150)


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
        print(build(stem, p['title'], p['body'], p['demo'], p['links'], demo_width=880,
                    text_width=760, text_height=p['text_height']))


if __name__ == '__main__':
    main()
