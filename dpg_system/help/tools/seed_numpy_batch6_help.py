"""Seed the help pages for the sixth batch of numpy nodes (2026-10-24): one page per node.

    python3 dpg_system/help/tools/seed_numpy_batch6_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

np.logspace, np.geomspace, np.fft.rfft2, np.fft.fft2, np.fft.irfft2,
np.fft.ifft2, np.isin, np.intersect1d, np.setdiff1d, np.union1d.

Like the earlier seeders it is not a make_* script, so rebuild.py never runs it,
and it will not write over an existing page without --force. Treat the pages as
hand-made once written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from seed_numpy_more_help import LIST, MSG, HM, C, N, see_also
from seed_numpy_batch2_help import METRO_SLOW, metro_row


PAGES = {}


def page(stem, title, body, demo, links, also, also_y, text_height=820):
    PAGES[stem] = dict(title=title, body=body.strip('\n'), demo=demo + see_also(also_y, also),
                       links=links, text_height=text_height)


IMG = lambda maxy=1.0, color='viridis': HM(210, 210, 1, -maxy if color == 'red-blue' else 0.0,
                                           maxy, '', color)


# ---------------------------------------------------------- logspace/geomspace
page('np.logspace', 'np.logspace - values spaced evenly on a log scale, by exponent',
"""
Makes values that grow by the same RATIO at every step, rather than by the same
amount: 1 10 100 1000, each ten times the last. You give the first and last
EXPONENTS - np.logspace 0 3 4 is 10 to the power of 0, 1, 2 and 3.

Equal ratios are how pitch, loudness, brightness and many other perceptions work:
an octave is a doubling wherever you start. Use it for frequency bands, a set of
time scales (10 ms to 10 s), zoom levels, a sweep of filter cutoffs, or test values
spanning many sizes.

np.geomspace does the same from the end values themselves, which is usually easier:
np.geomspace 1 1000 4 is the same as np.logspace 0 3 4.

SYNTAX:
np.logspace <start exponent> <stop exponent> <steps>
Defaults: 0, 3, 4 steps, base 10.

EXAMPLE:
np.logspace 0 7 8

INPUTS and PARAMETERS:

(button):
Makes the values and sends them.

start, stop (float boxes, also inlets):
The first and last exponents. Both ends are included.

steps (int box, also an inlet):
How many values.

base (float box, also an inlet):
What is raised to the exponents: 10 (the default), 2 for doublings, 2.71828 (e) for
natural growth.

OUTPUTS:

logspace out:
The values.

READING THE DEMO:
Made at load: 10 to the powers 0 to 3, 1 10 100 1000; and with base 2, the powers 0 to 7:
1 2 4 8 16 32 64 128.

RELATED:
np.geomspace takes the end values instead of exponents. np.linspace spaces values by
equal amounts. t.logspace is the PyTorch version.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', '10 to the 0 ... 3', 140, 53),
    N('l1', 'np.logspace 0 3 4', 140, 92, 170, 170),
    N('o1', 'list', 340, 92, 260, 42, LIST),
    C('c3', '2 to the 0 ... 7', 140, 310),
    N('l2', 'np.logspace 0 7 8', 140, 345, 170, 170, {'base': 2.0}),
    N('o2', 'list', 340, 345, 330, 42, LIST),
],
[('lb', 'out', 'l1', ''), ('lb', 'out', 'l2', ''), ('l1', 'logspace out', 'o1', ''),
 ('l2', 'logspace out', 'o2', '')],
['np.geomspace', 'np.linspace', 't.logspace', 'exp'], 560, 900)


page('np.geomspace', 'np.geomspace - values from one to another in equal ratios',
"""
Makes values from start to stop that grow by the same RATIO at every step: a
geometric sequence. np.geomspace 220 1760 4 is 220 440 880 1760 - each double the last:
the A of four octaves.

Equal ratios are how pitch, loudness, brightness and many other perceptions work, so
a control or a set of bands spaced this way feels even, where equal steps (np.linspace)
feel crowded at one end. Use it for musical intervals, octave or third-octave bands, a
range of time scales, zoom levels, or delays that grow by a fixed factor.

SYNTAX:
np.geomspace <start> <stop> <steps>
Defaults: 1, 1000, 4 steps.

EXAMPLE:
np.geomspace 20 20000 11

INPUTS and PARAMETERS:

(button):
Makes the values and sends them.

start, stop (float boxes, also inlets):
The first and last values, both included. Neither can be 0, and both must be on the
same side of zero (a ratio cannot take you across it).

steps (int box, also an inlet):
How many values.

OUTPUTS:

geomspace out:
The values.

READING THE DEMO:
Made at load: four octaves of A from 220 Hz, 220 440 880 1760; and 1 to 1000 in four
steps, 1 10 100 1000. The 880 shows as 880.0000000000001: values reached by
multiplying ratios carry a little rounding in their last digits. It makes no audible
or visible difference; put round after the node if you need whole numbers.

RELATED:
np.logspace takes exponents instead of end values. np.linspace spaces by equal amounts.
log turns equal ratios back into equal steps.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', 'four octaves of A', 140, 53),
    N('g1', 'np.geomspace 220 1760 4', 140, 92, 220, 150),
    N('o1', 'list', 390, 92, 280, 42, LIST),
    C('c3', '1 to 1000 in four steps', 140, 290),
    N('g2', 'np.geomspace 1 1000 4', 140, 325, 220, 150),
    N('o2', 'list', 390, 325, 280, 42, LIST),
],
[('lb', 'out', 'g1', ''), ('lb', 'out', 'g2', ''), ('g1', 'geomspace out', 'o1', ''),
 ('g2', 'geomspace out', 'o2', '')],
['np.logspace', 'np.linspace', 'log', 'exp'], 540, 900)


# ---------------------------------------------------------------- 2-D FFT
STRIPES = [
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', '0 to 31', 140, 53),
    N('ar', 'np.arange 32', 140, 92, 150, 110),
    C('c3', 'a 32 x 32 grid', 320, 53),
    N('mg', 'np.meshgrid', 320, 92, 150, 90),
]


def fft2_page(stem, real):
    other = 'np.fft.fft2' if real else 'np.fft.rfft2'
    inverse = 'np.fft.irfft2' if real else 'np.fft.ifft2'
    stripe_text = ('vertical stripes - x only, 4 cycles across' if real else
                   'diagonal stripes - x + y, 4 cycles each way')
    page(stem, f'{stem} - the spatial frequencies of an image' + (' (real)' if real else ''),
f"""
The Fourier transform in two dimensions: breaks a picture, a heat map or any grid of
values into waves that run across it - stripes of every spacing and direction - and
says how much of each there is.

Broad, gentle variation sits near the centre of the (centred) spectrum; fine detail and
sharp edges further out. Stripes show as bright points, at a distance from the centre
set by how close the stripes are and in a direction at right angles to them. Removing
the outer part blurs; removing the centre leaves only the edges - filters that are hard
to build any other way. It also finds repeating patterns: the rhythm of a floor
pattern, the grid of a woven texture.

{'np.fft.rfft2 is for real values (ordinary images and grids), and keeps only half the spectrum along the last axis, since a real image mirrors itself.' if real else 'np.fft.fft2 gives the full spectrum, and works on complex grids as well. For an ordinary real image np.fft.rfft2 gives the same in half the space; the full spectrum is symmetric about its centre.'}

SYNTAX:
{stem}

INPUTS and PARAMETERS:

in:
A 2-D array (or a stack of them: the last two axes are used). Every input is
transformed and sent.

centre:
An option, under the node. On (the default), the magnitude and phase outlets are
rearranged so the steady average sits in the middle and the slowest waves around it
- the way spectra are usually looked at. The spectrum outlet is never rearranged, so it
always goes straight into {inverse}.

norm:
An option, under the node: how the result is scaled, as for the 1-D transforms.

OUTPUTS:

spectrum:
The complex spectrum, in numpy's order, ready for {inverse}.

magnitude:
How much of each wave there is.

phase:
Where each wave starts, in radians. Sent before magnitude, which is sent before
spectrum.

SEEING THE MAGNITUDE:
The steady average and a few strong waves can be thousands of times larger than the
rest; log1p of the magnitude, as in the demo, brings the faint ones into view.

READING THE DEMO:
{stripe_text}: sin of each position times 45 degrees (left). Its spectrum (right,
through log1p) is dark except {'a single bright point on the middle row, four columns in from the left edge: four cycles across, none down (the real version keeps only the positive half across, so its left edge is zero, not the centre)' if real else 'two bright points on a diagonal, four steps each way from the centre and mirrored through it: four cycles across and four down'}.

RELATED:
{other} {'gives the full, mirrored spectrum' if real else 'is the version for real images'}. {inverse} turns a spectrum back into an image.
np.fft.rfft and np.fft.fft are the 1-D transforms.
""",
    STRIPES + [
        C('c4', 'stripes', 500, 53),
    ] + ([N('ml', '* 45', 500, 92, 129, 70)] if real else
         [N('ad', '+', 500, 92, 129, 70), N('ml', '* 45', 500, 180, 129, 70)]) + [
        N('sn', 'sin', 660, 92, 129, 70),
        C('c5', 'the image', 28, 290),
        N('hi', 'heat_map', 28, 325, 228, 250, IMG(1.0, 'red-blue')),
        C('c6', 'its spectrum', 290, 290),
        N('ff', stem, 290, 325, 150, 110),
        C('c7', 'log1p of the magnitude', 470, 290),
        N('lg', 'log1p', 470, 325, 129, 70),
        N('hs', 'heat_map', 470, 420, 228, 250, IMG(6.5, 'plasma')),
    ],
    [('lb', 'out', 'ar', ''), ('ar', 'arange out', 'mg', 'x')] +
    ([('mg', 'X', 'ml', 'in')] if real else
     [('mg', 'Y', 'ad', 'operand', 0, 1), ('mg', 'X', 'ad', 'in'), ('ad', 'result', 'ml', 'in')]) +
    [('ml', 'result', 'sn', 'in'), ('sn', 'out', 'hi', 'y'), ('sn', 'out', 'ff', 'in'),
     ('ff', 'magnitude', 'lg', 'in'), ('lg', 'result', 'hs', 'y')],
    [other, inverse, 'np.fft.rfft', 'log1p'], 730, 1150)


fft2_page('np.fft.rfft2', True)
fft2_page('np.fft.fft2', False)


def ifft2_page(stem, real):
    forward = 'np.fft.rfft2' if real else 'np.fft.fft2'
    page(stem, f'{stem} - an image from its spatial frequencies',
f"""
Turns a 2-D spectrum from {forward} back into an image: every wave across the grid is
rebuilt at its strength and position and all of them are added up.

Round trip it gives back the original exactly. The point is to change the spectrum in
between: keep only the middle (the slow waves) to blur; remove the middle to keep only
edges and detail; remove one bright point to take out a regular pattern - a mesh, a
flicker, a scan line - while leaving everything else.

SYNTAX:
{stem}{' <rows> <columns>' if real else ''}

INPUTS and PARAMETERS:

spectrum:
The complex spectrum, straight from {forward}'s spectrum outlet (not the centred
magnitude). Every input is transformed and sent.
""" + ("""
shape (text box, also an inlet):
The size of the image to make: rows columns. Empty (the default) makes the last axis
even - 2 x (columns - 1) - which is right for any image with an even number of
columns; give the size for an odd one.
""" if real else '') + f"""
norm:
An option, under the node: must match the norm used for {forward}.

OUTPUTS:

signal:
The image. {'Always real.' if real else 'Sent as real numbers when the imaginary part is only rounding (any spectrum from a real image, unchanged); otherwise complex.'}

READING THE DEMO:
16 x 16 random values (left) go to their spectrum and straight back (right): the same
picture, a new one every two seconds. Put any change to the spectrum between the two
nodes to see what it does to the image.

RELATED:
{forward} makes the spectrum. np.fft.irfft and np.fft.ifft are the 1-D inverses.
""",
    metro_row(label='every two seconds', props={'on': True, 'period': 2000.0, 'units': 'milliseconds'},
              init='metro 2000') + [
        C('c1', '16 x 16 random values', 180, 53),
        N('rn', 'np.rand 16 16', 180, 92, 150, 130),
        C('c2', 'to frequencies', 380, 53),
        N('ff', forward, 380, 92, 150, 110),
        C('c3', 'and back', 580, 53),
        N('if', stem, 580, 92, 150, 110),
        C('c4', 'in', 28, 290),
        N('h1', 'heat_map', 28, 325, 228, 250, IMG(1.0)),
        C('c5', 'out: the same picture', 470, 290),
        N('h2', 'heat_map', 470, 325, 228, 250, IMG(1.0)),
    ],
    [('mt', '', 'rn', ''), ('rn', 'random array', 'ff', 'in'), ('rn', 'random array', 'h1', 'y'),
     ('ff', 'spectrum', 'if', 'spectrum'), ('if', 'signal', 'h2', 'y')],
    [forward, 'np.fft.irfft', 'np.fft.ifft', 'np.real'], 640, 900)


ifft2_page('np.fft.irfft2', True)
ifft2_page('np.fft.ifft2', False)


# ------------------------------------------------------------------- isin
page('np.isin', 'np.isin - which elements belong to a set',
"""
Marks each element of an array that is one of a set of values: true where it is, false
where it is not.

Use it to pick out the joints, channels, notes or labels you care about from a longer
list - which of these readings come from the arm joints, which notes are in the scale,
which zones are active - and then select them with np.where or np.argwhere, or count
them with np.sum.

SYNTAX:
np.isin <set: numbers or words ...>
Default: no set (send one to the set inlet).

EXAMPLES:
np.isin 0 4 7
np.isin do mi sol

INPUTS and PARAMETERS:

in:
The elements to test: numbers or words, any shape. Every input is tested and sent.

set:
The values that count, replacing any given as arguments. Held.

invert:
An option, under the node. On, the answer is reversed: true for elements NOT in the
set.

OUTPUTS:

mask:
True or false for each element, the shape of the input.

READING THE DEMO:
Top: which of 1 to 6 are even - in the set 2 4 6. Bottom: which of the notes do re mi fa
sol are in the chord do mi sol. A metro re-sends both twice a second.

RELATED:
np.intersect1d gives the shared values themselves. == tests against a single value.
np.where and np.argwhere use the mask.
""",
metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
    C('c1', 'six numbers', 180, 53),
    N('m1', 'message', 180, 92, 160, 42, MSG('1 2 3 4 5 6')),
    C('c2', 'in the set 2 4 6?', 380, 53),
    N('i1', 'np.isin 2 4 6', 380, 92, 150, 70),
    N('l1', 'list', 380, 180, 400, 42, LIST),
    C('c3', 'five notes', 180, 270),
    N('m2', 'message', 180, 305, 180, 42, MSG('do re mi fa sol')),
    C('c4', 'in the chord do mi sol?', 380, 270),
    N('i2', 'np.isin do mi sol', 380, 305, 170, 70),
    N('l2', 'list', 380, 390, 360, 42, LIST),
],
[('mt', '', 'm1', ''), ('mt', '', 'm2', ''), ('m1', 'message out', 'i1', 'in'),
 ('m2', 'message out', 'i2', 'in'), ('i1', 'mask', 'l1', ''), ('i2', 'mask', 'l2', '')],
['np.intersect1d', 'np.where', 'equal', 'np.unique'], 490, 860)


# -------------------------------------------------------------- set ops
SETS = {
    'np.intersect1d': ('the values two lists share', 'in both', 'in both', '3 4',
                       'the values in both lists', """Use it to find what two selections have in common: the joints active in both
performers, the notes shared by two chords, the channels flagged by two different
tests."""),
    'np.setdiff1d': ('the values in one list but not the other', 'only in first', 'only in A', '1 2',
                     'the values in the first list that are not in the second', """Use it to take a selection away from a whole: every joint except the ones already
used, the notes of a scale not yet played, the channels that passed a test (all minus
the failures). The order matters: A minus B is not B minus A."""),
    'np.union1d': ('every value in either list', 'in either', 'in either', '1 2 3 4 5',
                   'every value that is in either list', """Use it to combine two selections into one, each value once: the joints used by
either hand, the notes of two chords together, the channels flagged by either test."""),
}

for stem, (gist, out, short, result, what, use) in SETS.items():
    others = [s for s in SETS if s != stem]
    page(stem, f'{stem} - {gist}',
f"""
Gives {what} - each value once, sorted from smallest to largest (or in alphabetical
order, for words).

{use}

These treat the lists as sets: how many times a value occurs, and where, does not
matter. Numbers and words both work.

SYNTAX:
{stem}

INPUTS and PARAMETERS:

in:
The first list, A (any shape is flattened). Every input is compared and sent - the
only inlet that makes the node send.

in 2:
The second list, B, held: send it first.

OUTPUTS:

{out}:
The result, sorted, each value once. It may be empty.

READING THE DEMO:
A is 1 2 3 4 and B is 3 4 5: {short}, {result}. A metro re-sends A twice a second;
edit either list and click it.

RELATED:
{others[0]} and {others[1]} are the other set operations. np.isin marks which
elements of a list are in a set. np.unique gives the values of one list, each once.
""",
    metro_row(label='twice a second', props=METRO_SLOW, init='metro 500') + [
        C('c1', 'A', 180, 53),
        N('ma', 'message', 180, 92, 140, 42, MSG('1 2 3 4')),
        C('c2', 'once at load', 28, 175),
        N('lb', 'load_bang', 28, 210, 88, 46),
        C('c3', 'B, held', 180, 175),
        N('mb', 'message', 180, 210, 120, 42, MSG('3 4 5')),
        C('c4', short, 380, 53),
        N('op', stem, 380, 92, 160, 70),
        N('ls', 'list', 580, 92, 200, 42, LIST),
    ],
    [('lb', 'out', 'mb', ''), ('mb', 'message out', 'op', 'in 2'), ('mt', '', 'ma', ''),
     ('ma', 'message out', 'op', 'in'), ('op', out, 'ls', '')],
    others + ['np.isin', 'np.unique'], 340, 760)


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
