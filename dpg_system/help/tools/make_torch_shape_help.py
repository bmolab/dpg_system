"""torch shape, joining and splitting: one page per node.

t.reshape, t.view, t.flatten (+ t.ravel), t.squeeze, t.unsqueeze, t.permute,
t.transpose, t.t, t.adjoint, t.cat, t.stack, t.hstack, t.vstack
(+ t.row_stack), t.dstack, t.column_stack, t.chunk, t.tensor_split.

These were two family pages (t.reshape, t.cat). Each node now has its own page
whose demo shows the tensor before and after as heat_maps, with the shapes as
text and a SEE ALSO row under it.

Aliases: t.flatten / t.ravel call torch.flatten / torch.ravel, which give the
same result for every input the node can receive (the node passes no
start/end dim), so they share the t.flatten page. t.vstack / t.row_stack both
run torch.vstack, so they share the t.vstack page. t.view and t.reshape share
a class but run different operations (view refuses non-contiguous tensors),
so they have separate pages, as do t.chunk / t.tensor_split.

Most demos use the capital F glyph (no symmetry, so every rearrangement reads)
made inside a make_F subpatcher, as on the np.flip page.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build


def HM(width, height, count, miny=0.0, maxy=1.0, fmt='%.2f', color='viridis'):
    """Property block for a heat_map that draws a whole array as one picture."""
    return {'color': color, 'width': width, 'height': height, 'sample count': count,
            'min y': miny, 'max y': maxy, 'update_mode': 'heat_map',
            'number format': fmt}


LIST = {'text in': '', 'font size': '24'}


def MSG(text):
    return {'text in': text, 'font size': '24'}


def see_also(y, stems, x=30):
    """A SEE ALSO row: one help_link per related page, left to right."""
    row = []
    for s in stems:
        w = int(len(s) * 7.7) + 30
        row.append({'key': 'sa_' + s, 'init': 'help_link ' + s, 'pos': (x, y),
                    'w': w, 'h': 30})
        x += w + 14
    row.append({'key': 'sa_c', 'comment': True, 'text': 'SEE ALSO - click to open',
                'pos': (x, y)})
    return row


METRO = {'on': True, 'period': 100.0, 'units': 'milliseconds'}

# A capital F, 7 rows by 6 columns: 1 is the letter, 0 the background. It has
# no symmetry at all, so every turn, fold and cut gives a different picture.
F_GLYPH = ('1 1 1 1 1 1  '
           '1 1 0 0 0 0  '
           '1 1 0 0 0 0  '
           '1 1 1 1 0 0  '
           '1 1 0 0 0 0  '
           '1 1 0 0 0 0  '
           '1 1 0 0 0 0')


def f_source(y=62, x=30, key='rs'):
    return [{'key': key, 'init': 'p make_F', 'pos': (x, y), 'w': 160, 'h': 70}]


# Inside make_F: metro -> message (the F as 42 numbers) -> t.reshape 7 6.
F_SUB = {
    'name': 'make_F',
    'host': 'rs',
    'demo': [
        {'key': 'met', 'init': 'metro 100', 'pos': (40, 40), 'w': 129, 'h': 70,
         'props': METRO},
        {'key': 'cm', 'init': 'comment re-sends the F ten times a second',
         'pos': (190, 50), 'w': 300, 'h': 30},
        {'key': 'msg', 'init': 'message', 'pos': (40, 130), 'w': 290, 'h': 42,
         'props': {'text in': F_GLYPH, 'font size': '24', 'width': 260}},
        {'key': 'cg', 'init': 'comment the letter F as 42 numbers, 1 = letter, 0 = background',
         'pos': (40, 185), 'w': 480, 'h': 30},
        {'key': 'rs', 'init': 't.reshape 7 6', 'pos': (40, 230), 'w': 260, 'h': 70},
        {'key': 'cr', 'init': 'comment folded into a tensor of 7 rows of 6',
         'pos': (320, 245), 'w': 330, 'h': 30},
        {'key': 'out', 'init': 'out F', 'pos': (40, 330), 'w': 150, 'h': 80},
    ],
    'links': [('met', '', 'msg', ''),
              ('msg', 'message out', 'rs', 'tensor in'),
              ('rs', 'output', 'out', 'F')],
}

# The same F, but sent only when the patch opens and when 'go' is banged -
# for the t.view page, whose failing chain would otherwise print an error ten
# times a second.
F_ONCE_SUB = {
    'name': 'make_F',
    'host': 'rs',
    'demo': [
        {'key': 'in', 'init': 'in go', 'pos': (40, 40), 'w': 150, 'h': 80},
        {'key': 'lb', 'init': 'load_bang', 'pos': (220, 40), 'w': 88, 'h': 46},
        {'key': 'cm', 'init': 'comment once on open, and on every bang',
         'pos': (330, 50), 'w': 300, 'h': 30},
        {'key': 'msg', 'init': 'message', 'pos': (40, 150), 'w': 290, 'h': 42,
         'props': {'text in': F_GLYPH, 'font size': '24', 'width': 260}},
        {'key': 'cg', 'init': 'comment the letter F as 42 numbers, 1 = letter, 0 = background',
         'pos': (40, 205), 'w': 480, 'h': 30},
        {'key': 'rs', 'init': 't.reshape 7 6', 'pos': (40, 250), 'w': 260, 'h': 70},
        {'key': 'out', 'init': 'out F', 'pos': (40, 350), 'w': 150, 'h': 80},
    ],
    'links': [('in', 'go', 'msg', ''), ('lb', 'out', 'msg', ''),
              ('msg', 'message out', 'rs', 'tensor in'),
              ('rs', 'output', 'out', 'F')],
}

# Inside make_ramp: load_bang -> t.linspace 0 1 42 -> t.reshape 7 6.
RAMP_SUB = {
    'name': 'make_ramp',
    'host': 'rs2',
    'demo': [
        {'key': 'lb', 'init': 'load_bang', 'pos': (40, 40), 'w': 88, 'h': 46},
        {'key': 'cl', 'init': 'comment once, when the patch opens',
         'pos': (150, 50), 'w': 280, 'h': 30},
        {'key': 'ls', 'init': 't.linspace 0 1 42', 'pos': (40, 110), 'w': 220, 'h': 200,
         'props': {'start': 0.0, 'stop': 1.0, 'steps': 42}},
        {'key': 'cs', 'init': 'comment 42 values from 0 to 1',
         'pos': (280, 125), 'w': 240, 'h': 30},
        {'key': 'rs', 'init': 't.reshape 7 6', 'pos': (40, 330), 'w': 260, 'h': 70},
        {'key': 'cr', 'init': 'comment folded into 7 rows of 6, the same shape as the F',
         'pos': (320, 345), 'w': 440, 'h': 30},
        {'key': 'out', 'init': 'out ramp', 'pos': (40, 430), 'w': 150, 'h': 80},
    ],
    'links': [('lb', 'out', 'ls', '###input'),
              ('ls', 'linspace out', 'rs', 'tensor in'),
              ('rs', 'output', 'out', 'ramp')],
}

# Inside make_layers: the F and a ramp, each 7 x 6, stacked into one
# tensor of shape [2, 7, 6] - two pictures as two layers.
LAYERS_SUB = {
    'name': 'make_layers',
    'host': 'ly',
    'demo': [
        {'key': 'met', 'init': 'metro 100', 'pos': (40, 40), 'w': 129, 'h': 70,
         'props': METRO},
        {'key': 'msg', 'init': 'message', 'pos': (40, 130), 'w': 290, 'h': 42,
         'props': {'text in': F_GLYPH, 'font size': '24', 'width': 260}},
        {'key': 'cg', 'init': 'comment layer 0: the letter F, 7 x 6',
         'pos': (40, 185), 'w': 300, 'h': 30},
        {'key': 'rs', 'init': 't.reshape 7 6', 'pos': (40, 230), 'w': 260, 'h': 70},
        {'key': 'lb', 'init': 'load_bang', 'pos': (420, 40), 'w': 88, 'h': 46},
        {'key': 'ls', 'init': 't.linspace 0 1 42', 'pos': (420, 100), 'w': 220, 'h': 200,
         'props': {'start': 0.0, 'stop': 1.0, 'steps': 42}},
        {'key': 'rs2', 'init': 't.reshape 7 6', 'pos': (420, 320), 'w': 260, 'h': 70},
        {'key': 'cr', 'init': 'comment layer 1: a ramp 0 .. 1, 7 x 6, made once',
         'pos': (420, 400), 'w': 380, 'h': 30},
        {'key': 'st', 'init': 't.stack', 'pos': (40, 450), 'w': 180, 'h': 100,
         'props': {'dim': 0}},
        {'key': 'cs', 'init': 'comment stacked: shape [2, 7, 6]',
         'pos': (240, 465), 'w': 280, 'h': 30},
        {'key': 'out', 'init': 'out layers', 'pos': (40, 580), 'w': 150, 'h': 80},
    ],
    'links': [('lb', 'out', 'ls', '###input'),
              ('ls', 'linspace out', 'rs2', 'tensor in'),
              ('rs2', 'output', 'st', 'tensor 2'),
              ('met', '', 'msg', ''),
              ('msg', 'message out', 'rs', 'tensor in'),
              ('rs', 'output', 'st', 'tensor 1'),
              ('st', 'output', 'out', 'layers')],
}


def ramp12(y=62):
    """metro 100 -> t.linspace 0 1 12 at the top of the page."""
    return [
        {'key': 'met', 'init': 'metro 100', 'pos': (30, y), 'w': 129, 'h': 70,
         'props': METRO},
        {'key': 'ls', 'init': 't.linspace 0 1 12', 'pos': (30, y + 90), 'w': 220, 'h': 200,
         'props': {'start': 0.0, 'stop': 1.0, 'steps': 12}},
        {'key': 'cr', 'comment': True, 'text': 'twelve values 0 .. 1, sent 10 times a second',
         'pos': (30, y + 95)},
    ]


RAMP12_LINKS = [('met', '', 'ls', '###input')]


def info_list(key, x, y):
    """t.info -> list: the shape of whatever reaches it."""
    return [
        {'key': 'i' + key, 'init': 't.info', 'pos': (x, y), 'w': 120, 'h': 120},
        {'key': 'l' + key, 'init': 'list', 'pos': (x + 70, y), 'w': 150, 'h': 42,
         'props': LIST},
    ]


def info_links(key, src, src_out):
    return [(src, src_out, 'i' + key, 'in'), ('i' + key, 'shape', 'l' + key, '')]


F_HM = HM(180, 210, 7, fmt='none')
STRIP = HM(520, 40, 1, fmt='none')

# =================================================================== t.reshape
body = """Rearranges a tensor into a new shape, keeping every number and its order.

Twelve numbers can be one row of twelve, three rows of four, four rows of
three or a column of twelve. t.reshape changes which of these you have;
the numbers themselves are untouched. Reach for it when a node wants its
data in a particular layout - a list of 3-D points as rows of three, an
image as rows and columns, a 2-D result as one long row.

HOW THE NUMBERS ARE PLACED:
The tensor is read row by row, as it would print, and the new shape is
filled row by row in the same order. So a ramp 0 .. 1 reshaped to 3 4
puts the first four values in row 0, the next four in row 1 and so on -
the ramp still reads left to right, top to bottom.

SYNTAX:
t.reshape <dim> <dim> ...
With no arguments the shape is -1: everything in one row, like t.flatten.

EXAMPLE:
t.reshape 3 4

INPUTS and PARAMETERS:

tensor in:
The tensor (a list, a numpy array or a number is turned into a float32
tensor on the cpu first). Receiving it is what makes the node send.

shape (text box, also an inlet):
The new shape. Any whole numbers in the text are the dimensions, so
3 4, [3, 4] and 3,4 all mean three rows of four. From a cord it takes a
string (as from the demo's string boxes), a list of numbers or a tensor
of whole numbers. The dimensions must multiply to the number of values
you have: 3 4 works for twelve and fails for ten. One dimension may be
-1, meaning "whatever fits": 6 -1 gives six rows of however many columns
are needed. An empty box passes the tensor through unchanged.
Changing the shape does NOT send by itself - the next tensor uses it. In
the demo the metro re-sends the ramp, so a click shows at once.
If the shape does not fit, a 'Shape Mismatch' message is printed and
nothing is sent; the heat_map keeps its last picture.

OUTPUTS:

output:
The same numbers in the new shape, same dtype and device.

RESHAPE OR VIEW?
t.reshape always works. When the tensor's memory is already in reading
order it gives a view of the same memory; otherwise (after t.t or
t.transpose, say) it quietly copies. t.view does only the first and
refuses the second - see its page.

READING THE DEMO:
The strip at the top is the ramp as it arrives: shape [12], one row.
Click 3 4 and the heat_map below shows it folded into three rows of four,
the values still in order reading across each row. 4 3 and 2 6 fold it
differently; 6 -1 lets t.reshape work out the 2; 12 1 makes a column -
twelve rows of one value, the ramp now running down the picture. t.info
prints each result's shape.

RELATED:
t.view is the same refold, but only without copying. t.flatten goes
back to one row. t.squeeze and t.unsqueeze remove or add a single axis
of length 1 without spelling out the whole shape. t.transpose is NOT a
reshape: it turns rows into columns, which moves the numbers.
np.reshape is the numpy version."""

demo = ramp12() + [
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 320), 'w': 540, 'h': 90,
     'props': HM(520, 40, 1)},
    {'key': 'c0', 'comment': True, 'text': 'in: shape [12], one row', 'pos': (30, 325)},
    {'key': 'm1', 'init': 'string', 'pos': (30, 430), 'w': 130, 'h': 42, 'props': MSG('3 4')},
    {'key': 'm2', 'init': 'string', 'pos': (175, 430), 'w': 130, 'h': 42, 'props': MSG('4 3')},
    {'key': 'm3', 'init': 'string', 'pos': (320, 430), 'w': 130, 'h': 42, 'props': MSG('2 6')},
    {'key': 'm4', 'init': 'string', 'pos': (465, 430), 'w': 130, 'h': 42, 'props': MSG('6 -1')},
    {'key': 'm5', 'init': 'string', 'pos': (610, 430), 'w': 130, 'h': 42, 'props': MSG('12 1')},
    {'key': 'c1', 'comment': True, 'text': 'click a shape, or type one in the box',
     'pos': (30, 435)},
    {'key': 'rs', 'init': 't.reshape 3 4', 'pos': (30, 500), 'w': 300, 'h': 70},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 590), 'w': 320, 'h': 230,
     'props': HM(300, 180, 3)},
    {'key': 'c2', 'comment': True, 'text': 'out: the same twelve values, refolded',
     'pos': (30, 595)},
] + info_list('o', 380, 590) + [
    {'key': 'c3', 'comment': True, 'text': 'its shape: [rows, columns]', 'pos': (380, 720)},
] + see_also(850, ['t.view', 't.flatten', 't.squeeze', 't.unsqueeze', 'np.reshape'])
links = RAMP12_LINKS + [
    ('ls', 'linspace out', 'hin', 'y'), ('ls', 'linspace out', 'rs', 'tensor in'),
    ('m1', 'string out', 'rs', 'shape'), ('m2', 'string out', 'rs', 'shape'),
    ('m3', 'string out', 'rs', 'shape'), ('m4', 'string out', 'rs', 'shape'),
    ('m5', 'string out', 'rs', 'shape'),
    ('rs', 'output', 'hout', 'y')] + info_links('o', 'rs', 'output')
print(build('t.reshape', 't.reshape - the same numbers in a new shape', body,
            demo, links, demo_width=760, text_width=760, text_height=920))

# ====================================================================== t.view
body = """Refolds a tensor into a new shape WITHOUT copying it - and refuses
when that is not possible.

It is t.reshape with a guarantee: the result is another way of reading
the very same memory, so it costs nothing at any size. The price is that
it only works when the tensor's numbers already sit in memory in reading
order (the tensor is "contiguous"). A fresh tensor always is. A tensor
that has been through t.t, t.transpose or t.permute usually is not: those
change how the memory is READ, not where the numbers are, so reading it
"row by row" would mean jumping about. t.view then fails, where
t.reshape would quietly make a copy.

Use t.view when you want that failure - when a copy would be a surprise
(a big tensor every frame) and you would rather know. Otherwise use
t.reshape.

SYNTAX:
t.view <dim> <dim> ...
With no arguments the shape is -1: everything in one row.

EXAMPLE:
t.view 6 7

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or numpy array is turned into a float32 tensor first).
Receiving it is what makes the node send.

shape (text box, also an inlet):
The new shape, exactly as on t.reshape: whole numbers in any form
(6 7, [6, 7], 6,7), one of them may be -1, and they must multiply to the
number of values. Changing it does NOT send by itself. An empty box
passes the tensor through.

OUTPUTS:

output:
The same numbers in the new shape, sharing the input's memory.
When the shape does not fit, OR the input is not contiguous, a
'Shape Mismatch - view size is not compatible...' message is printed and
nothing is sent.

READING THE DEMO:
The F (7 rows of 6) is made inside make_F when the patch opens; click the
button to send it again. t.view 6 7 refolds it into 6 rows of 7, reading
row by row - the letter breaks up, because the rows now wrap at 7, but
every number is in its place. On the right the F goes through t.t first
(turned on its side, 6 by 7). t.is_contiguous says False, and t.view 42
on it sends nothing - its strip stays empty and the console prints the
error. Through t.contiguous first (which copies it into reading order)
the same t.view 42 works: the strip is the turned F, row by row.

RELATED:
t.reshape does the same refold and copies when it has to. t.contiguous
makes the copy explicitly; t.is_contiguous tells you whether one is
needed. t.t, t.transpose and t.permute are what make a tensor
non-contiguous. np.reshape is the numpy version."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
] + f_source(y=130) + [
    {'key': 'c0', 'comment': True, 'text': 'click to send the F again', 'pos': (30, 67)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 230), 'w': 200, 'h': 260,
     'props': F_HM},
    {'key': 'c1', 'comment': True, 'text': 'the F, 7 x 6', 'pos': (30, 235)},
    {'key': 'vw', 'init': 't.view 6 7', 'pos': (30, 520), 'w': 280, 'h': 70},
    {'key': 'hv', 'init': 'heat_map', 'pos': (30, 610), 'w': 220, 'h': 240,
     'props': HM(200, 190, 6, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 'viewed as 6 x 7: same order, rows wrap at 7',
     'pos': (30, 615)},
    {'key': 'tt', 'init': 't.t', 'pos': (340, 230), 'w': 140, 'h': 70},
    {'key': 'c3', 'comment': True, 'text': 'turned: 6 x 7, NOT contiguous', 'pos': (340, 235)},
    {'key': 'ic', 'init': 't.is_contiguous', 'pos': (340, 320), 'w': 160, 'h': 70},
    {'key': 'lc', 'init': 'list', 'pos': (520, 320), 'w': 120, 'h': 42, 'props': LIST},
    {'key': 'v2', 'init': 't.view 42', 'pos': (340, 410), 'w': 280, 'h': 70},
    {'key': 'c4', 'comment': True, 'text': 'fails: nothing sent, error printed',
     'pos': (340, 415)},
    {'key': 'ct', 'init': 't.contiguous', 'pos': (340, 520), 'w': 150, 'h': 70},
    {'key': 'v3', 'init': 't.view 42', 'pos': (340, 610), 'w': 280, 'h': 70},
    {'key': 'c5', 'comment': True, 'text': 'copied into order first: works',
     'pos': (340, 615)},
    {'key': 'hf', 'init': 'heat_map', 'pos': (30, 880), 'w': 640, 'h': 90,
     'props': HM(620, 40, 1, fmt='none')},
    {'key': 'c6', 'comment': True, 'text': 't.view 42 straight after t.t: stays empty',
     'pos': (30, 885)},
    {'key': 'hg', 'init': 'heat_map', 'pos': (30, 990), 'w': 640, 'h': 90,
     'props': HM(620, 40, 1, fmt='none')},
    {'key': 'c7', 'comment': True, 'text': 'after t.contiguous: the turned F, row by row',
     'pos': (30, 995)},
] + see_also(1110, ['t.reshape', 't.contiguous', 't.is_contiguous', 't.t', 'np.reshape'])
links = [('btn', '', 'rs', 'go'),
         ('rs', 'F', 'hin', 'y'), ('rs', 'F', 'vw', 'tensor in'),
         ('vw', 'output', 'hv', 'y'),
         ('rs', 'F', 'tt', 'tensor in'),
         ('tt', 'tensor out', 'ic', 'tensor in'), ('ic', 'result', 'lc', ''),
         ('tt', 'tensor out', 'v2', 'tensor in'), ('v2', 'output', 'hf', 'y'),
         ('tt', 'tensor out', 'ct', 'tensor in'), ('ct', 'tensor out', 'v3', 'tensor in'),
         ('v3', 'output', 'hg', 'y')]
print(build('t.view', 't.view - a new shape without copying, or not at all', body,
            demo, links, demo_width=700, text_width=760, text_height=960,
            subpatch=F_ONCE_SUB))

# =================================================================== t.flatten
body = """Collapses a tensor of any shape into one long row, reading it row by
row.

t.flatten and t.ravel are two names for the same node: they give the same
numbers in the same order, for every tensor.

Use it when a node wants a plain list of values - to plot every value of
a grid as one line, to send an image's pixels as a single list, or to
turn the result of a 2-D calculation back into a 1-D signal. Seven rows of
six become one row of 42.

THE ORDER IS THE READING ORDER, NOT THE MEMORY ORDER:
Row 0 first, left to right, then row 1, and so on - as the tensor would
print. A tensor that has been turned with t.t is read in its NEW
orientation, so its strip runs down the original columns. (Where the
memory is already in that order it is a view and costs nothing;
otherwise a copy is made.)

SYNTAX:
t.flatten
t.ravel
No arguments. (torch.flatten's start and end dims are not available
here; to flatten only some axes, use t.reshape with -1.)

INPUTS:

tensor in:
The tensor (a list or numpy array becomes a float32 tensor first).
Receiving it makes the node send. A single number comes out as a
1-element tensor.

OUTPUTS:

tensor out:
A 1-D tensor holding every value, same dtype and device.

READING THE DEMO:
The F (7 by 6) flattened is one strip of 42 cells: the top bar is the
first six yellow cells, then each row's stem and arm in turn. Below, the
F goes through t.t (turned 6 by 7) before t.ravel: the strip now reads
the F's COLUMNS - two long yellow runs for the stem, then the arms.

RELATED:
t.reshape -1 does the same, and t.reshape can then fold it back.
t.view -1 does it only without copying. t.squeeze removes only
length-1 axes. flatten (np.ravel) is the numpy version."""

demo = f_source() + [
    {'key': 'c0', 'comment': True, 'text': 'the F, 7 x 6 - open make_F to see how it is made',
     'pos': (30, 160)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 200), 'w': 200, 'h': 260,
     'props': F_HM},
    {'key': 'fl', 'init': 't.flatten', 'pos': (30, 490), 'w': 150, 'h': 70},
    {'key': 'hf', 'init': 'heat_map', 'pos': (30, 580), 'w': 640, 'h': 90,
     'props': HM(620, 40, 1, fmt='none')},
    {'key': 'c1', 'comment': True, 'text': 't.flatten: 42 in a row, row 0 first',
     'pos': (30, 585)},
] + info_list('f', 210, 490) + [
    {'key': 'tt', 'init': 't.t', 'pos': (30, 700), 'w': 140, 'h': 70},
    {'key': 'rv', 'init': 't.ravel', 'pos': (30, 790), 'w': 150, 'h': 70},
    {'key': 'c2', 'comment': True, 'text': 'turned first, then t.ravel', 'pos': (30, 795)},
    {'key': 'hr', 'init': 'heat_map', 'pos': (30, 880), 'w': 640, 'h': 90,
     'props': HM(620, 40, 1, fmt='none')},
    {'key': 'c3', 'comment': True, 'text': 'reads the columns of the F, because it was turned',
     'pos': (30, 885)},
] + see_also(1000, ['t.reshape', 't.view', 't.squeeze', 'flatten'])
links = [('rs', 'F', 'hin', 'y'), ('rs', 'F', 'fl', 'tensor in'),
         ('fl', 'tensor out', 'hf', 'y'),
         ('rs', 'F', 'tt', 'tensor in'), ('tt', 'tensor out', 'rv', 'tensor in'),
         ('rv', 'tensor out', 'hr', 'y')] + info_links('f', 'fl', 'tensor out')
print(build('t.flatten', 't.flatten / t.ravel - one long row', body,
            demo, links, demo_width=700, text_width=760, text_height=820,
            subpatch=F_SUB))

# =================================================================== t.squeeze
body = """Removes axes of length 1 from a tensor's shape, leaving the numbers
alone.

A column of twelve values has shape [12, 1]; a row that came out of a
batch has shape [1, 12]. Both hold the same twelve numbers as a plain
[12], but many nodes treat them differently - a [12, 1] is a 2-D table
with one column, not a list. t.squeeze drops those stray 1s.

WHICH AXES:
With no argument, EVERY axis of length 1 goes: [1, 12, 1] becomes [12].
With an argument (t.squeeze 0), only that axis is considered, and only
if its length really is 1 - otherwise the tensor passes through
unchanged. Naming the axis is safer in a patch whose sizes can change: a
batch of one squeezes away as you want, but so would a list that happens
to hold one value.

SYNTAX:
t.squeeze
t.squeeze <dim: int>

EXAMPLE:
t.squeeze 0

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or numpy array becomes a float32 tensor first).
Receiving it makes the node send.

dim (int box, also an inlet - only when an argument was given):
The one axis to remove. Negative counts from the end (-1 is the last).
An axis the tensor does not have sends nothing. Changing it does NOT
send by itself; the next tensor uses it. A bare t.squeeze has no dim
box: retype it with an argument to get one.

OUTPUTS:

output:
The same numbers, with the length-1 axes gone.

READING THE DEMO:
The ramp of twelve is folded by t.reshape into a column, [12, 1] -
twelve rows of one value. t.squeeze removes the 1: the shape is [12] and
the heat_map draws it as a row. t.squeeze 0 only looks at axis 0, which
has length 12, so it changes nothing ([12, 1] still). Click 1 12: now
both remove a 1 (axis 0 is the length-1 one). Click 3 4: nothing to
remove, both pass it through.

RELATED:
t.unsqueeze adds a length-1 axis - the reverse. t.reshape can set any
shape. t.flatten collapses every axis, not just the length-1 ones.
np.squeeze is the numpy version."""

demo = ramp12() + [
    {'key': 'm1', 'init': 'string', 'pos': (300, 62), 'w': 130, 'h': 42, 'props': MSG('12 1')},
    {'key': 'm2', 'init': 'string', 'pos': (445, 62), 'w': 130, 'h': 42, 'props': MSG('1 12')},
    {'key': 'm3', 'init': 'string', 'pos': (590, 62), 'w': 130, 'h': 42, 'props': MSG('3 4')},
    {'key': 'c1', 'comment': True, 'text': 'choose the input shape', 'pos': (300, 67)},
    {'key': 'rs', 'init': 't.reshape 12 1', 'pos': (300, 150), 'w': 300, 'h': 70},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 320), 'w': 240, 'h': 250,
     'props': HM(220, 200, 12)},
    {'key': 'c0', 'comment': True, 'text': 'in: a column, [12, 1]', 'pos': (30, 325)},
] + info_list('i', 300, 320) + [
    {'key': 'sq', 'init': 't.squeeze', 'pos': (30, 600), 'w': 150, 'h': 70},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 690), 'w': 540, 'h': 90,
     'props': HM(520, 40, 1)},
    {'key': 'c2', 'comment': True, 'text': 't.squeeze: every length-1 axis removed',
     'pos': (30, 695)},
] + info_list('o', 300, 600) + [
    {'key': 'sq0', 'init': 't.squeeze 0', 'pos': (30, 810), 'w': 150, 'h': 100,
     'props': {'dim': 0}},
    {'key': 'c3', 'comment': True, 'text': 't.squeeze 0: only axis 0, only if it is 1',
     'pos': (30, 815)},
] + info_list('z', 300, 810) + see_also(960, ['t.unsqueeze', 't.reshape', 't.flatten',
                                              'np.squeeze'])
links = RAMP12_LINKS + [
    ('m1', 'string out', 'rs', 'shape'), ('m2', 'string out', 'rs', 'shape'),
    ('m3', 'string out', 'rs', 'shape'),
    ('ls', 'linspace out', 'rs', 'tensor in'), ('rs', 'output', 'hin', 'y'),
    ('rs', 'output', 'sq', 'tensor in'), ('sq', 'output', 'hout', 'y'),
    ('rs', 'output', 'sq0', 'tensor in')] + info_links('i', 'rs', 'output') + \
    info_links('o', 'sq', 'output') + info_links('z', 'sq0', 'output')
print(build('t.squeeze', 't.squeeze - remove axes of length 1', body,
            demo, links, demo_width=740, text_width=760, text_height=880))

# ================================================================= t.unsqueeze
body = """Adds an axis of length 1 to a tensor's shape, at the position you
choose, leaving the numbers alone.

Twelve values as [12] are a list. As [1, 12] they are a table of one
row; as [12, 1], a table of one column. Same numbers, but a node that
wants a 2-D input, or a batch of inputs, needs one of the latter.
t.unsqueeze makes it.

WHERE THE NEW AXIS GOES:
dim 0 puts the 1 in front: [12] becomes [1, 12], a single row - the usual
way to make a "batch of one". dim 1 (for a 1-D tensor, the end) puts it
behind: [12, 1], a column. A negative dim counts from the end: -1 always
puts the new axis last. For a [7, 6] tensor the choices are 0, 1 and 2
(or -3 .. -1), giving [1, 7, 6], [7, 1, 6] and [7, 6, 1].

SYNTAX:
t.unsqueeze <dim: int>
A bare t.unsqueeze always uses dim 0 and has no dim box; give an
argument to get the box.

EXAMPLE:
t.unsqueeze 1

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or numpy array becomes a float32 tensor first).
Receiving it makes the node send.

dim (int box, also an inlet - only when an argument was given):
Where to insert the axis. A dim outside the allowed range sends
nothing. Changing it does NOT send by itself; in the demo the metro
re-sends the ramp, so the picture follows.

OUTPUTS:

output:
The same numbers with one more axis, of length 1.

READING THE DEMO:
The ramp arrives as [12], a row. t.unsqueeze 1 makes it [12, 1]: twelve
rows of one value, and the heat_map draws a column. Type 0 into dim and
it is [1, 12] - one row of twelve, which looks like the input again but
is now 2-D (t.info shows the extra 1). -1 is the same as 1 here.

RELATED:
t.squeeze removes length-1 axes - the reverse. t.reshape can set any
shape, including these. t.stack adds a new axis while joining several
tensors. np.expand_dims (np.unsqueeze) is the numpy version."""

demo = ramp12() + [
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 320), 'w': 540, 'h': 90,
     'props': HM(520, 40, 1)},
    {'key': 'c0', 'comment': True, 'text': 'in: [12], one row', 'pos': (30, 325)},
] + info_list('i', 300, 62) + [
    {'key': 'us', 'init': 't.unsqueeze 1', 'pos': (30, 440), 'w': 150, 'h': 100,
     'props': {'dim': 1}},
    {'key': 'c1', 'comment': True, 'text': 'type 0, 1 or -1 into dim', 'pos': (30, 445)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 570), 'w': 240, 'h': 250,
     'props': HM(220, 200, 12)},
    {'key': 'c2', 'comment': True, 'text': 'out: one more axis, of length 1',
     'pos': (30, 575)},
] + info_list('o', 300, 570) + see_also(850, ['t.squeeze', 't.reshape', 't.stack',
                                              'np.expand_dims'])
links = RAMP12_LINKS + [
    ('ls', 'linspace out', 'hin', 'y'), ('ls', 'linspace out', 'us', 'tensor in'),
    ('us', 'output', 'hout', 'y')] + info_links('i', 'ls', 'linspace out') + \
    info_links('o', 'us', 'output')
print(build('t.unsqueeze', 't.unsqueeze - add an axis of length 1', body,
            demo, links, demo_width=620, text_width=760, text_height=820))

# =================================================================== t.permute
body = """Puts a tensor's axes in a new order, all at once.

You list every axis by its old number, in the order you want them: for a
tensor of shape [2, 7, 6], permute 1 2 0 gives [7, 6, 2] - old axis 1
first, then old axis 2, then old axis 0. The numbers move with their
axes; nothing is lost or added.

The everyday use is changing a layout convention. Image libraries
disagree: some keep colour channels first ([channels, height, width]),
others last ([height, width, channels]). permute 1 2 0 turns the first
into the second, permute 2 0 1 the second into the first. The same goes
for [batch, time, joint] versus [batch, joint, time]. For a 2-D tensor,
permute 1 0 is simply t.t.

SYNTAX:
t.permute <axis> <axis> ...

EXAMPLE:
t.permute 1 2 0

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or numpy array becomes a float32 tensor first).
Receiving it makes the node send.

permute (text box, also an inlet):
The new order, as whole numbers in any form (1 2 0, [1, 2, 0], 1,2,0).
It must name EVERY axis exactly once. If it lists the wrong number of
axes the tensor is passed through UNCHANGED, silently - check the shape
when a permute seems to do nothing. A repeated axis prints a 'Permute
Error' and sends nothing. Negative numbers count from the end. An empty
box passes the tensor through. Changing it does NOT send by itself.

OUTPUTS:

permuted tensor out:
The tensor with its axes reordered. Like t.t, this changes how the
memory is read rather than moving it, so the result is usually not
contiguous: t.view will refuse it (use t.reshape, or t.contiguous first).

READING THE DEMO:
make_layers holds two pictures as one tensor [2, 7, 6]: layer 0 is the
F, layer 1 a ramp. The top heat_map shows them one above the other.
t.permute 1 2 0 moves the layer axis to the end: [7, 6, 2], a 7 by 6
picture with two values in every cell, like an image with two channels.
The two lower heat_maps take channel 0 and channel 1 out again (t[]
:, :, 0 and :, :, 1) - the F and the ramp, intact. Click 2 1 0: rows and
columns swap as well ([6, 7, 2]) and both pictures come out turned on
their side.

RELATED:
t.transpose swaps just two axes; t.t and t.adjoint swap the last two.
t.reshape changes the shape without moving numbers between axes.
np.transpose (with an axes list) is the numpy version."""

demo = [
    {'key': 'ly', 'init': 'p make_layers', 'pos': (30, 62), 'w': 180, 'h': 70},
    {'key': 'c0', 'comment': True, 'text': 'two pictures as layers: [2, 7, 6]',
     'pos': (30, 160)},
] + info_list('i', 300, 62) + [
    {'key': 'r14', 'init': 't.reshape 14 6', 'pos': (30, 200), 'w': 300, 'h': 70},
    {'key': 'c1', 'comment': True, 'text': 'only for display: layer 0 above layer 1',
     'pos': (30, 205)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 290), 'w': 200, 'h': 380,
     'props': HM(180, 330, 14, fmt='none')},
    {'key': 'm1', 'init': 'string', 'pos': (300, 290), 'w': 130, 'h': 42,
     'props': MSG('1 2 0')},
    {'key': 'm2', 'init': 'string', 'pos': (445, 290), 'w': 130, 'h': 42,
     'props': MSG('2 1 0')},
    {'key': 'c2', 'comment': True, 'text': 'click an order', 'pos': (300, 295)},
    {'key': 'pm', 'init': 't.permute 1 2 0', 'pos': (300, 360), 'w': 300, 'h': 70},
] + info_list('o', 300, 450) + [
    {'key': 'c3', 'comment': True, 'text': 'the new shape', 'pos': (300, 580)},
    {'key': 'g0', 'init': 't[] :, :, 0', 'pos': (30, 700), 'w': 300, 'h': 70,
     'props': {'indices': ':, :, 0'}},
    {'key': 'g1', 'init': 't[] :, :, 1', 'pos': (360, 700), 'w': 300, 'h': 70,
     'props': {'indices': ':, :, 1'}},
    {'key': 'h0', 'init': 'heat_map', 'pos': (30, 790), 'w': 230, 'h': 260,
     'props': HM(210, 210, 7, fmt='none')},
    {'key': 'h1', 'init': 'heat_map', 'pos': (360, 790), 'w': 230, 'h': 260,
     'props': HM(210, 210, 7, fmt='none')},
    {'key': 'c4', 'comment': True, 'text': 'channel 0 and channel 1 of the permuted tensor',
     'pos': (30, 1070)},
] + see_also(1120, ['t.transpose', 't.t', 't.adjoint', 't.reshape', 'np.transpose'])
links = [('ly', 'layers', 'r14', 'tensor in'), ('r14', 'output', 'hin', 'y'),
         ('m1', 'string out', 'pm', 'permute'), ('m2', 'string out', 'pm', 'permute'),
         ('ly', 'layers', 'pm', 'tensor in'),
         ('pm', 'permuted tensor out', 'g0', 'tensor in'),
         ('pm', 'permuted tensor out', 'g1', 'tensor in'),
         ('g0', 'output', 'h0', 'y'), ('g1', 'output', 'h1', 'y')] + \
    info_links('i', 'ly', 'layers') + info_links('o', 'pm', 'permuted tensor out')
print(build('t.permute', 't.permute - put the axes in a new order', body,
            demo, links, demo_width=700, text_width=760, text_height=980,
            subpatch=LAYERS_SUB))

# ================================================================= t.transpose
body = """Swaps two axes of a tensor. For a 2-D tensor: rows become columns.

The default, axes 0 and 1, turns a 7 by 6 picture into a 6 by 7 one -
row 0 becomes column 0, so the picture is flipped across its diagonal
(the top-left to bottom-right line stays put). It is not a rotation:
compare t.flip and t.roll on the t.roll page.

For a tensor with more axes you choose which two swap - [batch, time,
joint] with transpose 1 2 becomes [batch, joint, time]. To reorder three
or more axes at once, use t.permute.

SYNTAX:
t.transpose <dim 1: int> <dim 2: int>
Defaults: 0 and 1.

EXAMPLE:
t.transpose 0 1

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or numpy array becomes a float32 tensor first).
Receiving it makes the node send. A 1-D tensor (or a single number) has
nothing to swap and is passed through unchanged.

dim 1 / dim 2 (int boxes, also inlets):
The two axes to swap; order does not matter (0 1 and 1 0 are the same).
The same axis twice gives the tensor back unchanged. Negative counts
from the end. An axis the tensor does not have prints an error and
sends nothing. Changing them does NOT send by itself; in the demo a
metro inside make_F re-sends the F ten times a second.

OUTPUTS:

permuted tensor out:
The tensor with the two axes swapped. Its memory is not moved, so the
result is not contiguous: t.view refuses it (use t.reshape, or
t.contiguous first).

READING THE DEMO:
The F (7 by 6) transposed lies on its side, 6 by 7: its top bar is now
the left column and its stem the top rows. t.info shows the shape swap.
Type 1 into dim 1 (so both are 1) and the F comes back unchanged.

RELATED:
t.t is the short form for 2-D tensors. t.adjoint swaps the last two axes
of any tensor (and conjugates complex ones). t.permute reorders all the
axes at once. np.transpose is the numpy version."""

demo = f_source() + [
    {'key': 'c0', 'comment': True, 'text': 'the F, 7 x 6 - open make_F to see how it is made',
     'pos': (30, 160)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 330), 'w': 230, 'h': 260,
     'props': HM(210, 210, 7, fmt='none')},
    {'key': 'tr', 'init': 't.transpose 0 1', 'pos': (340, 200), 'w': 200, 'h': 120,
     'props': {'dim 1': 0, 'dim 2': 1}},
    {'key': 'c1', 'comment': True, 'text': 'which two axes swap', 'pos': (340, 205)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (340, 330), 'w': 230, 'h': 260,
     'props': HM(210, 210, 6, fmt='none')},
] + info_list('i', 30, 620) + info_list('o', 340, 620) + [
    {'key': 'c2', 'comment': True, 'text': 'left: in. right: transposed',
     'pos': (30, 760)},
] + see_also(810, ['t.t', 't.permute', 't.adjoint', 't.view', 'np.transpose'])
links = [('rs', 'F', 'hin', 'y'), ('rs', 'F', 'tr', 'tensor in'),
         ('tr', 'permuted tensor out', 'hout', 'y')] + info_links('i', 'rs', 'F') + \
    info_links('o', 'tr', 'permuted tensor out')
print(build('t.transpose', 't.transpose - swap two axes', body,
            demo, links, demo_width=620, text_width=760, text_height=820,
            subpatch=F_SUB))

# ========================================================================= t.t
body = """Transposes a 2-D tensor: rows become columns.

The short form of t.transpose 0 1, for the common case. A 7 by 6 tensor
becomes 6 by 7; row 0 becomes column 0, so a picture is flipped across
its diagonal.

SYNTAX:
t.t
No arguments.

INPUTS:

tensor in:
The tensor (a list or numpy array becomes a float32 tensor first).
Receiving it makes the node send.
A 1-D tensor (or a single number) comes back unchanged - a row has no
columns to swap with. A tensor with 3 or more axes is an error: the
console prints 't() expects a tensor with <= 2 dimensions' and nothing
is sent. Use t.transpose or t.adjoint for those.

OUTPUTS:

tensor out:
The transposed tensor. Its memory is not moved, so it is not
contiguous: t.view refuses it (use t.reshape, or t.contiguous first).

READING THE DEMO:
The F (7 by 6) turned through t.t lies on its side, 6 by 7, its top bar
now down the left. t.info shows the shapes swap. Put the result through
a second t.t and it would stand upright again - transposing twice is no
change.

RELATED:
t.transpose swaps any two axes. t.adjoint swaps the last two axes of a
tensor of any size. t.permute reorders all the axes. np.transpose is the
numpy version."""

demo = f_source() + [
    {'key': 'c0', 'comment': True, 'text': 'the F, 7 x 6 - open make_F to see how it is made',
     'pos': (30, 160)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 300), 'w': 230, 'h': 260,
     'props': HM(210, 210, 7, fmt='none')},
    {'key': 'tt', 'init': 't.t', 'pos': (340, 210), 'w': 140, 'h': 70},
    {'key': 'hout', 'init': 'heat_map', 'pos': (340, 300), 'w': 230, 'h': 260,
     'props': HM(210, 210, 6, fmt='none')},
] + info_list('i', 30, 590) + info_list('o', 340, 590) + [
    {'key': 'c2', 'comment': True, 'text': 'left: in. right: rows and columns swapped',
     'pos': (30, 730)},
] + see_also(780, ['t.transpose', 't.permute', 't.adjoint', 'np.transpose'])
links = [('rs', 'F', 'hin', 'y'), ('rs', 'F', 'tt', 'tensor in'),
         ('tt', 'tensor out', 'hout', 'y')] + info_links('i', 'rs', 'F') + \
    info_links('o', 'tt', 'tensor out')
print(build('t.t', 't.t - transpose a 2-D tensor', body,
            demo, links, demo_width=620, text_width=760, text_height=640,
            subpatch=F_SUB))

# =================================================================== t.adjoint
body = """Swaps the last two axes of a tensor, and for complex numbers also
takes the conjugate.

On an ordinary (real) tensor it is a transpose of the LAST TWO axes,
whatever the tensor's size: a 7 by 6 becomes 6 by 7, and a stack of
pictures [2, 7, 6] becomes [2, 6, 7] - every picture transposed, the
stack left in order. That is what t.t cannot do (it only takes 2-D
tensors), which makes t.adjoint the way to transpose a whole batch of
matrices.

For a complex tensor it also flips the sign of every imaginary part (the
"conjugate transpose", or Hermitian adjoint, of linear algebra). That
only matters if you are working with complex numbers, from t.fft for
instance.

SYNTAX:
t.adjoint
No arguments.

INPUTS:

tensor in:
The tensor (a list or numpy array becomes a float32 tensor first).
Receiving it makes the node send. It needs at least 2 axes: a 1-D
tensor is an error ('only supported on matrices or batches of
matrices' is printed, nothing is sent).

OUTPUTS:

tensor out:
The tensor with its last two axes swapped (and conjugated if complex).
Like t.t, the memory is not moved, so it is not contiguous.

READING THE DEMO:
make_layers holds two pictures as one tensor [2, 7, 6]: the F and a
ramp. The left heat_map shows them one above the other. t.adjoint gives
[2, 6, 7] (t.info), and the right heat_map shows both again, one above
the other - each turned on its side, the F still on top. (The two
t.reshape nodes only lay the layers out for display.)

RELATED:
t.t transposes a 2-D tensor. t.transpose swaps any two axes you name;
t.transpose -2 -1 is t.adjoint without the conjugate. t.permute
reorders all the axes. np.transpose is the numpy version."""

demo = [
    {'key': 'ly', 'init': 'p make_layers', 'pos': (30, 62), 'w': 180, 'h': 70},
    {'key': 'c0', 'comment': True, 'text': 'two pictures as layers: [2, 7, 6]',
     'pos': (30, 160)},
    {'key': 'r14', 'init': 't.reshape 14 6', 'pos': (30, 200), 'w': 280, 'h': 70},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 290), 'w': 200, 'h': 380,
     'props': HM(180, 330, 14, fmt='none')},
    {'key': 'c1', 'comment': True, 'text': 'in: layer 0 above layer 1', 'pos': (30, 690)},
    {'key': 'adj', 'init': 't.adjoint', 'pos': (340, 62), 'w': 150, 'h': 70},
    {'key': 'r12', 'init': 't.reshape 12 7', 'pos': (340, 200), 'w': 280, 'h': 70},
    {'key': 'hout', 'init': 'heat_map', 'pos': (340, 290), 'w': 230, 'h': 340,
     'props': HM(210, 290, 12, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 'out: each layer transposed', 'pos': (340, 690)},
] + info_list('o', 340, 730) + see_also(880, ['t.t', 't.transpose', 't.permute',
                                              'np.transpose'])
links = [('ly', 'layers', 'r14', 'tensor in'), ('r14', 'output', 'hin', 'y'),
         ('ly', 'layers', 'adj', 'tensor in'), ('adj', 'tensor out', 'r12', 'tensor in'),
         ('r12', 'output', 'hout', 'y')] + info_links('o', 'adj', 'tensor out')
print(build('t.adjoint', 't.adjoint - swap the last two axes (and conjugate)', body,
            demo, links, demo_width=660, text_width=760, text_height=840,
            subpatch=LAYERS_SUB))


# ---------------------------------------------------------------- joining
def two_pictures():
    """The F (make_F, left) and the ramp (make_ramp, right), each shown."""
    return f_source() + [
        {'key': 'c0', 'comment': True, 'text': 'tensor 1: the F, 7 x 6', 'pos': (30, 160)},
        {'key': 'rs2', 'init': 'p make_ramp', 'pos': (330, 62), 'w': 180, 'h': 70},
        {'key': 'c1', 'comment': True, 'text': 'tensor 2: a ramp, 7 x 6, made once',
         'pos': (330, 160)},
        {'key': 'hin', 'init': 'heat_map', 'pos': (30, 200), 'w': 200, 'h': 260,
         'props': F_HM},
        {'key': 'hin2', 'init': 'heat_map', 'pos': (330, 200), 'w': 200, 'h': 260,
         'props': F_HM},
    ]


def two_links(node):
    """Ramp into tensor 2 first (it is stored), then the F into tensor 1."""
    return [('rs2', 'ramp', 'hin2', 'y'), ('rs2', 'ramp', node, 'tensor 2'),
            ('rs', 'F', 'hin', 'y'), ('rs', 'F', node, 'tensor 1')]


INPUTS_COUNT = """tensor 1:
The first tensor - it goes first (on top, or on the left). Receiving it
makes the node join and send. A list or numpy array becomes a float32
tensor.

tensor 2, tensor 3, ...:
The others, in order. They are only stored, so they must arrive BEFORE
tensor 1. Each is converted to tensor 1's dtype and device."""

# ======================================================================= t.cat
body = """Joins tensors end to end along an axis they already have.

Use it to put two pictures side by side or one above the other, to
append new rows to a table, or to join two lists into one longer list.
The result has the same number of axes as the inputs - it is just longer
along the one you join on. (To keep the tensors apart as layers of a new
axis, use t.stack.)

THE SHAPES MUST FIT:
The tensors must match in every axis except the one being joined: two
7 by 6 tensors join along dim 0 into 14 by 6, or along dim 1 into
7 by 12. A 7 by 6 and a 3 by 6 join along dim 0 (10 by 6), but not
along dim 1.

SYNTAX:
t.cat <number of inputs: int>
Default 2. The argument is how many tensor inlets the node has, NOT the
dim: t.cat 3 joins three tensors. (t.cat 0 or t.cat 1 gives a node with
only one inlet, which just passes its tensor through.) The dim is set
in the dim box.

EXAMPLE:
t.cat 2

INPUTS and PARAMETERS:

""" + INPUTS_COUNT + """

dim (int box, also an inlet):
The axis to join along: 0 (the default) joins below - more rows; 1 joins
to the right - more columns. Negative counts from the end. A dim the
tensors do not have sends nothing. Changing it does NOT send by itself;
in the demo a metro inside make_F re-sends the F ten times a second.

OUTPUTS:

output:
The joined tensor.

IF A SHAPE DOES NOT FIT:
Unlike np.concatenate, t.cat does not stop: a tensor whose shape does
not fit is LEFT OUT, silently, and the rest are joined. With two inputs
that means tensor 1 comes out alone, unchanged - so if the output is
the same size as tensor 1, check the other's shape. The same happens
before anything has reached tensor 2.

READING THE DEMO:
tensor 1 is the F (7 by 6), tensor 2 a ramp of the same shape. With
dim 0 the ramp hangs below the F (14 by 6); type 1 into dim and it sits
to its right (7 by 12). t.info shows the shape.

RELATED:
t.stack joins along a NEW axis. t.vstack and t.hstack are t.cat with
the axis chosen for you. t.chunk and t.tensor_split cut a tensor into
pieces - the reverse. np.concatenate is the numpy version."""

demo = two_pictures() + [
    {'key': 'cc', 'init': 't.cat', 'pos': (30, 490), 'w': 180, 'h': 120,
     'props': {'dim': 0}},
    {'key': 'c2', 'comment': True, 'text': 'type 0 or 1 into dim', 'pos': (30, 495)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 640), 'w': 330, 'h': 350,
     'props': HM(310, 300, 14, fmt='none')},
] + info_list('o', 390, 640) + [
    {'key': 'c3', 'comment': True, 'text': 'the joined tensor and its shape',
     'pos': (390, 780)},
] + see_also(1020, ['t.stack', 't.vstack', 't.hstack', 't.chunk', 'np.concatenate'])
links = two_links('cc') + [('cc', 'output', 'hout', 'y')] + info_links('o', 'cc', 'output')
print(build('t.cat', 't.cat - join tensors along an axis they have', body,
            demo, links, demo_width=660, text_width=760, text_height=1000,
            subpatch=[F_SUB, RAMP_SUB]))

# ===================================================================== t.stack
body = """Piles tensors of the same shape on top of each other, along a NEW
axis.

The result has one more axis than the inputs, and the tensors stay
separate inside it: stacking two rows of 8 gives a 2 by 8 tensor - row 0
is the first, row 1 the second. Use it to turn several signals into one
table, to pair up x and y lists, or to build a batch out of single
frames.

STACK OR CAT?
Joining two rows of 8 with t.cat gives ONE row of 16 - the second
carries straight on from the first. With t.stack it gives 2 rows of 8,
the originals kept apart. cat extends an axis; stack adds one. The demo
does both to the same two rows.

SYNTAX:
t.stack <number of inputs: int>
Default 2. The argument is how many tensor inlets the node has, NOT the
dim (which is set in the dim box).

EXAMPLE:
t.stack 3

INPUTS and PARAMETERS:

""" + INPUTS_COUNT + """
Each must have exactly the same shape as tensor 1. One that does not is
LEFT OUT, silently, and the rest are stacked - so a stack that comes out
with fewer layers than inputs means a shape did not match. Before
anything reaches tensor 2, tensor 1 is stacked alone (a new axis of
length 1).

dim (int box, also an inlet):
Where the new axis goes. 0 (the default) puts it first: the tensors
become the rows (for 1-D inputs) or the layers (for 2-D inputs). The
new axis can also go last - for two rows of 8, dim 1 gives 8 rows of 2,
the inputs becoming the columns. -1 also means last. Changing it does
NOT send by itself; in the demo the metro re-sends.

OUTPUTS:

output:
The stacked tensor, with one more axis than the inputs.

READING THE DEMO:
tensor 1 is a ramp rising across 8 cells (drag stop on its t.linspace);
tensor 2 the same ramp falling, made once when the patch opens.
t.stack gives a 2 by 8 picture - rising on top, falling below; type 1
into dim and it is 8 by 2, the two side by side as columns. t.cat of the
same two gives a single strip of 16. Keep steps at 8: the two must have
the same shape.

RELATED:
t.cat joins along an existing axis. t.unsqueeze adds a length-1 axis to
a single tensor. t.vstack and t.dstack stack 2-D pictures by rows or in
depth. t.chunk takes a tensor apart. np.stack is the numpy version."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70, 'props': METRO},
    {'key': 'lsa', 'init': 't.linspace 0 1 8', 'pos': (30, 150), 'w': 220, 'h': 200,
     'props': {'start': 0.0, 'stop': 1.0, 'steps': 8}},
    {'key': 'c0', 'comment': True, 'text': 'tensor 1: rising. drag stop', 'pos': (30, 155)},
    {'key': 'lb', 'init': 'load_bang', 'pos': (330, 62), 'w': 88, 'h': 46},
    {'key': 'lsb', 'init': 't.linspace 1 0 8', 'pos': (330, 150), 'w': 220, 'h': 200,
     'props': {'start': 1.0, 'stop': 0.0, 'steps': 8}},
    {'key': 'c1', 'comment': True, 'text': 'tensor 2: falling, made once', 'pos': (330, 155)},
    {'key': 'st', 'init': 't.stack', 'pos': (30, 380), 'w': 180, 'h': 120,
     'props': {'dim': 0}},
    {'key': 'c2', 'comment': True, 'text': 'type 0 or 1 into dim', 'pos': (30, 385)},
    {'key': 'hst', 'init': 'heat_map', 'pos': (30, 520), 'w': 540, 'h': 170,
     'props': HM(520, 120, 2)},
    {'key': 'c3', 'comment': True, 'text': 'stack: 2 rows of 8, kept apart',
     'pos': (30, 525)},
] + info_list('s', 30, 710) + [
    {'key': 'cc', 'init': 't.cat', 'pos': (30, 850), 'w': 180, 'h': 120,
     'props': {'dim': 0}},
    {'key': 'c4', 'comment': True, 'text': 'for contrast: the same two, through t.cat',
     'pos': (30, 855)},
    {'key': 'hcc', 'init': 'heat_map', 'pos': (30, 990), 'w': 540, 'h': 90,
     'props': HM(520, 40, 1, fmt='none')},
] + info_list('c', 30, 1100) + [
    {'key': 'c5', 'comment': True, 'text': 'cat: one row of 16', 'pos': (330, 1105)},
] + see_also(1250, ['t.cat', 't.unsqueeze', 't.vstack', 't.chunk', 'np.stack'])
links = [('lb', 'out', 'lsb', '###input'),
         ('lsb', 'linspace out', 'st', 'tensor 2'), ('lsb', 'linspace out', 'cc', 'tensor 2'),
         ('met', '', 'lsa', '###input'),
         ('lsa', 'linspace out', 'st', 'tensor 1'), ('lsa', 'linspace out', 'cc', 'tensor 1'),
         ('st', 'output', 'hst', 'y'), ('cc', 'output', 'hcc', 'y')] + \
    info_links('s', 'st', 'output') + info_links('c', 'cc', 'output')
print(build('t.stack', 't.stack - join tensors along a new axis', body,
            demo, links, demo_width=620, text_width=760, text_height=1080))


# ------------------------------------------------- the named stacks
def stack_inputs(extra=''):
    return """tensor 1:
The first tensor - it goes first. Receiving it makes the node join and
send. A list or numpy array becomes a float32 tensor.

tensor 2, tensor 3, ...:
The others, in order. They are only stored, so they must arrive BEFORE
tensor 1. Each is converted to tensor 1's dtype and device. Until
something has reached tensor 2, tensor 1 is joined with nothing and
comes out on its own.

ALL THE SAME SHAPE:
This node insists that every tensor has exactly the same shape as
tensor 1 (torch itself is looser). If one does not, nothing is sent at
all; with the app's verbose setting on, a message says so.""" + extra


STACK_SYNTAX = """SYNTAX:
{name} <number of inputs: int>
Default 2. The argument is how many tensor inlets the node has; there
is no dim to set - the name chooses the axis."""

# ==================================================================== t.hstack
body = """Joins tensors side by side: along the columns for 2-D tensors, end to
end for 1-D ones.

For two 7 by 6 pictures the result is 7 by 12 - the second picture to
the right of the first. It is t.cat with dim 1 for anything of 2 or more
axes, and t.cat with dim 0 for rows (1-D), which is the reading of
"horizontal" in each case: two rows of 8 become one row of 16.

""" + STACK_SYNTAX.format(name='t.hstack') + """

EXAMPLE:
t.hstack

INPUTS:

""" + stack_inputs() + """

OUTPUTS:

stacked tensors:
The joined tensor: same number of axes, wider.

READING THE DEMO:
tensor 1 is the F (7 by 6), tensor 2 a ramp of the same shape.
t.hstack puts the ramp to the right of the F: 7 by 12 (t.info).

RELATED:
t.vstack joins one above the other; t.dstack in depth. t.column_stack
is meant to turn rows into columns (see its page for what this node
does). t.cat lets you choose the axis. np.concatenate is the numpy
version."""

demo = two_pictures() + [
    {'key': 'hs', 'init': 't.hstack', 'pos': (30, 490), 'w': 180, 'h': 100},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 620), 'w': 440, 'h': 260,
     'props': HM(420, 210, 7, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 'side by side: 7 x 12', 'pos': (30, 625)},
] + info_list('o', 30, 900) + see_also(1050, ['t.vstack', 't.dstack', 't.column_stack',
                                              't.cat', 'np.concatenate'])
links = two_links('hs') + [('hs', 'stacked tensors', 'hout', 'y')] + \
    info_links('o', 'hs', 'stacked tensors')
print(build('t.hstack', 't.hstack - join side by side', body,
            demo, links, demo_width=620, text_width=760, text_height=760,
            subpatch=[F_SUB, RAMP_SUB]))

# ==================================================================== t.vstack
body = """Joins tensors one above the other, along the rows.

t.vstack and t.row_stack are two names for the same node: both run
torch.vstack.

For two 7 by 6 pictures the result is 14 by 6 - the second below the
first. It is t.cat with dim 0 for tensors of 2 or more axes. Rows (1-D
tensors) are first made into tables of one row, so two rows of 8 become
a 2 by 8 table - the same as t.stack would give, which is why it is
called row_stack.

""" + STACK_SYNTAX.format(name='t.vstack') + """
t.row_stack takes the same argument.

EXAMPLE:
t.vstack

INPUTS:

""" + stack_inputs() + """

OUTPUTS:

stacked tensors:
The joined tensor, taller.

READING THE DEMO:
tensor 1 is the F (7 by 6), tensor 2 a ramp of the same shape.
t.vstack hangs the ramp below the F: 14 by 6 (t.info).

RELATED:
t.hstack joins side by side; t.dstack in depth. t.cat lets you choose
the axis, t.stack adds a new one. np.concatenate and np.stack are the
numpy versions."""

demo = two_pictures() + [
    {'key': 'vs', 'init': 't.vstack', 'pos': (30, 490), 'w': 180, 'h': 100},
    {'key': 'hout', 'init': 'heat_map', 'pos': (30, 620), 'w': 220, 'h': 440,
     'props': HM(200, 390, 14, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 'one above the other: 14 x 6', 'pos': (30, 625)},
] + info_list('o', 300, 620) + see_also(1090, ['t.hstack', 't.dstack', 't.stack', 't.cat',
                                               'np.concatenate'])
links = two_links('vs') + [('vs', 'stacked tensors', 'hout', 'y')] + \
    info_links('o', 'vs', 'stacked tensors')
print(build('t.vstack', 't.vstack / t.row_stack - join one above the other', body,
            demo, links, demo_width=620, text_width=760, text_height=760,
            subpatch=[F_SUB, RAMP_SUB]))

# ==================================================================== t.dstack
body = """Joins tensors in DEPTH: along a third axis, so every cell holds one
value from each input.

Two 7 by 6 pictures become one 7 by 6 by 2 tensor: the same rows and
columns, with two values in each cell - like an image with two colour
channels. Three greyscale pictures dstacked are an RGB image of shape
[height, width, 3]. Rows (1-D) and 2-D tables are first given the extra
axes they need, so two rows of 8 become [1, 8, 2].

""" + STACK_SYNTAX.format(name='t.dstack') + """

EXAMPLE:
t.dstack 3

INPUTS:

""" + stack_inputs() + """

OUTPUTS:

stacked tensors:
The joined tensor, with the inputs along the third axis (axis 2).

READING THE DEMO:
tensor 1 is the F, tensor 2 a ramp, both 7 by 6. t.dstack makes
[7, 6, 2] (t.info) - one picture with two values per cell, which a
heat_map cannot draw directly. The two lower heat_maps take it apart
again with t[] (:, :, 0 is the first value of every cell, :, :, 1 the
second): the F and the ramp, each intact - dstack keeps the pictures
apart cell by cell, it does not mix them.

RELATED:
t.hstack and t.vstack join side by side and one above the other.
t.stack with dim 2 does the same for 2-D tensors, and lets you choose
the axis. t.permute moves that channel axis to the front if a node
wants channels first. np.stack is the numpy version."""

demo = two_pictures() + [
    {'key': 'ds', 'init': 't.dstack', 'pos': (30, 490), 'w': 180, 'h': 100},
] + info_list('o', 300, 490) + [
    {'key': 'c2', 'comment': True, 'text': 'one picture, two values per cell',
     'pos': (300, 620)},
    {'key': 'g0', 'init': 't[] :, :, 0', 'pos': (30, 670), 'w': 280, 'h': 70,
     'props': {'indices': ':, :, 0'}},
    {'key': 'g1', 'init': 't[] :, :, 1', 'pos': (330, 670), 'w': 280, 'h': 70,
     'props': {'indices': ':, :, 1'}},
    {'key': 'h0', 'init': 'heat_map', 'pos': (30, 760), 'w': 200, 'h': 260,
     'props': F_HM},
    {'key': 'h1', 'init': 'heat_map', 'pos': (330, 760), 'w': 200, 'h': 260,
     'props': F_HM},
    {'key': 'c3', 'comment': True, 'text': 'depth 0 and depth 1, taken out again',
     'pos': (30, 1040)},
] + see_also(1090, ['t.hstack', 't.vstack', 't.stack', 't.permute', 'np.stack'])
links = two_links('ds') + [('ds', 'stacked tensors', 'g0', 'tensor in'),
                           ('ds', 'stacked tensors', 'g1', 'tensor in'),
                           ('g0', 'output', 'h0', 'y'), ('g1', 'output', 'h1', 'y')] + \
    info_links('o', 'ds', 'stacked tensors')
print(build('t.dstack', 't.dstack - join in depth', body,
            demo, links, demo_width=660, text_width=760, text_height=860,
            subpatch=[F_SUB, RAMP_SUB]))

# =============================================================== t.column_stack
body = """Meant to join rows as COLUMNS - but at present it does exactly what
t.hstack does.

torch.column_stack turns each 1-D input into a column and puts the
columns side by side: two rows of 8 should give an 8 by 2 table, one
input per column. That is the usual way to pair up x and y lists into
points.

WHAT THIS NODE ACTUALLY DOES:
The node is built on t.hstack's code and never switches to
torch.column_stack, so it runs torch.hstack. For 2-D inputs the two
agree (two 7 by 6 pictures become 7 by 12 either way). For 1-D inputs
they do not: two rows of 8 come out as ONE row of 16, not 8 by 2. Until
that is fixed, use t.stack with dim 1 to get the columns - as the demo
does.

""" + STACK_SYNTAX.format(name='t.column_stack') + """

EXAMPLE:
t.column_stack

INPUTS:

""" + stack_inputs() + """

OUTPUTS:

stacked tensors:
The joined tensor (as t.hstack, above).

READING THE DEMO:
tensor 1 is a ramp rising across 8 cells, tensor 2 the same falling.
t.column_stack gives one strip of 16 - the t.hstack result. t.stack
with dim 1 gives what column_stack is supposed to: 8 rows of 2, rising
on the left and falling on the right.

RELATED:
t.hstack is what this node runs. t.stack (dim 1) pairs rows into
columns. t.vstack pairs them as rows. np.stack (axis 1) is the numpy
way to make the columns."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70, 'props': METRO},
    {'key': 'lsa', 'init': 't.linspace 0 1 8', 'pos': (30, 150), 'w': 220, 'h': 200,
     'props': {'start': 0.0, 'stop': 1.0, 'steps': 8}},
    {'key': 'c0', 'comment': True, 'text': 'tensor 1: rising', 'pos': (30, 155)},
    {'key': 'lb', 'init': 'load_bang', 'pos': (330, 62), 'w': 88, 'h': 46},
    {'key': 'lsb', 'init': 't.linspace 1 0 8', 'pos': (330, 150), 'w': 220, 'h': 200,
     'props': {'start': 1.0, 'stop': 0.0, 'steps': 8}},
    {'key': 'c1', 'comment': True, 'text': 'tensor 2: falling, made once', 'pos': (330, 155)},
    {'key': 'cs', 'init': 't.column_stack', 'pos': (30, 380), 'w': 180, 'h': 100},
    {'key': 'hcs', 'init': 'heat_map', 'pos': (30, 500), 'w': 540, 'h': 90,
     'props': HM(520, 40, 1, fmt='none')},
    {'key': 'c2', 'comment': True, 'text': 't.column_stack: one row of 16 (the hstack result)',
     'pos': (30, 505)},
] + info_list('a', 300, 380) + [
    {'key': 'st', 'init': 't.stack', 'pos': (30, 620), 'w': 180, 'h': 120,
     'props': {'dim': 1}},
    {'key': 'c3', 'comment': True, 'text': 't.stack, dim 1: the columns you wanted',
     'pos': (30, 625)},
    {'key': 'hst', 'init': 'heat_map', 'pos': (30, 760), 'w': 200, 'h': 300,
     'props': HM(180, 250, 8)},
] + info_list('b', 300, 760) + see_also(1090, ['t.hstack', 't.stack', 't.vstack', 'np.stack'])
links = [('lb', 'out', 'lsb', '###input'),
         ('lsb', 'linspace out', 'cs', 'tensor 2'), ('lsb', 'linspace out', 'st', 'tensor 2'),
         ('met', '', 'lsa', '###input'),
         ('lsa', 'linspace out', 'cs', 'tensor 1'), ('lsa', 'linspace out', 'st', 'tensor 1'),
         ('cs', 'stacked tensors', 'hcs', 'y'), ('st', 'output', 'hst', 'y')] + \
    info_links('a', 'cs', 'stacked tensors') + info_links('b', 'st', 'output')
print(build('t.column_stack', 't.column_stack - rows as columns (currently as t.hstack)',
            body, demo, links, demo_width=620, text_width=760, text_height=860))


# ------------------------------------------------- splitting
def split_body(name, other, sizes7, how, contrast):
    return f"""Cuts a tensor into a number of pieces along one axis, and sends each
piece from its own outlet.

Use it to separate a tensor that holds several things side by side - the
left and right halves of a picture, the x, y and z columns of a list of
points, the first and second half of a recording.

{how}

THE NUMBER OF PIECES IS FIXED:
It is the first argument, and it decides how many outlets the node has
(tensor 0, tensor 1, ...), so it cannot change on a live node - retype
the node to change it. The 'split' box among the node's options shows
the count, but editing it does nothing.

SYNTAX:
{name} <pieces: int> <dim: int>
Defaults: 2 pieces, dim 0.

EXAMPLE:
{name} 3

INPUTS and PARAMETERS:

tensor in:
The tensor (a list or numpy array becomes a float32 tensor first).
Every tensor that arrives is cut and the pieces sent, tensor 0 first.
Only this inlet makes the node send.

dim (int box, also an inlet):
The axis to cut across: 0 cuts the rows into groups (horizontal cuts),
1 cuts the columns into groups (vertical cuts). Negative counts from the
end. If the tensor does not have that axis, or the axis is shorter than
the number of pieces, nothing is sent. Changing it does NOT send by
itself; in the demo a metro inside make_F re-sends the F ten times a
second.

OUTPUTS:

tensor 0, tensor 1, ...:
The pieces, in order. Each has as many axes as the input; only its
length along dim is smaller. Each is a view of the input, not a copy.

READING THE DEMO:
{name} 3 cuts the 7 by 6 F into three bands of rows: {sizes7}. t.info
shows each piece's shape. Type 1 into dim and it cuts the 6 columns
instead, three slices 2 columns wide: the stem, then the two halves of
the arms.

{contrast}"""


def split_demo(node):
    return f_source() + [
        {'key': 'c0', 'comment': True, 'text': 'the F, 7 x 6 - open make_F to see how it is made',
         'pos': (30, 160)},
        {'key': 'hin', 'init': 'heat_map', 'pos': (380, 62), 'w': 200, 'h': 260,
         'props': F_HM},
        {'key': 'sp', 'init': node + ' 3', 'pos': (30, 350), 'w': 180, 'h': 140,
         'props': {'dim': 0}},
        {'key': 'c1', 'comment': True, 'text': 'type 0 or 1 into dim', 'pos': (30, 355)},
        {'key': 'h0', 'init': 'heat_map', 'pos': (30, 510), 'w': 190, 'h': 190,
         'props': HM(170, 140, 3, fmt='none')},
        {'key': 'h1', 'init': 'heat_map', 'pos': (240, 510), 'w': 190, 'h': 190,
         'props': HM(170, 140, 3, fmt='none')},
        {'key': 'h2', 'init': 'heat_map', 'pos': (450, 510), 'w': 190, 'h': 190,
         'props': HM(170, 140, 1, fmt='none')},
        {'key': 'i0', 'init': 't.info', 'pos': (30, 720), 'w': 120, 'h': 120},
        {'key': 'i1', 'init': 't.info', 'pos': (240, 720), 'w': 120, 'h': 120},
        {'key': 'i2', 'init': 't.info', 'pos': (450, 720), 'w': 120, 'h': 120},
        {'key': 'l0', 'init': 'list', 'pos': (30, 860), 'w': 150, 'h': 42, 'props': LIST},
        {'key': 'l1', 'init': 'list', 'pos': (240, 860), 'w': 150, 'h': 42, 'props': LIST},
        {'key': 'l2', 'init': 'list', 'pos': (450, 860), 'w': 150, 'h': 42, 'props': LIST},
        {'key': 'c2', 'comment': True, 'text': 'tensor 0, tensor 1, tensor 2 and their shapes',
         'pos': (30, 920)},
    ]


def split_links():
    out = [('rs', 'F', 'hin', 'y'), ('rs', 'F', 'sp', 'tensor in')]
    for i in range(3):
        out += [('sp', f'tensor {i}', f'h{i}', 'y'), ('sp', f'tensor {i}', f'i{i}', 'in'),
                (f'i{i}', 'shape', f'l{i}', '')]
    return out


# ===================================================================== t.chunk
body = split_body(
    't.chunk', 't.tensor_split',
    'the top 3 rows (the bar and the start of the stem), the next 3 (the '
    'middle arm with the stem above and below it), and a last piece of a '
    'single row',
    """HOW THE PIECES ARE SIZED:
t.chunk makes every piece the same size - the length divided by the
number of pieces, rounded UP - and the last piece takes what is left.
So 7 rows in 3 pieces are 3, 3 and 1. The surprise: it can give FEWER
pieces than you asked for. 6 rows in 4 pieces are 2, 2 and 2 - only
three pieces, because a fourth would be empty - and the unused outlet
sends nothing (its display keeps whatever it showed before).
t.tensor_split never does that: it spreads the remainder over the first
pieces (7 in 3 gives 3, 2, 2) and always gives the count you asked for.""",
    """RELATED:
t.tensor_split cuts into pieces as even as possible (3, 2, 2 here).
t.cat puts pieces back together. t.narrow takes out one piece by start
and length. np.split is the numpy version (np.split sizes its pieces
like t.tensor_split).""")
demo = split_demo('t.chunk') + see_also(970, ['t.tensor_split', 't.cat', 't.narrow',
                                              'np.split'])
print(build('t.chunk', 't.chunk - cut into equal pieces', body,
            demo, split_links(), demo_width=680, text_width=760, text_height=1000,
            subpatch=F_SUB))

# ============================================================== t.tensor_split
body = split_body(
    't.tensor_split', 't.chunk',
    'the top 3 rows (the bar and the start of the stem), then 2 rows (the '
    'middle arm and the stem below it), then the last 2 rows of stem',
    """HOW THE PIECES ARE SIZED:
t.tensor_split makes the pieces as even as it can, giving the first
ones one extra when the length does not divide: 7 rows in 3 pieces are
3, 2 and 2; 6 rows in 4 pieces are 2, 2, 1 and 1. It always gives
exactly the number of pieces you asked for. (t.chunk sizes them
differently - 3, 3 and 1 for the same F - and can give fewer.)
torch.tensor_split can also cut at a list of positions; this node only
takes a number of pieces.""",
    """RELATED:
t.chunk cuts into pieces of one size (3, 3, 1 here), and may give fewer.
t.cat puts pieces back together. t.narrow takes out one piece by start
and length. np.split is the numpy version, and sizes its pieces the
same way.""")
demo = split_demo('t.tensor_split') + see_also(970, ['t.chunk', 't.cat', 't.narrow',
                                                     'np.split'])
print(build('t.tensor_split', 't.tensor_split - cut into pieces as even as possible', body,
            demo, split_links(), demo_width=680, text_width=760, text_height=1000,
            subpatch=F_SUB))
