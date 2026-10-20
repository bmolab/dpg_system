"""torch tensor basics: tensor (and its alias t.to), t.detach, t.info,
t.numel, t.contiguous, t.is_contiguous, t.buffer, t.rolling_buffer -- one
page each. The tensor, t.detach, t.buffer and t.rolling_buffer pages are
hand-made now (see tools/README.md), so this script no longer builds them.
t.data_set's page is retired: the node is never registered.

These were three family pages (tensor, t.info, t.buffer) plus a t.data_set
page in make_small_help.py. Each node now has its own page whose demo shows
what the node does to a tensor: the values before and after as heat_maps,
the metadata as text, the buffers filling over time with rows = time.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import INT


def HM(width, height, count, miny=0.0, maxy=1.0, fmt='%.2f', color='viridis'):
    """Property block for a heat_map that draws a whole array as one picture."""
    return {'color': color, 'width': width, 'height': height, 'sample count': count,
            'min y': miny, 'max y': maxy, 'update_mode': 'heat_map',
            'number format': fmt}


LIST = {'text in': '', 'font size': '24'}
METRO = {'on': True, 'period': 100.0, 'units': 'milliseconds'}


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


def th(body):
    """Prose box height: about 14 px a line at text size 30, plus a margin."""
    return (body.count('\n') + 1) * 14 + 40


def shape_source(x=30, y=62):
    """A menu of shapes for t.rand, and a metro that keeps generating.

    The shapes are written 3x4 rather than '3 4': a message would send a
    list, and t.rand's shape inlet only accepts text (see the report)."""
    demo = [
        {'key': 'men', 'init': 'menu 3x4 6x8 1x16 2x3x4', 'pos': (x, y), 'w': 140, 'h': 46},
        {'key': 'cmen', 'comment': True, 'text': 'pick a shape for t.rand', 'pos': (x + 160, y + 8)},
        {'key': 'met', 'init': 'metro 300', 'pos': (x, y + 60), 'w': 129, 'h': 70,
         'props': dict(METRO, period=300.0)},
        {'key': 'rnd', 'init': 't.rand 3 4', 'pos': (x, y + 150), 'w': 230, 'h': 180,
         'props': {'shape': '[3, 4]', 'min': 0.0, 'max': 1.0}},
    ]
    links = [('men', '', 'rnd', 'shape'), ('met', '', 'rnd', '###input')]
    return demo, links


# tensor: its page is hand-made now (see tools/README.md).

# t.detach: its page is hand-made now (see tools/README.md).

# =================================================================== t.info
body = """Reports what a tensor is: its shape, its dtype, the device it lives
on, and whether it is tracking gradients - each on its own outlet.

It is the first thing to patch in when a torch node refuses a tensor or
gives a surprising result. The cause is nearly always one of these four:
the shape is not what you thought, the type is wrong (integers where
floats were expected), it is on a different device from the tensor it
meets, or it is carrying gradient history. The outlet labels change to
show the values, so the node itself is a display.

SYNTAX:
t.info

EXAMPLE:
t.info

INPUTS and PARAMETERS:

in:
The tensor. Every tensor that arrives is reported. A list or NumPy array
is turned into a float32 cpu tensor first and that is what is reported -
so t.info on a NumPy array always says float32, whatever the array's own
type. An empty tensor is reported like any other: its shape has a 0 in
it ([0] for an empty 1-D tensor). An empty list is reported the same
way, as [0].

OUTPUTS (sent right to left - grad first, shape last):

shape:
The size of each dimension, as a list: [3, 4] is 3 rows of 4.

dtype:
The type, as text: torch.float32, torch.int64 ...

device:
Where it lives, as text: cpu, or mps:0, cuda:0 for a GPU.

grad:
True if the tensor tracks gradients, False if not. The outlet's label
reads "grad" or "inference" to match.

READING THE DEMO:
A metro makes a fresh t.rand tensor three times a second; pick a shape
from the menu to change it (2x3x4 has three dimensions). The tensor then
passes through t.to, whose dtype and device menus you can change, and
whose requires_grad box you can tick. Each change appears in the four
displays and in the outlet labels.

RELATED:
t.numel gives the total number of values. t.is_contiguous tells you how
the values are laid out in memory. tensor (t.to) changes the dtype,
device and requires_grad that this reports. np.shape is the NumPy
version of the shape outlet."""

demo, links = shape_source()
demo += [
    {'key': 'to', 'init': 't.to', 'pos': (290, 212), 'w': 230, 'h': 150},
    {'key': 'cto', 'comment': True, 'text': 'change dtype, device, requires_grad here',
     'pos': (290, 372)},
    {'key': 'inf', 'init': 't.info', 'pos': (30, 410), 'w': 300, 'h': 140},
    {'key': 'lsh', 'init': 'list', 'pos': (30, 570), 'w': 140, 'h': 42, 'props': LIST},
    {'key': 'csh2', 'comment': True, 'text': 'shape', 'pos': (30, 620)},
    {'key': 'sdt', 'init': 'string', 'pos': (190, 570), 'w': 180, 'h': 42, 'props': LIST},
    {'key': 'cdt', 'comment': True, 'text': 'dtype', 'pos': (190, 620)},
    {'key': 'sdv', 'init': 'string', 'pos': (410, 570), 'w': 130, 'h': 42, 'props': LIST},
    {'key': 'cdv', 'comment': True, 'text': 'device', 'pos': (410, 620)},
    {'key': 'lg', 'init': 'string', 'pos': (570, 570), 'w': 130, 'h': 42, 'props': LIST},
    {'key': 'cg', 'comment': True, 'text': 'grad', 'pos': (570, 620)},
]
links += [('rnd', 'random tensor', 'to', 'in'), ('to', 'tensor out', 'inf', 'in'),
          ('inf', 'shape', 'lsh', ''), ('inf', 'dtype', 'sdt', ''),
          ('inf', 'device', 'sdv', ''), ('inf', 'grad', 'lg', '')]
demo += see_also(670, ['t.numel', 'tensor', 't.is_contiguous', 'np.shape'])
print(build('t.info', 't.info - shape, dtype, device and gradient tracking',
            body, demo, links, demo_width=720, text_width=760, text_height=th(body)))

# =================================================================== t.numel
body = """Counts the values in a tensor: the product of its shape.

A 3 by 4 tensor holds 12 values, a 2 by 3 by 4 one holds 24. Use it to
check that a reshape can work before trying it (the new shape must
multiply to the same number), to size a buffer to match, or to scale a
sum into an average.

SYNTAX:
t.numel

EXAMPLE:
t.numel

INPUTS and PARAMETERS:

tensor in:
The tensor. Every tensor that arrives is counted. A list or NumPy array
is turned into a tensor first, so t.numel counts a list's numbers too.
An empty tensor or an empty list gives 0.

OUTPUTS:

numel out:
The number of values, as an int.

READING THE DEMO:
A metro makes a fresh t.rand tensor three times a second. Pick a shape
from the menu: the heat_map shows the tensor (rows by columns) so you
can count the cells, t.info shows the shape, and t.numel the count - 12
for 3x4, 48 for 6x8, 16 for 1x16. 2x3x4 has three dimensions, which the
heat_map cannot draw as one picture, but t.numel still answers 24.

RELATED:
t.info gives the shape itself. t.reshape and t.view need a new shape
with the same count. np.shape gives a NumPy array's shape."""

demo, links = shape_source()
demo += [
    {'key': 'hm', 'init': 'heat_map', 'pos': (290, 212), 'w': 340, 'h': 230,
     'props': HM(320, 200, 3, fmt='none')},
    {'key': 'chm', 'comment': True, 'text': 'the tensor: count the cells',
     'pos': (290, 450)},
    {'key': 'ne', 'init': 't.numel', 'pos': (30, 410), 'w': 150, 'h': 70},
    {'key': 'ine', 'init': 'int', 'pos': (30, 500), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'cne', 'comment': True, 'text': 'how many values', 'pos': (30, 550)},
    {'key': 'inf', 'init': 't.info', 'pos': (290, 490), 'w': 300, 'h': 140},
    {'key': 'lsh', 'init': 'list', 'pos': (290, 650), 'w': 140, 'h': 42, 'props': LIST},
    {'key': 'csh2', 'comment': True, 'text': 'its shape', 'pos': (440, 657)},
]
links += [('rnd', 'random tensor', 'hm', 'y'), ('rnd', 'random tensor', 'ne', 'tensor in'),
          ('ne', 'numel out', 'ine', ''), ('rnd', 'random tensor', 'inf', 'in'),
          ('inf', 'shape', 'lsh', '')]
demo += see_also(720, ['t.info', 't.reshape', 't.view', 'np.shape'])
print(build('t.numel', 't.numel - how many values a tensor holds',
            body, demo, links, demo_width=660, text_width=760, text_height=th(body)))

# ============================================================== t.contiguous
CONTIG_TEXT = """WHAT CONTIGUOUS MEANS:
A tensor's values live in one flat run of memory, and its shape says how
to read that run as rows and columns. A fresh tensor is stored row after
row, in order - contiguous. t.t, t.transpose, t.permute and slicing out
some columns (t.narrow on dimension 1) do not move any values: they
change how the same memory is read, which is why they cost nothing at
any size. The result shows the right numbers, but they are no longer
stored row after row - it is not contiguous."""

body = """Makes a tensor contiguous: if its values are not stored row after
row in memory, it copies them so that they are. The numbers, and the
shape, do not change.

Most operations do not care how a tensor is stored. t.view does: it
only reinterprets memory, so it fails on a tensor that is not contiguous
(after a transpose, for instance), and some other code - exporting the
data, handing it to a library - expects contiguous memory too. Put
t.contiguous just before such a node.

""" + CONTIG_TEXT + """

SYNTAX:
t.contiguous

EXAMPLE:
t.contiguous

INPUTS and PARAMETERS:

tensor in:
The tensor. Every tensor that arrives is sent on. If it is already
contiguous it is passed through unchanged, with no copy; otherwise a
contiguous copy is sent. A list or NumPy array is turned into a tensor
first (and is then contiguous already).

OUTPUTS:

tensor out:
The same numbers and shape, stored row after row.

READING THE DEMO:
t.rand makes a 4 by 8 tensor (click t.rand's own button for new values)
and t.t transposes it to 8 by 4. The left picture is the transposed
tensor; t.is_contiguous says False. t.view -1, asked to lay it out as
one row, fails (the console prints the reason) and its strip stays
empty. After t.contiguous the picture is identical, t.is_contiguous says
True, and the same t.view -1 now works.

RELATED:
t.is_contiguous only asks the question. t.reshape is t.view that copies
by itself when it has to, so it works on anything. t.t and t.transpose
make non-contiguous tensors. NumPy has no node for this: np nodes
take care of memory layout themselves."""


def contig_demo():
    demo = [
        {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
        {'key': 'rnd', 'init': 't.rand 4 8', 'pos': (30, 130), 'w': 230, 'h': 180,
         'props': {'shape': '[4, 8]', 'min': 0.0, 'max': 1.0}},
        {'key': 'tt', 'init': 't.t', 'pos': (30, 330), 'w': 120, 'h': 70},
        {'key': 'ctt', 'comment': True, 'text': 'transposed: 8 rows of 4',
         'pos': (170, 340)},
        {'key': 'h1', 'init': 'heat_map', 'pos': (30, 420), 'w': 180, 'h': 290,
         'props': HM(160, 260, 8, fmt='none')},
        {'key': 'ic1', 'init': 't.is_contiguous', 'pos': (230, 420), 'w': 200, 'h': 70},
        {'key': 'l1', 'init': 'string', 'pos': (230, 505), 'w': 90, 'h': 42, 'props': LIST},
        {'key': 'v1', 'init': 't.view -1', 'pos': (230, 570), 'w': 260, 'h': 100},
        {'key': 'cv1', 'comment': True, 'text': 'fails: nothing comes out',
         'pos': (230, 680)},
        {'key': 'ct', 'init': 't.contiguous', 'pos': (30, 740), 'w': 180, 'h': 70},
        {'key': 'cct', 'comment': True, 'text': 'copies into row-after-row order',
         'pos': (230, 750)},
        {'key': 'h2', 'init': 'heat_map', 'pos': (30, 830), 'w': 180, 'h': 290,
         'props': HM(160, 260, 8, fmt='none')},
        {'key': 'ch2', 'comment': True, 'text': 'the same picture', 'pos': (30, 1130)},
        {'key': 'ic2', 'init': 't.is_contiguous', 'pos': (230, 830), 'w': 200, 'h': 70},
        {'key': 'l2', 'init': 'string', 'pos': (230, 915), 'w': 90, 'h': 42, 'props': LIST},
        {'key': 'v2', 'init': 't.view -1', 'pos': (230, 980), 'w': 260, 'h': 100},
    ]
    strips = [
        {'key': 'hv1', 'init': 'heat_map', 'pos': (510, 570), 'w': 340, 'h': 60,
         'props': HM(320, 30, 1, fmt='none')},
        {'key': 'hv2', 'init': 'heat_map', 'pos': (510, 980), 'w': 340, 'h': 60,
         'props': HM(320, 30, 1, fmt='none')},
        {'key': 'chv2', 'comment': True, 'text': 'works: the 32 values in one row',
         'pos': (510, 1050)},
    ]
    links = [('lb', 'out', 'rnd', '###input'), ('rnd', 'random tensor', 'tt', 'tensor in'),
             ('tt', 'tensor out', 'h1', 'y'), ('tt', 'tensor out', 'ic1', 'tensor in'),
             ('ic1', 'result', 'l1', ''), ('tt', 'tensor out', 'v1', 'tensor in'),
             ('v1', 'output', 'hv1', 'y'),
             ('tt', 'tensor out', 'ct', 'tensor in'),
             ('ct', 'tensor out', 'h2', 'y'), ('ct', 'tensor out', 'ic2', 'tensor in'),
             ('ic2', 'result', 'l2', ''), ('ct', 'tensor out', 'v2', 'tensor in'),
             ('v2', 'output', 'hv2', 'y')]
    return demo + strips, links


demo, links = contig_demo()
demo += see_also(1170, ['t.is_contiguous', 't.view', 't.reshape', 't.transpose',
                        't.t'])
print(build('t.contiguous', 't.contiguous - store a tensor row after row',
            body, demo, links, demo_width=870, text_width=760, text_height=th(body)))

# =========================================================== t.is_contiguous
body = """Says whether a tensor's values are stored row after row in memory
(True) or not (False). It changes nothing.

Use it to find out whether a t.view will work, or whether a t.contiguous
in front of something would actually copy - on a large tensor that copy
is worth avoiding when it is not needed.

""" + CONTIG_TEXT + """

SYNTAX:
t.is_contiguous

EXAMPLE:
t.is_contiguous

INPUTS and PARAMETERS:

tensor in:
The tensor. Every tensor that arrives is tested and the answer sent. A
list or NumPy array is turned into a tensor first, which is always
contiguous. An empty tensor is ignored.

OUTPUTS:

result:
True or False.

READING THE DEMO:
One 4 by 8 t.rand tensor (click its own button for new values) goes
four ways, each into a t.is_contiguous:
as it is - True;
transposed by t.t - False, the same memory read column-first;
the first 3 COLUMNS cut out by t.narrow 1 - False, each row now skips
the values of the columns left out;
the first 2 ROWS cut out by t.narrow 0 - True, those rows are still one
unbroken run.
t.flip also rearranges, but it copies, so its result is True.

RELATED:
t.contiguous makes a contiguous copy. t.view needs a contiguous input;
t.reshape does not. t.info reports the other things worth knowing about
a tensor."""


def ic_branch(key, x, init, props, note, inlet='tensor in', out='tensor out'):
    d = []
    if init is not None:
        d.append({'key': key, 'init': init, 'pos': (x, 330), 'w': 170, 'h': 120,
                  'props': props})
    d += [
        {'key': key + 'h', 'init': 'heat_map', 'pos': (x, 470), 'w': 170, 'h': 150,
         'props': HM(150, 120, 4, fmt='none')},
        {'key': key + 'ic', 'init': 't.is_contiguous', 'pos': (x, 640), 'w': 170, 'h': 70},
        {'key': key + 'l', 'init': 'string', 'pos': (x, 725), 'w': 90, 'h': 42, 'props': LIST},
        {'key': key + 'c', 'comment': True, 'text': note, 'pos': (x, 780)},
    ]
    return d


demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 't.rand 4 8', 'pos': (30, 130), 'w': 230, 'h': 180,
     'props': {'shape': '[4, 8]', 'min': 0.0, 'max': 1.0}},
    {'key': 'crnd', 'comment': True, 'text': '4 rows of 8, fresh', 'pos': (280, 140)},
]
demo += ic_branch('a', 30, None, None, 'as made')
demo += ic_branch('b', 220, 't.t', {}, 'transposed')
demo += ic_branch('c', 410, 't.narrow 1', {'dim': 1, 'start': 0, 'length': 3},
                  'first 3 columns')
demo += ic_branch('d', 600, 't.narrow 0', {'dim': 0, 'start': 0, 'length': 2},
                  'first 2 rows')
demo += ic_branch('e', 790, 't.flip', {'flip dims': '1'}, 'flipped (a copy)')
links = [('lb', 'out', 'rnd', '###input'),
         ('rnd', 'random tensor', 'ah', 'y'), ('rnd', 'random tensor', 'aic', 'tensor in')]
for k, out in (('b', 'tensor out'), ('c', 'tensor out'), ('d', 'tensor out'),
               ('e', 'output')):
    links += [('rnd', 'random tensor', k, 'tensor in'), (k, out, k + 'h', 'y'),
              (k, out, k + 'ic', 'tensor in')]
for k in 'abcde':
    links.append((k + 'ic', 'result', k + 'l', ''))
demo += see_also(830, ['t.contiguous', 't.view', 't.reshape', 't.narrow', 't.info'])
print(build('t.is_contiguous', 't.is_contiguous - is it stored row after row?',
            body, demo, links, demo_width=980, text_width=760, text_height=th(body)))

# t.buffer, t.rolling_buffer: their pages are hand-made now (see tools/README.md).

# t.data_set: its page is retired (see tools/README.md).
