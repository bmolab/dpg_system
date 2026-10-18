"""np averages and spread: one page per node.

np.var (the np.mean, np.median, np.nanmean, np.nanmedian, np.std and
np.robust_mean pages are hand-made). These were part of the np_stats family page;
each now has its own page whose demo shows how its statistic DIFFERS from its
neighbours - a spread that var follows squared - with a SEE ALSO row under
it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import INT, FLT


def HM(width, height, count, miny=0.0, maxy=1.0, fmt='%.2f', color='viridis'):
    """Property block for a heat_map that draws a whole array as one picture."""
    return {'color': color, 'width': width, 'height': height, 'sample count': count,
            'min y': miny, 'max y': maxy, 'update_mode': 'heat_map',
            'number format': fmt}


LIST = {'text in': '', 'font size': '24'}


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


def FLOATW(v=None):
    p = dict(FLT, **{'width': 110})
    if v is not None:
        p[''] = v
    return p


# --------------------------------------------------------------- shared demo data

# Nine readings: eight close to 5 and one wild one (20) - a sensor glitch.
READINGS = [4.8, 5.1, 4.9, 5.2, 5.0, 4.7, 5.3, 5.1, 20.0]


def readings_vector(key='vec', pos=(30, 130)):
    """A vector of the nine readings. Every row can be dragged; each drag sends
    the whole array again."""
    props = {'[%d]' % i: v for i, v in enumerate(READINGS)}
    props.update({'component count': 9, 'number format': '%.2f',
                  'output type': 'numpy', 'width': 90})
    return {'key': key, 'init': 'vector 9', 'pos': pos, 'w': 120, 'h': 300,
            'props': props}


# Inside make_noise: 64 random values scaled so their mean is exactly 0 and
# their std exactly 1, then multiplied by 'spread' and moved by 'shift'.
NOISE_SUB = {
    'name': 'make_noise',
    'host': 'nz',
    'demo': [
        {'key': 'lb', 'init': 'load_bang', 'pos': (40, 40), 'w': 88, 'h': 46},
        {'key': 'cl', 'init': 'comment once, when the patch opens',
         'pos': (150, 50), 'w': 280, 'h': 30},
        {'key': 'rnd', 'init': 'np.rand 64', 'pos': (40, 110), 'w': 160, 'h': 140,
         'props': {'min': -1.0, 'max': 1.0, 'dim 0': 64, 'dtype': 'float32'}},
        {'key': 'tr', 'init': 't array array array', 'pos': (40, 280), 'w': 200, 'h': 46},
        {'key': 'ct', 'init': 'comment right to left: std, then mean, then the values',
         'pos': (260, 285), 'w': 440, 'h': 30},
        {'key': 'sd', 'init': 'np.std', 'pos': (420, 340), 'w': 120, 'h': 70},
        {'key': 'mn', 'init': 'np.mean', 'pos': (260, 340), 'w': 120, 'h': 70},
        {'key': 'sub', 'init': '- 0.0', 'pos': (40, 440), 'w': 120, 'h': 46},
        {'key': 'div', 'init': '/ 1.0', 'pos': (40, 510), 'w': 120, 'h': 46},
        {'key': 'cz', 'init': 'comment now mean 0, std 1',
         'pos': (180, 515), 'w': 240, 'h': 30},
        {'key': 'var', 'init': 'var np_spread_help_noise', 'pos': (40, 580), 'w': 240, 'h': 70,
         'props': {'name': 'np_spread_help_noise'}},
        {'key': 'insp', 'init': 'in spread', 'pos': (460, 440), 'w': 150, 'h': 80},
        {'key': 'insh', 'init': 'in shift', 'pos': (640, 440), 'w': 150, 'h': 80},
        {'key': 'tsp', 'init': 't bang float', 'pos': (460, 540), 'w': 120, 'h': 46},
        {'key': 'tsh', 'init': 't bang float', 'pos': (640, 540), 'w': 120, 'h': 46},
        {'key': 'mul', 'init': '* 1.0', 'pos': (40, 680), 'w': 120, 'h': 46},
        {'key': 'add', 'init': '+ 0.0', 'pos': (40, 750), 'w': 120, 'h': 46},
        {'key': 'cm', 'init': 'comment times spread, plus shift',
         'pos': (180, 715), 'w': 280, 'h': 30},
        {'key': 'out', 'init': 'out data', 'pos': (40, 830), 'w': 150, 'h': 80},
    ],
    'links': [('lb', 'out', 'rnd', ''),
              ('rnd', 'random array', 'tr', ''),
              ('tr', 'array', 'sd', 'in', 2), ('tr', 'array', 'mn', 'in', 1),
              ('tr', 'array', 'sub', 'in', 0),
              ('sd', 'std', 'div', 'operand'), ('mn', 'mean', 'sub', 'operand'),
              ('sub', 'result', 'div', 'in'), ('div', 'result', 'var', 'in'),
              ('insp', 'spread', 'tsp', ''), ('insh', 'shift', 'tsh', ''),
              ('tsp', 'float', 'mul', 'operand'), ('tsp', 'bang', 'var', 'in'),
              ('tsh', 'float', 'add', 'operand'), ('tsh', 'bang', 'var', 'in'),
              ('var', 'out', 'mul', 'in'), ('mul', 'result', 'add', 'in'),
              ('add', 'result', 'out', 'data')],
}


def noise_controls():
    """Two sliders into make_noise, and the 64 values drawn as points."""
    return [
        {'key': 'ssp', 'init': 'slider', 'pos': (30, 62), 'w': 220, 'h': 60,
         'props': {'': 1.0, 'min': 0.0, 'max': 3.0, 'format': '%.2f', 'width': 200}},
        {'key': 'csp', 'comment': True, 'text': 'spread: how far the values scatter',
         'pos': (30, 125)},
        {'key': 'ssh', 'init': 'slider', 'pos': (290, 62), 'w': 220, 'h': 60,
         'props': {'': 0.0, 'min': -3.0, 'max': 3.0, 'format': '%.2f', 'width': 200}},
        {'key': 'csh', 'comment': True, 'text': 'shift: moves them all up or down',
         'pos': (290, 125)},
        {'key': 'nz', 'init': 'p make_noise', 'pos': (30, 170), 'w': 220, 'h': 70},
        {'key': 'cnz', 'comment': True, 'text': '64 random values - open it to see how',
         'pos': (270, 180)},
        {'key': 'pl', 'init': 'plot', 'pos': (30, 270), 'w': 500, 'h': 260,
         'props': {'color': 'none', 'width': 480, 'height': 210, 'style': 'scatter',
                   'update style': 'buffer holds one sample of input',
                   'sample count': 64, 'min x': 0.0, 'max x': 64.0,
                   'min y': -6.0, 'max y': 6.0}},
    ]


NOISE_LINKS = [('ssp', '', 'nz', 'spread'), ('ssh', '', 'nz', 'shift'),
               ('nz', 'data', 'pl', 'y')]


AXIS_TEXT = """THE axis MENU:
With no axis (the default) the whole array is reduced to one number.
With axis 0 the reduction runs DOWN each column, giving one result per
column; with axis 1 it runs ALONG each row, giving one result per row.
For a table of 5 frames (rows) of 4 sensors (columns), axis 0 answers
"what did each sensor read on average", axis 1 "what did all the sensors
read on average at each frame". The axis named is the one that
disappears: a 5 by 4 array gives 4 values on axis 0, 5 on axis 1."""

AXIS_INLET = """axis (menu, also an inlet):
Which way to reduce - see THE axis MENU. Changing it does NOT send by
itself; the next array uses it. The menu offers None (the whole array)
and the axes the last array had (None and 0 for a 1-D array); a message
None into the axis inlet also returns to the whole array. An axis the
array does not have is treated as its last axis for that array only:
the choice is kept, so one 1-D array through np.mean 1 does not stop
the 2-D arrays that follow from being averaged along axis 1."""

# np.mean: its page is hand-made now (see tools/README.md).

# np.median: its page is hand-made now (see tools/README.md).

# np.nanmean and np.nanmedian: their pages are hand-made now (see tools/README.md).

# ===================================================================== spread pages
STD_DEMO_TEXT = """THE DEMO:
make_noise makes 64 random values, scaled so that at spread 1 and shift
0 their mean is exactly 0 and their standard deviation exactly 1. The
spread slider multiplies them all; the shift slider adds to them all.
The plot shows the 64 values as points."""

# np.std: its page is hand-made now (see tools/README.md); STD_DEMO_TEXT stays for np.var.

# ========================================================================= np.var
body = """Gives the variance of an array: the average of the squared distances
of its values from their mean.

The variance is the standard deviation squared. It carries the same
information as np.std - how spread out the values are - but in squared
units, so it grows much faster: double the spread and the variance goes
up four times. Reach for it where the maths wants it: variances of
independent noises add (stds do not), so combining or comparing noise
levels, weighting by 1 / variance, or fitting is done in variance. For
a number to read or a threshold to set, np.std, in the data's own units,
is usually easier.

HOW IT IS WORKED OUT:
Take the mean; take each value's distance from it; square the distances
and average them. It divides by the number of values N (the "population"
variance); some tools divide by N - 1, which gives a larger answer for
small arrays: for 1 2 3 4 np.var gives 1.25, a spreadsheet's VAR 1.667.

""" + AXIS_TEXT + """

SYNTAX:
np.var <axis: int>
With no argument there is no axis: one variance for the whole array.

EXAMPLE:
np.var

INPUTS and PARAMETERS:

in:
The array (a list or a single number is turned into one). Every array
that arrives is measured and the result sent - the only inlet that makes
the node send.

""" + AXIS_INLET.replace('np.mean 1', 'np.var 1') + """

OUTPUTS:

var:
With no axis, one number, never negative. With an axis, an array one
dimension smaller: one variance per column (axis 0) or per row (axis 1).

""" + STD_DEMO_TEXT + """

READING THE DEMO:
np.var and np.std watch the same 64 values. Drag spread to 2: the std
reads 2 and the variance 4. Spread 3: std 3, variance 9. The std through
pow 2 (squared) reads the same as np.var, always. Below spread 1 the
variance is the SMALLER of the two (0.5 squared is 0.25). Drag shift and
neither moves - moving all the values together does not spread them.

RELATED:
np.std is the square root of the variance, in the data's own units.
np.mean is the centre the distances are measured from. pow raises a
number to a power - pow 2 squares, pow 0.5 takes the square root, which
turns a variance back into a std."""

demo = noise_controls() + [
    {'key': 'vr', 'init': 'np.var', 'pos': (30, 560), 'w': 130, 'h': 70},
    {'key': 'sd', 'init': 'np.std', 'pos': (290, 560), 'w': 130, 'h': 70},
    {'key': 'fvr', 'init': 'float', 'pos': (30, 650), 'w': 130, 'h': 42, 'props': FLOATW()},
    {'key': 'fsd', 'init': 'float', 'pos': (290, 650), 'w': 130, 'h': 42, 'props': FLOATW()},
    {'key': 'cf1', 'comment': True, 'text': 'variance', 'pos': (30, 700)},
    {'key': 'cf2', 'comment': True, 'text': 'standard deviation', 'pos': (290, 700)},
    {'key': 'sq', 'init': 'pow 2', 'pos': (290, 740), 'w': 120, 'h': 46},
    {'key': 'fsq', 'init': 'float', 'pos': (290, 810), 'w': 130, 'h': 42, 'props': FLOATW()},
    {'key': 'csq', 'comment': True, 'text': 'std squared - the same as var', 'pos': (440, 815)},
] + see_also(890, ['np.std', 'np.mean', 'pow'])
links = NOISE_LINKS + [('nz', 'data', 'vr', 'in'), ('nz', 'data', 'sd', 'in'),
                       ('vr', 'var', 'fvr', ''), ('sd', 'std', 'fsd', ''),
                       ('sd', 'std', 'sq', 'in'), ('sq', 'result', 'fsq', '')]
print(build('np.var', 'np.var - variance, the spread squared', body, demo, links,
            demo_width=640, text_width=760, text_height=1200, subpatch=NOISE_SUB))

# np.robust_mean: its page is hand-made now (see tools/README.md).
