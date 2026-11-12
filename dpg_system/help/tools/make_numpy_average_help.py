"""np averages and spread: one page per node.

None any more: the np.mean, np.median, np.nanmean, np.nanmedian, np.std,
np.var and np.robust_mean pages are all hand-made now (see tools/README.md).
The shared pieces below are kept for reference; nothing calls build().
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

# np.var: its page is hand-made now (see tools/README.md).

# np.robust_mean: its page is hand-made now (see tools/README.md).
