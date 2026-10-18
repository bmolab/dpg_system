"""Random tensors with a shape: the in-place fills (t.normal, t.uniform,
t.cauchy, t.log_normal, t.exponential, t.geometric, t.bernoulli, t.poisson)
and the t.dist.* distribution generators -- one page each.

These replace two hand-made family pages (t.distributions, t.dist.distributions)
that showed every node as one small heat_map of noise. A heat_map of noise looks
much the same whatever the distribution, so each page here shows the samples
TWICE: as a 64 x 64 heat_map (a picture of the field) and as a histogram
(t.histc into a bar plot), which is what actually tells one distribution from
another. Every parameter has a slider wired through 't bang float', so dragging
it re-samples at once and the histogram changes shape as you drag.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import FLT


def HM(width, height, count, miny=0.0, maxy=1.0, fmt='none', color='viridis'):
    """Property block for a heat_map that draws a whole array as one picture."""
    return {'color': color, 'width': width, 'height': height, 'sample count': count,
            'min y': miny, 'max y': maxy, 'update_mode': 'heat_map',
            'number format': fmt}


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


def sub_comments(sub):
    """Inside a subpatch build() takes comments as 'comment <text>' nodes."""
    for d in sub['demo']:
        if d.pop('comment', False):
            text = d.pop('text')
            d['init'] = 'comment ' + text
            d['w'] = int(len(text) * 7.7) + 20
            d['h'] = 30
    return sub


def th(body):
    """Prose box height: about 15 px a line at text size 30, plus a margin."""
    return (body.count('\n') + 1) * 15 + 40


def fmt_num(v):
    return ('%g' % v)


def HIST_PLOT(bins, maxy, width=520, height=230):
    """A bar plot that draws one histogram tensor, first bin on the left."""
    return {'color': 'none', 'width': width, 'height': height, 'style': 'bar',
            'update style': 'buffer holds one sample of input',
            'sample count': bins, 'min x': 0.0, 'max x': float(bins),
            'min y': 0.0, 'max y': float(maxy)}


SIDE = 64          # the sample field is SIDE x SIDE
N = SIDE * SIDE    # 4096 samples per click


# ----------------------------------------------------------- the demo builder
def sampler_demo(node_init, node_props, params, out_name, hist, hm, source=None,
                 hm_text=None, hist_text=None, node_h=None, extra=None):
    """Sliders -> t bang float -> parameter inlet + a click; the samples go to a
    heat_map (the field) and to t.histc -> bar plot (the shape).

    params: list of (inlet, default, slider min, slider max, comment)
    hist:   (lo, hi, bins, max y)
    hm:     (lo, hi, colour)
    source: None -> the bang clicks the node itself (t.dist.*);
            'zeros' -> a t.zeros 64 64 is clicked and its tensor is filled
            by the node (the in-place fills).
    Returns (demo, links, y below the plot)."""
    demo = [{'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46}]
    links = []
    trig = 'src' if source else 'nd'
    trig_in = '###input'
    x = 150
    for i, (inlet, dflt, lo, hi, text) in enumerate(params):
        k = 'p%d' % i
        demo += [
            {'key': 's' + k, 'init': 'slider', 'pos': (x, 62), 'w': 220, 'h': 60,
             'props': {'': float(dflt), 'min': float(lo), 'max': float(hi),
                       'format': '%.2f', 'width': 200}},
            {'key': 'c' + k, 'comment': True, 'text': text, 'pos': (x, 125)},
            {'key': 't' + k, 'init': 't bang float', 'pos': (x, 160), 'w': 110, 'h': 46},
        ]
        links += [('s' + k, '', 't' + k, ''), ('t' + k, 'float', 'nd', inlet),
                  ('t' + k, 'bang', trig, trig_in)]
        x += 250
    y = 240
    links.append(('lb', 'out', trig, trig_in))
    if source:
        demo += [
            {'key': 'src', 'init': 't.zeros %d %d' % (SIDE, SIDE), 'pos': (30, y),
             'w': 170, 'h': 110},
            {'key': 'csrc', 'comment': True,
             'text': 'a 64 x 64 tensor: only its shape and type are used',
             'pos': (220, y + 10)},
        ]
        links.append(('src', 'tensor of zeros', 'nd', 'tensor in'))
        y += 140
    nh = node_h or (90 + 30 * len(params))
    demo += [
        {'key': 'nd', 'init': node_init, 'pos': (30, y), 'w': 240, 'h': nh,
         'props': node_props},
        {'key': 'hm', 'init': 'heat_map', 'pos': (300, y), 'w': 250, 'h': 270,
         'props': HM(230, 230, SIDE, hm[0], hm[1], color=hm[2])},
        {'key': 'chm', 'comment': True,
         'text': hm_text or ('the field: %s dark, %s bright' % (fmt_num(hm[0]), fmt_num(hm[1]))),
         'pos': (300, y + 280)},
    ]
    links.append(('nd', out_name, 'hm', 'y'))
    y += max(nh, 300) + 30
    if hist is not None:
        lo, hi, bins, maxy = hist
        demo += [
            {'key': 'hc', 'init': 't.histc', 'pos': (30, y), 'w': 220, 'h': 150,
             'props': {'bin count': bins, 'min': float(lo), 'max': float(hi)}},
            {'key': 'chc', 'comment': True,
             'text': 'counts the %d samples in %d equal bins' % (N, bins),
             'pos': (270, y + 10)},
            {'key': 'pl', 'init': 'plot', 'pos': (30, y + 170), 'w': 540, 'h': 280,
             'props': HIST_PLOT(bins, maxy)},
            {'key': 'cpl', 'comment': True,
             'text': hist_text or ('the histogram: left edge %s, right edge %s'
                                   % (fmt_num(lo), fmt_num(hi))),
             'pos': (30, y + 455)},
        ]
        links += [('nd', out_name, 'hc', ''), ('hc', 'histogram tensor out', 'pl', 'y')]
        y += 500
    if extra:
        d2, l2, y = extra(y)
        demo += d2
        links += l2
    return demo, links, y


# ------------------------------------------------------------- shared prose
DIST_IN = """in (button):
The small button at the top left of the node. Click it, or send it
anything, for a fresh tensor of samples. Only this inlet makes the node
send.

shape (text, also an inlet):
The size of the tensor of samples, as whole numbers: [64, 64] is a 64
by 64 grid, [1000] a row of a thousand. Numbers typed after the node
name set it (t.dist.{n} 64 64); with none it is [1], a single sample.
A list sent into the inlet works too. A new shape does not send by
itself - the next click uses it."""

DIST_PARAM_NOTE = """Each parameter is a drag float on the node and also an inlet. Setting
one rebuilds the distribution but does NOT send; the next click samples
from it. In the demo each slider goes through 't bang float', which
passes the number to the parameter first and THEN clicks the node, so
dragging a slider re-samples at once. A value the distribution cannot
take (see each parameter) is refused with an error in the console, and
the node goes on sampling with the last value that worked."""

DIST_OUT = """OUTPUTS:

random tensor:
A float32 tensor on the cpu, in the shape set, filled with fresh
independent samples on every click."""

FILL_IN = """tensor in:
The tensor to fill. Only its SHAPE, dtype and device are used - its
values are ignored, and it is not changed (the node fills a copy). A
list or number is turned into a tensor first. Every tensor that arrives
is filled and sent; this is the only inlet that makes the node send."""

FILL_PARAM_NOTE = """The parameters are drag floats on the node and also inlets. They are
read when the next tensor arrives; setting one does not send. In the
demo each slider goes through 't bang float', which passes the number
to the parameter first and THEN clicks the t.zeros, so a fresh tensor
arrives and is filled at once."""

FILL_DTYPE = """The input must have a floating-point dtype (float32, the default for
the t.* generators, is right). An integer tensor - from t.zeros with
dtype int64, say - raises an error in the console and nothing is sent."""

FILL_OUT = """OUTPUTS:

tensor out:
A new tensor, the same shape, dtype and device as the input, filled
with fresh independent samples."""

HIST_READ = """THE HISTOGRAM:
t.histc divides the range from its min to its max into equal bins and
counts how many of the 4096 samples land in each; the bar plot draws
the counts, the first bin at the left. Its shape is the shape of the
distribution. Samples outside min..max are not counted at all."""


def dist_body(name, first, when, params_text, demo_text, related, extra=''):
    return (first + '\n\n' + when + '\n\n' + extra +
            'SYNTAX:\nt.dist.%s <dim 0: int> <dim 1: int> ...\n' % name +
            'Whole numbers give the shape of each batch of samples; with none\n'
            'the node sends one sample at a time.\n\n'
            'EXAMPLE:\nt.dist.%s 64 64\n\n' % name +
            'INPUTS and PARAMETERS:\n\n' + DIST_IN.replace('{n}', name) + '\n\n' +
            params_text + '\n\n' + DIST_PARAM_NOTE + '\n\n' + DIST_OUT + '\n\n' +
            HIST_READ + '\n\nREADING THE DEMO:\n' + demo_text + '\n\nRELATED:\n' + related)


def fill_body(name, first, when, params_text, demo_text, related, extra='',
              dtype_note=FILL_DTYPE):
    return (first + '\n\n' + when + '\n\n' + extra +
            'SYNTAX:\nt.%s\nThe node takes no arguments.\n\n' % name +
            'INPUTS and PARAMETERS:\n\n' + FILL_IN + '\n' + dtype_note + '\n\n' +
            params_text + ('\n\n' + FILL_PARAM_NOTE if params_text else '') + '\n\n' +
            FILL_OUT + '\n\n' + HIST_READ + '\n\nREADING THE DEMO:\n' + demo_text +
            '\n\nRELATED:\n' + related)


def write(name, title, body, demo, links, y, see, subpatch=None):
    demo = demo + see_also(y + 20, see)
    print(build(name, title, body, demo, links, demo_width=600, text_width=760,
                text_height=th(body), subpatch=subpatch))


PAGES = []   # filled below; run in order at the end


# ================================================================ t.dist.normal
body = dist_body('normal',
    "Makes a tensor of random samples from the NORMAL distribution - the bell\ncurve.",
    """Most values fall near the middle (loc) and fewer and fewer further
out: about two thirds within one scale of loc, nineteen in twenty within
two. Reach for it for natural-looking jitter, measurement noise, a
random walk's steps, or random weights - anything that should mostly be
close to a value and only rarely far from it. For noise where every
value in a range is equally likely, use t.dist.uniform or t.rand.""",
    """loc (drag float, also an inlet):
The middle of the bell - the mean, and the most likely value. Any
number. Default 0.

scale (drag float, also an inlet):
The spread - the standard deviation. Must be above 0. Default 1.""",
    """load_bang draws the first 4096 samples when the page opens; click the
node's own button for more. The heat_map shows them as a 64 x 64 field
(-3 dark, 3 bright): grey-green mush, mostly mid tones. The histogram
underneath is the bell. Drag loc and the bell slides sideways; drag
scale and it widens and flattens (the same number of samples spread
over more bins) or narrows into a tall spike.""",
    """t.normal fills an existing tensor with the same distribution.
t.dist.log_normal is the exponential of a normal, t.dist.half_normal
its absolute value. t.dist.cauchy and t.dist.laplace are bells with
heavier tails. t.rand and np.rand are uniform.""")
demo, links, y = sampler_demo(
    't.dist.normal %d %d' % (SIDE, SIDE), {'loc': 0.0, 'scale': 1.0},
    [('loc', 0.0, -4.0, 4.0, 'loc: where the middle is'),
     ('scale', 1.0, 0.4, 3.0, 'scale: how wide the bell is')],
    'random tensor', (-8.0, 8.0, 64, 1000.0), (-3.0, 3.0, 'viridis'))
write('t.dist.normal', 't.dist.normal - samples from a bell curve', body, demo, links, y,
      ['t.normal', 't.dist.log_normal', 't.dist.laplace', 't.dist.cauchy', 't.rand', 'np.rand'])

# ===================================================================== t.normal
body = fill_body('normal',
    "Fills a tensor with random samples from the NORMAL distribution (the bell\ncurve), keeping the tensor's shape.",
    """Send it any tensor and it sends back a tensor of the same shape and
type, every element replaced by an independent bell-curve sample. Use
it to add jitter shaped like the data you already have, or to make
normal noise of whatever size arrives. To make samples from nothing,
with the shape typed into the node, use t.dist.normal.""",
    """mean (drag float, also an inlet):
The middle of the bell. Default 0.

std (drag float, also an inlet):
The spread - the standard deviation. Default 1. 0 gives every element
exactly the mean; a negative std is an error (printed in the console)
and nothing is sent.""",
    """load_bang clicks t.zeros once when the page opens: its 64 x 64 tensor
of zeros goes into t.normal, which sends back 4096 bell-curve samples.
The heat_map shows the field (-3 dark, 3 bright); the histogram under
it is the bell. Drag mean and the bell slides sideways; drag std and it
widens or narrows. The zeros themselves never show - only their shape
matters.""",
    """t.dist.normal makes the same samples without an input tensor.
t.uniform, t.cauchy and t.log_normal are the other fills. t.rand and
np.rand make uniform noise.""")
demo, links, y = sampler_demo(
    't.normal', {'mean': 0.0, 'std': 1.0},
    [('mean', 0.0, -4.0, 4.0, 'mean: where the middle is'),
     ('std', 1.0, 0.4, 3.0, 'std: how wide the bell is')],
    'tensor out', (-8.0, 8.0, 64, 1000.0), (-3.0, 3.0, 'viridis'), source='zeros')
write('t.normal', 't.normal - fill a tensor with bell-curve noise', body, demo, links, y,
      ['t.dist.normal', 't.uniform', 't.log_normal', 't.rand', 'np.rand'])


DEMO_START = """load_bang draws the first 4096 samples when the page opens; click the
node's own button for more. """

FILL_START = """load_bang clicks t.zeros once when the page opens, and its 64 x 64
tensor of zeros comes back from the node as 4096 samples; the zeros
themselves never show, only their shape matters. """


def dist_page(name, title, first, when, params_text, demo_text, related, see,
              node_props, params, hist, hm, hm_text=None, hist_text=None, extra=None,
              extra_prose=''):
    body = dist_body(name, first, when, params_text, DEMO_START + demo_text, related,
                     extra=extra_prose)
    demo, links, y = sampler_demo('t.dist.%s %d %d' % (name, SIDE, SIDE), node_props,
                                  params, 'random tensor', hist, hm, hm_text=hm_text,
                                  hist_text=hist_text, extra=extra)
    write('t.dist.' + name, title, body, demo, links, y, see)


def fill_page(name, title, first, when, params_text, demo_text, related, see,
              node_props, params, hist, hm, hm_text=None, hist_text=None,
              dtype_note=FILL_DTYPE, extra_prose=''):
    body = fill_body(name, first, when, params_text, FILL_START + demo_text, related,
                     extra=extra_prose, dtype_note=dtype_note)
    demo, links, y = sampler_demo('t.' + name, node_props, params, 'tensor out', hist, hm,
                                  source='zeros', hm_text=hm_text, hist_text=hist_text)
    write('t.' + name, title, body, demo, links, y, see)


# ============================================================ t.dist.log_normal
dist_page('log_normal', 't.dist.log_normal - samples whose log is a bell curve',
    "Makes a tensor of random samples from the LOG-NORMAL distribution: always\npositive, bunched low, with a long tail of large values.",
    """Each sample is e raised to the power of a normal (bell-curve) sample,
so the LOGARITHMS of the values form a bell. It suits quantities that
grow by multiplying rather than adding - sizes, durations, prices,
loudness, the spacing of events - where a value twice the typical one
is as likely as one half of it, and nothing is ever negative.""",
    """loc (drag float, also an inlet):
The middle of the bell of the LOGARITHMS, not of the values. The median
value is e to the power loc (loc 0 gives a median of 1). Any number.
Default 0.

scale (drag float, also an inlet):
The spread of the logarithms. Must be above 0. Small values give a
nearly symmetrical hump; larger ones a sharp peak near 0 and a very long
tail. Default 1.""",
    """The heat_map (0 dark, 5 bright) is mostly dark with scattered bright
specks - the tail. The histogram rises steeply from 0, peaks below 1
and trails off to the right; samples above 10 run off its right edge.
Raise loc and the whole shape stretches to the right; lower scale and
the tail shortens into a lopsided hump.""",
    """t.log_normal fills an existing tensor with the same distribution.
t.dist.normal is the bell it is built on. t.dist.gamma, t.dist.weibull
and t.dist.pareto are other positive distributions with a long tail.""",
    ['t.log_normal', 't.dist.normal', 't.dist.gamma', 't.dist.pareto', 't.rand'],
    {'loc': 0.0, 'scale': 1.0},
    [('loc', 0.0, -0.5, 1.5, 'loc: the middle of the logarithms'),
     ('scale', 1.0, 0.5, 1.5, 'scale: the spread of the logarithms')],
    (0.0, 10.0, 64, 1000.0), (0.0, 5.0, 'viridis'))

# ================================================================ t.dist.cauchy
dist_page('cauchy', 't.dist.cauchy - a bell with enormous tails',
    "Makes a tensor of random samples from the CAUCHY distribution: a peak\nlike a bell curve's, but with tails so heavy that huge values keep turning up.",
    """Near the middle it looks much like t.dist.normal; far out it does not.
A normal sample ten widths from the middle essentially never happens; a
Cauchy one does about one time in sixteen, and now and then one is
hundreds of widths out. It has no average - the mean of a batch jumps
about however many samples you take. Use it for jitter with occasional
wild jumps, or to test how a chain copes with outliers.""",
    """loc (drag float, also an inlet):
The middle - the peak and the median. Any number. Default 0.

scale (drag float, also an inlet):
The half-width: half the samples fall within loc plus or minus scale.
Must be above 0. Default 1.""",
    """The heat_map (-3 dark, 3 bright) has a scatter of fully black and
fully white cells - the tails, far outside the colour range. The
histogram is a narrow peak with wide flat skirts; a few samples in
every batch land outside -10..10 and are not counted. Drag scale and
the peak widens or sharpens; drag loc and it slides.""",
    """t.cauchy fills an existing tensor with the same distribution.
t.dist.normal has the same middle and light tails; t.dist.half_cauchy
is the positive half of this one; t.dist.laplace sits between them.""",
    ['t.cauchy', 't.dist.normal', 't.dist.half_cauchy', 't.dist.laplace', 't.rand'],
    {'loc': 0.0, 'scale': 1.0},
    [('loc', 0.0, -4.0, 4.0, 'loc: where the peak is'),
     ('scale', 1.0, 0.5, 3.0, 'scale: how wide the peak is')],
    (-10.0, 10.0, 64, 900.0), (-3.0, 3.0, 'viridis'))

# ================================================================ t.dist.laplace
dist_page('laplace', 't.dist.laplace - a sharp peak with straight sides',
    "Makes a tensor of random samples from the LAPLACE distribution: a pointed\npeak falling away evenly on both sides.",
    """It is two exponential distributions back to back - one to the right of
loc, its mirror to the left - so the peak comes to a point rather than
the rounded top of a bell, and the tails, though falling fast, are
heavier than a normal's. It models the DIFFERENCE between two random
waiting times, and errors that are usually tiny but now and then
moderate.""",
    """loc (drag float, also an inlet):
The middle - the peak, the mean and the median. Any number. Default 0.

scale (drag float, also an inlet):
The spread. Must be above 0. The average distance from loc is scale;
the standard deviation is about 1.41 times scale. Default 1.""",
    """The heat_map (-3 dark, 3 bright) is mid-tones with a few strong cells.
The histogram is a tent: a sharp point at loc and sides that slope down
almost in straight lines (they are exponential curves). Compare it with
t.dist.normal's rounded bell at the same settings. Drag scale to widen
the tent, loc to move it.""",
    """t.dist.normal is the rounded bell; t.dist.exponential is one side of
this tent; t.dist.cauchy has much heavier tails.""",
    ['t.dist.normal', 't.dist.exponential', 't.dist.cauchy', 't.normal', 't.rand'],
    {'loc': 0.0, 'scale': 1.0},
    [('loc', 0.0, -4.0, 4.0, 'loc: where the point is'),
     ('scale', 1.0, 0.6, 3.0, 'scale: how wide the tent is')],
    (-8.0, 8.0, 64, 1000.0), (-3.0, 3.0, 'viridis'))

# ================================================================= t.dist.gumbel
dist_page('gumbel', 't.dist.gumbel - the largest of many',
    "Makes a tensor of random samples from the GUMBEL distribution: a lopsided\nhump with a longer tail to the right.",
    """If you take the LARGEST of a big batch of ordinary random values, again
and again, the largest values follow this shape - so it models the
highest flood of each year, the strongest gust of each storm, the
loudest moment of each take. In machine learning it is the noise added
to scores to pick one at random in proportion to its weight (the
'Gumbel max trick').

THE OTHER SPELLING:
t.dist.gumble is the same node under a misspelt name, kept so that old
patches still load; it gives exactly the same distribution. Use
t.dist.gumbel in new patches.

""",
    """loc (drag float, also an inlet):
The peak - the most likely value. The mean lies a little to its right
(loc plus 0.58 times scale). Any number. Default 0.

scale (drag float, also an inlet):
The spread. Must be above 0. Default 1.""",
    """The heat_map (-2 dark, 4 bright) is mid-dark with bright specks. The
histogram rises steeply on the left, peaks at loc and slopes down more
gently to the right - the tail of unusually large maxima. Drag loc to
move it, scale to stretch it.""",
    """t.dist.normal is the symmetrical bell; t.dist.weibull and
t.dist.log_normal are other lopsided shapes, but only positive.""",
    ['t.dist.normal', 't.dist.weibull', 't.dist.log_normal', 't.dist.laplace', 't.rand'],
    {'loc': 0.0, 'scale': 1.0},
    [('loc', 0.0, -3.0, 3.0, 'loc: where the peak is'),
     ('scale', 1.0, 0.5, 3.0, 'scale: how wide the hump is')],
    (-5.0, 10.0, 64, 800.0), (-2.0, 4.0, 'viridis'))

# ============================================================ t.dist.half_normal
dist_page('half_normal', 't.dist.half_normal - the size of a bell-curve sample',
    "Makes a tensor of random samples from the HALF-NORMAL distribution: the\nabsolute values of bell-curve samples - never negative, most near 0.",
    """Take normal samples centred on 0 and drop their signs: that is this
distribution. It is the size of a random deviation when only its size
matters - how far a hand wanders from a target, the magnitude of a
random jolt - and a good positive noise that is usually small.""",
    """scale (drag float, also an inlet):
The scale of the bell before it is folded - its standard deviation.
Must be above 0. The average sample is about 0.8 times scale. Default
1.""",
    """The heat_map (0 dark, 3 bright) is mostly dark. The histogram is the
right half of a bell: tallest at 0 and falling smoothly away. Drag
scale and it stretches or squeezes, always starting at 0.""",
    """t.dist.normal is the unfolded bell; t.dist.half_cauchy is the same
idea with heavy tails; t.dist.exponential also starts at its peak at 0
but falls away more slowly.""",
    ['t.dist.normal', 't.dist.half_cauchy', 't.dist.exponential', 't.normal', 't.rand'],
    {'scale': 1.0},
    [('scale', 1.0, 0.5, 3.0, 'scale: how far out it reaches')],
    (0.0, 8.0, 64, 900.0), (0.0, 3.0, 'viridis'))

# ============================================================ t.dist.half_cauchy
dist_page('half_cauchy', 't.dist.half_cauchy - positive, with huge outliers',
    "Makes a tensor of random samples from the HALF-CAUCHY distribution: the\nabsolute values of Cauchy samples - positive, mostly small, now and then enormous.",
    """Fold t.dist.cauchy at 0 and this is what you get. Half the samples are
below scale, but the tail never really dies away: one in ten is above
about six times scale, one in a hundred above sixty. Use it for
positive values that are normally modest but must sometimes be wild.""",
    """scale (drag float, also an inlet):
The median: half the samples fall below it. Must be above 0. Default 1.""",
    """The heat_map (0 dark, 4 bright) is dark with a sprinkling of
saturated bright cells - the outliers. The histogram peaks at 0 and
falls away, but the bars far to the right never quite reach zero, and
about one sample in sixteen is beyond 10 and not counted at all. Drag
scale to stretch it.""",
    """t.dist.cauchy is the unfolded version; t.dist.half_normal folds a bell
with light tails instead; t.dist.pareto is another heavy-tailed
positive distribution.""",
    ['t.dist.cauchy', 't.dist.half_normal', 't.dist.pareto', 't.cauchy', 't.rand'],
    {'scale': 1.0},
    [('scale', 1.0, 0.5, 4.0, 'scale: the median size')],
    (0.0, 10.0, 64, 900.0), (0.0, 4.0, 'viridis'))

# ============================================================ t.dist.exponential
dist_page('exponential', 't.dist.exponential - waiting times',
    "Makes a tensor of random samples from the EXPONENTIAL distribution: the\ntime you wait for something that happens at a steady random rate.",
    """Short waits are the most likely, long ones rarer and rarer, each extra
moment of waiting as likely to end the wait as the last. Use it for
the gaps between random events - clicks of a Geiger counter, raindrops,
the timing of generated notes - or for positive noise that is usually
small.""",
    """rate (drag float, also an inlet):
How often the events happen, per unit of time. Must be above 0. The
average wait is 1 / rate, so a higher rate means shorter waits.
Default 1.""",
    """The heat_map (0 dark, 3 bright) is mostly dark with brighter specks.
The histogram is tallest at 0 and falls away in a smooth curve, halving
every 0.69 / rate. Raise rate and it crowds against 0; lower it and the
curve stretches out to the right.""",
    """t.exponential fills an existing tensor with the same distribution.
t.dist.poisson counts how many such events land in a fixed time.
t.dist.gamma is the wait for several events; t.dist.weibull generalises
this curve; t.dist.laplace is two of them back to back.""",
    ['t.exponential', 't.dist.poisson', 't.dist.gamma', 't.dist.weibull', 't.rand'],
    {'rate': 1.0},
    [('rate', 1.0, 0.2, 2.0, 'rate: events per unit of time')],
    (0.0, 10.0, 64, 1200.0), (0.0, 3.0, 'viridis'))

# ================================================================ t.dist.gamma
dist_page('gamma', 't.dist.gamma - the wait for several events',
    "Makes a tensor of random samples from the GAMMA distribution: positive\nvalues in a lopsided hump whose shape you choose.",
    """With a whole-number concentration k it is the total time you wait for
k random events happening at the given rate - the sum of k exponential
waits. k 1 is the exponential itself (peak at 0); as k grows the peak
moves out and the hump becomes more and more like a bell. It is a
flexible general-purpose shape for positive quantities.""",
    """concentration (drag float, also an inlet):
The shape, k. Must be above 0. Below 1 the samples pile up against 0;
1 is the exponential; above 1 there is a hump. Default 0.5 on a new
node; the demo sets 2.

rate (drag float, also an inlet):
How fast the events come. Must be above 0. The mean is concentration /
rate, so raising rate squeezes the whole shape towards 0. Default 0.5
on a new node; the demo sets 1.""",
    """The heat_map (0 dark, 6 bright) is a speckle of mid tones. The
histogram starts at 0, rises to a hump at (concentration - 1) / rate
and trails off to the right. Raise concentration and the hump moves out
and grows rounder; raise rate and everything shrinks towards 0.""",
    """t.dist.exponential is the case concentration 1. t.dist.chi2 is a
gamma with rate 0.5. t.dist.log_normal and t.dist.weibull are other
positive humps.""",
    ['t.dist.exponential', 't.dist.chi2', 't.dist.log_normal', 't.dist.weibull', 't.dist.beta'],
    {'concentration': 2.0, 'rate': 1.0},
    [('concentration', 2.0, 1.5, 8.0, 'concentration: the shape'),
     ('rate', 1.0, 0.7, 2.0, 'rate: squeezes it towards 0')],
    (0.0, 15.0, 64, 1000.0), (0.0, 6.0, 'viridis'))

# ================================================================== t.dist.chi2
dist_page('chi2', 't.dist.chi2 - sums of squared bell-curve samples',
    "Makes a tensor of random samples from the CHI-SQUARED distribution: the sum\nof the squares of several independent bell-curve samples.",
    """With df 3 each sample is what you get by squaring three standard
normal samples and adding them up - the squared length of a random 3-D
vector. It is the distribution of a sum of squared errors, and the
reference curve for chi-squared tests. Always positive; the mean is df.""",
    """df (drag float, also an inlet):
Degrees of freedom - how many squared samples are added. Must be above
0; it need not be a whole number. 1 and 2 pile up against 0; from 3 up
there is a hump near df - 2 that moves out and becomes more bell-like.
Default 1 on a new node; the demo sets 3.""",
    """The heat_map (0 dark, 8 bright) is a speckle of mostly dark tones. The
histogram starts at 0, humps at df - 2 and trails to the right; the
mean is df. Drag df up and the hump moves right and widens.""",
    """t.dist.gamma is the general family this belongs to (rate 0.5,
concentration df / 2). t.dist.fisher_snedecor is a ratio of two of
these. t.dist.normal is what is being squared.""",
    ['t.dist.gamma', 't.dist.fisher_snedecor', 't.dist.normal', 't.dist.exponential', 't.rand'],
    {'df': 3.0},
    [('df', 3.0, 2.0, 12.0, 'df: how many squares are summed')],
    (0.0, 20.0, 64, 700.0), (0.0, 8.0, 'viridis'))

# ======================================================== t.dist.fisher_snedecor
dist_page('fisher_snedecor', 't.dist.fisher_snedecor - the ratio of two variances',
    "Makes a tensor of random samples from the F distribution (Fisher-Snedecor):\nthe ratio of two independent chi-squared samples, each divided by its df.",
    """It is the distribution of the ratio of two estimated variances, and so
the reference curve of the F-test and of analysis of variance. As a
generator it gives positive values with a hump near 1 and a long tail
to the right.""",
    """df1 (drag float, also an inlet):
Degrees of freedom of the top of the ratio. Must be above 0. Small
values push the samples towards 0. Default 0.5 on a new node; the demo
sets 5.

df2 (drag float, also an inlet):
Degrees of freedom of the bottom. Must be above 0. Small values make
the tail very long. Default 0.5 on a new node; the demo sets 10.

The defaults of a new node (0.5 and 0.5) give an extremely
heavy-tailed shape - most samples near 0 and some astronomically large
- so set both to something like 5 and 10 first.""",
    """The heat_map (0 dark, 3 bright) is mid-dark with bright specks. The
histogram rises from 0 to a hump just below 1 and trails off to the
right. Raise both dfs and the hump narrows around 1; lower df2 and the
right tail grows long.""",
    """t.dist.chi2 is the distribution being divided. t.dist.gamma and
t.dist.log_normal are other positive humps.""",
    ['t.dist.chi2', 't.dist.gamma', 't.dist.log_normal', 't.dist.beta', 't.rand'],
    {'df1': 5.0, 'df2': 10.0},
    [('df1', 5.0, 2.0, 30.0, 'df1: degrees of freedom on top'),
     ('df2', 10.0, 2.0, 30.0, 'df2: degrees of freedom below')],
    (0.0, 5.0, 64, 600.0), (0.0, 3.0, 'viridis'))

# ================================================================ t.dist.weibull
dist_page('weibull', 't.dist.weibull - lifetimes and wind speeds',
    "Makes a tensor of random samples from the WEIBULL distribution: positive\nvalues whose shape runs from a spike at 0 to a narrow hump.",
    """It describes how long things last when the chance of failing changes
with age, and is the standard model for wind speeds and particle
sizes. Concentration 1 is the exponential; below 1 it piles up at 0
with a long tail; around 3.5 it is close to a bell; above that it is a
narrow hump just below scale.""",
    """scale (drag float, also an inlet):
Stretches the whole shape; about two thirds of the samples fall below
it. Must be above 0. Default 1.

concentration (drag float, also an inlet):
The shape. Must be above 0. Default 1 (the exponential).""",
    """The heat_map (0 dark, 3 bright) changes character as you drag
concentration: speckled and dark at 1, smooth mid tones at 5. The
histogram goes from a curve falling away from 0 (concentration 1) to a
lopsided hump (2) to a narrow peak near scale (5). Drag scale to
stretch it sideways.""",
    """t.dist.exponential is concentration 1. t.dist.gamma and
t.dist.log_normal are other positive humps; t.dist.gumbel describes
largest values rather than lifetimes.""",
    ['t.dist.exponential', 't.dist.gamma', 't.dist.log_normal', 't.dist.gumbel', 't.rand'],
    {'scale': 1.0, 'concentration': 1.0},
    [('scale', 1.0, 0.7, 3.0, 'scale: stretches it sideways'),
     ('concentration', 1.0, 0.7, 5.0, 'concentration: the shape')],
    (0.0, 5.0, 64, 1000.0), (0.0, 3.0, 'viridis'))

# ================================================================= t.dist.pareto
dist_page('pareto', 't.dist.pareto - the power law',
    "Makes a tensor of random samples from the PARETO distribution: values that\nnever fall below scale, mostly close to it, with a very long tail.",
    """It is the '80-20' power law: most samples are small but a few are
huge, and the huge ones account for much of the total. City sizes,
incomes, file sizes and the loudness of rare events follow it. Every
doubling of the value is the same factor less likely, however far out
you go.""",
    """scale (drag float, also an inlet):
The smallest possible value - every sample is at least this. Must be
above 0. Default 1.

alpha (drag float, also an inlet):
How fast the tail falls away. Must be above 0. Smaller alpha means a
heavier tail; with alpha 1 or below the mean is infinite and a batch
can hold some astronomically large samples. Default 1.""",
    """The heat_map (0 dark, 5 bright) is dark with a scatter of saturated
cells. The histogram is empty below scale, jumps to its peak at scale
and falls away in a long tail; with alpha 1 about one sample in ten is
beyond 10 and not counted. Raise alpha and the tail shortens; raise
scale and the whole thing moves right. With small scale and large
alpha the peak can rise above the top of the plot.""",
    """t.dist.half_cauchy is another heavy-tailed positive distribution;
t.dist.exponential falls away much faster; t.dist.log_normal is a
milder long tail.""",
    ['t.dist.half_cauchy', 't.dist.exponential', 't.dist.log_normal', 't.dist.weibull', 't.rand'],
    {'scale': 1.0, 'alpha': 1.0},
    [('scale', 1.0, 0.5, 4.0, 'scale: the smallest value'),
     ('alpha', 1.0, 0.5, 4.0, 'alpha: how fast the tail falls')],
    (0.0, 10.0, 64, 1500.0), (0.0, 5.0, 'viridis'))

# ================================================================ t.dist.uniform
dist_page('uniform', 't.dist.uniform - every value equally likely',
    "Makes a tensor of random samples from the UNIFORM distribution: any value\nbetween low and high, all equally likely.",
    """This is plain noise in a range, with no value favoured over another -
the same thing t.rand makes, but with the range as live inlets and the
same interface as the other t.dist nodes, so you can swap a shaped
distribution in and out of a chain.""",
    """low (drag float, also an inlet):
The bottom of the range; samples can equal it. Default 0.

high (drag float, also an inlet):
The top of the range; samples are always below it. It must be above
low - a high at or below low is refused (an error in the console) and
the node keeps its last range. That is why the demo's sliders do not
overlap. Default 1.""",
    """The heat_map (-1 dark, 3 bright) is even noise with no structure. The
histogram is a flat-topped block from low to high, the bars jittering
around the same height; outside it there is nothing. Drag low and high
and the block widens (and lowers, as the same 4096 samples spread over
more bins) or narrows (and rises).""",
    """t.uniform fills an existing tensor the same way. t.rand and np.rand
are the plain uniform generators. Every other t.dist node shows a
shape; this one shows the absence of one.""",
    ['t.uniform', 't.rand', 'np.rand', 't.dist.normal', 't.dist.beta'],
    {'low': 0.0, 'high': 1.0},
    [('low', 0.0, -1.0, 0.5, 'low: the bottom of the range'),
     ('high', 1.0, 1.0, 3.0, 'high: the top of the range')],
    (-1.0, 3.0, 64, 600.0), (-1.0, 3.0, 'viridis'))

# =================================================================== t.dist.beta
dist_page('beta', 't.dist.beta - shapes between 0 and 1',
    "Makes a tensor of random samples from the BETA distribution: values between\n0 and 1 in a shape set by two numbers.",
    """It is the most flexible distribution on 0..1. Both parameters 1 give a
flat line; both above 1 a hump; both below 1 a U, piled up at the two
ends; unequal values lean the shape towards one end. Use it for random
proportions, probabilities, mix amounts and positions along a line,
wherever the result must stay inside 0..1 but should not be flat.""",
    """alpha (drag float, also an inlet):
Pulls the samples towards 1. Must be above 0. Default 0.5.

beta (drag float, also an inlet):
Pulls the samples towards 0. Must be above 0. Default 0.5.

The mean is alpha / (alpha + beta); the larger both are, the narrower
the hump around it. (In torch's terms alpha is concentration1 and beta
concentration0.)""",
    """At the defaults (0.5 and 0.5) the heat_map (0 dark, 1 bright) is
nearly all very dark and very bright cells, and the histogram is a U:
tall bars at both ends, low in the middle (the end bars can run off
the top of the plot). Drag both to 1 for a flat line, both to 5 for a
hump in the middle, then make one larger than the other and the hump
slides towards that end.""",
    """t.dist.kumaraswamy makes very similar shapes. t.dist.uniform is beta
with both 1. t.dist.continuous_bernoulli is another 0..1 shape with a
single parameter.""",
    ['t.dist.kumaraswamy', 't.dist.uniform', 't.dist.continuous_bernoulli', 't.dist.gamma', 't.rand'],
    {'alpha': 0.5, 'beta': 0.5},
    [('alpha', 0.5, 0.5, 6.0, 'alpha: pulls towards 1'),
     ('beta', 0.5, 0.5, 6.0, 'beta: pulls towards 0')],
    (0.0, 1.0, 64, 400.0), (0.0, 1.0, 'viridis'))

# ============================================================ t.dist.kumaraswamy
dist_page('kumaraswamy', 't.dist.kumaraswamy - beta-like shapes, made simply',
    "Makes a tensor of random samples from the KUMARASWAMY distribution: values\nbetween 0 and 1 in a shape set by two numbers.",
    """It makes the same family of shapes as t.dist.beta - flat, humped,
U-shaped, leaning to either end - but from a formula simple enough to
invert by hand: a sample is (1 - (1 - u) to the power 1/beta) to the
power 1/alpha, for a uniform u. Its parameters do not mean exactly the
same as beta's, so the same numbers give a somewhat different shape.""",
    """alpha (drag float, also an inlet):
Larger values push the samples towards 1. Must be above 0. Default 1.

beta (drag float, also an inlet):
Larger values push the samples towards 0. Must be above 0. Default 1.

Both 1 give the flat line; both below 1 a U; both above 1 a hump.""",
    """At the defaults (1 and 1) the heat_map (0 dark, 1 bright) is even
noise and the histogram flat. Drag alpha up and the samples crowd
towards 1; drag beta up and they crowd towards 0; both up gives a hump;
both below 1 piles them at the ends.""",
    """t.dist.beta is the better-known distribution with the same range of
shapes. t.dist.uniform is the flat case; t.dist.continuous_bernoulli
is another 0..1 shape.""",
    ['t.dist.beta', 't.dist.uniform', 't.dist.continuous_bernoulli', 't.rand'],
    {'alpha': 1.0, 'beta': 1.0},
    [('alpha', 1.0, 0.5, 6.0, 'alpha: pushes towards 1'),
     ('beta', 1.0, 0.5, 6.0, 'beta: pushes towards 0')],
    (0.0, 1.0, 64, 400.0), (0.0, 1.0, 'viridis'))

# ============================================================== t.dist.von_mises
dist_page('von_mises', 't.dist.von_mises - a bell curve round a circle',
    "Makes a tensor of random ANGLES from the VON MISES distribution: a bell curve\nwrapped round the circle, in radians.",
    """It is what a bell curve becomes when the values are directions: the
samples cluster around one angle, and the far side of the circle is
where the two tails meet. Use it for random headings, wind directions,
phases, or jitter on a rotation. The samples are in radians, from -pi
to pi (-3.14 to 3.14), whatever loc is.""",
    """concentration (drag float, also an inlet):
How tightly the angles cluster. Must be above 0. Near 0 every
direction is equally likely; around 10 they are within about a third
of a radian (18 degrees) of loc. Default 1. (It is the first parameter
on this node, before loc.)

loc (drag float, also an inlet):
The favoured direction, in radians. Values beyond pi are wrapped round.
Default 1 (not 0).""",
    """The histogram spans the full circle, -pi on the left to pi on the
right. At concentration 1 it is a broad hump around loc 1 with a low
floor reaching right round; drag concentration up and it narrows to a
peak; down towards 0 and it flattens to uniform. Drag loc past either
end and the hump wraps round to the other side - the two ends of the
plot are the same direction. The heat_map (-pi dark, pi bright) shows
the angles as tones.""",
    """t.dist.normal is the straight-line version; t.dist.uniform with low
-3.14 and high 3.14 gives a uniformly random direction.""",
    ['t.dist.normal', 't.dist.uniform', 't.rand', 'np.rand'],
    {'concentration': 1.0, 'loc': 1.0},
    [('concentration', 1.0, 0.01, 10.0, 'concentration: how tight'),
     ('loc', 1.0, -3.0, 3.0, 'loc: the favoured angle')],
    (-3.1416, 3.1416, 64, 550.0), (-3.1416, 3.1416, 'viridis'),
    hist_text='the histogram: left edge -pi, right edge pi')


def mean_extra(src_out, text, src='nd'):
    """t.mean of the samples into a float: the fraction of ones, the average count."""
    def f(y):
        demo = [
            {'key': 'mn', 'init': 't.mean', 'pos': (30, y), 'w': 160, 'h': 100},
            {'key': 'fm', 'init': 'float', 'pos': (30, y + 120), 'w': 127, 'h': 42,
             'props': FLT},
            {'key': 'cfm', 'comment': True, 'text': text, 'pos': (180, y + 125)},
        ]
        links = [(src, src_out, 'mn', 'tensor in'), ('mn', 'output', 'fm', '')]
        return demo, links, y + 180
    return f


# ============================================================== t.dist.bernoulli
dist_page('bernoulli', 't.dist.bernoulli - random coin flips',
    "Makes a tensor of random 0s and 1s: each element is 1 with the probability\nyou set, otherwise 0.",
    """Each element is an independent biased coin flip. Use it for random
masks (which cells to keep, which to drop), random triggers, sparse
patterns of dots, or a yes/no decision whose odds you control.""",
    """probability (drag float, also an inlet):
The chance that each element is 1, from 0 (all zeros) to 1 (all ones).
Outside 0..1 is refused (an error in the console) and the node keeps
its last value. Default 0.5.

THE logits FORM:
t.dist.bernoulli logits (any word after the name other than probs)
replaces the probability with a 'logits' parameter, default 1: the
probability is then 1 / (1 + e to the power -logits), so 0 means even
odds, large positive values nearly always 1, large negative nearly
always 0. Useful when the odds come out of a neural network.""",
    """The heat_map shows the coin flips: 1 white, 0 black. At 0.5 it is
salt and pepper; drag probability towards 0 and the white dots thin
out, towards 1 the black ones do. The histogram has just two bars - the
count of 0s on the left, of 1s on the right - and t.mean of the samples
is the fraction that came up 1, close to the probability you set.""",
    """t.bernoulli makes a 0/1 flip for every element of a tensor of
probabilities - a different probability for each element.
t.dist.continuous_bernoulli is a smooth 0..1 cousin. t.rand compared
with a threshold does the same job by hand.""",
    ['t.bernoulli', 't.dist.continuous_bernoulli', 't.dist.poisson', 't.rand', 'np.rand'],
    {'probability': 0.5},
    [('probability', 0.5, 0.0, 1.0, 'probability: the chance of a 1')],
    (0.0, 1.0, 2, 4096.0), (0.0, 1.0, 'hot'),
    hm_text='the flips: 1 white, 0 black',
    hist_text='two bars: how many 0s (left) and 1s (right)',
    extra=mean_extra('random tensor', 'the fraction of 1s'))

# =================================================== t.dist.continuous_bernoulli
dist_page('continuous_bernoulli', 't.dist.continuous_bernoulli - a sloping line on 0..1',
    "Makes a tensor of random samples between 0 and 1 whose density slopes\nup towards 1 or down towards 0, set by one probability.",
    """It is the smooth relative of the Bernoulli coin flip: instead of
exactly 0 or 1, any value in between, leaning towards 1 when the
probability is above 0.5 and towards 0 when it is below. At 0.5 it is
flat (uniform). It was introduced for neural networks whose outputs are
grey levels rather than strict black and white.""",
    """probability (drag float, also an inlet):
Which way and how steeply the shape leans: above 0.5 towards 1, below
0.5 towards 0, 0.5 flat. Keep it within 0..1: values outside are not
refused, but they mean nothing. It is NOT the mean - a probability of
0.9 gives a mean of about 0.67. Default 0.5.

THE logits FORM:
t.dist.continuous_bernoulli logits (any word after the name other than
probs) takes the lean as a 'logits' parameter instead, default 1, where
0 is flat.""",
    """At 0.5 the heat_map (0 dark, 1 bright) is even noise and the histogram
flat. Drag probability up and the bars on the right grow while those on
the left shrink - a sloping curve; drag it down and the slope reverses.
Unlike t.dist.bernoulli, there is never a pile of exact 0s and 1s.""",
    """t.dist.bernoulli gives exact 0s and 1s. t.dist.beta and
t.dist.kumaraswamy make more varied shapes on 0..1; t.dist.uniform is
the flat case.""",
    ['t.dist.bernoulli', 't.dist.beta', 't.dist.kumaraswamy', 't.dist.uniform', 't.rand'],
    {'probability': 0.5},
    [('probability', 0.5, 0.02, 0.98, 'probability: which way it leans')],
    (0.0, 1.0, 64, 300.0), (0.0, 1.0, 'viridis'))

# ================================================================ t.dist.poisson
dist_page('poisson', 't.dist.poisson - how many random events',
    "Makes a tensor of random whole-number COUNTS from the POISSON distribution:\nhow many events happen in a fixed time when they come at a steady random rate.",
    """If clicks come at random at an average of 3 a second, the number in
any one second is a Poisson sample with rate 3: often 2, 3 or 4,
sometimes 0, now and then 8. Use it for the number of notes to play in
a bar, of particles to emit in a frame, of photons on a pixel (shot
noise) - any count of independent random events.""",
    """rate (drag float, also an inlet):
The average count. Must be above 0. Default 1 on a new node; the demo
sets 3.""",
    """The heat_map (0 dark, 10 bright) shows the counts as stepped tones.
The histogram has one bar per whole number, 0 at the left to 20 at the
right: at rate 3 the bars for 2 and 3 are the tallest. Drag rate up and
the hump moves right, widens and becomes more like a bell; drag it down
towards 1 and the 0 and 1 bars dominate. The counts come out as floats
(2.0, 3.0 ...), always whole.""",
    """t.poisson draws a count for every element of a tensor of rates.
t.dist.exponential is the waiting time between the same events.
t.dist.bernoulli is a single yes/no event.""",
    ['t.poisson', 't.dist.exponential', 't.dist.bernoulli', 't.dist.normal', 't.rand'],
    {'rate': 3.0},
    [('rate', 3.0, 1.0, 15.0, 'rate: the average count')],
    (-0.5, 20.5, 21, 1600.0), (0.0, 10.0, 'viridis'),
    hist_text='one bar per count: 0 at the left, 20 at the right',
    extra_prose='')

# ==================================================================== t.uniform
fill_page('uniform', 't.uniform - fill a tensor with even noise',
    "Fills a tensor with random values spread evenly between from and to,\nkeeping the tensor's shape.",
    """Send it any tensor and it sends back a tensor of the same shape and
type, every element replaced by an independent sample from the range,
every value equally likely. Use it to make uniform noise the size of
data you already have. To make the noise from nothing, with the shape
typed into the node, use t.rand or t.dist.uniform.""",
    """from (drag float, also an inlet):
The bottom of the range; samples can equal it. Default 0.

to (drag float, also an inlet):
The top of the range; samples are always below it. It must not be
below from: if it is, an error is printed in the console and nothing
is sent. Equal values give every element that value. Default 1.""",
    """The heat_map (-1 dark, 3 bright) is even noise. The histogram is a
flat block from 'from' to 'to'. Drag the sliders and the block widens
and lowers, or narrows and rises.""",
    """t.dist.uniform and t.rand make the same noise without an input tensor;
np.rand is the NumPy version. t.normal fills with a bell curve
instead.""",
    ['t.dist.uniform', 't.rand', 'np.rand', 't.normal', 't.zeros'],
    {'from': 0.0, 'to': 1.0},
    [('from', 0.0, -1.0, 0.5, 'from: the bottom of the range'),
     ('to', 1.0, 1.0, 3.0, 'to: the top of the range')],
    (-1.0, 3.0, 64, 600.0), (-1.0, 3.0, 'viridis'))

# ===================================================================== t.cauchy
fill_page('cauchy', 't.cauchy - fill a tensor with heavy-tailed noise',
    "Fills a tensor with random samples from the CAUCHY distribution - a peak\nlike a bell curve's with enormous tails - keeping the tensor's shape.",
    """Send it any tensor and it sends back a tensor of the same shape and
type, every element replaced by an independent Cauchy sample: mostly
near the median, but with wild outliers far more often than a bell
curve gives. To make the samples without an input tensor, use
t.dist.cauchy.""",
    """median (drag float, also an inlet):
The middle of the peak. Default 0.

sigma (drag float, also an inlet):
The half-width: half the samples fall within median plus or minus
sigma. Must be above 0; 0 or below is an error (printed in the
console) and nothing is sent. Default 1.""",
    """The heat_map (-3 dark, 3 bright) has scattered fully black and fully
white cells - the outliers. The histogram is a narrow peak with wide
skirts; a few samples land outside -10..10 and are not counted. Drag
sigma to widen the peak, median to move it.""",
    """t.dist.cauchy makes the same samples from a typed shape. t.normal fills
with a bell curve, which has light tails.""",
    ['t.dist.cauchy', 't.normal', 't.dist.half_cauchy', 't.uniform', 't.rand'],
    {'median': 0.0, 'sigma': 1.0},
    [('median', 0.0, -4.0, 4.0, 'median: where the peak is'),
     ('sigma', 1.0, 0.5, 3.0, 'sigma: how wide the peak is')],
    (-10.0, 10.0, 64, 900.0), (-3.0, 3.0, 'viridis'))

# ================================================================= t.log_normal
fill_page('log_normal', 't.log_normal - fill a tensor with log-normal noise',
    "Fills a tensor with random samples from the LOG-NORMAL distribution -\nalways positive, bunched low, with a long tail - keeping the tensor's shape.",
    """Each element becomes e raised to the power of a bell-curve sample, so
the LOGARITHMS of the values form a bell. Use it for positive random
sizes, durations or gains the size of data you already have. To make
the samples without an input tensor, use t.dist.log_normal.""",
    """mean (drag float, also an inlet):
The mean of the LOGARITHMS, not of the values: the median value is e
to the power mean. Default 0 (a median of 1).

std (drag float, also an inlet):
The spread of the logarithms. Must be above 0; 0 or below is an error
(printed in the console) and nothing is sent. Default 1.""",
    """The heat_map (0 dark, 5 bright) is mostly dark with bright specks. The
histogram rises steeply from 0, peaks below 1 and trails off to the
right. Raise mean and the shape stretches out; lower std and the tail
shortens.""",
    """t.dist.log_normal makes the same samples from a typed shape. t.normal
fills with the bell this is built on; t.exponential is another
positive fill.""",
    ['t.dist.log_normal', 't.normal', 't.exponential', 't.rand'],
    {'mean': 0.0, 'std': 1.0},
    [('mean', 0.0, -0.5, 1.5, 'mean: the middle of the logarithms'),
     ('std', 1.0, 0.5, 1.5, 'std: the spread of the logarithms')],
    (0.0, 10.0, 64, 1000.0), (0.0, 5.0, 'viridis'))

# ================================================================ t.exponential
fill_page('exponential', 't.exponential - fill a tensor with waiting times',
    "Fills a tensor with random samples from the EXPONENTIAL distribution - the\nwait for an event that comes at a steady random rate - keeping the tensor's shape.",
    """Each element becomes an independent waiting time: short ones the most
likely, long ones rarer and rarer. Use it for random gaps between
events, or for positive noise that is usually small, the size of data
you already have. To make the samples without an input tensor, use
t.dist.exponential.""",
    """lambda (drag float, also an inlet):
The rate of the events. The average wait is 1 / lambda. The widget will
not go below 0.000001. Default 1.""",
    """The heat_map (0 dark, 3 bright) is mostly dark with brighter specks.
The histogram is tallest at 0 and falls away in a smooth curve. Raise
lambda and it crowds against 0; lower it and it stretches out.""",
    """t.dist.exponential makes the same samples from a typed shape.
t.geometric is the whole-number version (how many tries until the
first success). t.poisson counts events instead of timing them.""",
    ['t.dist.exponential', 't.geometric', 't.poisson', 't.log_normal', 't.rand'],
    {'lambda': 1.0},
    [('lambda', 1.0, 0.2, 2.0, 'lambda: events per unit of time')],
    (0.0, 10.0, 64, 1200.0), (0.0, 3.0, 'viridis'))

# ================================================================== t.geometric
fill_page('geometric', 't.geometric - fill a tensor with try-counts',
    "Fills a tensor with random whole numbers from the GEOMETRIC distribution:\nhow many tries it takes to get the first success, keeping the tensor's shape.",
    """Each element becomes the number of the try on which a coin with
success chance p first comes up: 1 (success at once) is the most
likely, 2 next, and so on, each count (1 - p) times as likely as the
one before. It is the whole-number cousin of the exponential wait. Use
it for random run lengths, repeat counts, or how many steps before a
change. There is no t.dist.geometric.""",
    """p (drag float, also an inlet):
The chance of success on each try, kept by the widget between 0.000001
and 0.999999. The average count is 1 / p. Default 0.5.""",
    """The heat_map (1 dark, 8 bright) shows the counts as stepped tones. The
histogram has one bar per count, 1 at the left to 20 at the right, each
bar a fixed fraction of the one before: at p 0.5 half the samples are
1, a quarter 2, an eighth 3. Drag p down and the bars spread to the
right; up towards 1 and nearly everything is 1.""",
    """t.exponential is the continuous version. t.dist.poisson and
t.poisson are the other whole-number counts. t.dist.bernoulli is a
single try.""",
    ['t.exponential', 't.poisson', 't.dist.poisson', 't.dist.bernoulli', 't.rand'],
    {'p': 0.5},
    [('p', 0.5, 0.05, 0.95, 'p: the chance of success per try')],
    (0.5, 20.5, 20, 4096.0), (1.0, 8.0, 'viridis'),
    hist_text='one bar per count: 1 at the left, 20 at the right',
    dtype_note="""Unlike the other fills it also works on an integer tensor, and the
counts then come out as integers; on a float tensor they are floats
(1.0, 2.0 ...), always whole.""")


# ------------------------------------------------- a ramp of probabilities/rates
def ramp_sub(top, host='ramp'):
    """make_ramp: a 64 x 64 tensor rising from 0 at the left edge to 'top' at
    the right, every row the same. t.linspace makes one row, t.repeat stacks
    64 of it, * scales it; a number into 'top' rescales and re-sends, a bang
    into 'again' re-sends."""
    return sub_comments({
        'name': 'make_ramp',
        'host': host,
        'demo': [
            {'key': 'lb', 'init': 'load_bang', 'pos': (40, 40), 'w': 88, 'h': 46},
            {'key': 'intop', 'init': 'in top', 'pos': (200, 40), 'w': 150, 'h': 80},
            {'key': 'inag', 'init': 'in again', 'pos': (400, 40), 'w': 150, 'h': 80},
            {'key': 'tbf', 'init': 't bang float', 'pos': (200, 150), 'w': 110, 'h': 46},
            {'key': 'lin', 'init': 't.linspace 0 1 %d' % SIDE, 'pos': (40, 230), 'w': 200, 'h': 120,
             'props': {'start': 0.0, 'stop': 1.0, 'steps': SIDE}},
            {'key': 'cl', 'comment': True, 'text': 'one row: 0 to 1 in 64 steps',
             'pos': (260, 240)},
            {'key': 'rep', 'init': 't.repeat %d 1' % SIDE, 'pos': (40, 380), 'w': 200, 'h': 70},
            {'key': 'cr', 'comment': True, 'text': '64 copies of the row: a 64 x 64 ramp',
             'pos': (260, 390)},
            {'key': 'mul', 'init': '* %s' % ('%.1f' % top), 'pos': (40, 480), 'w': 150, 'h': 70},
            {'key': 'cm', 'comment': True, 'text': 'scaled so the right edge is top',
             'pos': (260, 490)},
            {'key': 'out', 'init': 'out ramp', 'pos': (40, 580), 'w': 150, 'h': 80},
        ],
        'links': [('lb', 'out', 'lin', '###input'),
                  ('intop', 'top', 'tbf', ''),
                  ('tbf', 'float', 'mul', 'operand'), ('tbf', 'bang', 'lin', '###input'),
                  ('inag', 'again', 'lin', '###input'),
                  ('lin', 'linspace out', 'rep', 'tensor in'),
                  ('rep', 'repeated tensor out', 'mul', 'in'),
                  ('mul', 'result', 'out', 'ramp')],
    })


# ==================================================================== t.bernoulli
body = """Turns a tensor of PROBABILITIES into a tensor of random 0s and 1s: each
element comes out 1 with the probability it holds, otherwise 0.

Unlike t.dist.bernoulli, which flips every coin with the same odds,
this node gives each element its own odds - the input IS the parameter.
Feed it a soft mask or a picture and you get random dots whose density
follows it: dithering, a fading-in pattern of sparks, dropout with a
different rate per cell, a random decision per joint weighted by how
fast it moves.

SYNTAX:
t.bernoulli
The node takes no arguments.

INPUTS and PARAMETERS:

tensor in:
The probabilities, one per element, each from 0 (never 1) to 1 (always
1). Every tensor that arrives is sampled and sent; this is the only
inlet. A list or number is turned into a tensor first. If any value is
below 0 or above 1 the node sends NOTHING (and says why only with the
app's verbose setting on), so clip a signal into 0..1 first (t.clamp).
The tensor must be floating-point; an integer one gives an error in
the console.

OUTPUTS:

tensor out:
0s and 1s, the same shape, dtype and device as the input - a float
tensor of 0.0 and 1.0, not booleans. A new random draw every time a
tensor arrives.

READING THE DEMO:
make_ramp sends a 64 x 64 ramp of probabilities: 0 at the left edge,
rising evenly to top at the right (left heat_map, black 0 to white 1).
t.bernoulli turns it into the right heat_map: no dots at all on the
left, sparse dots in the middle, solid at the right if top is 1. Drag
top and the whole ramp, and the dot density with it, scales down.
Click 'again' for a fresh draw from the same ramp. The two floats are
the mean of the probabilities and the fraction of 1s that came out -
always very close.

RELATED:
t.dist.bernoulli flips coins with one shared probability, from a typed
shape. t.poisson does the same per-element trick for counts. t.rand
with a > comparison does this by hand."""
demo = [
    {'key': 'stop', 'init': 'slider', 'pos': (30, 62), 'w': 220, 'h': 60,
     'props': {'': 1.0, 'min': 0.0, 'max': 1.0, 'format': '%.2f', 'width': 200}},
    {'key': 'ctop', 'comment': True, 'text': 'top: the probability at the right edge',
     'pos': (30, 125)},
    {'key': 'bag', 'init': 'button', 'pos': (290, 62), 'w': 45, 'h': 42},
    {'key': 'cbag', 'comment': True, 'text': 'again: a fresh draw', 'pos': (345, 70)},
    {'key': 'ramp', 'init': 'p make_ramp', 'pos': (30, 170), 'w': 220, 'h': 70},
    {'key': 'cramp', 'comment': True, 'text': 'a 64 x 64 ramp - open it to see how',
     'pos': (270, 180)},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 270), 'w': 250, 'h': 270,
     'props': HM(230, 230, SIDE, 0.0, 1.0, color='hot')},
    {'key': 'nd', 'init': 't.bernoulli', 'pos': (310, 270), 'w': 150, 'h': 70},
    {'key': 'hout', 'init': 'heat_map', 'pos': (310, 360), 'w': 250, 'h': 270,
     'props': HM(230, 230, SIDE, 0.0, 1.0, color='hot')},
    {'key': 'chin', 'comment': True,
     'text': 'left: the probabilities; right: the 0s and 1s drawn from them',
     'pos': (30, 640)},
    {'key': 'mi', 'init': 't.mean', 'pos': (30, 680), 'w': 160, 'h': 100},
    {'key': 'mo', 'init': 't.mean', 'pos': (310, 680), 'w': 160, 'h': 100},
    {'key': 'fi', 'init': 'float', 'pos': (30, 800), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'fo', 'init': 'float', 'pos': (310, 800), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'cf', 'comment': True,
     'text': 'mean probability, and the fraction of 1s', 'pos': (30, 850)},
]
links = [('stop', '', 'ramp', 'top'), ('bag', '', 'ramp', 'again'),
         ('ramp', 'ramp', 'hin', 'y'), ('ramp', 'ramp', 'nd', 'tensor in'),
         ('nd', 'tensor out', 'hout', 'y'),
         ('ramp', 'ramp', 'mi', 'tensor in'), ('nd', 'tensor out', 'mo', 'tensor in'),
         ('mi', 'output', 'fi', ''), ('mo', 'output', 'fo', '')]
demo += see_also(900, ['t.dist.bernoulli', 't.poisson', 't.rand', 't.clamp', 'np.rand'])
print(build('t.bernoulli', 't.bernoulli - a coin flip per element', body, demo, links,
            demo_width=600, text_width=760, text_height=th(body), subpatch=ramp_sub(1.0)))

# ===================================================================== t.poisson
body = """Turns a tensor of RATES into a tensor of random whole-number COUNTS:
each element is a Poisson sample whose average is the rate it holds.

Unlike t.dist.poisson, which uses one rate for every element, this node
gives each element its own rate - the input IS the parameter. Feed it
a picture of brightness and you get photon shot noise; feed it how busy
each region should be and you get how many events each one gets this
frame.

SYNTAX:
t.poisson
The node takes no arguments.

INPUTS and PARAMETERS:

tensor in:
The rates, one per element, each 0 or above (0 always gives 0). Every
tensor that arrives is sampled and sent; this is the only inlet. A list
or number is turned into a tensor first. If any value is negative the
node sends NOTHING (and says why only with the app's verbose setting
on). The tensor must be floating-point; an integer one gives an error
in the console.

OUTPUTS:

tensor out:
Counts, the same shape, dtype and device as the input: whole numbers
stored as floats (0.0, 1.0, 2.0 ...). A new draw every time a tensor
arrives.

""" + HIST_READ + """

READING THE DEMO:
Top: the rate slider sets t.full's value and clicks it, so a 64 x 64
tensor holding the SAME rate everywhere goes in. Its counts are the
Poisson distribution itself: the heat_map (0 dark, 10 bright) is
stepped speckle, and the histogram has one bar per count, 0 at the
left to 20 at the right, peaking near the rate. Drag rate and the hump
moves and widens.
Bottom: make_ramp sends rates rising from 0 at the left to top at the
right (left heat_map), and the counts (right heat_map, same colours)
are dark and sparse at the left and brighter and noisier to the right
- each element drew from its own rate. Click 'again' for a fresh draw.

RELATED:
t.dist.poisson makes Poisson samples with one rate from a typed shape.
t.bernoulli is the per-element yes/no version. t.geometric and
t.exponential are other counts and waits."""
demo = [
    {'key': 'lb', 'init': 'load_bang', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'srt', 'init': 'slider', 'pos': (150, 62), 'w': 220, 'h': 60,
     'props': {'': 3.0, 'min': 1.0, 'max': 15.0, 'format': '%.2f', 'width': 200}},
    {'key': 'csrt', 'comment': True, 'text': 'rate: the same everywhere', 'pos': (150, 125)},
    {'key': 'trt', 'init': 't bang float', 'pos': (150, 160), 'w': 110, 'h': 46},
    {'key': 'full', 'init': 't.full %d %d' % (SIDE, SIDE), 'pos': (30, 240), 'w': 170, 'h': 120,
     'props': {'value': 3.0, 'dim 0': SIDE, 'dim 1': SIDE}},
    {'key': 'cfull', 'comment': True, 'text': '64 x 64, every element = rate',
     'pos': (220, 250)},
    {'key': 'nd', 'init': 't.poisson', 'pos': (30, 390), 'w': 150, 'h': 70},
    {'key': 'hm', 'init': 'heat_map', 'pos': (300, 390), 'w': 250, 'h': 270,
     'props': HM(230, 230, SIDE, 0.0, 10.0)},
    {'key': 'chm', 'comment': True, 'text': 'the counts: 0 dark, 10 bright',
     'pos': (300, 670)},
    {'key': 'hc', 'init': 't.histc', 'pos': (30, 490), 'w': 220, 'h': 150,
     'props': {'bin count': 21, 'min': -0.5, 'max': 20.5}},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 710), 'w': 540, 'h': 280,
     'props': HIST_PLOT(21, 1600.0)},
    {'key': 'cpl', 'comment': True, 'text': 'one bar per count: 0 at the left, 20 at the right',
     'pos': (30, 995)},
    # the ramp
    {'key': 'stop', 'init': 'slider', 'pos': (30, 1050), 'w': 220, 'h': 60,
     'props': {'': 10.0, 'min': 0.0, 'max': 20.0, 'format': '%.2f', 'width': 200}},
    {'key': 'ctop', 'comment': True, 'text': 'top: the rate at the right edge',
     'pos': (30, 1113)},
    {'key': 'bag', 'init': 'button', 'pos': (290, 1050), 'w': 45, 'h': 42},
    {'key': 'cbag', 'comment': True, 'text': 'again: a fresh draw', 'pos': (345, 1058)},
    {'key': 'ramp', 'init': 'p make_ramp', 'pos': (30, 1150), 'w': 220, 'h': 70},
    {'key': 'nd2', 'init': 't.poisson', 'pos': (310, 1150), 'w': 150, 'h': 70},
    {'key': 'hin', 'init': 'heat_map', 'pos': (30, 1250), 'w': 250, 'h': 270,
     'props': HM(230, 230, SIDE, 0.0, 20.0)},
    {'key': 'hout', 'init': 'heat_map', 'pos': (310, 1250), 'w': 250, 'h': 270,
     'props': HM(230, 230, SIDE, 0.0, 20.0)},
    {'key': 'chin', 'comment': True,
     'text': 'left: the rates; right: the counts drawn from them (0 dark, 20 bright)',
     'pos': (30, 1530)},
]
links = [('lb', 'out', 'full', '###input'), ('srt', '', 'trt', ''),
         ('trt', 'float', 'full', 'value'), ('trt', 'bang', 'full', '###input'),
         ('full', 'filled tensor', 'nd', 'tensor in'),
         ('nd', 'tensor out', 'hm', 'y'), ('nd', 'tensor out', 'hc', ''),
         ('hc', 'histogram tensor out', 'pl', 'y'),
         ('stop', '', 'ramp', 'top'), ('bag', '', 'ramp', 'again'),
         ('ramp', 'ramp', 'hin', 'y'), ('ramp', 'ramp', 'nd2', 'tensor in'),
         ('nd2', 'tensor out', 'hout', 'y')]
demo += see_also(1580, ['t.dist.poisson', 't.bernoulli', 't.geometric', 't.full', 't.rand'])
print(build('t.poisson', 't.poisson - a random count per element', body, demo, links,
            demo_width=600, text_width=760, text_height=th(body), subpatch=ramp_sub(10.0)))
