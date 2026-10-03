"""plot; heat_map and heat_scroll - looking at numbers."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

HM = lambda n=32, lo=0.0, hi=1.0, fmt='%.2f', mode='heat_map', col='viridis': {
    'color': col, 'width': 200, 'height': 100, 'sample count': n,
    'min y': lo, 'max y': hi, 'update_mode': mode, 'number format': fmt}


# ----------------------------------------------------------------------- plot
body = """plot draws a graph of the numbers going past - the quickest way to see what a
signal is actually doing.

SYNTAX:
plot

No arguments; everything is set in the node's options.

EXAMPLE:
plot

'update style' - WHAT AN INCOMING VALUE MEANS:
A three-way choice, the same one as on buffer. Getting it wrong is the commonest
reason a plot looks wrong after a bad range.

input is stream of samples        each value is one more moment. The ordinary
                                  case: a number arrives, the graph scrolls on
                                  by one. An array is taken as several moments
                                  in a row - and an array LONGER than the sample
                                  count makes the plot grow to fit it.
input is multi-channel sample     each array is one moment of several signals
                                  at once, drawn as one trace per element. A
                                  two-dimensional array of channels by samples
                                  replaces the whole display at once.
buffer holds one sample of input  the whole array is the graph, redrawn each
                                  time - a shape rather than a history. The
                                  sample count follows the array's length.

SET min y AND max y:
The graph's height is stretched between these two numbers, and anything outside
them is off the top or the bottom. They start at -1 and 1. A signal running 0 to
0.1 on that scale is a barely-visible line near the middle; a signal running 0 to
3000 is off the top. If a plot looks blank or flat, check these first.

'sample count' IS A LENGTH OF TIME:
In stream mode it is how many samples are kept, so at sixty frames a second,
60 is one second and 600 is ten. Default 200. Think of it as the width of the
window you are looking through rather than as a number.

'style':
line, scatter, stair, stem or bar. stem and scatter are much easier to read for
anything sparse or event-like; line implies a continuity that may not be there.
Changing the style quietly puts update style back to stream of samples, although
the menu still shows your old choice - pick update style again after changing
the style.

SEND IT 'dump' TO GET THE DATA BACK OUT:
Send the word 'dump' to the inlet and the collected history comes out of the
outlet as a NumPy array, oldest first. That turns a display into a recorder:
watch something happen, then dump it to save, analyse, or feed to something that
wants a whole window rather than a stream. With several traces, only the first
trace is sent. Nothing comes out of the outlet at any other time.

INPUTS and PARAMETERS:

y:
The data - a number, a list, a NumPy array or a PyTorch tensor - or the word
'dump'. This is the only inlet.

In the options:

color:
The colour map used for the traces when there are several. 'none' uses the
standard colours; changing the style sets this back to 'none'.

width / height:
The size of the display, in pixels. You can also drag the bar under the graph.

style:
line, scatter, stair, stem or bar.

update style:
What an incoming value means - see above.

sample count:
How many samples are shown.

min x / max x:
The stretch of samples shown, from 0 to the sample count. Put back to the whole
buffer whenever the sample count changes.

min y / max y:
The range of values shown. Set these.

OUTPUTS:

The single outlet sends the buffer, and only in reply to 'dump'.

RELATED:
heat_map         an array as a block of colour; heat_scroll keeps its history
buffer           keeps the same kind of history without drawing it
rolling_buffer   a scrolling history that it passes on as an array
profile          a plot you can draw into with the mouse, to make a curve by hand"""

demo = [
    {'key': 'sig', 'init': 'signal', 'pos': (30, 62), 'w': 129, 'h': 78,
     'props': SIG('sin', 3.0)},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 155), 'w': 208, 'h': 178,
     'props': PLOT(-1.0, 1.0, 200)},
    {'key': 'c0', 'comment': True, 'text': 'one value at a time, scrolling',
     'pos': (300, 155)},
    {'key': 'pl2', 'init': 'plot', 'pos': (30, 360), 'w': 208, 'h': 178,
     'props': PLOT(-1.0, 1.0, 200, 'stem')},
    {'key': 'c1', 'comment': True, 'text': 'the same data as stem - easier to read\nfor anything sparse or event-like',
     'pos': (300, 360)},
    {'key': 'btn', 'init': 'button', 'pos': (30, 570), 'w': 45, 'h': 42},
    {'key': 'cb', 'comment': True, 'text': 'click for a new array of 32 numbers',
     'pos': (300, 570)},
    {'key': 'rnd', 'init': 'np.rand 32', 'pos': (30, 625), 'w': 123, 'h': 84},
    {'key': 'pl3', 'init': 'plot', 'pos': (30, 730), 'w': 208, 'h': 178,
     'props': {'color': 'none', 'width': 200, 'height': 128, 'style': 'bar',
               'update style': 'buffer holds one sample of input', 'sample count': 32,
               'min x': 0.0, 'max x': 32.0, 'min y': 0.0, 'max y': 1.0}},
    {'key': 'c2', 'comment': True, 'text': "update style 'buffer holds one sample of input':\nthe array IS the graph, drawn as bars",
     'pos': (300, 730)},
    {'key': 'm1', 'init': 'message', 'pos': (30, 940), 'w': 130, 'h': 42,
     'props': {'text in': 'dump', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': "click 'dump' and the top plot hands back\nits whole history - a display becomes\na recorder",
     'pos': (300, 940)},
    {'key': 'inf', 'init': 'info', 'pos': (30, 1000), 'w': 235, 'h': 42},
]
links = [('sig', '', 'pl', 'y'), ('sig', '', 'pl2', 'y'),
         ('btn', '', 'rnd', ''),
         ('rnd', 'random array', 'pl3', 'y'),
         ('m1', 'message out', 'pl', 'y'),
         ('pl', '', 'inf', 'in')]
print(build('plot', 'plot - a graph of the numbers going past', body,
            demo, links, demo_width=680, text_width=800, text_height=1430))

# ------------------------------------------------------------------- heat_map
body = """heat_map shows an array as a block of colour - each number one cell, its colour
set by its value. heat_scroll is the same node, keeping history.

THE NODES:

heat_map      the array IS the picture, replaced whole each time
heat_scroll   each array is one COLUMN, and earlier columns stay, scrolling

They are one node. The name you type sets which way it starts, and the
'update_mode' option switches between them afterwards - so a heat_map can become
a heat_scroll without being replaced.

The difference is what an incoming array MEANS:

heat_map      Send 32 numbers and you see 32 cells, replaced entirely by the
              next array. A two-dimensional array is shown as a grid. Use it
              for something that has a shape now - a spectrum, a matrix, a set
              of joint values. The display adjusts itself to the array's size.
heat_scroll   Each array, whatever its shape, is laid out as one column of
              cells, and the earlier columns stay on screen. Use it for
              something whose history matters - a spectrum over time, a body's
              joints over the last few seconds.

Switching to heat_scroll sets the sample count to 200 if it was 1, because a
scroll of one column is not a scroll.

SET min y AND max y, OR YOU WILL SEE NOTHING:
This is the commonest reason a heat map looks broken. The colours are stretched
between these two numbers - min y gets the colour at one end of the map, max y
the colour at the other - and anything outside is flattened to the end colour.
They start at 0 and 1.

A signal running 0 to 0.1 on a 0-to-1 scale is a barely-visible smudge; a signal
running 0 to 3000 on the same scale is a solid block. Neither is wrong, and
neither tells you anything. Find out what range your data actually occupies and
set these to match. If a display is blank, uniform or saturated, check this
before suspecting anything upstream.

THE COLOUR MAPS:
viridis is the default and the sensible choice for data, because it is even -
equal steps in value look like equal steps in colour, and it survives being
printed in grey. jet is the familiar rainbow and is the one to avoid for anything
quantitative: it invents boundaries where the data has none. greys, hot and cool
suit quantities that run from nothing upwards; red-blue and the other two-sided
maps suit values where zero is meaningful and you want to see which side of it
you are on - set min y and max y the same distance either side of zero.

SYNTAX:
heat_map
heat_scroll

No arguments; everything is set in the node's options.

EXAMPLE:
heat_scroll

INPUTS and PARAMETERS:

y:
The data - a NumPy array, a list, a PyTorch tensor or a single number - or the
word 'dump'. This is the only inlet.

In the options:

color:
Which colour map. Default viridis.

width / height:
The size of the display, in pixels. You can also drag the bar under it.

sample count:
For heat_map, the number of cells across - set automatically from the array.
For heat_scroll, how many columns of history are kept.

min y / max y:
The values given the two end colours. Set these.

update_mode:
heat_map or heat_scroll.

number format:
How each cell's value is written on it, such as %.2f; empty for no numbers.
Default %.3f. The numbers are hidden automatically while the cells are narrower
than 40 pixels or shorter than 16, and come back when there is room.

SEND IT 'dump' TO GET THE DATA BACK OUT:
Send the word 'dump' and the buffer comes out of the outlet as a NumPy array: for
heat_map, the array last shown (for a grid, only its first row); for heat_scroll,
only the FIRST row - the history of the first element. Nothing comes out of the
outlet at any other time.

OUTPUTS:

The single outlet sends the buffer, and only in reply to 'dump'.

RELATED:
plot       a graph of values over time
spectrum   a natural source: the energy in each frequency band
vector     shows an array as numbers rather than colours"""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'cb', 'comment': True, 'text': 'click for a new array of 32 numbers',
     'pos': (300, 62)},
    {'key': 'rnd', 'init': 'np.rand 32', 'pos': (30, 117), 'w': 123, 'h': 84},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 225), 'w': 208, 'h': 150,
     'props': HM(32)},
    {'key': 'c0', 'comment': True, 'text': 'heat_map: the array IS the picture,\nreplaced whole each time',
     'pos': (300, 225)},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (30, 400), 'w': 208, 'h': 150,
     'props': HM(32, 0.0, 20.0)},
    {'key': 'c1', 'comment': True, 'text': 'the SAME data with max y at 20 -\nnearly blank. If a display looks wrong,\ncheck min y and max y first',
     'pos': (300, 400)},
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 580), 'w': 45, 'h': 42, 'props': {'': True}},
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 635), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 100.0, 'units': 'milliseconds'}},
    {'key': 'c2', 'comment': True, 'text': 'ten new arrays a second',
     'pos': (300, 635)},
    {'key': 'rnd2', 'init': 'np.rand 8', 'pos': (30, 720), 'w': 123, 'h': 84},
    {'key': 'hs', 'init': 'heat_scroll', 'pos': (30, 825), 'w': 208, 'h': 150,
     'props': HM(60, 0.0, 1.0, '%.2f', 'heat_scroll')},
    {'key': 'c3', 'comment': True, 'text': 'heat_scroll: each array of 8 is one column,\nand the last 60 stay on screen',
     'pos': (300, 825)},
    {'key': 'm1', 'init': 'message', 'pos': (30, 1005), 'w': 130, 'h': 42,
     'props': {'text in': 'dump', 'font size': '24'}},
    {'key': 'c4', 'comment': True, 'text': "click 'dump' and the top heat_map\nhands back the array it is showing",
     'pos': (300, 1005)},
    {'key': 'inf', 'init': 'info', 'pos': (30, 1065), 'w': 235, 'h': 42},
]
links = [('btn', '', 'rnd', ''),
         ('rnd', 'random array', 'hm', 'y'),
         ('rnd', 'random array', 'hm2', 'y'),
         ('tog', '', 'met', 'on'), ('met', '', 'rnd2', ''),
         ('rnd2', 'random array', 'hs', 'y'),
         ('m1', 'message out', 'hm', 'y'),
         ('hm', '', 'inf', 'in')]
print(build('heat_map', 'heat_map, heat_scroll - an array as colour', body,
            demo, links, demo_width=680, text_width=800, text_height=1520))
