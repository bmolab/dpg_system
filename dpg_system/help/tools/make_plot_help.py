"""plot, heat_map, heat_scroll - looking at numbers."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

HM = lambda n=32, lo=0.0, hi=1.0, fmt='%.2f', mode='heat_map', col='viridis': {
    'color': col, 'width': 200, 'height': 100, 'sample count': n,
    'min y': lo, 'max y': hi, 'update_mode': mode, 'number format': fmt}

body = """These are how you see what is actually going past.

THE NODES:

plot         a graph of values over time
heat_map     an array as a block of colour, redrawn each time
heat_scroll  the same, but keeping history and scrolling

heat_map AND heat_scroll ARE ONE NODE:
The name you type sets which way it starts, and 'update_mode' switches between
them afterwards - so a heat_map can become a heat_scroll without being replaced.

The difference is what an incoming array MEANS:

heat_map     the array IS the picture. Send 32 numbers and you see 32 cells,
             replaced entirely next time. Use it for something that has a shape
             now - a spectrum, a matrix, a set of joint values.
heat_scroll  the array is one COLUMN, and previous columns stay. Use it for
             something whose history matters - a spectrum over time, a body's
             joints over the last few seconds.

Switching to heat_scroll bumps the sample count to 200 if it was 1, because a
scroll of one column is not a scroll.

SET min y AND max y, OR YOU WILL SEE NOTHING:
This is the single commonest reason a display looks broken. The colours and the
graph height are stretched between these two numbers, and anything outside is
flattened against the edges.

A signal running 0 to 0.1 shown on a 0-to-1 scale is a barely-visible smudge; a
signal running 0 to 3000 on the same scale is a solid block. Neither is wrong,
and neither tells you anything.

So find out what range your data actually occupies and set these to match. If a
display is blank, uniform, or saturated, check this before suspecting anything
upstream - it is almost always this.

'update style' ON plot - THE SAME THREE-WAY CHOICE AS buffer:
input is stream of samples        each value is one more moment. The ordinary
                                  case: a scalar arrives, the graph scrolls.
input is multi-channel sample     one moment of several signals at once, drawn
                                  as several traces.
buffer holds one sample of input  the whole array is the graph, redrawn each
                                  time - a shape rather than a history.

Getting this wrong is the second commonest reason a plot looks wrong: an array
sent to a plot expecting a stream is drawn as a burst of successive samples
rather than as the shape it is.

'style' picks line, scatter, stair, stem or bar. stem and scatter are much
easier to read for anything sparse or event-like; line implies a continuity that
may not be there.

'sample count' IS A LENGTH OF TIME:
It is how many samples are kept, so at sixty frames a second, 60 is one second
and 600 is ten. Think of it as the width of the window you are looking through
rather than as a number.

SEND IT 'dump' TO GET THE DATA BACK OUT:
All three hold a buffer, and they will hand it over: send the word 'dump' to the
inlet and the collected history comes out of the outlet.

That turns a display into a recorder. Watch something happen, then dump the
buffer to save it, analyse it, or feed it to something that wants a whole
window rather than a stream.

Nothing comes out of the outlet at any other time - these are displays first,
and they only speak when asked.

THE COLOUR MAPS:
Seventeen of them. viridis is the sensible default for data, because it is even
- equal steps in value look like equal steps in colour, and it survives being
printed in grey. jet is the familiar rainbow and is the one to avoid for
anything quantitative: it invents boundaries where the data has none.

greys, hot and cool for single-ended quantities; red-blue and the other
diverging maps when zero is meaningful and you want to see which side of it you
are on.

SYNTAX:
plot
heat_map
heat_scroll

EXAMPLE:
heat_scroll

INPUTS and PARAMETERS:

y:
The data. This is the only inlet.

min y / max y:
The range being displayed. Set these.

sample count:
How much history to keep.

width / height:
The size of the display.

color:
Which colour map.

style (plot):
line, scatter, stair, stem or bar.

update style (plot) / update_mode (heat):
What an incoming value means, and static against scrolling.

number format:
How the numbers are written when they are shown.

OUTPUTS: 

The single outlet sends the whole buffer, and only in reply to 'dump'.

RELATED:
buffer and rolling_buffer keep the same kind of history without drawing it.
profile is a plot you can draw into with the mouse, to make a curve by hand."""

demo = [
    {'key': 'sig', 'init': 'signal', 'pos': (30, 62), 'w': 129, 'h': 78,
     'props': SIG('sin', 3.0)},
    {'key': 'pl', 'init': 'plot', 'pos': (30, 155), 'w': 300, 'h': 180,
     'props': PLOT(-1.0, 1.0, 200)},
    {'key': 'c0', 'comment': True, 'text': 'one value at a time, scrolling',
     'pos': (30, 345)},

    {'key': 'pl2', 'init': 'plot', 'pos': (380, 155), 'w': 300, 'h': 180,
     'props': PLOT(-1.0, 1.0, 200, 'stem')},
    {'key': 'c1', 'comment': True, 'text': 'the same data as stem - easier to read',
     'pos': (380, 345)},
    {'key': 'c2', 'comment': True, 'text': 'for anything sparse or event-like',
     'pos': (380, 375)},

    {'key': 'btn', 'init': 'button', 'pos': (30, 420), 'w': 88, 'h': 46},
    {'key': 'rnd', 'init': 'np.rand 32', 'pos': (30, 480), 'w': 180, 'h': 180},
    {'key': 'hm', 'init': 'heat_map', 'pos': (250, 480), 'w': 208, 'h': 148,
     'props': HM(32)},
    {'key': 'c3', 'comment': True, 'text': 'heat_map: the array IS the picture,',
     'pos': (250, 640)},
    {'key': 'c4', 'comment': True, 'text': 'replaced whole each time',
     'pos': (250, 670)},

    {'key': 'hs', 'init': 'heat_scroll', 'pos': (30, 720), 'w': 208, 'h': 148,
     'props': HM(120, -1.0, 1.0, '%.2f', 'heat_scroll')},
    {'key': 'c5', 'comment': True, 'text': 'heat_scroll: each array is one column,',
     'pos': (30, 880)},
    {'key': 'c6', 'comment': True, 'text': 'and the past stays on screen',
     'pos': (30, 910)},

    {'key': 'hm2', 'init': 'heat_map', 'pos': (480, 480), 'w': 208, 'h': 148,
     'props': HM(32, 0.0, 20.0)},
    {'key': 'c7', 'comment': True, 'text': 'the SAME data with max y at 20 -',
     'pos': (480, 640)},
    {'key': 'c8', 'comment': True, 'text': 'nearly blank. If a display looks wrong,',
     'pos': (480, 670)},
    {'key': 'c9', 'comment': True, 'text': 'check min y and max y first',
     'pos': (480, 700)},

    {'key': 'm1', 'init': 'message', 'pos': (480, 760), 'w': 140, 'h': 42,
     'props': {'text in': 'dump', 'font size': '24'}},
    {'key': 'inf', 'init': 'info', 'pos': (480, 820), 'w': 240, 'h': 80},
    {'key': 'c10', 'comment': True, 'text': "click 'dump' and the plot hands back",
     'pos': (480, 915)},
    {'key': 'c11', 'comment': True, 'text': 'its whole buffer - a display becomes',
     'pos': (480, 945)},
    {'key': 'c12', 'comment': True, 'text': 'a recorder', 'pos': (480, 975)},
]
links = [('sig', '', 'pl', 'y'), ('sig', '', 'pl2', 'y'),
         ('sig', '', 'hs', 'y'),
         ('btn', '', 'rnd', ''),
         ('rnd', 'random array', 'hm', 'y'),
         ('rnd', 'random array', 'hm2', 'y'),
         ('m1', 'message out', 'pl', 'y'),
         ('pl', '', 'inf', 'in')]
print(build('plot', 'plot, heat_map, heat_scroll - seeing the numbers', body,
            demo, links, demo_width=760, text_width=800, text_height=770))
