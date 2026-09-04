"""Streaming wavelet transform."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """A wavelet transform that keeps up with a live signal.

THE NODES:

t.ultracwt   the torch one - runs on the GPU, and reports phase as well
ultracwt     the numpy one

HOW THESE DIFFER FROM cwt:
The cwt node takes a whole window and transforms all of it. These take ONE
SAMPLE at a time and produce the newest column of the picture, keeping what came
before.

The saving is the point. A full transform of a window recomputes everything on
every frame, most of which has not changed; this computes only the new edge -
"only the boundary of the triangle", as the source puts it. That is what makes a
wavelet transform affordable at frame rate on a live signal.

So: cwt for a recorded window you want to look at, ultracwt for something
happening now.

'scales' IS THE FREQUENCY AXIS:
A list of wavelet widths. Narrow wavelets answer to fast changes, wide ones to
slow, so the list you give is the set of time-scales the transform will report -
one row of output per entry.

Powers of two, or steps between them, are the usual choice, because the interest
in a signal tends to be spread evenly across octaves rather than evenly across
frequencies. You can send the list from another node as well as typing it.

'scale based attenuation' SHOULD USUALLY STAY ON:
A wide wavelet integrates over more samples, so without correction the slow
scales simply produce bigger numbers than the fast ones and the picture is a
gradient rather than a reading. Attenuation divides each scale by its width, so
the rows become comparable to one another.

Turn it off only if you want the raw energy, and expect the low rows to dominate
everything you do with it.

'unskew' - EVERY SCALE HAS A DIFFERENT LATENCY:
This is the subtle one, and it matters for anything that has to line up in time.

A wide wavelet needs a long stretch of signal, so the answer it gives now
actually describes a moment further in the past than the answer from a narrow
one. Left alone, an event appears in the fast rows immediately and in the slow
rows some time later - the picture is smeared diagonally.

With unskew on, each scale's value is written back into the position it really
belongs to, so a single event lines up vertically across all the rows.
'unskew_scale' sets how far back each scale is assumed to refer, as a multiple
of its width.

There is a cost: with unskew on the node sends the whole two-dimensional frame
rather than just the newest column, because it is now revising the recent past
as well as adding to it.

'frame size' is how much history the node holds to work in.

A NOTE ON 'phase out':
On t.ultracwt this outlet carries the IMAGINARY component of the convolution,
not the phase angle in radians. It moves with the phase and is bounded by the
magnitude rather than by pi, so it is usable as an oscillating signal but is not
an angle and should not be treated as one.

The numpy node has no phase outlet, and a 'mode' option instead - normal, half,
or unskewed.

SYNTAX:
t.ultracwt
ultracwt

EXAMPLE:
t.ultracwt

INPUTS and PARAMETERS:

in 1:
One sample, or a frame. Receiving it advances the transform.

scales:
The wavelet widths - the frequency axis. A list, typed or sent.

frame size:
How much history to hold.

unskew / unskew_scale:
Line the scales up in time, and how far back each is taken to refer.

scale based attenuation:
Divide each scale by its width so the rows are comparable. Leave on.

device / dtype (t.ultracwt):
Where it runs.

OUTPUTS: 

cwt out:
The magnitudes - one value per scale, or the whole frame when unskewing.

phase out (t.ultracwt):
The imaginary component. See the note above.

RELATED:
cwt for a whole window at once, when the signal is already recorded.
heat_scroll is the natural destination - each frame is a column, and the history
scrolls."""

demo = [
    {'key': 'sig', 'init': 'signal', 'pos': (30, 62), 'w': 129, 'h': 78,
     'props': SIG('sin', 2.0)},
    {'key': 'c0', 'comment': True, 'text': 'one sample at a time, live',
     'pos': (30, 155)},

    {'key': 'sc', 'init': 'string', 'pos': (200, 62), 'w': 380, 'h': 42,
     'props': {'text in': '2 4 8 16 32 64', 'font size': '24', 'width': 340}},
    {'key': 'c1', 'comment': True, 'text': 'the scales - one output row each.\nPowers of two, because interest in a\nsignal spreads evenly across octaves',
     'pos': (200, 112)},

    {'key': 'uc', 'init': 't.ultracwt', 'pos': (30, 220), 'w': 320, 'h': 320},
    {'key': 'c4', 'comment': True, 'text': 'attenuation ON, or the slow rows just\nproduce bigger numbers than the fast\nones and the picture is a gradient',
     'pos': (30, 555)},

    {'key': 'hs', 'init': 'heat_scroll', 'pos': (400, 220), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 200,
               'min y': 0.0, 'max y': 0.15, 'update_mode': 'heat_scroll',
               'number format': '%.3f'}},
    {'key': 'c7', 'comment': True, 'text': 'heat_scroll is the natural destination:\neach frame is a column, history scrolls\nset max y to the range you actually see\nor it reads as blank - about 0.1 here',
     'pos': (400, 380)},

    {'key': 'c11', 'comment': True, 'text': "'unskew' lines the scales up in time.\nA wide wavelet answers about a moment\nfurther back, so without it an event\nsmears diagonally down the picture",
     'pos': (30, 665)},
]
links = [('sig', '', 'uc', 'in 1'),
         ('sc', 'string out', 'uc', 'scales'),
         ('uc', 'cwt out', 'hs', 'y')]
print(build('t.ultracwt', 'ultracwt - wavelets that keep up', body,
            demo, links, demo_width=700, text_width=800, text_height=760))
