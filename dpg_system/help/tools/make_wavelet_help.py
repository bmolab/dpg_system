"""t.cwt - the torch wavelet transform."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """A wavelet transform of a whole window, on the GPU, properly parameterised.

THE NODE:

t.cwt   a Morlet continuous wavelet transform, in torch

THERE ARE THREE WAVELET TRANSFORMS HERE - THIS IS THE CAREFUL ONE:

cwt          a whole window, several wavelet families to choose from
t.ultracwt   streaming - one sample at a time, only the newest column
t.cwt        a whole window, Morlet only, with the real parameters exposed

Use t.ultracwt when the signal is arriving now and you need to keep up. Use cwt
to try different wavelet shapes quickly. Use this one when the analysis matters
and you want to set it correctly - it is the only one of the three that lets you
say what a sample is worth in time, and it runs on the GPU in batches.

THE THREE SETTINGS ARE dt, dj AND w0 IN DISGUISE:
The labels are friendly; the quantities are the standard ones, and if you know
wavelets these are what you are looking for.

'sample_scaling' is dt - how much time one sample represents. 0.01 means a
hundred samples a second. It does NOT change the shape of the answer; it decides
what the rows MEAN. Set it wrong and every frequency you read off is wrong by
the same factor, silently, because the picture looks identical.

This is the setting to get right first, and the one most likely to be left at
its default by mistake.

'scale_distribution' is dj - how finely the scales are sampled, in octaves.
0.125 is an eighth of an octave, so eight rows per octave. This is what decides
how many rows you get:

    dj 0.25     33 rows
    dj 0.125    65 rows      (the default)
    dj 0.0625   129 rows

Halving it doubles the rows and doubles the work. Finer is smoother to look at
and tells you very little more.

'wavelet_constant' is w0 - the number of oscillations in the wavelet, and so the
trade between resolution in time and in frequency. 6 is the conventional choice
and is what most published work uses. Higher makes it sharper in frequency and
blurrier in time; lower does the reverse. It also shifts the range of scales, so
the row count changes with it - w0 12 gave 113 rows where 6 gave 129.

You cannot have both resolutions at once. That is a property of the world rather
than a limitation of the software, and w0 is where you choose your side of it.

'unbias' corrects the power spectrum so that scales are comparable to one
another. Without it the large scales read higher simply for being large. Turn it
on if you intend to compare rows; leave it off if you want the raw transform.

THE MAGNITUDES ARE NOT NORMALISED:
Nothing scales the answer to a convenient range - it depends on the amplitude of
your signal and on the settings. On the demo here the values run 0 to about 27
with a mean near 2.6, which is nowhere near the 0-to-1 a display defaults to.

So read the range off the data before setting min y and max y on whatever you
are drawing it with, or the picture will be uniformly saturated and tell you
nothing. This is the commonest way a correct transform looks broken.

THE OUTPUT HAS A BATCH DIMENSION:
The answer is (batch, scales, time) - so a single signal comes back as
(1, 65, 512) rather than (65, 512). That leading 1 will surprise anything
expecting a plain image; a t.squeeze in front of a heat_map sorts it out.

The cwt node in matrix_nodes has no such dimension, which is worth remembering
when swapping one for the other.

SYNTAX:
t.cwt

EXAMPLE:
t.cwt

INPUTS and PARAMETERS:

tensor in:
A window of signal. Receiving it does the transform.

sample_scaling:
dt - seconds per sample. Sets what the frequencies mean. Get this right.

scale_distribution:
dj - octave spacing of the scales. Sets how many rows.

wavelet_constant:
w0 - oscillations in the wavelet. 6 unless you have a reason. Trades time
resolution against frequency resolution.

unbias:
Make the scales comparable to one another.

OUTPUTS: 

wavelets out:
(batch, scales, time).

RELATED:
t.ultracwt for a live signal, where only the newest column is computed.
cwt for trying other wavelet shapes on a recorded window.
heat_map or heat_scroll to look at the result - through a t.squeeze, because of
the batch dimension."""

demo = [
    {'key': 'sig', 'init': 'signal', 'pos': (30, 62), 'w': 129, 'h': 78,
     'props': SIG('sin', 2.0)},
    {'key': 'rb', 'init': 'rolling_buffer 256', 'pos': (30, 155), 'w': 300, 'h': 200,
     'props': {'sample count': 256, 'update style': 'input is stream of samples'}},
    {'key': 'c0', 'comment': True, 'text': 'a window to transform - this node takes\nthe whole thing at once, unlike ultracwt',
     'pos': (30, 370)},

    {'key': 'cw', 'init': 't.cwt', 'pos': (30, 445), 'w': 320, 'h': 260},
    {'key': 'c2', 'comment': True, 'text': 'sample_scaling is dt - seconds per\nsample. It does not change the picture,\nonly what the rows MEAN in Hz - so a\nwrong value is wrong silently',
     'pos': (30, 720)},

    {'key': 'inf', 'init': 'info', 'pos': (400, 445), 'w': 260, 'h': 80},
    {'key': 'c6', 'comment': True, 'text': 'note the leading 1: the answer is\nbatch by scales by time, not a plain\nimage. Squeeze it before displaying',
     'pos': (400, 540)},

    {'key': 'sq', 'init': 't.squeeze', 'pos': (400, 650), 'w': 160, 'h': 80},
    {'key': 'hm', 'init': 'heat_map', 'pos': (400, 745), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 256,
               'min y': 0.0, 'max y': 10.0, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c9', 'comment': True, 'text': 'scale_distribution sets the row count:\n0.25 -> 33 rows, 0.125 -> 65, 0.0625\n-> 129. Halving it doubles the work\nwavelet_constant is w0 - 6 is standard.\nHigher is sharper in frequency and\nblurrier in time. You cannot have both',
     'pos': (400, 905)},
]
links = [('sig', '', 'rb', 'input'),
         ('rb', 'output', 'cw', 'tensor in'),
         ('cw', 'wavelets out', 'inf', 'in'),
         ('cw', 'wavelets out', 'sq', 'tensor in'),
         ('sq', 'output', 'hm', 'y')]
print(build('t.cwt', 't.cwt - wavelets, properly parameterised', body,
            demo, links, demo_width=700, text_width=800, text_height=760))
