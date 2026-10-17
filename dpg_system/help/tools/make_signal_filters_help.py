"""diff_filter, band_pass, spectrum, adaptive_filter, kinetic_filter (multi_filter, one_euro_filter and kalman_filter are hand-made)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# multi_filter_help.json is laid out by hand now (David's version) and is no longer
# generated here; regenerating it would overwrite that layout.

# ---------------------------------------------------------------- diff_filter
def MULTI_PLOT(miny=-1.0, maxy=1.0):
    """A plot that draws each element of a list as its own line."""
    return dict(PLOT(miny, maxy), **{'width': 300, 'update style': 'input is multi-channel sample'})


body = """diff_filter tells you how much a signal is MOVING at a chosen timescale - and
settles back to zero whenever the signal holds still.

Inside, it smooths the signal the way the filter node does, at two degrees: a
quick one and a slow one. It sends the GAP between them. While the signal holds
steady, both have caught up and the gap is zero. When the signal moves, the quick
filter follows at once and the slow one lags, so the gap opens - then closes again
as the slow one catches up. Rising movement gives a positive gap, falling movement
a negative one.

The top demo shows exactly this. A square wave jumps between -0.7 and 0.7.
"multi_filter 0 0.8 0.97" draws the wave itself (a degree of 0 is no smoothing),
a quick smoothing and a slow one. "diff_filter 0.8 0.97" sends the gap between
the quick and slow lines: a spike at each jump, dying away while the wave holds.

So it is a smoothed "how fast is this changing", with jitter taken out. It ignores
WHERE the signal is and reports only how it moves, which makes it good for onsets:
a hand starting to move, a sound starting to swell, a step in a sensor reading.
A slow drift barely registers.

BANDS IN TIME:
Give it more than two degrees and it sends one gap for each neighbouring pair, so
N degrees give N - 1 values. Each catches movement between those two timescales:
anything quicker than the quick filter can follow is smoothed away, and anything
slower than the slow filter's lag leaves almost no gap.

In the lower demo, a quick wobble rides on a slow swell. "diff_filter 0.3 0.85
0.99" pulls them apart: its first value (the 0.3 to 0.85 gap) mostly follows the
wobble, its second (0.85 to 0.99) mostly the swell. The split is soft, not a
brick wall - each band lets a little of the other through.

As a rough guide, at 60 values a second a degree d responds over about 1 / (1 - d)
values: 0.8 over 5 (a twelfth of a second), 0.97 over 33 (half a second), 0.99 over
100 (nearly two seconds).

SYNTAX:
diff_filter <degree> <degree> ...

EXAMPLES:
diff_filter 0.8 0.97          one value: movement over a tenth to half a second
diff_filter 0.3 0.85 0.99     two values: quick movement, then slow movement

INPUTS and PARAMETERS:

in:
A single number. Receiving data here triggers the node. diff_filter spreads one
signal over several filters; it does not filter each element of an array.

filter 0, filter 1, ...:
One inlet per degree, from 0.0 to 1.0, as on the filter node: higher is smoother
and slower. Give them from quickest to slowest, so that each gap is quick minus
slow and rising movement comes out positive.

MESSAGES:

set <value> <value> ...
Sets the filters' internal values, one per filter.

clear
Sets every filter back to zero.

OUTPUTS:

out:
A NumPy array with one value fewer than there are degrees: each filter minus the
next, slower one.

STARTING UP:
Every filter starts at zero, so a signal that begins far from zero looks, for a
moment, as if it had just jumped there - a large first gap. Send set with the
starting value for each filter to begin without it.

diff_filter_bank is the same node under another name.

SEE ALSO:
multi_filter    the smoothed values themselves
filter          one smoothing
diff            the plain change from one value to the next, unsmoothed
band_pass       a sharper split by speed"""

demo = [
    {'key': 'sq', 'init': 'signal 4.0 square', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': SIG('square', 4.0, 0.7)},
    {'key': 'mf', 'init': 'multi_filter 0.0 0.8 0.97', 'pos': (30, 160), 'w': 160, 'h': 84,
     'props': {'filter 0': 0.0, 'filter 1': 0.8, 'filter 2': 0.97}},
    {'key': 'p1', 'init': 'plot', 'pos': (240, 62), 'w': 308, 'h': 178, 'props': MULTI_PLOT()},
    {'key': 'c1', 'comment': True, 'text': 'the wave, a quick smoothing and a slow one',
     'pos': (240, 250)},
    {'key': 'df', 'init': 'diff_filter 0.8 0.97', 'pos': (30, 300), 'w': 160, 'h': 70,
     'props': {'filter 0': 0.8, 'filter 1': 0.97}},
    {'key': 'p2', 'init': 'plot', 'pos': (240, 300), 'w': 308, 'h': 178, 'props': MULTI_PLOT()},
    {'key': 'c2', 'comment': True, 'text': 'the gap between them: a spike at each jump, then zero',
     'pos': (240, 488)},

    {'key': 'wob', 'init': 'signal 0.4 sin', 'pos': (30, 560), 'w': 129, 'h': 70,
     'props': SIG('sin', 0.4, 0.3)},
    {'key': 'swl', 'init': 'signal 8.0 sin', 'pos': (30, 650), 'w': 129, 'h': 70,
     'props': SIG('sin', 8.0, 1.0)},
    {'key': 'add', 'init': '+ 0.0', 'pos': (190, 600), 'w': 88, 'h': 56, 'props': {'operand': 0.0}},
    {'key': 'p3', 'init': 'plot', 'pos': (310, 560), 'w': 208, 'h': 178, 'props': PLOT(-1.5, 1.5)},
    {'key': 'c3', 'comment': True, 'text': 'a quick wobble riding on a slow swell', 'pos': (310, 748)},
    {'key': 'df2', 'init': 'diff_filter 0.3 0.85 0.99', 'pos': (30, 800), 'w': 160, 'h': 84,
     'props': {'filter 0': 0.3, 'filter 1': 0.85, 'filter 2': 0.99}},
    {'key': 'p4', 'init': 'plot', 'pos': (240, 800), 'w': 308, 'h': 178, 'props': MULTI_PLOT(-1.2, 1.2)},
    {'key': 'c4', 'comment': True, 'text': 'first value: mostly the wobble\nsecond value: mostly the swell',
     'pos': (240, 988)},
]
links = [('sq', '', 'mf', 'in'), ('mf', 'out', 'p1', 'y'),
         ('sq', '', 'df', 'in'), ('df', 'out', 'p2', 'y'),
         ('wob', '', 'add', 'in'), ('swl', '', 'add', 'operand'),
         ('add', 'result', 'p3', 'y'), ('add', 'result', 'df2', 'in'), ('df2', 'out', 'p4', 'y')]
print(build('diff_filter', 'diff_filter - how much is it moving, at this timescale', body,
            demo, links, demo_width=580, text_width=820, text_height=1100))


# ------------------------------------------------------------------ band_pass
body = """band_pass selects part of a signal by how FAST it wiggles, rather than by how big it is.

A slow drift and a fast tremor can sit on top of each other in the same stream, at
the same size. No threshold separates them, and no amount of smoothing separates
them cleanly either. What tells them apart is frequency.

These are true digital filters - Butterworth or Chebyshev designs, at an order you
choose. A filter like this has to know how fast samples arrive, so you must tell
it the sample frequency, and getting that wrong makes every other setting wrong
with it.

THE NODES:

band_pass    one filter, whose type you choose: bandpass, lowpass, highpass
             or bandstop
filter_bank  a row of bandpass filters spread across a range, sending the
             filtered signal from every band at once

Use band_pass to keep or remove one range. Use filter_bank when you want the
signal split into several ranges and want to keep working with the pieces.
If you only want to know HOW MUCH is going on in each range, use spectrum.

SYNTAX:
band_pass
filter_bank

Neither takes arguments; everything is set on the node.

EXAMPLE:
band_pass, with filter type set to lowpass and high set to 1.0

INPUTS and PARAMETERS:

signal:
The value to filter. Receiving data here triggers the node.
A single number per frame - these work over time, on a stream. An array is not
filtered element by element; to filter several channels, use one node per channel.

sample freq:
How many values arrive per second. Default 60.
Get this right first: the filter's idea of what "10 Hz" means comes entirely from
this number. If your data arrives 100 times a second and this says 60, every
frequency setting lands in the wrong place.

low / high:
Frequencies in Hz (cycles per second).
On band_pass they are the two edges of the one filter - default 10 and 20.
lowpass uses only high, and highpass uses only low.
On filter_bank they are the two ends of the whole spread - default 1 and 20 - and
the bands between them are spaced evenly on a logarithmic scale, so each band
covers the same ratio of frequencies rather than the same number of Hz.

filter type (band_pass only):
bandpass     keeps what lies between low and high
bandstop     removes what lies between low and high
lowpass      keeps everything slower than high
highpass     keeps everything faster than low

band count (filter_bank only):
How many bands. Default 8. Changing it rebuilds the bank at once.

filter design:
butter is smooth and well behaved and is the sensible default.
cheby1 cuts off more sharply, at the cost of a ripple of about one decibel across
the range it keeps. cheby2 is flat across the range it keeps and turns down the
range it rejects by about 40 decibels (to about a hundredth of its size), with a
little unevenness out there instead.

order:
How steeply the filter cuts off, 1 to 8. Default 5. Higher is sharper, and also
rings more and reacts more slowly. Raise it only if a lower order really is not
separating what you need.

Changing any of these rebuilds the filter from scratch, so the output jumps
briefly while it settles again.

OUTPUTS:

filtered:
band_pass sends the filtered signal, one number per input.
filter_bank sends a NumPy array with one filtered signal per band, slowest band
first. These are signals, swinging above and below zero - not amounts of energy.

LIMITS WORTH KNOWING:
No filter can work above half the sample frequency - the Nyquist limit. At 60
values a second that is 30 Hz. If high is set at or above it, the node quietly
uses one Hz below the limit instead. If low ends up above high, low is quietly moved to
half of high. The widgets keep showing what you typed.

RELATED:
spectrum      the same row of bands as filter_bank, reporting the energy in each
filter        a simple one-pole smoother - a gentle lowpass with one setting
diff_filter   bands in time built from plain smoothing filters"""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 4.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 4.0, 1.0)},
    {'key': 'ca', 'comment': True, 'text': 'slow sway, once every four seconds', 'pos': (260, 132)},
    {'key': 'sig2', 'init': 'signal 0.125 sin', 'pos': (30, 222), 'w': 129, 'h': 78,
     'props': SIG('sin', 0.125, 0.3)},
    {'key': 'cb', 'comment': True, 'text': 'fast tremor, eight times a second', 'pos': (260, 222)},
    {'key': 'add', 'init': '+ 0.0', 'pos': (30, 312), 'w': 120, 'h': 70,
     'props': {'operand': 0.0}},
    {'key': 'p0', 'init': 'plot', 'pos': (30, 395), 'w': 208, 'h': 178,
     'props': PLOT(-1.5, 1.5)},
    {'key': 'c0', 'comment': True, 'text': 'the two mixed together', 'pos': (260, 395)},
    {'key': 'bp', 'init': 'band_pass', 'pos': (30, 600), 'w': 190, 'h': 150,
     'props': {'filter type': 'lowpass', 'filter design': 'butter', 'order': 5,
               'low': 0.5, 'high': 1.0, 'sample freq': 60.0}},
    {'key': 'c1', 'comment': True, 'text': 'lowpass: keep what is slower than high (1 Hz)',
     'pos': (260, 600)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 775), 'w': 208, 'h': 178,
     'props': PLOT(-1.5, 1.5)},
    {'key': 'c2', 'comment': True, 'text': 'the sway survives, the tremor goes\nswitch filter type to highpass to keep\nonly the tremor (faster than low, 0.5 Hz)',
     'pos': (260, 775)},
    {'key': 'fb', 'init': 'filter_bank', 'pos': (30, 980), 'w': 190, 'h': 150,
     'props': {'band count': 8, 'filter design': 'butter', 'order': 5,
               'low': 1.0, 'high': 20.0, 'sample freq': 60.0}},
    {'key': 'c3', 'comment': True, 'text': 'the same mix, split into eight bands\nfrom 1 to 20 Hz',
     'pos': (260, 980)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 1155), 'w': 208, 'h': 150,
     'props': {'color': 'red-blue', 'width': 200, 'height': 100, 'sample count': 8,
               'min y': -0.3, 'max y': 0.3, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c4', 'comment': True, 'text': 'each band\'s own filtered signal, slowest at the left:\nthe band holding the tremor swings back and forth',
     'pos': (260, 1155)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'), ('tt', '1', 'sig2', 'on'),
         ('sig', '', 'add', 'in'), ('sig2', '', 'add', 'operand'),
         ('add', 'result', 'p0', 'y'),
         ('add', 'result', 'bp', 'signal'), ('bp', 'filtered', 'p1', 'y'),
         ('add', 'result', 'fb', 'signal'), ('fb', 'filtered', 'hm', 'y')]
print(build('band_pass', 'band_pass - select by frequency, not by size', body,
            demo, links, demo_width=660, text_width=830, text_height=1430))

# ------------------------------------------------------------------- spectrum
body = """spectrum tells you how much movement there is in each range of speeds - a running
picture of WHERE the activity is, from slow to fast.

It splits the incoming signal into a row of frequency bands, exactly as
filter_bank does, but instead of sending the filtered signals it sends the ENERGY
in each band: how strongly the signal is moving at that speed right now.

That is the question to ask when you want to know whether a movement is a slow
sway or a fast shake, or how lively something is, without caring about the exact
shape of the wave.

HOW THE ENERGY IS MEASURED:
A filtered band is a wave swinging above and below zero, so its value alone keeps
passing through zero even while the band is busy. spectrum combines the value with
how fast it is changing, scaled to suit the band's frequency, which gives a steady
reading instead of a flicker. A steady wave of size A (swinging from minus A to
plus A) in the middle of a band reads A squared - a wave of size 1 reads 1, and a
wave of size 0.3 reads 0.09 - at any sample frequency.

SYNTAX:
spectrum

No arguments; everything is set on the node.

EXAMPLE:
spectrum, at its defaults: 8 bands from 1 to 20 Hz, at 60 samples a second

INPUTS and PARAMETERS:

signal:
The value to analyse. Receiving data here triggers the node. A single number per
frame - the picture is built over time, from a stream.

sample freq:
How many values arrive per second. Default 60. Get this right first: every band's
position, and the scaling that steadies the reading, comes from it.

low / high:
The two ends of the whole spread of bands, in Hz (cycles per second). Default 1
and 20. The bands are spaced evenly on a logarithmic scale between them, so each
covers the same ratio of frequencies. The default band edges are about
1, 1.5, 2.1, 3.1, 4.5, 6.5, 9.5, 13.8 and 20 Hz.

band count:
How many bands to divide the range into. Default 8.

filter design:
butter is the sensible default. cheby1 gives sharper band edges with about one
decibel of ripple. cheby2 keeps each band flat and turns down what lies outside
it by about 40 decibels.

order:
How sharply each band's edges cut off, 1 to 8. Default 5.

Changing any of these rebuilds the filters, so the readings jump briefly while
they settle again.

OUTPUTS:

spectrum:
A NumPy array with one energy value per band, slowest band first, never negative.
Send it to a heat_map, or to a plot set to 'buffer holds one sample of input', to
see it - and set their max y to suit: small movements give small numbers, because
the reading goes with the SQUARE of the size.

LIMITS WORTH KNOWING:
No filter can work above half the sample frequency - at 60 values a second,
30 Hz. A high setting at or above that is quietly moved to one Hz below it, and
a low setting above high is quietly moved to half of high.

A movement on the border between two bands is shared between them, and anything
outside low to high is not reported at all.

RELATED:
filter_bank   the same bands, sending the filtered signals themselves
band_pass     one filter, to keep or remove a single range
heat_map      the natural display for a spectrum; heat_scroll shows it over time
t.fft         a full frequency analysis of a whole window of samples at once"""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 4.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 4.0, 1.0)},
    {'key': 'ca', 'comment': True, 'text': 'slow sway, once every four seconds', 'pos': (260, 132)},
    {'key': 'sig2', 'init': 'signal 0.125 sin', 'pos': (30, 222), 'w': 129, 'h': 78,
     'props': SIG('sin', 0.125, 0.3)},
    {'key': 'cb', 'comment': True, 'text': 'fast tremor of size 0.3, eight times a second\ndrag its period to move it between bands',
     'pos': (260, 222)},
    {'key': 'add', 'init': '+ 0.0', 'pos': (30, 312), 'w': 120, 'h': 70,
     'props': {'operand': 0.0}},
    {'key': 'c0', 'comment': True, 'text': 'the two mixed together', 'pos': (260, 312)},
    {'key': 'spec', 'init': 'spectrum', 'pos': (30, 400), 'w': 190, 'h': 150,
     'props': {'band count': 8, 'filter design': 'butter', 'order': 5,
               'low': 1.0, 'high': 20.0, 'sample freq': 60.0}},
    {'key': 'c1', 'comment': True, 'text': 'eight bands from 1 to 20 Hz', 'pos': (260, 400)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 575), 'w': 208, 'h': 150,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 8,
               'min y': 0.0, 'max y': 0.1, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c2', 'comment': True, 'text': 'the tremor lights the band from 6.5 to 9.5 Hz,\nreading about 0.09 - 0.3 squared. The sway is\nslower than 1 Hz, so it does not show at all',
     'pos': (260, 575)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'), ('tt', '1', 'sig2', 'on'),
         ('sig', '', 'add', 'in'), ('sig2', '', 'add', 'operand'),
         ('add', 'result', 'spec', 'signal'), ('spec', 'spectrum', 'hm', 'y')]
print(build('spectrum', 'spectrum - how much movement at each speed', body,
            demo, links, demo_width=660, text_width=830, text_height=1270))

# ------------------------------------------------------------- adaptive_filter
body = """adaptive_filter smooths a signal by an amount that changes with how far it is moving.

A plain filter forces one choice on you: smooth enough to kill the noise, or
responsive enough to keep up with real movement. It cannot have both, because it
cannot tell the two apart.

adaptive_filter makes a simple assumption that lets it try: when the input stays
close to the output, the difference is probably noise, so smooth hard; when the
input pulls far away from it, that is probably real movement, so follow quickly.
The result keeps up with sharp gestures and sits still when nothing is happening.

THE NODES:

adaptive_filter             for numbers, NumPy arrays and PyTorch tensors
adaptive_quaternion_filter  a variant for one quaternion (a rotation), which
                            keeps its output at length 1

HOW IT DECIDES:
On every input it measures the gap between the new value and its current output,
divides it by 'signal range', and smooths that gap a little ('offset response').
It raises the result to the power 'power' to get the share of the new value to
take this time, capped at 'responsiveness'. Small gaps give tiny steps and heavy
smoothing; a gap the size of 'signal range' or larger gives the full
'responsiveness'. For an array the gap is averaged over all the elements, so the
whole array is smoothed by one shared amount.

SYNTAX:
adaptive_filter <power: float>
adaptive_quaternion_filter <power: float>

EXAMPLE:
adaptive_filter 2.0

INPUTS and PARAMETERS - adaptive_filter:

in:
The value to filter. Receiving data here triggers the node.

power:
How sharply the filter tells small gaps from large ones. Default 2.0, or the
argument. At 0 it is a plain filter that always takes the 'responsiveness' share.
Higher values smooth small gaps harder and wait for a gap near 'signal range'
before opening up - a sharper divide between noise and movement.

responsiveness:
The largest share of the new value the filter will take in one step, 0 to 1.
Default 0.95. This is the FAST end of the trade: lower it to keep some smoothing
even during big movements.

signal range:
The size of gap that counts as real movement. Default 1.0. Set it to roughly the
size of the movements you care about - this is the most important setting.
Too large and everything looks like noise, so the output lags; too small and
everything looks like movement, so the noise comes through.

offset response:
Smoothing of the measured gap, from 0 to 1. Default 0.9. Higher stops a single
noisy value from opening the filter, but reacts a little later to real movement.

smooth response:
Smoothing of the share taken, from 0 to 1. Default 0. Raise it if the filter
visibly switches between calm and quick too abruptly.

reset response:
A button that throws the current output away; it restarts from zero and catches
up with the input.

INPUTS and PARAMETERS - adaptive_quaternion_filter:

The same idea, with these differences:

in:
A quaternion as four numbers - a list, NumPy array or PyTorch tensor. Lists are
converted. The output is rescaled to length 1 after every step, so it stays a
valid rotation.

power:
Default 1.0.

responsiveness:
Here this is the most smoothing applied, used when the input is still - the share
of the PREVIOUS output kept on each step. Default 0.05, which smooths very little;
raise it towards 0.9 for real smoothing. This is the opposite sense from
adaptive_filter.

signal range:
Default 0.1. The gap is the average difference between the four numbers of the
new quaternion and those of the output.

offset response / smooth response:
As on adaptive_filter; both default to 0.5.

noise floor:
Default 0. A gap smaller than this counts as stillness and gets the full
smoothing; only the part of a gap above the floor makes the filter follow.
Raise it to just above the jitter of a still sensor - the output stops
trembling, while a real movement, well above the floor, still comes through.

A quaternion and its negative are the same rotation, but this filter does not
know that: if your source flips sign, the filter treats it as a sudden big
movement and follows it at once, unsmoothed.

It takes one quaternion or an array of several - a whole pose. Each quaternion
in an array is smoothed by its own amount and kept at length 1 on its own. The
filter starts from the first value it receives.

OUTPUTS:

degree out:
How much the filter followed on this frame. On adaptive_filter it is the share of
the new value taken: near 0 when still, rising towards 'responsiveness' on a big
movement. Worth plotting while tuning - it shows you directly when the filter is
opening up and when it is clamping down. On adaptive_quaternion_filter it is the
share of the old output kept, so it reads the other way up.

out:
The filtered value.

RELATED:
one_euro_filter   the published One Euro filter: the same idea, adapting to
                  speed, with a well-documented tuning procedure
filter            a plain one-pole smoother, for comparison
kalman_filter     smoothing by predicting where a smoothly moving value should be"""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 5.0 square', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('square', 5.0, 0.7)},
    {'key': 'ca', 'comment': True, 'text': 'sharp steps', 'pos': (260, 132)},
    {'key': 'nz', 'init': 'signal 1.0 random', 'pos': (30, 222), 'w': 129, 'h': 78,
     'props': SIG('random', 1.0, 0.15)},
    {'key': 'cb', 'comment': True, 'text': 'random noise', 'pos': (260, 222)},
    {'key': 'add', 'init': '+ 0.0', 'pos': (30, 312), 'w': 120, 'h': 70,
     'props': {'operand': 0.0}},
    {'key': 'p0', 'init': 'plot', 'pos': (30, 395), 'w': 208, 'h': 178,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c0', 'comment': True, 'text': 'sharp steps buried in noise', 'pos': (260, 395)},
    {'key': 'af', 'init': 'adaptive_filter 2.0', 'pos': (30, 600), 'w': 200, 'h': 210,
     'props': {'power': 2.0, 'responsiveness': 0.95, 'signal range': 1.0,
               'smooth response': 0.0, 'offset response': 0.9}},
    {'key': 'c1', 'comment': True, 'text': 'power 2, everything else at its defaults',
     'pos': (260, 600)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 840), 'w': 208, 'h': 178,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c2', 'comment': True, 'text': 'out: still between steps, quick at the edges',
     'pos': (260, 840)},
    {'key': 'p2', 'init': 'plot', 'pos': (30, 1045), 'w': 208, 'h': 178,
     'props': PLOT(0.0, 1.0)},
    {'key': 'c3', 'comment': True, 'text': 'degree out: near 0 while the input is still,\nleaping up on each step',
     'pos': (260, 1045)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'), ('tt', '1', 'nz', 'on'),
         ('sig', '', 'add', 'in'), ('nz', '', 'add', 'operand'),
         ('add', 'result', 'p0', 'y'),
         ('add', 'result', 'af', 'in'), ('af', 'out', 'p1', 'y'),
         ('af', 'degree out', 'p2', 'y')]
print(build('adaptive_filter', 'adaptive_filter - smooth when still, quick when moving',
            body, demo, links, demo_width=660, text_width=820, text_height=1820))

# one_euro_filter_help.json is laid out by hand now (David's version) and is no longer
# generated here; regenerating it would overwrite that layout.

# -------------------------------------------------------------- kinetic_filter
body = """kinetic_filter smooths a signal by refusing to let it move in ways a physical 
object could not.

Instead of averaging, it carries a position, a velocity and an acceleration, and 
moves them towards the incoming value under limits you set. A spike in the input 
does not get averaged away - it simply cannot be followed, because reaching it 
would need an impossible change of acceleration.

It hands you all three - position, velocity and acceleration - so you get the 
derivatives of the smoothed motion as well as the smoothed value.

There is no damping. The output chases the target as hard as the limits allow, 
so it overshoots a sudden step and swings back before it settles. For plain 
responsive smoothing, one_euro_filter is usually the better choice.

SYNTAX:
kinetic_filter

INPUTS:

in:
The target. Receiving data here triggers the node. Takes single numbers.

reset:
A button that returns position, velocity and acceleration to zero.

max delta accel:
The largest change in acceleration allowed per step - the jerk limit, 
and the main smoothing control.

max accel / max velocity:
Ceilings on acceleration and speed, per step.

All the limits are per step, not per second, so the same settings behave 
differently at different input rates.

OUTPUTS: 

accel out / velocity out / position out:
The smoothed value and its first two derivatives. 
The velocity outlet is a much cleaner speed estimate than putting diff after a 
filter, because it is part of the model rather than a difference of noisy 
samples.

TUNING:
Start loose - limits high enough that nothing is constrained - and bring them 
down until the jitter goes. The first limit that changes anything is the one 
doing the work. If the output lags or swings badly, that limit is too tight; 
if it still jitters, it is not tight enough."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 5.0 square', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('square', 5.0, 0.7)},
    {'key': 'c0', 'comment': True, 'text': 'an abrupt step', 'pos': (30, 215)},
    {'key': 'p0', 'init': 'plot', 'pos': (30, 250), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'kf', 'init': 'kinetic_filter', 'pos': (30, 450), 'w': 190, 'h': 150,
     'props': {'max delta accel': 0.001, 'max accel': 0.1, 'max velocity': 1.0}},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 650), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c1', 'comment': True, 'text': 'it accelerates, travels, overshoots\nand swings back',
     'pos': (30, 835)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'p0', 'y'),
         ('sig', '', 'kf', 'in'), ('kf', 'position out', 'p1', 'y')]
print(build('kinetic_filter', 'kinetic_filter - smooth by obeying physics', body,
            demo, links, demo_width=430, text_width=820, text_height=780))

# kalman_filter_help.json is laid out by hand now (David's version) and is no longer
# generated here; regenerating it would overwrite that layout.

