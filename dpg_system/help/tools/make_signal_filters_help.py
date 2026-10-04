"""multi_filter, band_pass, spectrum, adaptive_filter, one_euro_filter, physics_filter, kalman_filter."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# --------------------------------------------------------------- multi_filter
body = """These nodes run several smoothing filters on the same signal at once, 
each at a different degree, and hand you all the results together.

Every one of them is the same one-pole filter as the filter node - move part of 
the way towards the new value each time - just several of them side by side. 
You give the degrees as arguments, and you get back a NumPy array with one 
element per filter.

The interesting one is diff_filter. Instead of the smoothed values it sends the 
DIFFERENCES between neighbouring filters: the fast one minus the slower one 
next to it, and so on. Each difference isolates the movement that happens 
between the two timescales - motion too quick for the slow filter to follow, 
but slow enough that the fast one keeps up. 

That is a band, in time rather than in frequency, and it is a cheap way to ask 
"is this signal moving on a scale of a tenth of a second, or a second, 
or ten seconds?" without doing any frequency analysis at all.

THE NODES:

multi_filter        the smoothed values, one per degree
diff_filter         the differences between neighbouring smoothed values
diff_filter_bank    the same as diff_filter

Note that with N degrees, multi_filter sends N values but diff_filter sends 
N minus 1 - there is one fewer gap than there are filters.

SYNTAX:
multi_filter <degree> <degree> ...
diff_filter <degree> <degree> ...

EXAMPLE:
diff_filter 0.5 0.9 0.99

INPUTS and PARAMETERS:

in:
The value to filter. Receiving data here triggers the node. 
A single number: these nodes spread one signal across many filters, 
they do not filter an array element by element.

filter 0, filter 1, ...:
One inlet per filter, holding that filter's degree, from 0.0 to 1.0 - 
the same meaning as on the filter node. Higher is smoother and laggier. 
Give them in increasing order so the differences come out in a sensible 
fast-to-slow sequence.

MESSAGES:

set <value> <value> ...
Forces the filters' internal values, one per filter.

clear
Sets every filter back to zero.

OUTPUTS: 

out:
A NumPy array - the smoothed values for multi_filter, 
the differences between neighbours for diff_filter."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 2.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 2.0)},
    {'key': 'c0', 'comment': True, 'text': 'one signal in', 'pos': (30, 215)},
    {'key': 'mf', 'init': 'multi_filter 0.5 0.9 0.99', 'pos': (30, 250), 'w': 190, 'h': 120,
     'props': {'filter 0': 0.5, 'filter 1': 0.9, 'filter 2': 0.99}},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 390), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 3,
               'min y': -1.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.3f'}},
    {'key': 'c1', 'comment': True, 'text': 'three timescales at once', 'pos': (30, 550)},
    {'key': 'df', 'init': 'diff_filter 0.5 0.9 0.99', 'pos': (30, 590), 'w': 190, 'h': 120,
     'props': {'filter 0': 0.5, 'filter 1': 0.9, 'filter 2': 0.99}},
    {'key': 'hm2', 'init': 'heat_map', 'pos': (30, 730), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 2,
               'min y': -0.5, 'max y': 0.5, 'update_mode': 'heat_map',
               'number format': '%.3f'}},
    {'key': 'c2', 'comment': True, 'text': 'two gaps between three filters',
     'pos': (30, 890)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'mf', 'in'), ('mf', 'out', 'hm', 'y'),
         ('sig', '', 'df', 'in'), ('df', 'out', 'hm2', 'y')]
print(build('multi_filter', 'multi_filter - many timescales at once', body,
            demo, links, demo_width=430, text_width=810, text_height=700))

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

# ------------------------------------------------------------- one_euro_filter
body = """one_euro_filter is the published One Euro filter (Casiez, Roussel and Vogel, 2012):
a smoothing filter that adapts to how fast the signal is moving.

It is a plain lowpass filter whose cutoff - the speed of change it lets through -
rises whenever the signal moves quickly. When the signal is barely moving the
cutoff is low and jitter is smoothed away. When it moves fast the cutoff rises
and the output keeps up instead of lagging. It is the standard answer to "smooth
when still, quick when moving", and it is easy to tune.

SYNTAX:
one_euro_filter <min_cutoff: float> <beta: float> <d_cutoff: float>

All three are optional, defaulting to 1.0, 0.0 and 1.0. Note that with beta at 0
the filter does not adapt at all: it is a plain lowpass at min_cutoff.

EXAMPLE:
one_euro_filter 1.0 0.5 1.0

INPUTS and PARAMETERS:

input:
The value to filter. Receiving data here triggers the node.
A number, a list, a NumPy array or a PyTorch tensor. Each element of an array
adapts on its own, by its own speed. A tensor comes back as the same kind of
tensor, on the same device.

min_cutoff:
The cutoff when the signal is still, in Hz (cycles per second). LOWER means
smoother. Tune this first: set beta to 0, then lower min_cutoff until the output
is steady when the input is at rest - but not so low that slow movements lag.

beta:
How much speed raises the cutoff. Tune this second: raise it until fast
movements stop lagging. Higher means more responsive when moving, and a little
more jitter while moving.

d_cutoff:
The cutoff, in Hz, of the smoothing applied to the filter's own speed estimate.
The default of 1.0 is almost always right.

dt:
The time between samples, in seconds. Default 1/60, for data arriving once a
frame. The filter's cutoffs are in Hz, so it has to know how far apart the
samples are: set dt to match the data - 1/100 for a 100 Hz sensor - and the
filter behaves the same however fast the samples are actually delivered, which
matters for data played back faster or slower than it was recorded.
Set dt to 0 to have the node time the arrivals itself, from the clock.

The first value received passes straight through, and the filter starts from it.

OUTPUTS:

out:
The filtered value, in the same form as the input.

RELATED:
adaptive_filter   the same idea with more settings, and an outlet showing how
                  hard it is smoothing
filter            a plain one-pole smoother, for comparison
kalman_filter     predicts where a smoothly moving value should be, then corrects
physics_filter    smooths by limiting speed and acceleration"""

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
    {'key': 'oe', 'init': 'one_euro_filter 1.0 0.5 1.0', 'pos': (30, 600), 'w': 190, 'h': 120,
     'props': {'min_cutoff': 1.0, 'beta': 0.5, 'd_cutoff': 1.0}},
    {'key': 'c1', 'comment': True, 'text': 'min_cutoff 1, beta 0.5, d_cutoff 1', 'pos': (260, 600)},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 745), 'w': 208, 'h': 178,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c2', 'comment': True, 'text': 'still between steps, quick at the edges\nset beta to 0 and the steps go soft',
     'pos': (260, 745)},
    {'key': 'flt', 'init': 'filter 0.9', 'pos': (30, 950), 'w': 130, 'h': 70,
     'props': {'degree': 0.9}},
    {'key': 'c3', 'comment': True, 'text': 'a plain filter, for comparison', 'pos': (260, 950)},
    {'key': 'p2', 'init': 'plot', 'pos': (30, 1035), 'w': 208, 'h': 178,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c4', 'comment': True, 'text': 'about as quiet, but the steps are smeared',
     'pos': (260, 1035)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'), ('tt', '1', 'nz', 'on'),
         ('sig', '', 'add', 'in'), ('nz', '', 'add', 'operand'),
         ('add', 'result', 'p0', 'y'),
         ('add', 'result', 'oe', 'input'), ('oe', 'out', 'p1', 'y'),
         ('add', 'result', 'flt', 'in'), ('flt', 'out', 'p2', 'y')]
print(build('one_euro_filter', 'one_euro_filter - the standard adaptive smoother',
            body, demo, links, demo_width=660, text_width=820, text_height=900))

# -------------------------------------------------------------- physics_filter
body = """These filters smooth a signal by refusing to let it move in ways a physical object could not.

Instead of averaging, they carry a position, a velocity and an acceleration, and 
move that towards the incoming value under limits you set. A spike in the input 
does not get averaged away - it simply cannot be followed, because reaching it 
would need an impossible acceleration.

That gives a very different character from a plain filter. Smooth output with no 
constant lag: the result tracks the input exactly while the input behaves 
plausibly, and falls behind only when the input does something abrupt. 
For anything driving a physical or apparently-physical thing - a motor, a camera 
move, a rendered object - it usually looks right where an averaging filter looks 
soggy.

THE NODES:

physics_filter   a spring-damper chase with velocity, acceleration and jerk limits
kinetic_filter   a simpler limiter, reporting position, velocity and acceleration 
                 separately

physics_filter is the one to reach for. kinetic_filter is useful when you want 
the derivatives as well as the smoothed value - it hands you all three.

SYNTAX:
physics_filter
kinetic_filter

INPUTS and PARAMETERS - physics_filter:

input:
The target to chase. Receiving data here triggers the node. 
Accepts single numbers, lists, NumPy arrays and PyTorch tensors.

max_vel / max_accel / max_jerk:
The limits, in units per second, per second squared, and per second cubed. 
Velocity caps how fast the output can travel, acceleration caps how fast it can 
change speed, and jerk caps how abruptly it can change acceleration - 
that last one is what removes the visible corners from the motion.

freq:
How stiff the spring is, in Hz. Higher chases harder and arrives sooner.

zeta:
The damping. Below 1 the output overshoots and springs back; at 1 it arrives 
without overshoot; above 1 it eases in slowly. Default 2.5 - firmly damped, 
no bounce.

dt:
Time between samples, in seconds. Default 1/60. The limits above are all 
per-second, so this has to be right for them to mean what they say.

INPUTS and PARAMETERS - kinetic_filter:

in:
The target. Receiving data here triggers the node.

max delta accel:
The largest change in acceleration allowed per step - the jerk limit, 
and the main smoothing control.

max accel / max velocity:
Ceilings on acceleration and speed.

reset:
A button that returns position, velocity and acceleration to zero.

OUTPUTS: 

out (physics_filter):
The smoothed value.

position out / velocity out / accel out (kinetic_filter):
The smoothed value and its first two derivatives. 
The velocity outlet is a much cleaner speed estimate than putting diff after a 
filter, because it is part of the model rather than a difference of noisy 
samples.

TUNING:
Start loose - limits high enough that nothing is constrained - and bring them 
down until the jitter goes. The first limit that changes anything is the one 
doing the work. If the output lags badly, that limit is too tight; if it still 
jitters, it is not tight enough."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 5.0 square', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('square', 5.0, 0.7)},
    {'key': 'c0', 'comment': True, 'text': 'an abrupt step', 'pos': (30, 215)},
    {'key': 'p0', 'init': 'plot', 'pos': (30, 250), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'pf', 'init': 'physics_filter', 'pos': (30, 450), 'w': 190, 'h': 180,
     'props': {'max_vel': 500.0, 'max_accel': 50.0, 'max_jerk': 500.0,
               'freq': 5.0, 'zeta': 2.5, 'dt': 0.0166}},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 650), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c1', 'comment': True, 'text': 'it accelerates, travels, and settles\ndrop zeta below 1 to make it overshoot',
     'pos': (30, 835)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('sig', '', 'p0', 'y'),
         ('sig', '', 'pf', 'input'), ('pf', 'out', 'p1', 'y')]
print(build('physics_filter', 'physics_filter - smooth by obeying physics', body,
            demo, links, demo_width=430, text_width=820, text_height=780))

# --------------------------------------------------------------- kalman_filter
body = """The kalman_filter node estimates where a signal really is, given that your 
measurements of it are noisy.

It is not an averaging filter. It carries a model of the thing being measured - 
here a value with a velocity and an acceleration - and on every frame it does 
two things: it PREDICTS where that model says the value should be now, then 
CORRECTS that prediction using the measurement that just arrived.

How much it trusts the measurement against its own prediction is the whole 
question, and it works that out for itself from two numbers you supply: how 
unpredictable you think the underlying thing is, and how noisy you think your 
measurements are. Say the measurements are bad and it leans on its model, 
producing a smooth, confident, slightly stubborn estimate. Say they are good 
and it follows them closely.

Because it predicts before it corrects, it does not lag the way an averaging 
filter does. On a signal that really does move smoothly, it can track with 
almost no delay while still rejecting a lot of noise - which no amount of 
tuning will get you from the filter node.

SYNTAX:
kalman_filter

EXAMPLE:
kalman_filter

INPUTS and PARAMETERS:

in:
The measurement. Receiving data here triggers the node. A single number.

process noise:
How much the underlying value is expected to wander on its own, as a 3 by 3 
matrix. Larger values say "this thing genuinely moves unpredictably", 
and the filter becomes more willing to believe the measurements.

measurement noise:
How noisy each reading is, as a single number. 
Larger values say "do not trust any one reading", and the filter leans harder 
on its own prediction - smoother, but slower to accept a real change.

These two are a ratio, not two independent settings. What matters is how big one 
is relative to the other; scaling both changes nothing.

OUTPUTS: 

out:
The estimated value.

kalman gain:
How much weight the filter is currently giving the measurement over its own 
prediction. Near zero it is ignoring your data and running on the model; 
larger and it is following the measurements. 
Watch this while tuning - it tells you which of the two numbers above is 
actually in charge.

WHEN THIS IS THE WRONG NODE:
The model assumes the value moves smoothly, with a velocity and acceleration 
that change gradually. For a signal that genuinely jumps - a switch, a step, a 
category - the model is wrong and the filter will fight the data. 
Reach for physics_filter when you want plausible motion, one_euro_filter when 
you want responsive smoothing without a model, and this when your signal really 
is a smoothly moving quantity that you are measuring badly."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 3.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 3.0, 0.7)},
    {'key': 'tog', 'init': 'toggle', 'pos': (200, 62), 'w': 45, 'h': 42, 'props': {'': True}},
    {'key': 'met', 'init': 'metro 16', 'pos': (200, 112), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 16.0, 'units': 'milliseconds'}},
    {'key': 'rnd', 'init': 'random.gauss 0.0 0.2', 'pos': (200, 192), 'w': 175, 'h': 100},
    {'key': 'add', 'init': '+ 0.0', 'pos': (30, 232), 'w': 120, 'h': 70,
     'props': {'operand': 0.0}},
    {'key': 'c0', 'comment': True, 'text': 'a smooth thing, measured badly', 'pos': (30, 312)},
    {'key': 'p0', 'init': 'plot', 'pos': (30, 350), 'w': 208, 'h': 176,
     'props': PLOT(-1.5, 1.5)},
    {'key': 'kf', 'init': 'kalman_filter', 'pos': (30, 550), 'w': 190, 'h': 100},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 670), 'w': 208, 'h': 176,
     'props': PLOT(-1.5, 1.5)},
    {'key': 'c1', 'comment': True, 'text': 'the estimate, with very little lag',
     'pos': (30, 855)},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'),
         ('tog', '', 'met', 'on'), ('met', '', 'rnd', 'trigger'),
         ('sig', '', 'add', 'in'), ('rnd', 'out', 'add', 'operand'),
         ('add', 'result', 'p0', 'y'),
         ('add', 'result', 'kf', 'in'), ('kf', 'out', 'p1', 'y')]
print(build('kalman_filter', 'kalman_filter - predict, then correct', body,
            demo, links, demo_width=440, text_width=820, text_height=740))
