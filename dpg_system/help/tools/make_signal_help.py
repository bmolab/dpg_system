import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# --------------------------------------------------------------------- signal
body = """The signal node is a free-running waveform generator. 
It is the usual way to get something moving in a patch without wiring up a clock.

Once switched on it produces a new value on every frame, tracing out the chosen 
shape over and over. Unlike metro, which sends a bang and leaves you to make the 
value, signal sends the value itself.

Its timing is based on the wall clock, not on a frame count, so the period means 
what it says however fast or slow the patch happens to be running.

SYNTAX:
signal <period: float> <shape: name>

Both arguments are optional and may be given in either order - 
a recognised shape name is taken as the shape, and any other number as the period.

EXAMPLE:
signal 2.0 sin

INPUTS and PARAMETERS:

on:
Starts and stops the generator. Nothing comes out while it is off. 
The phase carries on from where it stopped.

period:
How long one complete cycle takes, IN SECONDS. 
A period of 2.0 means one full cycle every two seconds; 
0.1 means ten cycles a second. Values of zero or less are clamped 
to a thousandth of a second.

shape:
The waveform, chosen from a menu:

sin        a smooth sine, starting at zero and rising
cos        the same wave a quarter cycle ahead, starting at its peak
saw        a straight ramp that jumps back at the end of each cycle
square     an abrupt alternation between the two extremes
triangle   a straight ramp up and a straight ramp back down
random     a fresh random value every frame, ignoring the period

OPTIONS:

range:
Scales the output. The waveform is generated between -1 and 1 
(or 0 and 1 when bipolar is off) and then multiplied by this. 
Set it to 360 for an angle, or 0.5 for a gentle modulation.

bipolar:
When checked, the wave swings either side of zero, from minus range to plus range. 
When unchecked, it is offset to sit entirely above zero, from 0 to range. 
Uncheck it when you are driving something that has no meaning below zero, 
like a brightness or a rate.

vector size:
When this is 1, the node sends one number per frame. 
Set it higher and the node sends a NumPy array of that many samples per frame 
instead, with the cycle spread across them - so the waveform continues correctly 
from frame to frame at a higher effective sample rate. 
This is how you feed a waveform to something that wants a buffer rather than 
a single reading.

OUTPUTS: 

out:
The current value of the waveform - a single float, or a NumPy array when 
vector size is greater than 1."""

demo = starter() + [
    {'key': 'sig', 'init': 'signal 2.0 sin', 'pos': (30, 132), 'w': 129, 'h': 78,
     'props': SIG('sin', 2.0)},
    {'key': 'p1', 'init': 'plot', 'pos': (210, 108), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.2)},
    {'key': 'c0', 'comment': True, 'text': 'change the shape and period and watch',
     'pos': (30, 218)},
    {'key': 'sig2', 'init': 'signal 3.0 triangle', 'pos': (30, 262), 'w': 129, 'h': 78,
     'props': SIG('triangle', 3.0, 1.0, False)},
    {'key': 'p2', 'init': 'plot', 'pos': (210, 300), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 1.2)},
    {'key': 'c1', 'comment': True, 'text': 'bipolar off: 0 to range, never negative',
     'pos': (30, 348)},
    {'key': 'f1', 'init': 'float', 'pos': (30, 400), 'w': 127, 'h': 42, 'props': FLT},
]
links = [('lb', 'out', 'tt', ''), ('tt', '1', 'sig', 'on'), ('tt', '1', 'sig2', 'on'),
         ('sig', '', 'p1', 'y'), ('sig2', '', 'p2', 'y'), ('sig2', '', 'f1', '')]
print(build('signal', 'signal - a free-running waveform', body, demo, links,
            demo_width=440, text_width=800, text_height=700))

# --------------------------------------------------------------------- random
body = """The random nodes produce a fresh random number each time they are triggered.

They do not run on their own. Send anything into the trigger inlet - typically 
from a metro - and one number comes out. The remaining inlets shape the 
distribution those numbers are drawn from, and hold their values between triggers.

random is the plain one: every value in the range equally likely. 
The random.* nodes each draw from a named statistical distribution, so that some 
outcomes are more likely than others. Which one you want depends on what the 
number is FOR - an even spread is rarely what natural variation looks like.

THE NODES:

random                  even spread across the range - every value equally likely
random.gauss            the bell curve: clustered around a mean, rare far away
random.normalvariate    the same bell curve by its formal name
random.lognormvariate   a bell curve in the logarithm - always positive, 
                        with a long tail to the right. Good for durations 
                        and sizes, which cannot go negative
random.expovariate      waiting times between events that happen at a steady 
                        average rate - many short gaps, a few long ones
random.paretovariate    the classic long tail: mostly small, occasionally huge
random.gammavariate     positive, skewed, adjustable in both shape and spread
random.betavariate      confined between 0 and 1, with the weight of the 
                        distribution wherever you put it - useful for a 
                        random proportion or probability
random.weibullvariate   time until something fails; the shape decides whether 
                        failure gets more or less likely as time passes
random.triangular       a spread between a low and a high limit, peaking at 
                        a value you choose - the simple way to say 
                        "around here, never outside these bounds"
random.vonmisesvariate  the bell curve wrapped around a circle, for angles, 
                        where 359 degrees and 1 degree are neighbours

SYNTAX:
random <range: float>
random.gauss <mean: float> <deviation: float>
random.triangular <low: float> <high: float> <peak: float>

EXAMPLE:
random.gauss 0.0 1.0

INPUTS and PARAMETERS:

trigger:
Receiving anything here produces one new random number. 
The value sent is ignored - only its arrival matters.

range (random only):
The upper limit. Values come out between 0 and range, 
or between minus range and plus range when the bipolar option is checked.

bipolar (random only):
When checked, the spread is centred on zero rather than starting there.

mean / dev:
On the bell-curve nodes: the centre of the distribution, and how far values 
typically stray from it. About two thirds of values land within one deviation 
of the mean. On random.vonmisesvariate these are named mu and kappa, 
and kappa works the other way round - larger kappa means MORE tightly clustered.

alpha / beta / lambda:
The shape parameters of the gamma, beta, Weibull, Pareto and exponential 
distributions. Drag them and watch the histogram; that is faster than 
reasoning about them.

low / high / mode:
On random.triangular: the two limits, and the most likely value in between.

OUTPUTS: 

out:
A single float. One value per trigger - these nodes do not fill arrays. 
To build up a spread, trigger repeatedly and collect the results."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 20), 'w': 45, 'h': 42,
     'props': {'': True}},
    {'key': 'met', 'init': 'metro 20', 'pos': (30, 68), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 20.0, 'units': 'milliseconds'}},
    {'key': 'c0', 'comment': True, 'text': 'trigger it fast', 'pos': (172, 86)},
    {'key': 'rg', 'init': 'random.gauss 0.0 1.0', 'pos': (30, 158), 'w': 175, 'h': 100},
    {'key': 'p1', 'init': 'plot', 'pos': (240, 140), 'w': 208, 'h': 176,
     'props': PLOT(-3.5, 3.5)},
    {'key': 'c1', 'comment': True, 'text': 'mostly near the mean, rarely far away',
     'pos': (30, 268)},
    {'key': 'rn', 'init': 'random 1.0', 'pos': (30, 315), 'w': 140, 'h': 80,
     'props': {'range': 1.0, 'bipolar': False}},
    {'key': 'p2', 'init': 'plot', 'pos': (240, 335), 'w': 208, 'h': 176,
     'props': PLOT(-0.2, 1.2)},
    {'key': 'c2', 'comment': True, 'text': 'plain random: no value preferred\ndrag mean and dev on the node above',
     'pos': (30, 405)},
]
links = [('tog', '', 'met', 'on'), ('met', '', 'rg', 'trigger'),
         ('met', '', 'rn', 'trigger'),
         ('rg', 'out', 'p1', 'y'), ('rn', 'out', 'p2', 'y')]
print(build('random', 'random - one new number each time you ask', body, demo, links,
            demo_width=470, text_width=830, text_height=740))
