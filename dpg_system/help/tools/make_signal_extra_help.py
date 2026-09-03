"""sample_hold and togedge (signal_nodes.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ---------------------------------------------------------------- sample_hold
sample_hold_body = """The sample_hold node either passes a value through or freezes the last one it saw.

It works like a camera shutter on a stream of data. 
While it is in "sample" mode, every value that arrives is stored and sent straight out. 
The moment you switch it to "hold" mode, it stops looking at the input 
and keeps sending the last value it captured, over and over, for as long as data keeps arriving.

Note that the node only sends something when a value arrives at its input. 
Holding does not stop the output — it stops the output from changing. 

You use this node to freeze a reading at an interesting moment, to latch a sensor value 
so that a later part of the patch can keep working with it, or to hold a control steady 
while you adjust something else.

SYNTAX:
sample_hold

EXAMPLE:
sample_hold

INPUTS and PARAMETERS:

sample/hold:
The mode switch. 
When checked, the node samples: incoming values pass through and are remembered. 
When unchecked, the node holds: incoming values are ignored and the stored value is sent instead.

input:
The data to be sampled. 
Receiving data here triggers the node. It accepts any kind of data - 
numbers, lists, strings, NumPy arrays or PyTorch tensors.

OUTPUTS: 

out:
The sampled value. 
In sample mode this is whatever just arrived. 
In hold mode this is the value that was stored when the mode last changed. 
Before anything has been sampled, the stored value is 0."""

demo = [
    {'key': 'lb',  'init': 'load_bang',  'pos': (30, 62),  'w': 88, 'h': 46},
    {'key': 't',   'init': 't 1',        'pos': (150, 66), 'w': 22, 'h': 46},
    {'key': 'sig', 'init': 'signal',     'pos': (30, 132), 'w': 129, 'h': 78,
     'props': {'on': True, 'period': 2.0, 'shape': 'sin', 'range': 1.0, 'bipolar': True}},
    {'key': 'c1',  'comment': True, 'text': 'a slow sine wave', 'pos': (30, 215)},
    {'key': 'tog', 'init': 'toggle',     'pos': (30, 262), 'w': 45, 'h': 42,
     'props': {'': True}},
    {'key': 'c2',  'comment': True, 'text': 'uncheck to freeze the value', 'pos': (86, 268)},
    {'key': 'sh',  'init': 'sample_hold', 'pos': (30, 318), 'w': 140, 'h': 70,
     'props': {'sample/hold': True}},
    {'key': 'plt', 'init': 'plot',       'pos': (250, 300), 'w': 208, 'h': 176,
     'props': {'color': 'none', 'width': 200, 'height': 128, 'style': 'line',
               'update style': 'input is stream of samples', 'sample count': 200,
               'min x': 0.0, 'max x': 200.0, 'min y': -1.0, 'max y': 1.0}},
]
links = [
    ('lb', 'out', 't', ''),
    ('t', '1', 'sig', 'on'),
    ('sig', '', 'sh', 'input'),
    ('tog', '', 'sh', 'sample/hold'),
    ('sh', 'out', 'plt', 'y'),
]
print(build('sample_hold', 'sample_hold - pass values through, or freeze the last one',
            sample_hold_body, demo, links, demo_width=480,
            text_width=760, text_height=600))

# -------------------------------------------------------------------- togedge
togedge_body = """The togedge node watches a stream of numbers and reports the moment it crosses zero.

It thinks of its input as being either off (zero or below) or on (above zero). 
Most of the time nothing happens. Only when the input changes from off to on, 
or from on to off, does the node send anything - a single bang out of the matching outlet.

This is the difference between a state and an event. 
A stream of numbers tells you what things are like right now; 
togedge tells you the instant they changed.

You use this node to start something when a sensor first goes over a threshold, 
to fire a one-off action when a button is pressed rather than repeatedly while it is held, 
or to count how many times a signal has crossed a line.

Note that it looks only at whether the value is greater than zero. 
To detect a crossing at some other level, subtract that level first, 
or feed the node the output of a comparison.

SYNTAX:
togedge

EXAMPLE:
togedge

INPUTS and PARAMETERS:

in:
The number to watch. 
Receiving data here triggers the node. 
Values greater than zero count as on, anything else counts as off.

OUTPUTS: 

on:
Sends a bang at the instant the input goes from off to on - the rising edge.

off:
Sends a bang at the instant the input goes from on to off - the falling edge. 

Nothing is sent on the frames in between, however long the input stays put."""

demo2 = [
    {'key': 'lb',  'init': 'load_bang', 'pos': (30, 62),  'w': 88, 'h': 46},
    {'key': 't',   'init': 't 1',       'pos': (150, 66), 'w': 22, 'h': 46},
    {'key': 'sig', 'init': 'signal',    'pos': (30, 132), 'w': 129, 'h': 78,
     'props': {'on': True, 'period': 2.0, 'shape': 'sin', 'range': 1.0, 'bipolar': True}},
    {'key': 'c0',  'comment': True, 'text': 'a sine wave, so it crosses zero twice per cycle',
     'pos': (30, 215)},
    {'key': 'te',  'init': 'togedge',   'pos': (30, 258), 'w': 110, 'h': 70},
    {'key': 'c1',  'comment': True, 'text': 'rising', 'pos': (188, 330)},
    {'key': 'c2',  'comment': True, 'text': 'falling', 'pos': (338, 330)},
    {'key': 'cnt1', 'init': 'counter',  'pos': (180, 368), 'w': 123, 'h': 84,
     'props': {'step': 1}},
    {'key': 'cnt2', 'init': 'counter',  'pos': (330, 368), 'w': 123, 'h': 84,
     'props': {'step': 1}},
    {'key': 'i1',  'init': 'int',       'pos': (180, 462), 'w': 127, 'h': 42,
     'props': {'format': '%d', 'width': 100, 'font size': '24'}},
    {'key': 'i2',  'init': 'int',       'pos': (330, 462), 'w': 127, 'h': 42,
     'props': {'format': '%d', 'width': 100, 'font size': '24'}},
]
links2 = [
    ('lb', 'out', 't', ''),
    ('t', '1', 'sig', 'on'),
    ('sig', '', 'te', ''),
    ('te', 'on', 'cnt1', 'input'),
    ('te', 'off', 'cnt2', 'input'),
    ('cnt1', 'count out', 'i1', ''),
    ('cnt2', 'count out', 'i2', ''),
]
print(build('togedge', 'togedge - report the moment a signal crosses zero',
            togedge_body, demo2, links2, demo_width=480,
            text_width=760, text_height=600))
