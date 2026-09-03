"""Torque gangs: groups of joints heard as one."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """A gang is a group of joints whose effort only makes sense together.

THE NODES:

torque_gang       one named group, as sound-ready numbers. 'gang' is the same
gang              the short spelling
torque_residual   everything the gangs in the patch are NOT describing

WHY GROUP JOINTS AT ALL:
Bending forward is not any one joint. It is distributed across spine1, spine2
and spine3, and no single one of them IS the movement. The same is true of a leg
pushing off - hip, knee and ankle together - and of the collar and shoulder,
which SMPL separates and no listener could hear apart.

So the useful quantity is often the group. A gang is a named weighted sum over
several joints' torque, and the weights carry three things at once: which axis,
which anatomical sign, and how much each joint contributes. Triple extension is
the clearest case, because hip, knee and ankle do not share a sign - the weights
absorb the flips so the whole push reads as one rising number.

THREE NUMBERS, BECAUSE GROUPING FORCES A CHOICE:
As soon as several joints contribute, they can oppose each other, and there are
two honest answers to "how much is happening":

net         the signed sum - opposing efforts cancel
total       the sum of magnitudes - nothing cancels
coherence   |net| / total, from 0 to 1

COHERENCE IS THE ONE GANGING GIVES YOU:
A spine hinging as a single unit and a spine curling at the waist while
extending at the chest have the SAME TOTAL and completely different coherence.
Per-joint torque cannot tell you that; nothing but a group can.

    hinge (100, 100, 100):   net 1.08   total 1.08   coherence 1.00
    curl  (100, 100, -100):  net 0.52   total 1.08   coherence 0.48

1.0 is the group acting as one thing. Towards 0 is internal counter-effort -
work being done against itself, a body bracing rather than moving.

The three are perceptually independent, which is what makes them good to patch:
net to pitch or direction, total to amplitude, coherence to consonance, noise or
detune spread.

Coherence only means something for a group of more than one. Several presets
are single-joint, and their coherence is identically 1 - do not patch those as
though the number carried information.

'stream' IS THE BIGGEST LEVER ON HOW IT SOUNDS:
The same weights on a different component of torque are a different instrument.

gravity    postural load - slow, continuous, drone material
dynamic    the effort of actually moving - transient, percussive
total      both together
passive    rarely non-zero; most gangs never see it

Spine flex on gravity is the weight of holding yourself up. Spine flex on
dynamic is the act of bending. Same gang, and they sound nothing alike.

'normalize' SHOULD USUALLY STAY ON:
Each joint's torque is divided by what that joint can produce before the sum is
taken. Without it the lumbar spine, which is enormously strong, swamps
everything and the gang becomes a single-joint signal wearing a group's name.
With it, every term is "fraction of this joint's capacity" and the weights are
free to be an aesthetic choice.

'surprise' IS THE CONTRADICTION DETECTOR:
The fourth outlet compares what the body is doing against a statistical model of
what bodies usually do, built from twenty million frames of motion capture. It
is high when this combination of efforts is unusual.

The informative signal is a gang contradicting what normally accompanies it -
rarity is not strangeness by itself. Note also that about half of all surprise
lies outside anything the named gangs can express, so a quiet surprise outlet
does not mean nothing unusual is happening.

torque_residual IS THE CONSCIENCE OF A GANG BANK:
It reports, per joint, the effort that no live gang in the patch accounted for.
Patch it and you can hear what your gangs are missing; it recompiles as you add
and remove them, so it always describes the bank you actually have.

A bank of gangs is a vocabulary, and a vocabulary leaves things out. This is how
you find out what.

THE NUMBERS ARE FROM MOTION CAPTURE, NOT FROM A SUIT:
The ranges and the surprise model were measured on the AMASS corpus. Dance is
barely represented in it, and a Shadow suit sees far more extreme movement, so
expect the constants to be wrong at the edges even though the anatomy behind
them holds.

SYNTAX:
torque_gang <preset> <side> <stream>
gang <preset>
torque_residual <stream>

EXAMPLE:
torque_gang spine_flex

INPUTS and PARAMETERS:

gang:
Which preset. Fifteen of them, from spine_flex and leg_push to counter_rotation
and contralateral_swing.

side:
Left, right, or whatever sides the preset offers.

stream:
gravity, dynamic, total or passive.

normalize / gender / invert:
Scale by joint capacity, which body model to size against, and flip the sign.

OUTPUTS: 

net / total / coherence:
Signed sum, unsigned sum, and how much of one the group is being.

surprise:
How unusual this combination is against the corpus.

residual / magnitude (torque_residual):
Per-joint effort no gang claimed.

RELATED:
smpl_torque produces the torque these consume.
The working notes in GANG_NOTES.md and CHARACTERIZATION_RESULTS.md carry the
formalism and what twenty million frames say about it."""

demo = [
    {'key': 'pose', 'init': 'smpl_pose', 'pos': (30, 62), 'w': 240, 'h': 140},
    {'key': 'tq', 'init': 'smpl_torque', 'pos': (30, 220), 'w': 300, 'h': 300},
    {'key': 'c0', 'comment': True, 'text': 'a pose stream, turned into torque -',
     'pos': (30, 535)},
    {'key': 'c0b', 'comment': True, 'text': 'which the gangs then read',
     'pos': (30, 565)},

    {'key': 'g1', 'init': 'torque_gang spine_flex', 'pos': (30, 610), 'w': 300, 'h': 300,
     'props': {'stream': 'dynamic', 'normalize': True}},
    {'key': 'c1', 'comment': True, 'text': 'spine_flex on the dynamic stream is the',
     'pos': (30, 925)},
    {'key': 'c2', 'comment': True, 'text': 'ACT of bending. Switch stream to gravity',
     'pos': (30, 955)},
    {'key': 'c3', 'comment': True, 'text': 'and it becomes postural load instead',
     'pos': (30, 985)},

    {'key': 'p1', 'init': 'plot', 'pos': (380, 380), 'w': 300, 'h': 180,
     'props': PLOT(-1.0, 1.0, 200)},
    {'key': 'c4', 'comment': True, 'text': 'net: signed, so opposing efforts cancel',
     'pos': (380, 570)},
    {'key': 'p2', 'init': 'plot', 'pos': (380, 615), 'w': 300, 'h': 180,
     'props': PLOT(0.0, 1.0, 200)},
    {'key': 'c5', 'comment': True, 'text': 'total: nothing cancels', 'pos': (380, 805)},
    {'key': 'p3', 'init': 'plot', 'pos': (380, 850), 'w': 300, 'h': 180,
     'props': PLOT(0.0, 1.0, 200)},
    {'key': 'c6', 'comment': True, 'text': 'coherence: 1 is the spine hinging as one',
     'pos': (380, 1040)},
    {'key': 'c7', 'comment': True, 'text': 'thing, low is it working against itself',
     'pos': (380, 1070)},

    {'key': 'f1', 'init': 'float', 'pos': (30, 1035), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c8', 'comment': True, 'text': 'surprise: unusual against 20M frames',
     'pos': (30, 1085)},

    {'key': 'res', 'init': 'torque_residual', 'pos': (30, 1135), 'w': 280, 'h': 200},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 1350), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 24,
               'min y': 0.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c9', 'comment': True, 'text': 'what the gangs above did NOT account',
     'pos': (30, 1510)},
    {'key': 'c10', 'comment': True, 'text': 'for - one value per joint',
     'pos': (30, 1540)},
]
links = [('pose', 'out', 'tq', 'pose'),
         ('tq', 'dynamic_torque_vectors', 'g1', 'dynamic'),
         ('tq', 'gravity_torque_vectors', 'g1', 'gravity'),
         ('tq', 'torque_vectors', 'g1', 'torque'),
         ('tq', 'torque_vectors', 'res', 'torque'),
         ('g1', 'net', 'p1', 'y'),
         ('g1', 'total', 'p2', 'y'),
         ('g1', 'coherence', 'p3', 'y'),
         ('g1', 'surprise', 'f1', ''),
         ('res', 'residual', 'hm', 'y')]
print(build('torque_gang', 'torque gangs - joints heard as one', body,
            demo, links, demo_width=740, text_width=810, text_height=800))
