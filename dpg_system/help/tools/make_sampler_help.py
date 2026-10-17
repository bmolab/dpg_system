"""the sampler engine and voices, the dynamic players, and the fader nodes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

FADE = """
THE fade LIST:
The dynamic players take a 'fade' input, and its shape is the thing that makes
this system work: a list of [sound_id, level] pairs. Send

    [[0, 0.8], [3, 0.2], [7, 0.0]]

and sound 0 plays at 0.8, sound 3 at 0.2, sound 7 is silent. Ids not mentioned
keep whatever level they had.

That means a whole polyphonic texture is ONE message, and anything that can
produce such a list can drive it. The fader nodes exist to produce them from
body data.
"""

# ------------------------------------------------------------- sampler_engine
body = """These are the sample playback engine and the direct control of its voices.

THE NODES:

sampler_engine       the engine itself: one per patch
sampler_voice        control one voice directly
multi_voice_sampler  control all voices from one node

sampler_engine IS GLOBAL:
It owns the voices and the audio output. There should be one in a patch, and
everything else refers to it rather than creating its own. 'master volume' is
the overall level and 'output level' reports what is actually coming out, which
is the quickest check that the engine is running at all.

'restart engine' rebuilds it - useful after changing the audio device, and the
first thing to try when the sampler has gone silent for no visible reason.

VOICE-AT-A-TIME VERSUS ALL-AT-ONCE:
sampler_voice addresses one voice by index and gives you everything about it -
the file, the play range, pitch, looping, volume. That is the node for a sound
you are treating individually, where you want its loop points and pitch under
direct control.

multi_voice_sampler addresses every voice from one node, choosing which with a
'voice index' inlet. Fewer nodes and one place to look, at the cost of only
being able to talk about one voice at a time.

WHEN TO USE THESE RATHER THAN THE DYNAMIC PLAYERS:
These require you to decide which voice plays what. The polyphonic, granular and
scratch samplers allocate voices for you - see their help patch - and that is
usually what you want when sounds are triggered by events rather than assigned
by hand.

Use these when a particular sound needs to stay in a particular voice, so its
state persists and you can keep changing it.

SYNTAX:
sampler_engine
sampler_voice
multi_voice_sampler

EXAMPLE:
sampler_voice

INPUTS and PARAMETERS:

load / path:
The file for this voice.

play:
Start it.

sample start / sample end:
Which part of the file to play.

pitch / volume:
Transposition and level.

loop / loop start:
Whether it repeats, and from where.

voice index (multi_voice_sampler):
Which voice the other inlets refer to.

master volume / stop engine / restart engine (sampler_engine):
The engine.

OUTPUTS: 

output level:
What the engine is producing - the fastest way to tell whether anything is
happening.

position / active_voices:
Where playback has reached, and how many voices are sounding.

RELATED:
sampler_osc~ in the synth system treats recorded material as a modular
oscillator instead, with filters, envelopes and spatialisation available to it.
Use that when the sound needs processing; use these when it needs playing."""

demo = [
    {'key': 'se', 'init': 'sampler_engine', 'pos': (30, 62), 'w': 260, 'h': 180},
    {'key': 'f1', 'init': 'float', 'pos': (30, 260), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c0', 'comment': True, 'text': 'one engine per patch\noutput level: is anything happening?', 'pos': (30, 315)},
    {'key': 'btn', 'init': 'button', 'pos': (30, 390), 'w': 88, 'h': 46},
    {'key': 'sv', 'init': 'sampler_voice', 'pos': (30, 455), 'w': 280, 'h': 300},
    {'key': 'c2', 'comment': True, 'text': 'load a file, then click play\none voice, held, with loop points', 'pos': (30, 770)},
    {'key': 'mv', 'init': 'multi_voice_sampler', 'pos': (395, 455), 'w': 300, 'h': 300},
    {'key': 'c4', 'comment': True, 'text': 'or every voice from one node,\none at a time via voice index',
     'pos': (360, 770)},
]
links = [('se', 'output level', 'f1', ''), ('btn', '', 'sv', 'play')]
print(build('sampler_engine', 'sampler_engine - the engine and its voices', body,
            demo, links, demo_width=700, text_width=800, text_height=720))

# -------------------------------------------------------- polyphonic_sampler
body = """These play sounds without you deciding which voice each one goes into.

THE NODES:

polyphonic_sampler~  allocate a voice per triggered sound
granular_sampler~    play a sound as a cloud of short grains
scratch_sampler~     play with the position under direct control

VOICE ALLOCATION IS THE POINT:
Trigger a sound and one of these finds a free voice for it. Trigger another
before the first finishes and it uses a different one. You never name a voice,
and 'active_voices' tells you how many are sounding.

'start_voice' and 'voice_count' set which slice of the engine's voices this node
may use, which is how two of these coexist without fighting over the same ones.

GRANULAR IS A DIFFERENT INSTRUMENT:
Rather than playing a sound from start to end, it plays many very short pieces
of it, overlapping. Slow the rate at which the read position advances and the
sound stretches without changing pitch; move the position by hand and you get a
sustained tone from one instant of the recording.

That makes a short sample into a continuous, playable texture - the useful thing
when a sound has to last as long as a movement does rather than as long as the
recording.

SCRATCH PUTS POSITION UNDER YOUR HAND:
The read position follows what you send it, forwards or backwards, fast or
slow. Patch a limb's position or a fader in and the recording becomes something
you move through rather than something that plays.

Because the pitch follows the speed, the character comes from the gesture -
which is exactly how a hand on a record behaves.

INTO THE ~ GRAPH, OR STRAIGHT OUT:
With nothing patched to 'left out' and 'right out', these play straight to the
speakers. Patch either outlet into any ~ object - fader_out~, vcf~, vst~,
record~ - and this node's voices leave the direct output and play only through
what you patched, so the sampler can be filtered, put through a plugin, faded
with the rest of the mix, or recorded. Pull the cord out and it plays straight
out again.

Give each sampler its own 'start_voice' range: two whose ranges overlap share
voices, and patching one would take the shared voices from the other.

The grain and scratch controls still arrive at frame rate. For grain position
moved at audio rate, by an lfo~ say, use sampler_osc~ in granular mode.
""" + FADE + """
SYNTAX (the plain name, without ~, still works for older patches):
polyphonic_sampler~
granular_sampler~
scratch_sampler~

EXAMPLE:
polyphonic_sampler~

INPUTS and PARAMETERS:

trigger / stop:
Start and stop a sound. A sound id plays that sound; a bang plays the one shown
in 'sound_id'.

load / load_set:
One file, or a set of them at once. A path sent to 'load' goes into the sound
shown in 'sound_id'; [sound_id, path] chooses. record~'s path outlet patched
here loads each new take as you stop recording.

sound_id:
Which loaded sound.

fade:
The [sound_id, level] list. See above.

sample_start / sample_end / volume:
The part to play and how loud.

start_voice / voice_count:
Which of the engine's voices this node may use.

OUTPUTS: 

active_voices:
How many are sounding. Worth watching - if it sits at the voice count, you have
run out and new triggers are stealing from old ones.

left out / right out:
The sound as a signal, for the ~ graph. Patched, they take this node's voices
off the direct output - see above.

RELATED:
The fader nodes produce fade lists from body data - see the effort_fader help
patch. sampler_engine owns the voices these allocate from. record~ makes new
samples from a microphone or from anything in the ~ graph."""

demo = [
    {'key': 'se', 'init': 'sampler_engine', 'pos': (30, 62), 'w': 260, 'h': 180},
    {'key': 'btn', 'init': 'button', 'pos': (30, 260), 'w': 88, 'h': 46},
    {'key': 'ps', 'init': 'polyphonic_sampler~', 'pos': (30, 325), 'w': 300, 'h': 320},
    {'key': 'i1', 'init': 'int', 'pos': (30, 840), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c0', 'comment': True, 'text': 'load a set, then trigger sound ids\nactive_voices at the limit means\nnew triggers are stealing old ones',
     'pos': (30, 895)},
    {'key': 'btn2', 'init': 'button', 'pos': (410, 260), 'w': 88, 'h': 46},
    {'key': 'gs', 'init': 'granular_sampler~', 'pos': (410, 325), 'w': 300, 'h': 320},
    {'key': 'c3', 'comment': True, 'text': 'a short sample becomes a texture\nthat lasts as long as the movement',
     'pos': (410, 880)},
    {'key': 'vcf', 'init': 'vcf~', 'pos': (790, 325), 'w': 220, 'h': 220},
    {'key': 'fo', 'init': 'fader_out~ 1 2', 'pos': (790, 520), 'w': 220, 'h': 220,
     'props': {'fader': 0.0}},
    {'key': 'c4', 'comment': True, 'text': 'patched into ~ objects, the grains\nplay only through them\nraise the fader to hear it',
     'pos': (790, 850)},
]
links = [('btn', '', 'ps', 'trigger'), ('ps', 'active_voices', 'i1', ''),
         ('btn2', '', 'gs', 'trigger'),
         ('gs', 'left out', 'vcf', 'left in'), ('gs', 'right out', 'vcf', 'right in'),
         ('vcf', 'left out', 'fo', 'left'), ('vcf', 'right out', 'fo', 'right')]
print(build('polyphonic_sampler', 'polyphonic_sampler~ - voices allocated for you',
            body, demo, links, demo_width=1060, text_width=800, text_height=740))

# ---------------------------------------------------------------- effort_fader
body = """These turn measurements of a moving body into sound levels.

They are the join between the motion side of this system and the sampler.
Each takes a body measurement and produces a fade list - [sound_id, level]
pairs - that a polyphonic sampler plays directly.

THE NODES:

effort_fader             from smpl_torque's combined_effort
muscle_activation_fader  from mgl_smpl_heatmap's muscle activations

effort_fader MAPS JOINTS TO SOUNDS:
smpl_torque's 'combined_effort' is one effort vector per joint, and its size
is CAPACITY-RELATIVE - a fraction of that joint's maximum torque rather than
an absolute figure. So a small joint working hard and a large one working
hard read similarly, which is what you want when the sound should reflect
exertion rather than size.

Three modes:
  magnitude          one fader per joint - the vector's length.
                     sound id = start_sid + joint
  components         three faders per joint - the size along each axis.
                     sound id = start_sid + joint * 3 + axis
  signed components  six faders per joint - each axis split into a
                     positive and a negative fader (x+, x-, y+, y-, z+, z-).
                     sound id = start_sid + joint * 6 + axis * 2 + (0 or 1)

Magnitude asks how hard a joint is working. Components asks in which
direction; signed components also tells pushing one way from pushing the
other, which is a different kind of engagement. A list of one number per
joint is accepted too: signed components then gives two faders per joint.

Its output is sparse: a fader is sent while it is above zero, once more at
0 on the frame it falls silent (so the voice stops), and is then left out
until it sounds again. The 'mapping' outlet sends a dict of fader names
(left_knee, left_knee_x, left_knee_x+ ...) to sound ids when the number of
faders changes - on the first input, and after a mode change. Changing
start_sid does not resend it, and does not silence sounds started under the
old ids.

muscle_activation_fader MAPS MUSCLES TO SOUNDS:
It takes the activation dict from mgl_smpl_heatmap and gives each muscle
name the next sound id, from start_sid up, the first time it sees it,
keeping that assignment from then on. So a muscle always plays the same
sound, without you writing out the mapping. The assignments are saved with
the patch; 'reset_mapping', or a change to start_sid, forgets them.

It sends a level for every muscle in the dict on each input, and the
'mapping' outlet sends the name-to-id dict each time, which is how you find
out which sound is which.

SHAPING MATTERS MORE THAN THE MAPPING:
Both have 'threshold', 'curve' and 'gain', and these do most of the work.
Raw activation or effort mapped straight to level gives a wash - everything
makes a bit of sound all the time, and nothing stands out.

A value below the threshold is silenced; the threshold is compared with the
raw value, before the curve. The rest is raised to the power 'curve' (above
1 makes level rise late and steeply), multiplied by 'gain', and capped at
1. Set the threshold first, by watching what a still body produces and
putting it above that.

'top_n' on effort_fader keeps only the strongest few. It subtracts the
level of the strongest fader left out from every level, so at most top_n
sound and each one fades in and out of the group smoothly rather than
jumping. It guarantees the texture never fills up, however the threshold is
set. 0 turns it off.

WHEN THE DATA STOPS:
If no input arrives for 'stale_timeout' seconds while something is
sounding, each node sends 0 to its faders (effort_fader to the ones that
were sounding, muscle_activation_fader to every muscle it knows), so the
sampler's voices are released rather than left hanging.
""" + FADE + """
SYNTAX:
effort_fader
muscle_activation_fader

Neither takes arguments.

EXAMPLE:
effort_fader

INPUTS and PARAMETERS:

effort (effort_fader):
One effort vector (or one number) per joint.

muscle activations (muscle_activation_fader):
A dict of muscle name to activation, 0 to 1.

mode (effort_fader):
magnitude, components or signed components.

start_sid:
The first sound id.

gain / threshold / curve:
The shaping. See above - this is where the work is.

top_n (effort_fader):
Keep only the strongest few. 0 for all.

stale_timeout:
Seconds without input before sounding faders are sent to 0. 0.25 by default.

reset_mapping (muscle_activation_fader):
Forget the name-to-id assignments and start again.

OUTPUTS:

fade:
The [sound_id, level] list, for a polyphonic sampler's 'fade' inlet.

mapping:
Which joint, axis or muscle got which sound id.

RELATED:
smpl_torque and mgl_smpl_heatmap produce the measurements;
polyphonic_sampler~ plays the fade list. crossfade_scanner makes a fade list
from a single number."""

demo = [
    {'key': 'met', 'init': 'metro 100', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 100.0, 'units': 'milliseconds'}},
    {'key': 'nr', 'init': 'np.rand 24 3', 'pos': (30, 150), 'w': 140, 'h': 110,
     'props': {'min': -0.6, 'max': 0.6}},
    {'key': 'c0', 'comment': True,
     'text': 'stands in for smpl_torque combined_effort:\n24 joints, an effort vector each',
     'pos': (30, 150)},
    {'key': 'ef', 'init': 'effort_fader', 'pos': (30, 290), 'w': 280, 'h': 300,
     'props': {'threshold': 0.6, 'top_n': 4}},
    {'key': 'c1', 'comment': True,
     'text': 'threshold silences the weak ones,\ntop_n keeps the strongest 4',
     'pos': (30, 290)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 610), 'w': 560, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'the fade list: [sound_id, level] pairs',
     'pos': (30, 660)},
    {'key': 'btn', 'init': 'button', 'pos': (30, 720), 'w': 88, 'h': 46},
    {'key': 'c3', 'comment': True, 'text': 'click to send three muscle activations',
     'pos': (30, 720)},
    # repeat_in_order: one click stores the text in the dict, THEN sends it on
    {'key': 'rp', 'init': 'repeat_in_order 2', 'pos': (30, 780), 'w': 190, 'h': 70},
    {'key': 's0', 'init': 'string', 'pos': (30, 870), 'w': 500, 'h': 42,
     'props': {'text in': '{"pectoralis": 0.8, "deltoid": 0.3, "biceps": 0.05}',
               'font size': '24'}},
    {'key': 'dc', 'init': 'dict', 'pos': (30, 930), 'w': 160, 'h': 180},
    {'key': 'mf', 'init': 'muscle_activation_fader', 'pos': (30, 1130), 'w': 280, 'h': 260,
     'props': {'threshold': 0.1, 'stale_timeout': 3.0}},
    {'key': 'c4', 'comment': True,
     'text': 'each muscle gets the next id on first sight;\nbiceps is under the threshold;\nno new data for 3 seconds sends zeros',
     'pos': (30, 1130)},
    {'key': 'l2', 'init': 'list', 'pos': (30, 1410), 'w': 400, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'l3', 'init': 'string', 'pos': (30, 1470), 'w': 500, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c5', 'comment': True, 'text': 'fade, then mapping',
     'pos': (30, 1410)},
]
links = [('met', '', 'nr', ''),
         ('nr', 'random array', 'ef', 'effort'),
         ('ef', 'fade', 'l1', ''),
         ('btn', '', 'rp', ''),
         ('rp', 'first', 's0', ''),
         ('rp', 'second', 'dc', 'send dict'),
         ('s0', 'string out', 'dc', 'store'),
         ('dc', 'dict out', 'mf', 'muscle activations'),
         ('mf', 'fade', 'l2', ''),
         ('mf', 'mapping', 'l3', '')]
print(build('effort_fader', 'effort_fader - body measurements as sound levels', body,
            demo, links, demo_width=620, text_width=810, text_height=820))

# ---------------------------------------------------------- crossfade_scanner
body = """crossfade_scanner turns one number from 0 to 1 into a walk through a chain
of sounds, as a fade list for a polyphonic sampler.

Any single continuous control - a limb height, a distance, a fader - becomes
a path through a set of sounds.

HOW THE RANGE IS DIVIDED:
0 to 1 is split into 'n' equal sections, one per sound. At 0 everything is
silent. Through the first section the first sound fades up to full. Through
each section after that the previous sound fades out as the next fades in.
At 1 the last sound is at full and the rest are silent. So with 4 sounds,
0.25 is the first alone, 0.375 is the first and second at half each, and
0.5 is the second alone.

'equal_power' bends each level along a quarter sine, so that at the middle
of a crossover both sounds sit at about 0.71 rather than 0.5 - the
loudness holds steady through the crossover instead of dipping.

WHICH SOUNDS:
By default the sound ids are 0 to n-1. Send a list of ids to 'sound_ids' to
choose them - its length becomes n. Changing 'n' afterwards goes back to 0
to n-1. The ids are saved with the patch.

WHAT IT SENDS:
The whole list - every sound with its level, silent ones at 0 - each time
the value, n, the ids or equal_power changes.

THE fade LIST:
[[sound_id, level], ...] is what a polyphonic sampler's 'fade' inlet takes:
each listed sound plays at its level, and a sound at 0 is silenced. So the
whole crossfade is one message.

SYNTAX:
crossfade_scanner

It takes no arguments.

EXAMPLE:
crossfade_scanner

INPUTS and PARAMETERS:

value:
The position along the chain, 0 to 1.

n:
How many sounds, 2 to 128. 4 by default.

sound_ids:
A list of sound ids; sets n to its length.

equal_power:
Hold the loudness through each crossover.

OUTPUTS:

fade:
The [sound_id, level] list.

RELATED:
polyphonic_sampler~ plays the list. effort_fader and muscle_activation_fader
make fade lists from body data. crossfade blends two values."""

demo = [
    {'key': 'sg', 'init': 'signal', 'pos': (30, 62), 'w': 180, 'h': 180,
     'props': SIG('triangle', period=12.0, rng=1.0, bipolar=False)},
    {'key': 'c0', 'comment': True, 'text': 'a slow sweep from 0 to 1 and back',
     'pos': (30, 62)},
    {'key': 'm1', 'init': 'message', 'pos': (340, 62), 'w': 200, 'h': 42,
     'props': {'text in': '10 11 12', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'click: three sounds, ids 10 to 12',
     'pos': (340, 110)},
    {'key': 'cs', 'init': 'crossfade_scanner', 'pos': (30, 270), 'w': 300, 'h': 180},
    {'key': 'c2', 'comment': True, 'text': 'tick equal_power and watch the middles',
     'pos': (30, 270)},
    {'key': 'l1', 'init': 'list', 'pos': (30, 470), 'w': 560, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'the fade list, for a polyphonic sampler',
     'pos': (30, 520)},
]
links = [('sg', '', 'cs', 'value'),
         ('m1', '', 'cs', 'sound_ids'),
         ('cs', 'fade', 'l1', '')]
print(build('crossfade_scanner', 'crossfade_scanner - one number walks through sounds',
            body, demo, links, demo_width=620, text_width=790, text_height=760))
