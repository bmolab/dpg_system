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
    {'key': 'c0', 'comment': True, 'text': 'one engine per patch', 'pos': (30, 315)},
    {'key': 'c1', 'comment': True, 'text': 'output level: is anything happening?',
     'pos': (30, 345)},
    {'key': 'btn', 'init': 'button', 'pos': (30, 390), 'w': 88, 'h': 46},
    {'key': 'sv', 'init': 'sampler_voice', 'pos': (30, 455), 'w': 280, 'h': 300},
    {'key': 'c2', 'comment': True, 'text': 'load a file, then click play', 'pos': (30, 770)},
    {'key': 'c3', 'comment': True, 'text': 'one voice, held, with loop points',
     'pos': (30, 800)},
    {'key': 'mv', 'init': 'multi_voice_sampler', 'pos': (360, 455), 'w': 300, 'h': 300},
    {'key': 'c4', 'comment': True, 'text': 'or every voice from one node,',
     'pos': (360, 770)},
    {'key': 'c5', 'comment': True, 'text': 'one at a time via voice index',
     'pos': (360, 800)},
]
links = [('se', 'output level', 'f1', ''), ('btn', '', 'sv', 'play')]
print(build('sampler_engine', 'sampler_engine - the engine and its voices', body,
            demo, links, demo_width=700, text_width=800, text_height=720))

# -------------------------------------------------------- polyphonic_sampler
body = """These play sounds without you deciding which voice each one goes into.

THE NODES:

polyphonic_sampler  allocate a voice per triggered sound
granular_sampler    play a sound as a cloud of short grains
scratch_sampler     play with the position under direct control

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
""" + FADE + """
SYNTAX:
polyphonic_sampler
granular_sampler
scratch_sampler

EXAMPLE:
polyphonic_sampler

INPUTS and PARAMETERS:

trigger / stop:
Start and stop a sound.

load / load_set:
One file, or a set of them at once.

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

RELATED:
The fader nodes produce fade lists from body data - see the effort_fader help
patch. sampler_engine owns the voices these allocate from."""

demo = [
    {'key': 'se', 'init': 'sampler_engine', 'pos': (30, 62), 'w': 260, 'h': 180},
    {'key': 'btn', 'init': 'button', 'pos': (30, 260), 'w': 88, 'h': 46},
    {'key': 'ps', 'init': 'polyphonic_sampler', 'pos': (30, 325), 'w': 300, 'h': 320},
    {'key': 'i1', 'init': 'int', 'pos': (30, 660), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c0', 'comment': True, 'text': 'load a set, then trigger sound ids',
     'pos': (30, 715)},
    {'key': 'c1', 'comment': True, 'text': 'active_voices at the limit means',
     'pos': (30, 745)},
    {'key': 'c2', 'comment': True, 'text': 'new triggers are stealing old ones',
     'pos': (30, 775)},
    {'key': 'gs', 'init': 'granular_sampler', 'pos': (380, 325), 'w': 300, 'h': 320},
    {'key': 'c3', 'comment': True, 'text': 'a short sample becomes a texture',
     'pos': (380, 660)},
    {'key': 'c4', 'comment': True, 'text': 'that lasts as long as the movement',
     'pos': (380, 690)},
]
links = [('btn', '', 'ps', 'trigger'), ('ps', 'active_voices', 'i1', '')]
print(build('polyphonic_sampler', 'polyphonic_sampler - voices allocated for you',
            body, demo, links, demo_width=720, text_width=800, text_height=740))

# ---------------------------------------------------------------- effort_fader
body = """These turn measurements of a moving body into sound levels.

They are the join between the motion side of this system and the sampler.
Each takes a body measurement and produces a fade list - [sound_id, level]
pairs - that a polyphonic sampler plays directly.

THE NODES:

effort_fader             from smpl_torque's combined_effort
muscle_activation_fader  from a muscle activation dict
crossfade_scanner        from a single number, 0 to 1

effort_fader MAPS JOINTS TO SOUNDS:
smpl_torque's 'combined_effort' is a per-joint effort vector, and its magnitude
is CAPACITY-RELATIVE - a fraction of that joint's maximum torque rather than an
absolute figure. So a small joint working hard and a large one working hard read
similarly, which is what you want when the sound should reflect exertion rather
than size.

Two modes:
  magnitude   one fader per joint - the vector's length.
              sound id = start_sid + joint
  components  three faders per joint - one per axis.
              sound id = start_sid + joint * 3 + axis

Magnitude asks how hard a joint is working. Components asks in which direction,
and gives you three times as many sounds to say it with.

muscle_activation_fader MAPS MUSCLES TO SOUNDS:
It takes the activation dict from mgl_smpl_heatmap and assigns each muscle name
a sound id on first sight, keeping that assignment stable across frames. So a
muscle always plays the same sound, without you writing out the mapping.

The 'mapping' outlet reports what it decided, which is how you find out which
sound is which.

SHAPING MATTERS MORE THAN THE MAPPING:
Both faders have 'threshold', 'curve' and 'gain', and these do most of the work.
Raw activation or effort mapped straight to level gives a wash - everything
makes a bit of sound all the time, and nothing stands out.

The threshold silences what is merely present, the curve decides how sharply
level rises with effort, and gain sets the top. Set the threshold first, by
watching what a still body produces and putting it above that.

'top_n' on effort_fader keeps only the loudest few joints, which is a blunter
version of the same idea and sometimes the better one - it guarantees the
texture never fills up regardless of how the thresholds are set.

crossfade_scanner IS THE SIMPLE CASE:
One number from 0 to 1 walks a crossfade along a chain of sounds - the first
fades up, then crossfades into the second, and so on. Any single continuous
control - a limb height, a distance, a fader - becomes a path through a set of
sounds. 'equal_power' keeps the loudness even through each crossover rather than
dipping in the middle.
""" + FADE + """
SYNTAX:
effort_fader
muscle_activation_fader
crossfade_scanner

EXAMPLE:
effort_fader

INPUTS and PARAMETERS:

effort / muscle activations / value:
The measurement.

start_sid:
Which sound id the mapping begins at.

mode (effort_fader):
magnitude or components.

threshold / curve / gain:
The shaping. See above - this is where the work is.

top_n (effort_fader):
Keep only the loudest few.

stale_timeout:
How long before a sound that has stopped being mentioned fades out.

reset_mapping (muscle_activation_fader):
Forget the name-to-id assignments and start again.

n / sound_ids / equal_power (crossfade_scanner):
How many sounds, which ids, and whether to hold the loudness through crossovers.

OUTPUTS: 

fade:
The [sound_id, level] list, for a polyphonic sampler.

mapping:
Which name or joint got which sound id."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'be', 'init': 'smpl_beta_editor', 'pos': (350, 62), 'w': 300, 'h': 340},
    {'key': 'ss', 'init': 'shadow_to_smpl', 'pos': (30, 400), 'w': 260, 'h': 140},
    {'key': 'st', 'init': 'smpl_torque', 'pos': (30, 560), 'w': 300, 'h': 340},
    {'key': 'ef', 'init': 'effort_fader', 'pos': (30, 920), 'w': 280, 'h': 300},
    {'key': 'c0', 'comment': True, 'text': 'set threshold above what a still body gives',
     'pos': (30, 1235)},
    {'key': 'c1', 'comment': True, 'text': 'otherwise everything sounds, always',
     'pos': (30, 1265)},
    {'key': 'se', 'init': 'sampler_engine', 'pos': (380, 560), 'w': 260, 'h': 180},
    {'key': 'ps', 'init': 'polyphonic_sampler', 'pos': (380, 760), 'w': 300, 'h': 320},
    {'key': 'c2', 'comment': True, 'text': 'the fade list plays the whole texture',
     'pos': (380, 1095)},
    {'key': 'c3', 'comment': True, 'text': 'in one message', 'pos': (380, 1125)},
]
links = [('sh', 'body 1 quaternions', 'ss', 'pose'),
         ('sh', 'body 1 positions', 'ss', 'positions'),
         ('be', 'config', 'ss', 'config'),
         ('ss', 'pose', 'st', 'pose'), ('ss', 'trans', 'st', 'trans'),
         ('be', 'config', 'st', 'config'),
         ('st', 'combined_effort', 'ef', 'effort'),
         ('ef', 'fade', 'ps', 'fade')]
print(build('effort_fader', 'effort_fader - a moving body playing sounds', body,
            demo, links, demo_width=700, text_width=810, text_height=820))
