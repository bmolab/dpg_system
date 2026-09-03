"""Non-semantic speech analysis: how something is said, not what is said."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

SUITE = """
THE FAMILY, AND WHAT EACH ONE HEARS:
speech_pitch          the note being spoken - f0, and whether there is a voice
speech_prosody        the shape of that pitch over a window - the melody
speech_envelope       loudness, on two time scales, and onsets
speech_spectral       where the energy sits - bright or dark, tonal or noisy
speech_voice_quality  how clear the voice is - breathy, rough, or clean

None of these know what is being said. They measure HOW, which is the part that
carries urgency, hesitation and effort - and the part a patch can respond to
without any of the machinery of understanding language.

THEY ANALYSE ON A CLOCK, NOT ON DATA:
Each of these keeps an internal ring buffer, and runs its analysis at
'analysis_fps' - measured against the wall clock. Audio arriving faster than
that is buffered, not analysed.

This is worth knowing because it means the outputs are NOT one-per-chunk. Push a
file through as fast as it will go and you get far fewer analysis frames than
chunks - the node is pacing itself in real time. It is built for a live stream,
where that is exactly right.
"""

SOURCE = """
FEEDING THEM:
t.audio_source is the usual source - a live input, giving audio tensors. The
demo here is wired that way, so it needs an actual input to show anything; with
nothing arriving the outputs simply stay quiet.
"""

# ---------------------------------------------------------------- speech_pitch
body = """These follow the pitch of a voice, and the melody it traces.

THE NODES:

speech_pitch    the note being spoken, right now
speech_prosody  what that note has been doing over the last second
""" + SUITE + """
f0 IS THE NOTE, voiced IS WHETHER THERE IS ONE:
f0 is the fundamental frequency in Hz - roughly 85 to 180 for an adult male
voice, 165 to 255 for an adult female one, higher for children.

But speech is not continuously pitched. Vowels have a pitch; 's', 'f' and 't' do
not, and neither do the gaps. So 'voiced' matters as much as f0: it says whether
there is a note to report at all, and 'voiced_prob' gives the confidence behind
that as a number rather than a yes or no.

Anything downstream should check voiced first. An f0 reading taken during an 's'
is not a quiet note, it is a meaningless one.

SET min_freq AND max_freq TO THE ACTUAL SPEAKER:
This is the single most useful adjustment here. Pitch trackers make OCTAVE
ERRORS - reporting half or double the real frequency - and a range that matches
the person talking removes most of them at a stroke.

Leaving the range wide does not make the node more general, it makes it less
reliable. Narrow it to the voice you actually have.

'voiced_attack' and 'voiced_release' are hysteresis on the voiced flag: how
quickly it is willing to say a voice has started, and how long it waits before
saying it has stopped. Longer release stops the flag chattering during the small
gaps inside continuous speech.

THREE BACKENDS, AND IT PICKS THE BEST ONE PRESENT:
parselmouth (Praat) is preferred, then pyin (librosa), then kaldi (torchaudio).
They differ in accuracy and cost rather than in what they mean, and the node
falls back to whichever is installed. If none is, it produces nothing.

speech_prosody TAKES THE CONTOUR, NOT THE NOTE:
Feed it 'f0_raw' - the recent history as an array - not 'f0'. Prosody is about
what the pitch has been DOING, and that question needs the contour.

Over 'window_sec' of history it reports:

pitch_slope        Hz per second - rising or falling, and how fast
pitch_range        the span covered in the window
pitch_variability  how much it wanders
pitch_mean         where it sits
intonation         a WORD: 'rising', 'falling', 'flat' or 'unvoiced'

intonation is a string, not a number - it is the slope already judged for you,
flat meaning under 5 Hz per second either way. A question rising at the end, a
statement falling: that is what this outlet is for.

SYNTAX:
speech_pitch
speech_prosody

EXAMPLE:
speech_pitch

INPUTS and PARAMETERS:

audio tensor in:
The live audio.

buffer_sec / analysis_fps:
How much history to analyse, and how often - in real seconds.

min_freq / max_freq:
The speaker's range. Narrow it.

f0_in (speech_prosody):
The f0 CONTOUR, from f0_raw.

window_sec / smoothing:
How much history the prosody covers, and how much it is smoothed.

OUTPUTS: 

f0 / f0_raw:
The current note, and the recent contour.

voiced / voiced_prob:
Whether there is a note, and the confidence.

pitch_slope / pitch_range / pitch_variability / pitch_mean:
The melody, as numbers.

intonation:
The melody, as a word.
""" + SOURCE + """
RELATED:
speech_envelope for how loud rather than how high.
speech_voice_quality for how clear the voice is - and it needs 'voiced' too."""

demo = [
    {'key': 'src', 'init': 't.audio_source', 'pos': (30, 62), 'w': 260, 'h': 180},
    {'key': 'pit', 'init': 'speech_pitch', 'pos': (30, 270), 'w': 280, 'h': 300},
    {'key': 'f1', 'init': 'float', 'pos': (340, 270), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c0', 'comment': True, 'text': 'f0 in Hz - but check voiced first',
     'pos': (340, 320)},
    {'key': 'f2', 'init': 'float', 'pos': (340, 365), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'voiced: 0 through s, f, t and silence',
     'pos': (340, 415)},
    {'key': 'pl', 'init': 'plot', 'pos': (340, 460), 'w': 300, 'h': 180,
     'props': PLOT(50.0, 350.0, 200)},
    {'key': 'c2', 'comment': True, 'text': 'narrow min_freq/max_freq to the actual',
     'pos': (30, 600)},
    {'key': 'c3', 'comment': True, 'text': 'speaker - it removes most octave errors',
     'pos': (30, 630)},

    {'key': 'pro', 'init': 'speech_prosody', 'pos': (30, 680), 'w': 280, 'h': 240},
    {'key': 'c4', 'comment': True, 'text': 'fed from f0_raw, the CONTOUR - prosody',
     'pos': (30, 950)},
    {'key': 'c5', 'comment': True, 'text': 'is about what the pitch has been doing',
     'pos': (30, 980)},
    {'key': 'f3', 'init': 'float', 'pos': (350, 680), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c6', 'comment': True, 'text': 'slope, Hz per second', 'pos': (350, 730)},
    {'key': 'l1', 'init': 'list', 'pos': (350, 775), 'w': 260, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c7', 'comment': True, 'text': "intonation is a WORD: rising, falling,",
     'pos': (350, 825)},
    {'key': 'c8', 'comment': True, 'text': "flat or unvoiced", 'pos': (350, 855)},
]
links = [('src', 'audio tensors', 'pit', 'audio tensor in'),
         ('pit', 'f0', 'f1', ''), ('pit', 'voiced', 'f2', ''),
         ('pit', 'f0', 'pl', 'y'),
         ('pit', 'f0_raw', 'pro', 'f0_in'),
         ('pro', 'pitch_slope', 'f3', ''),
         ('pro', 'intonation', 'l1', '')]
print(build('speech_pitch', 'speech_pitch and speech_prosody - the melody', body,
            demo, links, demo_width=680, text_width=810, text_height=780))

# ------------------------------------------------------------- speech_envelope
body = """Loudness, followed on two time scales at once - and the onsets between them.

THE NODE:

speech_envelope   envelope, slow volume, crest factor, and an onset trigger
""" + SUITE + """
THE TWO TIME SCALES ARE THE WHOLE IDEA:
'envelope' is fast. It follows the actual shape of speech - each syllable rising
and falling.

'volume_db' is slow. It follows the ambient level over seconds, and is meant to
sit still while someone talks - it is the room, not the voice.

An ONSET is what happens when the fast one gets more than 'onset_threshold_db'
above the slow one: something just happened that is louder than the background
has been. That is a far better trigger than a fixed threshold, because it adapts
- move to a noisier room and the baseline rises with it, and the onsets keep
meaning the same thing.

The default is 6 dB, which is roughly "twice as loud as it has been".

WHAT min_cutoff AND beta ACTUALLY DO:
Both filters here are One Euro filters, and the two numbers trade steadiness
against lag in a way a single smoothing amount cannot:

min_cutoff  how much smoothing when the signal is STILL. Lower is smoother and
            calmer, at the cost of being slower to notice a change.
beta        how much the filter opens up when the signal is MOVING. Higher means
            it stops smoothing and follows, so fast changes are not lagged.

That is why the defaults differ so sharply between the two followers.
Envelope: min_cutoff 1.0, beta 0.5 - lively, follows syllables. Volume:
min_cutoff 0.1, beta 0.01 - almost immovable, which is what a baseline is for.

If onsets are being missed, raise env_beta before touching the threshold. If the
baseline is drifting up during long speech, lower vol_beta.

CREST FACTOR IS PEAKINESS:
Peak divided by average, so it says whether the energy arrives in spikes or
steadily. Some measured reference points, which are worth having:

pure sine tone       1.41   (exactly the square root of 2)
voice-like harmonics 1.77
white noise          3.45

Consonants and plosives are peaky, sustained vowels are not. A rising crest
factor with the envelope steady means the character of the sound changed without
the loudness changing.

SYNTAX:
speech_envelope

EXAMPLE:
speech_envelope

INPUTS and PARAMETERS:

audio tensor in:
The live audio.

frame_hop:
Samples per envelope frame - how finely the envelope is sampled.

env_min_cutoff / env_beta:
The fast follower. Smoothing when still, and how much it opens up when moving.

vol_min_cutoff / vol_beta:
The slow baseline. Both small, deliberately.

onset_threshold_db:
How far above the baseline counts as an onset. 6 dB by default.

OUTPUTS: 

envelope / envelope_db:
The fast loudness, linear and in dB.

volume_db:
The slow baseline.

crest_factor:
Peak over average - how spiky the sound is.

onset:
A one-shot trigger when the envelope jumps clear of the baseline.
""" + SOURCE + """
RELATED:
speech_pitch for how high rather than how loud.
Feeding onset to a counter or a sample_hold is the usual way to make something
happen once per utterance."""

demo = [
    {'key': 'src', 'init': 't.audio_source', 'pos': (30, 62), 'w': 260, 'h': 180},
    {'key': 'env', 'init': 'speech_envelope', 'pos': (30, 270), 'w': 290, 'h': 300},
    {'key': 'pl', 'init': 'plot', 'pos': (360, 270), 'w': 300, 'h': 180,
     'props': PLOT(-60.0, 0.0, 200)},
    {'key': 'c0', 'comment': True, 'text': 'the fast one: syllables', 'pos': (360, 460)},
    {'key': 'pl2', 'init': 'plot', 'pos': (360, 505), 'w': 300, 'h': 180,
     'props': PLOT(-60.0, 0.0, 200)},
    {'key': 'c1', 'comment': True, 'text': 'the slow one: the room. It should sit',
     'pos': (360, 695)},
    {'key': 'c2', 'comment': True, 'text': 'still while someone is talking',
     'pos': (360, 725)},

    {'key': 'f1', 'init': 'float', 'pos': (30, 610), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c3', 'comment': True, 'text': 'crest factor: sine 1.41, voice 1.77,',
     'pos': (30, 660)},
    {'key': 'c4', 'comment': True, 'text': 'white noise 3.45 - how peaky it is',
     'pos': (30, 690)},

    {'key': 'trg', 'init': 'trigger', 'pos': (30, 740), 'w': 200, 'h': 160,
     'props': {'threshold': 0.5, 'release threshold': 0.25,
               'trigger mode': 'output bang'}},
    {'key': 'cnt', 'init': 'counter', 'pos': (30, 915), 'w': 180, 'h': 110},
    {'key': 'i1', 'init': 'int', 'pos': (30, 1040), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c5b', 'comment': True, 'text': 'onset is a continuous 0/1 stream, not a',
     'pos': (250, 740)},
    {'key': 'c5c', 'comment': True, 'text': 'bang - a trigger in output bang mode',
     'pos': (250, 770)},
    {'key': 'c5d', 'comment': True, 'text': 'turns it into one count per onset',
     'pos': (250, 800)},
    {'key': 'c5', 'comment': True, 'text': 'onset fires when the fast one gets 6 dB',
     'pos': (30, 1090)},
    {'key': 'c6', 'comment': True, 'text': 'clear of the slow one - so it adapts to',
     'pos': (30, 1120)},
    {'key': 'c7', 'comment': True, 'text': 'the room instead of a fixed threshold',
     'pos': (30, 1150)},
]
links = [('src', 'audio tensors', 'env', 'audio tensor in'),
         ('env', 'envelope_db', 'pl', 'y'),
         ('env', 'volume_db', 'pl2', 'y'),
         ('env', 'crest_factor', 'f1', ''),
         ('env', 'onset', 'trg', 'input'), ('trg', 'out', 'cnt', 'input'),
         ('cnt', 'count out', 'i1', '')]
print(build('speech_envelope', 'speech_envelope - loudness on two time scales', body,
            demo, links, demo_width=700, text_width=810, text_height=780))

# ------------------------------------------------------------- speech_spectral
body = """Where the energy sits in the sound, and how clear the voice producing it is.

THE NODES:

speech_spectral       bright or dark, tonal or noisy
speech_voice_quality  breathy, rough, or clean
""" + SUITE + """
THE SPECTRAL MEASURES, IN PLAIN TERMS:

centroid    the balance point of the spectrum, in Hz. This is BRIGHTNESS - an
            's' is bright, a vowel is dark, and the number moves a long way
            between them.
bandwidth   how spread out the energy is around that point. Narrow means the
            sound is concentrated, wide means it is not.
rolloff     the frequency below which most of the energy sits ('most' being
            rolloff_pct). A more robust brightness measure than centroid when
            there is hiss about.
flatness    tonal against noisy. 0 is a pure tone, 1 would be perfectly flat
            white noise.
contrast    peak-to-valley across sub-bands, as an array - how much structure
            there is at each region of the spectrum.
mfcc        the standard compact description of spectral shape, as an array.
            This is what you feed a classifier.

READ flatness RELATIVELY, NOT AGAINST 1:
The scale says 0 to 1, but the top is theoretical. Measured on this node,
actual white noise reads about 0.56, not 1.0 - so treat 0.5 as "as noisy as
things get" rather than waiting for a number that will never arrive.

Some measured points to calibrate against:

                        flatness   centroid
voice-like harmonics      0.00       367 Hz
those harmonics + noise   0.03      2803 Hz
white noise               0.56      3992 Hz

Notice how much further the CENTROID moves than the flatness. For telling
breathy from clear, brightness is often the more sensitive instrument.

VOICE QUALITY IS ABOUT THE SOURCE, NOT THE SHAPE:
HNR is harmonics against noise, in dB - how much of the sound is a clean
periodic voice and how much is breath and turbulence. High is clear, low is
breathy or rough. Real speech generally falls somewhere around 10 to 25 dB.

jitter is cycle-to-cycle wobble in the PERIOD, shimmer the same in AMPLITUDE.
Both rise with roughness and strain. hnr_smooth is HNR with the jitter of the
measurement itself taken off, for when you want a slow-moving quantity to drive
something.

THE TRAP: ZERO JITTER CAN MEAN NO VOICE AT ALL:
These measures only mean anything while there IS a voice. Fed white noise, this
node reports jitter of exactly 0.0 - not because the voice is perfect, but
because there is no periodicity to measure the variation of. HNR gives the game
away in that case, at about -6 dB.

So gate on speech_pitch's 'voiced' before believing jitter or shimmer. A zero
during silence is not a clean voice.

SYNTAX:
speech_spectral
speech_voice_quality

EXAMPLE:
speech_spectral

INPUTS and PARAMETERS:

audio tensor in:
The live audio.

n_fft:
The analysis window. Bigger means finer frequency detail and coarser timing.

rolloff_pct / n_mfcc:
Where to put the rolloff point, and how many coefficients to report.

min_freq / max_freq (speech_voice_quality):
The speaker's range again - the same reasoning as speech_pitch.

smoothing_sec:
How much history hnr_smooth averages over.

OUTPUTS: 

centroid / bandwidth / rolloff / flatness:
Single numbers describing the spectrum.

contrast / mfcc:
Arrays.

hnr / hnr_smooth / jitter / shimmer:
Voice quality. Meaningless unless voiced.
""" + SOURCE + """
RELATED:
speech_pitch, whose 'voiced' outlet is what should be gating this one.
t.mfcc and the torchaudio nodes if you want the spectral machinery directly."""

demo = [
    {'key': 'src', 'init': 't.audio_source', 'pos': (30, 62), 'w': 260, 'h': 180},
    {'key': 'sp', 'init': 'speech_spectral', 'pos': (30, 270), 'w': 290, 'h': 280},
    {'key': 'pl', 'init': 'plot', 'pos': (360, 270), 'w': 300, 'h': 180,
     'props': PLOT(0.0, 5000.0, 200)},
    {'key': 'c0', 'comment': True, 'text': 'centroid: brightness. An s is bright,',
     'pos': (360, 460)},
    {'key': 'c1', 'comment': True, 'text': 'a vowel is dark', 'pos': (360, 490)},
    {'key': 'f1', 'init': 'float', 'pos': (360, 535), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c2', 'comment': True, 'text': 'flatness: 0 tonal. White noise reads',
     'pos': (360, 585)},
    {'key': 'c3', 'comment': True, 'text': 'only about 0.56, so read it relatively',
     'pos': (360, 615)},

    {'key': 'vq', 'init': 'speech_voice_quality', 'pos': (30, 590), 'w': 290, 'h': 240},
    {'key': 'f2', 'init': 'float', 'pos': (360, 680), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c4', 'comment': True, 'text': 'HNR in dB: real speech is about 10-25.',
     'pos': (360, 730)},
    {'key': 'c5', 'comment': True, 'text': 'White noise reads about -6', 'pos': (360, 760)},
    {'key': 'f3', 'init': 'float', 'pos': (360, 805), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c6', 'comment': True, 'text': 'jitter - but a 0 here can mean NO VOICE,',
     'pos': (360, 855)},
    {'key': 'c7', 'comment': True, 'text': 'not a perfect one. Gate on voiced.',
     'pos': (360, 885)},
    {'key': 'pit', 'init': 'speech_pitch', 'pos': (30, 870), 'w': 280, 'h': 300},
    {'key': 'c8', 'comment': True, 'text': "this is here for its 'voiced' outlet -",
     'pos': (30, 1200)},
    {'key': 'c9', 'comment': True, 'text': 'nothing above should be believed without it',
     'pos': (30, 1230)},
]
links = [('src', 'audio tensors', 'sp', 'audio tensor in'),
         ('sp', 'centroid', 'pl', 'y'), ('sp', 'flatness', 'f1', ''),
         ('src', 'audio tensors', 'vq', 'audio tensor in'),
         ('vq', 'hnr', 'f2', ''), ('vq', 'jitter', 'f3', ''),
         ('src', 'audio tensors', 'pit', 'audio tensor in')]
print(build('speech_spectral', 'speech_spectral and voice quality - timbre', body,
            demo, links, demo_width=700, text_width=810, text_height=790))
