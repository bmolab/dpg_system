
"""
Audio-rate modular synthesis core for dpg_system.

The design separates two rates that must not be confused:

  * The node layer runs on the main thread at GUI frame rate (~60 Hz). It owns
    widgets, patch cords, save/load and all user interaction.
  * The DSP layer runs inside the PortAudio callback at the sample rate, in
    blocks. It is a flat, pre-ordered list of Unit objects rendering into
    preallocated numpy buffers.

Patch cords declare topology; they never carry audio. When a signal connection
is made or broken, the compiler walks the patch, topologically sorts the units
and hands the audio thread a new SynthProgram. Units persist across recompiles
(each node owns its unit for life), so filter/envelope/oscillator state is
continuous -- only the execution order is rebuilt.

Cost per block is a handful of numpy calls per unit on `frames`-length arrays.
At 512 frames / 44.1 kHz the budget is 11.6 ms; a unit costs single-digit
microseconds, so a few hundred units fit comfortably. The thing to avoid is
per-sample Python, which is why the one inherently recursive unit (the filter)
uses a numba kernel and pink noise goes through lfilter.
"""

import math
import os
import platform
import struct
import threading
import time

import numpy as np

try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    njit = None
    _HAVE_NUMBA = False


# Buffers are allocated once at MAX_BLOCK and used as views of length `frames`.
# PortAudio is opened with a fixed blocksize, but a device can still hand us a
# short block, and we must never allocate on the audio thread.
MAX_BLOCK = 4096

DEFAULT_SAMPLE_RATE = 44100

# Reused index ramp so per-block curve generation does not rebuild an arange.
_INDEX_RAMP = np.arange(1, MAX_BLOCK + 1, dtype=np.float64)


# ----------------------------------------------------------------------------
# Signals and inlets
# ----------------------------------------------------------------------------

class Signal:
    """One block of audio, with a constant fast path.

    A unit that produces an unchanging value this block sets `constant` and
    `value` and leaves `data` untouched; consumers branch on the flag and do
    scalar arithmetic instead of filling an array. This is the ar/kr
    distinction from SuperCollider, decided per block rather than at patch
    time, so an unmodulated parameter costs nothing while remaining patchable.
    """
    __slots__ = ('data', 'constant', 'value')

    def __init__(self):
        self.data = np.zeros(MAX_BLOCK, dtype=np.float32)
        self.constant = True
        self.value = 0.0

    def set_constant(self, value):
        self.constant = True
        self.value = float(value)

    def array(self, frames):
        """Materialize this signal as an array view, filling it if constant."""
        if self.constant:
            self.data[:frames] = self.value
        return self.data[:frames]


class Inlet:
    """A unit input: a knob value plus any number of patched signal sources.

    This is the analog triad in one object -- `base` is the panel knob,
    `sources` are the patched CV cords, `depth` is the attenuverter. Multiple
    cords into one inlet sum, as they would through a passive multiple.

    Audio inputs use the same class with base 0 and depth 1; there is no
    structural difference between an audio path and a modulation path, which
    is the point of a modular system.
    """
    __slots__ = ('sources', 'base', 'depth', 'out', 'min', 'max')

    def __init__(self, base=0.0, depth=1.0, minimum=None, maximum=None):
        self.sources = []
        self.base = float(base)
        self.depth = float(depth)
        self.min = minimum
        self.max = maximum
        self.out = Signal()

    def _clamp(self, value):
        if self.min is not None and value < self.min:
            return self.min
        if self.max is not None and value > self.max:
            return self.max
        return value

    def eval(self, frames):
        """Resolve this inlet for the current block and return its Signal."""
        out = self.out
        sources = self.sources
        count = len(sources)
        base = self.base

        if count == 0:
            out.constant = True
            out.value = self._clamp(base)
            return out

        depth = self.depth

        if count == 1:
            source = sources[0]
            if source.constant:
                out.constant = True
                out.value = self._clamp(base + depth * source.value)
                return out
            buffer = out.data[:frames]
            np.multiply(source.data[:frames], depth, out=buffer)
            if base != 0.0:
                buffer += base
            if self.min is not None or self.max is not None:
                np.clip(buffer, self.min, self.max, out=buffer)
            out.constant = False
            return out

        # Several cords into one inlet: sum the varying ones, fold the
        # constant ones into a scalar so we only touch memory once.
        scalar_sum = 0.0
        varying = []
        for source in sources:
            if source.constant:
                scalar_sum += source.value
            else:
                varying.append(source)

        if not varying:
            out.constant = True
            out.value = self._clamp(base + depth * scalar_sum)
            return out

        buffer = out.data[:frames]
        np.copyto(buffer, varying[0].data[:frames])
        for source in varying[1:]:
            buffer += source.data[:frames]
        if scalar_sum != 0.0:
            buffer += scalar_sum
        if depth != 1.0:
            buffer *= depth
        if base != 0.0:
            buffer += base
        if self.min is not None or self.max is not None:
            np.clip(buffer, self.min, self.max, out=buffer)
        out.constant = False
        return out


# ----------------------------------------------------------------------------
# Unit base
# ----------------------------------------------------------------------------

class Unit:
    """Base class for every DSP object. render() runs on the audio thread."""

    # Seconds to fade in or out when a unit is switched on or off. Short
    # enough to feel like a switch, long enough not to be a step.
    GATE_SECONDS = 0.006

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        self.sample_rate = float(sample_rate)
        self.inlets = []
        self.outlets = []
        self._sync_armed = True
        self.enabled = True
        self._switched_off = False
        self._gate_level = 1.0
        self._gate_ramp = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._level_glide = 1.0
        self._level_ramp = np.zeros(MAX_BLOCK, dtype=np.float64)

    def new_inlet(self, base=0.0, depth=1.0, minimum=None, maximum=None):
        inlet = Inlet(base, depth, minimum, maximum)
        self.inlets.append(inlet)
        return inlet

    def new_outlet(self):
        outlet = Signal()
        self.outlets.append(outlet)
        return outlet

    def render(self, frames):
        pass

    def reset(self):
        pass

    @staticmethod
    def _decay_ramp(coefficient, frames, scratch):
        """coefficient**(1..frames), in closed form.

        Exponential segments are generated this way rather than by per-sample
        recursion: same result, one vectorized pass.
        """
        view = scratch[:frames]
        if coefficient <= 0.0:
            view[:] = 0.0
            return view
        np.multiply(_INDEX_RAMP[:frames], math.log(coefficient), out=view)
        np.exp(view, out=view)
        return view

    # -- enable ---------------------------------------------------------------
    #
    # Switching a source off is worth more than a mute: once it has faded out
    # there is nothing to render, so the block is skipped entirely and the
    # outlets go constant -- which makes everything downstream cheap too, since
    # a constant input takes the scalar path through every unit it reaches.
    # A disabled voice therefore costs almost nothing, which is the point of
    # having the switch at all when there are two dozen of them.
    #
    # It cannot simply stop, though. A running oscillator cut to zero between
    # two samples is a step in the waveform, which is a click -- so it fades,
    # and only stops once the fade has finished.

    def gate(self, frames):
        """How this block should be scaled: 1.0, 0.0, or a ramp between.

        1.0 means render normally and do nothing. 0.0 means the unit is off
        and settled, so there is nothing to render at all. Anything else is a
        per-sample ramp to multiply the output by.
        """
        target = 1.0 if self.enabled else 0.0
        level = self._gate_level
        if level == target:
            return target

        span = 1.0 / max(1.0, Unit.GATE_SECONDS * self.sample_rate)
        ramp = self._gate_ramp[:frames]
        np.multiply(_INDEX_RAMP[:frames], span if target > level else -span,
                    out=ramp)
        ramp += level
        np.clip(ramp, min(level, target), max(level, target), out=ramp)
        self._gate_level = float(ramp[-1])
        return ramp

    def silence(self, frames):
        """Put every outlet at constant zero. Used when a unit is switched off."""
        for signal in self.outlets:
            signal.set_constant(0.0)

    def deactivate(self):
        """Called once when a unit has finished switching off.

        For anything holding a history that stops making sense once the unit
        stops -- a delay line, chiefly -- this is where to drop it. A delay
        that simply stopped writing would leave a seam in its buffer between
        what it had before and what it gets afterwards, and the read head
        crosses that seam a delay time after coming back, which is long after
        any fade has finished and so is heard as a click.
        """

    def bypass_pairs(self):
        """(inlet, outlet) pairs to carry straight through when switched off.

        This is what separates a source from a processor. A source has none,
        so switching it off leaves silence. A processor names the way its
        input reaches its output, so switching it off leaves the signal alone
        instead of removing it -- which is what bypass has to mean, and why it
        cannot simply be the same switch.

        Units whose right channel mirrors the left when nothing is patched to
        it say so here, so a bypassed mono chain does not lose one side.
        """
        return ()

    def run(self, frames):
        """What the program calls. render() is the unit's own business.

        The whole switch lives here rather than in each unit's render, so
        every unit gets it by naming its dry paths and nothing else, and the
        cost when everything is switched on is one float comparison.
        """
        level = self.gate(frames)
        if isinstance(level, float):
            if level == 1.0:
                self._switched_off = False
                self.render(frames)
                return
            if not self._switched_off:
                self._switched_off = True
                self.deactivate()
            pairs = self.bypass_pairs()
            if pairs:
                self._carry_dry(frames, pairs)
            else:
                self.silence(frames)
            return
        self._switched_off = False

        # Part way between. The unit has to run either way, since what is
        # being faded between is its output and either silence or its input.
        self.render(frames)
        pairs = self.bypass_pairs()
        if pairs:
            self._blend_dry(frames, level, pairs)
            return
        for signal in self.outlets:
            buffer = signal.array(frames)
            np.multiply(buffer, level, out=buffer, casting='unsafe')
            signal.constant = False

    def _carry_dry(self, frames, pairs):
        """Fully bypassed: the input, and none of the work."""
        for inlet, outlet in pairs:
            source = inlet.eval(frames)
            if source.constant:
                outlet.set_constant(source.value)
            else:
                np.copyto(outlet.data[:frames], source.data[:frames])
                outlet.constant = False

    def _blend_dry(self, frames, level, pairs):
        """Part way in or out: cross from the input to the processed signal.

        A filter cut straight to dry would step by whatever it was removing,
        which is a click; crossing over a few milliseconds is not.
        """
        for inlet, outlet in pairs:
            dry = inlet.eval(frames).array(frames)
            wet = outlet.array(frames)
            np.subtract(wet, dry, out=wet)
            np.multiply(wet, level, out=wet)
            np.add(wet, dry, out=wet)
            outlet.constant = False

    # -- shared oscillator machinery -----------------------------------------
    #
    # Pitch and hard sync mean the same thing wherever they appear, so every
    # oscillator resolves them through these rather than through its own copy.

    def _build_increment(self, increment, frequency, pitch, linear_fm, frames):
        """Phase increment in cycles per sample, into `increment`.

        Frequency in Hz, scaled by an exponential inlet in octaves (1.0 = up
        an octave, matching a 1V/oct input) and offset by a linear FM inlet in
        Hz. Clamped below Nyquist, which is also what PolyBLEP needs.
        """
        if pitch.constant:
            multiplier = 2.0 ** pitch.value
            if frequency.constant:
                increment[:] = frequency.value * multiplier
            else:
                np.multiply(frequency.data[:frames], multiplier, out=increment,
                            casting='unsafe')
        else:
            np.multiply(pitch.data[:frames], math.log(2.0), out=increment,
                        casting='unsafe')
            np.exp(increment, out=increment)
            if frequency.constant:
                increment *= frequency.value
            else:
                increment *= frequency.data[:frames]

        if linear_fm.constant:
            if linear_fm.value != 0.0:
                increment += linear_fm.value
        else:
            increment += linear_fm.data[:frames]

        limit = self.sample_rate * 0.49
        np.clip(increment, -limit, limit, out=increment)
        increment /= self.sample_rate

    def _apply_level(self, buffer, level, frames):
        """Scale a rendered block by a level inlet.

        The physical models' loudness is emergent -- a strike rings to its
        table, a bow blooms to wherever the friction settles -- so they
        carry an output level rather than sending every voice through a
        vca~. A knob steps once a block like any control, so a constant
        level glides over a few blocks; a patched signal is already audio
        rate and is applied as it arrives.
        """
        if level.constant:
            target = min(2.0, max(0.0, level.value))
            start = self._level_glide
            landing = start + (target - start) * 0.35
            self._level_glide = landing
            if start == landing:
                if landing != 1.0:
                    buffer *= landing
                return
            # Ramped across the block, not stepped at its edge: a glide
            # applied as one factor per block is still a staircase.
            ramp = self._level_ramp[:frames]
            np.multiply(_INDEX_RAMP[:frames], (landing - start) / frames,
                        out=ramp)
            ramp += start
            buffer *= ramp
        else:
            buffer *= level.data[:frames]

    def _build_hertz(self, curve, frequency, pitch, frames, minimum):
        """Frequency in Hz into `curve`, scaled by the exponential pitch inlet.

        The physical models want a frequency curve rather than a phase
        increment: what they tune is a delay length, not a phase step. Same
        octave semantics as _build_increment, clamped into the band where a
        delay-line model can actually play.
        """
        if frequency.constant and pitch.constant:
            curve[:] = frequency.value * (2.0 ** pitch.value)
        elif pitch.constant:
            np.multiply(frequency.data[:frames], 2.0 ** pitch.value,
                        out=curve, casting='unsafe')
        else:
            np.multiply(pitch.data[:frames], math.log(2.0), out=curve,
                        casting='unsafe')
            np.exp(curve, out=curve)
            if frequency.constant:
                curve *= frequency.value
            else:
                curve *= frequency.data[:frames]
        np.clip(curve, minimum, self.sample_rate * 0.4, out=curve)

    def _sync_segments(self, sync, frames):
        """(end_index, reset_at_segment_start) covering the whole block.

        Splits the block at every rising edge of the sync inlet so a reset
        lands on the sample it happened on rather than at a block boundary.
        """
        if sync.constant:
            high = sync.value >= 0.5
            reset = high and self._sync_armed
            self._sync_armed = not high
            return ((frames, reset),)

        above = sync.data[:frames] >= 0.5
        edges = np.flatnonzero(above[1:] & ~above[:-1]) + 1

        starts = [(0, bool(above[0]) and self._sync_armed)]
        for edge in edges:
            starts.append((int(edge), True))

        segments = []
        for index, (begin, reset) in enumerate(starts):
            end = starts[index + 1][0] if index + 1 < len(starts) else frames
            segments.append((end, reset))

        self._sync_armed = not bool(above[-1])
        return tuple(segments)


# ----------------------------------------------------------------------------
# sig~  --  control value into the audio graph, with glide
# ----------------------------------------------------------------------------

class SigUnit(Unit):
    """Smooths a control-rate value (effort data, a fader) into a signal.

    Without slew a 60 Hz control stream stepping a VCA gain zippers audibly.
    The glide is a one-pole approach evaluated in closed form, so the ramp is
    exact rather than block-quantized.
    """

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.target = 0.0
        self.current = 0.0
        self.glide = 0.02        # seconds to reach ~63% of a step
        self.scale = 1.0
        self.offset = 0.0
        self.out = self.new_outlet()
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def render(self, frames):
        target = self.target * self.scale + self.offset
        current = self.current

        if self.glide <= 0.0 or abs(target - current) < 1.0e-7:
            self.current = target
            self.out.set_constant(target)
            return

        coefficient = math.exp(-1.0 / max(1.0, self.glide * self.sample_rate))
        ramp = self._decay_ramp(coefficient, frames, self._scratch)

        buffer = self.out.data[:frames]
        np.multiply(ramp, (current - target), out=buffer, casting='unsafe')
        buffer += target
        self.current = float(buffer[-1])
        self.out.constant = False


# ----------------------------------------------------------------------------
# formant~  --  a vowel as a bank of resonances
# ----------------------------------------------------------------------------

# The vowels, as the centre frequencies of their first three formants for an
# average adult male voice. These are the textbook averages (Peterson & Barney,
# and reproduced everywhere since); treat them as a well-placed starting point
# rather than gospel -- 'shift' moves the whole set, and the morph runs
# between them, so the exact numbers matter less than their relationships.
#
# F1 tracks how open the mouth is, F2 how far forward the tongue sits. That is
# why 'i' and 'u' both have a low F1 but sit at opposite ends of F2, and it is
# what makes a morph across this table sound like a mouth moving.
FORMANT_VOWELS = ('a', 'e', 'i', 'o', 'u')
_VOWEL_FORMANTS = {
    'a': (730.0, 1090.0, 2440.0),
    'e': (530.0, 1840.0, 2480.0),
    'i': (270.0, 2290.0, 3010.0),
    'o': (570.0, 840.0, 2410.0),
    'u': (300.0, 870.0, 2240.0),
}
# Two fixed upper resonances for air and presence. They do not identify the
# vowel, but without them the bank sounds like a filter rather than a throat.
_UPPER_FORMANTS = (3300.0, 3850.0)
# Relative weight per formant, about -6 dB a step.
_FORMANT_GAINS = (1.0, 0.5, 0.25, 0.125, 0.0625)


class FormantUnit(Unit):
    """A vowel, as five resonances in parallel.

    'vowel' runs 0..1 across a, e, i, o, u -- and runs *between* them, rather
    than switching: the formants are interpolated in the log domain, so a slow
    sweep is a mouth changing shape rather than a crossfade between two mouths.
    'shift' multiplies every formant frequency at once, which is the size of
    the head making the sound: below 1 a larger one, above 1 a smaller.

    'q' is how sharp the resonances are. Low values are a vowel-ish colour;
    high values ring, and past about 20 the bank starts to sing on its own with
    whatever the input excites. Each band is normalised for its own Q, so
    turning it up sharpens the vowel rather than simply making it louder.

    Feed it something harmonically dense -- a saw, better still a detuned
    unison stack, or noise for a whisper. A sine has nothing at the formant
    frequencies to resonate.

    Stereo when 'right in' is patched, on one set of coefficients, like vcf~.
    """

    BANDS = 5

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.right_in = self.new_inlet()
        self.vowel_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.shift_in = self.new_inlet(base=1.0, minimum=0.05)
        self.q_in = self.new_inlet(base=8.0, minimum=0.5)

        self.out = self.new_outlet()
        self.right = self.new_outlet()

        bands = FormantUnit.BANDS
        self._a1 = np.zeros(bands, dtype=np.float64)
        self._a2 = np.zeros(bands, dtype=np.float64)
        self._a3 = np.zeros(bands, dtype=np.float64)
        self._gains = np.zeros(bands, dtype=np.float64)
        self._ic1 = np.zeros(bands, dtype=np.float64)
        self._ic2 = np.zeros(bands, dtype=np.float64)
        self._ic1_right = np.zeros(bands, dtype=np.float64)
        self._ic2_right = np.zeros(bands, dtype=np.float64)

        self._x = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        # The frequencies the bank last ran with, for the node to display.
        self.frequencies = [0.0] * bands

    def reset(self):
        self._ic1[:] = 0.0
        self._ic2[:] = 0.0
        self._ic1_right[:] = 0.0
        self._ic2_right[:] = 0.0

    def _mirror(self, frames):
        if self.out.constant:
            self.right.set_constant(self.out.value)
            return
        np.copyto(self.right.data[:frames], self.out.data[:frames])
        self.right.constant = False

    def _build_coefficients(self, vowel, shift, q):
        """Where the five resonances sit for this vowel, and how sharp."""
        position = min(1.0, max(0.0, vowel)) * (len(FORMANT_VOWELS) - 1)
        lower = int(position)
        upper = min(lower + 1, len(FORMANT_VOWELS) - 1)
        blend = position - lower
        first = _VOWEL_FORMANTS[FORMANT_VOWELS[lower]]
        second = _VOWEL_FORMANTS[FORMANT_VOWELS[upper]]

        limit = self.sample_rate * 0.45
        quality = max(0.5, q)
        k = 1.0 / quality

        for band in range(FormantUnit.BANDS):
            if band < len(first):
                # Interpolated as a ratio rather than a difference: halfway
                # between 300 Hz and 2300 Hz belongs at 830, not 1300, and the
                # linear reading sweeps through vowels that are not on the way.
                frequency = math.exp(math.log(first[band]) * (1.0 - blend)
                                     + math.log(second[band]) * blend)
            else:
                frequency = _UPPER_FORMANTS[band - len(first)]
            frequency = min(limit, max(20.0, frequency * shift))
            self.frequencies[band] = frequency

            g = math.tan(math.pi * frequency / self.sample_rate)
            a1 = 1.0 / (1.0 + g * (g + k))
            self._a1[band] = a1
            self._a2[band] = g * a1
            self._a3[band] = g * g * a1
            # The bandpass tap peaks at Q, so fold k back in and each formant
            # arrives at the weight it was given whatever the sharpness.
            self._gains[band] = _FORMANT_GAINS[band] * k


    def bypass_pairs(self):
        if self.right_in.sources:
            return ((self.signal_in, self.out), (self.right_in, self.right))
        return ((self.signal_in, self.out), (self.signal_in, self.right))

    def render(self, frames):
        signal = self.signal_in.eval(frames)
        vowel = self.vowel_in.eval(frames)
        shift = self.shift_in.eval(frames)
        q = self.q_in.eval(frames)

        out = self.out
        stereo = bool(self.right_in.sources)

        if signal.constant and signal.value == 0.0 and not stereo:
            out.set_constant(0.0)
            self.right.set_constant(0.0)
            return

        self._build_coefficients(
            vowel.value if vowel.constant else float(vowel.data[frames - 1]),
            shift.value if shift.constant else float(shift.data[frames - 1]),
            q.value if q.constant else float(q.data[frames - 1]))

        if not _svf_ready.is_set():
            # Kernel still compiling, or numba missing: pass audio rather than
            # stall the callback.
            np.copyto(out.data[:frames], signal.array(frames))
            out.constant = False
            self._mirror(frames)
            return

        source = self._x[:frames]
        result = self._y[:frames]

        np.copyto(source, signal.array(frames), casting='unsafe')
        _formant_bank(source, self._a1, self._a2, self._a3, self._ic1,
                      self._ic2, self._gains, result)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False

        if not stereo:
            self._mirror(frames)
            return

        np.copyto(source, self.right_in.eval(frames).array(frames),
                  casting='unsafe')
        _formant_bank(source, self._a1, self._a2, self._a3, self._ic1_right,
                      self._ic2_right, self._gains, result)
        np.copyto(self.right.data[:frames], result, casting='unsafe')
        self.right.constant = False


# ----------------------------------------------------------------------------
# vocoder~  --  one signal's spectrum imposed on another
# ----------------------------------------------------------------------------

class VocoderUnit(Unit):
    """A filterbank analysing one signal and shaping another with it.

    The modulator is split into bands; each band drives an envelope follower;
    the carrier goes through the same bands with those envelopes as gains. What
    comes out has the carrier's pitch and the modulator's shape.

    Both banks are one numba kernel apiece, so the band count is close to free
    -- 16 bands cost about as much as a single vcf~. The expensive thing here
    is not the filtering, it is having 32 of anything at the node layer, which
    is why this is one object.

    Two details separate a vocoder that speaks from one that mumbles. The
    envelopes are followed and applied per sample, not per block, because at
    block rate the gains zipper exactly when they are moving most. And the
    bands are spaced geometrically, because that is how hearing is spaced --
    linear spacing spends most of its bands above the range where vowels are
    told apart.

    The band envelopes are readable from outside, and can be supplied from
    outside instead, which turns the bank into a spectral mapping surface for
    whatever else is to hand rather than a speech effect.
    """

    MAX_BANDS = 32

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.modulator_in = self.new_inlet()
        self.carrier_in = self.new_inlet()
        self.right_carrier_in = self.new_inlet()
        self.attack_in = self.new_inlet(base=0.002, minimum=0.0)   # seconds
        self.release_in = self.new_inlet(base=0.04, minimum=0.0)   # seconds
        self.sibilance_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0)

        self.bands = 16
        self.low = 120.0
        self.high = 8000.0
        self.q = 6.0
        self.freeze = False
        self.external = False       # take band gains from the node, not analysis

        self.out = self.new_outlet()
        self.right = self.new_outlet()

        size = VocoderUnit.MAX_BANDS
        self._a1 = np.zeros(size, dtype=np.float64)
        self._a2 = np.zeros(size, dtype=np.float64)
        self._a3 = np.zeros(size, dtype=np.float64)
        self._weights = np.zeros(size, dtype=np.float64)
        self._m_ic1 = np.zeros(size, dtype=np.float64)
        self._m_ic2 = np.zeros(size, dtype=np.float64)
        self._c_ic1 = np.zeros(size, dtype=np.float64)
        self._c_ic2 = np.zeros(size, dtype=np.float64)
        self._r_ic1 = np.zeros(size, dtype=np.float64)
        self._r_ic2 = np.zeros(size, dtype=np.float64)
        self.envelopes = np.zeros(size, dtype=np.float64)
        self.supplied = np.zeros(size, dtype=np.float64)
        self._env_block = np.zeros((size, MAX_BLOCK), dtype=np.float64)

        self._modulator = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._carrier = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._breath = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._result = np.zeros(MAX_BLOCK, dtype=np.float64)

        self._laid_out = ()

    def reset(self):
        for state in (self._m_ic1, self._m_ic2, self._c_ic1, self._c_ic2,
                      self._r_ic1, self._r_ic2):
            state[:] = 0.0
        self.envelopes[:] = 0.0

    def band_count(self):
        return max(2, min(int(self.bands), VocoderUnit.MAX_BANDS))

    def band_frequencies(self):
        count = self.band_count()
        low = max(20.0, min(self.low, self.high))
        high = min(self.sample_rate * 0.45, max(self.low, self.high))
        if high <= low:
            high = low * 2.0
        return np.geomspace(low, high, count)

    def _build_bank(self):
        """Coefficients for the current band layout, rebuilt only on change."""
        count = self.band_count()
        signature = (count, self.low, self.high, self.q, self.sample_rate)
        if signature == self._laid_out:
            return count
        self._laid_out = signature

        k = 1.0 / max(0.5, self.q)
        for band, frequency in enumerate(self.band_frequencies()):
            g = math.tan(math.pi * float(frequency) / self.sample_rate)
            a1 = 1.0 / (1.0 + g * (g + k))
            self._a1[band] = a1
            self._a2[band] = g * a1
            self._a3[band] = g * g * a1
            # The bandpass tap peaks at Q; normalise so band count and
            # sharpness do not each become a volume control.
            self._weights[band] = k
        return count

    def _mirror(self, frames):
        if self.out.constant:
            self.right.set_constant(self.out.value)
            return
        np.copyto(self.right.data[:frames], self.out.data[:frames])
        self.right.constant = False

    @staticmethod
    def _one_pole(seconds, sample_rate):
        """Per-sample coefficient for a time constant in seconds."""
        samples = max(1.0, seconds * sample_rate)
        return 1.0 - math.exp(-1.0 / samples)


    def bypass_pairs(self):
        # The carrier is the signal; the modulator only shapes it, so bypass
        # hands the carrier back untouched.
        if self.right_carrier_in.sources:
            return ((self.carrier_in, self.out),
                    (self.right_carrier_in, self.right))
        return ((self.carrier_in, self.out), (self.carrier_in, self.right))

    def render(self, frames):
        carrier = self.carrier_in.eval(frames)
        modulator = self.modulator_in.eval(frames)
        attack = self.attack_in.eval(frames)
        release = self.release_in.eval(frames)
        sibilance = self.sibilance_in.eval(frames)
        level = self.level_in.eval(frames)

        out = self.out
        stereo = bool(self.right_carrier_in.sources)
        count = self._build_bank()

        if not _svf_ready.is_set():
            np.copyto(out.data[:frames], carrier.array(frames))
            out.constant = False
            self._mirror(frames)
            return

        envelope_block = self._env_block[:count, :frames]

        if self.external:
            # Gains handed in from the patch: the same bank, told what to do
            # rather than listening for it.
            for band in range(count):
                envelope_block[band, :] = self.supplied[band]
                self.envelopes[band] = self.supplied[band]
        else:
            source = self._modulator[:frames]
            np.copyto(source, modulator.array(frames), casting='unsafe')
            attack_time = (attack.value if attack.constant
                           else float(attack.data[0]))
            release_time = (release.value if release.constant
                            else float(release.data[0]))
            _vocoder_analyse(source, self._a1[:count], self._a2[:count],
                             self._a3[:count], self._m_ic1[:count],
                             self._m_ic2[:count], self.envelopes[:count],
                             self._one_pole(attack_time, self.sample_rate),
                             self._one_pole(release_time, self.sample_rate),
                             0.0 if self.freeze else 1.0, envelope_block)

        voice = self._carrier[:frames]
        np.copyto(voice, carrier.array(frames), casting='unsafe')

        breath_amount = (sibilance.value if sibilance.constant
                         else float(sibilance.data[0]))
        breath_amount = min(1.0, max(0.0, breath_amount))
        if breath_amount > 0.0:
            breath = self._breath[:frames]
            breath[:] = np.random.random(frames) * 2.0 - 1.0
            np.multiply(breath, breath_amount, out=breath)
            np.multiply(voice, 1.0 - breath_amount, out=self._result[:frames])
            breath += self._result[:frames]
            # Noise replaces the carrier only in the top third of the range,
            # where consonants live and pitch does not.
            split = max(1, (count * 2) // 3)
        else:
            breath = voice
            split = count

        gain = level.value if level.constant else float(level.data[0])
        result = self._result[:frames]

        _vocoder_synthesise(voice, breath, split, self._a1[:count],
                            self._a2[:count], self._a3[:count],
                            self._c_ic1[:count], self._c_ic2[:count],
                            envelope_block, self._weights[:count], result)
        if gain != 1.0:
            np.multiply(result, gain, out=result)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False

        if not stereo:
            self._mirror(frames)
            return

        right_voice = self._carrier[:frames]
        np.copyto(right_voice, self.right_carrier_in.eval(frames).array(frames),
                  casting='unsafe')
        _vocoder_synthesise(right_voice, right_voice, count, self._a1[:count],
                            self._a2[:count], self._a3[:count],
                            self._r_ic1[:count], self._r_ic2[:count],
                            envelope_block, self._weights[:count], result)
        if gain != 1.0:
            np.multiply(result, gain, out=result)
        np.copyto(self.right.data[:frames], result, casting='unsafe')
        self.right.constant = False


# ----------------------------------------------------------------------------
# one_euro~  --  smoothing that gets out of the way when you move
# ----------------------------------------------------------------------------

class OneEuroUnit(Unit):
    """The one euro filter: a low-pass whose cutoff rises with speed.

    Any fixed smoothing has to choose between passing jitter and lagging
    behind a gesture, because those are the same setting. This chooses per
    sample: at rest the cutoff drops to 'min cutoff' and the signal settles
    hard, and as it starts moving the cutoff opens in proportion to its speed,
    so the lag falls away exactly when it would have been noticed. It was
    designed for interactive motion data (Casiello and Roussel, CHI 2012),
    which is what effort data is.

    Two controls, and they do separate jobs. 'min cutoff' is how still a
    resting signal is -- lower is calmer and slower to set off. 'beta' is how
    readily it gets out of the way -- raise it if fast gestures feel dragged,
    lower it if noise survives the movement.

    After ramp~, this rounds the corners between one frame's move and the
    next. On its own it is the thing to put between a jittery control stream
    and anything that will make a sound of it.
    """

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.right_in = self.new_inlet()
        self.min_cutoff_in = self.new_inlet(base=1.0, minimum=0.0001)   # Hz
        self.beta_in = self.new_inlet(base=1.0, minimum=0.0)
        self.derivative_cutoff = 1.0                                    # Hz

        self.out = self.new_outlet()
        self.right = self.new_outlet()

        # previous raw sample, smoothed speed, smoothed output, primed flag
        self._state = [0.0, 0.0, 0.0, 0.0]
        self._right_state = [0.0, 0.0, 0.0, 0.0]

        self._x = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self._state = [0.0, 0.0, 0.0, 0.0]
        self._right_state = [0.0, 0.0, 0.0, 0.0]

    def _mirror(self, frames):
        if self.out.constant:
            self.right.set_constant(self.out.value)
            return
        np.copyto(self.right.data[:frames], self.out.data[:frames])
        self.right.constant = False

    def _filter(self, signal, state, destination, period, min_cutoff,
                beta, frames):
        source = self._x[:frames]
        np.copyto(source, signal.array(frames), casting='unsafe')
        result = self._y[:frames]
        state[0], state[1], state[2], state[3] = _one_euro(
            source, result, period, min_cutoff, beta,
            self.derivative_cutoff, state[0], state[1], state[2], state[3])
        np.copyto(destination.data[:frames], result, casting='unsafe')
        destination.constant = False


    def bypass_pairs(self):
        if self.right_in.sources:
            return ((self.signal_in, self.out), (self.right_in, self.right))
        return ((self.signal_in, self.out), (self.signal_in, self.right))

    def render(self, frames):
        signal = self.signal_in.eval(frames)
        min_cutoff = self.min_cutoff_in.eval(frames)
        beta = self.beta_in.eval(frames)
        stereo = bool(self.right_in.sources)

        if not _svf_ready.is_set():
            np.copyto(self.out.data[:frames], signal.array(frames))
            self.out.constant = False
            self._mirror(frames)
            return

        period = 1.0 / self.sample_rate
        cutoff_value = max(0.0001, min_cutoff.value if min_cutoff.constant
                           else float(min_cutoff.data[0]))
        beta_value = max(0.0, beta.value if beta.constant
                         else float(beta.data[0]))

        self._filter(signal, self._state, self.out, period,
                     cutoff_value, beta_value, frames)
        if stereo:
            self._filter(self.right_in.eval(frames), self._right_state,
                         self.right, period, cutoff_value, beta_value,
                         frames)
        else:
            self._mirror(frames)


# ----------------------------------------------------------------------------
# ramp~  --  linear move to a target, arriving on schedule
# ----------------------------------------------------------------------------

class PhasorUnit(Unit):
    """A 0..1 ramp, built to drive sampler_osc~ position.

    An lfo~ set to a unipolar ramp is already a phasor, but two things make it
    awkward for scanning a sample. The rate has to be worked out by hand from
    the file's length (a 7.3 s sample needs 0.137 Hz), and there is nothing to
    tell the rest of the patch when a cycle turns over.

    So this takes a 'period' in seconds as well as a frequency: patch
    sampler_osc~'s 'length' outlet straight into it and one cycle scans the
    whole file at natural speed, whatever file is loaded. Slow the period down
    from there and you are time-stretching; freeze it and the sound holds at a
    position; run it negative and it plays backwards.

    The 'wrap' outlet emits a one-sample pulse each time the ramp turns over,
    ready for an adsr~ trigger or anything else that wants to fire per cycle.
    """

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.frequency_in = self.new_inlet(base=1.0)     # Hz
        self.period_in = self.new_inlet(base=0.0)        # seconds; >0 wins
        self.phase_in = self.new_inlet(base=0.0)         # offset, in cycles
        self.start_in = self.new_inlet(base=0.0)
        self.end_in = self.new_inlet(base=1.0)
        self.reset_in = self.new_inlet(base=0.0)

        self.phase = 0.0
        self.start_phase = 0.0
        self._reset_armed = True

        self.out = self.new_outlet()
        self.wrap = self.new_outlet()
        self._phase = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._increment = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self.phase = self.start_phase

    def render(self, frames):
        frequency = self.frequency_in.eval(frames)
        period = self.period_in.eval(frames)
        offset = self.phase_in.eval(frames)
        low = self.start_in.eval(frames)
        high = self.end_in.eval(frames)
        reset = self.reset_in.eval(frames)

        edge = reset.value if reset.constant else float(reset.data[frames - 1])
        if edge >= 0.5:
            if self._reset_armed:
                self.phase = self.start_phase
                self._reset_armed = False
        else:
            self._reset_armed = True

        increment = self._increment[:frames]
        # A period in seconds takes precedence, so 'length' can be patched
        # straight in without also clearing the frequency knob.
        period_value = period.value if period.constant else float(period.data[0])
        if period_value > 0.0:
            if period.constant:
                increment[:] = 1.0 / (period_value * self.sample_rate)
            else:
                np.copyto(increment, period.data[:frames], casting='unsafe')
                np.maximum(increment, 1.0e-6, out=increment)
                np.reciprocal(increment, out=increment)
                increment /= self.sample_rate
        elif frequency.constant:
            increment[:] = frequency.value / self.sample_rate
        else:
            np.divide(frequency.data[:frames], self.sample_rate, out=increment,
                      casting='unsafe')

        np.nan_to_num(increment, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        phase = self._phase[:frames]
        np.cumsum(increment, out=phase)
        phase += self.phase
        if offset.constant:
            if offset.value != 0.0:
                phase += offset.value
        else:
            phase += offset.data[:frames]
        np.mod(phase, 1.0, out=phase)

        previous = self.phase
        self.phase = float(phase[-1])

        # A turn-over is a step against the direction of travel.
        wrap = self.wrap
        wrap_buffer = wrap.data[:frames]
        wrap_buffer[:] = 0.0
        forward = increment[0] >= 0.0
        steps = np.diff(phase)
        if forward:
            indices = np.flatnonzero(steps < -0.5) + 1
            first = (phase[0] - previous) < -0.5
        else:
            indices = np.flatnonzero(steps > 0.5) + 1
            first = (phase[0] - previous) > 0.5
        if indices.size:
            wrap_buffer[indices] = 1.0
        if first:
            wrap_buffer[0] = 1.0
        wrap.constant = False

        out = self.out
        buffer = out.data[:frames]
        low_value = low.value if low.constant else low.data[:frames]
        high_value = high.value if high.constant else high.data[:frames]
        np.multiply(phase, (high_value - low_value), out=buffer,
                    casting='unsafe')
        buffer += low_value
        out.constant = False


class RampUnit(Unit):
    """Straight-line ramp to a target, reaching it exactly at the end of time.

    sig~ smooths a control stream with a one-pole glide, which approaches its
    target asymptotically -- the right thing for de-zippering, where when it
    arrives does not matter. This is the other kind of move: a straight line
    that lands on schedule, for when the move itself is the gesture. A
    portamento, a sweep timed to a beat, a fade of a known length.

    A new target starts a new ramp from wherever the output currently sits, so
    re-aiming mid-move never steps. The ramp time is read when the move
    starts, so changing it mid-flight affects the next move, not this one.
    """

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.target_in = self.new_inlet()
        self.time_in = self.new_inlet(base=0.1, minimum=0.0)
        self.trigger_in = self.new_inlet()

        self.threshold = 0.5
        self.current = 0.0
        self.arrive_count = 0     # bumped when a move lands; the node bangs

        # With auto_time set, the move takes as long as the gap between the
        # values arriving, which the node measures and writes here. The audio
        # thread only reads it, and a float assignment lands whole, so no
        # handshake is needed for it.
        self.auto_time = False
        self.measured_time = 0.0

        self._goal = 0.0
        self._remaining = 0       # samples left to run
        self._increment = 0.0
        self._trigger_armed = True

        # Main-thread requests, served on the next block. Counters rather than
        # flags for the usual reason: a read-modify-write shared by two threads
        # can lose one.
        self._jump_requests = 0
        self._jump_served = 0
        self._restart_requests = 0
        self._restart_served = 0

        self.out = self.new_outlet()
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self._remaining = 0

    def jump(self):
        """Node layer: snap to the target now, without ramping."""
        self._jump_requests += 1

    def restart(self):
        """Node layer: run the move again from where the output sits."""
        self._restart_requests += 1

    def _begin(self, goal, seconds):
        self._goal = goal
        samples = int(seconds * self.sample_rate)
        if samples < 1:
            self.current = goal
            self._remaining = 0
            self.arrive_count += 1
            return
        self._remaining = samples
        self._increment = (goal - self.current) / samples


    def bypass_pairs(self):
        # A ramp is a slew on its target, not a source of its own, so leaving
        # it alone means handing the target back unsmoothed -- the step it
        # would have ramped to. Silence would be the wrong answer: there is an
        # input here, it is simply a target rather than a waveform.
        return ((self.target_in, self.out),)

    def render(self, frames):
        target = self.target_in.eval(frames)
        time_in = self.time_in.eval(frames)
        trigger = self.trigger_in.eval(frames)

        goal = target.value if target.constant else float(target.data[frames - 1])
        if self.auto_time and self.measured_time > 0.0:
            seconds = self.measured_time
        else:
            # Also the fallback before enough values have arrived to time the
            # stream, which is why the manual setting still matters in auto.
            seconds = time_in.value if time_in.constant else float(time_in.data[0])

        if self._jump_requests != self._jump_served:
            self._jump_served = self._jump_requests
            self._goal = goal
            self.current = goal
            self._remaining = 0

        high = (trigger.value if trigger.constant
                else float(trigger.data[frames - 1])) >= self.threshold
        fire = False
        if high:
            if self._trigger_armed:
                fire = True
            self._trigger_armed = False
        else:
            self._trigger_armed = True

        if self._restart_requests != self._restart_served:
            self._restart_served = self._restart_requests
            fire = True

        if fire or goal != self._goal:
            self._begin(goal, seconds)

        out = self.out
        if self._remaining <= 0:
            out.set_constant(self.current)
            return

        buffer = out.data[:frames]
        count = min(frames, self._remaining)
        segment = self._scratch[:count]
        np.multiply(_INDEX_RAMP[:count], self._increment, out=segment)
        segment += self.current
        np.copyto(buffer[:count], segment, casting='unsafe')

        self.current = float(segment[-1])
        self._remaining -= count
        if self._remaining <= 0:
            # Land on the goal itself rather than wherever the accumulated
            # increments happened to arrive.
            self.current = self._goal
            buffer[count - 1] = self._goal
            self.arrive_count += 1
        if count < frames:
            buffer[count:] = self.current
        out.constant = False


# ----------------------------------------------------------------------------
# vca~
# ----------------------------------------------------------------------------

class VcaUnit(Unit):
    """Voltage controlled amplifier: signal in, gain in, signal out.

    Stereo when, and only when, something is patched to the right inlet: the
    gain curve is computed once and applied to both channels, so the two can
    never drift out of step the way a pair of separate vca~ can. Left alone,
    the right inlet costs nothing and the right outlet simply carries the same
    signal as the left, so a mono patch neither pays for stereo nor has to
    know about it.
    """

    LINEAR = 0
    EXPONENTIAL = 1

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.right_in = self.new_inlet()
        self.gain_in = self.new_inlet(base=1.0)
        self.response = VcaUnit.LINEAR
        self.out = self.new_outlet()
        self.right = self.new_outlet()
        self._gain = np.zeros(MAX_BLOCK, dtype=np.float32)

    def _mirror(self, frames):
        """Right carries the left channel when nothing is patched to it."""
        if self.out.constant:
            self.right.set_constant(self.out.value)
            return
        np.copyto(self.right.data[:frames], self.out.data[:frames])
        self.right.constant = False

    @staticmethod
    def _apply_constant(signal, out, gain_value, frames):
        if signal.constant:
            out.set_constant(signal.value * gain_value)
            return
        np.multiply(signal.data[:frames], gain_value, out=out.data[:frames])
        out.constant = False

    @staticmethod
    def _apply_curve(signal, out, curve, frames):
        np.multiply(signal.array(frames), curve, out=out.data[:frames])
        out.constant = False


    def bypass_pairs(self):
        if self.right_in.sources:
            return ((self.signal_in, self.out), (self.right_in, self.right))
        return ((self.signal_in, self.out), (self.signal_in, self.right))

    def render(self, frames):
        signal = self.signal_in.eval(frames)
        gain = self.gain_in.eval(frames)
        stereo = bool(self.right_in.sources)

        if gain.constant:
            gain_value = max(0.0, gain.value)
            if self.response == VcaUnit.EXPONENTIAL:
                gain_value = gain_value ** 3
            self._apply_constant(signal, self.out, gain_value, frames)
            if stereo:
                self._apply_constant(self.right_in.eval(frames), self.right,
                                     gain_value, frames)
            else:
                self._mirror(frames)
            return

        # Copy the gain curve before shaping so we do not scribble on the
        # inlet's buffer, which other consumers of the same cord still need.
        curve = self._gain[:frames]
        np.copyto(curve, gain.data[:frames])
        np.clip(curve, 0.0, None, out=curve)
        if self.response == VcaUnit.EXPONENTIAL:
            # Perceptually closer to an exponential VCA than a raw multiply,
            # and far cheaper than a real dB conversion.
            curve *= curve * curve

        self._apply_curve(signal, self.out, curve, frames)
        if stereo:
            self._apply_curve(self.right_in.eval(frames), self.right, curve,
                              frames)
        else:
            self._mirror(frames)


# ----------------------------------------------------------------------------
# adsr~
# ----------------------------------------------------------------------------

class AdsrUnit(Unit):
    """Audio-rate ADSR driven by a gate.

    The gate is a signal, so it can come from a control value through sig~, an
    LFO, a comparator, or another envelope. Crossings of the gate threshold are
    located within the block and the envelope is rendered in segments between
    them, so note-on timing is sample-accurate rather than block-quantized.

    Each stage is an exponential approach to an overshoot target -- the way an
    analog RC envelope actually behaves -- generated in closed form. Stage
    times are sampled once per segment, so A/D/S/R modulate at control rate but
    not audio rate; that keeps the coefficient math out of the inner loop.
    """

    IDLE, ATTACK, DECAY, SUSTAIN, RELEASE = 0, 1, 2, 3, 4

    # Curvature of each segment, as the exponent of a normalized 0..1 shape.
    # Small is nearly straight, large is a sharp knee. These match the feel of
    # the overshoot constants they replace (0.3 and 0.0001 respectively).
    ATTACK_CURVE = 1.466
    DECAY_CURVE = 9.21

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.gate_in = self.new_inlet()
        self.trigger_in = self.new_inlet()
        self.attack_in = self.new_inlet(base=0.01, minimum=0.0)
        self.decay_in = self.new_inlet(base=0.1, minimum=0.0)
        self.sustain_in = self.new_inlet(base=0.7, minimum=0.0, maximum=1.0)
        self.release_in = self.new_inlet(base=0.3, minimum=0.0)

        self.threshold = 0.5
        self.retrigger = True     # re-gate restarts attack from current level
        self.legato = False       # re-gate while sounding is ignored

        self.stage = AdsrUnit.IDLE
        self.level = 0.0
        self.gate_open = False
        self.finish_count = 0     # bumped when a tail reaches silence

        # Where the current stage started, and how far through it we are.
        # Progress is what makes stage times exact: it advances at a fixed
        # rate and the stage ends when it reaches 1, so a segment spanning
        # several blocks still finishes on schedule.
        self._stage_level = 0.0
        self._stage_progress = 0.0

        # One-shot state. A fired envelope runs attack -> decay -> release
        # without waiting for anything to let go of it.
        self._one_shot = False
        self._trigger_armed = True
        # Main thread requests a shot by bumping a counter the audio thread
        # consumes. A single int increment orders correctly under the GIL,
        # where setting stage and the flag separately could interleave.
        self._fire_requests = 0
        self._fire_served = 0

        self.out = self.new_outlet()
        self._ramp = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._segment = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self.stage = AdsrUnit.IDLE
        self.level = 0.0
        self.gate_open = False
        self._stage_level = 0.0
        self._stage_progress = 0.0

    def _enter_stage(self, stage):
        """Begin a stage from wherever the envelope currently sits."""
        self.stage = stage
        self._stage_level = self.level
        self._stage_progress = 0.0

    def trigger(self, gate_open):
        """Direct gate from the node layer, bypassing the signal inlet."""
        self._set_gate(bool(gate_open))

    def fire(self):
        """Request one shot from the node layer. Served on the next block."""
        self._fire_requests += 1

    def _begin_one_shot(self):
        self._one_shot = True
        self._enter_stage(AdsrUnit.ATTACK)

    def _set_gate(self, open_now):
        if open_now == self.gate_open:
            return
        self.gate_open = open_now
        if open_now:
            # A held gate takes command back from a one-shot in flight.
            self._one_shot = False
            if self.legato and self.stage in (AdsrUnit.ATTACK, AdsrUnit.DECAY,
                                              AdsrUnit.SUSTAIN):
                return
            if not self.retrigger and self.stage != AdsrUnit.IDLE:
                return
            self._enter_stage(AdsrUnit.ATTACK)
        elif self.stage != AdsrUnit.IDLE:
            self._enter_stage(AdsrUnit.RELEASE)

    def _trigger_edges(self, trigger, frames):
        """Rising edges of the trigger inlet inside this block."""
        above = trigger.data[:frames] >= self.threshold
        rising = np.flatnonzero(above[1:] & ~above[:-1]) + 1
        return rising

    def _coefficient(self, seconds, ratio):
        samples = max(1.0, seconds * self.sample_rate)
        return math.exp(-math.log((1.0 + ratio) / ratio) / samples)

    def _gate_events(self, gate, frames):
        """(index, state) for every gate transition inside this block."""
        values = gate.data[:frames]
        above = values >= self.threshold
        changes = np.flatnonzero(above[1:] != above[:-1]) + 1
        if changes.size == 0:
            return ()
        return tuple((int(index), bool(above[index])) for index in changes)

    def render(self, frames):
        gate = self.gate_in.eval(frames)
        attack = self.attack_in.eval(frames)
        decay = self.decay_in.eval(frames)
        sustain = self.sustain_in.eval(frames)
        release = self.release_in.eval(frames)

        attack_time = attack.value if attack.constant else float(attack.data[0])
        decay_time = decay.value if decay.constant else float(decay.data[0])
        sustain_level = sustain.value if sustain.constant else float(sustain.data[0])
        release_time = release.value if release.constant else float(release.data[0])

        # Serve any shots requested from the node layer since the last block.
        if self._fire_requests != self._fire_served:
            self._fire_served = self._fire_requests
            self._begin_one_shot()

        # Events are (index, is_gate, value), merged so a gate change and a
        # trigger edge in the same block are both applied at their own sample.
        events = []

        # A constant gate inlet drives the envelope too, so a plain value or a
        # sig~ works as a gate exactly as a patched square wave would.
        if gate.constant:
            self._set_gate(gate.value >= self.threshold)
        else:
            self._set_gate(bool(gate.data[0] >= self.threshold))
            for index, state in self._gate_events(gate, frames):
                events.append((index, True, state))

        trigger = self.trigger_in.eval(frames)
        if trigger.constant:
            high = trigger.value >= self.threshold
            if high and self._trigger_armed:
                self._begin_one_shot()
            self._trigger_armed = not high
        else:
            if bool(trigger.data[0] >= self.threshold) and self._trigger_armed:
                self._begin_one_shot()
            for index in self._trigger_edges(trigger, frames):
                events.append((int(index), False, True))
            self._trigger_armed = not bool(
                trigger.data[frames - 1] >= self.threshold)

        if len(events) > 1:
            events.sort(key=lambda item: item[0])

        out = self.out
        buffer = out.data[:frames]
        was_active = self.stage != AdsrUnit.IDLE

        position = 0
        event_index = 0
        # Each pass either advances `position` or changes stage; both are
        # bounded, and the guard keeps a pathological parameter set from
        # spinning on the audio thread.
        for _ in range(len(events) + 16):
            if position >= frames:
                break
            while event_index < len(events) and events[event_index][0] <= position:
                _, is_gate, value = events[event_index]
                if is_gate:
                    self._set_gate(value)
                else:
                    self._begin_one_shot()
                event_index += 1
            if event_index < len(events):
                segment_end = events[event_index][0]
            else:
                segment_end = frames
            position = self._render_segment(
                buffer, position, segment_end,
                attack_time, decay_time, sustain_level, release_time)
        if position < frames:
            buffer[position:] = self.level

        if self.stage == AdsrUnit.IDLE:
            self._one_shot = False
            if was_active:
                self.finish_count += 1
        out.constant = False

    def _render_segment(self, buffer, start, end, attack_time, decay_time,
                        sustain_level, release_time):
        """Fill buffer[start:end], returning where rendering actually reached.

        A stage transition inside the segment stops the fill at the crossing
        and returns there, so the caller re-enters with the new stage and the
        next curve begins on the correct sample rather than the next block.
        """
        length = end - start
        if length <= 0:
            return end

        stage = self.stage

        if stage == AdsrUnit.IDLE:
            buffer[start:end] = 0.0
            self.level = 0.0
            return end

        if stage == AdsrUnit.SUSTAIN:
            buffer[start:end] = sustain_level
            self.level = sustain_level
            return end

        # A one-shot does not wait to be let go: decay runs straight on into
        # release, so attack/decay/release shape the whole hit.
        after_decay = AdsrUnit.RELEASE if self._one_shot else AdsrUnit.SUSTAIN

        if stage == AdsrUnit.ATTACK:
            duration, goal, curve = attack_time, 1.0, AdsrUnit.ATTACK_CURVE
            next_stage = AdsrUnit.DECAY
        elif stage == AdsrUnit.DECAY:
            duration, goal, curve = decay_time, sustain_level, AdsrUnit.DECAY_CURVE
            next_stage = after_decay
        else:  # RELEASE
            duration, goal, curve = release_time, 0.0, AdsrUnit.DECAY_CURVE
            next_stage = AdsrUnit.IDLE

        if duration <= 0.0:
            self.level = goal
            self._enter_stage(next_stage)
            return start

        # Progress advances at a fixed rate and the stage ends when it reaches
        # 1, so the stage lasts exactly `duration` no matter how many blocks it
        # is split across or what level it started from.
        step = 1.0 / (duration * self.sample_rate)
        progress = self._stage_progress
        to_finish = (1.0 - progress) / step
        count = min(length, max(1, int(math.ceil(to_finish))))

        phase = self._ramp[:count]
        np.multiply(_INDEX_RAMP[:count], step, out=phase)
        phase += progress
        np.clip(phase, 0.0, 1.0, out=phase)

        # Normalized exponential shape: 0 at the start, exactly 1 at the end.
        segment = self._segment[:count]
        np.multiply(phase, -curve, out=segment)
        np.exp(segment, out=segment)
        segment -= 1.0
        segment *= (goal - self._stage_level) / (math.exp(-curve) - 1.0)
        segment += self._stage_level

        buffer[start:start + count] = segment
        self.level = float(segment[-1])
        self._stage_progress = min(1.0, progress + count * step)

        if self._stage_progress >= 1.0:
            self.level = goal
            self._enter_stage(next_stage)

        return start + count


# ----------------------------------------------------------------------------
# lfo~
# ----------------------------------------------------------------------------

LFO_SHAPES = ('sine', 'triangle', 'saw', 'ramp', 'square', 'sample_hold',
              'smooth_random', 'noise')


class LfoUnit(Unit):
    """Low frequency oscillator: the VCO engine without band limiting, plus
    the random shapes that make modulation interesting.

    Runs at audio rate like everything else, so it can be pushed into the audio
    range for sidebands or down to a fraction of a Hz for slow drift.
    """

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.rate_in = self.new_inlet(base=1.0)
        self.depth_in = self.new_inlet(base=1.0)
        self.offset_in = self.new_inlet(base=0.0)
        self.width_in = self.new_inlet(base=0.5, minimum=0.01, maximum=0.99)
        self.reset_in = self.new_inlet()

        self.shape = 'sine'
        self.bipolar = True
        self.start_phase = 0.0

        self.phase = 0.0
        self._held = 0.0
        self._previous_held = 0.0
        self._reset_armed = True

        self.out = self.new_outlet()
        self._phase = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._work = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self.phase = self.start_phase
        self._held = 0.0
        self._previous_held = 0.0

    def render(self, frames):
        rate = self.rate_in.eval(frames)
        depth = self.depth_in.eval(frames)
        offset = self.offset_in.eval(frames)
        width = self.width_in.eval(frames)
        reset = self.reset_in.eval(frames)

        # A rising edge on the reset inlet restarts the cycle.
        reset_value = reset.value if reset.constant else float(reset.data[frames - 1])
        if reset_value >= 0.5:
            if self._reset_armed:
                self.phase = self.start_phase
                self._reset_armed = False
        else:
            self._reset_armed = True

        out = self.out
        buffer = out.data[:frames]

        if self.shape == 'noise':
            buffer[:] = np.random.random(frames) * 2.0 - 1.0
        else:
            phase = self._advance_phase(rate, frames)
            np.copyto(buffer, self._shape_values(phase, width, frames),
                      casting='unsafe')

        if not self.bipolar:
            buffer += 1.0
            buffer *= 0.5

        if depth.constant:
            if depth.value != 1.0:
                buffer *= depth.value
        else:
            buffer *= depth.data[:frames]

        if offset.constant:
            if offset.value != 0.0:
                buffer += offset.value
        else:
            buffer += offset.data[:frames]

        out.constant = False

    def _advance_phase(self, rate, frames):
        """Accumulate phase across the block, allowing audio-rate rate CV."""
        phase = self._phase[:frames]
        if rate.constant:
            np.multiply(_INDEX_RAMP[:frames], rate.value / self.sample_rate,
                        out=phase)
        else:
            np.divide(rate.data[:frames], self.sample_rate, out=phase,
                      casting='unsafe')
            np.cumsum(phase, out=phase)
        phase += self.phase
        np.mod(phase, 1.0, out=phase)
        self.phase = float(phase[-1])
        return phase

    def _shape_values(self, phase, width, frames):
        work = self._work[:frames]
        shape = self.shape

        if shape == 'sine':
            np.multiply(phase, 2.0 * math.pi, out=work)
            np.sin(work, out=work)
            return work
        if shape == 'triangle':
            np.subtract(phase, 0.5, out=work)
            np.abs(work, out=work)
            work *= 4.0
            work -= 1.0
            return work
        if shape == 'saw':
            np.multiply(phase, -2.0, out=work)
            work += 1.0
            return work
        if shape == 'ramp':
            np.multiply(phase, 2.0, out=work)
            work -= 1.0
            return work
        if shape == 'square':
            duty = width.value if width.constant else float(width.data[0])
            np.copyto(work, np.where(phase < duty, 1.0, -1.0))
            return work

        # Random shapes step once per cycle; find the wrap inside the block.
        wrapped = np.flatnonzero(np.diff(phase) < 0.0)

        if shape == 'sample_hold':
            start = 0
            for index in wrapped:
                cut = int(index) + 1
                work[start:cut] = self._held
                self._held = np.random.random() * 2.0 - 1.0
                start = cut
            work[start:] = self._held
            return work

        # smooth_random: ramp between successive random targets across a cycle.
        start = 0
        for index in wrapped:
            cut = int(index) + 1
            work[start:cut] = (self._previous_held
                               + (self._held - self._previous_held) * phase[start:cut])
            self._previous_held = self._held
            self._held = np.random.random() * 2.0 - 1.0
            start = cut
        work[start:] = (self._previous_held
                        + (self._held - self._previous_held) * phase[start:])
        return work


# ----------------------------------------------------------------------------
# clock~
# ----------------------------------------------------------------------------

class ClockUnit(Unit):
    """Pulse train for the audio graph, tick count for the node layer.

    The signal outlet is a 0/1 gate whose rising edge lands on the sample the
    phase wraps, so an adsr~ trigger patched from here is as tight as the
    sample rate allows. The node layer cannot see individual samples, so ticks
    are also counted into `tick_count`; the node compares it once a frame and
    sends that many bangs. Nothing is dropped when the clock runs faster than
    the GUI -- a 20 Hz clock still delivers 20 bangs a second, arriving in
    bursts of one or two per frame.

    Rate is an ordinary inlet, so it can be swept by an envelope or an LFO, and
    since phase accumulates per sample the sweep is continuous rather than
    stepped at block boundaries.
    """

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.rate_in = self.new_inlet(base=2.0)
        self.width_in = self.new_inlet(base=0.5, minimum=0.001, maximum=0.999)
        self.reset_in = self.new_inlet()

        self.running = False
        self.start_phase = 0.0

        self.phase = 0.0
        self.tick_count = 0       # written by the audio thread, read by the node
        self._reset_armed = True

        # The node asks for a downbeat by bumping a counter rather than by
        # touching phase and tick_count itself: `tick_count += 1` on two threads
        # can lose an increment between its load and its store.
        self._restart_requests = 0
        self._restart_served = 0

        self.out = self.new_outlet()
        self._phase = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self.phase = self.start_phase

    def restart(self):
        """Node layer: put the clock on the downbeat, ticking immediately."""
        self._restart_requests += 1

    def _downbeat(self):
        self.phase = self.start_phase
        self.tick_count += 1

    def render(self, frames):
        if self._restart_requests != self._restart_served:
            self._restart_served = self._restart_requests
            self._downbeat()

        if not self.running:
            # Phase is held rather than cleared, so a clock that is stopped and
            # started again without a reset carries on where it left off.
            self.out.set_constant(0.0)
            return

        rate = self.rate_in.eval(frames)
        width = self.width_in.eval(frames)
        reset = self.reset_in.eval(frames)

        reset_value = reset.value if reset.constant else float(reset.data[frames - 1])
        if reset_value >= 0.5:
            if self._reset_armed:
                self._downbeat()
                self._reset_armed = False
        else:
            self._reset_armed = True

        phase = self._phase[:frames]
        if rate.constant:
            np.multiply(_INDEX_RAMP[:frames], rate.value / self.sample_rate,
                        out=phase)
        else:
            np.divide(rate.data[:frames], self.sample_rate, out=phase,
                      casting='unsafe')
            np.cumsum(phase, out=phase)
        phase += self.phase

        # Stored phase is always in [0, 1), so the integer part of where the
        # block ends is exactly how many cycles it crossed. Counting this way
        # stays correct at rates fast enough to wrap several times in one
        # block, where finding edges in the output would give the same answer
        # for more work -- and it counts backwards cycles under a negative rate.
        cycles = int(math.floor(float(phase[-1])))
        if cycles:
            self.tick_count += abs(cycles)

        np.mod(phase, 1.0, out=phase)
        self.phase = float(phase[-1])

        duty = width.value if width.constant else float(width.data[0])
        buffer = self.out.data[:frames]
        np.copyto(buffer, np.where(phase < duty, 1.0, 0.0), casting='unsafe')
        self.out.constant = False


# ----------------------------------------------------------------------------
# vco~
# ----------------------------------------------------------------------------

VCO_SHAPES = ('saw', 'square', 'triangle', 'sine', 'noise', 'pink')

# Paul Kellet's pink noise filter, expressed as six parallel one-poles so it
# can run through lfilter instead of a per-sample Python loop.
_PINK_POLES = (0.99886, 0.99332, 0.96900, 0.86650, 0.55000, -0.7616)
_PINK_GAINS = (0.0555179, 0.0750759, 0.1538520, 0.3104856, 0.5329522, -0.0168980)


def _polyblep(phase, increment, out):
    """Band-limited step correction for saw and pulse discontinuities."""
    out[:] = 0.0
    rising = phase < increment
    if np.any(rising):
        t = phase[rising] / increment[rising]
        out[rising] = t + t - t * t - 1.0
    falling = phase > (1.0 - increment)
    if np.any(falling):
        t = (phase[falling] - 1.0) / increment[falling]
        out[falling] = t * t + t + t + 1.0
    return out


class VcoUnit(Unit):
    """Voltage controlled oscillator with PolyBLEP antialiasing.

    Pitch is a base frequency in Hz multiplied by an exponential inlet in
    octaves (1.0 = up an octave, matching a 1V/oct input), with a separate
    linear FM inlet in Hz for clangorous/bell tones. Hard sync resets phase on
    a rising edge of the sync inlet; the block is split at sync events so the
    reset lands on the right sample.

    Raise `voices` and it becomes a unison stack: several oscillators detuned
    symmetrically about the written pitch and spread across the stereo field.
    They share one modulation section -- the increment is built once and
    scaled per voice -- which is about a quarter cheaper than the same stack
    patched as separate oscillators through pans and a mixer. The rest is the
    band-limited shape rendering, which is a fixed cost per voice.

    At one voice this is the plain oscillator it always was, on the same code
    path, and the right outlet simply carries the same signal.
    """

    MAX_VOICES = 8
    # How fast a voice's drift wanders, as a one-pole coefficient per block.
    # 0.01 at 86 blocks/sec is a time constant of about a second: slow enough
    # to be movement rather than vibrato.
    DRIFT_RATE = 0.01

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.frequency_in = self.new_inlet(base=110.0)
        self.pitch_in = self.new_inlet(base=0.0)          # octaves
        self.linear_fm_in = self.new_inlet(base=0.0)      # Hz
        self.width_in = self.new_inlet(base=0.5, minimum=0.01, maximum=0.99)
        self.phase_mod_in = self.new_inlet(base=0.0)      # cycles
        self.sync_in = self.new_inlet(base=0.0)
        self.detune_in = self.new_inlet(base=10.0, minimum=0.0)   # cents

        self.shape = 'saw'
        self.phase = 0.0
        self.start_phase = 0.0
        self._sync_armed = True
        self._pink_state = [np.zeros(1) for _ in _PINK_POLES]
        self._pink_last = 0.0

        self.voices = 1
        self.spread = 0.0          # 0 = all voices centred
        self.drift = 0.0           # cents of slow per-voice wander
        self._phases = [0.0] * VcoUnit.MAX_VOICES
        self._drift_state = [0.0] * VcoUnit.MAX_VOICES
        self._offsets = [0.0]
        self._laid_out = 0         # voice count self._offsets was built for
        self._started = 0          # voices that have been given a start phase

        self.out = self.new_outlet()
        self.right = self.new_outlet()
        self._phase = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._increment = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._work = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._blep = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._voice_increment = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._voice = np.zeros(MAX_BLOCK, dtype=np.float32)
        self._scaled = np.zeros(MAX_BLOCK, dtype=np.float32)

    def reset(self):
        self.phase = self.start_phase
        self._started = 0
        self._spread_phases(max(1, min(self.voices, VcoUnit.MAX_VOICES)))

    # -- unison layout -------------------------------------------------------

    def _spread_phases(self, count):
        """Give each voice its own starting point around the cycle.

        Voices all starting together sum coherently for the first moments of a
        note, which reads as a phasey attack before the detuning pulls them
        apart. Only voices that have not run yet are placed, so raising the
        voice count mid-note does not restart the ones already sounding.
        """
        for index in range(self._started, count):
            self._phases[index] = (self.start_phase + index / count) % 1.0
        self._started = max(self._started, count)

    def _voice_offsets(self, count):
        """Detune positions, -1..1, symmetric about the written pitch.

        An odd voice count leaves the middle voice exactly in tune, so the
        note keeps a stable centre however wide the detuning goes.
        """
        if self._laid_out != count:
            if count < 2:
                self._offsets = [0.0]
            else:
                self._offsets = [(2.0 * i / (count - 1)) - 1.0
                                 for i in range(count)]
            self._laid_out = count
        return self._offsets

    def _advance_drift(self, index):
        """One slow random wander per voice, in cents."""
        if self.drift <= 0.0:
            return 0.0
        target = np.random.random() * 2.0 - 1.0
        state = self._drift_state[index]
        state += (target - state) * VcoUnit.DRIFT_RATE
        self._drift_state[index] = state
        return state * self.drift

    # -- rendering -----------------------------------------------------------

    def render(self, frames):
        out = self.out
        right_out = self.right
        buffer = out.data[:frames]

        if self.shape == 'noise':
            buffer[:] = np.random.random(frames) * 2.0 - 1.0
            np.copyto(right_out.data[:frames], buffer)
            out.constant = False
            right_out.constant = False
            return
        if self.shape == 'pink':
            self._render_pink(buffer, frames)
            np.copyto(right_out.data[:frames], buffer)
            out.constant = False
            right_out.constant = False
            return

        frequency = self.frequency_in.eval(frames)
        pitch = self.pitch_in.eval(frames)
        linear_fm = self.linear_fm_in.eval(frames)
        width = self.width_in.eval(frames)
        phase_mod = self.phase_mod_in.eval(frames)
        sync = self.sync_in.eval(frames)

        increment = self._increment[:frames]
        self._build_increment(increment, frequency, pitch, linear_fm, frames)

        count = max(1, min(int(self.voices), VcoUnit.MAX_VOICES))
        if count > 1:
            self._render_unison(buffer, right_out.data[:frames], increment,
                                width, phase_mod, sync, count, frames)
            out.constant = False
            right_out.constant = False
            return

        phase = self._phase[:frames]
        start = 0
        for segment_end, do_reset in self._sync_segments(sync, frames):
            if do_reset:
                self.phase = self.start_phase
            if segment_end > start:
                view = phase[start:segment_end]
                np.cumsum(increment[start:segment_end], out=view)
                view += self.phase
                np.mod(view, 1.0, out=view)
                self.phase = float(view[-1])
            start = segment_end

        if not phase_mod.constant:
            phase = phase + phase_mod.data[:frames]
            np.mod(phase, 1.0, out=phase)
        elif phase_mod.value != 0.0:
            phase = (phase + phase_mod.value) % 1.0

        self._render_shape(buffer, phase, increment, width, frames)
        np.copyto(right_out.data[:frames], buffer)
        out.constant = False
        right_out.constant = False

    def _render_unison(self, left, right, increment, width, phase_mod, sync,
                       count, frames):
        detune = self.detune_in.eval(frames)
        cents = detune.value if detune.constant else float(detune.data[0])

        self._spread_phases(count)
        offsets = self._voice_offsets(count)
        segments = list(self._sync_segments(sync, frames))
        spread = min(1.0, max(0.0, self.spread))
        mono = spread <= 0.0

        # Voices drift in and out of phase with each other, so the sum grows
        # as the square root of their number rather than in proportion to it.
        gain = 1.0 / math.sqrt(count)

        left[:] = 0.0
        if not mono:
            right[:] = 0.0

        voice_increment = self._voice_increment[:frames]
        voice = self._voice[:frames]
        scaled = self._scaled[:frames]
        phase = self._phase[:frames]

        for index in range(count):
            ratio = 2.0 ** ((cents * offsets[index]
                             + self._advance_drift(index)) / 1200.0)
            np.multiply(increment, ratio, out=voice_increment)

            start = 0
            for segment_end, do_reset in segments:
                if do_reset:
                    # Sync restarts the whole stack together, keeping the
                    # voices' phase relationship rather than collapsing it.
                    self._phases[index] = (self.start_phase
                                           + index / count) % 1.0
                if segment_end > start:
                    view = phase[start:segment_end]
                    np.cumsum(voice_increment[start:segment_end], out=view)
                    view += self._phases[index]
                    np.mod(view, 1.0, out=view)
                    self._phases[index] = float(view[-1])
                start = segment_end

            if not phase_mod.constant:
                np.add(phase, phase_mod.data[:frames], out=phase)
                np.mod(phase, 1.0, out=phase)
            elif phase_mod.value != 0.0:
                phase += phase_mod.value
                np.mod(phase, 1.0, out=phase)

            self._render_shape(voice, phase, voice_increment, width, frames)

            if mono:
                np.multiply(voice, gain, out=scaled)
                left += scaled
                continue

            # Equal power across the field, matching pan~, so a voice does not
            # dip in level as it travels.
            angle = (offsets[index] * spread + 1.0) * 0.25 * math.pi
            np.multiply(voice, gain * math.cos(angle), out=scaled)
            left += scaled
            np.multiply(voice, gain * math.sin(angle), out=scaled)
            right += scaled

        if mono:
            np.copyto(right, left)

    def _render_shape(self, buffer, phase, increment, width, frames):
        work = self._work[:frames]
        blep = self._blep[:frames]
        shape = self.shape
        magnitude = np.abs(increment)

        if shape == 'sine':
            np.multiply(phase, 2.0 * math.pi, out=work)
            np.sin(work, out=work)
            np.copyto(buffer, work, casting='unsafe')
            return

        if shape == 'triangle':
            # Naive triangle: harmonics fall off at 12 dB/octave, so aliasing
            # sits far enough down to leave unbandlimited.
            np.subtract(phase, 0.5, out=work)
            np.abs(work, out=work)
            work *= 4.0
            work -= 1.0
            np.copyto(buffer, work, casting='unsafe')
            return

        if shape == 'saw':
            np.multiply(phase, 2.0, out=work)
            work -= 1.0
            _polyblep(phase, magnitude, blep)
            work -= blep
            np.copyto(buffer, work, casting='unsafe')
            return

        # square / pulse: two band-limited steps, one at each edge.
        duty = width.value if width.constant else float(width.data[0])
        np.copyto(work, np.where(phase < duty, 1.0, -1.0))
        _polyblep(phase, magnitude, blep)
        work += blep
        _polyblep((phase - duty) % 1.0, magnitude, blep)
        work -= blep
        np.copyto(buffer, work, casting='unsafe')

    def _render_pink(self, buffer, frames):
        white = np.random.random(frames) * 2.0 - 1.0
        if scipy_signal is None:
            np.copyto(buffer, white, casting='unsafe')
            return
        total = np.zeros(frames, dtype=np.float64)
        for index, (pole, gain) in enumerate(zip(_PINK_POLES, _PINK_GAINS)):
            state = self._pink_state[index]
            filtered, state = scipy_signal.lfilter([gain], [1.0, -pole], white,
                                                   zi=state)
            self._pink_state[index] = state
            total += filtered
        # The b6 term is simply the previous white sample, scaled.
        delayed = np.empty(frames, dtype=np.float64)
        delayed[0] = self._pink_last
        delayed[1:] = white[:-1]
        self._pink_last = float(white[-1])
        total += delayed * 0.115926
        total += white * 0.5362
        total *= 0.11
        np.copyto(buffer, total, casting='unsafe')


# ----------------------------------------------------------------------------
# additive~  --  a spectrum, sounded
# ----------------------------------------------------------------------------

class AdditiveUnit(Unit):
    """A drawn spectrum played as an oscillator.

    The partials are given as amplitudes against partial index -- partial 1 is
    the fundamental, 2 is the octave, and so on -- and the unit sounds their
    sum. Everything else here shapes that list: `tilt` weights it by a slope in
    dB per octave, `balance` fades between the odd and even partials, `count`
    says how many of them sound, and `stretch` bends where they sit.

    Two render paths, chosen per block by whether `stretch` is zero:

      Harmonic. With every partial at an exact multiple of the fundamental the
      sum is periodic at the fundamental, so it is baked into a single cycle by
      one inverse FFT and read back with the phase accumulator. That costs the
      same whether it is eight partials or five hundred, and the table is built
      for exactly as many partials as fit under Nyquist at this block's
      fastest increment -- band limiting that is exact rather than fitted, and
      that follows a pitch sweep by the sample.

      Stretched. Once the partials are not multiples of anything the sum is not
      periodic and there is no table to bake, so it falls back to an actual
      oscillator bank -- one phase accumulator per partial, the whole bank
      advanced as a matrix. That is real work per partial, which is why the
      bank is capped well below the harmonic path's partial count.

    The table is rebuilt whenever the spectrum or the controls shaping it move,
    which for a swept control is once a block; a rebuild crossfades into the
    old table across that block, so a spectrum can be modulated without the
    waveform stepping at block boundaries.

    `stretch` is the exponent in ratio = k ** (1 + stretch). Zero is harmonic;
    small positive values are the stiffness of a real string, which is what
    makes a piano's top octave sound in tune with itself; larger values walk
    out through bells and gongs; negative values compress the partials
    together instead.

    Phase mode matters more than it looks. The same magnitude spectrum with
    every partial aligned is a narrow spike once per cycle with very little in
    between -- a high crest factor, so it must be scaled down hard to fit, and
    it sounds like a buzz. Random or Schroeder phases spread the same energy
    across the cycle. Schroeder is the deterministic version, a quadratic phase
    sweep whose crest factor is close to the theoretical minimum.
    """

    # The harmonic path holds any of these for the price of one lookup. The
    # bank pays per partial per sample, so it stops much sooner.
    MAX_PARTIALS = 512
    BANK_PARTIALS = 64
    # The bank runs in chunks so its matrix can be preallocated at a fixed
    # size rather than at MAX_BLOCK, which would be 16x larger for nothing.
    BANK_CHUNK = 256

    TABLE_SIZE = 4096
    SPECTRUM_POINTS = 512

    PHASE_MODES = ('aligned', 'random', 'schroeder')
    NORMALIZE_MODES = ('none', 'rms', 'peak')
    STRETCH_SPAN, FIXED_SPAN = 0, 1

    # What 'rms' normalises to. Roughly a vco~ saw, so swapping one for the
    # other does not move the level of a patch.
    RMS_TARGET = 0.5
    # Most a normaliser may lift a quiet spectrum, about 18 dB. Past this it
    # stops lifting, so a spectrum being faded out is allowed to fade out
    # rather than being held up and then dropped.
    MAX_BOOST = 8.0
    # dB per octave, as a factor on partial index: 20 * log10(2).
    DB_PER_OCTAVE = 6.020599913279624

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.frequency_in = self.new_inlet(base=110.0)
        self.pitch_in = self.new_inlet(base=0.0)              # octaves
        self.linear_fm_in = self.new_inlet(base=0.0)          # Hz
        self.tilt_in = self.new_inlet(base=-6.0)              # dB per octave
        self.partials_in = self.new_inlet(
            base=32.0, minimum=1.0, maximum=float(AdditiveUnit.MAX_PARTIALS))
        self.balance_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.stretch_in = self.new_inlet(base=0.0)
        self.spread_in = self.new_inlet(base=1.0, minimum=0.0, maximum=4.0)
        self.phase_mod_in = self.new_inlet(base=0.0)          # cycles
        self.sync_in = self.new_inlet(base=0.0)

        self.phase_mode = 0
        self.normalize = 1                # rms
        self.spectrum_span = AdditiveUnit.STRETCH_SPAN
        self.phase = 0.0
        self.start_phase = 0.0
        # Seconds for a phase to turn half a circle -- the speed limit, not a
        # lag, so only a jump is rationed. Long enough that a step is a move
        # rather than a lurch, short enough still to feel like a switch.
        self.phase_glide = 0.08

        # The drawn curve, sampled uniformly across its x axis. Flat to begin
        # with, which with the default tilt is a band-limited saw.
        self.spectrum = np.ones(AdditiveUnit.SPECTRUM_POINTS, dtype=np.float64)
        self._spectrum_x = np.linspace(0.0, 1.0, AdditiveUnit.SPECTRUM_POINTS)
        self._spectrum_generation = 0

        self.out = self.new_outlet()

        # -- per-partial working space --
        self._k = np.arange(1.0, AdditiveUnit.MAX_PARTIALS + 1.0)
        self._weights = np.zeros(AdditiveUnit.MAX_PARTIALS)
        self._ratios = np.zeros(AdditiveUnit.MAX_PARTIALS)
        self._positions = np.zeros(AdditiveUnit.MAX_PARTIALS)
        # Random phases are drawn once and kept. Redrawing them on a rebuild
        # would change the waveform under a held note, which is a click.
        self._random_phases = (np.random.RandomState(20240501)
                               .uniform(0.0, 2.0 * math.pi,
                                        AdditiveUnit.MAX_PARTIALS))
        # The phases actually in force, what the settings ask for, and the
        # rotation between them for this block.
        self._dispersion_now = np.zeros(AdditiveUnit.MAX_PARTIALS)
        self._dispersion_want = np.zeros(AdditiveUnit.MAX_PARTIALS)
        self._dispersion_step = np.zeros(AdditiveUnit.MAX_PARTIALS)
        self._phase_ready = 0            # partials that have sounded
        self._phase_epoch = 0            # bumped while the phases are moving
        # What the phases were last resolved for, so a settled node can skip
        # the whole business -- which is the usual case, and this runs in the
        # audio callback once per voice per block.
        self._phase_settled = False
        self._phase_count = -1
        self._phase_spread = None
        self._phase_law = -1

        # -- harmonic path --
        # Two tables, used alternately, so the one being crossfaded out is
        # still intact while the new one is written.
        size = AdditiveUnit.TABLE_SIZE
        self._tables = [np.zeros(size + 1), np.zeros(size + 1)]
        self._live = 0
        self._table_key = None
        self._bins = np.zeros(size // 2 + 1, dtype=np.complex128)

        # -- bank path --
        self._bank = np.zeros((AdditiveUnit.BANK_PARTIALS,
                               AdditiveUnit.BANK_CHUNK))
        self._bank_phase = np.zeros(AdditiveUnit.BANK_PARTIALS)
        self._offsets = np.zeros(AdditiveUnit.BANK_PARTIALS)
        self._bank_fixed = np.zeros(AdditiveUnit.BANK_PARTIALS)
        self._bank_ratio = np.zeros(AdditiveUnit.BANK_PARTIALS)
        self._cumulative = np.zeros(AdditiveUnit.BANK_CHUNK)

        # -- block working space --
        self._phase = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._increment = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._mix = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._fresh = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._stale = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._index = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._floor = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._integer = np.zeros(MAX_BLOCK, dtype=np.int32)
        self._lower = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._upper = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._pm_last = 0.0

    def reset(self):
        self.phase = self.start_phase
        self._bank_phase[:] = self.start_phase
        self._pm_last = 0.0
        # An explicit reset is the one place a phase may jump: nothing is
        # sounding through it, so there is nothing for a glide to protect.
        self._phase_ready = 0
        self._phase_settled = False

    # -- the spectrum --------------------------------------------------------

    def set_spectrum(self, values):
        """Main thread: swap in a new set of partial amplitudes.

        Assigned whole, so the audio thread sees one spectrum or the other and
        never half of each. The generation bump is what invalidates the table.
        """
        curve = np.asarray(values, dtype=np.float64).reshape(-1)
        if curve.size < 2:
            return
        if curve.size != AdditiveUnit.SPECTRUM_POINTS:
            curve = np.interp(self._spectrum_x,
                              np.linspace(0.0, 1.0, curve.size), curve)
        self.spectrum = np.clip(curve, 0.0, None)
        self._spectrum_generation += 1

    @staticmethod
    def _scalar(signal):
        """Spectral controls are read once a block, not once a sample.

        Rebuilding the table costs about as much as a block of lookups, so
        there is no sense resolving these per sample -- and the bank's weights
        are per partial, which has no per-sample meaning either.
        """
        return signal.value if signal.constant else float(signal.data[0])

    def _partial_weights(self, count, tilt, balance, ratios, limit, partials):
        """Amplitude per sounding partial, and how many there are.

        The drawn curve, weighted by the tilt, faded between odd and even, and
        taken down by two soft edges: one at the requested partial count, one
        at Nyquist. Both are ramps a partial wide rather than cutoffs, so a
        partial arriving or leaving does so without a step.
        """
        highest = max(1, int(math.ceil(min(count, partials))))
        k = self._k[:highest]
        weights = self._weights[:highest]

        # Where each partial reads the drawn curve. Spanning the count means
        # the shape stretches to fit however many partials sound; spanning the
        # whole range means raising the count extends the spectrum instead.
        #
        # The count is used as it is, fractional part and all. Rounding it up
        # to whole partials first -- which is what the array has to be -- moves
        # every partial's reading of the curve at once each time the count
        # crosses an integer, so a drawn shape lurches its way through a count
        # sweep. Only the array length may be whole; the shape may not.
        positions = self._positions[:highest]
        if self.spectrum_span == AdditiveUnit.FIXED_SPAN:
            divisor = float(AdditiveUnit.MAX_PARTIALS - 1)
        else:
            divisor = max(1.0, count - 1.0)
        np.subtract(k, 1.0, out=positions)
        positions /= divisor
        np.copyto(weights, np.interp(positions, self._spectrum_x, self.spectrum))

        if tilt != 0.0:
            np.multiply(weights, np.power(k, tilt / AdditiveUnit.DB_PER_OCTAVE),
                        out=weights)

        # 0 leaves only the odd partials (a square/clarinet family), 1 only the
        # even, 0.5 all of them. Between those it is a fade, not a switch.
        if balance != 0.5:
            odd_gain = min(1.0, 2.0 - 2.0 * balance)
            even_gain = min(1.0, 2.0 * balance)
            weights[0::2] *= odd_gain      # k = 1, 3, 5 ... the odd partials
            weights[1::2] *= even_gain

        # Soft edge at the requested count.
        edge = self._positions[:highest]
        np.subtract(count + 1.0, k, out=edge)
        np.clip(edge, 0.0, 1.0, out=edge)
        weights *= edge

        # Soft edge at Nyquist. `limit` is the highest partial *ratio* that
        # fits, so for the bank this bites on the stretched positions rather
        # than on the index.
        np.subtract(limit + 1.0, ratios[:highest], out=edge)
        np.clip(edge, 0.0, 1.0, out=edge)
        weights *= edge

        # Normalising divides by the level, so a spectrum on its way to
        # silence asks for an ever larger boost -- and a guard that gives up
        # below some threshold turns that into a jump from full level to
        # nothing. Both are audible, and both fire in ordinary use: draw a
        # curve that reaches zero, or take the tilt or the partial count far
        # enough, and the sounding partials get arbitrarily quiet.
        #
        # The boost is capped instead. Above the cap the level is held where
        # it is asked for; below it the gain stops rising, so a spectrum
        # fading out fades out, and silence stays silent.
        if self.normalize == 1:                      # rms
            level = math.sqrt(max(float(np.dot(weights, weights)), 0.0) * 0.5)
            floor = AdditiveUnit.RMS_TARGET / AdditiveUnit.MAX_BOOST
            weights *= AdditiveUnit.RMS_TARGET / max(level, floor)
        elif self.normalize == 2:                    # peak
            # The sum is what the partials reach if they ever line up, so
            # scaling by it cannot overshoot. Computed from the weights alone,
            # which is the only definition both render paths can share.
            total = float(np.sum(weights))
            weights *= 1.0 / max(total, 1.0 / AdditiveUnit.MAX_BOOST)

        return weights, highest

    # -- phase ---------------------------------------------------------------
    #
    # Phase is a continuous quantity here, in both senses: 'spread' scales the
    # chosen law rather than switching it on, and whatever the settings ask for
    # is reached by rotating rather than by jumping.
    #
    # Both matter for the same reason. Every partial changing phase at once is
    # a step in the waveform, which is a click; and crossfading between two
    # sets of phases is worse than it sounds, because the same partial at two
    # phases partly cancels itself halfway through, so a blend sweeps a notch
    # through the spectrum. Rotating each partial to where it is wanted avoids
    # both -- amplitudes never move, and a phase turning slowly is heard as a
    # momentary detune of a fraction of a cent, if at all.

    def _dispersion(self, count):
        """The phase pattern of the chosen law, before 'spread' scales it.

        Zero for 'aligned', so at spread 0 every law agrees and changing the
        law is silent. That is the useful property: spread becomes the control
        and the law becomes a choice of what it spreads towards.
        """
        want = self._dispersion_want[:count]
        mode = self.phase_mode
        if mode == 1:
            np.copyto(want, self._random_phases[:count])
        elif mode == 2:
            # Schroeder: a quadratic sweep across the partials, which spreads
            # the cycle's energy about as evenly as it can be spread.
            k = self._k[:count]
            np.multiply(k, k - 1.0, out=want)
            want *= -math.pi / count
        else:
            want[:] = 0.0
        return want

    def _advance_phase(self, count, spread, frames):
        """Rotate the applied phases toward the ones the settings ask for.

        Returns (applied, step, moving): the phases in force for this block,
        how far each moved to get there, and whether anything is still moving.
        The step is what the bank needs -- it turns the rotation into a
        frequency, so the phases arrive smoothly within the block rather than
        stepping at its edge.
        """
        if (self._phase_settled and count == self._phase_count
                and spread == self._phase_spread
                and self.phase_mode == self._phase_law):
            # Arrived, and nothing has asked for anything different. The step
            # is already zero, so the bank folds in no rotation and the table
            # keeps its key.
            return (self._dispersion_now[:count],
                    self._dispersion_step[:count], False)
        self._phase_count = count
        self._phase_spread = spread
        self._phase_law = self.phase_mode

        want = self._dispersion(count)
        if spread != 1.0:
            want *= spread
        now = self._dispersion_now[:count]
        step = self._dispersion_step[:count]

        # A partial that has just come into range starts where it is wanted,
        # rather than gliding up from a phase it never sounded at.
        if count > self._phase_ready:
            now[self._phase_ready:count] = want[self._phase_ready:count]
            self._phase_ready = count

        np.subtract(want, now, out=step)
        # A phase is an angle, so go the short way round: +3pi/2 is really
        # -pi/2, and rotating the long way would take four times as long for
        # the same destination.
        step += math.pi
        np.mod(step, 2.0 * math.pi, out=step)
        step -= math.pi

        # A speed limit rather than a lag. A lag would smooth the jumps and
        # also drag every continuous move -- spread swept by hand or by an
        # LFO would arrive late and come out shallower than it was asked for.
        # Limiting the speed instead leaves anything slower than the limit
        # exactly as it is and only stretches what could not be done in time,
        # so the control stays immediate and only a jump is rationed.
        glide = self.phase_glide
        if glide > 0.0:
            reach = math.pi * frames / (glide * self.sample_rate)
            np.clip(step, -reach, reach, out=step)
        moving = bool(np.any(step))
        np.add(now, step, out=now)
        self._phase_settled = not moving
        return now, step, moving

    # -- rendering -----------------------------------------------------------

    def render(self, frames):
        out = self.out
        buffer = out.data[:frames]

        frequency = self.frequency_in.eval(frames)
        pitch = self.pitch_in.eval(frames)
        linear_fm = self.linear_fm_in.eval(frames)
        phase_mod = self.phase_mod_in.eval(frames)
        sync = self.sync_in.eval(frames)

        increment = self._increment[:frames]
        self._build_increment(increment, frequency, pitch, linear_fm, frames)
        self._fold_phase_mod(increment, phase_mod, frames)

        count = min(max(self._scalar(self.partials_in.eval(frames)), 1.0),
                    float(AdditiveUnit.MAX_PARTIALS))
        tilt = self._scalar(self.tilt_in.eval(frames))
        balance = min(1.0, max(0.0, self._scalar(self.balance_in.eval(frames))))
        stretch = self._scalar(self.stretch_in.eval(frames))
        spread = max(0.0, self._scalar(self.spread_in.eval(frames)))

        # The highest partial that stays below Nyquist, in multiples of the
        # fundamental, taken at this block's fastest increment so a sweep is
        # limited by where it is going rather than where it started.
        fastest = float(np.max(np.abs(increment))) if frames else 0.0
        if fastest > 1.0e-9:
            limit = 0.5 / fastest
        else:
            limit = float(AdditiveUnit.MAX_PARTIALS)

        segments = self._sync_segments(sync, frames)

        if abs(stretch) < 1.0e-4:
            self._render_table(buffer, increment, segments, count, tilt,
                               balance, spread, limit, frames)
        else:
            self._render_bank(buffer, increment, segments, count, tilt,
                              balance, stretch, spread, limit, frames)
        out.constant = False

    def _fold_phase_mod(self, increment, phase_mod, frames):
        """Add the phase modulation to the increment as its own difference.

        Accumulating the difference gives back the modulation exactly, so the
        one phase accumulator carries both -- and in the bank, where each
        partial runs at its own rate, the modulation comes out scaled by that
        rate, which is what phase modulation of the fundamental means. Adding
        it after the accumulator instead would work only at ratio 1, and would
        step every partial that is not an integer multiple every time the
        fundamental's phase wrapped.
        """
        if phase_mod.constant:
            if phase_mod.value != self._pm_last:
                # A knob, moved. The whole change put on one sample would be
                # exactly that: the phase jumping between two samples, which
                # is a step in the waveform and a click -- once per frame for
                # as long as the knob is being dragged. Spread across the
                # block it is a phase ramp instead, i.e. a moment of pitch,
                # which is the only thing a phase move can honestly be.
                increment += (phase_mod.value - self._pm_last) / frames
                self._pm_last = phase_mod.value
        else:
            data = phase_mod.data[:frames]
            previous = self._pm_last
            self._pm_last = float(data[-1])
            increment[1:] += np.diff(data)
            increment[0] += float(data[0]) - previous
        np.clip(increment, -0.5, 0.5, out=increment)

    def _accumulate(self, phase, increment, segments, frames):
        """Fundamental phase for the block, wrapped, honouring sync."""
        start = 0
        for segment_end, do_reset in segments:
            if do_reset:
                self.phase = self.start_phase
            if segment_end > start:
                view = phase[start:segment_end]
                np.cumsum(increment[start:segment_end], out=view)
                view += self.phase
                np.mod(view, 1.0, out=view)
                self.phase = float(view[-1])
            start = segment_end
        return phase

    # -- harmonic path -------------------------------------------------------

    def _render_table(self, buffer, increment, segments, count, tilt, balance,
                      spread, limit, frames):
        # The phases rotate whether or not the table is rebuilt, so this runs
        # before the key is compared -- and while they are moving it is what
        # invalidates the key, so each block bakes the phases it has reached
        # and crossfades from the ones before. Consecutive tables in a glide
        # differ by a fraction of a radian per partial, which is exactly the
        # case the crossfade handles cleanly.
        highest = max(1, int(math.ceil(min(count,
                                           AdditiveUnit.MAX_PARTIALS))))
        _, _, moving = self._advance_phase(highest, spread, frames)
        if moving:
            self._phase_epoch += 1

        table, previous = self._ensure_table(count, tilt, balance, limit)
        phase = self._accumulate(self._phase[:frames], increment, segments,
                                 frames)

        # Keep the bank's per-partial accumulators in step with the
        # fundamental, against the block where the stretch leaves zero and the
        # bank takes over. Every partial here is an exact multiple, so partial
        # k's phase is k times the fundamental's -- wrapping first is what
        # being harmonic means. Without this the bank would start each partial
        # from wherever it was last left, which is a step in every one of them
        # at once: the click on taking the stretch off zero. (The other
        # direction is already covered -- the bank keeps self.phase running.)
        bank = self._bank_phase
        np.multiply(self._k[:AdditiveUnit.BANK_PARTIALS], self.phase, out=bank)
        np.mod(bank, 1.0, out=bank)

        fresh = self._gather(phase, table, self._fresh[:frames], frames)
        if previous is None:
            np.copyto(buffer, fresh, casting='unsafe')
            return

        # The table changed under a running phase, which is a step in the
        # output wherever the two disagree. Crossfade across the block: for a
        # control being swept the two tables are nearly the same and this
        # costs a pass to prove it, and for a control being jumped it is the
        # difference between a move and a click.
        stale = self._gather(phase, previous, self._stale[:frames], frames)
        blend = self._mix[:frames]
        np.multiply(_INDEX_RAMP[:frames], 1.0 / frames, out=blend)
        np.subtract(fresh, stale, out=fresh)
        np.multiply(fresh, blend, out=fresh)
        np.add(stale, fresh, out=fresh)
        np.copyto(buffer, fresh, casting='unsafe')

    def _ensure_table(self, count, tilt, balance, limit):
        """The single cycle for these controls, and the one it replaces.

        Returns (table, previous) where previous is None if nothing was
        rebuilt -- which is the common case, since none of these move unless
        something is being swept.
        """
        # Nyquist only enters the key once it actually bites; otherwise a
        # pitch sweep under a modest partial count would rebuild every block
        # to produce the same table.
        key = (round(count, 2), round(min(limit, count + 1.0), 2),
               round(tilt, 2), round(balance, 3), self._spectrum_generation,
               self._phase_epoch, self.normalize, self.spectrum_span)
        if key == self._table_key:
            return self._tables[self._live], None

        previous = self._tables[self._live] if self._table_key is not None else None
        target = self._tables[1 - self._live]
        self._build_table(target, count, tilt, balance, limit)
        self._live = 1 - self._live
        self._table_key = key
        return target, previous

    def _build_table(self, target, count, tilt, balance, limit):
        weights, highest = self._partial_weights(
            count, tilt, balance, self._k, limit, AdditiveUnit.MAX_PARTIALS)

        bins = self._bins
        bins[:] = 0.0
        # The bins want the cosine convention; a quarter turn back off the
        # applied phases makes partial 1 a sine, so a reset or a sync still
        # lands on a zero crossing.
        phases = self._dispersion_now[:highest] - math.pi * 0.5
        bins[1:highest + 1] = weights * np.exp(1j * phases)

        size = AdditiveUnit.TABLE_SIZE
        # irfft carries a 1/size, and a single bin appears in the result at
        # twice its own magnitude once its conjugate is counted, so this scale
        # makes a bin of 1.0 come out as a partial of amplitude 1.0.
        cycle = np.fft.irfft(bins, n=size) * (size * 0.5)
        target[:size] = cycle
        # One entry past the end, holding the wrap, so index + 1 is always in
        # bounds and the last interval interpolates into the next cycle.
        target[size] = cycle[0]

    def _gather(self, phase, table, out, frames):
        """Read the table at `phase` (0..1) with linear interpolation."""
        size = table.size - 1
        index = self._index[:frames]
        np.multiply(phase, size, out=index)
        np.nan_to_num(index, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(index, 0.0, float(size), out=index)

        whole = self._floor[:frames]
        np.floor(index, out=whole)
        integer = self._integer[:frames]
        np.copyto(integer, whole, casting='unsafe')
        np.subtract(index, whole, out=index)          # index is now the fraction

        lower = self._lower[:frames]
        upper = self._upper[:frames]
        np.take(table, integer, out=lower, mode='clip')
        np.add(integer, 1, out=integer)
        np.take(table, integer, out=upper, mode='clip')

        np.subtract(upper, lower, out=upper)
        np.multiply(upper, index, out=upper)
        np.add(lower, upper, out=out)
        return out

    # -- stretched path ------------------------------------------------------

    def _render_bank(self, buffer, increment, segments, count, tilt, balance,
                     stretch, spread, limit, frames):
        highest = max(1, int(math.ceil(min(count,
                                           AdditiveUnit.BANK_PARTIALS))))
        ratios = self._ratios[:highest]
        np.power(self._k[:highest], 1.0 + stretch, out=ratios)

        weights, highest = self._partial_weights(
            count, tilt, balance, ratios, limit,
            AdditiveUnit.BANK_PARTIALS)
        ratios = ratios[:highest]

        # Partial k's phase is ratio_k times the fundamental's, so the block is
        # one 1-D accumulation scaled out across the partials rather than an
        # accumulation per partial -- which measured about four times cheaper,
        # accumulation and wrapping being the expensive parts of a bank and
        # the sine much the cheapest.
        #
        # Nothing is wrapped into a cycle here either: sin takes the radians
        # as they come. Un-wrapped, they reach a few hundred thousand within a
        # chunk, where a double still resolves about 5e-11 of a radian. Only
        # the phase carried between chunks is wrapped, and that is one short
        # vector rather than the whole matrix.
        #
        # What must not be done is to scale the *wrapped* fundamental phase:
        # every partial that is not an integer multiple would then step every
        # time the fundamental turned over, which is a click per cycle.
        phases = self._bank_phase[:highest]
        offsets = self._offsets[:highest]
        mix = self._mix[:frames]
        tau = 2.0 * math.pi

        # A phase rotation is a frequency. Rather than stepping the offsets at
        # the block edge -- which is the click this whole mechanism exists to
        # avoid -- the rotation owed this block is folded into each partial's
        # rate, so it arrives smoothly across the block and costs one add per
        # partial. The chunk length cancels out of that rate, so every chunk
        # can use the same one: the share of the rotation a chunk performs is
        # proportional to the share of the block it covers.
        applied, step, _ = self._advance_phase(highest, spread, frames)
        fixed = self._bank_fixed[:highest]
        np.subtract(applied, step, out=fixed)      # where this block starts
        travelled = float(np.sum(increment[:frames])) if frames else 0.0
        gliding = abs(travelled) > 1.0e-12 and bool(np.any(step))
        if gliding:
            rate = self._bank_ratio[:highest]
            np.multiply(step, 1.0 / (tau * travelled), out=rate)
            rate += ratios
        else:
            # Either nothing is turning -- so there is no rate to hide the
            # rotation in, and nothing audible to hide it from -- or there is
            # no rotation owed. Either way, land on the new phases.
            rate = ratios
            np.copyto(fixed, applied)

        start = 0
        for segment_end, do_reset in segments:
            if do_reset:
                phases[:] = self.start_phase
            begin = start
            while begin < segment_end:
                stop = min(begin + AdditiveUnit.BANK_CHUNK, segment_end)
                span = stop - begin
                block = self._bank[:highest, :span]
                travel = self._cumulative[:span]
                np.cumsum(increment[begin:stop], out=travel)
                advance = float(travel[-1])
                travel *= tau

                np.multiply(rate[:, None], travel[None, :], out=block)
                np.multiply(phases, tau, out=offsets)
                offsets += fixed
                block += offsets[:, None]
                np.sin(block, out=block)
                mix[begin:stop] = weights @ block

                # Carry each partial's phase into the next chunk.
                np.multiply(ratios, advance, out=offsets)
                phases += offsets
                np.mod(phases, 1.0, out=phases)
                if gliding:
                    # And the rotation performed so far with it, or the next
                    # chunk would start the glide over rather than continue it.
                    np.multiply(step, advance / travelled, out=offsets)
                    fixed += offsets
                begin = stop
            start = segment_end

        # The fundamental's own accumulator is kept in step so that leaving
        # the stretch does not restart the phase.
        self._accumulate(self._phase[:frames], increment, segments, frames)
        np.copyto(buffer, mix, casting='unsafe')


# ----------------------------------------------------------------------------
# shaper~  --  transfer function over the whole block
# ----------------------------------------------------------------------------

class ShaperUnit(Unit):
    """A curve applied to every sample: table lookup with linear interpolation.

    The node layer bakes the breakpoint function -- curvature and all -- onto a
    uniformly spaced table whenever it changes, so the audio thread never sees
    breakpoints, easing or searching. It maps the input to a table index,
    gathers the two neighbouring entries and interpolates between them, which
    is a dozen vector passes over the block: about 12 us per 512 frames,
    against 300 us for the same lookup written as a Python loop and 32 us for
    np.interp, which pays a binary search per sample that uniform spacing makes
    unnecessary.

    An unmodulated input costs nothing at all -- a constant signal takes one
    scalar lookup and the output stays constant for the block.

    Note that this is waveshaping: a curved transfer function generates
    harmonics above the input's own, which fold back if they pass Nyquist.
    That is inherent to the operation, not a defect of the table.
    """

    TABLE_SIZE = 4096
    CLIP, WRAP, FOLD = 0, 1, 2

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.in_low_in = self.new_inlet(base=-1.0)
        self.in_high_in = self.new_inlet(base=1.0)

        self.range_mode = ShaperUnit.CLIP
        # One entry more than the size, so index+1 is always in bounds. The
        # identity to begin with, so an unloaded shaper passes signal through.
        self.table = np.linspace(-1.0, 1.0, ShaperUnit.TABLE_SIZE + 1)

        self.out = self.new_outlet()
        self._index = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._floor = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._integer = np.zeros(MAX_BLOCK, dtype=np.int32)
        self._low = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._high = np.zeros(MAX_BLOCK, dtype=np.float64)

    def set_table(self, values):
        """Main thread: swap in a new curve.

        Assigned as one whole array, so the audio thread sees either the old
        table or the new one and never a half-written mixture.
        """
        table = np.asarray(values, dtype=np.float64).reshape(-1)
        if table.size >= 2:
            self.table = table

    def _position(self, value, low, high):
        """Input value to 0..1 across the table, honouring the range mode."""
        span = high - low
        position = (value - low) / span if span else 0.0
        if not math.isfinite(position):
            return 0.0
        if self.range_mode == ShaperUnit.WRAP:
            position = position % 1.0
        elif self.range_mode == ShaperUnit.FOLD:
            position = 1.0 - abs((position % 2.0) - 1.0)
        return min(1.0, max(0.0, position))

    def lookup(self, value, low=None, high=None):
        """Scalar read. Used for constant blocks and by the node's display."""
        if low is None:
            low = self.in_low_in.base
        if high is None:
            high = self.in_high_in.base
        table = self.table
        size = table.size - 1
        index = self._position(value, low, high) * size
        whole = int(index)
        if whole >= size:
            return float(table[size])
        fraction = index - whole
        return float(table[whole] + fraction * (table[whole + 1] - table[whole]))


    def bypass_pairs(self):
        return ((self.signal_in, self.out),)

    def render(self, frames):
        signal = self.signal_in.eval(frames)
        low_in = self.in_low_in.eval(frames)
        high_in = self.in_high_in.eval(frames)

        low = low_in.value if low_in.constant else float(low_in.data[0])
        high = high_in.value if high_in.constant else float(high_in.data[0])

        if signal.constant:
            self.out.set_constant(self.lookup(signal.value, low, high))
            return

        table = self.table
        size = table.size - 1
        span = high - low
        scale = (1.0 / span) if span else 0.0

        index = self._index[:frames]
        np.subtract(signal.data[:frames], low, out=index, casting='unsafe')
        index *= scale

        mode = self.range_mode
        if mode == ShaperUnit.WRAP:
            np.mod(index, 1.0, out=index)
        elif mode == ShaperUnit.FOLD:
            np.mod(index, 2.0, out=index)
            np.subtract(index, 1.0, out=index)
            np.abs(index, out=index)
            np.subtract(1.0, index, out=index)

        # A NaN arriving from upstream would otherwise become a garbage integer
        # index; np.take is also asked to clip, so no index can escape the table.
        np.nan_to_num(index, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.multiply(index, size, out=index)
        np.clip(index, 0.0, float(size), out=index)

        whole = self._floor[:frames]
        np.floor(index, out=whole)
        integer = self._integer[:frames]
        np.copyto(integer, whole, casting='unsafe')
        np.subtract(index, whole, out=index)      # index is now the fraction

        lower = self._low[:frames]
        upper = self._high[:frames]
        np.take(table, integer, out=lower, mode='clip')
        np.add(integer, 1, out=integer)
        np.take(table, integer, out=upper, mode='clip')

        np.subtract(upper, lower, out=upper)
        np.multiply(upper, index, out=upper)
        np.add(lower, upper, out=lower)

        np.copyto(self.out.data[:frames], lower, casting='unsafe')
        self.out.constant = False


# ----------------------------------------------------------------------------
# vcf~
# ----------------------------------------------------------------------------

# Topology-preserving state variable filter (Andy Simper / Cytomic). Stable
# under fast per-sample cutoff modulation, which is the whole reason this runs
# under numba instead of using fixed coefficients per block.
def _svf_kernel_source(x, g, k, ic1, ic2, mode, out):
    for i in range(x.shape[0]):
        gi = g[i]
        ki = k[i]
        a1 = 1.0 / (1.0 + gi * (gi + ki))
        a2 = gi * a1
        a3 = gi * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            out[i] = v2
        elif mode == 1:
            out[i] = x[i] - ki * v1 - v2
        elif mode == 2:
            out[i] = v1
        else:
            out[i] = x[i] - ki * v1
    return ic1, ic2


if _HAVE_NUMBA:
    _svf_kernel = njit(cache=True, fastmath=True)(_svf_kernel_source)
else:
    _svf_kernel = _svf_kernel_source

_svf_ready = threading.Event()


def _formant_bank_source(x, a1, a2, a3, ic1, ic2, gains, out):
    """A parallel bank of TPT state variable filters, bandpass taps summed.

    One kernel for the whole bank rather than one filter at a time: the bands
    are independent, so the inner loop vectorises, and the input sample is read
    once for all of them. A five-band bank costs about 8 us per 512 frames --
    less than the numpy plumbing around a single vcf~.

    Coefficients are per band and hold for the block, so they are computed
    outside and passed in rather than being rebuilt per sample.
    """
    bands = a1.shape[0]
    for i in range(x.shape[0]):
        sample = x[i]
        total = 0.0
        for b in range(bands):
            v3 = sample - ic2[b]
            v1 = a1[b] * ic1[b] + a2[b] * v3
            v2 = ic2[b] + a2[b] * ic1[b] + a3[b] * v3
            ic1[b] = 2.0 * v1 - ic1[b]
            ic2[b] = 2.0 * v2 - ic2[b]
            total += v1 * gains[b]
        out[i] = total


if _HAVE_NUMBA:
    _formant_bank = njit(cache=True, fastmath=True)(_formant_bank_source)
else:
    _formant_bank = _formant_bank_source


def _vocoder_analyse_source(x, a1, a2, a3, ic1, ic2, env, attack, release,
                            follow, env_out):
    """Per-band envelope of the modulator, tracked sample by sample.

    Attack and release are separate one-poles: a follower that rises fast and
    falls slowly is what makes consonants arrive intact while vowels hold.
    `follow` at 0 freezes the envelopes where they are, which turns the current
    spectral shape into a fixed filter -- the bank keeps its vowel after the
    voice stops.

    The envelope is written per sample rather than per block. A block-rate gain
    on a 512-frame buffer zippers audibly on speech, where the whole point is
    that the gains move fast.
    """
    bands = a1.shape[0]
    for i in range(x.shape[0]):
        sample = x[i]
        for b in range(bands):
            v3 = sample - ic2[b]
            v1 = a1[b] * ic1[b] + a2[b] * v3
            v2 = ic2[b] + a2[b] * ic1[b] + a3[b] * v3
            ic1[b] = 2.0 * v1 - ic1[b]
            ic2[b] = 2.0 * v2 - ic2[b]
            magnitude = abs(v1)
            level = env[b]
            coefficient = attack if magnitude > level else release
            level = level + (magnitude - level) * coefficient * follow
            env[b] = level
            env_out[b, i] = level


def _vocoder_synthesise_source(low, high, split, a1, a2, a3, ic1, ic2,
                               env_out, weights, out):
    """The carrier through the same bank, each band scaled by its envelope.

    Bands at or above `split` read the second input instead of the first,
    which is how the sibilance path works: noise mixed into the carrier for
    the top of the range only, so 's' and 't' survive without the whole voice
    turning breathy.
    """
    bands = a1.shape[0]
    for i in range(out.shape[0]):
        total = 0.0
        for b in range(bands):
            sample = low[i] if b < split else high[i]
            v3 = sample - ic2[b]
            v1 = a1[b] * ic1[b] + a2[b] * v3
            v2 = ic2[b] + a2[b] * ic1[b] + a3[b] * v3
            ic1[b] = 2.0 * v1 - ic1[b]
            ic2[b] = 2.0 * v2 - ic2[b]
            total += v1 * env_out[b, i] * weights[b]
        out[i] = total


if _HAVE_NUMBA:
    _vocoder_analyse = njit(cache=True, fastmath=True)(_vocoder_analyse_source)
    _vocoder_synthesise = njit(cache=True,
                               fastmath=True)(_vocoder_synthesise_source)
else:
    _vocoder_analyse = _vocoder_analyse_source
    _vocoder_synthesise = _vocoder_synthesise_source


def _cubic_read_source(buffer, size, position):
    """Catmull-Rom read at a fractional position, wrapping the buffer.

    Linear interpolation would be cheaper, but its error is a lowpass that
    varies with the fractional part -- so a delay time being modulated would
    scrub the high end in time with the modulation, which is audible as a
    warble quite apart from the pitch shift that is wanted.
    """
    i1 = int(position)
    frac = position - i1
    i0 = i1 - 1
    if i0 < 0:
        i0 += size
    i2 = i1 + 1
    if i2 >= size:
        i2 -= size
    i3 = i1 + 2
    if i3 >= size:
        i3 -= size
    y0 = buffer[i0]
    y1 = buffer[i1]
    y2 = buffer[i2]
    y3 = buffer[i3]
    c1 = 0.5 * (y2 - y0)
    c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)
    return ((c3 * frac + c2) * frac + c1) * frac + y1


def _delay_kernel_source(x, buffer, write, delay, feedback, damping, out,
                         low, phase, held_a, held_b, fade_step, mode, freeze):
    """One delay line with damped feedback, sample by sample.

    Recursive by nature -- what is written now depends on what was read now --
    so this is the one shape that cannot be vectorised, which is why it runs
    here rather than in numpy.

    mode 0 slides: one read head, so changing the delay time drags the read
    point through the buffer and the pitch goes with it. That is what tape
    does, and with a body driving the time it is the whole point.

    mode 1 crossfades: two heads, each picking up the current time when its
    turn comes round, mixed across a fixed period. The time changes without
    the pitch, at the cost of a little comb while the two disagree. Holding
    still, both heads sit at the same place and it is exactly mode 0.
    """
    size = buffer.shape[0]
    limit = size - 3.0
    for i in range(x.shape[0]):
        want = delay[i]
        if want < 1.0:
            want = 1.0
        elif want > limit:
            want = limit

        if held_a <= 0.0:
            held_a = want              # first block: start where we are asked

        if mode == 0:
            held_a = want
            read = write - want
            if read < 0.0:
                read += size
            y = _cubic_read(buffer, size, read)
        else:
            # Crossfade only when the time has actually moved, and hold a
            # single head the rest of the time. Blending two heads
            # continuously -- which is the obvious way to write this -- would
            # mean the output was always a mixture of two delays, so the pitch
            # would sit between them and the mode would never do the one thing
            # it exists for.
            if phase <= 0.0 and (want - held_a > 1.0 or held_a - want > 1.0):
                held_b = want
                phase = fade_step
            read = write - held_a
            if read < 0.0:
                read += size
            y = _cubic_read(buffer, size, read)
            if phase > 0.0:
                read = write - held_b
                if read < 0.0:
                    read += size
                y += (_cubic_read(buffer, size, read) - y) * phase
                phase += fade_step
                if phase >= 1.0:
                    held_a = held_b
                    phase = 0.0

        # One pole in the loop. At damping 0 it is transparent, so a clean
        # delay stays clean; opening it makes each repeat darker than the one
        # before, which is what stops a long feedback becoming a shriek.
        low += (y - low) * (1.0 - damping[i])
        fed = low * feedback[i]

        # Feedback over unity is worth having -- it is how a delay becomes an
        # oscillator -- so the loop needs a stop that is not a hard clip. This
        # is exactly linear below 1.5 and bends smoothly above it, so ordinary
        # levels are untouched and a runaway settles instead of exploding.
        if fed > 1.5:
            fed = 1.5 + np.tanh(fed - 1.5)
        elif fed < -1.5:
            fed = -1.5 - np.tanh(-fed - 1.5)

        if freeze != 0:
            buffer[write] = y          # the loop keeps what it has, and only that
        else:
            buffer[write] = x[i] + fed
        out[i] = y

        write += 1
        if write >= size:
            write = 0
    return write, low, phase, held_a, held_b


def _one_euro_source(x, out, period, min_cutoff, beta, d_cutoff,
                     previous, speed, smoothed, primed):
    """The one euro filter, sample by sample.

    A low-pass whose cutoff rises with the signal's own speed. The speed
    estimate is itself low-passed, at a fixed cutoff, so that noise in the
    signal cannot open the filter up and let more of itself through.

    A cutoff becomes a coefficient as (2*pi*fc*Te) / (1 + 2*pi*fc*Te), which
    is one multiply and one divide -- no tan or exp in the inner loop. Only
    the signal's own cutoff changes per sample; everything else is lifted out.
    """
    scale = 2.0 * math.pi * period
    speed_weight = (scale * d_cutoff) / (1.0 + scale * d_cutoff)
    rate = 1.0 / period

    for i in range(x.shape[0]):
        sample = x[i]
        if primed == 0.0:
            previous = sample
            speed = 0.0
            smoothed = sample
            primed = 1.0
            out[i] = sample
            continue

        # Speed of the raw signal, smoothed at a fixed cutoff.
        speed = speed + ((sample - previous) * rate - speed) * speed_weight
        previous = sample

        # The faster it is moving, the wider the filter opens.
        cutoff = min_cutoff + beta * abs(speed)
        weight = (scale * cutoff) / (1.0 + scale * cutoff)
        smoothed = smoothed + (sample - smoothed) * weight
        out[i] = smoothed
    return previous, speed, smoothed, primed


if _HAVE_NUMBA:
    _one_euro = njit(cache=True, fastmath=True)(_one_euro_source)
else:
    _one_euro = _one_euro_source

if _HAVE_NUMBA:
    _cubic_read = njit(cache=True, fastmath=True, inline='always')(
        _cubic_read_source)
    _delay_kernel = njit(cache=True, fastmath=True)(_delay_kernel_source)
else:
    _cubic_read = _cubic_read_source
    _delay_kernel = _delay_kernel_source


def _warm_up_filter():
    """Compile the filter kernels off the audio thread.

    numba's first call triggers an LLVM compile of roughly a second. Doing that
    inside the PortAudio callback would drop buffers, so it happens on a worker
    thread and vcf~ passes audio through dry until the kernel is live. With
    cache=True this only costs anything on the first run after an install.
    """
    if not _HAVE_NUMBA:
        print('synth_core: numba unavailable, vcf~ will pass audio through')
        return
    try:
        dummy = np.zeros(8, dtype=np.float32)
        coefficients = np.full(8, 0.1, dtype=np.float64)
        output = np.zeros(8, dtype=np.float64)
        _svf_kernel(dummy, coefficients, coefficients, 0.0, 0.0, 0, output)
        bank = np.full(2, 0.1, dtype=np.float64)
        state = np.zeros(2, dtype=np.float64)
        wide = dummy.astype(np.float64)
        _formant_bank(wide, bank, bank, bank, state, state.copy(), bank, output)
        envelopes = np.zeros((2, 8), dtype=np.float64)
        _vocoder_analyse(wide, bank, bank, bank, state.copy(), state.copy(),
                         bank.copy(), 0.1, 0.01, 1.0, envelopes)
        _vocoder_synthesise(wide, wide, 1, bank, bank, bank, state.copy(),
                            state.copy(), envelopes, bank, output)
        _one_euro(wide, output, 1.0 / DEFAULT_SAMPLE_RATE, 1.0, 1.0, 1.0,
                  0.0, 0.0, 0.0, 0.0)
        line = np.zeros(64, dtype=np.float64)
        taps = np.full(wide.shape[0], 8.0)
        zeros = np.zeros(wide.shape[0])
        for shape in (0, 1):
            _delay_kernel(wide, line, 0, taps, zeros, zeros, output,
                          0.0, 0.0, 8.0, 8.0, 0.01, shape, 0)
        allpass = np.zeros(2, dtype=np.float64)
        gains = np.full(wide.shape[0], 0.9)
        _string_kernel(wide, line, 0, line.copy(), 0, taps, gains, zeros,
                       0.2, 0.1, allpass, allpass.copy(), 1.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, output)
        _modal_kernel(wide, wide.copy(), bank.copy(), bank.copy(),
                      bank.copy(), bank.copy(), state.copy(), state.copy(),
                      0.0, 0.0, output, state.copy(), 0.0, 0.0, 0.0)
        breath = np.full(wide.shape[0], 0.8)
        for shape in (0, 1):
            _wind_kernel(breath, zeros, line, 0, line.copy(), 0,
                         line.copy(), 0, taps, taps, zeros,
                         -0.3, 0.6, 0.6, shape,
                         0.0, 0.0, 0.0, 0.0, 0.0, output)
        _bow_kernel(breath, breath, line.copy(), line.copy(), 0,
                    taps, taps, zeros, 0.995, 0.0, 0.0, 0.0, output)
        _brass_kernel(breath, zeros, line, 0, line.copy(), 0, taps,
                      zeros, 1.9, -0.98, 0.04, 0.5,
                      0.95, 0.3, 0.08, 1.8, -0.88, 0.05,
                      1.5, -0.9, 0.06, 0.2, 0.8, 0.8,
                      line.copy(), 40, 0.6, 0.7,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0, 0.0, 0.0, output)
        _rub_kernel(breath, breath.copy(), breath.copy(), bank.copy(),
                    bank.copy(), bank.copy(), bank.copy(), state.copy(),
                    state.copy(), state.copy(), 0.995, 0.0, 0.0, output,
                    0.2, 0.05, 0.3, 0.5)
        members = np.full(8, 0.5)
        _shaker_kernel(breath, 0.01, 0.999, 0.99, 0.4, 1.0, 0.3, members,
                       0.9, members.copy(), members.copy(), members.copy(),
                       0.0, members.copy(), members.copy(), members.copy(),
                       members.copy(), members.copy(), members.copy(),
                       members.copy(), np.uint64(12345),
                       output.copy(), output,
                       1.0, 1.0, 1.0, 0.0, 1.0, 0.01, 1.1, 0.0, 0.0,
                       0.0, 0.5, 40.0, 1.0, 1, 1.25, members.copy(), 0.0)
        _whoosh_kernel(breath, breath.copy(), breath.copy(), 0.4,
                       0.99, 0.02, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0,
                       np.uint64(99), output)
        _noise_kernel(breath, 0.35, 0.5, 0.6, 0.01, 0.001, 0.002,
                      0.5, 0.1, 0.2, 0.001, 0.99, 0.9, 0.3, 0.999,
                      0.99, 1.5, -0.87, 0.06, 0.05, 0.3, 1.2, 0.05,
                      0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      np.uint64(7), output)
        _bounce_kernel(breath, 1.0e-6, 0.7, 0.9, 100.0, 1000.0, 1.0e-4,
                       0.5, 0.0, 1.0, 0.0, 0.0, 0.0,
                       np.uint64(3), output)
        _bubbles_kernel(breath, 0.002, 500.0, 1.0, 0.5, 0.3,
                        0.3, 0.3, 0.5, 1.0, 0.99, 0.05, 44100.0,
                        state.copy(), state.copy(), state.copy(),
                        state.copy(), state.copy(), state.copy(),
                        state.copy(), state.copy(), state.copy(),
                        state.copy(), state.copy(), state.copy(),
                        0.0, 0.0, 1.0, np.uint64(21), output)
        _motor_kernel(breath, 0.001, 4, 5, 0.5, 0.4, 0.3, state.copy(),
                      0.2, 0.35, 0.3, 0.2, 0.1, 6.7, 0.97,
                      0.4, 0.96, 0.05, 0.1, 1000.0, 2000.0,
                      0.3, 0.02, 0.01, 0.5,
                      1.9, -0.97, 0.03, 1.9, -0.96, 0.04,
                      0.05, 0.0, -1, -1, 1.0, 1.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.4, 0.0,
                      0.0, 0.1, 0.0, 0.0, 0.99, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.99, 0.0, 0.0,
                      np.uint64(5), output)
        _drum_kernel(breath, breath.copy(), bank.copy(), bank.copy(),
                     bank.copy(), bank.copy(), bank.copy(),
                     state.copy(), state.copy(),
                     0.2, 0.01, 0.5, 0.1, 0.015, 0.3,
                     0.1, 0.1, 1.9, -0.92, 0.99,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     np.uint64(11), output)
        ap = np.zeros(3)
        _strain_kernel(wide, 0.01, 0.5, 0.6, 10.0, 0.5, 100.0, 1.0, 0.2,
                       0.3, 3.0, 0.05, 0.3, 20.0, 44100.0, 0.2,
                       0.3, 0.5, 0.3, 0.01, 1.0, 1.0, 0.9,
                       10.0, 0.01, 0.5, 1.7,
                       bank.copy(), bank.copy(),
                       bank.copy(), bank.copy(), bank.copy(), bank.copy(),
                       bank.copy(),
                       state.copy(), state.copy(), ap, ap.copy(),
                       0.0, 0.0, 0.01, 0.0, 0.0, 0.99, 0.0, 1.0, 0.0,
                       0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0,
                       0.99, 100.0, 0.0,
                       1.0e-6, np.uint64(777), 0.2, output.copy(),
                       output)
        quads = np.zeros((4, 5))
        quads[:, 0] = 1.0
        _clean_kernel(wide, quads, np.zeros((4, 2)), output)
        _svf_ready.set()
    except Exception as error:
        print('synth_core: filter kernel warm-up failed (' + str(error) + ')')


_warm_up_started = False


def start_filter_warm_up():
    global _warm_up_started
    if _warm_up_started or _svf_ready.is_set():
        return
    _warm_up_started = True
    threading.Thread(target=_warm_up_filter, daemon=True,
                     name='synth-filter-warmup').start()


class VcfUnit(Unit):
    """Resonant multimode filter with true per-sample cutoff modulation.

    Stereo when something is patched to the right inlet. The cutoff curve and
    the resonance coefficient -- the tan() prewarp and the rest of the setup,
    which is most of the work outside the kernel -- are computed once and used
    for both channels; only the recursive kernel and its two state variables
    are per channel. A pair of separate vcf~ would do that setup twice and
    still leave you to keep two cutoff cords in step.

    Unpatched, the right inlet costs a list check and the right outlet carries
    the left channel.
    """

    MODES = ('lowpass', 'highpass', 'bandpass', 'notch')

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.right_in = self.new_inlet()
        self.cutoff_in = self.new_inlet(base=1000.0)          # Hz
        self.tracking_in = self.new_inlet(base=0.0)           # octaves
        self.resonance_in = self.new_inlet(base=0.0, minimum=0.0, maximum=0.99)
        self.drive_in = self.new_inlet(base=1.0, minimum=0.0)

        self.mode = 0
        self._ic1 = 0.0
        self._ic2 = 0.0
        self._ic1_right = 0.0
        self._ic2_right = 0.0

        self.out = self.new_outlet()
        self.right = self.new_outlet()
        self._g = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._k = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._x = np.zeros(MAX_BLOCK, dtype=np.float32)
        self._x_right = np.zeros(MAX_BLOCK, dtype=np.float32)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self._ic1 = 0.0
        self._ic2 = 0.0
        self._ic1_right = 0.0
        self._ic2_right = 0.0

    def _mirror(self, frames):
        """Right carries the left channel when nothing is patched to it."""
        if self.out.constant:
            self.right.set_constant(self.out.value)
            return
        np.copyto(self.right.data[:frames], self.out.data[:frames])
        self.right.constant = False

    def _drive_into(self, signal, scratch, drive, frames):
        """Copy a channel in, saturating it on the way if drive is up."""
        np.copyto(scratch, signal.array(frames))
        if drive.constant:
            if drive.value != 1.0:
                np.multiply(scratch, drive.value, out=scratch)
                np.tanh(scratch, out=scratch)
        else:
            np.multiply(scratch, drive.data[:frames], out=scratch)
            np.tanh(scratch, out=scratch)
        return scratch


    def bypass_pairs(self):
        if self.right_in.sources:
            return ((self.signal_in, self.out), (self.right_in, self.right))
        return ((self.signal_in, self.out), (self.signal_in, self.right))

    def render(self, frames):
        signal = self.signal_in.eval(frames)
        cutoff = self.cutoff_in.eval(frames)
        tracking = self.tracking_in.eval(frames)
        resonance = self.resonance_in.eval(frames)
        drive = self.drive_in.eval(frames)

        out = self.out
        buffer = out.data[:frames]
        stereo = bool(self.right_in.sources)

        if signal.constant and signal.value == 0.0 and not stereo:
            out.set_constant(0.0)
            self.right.set_constant(0.0)
            return

        source = self._drive_into(signal, self._x[:frames], drive, frames)
        right_source = None
        if stereo:
            right_source = self._drive_into(self.right_in.eval(frames),
                                            self._x_right[:frames], drive,
                                            frames)

        if not _svf_ready.is_set():
            # Kernel still compiling (or numba missing): pass audio rather
            # than stall the callback or emit a click.
            np.copyto(buffer, source)
            out.constant = False
            if stereo:
                np.copyto(self.right.data[:frames], right_source)
                self.right.constant = False
            else:
                self._mirror(frames)
            return

        g = self._g[:frames]
        if cutoff.constant:
            g[:] = cutoff.value
        else:
            np.copyto(g, cutoff.data[:frames], casting='unsafe')

        if tracking.constant:
            if tracking.value != 0.0:
                g *= 2.0 ** tracking.value
        else:
            scratch = self._scratch[:frames]
            np.multiply(tracking.data[:frames], math.log(2.0), out=scratch,
                        casting='unsafe')
            np.exp(scratch, out=scratch)
            g *= scratch

        # tan() blows up at Nyquist; keep the prewarped cutoff in range.
        np.clip(g, 1.0, self.sample_rate * 0.49, out=g)
        g *= math.pi / self.sample_rate
        np.tan(g, out=g)

        k = self._k[:frames]
        if resonance.constant:
            k[:] = 2.0 - 2.0 * min(0.99, max(0.0, resonance.value))
        else:
            scratch = self._scratch[:frames]
            np.copyto(scratch, resonance.data[:frames], casting='unsafe')
            np.clip(scratch, 0.0, 0.99, out=scratch)
            np.multiply(scratch, -2.0, out=k)
            k += 2.0
        np.clip(k, 0.02, 2.0, out=k)

        result = self._y[:frames]
        self._ic1, self._ic2 = _svf_kernel(source, g, k, self._ic1, self._ic2,
                                           self.mode, result)
        np.copyto(buffer, result, casting='unsafe')
        out.constant = False

        if not stereo:
            self._mirror(frames)
            return

        # Same coefficients, its own state: the two channels are the same
        # filter, not two filters that happen to be set alike.
        self._ic1_right, self._ic2_right = _svf_kernel(
            right_source, g, k, self._ic1_right, self._ic2_right, self.mode,
            result)
        np.copyto(self.right.data[:frames], result, casting='unsafe')
        self.right.constant = False


# ----------------------------------------------------------------------------
# delay~  --  a buffer read behind itself
# ----------------------------------------------------------------------------

class DelayUnit(Unit):
    """Delay line with damped feedback and an audio-rate delay time.

    Feedback is part of the unit rather than something you patch, and it has
    to be: a cord from the outlet back to the inlet is a cycle, and the
    compiler runs a cycle a block late, so the shortest delay a patched loop
    can make is one block -- around 12 ms. Everything interesting below that
    is therefore only reachable from inside. A few ms with feedback is a comb
    filter; a few ms with feedback and damping is a plucked string; under a
    millisecond, modulated, is a flanger.

    'damping' is a one pole in the loop, so each repeat is darker than the one
    before. At 0 it is transparent and a clean delay stays clean. It is what
    makes a long feedback decay like something in a room rather than build to
    a shriek, and it is the difference between a comb filter and a string.

    Feedback past 1 is allowed. The loop has a soft stop that is exactly
    linear below 1.5 and bends above it, so a delay pushed into oscillation
    settles at a level instead of running away, and ordinary levels are not
    coloured on the way.

    'time' is an audio-rate inlet and the read is interpolated, so it can be
    modulated as hard as you like. In 'slide' the read head moves through the
    buffer and the pitch moves with it -- tape, and doppler: effort driving
    the delay time becomes pitch. In 'fade' two heads take turns picking up
    the new time and crossfade, so the time changes without the pitch, at the
    cost of a little comb while the two disagree. Standing still the two modes
    are identical.

    'freeze' stops the input and loops what is in the buffer.

    Stereo when something is patched to the right inlet: two buffers, one set
    of times and gains, so the channels cannot drift apart.
    """

    SLIDE, FADE = 0, 1
    MODES = ('slide', 'fade')
    # How long a crossfade takes when the delay time moves in 'fade'.
    FADE_SECONDS = 0.02

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.right_in = self.new_inlet()
        self.time_in = self.new_inlet(base=0.25, minimum=0.0)      # seconds
        self.feedback_in = self.new_inlet(base=0.0, minimum=-1.2, maximum=1.2)
        self.damping_in = self.new_inlet(base=0.0, minimum=0.0, maximum=0.999)
        self.freeze_in = self.new_inlet(base=0.0)

        self.mode = DelayUnit.SLIDE
        self.max_delay = 2.0
        self._allocate(self.max_delay)

        self.out = self.new_outlet()
        self.right = self.new_outlet()
        self._samples = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._feedback = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._damping = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._work = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._input = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._entering = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._last_tap = None
        self._write_fade = 0
        self._warned = False

    # -- buffers -------------------------------------------------------------

    def _allocate(self, seconds):
        size = max(64, int(self.sample_rate * max(0.001, seconds)) + 4)
        self.left_line = np.zeros(size, dtype=np.float64)
        self.right_line = np.zeros(size, dtype=np.float64)
        self._write_left = 0
        self._write_right = 0
        self._low_left = 0.0
        self._low_right = 0.0
        self._phase_left = 0.0
        self._phase_right = 0.0
        # Zero means 'not started': the kernel adopts the first delay it is
        # asked for rather than gliding to it from an arbitrary place.
        self._head_a_left = 0.0
        self._head_b_left = 0.0
        self._head_a_right = 0.0
        self._head_b_right = 0.0

    def set_max_delay(self, seconds):
        """Main thread: resize the line. Whole new buffers, assigned at once."""
        seconds = max(0.001, float(seconds))
        if abs(seconds - self.max_delay) < 1.0e-6:
            return
        self.max_delay = seconds
        self._allocate(seconds)

    def reset(self):
        self.left_line[:] = 0.0
        self.right_line[:] = 0.0
        self._low_left = 0.0
        self._low_right = 0.0

    # -- rendering -----------------------------------------------------------



    def deactivate(self):
        # Bypassing drops the tail. Keeping it would mean going on writing,
        # which is most of the work this switch exists to avoid, and coming
        # back to a line with a hole in it clicks.
        self.reset()
        # An emptied line still has one edge in it: the boundary between the
        # zeros and the first thing written after coming back, which the read
        # head reaches one delay time later -- long after any fade of the
        # output has finished, so it arrives unprotected. Ramping what is
        # written removes the edge itself rather than trying to cover it.
        self._write_fade = int(Unit.GATE_SECONDS * self.sample_rate)

    def bypass_pairs(self):
        if self.right_in.sources:
            return ((self.signal_in, self.out), (self.right_in, self.right))
        return ((self.signal_in, self.out), (self.signal_in, self.right))

    def render(self, frames):
        signal = self.signal_in.eval(frames)
        right_in = self.right_in.eval(frames)
        time = self.time_in.eval(frames)
        feedback = self.feedback_in.eval(frames)
        damping = self.damping_in.eval(frames)
        freeze = self.freeze_in.eval(frames)

        size = self.left_line.shape[0]
        taps = self._samples[:frames]
        if time.constant:
            # A knob, held between GUI frames. Jumping the read point to the
            # new value for the whole block would move it by tens of samples
            # at a stroke -- a step in the output, once a block, for as long
            # as the knob is moving. Ramped across the block it is a moment of
            # pitch instead, which is what moving a tape head sounds like.
            target = time.value * self.sample_rate
            start = target if self._last_tap is None else self._last_tap
            np.multiply(_INDEX_RAMP[:frames], (target - start) / frames,
                        out=taps)
            taps += start
            self._last_tap = target
        else:
            np.multiply(time.data[:frames], self.sample_rate, out=taps,
                        casting='unsafe')
            self._last_tap = float(taps[-1])
        np.nan_to_num(taps, copy=False, nan=1.0, posinf=0.0, neginf=0.0)
        np.clip(taps, 1.0, size - 3.0, out=taps)

        gains = self._feedback[:frames]
        if feedback.constant:
            gains[:] = feedback.value
        else:
            np.copyto(gains, feedback.data[:frames], casting='unsafe')
        np.clip(gains, -1.2, 1.2, out=gains)

        dark = self._damping[:frames]
        if damping.constant:
            dark[:] = damping.value
        else:
            np.copyto(dark, damping.data[:frames], casting='unsafe')
        np.clip(dark, 0.0, 0.999, out=dark)

        held = 1 if (freeze.value >= 0.5 if freeze.constant
                     else float(freeze.data[0]) >= 0.5) else 0
        stereo = len(self.right_in.sources) > 0
        step = 1.0 / max(1.0, DelayUnit.FADE_SECONDS * self.sample_rate)

        entering = None
        if self._write_fade > 0:
            total = Unit.GATE_SECONDS * self.sample_rate
            count = min(frames, self._write_fade)
            entering = self._entering[:frames]
            entering[:] = 1.0
            np.multiply(_INDEX_RAMP[:count],
                        1.0 / total, out=entering[:count])
            entering[:count] += (total - self._write_fade) / total
            np.clip(entering, 0.0, 1.0, out=entering)

        source = self._input[:frames]
        if signal.constant:
            source[:] = signal.value
        else:
            np.copyto(source, signal.data[:frames], casting='unsafe')
        if entering is not None:
            source *= entering

        work = self._work[:frames]
        if _HAVE_NUMBA and _svf_ready.is_set():
            (self._write_left, self._low_left, self._phase_left,
             self._head_a_left, self._head_b_left) = _delay_kernel(
                source, self.left_line, self._write_left, taps, gains, dark,
                work, self._low_left, self._phase_left, self._head_a_left,
                self._head_b_left, step, self.mode, held)
        else:
            self._render_plain(source, self.left_line, taps, work, frames,
                               left=True)
        np.copyto(self.out.data[:frames], work, casting='unsafe')
        self.out.constant = False

        if not stereo:
            np.copyto(self.right.data[:frames], self.out.data[:frames])
            self.right.constant = False
            self._spend_write_fade(frames)
            return

        if right_in.constant:
            source[:] = right_in.value
        else:
            np.copyto(source, right_in.data[:frames], casting='unsafe')
        if entering is not None:
            source *= entering
        if _HAVE_NUMBA and _svf_ready.is_set():
            (self._write_right, self._low_right, self._phase_right,
             self._head_a_right, self._head_b_right) = _delay_kernel(
                source, self.right_line, self._write_right, taps, gains, dark,
                work, self._low_right, self._phase_right, self._head_a_right,
                self._head_b_right, step, self.mode, held)
        else:
            self._render_plain(source, self.right_line, taps, work, frames,
                               left=False)
        np.copyto(self.right.data[:frames], work, casting='unsafe')
        self.right.constant = False
        self._spend_write_fade(frames)

    def _spend_write_fade(self, frames):
        if self._write_fade > 0:
            self._write_fade = max(0, self._write_fade - frames)

    def _render_plain(self, source, line, taps, out, frames, left):
        """Delay without feedback, for before the kernel is compiled.

        numba's first call costs about a second and cannot happen on the audio
        thread, so for the moment before it is ready -- and permanently, if
        numba is not installed -- the line still delays, vectorised, and only
        the loop is missing. Reads are clamped to a block so that a read
        cannot want a sample this same block is still writing.
        """
        if not self._warned and not _HAVE_NUMBA:
            self._warned = True
            print('synth_core: numba unavailable, delay~ runs without feedback')
        size = line.shape[0]
        write = self._write_left if left else self._write_right

        positions = np.clip(taps, float(frames), size - 3.0)
        read = (write + _INDEX_RAMP[:frames] - 1.0) - positions
        np.mod(read, float(size), out=read)

        base = read.astype(np.int64)
        frac = read - base
        gather = np.empty((4, frames))
        for offset in range(-1, 3):
            np.take(line, (base + offset) % size, out=gather[offset + 1])
        y0, y1, y2, y3 = gather
        c1 = 0.5 * (y2 - y0)
        c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
        c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)
        np.copyto(out, ((c3 * frac + c2) * frac + c1) * frac + y1)

        index = (write + np.arange(frames)) % size
        line[index] = source
        if left:
            self._write_left = int((write + frames) % size)
        else:
            self._write_right = int((write + frames) % size)


# ----------------------------------------------------------------------------
# fold~  --  a nonlinearity that does not fizz
# ----------------------------------------------------------------------------

def _fold_shape(v, shape):
    """The transfer function itself.

    Ordered gentlest to harshest, which is measurable rather than a matter of
    taste: driven six times over, a sine into each of these comes out with a
    spectral centroid of about 140 Hz, 155, 400 and 740, and with 11, 13, 68
    and 87 percent of its energy above the fundamental.
    """
    if shape == 1:
        return np.clip(v, -1.0, 1.0)
    if shape == 2:
        return np.sin(v)
    if shape == 3:
        p = np.mod(v + 1.0, 4.0)
        return np.where(p < 2.0, p - 1.0, 3.0 - p)
    return np.tanh(v)


def _fold_integral(v, shape):
    """Its antiderivative, which is what makes the anti-aliasing possible.

    Every shape here was chosen partly because this exists in closed form. A
    curve without one can still be drawn in shaper~; it just cannot be
    band-limited this cheaply.
    """
    if shape == 1:
        magnitude = np.abs(v)
        return np.where(magnitude <= 1.0, v * v * 0.5, magnitude - 0.5)
    if shape == 2:
        return -np.cos(v)
    if shape == 3:
        p = np.mod(v + 1.0, 4.0)
        return np.where(p < 2.0, p * p * 0.5 - p, 3.0 * p - p * p * 0.5 - 4.0)
    # log(cosh(v)), written so that it does not overflow for loud input.
    magnitude = np.abs(v)
    return magnitude + np.log1p(np.exp(-2.0 * magnitude)) - math.log(2.0)


_OVERSAMPLE_FIR = {}


def _oversample_filter(factor):
    """Half-band-ish lowpass for going up and coming back down.

    Designed once per factor and shared: it is the same filter both ways, and
    at 63 taps it is cheap enough to run twice per sample at the raised rate.
    """
    if factor not in _OVERSAMPLE_FIR:
        if scipy_signal is None:
            _OVERSAMPLE_FIR[factor] = None
        else:
            _OVERSAMPLE_FIR[factor] = scipy_signal.firwin(63, 0.9 / factor)
    return _OVERSAMPLE_FIR[factor]


class FoldUnit(Unit):
    """Saturation and wavefolding, with the aliasing taken out.

    Any nonlinearity makes harmonics above the ones it was given, and the ones
    that land past Nyquist fold back down as tones that are not related to the
    pitch and do not move with it. That is the fizz around bright distorted
    sound, and it is why shaper~ warns about it: a drawn curve cannot be
    band-limited, because band-limiting needs to know the curve's integral.

    These four do have integrals, in closed form, so this uses the standard
    first-order trick: instead of asking what the curve does at this sample,
    ask what it did *on average between the last sample and this one*, which
    is the difference of the antiderivative over the difference of the input.
    A corner is then crossed rather than landed on, and what would have been
    aliasing is mostly not generated. It costs one extra evaluation and half a
    sample of delay, and it is worth about 25 dB where it matters. Off, this
    is an ordinary waveshaper again -- worth hearing the difference.

    'shape' runs along the four of them and, like formant~'s vowel, runs
    *between* them rather than switching:

      0 tanh   soft saturation. Odd harmonics, gently, and a ceiling.
      1 clip   hard clipping. The same ceiling arrived at abruptly, so many
               more harmonics, and the classic sound of too much gain.
      2 sine   sine folding. Past the limit the signal turns back on itself
               instead of stopping, but smoothly -- so the harmonics do not
               merely grow with drive, they sweep and change places, without
               the very high ones a corner would make.
      3 fold   triangle wavefolding. The same turning back, with corners: the
               brightest of the four by a long way, and the one whose timbre
               moves most under a drive envelope.

    The order is not arbitrary, and not a matter of taste either -- it is the
    order the four come in when you measure them. Each step is one decision:
    0 to 1 is how sharp the knee is, 1 to 2 is whether the signal stops at the
    limit or turns back through it, 2 to 3 is how sharp that turn is. So the
    run only ever gets harsher, which is what a control that is swept upwards
    should do, and 2.5 is a fold with its corners taken off -- the ground
    between a sine's roundness and a triangle's bright edges. The position is
    an audio-rate inlet like any other: a shape that moves with what drives it.

    Mixing two curves is exactly what lets the anti-aliasing survive it:
    integration is linear, so the antiderivative of the mixture is the mixture
    of the antiderivatives, and the trick below still applies at any point
    along the run. Only two of the four are ever evaluated -- the pair the
    block sits between.

    'bias' pushes the signal off centre before shaping. A symmetrical curve
    makes only odd harmonics and can sound hollow; asymmetry brings in the
    even ones, which is most of what 'warm' means. It also makes DC, so there
    is a blocker on the way out.

    How much anti-aliasing is enough depends on the shape, and measurably so.
    On a 3 kHz tone driven hard, the aliasing sits at about -25 dB for tanh and
    -21 dB for hard clip before anything is done about it, and the
    antiderivative trick takes 6 to 8 dB off that -- audible, and enough for
    saturation. Folding is a different matter: it makes far more harmonics and
    they run far higher, so the same tone folded lands its aliasing at roughly
    the level of the signal itself, and 8 dB off that is still ruin. That is
    what 'oversample' is for. Running the shaper at two or four times the rate
    and filtering on the way back costs about four times as much and takes 30
    dB or more off, which is the difference between a folder that can be swept
    and one that can only be used quietly. Leave it at 1 for saturation; raise
    it for folding, or whenever a bright source starts to fizz.
    """

    SHAPES = ('tanh', 'clip', 'sine', 'fold')
    FACTORS = (1, 2, 4)
    # Below this the antiderivative difference is two nearly equal numbers
    # over a nearly zero one, so fall back to reading the curve directly.
    ADAA_FLOOR = 1.0e-5

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.right_in = self.new_inlet()
        self.drive_in = self.new_inlet(base=1.0, minimum=0.0)
        self.bias_in = self.new_inlet(base=0.0)
        self.level_in = self.new_inlet(base=1.0)
        self.shape_in = self.new_inlet(
            base=0.0, minimum=0.0, maximum=float(len(FoldUnit.SHAPES) - 1))

        self.antialias = True
        self.block_dc = True
        self.oversample = 1

        self.out = self.new_outlet()
        self.right = self.new_outlet()
        self._previous_left = 0.0
        self._previous_right = 0.0
        self._dc_left = np.zeros(1)
        self._dc_right = np.zeros(1)
        self._driven = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._earlier = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._morph = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._last_shape = None
        self._blend_low = 0
        self._blend_weight = 0.0
        self._raised = None
        self._up_state = [None, None]
        self._down_state = [None, None]

    def set_oversample(self, factor):
        """Main thread: change the internal rate, and size the buffer for it."""
        factor = int(factor)
        if factor not in FoldUnit.FACTORS or factor == self.oversample:
            return
        self.oversample = factor
        if factor > 1:
            self._raised = np.zeros(MAX_BLOCK * factor, dtype=np.float64)
        else:
            self._raised = None
        self._up_state = [None, None]
        self._down_state = [None, None]

    def reset(self):
        self._previous_left = 0.0
        self._previous_right = 0.0
        self._dc_left = np.zeros(1)
        self._dc_right = np.zeros(1)
        self._up_state = [None, None]
        self._down_state = [None, None]

    def _blend(self, values, integral, weight):
        """The shape, or its antiderivative, somewhere between two of them.

        Integrating is linear, so the antiderivative of a mixture is the
        mixture of the antiderivatives -- which is why the shapes can be
        morphed without giving up the anti-aliasing. A curve that had to be
        blended some other way would have no integral to speak of.
        """
        low = self._blend_low
        table = _fold_integral if integral else _fold_shape
        if isinstance(weight, float):
            if weight <= 0.0:
                return table(values, low)
            if weight >= 1.0:
                return table(values, low + 1)
        under = table(values, low)
        over = table(values, low + 1)
        return under + (over - under) * weight

    def _resolve_shape(self, frames):
        """Where along the list of shapes we are, and between which two.

        Ramped across the block when it comes from a knob, for the same reason
        every other setting here is: a curve swapped at a block boundary is a
        step in the output, once per frame for as long as it is being moved.
        """
        shape = self.shape_in.eval(frames)
        highest = float(len(FoldUnit.SHAPES) - 1)
        morph = self._morph[:frames]
        if shape.constant:
            target = min(highest, max(0.0, shape.value))
            start = target if self._last_shape is None else self._last_shape
            np.multiply(_INDEX_RAMP[:frames], (target - start) / frames,
                        out=morph)
            morph += start
            self._last_shape = target
        else:
            np.copyto(morph, shape.data[:frames], casting='unsafe')
            self._last_shape = float(morph[-1])
        np.clip(morph, 0.0, highest, out=morph)

        # Only ever two shapes are needed: the pair the block sits between.
        low = int(math.floor(float(morph.min())))
        self._blend_low = max(0, min(low, len(FoldUnit.SHAPES) - 2))
        weight = np.clip(morph - self._blend_low, 0.0, 1.0)
        first = float(weight[0])
        if float(weight.min()) == first and float(weight.max()) == first:
            self._blend_weight = first          # a scalar costs one curve less
        else:
            self._blend_weight = weight


    def bypass_pairs(self):
        if self.right_in.sources:
            return ((self.signal_in, self.out), (self.right_in, self.right))
        return ((self.signal_in, self.out), (self.signal_in, self.right))

    def render(self, frames):
        signal = self.signal_in.eval(frames)
        right_in = self.right_in.eval(frames)
        drive = self.drive_in.eval(frames)
        bias = self.bias_in.eval(frames)
        level = self.level_in.eval(frames)
        self._resolve_shape(frames)

        stereo = len(self.right_in.sources) > 0
        result = self._shape_channel(signal, drive, bias, level, frames, True)
        np.copyto(self.out.data[:frames], result, casting='unsafe')
        self.out.constant = False

        if not stereo:
            np.copyto(self.right.data[:frames], self.out.data[:frames])
            self.right.constant = False
            return
        result = self._shape_channel(right_in, drive, bias, level, frames,
                                     False)
        np.copyto(self.right.data[:frames], result, casting='unsafe')
        self.right.constant = False

    def _shape_channel(self, signal, drive, bias, level, frames, left):
        driven = self._driven[:frames]
        if signal.constant:
            driven[:] = signal.value
        else:
            np.copyto(driven, signal.data[:frames], casting='unsafe')
        if drive.constant:
            if drive.value != 1.0:
                driven *= drive.value
        else:
            driven *= drive.data[:frames]
        if bias.constant:
            if bias.value != 0.0:
                driven += bias.value
        else:
            driven += bias.data[:frames]

        factor = self.oversample
        if factor > 1 and scipy_signal is not None:
            shaped = self._shape_raised(driven, frames, factor, left)
        else:
            shaped = self._shape_at_rate(driven, left, self._blend_weight)

        if self.block_dc:
            shaped = self._remove_dc(shaped, left)
        if level.constant:
            if level.value != 1.0:
                shaped = shaped * level.value
        else:
            shaped = shaped * level.data[:frames]
        return shaped

    def _shape_at_rate(self, driven, left, weight):
        """The curve, applied to whatever rate these samples are at."""
        if not self.antialias:
            result = self._blend(driven, False, weight)
            if left:
                self._previous_left = float(driven[-1])
            else:
                self._previous_right = float(driven[-1])
            return result

        count = driven.shape[0]
        earlier = np.empty(count)
        earlier[0] = self._previous_left if left else self._previous_right
        earlier[1:] = driven[:-1]
        difference = driven - earlier
        # Where two samples are nearly equal the quotient is two almost equal
        # numbers over an almost zero one, so read the curve at the midpoint
        # instead -- which is the limit the quotient is heading for anyway.
        steady = np.abs(difference) < FoldUnit.ADAA_FLOOR
        safe = np.where(steady, 1.0, difference)
        result = np.where(
            steady,
            self._blend((driven + earlier) * 0.5, False, weight),
            (self._blend(driven, True, weight)
             - self._blend(earlier, True, weight)) / safe)
        if left:
            self._previous_left = float(driven[-1])
        else:
            self._previous_right = float(driven[-1])
        return result

    def _shape_raised(self, driven, frames, factor, left):
        """Shape at a multiple of the sample rate, then come back down.

        Zero-stuff, filter, shape, filter, keep every nth. Both filters carry
        their state between blocks, so this streams -- the one-shot resamplers
        would restart their history every block and buzz at the block rate.
        """
        taps = _oversample_filter(factor)
        channel = 0 if left else 1
        raised = self._raised[:frames * factor]
        raised[:] = 0.0
        raised[::factor] = driven

        state = self._up_state[channel]
        if state is None:
            state = np.zeros(len(taps) - 1)
        raised, state = scipy_signal.lfilter(taps * factor, 1.0, raised,
                                             zi=state)
        self._up_state[channel] = state

        # The morph is a control, not audio, so holding each of its values for
        # the run of raised samples it belongs to is exact enough.
        weight = self._blend_weight
        if not isinstance(weight, float):
            weight = np.repeat(weight, factor)
        shaped = self._shape_at_rate(raised, left, weight)

        state = self._down_state[channel]
        if state is None:
            state = np.zeros(len(taps) - 1)
        shaped, state = scipy_signal.lfilter(taps, 1.0, shaped, zi=state)
        self._down_state[channel] = state
        return shaped[::factor]

    def _remove_dc(self, values, left):
        """Bias makes DC, and DC eats headroom without being audible."""
        if scipy_signal is None:
            return values
        pole = math.exp(-2.0 * math.pi * 10.0 / self.sample_rate)
        state = self._dc_left if left else self._dc_right
        filtered, state = scipy_signal.lfilter([1.0, -1.0], [1.0, -pole],
                                               values, zi=state)
        if left:
            self._dc_left = state
        else:
            self._dc_right = state
        return filtered


# ----------------------------------------------------------------------------
# crush~  --  fewer bits, fewer samples
# ----------------------------------------------------------------------------

class CrushUnit(Unit):
    """Bit depth and sample rate reduction.

    Not a curve, which is why it is its own object: quantising is a staircase
    whose step size is fixed in amplitude, and holding is a staircase in time.
    Neither can be drawn as a transfer function, and they sound nothing alike.

    'bits' quantises the amplitude. The error is roughly noise at high
    settings and plainly harmonic at low ones, and unlike tape hiss it is
    loudest when the signal is loud.

    'rate' is the more useful of the two. Holding each sample until the next
    one is due is a sample and hold at audio rate, and the images it makes
    around that rate are inharmonic and do not move with the pitch -- so
    sweeping it against a held note is a whole instrument in itself. Set at or
    above the sample rate it does nothing.

    Both are audio-rate inlets. Neither is anti-aliased, and neither should
    be: the aliasing is the effect.
    """

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.right_in = self.new_inlet()
        self.bits_in = self.new_inlet(base=24.0, minimum=1.0, maximum=24.0)
        self.rate_in = self.new_inlet(base=DEFAULT_SAMPLE_RATE, minimum=1.0)

        self.out = self.new_outlet()
        self.right = self.new_outlet()
        self._phase = 0.0
        self._held_left = 0.0
        self._held_right = 0.0
        self._work = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._steps = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._index = np.zeros(MAX_BLOCK, dtype=np.int64)

    def reset(self):
        self._phase = 0.0
        self._held_left = 0.0
        self._held_right = 0.0


    def bypass_pairs(self):
        if self.right_in.sources:
            return ((self.signal_in, self.out), (self.right_in, self.right))
        return ((self.signal_in, self.out), (self.signal_in, self.right))

    def render(self, frames):
        signal = self.signal_in.eval(frames)
        right_in = self.right_in.eval(frames)
        bits = self.bits_in.eval(frames)
        rate = self.rate_in.eval(frames)

        # Where the held samples fall. A phase runs at the reduction rate and
        # a new sample is taken every time it passes a whole number, so the
        # rate is free to move within the block; the index of the most recent
        # crossing is then carried forward, which is the hold.
        advance = self._steps[:frames]
        if rate.constant:
            advance[:] = rate.value / self.sample_rate
        else:
            np.multiply(rate.data[:frames], 1.0 / self.sample_rate,
                        out=advance, casting='unsafe')
        np.nan_to_num(advance, copy=False, nan=1.0, posinf=1.0, neginf=0.0)
        np.clip(advance, 0.0, 1.0, out=advance)
        np.cumsum(advance, out=advance)
        advance += self._phase
        whole = np.floor(advance)
        self._phase = float(advance[-1] - whole[-1])

        taken = self._index[:frames]
        np.copyto(taken, np.arange(frames))
        fresh = np.empty(frames, dtype=bool)
        fresh[0] = whole[0] >= 1.0
        np.greater(whole[1:], whole[:-1], out=fresh[1:])
        taken[~fresh] = -1
        np.maximum.accumulate(taken, out=taken)

        levels = 0.0
        if bits.constant:
            levels = 2.0 ** (min(24.0, max(1.0, bits.value)) - 1.0)
        stereo = len(self.right_in.sources) > 0

        result = self._crush(signal, taken, bits, levels, frames, True)
        np.copyto(self.out.data[:frames], result, casting='unsafe')
        self.out.constant = False
        if not stereo:
            np.copyto(self.right.data[:frames], self.out.data[:frames])
            self.right.constant = False
            return
        result = self._crush(right_in, taken, bits, levels, frames, False)
        np.copyto(self.right.data[:frames], result, casting='unsafe')
        self.right.constant = False

    def _crush(self, signal, taken, bits, levels, frames, left):
        work = self._work[:frames]
        if signal.constant:
            work[:] = signal.value
        else:
            np.copyto(work, signal.data[:frames], casting='unsafe')

        previous = self._held_left if left else self._held_right
        held = np.where(taken >= 0, work[np.maximum(taken, 0)], previous)
        if left:
            self._held_left = float(held[-1])
        else:
            self._held_right = float(held[-1])

        if not bits.constant:
            steps = np.clip(bits.data[:frames], 1.0, 24.0) - 1.0
            levels = np.exp2(steps)
        if np.all(levels >= 8388608.0):
            return held
        return np.round(held * levels) / levels


# ----------------------------------------------------------------------------
# scaler~ / mix~ / pan~ / audio_out~ / snapshot~
# ----------------------------------------------------------------------------

class ScalerUnit(Unit):
    """Map a signal from one range into another, with curve control.

    Every modulation inlet already does linear scaling through its base and
    depth (base + depth * cv), so this is not needed to hit a range. It exists
    for the two things that arithmetic cannot express: a response curve, and
    an explicit input range so the mapping reads off the node face instead of
    living in a collapsed options panel.

    'exponential' output mode maps equal input steps to equal *ratios* rather
    than equal differences, which is what frequency-like destinations want --
    a linear 200..4200 Hz sweep spends almost all its travel in the top
    octaves, an exponential one is evenly spread.
    """

    LINEAR = 0
    EXPONENTIAL = 1

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.in_low_in = self.new_inlet(base=0.0)
        self.in_high_in = self.new_inlet(base=1.0)
        self.out_low_in = self.new_inlet(base=0.0)
        self.out_high_in = self.new_inlet(base=1.0)
        self.curve_in = self.new_inlet(base=1.0, minimum=0.01, maximum=16.0)

        self.mode = ScalerUnit.LINEAR
        self.clip = True

        self.out = self.new_outlet()
        self._work = np.zeros(MAX_BLOCK, dtype=np.float64)

    @staticmethod
    def _operand(signal, frames):
        """A signal as either a python float or an array, whichever it is."""
        if signal.constant:
            return signal.value
        return signal.data[:frames]

    def _map_scalar(self, x, in_low, in_high, out_low, out_high, curve):
        span = in_high - in_low
        if abs(span) < 1.0e-12:
            span = 1.0e-12
        t = (x - in_low) / span
        if curve != 1.0:
            t = math.copysign(abs(t) ** curve, t)
        if self.mode == ScalerUnit.EXPONENTIAL and out_low > 0.0 and out_high > 0.0:
            value = out_low * (out_high / out_low) ** t
        else:
            value = out_low + t * (out_high - out_low)
        if self.clip:
            low = min(out_low, out_high)
            high = max(out_low, out_high)
            value = min(max(value, low), high)
        return value


    def bypass_pairs(self):
        return ((self.signal_in, self.out),)

    def render(self, frames):
        signal = self.signal_in.eval(frames)
        in_low = self.in_low_in.eval(frames)
        in_high = self.in_high_in.eval(frames)
        out_low = self.out_low_in.eval(frames)
        out_high = self.out_high_in.eval(frames)
        curve = self.curve_in.eval(frames)

        out = self.out

        if (signal.constant and in_low.constant and in_high.constant
                and out_low.constant and out_high.constant and curve.constant):
            out.set_constant(self._map_scalar(
                signal.value, in_low.value, in_high.value,
                out_low.value, out_high.value, curve.value))
            return

        low_in = self._operand(in_low, frames)
        high_in = self._operand(in_high, frames)
        low_out = self._operand(out_low, frames)
        high_out = self._operand(out_high, frames)
        shape = self._operand(curve, frames)

        work = self._work[:frames]
        np.copyto(work, signal.array(frames), casting='unsafe')
        work -= low_in

        span = high_in - low_in
        if np.isscalar(span):
            work /= span if abs(span) > 1.0e-12 else 1.0e-12
        else:
            work /= np.where(np.abs(span) < 1.0e-12, 1.0e-12, span)

        # Sign-preserving power, so a curve on an out-of-range input stays
        # monotonic instead of turning into NaN.
        if np.isscalar(shape):
            if shape != 1.0:
                np.copysign(np.abs(work) ** shape, work, out=work)
        else:
            np.copysign(np.abs(work) ** shape, work, out=work)

        exponential = (self.mode == ScalerUnit.EXPONENTIAL
                       and np.all(low_out > 0.0) and np.all(high_out > 0.0))
        if exponential:
            np.power(high_out / low_out, work, out=work)
            work *= low_out
        else:
            work *= (high_out - low_out)
            work += low_out

        if self.clip:
            np.clip(work, np.minimum(low_out, high_out),
                    np.maximum(low_out, high_out), out=work)

        np.copyto(out.data[:frames], work, casting='unsafe')
        out.constant = False


class MultUnit(Unit):
    """Multiply signals together. Ring modulation, AM, envelope-shaped LFOs.

    Distinct from vca~, which is an *amplifier*: vca~ clamps negative gain (an
    amplifier cannot have negative amplification) and its knob sums with the
    CV. Both behaviours are right for an amp and wrong for a product, so
    multiplying two bipolar signals needs this instead.

    Each input's knob is a plain scale factor on that input, used on its own
    when nothing is patched there. An unpatched input is therefore identity at
    1.0 and never zeroes the product.
    """

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, inputs=2):
        super().__init__(sample_rate)
        # base 0 / depth 1 so eval() yields the bare sum of patched cords; the
        # knob lives in `factors` instead, because here it scales rather than
        # offsets.
        self.signal_inlets = [self.new_inlet() for _ in range(inputs)]
        self.factors = [1.0] * inputs
        self.out = self.new_outlet()


    def bypass_pairs(self):
        # The first inlet is the signal; the rest are what it is multiplied
        # by, so bypass hands the first one back.
        return ((self.signal_inlets[0], self.out),)

    def render(self, frames):
        scalar = 1.0
        varying = []

        for index, inlet in enumerate(self.signal_inlets):
            factor = self.factors[index]
            if not inlet.sources:
                scalar *= factor
                continue
            signal = inlet.eval(frames)
            if signal.constant:
                scalar *= signal.value * factor
            else:
                varying.append((signal, factor))

        out = self.out

        if not varying:
            out.set_constant(scalar)
            return

        buffer = out.data[:frames]
        first, first_factor = varying[0]
        np.multiply(first.data[:frames], first_factor, out=buffer)
        for signal, factor in varying[1:]:
            buffer *= signal.data[:frames]
            if factor != 1.0:
                buffer *= factor
        if scalar != 1.0:
            buffer *= scalar
        out.constant = False


class MixUnit(Unit):
    """Signal inputs with individual levels, plus a master level."""

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, channels=4):
        super().__init__(sample_rate)
        self.channel_inlets = []
        self.level_inlets = []
        for index in range(channels):
            self.channel_inlets.append(self.new_inlet())
            self.level_inlets.append(self.new_inlet(base=1.0))
        self.master_in = self.new_inlet(base=1.0)
        self.out = self.new_outlet()
        self._accumulator = np.zeros(MAX_BLOCK, dtype=np.float32)

    def render(self, frames):
        accumulator = self._accumulator[:frames]
        accumulator[:] = 0.0
        any_varying = False
        constant_total = 0.0

        for signal_inlet, level_inlet in zip(self.channel_inlets, self.level_inlets):
            signal = signal_inlet.eval(frames)
            level = level_inlet.eval(frames)
            if signal.constant and level.constant:
                constant_total += signal.value * level.value
                continue
            any_varying = True
            if level.constant:
                if level.value == 0.0:
                    continue
                accumulator += signal.data[:frames] * level.value
            else:
                accumulator += signal.array(frames) * level.array(frames)

        master = self.master_in.eval(frames)
        out = self.out

        if not any_varying:
            if master.constant:
                out.set_constant(constant_total * master.value)
                return
            np.multiply(master.data[:frames], constant_total,
                        out=out.data[:frames])
            out.constant = False
            return

        if constant_total != 0.0:
            accumulator += constant_total

        buffer = out.data[:frames]
        if master.constant:
            np.multiply(accumulator, master.value, out=buffer)
        else:
            np.multiply(accumulator, master.data[:frames], out=buffer)
        out.constant = False


class PanUnit(Unit):
    """Equal-power stereo panner. Position -1 is hard left, +1 hard right."""

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.position_in = self.new_inlet(base=0.0, minimum=-1.0, maximum=1.0)
        self.left = self.new_outlet()
        self.right = self.new_outlet()
        self._angle = np.zeros(MAX_BLOCK, dtype=np.float64)


    def bypass_pairs(self):
        return ((self.signal_in, self.left), (self.signal_in, self.right))

    def render(self, frames):
        signal = self.signal_in.eval(frames)
        position = self.position_in.eval(frames)

        if position.constant:
            angle = (position.value + 1.0) * (math.pi * 0.25)
            left_gain = math.cos(angle)
            right_gain = math.sin(angle)
            if signal.constant:
                self.left.set_constant(signal.value * left_gain)
                self.right.set_constant(signal.value * right_gain)
                return
            source = signal.data[:frames]
            np.multiply(source, left_gain, out=self.left.data[:frames])
            np.multiply(source, right_gain, out=self.right.data[:frames])
        else:
            angle = self._angle[:frames]
            np.copyto(angle, position.data[:frames], casting='unsafe')
            angle += 1.0
            angle *= math.pi * 0.25
            source = signal.array(frames)
            np.multiply(source, np.cos(angle), out=self.left.data[:frames],
                        casting='unsafe')
            np.multiply(source, np.sin(angle), out=self.right.data[:frames],
                        casting='unsafe')

        self.left.constant = False
        self.right.constant = False


class VuUnit(Unit):
    """Level meter: a tap on the signal, with eyes and no hands.

    A meter is a gauge, so it is not even in the signal path: branch a
    cord into it from anywhere and the chain it watches is untouched by
    construction -- there are no audio outlets to route through. Per
    block it takes the RMS and the absolute peak of each channel and
    smooths them the way a meter needle moves: rising fast enough to
    catch a transient (~25 ms), falling slowly enough to read (~300 ms),
    the peak held for most of a second before it lets go.

    Unpatched, the right channel reads as the left, so a mono tap fills
    both bars rather than leaving one dark.
    """

    ATTACK_SECONDS = 0.025
    RELEASE_SECONDS = 0.3
    PEAK_HOLD_SECONDS = 0.8
    # Linear fraction kept after one second of falling: -12 dB/s, the
    # customary PPM fallback -- readable, without lying for long.
    PEAK_FALL_PER_SECOND = 0.25

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.right_in = self.new_inlet()
        self.levels = [0.0, 0.0]        # smoothed rms per channel
        self.peaks = [0.0, 0.0]         # held peaks per channel
        self._hold = [0.0, 0.0]
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def _watch(self, buffer, channel, seconds):
        scratch = self._scratch[:buffer.shape[0]]
        np.multiply(buffer, buffer, out=scratch, casting='unsafe')
        rms = math.sqrt(float(np.mean(scratch)))
        peak = math.sqrt(float(scratch.max()))
        smoothed = self.levels[channel]
        if rms > smoothed:
            k = 1.0 - math.exp(-seconds / VuUnit.ATTACK_SECONDS)
        else:
            k = 1.0 - math.exp(-seconds / VuUnit.RELEASE_SECONDS)
        self.levels[channel] = smoothed + (rms - smoothed) * k
        if peak >= self.peaks[channel]:
            self.peaks[channel] = peak
            self._hold[channel] = 0.0
        else:
            self._hold[channel] += seconds
            if self._hold[channel] > VuUnit.PEAK_HOLD_SECONDS:
                self.peaks[channel] *= VuUnit.PEAK_FALL_PER_SECOND ** seconds

    def render(self, frames):
        seconds = frames / self.sample_rate
        signal = self.signal_in.eval(frames)
        self._watch(signal.array(frames), 0, seconds)
        if self.right_in.sources:
            self._watch(self.right_in.eval(frames).array(frames), 1, seconds)
        else:
            self.levels[1] = self.levels[0]
            self.peaks[1] = self.peaks[0]


def _clean_kernel_source(x, coeffs, states, out):
    """Four biquads in cascade, sample by sample: the conditioning filter.

    coeffs is [section, (b0, b1, b2, a1, a2)], states [section, 2], both
    per channel. Transposed direct form II, which keeps the state small
    and well-behaved when coefficients move between blocks.
    """
    sections = coeffs.shape[0]
    for i in range(x.shape[0]):
        value = x[i]
        for s in range(sections):
            b0 = coeffs[s, 0]
            b1 = coeffs[s, 1]
            b2 = coeffs[s, 2]
            a1 = coeffs[s, 3]
            a2 = coeffs[s, 4]
            y = b0 * value + states[s, 0]
            states[s, 0] = b1 * value - a1 * y + states[s, 1]
            states[s, 1] = b2 * value - a2 * y
            value = y
        out[i] = value
    return 0


if _HAVE_NUMBA:
    _clean_kernel = njit(cache=True, fastmath=True)(_clean_kernel_source)
else:
    _clean_kernel = _clean_kernel_source


class CleanUnit(Unit):
    """Conditioner: takes off what no patch means to keep.

    The physical models are honest about infrasound -- a bow at 5 Hz, a
    drive leaning on a low mode -- and honesty eats headroom. This is the
    hygiene stage of a channel strip: a fourth-order Butterworth highpass
    under the music and the same lowpass over it, 24 dB per octave each,
    flat and resonance-free in between. Not an instrument and not an EQ;
    it removes what was never meant, and passes everything that was.

    'low cut' defaults just under the lowest audible fundamental; pull it
    down when the subsonics ARE the material. 'high cut' catches the
    aliasing-adjacent fizz of hard folding and crushing. Both are inlets,
    so a patch can duck its own mud.

    Stereo the way vcf~ is: one set of coefficients, two channels of
    state, the right outlet carrying the left signal until something is
    patched to the right inlet. Bypassed, the signal passes untouched.
    """

    SECTIONS = 4
    # Butterworth Q pairs for a 4th-order response, one pair per slope.
    Q_PAIR = (0.5411961, 1.3065630)

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.right_in = self.new_inlet()
        self.low_in = self.new_inlet(base=25.0, minimum=5.0, maximum=300.0)
        self.high_in = self.new_inlet(base=16000.0, minimum=1000.0,
                                      maximum=20000.0)
        self.out = self.new_outlet()
        self.right = self.new_outlet()
        self._coeffs = np.zeros((CleanUnit.SECTIONS, 5))
        self._states = np.zeros((CleanUnit.SECTIONS, 2))
        self._states_right = np.zeros((CleanUnit.SECTIONS, 2))
        self._x = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._last_low = 0.0
        self._last_high = 0.0

    def reset(self):
        self._states[:, :] = 0.0
        self._states_right[:, :] = 0.0

    def deactivate(self):
        self.reset()

    def bypass_pairs(self):
        if self.right_in.sources:
            return ((self.signal_in, self.out), (self.right_in, self.right))
        return ((self.signal_in, self.out), (self.signal_in, self.right))

    def _biquad(self, section, kind, frequency, q):
        """RBJ cookbook coefficients, normalized by a0."""
        frequency = min(frequency, self.sample_rate * 0.45)
        w0 = 2.0 * math.pi * frequency / self.sample_rate
        cw = math.cos(w0)
        alpha = math.sin(w0) / (2.0 * q)
        if kind == 'high':
            b0 = (1.0 + cw) * 0.5
            b1 = -(1.0 + cw)
            b2 = (1.0 + cw) * 0.5
        else:
            b0 = (1.0 - cw) * 0.5
            b1 = 1.0 - cw
            b2 = (1.0 - cw) * 0.5
        a0 = 1.0 + alpha
        self._coeffs[section, 0] = b0 / a0
        self._coeffs[section, 1] = b1 / a0
        self._coeffs[section, 2] = b2 / a0
        self._coeffs[section, 3] = (-2.0 * cw) / a0
        self._coeffs[section, 4] = (1.0 - alpha) / a0

    def _update_coefficients(self, low, high):
        if low == self._last_low and high == self._last_high:
            return
        self._last_low = low
        self._last_high = high
        self._biquad(0, 'high', low, CleanUnit.Q_PAIR[0])
        self._biquad(1, 'high', low, CleanUnit.Q_PAIR[1])
        self._biquad(2, 'low', high, CleanUnit.Q_PAIR[0])
        self._biquad(3, 'low', high, CleanUnit.Q_PAIR[1])

    def _mirror(self, frames):
        if self.out.constant:
            self.right.set_constant(self.out.value)
            return
        np.copyto(self.right.data[:frames], self.out.data[:frames])
        self.right.constant = False

    def render(self, frames):
        signal = self.signal_in.eval(frames)
        low = self.low_in.eval(frames)
        high = self.high_in.eval(frames)

        out = self.out
        if not _svf_ready.is_set():
            # Kernel still compiling: pass audio rather than stall or click.
            source = signal.array(frames)
            np.copyto(out.data[:frames], source)
            out.constant = False
            self._mirror(frames)
            return

        stereo = bool(self.right_in.sources)
        if signal.constant and signal.value == 0.0 and not stereo:
            out.set_constant(0.0)
            self.right.set_constant(0.0)
            return

        low_now = low.value if low.constant else float(low.data[0])
        high_now = high.value if high.constant else float(high.data[0])
        self._update_coefficients(min(300.0, max(5.0, low_now)),
                                  min(20000.0, max(1000.0, high_now)))

        source = self._x[:frames]
        np.copyto(source, signal.array(frames), casting='unsafe')
        result = self._y[:frames]
        _clean_kernel(source, self._coeffs, self._states, result)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False

        if not stereo:
            self._mirror(frames)
            return
        np.copyto(source, self.right_in.eval(frames).array(frames),
                  casting='unsafe')
        _clean_kernel(source, self._coeffs, self._states_right, result)
        np.copyto(self.right.data[:frames], result, casting='unsafe')
        self.right.constant = False


class SpaceUnit(Unit):
    """Spatializer: a stereo signal onto a set of speakers, as outlets.

    This used to live inside audio_out~, which made the terminus a mixture
    of concerns; here it is its own processor, and the output stage is just
    a socket. One outlet per speaker, fixed when the unit is made; patch
    them to audio_out~'s inputs, several place~ into one output, summing at
    the inlets as everywhere.

    'ring' reads the outlets as speakers equally spaced around a circle, in
    order, panned pairwise between neighbours -- at any moment a source is
    in at most two speakers, which keeps the image sharp as it moves. 'pan'
    is azimuth: 0 front centre, +-0.5 the sides, +-1 the rear, where the
    ends meet.

    'corners' reads them as the corners of the room: bottom front-left,
    front-right, rear-left, rear-right, then the same four on top. Position
    is three equal-power faders -- left/right, front/rear, top/bottom --
    and a speaker's gain is the product of its axes, so power holds
    anywhere in the space. Four speakers are one layer; counts other than
    4 or 8 have no corners and fall back to the ring.

    Two speakers are neither ring nor corners but a pair: pan runs hard
    left at -1 to hard right at +1, equal-power, clamped -- on a ring of
    two, the extremes would meet at the rear and centre the image again,
    which is not what a pair means.

    Stereo is a fact, not a switch: patch the right inlet and the source
    occupies two points held apart by 'width'; leave it unpatched and the
    source is a single point that ignores width. Gains move at block rate
    and are ramped across each block, so a swept pan is click-free.

    Bypassed, left and right pass to the first two outlets untouched.
    """

    MAX_SPEAKERS = 16
    SPACES = ('ring', 'corners')

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, count=4):
        super().__init__(sample_rate)
        self.count = max(2, min(SpaceUnit.MAX_SPEAKERS, int(count)))
        self.signal_in = self.new_inlet()
        self.right_in = self.new_inlet()
        self.position_in = self.new_inlet(base=0.0, minimum=-1.0, maximum=1.0)
        self.width_in = self.new_inlet(base=2.0 / self.count,
                                       minimum=0.0, maximum=2.0)
        self.depth_in = self.new_inlet(base=0.0, minimum=-1.0, maximum=1.0)
        self.height_in = self.new_inlet(base=0.0, minimum=-1.0, maximum=1.0)
        self.space = 'ring'
        self.outs = [self.new_outlet() for _ in range(self.count)]
        self._target = np.zeros((2, self.count))
        self._previous = np.zeros((2, self.count))
        self._left = np.zeros(MAX_BLOCK, dtype=np.float32)
        self._right = np.zeros(MAX_BLOCK, dtype=np.float32)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float32)
        self._ramp = np.zeros(MAX_BLOCK, dtype=np.float32)

    def active_space(self):
        # Two speakers are a stereo pair, whatever the combo says: on a
        # ring of two, +-1 would be the rear point -- equidistant between
        # both speakers and so centred again -- when what a pair means by
        # pan is hard left to hard right, clamped, no wrap.
        if self.count == 2:
            return 'pair'
        if self.space == 'corners' and self.count in (4, 8):
            return 'corners'
        return 'ring'

    def _pair_gains(self, pan, row):
        left, right = self._axis_gains(pan)
        self._target[row, 0] = left
        self._target[row, 1] = right

    @staticmethod
    def _axis_gains(value):
        """Equal-power split of one axis: (toward -1, toward +1)."""
        if value < -1.0:
            value = -1.0
        elif value > 1.0:
            value = 1.0
        angle = (value + 1.0) * (math.pi * 0.25)
        return math.cos(angle), math.sin(angle)

    def _ring_gains(self, pan, row):
        """Pairwise pan around the ring; pan wraps rather than clamps."""
        count = self.count
        self._target[row, :] = 0.0
        azimuth = (pan * 0.5 + 0.5 / count) % 1.0
        scaled = azimuth * count
        index = int(scaled) % count
        fraction = scaled - int(scaled)
        angle = fraction * (math.pi * 0.5)
        self._target[row, index] = math.cos(angle)
        self._target[row, (index + 1) % count] = math.sin(angle)

    def _corner_gains(self, x, y, z, row):
        """Corner order is binary: bit 0 right, bit 1 rear, bit 2 top layer."""
        toward_left, toward_right = self._axis_gains(x)
        toward_front, toward_rear = self._axis_gains(y)
        toward_top, toward_bottom = self._axis_gains(z)
        for index in range(self.count):
            gain = toward_right if index & 1 else toward_left
            gain *= toward_rear if index & 2 else toward_front
            if self.count == 8:
                gain *= toward_top if index & 4 else toward_bottom
            self._target[row, index] = gain

    @staticmethod
    def _scalar(signal, frames):
        return signal.value if signal.constant else float(
            signal.data[frames - 1])

    def bypass_pairs(self):
        if self.right_in.sources:
            return ((self.signal_in, self.outs[0]),
                    (self.right_in, self.outs[1]))
        return ((self.signal_in, self.outs[0]),
                (self.signal_in, self.outs[1]))

    def deactivate(self):
        self._previous[:, :] = 0.0

    def render(self, frames):
        signal = self.signal_in.eval(frames)
        position = self.position_in.eval(frames)

        left = self._left[:frames]
        np.copyto(left, signal.array(frames))
        stereo = bool(self.right_in.sources)
        right = self._right[:frames]
        if stereo:
            np.copyto(right, self.right_in.eval(frames).array(frames))

        pan = self._scalar(position, frames)
        if stereo:
            width = self._scalar(self.width_in.eval(frames), frames)
            positions = (pan - width * 0.5, pan + width * 0.5)
        else:
            positions = (pan, None)
        space = self.active_space()
        if space == 'pair':
            for row, at in enumerate(positions):
                if at is None:
                    self._target[row, :] = 0.0
                else:
                    self._pair_gains(at, row)
        elif space == 'ring':
            for row, at in enumerate(positions):
                if at is None:
                    self._target[row, :] = 0.0
                else:
                    self._ring_gains(at, row)
        else:
            depth = self._scalar(self.depth_in.eval(frames), frames)
            height = self._scalar(self.height_in.eval(frames), frames)
            for row, at in enumerate(positions):
                if at is None:
                    self._target[row, :] = 0.0
                else:
                    self._corner_gains(at, depth, height, row)

        ramp = self._ramp[:frames]
        np.multiply(_INDEX_RAMP[:frames], 1.0 / frames, out=ramp,
                    casting='unsafe')
        scratch = self._scratch[:frames]
        sources = (left, right if stereo else left)
        for speaker in range(self.count):
            outlet = self.outs[speaker]
            wrote = False
            for row in (0, 1):
                begin = self._previous[row, speaker]
                end = self._target[row, speaker]
                if begin == 0.0 and end == 0.0:
                    continue
                if row == 1 and not stereo:
                    continue
                if begin == end:
                    np.multiply(sources[row], end, out=scratch)
                else:
                    np.multiply(ramp, end - begin, out=scratch)
                    scratch += begin
                    scratch *= sources[row]
                if wrote:
                    outlet.data[:frames] += scratch
                else:
                    np.copyto(outlet.data[:frames], scratch)
                    wrote = True
            if wrote:
                outlet.constant = False
            else:
                outlet.set_constant(0.0)
        self._previous[:, :] = self._target[:, :]


class AudioOutUnit(Unit):
    """Terminus: a wall socket, one input per device channel.

    This used to carry level, pan and a whole spatializer; those are
    fader~ and place~ now, and what remains is only what a socket needs --
    which inputs land on which device columns, and a mute. Input k mixes
    into the k-th listed channel, several nodes may address the same
    channel, and a channel the current device does not have is silent
    rather than an error, so an eight-channel patch still runs on a stereo
    laptop and sounds again when the rig is back.
    """

    MAX_CHANNELS = 16

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, count=2):
        super().__init__(sample_rate)
        self.count = max(1, min(AudioOutUnit.MAX_CHANNELS, int(count)))
        self.ins = [self.new_inlet() for _ in range(self.count)]
        # The stereo names survive for anything that addressed them.
        self.signal_in = self.ins[0]
        self.right_in = self.ins[1] if self.count > 1 else self.ins[0]
        self.channels = list(range(self.count))
        self.muted = False
        self.peak = 0.0
        self._buffers = np.zeros((self.count, MAX_BLOCK), dtype=np.float32)
        self._live = [False] * self.count
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float32)

    def render(self, frames):
        if self.muted:
            self.peak = 0.0
            return
        peak = 0.0
        scratch = self._scratch[:frames]
        for index, inlet in enumerate(self.ins):
            signal = inlet.eval(frames)
            if signal.constant and signal.value == 0.0:
                self._live[index] = False
                continue
            self._live[index] = True
            row = self._buffers[index, :frames]
            np.copyto(row, signal.array(frames))
            np.abs(row, out=scratch)
            peak = max(peak, float(scratch.max()))
        self.peak = peak

    def mix_into(self, mix, frames):
        if self.muted:
            return
        available = mix.shape[1]
        for index in range(self.count):
            if not self._live[index]:
                continue
            if index >= len(self.channels):
                continue
            channel = self.channels[index]
            if 0 <= channel < available:
                mix[:frames, channel] += self._buffers[index, :frames]


SAMPLER_MODES = ('loop', 'oneshot', 'scrub', 'follow', 'granular')


class SamplerBuffer:
    """A loaded sample, immutable once built.

    Built on the main thread and swapped into a SamplerOscUnit as one object,
    so the audio thread either sees the whole old sample or the whole new one.
    Channels are kept separate and padded by a sample so interpolation can read
    idx+1 without a bounds test in the inner path.
    """

    __slots__ = ('left', 'right', 'frames', 'source_rate', 'path', 'stereo')

    def __init__(self, left, right=None, source_rate=DEFAULT_SAMPLE_RATE, path=''):
        left = np.asarray(left, dtype=np.float32).reshape(-1)
        self.stereo = right is not None
        if right is None:
            right = left
        else:
            right = np.asarray(right, dtype=np.float32).reshape(-1)
            if len(right) != len(left):
                size = min(len(left), len(right))
                left, right = left[:size], right[:size]

        self.frames = int(len(left))
        if self.frames:
            self.left = np.concatenate([left, left[-1:]])
            self.right = np.concatenate([right, right[-1:]])
        else:
            self.left = np.zeros(2, dtype=np.float32)
            self.right = self.left
        self.source_rate = float(source_rate)
        self.path = path


class SamplerOscUnit(Unit):
    """The sampler recast as an oscillator: recorded material under CV.

    Playback rate combines a linear 'rate' multiplier with an exponential
    'pitch' inlet in octaves, matching vco~, so the same envelope or LFO drives
    either interchangeably. Source files are never resampled -- the file's own
    rate is folded into the increment, which is exact and costs nothing.

    Modes differ in what drives the playhead:

      loop      free-running through the loop window, wrapping
      oneshot   plays once from the trigger and stops
      scrub     playhead follows the position inlet directly, turntable style.
                Holding still emits a constant sample value, i.e. DC -- that is
                what a stopped record does, but patch a highpass if it matters.
      follow    position inlet is a target the playhead chases through a
                spring, so motion becomes playback speed. This is the one that
                suits effort data: the material moves when the body does.
      granular  grains sprayed around the position inlet.

    Loop points and grain settings are read once per block. Rate, pitch and
    position are true audio-rate inlets.
    """

    MAX_GRAINS = 128
    # Hard bound on playback speed. Also the sanitising clamp: a NaN or inf
    # arriving from upstream CV would otherwise poison self.position for good
    # and take the audio thread down on the next gather.
    MAX_RATE = 64.0

    class _Grain:
        __slots__ = ('position', 'increment', 'age', 'duration', 'amplitude')

        def __init__(self, position, increment, duration, amplitude):
            self.position = position
            self.increment = increment
            self.age = 0
            self.duration = duration
            self.amplitude = amplitude

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.rate_in = self.new_inlet(base=1.0)
        self.pitch_in = self.new_inlet(base=0.0)                       # octaves
        self.position_in = self.new_inlet(base=0.0)                    # 0..1
        self.loop_start_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.loop_end_in = self.new_inlet(base=1.0, minimum=0.0, maximum=1.0)
        self.grain_size_in = self.new_inlet(base=0.08, minimum=0.001)  # seconds
        self.grain_rate_in = self.new_inlet(base=20.0, minimum=0.0)    # per second
        self.jitter_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.trigger_in = self.new_inlet(base=0.0)

        self.sample = None
        self.mode = 0
        self.crossfade = 0.005      # seconds, loop seam
        self.follow_speed = 8.0     # spring rate for 'follow'
        self.reverse = False

        self.position = 0.0         # in source samples
        self.velocity = 0.0         # source samples per output sample
        self.playing = True
        self._trigger_armed = True
        self._grains = []
        self._grain_debt = 0.0

        self.left = self.new_outlet()
        self.right = self.new_outlet()

        # Scrub declick: a position inlet can jump (a phasor turning over, a
        # gesture snapping), and an absolute jump in the playhead is a step in
        # the output, which is a click. These carry the pre-jump trajectory so
        # it can be crossfaded out across the discontinuity.
        self._scrub_last = None
        self._scrub_delta = 0.0
        self._fade_position = 0.0
        self._fade_delta = 0.0
        self._fade_remaining = 0
        self._fade_total = 0

        self._pos = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._inc = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._deltas = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._fade_pos = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._fade_buffer = np.zeros(MAX_BLOCK, dtype=np.float32)
        self._left = np.zeros(MAX_BLOCK, dtype=np.float32)
        self._right = np.zeros(MAX_BLOCK, dtype=np.float32)

    # -- helpers ------------------------------------------------------------

    def reset(self):
        self.position = 0.0
        self.velocity = 0.0
        self.playing = True
        self._grains = []

    def trigger(self):
        """Restart from the top of the loop window."""
        sample = self.sample
        self.position = 0.0
        if sample is not None:
            start, _ = self._window(sample)
            self.position = float(start)
        self.velocity = 0.0
        self.playing = True
        self._grains = []

    def _window(self, sample):
        """Loop start/end in source samples, read at control rate."""
        total = sample.frames
        low = self.loop_start_in.eval(1)
        high = self.loop_end_in.eval(1)
        start = (low.value if low.constant else float(low.data[0])) * total
        end = (high.value if high.constant else float(high.data[0])) * total
        start = max(0.0, min(start, total - 2.0))
        end = max(start + 2.0, min(end, float(total)))
        return start, end

    @staticmethod
    def _gather(data, positions, out):
        index = positions.astype(np.int64)
        fraction = positions - index
        low = data[index]
        high = data[index + 1]
        np.multiply(high - low, fraction, out=out, casting='unsafe')
        out += low

    # -- render -------------------------------------------------------------

    def render(self, frames):
        sample = self.sample          # single read: the swap is atomic
        left_out, right_out = self.left, self.right

        if sample is None or sample.frames < 2:
            left_out.set_constant(0.0)
            right_out.set_constant(0.0)
            return

        trigger = self.trigger_in.eval(frames)
        level = trigger.value if trigger.constant else float(trigger.data[frames - 1])
        if level >= 0.5:
            if self._trigger_armed:
                self._trigger_armed = False
                self.trigger()
        else:
            self._trigger_armed = True

        start, end = self._window(sample)
        mode = SAMPLER_MODES[self.mode] if self.mode < len(SAMPLER_MODES) else 'loop'

        if mode == 'granular':
            self._render_granular(sample, start, end, frames)
        else:
            self._render_playhead(sample, start, end, mode, frames)

        np.copyto(left_out.data[:frames], self._left[:frames])
        np.copyto(right_out.data[:frames], self._right[:frames])
        left_out.constant = False
        right_out.constant = False

    def _increment(self, sample, frames):
        """Source samples consumed per output sample, as an array."""
        rate = self.rate_in.eval(frames)
        pitch = self.pitch_in.eval(frames)
        increment = self._inc[:frames]

        scale = sample.source_rate / self.sample_rate
        if pitch.constant:
            scale *= 2.0 ** pitch.value
            if rate.constant:
                increment[:] = rate.value * scale
            else:
                np.multiply(rate.data[:frames], scale, out=increment,
                            casting='unsafe')
        else:
            np.multiply(pitch.data[:frames], math.log(2.0), out=increment,
                        casting='unsafe')
            np.exp(increment, out=increment)
            increment *= scale
            if rate.constant:
                increment *= rate.value
            else:
                increment *= rate.data[:frames]

        if self.reverse:
            increment *= -1.0

        limit = SamplerOscUnit.MAX_RATE
        np.nan_to_num(increment, copy=False, nan=0.0, posinf=limit, neginf=-limit)
        np.clip(increment, -limit, limit, out=increment)
        return increment

    def _render_playhead(self, sample, start, end, mode, frames):
        positions = self._pos[:frames]
        span = end - start

        if mode in ('loop', 'oneshot'):
            increment = self._increment(sample, frames)
            np.cumsum(increment, out=positions)
            positions += self.position

            if mode == 'loop':
                np.subtract(positions, start, out=positions)
                np.mod(positions, span, out=positions)
                positions += start
                self.position = float(positions[-1])
            else:
                if not self.playing:
                    self._left[:frames] = 0.0
                    self._right[:frames] = 0.0
                    return
                self.position = float(positions[-1])
                if self.position >= end or self.position < start:
                    self.playing = False
                np.clip(positions, start, end - 1.0, out=positions)

        else:
            target = self.position_in.eval(frames)
            if mode == 'scrub':
                if target.constant:
                    positions[:] = start + target.value * span
                else:
                    np.multiply(target.data[:frames], span, out=positions,
                                casting='unsafe')
                    positions += start
                # A record has ends: hold at them rather than wrapping, so
                # position 1.0 means the end of the window and not the start.
                np.nan_to_num(positions, copy=False, nan=start,
                              posinf=end, neginf=start)
                np.clip(positions, start, end - 1.0, out=positions)
                self.position = float(positions[-1])
                self._finish_playhead(sample, positions, start, mode, frames)
                return
            else:  # follow
                goal = start + span * (target.value if target.constant
                                       else float(target.data[frames - 1]))
                # Critically damped spring: motion of the target becomes
                # playback speed, so the material only sounds while it moves.
                #
                # Solved in closed form across the block rather than stepped.
                # Stepping it is only stable while follow_speed * block_time
                # stays under about 1 -- past that the velocity diverges within
                # a few blocks and the playhead reads noise. The closed form is
                # exact at any speed and never overshoots, and it gives a
                # per-sample curve instead of one straight line per block.
                #
                #   x(t) = goal + e^(-wt) * (d + (v + w*d) * t)
                #
                # with d the displacement from the goal and v the current
                # velocity, both at the top of the block.
                omega = max(0.01, self.follow_speed)
                displacement = self.position - goal
                velocity = self.velocity * self.sample_rate
                coefficient = velocity + omega * displacement

                times = self._inc[:frames]
                np.multiply(_INDEX_RAMP[:frames], 1.0 / self.sample_rate,
                            out=times)
                # positions carries the decay term until the two are combined.
                np.multiply(times, -omega, out=positions)
                np.exp(positions, out=positions)
                decay_end = float(positions[-1])

                np.multiply(times, coefficient, out=times)
                times += displacement
                polynomial_end = float(times[-1])

                positions *= times
                positions += goal

                self.position = float(positions[-1])
                self.velocity = (decay_end
                                 * (coefficient - omega * polynomial_end)
                                 / self.sample_rate)
                if not math.isfinite(self.velocity):
                    self.velocity = 0.0

            np.nan_to_num(positions, copy=False, nan=start,
                          posinf=start, neginf=start)
            np.subtract(positions, start, out=positions)
            np.mod(positions, span, out=positions)
            positions += start

        self._finish_playhead(sample, positions, start, mode, frames,
                              end=end)

    def _finish_playhead(self, sample, positions, start, mode, frames, end=None):
        """Sanitise, gather both channels, and blend the loop seam."""
        np.nan_to_num(positions, copy=False, nan=start, posinf=start,
                      neginf=start)
        np.clip(positions, 0.0, sample.frames - 1.0, out=positions)
        if not math.isfinite(self.position):
            self.position = start
        if not math.isfinite(self.velocity):
            self.velocity = 0.0

        left = self._left[:frames]
        right = self._right[:frames]
        self._gather(sample.left, positions, left)
        if sample.stereo:
            self._gather(sample.right, positions, right)
        else:
            np.copyto(right, left)

        if mode == 'loop' and end is not None:
            self._apply_seam_crossfade(sample, positions, start, end, frames)
        elif mode == 'scrub':
            self._apply_scrub_declick(sample, positions, frames)

    def _apply_scrub_declick(self, sample, positions, frames):
        """Crossfade across jumps in the position inlet.

        A phasor driving position turns over once a cycle, and that turn-over
        is a full-scale step in the playhead: measured at 16x the ordinary
        sample-to-sample motion, i.e. an audible click every cycle. Rather than
        cut straight to the new position, the pre-jump trajectory is continued
        at its own speed and faded out underneath the new one.

        The threshold is the crossfade length itself, which separates cleanly:
        ordinary playback moves at most MAX_RATE samples per sample, far below
        a fade window's worth of material.
        """
        fade_total = int(self.crossfade * self.sample_rate)
        if fade_total < 8:
            self._fade_remaining = 0
            self._scrub_last = float(positions[frames - 1])
            self._scrub_delta = (float(positions[frames - 1] - positions[frames - 2])
                                 if frames > 1 else 0.0)
            return

        threshold = max(float(SamplerOscUnit.MAX_RATE) * 4.0,
                        self.crossfade * sample.source_rate)

        deltas = self._deltas[:frames]
        if frames > 1:
            np.subtract(positions[1:], positions[:-1], out=deltas[1:])
        deltas[0] = (positions[0] - self._scrub_last
                     if self._scrub_last is not None else 0.0)

        jumps = np.flatnonzero(np.abs(deltas) > threshold)
        fade_start = -1

        if jumps.size:
            index = int(jumps[0])
            if index == 0:
                base = self._scrub_last
                delta = self._scrub_delta
            else:
                base = float(positions[index - 1])
                delta = (float(deltas[index - 1]) if index >= 2
                         else self._scrub_delta)
            self._fade_position = base if base is not None else float(positions[0])
            self._fade_delta = delta
            self._fade_remaining = fade_total
            self._fade_total = fade_total
            fade_start = index
        elif self._fade_remaining > 0:
            fade_start = 0

        self._scrub_last = float(positions[frames - 1])
        self._scrub_delta = (float(deltas[frames - 1]) if frames > 1 else 0.0)

        if fade_start < 0 or self._fade_remaining <= 0:
            return

        count = min(frames - fade_start, self._fade_remaining)
        if count <= 0:
            return

        # Continue the old playhead at the speed it was travelling.
        old = self._fade_pos[:count]
        np.multiply(_INDEX_RAMP[:count], self._fade_delta, out=old)
        old += self._fade_position
        np.clip(old, 0.0, sample.frames - 1.0, out=old)

        done = self._fade_total - self._fade_remaining
        alpha = self._deltas[:count]
        np.multiply(_INDEX_RAMP[:count], 1.0 / self._fade_total, out=alpha)
        alpha += done / self._fade_total
        np.clip(alpha, 0.0, 1.0, out=alpha)

        blend = self._fade_buffer[:count]
        stop = fade_start + count

        self._gather(sample.left, old, blend)
        segment = self._left[fade_start:stop]
        segment *= alpha
        segment += blend * (1.0 - alpha)

        if sample.stereo:
            self._gather(sample.right, old, blend)
        segment = self._right[fade_start:stop]
        segment *= alpha
        segment += blend * (1.0 - alpha)

        self._fade_position = float(old[-1]) + self._fade_delta
        self._fade_remaining -= count

    def _apply_seam_crossfade(self, sample, positions, start, end, frames):
        """Blend the pre-loop-end tail with the post-wrap head.

        Without this, a loop whose endpoints do not match emits a step -- and a
        step is a click.
        """
        fade = self.crossfade * sample.source_rate
        if fade <= 1.0 or fade >= (end - start):
            return
        edge = end - fade
        mask = positions >= edge
        if not np.any(mask):
            return

        tail = positions[mask]
        alpha = (tail - edge) / fade
        wrapped = tail - (end - start)
        np.clip(wrapped, 0.0, sample.frames - 1.0, out=wrapped)

        blend = np.empty(len(tail), dtype=np.float32)
        self._gather(sample.left, wrapped, blend)
        left = self._left[:frames]
        left[mask] = left[mask] * (1.0 - alpha) + blend * alpha
        if sample.stereo:
            self._gather(sample.right, wrapped, blend)
            right = self._right[:frames]
            right[mask] = right[mask] * (1.0 - alpha) + blend * alpha
        else:
            self._right[:frames][mask] = left[mask]

    def _render_granular(self, sample, start, end, frames):
        left = self._left[:frames]
        right = self._right[:frames]
        left[:] = 0.0
        right[:] = 0.0

        span = end - start
        size = self.grain_size_in.eval(1)
        rate = self.grain_rate_in.eval(1)
        jitter = self.jitter_in.eval(1)
        position = self.position_in.eval(frames)

        duration = max(1, int((size.value if size.constant
                               else float(size.data[0])) * self.sample_rate))
        density = rate.value if rate.constant else float(rate.data[0])
        spread = jitter.value if jitter.constant else float(jitter.data[0])
        centre = position.value if position.constant else float(position.data[0])
        increment = self._increment(sample, frames)
        grain_increment = float(increment[0])

        # Spawn on a running debt so fractional grains-per-block accumulate
        # instead of being rounded away.
        if density > 0.0:
            self._grain_debt += density * frames / self.sample_rate
            while self._grain_debt >= 1.0 and len(self._grains) < self.MAX_GRAINS:
                self._grain_debt -= 1.0
                offset = centre
                if spread > 0.0:
                    offset += (np.random.random() - 0.5) * 2.0 * spread
                offset -= math.floor(offset)
                self._grains.append(self._Grain(
                    start + offset * span, grain_increment, duration, 1.0))
        else:
            self._grain_debt = 0.0

        if not self._grains:
            return

        alive = []
        limit = sample.frames - 1.0
        for grain in self._grains:
            remaining = grain.duration - grain.age
            if remaining <= 0:
                continue
            count = min(frames, remaining)

            positions = self._pos[:count]
            np.multiply(_INDEX_RAMP[:count], grain.increment, out=positions)
            positions += grain.position
            np.clip(positions, 0.0, limit, out=positions)

            chunk = np.empty(count, dtype=np.float32)
            self._gather(sample.left, positions, chunk)

            # Hann window over the grain's own lifetime.
            phase = (grain.age + _INDEX_RAMP[:count]) / grain.duration
            window = 0.5 - 0.5 * np.cos(2.0 * math.pi * phase)
            chunk *= window
            chunk *= grain.amplitude
            left[:count] += chunk

            if sample.stereo:
                chunk_right = np.empty(count, dtype=np.float32)
                self._gather(sample.right, positions, chunk_right)
                chunk_right *= window
                chunk_right *= grain.amplitude
                right[:count] += chunk_right
            else:
                right[:count] += chunk

            grain.age += count
            grain.position = float(positions[-1]) + grain.increment
            if grain.age < grain.duration:
                alive.append(grain)

        self._grains = alive


# ----------------------------------------------------------------------------
# string~ / modal~  --  physical models
# ----------------------------------------------------------------------------
#
# Two shapes cover most struck and plucked acoustics. A string is a delay loop
# with a little loss -- energy bounces between the ends, darkening as it goes --
# and everything from guitar to pipe is that loop with different reflections.
# A bell is not a loop at all but a set of independent resonances, each ringing
# down at its own rate; bars, membranes and bowls are the same bank with
# different tuning tables. Between them: pluck a string, strike a bell, and
# both take arbitrary audio as excitation, which is where a body driving a
# resonator gets interesting.


def _excitation_events(trigger, frames, threshold, armed):
    """Rising edges of a trigger signal, with the level at each crossing.

    Returns ((index, level), ...) and the new armed state. The level is the
    trigger's own height at the crossing sample, so the same cord carries
    timing and velocity: a taller trigger strikes harder, and an envelope or
    effort value patched here plays dynamics without a second connection.
    """
    if trigger.constant:
        high = trigger.value >= threshold
        if high and armed:
            return ((0, abs(trigger.value)),), False
        return (), not high
    data = trigger.data[:frames]
    above = data >= threshold
    events = []
    if above[0] and armed:
        events.append((0, abs(float(data[0]))))
    edges = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    for edge in edges:
        events.append((int(edge), abs(float(data[edge]))))
    return tuple(events), not bool(above[-1])


def _string_kernel_source(x, line, write, ex_line, ex_write, delay, gain,
                          damp, position, stiffness, ap_x, ap_y, polarity,
                          low, dc_x, dc_y, in_x, in_y, out):
    """A waveguide string: a delay loop with loss, sample by sample.

    The loop is the string. What is written now depends on what is read now,
    so like the delay kernel this cannot be vectorised. Per sample: read the
    far end of the line, darken it through a one pole (the string's internal
    loss -- high partials die first, which is most of what makes a decaying
    note sound plucked rather than filtered), scale by the round-trip gain
    that sets the decay time, disperse it through a short allpass chain
    (stiffness: high partials travel faster in a stiff string, which is the
    piano's inharmonicity), and write it back in with the new excitation.

    The excitation passes through a comb before it enters -- a pluck at 1/5 of
    the length cannot excite the modes with a node there, and cancelling a
    delayed copy of the excitation is exactly that. Position 0 turns it off.

    Two DC blockers: one on the excitation (a slow signal patched in as an
    exciter would otherwise be multiplied by the loop's DC gain and pin the
    line against its soft stop), one on the output (the loop can hold a small
    standing offset that is nothing musical).

    The soft stop from the delay kernel guards the loop: gains reach 1 when
    long decays meet a bright setting, and the failure should be a settled
    oscillation rather than a runaway.
    """
    size = line.shape[0]
    ex_size = ex_line.shape[0]
    limit = size - 3.0
    for i in range(x.shape[0]):
        want = delay[i]
        if want < 2.0:
            want = 2.0
        elif want > limit:
            want = limit

        e = x[i]
        hp = e - in_x + 0.995 * in_y
        in_x = e
        in_y = hp
        ex_line[ex_write] = hp
        if position > 0.0:
            back = want * position
            if back < 1.0:
                back = 1.0
            elif back > ex_size - 3.0:
                back = ex_size - 3.0
            read = ex_write - back
            if read < 0.0:
                read += ex_size
            hp -= _cubic_read(ex_line, ex_size, read)
        ex_write += 1
        if ex_write >= ex_size:
            ex_write = 0

        read = write - want
        if read < 0.0:
            read += size
        y = _cubic_read(line, size, read)

        low += (y - low) * (1.0 - damp[i])
        fed = low * gain[i] * polarity

        for k in range(ap_x.shape[0]):
            v = -stiffness * fed + ap_x[k] + stiffness * ap_y[k]
            ap_x[k] = fed
            ap_y[k] = v
            fed = v

        if fed > 1.5:
            fed = 1.5 + np.tanh(fed - 1.5)
        elif fed < -1.5:
            fed = -1.5 - np.tanh(-fed - 1.5)

        line[write] = hp + fed

        o = y - dc_x + 0.995 * dc_y
        dc_x = y
        dc_y = o
        out[i] = o

        write += 1
        if write >= size:
            write = 0
    return write, ex_write, low, dc_x, dc_y, in_x, in_y


def _modal_kernel_source(x, pulse, b1, b2, gains, strike_gains, s1, s2,
                         dc_x, dc_y, out, rungs, pair_phase, pair_step,
                         pair_out):
    """A bank of two-pole resonators: struck through one gain, driven
    through another.

    Same shape as the formant bank -- modes are independent, so the inner
    loop vectorises and the input is read once for all of them -- but where
    a formant filters what passes through it, a mode *rings*: the pole radius
    is set from a decay time, so an impulse in produces a tone that dies away
    on its own. Coefficients hold for the block and are computed outside.

    The two inputs are the same bank meaning two different things, and they
    cannot share a gain. A resonator's steady-state gain at its own frequency
    exceeds its impulse-response amplitude by its Q -- thousands, at a long
    decay -- so gains that make a strike ring at its table weight make a
    sustained tone parked on a mode a runaway, and gains that bound the
    steady state make the first half-second of bowing inaudible. The strike
    path (the internal mallet) is impulse-normalized; the audio path is
    normalized by sqrt(1-r) -- heard at once, growing while held -- and the
    growth is bounded by the soft stop on the states rather than by the
    gain.

    The DC blocker guards the audio path: a low mode with a long decay still
    has real gain near DC, and a slow control signal used as a drive would
    ride the bank up on its offset. The strike pulse skips it -- a mallet
    tap is one-sided by nature and has nothing to block.
    """
    modes = b1.shape[0]
    for i in range(x.shape[0]):
        e = x[i]
        hp = e - dc_x + 0.995 * dc_y
        dc_x = e
        dc_y = hp
        tap = pulse[i]
        total = 0.0
        for m in range(modes):
            # The RECURSIVE part alone is what the mode is DOING. Adding
            # the driving term into the output as well gives every mode
            # a path around its own filter, and at lag zero those paths
            # add COHERENTLY across the bank while the rings they excite
            # dephase within a sample -- so eight modes pass eight times
            # the excitation against one mode's worth of tone. Measured
            # on the plate bank the leak came to 100.6% of the whole
            # ring peak: the drive arriving as loudly as the resonance
            # it was meant to be feeding, unfiltered and so flat, and
            # plainly audible with the dry mix at zero.
            #
            # A struck plate does not radiate the mallet. The drive goes
            # into the mode's STATE and is heard through it.
            rung = b1[m] * s1[m] + b2[m] * s2[m]
            y = gains[m] * hp + strike_gains[m] * tap + rung
            # The soft stop from the delay loop, on each mode's state: a
            # drive parked on a resonance grows toward this and settles
            # against it instead of running away, and everything below the
            # knee stays exactly linear -- struck rings never touch it.
            if y > 1.5:
                y = 1.5 + np.tanh(y - 1.5)
            elif y < -1.5:
                y = -1.5 - np.tanh(-y - 1.5)
            s2[m] = s1[m]
            s1[m] = y
            rungs[m] = rung
            if pair_out < 0.5:
                total += rung
        if pair_out > 0.5:
            # The modes come in pairs whose split is fixed in the frame
            # of whatever is loading them -- for a vessel, the water. If
            # that frame TURNS, the pattern turns with it, and a fixed
            # listener hears the bellies and the nodes go past. So the
            # pair is mixed at the pickup rather than summed, and the
            # mix angle walks. Summing them is what a still frame does,
            # and it is what everything but a swirled vessel wants.
            cs = math.cos(pair_phase)
            sn = math.sin(pair_phase)
            for m in range(0, modes - 1, 2):
                total += cs * rungs[m] + sn * rungs[m + 1]
            if modes % 2 == 1:
                total += rungs[modes - 1]
            pair_phase += pair_step
            if pair_phase > 6.283185307179586:
                pair_phase -= 6.283185307179586
            elif pair_phase < 0.0:
                pair_phase += 6.283185307179586
        out[i] = total
    return dc_x, dc_y, pair_phase


if _HAVE_NUMBA:
    _string_kernel = njit(cache=True, fastmath=True)(_string_kernel_source)
    _modal_kernel = njit(cache=True, fastmath=True)(_modal_kernel_source)
else:
    _string_kernel = _string_kernel_source
    _modal_kernel = _modal_kernel_source


class StringUnit(Unit):
    """Plucked string by extended Karplus-Strong, with a tube mode.

    The classic algorithm is a delay line the length of one period with a
    lossy loop; everything else here is what forty years of extensions found
    worth adding. 'decay' is the time to -60 dB, held constant across pitch
    by computing the round-trip gain from the current period -- without that
    a string tuned up decays faster, which is physical but unplayable.
    'brightness' is the loop's internal loss, 'position' the pluck-point comb,
    'stiffness' a dispersion chain that stretches the upper partials sharp,
    toward piano and away from nylon.

    Excitation is two things at once, and either alone works. A rising edge
    on 'trigger' injects one period of noise -- the pluck, its level taken
    from the trigger's own height, its spectrum set by pluck_color between
    fresh white noise and a darker, mellower burst. And whatever arrives at
    the audio inlet enters the loop continuously, so a noise envelope bows
    the string, a click strikes it, and a body's effort stream plays it as a
    sustained instrument no burst could.

    'tube' flips the sign of the reflection, which is what the closed end of
    a pipe does: even harmonics cancel and the fundamental sits an octave
    below the loop length, so the delay is halved to keep the pitch. Blow it
    with noise at the audio inlet rather than plucking it.

    The delay is compensated for the group delay of the loop filter and the
    dispersion chain, so the string plays in tune until the compensation runs
    into the 2-sample floor at the very top of the range.
    """

    MODES = ('string', 'tube')
    MIN_FREQUENCY = 20.0
    DISPERSION_STAGES = 4
    NOISE_SAMPLES = 1 << 16

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.excite_in = self.new_inlet()
        # How keenly the string hears what is patched in. Unity by
        # default, so a string that never had this sounds as it did.
        self.sensitivity_in = self.new_inlet(base=1.0, minimum=0.0,
                                             maximum=8.0)
        self.trigger_in = self.new_inlet()
        self.frequency_in = self.new_inlet(base=220.0,
                                           minimum=StringUnit.MIN_FREQUENCY)
        self.pitch_in = self.new_inlet()
        self.decay_in = self.new_inlet(base=2.0, minimum=0.01, maximum=60.0)
        self.brightness_in = self.new_inlet(base=0.7, minimum=0.0, maximum=1.0)
        self.position_in = self.new_inlet(base=0.2, minimum=0.0, maximum=0.5)
        self.stiffness_in = self.new_inlet(base=0.0, minimum=0.0, maximum=0.9)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        self.mode = 0
        self.pluck_color = 0.3
        self.threshold = 0.5

        size = int(self.sample_rate / StringUnit.MIN_FREQUENCY) + 8
        self.line = np.zeros(size, dtype=np.float64)
        self.ex_line = np.zeros(size, dtype=np.float64)
        self._write = 0
        self._ex_write = 0
        self._low = 0.0
        self._dc_x = 0.0
        self._dc_y = 0.0
        self._in_x = 0.0
        self._in_y = 0.0
        self._ap_x = np.zeros(StringUnit.DISPERSION_STAGES, dtype=np.float64)
        self._ap_y = np.zeros(StringUnit.DISPERSION_STAGES, dtype=np.float64)

        # Two noise tables made once: white, and the same noise through a one
        # pole. pluck_color crossfades between them, which is how the burst
        # gets a variable spectrum without running a filter on the audio
        # thread. The read point advances table-length-agnostically, so
        # successive plucks draw different noise but a patch reloaded sounds
        # the same.
        generator = np.random.default_rng(20260807)
        white = generator.uniform(-1.0, 1.0, StringUnit.NOISE_SAMPLES)
        dark = np.empty_like(white)
        coefficient = math.exp(-2.0 * math.pi * 800.0 / self.sample_rate)
        if scipy_signal is not None:
            dark[:] = scipy_signal.lfilter([1.0 - coefficient],
                                           [1.0, -coefficient], white)
        else:
            level = 0.0
            for index in range(white.shape[0]):
                level += (white[index] - level) * (1.0 - coefficient)
                dark[index] = level
        dark *= float(np.std(white) / max(1.0e-9, np.std(dark)))
        self._white = white
        self._dark = dark
        self._noise_at = 0
        self._burst_remaining = 0
        self._burst_amp = 0.0
        self._trigger_armed = True
        self._fire_requests = 0
        self._fire_served = 0
        self._quiet = True

        self.out = self.new_outlet()
        self._exc = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._sense_glide = 1.0
        self._sense_ramp = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._freq = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._delay = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._gain = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._damp = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._noise_a = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._noise_b = np.zeros(MAX_BLOCK, dtype=np.float64)

    def bypass_pairs(self):
        # A string has an audio input, so its switch is a bypass: standing
        # aside means the excitation reaches the output unplayed, not that
        # the signal disappears.
        return ((self.excite_in, self.out),)

    def fire(self):
        """Request one pluck from the node layer. Served on the next block."""
        self._fire_requests += 1

    def reset(self):
        self.line[:] = 0.0
        self.ex_line[:] = 0.0
        self._low = 0.0
        self._dc_x = 0.0
        self._dc_y = 0.0
        self._in_x = 0.0
        self._in_y = 0.0
        self._ap_x[:] = 0.0
        self._ap_y[:] = 0.0
        self._burst_remaining = 0
        self._quiet = True

    def deactivate(self):
        # Same reasoning as the delay: coming back to a stale line is a seam
        # the read head finds later, long after the fade has finished.
        self.reset()

    def _add_burst(self, exc, start, stop):
        """Mix the active noise burst into exc[start:stop], advancing it."""
        remaining = self._burst_remaining
        if remaining <= 0 or stop <= start:
            return
        count = min(stop - start, remaining)
        color = min(1.0, max(0.0, self.pluck_color))
        amp = self._burst_amp
        at = start
        left = count
        while left > 0:
            position = self._noise_at
            chunk = min(left, StringUnit.NOISE_SAMPLES - position)
            piece = self._noise_a[:chunk]
            np.multiply(self._white[position:position + chunk],
                        (1.0 - color) * amp, out=piece)
            if color > 0.0:
                shade = self._noise_b[:chunk]
                np.multiply(self._dark[position:position + chunk],
                            color * amp, out=shade)
                piece += shade
            exc[at:at + chunk] += piece
            self._noise_at = (position + chunk) % StringUnit.NOISE_SAMPLES
            at += chunk
            left -= chunk
        self._burst_remaining = remaining - count

    def render(self, frames):
        signal = self.excite_in.eval(frames)
        trigger = self.trigger_in.eval(frames)
        frequency = self.frequency_in.eval(frames)
        pitch = self.pitch_in.eval(frames)
        decay = self.decay_in.eval(frames)
        brightness = self.brightness_in.eval(frames)
        position = self.position_in.eval(frames)
        stiffness = self.stiffness_in.eval(frames)
        out_level = self.level_in.eval(frames)

        out = self.out
        if not _svf_ready.is_set():
            # Kernel still compiling (or numba missing): a silent string
            # rather than a stalled callback.
            out.set_constant(0.0)
            return

        exc = self._exc[:frames]
        # Sensitivity scales what ARRIVES, applied before the unit's own
        # mallet is written into the same buffer -- turning the inlet down
        # must not soften a strike the node makes itself. A knob move
        # glides across the block rather than stepping at its edge: one
        # factor per block is a staircase, and a staircase on a sustained
        # excitation is a zipper.
        sense = self.sensitivity_in.eval(frames)
        if signal.constant:
            exc[:] = signal.value
            silent_input = signal.value == 0.0
        else:
            np.copyto(exc, signal.data[:frames])
            silent_input = False
        if sense.constant:
            target = min(8.0, max(0.0, sense.value))
            start = self._sense_glide
            landing = start + (target - start) * 0.35
            self._sense_glide = landing
            if start == landing:
                if landing != 1.0:
                    exc *= landing
                silent_input = silent_input or landing == 0.0
            else:
                ramp = self._sense_ramp[:frames]
                np.multiply(_INDEX_RAMP[:frames],
                            (landing - start) / frames, out=ramp)
                ramp += start
                exc *= ramp
                silent_input = False
        else:
            np.clip(sense.data[:frames], 0.0, 8.0,
                    out=self._sense_ramp[:frames])
            exc *= self._sense_ramp[:frames]
            self._sense_glide = float(self._sense_ramp[frames - 1])
            silent_input = False

        events, self._trigger_armed = _excitation_events(
            trigger, frames, self.threshold, self._trigger_armed)
        if self._fire_requests != self._fire_served:
            self._fire_served = self._fire_requests
            events = ((0, 1.0),) + events

        # A settled string with nothing exciting it renders nothing, so a
        # patch full of idle strings costs what an idle patch should.
        if (self._quiet and not events and silent_input
                and self._burst_remaining <= 0):
            out.set_constant(0.0)
            return

        freq = self._freq[:frames]
        self._build_hertz(freq, frequency, pitch, frames,
                          StringUnit.MIN_FREQUENCY)

        # The pluck is one period of noise -- the burst the original
        # algorithm filled its buffer with, delivered through the inlet so
        # it passes the position comb like any other excitation.
        period = self.sample_rate / float(freq[0])
        cursor = 0
        for index, amp in events:
            self._add_burst(exc, cursor, index)
            cursor = index
            self._burst_remaining = int(min(max(32.0, period), 4096.0))
            # Half, so a velocity-1 pluck peaks near the +-1 the oscillators
            # put out rather than twice it.
            self._burst_amp = min(2.0, amp) * 0.5
        self._add_burst(exc, cursor, frames)

        damp = self._damp[:frames]
        if brightness.constant:
            damp[:] = 1.0 - brightness.value
        else:
            np.subtract(1.0, brightness.data[:frames], out=damp,
                        casting='unsafe')
        np.clip(damp, 0.0, 0.95, out=damp)

        # Round-trip gain from the decay time: -60 dB after f * t60 trips.
        gain = self._gain[:frames]
        if decay.constant:
            np.multiply(freq, max(0.01, decay.value), out=gain)
        else:
            scratch = self._scratch[:frames]
            np.copyto(scratch, decay.data[:frames], casting='unsafe')
            np.clip(scratch, 0.01, 60.0, out=scratch)
            np.multiply(freq, scratch, out=gain)
        np.divide(-6.907755, gain, out=gain)
        np.exp(gain, out=gain)

        stiff = stiffness.value if stiffness.constant else float(
            stiffness.data[0])
        stiff = min(0.9, max(0.0, stiff))

        taps = self._delay[:frames]
        np.divide(self.sample_rate, freq, out=taps)
        polarity = 1.0
        if self.mode == 1:
            # A closed pipe reflects inverted; the octave that costs is
            # bought back by halving the line.
            taps *= 0.5
            polarity = -1.0
        # What the loop already delays, the line must not: the one pole's
        # group delay plus one sample and change per allpass stage.
        darkness = float(damp[0])
        compensation = darkness / (1.0 - darkness)
        compensation += StringUnit.DISPERSION_STAGES * (1.0 + stiff) / (
            1.0 - stiff)
        taps -= compensation
        np.clip(taps, 2.0, self.line.shape[0] - 3.0, out=taps)

        pluck_point = position.value if position.constant else float(
            position.data[0])
        pluck_point = min(0.5, max(0.0, pluck_point))

        result = self._y[:frames]
        (self._write, self._ex_write, self._low, self._dc_x, self._dc_y,
         self._in_x, self._in_y) = _string_kernel(
            exc, self.line, self._write, self.ex_line, self._ex_write,
            taps, gain, damp, pluck_point, stiff, self._ap_x, self._ap_y,
            polarity, self._low, self._dc_x, self._dc_y, self._in_x,
            self._in_y, result)

        self._apply_level(result, out_level, frames)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        scratch = self._scratch[:frames]
        np.abs(result, out=scratch)
        self._quiet = bool(scratch.max() < 1.0e-5)


# Twenty log ten of two: one octave's worth of decibels, for controls
# written in dB per octave.
DB_PER_OCTAVE = 6.020599913279624


class ModalUnit(Unit):
    """A struck object as a bank of ringing modes.

    Bells, bars, bowls, membranes: none of them is a loop, all of them are a
    handful of resonances, each with its own frequency ratio, weight and
    decay. The mode table carries those three columns and is what makes one
    material sound unlike another; the node layer owns the tables, this unit
    just rings whatever it is given via set_modes().

    'frequency' places the first mode and the rest follow their ratios, so a
    bell transposes as an object rather than as a chord of independent
    partials. 'decay' is the -60 dB time of that first mode; the others scale
    by their table entries, which is where 'low modes outlast high ones'
    lives. 'brightness' tilts the mode weights around the fundamental, and
    'position' imposes the node pattern of striking off-centre -- weights go
    as sin(m * pi * position), so some modes vanish exactly as they do under
    a mallet at their node. Position 0 is the idealized uniform strike.

    A strike is a raised-cosine tap whose width comes from 'hardness': a
    soft mallet is a wide pulse that cannot excite the high modes, a hard
    one is nearly a click that reaches them all -- which is the physical
    reason hardness reads as brightness of attack. Level rides on the
    trigger's height, and anything at the audio inlet excites the bank
    continuously: noise bows it, a body's effort stream makes it a resonator
    for movement.

    The two ways in are normalized differently, because they mean different
    things. The trigger is a mallet: impulse-normalized, so every mode rings
    up to its table weight wherever it sits in frequency, and a strike
    sounds the same at any decay. The excite inlet is heard rather than
    struck, normalized by sqrt(1-r): bowing speaks the moment it touches,
    swells while it is held, and an excitation parked on a mode settles
    against the soft stop on the mode states instead of being multiplied
    out to the mode's Q -- which at a three-second decay would be
    thousands. 'sensitivity' is how keenly that inlet is heard, and it sits
    directly beneath it. The price of the normalization is that a click
    patched in rings only faintly; strikes belong to the trigger, which
    carries velocity anyway. Modes excited past Nyquist are muted rather
    than folded.

    'dry' passes the audio input straight through alongside the ring, and it
    is what separates the unit's two lives. At 0 (the default) the bank IS
    the instrument, and only the modes speak. Opened up -- with the
    frequency fixed rather than tracking, and the decay short enough that
    each mode widens from a partial into a formant -- the bank becomes a
    body: the input carries the note, the modes color it, exactly a bow~
    into a violin box. The dry tap is the input as patched, before the DC
    blocker and without the internal strike, whose click belongs to the
    struck sound and not to a pass-through.
    """

    MAX_MODES = 24
    # Default level of the excite path against the sqrt(1-r)
    # normalization: how keenly the body hears what is patched in, before
    # the state soft-stop has its say. 'sensitivity' scales from here, and
    # 0.7 is kept as the default so that every patch saved when this was
    # called 'drive' sounds exactly as it did.
    SENSE_GAIN = 0.7

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.excite_in = self.new_inlet()
        self.trigger_in = self.new_inlet()
        self.frequency_in = self.new_inlet(base=220.0, minimum=1.0)
        self.pitch_in = self.new_inlet()
        self.decay_in = self.new_inlet(base=3.0, minimum=0.01, maximum=60.0)
        self.brightness_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        # The same axis 'brightness' works on, in a unit that means
        # something: decibels per octave of the mode's own frequency
        # ratio, as additive~ tilts its partials. Brightness is that
        # operation on a nought-to-one knob spanning plus or minus six;
        # this reaches much further and is worth reading off a number.
        # They multiply, so either alone is enough.
        self.tilt_in = self.new_inlet(base=0.0, minimum=-24.0, maximum=24.0)
        self.hardness_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.position_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        # Reaches well past unity: a mode bank up in the kilohertz rings
        # small, and a sparse excitation into it needs real gain. The old
        # ceiling of 2.0 was not enough to balance one against another.
        self.sensitivity_in = self.new_inlet(base=ModalUnit.SENSE_GAIN,
                                             minimum=0.0, maximum=8.0)
        self.dry_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        self.threshold = 0.5
        # ratio, weight, decay multiple -- one row per mode. Replaced whole
        # via set_modes so the audio thread never sees a half-updated table.
        self._modes = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        self._weight_norm = 1.0

        self._s1 = np.zeros(ModalUnit.MAX_MODES, dtype=np.float64)
        self._s2 = np.zeros(ModalUnit.MAX_MODES, dtype=np.float64)
        self._b1 = np.zeros(ModalUnit.MAX_MODES, dtype=np.float64)
        self._b2 = np.zeros(ModalUnit.MAX_MODES, dtype=np.float64)
        self._gains = np.zeros(ModalUnit.MAX_MODES, dtype=np.float64)
        self._gains_live = np.zeros(ModalUnit.MAX_MODES, dtype=np.float64)
        self._rungs = np.zeros(ModalUnit.MAX_MODES, dtype=np.float64)
        # Only a swirled vessel mixes its pairs; everything else sums.
        self._pair_phase = 0.0
        self._pair_step = 0.0
        self._pair_out = 0.0
        self._live_count = 0
        self._coef_key = None
        self._level_live = ModalUnit.SENSE_GAIN
        self._drive_gains = np.zeros(ModalUnit.MAX_MODES, dtype=np.float64)
        self._fm = np.zeros(ModalUnit.MAX_MODES, dtype=np.float64)
        self._theta = np.zeros(ModalUnit.MAX_MODES, dtype=np.float64)
        self._radius = np.zeros(ModalUnit.MAX_MODES, dtype=np.float64)
        self._mode_scratch = np.zeros(ModalUnit.MAX_MODES, dtype=np.float64)

        self._dc_x = 0.0
        self._dc_y = 0.0
        self._pulse_remaining = 0
        self._pulse_length = 1
        self._pulse_at = 0
        self._pulse_amp = 0.0
        self._trigger_armed = True
        self._fire_requests = 0
        self._fire_served = 0
        self._quiet = True

        self.out = self.new_outlet()
        self._exc = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._pulse = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def _geometry(self, frames):
        """The bank as it stands: the table, a pitch scale, and a key
        that says whether either has moved. vessel~ overrides this."""
        return self._modes, 1.0, 0.0

    def set_modes(self, table):
        """Main thread: adopt a mode table, rows of (ratio, weight, decay).

        Coefficients are rebuilt from the table every block, so a mode edited
        while it is ringing retunes live -- dragging a stem in the editor
        glisses the partial rather than cutting it. Only a change in how many
        modes there are clears the ring: the states would otherwise carry
        energy from one mode's history into what is now a different mode.
        """
        rows = [row for row in table[:ModalUnit.MAX_MODES]]
        if not rows:
            rows = [(1.0, 1.0, 1.0)]
        fresh = np.array(rows, dtype=np.float64)
        resized = fresh.shape[0] != self._modes.shape[0]
        self._modes = fresh
        self._weight_norm = max(1.0, float(np.sum(np.abs(fresh[:, 1]))))
        if resized:
            self._s1[:] = 0.0
            self._s2[:] = 0.0

    def bypass_pairs(self):
        # Same reasoning as the string: an object with an audio input
        # bypasses to that input, not to silence.
        return ((self.excite_in, self.out),)

    def fire(self):
        """Request one strike from the node layer. Served on the next block."""
        self._fire_requests += 1

    def reset(self):
        self._s1[:] = 0.0
        self._s2[:] = 0.0
        self._dc_x = 0.0
        self._dc_y = 0.0
        self._pulse_remaining = 0
        self._quiet = True

    def deactivate(self):
        self.reset()

    def _add_pulse(self, exc, start, stop):
        """Mix the active strike pulse into exc[start:stop], advancing it."""
        remaining = self._pulse_remaining
        if remaining <= 0 or stop <= start:
            return
        count = min(stop - start, remaining)
        window = self._scratch[:count]
        np.add(_INDEX_RAMP[:count], float(self._pulse_at - 1), out=window)
        window *= 2.0 * math.pi / self._pulse_length
        np.cos(window, out=window)
        np.subtract(1.0, window, out=window)
        window *= 0.5 * self._pulse_amp
        exc[start:start + count] += window
        self._pulse_at += count
        self._pulse_remaining = remaining - count

    def render(self, frames):
        signal = self.excite_in.eval(frames)
        trigger = self.trigger_in.eval(frames)
        frequency = self.frequency_in.eval(frames)
        pitch = self.pitch_in.eval(frames)
        decay = self.decay_in.eval(frames)
        brightness = self.brightness_in.eval(frames)
        hardness = self.hardness_in.eval(frames)
        position = self.position_in.eval(frames)
        sense_level = self.sensitivity_in.eval(frames)
        dry = self.dry_in.eval(frames)
        out_level = self.level_in.eval(frames)

        out = self.out
        if not _svf_ready.is_set():
            out.set_constant(0.0)
            return

        exc = self._exc[:frames]
        if signal.constant:
            exc[:] = signal.value
            silent_input = signal.value == 0.0
        else:
            np.copyto(exc, signal.data[:frames])
            silent_input = False

        events, self._trigger_armed = _excitation_events(
            trigger, frames, self.threshold, self._trigger_armed)
        if self._fire_requests != self._fire_served:
            self._fire_served = self._fire_requests
            events = ((0, 1.0),) + events

        if (self._quiet and not events and silent_input
                and self._pulse_remaining <= 0):
            out.set_constant(0.0)
            return

        # Mallet width from hardness: 8 ms of felt down to a third of a
        # millisecond of wood, exponentially, since hardness is heard as the
        # ratio of width to period rather than as milliseconds.
        hard = hardness.value if hardness.constant else float(hardness.data[0])
        hard = min(1.0, max(0.0, hard))
        width = int(max(8.0, 0.008 * (0.04 ** hard) * self.sample_rate))
        pulse = self._pulse[:frames]
        pulse[:] = 0.0
        cursor = 0
        for index, amp in events:
            self._add_pulse(pulse, cursor, index)
            cursor = index
            self._pulse_length = width
            self._pulse_remaining = width
            self._pulse_at = 0
            # Area-normalized: a mallet integrates over its dwell, so without
            # this a soft strike would land tens of times harder than a hard
            # one of the same velocity.
            self._pulse_amp = min(2.0, amp) * 2.0 / width
        self._add_pulse(pulse, cursor, frames)

        # Coefficients hold for the block: a handful of vector ops over at
        # most MAX_MODES values, into preallocated views.
        # A subclass can reshape the bank before it is used -- vessel~
        # splits every mode into a pair and pulls the whole thing flat
        # as it fills. Here it is the table exactly as given.
        modes, geom_scale, geom_key = self._geometry(frames)
        count = modes.shape[0]
        ratios = modes[:, 0]
        weights = modes[:, 1]
        decay_scale = modes[:, 2]

        f0 = frequency.value if frequency.constant else float(
            frequency.data[0])
        if not pitch.constant:
            f0 *= 2.0 ** float(pitch.data[0])
        elif pitch.value != 0.0:
            f0 *= 2.0 ** pitch.value
        f0 *= geom_scale
        f0 = min(self.sample_rate * 0.45, max(1.0, f0))

        seconds = decay.value if decay.constant else float(decay.data[0])
        seconds = min(60.0, max(0.01, seconds))

        bright = brightness.value if brightness.constant else float(
            brightness.data[0])
        tilt_db = self.tilt_in.eval(frames)
        tilt_db = (tilt_db.value if tilt_db.constant
                   else float(tilt_db.data[0]))
        tilt_db = min(24.0, max(-24.0, tilt_db))
        struck_at = position.value if position.constant else float(
            position.data[0])
        struck_at = min(1.0, max(0.0, struck_at))
        # The geometry holds until a knob moves: on a struck object at
        # rest the table plumbing here was most of the block's python
        # time. The gain glide below still runs, so sweeps stay smooth.
        coef_key = (count, f0, seconds, bright, round(tilt_db, 3),
                    struck_at, id(self._modes), geom_key)
        theta = self._theta[:count]
        radius = self._radius[:count]
        b1 = self._b1[:count]
        b2 = self._b2[:count]
        gains = self._gains[:count]
        if coef_key != self._coef_key:
            self._coef_key = coef_key
            fm = self._fm[:count]
            np.multiply(ratios, f0, out=fm)
            limit = self.sample_rate * 0.45

            np.clip(fm, 1.0, limit, out=theta)
            theta *= 2.0 * math.pi / self.sample_rate

            np.multiply(decay_scale, seconds * self.sample_rate,
                        out=radius)
            np.clip(radius, 1.0, None, out=radius)
            np.divide(-6.907755, radius, out=radius)
            np.exp(radius, out=radius)

            np.cos(theta, out=b1)
            b1 *= radius
            b1 *= 2.0
            np.multiply(radius, radius, out=b2)
            np.negative(b2, out=b2)

            np.sin(theta, out=gains)
            gains *= weights
            # Divided through by the table's total weight, a velocity-1
            # strike peaks near +-1 whatever the material and however
            # many modes ring.
            gains /= self._weight_norm
            # Mute rather than fold whatever the transposition pushed
            # past Nyquist. The comparison writes 0/1 straight into a
            # float scratch.
            alive = self._mode_scratch[:count]
            np.less_equal(fm, limit, out=alive, casting='unsafe')
            gains *= alive

            # Brightness and tilt are the same operation, so they are
            # done as one: a power of the mode's frequency ratio.
            tilt = (min(1.0, max(0.0, bright)) - 0.5) * 2.0
            tilt += tilt_db / DB_PER_OCTAVE
            if tilt != 0.0:
                shape = self._mode_scratch[:count]
                np.power(ratios, tilt, out=shape)
                # Tilting must not double as a volume control. A slope
                # of a few decibels an octave over a table reaching to
                # the twelfth ratio is a factor of hundreds on the top
                # mode; left alone that is a level jump, and a
                # dangerous one. Normalized to hold the bank's power,
                # tilt moves weight between the modes and nothing else.
                power = float(np.sqrt(np.mean(shape * shape)))
                if power > 1.0e-9:
                    shape /= power
                gains *= shape

            if struck_at > 0.0:
                pattern = self._mode_scratch[:count]
                np.multiply(_INDEX_RAMP[:count], math.pi * struck_at,
                            out=pattern)
                np.sin(pattern, out=pattern)
                np.abs(pattern, out=pattern)
                # Position 0 means the idealized uniform strike, but
                # the node pattern's own limit there is every weight at
                # zero -- a cliff. The first twentieth of the travel
                # crossfades between the two readings, so leaving 0 is
                # a slope rather than a step.
                blend = min(1.0, struck_at / 0.05)
                if blend < 1.0:
                    pattern *= blend
                    pattern += 1.0 - blend
                gains *= pattern

        # Gain-shaping controls -- position, brightness, the table's own
        # weights -- arrive as block-rate steps, and a step in input gain
        # under a sustained drive is a click once a block for as long as
        # the knob moves. The gains the kernel sees glide toward their
        # target over a few blocks instead; a change of mode count adopts
        # the target at once, since gliding between different modes would
        # bleed one mode's level into another's.
        live = self._gains_live[:count]
        if count != self._live_count:
            np.copyto(live, gains)
            self._live_count = count
        else:
            step = self._mode_scratch[:count]
            np.subtract(gains, live, out=step)
            step *= 0.35
            live += step

        # The audio path is a drive, not a mallet, and its gain is the
        # compromise a linear resonator cannot make on its own. Impulse
        # normalization (none) makes a sustained drive explode by the mode's
        # Q; full filter normalization (1-r) makes the first half-second of
        # bowing inaudible at any long decay. sqrt(1-r) splits the
        # difference -- the drive is heard the moment it arrives, grows as
        # it is held, and what would have grown past bounds is taken by the
        # soft stop on the mode states in the kernel.
        level = (sense_level.value if sense_level.constant
                 else float(sense_level.data[0]))
        level = min(2.0, max(0.0, level))
        self._level_live += (level - self._level_live) * 0.35
        drive = self._drive_gains[:count]
        np.subtract(1.0, radius, out=drive)
        np.sqrt(drive, out=drive)
        drive *= live
        drive *= self._level_live

        result = self._y[:frames]
        self._dc_x, self._dc_y, self._pair_phase = _modal_kernel(
            exc, pulse, b1, b2, drive, live, self._s1[:count],
            self._s2[:count], self._dc_x, self._dc_y, result,
            self._rungs[:count], self._pair_phase, self._pair_step,
            self._pair_out)

        # The dry tap: the input as patched, not the excitation buffer --
        # the strike pulse stays out of it.
        if dry.constant:
            if dry.value != 0.0:
                if signal.constant:
                    if signal.value != 0.0:
                        result += dry.value * signal.value
                else:
                    scratch = self._scratch[:frames]
                    np.multiply(signal.data[:frames], dry.value, out=scratch)
                    result += scratch
        else:
            scratch = self._scratch[:frames]
            if signal.constant:
                np.multiply(dry.data[:frames], signal.value, out=scratch)
            else:
                np.multiply(dry.data[:frames], signal.data[:frames],
                            out=scratch)
            result += scratch

        self._apply_level(result, out_level, frames)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        scratch = self._scratch[:frames]
        np.abs(result, out=scratch)
        self._quiet = bool(scratch.max() < 1.0e-5)


def _rub_kernel_source(velocity, contact, force, b1, b2, inject, pickup,
                       s1, s2, free, dc_pole, dc_x, dc_y, out,
                       stick_max, stribeck, mu_floor, sens):
    """Friction closed around the modal bank: bowed glass, sample by sample.

    Where bow~'s friction negotiates with a string's delay lines, this one
    negotiates with the modes themselves. Per sample: every mode rings
    freely from its own history, their velocities sum into the surface the
    bow hair is touching, and the force the contact exerts is poured back
    into every mode. The loop is why a bowed mode blooms rather than just
    being filtered noise -- each slip lands in phase with the motion that
    caused it.

    The contact STICKS or it SLIPS, and which one it is gets decided
    rather than smoothed over. This used to be a single smooth curve of
    force against relative speed, which never holds the surface: with no
    stick the amplitude is set by where friction input balances mode
    damping, and that balance barely moves with bow speed. So the thing
    had a cliff in it -- silence below the threshold where oscillation
    can start, then very nearly full voice above it, twenty-five
    decibels of arrival for a one-and-a-half-fold change in speed, which
    is not something a hand can aim.

    A real bowed oscillator is loud in proportion to bow speed, and the
    reason is kinematic rather than energetic: through the stuck part of
    every cycle the surface is CARRIED at the speed of the hair, so the
    distance it travels -- and thus the amplitude -- goes with that
    speed. It cannot be got from a curve that never quite grips.

    So: the force needed to hold the surface with the hair is worked out
    directly. The injection lands in the state about to be written, so
    the surface velocity after it is what it is now plus the force times
    a sensitivity the bank fixes, and the holding force follows from
    that with no loop to chase. If that force is within what the contact
    can supply -- how hard the hair is pressed, which is what 'force'
    now means -- the surface is held and the two move together. When it
    is not, the contact lets go and the force falls down a Stribeck
    curve: highest as it breaks away, settling toward a sliding floor as
    the speeds diverge. The negative slope in between is what pumps the
    mode, and it is where the sound comes from.
    """
    modes = b1.shape[0]
    for i in range(velocity.shape[0]):
        surface = 0.0
        for m in range(modes):
            ring = b1[m] * s1[m] + b2[m] * s2[m]
            free[m] = ring
            surface += pickup[m] * (ring - s1[m])
        v = velocity[i]
        # What the contact can hold with: the normal force it is pressed
        # on with, faded out as the hair lifts off a stopping bow.
        grip = stick_max * force[i] * contact[i]
        if sens > 1.0e-12:
            held = (v - surface) / sens
        else:
            held = 0.0
        if held <= grip and held >= -grip:
            # Stuck: the surface goes where the hair goes.
            friction = held
        else:
            # Slipping. The force and the speed difference each depend
            # on the other -- the bank's load line crossing the friction
            # curve -- so it is settled by a few passes rather than
            # taken from the last sample, which would put a delay inside
            # the nonlinearity and blunt the break-away.
            friction = grip if held > 0.0 else -grip
            for _ in range(3):
                dv = v - surface - sens * friction
                rel = dv / stribeck
                mu = mu_floor + (1.0 - mu_floor) / (1.0 + rel * rel)
                friction = grip * mu if dv > 0.0 else -grip * mu
        total = 0.0
        for m in range(modes):
            y = free[m] + inject[m] * friction
            if y > 1.5:
                y = 1.5 + np.tanh(y - 1.5)
            elif y < -1.5:
                y = -1.5 - np.tanh(-y - 1.5)
            s2[m] = s1[m]
            s1[m] = y
            total += y
        o = total - dc_x + dc_pole * dc_y
        dc_x = total
        dc_y = o
        out[i] = o
    return dc_x, dc_y


if _HAVE_NUMBA:
    _rub_kernel = njit(cache=True, fastmath=True)(_rub_kernel_source)
else:
    _rub_kernel = _rub_kernel_source


class VesselUnit(ModalUnit):
    """A vessel with water in it: a glass, a bowl, a can, tipped.

    Three things happen when you put water in a ringing vessel, and they
    do not agree with each other.

    Water touching a vibrating wall has to move WITH it. That is mass
    without stiffness, so every mode falls -- and it is weighted by the
    SQUARE of the wall's amplitude, which for a shell held at its base
    and free at its rim goes as the height squared. The loading integral
    is therefore the fill to the FIFTH power: water in the bottom third
    is worth almost nothing and the last centimetre under the rim is
    worth everything. Empty to full is about ten semitones, and two
    thirds of that arrives in the top quarter of the fill.

    TIPPING barely moves the pitch at all -- under a semitone at thirty
    degrees, which was not what I expected. What tipping does is make
    the loading uneven AROUND the wall, and that splits the modes. An
    upright vessel has them in degenerate pairs, cos n0 and sin n0 at
    one frequency, standing at whatever angle they like; break the
    symmetry and the pair comes apart, and two close frequencies BEAT.
    Which perturbation splits which pair is not free: a pair of order n
    is split by the 2n-th harmonic of the loading and by nothing else. A
    tilted plane is almost entirely a first harmonic, so at small tips
    almost nothing happens -- beat periods of minutes. The fourth
    harmonic only really arrives once the water line runs into the base
    or the rim, and then it arrives fast. Tip a little and it sits
    still; past about twenty degrees it starts to warble; tip far and it
    flutters. That threshold is the geometry, not a curve.

    And tipping sets the water SLOSHING, at three-ish hertz almost
    regardless of how full it is -- the tanh in the wave speed saturates
    once the depth passes a third of the radius. That rides on top as a
    slow waver in pitch while it settles, so a quick tilt warbles and
    comes to rest the way a real one does.

    'fill' is how full, 'tip' the angle in degrees, 'size' the radius in
    metres -- which sets the slosh rate and nothing else, the ringing
    pitch being 'frequency' as everywhere.
    """

    # Water's effective mass over the shell's own with the vessel full,
    # for a glass about a millimetre and a half thick. Empty to full is
    # 1/sqrt(1+this), about ten semitones.
    MU_FULL = 2.3333
    # How tall the vessel is against its radius. Sets how far the water
    # line swings for a given tip, and so how soon it clips.
    ASPECT = 3.43
    # First antisymmetric sloshing mode of a cylinder: k = 1.841 / R.
    SLOSH_K = 1.841
    # How hard a change of tip throws the water, and how fast the slop
    # dies away.
    SLOSH_KICK = 0.55
    SLOSH_DECAY = 1.6
    # A swirl does not just point the tilt somewhere new -- it pushes
    # the water round, over and over, at whatever rate the hand goes.
    # Near the sloshing rate that pushing is resonant and the water
    # climbs the wall, which is the entire reason anybody swirls a
    # glass. Nothing here does the climbing; the slosh oscillator is
    # simply driven, and being a resonance it answers loudest when it
    # is asked at its own rate.
    SWIRL_DRIVE = 2.4
    # Points around the wall for the harmonic that does the splitting.
    # It has converged by a hundred and twenty-eight.
    RIM_POINTS = 128
    # How fast the pickup walks round the split pair for each turn of
    # the vessel. The pattern that splits a pair of order two has four
    # bellies, so it is four -- and the sidebands come out spaced at
    # four times the swirl, which is what the coupled pair does when it
    # is integrated directly.
    SWIRL_ORDER = 4.0

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.fill_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.tip_in = self.new_inlet(base=0.0, minimum=0.0, maximum=60.0)
        self.size_in = self.new_inlet(base=0.035, minimum=0.005,
                                      maximum=0.5)
        # Where the water sits low, against where the vessel is struck.
        # A quarter of the pattern's period is the default because that
        # is what equal weights meant before there was a control for it.
        self.turn_in = self.new_inlet(base=22.5, minimum=0.0,
                                      maximum=360.0)
        # Turns per second of the low point going round the rim. A
        # swirl, as against a tilt held still.
        self.swirl_in = self.new_inlet(base=0.0, minimum=-8.0,
                                       maximum=8.0)
        # Every mode becomes two, so the table it is given can hold half
        # what modal~'s can. The materials all fit.
        self._pair = np.zeros((ModalUnit.MAX_MODES, 3), dtype=np.float64)
        self._theta_ring = (np.arange(VesselUnit.RIM_POINTS)
                            * (2.0 * math.pi / VesselUnit.RIM_POINTS))
        self._cos_ring = np.cos(self._theta_ring)
        self._cos4_ring = np.cos(4.0 * self._theta_ring)
        self._wet = np.zeros(VesselUnit.RIM_POINTS, dtype=np.float64)
        self._slosh_x = 0.0
        self._slosh_v = 0.0
        self._tip_last = None
        self._swirl_phase = 0.0
        self._geom_cache = None

    def reset(self):
        super().reset()
        self._slosh_x = 0.0
        self._slosh_v = 0.0
        self._tip_last = None
        self._swirl_phase = 0.0
        self._geom_cache = None

    def _loading(self, fill, tip_deg):
        """Mean loading and the fourth harmonic of it, around the wall.

        The loading is the water line to the fifth power, so it can be
        had directly -- no integral up the wall. Clipping at the base
        and the rim is where the fourth harmonic comes from, so it is
        kept rather than expanded away, and the plane is shifted until
        the volume comes back to what it was.
        """
        span = math.tan(math.radians(max(0.0, tip_deg))) / VesselUnit.ASPECT
        level = fill
        ring = self._wet
        for _ in range(24):
            np.multiply(self._cos_ring, span, out=ring)
            ring += level
            np.clip(ring, 0.0, 1.0, out=ring)
            level += fill - float(ring.mean())
        np.multiply(self._cos_ring, span, out=ring)
        ring += level
        np.clip(ring, 0.0, 1.0, out=ring)
        np.power(ring, 5.0, out=ring)
        mean = float(ring.mean())
        if mean <= 1.0e-12:
            return 0.0, 0.0
        fourth = abs(2.0 * float(np.dot(ring, self._cos4_ring))
                     / (VesselUnit.RIM_POINTS * mean))
        return mean, fourth

    def _geometry(self, frames):
        fill = self.fill_in.eval(frames)
        fill = fill.value if fill.constant else float(fill.data[0])
        fill = min(1.0, max(0.0, fill))
        tip = self.tip_in.eval(frames)
        tip = tip.value if tip.constant else float(tip.data[0])
        tip = min(60.0, max(0.0, tip))
        radius = self.size_in.eval(frames)
        radius = radius.value if radius.constant else float(radius.data[0])
        radius = min(0.5, max(0.005, radius))
        turn = self.turn_in.eval(frames)
        turn = turn.value if turn.constant else float(turn.data[0])
        swirl = self.swirl_in.eval(frames)
        swirl = swirl.value if swirl.constant else float(swirl.data[0])
        swirl = min(8.0, max(-8.0, swirl))
        # Held still, the pair is summed as everything else is summed.
        # Swirled, the pickup walks round it -- and the walking is what
        # puts the sidebands there.
        if swirl == 0.0:
            self._pair_out = 0.0
            self._pair_step = 0.0
        else:
            self._pair_out = 1.0
            self._pair_step = (VesselUnit.SWIRL_ORDER * 2.0 * math.pi
                               * swirl / self.sample_rate)

        # Sloshing is set going by a CHANGE of tip, not by tip itself: a
        # vessel held at an angle is still, one just moved is not.
        if self._tip_last is None:
            self._tip_last = tip
        moved = tip - self._tip_last
        self._tip_last = tip
        depth = max(1.0e-4, fill) * VesselUnit.ASPECT * radius
        k = VesselUnit.SLOSH_K / radius
        omega = math.sqrt(9.80665 * k * math.tanh(k * depth))
        dt = frames / self.sample_rate
        self._slosh_v += (moved / 60.0) * VesselUnit.SLOSH_KICK
        # And the swirl pushes it round and round.
        self._swirl_phase += 2.0 * math.pi * swirl * dt
        if self._swirl_phase > 6.283185307179586:
            self._swirl_phase -= 6.283185307179586
        elif self._swirl_phase < 0.0:
            self._swirl_phase += 6.283185307179586
        if swirl != 0.0:
            self._slosh_v += (VesselUnit.SWIRL_DRIVE * abs(swirl)
                              * (tip / 60.0)
                              * math.sin(self._swirl_phase) * dt)
        self._slosh_v -= (omega * omega * self._slosh_x
                          + 2.0 * VesselUnit.SLOSH_DECAY * self._slosh_v) * dt
        self._slosh_x += self._slosh_v * dt
        if self._slosh_x > 0.5:
            self._slosh_x = 0.5
        elif self._slosh_x < -0.5:
            self._slosh_x = -0.5
        wet = min(1.0, max(0.0, fill + self._slosh_x))

        key = (round(wet, 4), round(tip, 3), round(turn, 2))
        if self._geom_cache is None or self._geom_cache[0] != key:
            mean, fourth = self._loading(wet, tip)
            mu = VesselUnit.MU_FULL * mean
            scale = 1.0 / math.sqrt(1.0 + mu)
            split = mu * fourth / (2.0 * (1.0 + mu))
            # Which of the split pair a blow wakes depends on where it
            # lands against where the water is. Strike where the pattern
            # has a belly and only that one answers -- one frequency, no
            # beat at all. Strike between them and both answer equally,
            # which is where the beat is deepest. It goes round every
            # ninety degrees because the pattern that splits a pair of
            # order two has four bellies.
            phase = math.radians(turn) * 2.0
            # AMPLITUDES, not shares of one. A blow between the bellies
            # puts as much in as a blow on one -- it just divides it
            # between the two, and two amplitudes add as their squares.
            # Splitting the weight instead made the middle of the turn
            # three decibels quieter than its ends, which is a knob
            # doubling as a volume control.
            near = abs(math.cos(phase))
            far = abs(math.sin(phase))
            self._geom_cache = (key, scale, split, near, far)
        _, scale, split, near, far = self._geom_cache

        source = self._modes
        count = min(source.shape[0], ModalUnit.MAX_MODES // 2)
        pair = self._pair[:count * 2]
        # The two members of a split pair share what the one mode had,
        # so filling a vessel does not also make it louder.
        for index in range(count):
            ratio, weight, decay = source[index]
            # The two share what the one mode had, so turning the
            # vessel moves the sound between them without making it
            # louder or quieter.
            pair[2 * index] = (ratio * (1.0 - split), weight * near, decay)
            pair[2 * index + 1] = (ratio * (1.0 + split), weight * far,
                                   decay)
        return pair, scale, (key, count)


class RubUnit(Unit):
    """Bowed modal object: glass, bowl, bar or bell under a bow.

    The complement of modal~'s mallet, and the second half of the pair
    bow~ began: friction fused with a resonator, this time the resonator
    being the mode table rather than a string. It shares modal~'s tables
    and bow~'s hands -- velocity is the bow's speed, force its weight (a
    heavier bow widens the sticking region and presses the tone quieter),
    position where on the object the hair lands, silencing the modes with
    a node there.

    Played gently it locks to the lowest live mode and sings nearly pure,
    which is what a wine glass does and why; faster bowing pulls the tone
    sharp and then breaks upward to higher modes -- the squeal -- and
    velocity past 1 is deliberately into that territory. Slowing to a stop
    lifts the bow, and the object rings down at its own decay; there is no
    trigger anywhere, and striking belongs to modal~.

    Mode tables arrive through set_modes() as everywhere; edits while
    sounding retune the ring live, count changes clear it.
    """

    MAX_MODES = 24
    MIN_FREQUENCY = 20.0
    # Friction-to-bank coupling: gentle enough that the capture is clean
    # and the fundamental regime wide. Found empirically, like everything
    # about a nonlinear oscillator's operating point.
    COUPLING = 0.5
    # User velocity 0..1 onto the internal range where the fundamental
    # regime lives; past 1 climbs into the mode-jump squeals.
    #
    # This is where the playable wedge SITS on the knob, and it is worth
    # knowing that moving it does not soften the step into oscillation,
    # only relocate it. At 0.3 the tone arrives a third of the way up
    # and the level spreads over ten decibels instead of twenty-five --
    # but a velocity of 0.7 then locks to a higher mode instead of the
    # fundamental, so the fundamental regime is what gets paid. Left
    # where it plays.
    VELOCITY_SCALE = 0.12
    # Below this internal speed the hair is lifting off: contact fades so
    # a stopped bow releases the ring instead of damping it dead.
    CONTACT_VELOCITY = 0.005
    # What the contact can hold with, per unit of 'force'. 'force' is
    # the normal force now -- how hard the hair is pressed -- rather
    # than a width for a curve, which is both what it is called and what
    # decides when the surface breaks away.
    STICK_FORCE = 0.2
    # The speed difference over which sliding friction falls from its
    # break-away value toward the floor. The negative slope across it is
    # what pumps the mode, and its width is what decides how far up the
    # bow speed the fundamental holds before the bank jumps to a higher
    # mode: narrow and it squeals almost at once, wide and it holds
    # through most of the range and jumps near the top, which is where
    # the jump belongs.
    STRIBECK = 0.2
    # Sliding friction as a fraction of break-away.
    MU_FLOOR = 0.3

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.velocity_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.5)
        self.force_in = self.new_inlet(base=0.4, minimum=0.0, maximum=1.0)
        self.position_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.frequency_in = self.new_inlet(base=440.0,
                                           minimum=RubUnit.MIN_FREQUENCY)
        self.pitch_in = self.new_inlet()
        self.decay_in = self.new_inlet(base=3.0, minimum=0.01, maximum=60.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        self._modes = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        self._weight_norm = 1.0

        self._s1 = np.zeros(RubUnit.MAX_MODES, dtype=np.float64)
        self._s2 = np.zeros(RubUnit.MAX_MODES, dtype=np.float64)
        self._b1 = np.zeros(RubUnit.MAX_MODES, dtype=np.float64)
        self._b2 = np.zeros(RubUnit.MAX_MODES, dtype=np.float64)
        self._inject = np.zeros(RubUnit.MAX_MODES, dtype=np.float64)
        self._pickup = np.zeros(RubUnit.MAX_MODES, dtype=np.float64)
        self._inject_live = np.zeros(RubUnit.MAX_MODES, dtype=np.float64)
        self._pickup_live = np.zeros(RubUnit.MAX_MODES, dtype=np.float64)
        self._live_count = 0
        self._coef_key = None
        self._free = np.zeros(RubUnit.MAX_MODES, dtype=np.float64)
        self._fm = np.zeros(RubUnit.MAX_MODES, dtype=np.float64)
        self._theta = np.zeros(RubUnit.MAX_MODES, dtype=np.float64)
        self._radius = np.zeros(RubUnit.MAX_MODES, dtype=np.float64)
        self._mode_scratch = np.zeros(RubUnit.MAX_MODES, dtype=np.float64)

        self._dc_x = 0.0
        self._dc_y = 0.0
        self._quiet = True

        self.out = self.new_outlet()
        self._vel = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._contact = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._force = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._freq = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def set_modes(self, table):
        """Main thread: adopt a mode table, rows of (ratio, weight, decay).

        Same live-edit contract as modal~: value edits retune the ring,
        only a change of mode count clears it.
        """
        rows = [row for row in table[:RubUnit.MAX_MODES]]
        if not rows:
            rows = [(1.0, 1.0, 1.0)]
        fresh = np.array(rows, dtype=np.float64)
        resized = fresh.shape[0] != self._modes.shape[0]
        self._modes = fresh
        self._weight_norm = max(1.0, float(np.sum(np.abs(fresh[:, 1]))))
        if resized:
            self._s1[:] = 0.0
            self._s2[:] = 0.0

    def reset(self):
        self._s1[:] = 0.0
        self._s2[:] = 0.0
        self._dc_x = 0.0
        self._dc_y = 0.0
        self._quiet = True

    def deactivate(self):
        self.reset()

    def render(self, frames):
        velocity = self.velocity_in.eval(frames)
        force = self.force_in.eval(frames)
        position = self.position_in.eval(frames)
        frequency = self.frequency_in.eval(frames)
        pitch = self.pitch_in.eval(frames)
        decay = self.decay_in.eval(frames)
        out_level = self.level_in.eval(frames)

        out = self.out
        if not _svf_ready.is_set():
            out.set_constant(0.0)
            return

        vel = self._vel[:frames]
        if velocity.constant:
            vel[:] = velocity.value
            idle = abs(velocity.value) < 1.0e-4
        else:
            np.copyto(vel, velocity.data[:frames])
            idle = False
        np.clip(vel, 0.0, 2.0, out=vel)
        vel *= RubUnit.VELOCITY_SCALE

        if self._quiet and idle:
            out.set_constant(0.0)
            return

        contact = self._contact[:frames]
        np.divide(vel, RubUnit.CONTACT_VELOCITY, out=contact)
        np.clip(contact, 0.0, 1.0, out=contact)

        push = self._force[:frames]
        if force.constant:
            push[:] = force.value
        else:
            np.copyto(push, force.data[:frames], casting='unsafe')
        np.clip(push, 0.0, 1.0, out=push)

        freq = self._freq[:frames]
        self._build_hertz(freq, frequency, pitch, frames,
                          RubUnit.MIN_FREQUENCY)
        f0 = float(freq[0])

        modes = self._modes
        count = modes.shape[0]
        ratios = modes[:, 0]
        weights = modes[:, 1]
        decay_scale = modes[:, 2]

        seconds = decay.value if decay.constant else float(decay.data[0])
        seconds = min(60.0, max(0.01, seconds))

        struck_at = position.value if position.constant else float(
            position.data[0])
        struck_at = min(1.0, max(0.0, struck_at))
        # The geometry holds until a knob moves; the glide below still
        # runs every block, so sweeps stay clickless. Most blocks the
        # table plumbing here cost more python time than the kernel.
        coef_key = (count, f0, seconds, struck_at, id(modes))
        theta = self._theta[:count]
        radius = self._radius[:count]
        b1 = self._b1[:count]
        b2 = self._b2[:count]
        inject = self._inject[:count]
        pickup = self._pickup[:count]
        if coef_key != self._coef_key:
            self._coef_key = coef_key
            fm = self._fm[:count]
            np.multiply(ratios, f0, out=fm)
            limit = self.sample_rate * 0.45

            np.clip(fm, 1.0, limit, out=theta)
            theta *= 2.0 * math.pi / self.sample_rate

            np.multiply(decay_scale, seconds * self.sample_rate,
                        out=radius)
            np.clip(radius, 1.0, None, out=radius)
            np.divide(-6.907755, radius, out=radius)
            np.exp(radius, out=radius)

            np.cos(theta, out=b1)
            b1 *= radius
            b1 *= 2.0
            np.multiply(radius, radius, out=b2)
            np.negative(b2, out=b2)

            np.sin(theta, out=inject)
            inject *= weights
            inject /= self._weight_norm
            inject *= RubUnit.COUPLING
            alive = self._mode_scratch[:count]
            np.less_equal(fm, limit, out=alive, casting='unsafe')
            inject *= alive

            pickup[:] = 1.0
            pickup *= alive
            if struck_at > 0.0:
                pattern = self._mode_scratch[:count]
                np.multiply(_INDEX_RAMP[:count], math.pi * struck_at,
                            out=pattern)
                np.sin(pattern, out=pattern)
                np.abs(pattern, out=pattern)
                # Continuous out of 0 for the same reason as modal~:
                # the pattern's limit there contradicts the uniform
                # reading.
                blend = min(1.0, struck_at / 0.05)
                if blend < 1.0:
                    pattern *= blend
                    pattern += 1.0 - blend
                # Both ends of the coupling: bowing at a mode's node
                # neither hears nor moves it, which is reciprocity.
                inject *= pattern
                pickup *= pattern

        # Coupling weights glide toward their targets, as modal~'s gains
        # do: a position sweep while bowing would otherwise step the
        # friction's grip on the modes once a block, which is a click.
        inject_live = self._inject_live[:count]
        pickup_live = self._pickup_live[:count]
        if count != self._live_count:
            np.copyto(inject_live, inject)
            np.copyto(pickup_live, pickup)
            self._live_count = count
        else:
            step = self._mode_scratch[:count]
            np.subtract(inject, inject_live, out=step)
            step *= 0.35
            inject_live += step
            np.subtract(pickup, pickup_live, out=step)
            step *= 0.35
            pickup_live += step

        corner = min(40.0, max(1.0, f0 * 0.25))
        dc_pole = math.exp(-2.0 * math.pi * corner / self.sample_rate)

        result = self._y[:frames]
        self._dc_x, self._dc_y = _rub_kernel(
            vel, contact, push, b1, b2, inject_live, pickup_live,
            self._s1[:count], self._s2[:count], self._free[:count],
            dc_pole, self._dc_x, self._dc_y, result,
            RubUnit.STICK_FORCE, RubUnit.STRIBECK, RubUnit.MU_FLOOR,
            float(np.dot(pickup_live, inject_live)))

        self._apply_level(result, out_level, frames)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        scratch = self._scratch[:frames]
        np.abs(result, out=scratch)
        self._quiet = bool(scratch.max() < 1.0e-5)


def _wind_kernel_source(pressure, noise_amt, noise, noise_at,
                        bore, bwrite, jet, jwrite,
                        bore_delay, jet_delay, damp,
                        slope, jet_refl, end_refl, mode,
                        low, dcb_x, dcb_y, dc_x, dc_y, out):
    """Breath through a nonlinear valve into a bore, sample by sample.

    Unlike the string, nothing here is triggered: the tone is a limit cycle.
    Breath pressure feeds a nonlinearity whose gain depends on the very wave
    coming back up the bore, and above a threshold the loop's small
    disturbances grow into oscillation -- which is what starting a note on a
    wind instrument is. The noise riding on the breath is not decoration; it
    is the perturbation the jet amplifies into speech, and the reed's
    breathiness.

    mode 0 is a reed on a closed bore, after Smith/Cook: the returning wave
    is reflected inverted, and the reed table -- a clipped linear reflection
    coefficient in the pressure difference -- lets more energy through as the
    player bites. Odd harmonics, speaks near half pressure, clarinet.

    mode 1 is an air jet across an open bore, after Cook's flute: the bore's
    return steers a jet whose own travel time (the jet line, a fraction of
    the bore set by embouchure) delays that steering, and the cubic
    x*(x*x - 1) is the jet switching sides. It needs most of a full breath
    before it speaks, overblows as the jet length leaves the middle of its
    range -- to the octave one way, the twelfth the other -- and
    has a narrow wolf just past full pressure where the jet's operating
    point crosses the outer zero of the cubic and the note cracks an octave
    down -- as the real instrument does.

    Both bores are clamped at +-2: the failure mode of a nonlinear loop
    driven hard should be saturation, never runaway.
    """
    bsize = bore.shape[0]
    jsize = jet.shape[0]
    blimit = bsize - 3.0
    jlimit = jsize - 3.0
    nsize = noise.shape[0]
    for i in range(pressure.shape[0]):
        breath = pressure[i]
        breath += breath * noise_amt[i] * noise[noise_at]
        noise_at += 1
        if noise_at >= nsize:
            noise_at = 0

        want = bore_delay[i]
        if want < 2.0:
            want = 2.0
        elif want > blimit:
            want = blimit
        read = bwrite - want
        if read < 0.0:
            read += bsize
        bore_out = _cubic_read(bore, bsize, read)

        low += (bore_out - low) * (1.0 - damp[i])

        if mode == 0:
            pd = -0.95 * low - breath
            r = 0.7 + slope * pd
            if r > 1.0:
                r = 1.0
            elif r < -1.0:
                r = -1.0
            v = breath + pd * r
        else:
            temp = low - dcb_x + 0.995 * dcb_y
            dcb_x = low
            dcb_y = temp

            jwant = jet_delay[i]
            if jwant < 2.0:
                jwant = 2.0
            elif jwant > jlimit:
                jwant = jlimit
            jread = jwrite - jwant
            if jread < 0.0:
                jread += jsize
            x = _cubic_read(jet, jsize, jread)
            jet[jwrite] = breath - jet_refl * temp
            jwrite += 1
            if jwrite >= jsize:
                jwrite = 0

            v = x * (x * x - 1.0)
            if v > 1.0:
                v = 1.0
            elif v < -1.0:
                v = -1.0
            v += end_refl * temp

        if v > 2.0:
            v = 2.0
        elif v < -2.0:
            v = -2.0
        bore[bwrite] = v

        o = bore_out - dc_x + 0.995 * dc_y
        dc_x = bore_out
        dc_y = o
        out[i] = o

        bwrite += 1
        if bwrite >= bsize:
            bwrite = 0
    return bwrite, jwrite, noise_at, low, dcb_x, dcb_y, dc_x, dc_y


if _HAVE_NUMBA:
    _wind_kernel = njit(cache=True, fastmath=True)(_wind_kernel_source)
else:
    _wind_kernel = _wind_kernel_source


def _brass_kernel_source(pressure, noise_amt, noise, noise_at,
                         bore, write, delay, damp,
                         lip_b1, lip_b2, lip_b0, bias,
                         refl_gain, mute, hp_k, fb1, fb2, fg,
                         bb1, bb2, bg, m_dir, m_buzz, m_stem,
                         cav, comb_d, comb_g, bark,
                         low, rb_x, rb_y, dp1, dp2, y1, y2, f1, f2,
                         g1, g2, hpst, cidx, dc_x, dc_y, out):
    """The lip valve against the bore, sample by sample.

    The lip is not a table but an oscillator: a bandpass resonator (it
    answers the pressure WAVE, not the pressure -- a lip held open by
    static breath is just open) whose displacement, plus the static
    aperture 'bias', sets the opening area, and the transmitted pressure
    is the crossfade between mouth and bore by that area -- a convex
    combination, bounded by construction. The drive is the returning
    wave minus the breath: the lip is blown open by the bore's reply,
    the outward-striking door, which is the sign that locks the coupled
    system ON the bore's harmonics rather than between them.

    Two vents keep the loop honest: the bell vents DC (a bore is open --
    static pressure escapes, waves return), without which the bore
    charges up to the breath and the pressure difference that drives
    everything collapses.

    The mute is a harmon: a cork-sealed cavity in the bell. It works
    on both sides. To the lips, the seal reflects more back across
    more of the spectrum -- the stuffed feel. To the room, the seal
    strips the body from the sound (a highpass: the fundamental
    barely escapes, which is the famous thinness), and what remains
    leaves through the small hole -- its own fixed buzzy resonance,
    stem out -- or through the stem, whose resonance is the surface
    the hand plays: 'wah' sweeps it, working on the thinned sound,
    which is why a harmon wah talks where a plain formant only
    filters.

    What the coupled system does from there is brass: the sounding pitch
    locks to whichever bore harmonic sits nearest the lip's resonance,
    a continuous lip sweep climbs the series in discrete steps, and too
    much breath cracks the lock down to the pedal. None of it is coded;
    it is what a valve with its own resonance does against a comb of
    bore modes.
    """
    size = bore.shape[0]
    limit = size - 3.0
    nsize = noise.shape[0]
    for i in range(pressure.shape[0]):
        want = delay[i]
        if want < 2.0:
            want = 2.0
        elif want > limit:
            want = limit
        read = write - want
        if read < 0.0:
            read += size
        p_bore = _cubic_read(bore, size, read)

        low += (p_bore - low) * (1.0 - damp[i])
        r = refl_gain * low
        refl = r - rb_x + 0.995 * rb_y
        rb_x = r
        rb_y = refl

        breath = pressure[i]
        breath += breath * noise_amt[i] * noise[noise_at]
        noise_at += 1
        if noise_at >= nsize:
            noise_at = 0

        dp = refl - breath
        y = lip_b0 * (dp - dp2) + lip_b1 * y1 + lip_b2 * y2
        dp2 = dp1
        dp1 = dp
        y2 = y1
        y1 = y

        opening = y + bias
        area = opening * opening
        if area > 1.0:
            area = 1.0
        v = area * breath + (1.0 - area) * refl
        if v > 2.0:
            v = 2.0
        elif v < -2.0:
            v = -2.0
        bore[write] = v

        o = p_bore - dc_x + 0.995 * dc_y
        dc_x = p_bore
        dc_y = o
        hpst += (o - hpst) * hp_k
        thin = o - hpst
        # The stem is a tube from INSIDE the seal: its resonance taps
        # the full bell sound, lows and all, which is why a closed
        # hand can still honk dark instead of vanishing.
        fm = fg * o + fb1 * f1 + fb2 * f2
        f2 = f1
        f1 = fm
        bz = bg * thin + bb1 * g1 + bb2 * g2
        g2 = g1
        g1 = bz
        esc = m_dir * thin + m_buzz * bz
        # The bark: sound squeezing through a small hole is not
        # linear -- louder moments snarl. Odd, DC-free, bounded.
        a = esc if esc >= 0.0 else -esc
        if a > 1.2:
            a = 1.2
        esc = esc * (1.0 + bark * a)
        # The hollowness AND the tin: the cavity is a shallow metal
        # space. Feedforward cuts the notches (cupped hands);
        # feedback makes it RING at its harmonics -- the small tin
        # chamber answering the jet.
        rd = cidx - comb_d
        if rd < 0:
            rd += cav.shape[0]
        held = cav[rd]
        hollow = (esc - comb_g * held) * 0.8
        cav[cidx] = esc + 0.55 * held
        cidx += 1
        if cidx >= cav.shape[0]:
            cidx = 0
        # The stem tube passes THROUGH the cork to open air: its voice
        # joins after the cavity, not inside it, which is why the wah
        # keeps its full vowel over the hollow. It barks on its own.
        sfm = m_stem * fm
        a2 = sfm if sfm >= 0.0 else -sfm
        if a2 > 1.2:
            a2 = 1.2
        sfm = sfm * (1.0 + bark * a2)
        out[i] = (1.0 - mute) * o + mute * (hollow + sfm)

        write += 1
        if write >= size:
            write = 0
    return (write, noise_at, low, rb_x, rb_y, dp1, dp2, y1, y2, f1, f2,
            g1, g2, hpst, cidx, dc_x, dc_y)


if _HAVE_NUMBA:
    _brass_kernel = njit(cache=True, fastmath=True)(_brass_kernel_source)
else:
    _brass_kernel = _brass_kernel_source


class BrassUnit(Unit):
    """A brass instrument: one bore, many notes, chosen by the lip.

    The third of the pressure-played instruments, and the one where the
    valve has a will of its own. wind~'s reed is fast and obedient;
    brass lips are an oscillator with their own resonance, and the note
    that sounds is a negotiation between that resonance and the bore's
    harmonic comb: the system locks to the nearest bore mode, so
    'frequency' is the size of the instrument (its pedal fundamental)
    and 'lip' -- tension, mapped across the first sixteen harmonics --
    picks which partial speaks. Sweep it and the pitch climbs the series
    in steps, like a bugler; nobody quantized anything.

    'pressure' is the breath, with the family's usual habits: a
    threshold to speak, dynamics above it, and -- being brass -- pushed
    hard it cracks the lock and blats down toward the pedal. 'breath'
    is turbulence, 'brightness' the bell's keeping of highs. No trigger
    anywhere: tonguing is an adsr~ on pressure, and an effort stream is
    an embouchure.
    """

    MODES = ()
    MIN_FREQUENCY = 20.0
    LIP_BIAS = 0.5
    LIP_GAIN = 4.0
    # The embouchure tightens as it climbs: Q rises with tension, which is
    # both the physiology and what keeps the high partials projecting.
    LIP_Q_LOW = 8.0
    LIP_Q_HIGH = 24.0
    MAX_HARMONIC = 16.0

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.pressure_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.5)
        self.lip_in = self.new_inlet(base=0.14, minimum=0.0, maximum=1.0)
        self.frequency_in = self.new_inlet(base=110.0,
                                           minimum=BrassUnit.MIN_FREQUENCY)
        self.pitch_in = self.new_inlet()
        self.brightness_in = self.new_inlet(base=0.7, minimum=0.0,
                                            maximum=1.0)
        self.noise_in = self.new_inlet(base=0.02, minimum=0.0, maximum=1.0)
        self.mute_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.stem_in = self.new_inlet(base=1.0, minimum=0.0, maximum=1.0)
        self.wah_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        size = int(self.sample_rate / BrassUnit.MIN_FREQUENCY) + 8
        self.bore = np.zeros(size, dtype=np.float64)
        self._write = 0
        self._low = 0.0
        self._rb_x = 0.0
        self._rb_y = 0.0
        self._dp1 = 0.0
        self._dp2 = 0.0
        self._y1 = 0.0
        self._y2 = 0.0
        self._f1 = 0.0
        self._f2 = 0.0
        self._g1 = 0.0
        self._g2 = 0.0
        self._hpst = 0.0
        self._cav = np.zeros(256, dtype=np.float64)
        self._cidx = 0
        self._wah_live = 0.5
        self._dc_x = 0.0
        self._dc_y = 0.0

        generator = np.random.default_rng(20260810)
        self._noise = generator.uniform(-1.0, 1.0, 1 << 16)
        self._noise_at = 0
        self._quiet = True

        self.out = self.new_outlet()
        self._press = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._namt = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._freq = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._delay = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._damp = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self.bore[:] = 0.0
        self._low = 0.0
        self._rb_x = 0.0
        self._rb_y = 0.0
        self._dp1 = 0.0
        self._dp2 = 0.0
        self._y1 = 0.0
        self._y2 = 0.0
        self._f1 = 0.0
        self._f2 = 0.0
        self._g1 = 0.0
        self._g2 = 0.0
        self._hpst = 0.0
        self._cav[:] = 0.0
        self._cidx = 0
        self._dc_x = 0.0
        self._dc_y = 0.0
        self._quiet = True

    def deactivate(self):
        self.reset()

    def render(self, frames):
        pressure = self.pressure_in.eval(frames)
        lip = self.lip_in.eval(frames)
        frequency = self.frequency_in.eval(frames)
        pitch = self.pitch_in.eval(frames)
        brightness = self.brightness_in.eval(frames)
        noise = self.noise_in.eval(frames)
        mute = self.mute_in.eval(frames)
        stem = self.stem_in.eval(frames)
        wah = self.wah_in.eval(frames)
        out_level = self.level_in.eval(frames)

        out = self.out
        if not _svf_ready.is_set():
            out.set_constant(0.0)
            return

        press = self._press[:frames]
        if pressure.constant:
            press[:] = pressure.value
            idle = abs(pressure.value) < 1.0e-4
        else:
            np.copyto(press, pressure.data[:frames])
            idle = False
        np.clip(press, 0.0, 2.0, out=press)

        if self._quiet and idle:
            out.set_constant(0.0)
            return

        freq = self._freq[:frames]
        self._build_hertz(freq, frequency, pitch, frames,
                          BrassUnit.MIN_FREQUENCY)
        f0 = float(freq[0])

        def scalar(signal, lo, hi):
            value = signal.value if signal.constant else float(signal.data[0])
            return min(hi, max(lo, value))

        mute_now = scalar(mute, 0.0, 1.0)
        stem_now = scalar(stem, 0.0, 1.0)
        wah_now = scalar(wah, 0.0, 1.0)
        # The formant glides rather than steps: a hand over a harmon
        # stem is not quantized to blocks.
        self._wah_live += (wah_now - self._wah_live) * 0.2

        damp = self._damp[:frames]
        if brightness.constant:
            damp[:] = 1.0 - brightness.value
        else:
            np.subtract(1.0, brightness.data[:frames], out=damp,
                        casting='unsafe')
        np.clip(damp, 0.0, 0.95, out=damp)
        # The mute's small aperture widens what the bell reflects: the
        # lips get more back, across more of the spectrum. Stuffiness.
        if mute_now > 0.0:
            damp *= 1.0 - 0.45 * mute_now

        namt = self._namt[:frames]
        if noise.constant:
            namt[:] = noise.value
        else:
            np.copyto(namt, noise.data[:frames], casting='unsafe')
        np.clip(namt, 0.0, 1.0, out=namt)

        taps = self._delay[:frames]
        np.divide(self.sample_rate, freq, out=taps)
        np.clip(taps, 2.0, self.bore.shape[0] - 3.0, out=taps)

        # The lip resonator, tuned along the harmonic series at constant
        # Q: its bandwidth scales with its frequency, so it can single
        # out a partial on a big bore and a small one alike.
        tension = lip.value if lip.constant else float(lip.data[0])
        tension = min(1.0, max(0.0, tension))
        harmonic = 1.0 + tension * (BrassUnit.MAX_HARMONIC - 1.0)
        f_lip = min(self.sample_rate * 0.45, harmonic * f0)
        theta = 2.0 * math.pi * f_lip / self.sample_rate
        lip_q = (BrassUnit.LIP_Q_LOW
                 + tension * (BrassUnit.LIP_Q_HIGH - BrassUnit.LIP_Q_LOW))
        r_lip = math.exp(-math.pi * f_lip / (lip_q * self.sample_rate))
        lip_b1 = 2.0 * r_lip * math.cos(theta)
        lip_b2 = -r_lip * r_lip
        lip_b0 = BrassUnit.LIP_GAIN * (1.0 - r_lip)

        refl_gain = 0.95 + 0.035 * mute_now
        # The mute cavity: a vocal formant on what escapes, swept by
        # wah from closed hand (dark) to open (bright). The crossfade
        # completes by seventy percent of the knob -- past that the
        # horn speaks only through the cavity, which is what a harmon
        # is -- and the formant is sharp enough to be a vowel.
        mute_mix = min(1.0, mute_now * 1.4)
        # The seal: the cavity strips the body from what escapes.
        hp_k = 1.0 - math.exp(-2.0 * math.pi * 600.0 / self.sample_rate)
        # The stem: the hand's surface, swept by wah on the THINNED
        # sound -- which is why the harmon vowel talks.
        f_mute = 260.0 * 9.0 ** self._wah_live
        th_f = 2.0 * math.pi * min(f_mute, 0.4 * self.sample_rate) \
            / self.sample_rate
        r_f = 0.965
        fb1 = 2.0 * r_f * math.cos(th_f)
        fb2 = -r_f * r_f
        fg = (1.0 - r_f) * math.sin(th_f) * 1.3
        # The small hole itself: fixed, buzzy, the stem-out voice.
        th_b = 2.0 * math.pi * min(2600.0, 0.4 * self.sample_rate) \
            / self.sample_rate
        r_b = 0.95
        bb1 = 2.0 * r_b * math.cos(th_b)
        bb2 = -r_b * r_b
        bg = (1.0 - r_b) * math.sin(th_b) * 1.2
        # The stem sits IN the hole but air still rushes through the
        # annulus around it: inserting the stem keeps the bright buzz
        # and ADDS the vocal tube over it. The high timbre stays.
        m_dir = 0.18 * (1.0 - 0.7 * stem_now)
        m_buzz = 1.3 * (1.0 - 0.2 * stem_now)
        m_stem = 2.3 * stem_now
        # The cavity's depth, well under a millisecond: notches and
        # ring harmonics every twelve hundred hertz or so. Tin.
        comb_d = max(8, min(250, int(0.0008 * self.sample_rate)))

        result = self._y[:frames]
        (self._write, self._noise_at, self._low, self._rb_x, self._rb_y,
         self._dp1, self._dp2, self._y1, self._y2, self._f1, self._f2,
         self._g1, self._g2, self._hpst, self._cidx,
         self._dc_x, self._dc_y) = _brass_kernel(
            press, namt, self._noise, self._noise_at,
            self.bore, self._write, taps, damp,
            lip_b1, lip_b2, lip_b0, BrassUnit.LIP_BIAS,
            refl_gain, mute_mix, hp_k, fb1, fb2, fg,
            bb1, bb2, bg, m_dir, m_buzz, m_stem,
            self._cav, comb_d, 0.65, 1.4,
            self._low, self._rb_x, self._rb_y, self._dp1, self._dp2,
            self._y1, self._y2, self._f1, self._f2,
            self._g1, self._g2, self._hpst, self._cidx,
            self._dc_x, self._dc_y, result)

        # Down to the family's level: a locked partial peaks near +-1
        # like the other instruments, with the blat above that.
        result *= 0.6
        self._apply_level(result, out_level, frames)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        scratch = self._scratch[:frames]
        np.abs(result, out=scratch)
        self._quiet = bool(scratch.max() < 1.0e-5)


def _bow_kernel_source(velocity, force, neck, bridge, write,
                       neck_delay, bridge_delay, damp, dc_pole,
                       low, dc_x, dc_y, out):
    """Bow against string: friction between two delay lines, sample by sample.

    The bow point cuts the string in two, so there are two lines -- bow to
    nut and bow to bridge -- each reflecting inverted, the bridge side
    through the loss filter. Where they meet, the bow: the velocity
    difference between bow hair and string passes through a friction curve
    that is flat near zero (sticking -- the string travels with the bow) and
    collapses as the difference grows (slipping). Stick, slip once per round
    trip, stick again: that alternation is Helmholtz motion, and the
    sawtooth at the bridge is what a bowed string is. Nothing here makes a
    sawtooth on purpose.

    The curve is Smith's: friction = (|dv * slope| + 0.75)^-4, clipped to 1.
    Slope comes from bow force -- pressing harder widens the sticking region,
    which is why force gates how fast the bow may move and still hold the
    fundamental (the Schelleng diagram, found empirically in the unit's
    velocity mapping).
    """
    size = neck.shape[0]
    limit = size - 3.0
    for i in range(velocity.shape[0]):
        nd = neck_delay[i]
        if nd < 2.0:
            nd = 2.0
        elif nd > limit:
            nd = limit
        bd = bridge_delay[i]
        if bd < 2.0:
            bd = 2.0
        elif bd > limit:
            bd = limit

        read = write - bd
        if read < 0.0:
            read += size
        bridge_out = _cubic_read(bridge, size, read)
        read = write - nd
        if read < 0.0:
            read += size
        neck_out = _cubic_read(neck, size, read)

        low += (bridge_out - low) * (1.0 - damp[i])
        bridge_refl = -0.95 * low
        nut_refl = -neck_out

        dv = velocity[i] - (bridge_refl + nut_refl)
        slope = 5.0 - 4.0 * force[i]
        t = abs(dv * slope) + 0.75
        c = 1.0 / (t * t * t * t)
        if c > 1.0:
            c = 1.0
        new_vel = dv * c

        v = bridge_refl + new_vel
        if v > 2.0:
            v = 2.0
        elif v < -2.0:
            v = -2.0
        neck[write] = v
        v = nut_refl + new_vel
        if v > 2.0:
            v = 2.0
        elif v < -2.0:
            v = -2.0
        bridge[write] = v

        o = bridge_out - dc_x + dc_pole * dc_y
        dc_x = bridge_out
        dc_y = o
        out[i] = o

        write += 1
        if write >= size:
            write = 0
    return write, low, dc_x, dc_y


if _HAVE_NUMBA:
    _bow_kernel = njit(cache=True, fastmath=True)(_bow_kernel_source)
else:
    _bow_kernel = _bow_kernel_source


class WindUnit(Unit):
    """Blown instrument: reed or flute, played entirely by pressure.

    There is no trigger anywhere on this unit, because a wind instrument has
    none. 'pressure' is the whole interface -- a knob to lean on, an adsr~
    for tongued notes, an LFO for breath vibrato, or an effort stream so that
    a body's exertion is literally what blows the note. The reed speaks from
    about half pressure; the flute wants most of a full breath, whispers
    filtered air below that, and cracks octaves when pushed past 1 -- all of
    it emergent from the model rather than programmed.

    'embouchure' is the mouth: on the reed it is the bite (reed stiffness,
    where it speaks and how reedy it is), on the flute the jet length, which
    selects the register: fundamental through the middle of the range, bent
    expressively flat and sharp along the way, breaking to the octave above
    ~0.8 and the twelfth at the very bottom. 'brightness' is bore loss as
    everywhere else, and 'breath' is how much turbulence rides on the
    pressure, from pure tone to half air.

    Tuning is calibrated: the reed's bore is compensated for its reflection
    filter (within a few cents across the range), the flute needs none.
    """

    MODES = ('reed', 'flute')
    MIN_FREQUENCY = 30.0
    NOISE_SAMPLES = 1 << 16
    # The flute's playable range is engineered so the wolf -- the jet's
    # outer-zero crossing -- sits just past nominal full breath.
    FLUTE_BREATH_SCALE = 0.92
    FLUTE_OUTPUT_SCALE = 0.5

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.pressure_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.5)
        self.frequency_in = self.new_inlet(base=220.0,
                                           minimum=WindUnit.MIN_FREQUENCY)
        self.pitch_in = self.new_inlet()
        self.embouchure_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.brightness_in = self.new_inlet(base=0.7, minimum=0.0, maximum=1.0)
        self.noise_in = self.new_inlet(base=0.06, minimum=0.0, maximum=1.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        self.mode = 0

        size = int(self.sample_rate / WindUnit.MIN_FREQUENCY) + 8
        self.bore = np.zeros(size, dtype=np.float64)
        self.jet = np.zeros(size, dtype=np.float64)
        self._bwrite = 0
        self._jwrite = 0
        self._low = 0.0
        self._dcb_x = 0.0
        self._dcb_y = 0.0
        self._dc_x = 0.0
        self._dc_y = 0.0

        generator = np.random.default_rng(20260808)
        self._noise = generator.uniform(-1.0, 1.0, WindUnit.NOISE_SAMPLES)
        self._noise_at = 0
        self._quiet = True

        self.out = self.new_outlet()
        self._press = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._namt = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._freq = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._bore_delay = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._jet_delay = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._damp = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self.bore[:] = 0.0
        self.jet[:] = 0.0
        self._low = 0.0
        self._dcb_x = 0.0
        self._dcb_y = 0.0
        self._dc_x = 0.0
        self._dc_y = 0.0
        self._quiet = True

    def deactivate(self):
        self.reset()

    def render(self, frames):
        pressure = self.pressure_in.eval(frames)
        frequency = self.frequency_in.eval(frames)
        pitch = self.pitch_in.eval(frames)
        embouchure = self.embouchure_in.eval(frames)
        brightness = self.brightness_in.eval(frames)
        noise = self.noise_in.eval(frames)
        out_level = self.level_in.eval(frames)

        out = self.out
        if not _svf_ready.is_set():
            out.set_constant(0.0)
            return

        press = self._press[:frames]
        if pressure.constant:
            press[:] = pressure.value
            idle = abs(pressure.value) < 1.0e-4
        else:
            np.copyto(press, pressure.data[:frames])
            idle = False
        np.clip(press, 0.0, 2.0, out=press)

        # No breath and nothing still sounding in the bore: skip the block.
        if self._quiet and idle:
            out.set_constant(0.0)
            return

        freq = self._freq[:frames]
        self._build_hertz(freq, frequency, pitch, frames,
                          WindUnit.MIN_FREQUENCY)

        damp = self._damp[:frames]
        if brightness.constant:
            damp[:] = 1.0 - brightness.value
        else:
            np.subtract(1.0, brightness.data[:frames], out=damp,
                        casting='unsafe')
        np.clip(damp, 0.0, 0.95, out=damp)

        namt = self._namt[:frames]
        if noise.constant:
            namt[:] = noise.value
        else:
            np.copyto(namt, noise.data[:frames], casting='unsafe')
        np.clip(namt, 0.0, 1.0, out=namt)

        emb = embouchure.value if embouchure.constant else float(
            embouchure.data[0])
        emb = min(1.0, max(0.0, emb))

        bore_delay = self._bore_delay[:frames]
        jet_delay = self._jet_delay[:frames]
        slope = 0.0
        scale = 1.0
        if self.mode == 0:
            # Closed bore: half-period line, inverted reflection. The bite
            # sets the reed table's slope.
            slope = -(0.16 + 0.28 * emb)
            np.divide(self.sample_rate * 0.5, freq, out=bore_delay)
            darkness = float(damp[0])
            bore_delay -= darkness / (1.0 - darkness)
        else:
            np.divide(self.sample_rate, freq, out=bore_delay)
            # The jet line is a fraction of the bore; embouchure slides it,
            # and with it which register the jet locks to.
            np.multiply(bore_delay, 0.2 + 0.6 * emb, out=jet_delay)
            press *= WindUnit.FLUTE_BREATH_SCALE
            scale = WindUnit.FLUTE_OUTPUT_SCALE
        np.clip(bore_delay, 2.0, self.bore.shape[0] - 3.0, out=bore_delay)

        result = self._y[:frames]
        (self._bwrite, self._jwrite, self._noise_at, self._low,
         self._dcb_x, self._dcb_y, self._dc_x, self._dc_y) = _wind_kernel(
            press, namt, self._noise, self._noise_at,
            self.bore, self._bwrite, self.jet, self._jwrite,
            bore_delay, jet_delay, damp,
            slope, 0.6, 0.6, self.mode,
            self._low, self._dcb_x, self._dcb_y, self._dc_x, self._dc_y,
            result)

        if scale != 1.0:
            result *= scale
        self._apply_level(result, out_level, frames)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        scratch = self._scratch[:frames]
        np.abs(result, out=scratch)
        self._quiet = bool(scratch.max() < 1.0e-5)


class BowUnit(Unit):
    """Bowed string, played by velocity and force -- no trigger, no pluck.

    'velocity' is the bow's speed across the string and 'force' how hard it
    presses. Fundamental tone lives on a diagonal of that plane -- a faster
    bow needs more weight behind it -- and the unit maps the nominal 0..1
    ranges onto that diagonal, so the middle of both sliders bows cleanly at
    any pitch while the edges stay expressive: velocity past 1 with a light
    bow breaks into the octave whistle, slow and heavy crushes into
    subharmonic scratch. Patched from effort data, the mapping means moving
    faster is bowing faster and pressing is leaning in, and the instrument
    misbehaves in the same directions the real one does.

    The internal bow speed also scales with 1/sqrt(pitch), which is what
    keeps the same gesture playable on a low string and a high one -- the
    empirical version of a player lightening the bow as they go up.

    'position' is where the bow lands between bridge (small) and fingerboard:
    sul ponticello to sul tasto, a timbre control. 'brightness' is string
    loss, as everywhere.

    What comes out is the bridge wave -- the raw string, deliberately without
    a violin body. A body is just a resonator, and resonators are patchable:
    bow~ into modal~ (wood) or formant~ is a violin; into a bell table it is
    a bowed bell, which no luthier will build you.

    The range runs down to 5 Hz, well below any string that plays a note:
    down there the stick-slip cycle slows from pitch into event, and the bow
    becomes a creak, a groan, a door hinge -- texture rather than tone. The
    output DC blocker tracks the fundamental (a quarter of it, capped where
    it always was) so the bottom octaves keep their weight instead of being
    filtered away by their own protection.
    """

    MIN_FREQUENCY = 5.0

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.velocity_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.5)
        self.force_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.position_in = self.new_inlet(base=0.127, minimum=0.05,
                                          maximum=0.4)
        self.frequency_in = self.new_inlet(base=220.0,
                                           minimum=BowUnit.MIN_FREQUENCY)
        self.pitch_in = self.new_inlet()
        self.brightness_in = self.new_inlet(base=0.75, minimum=0.0,
                                            maximum=1.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        size = int(self.sample_rate / BowUnit.MIN_FREQUENCY) + 8
        self.neck = np.zeros(size, dtype=np.float64)
        self.bridge = np.zeros(size, dtype=np.float64)
        self._write = 0
        self._low = 0.0
        self._dc_x = 0.0
        self._dc_y = 0.0
        self._quiet = True

        self.out = self.new_outlet()
        self._vel = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._force = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._freq = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._neck_delay = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._bridge_delay = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._damp = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self.neck[:] = 0.0
        self.bridge[:] = 0.0
        self._low = 0.0
        self._dc_x = 0.0
        self._dc_y = 0.0
        self._quiet = True

    def deactivate(self):
        self.reset()

    def render(self, frames):
        velocity = self.velocity_in.eval(frames)
        force = self.force_in.eval(frames)
        position = self.position_in.eval(frames)
        frequency = self.frequency_in.eval(frames)
        pitch = self.pitch_in.eval(frames)
        brightness = self.brightness_in.eval(frames)
        out_level = self.level_in.eval(frames)

        out = self.out
        if not _svf_ready.is_set():
            out.set_constant(0.0)
            return

        vel = self._vel[:frames]
        if velocity.constant:
            vel[:] = velocity.value
            idle = abs(velocity.value) < 1.0e-4
        else:
            np.copyto(vel, velocity.data[:frames])
            idle = False
        np.clip(vel, 0.0, 2.0, out=vel)

        # Bow lifted and string rung down: nothing to do.
        if self._quiet and idle:
            out.set_constant(0.0)
            return

        push = self._force[:frames]
        if force.constant:
            push[:] = force.value
        else:
            np.copyto(push, force.data[:frames], casting='unsafe')
        np.clip(push, 0.0, 1.0, out=push)

        freq = self._freq[:frames]
        self._build_hertz(freq, frequency, pitch, frames,
                          BowUnit.MIN_FREQUENCY)

        # Velocity onto the Schelleng diagonal: what the string will accept
        # scales with force, and shrinks as the pitch rises. Past 1 the
        # mapping lets go on purpose -- the overflow bypasses the coupling,
        # more readily the lighter the bow, which is where the octave
        # whistle lives. A heavy bow driven past 1 just plays louder.
        f0 = float(freq[0])
        reach = min(1.5, (220.0 / f0) ** 0.5)
        scratch = self._scratch[:frames]
        over = self._y[:frames]
        np.subtract(vel, 1.0, out=over)
        np.clip(over, 0.0, None, out=over)
        np.clip(vel, 0.0, 1.0, out=vel)
        np.multiply(push, 0.09, out=scratch)
        scratch += 0.05
        vel *= scratch
        np.multiply(push, -0.125, out=scratch)
        scratch += 0.25
        over *= scratch
        vel += over
        vel *= reach

        damp = self._damp[:frames]
        if brightness.constant:
            damp[:] = 1.0 - brightness.value
        else:
            np.subtract(1.0, brightness.data[:frames], out=damp,
                        casting='unsafe')
        np.clip(damp, 0.0, 0.95, out=damp)

        beta = position.value if position.constant else float(
            position.data[0])
        beta = min(0.4, max(0.05, beta))

        neck_delay = self._neck_delay[:frames]
        bridge_delay = self._bridge_delay[:frames]
        np.divide(self.sample_rate, freq, out=neck_delay)
        np.multiply(neck_delay, beta, out=bridge_delay)
        neck_delay *= 1.0 - beta

        # The output DC blocker follows the note down: a fixed corner sat
        # near 35 Hz and would thin every fundamental below it.
        corner = min(40.0, max(1.0, f0 * 0.25))
        dc_pole = math.exp(-2.0 * math.pi * corner / self.sample_rate)

        result = self._y[:frames]
        self._write, self._low, self._dc_x, self._dc_y = _bow_kernel(
            vel, push, self.neck, self.bridge, self._write,
            neck_delay, bridge_delay, damp, dc_pole,
            self._low, self._dc_x, self._dc_y, result)

        result *= 1.5
        self._apply_level(result, out_level, frames)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        np.abs(result, out=scratch)
        self._quiet = bool(scratch.max() < 1.0e-5)


def _fast_sin_source(x):
    """Odd minimax-ish sine on a wrapped phase, for voice stacks.

    Branchless parabolic sine (the classic two-parabola form with one
    refinement): a few nanoseconds where libm takes ten and branches
    cost more than math. With sixteen oscillators per sample
    (bubbles~), that is most of the kernel. Harmonics sit near -44 dB
    -- inaudible under anything born of noise. Input must already be
    wrapped to [0, 2pi)."""
    if x > 3.141592653589793:
        x -= 6.283185307179586
    y = 1.2732395447351628 * x - 0.4052847345693511 * x * abs(x)
    return 0.225 * (y * abs(y) - y) + y


if _HAVE_NUMBA:
    _fast_sin = njit(cache=True, fastmath=True, inline='always')(
        _fast_sin_source)
else:
    _fast_sin = _fast_sin_source


def _rand01_source(state):
    """xorshift64*: one uniform draw in [0, 1) and the state that follows.

    A table of precomputed randoms loops -- at one entry per sample even a
    generous table repeats every few seconds, which at low density is an
    audibly identical pattern of rain. This is a real generator: period
    2^64, a handful of integer ops, nothing to loop.
    """
    state ^= state >> np.uint64(12)
    state ^= state << np.uint64(25)
    state ^= state >> np.uint64(27)
    scrambled = (state * np.uint64(2685821657736338717)) >> np.uint64(11)
    return state, np.float64(scrambled) * (1.0 / 9007199254740992.0)


if _HAVE_NUMBA:
    _rand01 = njit(cache=True, inline='always')(_rand01_source)
else:
    _rand01 = _rand01_source


def _shaker_kernel_source(shake, rate_per_sample, energy_decay, grain_decay,
                          vary, amp, attack_k, thetas, radius, b1s, b1zs,
                          g2s, energy, sounds, gds, envs, y1, y2, z1, z2,
                          rng, out_raw, out,
                          head_on, rate_boost, fine, hold, hold_scale,
                          hold_k, throw_kick, shake_prev, slide_base,
                          slide_amp,
                          slide_tail,
                          slide_max, slide_norm, slide_least, hurst_inv,
                          slide_ring, slide_head):
    """Cook's PhISEM, sample by sample -- with a polyphonic vessel.

    'energy' is how agitated the beans are, pumped by the gesture and
    settling on its own; each sample a collision happens or does not,
    with probability rising with agitation and bean count.

    The vessel is not one resonator but a small ensemble: a tambourine
    is a dozen jingles, each with its own pitch, and a collision strikes
    ONE of them -- its own grain envelope, its own ring -- while the
    others keep sounding. The members' tunings are fixed per instance
    (the jingle knob spreads them), which is what makes overlapping
    decays at distinct pitches: polyphony, where a single retuned
    resonator could only ever play the last note.

    Coefficients are derived from angle and radius together, per block,
    never stored across a radius change.
    """
    members = thetas.shape[0]
    pump = 1.0 - energy_decay
    b2 = -radius * radius
    # Energy-constant with respect to ring time: each grain's ring holds
    # its loudness as resonance stretches it, rather than thinning away.
    vgain = 0.3 * math.sqrt((1.0 - radius) / 0.15)
    # Tunings are fixed within a block, so the trig lives out here.
    # The second stage steepens the skirt so the strike's noise burst
    # arrives already in the bell's voice -- but at a fixed BROAD
    # resonance (about a millisecond), because a second narrow stage
    # would swell over its own ring time and a struck bell does not
    # swell. It is the clapper contact shaped by the bell's body.
    # Normalized to unity at its peak, it reshapes without relevelling.
    r2 = radius
    if r2 > 0.977:
        r2 = 0.977
    b2z = -r2 * r2
    for m in range(members):
        b1s[m] = 2.0 * radius * math.cos(thetas[m])
        b1zs[m] = 2.0 * r2 * math.cos(thetas[m])
        # 2.5x makeup: the steeper skirt sheds broadband energy, and
        # the family should sit at the level it always did.
        g2s[m] = 2.5 * (1.0 - r2) * math.sin(thetas[m])
    ring_n = slide_ring.shape[0]
    head = int(slide_head)
    for i in range(shake.shape[0]):
        # A shake and a swirl both agitate the beans; what differs is
        # HOW. A shake throws the whole handful at one end and they
        # arrive together, so the ticks come in bursts and die between
        # them. A swirl keeps them tumbling continuously -- over each
        # other as much as against the wall -- so the agitation never
        # falls away between strokes and the impacts are finer and more
        # numerous, being bean on bean rather than handful on shell.
        #
        # It is NOT beans pinned to the wall sliding round it. That was
        # the first guess here and it is wrong: they tumble.
        if hold > 0.5:
            # HELD: the gesture IS the agitation. A steady hand gives a
            # steady wash and letting go stops it at the settle rate.
            energy += (shake[i] * hold_scale - energy) * hold_k
        else:
            # THROWN: the beans answer the CHANGE, not the level, which
            # is what spin~ means by the same word. A rise throws them
            # and they carry on by themselves; holding the hand still --
            # at any height at all -- adds nothing more, because a
            # shaker held out at arm's length is not being shaken.
            #
            # Pumping from the LEVEL, as this did at first, gives a
            # steady state proportional to the gesture and a tail at the
            # settle rate -- which is what holding already does. Two
            # names for one behaviour, and no way to tell them apart.
            # Either way. Shaking is back AND forth, and the beans are
            # thrown just as hard on the return -- taking only the rise
            # threw them on half the strokes and let them settle
            # through the other half.
            rise = shake[i] - shake_prev
            if rise < 0.0:
                rise = -rise
            # The stroke goes straight in. Multiplied by 'pump' -- which
            # is one minus the settle decay, a ten-thousandth -- a whole
            # sweep of the hand landed twenty times under what holding
            # the same number gives, and how hard a stroke threw the
            # beans depended on how long they took to settle afterwards.
            # A stroke is a stroke; settle says how it DIES.
            energy = energy * energy_decay
            energy += rise * throw_kick
        shake_prev = shake[i]
        rng, draw = _rand01(rng)
        if draw < rate_per_sample * energy * rate_boost:
            rng, pick = _rand01(rng)
            member = int(pick * members)
            if member >= members:
                member = members - 1
            rng, strength = _rand01(rng)
            # Only what goes INTO the shell rings it. A glancing bean
            # keeps most of its speed along the wall and gives up little
            # of it to the wall, so the tick fades as the angle opens.
            sounds[member] += (amp * fine * head_on * energy
                               * (0.5 + 0.5 * strength))
            if sounds[member] > 100.0:
                sounds[member] = 100.0
            if vary > 0.0:
                rng, size = _rand01(rng)
                gds[member] = grain_decay ** (2.0 ** ((0.5 - size)
                                                     * 2.0 * vary))
            else:
                gds[member] = grain_decay
        # And what they do instead is SLIDE. A bean pinned to the wall
        # and dragged round it is a contact crossing a rough surface,
        # which is the same thing a coin's rim does -- so it is the same
        # grain: sizes drawn from a power law because a surface has no
        # characteristic asperity, and each one lasting as long as it is
        # big, because a wide feature takes longer to cross. Firing them
        # all as single samples would make clicks, not friction.
        # And what it keeps, it drags along the wall. The two are one
        # impact seen at an angle: head on it is a tick, tangential it
        # is a graze, and everything between is both. How much there is
        # of either is the AGITATION -- there is no separate swirling
        # gesture, because you cannot shake a maraca while you are
        # rolling it and you cannot roll it while you are shaking it.
        slide = 0.0
        slide_dens = slide_base * energy
        if slide_dens > 0.45:
            slide_dens = 0.45
        if slide_dens > 0.0:
            rng, su = _rand01(rng)
            if su < slide_dens:
                rng, su = _rand01(rng)
                if su < 1.0e-9:
                    su = 1.0e-9
                size = su ** (-slide_tail)
                if size > slide_max:
                    size = slide_max
                rng, su = _rand01(rng)
                peak = size * slide_amp * slide_norm
                if su <= 0.5:
                    peak = -peak
                dur = int(size ** hurst_inv)
                if dur < slide_least:
                    dur = slide_least
                if dur > ring_n - 1:
                    dur = ring_n - 1
                if dur == 1:
                    slide_ring[head] += peak
                else:
                    scale = 2.0 * peak / dur
                    at = head
                    for q in range(dur):
                        slide_ring[at] += scale * 0.5 * (
                            1.0 - math.cos(6.283185307179586
                                           * (q + 0.5) / dur))
                        at += 1
                        if at >= ring_n:
                            at = 0
            # Shared across the ensemble, not handed to each of them:
            # the heap scuffs one shell, and there are eight members
            # standing for it. Added per member it came out eight times
            # over and swamped everything else in the unit.
            slide = slide_ring[head] / members
            slide_ring[head] = 0.0
            head += 1
            if head >= ring_n:
                head = 0
        rng, nz = _rand01(rng)
        noise = 2.0 * nz - 1.0
        raw = 0.0
        total = 0.0
        for m in range(members):
            sounds[m] *= gds[m]
            # A soft contact rises slowly as well as decaying slowly:
            # the envelope follows its target with an attack lag scaled
            # to the grain length, so hardness 0 has no leading edge.
            envs[m] += (sounds[m] - envs[m]) * attack_k
            grain = envs[m] * noise + slide
            raw += grain
            y = vgain * grain + b1s[m] * y1[m] + b2 * y2[m]
            y2[m] = y1[m]
            y1[m] = y
            z = g2s[m] * y + b1zs[m] * z1[m] + b2z * z2[m]
            z2[m] = z1[m]
            z1[m] = z
            total += z
        out_raw[i] = raw
        out[i] = total
    return energy, rng, head, shake_prev


if _HAVE_NUMBA:
    _shaker_kernel = njit(cache=True, fastmath=True)(_shaker_kernel_source)
else:
    _shaker_kernel = _shaker_kernel_source


def _rattle_kernel_source(ax, ay, az, wx, wy, wz, pos, vel,
                          count, half_x, half_y, half_z, grain,
                          sizes, spring, spin_phase, scatter, tumble,
                          held, support, touching, release,
                          restitution, grip, texture, texture_drag,
                          asperity, asperity_rough, skin, tilt,
                          hold_rough,
                          grav, box,
                          speed_cap,
                          rest_speed, rest_steps, slide_gain,
                          decim, dt, knock_gain, scrape_gain,
                          contact, window, ring_knock, ring_scrape,
                          ring_head,
                          spin_prev_x, spin_prev_y, spin_prev_z,
                          ang_prev_x, ang_prev_y, ang_prev_z,
                          down_x, down_y, down_z, rng,
                          out, knock_out, scrape_out):
    """Loose things in a shaken container, actually simulated.

    Not a collision RATE driven by an agitation, which is what a PhISEM
    shaker is -- particles, with positions, in a box or an ellipsoid,
    hit by the walls when the walls come to them. Everything about how a
    gesture sounds then follows from the gesture instead of from a curve
    somebody fitted: shaking along a line and swirling in a circle are
    the same simulation given a line and a circle, and the difference
    between them -- more glancing contact, an envelope that stops
    pulsing -- comes out on its own. Measured on the prototype: two and
    a half times as much tangential as normal contact driven along a
    line, four times driven round a circle, and the envelope swing
    falling from 0.33 to 0.05.

    In the container's frame the particles are pushed by everything the
    container does. Translating, that is minus its acceleration.
    ROTATING, it is three more: the centrifugal push outward, the
    Coriolis deflection of anything already moving, and the Euler shove
    when the spin itself changes. All three are what a swirl actually
    is, and without them a rotated container does nothing at all.

    Gravity points down and stays there. Because all of this is worked
    out in the CONTAINER's coordinates, though, that unmoving direction
    sweeps around as the container turns under it -- so it is carried
    along at minus the spin. Nothing rotates but the frame, and that
    sweep is what a turned container does to what is in it: the contents
    are drawn down, the wall comes round to meet them, and they drag
    along it.

    Inter-particle collisions are left out on purpose. They are most of
    the cost and least of the sound: what is heard is the wall.
    """
    ring_n = ring_knock.shape[0]
    head = int(ring_head)
    steps = 0
    two_pi = 6.283185307179586
    width = int(contact)
    if width < 2:
        width = 2
    if width > ring_n - 1:
        width = ring_n - 1
    for i in range(ax.shape[0]):
        if steps <= 0:
            steps = decim
            # --- one step of the world ---
            # 'turn' is an ANGLE -- where the container is POINTING
            # about each axis, in radians -- and not a rate. Which way
            # is down follows straight from it, with nothing integrated
            # and so nothing to drift: a container held at a tilt holds
            # gravity at exactly that tilt, for ever. Turning about the
            # vertical does not move it at all, which is right.
            #
            # Gravity itself never moves. It is these coordinates that
            # turn under it, and that sweep is the whole of what
            # turning a container does to what is in it: the contents
            # are drawn down, the wall comes round to meet them, and
            # they drag along it.
            tax = wx[i]
            tay = wy[i]
            sa = math.sin(tax)
            ca = math.cos(tax)
            sb = math.sin(tay)
            cb = math.cos(tay)
            down_x = sb
            down_y = -cb * sa
            down_z = -cb * ca
            # The centrifugal, Coriolis and Euler terms want a RATE, so
            # it is differenced out of the angle, and the rate is
            # differenced again for Euler.
            sx = (wx[i] - ang_prev_x) / dt
            sy = (wy[i] - ang_prev_y) / dt
            sz = (wz[i] - ang_prev_z) / dt
            ang_prev_x = wx[i]
            ang_prev_y = wy[i]
            ang_prev_z = wz[i]
            ex = (sx - spin_prev_x) / dt
            ey = (sy - spin_prev_y) / dt
            ez = (sz - spin_prev_z) / dt
            spin_prev_x = sx
            spin_prev_y = sy
            spin_prev_z = sz
            hit_k = 0.0
            hit_s = 0.0
            rub_pow = 0.0
            rub_wsum = 0.0
            tick_pow = 0.0
            gx = -ax[i] + grav * down_x
            gy = -ay[i] + grav * down_y
            gz = -az[i] + grav * down_z
            for p in range(count):
                # Each one its own size. Identical particles in one
                # shared field keep step -- they differ only in where
                # they started, so they arrive together and a hundred of
                # them sound like eight. Real grains are not identical.
                #
                # This used to wobble with the tumble as well, standing
                # the surface off the centre by a changing amount. That
                # made the wall bob in and out under anything sliding
                # along it, and every bob read as an arrival: a slow
                # turn on smooth glass came out 30 to 1 knocks over
                # slide, which is backwards. A thing's irregularity
                # belongs in what HOLDS it, below, not in where the
                # wall is.
                # What actually resists here: the coefficient plus
                # what it costs to ride over the roughness. Without the
                # second term texture could do nothing at all at low
                # friction -- a thing only ever comes to rest when its
                # speed falls under grip*press*dt, so with no grip it
                # never rested, never caught, and the support was never
                # drawn. A rough surface stops things whether or not it
                # is sticky.
                mu = grip + texture * texture_drag
                grain_p = grain * sizes[p]
                rx = pos[3 * p]
                ry = pos[3 * p + 1]
                rz = pos[3 * p + 2]
                vx = vel[3 * p]
                vy = vel[3 * p + 1]
                vz = vel[3 * p + 2]
                # spin x r, then spin x that: the centrifugal push
                cx = sy * rz - sz * ry
                cy = sz * rx - sx * rz
                cz = sx * ry - sy * rx
                fx = gx - (sy * cz - sz * cy)
                fy = gy - (sz * cx - sx * cz)
                fz = gz - (sx * cy - sy * cx)
                # Coriolis, on whatever is already moving
                fx -= 2.0 * (sy * vz - sz * vy)
                fy -= 2.0 * (sz * vx - sx * vz)
                fz -= 2.0 * (sx * vy - sy * vx)
                # Euler, when the spin itself is changing
                fx -= ey * rz - ez * ry
                fy -= ez * rx - ex * rz
                fz -= ex * ry - ey * rx
                vx += fx * dt
                vy += fy * dt
                vz += fz * dt
                # Nothing in a shaken container outruns the shaking by
                # much. Without this a perfectly slippery wall -- which
                # is what no grip at all asks for -- never takes any
                # sideways speed away, so it accumulates with every
                # push: a box measured seventeen times its own level
                # standing still, and any parameter moved near that
                # corner threw a spike. It is also the guard against a
                # wall arriving on something at speed, which is what
                # resizing the container does.
                # What counts as 'at rest' is not a fixed speed: it is
                # whatever the push will give back in one step. A ball
                # dropped on a floor bounces an infinite number of times
                # in finite time, and a fixed threshold either cuts that
                # off too early -- silencing a swirl, whose particles
                # PRESS rather than arrive -- or too late, and then a
                # single object micro-bounces three hundred times a
                # second instead of sounding like one object.
                push = math.sqrt(fx * fx + fy * fy + fz * fz)
                floor_v = push * dt * rest_steps
                if floor_v < rest_speed:
                    floor_v = rest_speed
                sp2 = vx * vx + vy * vy + vz * vz
                if sp2 > speed_cap * speed_cap:
                    sc_ = speed_cap / math.sqrt(sp2)
                    vx *= sc_
                    vy *= sc_
                    vz *= sc_
                rx += vx * dt
                ry += vy * dt
                rz += vz * dt
                touch = 0.0
                nx = 0.0
                ny = 0.0
                nz = 0.0
                if box > 0.5:
                    # Which wall, and which way it faces. The contact
                    # itself is decided below, the same way for every
                    # shape -- a wall is a wall.
                    #
                    # Touching means AT the wall, not past it. Asking
                    # for strictly past it, a thing that had come to
                    # rest sat exactly on the line and so counted as
                    # not touching at all: held and the contact itself
                    # were both cleared, every other step, for ever.
                    # The support got redrawn twice a control period
                    # instead of once per catch, so roughness came out
                    # as fast noise rather than catch-and-release, and
                    # contacts kept re-registering as new. Hence a skin.
                    lim = half_x - grain_p
                    if rx > lim - skin * lim:
                        if rx > lim:
                            rx = lim
                        touch = 1.0
                        nx = 1.0
                    elif rx < skin * lim - lim:
                        if rx < -lim:
                            rx = -lim
                        touch = 1.0
                        nx = -1.0
                    lim = half_y - grain_p
                    if ry > lim - skin * lim:
                        if ry > lim:
                            ry = lim
                        touch = 1.0
                        ny = 1.0
                    elif ry < skin * lim - lim:
                        if ry < -lim:
                            ry = -lim
                        touch = 1.0
                        ny = -1.0
                    lim = half_z - grain_p
                    if rz > lim - skin * lim:
                        if rz > lim:
                            rz = lim
                        touch = 1.0
                        nz = 1.0
                    elif rz < skin * lim - lim:
                        if rz < -lim:
                            rz = -lim
                        touch = 1.0
                        nz = -1.0
                    if touch > 0.5:
                        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
                        nx /= nl
                        ny /= nl
                        nz /= nl
                else:
                    qx = (half_x - grain_p)
                    qy = (half_y - grain_p)
                    qz = (half_z - grain_p)
                    ux = rx / qx
                    uy = ry / qy
                    uz = rz / qz
                    d2 = ux * ux + uy * uy + uz * uz
                    if d2 > (1.0 - skin) * (1.0 - skin):
                        if d2 > 1.0:
                            d = math.sqrt(d2)
                            rx /= d
                            ry /= d
                            rz /= d
                        # The surface's own normal, which on an egg is
                        # not the way out from the middle.
                        nx = rx / (qx * qx)
                        ny = ry / (qy * qy)
                        nz = rz / (qz * qz)
                        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
                        if nl > 1.0e-12:
                            nx /= nl
                            ny /= nl
                            nz /= nl
                            touch = 1.0

                if touch > 0.5:
                        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
                        nx /= nl
                        ny /= nl
                        nz /= nl
                else:
                    qx = (half_x - grain_p)
                    qy = (half_y - grain_p)
                    qz = (half_z - grain_p)
                    ux = rx / qx
                    uy = ry / qy
                    uz = rz / qz
                    d2 = ux * ux + uy * uy + uz * uz
                    if d2 > 1.0:
                        d = math.sqrt(d2)
                        rx /= d
                        ry /= d
                        rz /= d
                        # The surface's own normal, which on an egg is
                        # not the way out from the middle.
                        nx = rx / (qx * qx)
                        ny = ry / (qy * qy)
                        nz = rz / (qz * qz)
                        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
                        if nl > 1.0e-12:
                            nx /= nl
                            ny /= nl
                            nz /= nl
                            touch = 1.0

                if touch > 0.5:
                    # ---- one contact, decided once ----
                    vn = vx * nx + vy * ny + vz * nz
                    press = fx * nx + fy * ny + fz * nz
                    # ARRIVING means a NEW contact, not a fast one.
                    # Judged on the normal speed instead, anything
                    # sliding inside a curved shell reports an arrival
                    # every step: it travels a straight chord between
                    # steps, lands a little outside the wall, and picks
                    # up an outward speed of about v*v*dt/R from the
                    # curvature alone. Against a fixed floor that
                    # crosses at sqrt(g * rest_steps * R), which here is
                    # a metre a second -- so everything sliding faster
                    # than walking pace knocked continuously, and with
                    # no friction to slow anything down it was ALL
                    # knock. A contact that was already there is not an
                    # arrival however fast it is going.
                    if touching[p] < 0.5 and vn > floor_v:
                        # ARRIVING. A blow, and it is no longer held.
                        held[p] = 0.0
                        hit_k = vn
                        rest_p = restitution * spring[p]
                        if rest_p > 0.95:
                            rest_p = 0.95
                        kk = (1.0 + rest_p) * vn
                        # Struck off-centre, an irregular thing comes
                        # off at a tilt -- but it is the IMPULSE that
                        # tilts, not the velocity. Tilting the whole
                        # velocity, a thing sliding fast ALONG the wall
                        # had that speed turned into speed away from
                        # it, so raising variety launched the handful
                        # into free flight and every landing was a
                        # knock. Tilting the impulse cannot do that:
                        # the impulse is zero when nothing arrived. And
                        # tilting it keeps its size, so this neither
                        # adds energy nor takes it -- scaling it
                        # instead was a restitution above one half the
                        # time, and scaling it only downward killed the
                        # bouncing outright at a whisker of variety.
                        bx = nx
                        by = ny
                        bz = nz
                        if scatter > 0.0:
                            rng, r1 = _rand01(rng)
                            rng, r2 = _rand01(rng)
                            rng, r3 = _rand01(rng)
                            cx = nx + tilt * scatter * (2.0 * r1 - 1.0)
                            cy = ny + tilt * scatter * (2.0 * r2 - 1.0)
                            cz = nz + tilt * scatter * (2.0 * r3 - 1.0)
                            cd = cx * nx + cy * ny + cz * nz
                            cl = math.sqrt(cx * cx + cy * cy + cz * cz)
                            if cd > 0.0 and cl > 1.0e-9:
                                bx = cx / cl
                                by = cy / cl
                                bz = cz / cl
                        vx -= kk * bx
                        vy -= kk * by
                        vz -= kk * bz
                        # What the wall takes sideways, bounded by
                        # Coulomb: what resists, times how hard it hit.
                        vnn = vx * nx + vy * ny + vz * nz
                        tx = vx - vnn * nx
                        ty = vy - vnn * ny
                        tz = vz - vnn * nz
                        ts = math.sqrt(tx * tx + ty * ty + tz * tz)
                        if ts > 1.0e-12:
                            take = mu * vn * (1.0 + rest_p)
                            if take > ts:
                                take = ts
                            # Part of the BLOW, not of the rubbing.
                            # A glancing blow does drag as it lands,
                            # but that drag is impulsive and lasts the
                            # CONTACT -- it is the contact's colour,
                            # and it moves with hardness. Sent to the
                            # scrape outlet it put a blow-shaped,
                            # hardness-following pulse into the one
                            # signal that is meant to be nothing but
                            # rubbing, which is audible and plain on a
                            # scope. The rub itself does not move with
                            # hardness at all, and should not. So the
                            # two impulses of one collision combine
                            # into one blow, in quadrature, the way
                            # perpendicular things do.
                            hit_k = math.sqrt(hit_k * hit_k
                                              + take * take)
                            f_ = take / ts
                            vx -= f_ * tx
                            vy -= f_ * ty
                            vz -= f_ * tz
                    elif press > 0.0:
                        touching[p] = 1.0
                        # RESTING on it, and the only question that
                        # matters is whether the slope it is sitting on
                        # still holds it.
                        #
                        # Coulomb says held while the sideways pull is
                        # under the grip times the press -- which is the
                        # angle of repose and nothing else. A smooth
                        # shell has one grip everywhere, so once a thing
                        # starts sliding it goes on sliding: a slow turn
                        # on smooth glass is a slide, and a slide is a
                        # continuous sound.
                        #
                        # A ROUGH shell does not. Every place it comes
                        # to rest holds differently, so it catches, is
                        # carried until that place no longer holds it,
                        # lets go, drops to the next -- and each letting
                        # go is a tap. That is the whole difference
                        # between a hiss and a rattle, and it is one
                        # number: how much the support varies from
                        # place to place.
                        vx -= vn * nx
                        vy -= vn * ny
                        vz -= vn * nz
                        ftn = fx - press * nx
                        fty = fy - press * ny
                        ftz = fz - press * nz
                        pull = math.sqrt(ftn * ftn + fty * fty + ftz * ftz)
                        ts = math.sqrt(vx * vx + vy * vy + vz * vz)
                        if held[p] > 0.5:
                            if pull > support[p] * press:
                                # Let go. On a rough shell that is a
                                # drop off whatever was holding it.
                                held[p] = 0.0
                                # It does not vanish, it DROPS -- off
                                # whatever was holding it, through a
                                # height the roughness sets, arriving
                                # at the speed that fall gives it. So
                                # a rougher shell ticks louder because
                                # its grains sit up higher, and a
                                # harder press ticks louder because it
                                # comes down faster. No gain needed.
                                # Gathered, not laid down separately.
                                # Letting go is a small dense event and
                                # there are hundreds a second; at a
                                # soft contact, writing each one its
                                # own long pulse was the whole of what
                                # was left of the load. An ARRIVAL is
                                # different -- it is one transient and
                                # it keeps its own place.
                                tick = release * math.sqrt(
                                    2.0 * press * texture * grain_p)
                                tick_pow += tick * tick
                            else:
                                vx = 0.0
                                vy = 0.0
                                vz = 0.0
                        else:
                            drag = mu * press * dt
                            if ts <= drag:
                                # Come to rest. What holds it here is
                                # its own patch of the wall.
                                vx = 0.0
                                vy = 0.0
                                vz = 0.0
                                held[p] = 1.0
                                # What this patch of wall will take
                                # before it lets go. Two things vary
                                # it: how rough the SHELL is, which
                                # differs from place to place, and how
                                # irregular the THING is, which is
                                # whichever of its corners it came to
                                # rest on.
                                rng, rs_ = _rand01(rng)
                                rng, rt_ = _rand01(rng)
                                # Roughness WIDENS this; it does not
                                # raise it. Added on top, every place
                                # held harder as the shell got rougher,
                                # so it let go LESS often -- releases
                                # fell 251 a second to 106 as texture
                                # went up, which is backwards. Spread
                                # symmetrically, a rougher shell has
                                # weaker places as well as stronger
                                # ones, and it is the weak ones that
                                # tick.
                                mu_s = (grip + texture * texture_drag
                                        * hold_rough)
                                support[p] = mu_s * (1.0
                                                     + tumble
                                                     * (2.0 * rt_ - 1.0)
                                                     + texture
                                                     * (2.0 * rs_ - 1.0))
                            else:
                                take = drag
                                # A FORCE, not the speed it took off
                                # this step. Emitted per control step,
                                # a per-step quantity carried the step
                                # SIZE into the sound: the rub came out
                                # 0.0070 / 0.0098 / 0.0140 at
                                # decimation 4 / 8 / 16, a root two per
                                # doubling, so how finely it was
                                # integrated set how loud it rubbed.
                                hit_s = (mu * press * slide_gain
                                         * (1.0 + tumble
                                            * math.sin(spin_phase[p])))
                                f_ = take / ts
                                vx -= f_ * vx
                                vy -= f_ * vy
                                vz -= f_ * vz
                    else:
                        # Pulled clean off the wall -- whatever was
                        # holding it is not holding it any more.
                        held[p] = 0.0
                        touching[p] = 0.0
                else:
                    held[p] = 0.0
                    touching[p] = 0.0
                # It turns as fast as it is travelling over the
                # surface: a thing of radius r going at v rolls at v/r.
                if grain_p > 1.0e-9:
                    spin_phase[p] += (math.sqrt(vx * vx + vy * vy
                                                + vz * vz)
                                      / grain_p) * dt
                    if spin_phase[p] > 6.283185307179586:
                        spin_phase[p] -= 6.283185307179586
                # Each blow lands where it fell in the block, and is
                # a contact of some width rather than a single sample.
                if hit_k > 0.0 or hit_s > 0.0:
                    rng, u01 = _rand01(rng)
                    at0 = head + int(u01 * decim)
                    if at0 >= ring_n:
                        at0 -= ring_n
                    # SIGNED. A wall only ever pushes, so a contact
                    # force really is one-sided -- but as an exciter
                    # that is just an offset. Added one way only, a
                    # slide put out one bump every control step for
                    # ever and they summed to a CONSTANT: measured, the
                    # slide output ran mean +0.0083 against an rms of
                    # 0.0084, never once went below zero, and had every
                    # spectral component under 10 Hz. It was not a
                    # sound at all, it was a force level, and the level
                    # stepping as things caught and let go is what read
                    # as the output jumping to random offsets. What
                    # radiates from a sliding contact is the part that
                    # FLUCTUATES; the steady drag does not. So each
                    # contact goes in with a sign, which costs nothing
                    # and makes both outlets zero-mean, the way the
                    # rest of the rack is.
                    if hit_k > 0.0:
                        rng, sgn = _rand01(rng)
                        # A blow carries an IMPULSE, and how hard the
                        # contact is decides how that impulse is spread
                        # in time -- not how much of it there is. A
                        # Hann hump of width W and peak A carries A*W/2,
                        # so the peak has to go as 1/W to keep it. Going
                        # as one over the root instead, the impulse GREW
                        # as the root of the width: a soft contact
                        # handed the resonator five times the momentum
                        # of a hard one, and since a mode below the
                        # contact bandwidth answers the impulse, that
                        # made hardness a fourteen decibel loudness
                        # control on everything low. It should change
                        # the colour and nothing else.
                        kp = 2.0 * hit_k * knock_gain / width
                        if sgn < 0.5:
                            kp = -kp
                        at = at0
                        for q in range(width):
                            shape = 0.5 * (1.0 - math.cos(two_pi
                                                          * (q + 0.5)
                                                          / width))
                            ring_knock[at] += kp * shape
                            at += 1
                            if at >= ring_n:
                                at = 0
                    if hit_s > 0.0:
                        # How wide a rub is, is not how wide a blow is.
                        # A blow lasts as long as the contact takes to
                        # spring back, which is what hardness sets. A
                        # rub is one bump of the surface going by, so
                        # it lasts the spacing over the speed -- and
                        # that means a faster slide comes out brighter
                        # on its own, which is what a faster slide
                        # does.
                        sw = ring_n - 1
                        rub = math.sqrt(vx * vx + vy * vy + vz * vz)
                        if rub > 1.0e-9:
                            # How far apart the bumps of the wall are
                            # grows with how rough it is: a polished
                            # shell has fine ones and hisses, a coarse
                            # one has them further apart and rasps
                            # lower. That is the vessel's own surface
                            # speaking, and until now it was a fixed
                            # constant -- so 'texture' changed how much
                            # the wall CAUGHT but nothing about what it
                            # sounded like.
                            sw = int(grain_p * asperity
                                     * (1.0 + texture * asperity_rough)
                                     / rub * decim / dt)
                        # A rub cannot be sharper than a CONTACT. It is
                        # a run of tiny impacts and every one of them
                        # is still a contact, so the same stiffness
                        # that stops a blow being sharper than this
                        # stops a rub too. Floored at two samples
                        # instead, a rub at speed went white -- finer
                        # than the hardest blow the model allows -- and
                        # came out at 5900 Hz against a blow's 432 Hz
                        # from what is supposed to be one material.
                        if sw < width:
                            sw = width
                        if sw > ring_n - 1:
                            sw = ring_n - 1
                        # GATHERED, not laid down here. Every rubbing
                        # thing writing its own shaped bump into the
                        # ring every control step cost the count times
                        # the width times the control rate -- eighteen
                        # million sample-writes a second on its own,
                        # and it went up as the contact softened, so
                        # dropping hardness doubled the load. They are
                        # independent noises, and independent noises
                        # add in POWER: one bump carrying the summed
                        # power is the same sound for a hundredth of
                        # the work. The width is carried along
                        # power-weighted so the result keeps the
                        # brightness of whatever is actually rubbing
                        # hardest.
                        hs2 = hit_s * hit_s
                        rub_pow += hs2
                        rub_wsum += hs2 * sw
                # Cleared PER PARTICLE. Cleared only once a control
                # step, as they were, the first thing to register a
                # blow left it standing and every thing after it in
                # that step laid the same blow down again -- a hundred
                # and twenty-eight times over in a full container. The
                # rub was gathered correctly and the blows were not,
                # which is exactly why they stood out of all
                # proportion to it.
                hit_k = 0.0
                hit_s = 0.0
                pos[3 * p] = rx
                pos[3 * p + 1] = ry
                pos[3 * p + 2] = rz
                vel[3 * p] = vx
                vel[3 * p + 1] = vy
                vel[3 * p + 2] = vz
            if tick_pow > 0.0:
                rng, u01 = _rand01(rng)
                at = head + int(u01 * decim)
                if at >= ring_n:
                    at -= ring_n
                rng, sgn = _rand01(rng)
                kp = (2.0 * math.sqrt(tick_pow) * knock_gain
                      / width)
                if sgn < 0.5:
                    kp = -kp
                for q in range(width):
                    shape = 0.5 * (1.0 - math.cos(two_pi * (q + 0.5)
                                                  / width))
                    ring_knock[at] += kp * shape
                    at += 1
                    if at >= ring_n:
                        at = 0
            if rub_pow > 0.0:
                # One bump for the whole handful. Its size is set so
                # the sum comes out exactly where all the separate ones
                # did: each of those carried power hit_s squared times
                # dt, and spreading that over a width takes a root.
                sw = int(rub_wsum / rub_pow)
                if sw < 2:
                    sw = 2
                if sw > ring_n - 1:
                    sw = ring_n - 1
                rng, u01 = _rand01(rng)
                at = head + int(u01 * decim)
                if at >= ring_n:
                    at -= ring_n
                rng, sgn = _rand01(rng)
                spv = scrape_gain * math.sqrt(dt * rub_pow / sw)
                if sgn < 0.5:
                    spv = -spv
                for q in range(sw):
                    shape = 0.5 * (1.0 - math.cos(two_pi * (q + 0.5)
                                                  / sw))
                    ring_scrape[at] += spv * shape
                    at += 1
                    if at >= ring_n:
                        at = 0
        steps -= 1
        k = ring_knock[head]
        sc = ring_scrape[head]
        ring_knock[head] = 0.0
        ring_scrape[head] = 0.0
        head += 1
        if head >= ring_n:
            head = 0
        knock_out[i] = k
        scrape_out[i] = sc
        out[i] = k + sc
    return (head, spin_prev_x, spin_prev_y, spin_prev_z,
            ang_prev_x, ang_prev_y, ang_prev_z,
            down_x, down_y, down_z, rng)


if _HAVE_NUMBA:
    _rattle_kernel = njit(cache=True, fastmath=True)(_rattle_kernel_source)
else:
    _rattle_kernel = _rattle_kernel_source


class RattleUnit(Unit):
    """Loose things in a container, shaken and turned.

    shaker~ is a collision RATE driven by an agitation -- Cook's PhISEM,
    and very cheap. This has particles instead: positions, velocities,
    and walls that hit them when the walls come to them. Everything
    about how a gesture sounds then follows from the gesture, rather
    than from a curve fitted to it. Shaking along a line and swirling in
    a circle are the same simulation given a line and a circle, and what
    separates them -- more glancing contact, an envelope that stops
    pulsing -- comes out on its own.

    The gesture is where the container GOES: 'shake x/y/z' is its
    acceleration and 'turn x/y/z' is how far it is TIPPED, in degrees
    -- an angle, not a rate. Three-axis
    movement drives it directly, which is what a body gives, and needs
    no translating into 'how agitated'.

    Turning it matters, and matters three ways. The centrifugal push
    throws everything outward; the Coriolis force deflects whatever is
    already moving, which is what makes a swirled container sound
    unlike a shaken one at the same speed; and the Euler force shoves
    when the turning itself changes. Without them a rotated container
    does nothing at all.

    'shape' is sphere, box or egg, and it is only a boundary test, so it
    costs nothing and changes a great deal: flat walls take a bean head
    on where a curved one lets it glance. An egg is the ellipsoid with
    one axis stretched -- and the normal is the surface's, not the way
    out from the middle, so the pointed end presents the angle it
    really presents.

    'knock' and 'scrape' come out separately as well as mixed -- the
    normal blow and what it drags along the wall. Patch either into a
    resonator.

    Inter-particle collisions are left out on purpose: most of the cost,
    least of the sound. What is heard is the wall.
    """

    MAX_PARTICLES = 256
    RING = 512
    # Impulse into signal, in the units the simulation actually works
    # in -- metres a second of arriving bean. Set so a brisk shake of a
    # few dozen of them sits where the rest of the rack sits: bow~ peaks
    # at 0.42, noise~ at 0.68, bounce~ at 0.39.
    GEE = 9.80665
    DEG = 0.017453292519943295
    # How the level climbs with how many things are in there, measured.
    DENSITY_LAW = 0.5
    DENSITY_REF = 1.097
    KNOCK_GAIN = 30.6
    SCRAPE_GAIN = 0.168
    SHAPES = ('sphere', 'box', 'egg')
    # How much longer an egg is than it is wide.
    EGG = 1.7
    CONTROL_DECIM = 8
    # The fastest anything in there is allowed to be going, in metres a
    # second. A hand shakes at a few; this is well clear of any real
    # gesture and only catches the corners where the arithmetic would
    # otherwise run away.
    SPEED_CAP = 12.0
    # Below this, arriving at a wall is resting on it. A settled
    # particle is pushed back into the floor by gravity every step, so
    # without a threshold each one rings the container at the control
    # rate for as long as it lies there -- most of the cost, and a noise
    # floor made of things lying still.
    #
    # It has to be SMALL. Set at a sixtieth of a metre a second it also
    # swallowed the swirl, because centrifugal force pins things to the
    # wall and they press rather than arrive: a circular drive produced
    # no collisions whatever. This is low enough to catch only what is
    # genuinely at rest.
    REST_SPEED = 0.001
    # How many steps' worth of the pressing force still counts as rest.
    # A bounce that will not outlast a few steps is a thing settling,
    # not a thing arriving.
    REST_STEPS = 3.0
    # Sliding contact is quieter than arriving, per unit of momentum
    # taken -- a graze is not a blow. Found by ear against the knocks.
    SLIDE_GAIN = 30.0
    # How far from round a grain is, at full 'variety' -- the fraction
    # by which its surface stands nearer or further as it turns. A
    # smooth sphere does not chatter; nothing else is smooth.
    TUMBLE = 0.35
    # How wide the spread of sizes is with variety all the way up, and
    # how wide it stays with variety at nothing. Nothing real comes in
    # a hundred identical copies, and perfectly identical things in one
    # shared field keep STEP -- they differ only in where they started,
    # so they arrive together and a hundred sound like eight. Leaving
    # no floor made variety-at-zero a degenerate case rather than a
    # plain one.
    SIZE_SPREAD = 0.18
    SIZE_FLOOR = 0.06
    # How much the springiness varies from one to the next.
    SPRING_SPREAD = 0.4
    # How far the rebound is allowed to tilt. Small on purpose: a big
    # tilt DISPERSES rather than decorrelates -- things get flung right
    # across the middle, fly a long way between contacts, and the
    # sound goes sparse and sporadic, which is the opposite of what
    # variety is for. Measured, a full tilt took the event rate DOWN
    # 110 -> 70 a second and made it lumpier; a quarter of one takes
    # it UP 110 -> 132 and makes it smoother, with the level unmoved.
    SCATTER_TILT = 0.2
    # How much of the roughness's RESISTANCE also shows up in what
    # holds a thing up. Roughness resists in full -- that is what makes
    # it rasp with no friction at all -- but carrying that straight
    # into the support meant a rougher shell held everything harder and
    # so let go LESS often: releases fell 287 a second to 150 as
    # texture went up, and the knocks quietened with them, which is
    # backwards. At a quarter the rate stays flat (313 / 316 / 322)
    # while each release grows (0.066 -> 0.098 -> 0.112), which is what
    # roughening something should do. Not zero: at zero nothing holds
    # at all on a frictionless shell, and roughness must still catch.
    HOLD_ROUGH = 0.25
    # How thick the skin is in which a thing counts as touching the
    # wall, as a fraction of the shell.
    CONTACT_SKIN = 0.002
    # How far apart the bumps of the surface are, against the size of
    # the things riding over them. Sets how bright a rub is at a given
    # speed.
    # How long a contact lasts, softest to hardest, in seconds.
    CONTACT_SOFT = 0.008
    CONTACT_HARD = 0.0000454
    # How far apart the bumps of the surface are, against the size of
    # a grain. At a fiftieth of a grain radius this was 120 microns on
    # a default grain -- coarse sandpaper -- and a crossing took long
    # enough that it, and not the contact, set how bright a rub could
    # be. Real surfaces are microns, so the CONTACT is what limits it,
    # which is what makes a small hard shaker hiss.
    ASPERITY = 0.002
    # How much coarser the bumps get as the shell roughens.
    ASPERITY_ROUGH = 4.0
    # How much riding over the roughness resists, against the friction
    # coefficient itself.
    TEXTURE_DRAG = 0.5
    # Letting go is a drop, not a blow -- this only says how much of
    # the roughness height it actually falls through.
    RELEASE = 0.6

    _seeded = 0

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.shake_x_in = self.new_inlet()
        self.shake_y_in = self.new_inlet()
        self.shake_z_in = self.new_inlet()
        self.turn_x_in = self.new_inlet()
        self.turn_y_in = self.new_inlet()
        self.turn_z_in = self.new_inlet()
        self.size_in = self.new_inlet(base=0.04, minimum=0.005,
                                      maximum=0.5)
        self.grain_in = self.new_inlet(base=0.15, minimum=0.0,
                                       maximum=0.6)
        self.bounce_in = self.new_inlet(base=0.55, minimum=0.0,
                                        maximum=0.95)
        # How much purchase the shell has: 0 is glass, 1 grips hard.
        self.friction_in = self.new_inlet(base=0.3, minimum=0.0,
                                          maximum=1.0)
        # And how rough it is, which is a different thing. A smooth
        # shell lets a thing slide and hiss; a rough one makes it ride
        # up and let go, over and over, which taps.
        self.texture_in = self.new_inlet(base=0.25, minimum=0.0,
                                         maximum=1.0)
        # 0.4 rather than 0.6, since the range grew a harder top: this
        # is the same contact the old 0.6 gave, so the default sound
        # sits where it was and the new hardness is added ABOVE it
        # instead of underneath everything.
        self.hardness_in = self.new_inlet(base=0.4, minimum=0.0,
                                          maximum=1.0)
        self.gravity_in = self.new_inlet(base=9.80665, minimum=0.0,
                                         maximum=40.0)
        # How unalike the things in there are, and how irregular. At 0
        # they are identical smooth spheres, which in one shared field
        # keep step and sound like a handful however many there are.
        self.variety_in = self.new_inlet(base=0.35, minimum=0.0,
                                         maximum=1.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        RattleUnit._seeded += 1
        rng = np.random.default_rng(4242 + RattleUnit._seeded)
        self._sizes = np.ones(RattleUnit.MAX_PARTICLES, dtype=np.float64)
        # How far each one is off the average size, fixed per handful.
        # 'variety' scales this: at zero they really are all the same,
        # which is what the control has always claimed and never did --
        # the spread was drawn once at full width and variety only ever
        # reached the rebound scattering.
        self._size_dev = np.zeros(RattleUnit.MAX_PARTICLES,
                                  dtype=np.float64)
        # ...and how springy each one is. Unalike does not only mean
        # different SIZES. Things that bounce alike keep step however
        # much their sizes differ, because what decides when a thing
        # next arrives is how high it came off the wall. Spreading the
        # springiness is what actually puts them out of phase with each
        # other, and it neither adds energy nor takes any.
        self._spring_dev = np.zeros(RattleUnit.MAX_PARTICLES,
                                    dtype=np.float64)
        self._spring = np.ones(RattleUnit.MAX_PARTICLES,
                               dtype=np.float64)
        self._variety_live = -1.0
        self._spin_phase = np.zeros(RattleUnit.MAX_PARTICLES,
                                    dtype=np.float64)
        # Whether each thing is being HELD by the patch of wall it is
        # sitting on, and how much that patch will take before it lets
        # go. A smooth shell holds the same everywhere; a rough one does
        # not, and that is what turns a slide into a rattle.
        self._held = np.zeros(RattleUnit.MAX_PARTICLES, dtype=np.float64)
        self._support = np.zeros(RattleUnit.MAX_PARTICLES,
                                 dtype=np.float64)
        # Whether it was against the wall LAST step. An arrival is a
        # contact that was not there before.
        self._touching = np.zeros(RattleUnit.MAX_PARTICLES,
                                  dtype=np.float64)
        self._pos = np.zeros(RattleUnit.MAX_PARTICLES * 3, dtype=np.float64)
        self._vel = np.zeros(RattleUnit.MAX_PARTICLES * 3, dtype=np.float64)
        self._rng_start = rng
        self._count = 48
        self.shape = 0
        self._scatter()
        self._ring_knock = np.zeros(RattleUnit.RING, dtype=np.float64)
        self._ring_scrape = np.zeros(RattleUnit.RING, dtype=np.float64)
        self._ring_head = 0.0
        self._window = np.zeros(RattleUnit.RING, dtype=np.float64)
        self._window_width = 0
        self._size_live = 0.04
        self._grain_live = 0.006
        self._friction_live = 0.3
        self._bounce_live = 0.55
        self._spin_prev = np.zeros(3, dtype=np.float64)
        # Where it was pointing last step, so the rate can be got out
        # of the angle. Primed from the first sample seen, or a patch
        # that starts already tilted would look like it got there
        # infinitely fast.
        self._ang_prev = np.zeros(3, dtype=np.float64)
        self._turn_primed = False
        # Which way is down, in the container's own frame.
        self._down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self._rng = np.uint64(9137 + RattleUnit._seeded * 7919)
        self._quiet = True

        self.out = self.new_outlet()
        self.knock = self.new_outlet()
        self.scrape = self.new_outlet()
        self._ax = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._ay = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._az = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._wx = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._wy = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._wz = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._yk = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._ys = np.zeros(MAX_BLOCK, dtype=np.float64)

    def _scatter(self):
        """Drop them in, well inside the wall so none starts embedded."""
        rng = np.random.default_rng(97 + RattleUnit._seeded)
        n = RattleUnit.MAX_PARTICLES
        # Well inside the wall, and inside THIS wall: scattered over a
        # fixed four tenths of a metre they landed ten times outside a
        # default-sized shell, so every one of them started embedded,
        # was clamped out, and the first few seconds were a crash
        # rather than a settle.
        self._pos[:] = (rng.uniform(-0.5, 0.5, n * 3)
                        * self.size_in.base)
        self._vel[:] = rng.normal(0.0, 0.05, n * 3)
        # A spread of sizes, fixed per handful: this is what they are,
        # not something that changes while they rattle.
        self._size_dev[:] = np.clip(rng.normal(0.0, 1.0, n), -2.0, 2.0)
        self._spring_dev[:] = np.clip(rng.normal(0.0, 1.0, n), -2.0, 2.0)
        self._variety_live = -1.0
        self._spin_phase[:] = rng.uniform(0.0, 2.0 * math.pi, n)

    def set_count(self, count):
        count = int(min(RattleUnit.MAX_PARTICLES, max(1, count)))
        if count != self._count:
            self._count = count
            self._scatter()

    def reset(self):
        self._scatter()
        self._ring_knock[:] = 0.0
        self._ring_scrape[:] = 0.0
        self._ring_head = 0.0
        self._spin_prev[:] = 0.0
        self._ang_prev[:] = 0.0
        self._turn_primed = False
        self._down[:] = (0.0, 0.0, -1.0)
        self._quiet = True

    def render(self, frames):
        out = self.out
        knock_out = self.knock
        scrape_out = self.scrape
        if not self.enabled:
            for signal in (out, knock_out, scrape_out):
                signal.set_constant(0.0)
            return

        def fill(inlet, buffer):
            got = inlet.eval(frames)
            if got.constant:
                buffer[:frames] = got.value
            else:
                np.copyto(buffer[:frames], got.data[:frames],
                          casting='unsafe')
            return buffer[:frames]

        # Shake arrives in gravities, not in metres per second squared.
        # Read raw it was silent below about 10, because nothing lifts
        # off the floor until the shake beats its own weight -- true,
        # but it put the playable range at 10 to 40 where every other
        # unit here works over 0.3 to 3.
        ax = fill(self.shake_x_in, self._ax)
        ay = fill(self.shake_y_in, self._ay)
        az = fill(self.shake_z_in, self._az)
        np.multiply(ax, RattleUnit.GEE, out=ax)
        np.multiply(ay, RattleUnit.GEE, out=ay)
        np.multiply(az, RattleUnit.GEE, out=az)
        # Turn arrives in DEGREES, which is how anyone thinks about
        # how far a thing is tipped.
        wx = fill(self.turn_x_in, self._wx)
        wy = fill(self.turn_y_in, self._wy)
        wz = fill(self.turn_z_in, self._wz)
        np.multiply(wx, RattleUnit.DEG, out=wx)
        np.multiply(wy, RattleUnit.DEG, out=wy)
        np.multiply(wz, RattleUnit.DEG, out=wz)
        if not self._turn_primed:
            self._turn_primed = True
            self._ang_prev[:] = (wx[0], wy[0], wz[0])

        def scalar(inlet, lo, hi):
            got = inlet.eval(1)
            value = got.value if got.constant else float(got.data[0])
            return min(hi, max(lo, value))

        size = scalar(self.size_in, 0.005, 0.5)
        grain = scalar(self.grain_in, 0.0, 0.6) * size
        bounce = scalar(self.bounce_in, 0.0, 0.95)
        grip = scalar(self.friction_in, 0.0, 1.0)
        hard = scalar(self.hardness_in, 0.0, 1.0)
        grav = scalar(self.gravity_in, 0.0, 40.0)
        level = self.level_in.eval(frames)

        # The container's geometry glides. Dragged, a knob steps the
        # walls once a block, and a wall that jumps is a wall that
        # arrives at whatever is standing there.
        for name, target in (('_size_live', size), ('_grain_live', grain),
                             ('_friction_live', grip),
                             ('_bounce_live', bounce)):
            was = getattr(self, name)
            setattr(self, name, was + (target - was)
                    * min(1.0, frames / (0.05 * self.sample_rate)))
        size = self._size_live
        grain = self._grain_live
        scatter = scalar(self.variety_in, 0.0, 1.0)
        if scatter != self._variety_live:
            self._variety_live = scatter
            np.multiply(
                self._size_dev,
                RattleUnit.SIZE_FLOOR
                + (RattleUnit.SIZE_SPREAD - RattleUnit.SIZE_FLOOR)
                * scatter,
                out=self._sizes)
            np.add(self._sizes, 1.0, out=self._sizes)
            np.clip(self._sizes, 0.25, 2.0, out=self._sizes)
            np.multiply(self._spring_dev,
                        RattleUnit.SPRING_SPREAD * scatter,
                        out=self._spring)
            np.add(self._spring, 1.0, out=self._spring)
            np.clip(self._spring, 0.15, 1.85, out=self._spring)
        grip = self._friction_live
        texture = scalar(self.texture_in, 0.0, 1.0)
        bounce = self._bounce_live
        half_x = half_y = half_z = size
        if self.shape == 2:
            half_z = size * RattleUnit.EGG
        box = 1.0 if self.shape == 1 else 0.0
        # HARDER at the top than the mallet range the rest of the rack
        # uses, and it has to be: nothing here is a mallet. Contact time
        # falls with mass, and a light bead on a stiff plastic shell
        # rings far shorter than the hardest mallet head -- so eight
        # milliseconds of something soft down to about a twentieth of a
        # millisecond, which is two samples and as near an impulse as
        # this can carry. Stopping where a mallet stops, a third of a
        # millisecond, the contact set a floor under the RUB as well
        # (a rub is a run of tiny contacts), and a small egg shaker
        # came out too dark at the top of its range to be white.
        #
        # In AUDIO samples: the ring is read one per sample, so dividing
        # this by the decimation made every contact eight times shorter
        # than it was supposed to be, and a bank of contacts that short
        # is a bank of clicks.
        contact = max(2.0, RattleUnit.CONTACT_SOFT
                      * (RattleUnit.CONTACT_HARD / RattleUnit.CONTACT_SOFT)
                      ** hard * self.sample_rate)
        width = int(min(RattleUnit.RING - 1, max(2, contact)))
        if width != self._window_width:
            self._window_width = width
            edge = (np.arange(width) + 0.5) * (2.0 * math.pi / width)
            self._window[:width] = 0.5 * (1.0 - np.cos(edge))
        # Density changes the texture, not the level -- the house rule
        # everywhere in the rack. Independent sources add in POWER, so
        # a square root does it, and measured that is exactly what it
        # is: 0.503 / 0.498 / 0.496 across the whole range of variety.
        #
        # It read 1.35 while every contact went in one way only and
        # summed coherently, and 0.85 while blows were being laid down
        # once per particle per control step instead of once each. Both
        # of those are gone, and the honest exponent turned up on its
        # own.
        #
        # Anchored at 48, where the level was matched against shaker~.
        # No floor: flooring it left a single object far quieter than a
        # handful, which is why one of them could not be heard bouncing
        # around.
        gain = (RattleUnit.DENSITY_REF
                / max(1.0, self._count) ** RattleUnit.DENSITY_LAW)

        result = self._y[:frames]
        rk = self._yk[:frames]
        rs = self._ys[:frames]
        (self._ring_head, self._spin_prev[0], self._spin_prev[1],
         self._spin_prev[2], self._ang_prev[0], self._ang_prev[1],
         self._ang_prev[2], self._down[0], self._down[1],
         self._down[2], rng_state) = _rattle_kernel(
            ax, ay, az, wx, wy, wz, self._pos, self._vel,
            self._count, half_x, half_y, half_z, grain,
            self._sizes, self._spring, self._spin_phase, scatter,
            RattleUnit.TUMBLE, self._held, self._support,
            self._touching,
            RattleUnit.RELEASE, bounce, grip, texture,
            RattleUnit.TEXTURE_DRAG, RattleUnit.ASPERITY,
            RattleUnit.ASPERITY_ROUGH,
            RattleUnit.CONTACT_SKIN, RattleUnit.SCATTER_TILT,
            RattleUnit.HOLD_ROUGH,
            grav, box,
            RattleUnit.SPEED_CAP,
            RattleUnit.REST_SPEED, RattleUnit.REST_STEPS,
            RattleUnit.SLIDE_GAIN,
            RattleUnit.CONTROL_DECIM,
            RattleUnit.CONTROL_DECIM / self.sample_rate,
            RattleUnit.KNOCK_GAIN * gain, RattleUnit.SCRAPE_GAIN * gain,
            contact, self._window, self._ring_knock, self._ring_scrape,
            self._ring_head, self._spin_prev[0], self._spin_prev[1],
            self._spin_prev[2], self._ang_prev[0], self._ang_prev[1],
            self._ang_prev[2], self._down[0], self._down[1],
            self._down[2], self._rng, result, rk, rs)
        self._rng = np.uint64(rng_state)

        if not np.isfinite(result).all():
            self.reset()
            for signal in (out, knock_out, scrape_out):
                signal.set_constant(0.0)
            return

        glide = self._level_glide
        self._apply_level(result, level, frames)
        advanced = self._level_glide
        self._level_glide = glide
        self._apply_level(rk, level, frames)
        self._level_glide = glide
        self._apply_level(rs, level, frames)
        self._level_glide = advanced
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        np.copyto(knock_out.data[:frames], rk, casting='unsafe')
        knock_out.constant = False
        np.copyto(scrape_out.data[:frames], rs, casting='unsafe')
        scrape_out.constant = False
        self._quiet = bool(np.abs(result).max() < 1.0e-6)


class ShakerUnit(Unit):
    """Shaken percussion: the texture family, played by agitation.

    'shake' is the whole interface, and it means exactly what it says: how
    hard the vessel is being moved, right now. A wrist flick is a burst of
    grains that settles as the beans do; a steady tremble is a sustained
    wash; stillness is silence. An effort stream patched here needs no
    translation at all -- shaking a sensor is shaking the shaker.

    'density' is collisions per second at full shake (a handful is
    countable ticks, thousands is rain), loudness-compensated so it
    changes texture rather than level. 'settle' is how long the beans
    keep moving after the gesture stops; 'hardness' how sharp each tick
    is. The vessel is a tunable resonance ('vessel', 'resonance') -- and
    it is plural: an ensemble of eight members, each with its own fixed
    tuning spread across the 'jingle' band, each collision striking one
    while the others keep ringing. A tambourine is a dozen jingles and
    sleighbells are a strap of bells; what makes them them is overlapping
    decays at distinct pitches, and that needs polyphony, not a single
    resonator retuned out from under its own ring. At jingle 0 the
    members coincide and the vessel is one voice again.

    'swirl' is not a second gesture to be mixed with the first. You
    cannot shake a maraca while you are rolling it, or roll it while you
    are shaking it -- there is one agitation, and what changes is the
    ANGLE it meets the shell at. Head on, a bean stops dead against the
    wall and rings it: the tick. Tangential, it keeps its speed along
    the wall and drags: the graze. Everything between is both, and this
    is where between.

    So opening it moves the sound from one to the other rather than
    adding a second one, and the impacts get finer and more numerous on
    the way -- a bean that skips rather than stops makes more contacts
    and smaller ones.

    Nothing here wobbles it. A real roll surges as the heap comes round,
    but that is a shape a hand makes, and a hand is what should make it:
    patch it from the movement, or from an LFO, into 'shake' or into
    this. An oscillator hidden in here would only be in the way.

    'grains out' carries the raw collisions before the vessel: patch it
    into modal~ (drive up, dry 0) and the beans rattle inside any object
    the table editor can draw. The coupling really is one-way -- beans
    excite the vessel, the vessel does not stir the beans -- so this is
    the rare physical seam an ordinary cord models honestly.
    """

    # How much finer and more numerous the impacts get as the angle
    # opens. A glancing bean does not stop dead against the wall, it
    # skips along it, so there are more contacts and each is smaller.
    FINE_RATE = 5.0
    # Contacts a second of bean dragging wall, per unit of agitation.
    # Nine thousand was the first guess and it was wrong the way dense
    # grains are always wrong: five thousand a second is not friction,
    # it is the central limit theorem, and it measured a kurtosis of
    # three, which is gaussian noise exactly.
    SLIDE_DENSITY = 2500.0
    SLIDE_GAIN = 0.5
    SLIDE_RING = 64
    HOLD_SCALE = 1.0
    SHAKE_MODES = ('throw', 'hold')
    # How much a whole sweep of the gesture is worth read as a stroke.
    THROW_KICK = 1.1
    # How much longer the beans stay agitated at a full tangential
    # angle. A swirl's speed never passes through zero, so nothing ever
    # lets them settle between strokes.
    SWIRL_SMOOTH = 5.0
    # How much of the lengthened settle to take back out of the stroke.
    # Not all of it: strokes only pile up once they start arriving
    # inside each other's tails, and how soon that happens depends on
    # how fast the hand is going -- which this cannot know. Taking the
    # whole of it out assumed they always overlap, and cost the roll
    # three and a half decibels the moment the angle left zero.
    # Found by measurement, not by argument: at 1 the roll lost three
    # and a half decibels the moment the angle left zero, and the whole
    # of that loss happened in the first quarter of the travel -- the
    # correction was arriving before there was anything to correct. This
    # holds it flat within about a decibel from a slow gesture to a fast
    # one.
    THROW_NORM = 0.85
    # What a bean gives the wall even side on. Zero is geometry; a
    # maraca rolled flat out still has beans in it.
    HEAD_FLOOR = 0.3

    _seeded = 0

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.shake_in = self.new_inlet(base=0.0, minimum=0.0, maximum=2.0)
        # Where the beans meet the shell, from head on to tangential.
        # Shake and roll are not two gestures to be mixed -- you cannot
        # shake a maraca while you are rolling it -- they are one
        # agitation arriving at an angle, and this is the angle.
        self.swirl_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.density_in = self.new_inlet(base=64.0, minimum=1.0,
                                         maximum=2000.0)
        self.settle_in = self.new_inlet(base=0.12, minimum=0.02, maximum=1.0)
        self.hardness_in = self.new_inlet(base=0.7, minimum=0.0, maximum=1.0)
        self.vessel_in = self.new_inlet(base=3200.0, minimum=100.0,
                                        maximum=12000.0)
        self.resonance_in = self.new_inlet(base=0.7, minimum=0.0, maximum=1.0)
        self.jingle_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.vary_in = self.new_inlet(base=0.4, minimum=0.0, maximum=1.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        # Each instance gets its own generator stream: deterministic from
        # run to run, different from shaker to shaker, and never looping.
        ShakerUnit._seeded += 1
        seed = (ShakerUnit._seeded * 0x9E3779B97F4A7C15) % (1 << 64)
        self._rng = np.uint64(seed if seed else 0x853C49E6748FEA9B)
        self._energy = 0.0
        self._quiet = True
        # The ensemble's tunings: fixed per instance, drawn from the same
        # seed, so this shaker's bells are ITS bells, every run.
        MEMBERS = 8
        spread_rng = np.random.RandomState(seed & 0xFFFFFFFF or 1)
        self._offsets = spread_rng.uniform(-0.5, 0.5, MEMBERS)
        self._thetas = np.zeros(MEMBERS, dtype=np.float64)
        self._sounds = np.zeros(MEMBERS, dtype=np.float64)
        self._envs = np.zeros(MEMBERS, dtype=np.float64)
        self._gds = np.full(MEMBERS, 0.99, dtype=np.float64)
        self._ry1 = np.zeros(MEMBERS, dtype=np.float64)
        self._ry2 = np.zeros(MEMBERS, dtype=np.float64)
        self._rz1 = np.zeros(MEMBERS, dtype=np.float64)
        self._rz2 = np.zeros(MEMBERS, dtype=np.float64)
        self._b1s = np.zeros(MEMBERS, dtype=np.float64)
        self._b1zs = np.zeros(MEMBERS, dtype=np.float64)
        self._g2s = np.zeros(MEMBERS, dtype=np.float64)

        self.out = self.new_outlet()
        self.grains = self.new_outlet()
        self._shake = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._raw = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._slide_ring = np.zeros(ShakerUnit.SLIDE_RING, dtype=np.float64)
        self._slide_head = 0.0
        # 0 throws the beans, 1 holds them where the gesture says.
        self.shake_mode = 0
        self._shake_prev = 0.0
        self._settle_live = 0.12
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self._energy = 0.0
        self._sounds[:] = 0.0
        self._envs[:] = 0.0
        self._ry1[:] = 0.0
        self._ry2[:] = 0.0
        self._rz1[:] = 0.0
        self._rz2[:] = 0.0
        self._quiet = True

    def deactivate(self):
        self.reset()

    def render(self, frames):
        shake = self.shake_in.eval(frames)
        density = self.density_in.eval(frames)
        settle = self.settle_in.eval(frames)
        hardness = self.hardness_in.eval(frames)
        vessel = self.vessel_in.eval(frames)
        resonance = self.resonance_in.eval(frames)
        jingle = self.jingle_in.eval(frames)
        vary = self.vary_in.eval(frames)
        out_level = self.level_in.eval(frames)

        out = self.out
        if not _svf_ready.is_set():
            out.set_constant(0.0)
            self.grains.set_constant(0.0)
            return

        gesture = self._shake[:frames]
        if shake.constant:
            gesture[:] = shake.value
            idle = abs(shake.value) < 1.0e-4
        else:
            np.copyto(gesture, shake.data[:frames])
            idle = False
        np.clip(gesture, 0.0, 2.0, out=gesture)

        # Skip only when the hand is still, the output has faded AND the
        # beans have stopped moving -- between sparse collisions the output
        # alone can look silent while the system is still agitated, and
        # cutting there would truncate the settle after a flick.
        if self._quiet and idle and self._energy < 1.0e-4:
            out.set_constant(0.0)
            self.grains.set_constant(0.0)
            return

        def scalar(signal, lo, hi):
            value = signal.value if signal.constant else float(signal.data[0])
            return min(hi, max(lo, value))

        beans = scalar(density, 1.0, 2000.0)
        settle_now = scalar(settle, 0.02, 1.0)
        hard = scalar(hardness, 0.0, 1.0)
        vessel_hz = scalar(vessel, 100.0, min(12000.0,
                                              self.sample_rate * 0.45))
        res = scalar(resonance, 0.0, 1.0)
        jingle_now = scalar(jingle, 0.0, 1.0)
        vary_now = scalar(vary, 0.0, 1.0)

        # A shake is ONE DIMENSIONAL: back and forth, so the hand stops
        # dead at each turnaround and the agitation pulses -- clear
        # peaks and troughs, which is what makes it a rhythm. A swirl is
        # sine AND cosine: the speed never passes through zero, so the
        # beans are never not being driven and the intensity has no
        # troughs to fall into. Its peaks are subtler and scrapier.
        #
        # So opening the angle lengthens how long they stay agitated.
        # That is the same lever as 'settle', deliberately: reaching for
        # settle to get a roll is the right instinct, and this is the
        # instinct built in.
        swirl_now = self.swirl_in.eval(frames)
        swirl_now = abs(swirl_now.value if swirl_now.constant
                        else float(swirl_now.data[0]))
        swirl_now = min(1.0, swirl_now)
        settle_now *= 1.0 + ShakerUnit.SWIRL_SMOOTH * swirl_now

        # Settle glides. Changing it does not reach back into beans that
        # are already moving: dropping it to end a roll used to cut the
        # ring off where it stood, which is a thing no hand can do to a
        # shaker. It arrives over a tenth of a second instead.
        self._settle_live += ((settle_now - self._settle_live)
                              * min(1.0, frames / (0.1 * self.sample_rate)))
        settle_now = self._settle_live
        energy_decay = math.exp(-1.0 / (settle_now * self.sample_rate))
        # Hard beans are short ticks: 20 ms of felt down to half a
        # millisecond of glass, exponentially.
        grain_seconds = 0.02 * (0.025 ** hard)
        grain_decay = math.exp(-1.0 / (grain_seconds * self.sample_rate))
        attack_k = 1.0 - math.exp(-1.0 / (0.15 * grain_seconds
                                          * self.sample_rate))
        # Density changes the texture, not the level: grain amplitude is
        # compensated so rain is not simply louder than a maraca.
        amp = math.sqrt(64.0 / max(8.0, beans))
        theta = 2.0 * math.pi * vessel_hz / self.sample_rate
        # Exponential from a 3 ms thud to a tenth-of-a-second bell ring:
        # the whole audible range of ring time, most of it living in the
        # top third of the knob, where tambourine and sleighbells are.
        radius = 1.0 - 0.15 * math.pow(750.0, -res)
        # Spread the ensemble across the jingle band around the vessel
        # knob. The members' offsets are fixed; jingle only opens the fan.
        # The kernel derives both coefficients from angle and radius
        # together, per block, so the filters are always self-consistent.
        np.multiply(self._offsets, jingle_now * 0.8, out=self._thetas)
        self._thetas += 1.0
        self._thetas *= theta
        np.clip(self._thetas, 1.0e-3, math.pi * 0.95, out=self._thetas)

        # Swirling pins the beans to the wall and they stop being
        # thrown at it -- they are dragged round it instead. The pull
        # that pins them is the hand's circle, not the shaker's, and it
        # beats gravity at just under two turns a second for a wrist.
        # Soft either side of that, because a hand does not cross it
        # cleanly and neither do the beans.
        swirl = swirl_now
        # One impact, resolved. Head on it rings the shell; tangential
        # it drags along it. Sine and cosine because the two are
        # components of the same speed, so opening the angle moves the
        # sound from one to the other rather than adding a second one.
        angle = swirl * 0.5 * math.pi
        # Not all the way to nothing. A fully tangential bean still
        # gives the wall SOME of itself -- the wall is curved, the beans
        # are round, and they carom off each other whatever the hand is
        # doing. A plain cosine reaches zero at the top of the knob and
        # the ticks vanish with it, so the last tenth of the travel fell
        # away to almost silence.
        head_on = (ShakerUnit.HEAD_FLOOR
                   + (1.0 - ShakerUnit.HEAD_FLOOR) * math.cos(angle))
        glancing = math.sin(angle)

        # A bean that skips rather than stops makes more contacts and
        # smaller ones: texture, not level.
        rate_boost = 1.0 + ShakerUnit.FINE_RATE * swirl
        fine = 1.0 / math.sqrt(rate_boost)
        slide_base = (ShakerUnit.SLIDE_DENSITY * glancing
                      / self.sample_rate)
        slide_amp = ShakerUnit.SLIDE_GAIN * glancing * amp
        tail_index = 2.3 - 1.0 * hard
        slide_tail = 1.0 / tail_index
        slide_max = 40.0
        clip = slide_max ** (-tail_index)
        power = 1.0 - 2.0 / tail_index
        spread = ((1.0 - clip ** power) / power if abs(power) > 1.0e-9
                  else -math.log(clip))
        slide_norm = 1.0 / math.sqrt(max(slide_max * slide_max * clip
                                         + spread, 1.0e-9))
        slide_least = int(round(1.0 + (1.0 - hard) * 7.0))
        # Thrown, the strokes keep arriving while the beans hold their
        # energy longer -- so rolling did not smooth the sound, it piled
        # it up: ten decibels by half travel, and the peak agitation
        # more than tripled. Rolling should change how the strokes JOIN,
        # not how much each one is worth, so the stroke is scaled
        # against how long it will now last. The tail still lengthens;
        # the level does not run away with it.
        throw_kick = ShakerUnit.THROW_KICK / (
            (1.0 + ShakerUnit.SWIRL_SMOOTH * swirl)
            ** ShakerUnit.THROW_NORM)
        hold = 1.0 if self.shake_mode else 0.0
        hold_k = min(0.5, 1.0 / max(1.0, settle_now * self.sample_rate))
        raw = self._raw[:frames]
        result = self._y[:frames]
        (self._energy, rng_state, self._slide_head,
         self._shake_prev) = _shaker_kernel(
            gesture, beans / self.sample_rate, energy_decay, grain_decay,
            vary_now, amp, attack_k, self._thetas, radius,
            self._b1s, self._b1zs, self._g2s,
            self._energy, self._sounds, self._gds, self._envs,
            self._ry1, self._ry2, self._rz1, self._rz2, self._rng,
            raw, result,
            head_on, rate_boost, fine, hold, ShakerUnit.HOLD_SCALE,
            hold_k, throw_kick, self._shake_prev,
            slide_base, slide_amp, slide_tail,
            slide_max,
            slide_norm, slide_least, 1.25,
            self._slide_ring, self._slide_head)
        # numba hands the state back as a Python int, and a bare int at or
        # above 2**63 fails the signed conversion on the way back in --
        # half of all states. It must go home as the unsigned it is.
        self._rng = np.uint64(rng_state)

        self._apply_level(result, out_level, frames)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        np.copyto(self.grains.data[:frames], raw, casting='unsafe')
        self.grains.constant = False
        scratch = self._scratch[:frames]
        np.abs(result, out=scratch)
        self._quiet = bool(scratch.max() < 1.0e-5)


def _whoosh_kernel_source(speed, omega, amp, wake_mix, shed_r, shed_g,
                          wake_k, smooth_k, vel, y1, y2, lp, rng, out):
    """Aeolian sound, sample by sample: motion through air.

    A moving object sheds vortices at a frequency set by its speed over
    its size, and the shedding sings: a resonant bandpass on turbulence,
    centred wherever the speed says, per sample. Behind it the wake --
    broadband, lowpassed, the hiss of stirred air. Speed is smoothed
    over a few milliseconds on the way in, so a control-rate effort
    stream drives it without zippering.
    """
    for i in range(speed.shape[0]):
        vel += (speed[i] - vel) * smooth_k
        rng, n1 = _rand01(rng)
        noise = 2.0 * n1 - 1.0
        w = omega[i] * vel
        if w > 2.8:
            w = 2.8
        b1 = 2.0 * shed_r * math.cos(w)
        y = shed_g * noise + b1 * y1 - shed_r * shed_r * y2
        y2 = y1
        y1 = y
        rng, n2 = _rand01(rng)
        lp += ((2.0 * n2 - 1.0) - lp) * wake_k
        out[i] = amp[i] * ((1.0 - wake_mix) * y + wake_mix * 2.5 * lp)
    return vel, y1, y2, lp, rng


if _HAVE_NUMBA:
    _whoosh_kernel = njit(cache=True, fastmath=True)(_whoosh_kernel_source)
else:
    _whoosh_kernel = _whoosh_kernel_source


class WhooshUnit(Unit):
    """Motion through air: the whoosh, with speed as its only player.

    The most legible mapping in the rack, because it is the physics
    itself: vortex shedding puts the pitch of the swish at speed over
    size, and aeolian radiation makes loudness rise steeply with speed
    -- slow motion whispers, fast motion roars, stillness is silent.
    Patch a limb's speed into 'speed' and the air does the rest.

    'size' is the object: a thin edge sings high, a thick limb rumbles.
    'edge' is how bladelike the shedding is -- a taut wire whistles, a
    hand merely swishes. 'wake' mixes in the broadband hiss of stirred
    air behind the object.
    """

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.speed_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.5)
        self.size_in = self.new_inlet(base=0.4, minimum=0.0, maximum=1.0)
        self.edge_in = self.new_inlet(base=0.4, minimum=0.0, maximum=1.0)
        self.wake_in = self.new_inlet(base=0.4, minimum=0.0, maximum=1.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        WhooshUnit._seeded = getattr(WhooshUnit, '_seeded', 0) + 1
        seed = (WhooshUnit._seeded * 0x9E3779B97F4A7C15) % (1 << 64)
        self._rng = np.uint64(seed if seed else 0x853C49E6748FEA9B)
        self._vel = 0.0
        self._y1 = 0.0
        self._y2 = 0.0
        self._lp = 0.0
        self._quiet = True

        self.out = self.new_outlet()
        self._speed = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._omega = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._amp = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self._vel = 0.0
        self._y1 = 0.0
        self._y2 = 0.0
        self._lp = 0.0
        self._quiet = True

    def render(self, frames):
        speed = self.speed_in.eval(frames)
        size = self.size_in.eval(frames)
        edge = self.edge_in.eval(frames)
        wake = self.wake_in.eval(frames)
        out_level = self.level_in.eval(frames)

        out = self.out
        if not _svf_ready.is_set():
            out.set_constant(0.0)
            return

        gust = self._speed[:frames]
        if speed.constant:
            gust[:] = speed.value
            still = abs(speed.value) < 1.0e-4
        else:
            np.copyto(gust, speed.data[:frames])
            still = False
        np.clip(gust, 0.0, 1.5, out=gust)

        if self._quiet and still and self._vel < 1.0e-4:
            out.set_constant(0.0)
            return

        def scalar(signal, lo, hi):
            value = signal.value if signal.constant else float(signal.data[0])
            return min(hi, max(lo, value))

        size_now = scalar(size, 0.0, 1.0)
        edge_now = scalar(edge, 0.0, 1.0)
        wake_now = scalar(wake, 0.0, 1.0)

        # The shedding line: a thin edge at full speed sings near 6 kHz, a
        # thick limb near 120 Hz, and the frequency is linear in speed as
        # Strouhal says it should be.
        f_full = 6000.0 * (0.02 ** size_now)
        omega = self._omega[:frames]
        omega[:] = 2.0 * math.pi * f_full / self.sample_rate

        # Aeolian loudness: steep in speed. The whisper-to-roar curve is
        # most of what makes a whoosh read as effort.
        amp = self._amp[:frames]
        np.power(gust, 2.5, out=amp)
        amp *= 0.85

        # The shedding's sharpness: a wire whistles, a hand swishes.
        q = 1.5 + edge_now * 28.0
        f_now = max(40.0, f_full * max(0.05, float(gust[0])))
        shed_r = math.exp(-math.pi * f_now / (q * self.sample_rate))
        shed_g = (1.0 - shed_r) * (1.2 + 2.0 * edge_now)
        wake_cut = 150.0 + 0.4 * f_now
        wake_k = 1.0 - math.exp(-2.0 * math.pi * wake_cut / self.sample_rate)
        smooth_k = 1.0 - math.exp(-1.0 / (0.004 * self.sample_rate))

        result = self._y[:frames]
        (self._vel, self._y1, self._y2, self._lp, rng_state) = _whoosh_kernel(
            gust, omega, amp, wake_now, shed_r, shed_g, wake_k, smooth_k,
            self._vel, self._y1, self._y2, self._lp, self._rng, result)
        self._rng = np.uint64(rng_state)

        self._apply_level(result, out_level, frames)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        scratch = self._scratch[:frames]
        np.abs(result, out=scratch)
        self._quiet = bool(scratch.max() < 1.0e-5)


def _noise_kernel_source(pressure, amp_scale, color_k, bright, hp_k,
                         p_close, p_open, gate_atk, gate_rel, gate_floor,
                         spit_gain, spit_decay, crk_dec, ch_w0, ch_fall,
                         ch_dec, ab1, ab2, ag, ag2, m_dry, m_res,
                         smooth_k,
                         pres, gate, is_open, spit, blocked, lp, hp,
                         a1, a2, a3, a4, crk, crk_lp, ch_ph, ch_w,
                         ch_amp, rng,
                         out):
    """A leak, sample by sample: pressure escaping through an orifice.

    The hiss is turbulence -- white noise through a one-pole whose
    cutoff is the color knob, power-compensated so color changes the
    timbre and never the level, brightening a little as pressure rises
    the way a harder leak does. And the jet SINGS: the aperture puts a
    resonant hump on the noise at the frequency its size dictates --
    a pinhole whistles high and tight, a wide gap breathes low and
    broad -- rising a little with pressure, as a harder jet does.

    Sputter is a telegraph process, not a wobble: the orifice blocks
    (condensation, debris) and blows back open. While it is blocked the
    pressure behind it builds, so the reopening SPITS -- an overshoot
    burst scaled by how long the blockage held. Blocked spells run
    shorter than open ones, partial at low sputter, hard dropouts at
    full. The gate reopens faster than it closes, because blow-through
    is abrupt and clogging is not.

    And a spit is an EVENT, not just a louder moment: it crackles --
    discrete micro-pops peppering its short life, the fire sound of
    droplets flashing -- and it can pip: a brief falling whistle as
    the pressure releases through the wet orifice, the kettle's chirp,
    pitched and present differently every time.
    """
    for i in range(pressure.shape[0]):
        pres += (pressure[i] - pres) * smooth_k
        rng, u = _rand01(rng)
        if is_open > 0.5:
            if u < p_close:
                is_open = 0.0
                blocked = 0.0
        else:
            blocked += 1.0
            if u < p_open:
                is_open = 1.0
                s = blocked * spit_gain
                if s > 1.2:
                    s = 1.2
                spit += s
                rng, uc = _rand01(rng)
                if uc < 0.7:
                    rng, uw = _rand01(rng)
                    ch_w = ch_w0 * (0.6 + 0.8 * uw)
                    rng, ua = _rand01(rng)
                    ch_amp = s * (0.5 + 0.9 * ua)
                    ch_ph = 0.0
        target = 1.0 if is_open > 0.5 else gate_floor
        if target > gate:
            gate += (target - gate) * gate_atk
        else:
            gate += (target - gate) * gate_rel
        spit *= spit_decay
        # The crackle: micro-pops arriving densely while the spit is
        # alive, each a tiny heavy-tailed click.
        rng, up = _rand01(rng)
        if up < 0.3 * spit:
            rng, up2 = _rand01(rng)
            crk += 0.5 + up2
            if crk > 3.0:
                crk = 3.0
        crk *= crk_dec
        # The pip: a falling whistle, gone in tens of milliseconds.
        pip = 0.0
        if ch_amp > 1.0e-4:
            ch_ph += ch_w
            if ch_ph > 6.283185307179586:
                ch_ph -= 6.283185307179586
            ch_w *= ch_fall
            pip = ch_amp * math.sin(ch_ph)
            ch_amp *= ch_dec
        rng, nz = _rand01(rng)
        white = 2.0 * nz - 1.0
        k = color_k * (0.4 + bright * pres)
        if k > 1.0:
            k = 1.0
        lp += (white - lp) * k
        cc = math.sqrt((2.0 - k) / k)
        v = lp * cc
        hp += (v - hp) * hp_k
        v -= hp
        apv = ag * v + ab1 * a1 + ab2 * a2
        a2 = a1
        a1 = apv
        # Twice through the jet's resonance: one two-pole's skirt
        # leaks a broadband floor beside a quiet whistle; two make
        # the hole pass its note and nothing else.
        apv2 = ag2 * apv + ab1 * a3 + ab2 * a4
        a4 = a3
        a3 = apv2
        v = m_dry * v + m_res * apv2
        rng, nc = _rand01(rng)
        coarse = 2.0 * nc - 1.0
        coarse = coarse * coarse * coarse
        # The crackle leaves through the same passage as the hiss: a
        # dark leak crackles dark, at the same compensated power.
        crk_lp += (coarse - crk_lp) * k
        loud = amp_scale * (pres ** 1.8)
        # A spit is STATIC, not a gust: its voice is the crackle, and
        # the noise path gets only a modest lift.
        out[i] = loud * (v * (gate + 0.35 * spit)
                         + 2.2 * crk * crk_lp * cc + 0.9 * pip)
    return (pres, gate, is_open, spit, blocked, lp, hp, a1, a2, a3, a4,
            crk, crk_lp, ch_ph, ch_w, ch_amp, rng)


if _HAVE_NUMBA:
    _noise_kernel = njit(cache=True, fastmath=True)(_noise_kernel_source)
else:
    _noise_kernel = _noise_kernel_source


class NoiseUnit(Unit):
    """A leak: the noise source the rack was missing, played by pressure.

    Every other unit keeps its noise inside; this one hands it to the
    patch. 'pressure' is the whole interface -- loudness rises steeply
    with it as turbulence does, brightness rises gently, stillness is
    silence -- and it is smoothed on the way in so a control-rate
    effort stream drives it without zippering.

    'color' tilts the spectrum from dark rumble to full white at
    constant power. 'sputter' breaks the flow up: a telegraph blockage
    that builds pressure while closed and spits on reopening, partial
    flutter at low values, hard dropouts at high. 'rate' is its tempo,
    slow gulps to buzzy flutter.

    Downstream it is a source like any other: through formant~ it is
    breath, through modal~ it is a rattling surface, through vcf~ it
    is classic subtractive hiss.
    """

    _seeded = 0

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.pressure_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)
        self.color_in = self.new_inlet(base=0.85, minimum=0.0, maximum=1.0)
        self.sputter_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.aperture_in = self.new_inlet(base=0.35, minimum=0.0,
                                          maximum=1.0)
        self.whistle_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.rate_in = self.new_inlet(base=10.0, minimum=0.2, maximum=200.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        NoiseUnit._seeded += 1
        seed = (NoiseUnit._seeded * 0x9E3779B97F4A7C15) % (1 << 64)
        self._rng = np.uint64(seed if seed else 0x853C49E6748FEA9B)
        self._pres = 0.0
        self._gate = 1.0
        self._open = 1.0
        self._spit = 0.0
        self._blocked = 0.0
        self._lp = 0.0
        self._hp = 0.0
        self._a1 = 0.0
        self._a2 = 0.0
        self._a3 = 0.0
        self._a4 = 0.0
        self._crk = 0.0
        self._crk_lp = 0.0
        self._ch_ph = 0.0
        self._ch_w = 0.0
        self._ch_amp = 0.0
        self._quiet = True

        self.out = self.new_outlet()
        self._pressure = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self._pres = 0.0
        self._gate = 1.0
        self._open = 1.0
        self._spit = 0.0
        self._lp = 0.0
        self._hp = 0.0
        self._a1 = 0.0
        self._a2 = 0.0
        self._a3 = 0.0
        self._a4 = 0.0
        self._crk = 0.0
        self._crk_lp = 0.0
        self._ch_amp = 0.0
        self._quiet = True

    def render(self, frames):
        pressure = self.pressure_in.eval(frames)
        color = self.color_in.eval(frames)
        sputter = self.sputter_in.eval(frames)
        aperture = self.aperture_in.eval(frames)
        whistle = self.whistle_in.eval(frames)
        rate = self.rate_in.eval(frames)
        out_level = self.level_in.eval(frames)

        out = self.out
        if not _svf_ready.is_set():
            out.set_constant(0.0)
            return

        drive = self._pressure[:frames]
        if pressure.constant:
            drive[:] = pressure.value
            idle = abs(pressure.value) < 1.0e-4
        else:
            np.copyto(drive, pressure.data[:frames])
            idle = False
        np.clip(drive, 0.0, 2.0, out=drive)

        if self._quiet and idle and self._pres < 1.0e-4:
            out.set_constant(0.0)
            return

        def scalar(signal, lo, hi):
            value = signal.value if signal.constant else float(signal.data[0])
            return min(hi, max(lo, value))

        color_now = scalar(color, 0.0, 1.0)
        sputter_now = scalar(sputter, 0.0, 1.0)
        aperture_now = scalar(aperture, 0.0, 1.0)
        whistle_now = scalar(whistle, 0.0, 1.0)
        rate_now = scalar(rate, 0.2, 200.0)

        # Dark rumble to full white, exponentially: 30 Hz to the top.
        cutoff = 30.0 * (800.0 ** color_now)
        color_k = 1.0 - math.exp(-2.0 * math.pi * cutoff / self.sample_rate)
        hp_k = 1.0 - math.exp(-2.0 * math.pi * 25.0 / self.sample_rate)
        # Blockage arrives only as sputter opens. Rate is a TEMPO: it
        # sets how often the flow breaks, and the gaps never stretch
        # past about sixty milliseconds however slow the tempo -- a
        # rare sputter is a rare NORMAL sputter, not a long outage.
        # Harder flow flutters faster: the tempo breathes with the
        # (smoothed) pressure.
        rate_eff = rate_now * (0.4 + 0.8 * min(1.5, self._pres))
        p_close = rate_eff * 1.2 * sputter_now / self.sample_rate
        p_open = max(16.0, rate_eff * 1.1) / self.sample_rate
        gate_atk = 1.0 - math.exp(-1.0 / (0.0012 * self.sample_rate))
        gate_rel = 1.0 - math.exp(-1.0 / (0.005 * self.sample_rate))
        gate_floor = 1.0 - sputter_now
        spit_gain = 1.2 * sputter_now / (0.06 * self.sample_rate)
        spit_decay = math.exp(-1.0 / (0.010 * self.sample_rate))
        # Crackle pops die in under a millisecond; the pip starts near
        # three kilohertz and falls an octave in a dozen milliseconds.
        crk_dec = math.exp(-1.0 / (0.0008 * self.sample_rate))
        ch_w0 = 2.0 * math.pi * min(2900.0, 0.4 * self.sample_rate) \
            / self.sample_rate
        ch_fall = math.exp(-1.0 / (0.017 * self.sample_rate))
        ch_dec = math.exp(-1.0 / (0.012 * self.sample_rate))
        # The jet's song: a pinhole at seven kilohertz down to a wide
        # gap near six hundred, exponentially, pulled up a little by
        # pressure the way a harder jet whistles sharper. A small hole
        # BLOCKS what does not fit through it: at the pinhole the
        # whistle is nearly the whole voice and it is tight; a wide
        # gap mostly just breathes.
        f_ap = 16000.0 * (0.032 ** aperture_now) \
            * (0.8 + 0.3 * min(1.5, self._pres))
        th_a = 2.0 * math.pi * min(f_ap, 0.45 * self.sample_rate) \
            / self.sample_rate
        # 'whistle' is the tightness of the jet's resonance: breathy
        # at 0, piercing near-pure at 1, the old feel at the middle.
        r_a = 0.90 + 0.08 * (1.0 - aperture_now) \
            + 0.18 * (whistle_now - 0.5)
        if r_a > 0.995:
            r_a = 0.995
        elif r_a < 0.6:
            r_a = 0.6
        ab1 = 2.0 * r_a * math.cos(th_a)
        ab2 = -r_a * r_a
        ag = (1.0 - r_a) * math.sin(th_a) * 1.1
        # Second pass, unity at its own peak: reshapes without
        # relevelling.
        ag2 = (1.0 - r_a) * math.sin(th_a)
        # No floor on the bypass: a pinprick passes ONLY its whistle.
        m_dry = 0.8 * aperture_now ** 1.3
        # Narrow bands carry little energy: makeup as the hole closes,
        # so the pinprick sings at a usable level -- and as the
        # whistle tightens, energy-constant against the mid-knob
        # reference, so tight does not mean vanishing.
        r_mid = 0.90 + 0.08 * (1.0 - aperture_now)
        m_res = 1.8 * (1.0 - 0.45 * aperture_now) \
            * (1.0 + 2.2 * (1.0 - aperture_now)) \
            * math.sqrt((1.0 - r_mid) / (1.0 - r_a))
        smooth_k = 1.0 - math.exp(-1.0 / (0.004 * self.sample_rate))

        result = self._y[:frames]
        (self._pres, self._gate, self._open, self._spit, self._blocked,
         self._lp, self._hp, self._a1, self._a2, self._a3, self._a4,
         self._crk, self._crk_lp, self._ch_ph, self._ch_w,
         self._ch_amp, rng_state) = _noise_kernel(
            drive, 0.35, color_k, 0.6, hp_k,
            p_close, p_open, gate_atk, gate_rel, gate_floor,
            spit_gain, spit_decay, crk_dec, ch_w0, ch_fall, ch_dec,
            ab1, ab2, ag, ag2, m_dry, m_res, smooth_k,
            self._pres, self._gate, self._open, self._spit, self._blocked,
            self._lp, self._hp, self._a1, self._a2, self._a3, self._a4,
            self._crk, self._crk_lp, self._ch_ph, self._ch_w,
            self._ch_amp, self._rng, result)
        self._rng = np.uint64(rng_state)

        self._apply_level(result, out_level, frames)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        scratch = self._scratch[:frames]
        np.abs(result, out=scratch)
        self._quiet = bool(scratch.max() < 1.0e-5)


def _bounce_kernel_source(drop, g, restitution, press_kill,
                          contact_samples, strike_scale, vmin,
                          h, v, armed, pulse_amp, pulse_at, pulse_len,
                          rng, out):
    """A dropped mallet, integrated at audio rate.

    Height and velocity are the whole state. Each impact reverses the
    velocity through the restitution, so the intervals and strengths
    shrink geometrically -- the accelerating cadence of a dropped ball
    and of a drum roll is not a pattern here, it is gravity. The strike
    itself is a half-cosine force pulse, the real shape of a mallet
    contact: click-free by construction, wider for soft mallets.

    Pressing kills the rebound; the ball is judged at rest when its
    next flight would be shorter than its own contact, which is where
    a roll's buzz ends.
    """
    for i in range(drop.shape[0]):
        d = drop[i]
        if armed > 0.5:
            if d > 1.0e-4:
                h = d
                v = 0.0
                armed = 0.0
        elif d <= 1.0e-4:
            armed = 1.0
        if h > 0.0 or v != 0.0:
            v -= g
            h += v
            if h <= 0.0 and v < 0.0:
                speed = -v
                h = 0.0
                if speed > vmin:
                    rng, u = _rand01(rng)
                    v = speed * restitution * press_kill \
                        * (0.94 + 0.12 * u)
                    pulse_amp = speed * strike_scale
                    if pulse_amp > 3.0:
                        pulse_amp = 3.0
                    pulse_at = 0.0
                    pulse_len = contact_samples
                else:
                    v = 0.0
        s = 0.0
        if pulse_at < pulse_len:
            s = pulse_amp * 0.5 \
                * (1.0 - math.cos(6.283185307179586 * pulse_at
                                  / pulse_len))
            pulse_at += 1.0
        out[i] = s
    return h, v, armed, pulse_amp, pulse_at, pulse_len, rng


if _HAVE_NUMBA:
    _bounce_kernel = njit(cache=True, fastmath=True)(_bounce_kernel_source)
else:
    _bounce_kernel = _bounce_kernel_source


class BounceUnit(Unit):
    """A dropped mallet: the excitation drum rolls are made of.

    'drop' is the gesture: rising from zero drops the mallet from that
    height, and everything after is gravity -- bounces accelerate and
    weaken geometrically until the buzz, exactly as a dropped stick
    does on a drum head. Patch an LFO and every cycle is a stroke;
    patch a hand's height and lowering it to the surface IS the roll.

    'gravity' is the first fall's time, 'bounce' the restitution,
    'press' the player leaning into the roll -- faster returns, deader
    rebound, sooner buzz -- and 'hardness' the contact time of each
    strike. The output is a train of half-cosine force pulses sized by
    impact speed: feed it to drum~'s or modal~'s excite input.
    """

    # Unit area is the right convention for a strike -- a drum rings by
    # the momentum it is given, so hardness colors a blow rather than
    # weighting it -- and area stays constant across hardness, which is
    # the part that matters.
    #
    # The LEVEL was an order of magnitude under everything else that
    # drives a resonator. At each unit's own full gesture: bow~ peaks at
    # 0.42, noise~ at 0.68, rub~ at 1.39, and this at 0.039. It only
    # sounded right into drum~ because drum~ was impulse-normalized and
    # handed the missing thirty decibels back -- which is the same fact
    # that made bowing a drum explode. With both banks normalized alike
    # this can be what it should have been: a dropped mallet in the same
    # company as a bow.
    STRIKE_LEVEL = 10.0

    _seeded = 0

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.drop_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.gravity_in = self.new_inlet(base=0.35, minimum=0.02,
                                         maximum=2.0)
        self.bounce_in = self.new_inlet(base=0.75, minimum=0.0,
                                        maximum=0.99)
        self.press_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.hardness_in = self.new_inlet(base=0.6, minimum=0.0,
                                          maximum=1.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        BounceUnit._seeded += 1
        seed = (BounceUnit._seeded * 0x9E3779B97F4A7C15) % (1 << 64)
        self._rng = np.uint64(seed if seed else 0x853C49E6748FEA9B)
        self._h = 0.0
        self._v = 0.0
        self._armed = 1.0
        self._pulse_amp = 0.0
        self._pulse_at = 0.0
        self._pulse_len = 0.0
        self._quiet = True

        self.out = self.new_outlet()
        self._drop = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self._h = 0.0
        self._v = 0.0
        self._armed = 1.0
        self._pulse_amp = 0.0
        self._pulse_at = 0.0
        self._pulse_len = 0.0
        self._quiet = True

    def render(self, frames):
        drop = self.drop_in.eval(frames)
        gravity = self.gravity_in.eval(frames)
        bounce = self.bounce_in.eval(frames)
        press = self.press_in.eval(frames)
        hardness = self.hardness_in.eval(frames)
        out_level = self.level_in.eval(frames)

        out = self.out
        if not _svf_ready.is_set():
            out.set_constant(0.0)
            return

        gesture = self._drop[:frames]
        if drop.constant:
            gesture[:] = drop.value
            idle = drop.value <= 1.0e-4
        else:
            np.copyto(gesture, drop.data[:frames])
            idle = False
        np.clip(gesture, 0.0, 1.0, out=gesture)

        if (self._quiet and idle and self._armed > 0.5
                and self._h <= 0.0 and self._v == 0.0):
            out.set_constant(0.0)
            return

        def scalar(signal, lo, hi):
            value = signal.value if signal.constant else float(signal.data[0])
            return min(hi, max(lo, value))

        fall = scalar(gravity, 0.02, 2.0)
        rest = scalar(bounce, 0.0, 0.99)
        press_now = scalar(press, 0.0, 1.0)
        hard = scalar(hardness, 0.0, 1.0)

        # Fall time in seconds from a full-height drop sets gravity;
        # pressing adds the player's weight to it.
        n_fall = fall * self.sample_rate
        g = 2.0 / (n_fall * n_fall) * (1.0 + 3.0 * press_now)
        press_kill = 1.0 - 0.7 * press_now
        # Contact: 8 ms of felt down to a third of a millisecond, the
        # mallet family's whole range, as everywhere in the rack.
        contact = max(4.0, 0.008 * (0.04 ** hard) * self.sample_rate)
        # Impact speed from a full drop is sqrt(2 g h): normalize so a
        # full-height drop strikes with unit AREA -- force integrates
        # over the contact, and the drum downstream rings by momentum,
        # so hardness changes the color of a strike and not its weight.
        full_speed = math.sqrt(2.0 * g)
        strike_scale = (BounceUnit.STRIKE_LEVEL / full_speed) * (2.0 / contact)
        # At rest when the next flight would be shorter than the
        # contact itself: flight = 2 v / g samples.
        vmin = 0.5 * g * contact

        result = self._y[:frames]
        (self._h, self._v, self._armed, self._pulse_amp, self._pulse_at,
         self._pulse_len, rng_state) = _bounce_kernel(
            gesture, g, rest, press_kill, contact, strike_scale, vmin,
            self._h, self._v, self._armed, self._pulse_amp,
            self._pulse_at, self._pulse_len, self._rng, result)
        self._rng = np.uint64(rng_state)

        self._apply_level(result, out_level, frames)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        scratch = self._scratch[:frames]
        np.abs(result, out=scratch)
        self._quiet = bool(scratch.max() < 1.0e-5)


def _drum_kernel_source(exc, pulse, b1_base, b1_slope, b2, gains,
                        strike_gains, s1, s2,
                        bend_amt, bend_k, env_a, env_r, gap, snare_gain,
                        hp_k, wire_g, wire_b1, wire_b2,
                        dc_pole, env, tune_dev, hp, wy1, wy2,
                        dc_x, dc_y, rng, out):
    """A membrane, with the two nonlinearities that make it a drum.

    Tension modulation: displacement stiffens the head, so a hard hit
    lands pitched sharp and glides down as the ring decays. The bank's
    own envelope drives a smoothed frequency deviation, applied as a
    first-order retune per sample -- cheap, and smooth enough that the
    bend is a glide rather than a zipper.

    Snares: bright noise gated by the ring's own envelope through a
    soft valve, with a mild metallic formant from the wires' resonance
    laid over it -- smooth in time, wires in spectrum. They rattle
    while the drum speaks and die with it, because the drum is what is
    shaking them. No timer anywhere.
    """
    modes = b1_base.shape[0]
    for i in range(exc.shape[0]):
        drive = exc[i]
        tap = pulse[i]
        total = 0.0
        for m in range(modes):
            b1m = b1_base[m] + b1_slope[m] * tune_dev
            # The recursive part alone is what the head is DOING. Adding
            # the drive to the output as well gives every mode a path
            # around its own filter, and at lag zero those add
            # coherently across the bank while the rings they excite
            # dephase within a sample -- the same leak modal~ had. A
            # drum does not radiate the mallet.
            rung = b1m * s1[m] + b2[m] * s2[m]
            y = gains[m] * drive + strike_gains[m] * tap + rung
            if y > 1.5:
                y = 1.5 + np.tanh(y - 1.5)
            elif y < -1.5:
                y = -1.5 - np.tanh(-y - 1.5)
            s2[m] = s1[m]
            s1[m] = y
            total += rung
        a = total if total >= 0.0 else -total
        if a > env:
            env += (a - env) * env_a
        else:
            env += (a - env) * env_r
        target = bend_amt * env * env
        if target > 0.5:
            target = 0.5
        tune_dev += (target - tune_dev) * bend_k
        rattle = 0.0
        if snare_gain > 0.0:
            rng, nz = _rand01(rng)
            noise = 2.0 * nz - 1.0
            hp += (noise - hp) * hp_k
            bright = noise - hp
            wy = wire_g * bright + wire_b1 * wy1 + wire_b2 * wy2
            if wy > 1.5:
                wy = 1.5 + np.tanh(wy - 1.5)
            elif wy < -1.5:
                wy = -1.5 - np.tanh(-wy - 1.5)
            wy2 = wy1
            wy1 = wy
            lift = env - gap
            if lift > 0.0:
                rattle = snare_gain * (lift / (lift + 0.15)) \
                    * (bright + 0.8 * wy)
        o = total + rattle
        od = o - dc_x + dc_pole * dc_y
        dc_x = o
        dc_y = od
        out[i] = od
    return env, tune_dev, hp, wy1, wy2, dc_x, dc_y, rng


if _HAVE_NUMBA:
    _drum_kernel = njit(cache=True, fastmath=True)(_drum_kernel_source)
else:
    _drum_kernel = _drum_kernel_source


class DrumUnit(Unit):
    """A drum: modal~'s membrane plus what modal~ cannot do.

    Two nonlinearities separate a drum from a bank of modes. The head
    stiffens as it moves, so a hard hit lands pitched sharp and bends
    down through its ring -- 'tension' is how much, and tabla and toms
    live on it. And 'snares' are wires shaken BY the head: their
    rattle rides the ring's own envelope through a soft valve, so they
    speak when the drum speaks and die when it dies.

    'hit' is the mallet: a trigger whose height is velocity, shaped by
    'hardness' into a raised-cosine contact. 'excite in' hears any
    audio -- bounce~ belongs here, and a roll is bounce~ pressed into
    the head. The mode table is drawn in the same editor as modal~;
    membrane and tabla tables are the natural starting points.
    """

    MAX_MODES = 24

    _seeded = 0

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.excite_in = self.new_inlet()
        # How keenly the head hears what is patched in. Unity by default,
        # so a drum that never had this control sounds exactly as it did.
        self.sensitivity_in = self.new_inlet(base=1.0, minimum=0.0,
                                             maximum=8.0)
        self.trigger_in = self.new_inlet()
        self.frequency_in = self.new_inlet(base=120.0, minimum=20.0)
        self.pitch_in = self.new_inlet()
        self.decay_in = self.new_inlet(base=0.5, minimum=0.01, maximum=30.0)
        self.hardness_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.position_in = self.new_inlet(base=0.35, minimum=0.0,
                                          maximum=1.0)
        self.tension_in = self.new_inlet(base=0.3, minimum=0.0, maximum=1.0)
        self.snares_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        DrumUnit._seeded += 1
        seed = (DrumUnit._seeded * 0x9E3779B97F4A7C15) % (1 << 64)
        self._rng = np.uint64(seed if seed else 0x853C49E6748FEA9B)

        self._modes = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        self._weight_norm = 1.0
        self._s1 = np.zeros(DrumUnit.MAX_MODES, dtype=np.float64)
        self._s2 = np.zeros(DrumUnit.MAX_MODES, dtype=np.float64)
        self._b1 = np.zeros(DrumUnit.MAX_MODES, dtype=np.float64)
        self._slope = np.zeros(DrumUnit.MAX_MODES, dtype=np.float64)
        self._b2 = np.zeros(DrumUnit.MAX_MODES, dtype=np.float64)
        self._gains = np.zeros(DrumUnit.MAX_MODES, dtype=np.float64)
        self._gains_live = np.zeros(DrumUnit.MAX_MODES, dtype=np.float64)
        self._live_count = 0
        self._fm = np.zeros(DrumUnit.MAX_MODES, dtype=np.float64)
        self._theta = np.zeros(DrumUnit.MAX_MODES, dtype=np.float64)
        self._radius = np.zeros(DrumUnit.MAX_MODES, dtype=np.float64)
        self._mode_scratch = np.zeros(DrumUnit.MAX_MODES, dtype=np.float64)

        self.threshold = 0.05
        self._trigger_armed = True
        self._fire_requests = 0
        self._fire_served = 0
        self._pulse_amp = 0.0
        self._pulse_at = 0
        self._pulse_remaining = 0
        self._pulse_length = 1.0
        self._coef_key = None

        self._env = 0.0
        self._tune_dev = 0.0
        self._hp = 0.0
        self._wy1 = 0.0
        self._wy2 = 0.0
        self._dc_x = 0.0
        self._dc_y = 0.0
        self._quiet = True

        self.out = self.new_outlet()
        self._exc = np.zeros(MAX_BLOCK, dtype=np.float64)
        # The mallet gets its own buffer. It used to be mixed into the
        # excitation and so shared its gain, which is what stopped the
        # two being normalized differently.
        self._pulse_buf = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._drive_gains = np.zeros(DrumUnit.MAX_MODES, dtype=np.float64)
        self._sense_glide = 1.0
        self._sense_ramp = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def set_modes(self, table):
        rows = [row for row in table[:DrumUnit.MAX_MODES]]
        if not rows:
            rows = [(1.0, 1.0, 1.0)]
        fresh = np.array(rows, dtype=np.float64)
        resized = fresh.shape[0] != self._modes.shape[0]
        self._modes = fresh
        self._weight_norm = max(1.0, float(np.sum(np.abs(fresh[:, 1]))))
        if resized:
            self._s1[:] = 0.0
            self._s2[:] = 0.0

    def bypass_pairs(self):
        return ((self.excite_in, self.out),)

    def fire(self):
        self._fire_requests += 1

    def reset(self):
        self._s1[:] = 0.0
        self._s2[:] = 0.0
        self._env = 0.0
        self._tune_dev = 0.0
        self._hp = 0.0
        self._wy1 = 0.0
        self._wy2 = 0.0
        self._dc_x = 0.0
        self._dc_y = 0.0
        self._pulse_remaining = 0
        self._quiet = True

    def deactivate(self):
        self.reset()

    def _add_pulse(self, exc, start, stop):
        remaining = self._pulse_remaining
        if remaining <= 0 or stop <= start:
            return
        count = min(stop - start, remaining)
        window = self._scratch[:count]
        np.add(_INDEX_RAMP[:count], float(self._pulse_at - 1), out=window)
        window *= 2.0 * math.pi / self._pulse_length
        np.cos(window, out=window)
        np.subtract(1.0, window, out=window)
        window *= 0.5 * self._pulse_amp
        exc[start:start + count] += window
        self._pulse_at += count
        self._pulse_remaining = remaining - count

    def render(self, frames):
        signal = self.excite_in.eval(frames)
        trigger = self.trigger_in.eval(frames)
        frequency = self.frequency_in.eval(frames)
        pitch = self.pitch_in.eval(frames)
        decay = self.decay_in.eval(frames)
        hardness = self.hardness_in.eval(frames)
        position = self.position_in.eval(frames)
        tension = self.tension_in.eval(frames)
        snares = self.snares_in.eval(frames)
        out_level = self.level_in.eval(frames)

        out = self.out
        if not _svf_ready.is_set():
            out.set_constant(0.0)
            return

        exc = self._exc[:frames]
        # Sensitivity scales what ARRIVES, applied before the unit's own
        # mallet is written into the same buffer -- turning the inlet down
        # must not soften a strike the node makes itself. A knob move
        # glides across the block rather than stepping at its edge: one
        # factor per block is a staircase, and a staircase on a sustained
        # excitation is a zipper.
        sense = self.sensitivity_in.eval(frames)
        if signal.constant:
            exc[:] = signal.value
            silent_input = signal.value == 0.0
        else:
            np.copyto(exc, signal.data[:frames])
            silent_input = False
        if sense.constant:
            target = min(8.0, max(0.0, sense.value))
            start = self._sense_glide
            landing = start + (target - start) * 0.35
            self._sense_glide = landing
            if start == landing:
                if landing != 1.0:
                    exc *= landing
                silent_input = silent_input or landing == 0.0
            else:
                ramp = self._sense_ramp[:frames]
                np.multiply(_INDEX_RAMP[:frames],
                            (landing - start) / frames, out=ramp)
                ramp += start
                exc *= ramp
                silent_input = False
        else:
            np.clip(sense.data[:frames], 0.0, 8.0,
                    out=self._sense_ramp[:frames])
            exc *= self._sense_ramp[:frames]
            self._sense_glide = float(self._sense_ramp[frames - 1])
            silent_input = False

        events, self._trigger_armed = _excitation_events(
            trigger, frames, self.threshold, self._trigger_armed)
        if self._fire_requests != self._fire_served:
            self._fire_served = self._fire_requests
            events = ((0, 1.0),) + events

        if (self._quiet and not events and silent_input
                and self._pulse_remaining <= 0):
            out.set_constant(0.0)
            return

        def scalar(signal_, lo, hi):
            value = signal_.value if signal_.constant \
                else float(signal_.data[0])
            return min(hi, max(lo, value))

        hard = scalar(hardness, 0.0, 1.0)
        pos = scalar(position, 0.0, 1.0)
        tension_now = scalar(tension, 0.0, 1.0)
        snares_now = scalar(snares, 0.0, 1.0)
        seconds = scalar(decay, 0.01, 30.0)

        # The mallet: raised-cosine contact, 8 ms of felt to a third
        # of a millisecond of stick.
        width = max(2.0, 0.008 * (0.04 ** hard) * self.sample_rate)
        pulse = self._pulse_buf[:frames]
        pulse[:] = 0.0
        offset = 0
        for when, height in events:
            self._add_pulse(pulse, offset, when)
            # Area-normalized, as modal~'s mallet: force integrates over
            # the dwell, so a soft strike must not land fifty times
            # harder than a hard one of the same velocity.
            self._pulse_amp = min(2.0, height) * 2.0 / width
            self._pulse_length = width
            self._pulse_at = 0
            self._pulse_remaining = int(width)
            offset = when
        self._add_pulse(pulse, offset, frames)

        curve = self._scratch[:frames]
        self._build_hertz(curve, frequency, pitch, frames, 20.0)
        f0 = float(curve[0])

        modes = self._modes
        count = modes.shape[0]
        ratios = modes[:, 0]
        weights = modes[:, 1]
        decay_scale = modes[:, 2]

        # The geometry holds until a knob moves: most blocks, on most
        # drums, nothing here changes, and this python-side plumbing
        # was costing more than the kernel.
        coef_key = (count, f0, seconds, pos, id(self._modes))
        theta = self._theta[:count]
        radius = self._radius[:count]
        b1 = self._b1[:count]
        b2 = self._b2[:count]
        gains = self._gains[:count]
        if coef_key != self._coef_key:
            self._coef_key = coef_key
            fm = self._fm[:count]
            np.multiply(ratios, f0, out=fm)
            limit = self.sample_rate * 0.45
            np.clip(fm, 1.0, limit, out=theta)
            theta *= 2.0 * math.pi / self.sample_rate
            np.multiply(decay_scale, seconds * self.sample_rate,
                        out=radius)
            np.clip(radius, 1.0, None, out=radius)
            np.divide(-6.907755, radius, out=radius)
            np.exp(radius, out=radius)
            np.cos(theta, out=b1)
            b1 *= radius
            b1 *= 2.0
            # The tension retune, linearized: d b1 / d tune at the
            # table's tuning, applied per sample against the smoothed
            # deviation.
            slope = self._slope[:count]
            np.sin(theta, out=slope)
            slope *= theta
            slope *= radius
            slope *= -2.0
            np.multiply(radius, radius, out=b2)
            np.negative(b2, out=b2)
            # Impulse-normalized, the mallet convention: sin(theta)
            # cancels the two-pole's 1/sin(theta) impulse-response
            # peak, so a unit-area strike rings each mode up to its
            # table weight wherever it sits in frequency. The excite
            # input is a strike train (bounce~), not a bow -- modal~
            # owns the bowing normalization.
            np.sin(theta, out=gains)
            gains *= weights
            gains /= self._weight_norm
            alive = self._mode_scratch[:count]
            np.less_equal(fm, limit, out=alive, casting='unsafe')
            gains *= alive
            if pos > 0.0:
                pattern = self._mode_scratch[:count]
                np.multiply(_INDEX_RAMP[:count], math.pi * pos,
                            out=pattern)
                np.sin(pattern, out=pattern)
                np.abs(pattern, out=pattern)
                blend = min(1.0, pos / 0.05)
                if blend < 1.0:
                    pattern *= blend
                    pattern += 1.0 - blend
                gains *= pattern
        live = self._gains_live[:count]
        if count != self._live_count:
            np.copyto(live, gains)
            self._live_count = count
        else:
            step = self._mode_scratch[:count]
            np.subtract(gains, live, out=step)
            step *= 0.35
            live += step

        # Two normalizations, because the two inputs mean two different
        # things -- the arrangement modal~ already had and this did not.
        #
        # 'live' is impulse-normalized, the mallet convention: a strike
        # of unit area rings each mode up to its table weight wherever
        # it sits. That is right for the trigger and wrong for anything
        # sustained, because a drive parked on a resonance is then
        # multiplied by the mode's Q -- which is why bowing a drum came
        # out thirty-two decibels over bowing modal~ and clipped. The
        # excite path gets the same sqrt(1-r) compromise modal~ uses, so
        # the SAME signal into either bank now arrives at the same
        # loudness. A stick on a bass string and a stick on a drum
        # should be about as loud as each other.
        drive_live = self._drive_gains[:count]
        np.subtract(1.0, radius, out=drive_live)
        np.sqrt(drive_live, out=drive_live)
        drive_live *= live

        env_a = 1.0 - math.exp(-1.0 / (0.001 * self.sample_rate))
        env_r = 1.0 - math.exp(-1.0 / (0.04 * self.sample_rate))
        bend_k = 1.0 - math.exp(-1.0 / (0.002 * self.sample_rate))
        # Referenced to the honest ring level of a full hit (env near
        # a half): up to a third sharp at full tension, the tabla and
        # rototom register, quadratic in the hit below it.
        bend_amt = tension_now * 2.5
        # The wires: chatter at a few thousand contacts a second at a
        # full ring, each grain rung through the wires' own metallic
        # band, up around 1.7 kHz with a snappy few-millisecond decay.
        # Wires: mostly the bright noise itself (smooth in time), with
        # a broad metallic formant laid over it. Chatter was tried and
        # heard as gravel; the truth of a snare buzz is closer to
        # colored noise than to countable grains.
        hp_k = 1.0 - math.exp(-2.0 * math.pi * 1200.0 / self.sample_rate)
        theta_w = 2.0 * math.pi * min(2200.0, 0.4 * self.sample_rate) \
            / self.sample_rate
        r_w = 0.9
        wire_b1 = 2.0 * r_w * math.cos(theta_w)
        wire_b2 = -r_w * r_w
        wire_g = 1.0 - r_w
        snare_gain = snares_now * 0.5
        corner = min(40.0, max(1.0, f0 * 0.25))
        dc_pole = math.exp(-2.0 * math.pi * corner / self.sample_rate)

        result = self._y[:frames]
        (self._env, self._tune_dev, self._hp, self._wy1, self._wy2,
         self._dc_x, self._dc_y, rng_state) = _drum_kernel(
            exc, pulse, b1, self._slope[:count], b2, drive_live, live,
            self._s1[:count], self._s2[:count],
            bend_amt, bend_k, env_a, env_r, 0.015, snare_gain,
            hp_k, wire_g, wire_b1, wire_b2,
            dc_pole, self._env, self._tune_dev, self._hp,
            self._wy1, self._wy2,
            self._dc_x, self._dc_y, self._rng, result)
        self._rng = np.uint64(rng_state)

        self._apply_level(result, out_level, frames)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        scratch = self._scratch[:frames]
        np.abs(result, out=scratch)
        self._quiet = bool(scratch.max() < 1.0e-5)


def _motor_kernel_source(speed, rate_norm, parts_a, parts_b, xfade,
                         w_frac, throb,
                         offsets, jit_amt, amp_lo, amp_hi,
                         grind_gain, grind_k, bp_ratio, bdec,
                         bw_th, bw_r, bw_g, p_squeal, sq_lo, sq_span,
                         beat, slip, flut_k, body_mix,
                         bb1a, bb2a, bga, bb1b, bb2b, bgb,
                         smooth_k, phase, prev_ka, prev_kb,
                         str_a, str_b, vel,
                         g_lp, bphase, benv, bw1, bw2, bw_w, bw_sw,
                         sq_ph, sq_w, sq_wr, sq_amp, sq_dec,
                         bt_phase, flut,
                         ya1, ya2, yb1, yb2, dc_pole, dc_x, dc_y,
                         rng, out):
    """Rotating machinery, sample by sample: speed and load as streams.

    The rotation phase advances with the (smoothed) speed; each of the
    'parts' firings per revolution is a raised-cosine pulse whose width
    is the tone -- narrow knocks to smooth hum. Every part carries a
    fixed strength offset (throb spreads them), so uneven firing beats
    at once per revolution: the idle lope of a real engine, not an LFO.
    Load adds per-firing jitter, and the grind underneath is BAD
    BEARINGS, not a noise floor: defects strike at a non-integer
    multiple of the rotation (so the rattle beats against the firing
    pattern, as real bearing faults do), each impact rung at its own
    drawn pitch in the bearing's metallic band and SWEEPING as it
    rings, harder under load, faster with speed. And sometimes an
    impact sticks and SINGS: a swept whine of thirty to a hundred
    milliseconds, likelier under load -- the squeak of the wheel that
    needs the grease. A whisper of broadband scrub remains beneath.
    The housing is two fixed broad resonances -- housings do not track
    RPM -- and a DC blocker takes out the pulse train's offset.
    """
    n_parts = offsets.shape[0]
    for i in range(speed.shape[0]):
        vel += (speed[i] - vel) * smooth_k
        f = vel * rate_norm
        phase += f
        if phase >= 1.0:
            phase -= 1.0
        # Fractional parts: two firing patterns share the rotation
        # phase and crossfade, so the count GLIDES -- an engine caught
        # between natures, continuous at the integers by construction.
        fpa = phase * parts_a
        ka = int(fpa)
        if ka >= parts_a:
            ka = parts_a - 1
        fra = fpa - ka
        if ka != prev_ka:
            prev_ka = ka
            rng, ua = _rand01(rng)
            str_a = (1.0 + throb * offsets[ka % n_parts]) \
                * (1.0 + jit_amt * (2.0 * ua - 1.0))
        pa = 0.0
        if fra < w_frac:
            pa = 0.5 * (1.0 - math.cos(6.283185307179586 * fra / w_frac))
        fpb = phase * parts_b
        kb = int(fpb)
        if kb >= parts_b:
            kb = parts_b - 1
        frb = fpb - kb
        if kb != prev_kb:
            prev_kb = kb
            rng, ub2 = _rand01(rng)
            str_b = (1.0 + throb * offsets[kb % n_parts]) \
                * (1.0 + jit_amt * (2.0 * ub2 - 1.0))
        pb = 0.0
        if frb < w_frac:
            pb = 0.5 * (1.0 - math.cos(6.283185307179586 * frb / w_frac))
        p = (1.0 - xfade) * pa * str_a + xfade * pb * str_b
        amp = 0.55 * vel ** 0.6 if vel > 0.0 else 0.0
        tone_sig = p * amp * (amp_lo + amp_hi)
        rng, nz = _rand01(rng)
        coarse = 2.0 * nz - 1.0
        coarse = coarse * coarse * coarse
        g_lp += (coarse - g_lp) * grind_k
        # The bearing: defect impacts at bp_ratio times the rotation,
        # each rung at its own pitch and sweeping while it rings.
        bphase += f * bp_ratio
        if bphase >= 1.0:
            bphase -= 1.0
            rng, ub = _rand01(rng)
            benv += 0.4 + 0.9 * ub
            if benv > 3.0:
                benv = 3.0
            rng, ud = _rand01(rng)
            bw_w = bw_th * (0.75 + 0.5 * ud)
            rng, ud2 = _rand01(rng)
            bw_sw = bw_w * 0.2 * (ud2 - 0.35) * (1.0 - bdec)
            rng, us = _rand01(rng)
            if us < p_squeal:
                rng, ua = _rand01(rng)
                sq_amp = 0.5 + 0.5 * ua
                rng, uda = _rand01(rng)
                dur = sq_lo + sq_span * uda
                sq_dec = math.exp(-1.0 / dur)
                rng, uw = _rand01(rng)
                sq_w = bw_th * (0.45 + 0.3 * uw)
                sq_wr = sq_w * 0.2 / dur
                sq_ph = 0.0
        benv *= bdec
        bw_w += bw_sw
        if bw_w < bw_th * 0.4:
            bw_w = bw_th * 0.4
        elif bw_w > bw_th * 1.7:
            bw_w = bw_th * 1.7
        b1w = 2.0 * bw_r * math.cos(bw_w)
        rng, nb = _rand01(rng)
        bhit = benv * (2.0 * nb - 1.0)
        bw = bw_g * bhit + b1w * bw1 - bw_r * bw_r * bw2
        bw2 = bw1
        bw1 = bw
        # Roughness for the squeal: slow noise that flutters both its
        # loudness and its pitch. A pure swept sine is a bird or a
        # drip; a bearing's whine wavers and rasps.
        rng, nf = _rand01(rng)
        flut += ((2.0 * nf - 1.0) - flut) * flut_k
        sq_sig = 0.0
        if sq_amp > 1.0e-4:
            sq_w += sq_wr
            sq_ph += sq_w * (1.0 + 0.12 * flut)
            if sq_ph > 6.283185307179586:
                sq_ph -= 6.283185307179586
            sq_sig = sq_amp * (0.6 + 0.7 * flut) * math.sin(sq_ph)
            sq_amp *= sq_dec
        # The narrow band keeps little of the impact's energy, so the
        # ring is driven hard to sit audibly under the tone.
        # The machine is what shakes the bearing: the whole grind
        # stack pumps with the firing pulses, which is what welds it
        # INTO the engine instead of leaving it beside one.
        raw = tone_sig + grind_gain * amp * (0.45 + 0.75 * p) \
            * (14.0 * bw + 0.3 * g_lp + 2.2 * sq_sig) * 1.2
        # The slip beat: a second shaft a few percent behind, beating
        # at slip times rotation -- so the slow breathing of the
        # machine speeds up and slows down WITH it.
        bt_phase += f * slip
        if bt_phase >= 1.0:
            bt_phase -= 1.0
        raw *= 1.0 - beat * 0.85 \
            * (0.5 - 0.5 * math.cos(6.283185307179586 * bt_phase))
        ya = bga * raw + bb1a * ya1 + bb2a * ya2
        ya2 = ya1
        ya1 = ya
        yb = bgb * raw + bb1b * yb1 + bb2b * yb2
        yb2 = yb1
        yb1 = yb
        o = raw * (1.0 - 0.5 * body_mix) + body_mix * (ya + yb)
        od = o - dc_x + dc_pole * dc_y
        dc_x = o
        dc_y = od
        out[i] = od
    return (phase, prev_ka, prev_kb, str_a, str_b, vel, g_lp,
            bphase, benv, bw1, bw2,
            bw_w, bw_sw, sq_ph, sq_w, sq_wr, sq_amp, sq_dec,
            bt_phase, flut,
            ya1, ya2, yb1, yb2, dc_x, dc_y, rng)


if _HAVE_NUMBA:
    _motor_kernel = njit(cache=True, fastmath=True)(_motor_kernel_source)
else:
    _motor_kernel = _motor_kernel_source


class MotorUnit(Unit):
    """A machine: rotation as pitch, load as violence.

    The second unit after whoosh~ whose mapping IS the physics, and the
    first that wants two effort streams at once: 'speed' is rotation
    (pitch linear in it, loudness rising, stillness silent), 'load' is
    torque (each firing punchier and less regular, the grind of
    bearings rising underneath). Velocity into one, torque into the
    other, and a joint is an engine.

    'rate' is the full-speed rotation in Hz. 'parts' is firings per
    revolution -- one is a thumper, four an engine, eight a turbine's
    whine. 'tone' widens the firing pulse from knock to electric hum.
    'throb' spreads the parts' fixed strengths so the engine lopes at
    once per revolution, which is what an idle shudder is. 'housing'
    is a fixed pair of broad body resonances; for a specific machine,
    patch through modal~ or formant~ instead.
    """

    _seeded = 0

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.speed_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.5)
        self.load_in = self.new_inlet(base=0.3, minimum=0.0, maximum=1.0)
        self.rate_in = self.new_inlet(base=45.0, minimum=2.0, maximum=200.0)
        self.parts_in = self.new_inlet(base=4.0, minimum=1.0, maximum=12.0)
        self.tone_in = self.new_inlet(base=0.4, minimum=0.0, maximum=1.0)
        self.throb_in = self.new_inlet(base=0.35, minimum=0.0, maximum=1.0)
        self.beat_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.slip_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.grind_in = self.new_inlet(base=0.4, minimum=0.0, maximum=1.0)
        self.housing_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        MotorUnit._seeded += 1
        seed = (MotorUnit._seeded * 0x9E3779B97F4A7C15) % (1 << 64)
        self._rng = np.uint64(seed if seed else 0x853C49E6748FEA9B)
        # Each part's fixed unevenness: this motor's cylinders, every
        # run, drawn once from the instance seed -- and this motor's
        # own slip, so no two beat alike.
        spread_rng = np.random.RandomState(seed & 0xFFFFFFFF or 1)
        self._offsets = spread_rng.uniform(-0.5, 0.5, 12)
        self._slip_flavor = 0.85 + 0.3 * spread_rng.uniform()
        self._phase = 0.0
        self._prev_ka = -1
        self._prev_kb = -1
        self._str_a = 1.0
        self._str_b = 1.0
        self._vel = 0.0
        self._g_lp = 0.0
        self._bphase = 0.0
        self._benv = 0.0
        self._bw1 = 0.0
        self._bw2 = 0.0
        self._bw_w = 0.4
        self._bw_sw = 0.0
        self._sq_ph = 0.0
        self._sq_w = 0.0
        self._sq_wr = 0.0
        self._sq_amp = 0.0
        self._sq_dec = 0.99
        self._bt_phase = 0.0
        self._flut = 0.0
        self._ya1 = 0.0
        self._ya2 = 0.0
        self._yb1 = 0.0
        self._yb2 = 0.0
        self._dc_x = 0.0
        self._dc_y = 0.0
        self._quiet = True

        self.out = self.new_outlet()
        self._speed = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self._phase = 0.0
        self._prev_ka = -1
        self._prev_kb = -1
        self._str_a = 1.0
        self._str_b = 1.0
        self._vel = 0.0
        self._g_lp = 0.0
        self._bphase = 0.0
        self._benv = 0.0
        self._bw1 = 0.0
        self._bw2 = 0.0
        self._bw_w = 0.4
        self._bw_sw = 0.0
        self._sq_amp = 0.0
        self._bt_phase = 0.0
        self._flut = 0.0
        self._ya1 = 0.0
        self._ya2 = 0.0
        self._yb1 = 0.0
        self._yb2 = 0.0
        self._dc_x = 0.0
        self._dc_y = 0.0
        self._quiet = True

    def render(self, frames):
        speed = self.speed_in.eval(frames)
        load = self.load_in.eval(frames)
        rate = self.rate_in.eval(frames)
        parts = self.parts_in.eval(frames)
        tone = self.tone_in.eval(frames)
        throb = self.throb_in.eval(frames)
        beat = self.beat_in.eval(frames)
        slip = self.slip_in.eval(frames)
        grind = self.grind_in.eval(frames)
        housing = self.housing_in.eval(frames)
        out_level = self.level_in.eval(frames)

        out = self.out
        if not _svf_ready.is_set():
            out.set_constant(0.0)
            return

        drive = self._speed[:frames]
        if speed.constant:
            drive[:] = speed.value
            idle = abs(speed.value) < 1.0e-4
        else:
            np.copyto(drive, speed.data[:frames])
            idle = False
        np.clip(drive, 0.0, 1.5, out=drive)

        if self._quiet and idle and self._vel < 1.0e-4:
            out.set_constant(0.0)
            return

        def scalar(signal, lo, hi):
            value = signal.value if signal.constant else float(signal.data[0])
            return min(hi, max(lo, value))

        load_now = scalar(load, 0.0, 1.0)
        rate_now = scalar(rate, 2.0, 200.0)
        parts_f = scalar(parts, 1.0, 12.0)
        parts_a = int(parts_f)
        xfade = parts_f - parts_a
        parts_b = parts_a + 1
        if parts_b > 12:
            parts_b = 12
            xfade = 0.0
        tone_now = scalar(tone, 0.0, 1.0)
        throb_now = scalar(throb, 0.0, 1.0)
        beat_now = scalar(beat, 0.0, 1.0)
        # Half a percent to eight percent of the rotation,
        # exponentially, with this instance's own flavor on top: the
        # breath from geological to seasick, still never quite shared
        # between two motors.
        slip_now = (0.002 * 100.0 ** scalar(slip, 0.0, 1.0)
                    * self._slip_flavor)
        grind_now = scalar(grind, 0.0, 1.0)
        housing_now = scalar(housing, 0.0, 1.0)

        rate_norm = rate_now / self.sample_rate
        # Knock to hum: the firing pulse widens from a sixth of its
        # interval to all of it, where it fuses into a near-sine.
        w_frac = 0.16 + 0.84 * tone_now
        # Load is punch and irregularity at once, as torque is.
        jit_amt = 0.35 * load_now
        amp_lo = 0.35
        amp_hi = 0.65 * load_now
        grind_gain = grind_now * (0.15 + 0.85 * load_now) * 0.8
        grind_cut = 400.0 + 2600.0 * min(1.0, float(drive[0]))
        grind_k = 1.0 - math.exp(-2.0 * math.pi * grind_cut
                                 / self.sample_rate)
        # The bearing's numbers: a defect-pass ratio that never lines
        # up with the firing pattern, a millisecond-and-a-half of ring
        # per impact, and a metallic band near 2.8 kHz at unity peak.
        bp_ratio = 6.71
        bdec = math.exp(-1.0 / (0.0015 * self.sample_rate))
        # Down out of the songbird register: machinery moans near two
        # kilohertz, it does not tweet at three.
        th_w = 2.0 * math.pi * min(1900.0, 0.4 * self.sample_rate) \
            / self.sample_rate
        r_w = 0.97
        bw_g = (1.0 - r_w) * math.sin(th_w) * 1.2
        # The squeak of the wheel: likelier under load, thirty to a
        # hundred milliseconds, rising as stick-slip squeals do.
        p_squeal = 0.0015 + 0.007 * load_now
        sq_lo = 0.03 * self.sample_rate
        sq_span = 0.07 * self.sample_rate
        flut_k = 1.0 - math.exp(-2.0 * math.pi * 90.0 / self.sample_rate)
        smooth_k = 1.0 - math.exp(-1.0 / (0.004 * self.sample_rate))
        # The housing: two fixed broad resonances. Housings do not
        # track RPM, which is why climbing through them reads as real.
        r_a = 0.985
        th_a = 2.0 * math.pi * 150.0 / self.sample_rate
        r_b = 0.98
        th_b = 2.0 * math.pi * 420.0 / self.sample_rate
        bb1a = 2.0 * r_a * math.cos(th_a)
        bb2a = -r_a * r_a
        # (1-r)*sin(theta): unity at the peak. The third time this
        # normalization has been learned; may it be the last.
        bga = (1.0 - r_a) * math.sin(th_a) * 1.2
        bb1b = 2.0 * r_b * math.cos(th_b)
        bb2b = -r_b * r_b
        bgb = (1.0 - r_b) * math.sin(th_b) * 1.2
        corner = min(30.0, max(2.0, rate_now * 0.3))
        dc_pole = math.exp(-2.0 * math.pi * corner / self.sample_rate)

        result = self._y[:frames]
        (self._phase, self._prev_ka, self._prev_kb,
         self._str_a, self._str_b, self._vel, self._g_lp,
         self._bphase, self._benv, self._bw1, self._bw2,
         self._bw_w, self._bw_sw, self._sq_ph, self._sq_w,
         self._sq_wr, self._sq_amp, self._sq_dec,
         self._bt_phase, self._flut,
         self._ya1, self._ya2, self._yb1, self._yb2,
         self._dc_x, self._dc_y, rng_state) = _motor_kernel(
            drive, rate_norm, parts_a, parts_b, xfade, w_frac, throb_now,
            self._offsets, jit_amt, amp_lo, amp_hi,
            grind_gain, grind_k, bp_ratio, bdec,
            th_w, r_w, bw_g, p_squeal, sq_lo, sq_span,
            beat_now, slip_now, flut_k, housing_now,
            bb1a, bb2a, bga, bb1b, bb2b, bgb,
            smooth_k, self._phase, self._prev_ka, self._prev_kb,
            self._str_a, self._str_b,
            self._vel, self._g_lp,
            self._bphase, self._benv, self._bw1, self._bw2,
            self._bw_w, self._bw_sw, self._sq_ph, self._sq_w,
            self._sq_wr, self._sq_amp, self._sq_dec,
            self._bt_phase, self._flut,
            self._ya1, self._ya2,
            self._yb1, self._yb2, dc_pole, self._dc_x, self._dc_y,
            self._rng, result)
        self._rng = np.uint64(rng_state)

        self._apply_level(result, out_level, frames)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        scratch = self._scratch[:frames]
        np.abs(result, out=scratch)
        self._quiet = bool(scratch.max() < 1.0e-5)


def _bubbles_kernel_source(flow, rate_norm, f_center, spread_oct,
                           chirp_amt, gulp, bloom_frac, amp_base, reg,
                           ring_mult, atk_dec, smooth_k,
                           srate, ph, w, wr, amp, dec, atk1,
                           ph2, w2, wr2, amp2, dec2, atk2, vel,
                           acc, nxt, rng, out):
    """Water, one bubble at a time: the Minnaert chorus.

    Each bubble is a decaying sine at the frequency its size dictates,
    and its pitch RISES as it dies -- the rate of rise tied to the
    decay rate, so small quick bubbles chirp fast and big glugs bend
    slowly. That inflection is what makes water sound like water; at
    chirp 0 the same voices are the pure pings of bubbles deep under
    the surface.

    Arrivals ride the (smoothed) flow through an accumulator whose
    fill threshold is drawn between exponential (a Poisson stream, the
    boil) and constant (metronomic, the glug-glug-glug of a dumped
    bottle) -- 'regular' is that draw, at identical mean rate all the
    way across, and nothing else touches the timing. 'gulp' is the
    onset twin of chirp: each birth carries a partial born well below
    the bubble that glides up into it and dies -- the bwup of a glug,
    fast rise, soft landing, time-scaled to the ring. Eight voices overlap, a new bubble
    taking the quietest slot: the gurgle is polyphony, as the
    tambourine taught.
    """
    voices = ph.shape[0]
    for i in range(flow.shape[0]):
        vel += (flow[i] - vel) * smooth_k
        acc += rate_norm * vel
        if acc >= nxt:
            acc -= nxt
            rng, ui = _rand01(rng)
            draw = -math.log(ui + 1.0e-12)
            nxt = reg + (1.0 - reg) * draw
            # The search starts at a random voice: a deterministic
            # weakest-slot rotation stamped an amplitude pattern with
            # period voices-times-interval on a dense stream -- heard
            # as periodic bursts that no water makes.
            rng, us = _rand01(rng)
            start = int(us * voices)
            if start >= voices:
                start = voices - 1
            weakest = start
            smallest = amp[start]
            for v in range(1, voices):
                idx = start + v
                if idx >= voices:
                    idx -= voices
                if amp[idx] < smallest:
                    smallest = amp[idx]
                    weakest = idx
            rng, u1 = _rand01(rng)
            rng, u2 = _rand01(rng)
            fr = f_center * 2.0 ** ((u1 + u2 - 1.0) * spread_oct)
            if fr > 0.4 * srate:
                fr = 0.4 * srate
            wv = 6.283185307179586 * fr / srate
            rng, u3 = _rand01(rng)
            cycles = (25.0 + 45.0 * u3) * ring_mult
            tau = cycles * srate / fr
            d = math.exp(-1.0 / tau)
            rng, u4 = _rand01(rng)
            # A vigorous stream makes stronger bubbles, which is what
            # keeps loudness rising with flow even once the pond is
            # full and masking caps the count.
            a0 = amp_base * (f_center / fr) ** 0.4 * (0.7 + 0.6 * u4) \
                * (0.55 + 0.6 * vel)
            if a0 > 0.8:
                a0 = 0.8
            # A full pond masks a new bubble: when even the quietest
            # voice still rings within an eighth of the newcomer's
            # strength, the newcomer is lost in the mass -- truncating
            # a live ring to make room was a click and a hole at once,
            # and at high rates a zipper of them.
            if smallest <= 0.15 * a0:
                ph[weakest] = 0.0
                w[weakest] = wv
                # The bubble's own onset is fast but not one sample:
                # under a millisecond of rise, which is what separates
                # a birth from a click.
                atk1[weakest] = 1.0
                # The rise: chirp times about half the frequency over
                # one ring time -- van den Doel's sigma in kernel terms.
                wr[weakest] = chirp_amt * 0.6 * wv / tau
                amp[weakest] = a0
                dec[weakest] = d
                # The glug: born well below the bubble and GLIDING up
                # into it -- fast rise, soft landing, a one-pole toward
                # the bubble's own (possibly chirping) frequency, gone
                # in a third of the ring. The bwup, time-scaled so fizz
                # flicks and glugs woop.
                ph2[weakest] = 0.0
                w2[weakest] = wv * 0.3
                tau2 = tau * 0.35
                wr2[weakest] = 1.0 - math.exp(-1.0 / (bloom_frac
                                                       * tau2))
                amp2[weakest] = a0 * gulp * 1.6
                dec2[weakest] = math.exp(-1.0 / tau2)
                # Resonance swells; only percussion starts loud. The
                # attack rides the same clock as the glide, so the
                # bloom peaks as the pitch lands.
                atk2[weakest] = 1.0
        s = 0.0
        for v in range(voices):
            if amp[v] > 1.0e-5:
                w[v] += wr[v]
                ph[v] += w[v]
                if ph[v] > 6.283185307179586:
                    ph[v] -= 6.283185307179586
                s += amp[v] * (1.0 - atk1[v]) * _fast_sin(ph[v])
                amp[v] *= dec[v]
                atk1[v] *= atk_dec
            if amp2[v] > 1.0e-5:
                w2[v] += (w[v] - w2[v]) * wr2[v]
                ph2[v] += w2[v]
                if ph2[v] > 6.283185307179586:
                    ph2[v] -= 6.283185307179586
                s += amp2[v] * (1.0 - atk2[v]) * _fast_sin(ph2[v])
                amp2[v] *= dec2[v]
                atk2[v] *= 1.0 - wr2[v]
        out[i] = s
    return vel, acc, nxt, rng


if _HAVE_NUMBA:
    _bubbles_kernel = njit(cache=True, fastmath=True)(_bubbles_kernel_source)
else:
    _bubbles_kernel = _bubbles_kernel_source


class BubblesUnit(Unit):
    """Liquid: bubbles as they actually sound, played by flow.

    'flow' is the whole interface -- arrival rate rides it, stillness
    is silence -- and each arrival is a bubble: a decaying sine at its
    Minnaert frequency whose pitch rises as it dies. 'size' places the
    band (fizz to glug), 'spread' widens the population, 'chirp' is
    the upward inflection (surface gurgle) down to pure submerged
    pings, 'gulp' clusters arrivals on the slow air-exchange cycle of
    a pouring bottle. Layer noise~ underneath for the splash.
    """

    VOICES = 8

    _seeded = 0

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.flow_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.5)
        self.size_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.spread_in = self.new_inlet(base=0.4, minimum=0.0, maximum=1.0)
        self.chirp_in = self.new_inlet(base=0.6, minimum=0.0, maximum=1.0)
        self.gulp_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.bloom_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.regular_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.decay_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.density_in = self.new_inlet(base=80.0, minimum=5.0,
                                         maximum=400.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        BubblesUnit._seeded += 1
        seed = (BubblesUnit._seeded * 0x9E3779B97F4A7C15) % (1 << 64)
        self._rng = np.uint64(seed if seed else 0x853C49E6748FEA9B)
        self._ph = np.zeros(BubblesUnit.VOICES, dtype=np.float64)
        self._w = np.zeros(BubblesUnit.VOICES, dtype=np.float64)
        self._wr = np.zeros(BubblesUnit.VOICES, dtype=np.float64)
        self._amp = np.zeros(BubblesUnit.VOICES, dtype=np.float64)
        self._dec = np.zeros(BubblesUnit.VOICES, dtype=np.float64)
        self._atk1 = np.zeros(BubblesUnit.VOICES, dtype=np.float64)
        self._ph2 = np.zeros(BubblesUnit.VOICES, dtype=np.float64)
        self._w2 = np.zeros(BubblesUnit.VOICES, dtype=np.float64)
        self._wr2 = np.zeros(BubblesUnit.VOICES, dtype=np.float64)
        self._amp2 = np.zeros(BubblesUnit.VOICES, dtype=np.float64)
        self._dec2 = np.zeros(BubblesUnit.VOICES, dtype=np.float64)
        self._atk2 = np.zeros(BubblesUnit.VOICES, dtype=np.float64)
        self._vel = 0.0
        self._acc = 0.0
        self._nxt = 1.0
        self._quiet = True

        self.out = self.new_outlet()
        self._flow = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self._ph[:] = 0.0
        self._w[:] = 0.0
        self._wr[:] = 0.0
        self._amp[:] = 0.0
        self._dec[:] = 0.0
        self._amp2[:] = 0.0
        self._vel = 0.0
        self._acc = 0.0
        self._nxt = 1.0
        self._quiet = True

    def render(self, frames):
        flow = self.flow_in.eval(frames)
        size = self.size_in.eval(frames)
        spread = self.spread_in.eval(frames)
        chirp = self.chirp_in.eval(frames)
        gulp = self.gulp_in.eval(frames)
        bloom = self.bloom_in.eval(frames)
        regular = self.regular_in.eval(frames)
        decay = self.decay_in.eval(frames)
        density = self.density_in.eval(frames)
        out_level = self.level_in.eval(frames)

        out = self.out
        if not _svf_ready.is_set():
            out.set_constant(0.0)
            return

        stream = self._flow[:frames]
        if flow.constant:
            stream[:] = flow.value
            idle = abs(flow.value) < 1.0e-4
        else:
            np.copyto(stream, flow.data[:frames])
            idle = False
        np.clip(stream, 0.0, 1.5, out=stream)

        if self._quiet and idle and self._vel < 1.0e-4:
            out.set_constant(0.0)
            return

        def scalar(signal, lo, hi):
            value = signal.value if signal.constant else float(signal.data[0])
            return min(hi, max(lo, value))

        size_now = scalar(size, 0.0, 1.0)
        spread_now = scalar(spread, 0.0, 1.0)
        chirp_now = scalar(chirp, 0.0, 1.0)
        gulp_now = scalar(gulp, 0.0, 1.0)
        bloom_now = scalar(bloom, 0.0, 1.0)
        reg_now = scalar(regular, 0.0, 1.0)
        decay_now = scalar(decay, 0.0, 1.0)
        dens = scalar(density, 5.0, 400.0)

        # Fizz at 5 kHz down to a 60 Hz glug: Minnaert, by the knob.
        f_center = 5000.0 * (0.012 ** size_now)
        spread_oct = spread_now * 2.5
        rate_norm = dens / self.sample_rate
        # Density is texture, not loudness, as with the shaker's beans.
        amp_base = 0.5 * math.sqrt(80.0 / max(20.0, dens))
        # A quarter of the physical ring up to four times it: dry drip
        # to droplet in a cave, exponentially about the truth.
        ring_mult = 16.0 ** (decay_now - 0.5)
        # The glug's shared glide-and-swell clock, as a fraction of its
        # life: a tenth (snappy blip) to four-fifths (lazy cavity),
        # with the old feel at the middle.
        bloom_frac = 0.1 * 8.0 ** bloom_now
        atk_dec = math.exp(-1.0 / (0.0007 * self.sample_rate))
        smooth_k = 1.0 - math.exp(-1.0 / (0.004 * self.sample_rate))

        result = self._y[:frames]
        (self._vel, self._acc, self._nxt,
         rng_state) = _bubbles_kernel(
            stream, rate_norm, f_center, spread_oct, chirp_now,
            gulp_now, bloom_frac, amp_base, reg_now, ring_mult,
            atk_dec, smooth_k,
            float(self.sample_rate),
            self._ph, self._w, self._wr, self._amp, self._dec, self._atk1,
            self._ph2, self._w2, self._wr2, self._amp2, self._dec2,
            self._atk2, self._vel, self._acc, self._nxt,
            self._rng, result)
        self._rng = np.uint64(rng_state)

        self._apply_level(result, out_level, frames)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        scratch = self._scratch[:frames]
        np.abs(result, out=scratch)
        self._quiet = bool(scratch.max() < 1.0e-5)


def _strain_kernel_source(strain, thresh, spread, alpha, size_cap,
                          habituate, grain_samples, amp_scale, chirp_a,
                          contact, sl, v50, vdg, pk, srate, vary_oct,
                          atk_k, mix, jgain, jit_k, macro_thresh, pop_amp,
                          gd2, grind_gain, vel_k, tex_k, tex_comp,
                          theta, radius, b1, b2, gains, ggains, ring,
                          s1, s2, ap_x, ap_y,
                          s_prev, stress, threshold, pulse, penv, gd,
                          stress2, threshold2, jit_lp, pulse2,
                          sv, tune, vel_env, g_lp,
                          pew_scale, pew_w, c_phase, c_time, c_amp, c_dec,
                          c_t0, max_strain, recover, rng, dry,
                          out_raw, out):
    """The strain engine: solids under stress, sample by sample.

    'strain' is not a control but the physical variable the model runs
    on -- a joint angle, a stretch, a load. Motion accumulates stress
    (displacement since the last release); stress against a randomized
    threshold releases an event; the event is a noise grain sized by
    what was released, dispersed through an allpass chain (flexural
    waves travel faster at high frequency, which is why lake ice
    chirps), and rung through the mode bank. No motion, no stress, no
    sound -- stillness is silent by construction, and a release rate
    that follows speed is what makes a creak legible as effort.

    The material remembers: strain beyond the old maximum releases at
    full strength, familiar territory at the habituated fraction, and
    the memory relaxes over tens of seconds. Paper is loud once; a
    hinge creaks every time; ice cracks and is done. One number per
    regime says which.
    """
    modes = b1.shape[0]
    stages = ap_x.shape[0]
    # Each release may come from a slightly different object -- a walk
    # crosses many boards, a crumple has many facets. 'tune' is this
    # event's member of the ensemble, and b1 is always derived from the
    # angle and the current radius together, never stored across a
    # radius change (the shaker taught that lesson).
    for m in range(modes):
        b1[m] = 2.0 * radius[m] * math.cos(theta[m] * tune)
    for i in range(strain.shape[0]):
        s = strain[i]
        ds = s - s_prev
        s_prev = s
        speed = abs(ds)
        stress += speed
        # The grind: continuous frictional shearing between the slips,
        # following how fast the strain is moving -- what breath is to
        # the winds, the scrub of surfaces is to a bend.
        vel_env += (speed - vel_env) * vel_k
        # Signed sliding velocity in strain-units per second: the hinge
        # face's actual motion, direction and all, for the friction loop.
        sv += (ds * srate - sv) * vel_k

        if stress > threshold:
            rng, u = _rand01(rng)
            if alpha > 0.0:
                size = (0.05 + u) ** (-alpha)
                if size > size_cap:
                    size = size_cap
            else:
                size = 0.5 + u
            novel = 1.0 if s > max_strain else habituate
            # A slip releases at most what static friction was holding:
            # sudden huge motion makes a loud event, not an unbounded one.
            overshoot = stress / thresh
            if overshoot > 3.0:
                overshoot = 3.0
            # A slip's duration is physical: the released distance over
            # the sliding speed. Sparse heavy slips (much stress, slow
            # slide) are long SCRAPES; dense light ones are short
            # grains; a sudden jump releases fast and stays a crack.
            # Fixed-length grains made every sparse event a knock --
            # nineteen wooden taps a second read as tap-dancing, not
            # creaking.
            span = size
            if span > 3.0:
                span = 3.0
            iv = 0.25 * threshold / (vel_env + 1.0e-12)
            lo = grain_samples * 0.4
            hi = grain_samples * 6.0
            if iv < lo:
                iv = lo
            elif iv > hi:
                iv = hi
            iv *= 0.4 + 0.8 * span
            gd = math.exp(-1.0 / iv)
            # Constant energy per released unit: a longer scrape is
            # proportionally quieter than a short knock.
            escale = 1.0
            if iv > grain_samples:
                escale = math.sqrt(grain_samples / iv)
            pulse += amp_scale * size * novel * overshoot * escale
            if pulse > 50.0:
                pulse = 50.0
            # Launch this event's pew: flexural dispersion arrives high
            # first and sweeps down as 1/t^2, longer the farther away the
            # fracture -- which is what 'chirp' scales, jittered so every
            # crack comes from somewhere else.
            if pew_scale > 0.0 and mix < 1.0:
                rng, u5 = _rand01(rng)
                tau = pew_scale * (0.6 + 0.8 * u5)
                c_dec = math.exp(-1.0 / tau)
                c_t0 = tau * 0.12
                c_time = 0.0
                # Micro events pew only as far as they still strike the
                # bank at all: with the squeal carrying them, nineteen
                # pews a second was the chirp knob 'going crazy'.
                c_amp = (amp_scale * size * novel * overshoot * 0.9
                         * (1.0 - mix))
                c_phase = 0.0
            stress = 0.0
            rng, u2 = _rand01(rng)
            threshold = thresh * (0.35 + spread * 1.3 * u2 * u2)

        # The second scale of release: most slips are micro (they live
        # in the squeal), but stress also accumulates toward the RARE
        # letting-go of the geometry itself -- the percussive pop a
        # real strain gives occasionally, not nineteen times a second.
        stress2 += speed
        if stress2 > threshold2:
            rng, up = _rand01(rng)
            # The pop has its own envelope, straight into the bank: it
            # is the geometry letting go, and no amount of squeal
            # routing should be able to swallow it. Its size shares the
            # motion's energy -- a release mid-heave is bigger than one
            # at the edge of stillness.
            h2 = sv if sv >= 0.0 else -sv
            punch = 0.5 + h2 / (h2 + 0.15)
            pa = pop_amp * (0.6 + 0.8 * up) * punch
            pulse2 += pa
            if pulse2 > 50.0:
                pulse2 = 50.0
            # And the release goes THROUGH the interface: the geometry
            # jumping is a spike of sliding velocity, so the squeal
            # yelps at the moment of the pop. This is the violence --
            # the pop and the squeal are the same event seen twice.
            rng, uk = _rand01(rng)
            kick = 0.6 * (0.8 + 0.7 * uk)
            if sv >= 0.0:
                sv += kick
            else:
                sv -= kick
            # A pop is the geometry actually shifting, so THIS is where
            # the ensemble retunes ('vary'). Retuning on every micro
            # slip stepped the pitch of a singing squeal nineteen times
            # a second, and each step was a click.
            if vary_oct > 0.0:
                rng, u4 = _rand01(rng)
                tune = 2.0 ** ((u4 - 0.5) * 2.0 * vary_oct)
                for m in range(modes):
                    b1[m] = 2.0 * radius[m] * math.cos(theta[m] * tune)
            if pew_scale > 0.0:
                rng, u6 = _rand01(rng)
                tau = pew_scale * (0.6 + 0.8 * u6)
                c_dec = math.exp(-1.0 / tau)
                c_t0 = tau * 0.12
                c_time = 0.0
                # Scaled to the sweep's length: a short dispersion is
                # a subtle zing on the pop, not a concentrated blast --
                # and the sweep is coherent, ringing the modes it
                # passes, so even the long ones need less to land.
                depth = tau / (0.08 * srate)
                if depth > 1.0:
                    depth = 1.0
                c_amp = pa * 0.28 * math.sqrt(depth)
                c_phase = 0.0
            stress2 = 0.0
            rng, u7 = _rand01(rng)
            threshold2 = macro_thresh * (0.4 + 1.2 * u7)

        if s > max_strain:
            max_strain = s
        else:
            max_strain += (s - max_strain) * recover

        pulse *= gd
        # A slip's release is not instantaneous: the envelope follows
        # its target with an attack lag scaled to the grain length, so
        # an event swells in over a fraction of its own life instead of
        # arriving as an edge (the sleighbells lesson, again).
        penv += (pulse - penv) * atk_k
        rng, nz = _rand01(rng)
        # The slip itself is broadband -- the TONE of a squeal is not in
        # the event but in the loop below, where the interface sings
        # against the bank.
        voice = 2.0 * nz - 1.0
        # Cubed noise is coarse where white noise is smooth: heavy-tailed,
        # gravelly, the texture of surfaces actually scrubbing.
        rng, gz = _rand01(rng)
        coarse = 2.0 * gz - 1.0
        coarse = coarse * coarse * coarse
        # Texture is where the grind sits spectrally: fine surfaces rub
        # dark, coarse ones bright. One pole, loudness-compensated, so
        # the knob moves the character and not the level.
        g_lp += (coarse - g_lp) * tex_k

        # Chirp is an event phenomenon -- a crack disperses, a steady
        # scrub does not -- so only the events pass the allpass chain.
        # With the squeal engaged, micro-slip energy crossfades OUT of
        # the bank and INTO the friction loop as slide jitter: the
        # micro-slips ARE the squeal's unsteadiness, not knocks on the
        # body. Pops arrive through 'pulse' too, and stay loud because
        # they are rare and big.
        pulse2 *= gd2
        ev_all = penv * voice
        ev = ev_all * (1.0 - mix) + pulse2 * voice
        # Slow: unsteadiness is the roughness of the surface passing
        # under the interface, tens of milliseconds, not a percussive
        # kick per micro-slip. The slips' energy drifts the operating
        # point and the squeal wavers.
        jit_lp += (ev_all * mix * jgain - jit_lp) * jit_k
        for k in range(stages):
            v = -chirp_a * ev + ap_x[k] + chirp_a * ap_y[k]
            ap_x[k] = ev
            ap_y[k] = v
            ev = v
        if c_amp > 1.0e-5:
            u = c_t0 / (c_t0 + c_time)
            # The sweep ENDS when the slow waves arrive: fading out as
            # it bottoms, rather than crawling through the low register
            # for seconds like a radio hunting for a station.
            if u < 0.08:
                c_amp = 0.0
            else:
                c_phase += pew_w * u * u
                attack = c_time / (0.02 * c_t0 + c_time + 1.0)
                fade = (u - 0.08) * 8.0
                if fade > 1.0:
                    fade = 1.0
                ev += c_amp * attack * fade * math.sin(c_phase)
                c_amp *= c_dec
                c_time += 1.0
        grind_sig = grind_gain * vel_env * g_lp * tex_comp
        raw = ev + grind_sig

        # The squeal, regenerative: rub~'s loop closed around this bank.
        # Every mode rings freely from its own history; their velocities
        # sum into the surface the hinge faces are touching; the slide
        # velocity against that surface goes through the friction curve;
        # and the force pours back into every mode. Nothing here sets
        # the squeal's pitch -- the loop locks onto a mode of the body
        # and jumps between modes as load and speed move, which is the
        # staircase brass taught us to trust. The interface engages only
        # while the strain is actually sliding (a Hill valve on speed),
        # so stillness stays silent by construction.
        surface = 0.0
        for m in range(modes):
            r_ = b1[m] * s1[m] + b2[m] * s2[m]
            ring[m] = r_
            surface += r_ - s1[m]
        h = sv if sv >= 0.0 else -sv
        # Squared Hill: engagement lets go decisively below hinge
        # speeds, so the turnarounds of a gesture pass through the
        # judder region quietly instead of knocking.
        engage = contact * (h * h / (h * h + v50 * v50))
        # pk puts the bank's motion in true velocity units (the raw
        # two-pole difference is smaller by the mode angle), so the
        # surface speaks the same language as the slide.
        dv = sv * vdg - surface * pk + jit_lp
        t = abs(dv * sl) + 0.75
        c = 1.0 / (t * t * t * t)
        if c > 1.0:
            c = 1.0
        fr = dv * c * engage

        # The grind excites the bank through its own gain vector: through
        # a comb of resonators, WHICH modes are rubbed is audible where a
        # spectral tilt of the noise is not.
        total = 0.0
        for m in range(modes):
            y = (gains[m] * (ev + fr) + ggains[m] * grind_sig
                 + ring[m])
            if y > 1.5:
                y = 1.5 + np.tanh(y - 1.5)
            elif y < -1.5:
                y = -1.5 - np.tanh(-y - 1.5)
            s2[m] = s1[m]
            s1[m] = y
            total += y
        out_raw[i] = raw
        out[i] = total + dry * raw
    return (s_prev, stress, threshold, pulse, penv, gd,
            stress2, threshold2, jit_lp, pulse2, sv,
            tune, vel_env, g_lp, c_phase, c_time, c_amp, c_dec, c_t0,
            max_strain, rng)


if _HAVE_NUMBA:
    _strain_kernel = njit(cache=True, fastmath=True)(_strain_kernel_source)
else:
    _strain_kernel = _strain_kernel_source


class StrainUnit(Unit):
    """Solids under stress: creak, crumple and crack from a strain input.

    The first unit whose input is effort itself rather than an
    instrument's controls: patch a joint angle or a stretch into
    'strain' and the model runs on it. Bending releases stick-slip
    events whose rate follows how fast you bend and whose loudness
    follows 'resist' -- tissue paper to oak door -- and stillness is
    silent by construction. The regime constants (set by the node's
    combo) make the difference between a hinge that creaks every time,
    paper that is loud only in new territory, and ice that cracks
    rarely, hugely, and once.

    'chirp' disperses each event the way a plate does -- flexural waves
    outrun their lows -- which is the lake-ice pew. The body is a mode
    table, drawn in the same editor as modal~ and rub~.

    'squeal' is regenerative: rub~'s friction loop closed around this
    unit's own bank, driven by the strain's actual sliding velocity.
    Nothing sets its pitch -- it locks onto a mode and jumps between
    modes as load and speed move, sings only in a window of sliding
    speed (slower judders, faster breaks up), and past the singing
    region it groans. All of that is what the physics does; none of it
    is coded as behavior.
    """

    MAX_MODES = 24
    CHIRP_STAGES = 12

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.strain_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.resist_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.stretch_in = self.new_inlet(base=0.3, minimum=-1.0, maximum=1.0)
        self.squeal_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.grind_in = self.new_inlet(base=0.2, minimum=0.0, maximum=1.0)
        self.texture_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.vary_in = self.new_inlet(base=0.15, minimum=0.0, maximum=1.0)
        self.chirp_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.pops_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.frequency_in = self.new_inlet(base=700.0, minimum=20.0)
        self.pitch_in = self.new_inlet()
        self.decay_in = self.new_inlet(base=0.4, minimum=0.01, maximum=60.0)
        self.dry_in = self.new_inlet(base=0.2, minimum=0.0, maximum=1.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        # Regime statistics, set by the node's combo.
        self.thresh = 0.004
        self.spread = 0.5
        self.alpha = 0.0
        self.size_cap = 2.0
        self.habituate = 0.6
        self.grain_seconds = 0.003
        self.amp = 0.25

        self._modes = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        self._weight_norm = 1.0
        self._s1 = np.zeros(StrainUnit.MAX_MODES, dtype=np.float64)
        self._s2 = np.zeros(StrainUnit.MAX_MODES, dtype=np.float64)
        self._b1 = np.zeros(StrainUnit.MAX_MODES, dtype=np.float64)
        self._b2 = np.zeros(StrainUnit.MAX_MODES, dtype=np.float64)
        self._gains = np.zeros(StrainUnit.MAX_MODES, dtype=np.float64)
        self._gains_live = np.zeros(StrainUnit.MAX_MODES, dtype=np.float64)
        self._ggains = np.zeros(StrainUnit.MAX_MODES, dtype=np.float64)
        self._live_count = 0
        self._fm = np.zeros(StrainUnit.MAX_MODES, dtype=np.float64)
        self._theta = np.zeros(StrainUnit.MAX_MODES, dtype=np.float64)
        self._radius = np.zeros(StrainUnit.MAX_MODES, dtype=np.float64)
        self._mode_scratch = np.zeros(StrainUnit.MAX_MODES, dtype=np.float64)
        self._ring = np.zeros(StrainUnit.MAX_MODES, dtype=np.float64)
        self._ap_x = np.zeros(StrainUnit.CHIRP_STAGES, dtype=np.float64)
        self._ap_y = np.zeros(StrainUnit.CHIRP_STAGES, dtype=np.float64)

        self._s_prev = 0.0
        self._stress = 0.0
        self._threshold = self.thresh
        self._pulse = 0.0
        self._penv = 0.0
        self._gd = 0.99
        self._stress2 = 0.0
        self._threshold2 = 1.0
        self._jit_lp = 0.0
        self._pulse2 = 0.0
        self._sv = 0.0
        self._tune = 1.0
        self._vel_env = 0.0
        self._g_lp = 0.0
        self._c_phase = 0.0
        self._c_time = 0.0
        self._c_amp = 0.0
        self._c_dec = 0.99
        self._c_t0 = 1.0
        self._max_strain = 0.0
        StrainUnit._seeded = getattr(StrainUnit, '_seeded', 0) + 1
        seed = (StrainUnit._seeded * 0x9E3779B97F4A7C15) % (1 << 64)
        self._rng = np.uint64(seed if seed else 0x853C49E6748FEA9B)
        self._quiet = True

        self.out = self.new_outlet()
        self.grains = self.new_outlet()
        self._raw = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._strain = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def set_modes(self, table):
        """Same live-edit contract as modal~ and rub~."""
        rows = [row for row in table[:StrainUnit.MAX_MODES]]
        if not rows:
            rows = [(1.0, 1.0, 1.0)]
        fresh = np.array(rows, dtype=np.float64)
        resized = fresh.shape[0] != self._modes.shape[0]
        self._modes = fresh
        self._weight_norm = max(1.0, float(np.sum(np.abs(fresh[:, 1]))))
        if resized:
            self._s1[:] = 0.0
            self._s2[:] = 0.0

    def reset(self):
        self._s1[:] = 0.0
        self._s2[:] = 0.0
        self._ap_x[:] = 0.0
        self._ap_y[:] = 0.0
        self._stress = 0.0
        self._pulse = 0.0
        self._penv = 0.0
        self._stress2 = 0.0
        self._jit_lp = 0.0
        self._pulse2 = 0.0
        self._sv = 0.0
        self._quiet = True

    def deactivate(self):
        self.reset()

    def render(self, frames):
        strain = self.strain_in.eval(frames)
        resist = self.resist_in.eval(frames)
        stretch = self.stretch_in.eval(frames)
        squeal = self.squeal_in.eval(frames)
        grind = self.grind_in.eval(frames)
        texture = self.texture_in.eval(frames)
        vary = self.vary_in.eval(frames)
        chirp = self.chirp_in.eval(frames)
        pops = self.pops_in.eval(frames)
        frequency = self.frequency_in.eval(frames)
        pitch = self.pitch_in.eval(frames)
        decay = self.decay_in.eval(frames)
        dry = self.dry_in.eval(frames)
        out_level = self.level_in.eval(frames)

        out = self.out
        if not _svf_ready.is_set():
            out.set_constant(0.0)
            return

        bend = self._strain[:frames]
        if strain.constant:
            bend[:] = strain.value
            still = True
        else:
            np.copyto(bend, strain.data[:frames])
            still = False
        np.clip(bend, 0.0, 1.0, out=bend)

        # Stillness makes no events; once the body has rung down there is
        # nothing left to render. A held strain is stillness too.
        if self._quiet and still and self._pulse < 1.0e-6:
            self._s_prev = float(bend[0])
            out.set_constant(0.0)
            self.grains.set_constant(0.0)
            return

        def scalar(signal, lo, hi):
            value = signal.value if signal.constant else float(signal.data[0])
            return min(hi, max(lo, value))

        resist_now = scalar(resist, 0.0, 1.0)
        chirp_now = scalar(chirp, 0.0, 1.0)
        seconds = scalar(decay, 0.01, 60.0)

        curve = self._scratch[:frames]
        self._build_hertz(curve, frequency, pitch, frames, 20.0)
        f0 = float(curve[0])
        # The body is under load: stress-stiffening shifts its resonances
        # with the strain itself, up to an octave across the full bend.
        # The coefficients rebuild per block over persistent ring states,
        # so the rings BEND as the bending continues -- a groan that rises
        # through the gesture rather than a row of identical pings.
        stretch_now = scalar(stretch, -1.0, 1.0)
        f0 *= 2.0 ** (stretch_now * float(bend[0]))

        modes = self._modes
        count = modes.shape[0]
        ratios = modes[:, 0]
        weights = modes[:, 1]
        decay_scale = modes[:, 2]

        fm = self._fm[:count]
        np.multiply(ratios, f0, out=fm)
        limit = self.sample_rate * 0.45
        theta = self._theta[:count]
        np.clip(fm, 1.0, limit, out=theta)
        theta *= 2.0 * math.pi / self.sample_rate
        radius = self._radius[:count]
        np.multiply(decay_scale, seconds * self.sample_rate, out=radius)
        np.clip(radius, 1.0, None, out=radius)
        np.divide(-6.907755, radius, out=radius)
        np.exp(radius, out=radius)
        # b1 is the kernel's to derive (per-event ensemble
        # retuning); only the ingredients are prepared here.
        b1 = self._b1[:count]
        b2 = self._b2[:count]
        np.multiply(radius, radius, out=b2)
        np.negative(b2, out=b2)
        gains = self._gains[:count]
        np.sin(theta, out=gains)
        gains *= weights
        gains /= self._weight_norm
        alive = self._mode_scratch[:count]
        np.less_equal(fm, limit, out=alive, casting='unsafe')
        gains *= alive
        live = self._gains_live[:count]
        if count != self._live_count:
            np.copyto(live, gains)
            self._live_count = count
        else:
            step = self._mode_scratch[:count]
            np.subtract(gains, live, out=step)
            step *= 0.35
            live += step

        # 'resist': tissue paper to oak door. Heavier resistance means
        # sparser, bigger, louder releases.
        thresh_eff = self.thresh * (4.0 ** (2.0 * resist_now - 1.0))
        amp_eff = self.amp * (2.0 ** (2.0 * resist_now - 1.0))
        grain_samples = self.grain_seconds * self.sample_rate
        atk_k = 1.0 - math.exp(-1.0 / (0.25 * max(4.0, grain_samples)))
        jit_k = 1.0 - math.exp(-1.0 / (0.03 * self.sample_rate))
        gd2 = math.exp(-1.0 / grain_samples)
        chirp_a = chirp_now * 0.5
        squeal_now = scalar(squeal, 0.0, 1.0)
        # The squeal is regenerative: 'squeal' is how firmly the
        # interface engages the body, 'resist' how wide its sticking
        # region is, and the pitch is in neither number -- the friction
        # loop locks onto whichever mode of the bank will carry it.
        # Placed from the measured map: the loop is damped below about
        # 0.5, sings from 0.7, and tips into groaning chaos past 1.2 --
        # so the knob's top third is the singing region and full squeal
        # leans on the door hard enough to groan. There is also a speed
        # window, as with a real hinge: too slow judders, too fast
        # breaks up, and neither is an error.
        contact = squeal_now * (0.4 + 0.6 * resist_now) * 1.75
        sl = 5.0 - 4.0 * resist_now
        # Micro-slips move house as the squeal engages: by squeal 0.5
        # they are entirely the loop's unsteadiness, and the bank hears
        # only the rare macro pops.
        mix = min(1.0, 2.0 * squeal_now)
        # How often the geometry lets go, exponentially: 0.5 is the
        # natural rate, 0 is never, 1 is a rolling grumble of releases.
        # An inlet, so an effort stream can DEMAND a pop.
        pops_now = scalar(pops, 0.0, 1.0)
        if pops_now < 0.005:
            macro_thresh = 1.0e12
            # The armed threshold predates the knob hitting zero: never
            # means never, including the one already in the chamber.
            if self._threshold2 < 1.0e11:
                self._threshold2 = 1.0e12
        else:
            macro_thresh = thresh_eff * 45.0 * (8.0 ** (1.0 - 2.0 * pops_now))
            # Thresholds redraw only at a release, so a knob turned UP
            # must re-arm the pending one -- otherwise a spell at zero
            # (or at rare) leaves an armed threshold too far away to
            # ever fire at the new rate.
            if self._threshold2 > macro_thresh * 1.6:
                self._threshold2 = macro_thresh
        pop_amp = amp_eff * 1.5
        pk = 0.5 * self.sample_rate / (2.0 * math.pi * f0)
        vary_oct = scalar(vary, 0.0, 1.0) * 0.5
        dry_now = scalar(dry, 0.0, 1.0)
        # Grind rides speed (in strain-units per second, enveloped over
        # ~20 ms) and leans on the load like everything frictional.
        # Full knob is a modest scrub: the useful range of a texture is
        # narrow, so the whole travel is spent inside it.
        grind_gain = (scalar(grind, 0.0, 1.0) * (0.4 + 0.8 * resist_now)
                      * 0.12 * self.sample_rate)
        vel_k = 1.0 - math.exp(-1.0 / (0.02 * self.sample_rate))
        tex_now = scalar(texture, 0.0, 1.0)
        tex_cut = 150.0 * (80.0 ** tex_now)
        tex_k = 1.0 - math.exp(-2.0 * math.pi * tex_cut / self.sample_rate)
        tex_comp = math.sqrt((2.0 - tex_k) / tex_k)
        # Dark rubs the low modes, bright the high: the grind's own
        # injection gains, tilted by texture and held at equal total.
        ggains = self._ggains[:count]
        np.power(ratios, (tex_now - 0.5) * 3.0, out=ggains)
        ggains *= live
        # Held at equal POWER, exactly: the two-pole's noise-energy
        # transfer has a closed form, and the one-pole pre-filter shapes
        # what each mode receives at its own frequency. Both go into the
        # balance, referenced to white injection through the live gains,
        # so turning the texture knob moves character and nothing else.
        r2 = radius * radius
        cos2 = np.cos(2.0 * theta)
        energy = (1.0 + r2) / ((1.0 - r2)
                               * (1.0 - 2.0 * r2 * cos2 + r2 * r2))
        one_minus_k = 1.0 - tex_k
        lp_psd = ((tex_comp * tex_k) ** 2
                  / (1.0 - 2.0 * one_minus_k * np.cos(theta)
                     + one_minus_k * one_minus_k))
        target = float(np.sum(live * live * energy))
        current = float(np.sum(ggains * ggains * lp_psd * energy))
        if current > 1.0e-18:
            ggains *= math.sqrt(target / current)
        # The pew: chirp scales how long each event's dispersive sweep
        # lasts (a farther fracture), starting well above the body and
        # sweeping down through it.
        # Squared: most of the knob's travel lives in short, subtle
        # dispersions, and the lake only opens up near the top.
        pew_scale = chirp_now * chirp_now * 0.35 * self.sample_rate
        pew_w = (2.0 * math.pi
                 * min(0.35 * self.sample_rate, f0 * 12.0)
                 / self.sample_rate)
        recover = 1.0 / (20.0 * self.sample_rate)

        result = self._y[:frames]
        (self._s_prev, self._stress, self._threshold, self._pulse,
         self._penv, self._gd, self._stress2, self._threshold2,
         self._jit_lp, self._pulse2, self._sv, self._tune,
         self._vel_env, self._g_lp, self._c_phase, self._c_time,
         self._c_amp, self._c_dec, self._c_t0, self._max_strain,
         rng_state) = _strain_kernel(
            bend, thresh_eff, self.spread, self.alpha, self.size_cap,
            self.habituate, grain_samples, amp_eff, chirp_a,
            contact, sl, 0.09, 0.45, pk, float(self.sample_rate), vary_oct,
            atk_k, mix, 0.5, jit_k, macro_thresh, pop_amp, gd2,
            grind_gain, vel_k, tex_k, tex_comp,
            theta, radius, b1, b2,
            live, ggains, self._ring[:count],
            self._s1[:count], self._s2[:count],
            self._ap_x, self._ap_y,
            self._s_prev, self._stress, self._threshold, self._pulse,
            self._penv, self._gd, self._stress2, self._threshold2,
            self._jit_lp, self._pulse2, self._sv, self._tune,
            self._vel_env, self._g_lp, pew_scale, pew_w,
            self._c_phase, self._c_time, self._c_amp, self._c_dec,
            self._c_t0, self._max_strain, recover,
            self._rng, dry_now, self._raw[:frames], result)
        self._rng = np.uint64(rng_state)

        self._apply_level(result, out_level, frames)
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        scratch = self._scratch[:frames]
        np.abs(result, out=scratch)
        self._quiet = bool(scratch.max() < 1.0e-5)
        np.copyto(self.grains.data[:frames], self._raw[:frames],
                  casting='unsafe')
        self.grains.constant = False


class StrokeUnit(Unit):
    """A bow arm: the coordinated velocity/force pair, as two outlets.

    Guettler's finding, made patchable: a clean bow stroke is not a
    velocity shape, it is a coordination -- clean attacks live in a wedge
    of the acceleration/force plane, which is why no single LFO waveform
    ever bows well. Velocity here is a trapezoid with raised-cosine
    corners, fast enough through the low-speed region that the string
    never settles into a wrong regime, never discontinuous because no arm
    is. Force is its counter-phase: 'lean' raises it exactly where
    velocity dips, which is the pressure a player keeps through the bow
    change so the widened sticking region carries the string across.

    'run' strokes continuously at 'rate'; 'dip' is how low the turnaround
    goes -- 1 is seamless legato, 0 lifts off the string between strokes.
    'gate' draws the bow while the gate is high and lifts on release, both
    over 'corner' seconds. A trigger fires one complete stroke from rest
    to rest in either mode. 'swell' arches the cruise of each stroke (and
    so, through 'lean', eases the force mid-stroke); it shapes run and
    triggered strokes, where there is a stroke to arch, not a held gate.

    'tick' pulses one sample high at each turnaround: bowing as a clock,
    for whatever should happen in time with the arm.

    The outputs are absolute positions of the arm, so the destination's
    own velocity and force knobs should sit at zero -- the inlet triad
    sums, and a knob left up would ride on top of the gesture.

    Gesture timing is block-granular (a stroke is seconds long; a block
    is twelve milliseconds), and rate and corner are read once per block.
    """

    MODES = ('run', 'gate')

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.rate_in = self.new_inlet(base=1.0, minimum=0.05, maximum=8.0)
        self.speed_in = self.new_inlet(base=0.8, minimum=0.0, maximum=1.5)
        self.dip_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.corner_in = self.new_inlet(base=0.03, minimum=0.005, maximum=0.3)
        self.force_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.lean_in = self.new_inlet(base=0.4, minimum=0.0, maximum=1.0)
        self.swell_in = self.new_inlet(base=0.2, minimum=0.0, maximum=1.0)
        self.gate_in = self.new_inlet()
        self.trigger_in = self.new_inlet()

        self.mode = 0
        self.threshold = 0.5
        self._phase = 0.0
        self._edge = 0.0            # gate mode: where the draw has reached
        self._one_shot = False
        self._gate_open = False
        self._trigger_armed = True
        self._fire_requests = 0
        self._fire_served = 0

        self.velocity_out = self.new_outlet()
        self.force_out = self.new_outlet()
        self.tick_out = self.new_outlet()
        self._phi = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._n = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._work = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._push = np.zeros(MAX_BLOCK, dtype=np.float64)

    def fire(self):
        """Request one stroke from the node layer. Served next block."""
        self._fire_requests += 1

    def reset(self):
        self._phase = 0.0
        self._edge = 0.0
        self._one_shot = False

    def render(self, frames):
        rate = self.rate_in.eval(frames)
        speed = self.speed_in.eval(frames)
        dip = self.dip_in.eval(frames)
        corner = self.corner_in.eval(frames)
        force = self.force_in.eval(frames)
        lean = self.lean_in.eval(frames)
        swell = self.swell_in.eval(frames)
        gate = self.gate_in.eval(frames)
        trigger = self.trigger_in.eval(frames)

        def scalar(signal):
            return signal.value if signal.constant else float(signal.data[0])

        rate_now = min(8.0, max(0.05, scalar(rate)))
        speed_now = min(1.5, max(0.0, scalar(speed)))
        dip_now = min(1.0, max(0.0, scalar(dip)))
        corner_now = min(0.3, max(0.005, scalar(corner)))
        force_now = min(1.0, max(0.0, scalar(force)))
        lean_now = min(1.0, max(0.0, scalar(lean)))
        swell_now = min(1.0, max(0.0, scalar(swell)))

        # Triggers: from the node layer, and rising through the inlet.
        fired = False
        if self._fire_requests != self._fire_served:
            self._fire_served = self._fire_requests
            fired = True
        high = scalar(trigger) >= self.threshold
        if high and self._trigger_armed:
            fired = True
        self._trigger_armed = not high
        if fired:
            self._one_shot = True
            self._phase = 0.0
            self._edge = 0.0

        gate_now = scalar(gate) >= self.threshold
        self._gate_open = gate_now

        tick_at = -1
        n = self._n[:frames]

        if self._one_shot or self.mode == 0:
            # A stroke is one turn of the phasor, shaped. The corner is
            # asked for in seconds and clamped as a fraction of the stroke,
            # since a corner longer than the stroke is not a corner.
            one_shot_block = self._one_shot
            increment = rate_now / self.sample_rate
            cf = min(0.45, max(0.02, corner_now * rate_now))
            phi = self._phi[:frames]
            np.multiply(_INDEX_RAMP[:frames], increment, out=phi)
            phi += self._phase - increment
            wrapped = phi >= 1.0
            if wrapped.any():
                tick_at = int(np.argmax(wrapped))
                phi -= np.floor(phi)
            self._phase = float(phi[-1]) + increment

            if one_shot_block and tick_at >= 0:
                # The stroke completes at the wrap and the arm comes to
                # rest; the tail of the block stays lifted.
                phi[tick_at:] = 0.0
                self._phase = 0.0
                self._edge = 0.0
                self._one_shot = False

            # A triggered stroke starts and ends at rest; a running one
            # only dips at its turnarounds.
            floor = 0.0 if one_shot_block else dip_now
            np.minimum(phi, 1.0 - phi, out=n)
            n /= cf
            np.clip(n, 0.0, 1.0, out=n)
            n *= math.pi
            np.cos(n, out=n)
            np.subtract(1.0, n, out=n)
            n *= 0.5 * (1.0 - floor)
            n += floor
            if swell_now > 0.0:
                arch = self._work[:frames]
                np.multiply(phi, math.pi, out=arch)
                np.sin(arch, out=arch)
                arch *= 0.4 * swell_now
                arch += 1.0
                n *= arch
        else:
            # Gate mode: the draw slews toward on or off over the corner
            # time, cosine-shaped so the start and the lift are arms too.
            step = 1.0 / max(1.0, corner_now * self.sample_rate)
            target_dir = step if gate_now else -step
            edge = self._edge
            path = self._phi[:frames]
            np.multiply(_INDEX_RAMP[:frames], target_dir, out=path)
            path += edge
            np.clip(path, 0.0, 1.0, out=path)
            self._edge = float(path[-1])
            np.multiply(path, math.pi, out=n)
            np.cos(n, out=n)
            np.subtract(1.0, n, out=n)
            n *= 0.5

        velocity = self.velocity_out
        buffer = velocity.data[:frames]
        np.multiply(n, speed_now, out=buffer, casting='unsafe')
        velocity.constant = False

        push = self._push[:frames]
        np.subtract(1.0, n, out=push)
        push *= lean_now
        push += 1.0
        push *= force_now
        np.clip(push, 0.0, 1.0, out=push)
        out_force = self.force_out
        np.copyto(out_force.data[:frames], push, casting='unsafe')
        out_force.constant = False

        tick = self.tick_out
        if tick_at >= 0:
            tick.data[:frames] = 0.0
            tick.data[tick_at] = 1.0
            tick.constant = False
        else:
            tick.set_constant(0.0)


def _disc_deriv_source(q2, q3, u1, u2, u3, radius, half_thick,
                       out_of_round, grav, cos_floor, u_cap):
    """The rolling coin's equations of motion. Derived, not devised.

    Kane's method on a uniform CYLINDER rolling without slipping on a
    plane. A knife-edge disc puts its centre of mass one radius from the
    contact, straight up the lean; a real coin's contact sits on the
    edge where the rim meets the lower face, so the centre is a radius
    away radially AND half a thickness along the symmetry axis. That one
    term is the whole difference, and it keeps the centre of mass from
    descending to nothing when the coin goes flat.

    The mass cancels, as it must for a rigid body falling under gravity,
    and so do the yaw and the spin angle -- only the lean matters,
    because the coin is symmetric and the table is uniform.

    Derived in dpg_system/tests/disc_reference.py and checked against it
    over five hundred random states, worst relative difference two parts
    in a hundred million million. With the thickness set to zero it
    reproduces the knife-edge disc exactly, which is checked there too.

    Clamped HERE rather than only on the finished step: RK4 evaluates
    this four times per step at states it invents along the way, and
    near the singularity those can run far past anything the coin
    reaches -- far enough to overflow, after which infinity minus
    infinity is a NaN that never leaves.
    """
    c = math.cos(q2)
    if c < cos_floor:
        c = cos_floor
    elif c > 1.0:
        c = 1.0
    if u3 > u_cap:
        u3 = u_cap
    elif u3 < -u_cap:
        u3 = -u_cap
    if u1 > u_cap:
        u1 = u_cap
    elif u1 < -u_cap:
        u1 = -u_cap
    if u2 > u_cap:
        u2 = u_cap
    elif u2 < -u_cap:
        u2 = -u_cap
    # A real coin's rim is not a circle, and that is geometry rather
    # than a coefficient: the distance from the centre to the contact
    # varies with where around the rim the contact currently sits.
    r = radius * (1.0 + out_of_round * math.cos(q3))
    th = half_thick
    s = math.sin(q2)
    t1 = u3 * s / c
    t2 = -t1 + u2
    t3 = r * r
    t4 = 0.25 * t3
    t5 = th * th
    t8 = t4 * u3
    t9 = u1 * u1
    t10 = th * u3
    t11 = r * u2 - t10
    t12 = 3.0 * t3
    t13 = 4.0 * t5
    return (u3 / c,
            u1,
            t2,
            (-grav * c * th + grav * r * s + t2 * t8
             + r * (t11 * u3 - t9 * th) - th * (-r * t9 - t1 * t11)
             + t8 * u2) / ((t3 + t5) + t4),
            -2.0 * r * u1 * (r * u3 + 2.0 * th * u2) / (t12 + t13),
            (1.0 / 12.0) * u1 * (4.0 * r * t10 + t1 * t12 + t1 * t13
                                 - 6.0 * t3 * u2)
            / ((1.0 / 3.0) * t5 + t4))


if _HAVE_NUMBA:
    _disc_deriv = njit(cache=True, inline='always')(_disc_deriv_source)
else:
    _disc_deriv = _disc_deriv_source


def _spin_real_kernel_source(gesture, radius, grav, tilt_full, tilt_flat,
                             lift_k, loss, law, twist, wobble, contact,
                             strike_scale, face_slap, ripple_gain,
                             scrape_gain, scrape_k, load_exp, sharpen,
                             sharp_k, bright_norm, stop_k, grain_dens,
                             grain_floor, grain_ceiling, sweep_ref,
                             half_thick, absolute, dead_zone, track_k,
                             drain_side,
                             inject_gain, balance_gain,
                             cast_least, cast_gain,
                             grain_tail, grain_max, grain_norm,
                             out_of_round, drain_k, nut_kick,
                             restitution, u_ref, roll_floor, fall_floor,
                             u3_ceiling,
                             u_cap, u_floor, load_head, flat_ref, lean_k,
                             follow_k, hop_gain, prof_depth, table_rough,
                             rough_depth, rough_norm,
                             q2_ceiling,
                             cos_floor, decim, dt,
                             sample_rate,
                             q1, q2, q3, u1, u2, u3, gesture_last, push,
                             lean_goal, mean_lean,
                             load_now, sweep_now, prec_now,
                             slap_amp, slap_at, slap_len, grip, lp, hp,
                             edge, landed, hop_v, hop_h, rng,
                             hurst_inv, grain_span, grain_least,
                             grain_head,
                             out, rate_out, face_out, grind_out,
                             strike_out, grain_ring, rim_amp, rim_phase,
                             rough_k, rough_state):
    """A disc settling, integrated from its own equations of motion.

    The hand-made version of this unit reproduced a coin by assembling
    behaviours: a rate law, a load shape, a nutation, an eccentric orbit,
    each fitted by ear until it sounded right. This one runs the disc
    instead, and reads the sound off it. The precession, the swing in the
    lean, the surge in contact load and the speed the contact sweeps the
    rim are not modelled here at all -- they are what the equations do.

    Dissipation is the one thing not derived, and cannot be: which loss
    dominates a settling coin (air in the gap, rolling friction, contact
    damping) is a modelling choice, and Moffatt wrote a paper about
    exactly that. So the rigid-body motion is integrated honestly and
    energy is drained from it to follow the settle law the node already
    had -- 'settle' and 'rush' therefore mean what they always meant,
    while everything fast comes from the disc.

    The mechanical motion tops out in the hundreds of hertz, so it is
    integrated once every few samples and read between.
    """
    two_pi = 6.283185307179586
    ring_n = grain_ring.shape[0]
    head = int(grain_head)
    count = 0
    for i in range(gesture.shape[0]):
        # 'spin' works by CHANGE, not by level. Rising, it puts energy
        # in; falling, it takes energy out; held still -- at any value
        # whatever -- it does nothing and the coin settles as it would
        # have anyway. A hand that stops moving stops playing.
        move = gesture[i] - gesture_last
        gesture_last = gesture[i]
        push += move
        if landed > 0.5 and push < 0.0:
            # A coin already at rest has no energy to take away, so a
            # falling gesture on a stopped coin does nothing. Letting it
            # accumulate downwards dug a hole the next throw had to
            # climb out of first -- release after a coin has landed and
            # an identical rise would only bring the total back to zero,
            # so the node went silent and stayed silent.
            push = 0.0
        want = tilt_full * gesture[i]
        if absolute > 0.5:
            # ABSOLUTE: the gesture is the lean it asks for, held for as
            # long as it is held. Sustained motion keeps the coin going;
            # letting go lets it settle. This is the older reading of
            # 'spin', kept because holding a coin open is a different
            # instrument from throwing one.
            push = 0.0
            if want > lean_goal:
                lean_goal = want if landed > 0.5 \
                    else lean_goal + (want - lean_goal) * lift_k
        # A coin has to be thrown far enough over to be a throw. The
        # flat limit is not that mark -- it is a thousandth of a radian,
        # so in hold mode the first hair of slider movement cast a coin
        # already at its end, which collapsed at once, landed, and was
        # thrown again by the next hair. Sliding the control up read as
        # a collapse and a resurrection over and over, while pinning it
        # straight to the top sounded perfect.
        cast_floor = tilt_full * cast_least
        if cast_floor < tilt_flat:
            cast_floor = tilt_flat
        if landed > 0.5 and (push > cast_least
                             or (absolute > 0.5 and lean_goal > cast_floor)):
            # At rest, a rise throws the coin, and how much of a rise
            # decides how far over. It waits for a real gesture rather
            # than a trickle: the loss law is steepest where the coin is
            # flattest, so a coin thrown barely over is already dead
            # when it lands -- at rush 1 the loss takes lean away some
            # eight hundred times faster than a slow hand can add it.
            if absolute < 0.5:
                lean_goal = tilt_full * push * cast_gain
                if lean_goal > tilt_full:
                    lean_goal = tilt_full
            push = 0.0
            if lean_goal > tilt_flat:
                # Only a coin that is actually AT REST gets cast. A
                # rising gesture on a coin already going must open it
                # up, not throw it again -- re-cast on every rising
                # sample and a held gesture rebuilds a fresh disc in
                # perfect steady roll thousands of times a second,
                # which is silent by construction and was the cause of
                # the long featureless openings.
                landed = 0.0
                # Cast the coin. 'twist' is the fraction of the steady
                # roll it was actually given -- and that is the whole of
                # it, because a disc spun below its equilibrium speed is
                # not in equilibrium, so the lean falls on its own. The
                # nutation does not have to be put there; under-spinning
                # IS the nutation.
                #
                # At 1 the coin is cast in exact steady roll and never
                # wobbles. At 0 it is given no spin at all: it does not
                # roll, it simply tips onto its face and bounces, which
                # is what a coin pushed over does. Everything between is
                # a real throw.
                #
                # A kick on the lean rate used to stand in for this,
                # while the roll was always cast at full speed -- so no
                # twist still meant a coin spinning perfectly and merely
                # wobbling, and it read as a fifth of a throw rather
                # than none of one.
                q2 = 1.5707963267948966 - lean_goal
                # The steady roll for a coin with thickness, in closed
                # form. With no thickness this is the knife edge's
                # 2*sqrt(g*cos(q2)/r) exactly.
                _s = math.sin(q2)
                _c = math.cos(q2)
                if _c < cos_floor:
                    _c = cos_floor
                _tn = _s / _c
                _num = grav * (radius * _s - half_thick * _c)
                _den = 0.25 * radius * radius * _tn \
                    + radius * half_thick + half_thick * half_thick * _tn
                if _num < 0.0:
                    _num = 0.0
                u3 = twist * math.sqrt(_num / _den)
                u2 = 0.0
                u1 = 0.0
                mean_lean = lean_goal
        hertz = 0.0
        rolling = 0.0
        struck = 0.0
        if landed < 0.5:
            count += 1
            if count >= decim:
                count = 0
                # --- the disc itself, one RK4 step ---
                a0, a1, a2, a3, a4, a5 = _disc_deriv(
                    q2, q3, u1, u2, u3, radius, half_thick, out_of_round,
                    grav, cos_floor, u_cap)
                b0, b1, b2, b3, b4, b5 = _disc_deriv(
                    q2 + 0.5*dt*a1, q3 + 0.5*dt*a2, u1 + 0.5*dt*a3,
                    u2 + 0.5*dt*a4, u3 + 0.5*dt*a5, radius, half_thick,
                    out_of_round, grav, cos_floor, u_cap)
                c0, c1, c2, c3, c4, c5 = _disc_deriv(
                    q2 + 0.5*dt*b1, q3 + 0.5*dt*b2, u1 + 0.5*dt*b3,
                    u2 + 0.5*dt*b4, u3 + 0.5*dt*b5, radius, half_thick,
                    out_of_round, grav, cos_floor, u_cap)
                e0, e1, e2, e3, e4, e5 = _disc_deriv(
                    q2 + dt*c1, q3 + dt*c2, u1 + dt*c3, u2 + dt*c4,
                    u3 + dt*c5, radius, half_thick, out_of_round, grav,
                    cos_floor, u_cap)
                q1 += (dt/6.0) * (a0 + 2.0*b0 + 2.0*c0 + e0)
                q2 += (dt/6.0) * (a1 + 2.0*b1 + 2.0*c1 + e1)
                q3 += (dt/6.0) * (a2 + 2.0*b2 + 2.0*c2 + e2)
                u1 += (dt/6.0) * (a3 + 2.0*b3 + 2.0*c3 + e3)
                u2 += (dt/6.0) * (a4 + 2.0*b4 + 2.0*c4 + e4)
                u3 += (dt/6.0) * (a5 + 2.0*b5 + 2.0*c5 + e5)
                # The lean swinging down to nothing is the face meeting
                # the table. That is a CONTACT, not the end -- the coin
                # bounces off it and goes on nutating, which is the whole
                # of what a badly cast one does. Treating it as the end
                # made a big nutation kill the disc on its first swing,
                # so the worse the cast, the less there was to hear: the
                # exact opposite of a real coin.
                # The bound is the coin standing on its edge, which is
                # where these equations stop meaning anything -- NOT the
                # lean the gesture asked for. Clamping at the request
                # leaves a nutation no headroom above it: hold the
                # gesture at maximum and every upswing is clipped, the
                # wobble bleeds away against the ceiling, and what was a
                # coin becomes a steady tone. The pump aims at the
                # request; this only stops a runaway.
                q2_open = q2_ceiling
                if q2 < q2_open:
                    q2 = q2_open
                    if u1 < 0.0:
                        u1 = 0.0
                q2_flat = 1.5707963267948966 - tilt_flat
                if q2 > q2_flat:
                    q2 = q2_flat
                    if u1 > 0.0:
                        # Coming down onto the face. Rebound, and let the
                        # blow be as hard as the arrival.
                        hit = u1
                        u1 = -u1 * restitution
                        # (A friction scrub on the roll was tried here,
                        # to make a coin that slams repeatedly lose its
                        # spin and settle sooner. It changes nothing
                        # measurable: near flat the term u1*tan(q2)*u3
                        # amplifies whatever roll is left faster than an
                        # impact can take it away, so the roll regrows
                        # between contacts. Removed rather than left in
                        # doing nothing.)
                        # As hard as it arrives. Scaling by the lean
                        # that was left made sense when the only impact
                        # was the coin finally lying down; for one that
                        # is BOUNCING it is wrong, and at high polish it
                        # divided every blow by forty-odd, which is why
                        # a coin with no spin -- all bounce and no roll
                        # -- could not be heard at all.
                        amp = strike_scale * face_slap * (hit / u_ref)
                        if amp > 3.0:
                            amp = 3.0
                        slap_amp = amp
                        slap_at = 0.0
                        slap_len = contact * 2.0
                    # Down when it has stopped BOUNCING. A coin whose
                    # lean has reached the limit and is no longer
                    # flopping has settled, whatever its roll is doing:
                    # the limit is where the surface stops it.
                    #
                    # Requiring the roll to be near zero as well looked
                    # right and was not. With a rough surface the limit
                    # is a large lean, and the steady roll AT that lean
                    # is still fast -- so the coin arrived at its limit
                    # with far too much roll to satisfy the test, the
                    # drain had nothing left to pull against, and it
                    # rolled there for ever. That is the whole of why a
                    # low polish never stopped.
                    if u1 > -fall_floor and u1 < fall_floor:
                        landed = 1.0
                if u1 > u_cap:
                    u1 = u_cap
                elif u1 < -u_cap:
                    u1 = -u_cap
                # The roll is cast in one direction and stays there. A
                # disc that reverses its precession mid-settle is the
                # controller pushing it through zero, not anything a
                # coin does, and it reads as the rate outlet going
                # negative.
                if u3 > u_cap:
                    u3 = u_cap
                elif u3 < u_floor:
                    u3 = u_floor
                # Wrapped properly, not by subtracting one turn. The
                # contact can sweep more than a full turn in a single
                # control step, and a single subtraction cannot keep up
                # with that -- the angle then grows without bound, and
                # long before it reaches anything a NaN check would
                # catch, the cosine of it has lost all its precision.
                # That is what turns 'face' into a stepped square wave:
                # not an overflow, an angle no longer worth taking a
                # cosine of.
                q1 = q1 % two_pi
                q3 = q3 % two_pi

                lean = 1.5707963267948966 - q2
                if lean < 0.0:
                    lean = 0.0
                # Per unit time, like the drain and the tracking: these
                # all ran once per control step, so the integration rate
                # set how fast the coin was averaged, carried and
                # drained. Written this way the kernel can be stepped
                # finer for accuracy without becoming a different coin.
                mean_k = lean_k * decim
                if mean_k > 0.5:
                    mean_k = 0.5
                mean_lean += (lean - mean_lean) * mean_k

                # --- the loss, which is the one thing not derived ---
                # A flopping coin does settle sooner, but that is not
                # something to write down here: its face contacts are
                # inelastic and the rebound already takes the energy. A
                # term for it as well would be counting the same loss
                # twice, and inventing the coefficient to do it.
                lean_goal -= loss * decim * lean_goal ** (-law)
                if lean_goal < tilt_flat:
                    lean_goal = tilt_flat
                # Energy in and energy out, by the same handle. The
                # steady roll goes as the square root of the lean, so
                # taking speed out lowers the coin and putting it in
                # opens it -- which makes 'spin' a fluid grip on the
                # tilt rather than a trigger. Hold a gesture and the
                # coin stays open, still nutating and still grinding;
                # release it and the loss law closes it.
                # Pull down when the coin sits above where it should
                # be, feed when it sits below -- but not near the mark.
                # These two act on a measurement that lags, against a
                # coin with dynamics of its own, so with no dead zone
                # between them they simply take turns overshooting: the
                # lean ends up swinging over half its range while its
                # goal stands still, which is hunting rather than
                # settling, and it sounds like a coin tumbling.
                over = (mean_lean - lean_goal) / lean_goal
                # The dead zone exists so the drain and the pump do not
                # take turns overshooting each other. It has to CLOSE as
                # the goal reaches the floor, though: down there the
                # gesture has released and there is no pump to fight,
                # and a band that stays open lets the coin park just
                # inside it -- a fraction above the floor, rolling at
                # full speed, with nothing left that can bring it down.
                # That is a coin nothing can stop.
                near_end = 1.0 - tilt_flat / lean_goal
                if near_end < 0.0:
                    near_end = 0.0
                drain_dead = dead_zone * near_end
                if over > drain_dead:
                    # Per unit TIME, not per control step. The lean goal
                    # above is already scaled by the step; this was not,
                    # so the integration rate quietly set how fast the
                    # coin gave up its energy -- the same settings ran
                    # 1.84 s at a decimation of eight and 0.75 s at one.
                    # 'settle' has to mean seconds, whatever the kernel
                    # is stepping at.
                    scale = 1.0 - drain_k * decim * (over - drain_dead)
                    if scale < 0.5:
                        scale = 0.5
                    # Losing energy means DESCENDING THE STEADY FAMILY,
                    # not slowing everything down together. Scaling the
                    # roll away with the rest left the coin with a tenth
                    # of the roll its lean calls for, so it could not
                    # precess -- it just rocked, and the contact rate,
                    # the loudness and the brightness all collapsed with
                    # it. The family says otherwise: as the lean closes
                    # from twenty degrees to a tenth the roll does fall,
                    # 29.9 to 2.2, but the precession RISES fourteenfold
                    # and the contact sweep fifteenfold, so the grind --
                    # which rides on its square root -- gains about
                    # twelve decibels on the way down. A real coin
                    # recorded doing this gains nine. The crescendo is
                    # the whole character of a settling coin, and it was
                    # missing because the roll was being drained instead
                    # of being allowed to follow the lean.
                    #
                    # The nutation is the deviation from this family and
                    # is left to damp on its own, so the coin still
                    # bounces on the way down.
                    # Aimed at the lean the LOSS is driving towards,
                    # not the one the coin is at. At the current lean
                    # the steady roll is by definition the one that
                    # holds it, so the coin simply stayed open and never
                    # settled at all. Aimed one step down the family it
                    # descends, which is what quasi-static dissipation
                    # means.
                    _q2g = 1.5707963267948966 - lean_goal
                    _ss = math.sin(_q2g)
                    _sc = math.cos(_q2g)
                    if _sc < cos_floor:
                        _sc = cos_floor
                    _st = _ss / _sc
                    _sn = grav * (radius * _ss - half_thick * _sc)
                    _sd = (0.25 * radius * radius * _st
                           + radius * half_thick
                           + half_thick * half_thick * _st)
                    if _sn < 0.0:
                        _sn = 0.0
                    if _sd > 0.0 and near_end > 0.0:
                        _roll_now = twist * math.sqrt(_sn / _sd)
                        _fk = follow_k * decim
                        if _fk > 0.5:
                            _fk = 0.5
                        u3 += (_roll_now - u3) * _fk
                    else:
                        # The family ends at the flat limit. Below it
                        # there is no steady roll to descend to -- the
                        # coin is lying down and the face is taking the
                        # energy -- so the roll is drained again. Left
                        # following, it parked at the limit rolling at
                        # the rate that limit calls for, and no coin
                        # ever came to rest.
                        u3 *= scale
                elif absolute > 0.5 and want > mean_lean:
                    # HOLD means the gesture IS the lean, so say so
                    # rather than chase it. A feedback loop on the
                    # energy was tried and re-tuned four times: every
                    # gain that tracked one gesture shape overshot or
                    # lagged another, because the coin's own dynamics
                    # sit inside the loop. Here the disc is carried
                    # towards the steady roll the request asks for, at
                    # a fixed rate, and the physics runs on top of it.
                    q2_want = 1.5707963267948966 - want
                    carry = track_k * decim
                    if carry > 0.5:
                        carry = 0.5
                    q2 += (q2_want - q2) * carry
                    _ws = math.sin(q2)
                    _wc = math.cos(q2)
                    if _wc < cos_floor:
                        _wc = cos_floor
                    _wt = _ws / _wc
                    _wn = grav * (radius * _ws - half_thick * _wc)
                    _wd = (0.25 * radius * radius * _wt
                           + radius * half_thick
                           + half_thick * half_thick * _wt)
                    if _wn < 0.0:
                        _wn = 0.0
                    # The roll it is carried to is the one TWIST asks
                    # for, not the balanced one. Carrying it to the
                    # steady roll regardless -- and damping the lean
                    # rate on the way -- is precisely what twist 1 means,
                    # so it overwrote the control on every step and
                    # every cast sounded perfectly spun. Under-spinning
                    # is the nutation, so the wobble is left alone.
                    u3_want = twist * math.sqrt(_wn / _wd) \
                        if _wd > 0.0 else u3
                    u3 += (u3_want - u3) * carry
                    scale = 1.0
                else:
                    scale = 1.0
                u1 *= scale
                u2 *= scale

                # What the hand did since the last step. Rising, it
                # spins the coin up AND balances it -- more roll, less
                # wobble -- which is the same direction as being thrown
                # well, so enough of a rise carries the coin back to the
                # balanced spin it would have had from a clean cast, at
                # a higher rate. Falling, it takes the roll away, which
                # pushes the coin towards its end and onto its face.
                if push > 0.0:
                    boost = 1.0 + push * inject_gain
                    if boost > 1.6:
                        boost = 1.6
                    u3 *= boost
                    u2 *= boost
                    steady = 1.0 - push * balance_gain
                    if steady < 0.0:
                        steady = 0.0
                    u1 *= steady
                    # And it opens the lean itself. Speeding the roll
                    # alone is not enough: the loss law is steepest
                    # where the coin is flattest, so a coin wound up
                    # from near the floor was crushed faster than the
                    # winding could lift it. A whole sweep of the hand
                    # is worth a whole lean.
                    lean_goal += push * tilt_full
                    if lean_goal > tilt_full:
                        lean_goal = tilt_full
                    if mean_lean > lean_goal:
                        lean_goal = mean_lean
                elif push < 0.0:
                    # Taking energy OUT is not the mirror of putting it
                    # in. Winding a coin up is gradual; the only way to
                    # take speed off one is to touch it, and a hand that
                    # touches it hard stops it dead. At the same
                    # strength as the injection a full release closed
                    # the lean by its whole range and cut the roll by
                    # more than half at a stroke -- so letting go of the
                    # control killed the coin instead of hurrying it.
                    # It hurries it now: a full release roughly halves
                    # what is left rather than ending it.
                    cut = 1.0 + push * inject_gain * drain_side
                    if cut < 0.93:
                        # No more than a small bite per step, so a snap
                        # release is a hand closing on the coin over a
                        # moment rather than a single grab.
                        cut = 0.93
                    u3 *= cut
                    u2 *= cut
                    # A fraction of what is LEFT, not of the full
                    # range. Subtracting a slice of the whole range
                    # wipes out a coin whose lean is already a small
                    # part of it -- so any release late in a settle
                    # slammed the target to the floor at a stroke,
                    # whatever the gain was set to. That is why making
                    # the drain gentler changed nothing.
                    lean_goal *= 1.0 + push * drain_side
                    if lean_goal < tilt_flat:
                        lean_goal = tilt_flat
                push = 0.0

                # A coin cannot be over-spun without limit, whichever
                # way the energy arrived. Excess spin is what stands a
                # coin UP -- that is where it goes -- and the lean is
                # capped, so the spin is capped with it. This belongs to
                # the coin, not to one reading of the gesture: put it
                # only in the throwing path and holding the gesture up
                # feeds the roll every step for ever, which intensifies
                # until there is nothing left to hear but quantization.
                if u3 > u3_ceiling:
                    u3 = u3_ceiling
                elif u3 < -u3_ceiling:
                    u3 = -u3_ceiling

                # --- what the disc is doing, read off it ---
                cq = math.cos(q2)
                if cq < 1.0e-6:
                    cq = 1.0e-6
                prec_now = u3 / cq
                # Contact load, in units of the disc's own weight: the
                # centre of mass rides at radius*cos(q2), so the force
                # is what its vertical acceleration asks of the table.
                _, _, dq3, du1, _, _ = _disc_deriv(
                    q2, q3, u1, u2, u3, radius, half_thick, out_of_round,
                    grav, cos_floor, u_cap)
                # The centre of mass rides at r(q3)*cos(q2), and r is
                # moving because the contact is travelling round an
                # out-of-round rim. Its second derivative carries a
                # sweep-rate-SQUARED term, which is why an untrue coin
                # bites harder and harder as the contact accelerates.
                sq = math.sin(q2)
                # The rim, as a rim actually is. This was one cosine --
                # a coin bent once, swelling smoothly under the contact
                # every turn. Real untrueness is nicked and uneven all
                # the way round, and a rolling coin meets its OWN rim
                # again every revolution, so that unevenness repeats
                # while the table underneath is always fresh. That is
                # what puts a rhythm at the rate of spin into a sound
                # whose grains are otherwise memoryless, and what makes
                # the contact jump rather than swell: a fine feature is
                # small but it is crossed fast, and the load goes as the
                # SECOND derivative, so the little ones hit hardest.
                # Amplitudes fall as k**-1.3, which is a self-affine
                # edge; the phases are fixed per coin, so each one is
                # its own coin and stays that coin.
                # Two different faults, and they were one control. The
                # first harmonic IS eccentricity -- the rim once round,
                # a coin off-centre or squashed -- and everything above
                # it is the edge PROFILE: nicks, burrs, a milled edge
                # worn unevenly. They come from different causes and
                # they sound different, so they get a knob each.
                rim_c = 0.0
                rim_s = 0.0
                rim_a = 0.0
                for _h in range(rim_amp.shape[0]):
                    _k = _h + 1.0
                    _wg = out_of_round if _h == 0 else prof_depth
                    _ang = _k * q3 + rim_phase[_h]
                    _ca = math.cos(_ang)
                    _sa = math.sin(_ang)
                    rim_c += _wg * rim_amp[_h] * _ca
                    rim_s += _wg * rim_amp[_h] * _k * _sa
                    rim_a += _wg * rim_amp[_h] * _k * _k * _ca
                r_now = radius * (1.0 + rim_c)
                # z = r*cos(q2) + half_thick*sin(q2): the coin's centre
                # cannot descend below half its own thickness.
                rd = -radius * rim_s * dq3
                # The sweep's own acceleration contributes here too, but
                # it is smaller than this term by about seventy to one
                # (the sweep changes over seconds, its square over
                # milliseconds), so it is left out.
                rdd = -radius * rim_a * dq3 * dq3
                # The rim's own contribution to lifting the coin, kept
                # apart so the hop can be decided on it alone.
                rim_lift = -rdd * cq
                zdd = (rdd * cq - 2.0 * rd * sq * u1
                       - r_now * (cq * u1 * u1 + sq * du1)
                       + half_thick * (cq * du1 - sq * u1 * u1))
                # Past a few times its own weight this is the
                # integrator near the singularity, not a coin -- but a
                # hard ceiling FLAT-TOPS, and a load pinned at its limit
                # for tens of milliseconds is heard as a burst with a
                # square edge on it. Bend it over instead: identity
                # while the coin is behaving, asymptotic once it is not.
                # Only the excess is bent, so a resting coin still reads
                # exactly its own weight.
                excess = zdd / grav
                if excess > 0.0:
                    excess = load_head * math.tanh(excess / load_head)
                held = 1.0 + excess
                # No adhesion: past zero the contact does not merely
                # stop pushing, it LETS GO -- and a coin that has let go
                # is in the air. Clamping at zero kept it glued to the
                # table through the very thing that lifts it.
                #
                # The rim forcing carries dq3 SQUARED, so a coin spun
                # hard rides its own unevenness far harder than a slow
                # one does and chatters, once per turn, off whatever is
                # least true about it. That is why the jumping is
                # loudest the moment a coin is thrown and fades as it
                # slows -- and why holding a fast spin brings it back.
                #
                # The attitude keeps running while the coin is off the
                # table, which is not exact: in the air there is no
                # contact and so no rolling constraint. The hops last
                # under a millisecond, and what is heard is the landing.
                if hop_h > 0.0:
                    hop_v -= grav * dt
                    hop_h += hop_v * dt
                    if hop_h <= 0.0:
                        hop_h = 0.0
                        if hop_v < 0.0:
                            amp = strike_scale * hop_gain * (-hop_v)
                            if amp > 3.0:
                                amp = 3.0
                            # A finished slap must not go on blocking
                            # the next arrival: slap_amp is never
                            # cleared, so comparing against a spent one
                            # silenced every hop after the first blow.
                            if slap_at >= slap_len or amp > slap_amp:
                                slap_amp = amp
                                slap_at = 0.0
                                slap_len = contact * 2.0
                        hop_v = 0.0
                # The rim repeats every revolution but the TABLE does
                # not: the coin comes down somewhere new each time. With
                # only the rim deciding, the chatter was a loop playing
                # over and over. What the surface adds is scaled by how
                # rough it is, which is what 'scrape' already means.
                rng, _tb = _rand01(rng)
                lift_now = rim_lift * (1.0 + table_rough * (2.0 * _tb - 1.0))
                if lift_now > grav:
                    # What throws the coin off the table is the RIM,
                    # not the total load. The rest of that load carries
                    # du1, and du1 carries the control loop's own kicks
                    # -- the drain scaling the rates, the rebound
                    # reversing them in a single sample -- which are not
                    # accelerations any coin undergoes. Deciding contact
                    # break on those had it exactly backwards: the
                    # chatter grew as the coin slowed and a faster held
                    # spin produced LESS of it.
                    #
                    # The rim term is clean and says what a coin says:
                    # it goes as the roll SQUARED, as the sine of the
                    # lean, and as how untrue the rim is. Fast, open and
                    # bent chatters; slow, flat or true does not.
                    hop_v += (lift_now - grav) * dt
                    if hop_v > 0.0:
                        hop_h = hop_v * dt
                load_now = held
                if load_now < 0.0:
                    load_now = 0.0
                if hop_h > 0.0:
                    # In the air there is nothing to grind against.
                    load_now = 0.0
                # How fast the contact travels around the rim, which is
                # the rate it meets new surface.
                sweep_now = u2 - math.tan(q2) * u3
                if sweep_now < 0.0:
                    sweep_now = -sweep_now

                # (A landing check stood here, left over from before
                # the face contact learned to rebound. It fired on the
                # same sample as the rebound -- which clamps the lean to
                # exactly this value -- so every bounce was converted
                # into a landing and the coin stopped on first contact.
                # The rebound above is the one that decides.)

            hertz = prec_now / two_pi
            lean = 1.5707963267948966 - q2
            if lean < 1.0e-9:
                lean = 1.0e-9
            # The load here is the real one, computed from the disc's
            # own vertical acceleration -- so unlike the voiced model
            # there is nothing to scale it by. The equations describe a
            # perfectly uniform disc, which means the surges come
            # entirely from the CAST and the coin's own untrueness is
            # not represented at all. 'wobble' has no physics to attach
            # to here, and does nothing.
            press = load_now
            # Grinding is what the RIM does. As the lean closes on the
            # limit the face comes down and takes the load off the rim,
            # and once the coin is bouncing on its face the rim is not
            # touching anything at all -- so there is nothing to grind.
            # Without this the sweep term, which carries a tan(q2) and
            # so runs away exactly there, kept driving a grind through
            # every face bounce: hundreds of hertz one instant and a few
            # the next, which is the noise heard at the end of a settle
            # and is not any coin. The face has its own voice already,
            # and it is the slap.
            # Grinding is what the RIM does, and the rim stops being a
            # rim once the coin is lying almost flat: at a lean of the
            # order of its own thickness-over-radius the contact is no
            # longer a point running round an edge, and the equations'
            # tan(q2) -- which multiplies the roll by a growing factor
            # every time the lean dips -- stops describing a coin. Left
            # open, one rock down to a fifteenth of a degree took the
            # roll from 1.4 to 5.5 and the sweep from 270 to 4130 in
            # three milliseconds, and the square root of that arrived
            # as a click. The coin still has a voice down there; it is
            # the face, and the face has the slap.
            flat = lean / flat_ref
            flat *= flat
            rim = flat / (1.0 + flat)
            # And while the FACE is down it is the face carrying the
            # coin, not the rim -- the rim takes the load back as it
            # rocks up off it. Without this the grind read the rebound
            # as a load surge: the bounce reverses the lean rate in a
            # single sample, so the vertical acceleration through it is
            # a numerical artefact of that discontinuity and drove the
            # load into its ceiling every time. The blow is already
            # spoken for; it is the slap.
            if slap_at < slap_len:
                rim *= slap_at / slap_len
            rolling = ripple_gain * math.sqrt(lean) * (press - 1.0) * rim
            grip = scrape_gain * math.sqrt(sweep_now / sweep_ref) \
                * press ** load_exp * rim
            edge = sharpen * (press - 1.0)
            if edge < 0.0:
                edge = 0.0
            elif edge > 1.0:
                edge = 1.0
            # Grain density follows the sweep: a contact that travels
            # meets new surface and fuses into a hiss, one that does not
            # arrives as separate hits.
            dens = grain_dens * (sweep_now / sweep_ref) * press
            if dens > 0.4:
                dens = 0.4
            elif dens < grain_floor:
                dens = grain_floor
            grain_amp = 1.0 / math.sqrt(dens)
            if grain_amp > grain_ceiling:
                grain_amp = grain_ceiling
            # The same asperity takes less time to cross when the
            # contact is travelling faster, so a slow scrape is darker
            # than a fast one -- floored, or a nearly stopped contact
            # would smear its grains into nothing.
            swp = sweep_now / sweep_ref
            if swp < 0.15:
                swp = 0.15
            dur_k = grain_span / swp
        else:
            grip -= grip * stop_k
            dens = grain_floor
            grain_amp = 1.0
            dur_k = grain_span
        # The surface a contact crosses is CONTINUOUS, and a self-affine
        # one has structure at every scale -- so the roughness it meets
        # drifts, in patches, on every timescale at once. Drawing each
        # grain independently of the last makes that drift white by
        # construction: statistically impulsive, and still a featureless
        # hiss to listen to, because nothing varies for longer than one
        # grain. Four one-poles spanning half a millisecond to half a
        # second, summed, give roughly the 1/f the surface has, and the
        # grains ride it.
        rng, _rn = _rand01(rng)
        _wn = 2.0 * _rn - 1.0
        texture = 0.0
        for _q in range(rough_k.shape[0]):
            rough_state[_q] += (_wn - rough_state[_q]) * rough_k[_q]
            texture += rough_state[_q]
        texture = 1.0 + rough_depth * texture * rough_norm
        if texture < 0.0:
            texture = 0.0
        if grip > 1.0e-9:
            rng, u = _rand01(rng)
            if u < dens:
                # A real surface is self-affine: its asperities have no
                # characteristic size, so the impulses a contact takes
                # from them follow a power law rather than all being
                # much the same. Drawing them uniformly was the reason
                # this sounded like noise and not like friction --
                # dense impulses of similar size ARE noise, by the
                # central limit theorem, at any density whatever.
                #
                # Below a tail index of two the variance diverges and
                # sums stop converging on a Gaussian, which is what
                # keeps the big events individually audible over the
                # hiss of the small ones however fast the contact runs.
                rng, u = _rand01(rng)
                if u < 1.0e-9:
                    u = 1.0e-9
                size = u ** (-grain_tail)
                if size > grain_max:
                    size = grain_max
                rng, u = _rand01(rng)
                peak = size * grain_amp * grain_norm * texture
                if u <= 0.5:
                    peak = -peak
                # An asperity is as WIDE as it is tall, by the same
                # self-affinity that set its height, and the contact
                # crosses it at a finite speed -- so a big grain takes
                # longer. Its momentum is what it is; delivering that
                # over a longer time makes it lower, not sharper. Firing
                # every grain as one sample instead made the rare huge
                # ones perfect impulses, which is exactly the sound that
                # does not belong: a click, flat to Nyquist, where the
                # grinding is narrowband and dark. A faster sweep
                # crosses the same feature sooner.
                span = size ** hurst_inv * dur_k
                dur = int(span)
                # Nothing the contact does can be sharper than the
                # contact itself: it has stiffness and the coin has
                # mass, so it cannot answer a feature crossed quicker
                # than it can deform. A hard coin on stone floors at a
                # sample or two; a soft one simply cannot make a sharp
                # grain, whatever it runs over. Without this floor the
                # sweep term shortened every grain back to a single
                # sample exactly where the coin spins fastest -- which
                # is where the clicks were heard.
                if dur < grain_least:
                    dur = grain_least
                if dur > ring_n - 1:
                    dur = ring_n - 1
                if dur == 1:
                    grain_ring[head] += peak
                else:
                    # Held momentum, spread under a raised cosine: the
                    # area is the same whatever the duration, so the
                    # peak of a long grain falls instead of towering.
                    scale = 2.0 * peak / dur
                    at = head
                    for k in range(dur):
                        grain_ring[at] += scale * 0.5 * (
                            1.0 - math.cos(two_pi * (k + 0.5) / dur))
                        at += 1
                        if at >= ring_n:
                            at = 0
            noise = grain_ring[head]
            grain_ring[head] = 0.0
            head += 1
            if head >= ring_n:
                head = 0
            lp += (noise - lp) * scrape_k
            base = noise - lp
            hp += (base - hp) * sharp_k
            bright = (base - hp) * bright_norm
            rolling += grip * (base + edge * (bright - base))
        if slap_at < slap_len:
            struck += slap_amp * 0.5 \
                * (1.0 - math.cos(two_pi * slap_at / slap_len))
            slap_at += 1.0
        out[i] = rolling + struck
        rate_out[i] = hertz
        face_out[i] = math.cos(q3)
        grind_out[i] = rolling
        strike_out[i] = struck
    return (q1, q2, q3, u1, u2, u3, gesture_last, push,
            lean_goal, mean_lean, load_now,
            sweep_now, prec_now, slap_amp, slap_at, slap_len, grip, lp,
            hp, edge, landed, hop_v, hop_h, rng, head)


if _HAVE_NUMBA:
    _spin_real_kernel = njit(cache=True, fastmath=True)(
        _spin_real_kernel_source)
else:
    _spin_real_kernel = _spin_real_kernel_source


def _spin_kernel_source(gesture, tilt_full, tilt_flat, lift_k, loss, law,
                        rate_scale, rate_ceiling, contact, strike_scale,
                        grain_tail, grain_max, grain_norm,
                        wobble, ecc, orbit_norm, nut_ratio, nut_decay,
                        nut_max, twist, grain_dens, grain_floor,
                        grain_ceiling, step_ref, face_slap, ripple_gain,
                        scrape_gain, scrape_k,
                        load_exp, sharpen, sharp_k, bright_norm, stop_k,
                        sample_rate, tilt, phase,
                        face, profile, nut_phase, nut_amp, rim_down,
                        slap_amp, slap_at, slap_len, grip, lp,
                        hp, edge, landed, rng,
                        out, rate_out, face_out, grind_out, strike_out):
    """A disc settling towards flat, integrated at audio rate.

    The tilt angle is the whole state, and one law does all the work: a
    disc rolling on its rim has its contact point race around that rim
    at a rate going as one over the square root of the tilt. So as the
    disc gives up its lean the sound rises without limit -- the runaway
    everyone has heard a dropped coin make. Nothing here is a ramp or an
    envelope; the acceleration is that exponent, the way bounce~'s
    quickening is gravity.

    A rolling disc does not strike anything, and -- tested against real
    coins -- it hardly ever leaves the table either. The contact is
    continuous and it is the LOAD on that contact that varies, so what
    is heard is one sound modulated by the rotation. Two rotations
    modulate it, and they move opposite ways. The rim passes under the
    contact at the precession rate, which is rising: that ripple in the
    load is the pitch. The disc's own mass sits unevenly and comes
    round at the face rate, which is FALLING, since the face turns at
    the tilt times the precession rate: that is the slow waver in
    intensity, dragging while the pitch runs away.

    An off-kilter coin flops hard onto its rim once a turn, and that
    flop is NOT a blow. The contact stays down; it is simply carrying
    several times the weight for a moment, and a loaded contact engages
    more of the surface and stiffens against it. So the grinding grows
    faster than the load and brightens as it grows -- a sharpening, not
    an impact. Deep wobble therefore buys a harder, sharper grind in
    the part of each turn where the heavy side is down, which is what a
    real coin does, and no separate mechanism is needed for it.

    Everything scales as the square root of the tilt, so the sound
    thins as it quickens: higher, denser and smaller, ending in the one
    real impact there is, when the face finally lands flat.
    """
    two_pi = 6.283185307179586
    for i in range(gesture.shape[0]):
        want = tilt_full * gesture[i]
        if want > tilt:
            if landed > 0.5:
                # Set down at that lean. A landed disc must not glide up
                # through every pitch on its way to being spun again.
                tilt = want
            else:
                tilt += (want - tilt) * lift_k
            if tilt > tilt_flat:
                landed = 0.0
            # Energy going in is a fresh cast of the coin, so it brings
            # its own unsteadiness with it: all the wobble a launch has
            # is the spin it was NOT given.
            fresh = (1.0 - twist) * nut_max
            if fresh > nut_amp:
                nut_amp = fresh
        hertz = 0.0
        rolling = 0.0
        struck = 0.0
        if landed < 0.5:
            # The losses, as a power of the lean that is left: which
            # power is the shape of the whole tail.
            tilt -= loss * tilt ** (-law)
            if tilt <= tilt_flat:
                tilt = 0.0
                landed = 1.0
                # Going flat is the one real impact: the whole face
                # lands at once, so it is broader and duller than
                # anything the rim does -- and only as loud as the lean
                # it had left, which is why a rough surface stops a coin
                # with a clack and a polished one lets it whisper away.
                slap_amp = strike_scale * math.sqrt(tilt_flat) * face_slap
                slap_at = 0.0
                slap_len = contact * 3.0
            else:
                # NUTATION -- the tilt itself swinging, and the whole
                # difference between a coin set spinning and a coin
                # merely dropped.
                #
                # Launch a disc into steady precession and it has none:
                # the contact drifts round the rim in a regular way and
                # the lean falls smoothly. Launch it short of that
                # condition and the lean oscillates instead, and since
                # the rate goes as one over the square root of the
                # lean, an oscillating lean warbles the RATE as well as
                # the load -- which is why a badly cast coin sounds
                # uneven in pitch and not merely uneven in loudness.
                #
                # Push it far enough and the lean swings all the way to
                # flat once a cycle. Then the face is what touches, the
                # rim is not rolling at all, and the coin rattles to a
                # stop rather than whirring: the no-spin drop, arrived
                # at from the same equations.
                base_step = rate_scale / math.sqrt(tilt)
                nut_phase += base_step * nut_ratio
                if nut_phase >= 1.0:
                    nut_phase -= 1.0
                dip = 0.5 * (1.0 - math.cos(two_pi * nut_phase))
                tilt_now = tilt * (1.0 - nut_amp * dip)
                if tilt_now < tilt_flat:
                    # The lean has swung right down: the face is on the
                    # table. This is the only place the rim genuinely
                    # leaves it, and it is a badly-cast coin's rattle
                    # rather than anything a well-spun one does.
                    tilt_now = tilt_flat
                    if rim_down < 0.5:
                        rim_down = 1.0
                        slap_amp = strike_scale * math.sqrt(tilt) \
                            * face_slap * 0.6
                        slap_at = 0.0
                        slap_len = contact * 2.0
                else:
                    rim_down = 0.0
                step = rate_scale / math.sqrt(tilt_now)
                if step > rate_ceiling:
                    step = rate_ceiling
                hertz = step * sample_rate
                face += step * tilt_now
                if face >= 1.0:
                    face -= 1.0
                # A badly spun coin does not trace a circle. Its contact
                # runs an eccentric orbit -- racing where the orbit pulls
                # tight, dawdling where it runs wide -- so the turn is
                # not uniform in time. Angular momentum does it: the rate
                # goes as one over the radius squared. And the tight side
                # is fixed in the coin's own body, so it drifts round at
                # the face rate rather than sitting still, which is why
                # the lurch never quite repeats.
                #
                # This is what makes a pulse that is symmetric in ANGLE
                # land lopsided in TIME. Nothing below is asymmetric; the
                # asymmetry is entirely in how fast the coin gets there.
                orbit = 1.0 + ecc * math.cos(two_pi * (phase - face))
                swing = orbit * orbit / orbit_norm
                phase += step * swing
                if phase >= 1.0:
                    phase -= 1.0
                # A disc left alone finds its steady precession, so the
                # swing in the lean bleeds away faster than the lean
                # itself does. What is cast badly does not stay bad
                # forever -- it merely starts that way.
                nut_amp -= nut_amp * nut_decay
                # Where the disc's own weight sits, coming round on the
                # face's slowing turn. Two harmonics, so it wavers
                # unevenly the way a real disc does.
                profile = (0.7 * math.cos(two_pi * face)
                           + 0.3 * math.cos(2.0 * two_pi * face + 0.7))
                # How unevenly the disc is made, as the swing in how
                # hard it presses. This reaches far past its own weight,
                # because an off-kilter coin flops onto its rim with
                # several times the load of a true one.
                depth = wobble * (0.6 + 0.4 * profile)
                # The rim passing under the contact: peaked, because a
                # contact riding over a high spot is not a sine, and the
                # harmonics of that peak are what give the whir a pitch
                # rather than just a rate. It presses hardest on the
                # tight side of the orbit, where it is also travelling
                # fastest -- so the flop is short, hard and early, and
                # the rest of the turn is the long easy part between.
                bump = 0.5 * (1.0 + math.cos(two_pi * phase))
                # It presses harder on the tight side, but only somewhat:
                # coupling the load to the orbit at full strength stacks
                # two multiplications on the same peak and the flop turns
                # into a detonation. Most of the lurch belongs in the
                # timing, which is where a real coin keeps it.
                press = 1.0 + (swing - 1.0) * 0.35
                load = 1.0 + depth * (2.0 * bump * bump - 1.0) * press
                if load < 0.0:
                    # Unloaded, but NOT airborne: a settling coin hardly
                    # ever leaves the table. It merely stops pressing,
                    # and a contact carrying no weight makes no sound.
                    load = 0.0
                # The ripple in the load is the sound: the disc shaken
                # once a revolution by its own rim.
                if rim_down > 0.5:
                    # The rim is off the table; only the face is on it.
                    load = 0.0
                rolling = ripple_gain * math.sqrt(tilt_now) * (load - 1.0)
                # And the roughness under it, heard at the speed the
                # contact travels and pressed by that same load. The
                # press does not merely scale it: a loaded contact
                # engages more of the surface and stiffens against it,
                # so it grows FASTER than the load and brightens as it
                # grows. That is what the hard flop of an off-kilter
                # coin actually is -- not a blow, but the grinding gone
                # sharp for the part of the turn that its heavy side is
                # down.
                grip = scrape_gain * math.sqrt(step) * load ** load_exp
                # How finely the contact is divided. A rolling rim
                # travels, so it meets new surface constantly and the
                # grains fuse; a coin with no spin does not travel at
                # all and each contact is a single hit. Load matters
                # too: a pressed contact touches more of the surface at
                # once. Grain size follows density so the power does
                # not, which is what makes the sparse end read as hits
                # rather than as a quieter hiss.
                dens = grain_dens * twist * (step / step_ref) * load
                if dens > 0.4:
                    dens = 0.4
                elif dens < grain_floor:
                    dens = grain_floor
                grain_amp = 1.0 / math.sqrt(dens)
                if grain_amp > grain_ceiling:
                    grain_amp = grain_ceiling
                # Colour, kept separate from level. Raising a filter's
                # corner brightens by REMOVING what is below it, which
                # thins a sound rather than sharpening it -- so the
                # loudness comes from the load above, and this only
                # crossfades the noise towards its own high end, level
                # matched. Sharper, not merely thinner.
                edge = sharpen * (load - 1.0)
                if edge < 0.0:
                    edge = 0.0
                elif edge > 1.0:
                    edge = 1.0
        else:
            # Landing does not cut the grinding off. Silencing a noise
            # in one sample is a step, and a step is a click -- so the
            # contact loses its grip over a couple of milliseconds,
            # which is short enough to still read as stopping dead.
            grip -= grip * stop_k
        if grip > 1.0e-9:
            # A contact is not a hiss, it is a stream of micro-impacts,
            # and whether it is HEARD as a hiss or as separate hits is
            # only their density. A rim rolling along the surface sweeps
            # thousands of asperities a second and fuses into noise; a
            # face coming down delivers one. Density is the whole
            # difference, so it is what twist moves -- and power is held
            # constant across it, which makes sparse grains big and
            # dense ones small, exactly as the same energy arriving in
            # fewer pieces must.
            #
            # It also fixes something white noise cannot: a contact is
            # granular and uneven, and noise is far too regular to be
            # one.
            rng, u = _rand01(rng)
            if u < dens:
                # A real surface is self-affine: its asperities have no
                # characteristic size, so the impulses a contact takes
                # from them follow a power law rather than all being
                # much the same. Drawing them uniformly was the reason
                # this sounded like noise and not like friction --
                # dense impulses of similar size ARE noise, by the
                # central limit theorem, at any density whatever.
                #
                # Below a tail index of two the variance diverges and
                # sums stop converging on a Gaussian, which is what
                # keeps the big events individually audible over the
                # hiss of the small ones however fast the contact runs.
                rng, u = _rand01(rng)
                if u < 1.0e-9:
                    u = 1.0e-9
                size = u ** (-grain_tail)
                if size > grain_max:
                    size = grain_max
                rng, u = _rand01(rng)
                if u > 0.5:
                    noise = size * grain_amp * grain_norm
                else:
                    noise = -size * grain_amp * grain_norm
            else:
                noise = 0.0
            lp += (noise - lp) * scrape_k
            base = noise - lp
            # The sharp tap sits high enough to be heard AS sharpness.
            # Trimming a few hundred hertz off noise that already runs to
            # twenty kilohertz changes nothing anyone can hear; the edge
            # of a loaded contact lives up where the surface is being
            # torn at, so this pole is in the kilohertz, and it is
            # scaled back up to sit level with what it replaces.
            hp += (base - hp) * sharp_k
            bright = (base - hp) * bright_norm
            rolling += grip * (base + edge * (bright - base))
        if slap_at < slap_len:
            struck += slap_amp * 0.5 \
                * (1.0 - math.cos(two_pi * slap_at / slap_len))
            slap_at += 1.0
        out[i] = rolling + struck
        rate_out[i] = hertz
        face_out[i] = profile
        grind_out[i] = rolling
        strike_out[i] = struck
    return (tilt, phase, face, profile, nut_phase, nut_amp, rim_down,
            slap_amp, slap_at, slap_len, grip, lp, hp, edge, landed, rng)


if _HAVE_NUMBA:
    _spin_kernel = njit(cache=True, fastmath=True)(_spin_kernel_source)
else:
    _spin_kernel = _spin_kernel_source


class SpinUnit(Unit):
    """A spinning disc settling: the sound that runs away.

    A coin dropped on a table, a plate set down spinning, a hubcap in
    the road. All of them do the same thing, and it is not a bounce:
    the contact point races around the rim at a rate that goes as one
    over the square root of the tilt, so as the lean bleeds away the
    sound accelerates without limit and then stops dead. That runaway
    is this unit's entire mechanism.

    Nor is it a series of blows, and a settling coin hardly ever leaves
    the table at all. A rolling disc keeps its contact; what varies is
    the LOAD on it, so this is one continuous sound modulated by the
    rotation. Two rotations modulate it and they move opposite ways:
    the rim passes under the contact at the precession rate, which
    rises, and that ripple is the pitch; the disc's own weight comes
    round at the face rate, which FALLS, and that is the slow waver in
    intensity.

    The hard flop of a well-spun but off-kilter coin is not an impact
    either -- the contact stays down and simply carries several times
    the weight for a moment. A loaded contact engages more of the
    surface and stiffens against it, so the grinding grows faster than
    the load and brightens while it grows. That sharpening IS the flop.

    But roughness is mostly not the coin's doing at all: it is the
    CAST. Spin a coin true on its edge and it rolls -- the contact
    drifts round the rim regularly, the lean falls smoothly, nothing
    ever leaves the table. Push one flat over with no spin and it
    never rolls at all: the lean swings past level every cycle, the
    face slaps, and it rattles to a stop. 'twist' is that continuum,
    and it works by nutation -- the lean itself oscillating rather than
    falling smoothly. Because the rate goes as one over the square root
    of the lean, an oscillating lean warbles the PITCH as well as the
    load, which is why a badly thrown coin sounds unsteady and not just
    uneven. And since it is spin that holds a disc in steady
    precession, the swing dies away in proportion to the twist it was
    given: with none, there is nothing to settle into.

    'spin' is the gesture, and it can only add energy: raising it leans
    the disc over that far, and everything after is loss. Hold it and
    the sound holds, at a pitch its own level sets; let go and the disc
    settles from wherever it had got to. A big gesture is a long low
    fall, a small one a short high one. This is why movement into
    'spin' gives a sound with a tail: the tail is what the disc does
    when the hand has stopped.

    'size' is the disc's radius in metres and sets every rate -- a coin
    turns about ten times a second at full lean, a dinner plate four.
    'settle' is the seconds from full lean to flat. 'rush' is which
    loss dominates, and so where in the tail the acceleration lives: at
    0 the loss is proportional to energy and the pitch glides up evenly
    across the whole tail, at seven tenths it is rolling friction and
    the rise leans towards the end, and at 1 it is the viscous-air
    exponent Moffatt derived for the toy, which holds the pitch almost
    still and then spends the last per cent of the settle on the entire
    scream. That last is honestly what a Euler disc does, and it wants
    long settle times to be heard as anything but a chirp.

    'wobble' is how unevenly the disc is made -- the swing in how hard
    it presses, reaching several times its own weight at full. It buys
    the tone (a true disc has no ripple to hear) and it buys the flop.
    'scrape' is the roughness under the contact, 'hardness' how sharply
    that roughness answers a load, and 'polish' how near flat it gets
    before it lands -- how high the whir climbs, and whether the end is
    a clack or a vanishing.

    Five outlets, because the dynamics are worth more than the sound.
    'out' is everything; 'grind' is the rolling, which is nearly all of
    it, and 'landing' carries the single impact at the end, alone, so
    it can have its own resonator and its own gain. 'rate' is the
    precession frequency in Hz, rising, and 'face' the disc's own
    profile coming round, falling -- two counter-moving controls out of
    one gesture.

    'landing' fires once per settle and is silent the rest of the time.
    That is not a fault to be corrected: there is exactly one impact in
    the life of a settling disc, and this is it.
    """

    # Full lean is half a radian, about thirty degrees: past that a disc
    # is falling over rather than settling, and the small-angle rate law
    # this is built on has stopped meaning anything.
    TILT_FULL = 0.5
    # What 'wobble' at 1 means as a swing in contact load: six times the
    # disc's weight at the peak of the heavy side. Set by ear against
    # real coins -- the flop of an off-kilter one is a big multiple of
    # its own weight, not a fraction of it.
    WOBBLE_DEPTH = 6.0
    # Two ways to be a coin. 'derived' integrates the disc's own
    # equations of motion and reads the sound off the result; 'voiced' is
    # the earlier model, whose behaviours were assembled and fitted by
    # ear. The first is the truth; the second is kept because it is what
    # patches saved before this expect, and because a thing that sounds
    # right is worth being able to go back to.
    MODELS = ('derived', 'voiced')
    # Two ways to read 'spin'. 'throw' works by CHANGE: rising injects,
    # falling drains, holding still does nothing and the coin settles --
    # a gesture throws a coin. 'hold' works by LEVEL: the gesture is the
    # lean it asks for and sustained motion keeps the coin going. They
    # are different instruments and both are worth having.
    SPIN_MODES = ('throw', 'hold')
    # How often the disc is stepped, in samples. Its motion tops out in
    # the hundreds of hertz, so a few kilohertz is ample and the audio
    # reads between steps.
    CONTROL_DECIM = 4
    # Half a coin's thickness as a fraction of its radius: a coin about
    # 1.75 mm thick with a 12 mm radius.
    COIN_ASPECT = 0.073
    # How much a unit of gesture movement is worth as energy, per
    # control step, and how much of it goes to balancing the spin
    # rather than merely speeding it. Rising fast enough should carry a
    # wobbling coin back to a clean balanced spin, which is what a hand
    # winding one up actually does.
    # How far from its goal the lean may sit before anything acts on
    # it. Without a dead zone the drain and the pump hunt each other.
    DEAD_ZONE = 0.18
    # How hard the hold tracks the gesture. This has to be strong
    # enough to FOLLOW the control, not merely lean against it: at a
    # tenth of this the coin was cast partway up a ramp, the target
    # jumped ahead of it, and the coin collapsed to its end and had to
    # be dragged back -- heard as an acceleration and a resurrection
    # every time the control was slid rather than jumped. Above about
    # five times it overshoots instead.
    # How quickly a held gesture carries the coin to the lean it asks
    # for, per control step -- about a tenth of a second.
    # Per sample; the kernel multiplies by the decimation.
    TRACK = 0.00025
    # How fast the lean the drain and pump work against is averaged,
    # per sample. Also multiplied by the decimation in the kernel.
    LEAN_AVERAGE = 0.0025
    # How fast the roll is carried to the one its lean calls for, per
    # sample. This is what makes a settling coin get LOUDER: the lean
    # closes, the steady roll for that lean brings the precession up
    # with it, and the grind rides the square root of the contact sweep.
    FOLLOW = 0.0004
    # How loudly a coin lands from one of its own hops. A fast spin
    # rides its rim hard enough to throw itself off the table once a
    # turn; this says what that arrival is worth against a face slap.
    # A coin throwing itself off the table is an ACCENT on the grinding,
    # not the sound itself. At 600 the landings ran to half again the
    # grind's own peak and simply took the piece over.
    #
    # The trigger is a THRESHOLD, so 'profile' cannot fade it in: below
    # the lift that clears gravity there is nothing at all, and above it
    # the blows arrive at full size. That is the coin and not a fault --
    # it either leaves the table or it does not -- but it does mean the
    # knob cannot be used to balance them. What can is the outlets:
    # 'grind' and 'landing' come out separately, so the two can be
    # weighed against each other outside and only 'out' has them
    # pre-mixed.
    HOP = 5.0
    # How unevenly the TABLE is made, as a fraction of the rim's own
    # lift, at full 'scrape'. Without it the chatter is the rim profile
    # played round and round without variation.
    TABLE_ROUGH = 0.7
    # How much the roughness the contact meets drifts as it travels.
    # Zero is a surface machined identically everywhere, which is what
    # independent grains amount to and what makes them read as hiss.
    # Low on purpose. The mechanism is real -- a travelling contact does
    # meet the surface in patches -- but measured against a recording of
    # real rolling coins the grind ALREADY drifts more than they do
    # (0.18 against 0.16 over 20 ms, where gaussian noise is 0.02), so
    # there is no gap here to close and turning this up only walks away
    # from the reference. It is a knob, not a correction.
    TEXTURE = 0.3
    # The timescales that drift happens on, in seconds. Spanning three
    # decades is what makes it scale-free rather than a wobble at one
    # rate.
    TEXTURE_TIMES = (0.0005, 0.005, 0.05, 0.5)
    # How many bumps round the rim. Low order only: this is the coin's
    # SHAPE -- bent, nicked, out of true -- not its micro-roughness,
    # which the grains already carry.
    RIM_HARMONICS = 8
    # Self-affine edge. The load rides the second derivative, so the
    # k-squared there outruns this and the fine features hit hardest,
    # which is the jumping rather than swelling.
    RIM_FALL = 1.3
    # How much weaker taking energy out is than putting it in.
    # Taking energy out is weaker than putting it in, and deliberately
    # so: winding a coin up is gradual, while the only way to take
    # speed off one is to touch it. At full strength a release shortens
    # what is left by about half; this is gentler than that, so letting
    # go of the control hurries the coin rather than ending it.
    DRAIN_SIDE = 0.6
    INJECT_GAIN = 2.5
    BALANCE_GAIN = 1.5
    # How much accumulated rise counts as a throw, and how far over that
    # throw puts the coin. The gain is one so that the whole of a
    # gesture maps to the whole of the lean -- above one it saturates,
    # and every throw past halfway casts the coin identically, which is
    # a control with no range in it.
    #
    # Note which way this runs: a BIGGER throw stands the coin up more,
    # and the rate goes as one over the root of the lean, so it starts
    # SLOWER -- and then sweeps further and lasts longer. A gentle throw
    # lands the coin nearly flat, where it buzzes fast and briefly.
    # That is the physics rather than the intuition.
    CAST_LEAST = 0.15
    CAST_GAIN = 1.0
    # How hard energy is pulled out to follow the settle law.
    # Gentle enough to be stable, firm enough to follow the settle law.
    # Measured: an order of magnitude above this and the disc is slammed
    # into its own singularity; an order below and the tail runs long.
    # Per SAMPLE, not per control step -- the kernel multiplies by the
    # decimation. Written per step it was eight times stronger at the
    # default rate than at the finest, which is how the integration rate
    # came to set what 'settle' meant. This is the old value divided by
    # the default decimation, so the coin behaves as it was tuned to.
    DRAIN = 0.00025
    # How much of the lean's arrival survives a face contact. A coin
    # flopping onto its face keeps a good deal of it, which is why a
    # badly cast one goes on flopping instead of stopping dead.
    REBOUND = 0.62
    # The nutation a cast withholds, as a fraction of the roll speed.
    # Now that a face contact rebounds instead of ending the disc, this
    # can be what the reference simulation says a real wobble is, rather
    # than what the integrator would tolerate.
    NUT_KICK = 0.35
    # The derived model's load is the real one, in units of the disc's
    # weight, so its raw scale has nothing to do with the voiced model's
    # invented one. This brings the two into line so switching between
    # them is a change of character and not of volume.
    DERIVED_GAIN = 0.004
    # How far out of round the contact's orbit is pulled at full wobble.
    # At 0.6 the contact runs some sixteen times faster through the tight
    # side of the turn than the wide one, which is a coin lurching badly
    # -- and the pulse it makes is short and hard where it races, long
    # and easy where it does not.
    ORBIT_ECCENTRIC = 0.6
    # How fast the lean swings against how fast the contact goes round.
    # This was invented by ear at 0.5. It is now MEASURED, from a
    # rolling-disc simulation derived by Kane's method and validated two
    # ways -- energy conserved to one part in 10^12, and steady
    # precession reproducing rate^2 = 4g/(radius*tilt) to better than
    # 0.1% at small tilt, a law that appears nowhere in its derivation.
    # Perturbing a steady roll and counting the lean's oscillations gives
    # 0.775, and remarkably flatly: 0.774 at two degrees of tilt, 0.777
    # at ten, 0.804 at thirty. The ear was 55% low.
    # See dpg_system/tests/disc_reference.py.
    NUTATION_RATIO = 0.775
    # And how quickly a disc finds its steady precession: a third of the
    # settle, so a bad cast is heard through the first half and is gone
    # by the end, which is what a coin does.
    NUTATION_SETTLE = 0.33
    # How far past level the lean can swing when a coin is merely pushed
    # over. It has to exceed one, or the face touches the table for a
    # single sample per cycle instead of lying on it for a real arc of
    # one -- which is the difference between a rattle and nothing.
    NUTATION_MAX = 1.8
    # Grains per sample at a full-lean roll: high enough that a rolling
    # contact fuses into a continuous sound rather than being heard as
    # separate events.
    GRAIN_DENSITY = 0.35
    # And the sparse end, where a contact is a handful of hits a second.
    # Also the floor that keeps grain size finite.
    GRAIN_FLOOR = 0.0004
    # A self-affine surface ties an asperity's height to its WIDTH: over
    # a lateral distance L the height wanders by about L**H. So a grain
    # of height h is about h**(1/H) wide, and the contact -- travelling
    # at a finite speed -- takes that much longer to cross it. Giving
    # every grain the same one-sample life, whatever its size, turned
    # the big ones into perfect impulses: flat to Nyquist, and audibly
    # nothing to do with the grinding around them. A big feature is
    # LOWER, not brighter. H near 0.8 is the usual measurement for worn
    # and machined surfaces.
    GRAIN_HURST = 0.8
    # The longest a grain may run, in samples at the reference sweep.
    # Also the length of the overlap-add ring the kernel schedules into.
    GRAIN_RING = 64
    # How much longer the softest contact's shortest grain is than the
    # hardest one's. Soft contacts cannot make sharp events at all.
    GRAIN_SOFT = 7.0
    # The most the contact load is allowed to reach, in coin weights.
    # Approached smoothly rather than run into.
    LOAD_MAX = 5.0
    # How the grinding answers that load. Amontons: friction force is
    # proportional to the normal load, and rough-contact models put the
    # exponent between two thirds and one, never above. Raise it and
    # every flop turns into a burst.
    LOAD_EXP = 1.0
    # The lean below which the rim stops being a rim, as a fraction of
    # the coin's own half-thickness over its radius. Below this the face
    # is coming down and the rolling description gives out.
    #
    # OFF, because what it was put in to hide is fixed at the source.
    # It was covering a coin that ROCKED through near-flat -- the roll
    # drained away from under it -- and the clicks that came of it. Now
    # the roll follows the lean down the steady family the coin
    # precesses instead, and the valve only cancels the crescendo that
    # descent is FOR: at 0.12 it turned a 16 dB rise into a 26 dB fall.
    # The mechanism is real (the rim does unload as the face comes down)
    # and is left here to be turned up if it is ever wanted.
    FLAT_SCALE = 0.0

    _seeded = 0

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.spin_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.size_in = self.new_inlet(base=0.012, minimum=0.004,
                                      maximum=0.6)
        self.settle_in = self.new_inlet(base=3.0, minimum=0.05,
                                        maximum=60.0)
        self.rush_in = self.new_inlet(base=0.4, minimum=0.0, maximum=1.0)
        # Enough of a swing to have a clear tone and an audible flop once
        # a turn, without being a coin that has been stepped on.
        self.wobble_in = self.new_inlet(base=0.35, minimum=0.0,
                                        maximum=1.0)
        # The edge itself: nicks, burrs, a milled rim worn unevenly.
        # A different fault from being off-centre and it sounds
        # different, so it gets its own control rather than sharing
        # 'wobble'. This is what makes the contact chatter.
        self.profile_in = self.new_inlet(base=0.35, minimum=0.0,
                                         maximum=1.0)
        # How cleanly it was set spinning. 1 is a coin spun true on its
        # edge; 0 is one simply pushed over, which never rolls at all
        # and only rattles.
        self.twist_in = self.new_inlet(base=0.8, minimum=0.0, maximum=1.0)
        self.scrape_in = self.new_inlet(base=0.35, minimum=0.0, maximum=1.0)
        self.hardness_in = self.new_inlet(base=0.75, minimum=0.0,
                                          maximum=1.0)
        self.polish_in = self.new_inlet(base=0.7, minimum=0.0, maximum=1.0)
        self.level_in = self.new_inlet(base=1.0, minimum=0.0, maximum=2.0)

        SpinUnit._seeded += 1
        seed = (SpinUnit._seeded * 0x9E3779B97F4A7C15) % (1 << 64)
        self._rng = np.uint64(seed if seed else 0x2545F4914F6CDD1D)
        self._tilt = 0.0
        self._phase = 0.0
        self._face = 0.0
        self._profile = 0.0
        self._nut_phase = 0.0
        self._nut_amp = 0.0
        self._rim_down = 0.0
        self._slap_amp = 0.0
        self._slap_at = 0.0
        self._slap_len = 0.0
        self._grip = 0.0
        self._lp = 0.0
        self._hp = 0.0
        self._edge = 0.0
        self._landed = 1.0
        self._quiet = True
        self.model = 1
        self.spin_mode = 0
        # The disc's own state, for the derived model.
        self._d_q1 = 0.0
        self._d_q2 = 1.5707963267948966
        self._d_q3 = 0.0
        self._d_u1 = 0.0
        self._d_u2 = 0.0
        self._d_u3 = 0.0
        self._gesture_last = 0.0
        self._push = 0.0
        self._lean_goal = 0.0
        self._mean_lean = 0.0
        self._load_now = 1.0
        self._sweep_now = 0.0
        self._prec_now = 0.0
        self._grain_head = 0.0
        self._hop_v = 0.0
        self._hop_h = 0.0

        self.out = self.new_outlet()
        self.grind = self.new_outlet()
        self.landing = self.new_outlet()
        self.rate = self.new_outlet()
        self.face = self.new_outlet()
        self._gesture = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._hz = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._roll = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._hit = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._turn = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)
        # Grains outlive the sample that starts them, so what is still
        # sounding has to be carried across the block edge.
        self._grain_ring = np.zeros(SpinUnit.GRAIN_RING, dtype=np.float64)
        # This coin's own rim: fixed once, so it stays the same coin.
        # Amplitudes falling as k**-1.3 are a self-affine edge; scaled
        # so the whole profile swings as far as the single cosine it
        # replaces, and 'wobble' still means the same fraction of the
        # radius.
        _tk = np.array([1.0 / max(1.0, t * self.sample_rate)
                        for t in SpinUnit.TEXTURE_TIMES], dtype=np.float64)
        self._rough_k = _tk
        self._rough_state = np.zeros(len(_tk), dtype=np.float64)
        # Each one-pole on white noise keeps k/2 of its variance, and
        # they are independent, so this brings the sum back to unity --
        # and the depth is divided out of the whole so that roughening
        # the texture does not also make it louder.
        self._rough_norm = 1.0 / math.sqrt(max(float((_tk / 2.0).sum()),
                                               1.0e-12))
        self._rough_norm /= math.sqrt(1.0 + SpinUnit.TEXTURE ** 2)
        _rk = np.arange(1, SpinUnit.RIM_HARMONICS + 1, dtype=np.float64)
        _ra = _rk ** -SpinUnit.RIM_FALL
        self._rim_amp = _ra / math.sqrt(float((_ra ** 2).sum()))
        SpinUnit._seeded += 1
        self._rim_phase = np.random.default_rng(
            8081 + SpinUnit._seeded).uniform(
                0.0, 2.0 * math.pi, SpinUnit.RIM_HARMONICS)

    def reset(self):
        self._tilt = 0.0
        self._phase = 0.0
        self._face = 0.0
        self._profile = 0.0
        self._nut_phase = 0.0
        self._nut_amp = 0.0
        self._rim_down = 0.0
        self._slap_amp = 0.0
        self._slap_at = 0.0
        self._slap_len = 0.0
        self._grip = 0.0
        self._lp = 0.0
        self._hp = 0.0
        self._edge = 0.0
        self._landed = 1.0
        self._quiet = True
        self._d_q1 = 0.0
        self._d_q2 = 1.5707963267948966
        self._d_q3 = 0.0
        self._d_u1 = 0.0
        self._d_u2 = 0.0
        self._d_u3 = 0.0
        self._gesture_last = 0.0
        self._push = 0.0
        self._lean_goal = 0.0
        self._mean_lean = 0.0
        self._load_now = 1.0
        self._sweep_now = 0.0
        self._prec_now = 0.0
        self._grain_head = 0.0
        self._hop_v = 0.0
        self._hop_h = 0.0
        self._grain_ring[:] = 0.0
        self._rough_state[:] = 0.0

    # Every number this unit carries between blocks. A single infinity
    # anywhere in here poisons all of them within a block or two, and
    # then stays -- which is a voice that has gone silent for the rest of
    # the session and cannot be got back without rebuilding the patch.
    _STATE = ('_tilt', '_phase', '_face', '_profile', '_nut_phase',
              '_nut_amp', '_rim_down', '_slap_amp', '_slap_at',
              '_slap_len', '_grip', '_lp', '_hp', '_edge', '_landed',
              '_d_q1', '_d_q2', '_d_q3', '_d_u1', '_d_u2', '_d_u3',
              '_lean_goal', '_mean_lean', '_load_now', '_sweep_now',
              '_prec_now', '_grain_head', '_hop_v', '_hop_h')

    def _state_is_sane(self):
        """True if nothing has gone to infinity or worse.

        A settling disc runs into a genuine singularity, and no amount of
        clamping proves that some corner of the parameter space cannot
        step over it. So rather than claim it cannot happen, the unit
        checks whether it has, and recovers.
        """
        for name in SpinUnit._STATE:
            value = getattr(self, name)
            if not (value == value) or value > 1.0e30 or value < -1.0e30:
                return False
        return True

    def render(self, frames):
        spin = self.spin_in.eval(frames)
        size = self.size_in.eval(frames)
        settle = self.settle_in.eval(frames)
        rush = self.rush_in.eval(frames)
        wobble = self.wobble_in.eval(frames)
        profile = self.profile_in.eval(frames)
        twist = self.twist_in.eval(frames)
        scrape = self.scrape_in.eval(frames)
        hardness = self.hardness_in.eval(frames)
        polish = self.polish_in.eval(frames)
        out_level = self.level_in.eval(frames)

        out = self.out
        grind_out = self.grind
        strike_out = self.landing
        rate_out = self.rate
        face_out = self.face
        if not _svf_ready.is_set():
            for signal in (out, grind_out, strike_out, rate_out, face_out):
                signal.set_constant(0.0)
            return

        gesture = self._gesture[:frames]
        if spin.constant:
            gesture[:] = spin.value
            idle = spin.value <= 1.0e-4
        else:
            np.copyto(gesture, spin.data[:frames])
            idle = False
        np.clip(gesture, 0.0, 1.0, out=gesture)

        if self._quiet and idle and self._landed > 0.5:
            for signal in (out, grind_out, strike_out, rate_out):
                signal.set_constant(0.0)
            # The face keeps whatever it stopped at: a control that
            # jumped to zero on landing would step whatever it drives.
            face_out.set_constant(self._profile)
            return

        def scalar(signal, lo, hi):
            value = signal.value if signal.constant else float(signal.data[0])
            return min(hi, max(lo, value))

        radius = scalar(size, 0.004, 0.6)
        tail = scalar(settle, 0.05, 60.0)
        # Wobble is the swing in how hard the rim presses, and it reaches
        # far past the disc's own weight: an off-kilter coin flops onto
        # its rim carrying several times what a true one does, and that
        # load is where the sharpness comes from. At 0 the disc is
        # perfectly true, the load never varies, and there is no tone at
        # all -- only a steady hiss, which is honestly what a perfect
        # disc on a perfect surface would make.
        twist_now = scalar(twist, 0.0, 1.0)
        # A coin pushed over stops quickly; one spun true rings on. So
        # the cast shortens the tail as well as roughening it -- 'settle'
        # is the time a cleanly spun disc takes, and a bad throw gets a
        # fraction of it.
        cast_tail = tail * (0.15 + 0.85 * twist_now)
        wob_raw = scalar(wobble, 0.0, 1.0)
        prof_raw = scalar(profile, 0.0, 1.0)
        wob = SpinUnit.WOBBLE_DEPTH * wob_raw
        # The same off-centre mass that swings the load also pulls the
        # contact's orbit out of round, so one control owns both. Squared,
        # so a well-spun coin stays round and the lurch belongs to the
        # badly spun end where it is actually heard.
        ecc = SpinUnit.ORBIT_ECCENTRIC * wob_raw * wob_raw
        # The turn has to keep its average rate whatever shape it takes,
        # or wobble would retune the disc. This is the mean of the rate
        # factor over a turn.
        orbit_norm = 1.0 + 0.5 * ecc * ecc
        grain_dens = SpinUnit.GRAIN_DENSITY
        grain_floor = SpinUnit.GRAIN_FLOOR
        grain_ceiling = 1.0 / math.sqrt(SpinUnit.GRAIN_FLOOR)
        nut_ratio = SpinUnit.NUTATION_RATIO
        # A disc settles into steady precession because its SPIN holds it
        # there; that is what gyroscopic stability is for. So the swing in
        # the lean bleeds away in proportion to the twist it was given --
        # and a coin merely pushed over, with no spin at all, never
        # settles into rolling. It flops until it stops, which is exactly
        # what the far end of this continuum should do.
        nut_decay = twist_now / max(1.0, SpinUnit.NUTATION_SETTLE
                                    * cast_tail * self.sample_rate)
        scr = scalar(scrape, 0.0, 1.0)
        hard = scalar(hardness, 0.0, 1.0)
        pol = scalar(polish, 0.0, 1.0)
        # Which power of the lean the losses go as. Minus one is loss
        # proportional to energy, a half is rolling friction, one is drag
        # on the face, two is Moffatt's viscous gap. Squared, because the
        # exponents above a half all sound like the same cliff: the knob
        # spends most of its travel where the tail's shape still changes.
        spread = scalar(rush, 0.0, 1.0)
        law = -1.0 + 3.0 * spread * spread

        tilt_full = SpinUnit.TILT_FULL
        # Flat is not zero: a rim has a thickness and a table has a
        # texture, and the disc lands when its lean is down among them.
        # This is the ceiling of the whole pitch sweep.
        tilt_flat = tilt_full * 10.0 ** -(0.7 + 2.3 * pol)
        # Precession: rate squared is four g over radius times tilt, in
        # radians per second. Held here as phase per sample.
        rate_scale = math.sqrt(4.0 * 9.80665 / radius) \
            / (6.283185307179586 * self.sample_rate)
        # 'settle' is the time from full lean to flat, so the loss
        # constant is that path divided by that time -- which keeps
        # settle and polish from pulling on each other.
        if abs(law + 1.0) > 1.0e-6:
            power = law + 1.0
            span = (tilt_full ** power - tilt_flat ** power) / power
        else:
            span = math.log(tilt_full / tilt_flat)
        loss = span / (cast_tail * self.sample_rate)
        # A nudge to a disc already going takes fifteen milliseconds to
        # arrive, so re-energizing glides rather than steps.
        lift_k = 1.0 / max(1.0, 0.015 * self.sample_rate)
        # Contact: six milliseconds of something soft down to under two
        # tenths for coin on stone.
        contact = max(4.0, 0.006 * (0.03 ** hard) * self.sample_rate)
        # The landing at full lean has unit area -- the same convention as
        # bounce~, so hardness is the colour of a contact, not its weight.
        strike_scale = (2.0 / contact) / math.sqrt(tilt_full)
        # The load ripple is the continuous voice, so it is referenced to
        # full lean rather than to a contact time: how hard the disc is
        # shaken does not depend on how sharp its rim is.
        ripple_gain = 0.015 / math.sqrt(tilt_full)
        # Grinding rises as the square root of the contact's speed,
        # referenced to full lean so 'scrape' means the same thing at
        # every size.
        step_full = rate_scale / math.sqrt(tilt_full)
        scrape_gain = scr * 0.01 / math.sqrt(step_full)
        scrape_k = min(0.9, 4000.0 / self.sample_rate)
        # How scale-free the surface is. A worn or machined surface has
        # asperities at every size, so its impulses are power-law
        # distributed; 'scrape' sets how heavy that tail is. Below an
        # index of two the variance diverges, which is the regime where
        # a contact keeps sounding granular instead of collapsing into
        # hiss -- so rougher does not merely mean louder, it means the
        # big events stay individually audible.
        tail_index = 2.3 - 1.0 * scr
        grain_tail = 1.0 / tail_index
        grain_max = 40.0
        # Power has to stay put while the tail changes, or 'scrape'
        # would double as a volume control. This is the mean square of
        # a Pareto draw clipped at grain_max, in closed form.
        clip = grain_max ** (-tail_index)
        power = 1.0 - 2.0 / tail_index
        if abs(power) > 1.0e-9:
            spread = (1.0 - clip ** power) / power
        else:
            spread = -math.log(clip)
        mean_square = grain_max * grain_max * clip + spread
        grain_norm = 1.0 / math.sqrt(max(mean_square, 1.0e-9))
        # Spreading a grain over its own width moves power about -- a
        # long grain holds its momentum but spends it slower -- so the
        # normalisation above is no longer the whole story. Integrate
        # what the shaped grains actually carry and put the level back.
        # Only 'scrape' moves it, so it is worked out once and kept.
        hurst_inv = 1.0 / SpinUnit.GRAIN_HURST
        grain_span = 1.0
        # The shortest grain the contact can pass. 'hardness' already
        # means how sharply a contact answers, so it sets this too.
        grain_least = int(round(1.0 + (1.0 - hard) * SpinUnit.GRAIN_SOFT))
        if grain_least < 1:
            grain_least = 1
        if getattr(self, '_shape_at', None) != (tail_index, grain_least):
            u = (np.arange(4096) + 0.5) / 4096.0
            sizes = np.minimum(u ** (-grain_tail), grain_max)
            durs = np.clip((sizes ** hurst_inv * grain_span).astype(int),
                           grain_least, SpinUnit.GRAIN_RING - 1)
            # A one-sample grain keeps its square; a raised cosine of
            # length D spends (3/8)D of the square of its own peak, and
            # that peak is 2/D of the momentum it was given.
            carried = np.where(durs > 1, 1.5 / durs, 1.0) * sizes * sizes
            self._shape_norm = float(
                math.sqrt(max((sizes * sizes).mean(), 1.0e-9)
                          / max(carried.mean(), 1.0e-30)))
            self._shape_at = (tail_index, grain_least)
        # Only the derived model shapes its grains, so only it gets the
        # compensation. Handing this to the voiced kernel too would make
        # it louder to correct for something it does not do.
        grain_norm_shaped = grain_norm * self._shape_norm
        # How the grinding answers the load. Both of these are what makes
        # a flop a sharpening rather than a swell: it grows faster than
        # the press, and it brightens while it grows. Hardness sets how
        # much of each, so a coin on stone flops sharply and something
        # soft merely leans on it.
        # Friction force is PROPORTIONAL to the normal load -- Amontons,
        # and every rough-contact model from Bowden and Tabor's plastic
        # junctions to Greenwood and Williamson's elastic ones puts the
        # exponent between two thirds and one. Never above. This used to
        # rise to 1.9 with hardness, which turned a five-fold load into
        # a twenty-fold burst: twenty-six decibels arriving in a few
        # milliseconds, which is a glitch and not a coin. Hardness earns
        # its keep on the SPECTRUM instead -- how short a grain the
        # contact can pass, and how much it brightens under load.
        # The voiced model is kept as an untouched reference, so it
        # keeps the exponent it was tuned with; only the derived model
        # is held to the physics.
        load_exp = 1.0 + 0.9 * hard
        load_exp_real = SpinUnit.LOAD_EXP
        sharpen = 0.1 + 0.5 * hard
        # A one-pole high-pass on white noise keeps only a fraction of
        # its power, so the bright tap is scaled back up to sit level
        # with the noise it replaces.
        sharp_k = min(0.9, 5000.0 * 6.283185307179586 / self.sample_rate)
        bright_norm = math.sqrt((2.0 - sharp_k) / sharp_k)
        stop_k = min(1.0, 1.0 / max(1.0, 0.002 * self.sample_rate))

        if self.model == 0:
            # The derived model: integrate the disc, read the sound off
            # it. Everything fast -- precession, the swing in the lean,
            # the surge in load, the speed the contact sweeps -- comes
            # out of the equations rather than being shaped here.
            decim = SpinUnit.CONTROL_DECIM
            dt = decim / self.sample_rate
            # The sweep at a full-lean steady roll, as the reference the
            # grain density and the grinding are measured against.
            q2_full = 1.5707963267948966 - tilt_full
            u3_full = 2.0 * math.sqrt(9.80665 * math.cos(q2_full) / radius)
            sweep_ref = abs(math.tan(q2_full) * u3_full)
            # The fastest the disc can legitimately turn, set by where it
            # is stopped: anything past that is the integrator, not the
            # coin.
            u_cap = 4.0 * math.sqrt(9.80665 / (radius * tilt_flat))
            # Zero, not a small positive number. The roll is held from
            # reversing, but it must be allowed to be NONE -- a coin
            # given no twist has no spin at all, and a floor above zero
            # seeds one. Near the singularity the dynamics then amplify
            # that seed exponentially, so the least-spun coin came out
            # spinning fastest of all.
            u_floor = 0.0
            # Room above the requested lean for the nutation to swing
            # into. Clamped at the request itself, a held gesture clips
            # every upswing and the wobble bleeds away against the
            # ceiling; with no ceiling at all the pump overshoots and
            # the coin tumbles. Half again is enough for the swing and
            # still bounds a runaway.
            # The lean rate a coin arrives with after falling from
            # full lean, which is what an impact is measured against.
            fall_speed = math.sqrt(2.0 * 0.8 * 9.80665
                                   * math.sin(q2_full) * tilt_full / radius)
            lean_ceiling = min(1.49, tilt_full * 1.6)
            q2_ceiling = 1.5707963267948966 - lean_ceiling
            # The derived model earns its shorter tail by flopping, so it
            # is given the settle time as asked rather than pre-shortened.
            plain_loss = span / (tail * self.sample_rate)
            # A coin is a cylinder, not a knife edge. Its thickness runs
            # about a seventh of its radius; this is half of that, the
            # offset from the contact edge to the centre along the
            # symmetry axis.
            half_thick = radius * SpinUnit.COIN_ASPECT
            # The fastest roll the ceiling lean can hold in steady
            # rolling: past this the coin would have to stand up
            # further, and it is not allowed to.
            _sc = math.sin(q2_ceiling)
            _cc = math.cos(q2_ceiling)
            _tc = _sc / max(_cc, 1.0e-6)
            _n = 9.80665 * (radius * _sc - half_thick * _cc)
            _d = (0.25 * radius * radius * _tc + radius * half_thick
                  + half_thick * half_thick * _tc)
            u3_ceiling = math.sqrt(max(_n, 0.0) / _d) if _d > 0.0 else u_cap
            # 'wobble' means something physical here: how far from round
            # the rim is, as a fraction of the radius. A tenth is a badly
            # made coin.
            out_of_round = 0.1 * wob_raw
            if sweep_ref < 1.0e-6:
                sweep_ref = 1.0e-6
            result = self._y[:frames]
            hertz = self._hz[:frames]
            rolling = self._roll[:frames]
            hits = self._hit[:frames]
            turning = self._turn[:frames]
            (self._d_q1, self._d_q2, self._d_q3, self._d_u1, self._d_u2,
             self._d_u3, self._gesture_last, self._push,
             self._lean_goal, self._mean_lean, self._load_now,
             self._sweep_now, self._prec_now, self._slap_amp,
             self._slap_at, self._slap_len, self._grip, self._lp,
             self._hp, self._edge, self._landed,
             self._hop_v, self._hop_h,
             rng_state, self._grain_head) = _spin_real_kernel(
                gesture, radius, 9.80665, tilt_full, tilt_flat, lift_k,
                plain_loss, law, twist_now, wob_raw, contact,
                strike_scale, 2.5,
                ripple_gain * SpinUnit.DERIVED_GAIN,
                scrape_gain * SpinUnit.DERIVED_GAIN, scrape_k,
                load_exp_real, sharpen,
                sharp_k, bright_norm, stop_k, grain_dens, grain_floor,
                grain_ceiling, sweep_ref, half_thick,
                1.0 if self.spin_mode else 0.0, SpinUnit.DEAD_ZONE,
                SpinUnit.TRACK, SpinUnit.DRAIN_SIDE,
                SpinUnit.INJECT_GAIN, SpinUnit.BALANCE_GAIN,
                SpinUnit.CAST_LEAST, SpinUnit.CAST_GAIN,
                grain_tail, grain_max,
                grain_norm_shaped, out_of_round,
                SpinUnit.DRAIN, SpinUnit.NUT_KICK, SpinUnit.REBOUND,
                fall_speed, u3_full * 0.05, fall_speed * 0.04,
                u3_ceiling,
                u_cap, u_floor, SpinUnit.LOAD_MAX - 1.0,
                max(1.0e-6, SpinUnit.FLAT_SCALE * half_thick / radius),
                SpinUnit.LEAN_AVERAGE, SpinUnit.FOLLOW, SpinUnit.HOP,
                0.1 * prof_raw, SpinUnit.TABLE_ROUGH * scr,
                SpinUnit.TEXTURE, self._rough_norm,
                q2_ceiling,
                math.sin(tilt_flat), decim, dt,
                self.sample_rate,
                self._d_q1, self._d_q2, self._d_q3, self._d_u1,
                self._d_u2, self._d_u3, self._gesture_last, self._push,
                self._lean_goal, self._mean_lean,
                self._load_now, self._sweep_now, self._prec_now,
                self._slap_amp, self._slap_at, self._slap_len, self._grip,
                self._lp, self._hp, self._edge, self._landed,
                self._hop_v, self._hop_h, self._rng,
                hurst_inv, grain_span, grain_least, self._grain_head,
                result, hertz, turning, rolling, hits, self._grain_ring,
                self._rim_amp, self._rim_phase,
                self._rough_k, self._rough_state)
            self._rng = np.uint64(rng_state)
            if not self._state_is_sane():
                # Something stepped over the singularity. Put the coin
                # down, silence this block, and carry on -- a voice that
                # cannot recover from one bad block is worse than one
                # that drops a block.
                self.reset()
                for signal in (out, grind_out, strike_out, rate_out,
                               face_out):
                    signal.set_constant(0.0)
                return
            glide = self._level_glide
            self._apply_level(result, out_level, frames)
            advanced = self._level_glide
            self._level_glide = glide
            self._apply_level(rolling, out_level, frames)
            self._level_glide = glide
            self._apply_level(hits, out_level, frames)
            self._level_glide = advanced
            np.copyto(out.data[:frames], result, casting='unsafe')
            out.constant = False
            np.copyto(grind_out.data[:frames], rolling, casting='unsafe')
            grind_out.constant = False
            np.copyto(strike_out.data[:frames], hits, casting='unsafe')
            strike_out.constant = False
            np.copyto(rate_out.data[:frames], hertz, casting='unsafe')
            rate_out.constant = False
            np.copyto(face_out.data[:frames], turning, casting='unsafe')
            face_out.constant = False
            scratch = self._scratch[:frames]
            np.abs(result, out=scratch)
            self._quiet = bool(scratch.max() < 1.0e-5)
            return

        result = self._y[:frames]
        hertz = self._hz[:frames]
        rolling = self._roll[:frames]
        hits = self._hit[:frames]
        turning = self._turn[:frames]
        (self._tilt, self._phase, self._face, self._profile,
         self._nut_phase, self._nut_amp, self._rim_down,
         self._slap_amp, self._slap_at, self._slap_len, self._grip,
         self._lp, self._hp, self._edge, self._landed,
         rng_state) = _spin_kernel(
            gesture, tilt_full, tilt_flat, lift_k, loss, law, rate_scale,
            0.4, contact, strike_scale, grain_tail, grain_max, grain_norm,
            wob, ecc, orbit_norm, nut_ratio,
            nut_decay, SpinUnit.NUTATION_MAX, twist_now, grain_dens,
            grain_floor, grain_ceiling, step_full, 2.5,
            ripple_gain, scrape_gain,
            scrape_k, load_exp, sharpen, sharp_k, bright_norm, stop_k,
            self.sample_rate,
            self._tilt, self._phase, self._face, self._profile,
            self._nut_phase, self._nut_amp, self._rim_down,
            self._slap_amp, self._slap_at, self._slap_len,
            self._grip, self._lp, self._hp, self._edge,
            self._landed, self._rng, result, hertz, turning, rolling, hits)
        self._rng = np.uint64(rng_state)

        # Level rides the sound, not the controls: a rate in Hz and a
        # profile between minus one and one mean what they say, and a
        # fader on them would be a lie. The glide is rewound between the
        # three, since it advances once per block and not once per
        # buffer -- otherwise each outlet would ride a different ramp
        # and the parts would stop summing to the whole.
        glide = self._level_glide
        self._apply_level(result, out_level, frames)
        advanced = self._level_glide
        self._level_glide = glide
        self._apply_level(rolling, out_level, frames)
        self._level_glide = glide
        self._apply_level(hits, out_level, frames)
        self._level_glide = advanced
        np.copyto(out.data[:frames], result, casting='unsafe')
        out.constant = False
        np.copyto(grind_out.data[:frames], rolling, casting='unsafe')
        grind_out.constant = False
        np.copyto(strike_out.data[:frames], hits, casting='unsafe')
        strike_out.constant = False
        np.copyto(rate_out.data[:frames], hertz, casting='unsafe')
        rate_out.constant = False
        np.copyto(face_out.data[:frames], turning, casting='unsafe')
        face_out.constant = False
        scratch = self._scratch[:frames]
        np.abs(result, out=scratch)
        self._quiet = bool(scratch.max() < 1.0e-5)


class FaderUnit(Unit):
    """A mixing-desk fader on the signal passing through.

    What separates a fader from a multiply is the taper. Loudness lives in
    dB, so the throw is dB-linear through the working range -- unity at
    three quarters of the travel where a desk puts its 0 mark, +6 dB of
    push above it, 60 dB of reach below -- and the last twentieth fades
    linearly to a true zero, which no finite number of decibels reaches.
    Equal distances on the handle are equal loudness changes, which is what
    makes a ride feel right.

    The gain glides and is ramped across each block, so riding the handle
    -- the entire point of a fader -- never zippers. The position is an
    inlet like any other: patch an envelope or an effort stream and the
    automation moves through the same taper as the hand, at control rate.

    Stereo the way vcf~ is: one fader, two channels, and the right outlet
    carries the left signal until something is patched to the right inlet.
    Bypassed, it stands aside and the signal passes untouched.
    """

    UNITY_POSITION = 0.75
    FLOOR_POSITION = 0.05

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.right_in = self.new_inlet()
        self.position_in = self.new_inlet(base=FaderUnit.UNITY_POSITION,
                                          minimum=0.0, maximum=1.0)
        self.out = self.new_outlet()
        self.right = self.new_outlet()
        self.pan_in = self.new_inlet(base=0.0, minimum=-1.0, maximum=1.0)
        self._pan_glide = 0.0
        self.levels = [0.0, 0.0]
        self.peaks = [0.0, 0.0]
        self._hold = [0.0, 0.0]
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    @staticmethod
    def taper(position):
        """Fader position 0..1 to linear gain, desk-law shaped."""
        position = min(1.0, max(0.0, position))
        if position >= FaderUnit.UNITY_POSITION:
            db = (position - FaderUnit.UNITY_POSITION) * 24.0
            return 10.0 ** (db / 20.0)
        if position <= FaderUnit.FLOOR_POSITION:
            floor_gain = 10.0 ** (-60.0 / 20.0)
            return floor_gain * position / FaderUnit.FLOOR_POSITION
        span = FaderUnit.UNITY_POSITION - FaderUnit.FLOOR_POSITION
        db = -60.0 * (FaderUnit.UNITY_POSITION - position) / span
        return 10.0 ** (db / 20.0)

    def current_db(self):
        """Where the glide actually is, for the node's readout."""
        if self._level_glide <= 1.0e-6:
            return None
        return 20.0 * math.log10(self._level_glide)

    # Post-fader metering, with vu~'s ballistics: the fader carries
    # its own eyes so a strip does not need a second node.
    def _meter(self, frames):
        seconds = frames / self.sample_rate
        for channel, outlet in ((0, self.out), (1, self.right)):
            if outlet.constant:
                # A constant block needs no numpy: rms and peak are its
                # magnitude -- and an idle strip is all constant zeros.
                rms = peak = abs(outlet.value)
            else:
                buffer = outlet.array(frames)
                scratch = self._scratch[:frames]
                np.multiply(buffer, buffer, out=scratch, casting='unsafe')
                rms = math.sqrt(float(np.mean(scratch)))
                peak = math.sqrt(float(scratch.max()))
            smoothed = self.levels[channel]
            if rms > smoothed:
                k = 1.0 - math.exp(-seconds / VuUnit.ATTACK_SECONDS)
            else:
                k = 1.0 - math.exp(-seconds / VuUnit.RELEASE_SECONDS)
            self.levels[channel] = smoothed + (rms - smoothed) * k
            if peak >= self.peaks[channel]:
                self.peaks[channel] = peak
                self._hold[channel] = 0.0
            else:
                self._hold[channel] += seconds
                if self._hold[channel] > VuUnit.PEAK_HOLD_SECONDS:
                    self.peaks[channel] *= \
                        VuUnit.PEAK_FALL_PER_SECOND ** seconds

    def bypass_pairs(self):
        if self.right_in.sources:
            return ((self.signal_in, self.out), (self.right_in, self.right))
        return ((self.signal_in, self.out), (self.signal_in, self.right))

    def _scale_into(self, source, outlet, start, landing, frames):
        if source.constant and start == landing:
            outlet.set_constant(source.value * landing)
            return
        buffer = outlet.data[:frames]
        np.copyto(buffer, source.array(frames))
        if start == landing:
            if landing != 1.0:
                buffer *= landing
        else:
            ramp = self._level_ramp[:frames]
            np.multiply(_INDEX_RAMP[:frames], (landing - start) / frames,
                        out=ramp)
            ramp += start
            buffer *= ramp
        outlet.constant = False

    def render(self, frames):
        signal = self.signal_in.eval(frames)
        right_in = self.right_in.eval(frames)
        position = self.position_in.eval(frames)
        pan = self.pan_in.eval(frames)

        p = position.value if position.constant else float(position.data[0])
        target = FaderUnit.taper(p)
        start = self._level_glide
        landing = start + (target - start) * 0.35
        if abs(landing - target) < 1.0e-6:
            landing = target
        self._level_glide = landing

        # Equal-power pan, referenced to unity at center so a patch
        # that never touches the knob hears exactly what it always
        # did; the panned-into side gains three dB at the extreme, as
        # a desk's balance does. Glided like the level: a knob is a
        # hand, not a step.
        pv = pan.value if pan.constant else float(pan.data[0])
        pv = min(1.0, max(-1.0, pv))
        pg0 = self._pan_glide
        pg1 = pg0 + (pv - pg0) * 0.35
        if abs(pg1 - pv) < 1.0e-6:
            pg1 = pv
        self._pan_glide = pg1
        a0 = (pg0 + 1.0) * math.pi * 0.25
        a1 = (pg1 + 1.0) * math.pi * 0.25
        root2 = 1.4142135623730951
        gl0, gl1 = math.cos(a0) * root2, math.cos(a1) * root2
        gr0, gr1 = math.sin(a0) * root2, math.sin(a1) * root2

        right_src = right_in if self.right_in.sources else signal
        self._scale_into(signal, self.out,
                         start * gl0, landing * gl1, frames)
        self._scale_into(right_src, self.right,
                         start * gr0, landing * gr1, frames)
        self._meter(frames)


class CaptureUnit(Unit):
    """Ring buffer of recent audio, readable as an array from the node world.

    Where snapshot~ samples one value per GUI frame, this keeps every sample,
    so nothing is lost between frames -- audio produces ~86 blocks/sec against
    a ~60 Hz GUI, and a naive "hand me the current block" reader both drops
    about a third of the blocks and duplicates others.

    Thread handoff is a single-producer/single-consumer ring with no lock,
    resting on two guarantees:

      * The audio thread writes the samples and only then publishes the new
        write index, so a reader touching strictly below that index can never
        observe a partial write.
      * The reader never comes within MAX_WINDOW samples of the write head's
        wrap point. Since the writer is rate-limited by the audio device, it
        needs MAX_WINDOW/sample_rate -- about 0.74 s -- to close that gap,
        against a reader copy measured in microseconds.

    The second guarantee is why reads are capped at half the ring rather than
    all of it: at a full-capacity window the headroom would be zero and a
    slow reader could be overtaken mid-copy.
    """

    CAPACITY = 65536      # ~1.5 s at 44.1 kHz, 256 kB of float32

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.capacity = CaptureUnit.CAPACITY
        # Largest readable window, and equally the largest backlog tolerated
        # before declaring an overrun. Leaves the other half as headroom.
        self.max_window = self.capacity // 2
        self._buffer = np.zeros(self.capacity, dtype=np.float32)
        self._write = 0
        # Size of the most recent audio block, so the node can align its
        # chunking to the quantum the audio thread actually delivers.
        self.last_block = 0

    @property
    def written(self):
        """Total samples committed so far. Monotonic."""
        return self._write

    def render(self, frames):
        if frames > self.capacity:
            return
        signal = self.signal_in.eval(frames)
        values = signal.array(frames)

        capacity = self.capacity
        write = self._write
        start = write % capacity
        end = start + frames

        if end <= capacity:
            self._buffer[start:end] = values
        else:
            first = capacity - start
            self._buffer[start:] = values[:first]
            self._buffer[:frames - first] = values[first:]

        self.last_block = frames
        # Publish last: everything below the new index is now complete.
        self._write = write + frames

    def _extract(self, begin, count):
        capacity = self.capacity
        out = np.empty(count, dtype=np.float32)
        start = begin % capacity
        end = start + count
        if end <= capacity:
            out[:] = self._buffer[start:end]
        else:
            first = capacity - start
            out[:first] = self._buffer[start:]
            out[first:] = self._buffer[:count - first]
        return out

    def read_latest(self, size):
        """The newest `size` samples. Overlaps or skips as the rates dictate."""
        write = self._write
        count = min(int(size), self.max_window, write)
        if count <= 0:
            return None
        return self._extract(write - count, count)

    def read_chunk(self, last_read, size):
        """Gapless, fixed length: exactly `size` samples, or nothing yet.

        Audio arrives in blocks and the GUI reads at a rate that does not
        divide into them -- about 1.44 blocks per frame at 512/44.1k. Handing
        over whatever happens to have accumulated would give a different array
        length every frame, which any downstream FFT, plot or buffer has to
        cope with for no benefit. Instead the remainder stays in the ring and
        joins the next chunk, so consumers see one constant length forever.

        Returns (array_or_None, new_last_read, dropped). `dropped` is nonzero
        when the reader fell more than MAX_WINDOW behind, meaning the patch
        could not keep up and samples were genuinely lost -- worth surfacing
        rather than hiding.
        """
        write = self._write
        dropped = 0
        available = write - last_read

        if available > self.max_window:
            dropped = available - self.max_window
            last_read = write - self.max_window
            available = self.max_window

        size = min(int(size), self.max_window)
        if size <= 0 or available < size:
            return None, last_read, dropped
        return self._extract(last_read, size), last_read + size, dropped


class SnapshotUnit(Unit):
    """Reads a signal back to the control layer for metering and display.

    Peak and RMS accumulate across every block and are cleared when the node
    reads them, rather than describing only the most recent block. Audio runs
    at ~86 blocks per second against a ~60 Hz GUI, so a per-block figure is
    read roughly two times in three -- and a percussive envelope shorter than
    12 ms can live and die inside one block, which a sampling reader misses
    entirely. Holding between reads means no transient can slip through.

    `value` stays instantaneous (the newest sample), which is what you want
    for following a slow control signal.
    """

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.value = 0.0
        self._peak_hold = 0.0
        self._square_sum = 0.0
        self._sample_count = 0

    def render(self, frames):
        signal = self.signal_in.eval(frames)

        if signal.constant:
            value = signal.value
            self.value = value
            magnitude = abs(value)
            if magnitude > self._peak_hold:
                self._peak_hold = magnitude
            self._square_sum += value * value * frames
            self._sample_count += frames
            return

        values = signal.data[:frames]
        self.value = float(values[-1])
        magnitude = float(np.max(np.abs(values)))
        if magnitude > self._peak_hold:
            self._peak_hold = magnitude
        self._square_sum += float(np.dot(values, values))
        self._sample_count += frames

    def take(self):
        """Main thread: value now, plus peak and RMS since the previous call.

        The clear-after-read races the audio thread in principle -- a block
        landing between the read and the reset loses its contribution. It
        costs at most one block of a meter reading, so it is not worth a lock
        on the audio path.
        """
        peak = self._peak_hold
        total = self._square_sum
        count = self._sample_count
        self._peak_hold = 0.0
        self._square_sum = 0.0
        self._sample_count = 0
        rms = math.sqrt(total / count) if count else 0.0
        return self.value, peak, rms


# ----------------------------------------------------------------------------
# vst~
# ----------------------------------------------------------------------------

try:
    import pedalboard
except ImportError:
    pedalboard = None


def plugin_hosting_available():
    return pedalboard is not None


# macOS keeps its two dozen built-in effects (AUMatrixReverb, AUDelay, the
# filters) in one shell bundle here. None of them can be hosted: pedalboard
# only scans /Library/Audio/Plug-Ins/Components and the user equivalent, and
# handed this bundle directly its scanner segfaults on a name lookup and hangs
# for ever on a named load -- tested on seven of them, all seven hung. A hang
# is worse than a refusal because it takes the GUI thread with it, so the
# whole directory is refused by name before anything is opened.
UNSCANNABLE_DIRECTORY = '/System/Library/Components'


def plugin_file_refusal(path):
    """Why this file must not be opened, or '' if it is safe to try."""
    if not path:
        return ''
    if os.path.abspath(path).startswith(UNSCANNABLE_DIRECTORY):
        return ("macOS's built-in AudioUnits live in " + UNSCANNABLE_DIRECTORY
                + ' and cannot be hosted -- the scanner hangs on that bundle. '
                'Install a plugin under /Library/Audio/Plug-Ins instead.')
    return ''


def installed_plugin_files():
    """Every VST3 and AudioUnit file the system knows about.

    Apple's own effects are deliberately not among them; see
    UNSCANNABLE_DIRECTORY.
    """
    if pedalboard is None:
        return []
    files = []
    for holder in (pedalboard.VST3Plugin, pedalboard.AudioUnitPlugin):
        try:
            files.extend(holder.installed_plugins)
        except Exception:
            continue
    return files


def find_plugin_file(fragment):
    """Resolve a path, or any distinctive part of a plugin's filename."""
    if not fragment:
        return None
    if os.path.exists(fragment):
        return fragment
    lowered = str(fragment).lower()
    for path in installed_plugin_files():
        if lowered in os.path.basename(path).lower():
            return path
    return None


def plugin_names_in_file(path):
    """The plugins inside one file. Usually one; shells hold many."""
    if pedalboard is None or plugin_file_refusal(path):
        return []
    holder = (pedalboard.AudioUnitPlugin if path.endswith('.component')
              else pedalboard.VST3Plugin)
    try:
        return list(holder.get_plugin_names_for_file(path))
    except Exception:
        return []


# Mach-O, enough of it to read the architectures out of a plugin binary.
_MACHO_CPU_NAMES = {0x01000007: 'x86_64', 0x0100000c: 'arm64',
                    0x00000007: 'i386', 0x0000000c: 'arm'}
_MACHO_FAT = (0xcafebabe, 0xcafebabf)
# Keyed by the magic as it reads big-endian, which is how the four bytes sit
# in the file. A file whose magic reads back 0xcffaedfe is little-endian, so
# its cputype must be read little-endian too -- getting this backwards turns
# x86_64 (0x01000007) into 0x07000001 and reports an architecture that has
# never existed.
_MACHO_THIN = {0xfeedface: '>I', 0xfeedfacf: '>I',
               0xcefaedfe: '<I', 0xcffaedfe: '<I'}


def _macho_arch_name(cpu, subtype):
    """arm64 and arm64e share a CPU type and differ only in the subtype."""
    name = _MACHO_CPU_NAMES.get(cpu)
    if name is None:
        return hex(cpu)
    if name == 'arm64' and (subtype & 0xff) == 2:
        return 'arm64e'
    return name


def plugin_architectures(path):
    """Which CPUs a plugin bundle was built for, as a set of names.

    Read out of the Mach-O header rather than shelled out to `lipo`, so it
    costs nothing and works whether or not the developer tools are installed.
    An empty set means we could not tell, which is treated as no evidence
    rather than as bad news.
    """
    binary = None
    holder = os.path.join(path, 'Contents', 'MacOS')
    if os.path.isdir(holder):
        for entry in sorted(os.listdir(holder)):
            candidate = os.path.join(holder, entry)
            if os.path.isfile(candidate):
                binary = candidate
                break
    if binary is None:
        return set()

    try:
        with open(binary, 'rb') as handle:
            head = handle.read(4)
            if len(head) < 4:
                return set()
            magic = struct.unpack('>I', head)[0]
            if magic in _MACHO_FAT:
                count = struct.unpack('>I', handle.read(4))[0]
                if count > 64:
                    return set()
                found = set()
                # fat_arch is cputype, cpusubtype, offset, size, align -- 20
                # bytes; fat_arch_64 widens offset and size and adds a
                # reserved word, making 32. Getting this stride wrong reads
                # the next entry's offset as a CPU type and invents
                # architectures that are not there.
                width = 32 if magic == 0xcafebabf else 20
                for _ in range(count):
                    entry = handle.read(width)
                    if len(entry) < 8:
                        break
                    cpu, subtype = struct.unpack('>II', entry[:8])
                    found.add(_macho_arch_name(cpu, subtype))
                return found
            if magic in _MACHO_THIN:
                order = _MACHO_THIN[magic]
                rest = handle.read(8)
                if len(rest) < 8:
                    return set()
                cpu, subtype = struct.unpack(order + order[-1], rest)
                return {_macho_arch_name(cpu, subtype)}
    except OSError:
        return set()
    return set()


def architecture_complaint(path):
    """Why this plugin cannot run in this process, or '' if that is not it.

    A plugin built for another CPU fails to load with nothing but 'scan
    failure' from the plugin format layer, which sends you looking for a bug
    in the host. It is worth the twenty lines above to be able to say what is
    actually wrong, because there is no fixing it in software: an arm64
    process cannot load an x86_64 bundle, and every plugin from before Apple
    Silicon is in exactly that position.
    """
    built_for = plugin_architectures(path)
    if not built_for:
        return ''
    ours = platform.machine()
    if ours in built_for:
        return ''
    return (os.path.basename(path) + ' is built for '
            + ', '.join(sorted(built_for)) + ' and this is an ' + ours
            + ' process, so it cannot be loaded at all -- no host running '
            'natively on this Mac can. Find a build with ' + ours
            + ' in it, or run the plugin in an external host and return its '
            'audio over a virtual device.')


def open_plugin(path, plugin_name=None, sample_rate=DEFAULT_SAMPLE_RATE):
    """Main thread only: load a plugin and find out how wide it is.

    A plugin's channel layout is fixed by the plugin, not chosen by the host:
    a mono effect handed a stereo block raises rather than adapting. There is
    no property to ask, so the only reliable way to find out is to try it,
    stereo first, and keep whichever shape it accepts.

    Loading is slow (hundreds of ms, sometimes seconds) and some plugins touch
    UI toolkits while initialising, so this belongs on the main thread at a
    moment when a stall is acceptable -- never on the audio thread.
    """
    if pedalboard is None:
        raise RuntimeError('pedalboard is not installed')
    refusal = plugin_file_refusal(path)
    if refusal:
        raise ValueError(refusal)
    try:
        plugin = pedalboard.load_plugin(path, plugin_name=plugin_name)
    except Exception as error:
        # Diagnose only once it has actually failed. Checking the header first
        # would let a mistake in the reader refuse a plugin that works.
        complaint = architecture_complaint(path)
        if complaint:
            raise ValueError(complaint)
        raise error
    if plugin.is_instrument and not plugin.is_effect:
        raise ValueError(str(plugin.name) + ' is an instrument; vst~ hosts '
                         'effects. Drive instruments from an external host.')

    channels = 0
    for count in (2, 1):
        probe = np.zeros((count, 64), dtype=np.float32)
        try:
            result = plugin.process(probe, sample_rate, buffer_size=64,
                                    reset=True)
        except Exception:
            continue
        if result.shape[1] == 64:
            channels = count
            break
    if channels == 0:
        raise ValueError(str(plugin.name) + ' accepted neither a mono nor a '
                         'stereo block')

    # Deliberately not reset() here, however tidy that would look. reset()
    # tears down the prepared state, and pedalboard only rebuilds it on a
    # process(reset=True) -- so a reset plugin driven the way the audio thread
    # drives it, with reset=False, returns zero frames for ever, silently and
    # without raising. The probe above already ran with reset=True, which is
    # what leaves it prepared; all it has swallowed is 64 samples of silence.
    return plugin, channels


class VstUnit(Unit):
    """Hosts a VST3 or AudioUnit effect as one unit in the graph.

    The plugin is an ordinary unit as far as the compiler is concerned -- it
    sorts, it gates, it bypasses. What is different is that its insides are
    somebody else's code, which the audio thread has to call and cannot
    trust: it may be slow, it may throw, and it may be doing something
    unbounded like reading a sample library from disk. So the call is timed
    and wrapped, and a plugin that misbehaves is dropped rather than allowed
    to keep costing the whole graph its deadline. Dropping it leaves the
    signal passing through, not silence, which is the same thing bypass does
    and the only failure a performance can absorb.

    Two things the surrounding architecture does not do for you:

    Latency. Plugins report a processing delay -- 46 ms for a Waves Clarity,
    216 ms for a neural effect -- and there is no delay compensation in the
    compiler. In series that is simply added delay. Against a parallel dry
    path it is a phase smear, and the 'mix' control here is exactly such a
    path, so on a plugin with real latency use mix at 1.0 and blend outside.
    `latency` carries the figure so the node can say so.

    Rate. Parameters are set once per block, ~86 Hz at 512 frames, because
    that is what a plugin parameter is -- automation, not a modulation input.
    Audio-rate movement patched to a parameter inlet is read at its last
    sample and nothing between. For anything that needs to move at audio rate,
    use the native units; they exist for that.
    """

    # A plugin taking more than half the block period on its own leaves too
    # little for the rest of the graph. One late block is a scheduling
    # accident, twenty is what the plugin costs, so tolerate a run of them
    # and count down on the good ones before giving up.
    OVERRUN_FRACTION = 0.5
    OVERRUN_LIMIT = 20

    # Parameter values below this much movement are not worth a call across
    # the plugin boundary; VST automation is 7-bit-ish in practice anyway.
    PARAMETER_EPSILON = 1.0 / 2048.0

    # Modulation slots. However many there are, they are created before any
    # plugin is loaded, because the node's ports have to exist at construction
    # to survive save and load -- which plugin parameter a slot drives is a
    # name stored in an option, not a port that appears and disappears.
    #
    # The count is worth spending: a slot that drives nothing is not in the
    # binding list at all and costs literally nothing per block, and a bound
    # one costs 0.33 us whether it moves or not, plus whatever the plugin
    # charges for the write (2.8 us on Supermassive) and only when the value
    # actually changes. Sixteen slots all moving every block is under half a
    # percent of the block period. Screen space is the scarce thing here, not
    # time.
    PARAMETER_SLOTS = 8

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, parameter_slots=None):
        super().__init__(sample_rate)
        if parameter_slots is None:
            parameter_slots = VstUnit.PARAMETER_SLOTS
        self.signal_in = self.new_inlet()
        self.right_in = self.new_inlet()
        self.mix_in = self.new_inlet(base=1.0, minimum=0.0, maximum=1.0)
        self.parameter_in = [self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
                             for _ in range(max(0, int(parameter_slots)))]
        self.out = self.new_outlet()
        self.right = self.new_outlet()

        # Read once at the top of render and assigned whole from the main
        # thread, the way the engine takes a new SynthProgram: the audio
        # thread either has the old plugin or the new one, never a half-built
        # one, and no lock is needed to say so.
        self._plugin = None
        self.plugin_name = ''
        self.channels = 1
        self.latency = 0
        self.error = ''
        self.cost_ms = 0.0

        self._bindings = ()
        self._pending_choices = []
        self._blocks = {}
        self._overruns = 0
        self._dry = np.zeros((2, MAX_BLOCK), dtype=np.float32)
        self._blend = np.zeros(MAX_BLOCK, dtype=np.float32)

    # -- main thread --------------------------------------------------------

    def attach(self, plugin, channels, name='', latency=0):
        """Install a loaded plugin, or None to go back to passing through."""
        self._plugin = None
        self._bindings = ()
        self._pending_choices = []
        self._blocks = {}
        self.channels = max(1, min(2, int(channels)))
        self.plugin_name = name
        self.latency = int(latency)
        self.cost_ms = 0.0
        self._overruns = 0
        self.error = ''
        self._plugin = plugin

    def bind_parameters(self, pairs):
        """(parameter object, inlet) pairs for render to drive per block.

        The parameter objects are held rather than looked up by name because
        `plugin.parameters` rebuilds the entire dictionary on every access --
        294 us measured on a 300-parameter plugin, which is a quarter of the
        block period at 512 frames, to read one number. A parameter object
        kept from that dictionary costs 2.3 us to set, which is affordable.
        """
        self._bindings = [[parameter, inlet, -1.0] for parameter, inlet in pairs]

    def set_choice(self, parameter, raw_value):
        """Ask for a discrete parameter (a mode, a sync division) to change.

        Queued rather than written here. The plugin belongs to the audio
        thread for as long as it is rendering, and a parameter written from
        under it is the kind of race that shows up once a fortnight in
        performance and never on the bench. The queue is drained at the top of
        the next block, so the change lands within a block either way -- which
        for something you choose from a menu is immediate.

        Discrete parameters are not evenly spaced and cannot be indexed by
        arithmetic: Supermassive's 23 reverb modes quantise unevenly and list
        'Gemini' at both ends. The caller works the value out with the
        parameter's own get_raw_value_for(), which is exact.
        """
        self._pending_choices.append((parameter, float(raw_value)))

    def plugin_loaded(self):
        return self._plugin is not None

    # -- audio thread -------------------------------------------------------

    def bypass_pairs(self):
        if self.right_in.sources:
            return ((self.signal_in, self.out), (self.right_in, self.right))
        return ((self.signal_in, self.out), (self.signal_in, self.right))

    def _block(self, frames):
        """A contiguous (channels, frames) buffer for this block size.

        It has to be contiguous. A view like scratch[:, :frames] of a wider
        two-channel buffer is not, and pedalboard does not reject one -- it
        returns a (2, 0) array, so the failure arrives as silence with no
        exception anywhere. Buffers are therefore kept whole, one per block
        size seen. PortAudio settles on a size immediately, so this allocates
        once or twice in the life of a patch and never again.
        """
        block = self._blocks.get(frames)
        if block is None or block.shape[0] != self.channels:
            block = np.zeros((self.channels, frames), dtype=np.float32)
            self._blocks[frames] = block
        return block

    def _apply_bindings(self, frames):
        """Push modulated parameters across, skipping the ones that held.

        Values are normalised 0..1, which is what VST automation is; the
        plugin maps that onto whatever its own range happens to be.
        """
        for binding in self._bindings:
            parameter, inlet, previous = binding
            signal = inlet.eval(frames)
            if signal.constant:
                value = signal.value
            else:
                value = float(signal.data[frames - 1])
            if value < 0.0:
                value = 0.0
            elif value > 1.0:
                value = 1.0
            if abs(value - previous) < VstUnit.PARAMETER_EPSILON:
                continue
            binding[2] = value
            try:
                parameter.raw_value = value
            except Exception:
                # A parameter that refuses a value is not worth killing the
                # plugin over, but it is worth not retrying every block.
                binding[2] = value

    def fail(self, message):
        """Drop the plugin. The graph keeps running with the signal passing."""
        self._plugin = None
        self.error = message

    def _pass_through(self, signal, right_signal, frames):
        for source, outlet in ((signal, self.out), (right_signal, self.right)):
            if source.constant:
                outlet.set_constant(source.value)
            else:
                np.copyto(outlet.data[:frames], source.data[:frames])
                outlet.constant = False

    def _drain_choices(self):
        """Apply queued menu changes. Rebinding the list is the whole lock:
        the main thread only ever appends, so nothing can be lost."""
        pending, self._pending_choices = self._pending_choices, []
        for parameter, value in pending:
            try:
                parameter.raw_value = value
            except Exception:
                pass

    def render(self, frames):
        plugin = self._plugin
        if self._pending_choices:
            self._drain_choices()
        signal = self.signal_in.eval(frames)
        right_signal = (self.right_in.eval(frames) if self.right_in.sources
                        else signal)

        if plugin is None:
            self._pass_through(signal, right_signal, frames)
            return

        stereo = self.channels > 1
        block = self._block(frames)
        np.copyto(block[0], signal.array(frames), casting='unsafe')
        if stereo:
            np.copyto(block[1], right_signal.array(frames), casting='unsafe')

        mix = self.mix_in.eval(frames)
        wet_only = mix.constant and mix.value >= 1.0
        if not wet_only:
            # Kept before processing: the plugin is handed our buffer and is
            # entitled to write through it.
            np.copyto(self._dry[0, :frames], block[0])
            if stereo:
                np.copyto(self._dry[1, :frames], block[1])

        self._apply_bindings(frames)

        start = time.perf_counter()
        try:
            wet = plugin.process(block, self.sample_rate, buffer_size=frames,
                                 reset=False)
        except Exception as error:
            self.fail(type(error).__name__ + ': ' + str(error))
            self._pass_through(signal, right_signal, frames)
            return
        self._watch(time.perf_counter() - start, frames)

        if wet.shape[1] != frames:
            self.fail('returned ' + str(wet.shape[1]) + ' of ' + str(frames)
                      + ' frames')
            self._pass_through(signal, right_signal, frames)
            return

        left_out = self.out.data[:frames]
        right_out = self.right.data[:frames]
        np.copyto(left_out, wet[0], casting='unsafe')
        self.out.constant = False
        self.right.constant = False

        if stereo:
            np.copyto(right_out, wet[1] if wet.shape[0] > 1 else wet[0],
                      casting='unsafe')
            if not wet_only:
                self._mix_dry(left_out, self._dry[0, :frames], mix, frames)
                self._mix_dry(right_out, self._dry[1, :frames], mix, frames)
            return

        if not wet_only:
            self._mix_dry(left_out, self._dry[0, :frames], mix, frames)
        np.copyto(right_out, left_out)

    def _mix_dry(self, wet, dry, mix, frames):
        """Crossfade in place. `dry` is scratch and may be consumed."""
        if mix.constant:
            amount = min(1.0, max(0.0, mix.value))
            wet *= amount
            dry *= (1.0 - amount)
            wet += dry
            return
        amount = mix.data[:frames]
        blend = self._blend[:frames]
        np.subtract(1.0, amount, out=blend)
        wet *= amount
        dry *= blend
        wet += dry

    def _watch(self, elapsed, frames):
        milliseconds = elapsed * 1000.0
        # Smoothed rather than averaged: the node only wants a number to show,
        # and a running figure costs one multiply where a window costs memory.
        self.cost_ms += (milliseconds - self.cost_ms) * 0.05
        budget = frames / self.sample_rate * 1000.0
        if milliseconds > budget * VstUnit.OVERRUN_FRACTION:
            self._overruns += 1
            if self._overruns >= VstUnit.OVERRUN_LIMIT:
                self.fail('too slow for the audio thread ('
                          + format(milliseconds, '.1f') + ' ms of a '
                          + format(budget, '.1f') + ' ms block)')
        elif self._overruns > 0:
            self._overruns -= 1


# ----------------------------------------------------------------------------
# Program and compiler
# ----------------------------------------------------------------------------

class SynthProgram:
    """A compiled, topologically ordered execution list."""

    __slots__ = ('units', 'sinks', 'generation')

    def __init__(self, units, sinks, generation=0):
        self.units = units
        self.sinks = sinks
        self.generation = generation

    def render(self, mix, frames):
        for unit in self.units:
            unit.run(frames)
        for sink in self.sinks:
            sink.mix_into(mix, frames)


class SynthGraph:
    """Registry of live synth nodes plus the compiler that orders their units.

    There is no per-node notification when a patch cord is made or broken, so
    rather than patching the core node classes we compare a cheap topology
    signature once per frame. The signature covers every signal inlet's parent
    list, which catches link creation, deletion, node deletion, patch load,
    paste and undo alike.
    """

    def __init__(self):
        self.nodes = []
        self.engine = None
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self._signature = None
        self._last_frame = -1
        self._generation = 0
        self.last_error = ''
        self.cycle_nodes = []

    # -- registration -------------------------------------------------------

    def register(self, node):
        if node not in self.nodes:
            self.nodes.append(node)
        self._signature = None

    def unregister(self, node):
        if node in self.nodes:
            self.nodes.remove(node)
        self._signature = None
        # Recompile at once: a deleted node's unit must stop rendering before
        # the node object goes away.
        self.compile()

    def attach_engine(self, engine):
        self.engine = engine
        if engine is not None:
            self.sample_rate = float(engine.sample_rate)

    # -- per-frame poll -----------------------------------------------------

    def tick(self, frame_number):
        """Called from every synth node's frame task; acts once per frame."""
        if frame_number == self._last_frame:
            return
        self._last_frame = frame_number
        signature = self._compute_signature()
        if signature != self._signature:
            self._signature = signature
            self.compile()

    def _compute_signature(self):
        parts = []
        for node in self.nodes:
            entry = [id(node)]
            for port in node.signal_inputs:
                entry.append(tuple(id(parent) for parent in port._parents))
            parts.append(tuple(entry))
        return tuple(parts)

    # -- compilation --------------------------------------------------------

    def compile(self):
        """Rebuild the execution order and hand it to the audio thread."""
        self.cycle_nodes = []
        self.last_error = ''

        units = []
        unit_owner = {}
        for node in self.nodes:
            unit = getattr(node, 'unit', None)
            if unit is None:
                continue
            units.append(unit)
            unit_owner[id(unit)] = node

        by_id = {id(unit): unit for unit in units}
        edges = set()

        # Resolve every signal inlet to the Signal objects feeding it, and
        # record the producing units as graph edges.
        for node in self.nodes:
            unit = getattr(node, 'unit', None)
            if unit is None:
                continue
            for port in node.signal_inputs:
                inlet = port.synth_inlet
                if inlet is None:
                    continue
                sources = []
                for parent in port._parents:
                    producer = getattr(parent, 'synth_signal', None)
                    if producer is None:
                        continue
                    sources.append(producer)
                    producer_unit = getattr(parent, 'synth_unit', None)
                    # A unit feeding itself would need an explicit one-block
                    # delay to be meaningful; skip the self edge so the sort
                    # still succeeds rather than reporting a false cycle.
                    if producer_unit is not None and producer_unit is not unit \
                            and id(producer_unit) in by_id:
                        edges.add((id(producer_unit), id(unit)))
                inlet.sources = sources

        indegree = {key: 0 for key in by_id}
        outgoing = {key: [] for key in by_id}
        for producer_id, consumer_id in edges:
            outgoing[producer_id].append(consumer_id)
            indegree[consumer_id] += 1

        ready = [unit for unit in units if indegree[id(unit)] == 0]
        ordered = []
        while ready:
            unit = ready.pop()
            ordered.append(unit)
            for consumer_id in outgoing[id(unit)]:
                indegree[consumer_id] -= 1
                if indegree[consumer_id] == 0:
                    ready.append(by_id[consumer_id])

        if len(ordered) != len(units):
            placed = {id(unit) for unit in ordered}
            stuck = [unit for unit in units if id(unit) not in placed]
            self.cycle_nodes = [unit_owner.get(id(unit)) for unit in stuck]
            self.last_error = ('feedback loop through ' + str(len(stuck))
                               + ' unit(s); needs an explicit delay')
            # Run them anyway, after everything else, reading the previous
            # block's buffers. The loop works, just with one block of latency.
            ordered.extend(stuck)

        sinks = [unit for unit in ordered if isinstance(unit, AudioOutUnit)]

        self._generation += 1
        program = SynthProgram(ordered, sinks, self._generation)

        engine = self.engine
        if engine is not None:
            # Plain attribute assignment; the audio thread reads it once at the
            # top of its callback, so the swap is atomic under the GIL.
            engine.program = program
        return program


synth_graph = SynthGraph()
