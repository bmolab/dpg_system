
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
                      0.0, 0.0, output)
        breath = np.full(wide.shape[0], 0.8)
        for shape in (0, 1):
            _wind_kernel(breath, zeros, line, 0, line.copy(), 0,
                         line.copy(), 0, taps, taps, zeros,
                         -0.3, 0.6, 0.6, shape,
                         0.0, 0.0, 0.0, 0.0, 0.0, output)
        _bow_kernel(breath, breath, line.copy(), line.copy(), 0,
                    taps, taps, zeros, 0.995, 0.0, 0.0, 0.0, output)
        _brass_kernel(breath, zeros, line, 0, line.copy(), 0, taps,
                      zeros, 1.9, -0.98, 0.04, 0.5, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, output)
        _rub_kernel(breath, breath.copy(), breath.copy(), bank.copy(),
                    bank.copy(), bank.copy(), bank.copy(), state.copy(),
                    state.copy(), state.copy(), 0.995, 0.0, 0.0, output)
        _shaker_kernel(breath, 0.01, 0.999, 0.99, 0.4, 1.0, 0.5, 0.9, 0.5,
                       0.0, 0.0, 0.5, 0.99, np.uint64(12345), 0.0, 0.0,
                       output.copy(), output)
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
                         dc_x, dc_y, out):
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
            y = (gains[m] * hp + strike_gains[m] * tap
                 + b1[m] * s1[m] + b2[m] * s2[m])
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
            total += y
        out[i] = total
    return dc_x, dc_y


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
    sounds the same at any decay. The audio inlet is a drive, normalized by
    sqrt(1-r): bowing is heard the moment it touches, swells while it is
    held, and a drive parked on a mode settles against the soft stop on the
    mode states instead of being multiplied out to the mode's Q -- which at
    a three-second decay would be thousands. The price is that a click
    patched into the audio inlet rings only faintly; strikes belong to the
    trigger, which carries velocity anyway. Modes driven past Nyquist are
    muted rather than folded.

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
    # Default level of the audio-drive path against the sqrt(1-r)
    # normalization: how hard bowing and driving speak, before the state
    # soft-stop has its say. The 'drive' inlet scales from here.
    DRIVE_GAIN = 0.7

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.excite_in = self.new_inlet()
        self.trigger_in = self.new_inlet()
        self.frequency_in = self.new_inlet(base=220.0, minimum=1.0)
        self.pitch_in = self.new_inlet()
        self.decay_in = self.new_inlet(base=3.0, minimum=0.01, maximum=60.0)
        self.brightness_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.hardness_in = self.new_inlet(base=0.5, minimum=0.0, maximum=1.0)
        self.position_in = self.new_inlet(base=0.0, minimum=0.0, maximum=1.0)
        self.drive_in = self.new_inlet(base=ModalUnit.DRIVE_GAIN,
                                       minimum=0.0, maximum=2.0)
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
        self._live_count = 0
        self._level_live = ModalUnit.DRIVE_GAIN
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
        drive_level = self.drive_in.eval(frames)
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
        modes = self._modes
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
        f0 = min(self.sample_rate * 0.45, max(1.0, f0))

        seconds = decay.value if decay.constant else float(decay.data[0])
        seconds = min(60.0, max(0.01, seconds))

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

        b1 = self._b1[:count]
        np.cos(theta, out=b1)
        b1 *= radius
        b1 *= 2.0
        b2 = self._b2[:count]
        np.multiply(radius, radius, out=b2)
        np.negative(b2, out=b2)

        gains = self._gains[:count]
        np.sin(theta, out=gains)
        gains *= weights
        # Divided through by the table's total weight, a velocity-1 strike
        # peaks near +-1 whatever the material and however many modes ring.
        gains /= self._weight_norm
        # Mute rather than fold whatever the transposition pushed past
        # Nyquist. The comparison writes 0/1 straight into a float scratch.
        alive = self._mode_scratch[:count]
        np.less_equal(fm, limit, out=alive, casting='unsafe')
        gains *= alive

        bright = brightness.value if brightness.constant else float(
            brightness.data[0])
        tilt = (min(1.0, max(0.0, bright)) - 0.5) * 2.0
        if tilt != 0.0:
            shape = self._mode_scratch[:count]
            np.power(ratios, tilt, out=shape)
            gains *= shape

        struck_at = position.value if position.constant else float(
            position.data[0])
        struck_at = min(1.0, max(0.0, struck_at))
        if struck_at > 0.0:
            pattern = self._mode_scratch[:count]
            np.multiply(_INDEX_RAMP[:count], math.pi * struck_at, out=pattern)
            np.sin(pattern, out=pattern)
            np.abs(pattern, out=pattern)
            # Position 0 means the idealized uniform strike, but the node
            # pattern's own limit there is every weight at zero -- a cliff.
            # The first twentieth of the travel crossfades between the two
            # readings, so leaving 0 is a slope rather than a step.
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
        level = (drive_level.value if drive_level.constant
                 else float(drive_level.data[0]))
        level = min(2.0, max(0.0, level))
        self._level_live += (level - self._level_live) * 0.35
        drive = self._drive_gains[:count]
        np.subtract(1.0, radius, out=drive)
        np.sqrt(drive, out=drive)
        drive *= live
        drive *= self._level_live

        result = self._y[:frames]
        self._dc_x, self._dc_y = _modal_kernel(
            exc, pulse, b1, b2, drive, live, self._s1[:count],
            self._s2[:count], self._dc_x, self._dc_y, result)

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
                       s1, s2, free, dc_pole, dc_x, dc_y, out):
    """Friction closed around the modal bank: bowed glass, sample by sample.

    Where bow~'s friction negotiates with a string's delay lines, this one
    negotiates with the modes themselves. Per sample: every mode rings
    freely from its own history, their velocities sum into the surface the
    bow hair is touching, the velocity difference goes through the same
    friction curve as bow~, and the resulting force is poured back into
    every mode. The loop is why a bowed mode blooms rather than just being
    filtered noise -- each slip lands in phase with the motion that caused
    it.

    What the loop does with an inharmonic table is the sound of bowed
    glass: the modes cannot phase-lock into a shared cycle the way a
    string's harmonics do, so the friction captures one -- nearly a pure
    tone -- and pushing harder or faster makes it jump modes rather than
    brighten. None of that is coded here; it is what the physics does.

    'contact' is how firmly the hair is on the surface, following bow
    speed: a stopping bow lifts off, so the glass rings out at its own
    decay instead of being damped dead by a parked bow. The DC blocker
    takes out the static deflection of a surface leaned on by a moving
    bow.
    """
    modes = b1.shape[0]
    for i in range(velocity.shape[0]):
        surface = 0.0
        for m in range(modes):
            ring = b1[m] * s1[m] + b2[m] * s2[m]
            free[m] = ring
            surface += pickup[m] * (ring - s1[m])
        dv = velocity[i] - surface
        sl = 5.0 - 4.0 * force[i]
        t = abs(dv * sl) + 0.75
        c = 1.0 / (t * t * t * t)
        if c > 1.0:
            c = 1.0
        friction = dv * c * contact[i]
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
    VELOCITY_SCALE = 0.12
    # Below this internal speed the hair is lifting off: contact fades so
    # a stopped bow releases the ring instead of damping it dead.
    CONTACT_VELOCITY = 0.005

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

        b1 = self._b1[:count]
        np.cos(theta, out=b1)
        b1 *= radius
        b1 *= 2.0
        b2 = self._b2[:count]
        np.multiply(radius, radius, out=b2)
        np.negative(b2, out=b2)

        inject = self._inject[:count]
        np.sin(theta, out=inject)
        inject *= weights
        inject /= self._weight_norm
        inject *= RubUnit.COUPLING
        alive = self._mode_scratch[:count]
        np.less_equal(fm, limit, out=alive, casting='unsafe')
        inject *= alive

        pickup = self._pickup[:count]
        pickup[:] = 1.0
        pickup *= alive
        struck_at = position.value if position.constant else float(
            position.data[0])
        struck_at = min(1.0, max(0.0, struck_at))
        if struck_at > 0.0:
            pattern = self._mode_scratch[:count]
            np.multiply(_INDEX_RAMP[:count], math.pi * struck_at, out=pattern)
            np.sin(pattern, out=pattern)
            np.abs(pattern, out=pattern)
            # Continuous out of 0 for the same reason as modal~: the
            # pattern's limit there contradicts the uniform reading.
            blend = min(1.0, struck_at / 0.05)
            if blend < 1.0:
                pattern *= blend
                pattern += 1.0 - blend
            # Both ends of the coupling: bowing at a mode's node neither
            # hears nor moves it, which is reciprocity.
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
            dc_pole, self._dc_x, self._dc_y, result)

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
                         low, rb_x, rb_y, dp1, dp2, y1, y2, dc_x, dc_y,
                         out):
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
        r = 0.95 * low
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
        out[i] = o

        write += 1
        if write >= size:
            write = 0
    return (write, noise_at, low, rb_x, rb_y, dp1, dp2, y1, y2, dc_x, dc_y)


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

        result = self._y[:frames]
        (self._write, self._noise_at, self._low, self._rb_x, self._rb_y,
         self._dp1, self._dp2, self._y1, self._y2, self._dc_x,
         self._dc_y) = _brass_kernel(
            press, namt, self._noise, self._noise_at,
            self.bore, self._write, taps, damp,
            lip_b1, lip_b2, lip_b0, BrassUnit.LIP_BIAS,
            self._low, self._rb_x, self._rb_y, self._dp1, self._dp2,
            self._y1, self._y2, self._dc_x, self._dc_y, result)

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
                          vary, amp, theta, radius, jingle,
                          energy, sound, th, gd, rng, y1, y2, out_raw, out):
    """Cook's PhISEM, sample by sample: percussion as statistics.

    Nothing here is a waveform. 'energy' is how agitated the beans are --
    pumped by the shaking gesture, settling on its own -- and each sample
    a collision either happens or does not, with a probability that rises
    with agitation and bean count. A collision tops up a fast-decaying
    grain envelope that gates raw noise: one tick of one bean. What the
    ear hears as maraca, cabasa or rain is only the statistics of those
    ticks, which is the whole insight of the model.

    The vessel is a single two-pole resonance. With 'jingle' up, every
    collision retunes it inside a band around the vessel frequency --
    Cook's trick for tambourines and sleigh bells, where each jingle
    struck is a different one.

    What persists across blocks is the retuned ANGLE, never a coefficient:
    both coefficients are derived here from the angle and the current
    radius together, so they always describe the same filter. A stored
    b1 meeting a b2 from a radius the knob has since moved would not --
    that mismatch can put a pole outside the circle, and a resonator gone
    unstable reaches float range in milliseconds.

    Every random quantity is its own draw from the generator: which
    sample collides, how hard, how long it rings, where the jingle lands,
    and the noise being gated. Sharing draws would correlate them --
    every loud tick also long and detuned -- and reusing a table would
    loop.
    """
    pump = 1.0 - energy_decay
    b1c = 2.0 * radius * math.cos(th)
    b2c = -radius * radius
    vgain = (1.0 - radius) * 2.0
    for i in range(shake.shape[0]):
        energy = energy * energy_decay + shake[i] * pump
        rng, draw = _rand01(rng)
        if draw < rate_per_sample * energy:
            rng, strength = _rand01(rng)
            sound += amp * energy * (0.5 + 0.5 * strength)
            # A pile of unlucky draws is loud; it must never be unbounded.
            if sound > 100.0:
                sound = 100.0
            # Beans are a size distribution, not a size: each collision
            # draws its own ring time, up to an octave either side of the
            # hardness setting. Uniform ticks are what makes a model sound
            # like a machine gun instead of a gourd.
            if vary > 0.0:
                rng, size = _rand01(rng)
                gd = grain_decay ** (2.0 ** ((0.5 - size) * 2.0 * vary))
            else:
                gd = grain_decay
            if jingle > 0.0:
                rng, where = _rand01(rng)
                th = theta * (1.0 + jingle * 0.8 * (where - 0.5))
                b1c = 2.0 * radius * math.cos(th)
        sound *= gd
        rng, hiss = _rand01(rng)
        grain = sound * (2.0 * hiss - 1.0)
        y = vgain * grain + b1c * y1 + b2c * y2
        y2 = y1
        y1 = y
        out_raw[i] = grain
        out[i] = y
    return energy, sound, th, gd, rng, y1, y2


if _HAVE_NUMBA:
    _shaker_kernel = njit(cache=True, fastmath=True)(_shaker_kernel_source)
else:
    _shaker_kernel = _shaker_kernel_source


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
    is. The vessel is a tunable resonance ('vessel', 'resonance'), and
    'jingle' retunes it per collision -- tambourines are many little
    bells, each collision striking a different one.

    'grains out' carries the raw collisions before the vessel: patch it
    into modal~ (drive up, dry 0) and the beans rattle inside any object
    the table editor can draw. The coupling really is one-way -- beans
    excite the vessel, the vessel does not stir the beans -- so this is
    the rare physical seam an ordinary cord models honestly.
    """

    _seeded = 0

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.shake_in = self.new_inlet(base=0.0, minimum=0.0, maximum=2.0)
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
        self._sound = 0.0
        self._th = 0.0
        self._gd = 0.99
        self._y1 = 0.0
        self._y2 = 0.0
        self._quiet = True

        self.out = self.new_outlet()
        self.grains = self.new_outlet()
        self._shake = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._raw = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self._energy = 0.0
        self._sound = 0.0
        self._y1 = 0.0
        self._y2 = 0.0
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

        energy_decay = math.exp(-1.0 / (settle_now * self.sample_rate))
        # Hard beans are short ticks: 20 ms of felt down to half a
        # millisecond of glass, exponentially.
        grain_seconds = 0.02 * (0.025 ** hard)
        grain_decay = math.exp(-1.0 / (grain_seconds * self.sample_rate))
        # Density changes the texture, not the level: grain amplitude is
        # compensated so rain is not simply louder than a maraca.
        amp = math.sqrt(64.0 / max(8.0, beans))
        theta = 2.0 * math.pi * vessel_hz / self.sample_rate
        radius = 0.85 + 0.145 * res
        # With no jingle the angle simply follows the knob; jingled, it
        # keeps the last collision's tuning until the next collision moves
        # it. Either way the kernel derives both coefficients from angle
        # and radius together, so the filter is always self-consistent.
        if jingle_now <= 0.0 or self._th == 0.0:
            self._th = theta

        raw = self._raw[:frames]
        result = self._y[:frames]
        (self._energy, self._sound, self._th, self._gd, rng_state,
         self._y1, self._y2) = _shaker_kernel(
            gesture, beans / self.sample_rate, energy_decay, grain_decay,
            vary_now, amp, theta, radius, jingle_now,
            self._energy, self._sound, self._th, self._gd, self._rng,
            self._y1, self._y2, raw, result)
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

        p = position.value if position.constant else float(position.data[0])
        target = FaderUnit.taper(p)
        start = self._level_glide
        landing = start + (target - start) * 0.35
        if abs(landing - target) < 1.0e-6:
            landing = target
        self._level_glide = landing

        self._scale_into(signal, self.out, start, landing, frames)
        if self.right_in.sources:
            self._scale_into(right_in, self.right, start, landing, frames)
        elif self.out.constant:
            self.right.set_constant(self.out.value)
        else:
            np.copyto(self.right.data[:frames], self.out.data[:frames])
            self.right.constant = False


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
