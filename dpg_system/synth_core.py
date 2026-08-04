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
import threading

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

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        self.sample_rate = float(sample_rate)
        self.inlets = []
        self.outlets = []

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
# ramp~  --  linear move to a target, arriving on schedule
# ----------------------------------------------------------------------------

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

    def render(self, frames):
        target = self.target_in.eval(frames)
        time_in = self.time_in.eval(frames)
        trigger = self.trigger_in.eval(frames)

        goal = target.value if target.constant else float(target.data[frames - 1])
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
    """Voltage controlled amplifier: signal in, gain in, signal out."""

    LINEAR = 0
    EXPONENTIAL = 1

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.gain_in = self.new_inlet(base=1.0)
        self.response = VcaUnit.LINEAR
        self.out = self.new_outlet()
        self._gain = np.zeros(MAX_BLOCK, dtype=np.float32)

    def render(self, frames):
        signal = self.signal_in.eval(frames)
        gain = self.gain_in.eval(frames)
        out = self.out

        if gain.constant:
            gain_value = max(0.0, gain.value)
            if self.response == VcaUnit.EXPONENTIAL:
                gain_value = gain_value ** 3
            if signal.constant:
                out.set_constant(signal.value * gain_value)
                return
            np.multiply(signal.data[:frames], gain_value, out=out.data[:frames])
            out.constant = False
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

        buffer = out.data[:frames]
        np.multiply(signal.array(frames), curve, out=buffer)
        out.constant = False


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
    """

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.frequency_in = self.new_inlet(base=110.0)
        self.pitch_in = self.new_inlet(base=0.0)          # octaves
        self.linear_fm_in = self.new_inlet(base=0.0)      # Hz
        self.width_in = self.new_inlet(base=0.5, minimum=0.01, maximum=0.99)
        self.phase_mod_in = self.new_inlet(base=0.0)      # cycles
        self.sync_in = self.new_inlet(base=0.0)

        self.shape = 'saw'
        self.phase = 0.0
        self.start_phase = 0.0
        self._sync_armed = True
        self._pink_state = [np.zeros(1) for _ in _PINK_POLES]
        self._pink_last = 0.0

        self.out = self.new_outlet()
        self._phase = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._increment = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._work = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._blep = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self.phase = self.start_phase

    def render(self, frames):
        out = self.out
        buffer = out.data[:frames]

        if self.shape == 'noise':
            buffer[:] = np.random.random(frames) * 2.0 - 1.0
            out.constant = False
            return
        if self.shape == 'pink':
            self._render_pink(buffer, frames)
            out.constant = False
            return

        frequency = self.frequency_in.eval(frames)
        pitch = self.pitch_in.eval(frames)
        linear_fm = self.linear_fm_in.eval(frames)
        width = self.width_in.eval(frames)
        phase_mod = self.phase_mod_in.eval(frames)
        sync = self.sync_in.eval(frames)

        increment = self._increment[:frames]
        self._build_increment(increment, frequency, pitch, linear_fm, frames)

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
        out.constant = False

    def _build_increment(self, increment, frequency, pitch, linear_fm, frames):
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

        # PolyBLEP assumes |increment| < 0.5; also keeps us under Nyquist.
        limit = self.sample_rate * 0.49
        np.clip(increment, -limit, limit, out=increment)
        increment /= self.sample_rate

    def _sync_segments(self, sync, frames):
        """(end_index, reset_at_segment_start) covering the whole block."""
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


def _warm_up_filter():
    """Compile the filter kernel off the audio thread.

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
    """Resonant multimode filter with true per-sample cutoff modulation."""

    MODES = ('lowpass', 'highpass', 'bandpass', 'notch')

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.cutoff_in = self.new_inlet(base=1000.0)          # Hz
        self.tracking_in = self.new_inlet(base=0.0)           # octaves
        self.resonance_in = self.new_inlet(base=0.0, minimum=0.0, maximum=0.99)
        self.drive_in = self.new_inlet(base=1.0, minimum=0.0)

        self.mode = 0
        self._ic1 = 0.0
        self._ic2 = 0.0

        self.out = self.new_outlet()
        self._g = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._k = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._x = np.zeros(MAX_BLOCK, dtype=np.float32)
        self._y = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._scratch = np.zeros(MAX_BLOCK, dtype=np.float64)

    def reset(self):
        self._ic1 = 0.0
        self._ic2 = 0.0

    def render(self, frames):
        signal = self.signal_in.eval(frames)
        cutoff = self.cutoff_in.eval(frames)
        tracking = self.tracking_in.eval(frames)
        resonance = self.resonance_in.eval(frames)
        drive = self.drive_in.eval(frames)

        out = self.out
        buffer = out.data[:frames]

        if signal.constant and signal.value == 0.0:
            out.set_constant(0.0)
            return

        source = self._x[:frames]
        np.copyto(source, signal.array(frames))

        if drive.constant:
            if drive.value != 1.0:
                np.multiply(source, drive.value, out=source)
                np.tanh(source, out=source)
        else:
            np.multiply(source, drive.data[:frames], out=source)
            np.tanh(source, out=source)

        if not _svf_ready.is_set():
            # Kernel still compiling (or numba missing): pass audio rather
            # than stall the callback or emit a click.
            np.copyto(buffer, source)
            out.constant = False
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


class AudioOutUnit(Unit):
    """Terminus. Mixes its input into the engine's output buffer."""

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate)
        self.signal_in = self.new_inlet()
        self.right_in = self.new_inlet()
        self.level_in = self.new_inlet(base=0.5, minimum=0.0)
        self.position_in = self.new_inlet(base=0.0, minimum=-1.0, maximum=1.0)
        self.stereo = False
        self.muted = False
        self.peak = 0.0
        self._left = np.zeros(MAX_BLOCK, dtype=np.float32)
        self._right = np.zeros(MAX_BLOCK, dtype=np.float32)

    def render(self, frames):
        if self.muted:
            self.peak = 0.0
            return

        signal = self.signal_in.eval(frames)
        level = self.level_in.eval(frames)
        position = self.position_in.eval(frames)

        left = self._left[:frames]
        right = self._right[:frames]

        np.copyto(left, signal.array(frames))
        if self.stereo:
            np.copyto(right, self.right_in.eval(frames).array(frames))
        else:
            np.copyto(right, left)

        if level.constant:
            if level.value != 1.0:
                left *= level.value
                right *= level.value
        else:
            gain = level.data[:frames]
            left *= gain
            right *= gain

        if position.constant:
            if position.value != 0.0:
                angle = (position.value + 1.0) * (math.pi * 0.25)
                left *= math.cos(angle) * math.sqrt(2.0)
                right *= math.sin(angle) * math.sqrt(2.0)
        else:
            angle = (position.data[:frames] + 1.0) * (math.pi * 0.25)
            left *= np.cos(angle) * math.sqrt(2.0)
            right *= np.sin(angle) * math.sqrt(2.0)

        self.peak = float(max(np.max(np.abs(left)), np.max(np.abs(right))))

    def mix_into(self, mix, frames):
        if self.muted:
            return
        mix[:frames, 0] += self._left[:frames]
        if mix.shape[1] > 1:
            mix[:frames, 1] += self._right[:frames]


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

        self._pos = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._inc = np.zeros(MAX_BLOCK, dtype=np.float64)
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
            unit.render(frames)
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
