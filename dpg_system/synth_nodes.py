"""
Modular synthesis nodes for dpg_system.

Signal objects carry a trailing ~ in the Max/Pd tradition, because signal cords
behave differently from control cords: a cord between two ~ objects declares
topology to the compiler in synth_core.py and never transports data itself.
Control cords into the same inlets still work normally and set the knob value.

Every modulatable parameter is the analog triad -- a knob (the inlet's base
value), a CV inlet (patch a signal to it), and an attenuverter in the options
(the 'depth' entry). An unpatched parameter costs nothing at render time, so
there is no penalty for exposing all of them.
"""

import dearpygui.dearpygui as dpg

from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.synth_core import (
    synth_graph, start_filter_warm_up,
    SigUnit, VcoUnit, VcfUnit, VcaUnit, AdsrUnit, LfoUnit, ClockUnit, RampUnit,
    ShaperUnit,
    MixUnit, MultUnit, PanUnit, AudioOutUnit, SnapshotUnit, ScalerUnit,
    CaptureUnit, SamplerOscUnit, SamplerBuffer,
    LFO_SHAPES, VCO_SHAPES, SAMPLER_MODES,
)

import os

import numpy as np

AUDIO_FILE_EXTENSIONS = ('.wav', '.aif', '.aiff', '.mp3', '.flac', '.ogg', '.m4a')


def register_synth_nodes():
    Node.app.register_node('audio_out~', AudioOutNode.factory)
    Node.app.register_node('sig~', SigNode.factory)
    Node.app.register_node('ramp~', RampNode.factory)
    Node.app.register_node('line~', RampNode.factory)
    Node.app.register_node('vco~', VcoNode.factory)
    Node.app.register_node('vcf~', VcfNode.factory)
    Node.app.register_node('vca~', VcaNode.factory)
    Node.app.register_node('adsr~', AdsrNode.factory)
    Node.app.register_node('lfo~', LfoNode.factory)
    Node.app.register_node('phasor~', LfoNode.factory)
    Node.app.register_node('clock~', ClockNode.factory)
    Node.app.register_node('metro~', ClockNode.factory)
    Node.app.register_node('mix~', MixNode.factory)
    Node.app.register_node('pan~', PanNode.factory)
    Node.app.register_node('shaper~', ShaperNode.factory)
    Node.app.register_node('lookup~', ShaperNode.factory)
    Node.app.register_node('envelope~', ShaperNode.factory)
    Node.app.register_node('scaler~', ScalerNode.factory)
    Node.app.register_node('scale~', ScalerNode.factory)
    Node.app.register_node('mult~', MultNode.factory)
    Node.app.register_node('*~', MultNode.factory)
    Node.app.register_node('ring~', MultNode.factory)
    Node.app.register_node('snapshot~', SnapshotNode.factory)
    Node.app.register_node('sampler_osc~', SamplerOscNode.factory)
    Node.app.register_node('capture~', CaptureNode.factory)
    Node.app.register_node('array~', CaptureNode.factory)
    Node.app.register_node('scope~', CaptureNode.factory)
    # Compile the filter kernel now, during startup, so the first vcf~ the
    # user patches is already band-limited rather than passing audio dry.
    start_filter_warm_up()


def ensure_engine():
    """Attach the shared SamplerEngine so synth and sampler use one stream.

    Opening a second output stream on the same device is a good way to get
    glitches and device contention, so the modular graph mixes into the engine
    the sampler nodes already own.
    """
    try:
        from dpg_system.sampler_nodes import SamplerEngineNode
        from dpg_system.sampler import SamplerEngine
    except ImportError as error:
        print('synth_nodes: sampler engine unavailable (' + str(error) + ')')
        return None

    engine = SamplerEngineNode.engine
    if engine is None:
        engine = SamplerEngine()
        if not engine.start():
            return None
        SamplerEngineNode.engine = engine
    if synth_graph.engine is not engine:
        synth_graph.attach_engine(engine)
        synth_graph.compile()
    return engine


class SynthNode(Node):
    """Base for every ~ object: owns a DSP unit and keeps it in sync.

    The unit is created once and lives for the node's lifetime. Recompiles
    reorder units but never replace them, so oscillator phase, filter state and
    envelope stage survive repatching.
    """

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = None
        self.signal_inputs = []
        self.signal_outputs = []
        self._parameter_bindings = []
        self._depth_bindings = []
        self._custom_bindings = []
        self._registered = False

    # -- port construction --------------------------------------------------

    def add_signal_input(self, label, inlet):
        """An inlet that only accepts a patched signal (an audio input)."""
        port = self.add_input(label)
        port.synth_inlet = inlet
        self.signal_inputs.append(port)
        return port

    def add_modulation_input(self, label, inlet, widget_type='drag_float',
                             default_value=None, minimum=None, maximum=None,
                             speed=None, attenuverter=True):
        """Knob + CV inlet + attenuverter for one parameter."""
        if default_value is None:
            default_value = inlet.base
        port = self.add_input(label, widget_type=widget_type,
                              default_value=default_value,
                              min=minimum, max=maximum,
                              callback=self.parameters_changed)
        if speed is not None and port.widget is not None:
            port.widget.speed = speed
        port.synth_inlet = inlet
        self.signal_inputs.append(port)
        self._parameter_bindings.append((port, inlet))

        if attenuverter:
            option = self.add_option(label + ' depth', widget_type='drag_float',
                                     default_value=inlet.depth,
                                     callback=self.parameters_changed)
            if option.widget is not None:
                option.widget.speed = 0.01
            self._depth_bindings.append((option, inlet))
        return port

    def add_scaling_signal_input(self, label, inlet, setter, default_value=1.0,
                                 speed=0.01):
        """A signal inlet whose knob scales rather than offsets.

        add_modulation_input binds its widget to inlet.base, which the inlet
        *adds* to the CV. Where the knob needs to mean something else -- a
        multiplier, say -- the value goes through `setter` instead.
        """
        port = self.add_input(label, widget_type='drag_float',
                              default_value=default_value,
                              callback=self.parameters_changed)
        if port.widget is not None:
            port.widget.speed = speed
        port.synth_inlet = inlet
        self.signal_inputs.append(port)
        self._custom_bindings.append((port, setter))
        return port

    def add_trigger_signal_input(self, label, inlet, callback):
        """A signal inlet that is also a button, and answers a bang.

        Patch a signal and the unit edge-detects it at sample accuracy; click
        the button or send a bang from an ordinary cord and `callback` runs.
        Both paths work at once, and the button's value never touches the
        inlet, so the two cannot fight.
        """
        port = self.add_input(label, widget_type='button', callback=callback)
        port.synth_inlet = inlet
        self.signal_inputs.append(port)
        return port

    def add_signal_output(self, label, signal):
        port = self.add_output(label)
        port.synth_signal = signal
        port.synth_unit = self.unit
        self.signal_outputs.append(port)
        return port

    def finish_synth_node(self):
        """Call at the end of __init__, once ports and the unit exist."""
        ensure_engine()
        synth_graph.register(self)
        self._registered = True
        self.add_frame_task()

    # -- parameter sync -----------------------------------------------------

    def parameters_changed(self):
        """Push every widget value into the unit.

        One callback for all parameters rather than a closure per widget: the
        cost is a handful of float assignments, and it stays correct no matter
        what order the loader restores widgets in.
        """
        if self.unit is None:
            return
        for port, inlet in self._parameter_bindings:
            inlet.base = any_to_float(port())
        for option, inlet in self._depth_bindings:
            inlet.depth = any_to_float(option())
        for port, setter in self._custom_bindings:
            setter(any_to_float(port()))
        self.sync_options()

    def sync_options(self):
        """Subclass hook for non-numeric settings (shape, mode, flags)."""
        pass

    def update_parameters_from_widgets(self):
        # Called by the loader once a patch has finished restoring widgets.
        self.parameters_changed()

    # -- lifecycle ----------------------------------------------------------

    def frame_task(self):
        # Any synth node drives the shared topology check; it acts once per
        # frame no matter how many nodes call it.
        synth_graph.tick(Node.app.frame_number)
        self.synth_frame_task()

    def synth_frame_task(self):
        pass

    def custom_cleanup(self):
        if self._registered:
            synth_graph.unregister(self)
            self._registered = False


# ----------------------------------------------------------------------------
# sig~
# ----------------------------------------------------------------------------

class SigNode(SynthNode):
    """Control value into the audio graph, with glide.

    This is the entry point for effort data. A 60 Hz stream stepping a VCA gain
    zippers; glide smooths it into a continuous signal without the value ever
    lagging more than the glide time.
    """

    @staticmethod
    def factory(name, data, args=None):
        return SigNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = SigUnit(synth_graph.sample_rate)

        initial = 0.0
        if args is not None and len(args) > 0:
            value, arg_type = decode_arg(args, 0)
            if arg_type in [float, int]:
                initial = float(value)
        self.unit.target = initial

        self.value_input = self.add_input('value', widget_type='drag_float',
                                          default_value=initial,
                                          triggers_execution=True,
                                          callback=self.value_changed)
        if self.value_input.widget is not None:
            self.value_input.widget.speed = 0.01

        self.glide_input = self.add_input('glide', widget_type='drag_float',
                                          default_value=self.unit.glide,
                                          min=0.0, callback=self.settings_changed)
        if self.glide_input.widget is not None:
            self.glide_input.widget.speed = 0.001

        self.scale_input = self.add_input('scale', widget_type='drag_float',
                                          default_value=1.0,
                                          callback=self.settings_changed)
        self.offset_input = self.add_input('offset', widget_type='drag_float',
                                           default_value=0.0,
                                           callback=self.settings_changed)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.finish_synth_node()

    def value_changed(self):
        self.unit.target = any_to_float(self.value_input())

    def settings_changed(self):
        self.unit.glide = max(0.0, any_to_float(self.glide_input()))
        self.unit.scale = any_to_float(self.scale_input())
        self.unit.offset = any_to_float(self.offset_input())

    def update_parameters_from_widgets(self):
        self.value_changed()
        self.settings_changed()

    def execute(self):
        self.unit.target = any_to_float(self.value_input())


# ----------------------------------------------------------------------------
# ramp~
# ----------------------------------------------------------------------------

class RampNode(SynthNode):
    """Linear ramp to a target over a set time. Also registered as line~.

    Send a value to 'target' and the output leaves where it is and arrives at
    the new value exactly `time` seconds later. Re-aim it mid-move and it
    starts a fresh line from wherever it had got to, so a stream of targets
    never steps.

    This is the counterpart to sig~, not a replacement: sig~'s glide is a
    one-pole approach, which never quite arrives and is the right tool for
    de-zippering a control stream. Reach for ramp~ when the timing of the move
    matters -- a step from shape_seq stretched across the beat, a filter sweep
    that must land with the next hit, a fade of a stated length.

    The 'done' outlet bangs on arrival, so ramps can be chained or used to
    time anything else. 'time' is read when a move begins; changing it affects
    the next move rather than the one in flight.

    Arguments: ramp~ <time in seconds> <starting value>, e.g. 'ramp~ 0.25'.
    """

    @staticmethod
    def factory(name, data, args=None):
        return RampNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = RampUnit(synth_graph.sample_rate)
        self._last_arrive_count = 0

        numbers = []
        if args is not None:
            for arg in args:
                try:
                    numbers.append(float(arg))
                except (ValueError, TypeError):
                    continue
        if len(numbers) > 0:
            self.unit.time_in.base = max(0.0, numbers[0])
        if len(numbers) > 1:
            # A starting value the output already holds, so the first target
            # ramps from it rather than from zero.
            self.unit.current = numbers[1]
            self.unit._goal = numbers[1]
            self.unit.target_in.base = numbers[1]

        self.add_modulation_input('target', self.unit.target_in, speed=0.01)
        self.add_modulation_input('time', self.unit.time_in,
                                  default_value=self.unit.time_in.base,
                                  minimum=0.0, speed=0.001, attenuverter=False)
        self.add_trigger_signal_input('trigger', self.unit.trigger_in,
                                      self.restart)

        self.jump_option = self.add_option('jump to target',
                                           widget_type='button', width=110,
                                           callback=self.jump_now)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.done_output = self.add_output('done')
        self.finish_synth_node()

    def restart(self):
        self.unit.restart()

    def jump_now(self):
        self.unit.jump()

    def synth_frame_task(self):
        # Several short ramps can land between GUI frames; report each arrival
        # rather than only the most recent state.
        count = self.unit.arrive_count
        if count != self._last_arrive_count:
            self._last_arrive_count = count
            self.done_output.send('bang')


# ----------------------------------------------------------------------------
# vco~
# ----------------------------------------------------------------------------

class VcoNode(SynthNode):
    """Band-limited oscillator.

    Pitch is a base frequency in Hz scaled by the exponential 'pitch' inlet in
    octaves (patch an envelope there for sweeps, an LFO for vibrato), with a
    separate linear FM inlet in Hz for inharmonic tones.
    """

    @staticmethod
    def factory(name, data, args=None):
        return VcoNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = VcoUnit(synth_graph.sample_rate)

        frequency = 110.0
        shape = 'saw'
        if args is not None:
            for arg in args:
                if arg in VCO_SHAPES:
                    shape = arg
                else:
                    try:
                        frequency = float(arg)
                    except (ValueError, TypeError):
                        continue
        self.unit.shape = shape
        self.unit.frequency_in.base = frequency

        self.add_modulation_input('frequency', self.unit.frequency_in,
                                  default_value=frequency, minimum=0.0,
                                  speed=1.0)
        self.add_modulation_input('pitch', self.unit.pitch_in, speed=0.01)
        self.add_modulation_input('linear fm', self.unit.linear_fm_in, speed=1.0)
        self.add_modulation_input('width', self.unit.width_in,
                                  minimum=0.01, maximum=0.99, speed=0.01)
        self.add_modulation_input('phase mod', self.unit.phase_mod_in, speed=0.01)
        self.add_signal_input('sync', self.unit.sync_in)

        self.shape_input = self.add_input('shape', widget_type='combo',
                                          default_value=shape,
                                          callback=self.parameters_changed)
        self.shape_input.widget.combo_items = list(VCO_SHAPES)

        self.phase_option = self.add_option('start phase', widget_type='drag_float',
                                            default_value=0.0, min=0.0, max=1.0,
                                            callback=self.parameters_changed)
        self.reset_option = self.add_option('reset phase', widget_type='button',
                                            callback=self.reset_phase)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.finish_synth_node()

    def sync_options(self):
        shape = any_to_string(self.shape_input())
        if shape in VCO_SHAPES:
            self.unit.shape = shape
        self.unit.start_phase = any_to_float(self.phase_option())

    def reset_phase(self):
        self.unit.reset()


# ----------------------------------------------------------------------------
# vcf~
# ----------------------------------------------------------------------------

class VcfNode(SynthNode):
    """Resonant multimode filter with per-sample cutoff modulation.

    'tracking' is an exponential cutoff input in octaves, so patching the same
    signal that drives a vco~'s pitch inlet makes the filter track the
    oscillator. 'drive' saturates into the filter for a dirtier tone.
    """

    @staticmethod
    def factory(name, data, args=None):
        return VcfNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = VcfUnit(synth_graph.sample_rate)

        mode = 'lowpass'
        cutoff = 1000.0
        if args is not None:
            for arg in args:
                if arg in VcfUnit.MODES:
                    mode = arg
                else:
                    try:
                        cutoff = float(arg)
                    except (ValueError, TypeError):
                        continue
        self.unit.mode = VcfUnit.MODES.index(mode)
        self.unit.cutoff_in.base = cutoff

        self.add_signal_input('in', self.unit.signal_in)
        self.add_modulation_input('cutoff', self.unit.cutoff_in,
                                  default_value=cutoff, minimum=1.0, speed=5.0)
        self.add_modulation_input('tracking', self.unit.tracking_in, speed=0.01)
        self.add_modulation_input('resonance', self.unit.resonance_in,
                                  minimum=0.0, maximum=0.99, speed=0.01)
        self.add_modulation_input('drive', self.unit.drive_in,
                                  minimum=0.0, speed=0.01)

        self.mode_input = self.add_input('mode', widget_type='combo',
                                         default_value=mode,
                                         callback=self.parameters_changed)
        self.mode_input.widget.combo_items = list(VcfUnit.MODES)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.finish_synth_node()

    def sync_options(self):
        mode = any_to_string(self.mode_input())
        if mode in VcfUnit.MODES:
            self.unit.mode = VcfUnit.MODES.index(mode)


# ----------------------------------------------------------------------------
# vca~
# ----------------------------------------------------------------------------

class VcaNode(SynthNode):
    """Voltage controlled amplifier.

    Gain is the sum of the knob and any patched CV, so the usual patch is knob
    at 0 with an adsr~ into the gain inlet.
    """

    RESPONSES = ('linear', 'exponential')

    @staticmethod
    def factory(name, data, args=None):
        return VcaNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = VcaUnit(synth_graph.sample_rate)

        response = 'linear'
        gain = 1.0
        if args is not None:
            for arg in args:
                if arg in VcaNode.RESPONSES:
                    response = arg
                else:
                    try:
                        gain = float(arg)
                    except (ValueError, TypeError):
                        continue
        self.unit.gain_in.base = gain
        self.unit.response = VcaNode.RESPONSES.index(response)

        self.add_signal_input('in', self.unit.signal_in)
        self.add_modulation_input('gain', self.unit.gain_in,
                                  default_value=gain, minimum=0.0, speed=0.01)

        self.response_input = self.add_input('response', widget_type='combo',
                                             default_value=response,
                                             callback=self.parameters_changed)
        self.response_input.widget.combo_items = list(VcaNode.RESPONSES)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.finish_synth_node()

    def sync_options(self):
        response = any_to_string(self.response_input())
        if response in VcaNode.RESPONSES:
            self.unit.response = VcaNode.RESPONSES.index(response)


# ----------------------------------------------------------------------------
# adsr~
# ----------------------------------------------------------------------------

class AdsrNode(SynthNode):
    """Audio-rate envelope generator, with both a gate and a one-shot trigger.

    The 'gate' inlet is the sustaining input: hold it up and the envelope goes
    attack, decay, then sits at sustain until it is let go. Tick the checkbox
    by hand, send it a 0/1 from the patch, or drive it from a sig~ carrying
    thresholded effort.

    The 'trigger' inlet fires one shot and needs nothing held: attack, decay,
    then straight on into release. That makes A/D/R the whole contour of a hit,
    with sustain acting as the level the two decay stages meet at -- set
    sustain to 0 for a plain AD percussive shape, or leave it up for a
    two-stage tail. Click it, send it a bang from any ordinary node, or patch
    a signal for sample-accurate triggering from a comparator or an LFO.

    A gate going up takes command back from a shot in flight, so the two can
    share one envelope without deadlocking each other. Gate transitions and
    trigger edges inside a block are both acted on at the sample they occur,
    so neither is quantized to the block boundary.
    """

    @staticmethod
    def factory(name, data, args=None):
        return AdsrNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = AdsrUnit(synth_graph.sample_rate)
        self._last_finish_count = 0

        times = []
        if args is not None:
            for arg in args:
                try:
                    times.append(float(arg))
                except (ValueError, TypeError):
                    continue
        if len(times) > 0:
            self.unit.attack_in.base = times[0]
        if len(times) > 1:
            self.unit.decay_in.base = times[1]
        if len(times) > 2:
            self.unit.sustain_in.base = times[2]
        if len(times) > 3:
            self.unit.release_in.base = times[3]

        self.add_modulation_input('gate', self.unit.gate_in,
                                  widget_type='checkbox', default_value=False,
                                  attenuverter=False)
        self.add_trigger_signal_input('trigger', self.unit.trigger_in,
                                      self.fire_once)
        self.add_modulation_input('attack', self.unit.attack_in,
                                  minimum=0.0, speed=0.001)
        self.add_modulation_input('decay', self.unit.decay_in,
                                  minimum=0.0, speed=0.001)
        self.add_modulation_input('sustain', self.unit.sustain_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('release', self.unit.release_in,
                                  minimum=0.0, speed=0.001)

        self.retrigger_option = self.add_option('retrigger', widget_type='checkbox',
                                                default_value=True,
                                                callback=self.parameters_changed)
        self.legato_option = self.add_option('legato', widget_type='checkbox',
                                             default_value=False,
                                             callback=self.parameters_changed)
        self.threshold_option = self.add_option('gate threshold',
                                                widget_type='drag_float',
                                                default_value=0.5,
                                                callback=self.parameters_changed)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.done_output = self.add_output('done')
        self.finish_synth_node()

    def fire_once(self):
        self.unit.fire()

    def sync_options(self):
        self.unit.retrigger = any_to_bool(self.retrigger_option())
        self.unit.legato = any_to_bool(self.legato_option())
        self.unit.threshold = any_to_float(self.threshold_option())

    def synth_frame_task(self):
        # The envelope can finish several times between GUI frames; report
        # each completion rather than only the most recent state.
        count = self.unit.finish_count
        if count != self._last_finish_count:
            self._last_finish_count = count
            self.done_output.send('bang')


# ----------------------------------------------------------------------------
# lfo~
# ----------------------------------------------------------------------------

class LfoNode(SynthNode):
    """Low frequency oscillator, running at audio rate.

    Nothing stops the rate from reaching the audio range, where it becomes an
    ordinary (unbandlimited) modulator for FM and AM.

    Registered as phasor~ as well, which starts as a unipolar 0..1 ramp. That
    is the shape to index a table or drive shaper~ with -- not vco~'s saw,
    whose band limiting is what makes it a good oscillator and a bad index.
    """

    @staticmethod
    def factory(name, data, args=None):
        return LfoNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = LfoUnit(synth_graph.sample_rate)

        rate = 1.0
        # phasor~ is the same oscillator wearing the name it is wanted under:
        # a unipolar 0..1 ramp for indexing tables and driving shaper~. It has
        # to be this one rather than vco~'s saw, because band limiting spreads
        # the wrap over a couple of samples with intermediate values, and a
        # lookup maps those through the middle of the curve -- a spike at every
        # cycle even when the curve's endpoints match.
        phasor = label == 'phasor~'
        shape = 'ramp' if phasor else 'sine'
        if args is not None:
            for arg in args:
                if arg in LFO_SHAPES:
                    shape = arg
                else:
                    try:
                        rate = float(arg)
                    except (ValueError, TypeError):
                        continue
        self.unit.shape = shape
        self.unit.rate_in.base = rate

        self.add_modulation_input('rate', self.unit.rate_in,
                                  default_value=rate, speed=0.01)
        self.add_modulation_input('depth', self.unit.depth_in, speed=0.01)
        self.add_modulation_input('offset', self.unit.offset_in, speed=0.01)
        self.add_modulation_input('width', self.unit.width_in,
                                  minimum=0.01, maximum=0.99, speed=0.01)
        self.add_signal_input('reset', self.unit.reset_in)

        self.shape_input = self.add_input('shape', widget_type='combo',
                                          default_value=shape,
                                          callback=self.parameters_changed)
        self.shape_input.widget.combo_items = list(LFO_SHAPES)

        # A phasor runs 0..1, which lines up with shaper~'s default input range.
        self.bipolar_option = self.add_option('bipolar', widget_type='checkbox',
                                              default_value=not phasor,
                                              callback=self.parameters_changed)
        self.phase_option = self.add_option('start phase', widget_type='drag_float',
                                            default_value=0.0, min=0.0, max=1.0,
                                            callback=self.parameters_changed)
        self.reset_option = self.add_option('reset now', widget_type='button',
                                            callback=self.reset_phase)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.finish_synth_node()

    def sync_options(self):
        shape = any_to_string(self.shape_input())
        if shape in LFO_SHAPES:
            self.unit.shape = shape
        self.unit.bipolar = any_to_bool(self.bipolar_option())
        self.unit.start_phase = any_to_float(self.phase_option())

    def reset_phase(self):
        self.unit.reset()


# ----------------------------------------------------------------------------
# clock~
# ----------------------------------------------------------------------------

CLOCK_UNITS = ('hz', 'bpm', 'ms', 'seconds')


class ClockNode(SynthNode):
    """Master clock: a pulse train for the audio graph, bangs for the patch.

    The 'trigger' signal outlet is a gate whose rising edge is exact to the
    sample, so patching it into an adsr~ trigger inlet fires the envelope with
    no block quantization. The 'bang' outlet is the same clock in the ordinary
    node world, for sequencers, counters and anything else that wants a beat.
    Both run from one phase, so audio and patch never drift apart.

    Rate is a modulation inlet like any other -- patch an lfo~ or an envelope
    at it for accelerandi and rubato, or leave it on the knob. The units option
    reads the knob as hz, bpm, a period in ms, or a period in seconds; patched
    CV is always in hz and sums on top.

    Starting the clock puts it on a downbeat and ticks at once rather than
    waiting out a period. Stopping holds the phase where it was, so a stop and
    start without a reset resumes mid-bar.

    Arguments: clock~ <rate> <units>, e.g. 'clock~ 120 bpm'. Also metro~.
    """

    # A GUI frame that runs long -- a patch load, a heavy node -- leaves a
    # backlog of ticks behind it. Firing all of them would spray bangs that no
    # longer mean anything musically, so the backlog is capped and the rest of
    # that stall is simply not heard.
    MAX_TICKS_PER_FRAME = 32

    @staticmethod
    def factory(name, data, args=None):
        return ClockNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = ClockUnit(synth_graph.sample_rate)
        self._served_ticks = self.unit.tick_count
        self._tick_index = 0
        self._was_running = False

        rate = 2.0
        units = 'hz'
        if args is not None:
            for arg in args:
                if arg in CLOCK_UNITS:
                    units = arg
                else:
                    try:
                        rate = float(arg)
                    except (ValueError, TypeError):
                        continue

        self.run_input = self.add_input('run', widget_type='checkbox',
                                        default_value=False,
                                        callback=self.parameters_changed)

        # Rate cannot use add_modulation_input: that binds the widget straight
        # to inlet.base, and this knob is read through the units option first.
        # The inlet is registered by hand so a patched signal still lands on it.
        self.rate_input = self.add_input('rate', widget_type='drag_float',
                                         default_value=rate,
                                         callback=self.parameters_changed)
        if self.rate_input.widget is not None:
            self.rate_input.widget.speed = 0.01
        self.rate_input.synth_inlet = self.unit.rate_in
        self.signal_inputs.append(self.rate_input)

        self.add_modulation_input('pulse width', self.unit.width_in,
                                  minimum=0.001, maximum=0.999, speed=0.01,
                                  attenuverter=False)
        self.add_trigger_signal_input('reset', self.unit.reset_in, self.restart)

        self.units_option = self.add_option('units', widget_type='combo',
                                            default_value=units,
                                            callback=self.parameters_changed)
        self.units_option.widget.combo_items = list(CLOCK_UNITS)
        self.rate_depth_option = self.add_option('rate depth',
                                                 widget_type='drag_float',
                                                 default_value=1.0,
                                                 callback=self.parameters_changed)
        if self.rate_depth_option.widget is not None:
            self.rate_depth_option.widget.speed = 0.01
        self._depth_bindings.append((self.rate_depth_option, self.unit.rate_in))

        self.trigger_output = self.add_signal_output('trigger', self.unit.out)
        self.bang_output = self.add_output('bang')
        self.count_output = self.add_output('count')
        self.finish_synth_node()

    def rate_in_hz(self):
        """The knob, read through the units option. Patched CV is always hz."""
        value = any_to_float(self.rate_input())
        units = any_to_string(self.units_option())
        if units == 'bpm':
            return value / 60.0
        if units == 'ms':
            return 1000.0 / value if abs(value) > 1.0e-9 else 0.0
        if units == 'seconds':
            return 1.0 / value if abs(value) > 1.0e-9 else 0.0
        return value

    def sync_options(self):
        self.unit.rate_in.base = self.rate_in_hz()

        running = any_to_bool(self.run_input())
        if running != self._was_running:
            self._was_running = running
            self.unit.running = running
            if running:
                # Start on the beat rather than wherever the phase happened to
                # sit, so the first note lands when the switch is thrown.
                self.unit.restart()

    def restart(self):
        self.unit.restart()

    def synth_frame_task(self):
        count = self.unit.tick_count
        if count == self._served_ticks:
            return
        pending = count - self._served_ticks
        self._served_ticks = count
        if pending > ClockNode.MAX_TICKS_PER_FRAME:
            pending = ClockNode.MAX_TICKS_PER_FRAME
        for _ in range(pending):
            self._tick_index += 1
            self.count_output.send(self._tick_index)
            self.bang_output.send('bang')


# ----------------------------------------------------------------------------
# mix~ / pan~
# ----------------------------------------------------------------------------

class MixNode(SynthNode):
    """Signal mixer with per-input levels and a master.

    'mix~ 6' gives six inputs. Cords summing into one inlet already mix, so
    this exists for the levels, which are themselves modulatable.
    """

    @staticmethod
    def factory(name, data, args=None):
        return MixNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        channels = 4
        if args is not None and len(args) > 0:
            value, arg_type = decode_arg(args, 0)
            if arg_type == int:
                channels = max(2, min(16, int(value)))
        self.unit = MixUnit(synth_graph.sample_rate, channels=channels)

        for index in range(channels):
            suffix = str(index + 1)
            self.add_signal_input('in ' + suffix, self.unit.channel_inlets[index])
            self.add_modulation_input('level ' + suffix,
                                      self.unit.level_inlets[index],
                                      minimum=0.0, speed=0.01,
                                      attenuverter=False)
        self.add_modulation_input('master', self.unit.master_in,
                                  minimum=0.0, speed=0.01, attenuverter=False)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.finish_synth_node()


class ScalerNode(SynthNode):
    """Map a signal from one range into another, with a response curve.

    The usual case is an envelope or LFO (0..1, or -1..1) driving something
    that wants engineering units. Note that a plain linear range change is
    already available without this node -- every modulation inlet computes
    base + depth * CV, so the knob is the low end and the 'depth' option in
    the inlet's options is the span.

    Reach for scaler~ when you need what that cannot do: a curve, an explicit
    input range, clamping, or exponential (equal-ratio) output. For filter
    cutoff specifically, prefer either exponential mode here or vcf~'s own
    'tracking' inlet, which is already in octaves -- a linear sweep from 200
    to 4200 Hz spends nearly all of its travel in the top octaves.

    Arguments: scaler~ <out low> <out high>, e.g. 'scaler~ 200 4000'.
    """

    MODES = ('linear', 'exponential')

    @staticmethod
    def factory(name, data, args=None):
        return ScalerNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = ScalerUnit(synth_graph.sample_rate)

        mode = 'linear'
        numbers = []
        if args is not None:
            for arg in args:
                if arg in ScalerNode.MODES:
                    mode = arg
                else:
                    try:
                        numbers.append(float(arg))
                    except (ValueError, TypeError):
                        continue
        if len(numbers) > 0:
            self.unit.out_low_in.base = numbers[0]
        if len(numbers) > 1:
            self.unit.out_high_in.base = numbers[1]
        if len(numbers) > 2:
            self.unit.in_low_in.base = numbers[2]
        if len(numbers) > 3:
            self.unit.in_high_in.base = numbers[3]
        self.unit.mode = ScalerNode.MODES.index(mode)

        self.add_signal_input('in', self.unit.signal_in)
        self.add_modulation_input('in low', self.unit.in_low_in,
                                  speed=0.01, attenuverter=False)
        self.add_modulation_input('in high', self.unit.in_high_in,
                                  speed=0.01, attenuverter=False)
        self.add_modulation_input('out low', self.unit.out_low_in,
                                  speed=0.01, attenuverter=False)
        self.add_modulation_input('out high', self.unit.out_high_in,
                                  speed=0.01, attenuverter=False)
        self.add_modulation_input('curve', self.unit.curve_in,
                                  minimum=0.01, maximum=16.0, speed=0.01,
                                  attenuverter=False)

        self.mode_input = self.add_input('mode', widget_type='combo',
                                         default_value=mode,
                                         callback=self.parameters_changed)
        self.mode_input.widget.combo_items = list(ScalerNode.MODES)

        self.clip_option = self.add_option('clip', widget_type='checkbox',
                                           default_value=True,
                                           callback=self.parameters_changed)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.finish_synth_node()

    def sync_options(self):
        mode = any_to_string(self.mode_input())
        if mode in ScalerNode.MODES:
            self.unit.mode = ScalerNode.MODES.index(mode)
        self.unit.clip = any_to_bool(self.clip_option())


class ShaperNode(SynthNode):
    """A breakpoint curve as a transfer function, applied to every sample.

    This is the envelope node's lookup at audio rate: where envelope maps one
    x to one y per message, shaper~ maps every sample of every block. Draw the
    curve on the node itself -- drag a point to move it, right-click to add or
    remove one, shift + left-drag a segment to bend it -- and the table behind
    it is rebuilt as you draw, so the sound follows the mouse. An envelope
    node's 'points out' patched into the 'points' inlet loads a curve too, for
    when the shape is being made or sequenced elsewhere.

    What it is for depends on what you feed it. On a control signal it is an
    arbitrary response curve: effort into a shape and out to a filter cutoff,
    with the exact bend you drew rather than a choice of linear or exponential.
    On an audio signal it is a waveshaper -- a curve that is not a straight
    line adds harmonics, and a hand-drawn one adds whatever you draw. Expect
    aliasing when shaping bright material; that is the operation, not the
    implementation.

    Sweep it with phasor~ to play the curve as a waveform. Use that rather
    than vco~'s saw: a band-limited saw crosses its wrap over a couple of
    samples at intermediate values, and a lookup maps those through the middle
    of the curve, so every cycle ends in a spike however well the endpoints
    match. Band limiting is a linear repair fitted to that one waveform, and
    nothing survives being bent afterwards.

    The curve's x axis is the input, from 'in low' to 'in high'; its y axis is
    the output, bounded by the 'out low' and 'out high' options. Input outside
    the range is held at the ends ('clip'), taken modulo the range ('wrap'), or
    reflected back and forth across it ('fold') -- the last two are worth
    trying with a signal that overshoots.

    Also registered as lookup~ and envelope~.

    Arguments: shaper~ <in low> <in high> and/or a range mode.
    """

    RANGE_MODES = ('clip', 'wrap', 'fold')
    TABLE_SAMPLES_PER_SEGMENT = 256

    @staticmethod
    def factory(name, data, args=None):
        return ShaperNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = ShaperUnit(synth_graph.sample_rate)

        mode = 'clip'
        numbers = []
        if args is not None:
            for arg in args:
                if arg in ShaperNode.RANGE_MODES:
                    mode = arg
                else:
                    try:
                        numbers.append(float(arg))
                    except (ValueError, TypeError):
                        continue
        in_low, in_high = 0.0, 1.0
        if len(numbers) > 0:
            in_low = numbers[0]
        if len(numbers) > 1:
            in_high = numbers[1]
        self.unit.in_low_in.base = in_low
        self.unit.in_high_in.base = in_high
        self.unit.range_mode = ShaperNode.RANGE_MODES.index(mode)

        self.plot_width = 200
        self.plot_height = 100
        # The editor lives in the interface module. Imported here rather than
        # at module scope: the DSP nodes otherwise have no business depending
        # on the GUI layer, and this node is the one place the two meet.
        from dpg_system.interface_nodes import BreakpointEditor
        # x is the input position across the range, so the editor's x axis is
        # always 0..1; y is the output value.
        self.editor = BreakpointEditor(x_max=1.0, y_min=0.0, y_max=1.0,
                                       width=self.plot_width,
                                       height=self.plot_height,
                                       on_change=self.curve_changed,
                                       name=label)
        # 'point 1 0.5 0.8' and friends, so the curve can be moved from a patch
        # rather than only by hand. See BreakpointEditor.handle_message.
        for name in BreakpointEditor.MESSAGES:
            self.message_handlers[name] = self.curve_message

        self.add_signal_input('in', self.unit.signal_in)
        self.add_modulation_input('in low', self.unit.in_low_in,
                                  default_value=in_low,
                                  speed=0.01, attenuverter=False)
        self.add_modulation_input('in high', self.unit.in_high_in,
                                  default_value=in_high,
                                  speed=0.01, attenuverter=False)
        self.points_input = self.add_input('points', callback=self.points_received)

        self.mode_input = self.add_input('range', widget_type='combo',
                                         default_value=mode,
                                         callback=self.parameters_changed)
        self.mode_input.widget.combo_items = list(ShaperNode.RANGE_MODES)

        self.curve_display = self.add_display('')
        self.curve_display.submit_callback = self.submit_display

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.points_output = self.add_output('points out')

        self.out_low_option = self.add_option('out low', widget_type='drag_float',
                                              default_value=0.0,
                                              callback=self.ranges_changed)
        self.out_high_option = self.add_option('out high', widget_type='drag_float',
                                               default_value=1.0,
                                               callback=self.ranges_changed)
        for option in (self.out_low_option, self.out_high_option):
            if option.widget is not None:
                option.widget.speed = 0.01
        self.reset_option = self.add_option('straight line',
                                            widget_type='button', width=110,
                                            callback=self.reset_curve)
        self.width_option = self.add_option('width', widget_type='drag_int',
                                            default_value=self.plot_width,
                                            callback=self.size_changed)
        self.height_option = self.add_option('height', widget_type='drag_int',
                                             default_value=self.plot_height,
                                             callback=self.size_changed)
        self.finish_synth_node()

    # -- display ------------------------------------------------------------

    def submit_display(self):
        self.editor.submit(self.curve_display.uuid,
                           width_option=self.width_option,
                           height_option=self.height_option)

    def custom_create(self, from_file):
        # Options only hold their real values once every element has been
        # created, so anything that reads one waits until here.
        self.editor.set_ranges(y_min=any_to_float(self.out_low_option()),
                               y_max=any_to_float(self.out_high_option()),
                               notify=False)
        self.editor.set_size(any_to_int(self.width_option()),
                             any_to_int(self.height_option()))
        self.build_table()

    def size_changed(self):
        self.editor.set_size(any_to_int(self.width_option()),
                             any_to_int(self.height_option()))

    def ranges_changed(self):
        low = any_to_float(self.out_low_option())
        high = any_to_float(self.out_high_option())
        if high <= low:
            high = low + 1.0
        self.editor.set_ranges(y_min=low, y_max=high, notify=False)
        self.build_table()

    def reset_curve(self):
        self.editor.set_points(self.editor.straight_line())

    # -- curve --------------------------------------------------------------

    def curve_message(self, message='', message_data=[]):
        self.editor.handle_message(message, message_data)

    def curve_changed(self):
        """The editor moved: rebake the table and pass the points on."""
        self.build_table()
        self.points_output.send(self.editor.get_points())

    def build_table(self):
        """Sample the drawn curve onto the unit's uniform table.

        breakpoint_line_data owns what a curved segment means, so sampling it
        rather than reimplementing the easing is what keeps shaper~, envelope
        and shape_seq agreeing about the same curve.
        """
        xs, ys = self.editor.line_data(ShaperNode.TABLE_SAMPLES_PER_SEGMENT)
        if len(xs) < 2:
            return
        # The table spans the editor's whole x axis, not the extent of the
        # points. A curve whose last point sits short of the right edge holds
        # its end value across the gap, exactly as reading the breakpoints
        # would -- which is what np.interp does outside its x range. Spanning
        # the points instead would stretch the curve to fill the input range
        # and move every other point's meaning with it.
        grid = np.linspace(0.0, self.editor.x_max, ShaperUnit.TABLE_SIZE + 1)
        self.unit.set_table(np.interp(grid, xs, ys))

    def points_received(self):
        """A breakpoint list from elsewhere -- an envelope node, a preset.

        The incoming x span is normalised onto the editor's 0..1, and the out
        range grows to fit the incoming y values, so a curve drawn against any
        ranges arrives looking like itself.
        """
        points = self.editor_points_from(self.points_input())
        if not points:
            return
        low = min(point[1] for point in points)
        high = max(point[1] for point in points)
        if high <= low:
            high = low + 1.0
        if low < self.editor.y_min or high > self.editor.y_max:
            self.out_low_option.widget.set(min(low, self.editor.y_min))
            self.out_high_option.widget.set(max(high, self.editor.y_max))
            self.editor.set_ranges(y_min=any_to_float(self.out_low_option()),
                                   y_max=any_to_float(self.out_high_option()),
                                   notify=False)
        self.editor.set_points(points)

    @staticmethod
    def editor_points_from(data):
        """[[x, y, curve], ...] as the editor wants it, x normalised to 0..1."""
        points = []
        if isinstance(data, np.ndarray):
            data = data.tolist()
        if not isinstance(data, (list, tuple)):
            return points
        for entry in data:
            if isinstance(entry, np.ndarray):
                entry = entry.tolist()
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            points.append([any_to_float(entry[0]), any_to_float(entry[1]),
                           any_to_float(entry[2]) if len(entry) > 2 else 0.0])
        if len(points) < 2:
            return []
        span_low = min(p[0] for p in points)
        span_high = max(p[0] for p in points)
        span = span_high - span_low
        if span <= 0.0:
            return []
        for point in points:
            point[0] = (point[0] - span_low) / span
        return points

    def sync_options(self):
        mode = any_to_string(self.mode_input())
        if mode in ShaperNode.RANGE_MODES:
            self.unit.range_mode = ShaperNode.RANGE_MODES.index(mode)

    def synth_frame_task(self):
        self.editor.poll()

    def save_custom(self, container):
        container['shaper_points'] = self.editor.get_points()

    def load_custom(self, container):
        if 'shaper_points' in container:
            self.editor.set_points(container['shaper_points'], notify=False)
            self.build_table()


class MultNode(SynthNode):
    """Multiply signals together: ring modulation, AM, shaped modulators.

    Use this rather than vca~ whenever either signal is bipolar. vca~ is an
    amplifier -- it clamps negative gain and its knob sums with the CV -- so
    an LFO into a vca~ gain inlet loses its negative half. mult~ keeps it.

    Each knob scales its own input and is used alone when nothing is patched
    there, so unused inputs sit at 1.0 and do not collapse the product.

    Arguments: mult~ <number of inputs>, default 2. Also registered as *~
    and ring~.
    """

    @staticmethod
    def factory(name, data, args=None):
        return MultNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        inputs = 2
        if args is not None and len(args) > 0:
            value, arg_type = decode_arg(args, 0)
            if arg_type == int:
                inputs = max(2, min(8, int(value)))
        self.unit = MultUnit(synth_graph.sample_rate, inputs=inputs)

        for index in range(inputs):
            self.add_scaling_signal_input(
                'in ' + str(index + 1), self.unit.signal_inlets[index],
                self._factor_setter(index))

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.finish_synth_node()

    def _factor_setter(self, index):
        def setter(value):
            self.unit.factors[index] = value
        return setter


class PanNode(SynthNode):
    """Equal-power panner: -1 hard left, +1 hard right."""

    @staticmethod
    def factory(name, data, args=None):
        return PanNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = PanUnit(synth_graph.sample_rate)

        self.add_signal_input('in', self.unit.signal_in)
        self.add_modulation_input('position', self.unit.position_in,
                                  minimum=-1.0, maximum=1.0, speed=0.01)

        self.left_output = self.add_signal_output('left', self.unit.left)
        self.right_output = self.add_signal_output('right', self.unit.right)
        self.finish_synth_node()


# ----------------------------------------------------------------------------
# audio_out~ / snapshot~
# ----------------------------------------------------------------------------

class AudioOutNode(SynthNode):
    """Graph terminus. Mixes into the shared SamplerEngine output stream.

    Nothing is heard without one of these. Every registered unit still renders
    whether or not it reaches an output -- a snapshot~ with no audio_out~ is a
    legitimate patch -- but only audio_out~ contributes to the stream.
    """

    @staticmethod
    def factory(name, data, args=None):
        return AudioOutNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = AudioOutUnit(synth_graph.sample_rate)

        level = 0.5
        if args is not None and len(args) > 0:
            value, arg_type = decode_arg(args, 0)
            if arg_type in [float, int]:
                level = float(value)
        self.unit.level_in.base = level

        self.add_signal_input('in', self.unit.signal_in)
        self.add_signal_input('right in', self.unit.right_in)
        self.add_modulation_input('level', self.unit.level_in,
                                  default_value=level, minimum=0.0, speed=0.01,
                                  attenuverter=False)
        self.add_modulation_input('pan', self.unit.position_in,
                                  minimum=-1.0, maximum=1.0, speed=0.01,
                                  attenuverter=False)

        self.mute_input = self.add_input('mute', widget_type='checkbox',
                                         default_value=False,
                                         callback=self.parameters_changed)
        self.stereo_option = self.add_option('stereo', widget_type='checkbox',
                                             default_value=False,
                                             callback=self.parameters_changed)

        self.level_output = self.add_output('peak')
        self.status_output = self.add_output('status')
        self._last_status = ''
        self.finish_synth_node()

    def sync_options(self):
        self.unit.muted = any_to_bool(self.mute_input())
        self.unit.stereo = any_to_bool(self.stereo_option())

    def synth_frame_task(self):
        # Re-attach if the sampler engine was restarted or replaced.
        ensure_engine()
        self.level_output.send(self.unit.peak)
        status = synth_graph.last_error
        if status != self._last_status:
            self._last_status = status
            self.status_output.send(status if status else 'ok')


class SnapshotNode(SynthNode):
    """Signal to float: the bridge back into the ordinary node world.

    Patch any ~ signal in and the current value appears on the node face and
    goes out the 'value' outlet at frame rate, ready for number boxes, math
    nodes, OSC, anything. An adsr~ or lfo~ becomes an ordinary float stream.

    'peak' and 'rms' cover the whole interval since the last output rather
    than a single audio block, so short transients cannot slip between GUI
    frames.

    By default output is sent only when the value changes, so a resting
    signal does not push 60 identical floats per second into the patch. Set
    mode to 'continuous' for an unconditional stream, raise 'deadband' to
    ignore small wobble, or bang the node to force a reading.
    """

    MODES = ('on change', 'continuous')

    @staticmethod
    def factory(name, data, args=None):
        return SnapshotNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = SnapshotUnit(synth_graph.sample_rate)
        self._last_sent = None
        self._last_text = ''

        self.add_signal_input('in', self.unit.signal_in)
        self.bang_input = self.add_input('bang', widget_type='button',
                                         callback=self.send_now)

        # A label widget is read-only, so it displays without inviting edits.
        self.value_display = self.add_property('value', widget_type='label',
                                               default_value='0.000')

        self.value_output = self.add_output('value')
        self.peak_output = self.add_output('peak')
        self.rms_output = self.add_output('rms')

        self.mode_option = self.add_option('mode', widget_type='combo',
                                           default_value='on change',
                                           callback=self.parameters_changed)
        self.mode_option.widget.combo_items = list(SnapshotNode.MODES)
        self.deadband_option = self.add_option('deadband', widget_type='drag_float',
                                               default_value=0.0, min=0.0)
        if self.deadband_option.widget is not None:
            self.deadband_option.widget.speed = 0.001
        self.precision_option = self.add_option('precision', widget_type='drag_int',
                                                default_value=3, min=0, max=8)
        self.finish_synth_node()

    def _display(self, value):
        text = '%.*f' % (int(any_to_int(self.precision_option())), value)
        if text != self._last_text:
            self._last_text = text
            if self.value_display.widget is not None:
                self.value_display.widget.set(text)

    def _should_send(self, value):
        if any_to_string(self.mode_option()) == 'continuous':
            return True
        if self._last_sent is None:
            return True
        return abs(value - self._last_sent) > any_to_float(self.deadband_option())

    def _emit(self, value, peak, rms):
        self._last_sent = value
        # Right to left, so anything driven by 'value' sees the matching peak
        # and rms already in place.
        self.rms_output.send(rms)
        self.peak_output.send(peak)
        self.value_output.send(value)

    def send_now(self):
        value, peak, rms = self.unit.take()
        self._display(value)
        self._emit(value, peak, rms)

    def synth_frame_task(self):
        value, peak, rms = self.unit.take()
        self._display(value)
        if self._should_send(value):
            self._emit(value, peak, rms)


def load_sample_buffer(path, sample_rate):
    """Read an audio file into a SamplerBuffer, on the calling (main) thread.

    The file's own rate is carried on the buffer rather than resampled: the
    playback increment folds it in exactly, which is both cheaper and lossless
    compared with converting the whole file up front.
    """
    if not path or not os.path.exists(path):
        print('sampler_osc~: file not found: ' + str(path))
        return None
    try:
        import torchaudio
    except ImportError:
        print('sampler_osc~: torchaudio unavailable, cannot load audio')
        return None
    try:
        waveform, source_rate = torchaudio.load(path)
        if not waveform.is_cpu:
            waveform = waveform.cpu()
        data = waveform.numpy()
    except Exception as error:
        print('sampler_osc~: could not load ' + str(path) + ' (' + str(error) + ')')
        return None

    if data.ndim == 1:
        return SamplerBuffer(data, None, source_rate, path)
    if data.shape[0] == 1:
        return SamplerBuffer(data[0], None, source_rate, path)
    return SamplerBuffer(data[0], data[1], source_rate, path)


class SamplerOscNode(SynthNode):
    """Recorded material as a modular oscillator.

    Playback rate is a linear 'rate' multiplier times an exponential 'pitch'
    inlet in octaves, so the same envelope or LFO that sweeps a vco~ sweeps
    this identically. Source files are not resampled; their rate folds into
    the increment exactly.

    Modes, which differ in what moves the playhead:

      loop      free-running through the loop window, wrapping, with a short
                crossfade across the seam so mismatched endpoints do not click
      oneshot   plays once per trigger and stops
      scrub     playhead sits wherever the position inlet says, turntable
                style. Held still it emits a constant sample value -- DC --
                exactly as a stopped record would
      follow    position is a target the playhead chases through a spring, so
                movement becomes playback speed. The one to reach for with
                effort data: the material sounds while the body moves
      granular  grains sprayed around the position inlet

    Arguments: sampler_osc~ <path> and/or a mode name.
    """

    @staticmethod
    def factory(name, data, args=None):
        return SamplerOscNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = SamplerOscUnit(synth_graph.sample_rate)
        self._pending_path = ''
        self._shown_name = ''

        mode = 'loop'
        path = ''
        if args is not None:
            for arg in args:
                if arg in SAMPLER_MODES:
                    mode = arg
                else:
                    path = arg
        self.unit.mode = SAMPLER_MODES.index(mode)

        self.add_modulation_input('rate', self.unit.rate_in, speed=0.01)
        self.add_modulation_input('pitch', self.unit.pitch_in, speed=0.01)
        self.add_modulation_input('position', self.unit.position_in, speed=0.01)
        self.add_modulation_input('loop start', self.unit.loop_start_in,
                                  minimum=0.0, maximum=1.0, speed=0.001,
                                  attenuverter=False)
        self.add_modulation_input('loop end', self.unit.loop_end_in,
                                  minimum=0.0, maximum=1.0, speed=0.001,
                                  attenuverter=False)
        self.add_modulation_input('grain size', self.unit.grain_size_in,
                                  minimum=0.001, speed=0.001, attenuverter=False)
        self.add_modulation_input('grain rate', self.unit.grain_rate_in,
                                  minimum=0.0, speed=0.5, attenuverter=False)
        self.add_modulation_input('jitter', self.unit.jitter_in,
                                  minimum=0.0, maximum=1.0, speed=0.01,
                                  attenuverter=False)
        self.add_signal_input('trigger', self.unit.trigger_in)

        self.mode_input = self.add_input('mode', widget_type='combo',
                                         default_value=mode,
                                         callback=self.parameters_changed)
        self.mode_input.widget.combo_items = list(SAMPLER_MODES)

        self.load_input = self.add_input('load', widget_type='button',
                                         callback=self.request_load)
        self.path_input = self.add_input('path', callback=self.path_received)
        self.start_input = self.add_input('start', widget_type='button',
                                          callback=self.retrigger)

        self.name_display = self.add_property('sample', widget_type='label',
                                              default_value='(no sample)')

        self.left_output = self.add_signal_output('left', self.unit.left)
        self.right_output = self.add_signal_output('right', self.unit.right)
        self.length_output = self.add_output('length')

        self.reverse_option = self.add_option('reverse', widget_type='checkbox',
                                              default_value=False,
                                              callback=self.parameters_changed)
        self.crossfade_option = self.add_option('loop crossfade',
                                                widget_type='drag_float',
                                                default_value=0.005, min=0.0,
                                                callback=self.parameters_changed)
        if self.crossfade_option.widget is not None:
            self.crossfade_option.widget.speed = 0.001
        self.follow_option = self.add_option('follow speed',
                                             widget_type='drag_float',
                                             default_value=8.0, min=0.01,
                                             callback=self.parameters_changed)
        self.finish_synth_node()

        if path:
            self.load_path(path)

    # -- settings -----------------------------------------------------------

    def sync_options(self):
        mode = any_to_string(self.mode_input())
        if mode in SAMPLER_MODES:
            self.unit.mode = SAMPLER_MODES.index(mode)
        self.unit.reverse = any_to_bool(self.reverse_option())
        self.unit.crossfade = max(0.0, any_to_float(self.crossfade_option()))
        self.unit.follow_speed = max(0.01, any_to_float(self.follow_option()))

    def retrigger(self):
        self.unit.trigger()

    # -- loading ------------------------------------------------------------

    def request_load(self):
        try:
            from dpg_system.node import LoadDialog
        except ImportError:
            return
        LoadDialog(self, self.load_path, extensions=list(AUDIO_FILE_EXTENSIONS))

    def path_received(self):
        value = self.path_input()
        if value is None or value == 'bang':
            self.request_load()
            return
        self.load_path(any_to_string(value))

    def load_path(self, path):
        buffer = load_sample_buffer(path, synth_graph.sample_rate)
        if buffer is None:
            return
        # One assignment: the audio thread reads self.sample once per block, so
        # it sees either the whole previous sample or the whole new one.
        self.unit.sample = buffer
        self.unit.trigger()
        self._pending_path = path
        name = os.path.basename(path)
        if name != self._shown_name:
            self._shown_name = name
            if self.name_display.widget is not None:
                self.name_display.widget.set(name)
        self.length_output.send(buffer.frames / buffer.source_rate)

    def save_custom(self, container):
        container['sample_path'] = self._pending_path

    def load_custom(self, container):
        path = container.get('sample_path', '')
        if path:
            self.load_path(path)


class CaptureNode(SynthNode):
    """Signal to numpy array: hands whole blocks of audio to the node world.

    snapshot~ gives one value per frame, which is right for following a slow
    control signal but throws away the waveform. capture~ keeps every sample
    in a ring buffer and emits an array, so plot, spectrum, numpy and torch
    nodes can work on the actual signal.

    Two modes, and the difference matters:

      latest      the newest `size` samples every frame. Frames and audio
                  blocks do not divide evenly, so successive reads overlap or
                  skip. Right for a scope or a spectrum display, where you
                  want a current window and do not care about continuity.

      continuous  gapless chunks of exactly `size` samples, in order, every
                  sample delivered once. A frame that has accumulated two
                  blocks sends two chunks; a partial remainder waits for the
                  next one, so the array length never varies. Right for
                  analysis, recording or anything cumulative. 'dropped'
                  reports samples genuinely lost when the patch fell behind.

    `size` defaults to 512, which is the engine's audio block. Keeping it at
    the block size (or a multiple) means each chunk corresponds to whole
    blocks as the audio thread produced them; other values still work and
    stay gapless, the boundaries just fall inside blocks.

    Arguments: capture~ <size> and/or 'latest' | 'continuous'.
    Also registered as array~ and scope~.
    """

    MODES = ('latest', 'continuous')
    SEND_MODES = ('every frame', 'on bang')

    @staticmethod
    def factory(name, data, args=None):
        return CaptureNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = CaptureUnit(synth_graph.sample_rate)

        size = 512
        mode = 'latest'
        if args is not None:
            for arg in args:
                if arg in CaptureNode.MODES:
                    mode = arg
                else:
                    try:
                        size = int(float(arg))
                    except (ValueError, TypeError):
                        continue
        size = max(16, min(self.unit.max_window, size))

        # Start where the signal is now, so continuous mode does not open by
        # dumping a buffer of silence recorded before the node existed.
        self._last_read = self.unit.written

        self.add_signal_input('in', self.unit.signal_in)
        self.bang_input = self.add_input('bang', widget_type='button',
                                         callback=self.send_now)

        self.array_output = self.add_array_output('array')
        self.dropped_output = self.add_output('dropped')

        self.size_option = self.add_option('size', widget_type='drag_int',
                                           default_value=size, min=16,
                                           max=self.unit.max_window)
        self.mode_option = self.add_option('mode', widget_type='combo',
                                           default_value=mode)
        self.mode_option.widget.combo_items = list(CaptureNode.MODES)
        self.send_option = self.add_option('send', widget_type='combo',
                                           default_value='every frame')
        self.send_option.widget.combo_items = list(CaptureNode.SEND_MODES)
        self.finish_synth_node()

    # More than one block can land between GUI frames, so a frame may owe
    # several chunks. The cap stops a stalled patch from dumping an unbounded
    # burst; anything beyond it is caught by the overrun report instead.
    MAX_CHUNKS_PER_FRAME = 16

    def _emit(self):
        size = max(16, min(self.unit.max_window, any_to_int(self.size_option())))

        if any_to_string(self.mode_option()) != 'continuous':
            data = self.unit.read_latest(size)
            if data is not None and data.size:
                self.array_output.send(data)
            return

        for _ in range(CaptureNode.MAX_CHUNKS_PER_FRAME):
            data, self._last_read, dropped = self.unit.read_chunk(
                self._last_read, size)
            if dropped:
                self.dropped_output.send(dropped)
            if data is None:
                return
            self.array_output.send(data)

    def send_now(self):
        self._emit()

    def synth_frame_task(self):
        if any_to_string(self.send_option()) == 'on bang':
            # Keep the read cursor current so switching back to streaming does
            # not immediately dump a large backlog.
            self._last_read = self.unit.written
            return
        self._emit()
