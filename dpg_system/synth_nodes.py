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
import time

from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.synth_core import (
    synth_graph, start_filter_warm_up,
    SigUnit, VcoUnit, VcfUnit, VcaUnit, AdsrUnit, LfoUnit, ClockUnit, RampUnit,
    AdditiveUnit, DelayUnit, FoldUnit, CrushUnit,
    ShaperUnit, FormantUnit, VocoderUnit, OneEuroUnit, FORMANT_VOWELS,
    MixUnit, MultUnit, PanUnit, AudioOutUnit, SnapshotUnit, ScalerUnit,
    CaptureUnit, SamplerOscUnit, SamplerBuffer, PhasorUnit, VstUnit,
    plugin_hosting_available, installed_plugin_files, find_plugin_file,
    plugin_names_in_file, open_plugin, plugin_file_refusal,
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
    Node.app.register_node('one_euro~', OneEuroNode.factory)
    Node.app.register_node('smooth~', OneEuroNode.factory)
    Node.app.register_node('vco~', VcoNode.factory)
    Node.app.register_node('additive~', AdditiveNode.factory)
    Node.app.register_node('spectrum~', AdditiveNode.factory)
    Node.app.register_node('delay~', DelayNode.factory)
    Node.app.register_node('echo~', DelayNode.factory)
    Node.app.register_node('fold~', FoldNode.factory)
    Node.app.register_node('distort~', FoldNode.factory)
    Node.app.register_node('crush~', CrushNode.factory)
    Node.app.register_node('decimate~', CrushNode.factory)
    Node.app.register_node('vcf~', VcfNode.factory)
    Node.app.register_node('formant~', FormantNode.factory)
    Node.app.register_node('vocoder~', VocoderNode.factory)
    Node.app.register_node('vowel~', FormantNode.factory)
    Node.app.register_node('vca~', VcaNode.factory)
    Node.app.register_node('adsr~', AdsrNode.factory)
    Node.app.register_node('lfo~', LfoNode.factory)
    Node.app.register_node('phasor~', PhasorNode.factory)
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
    Node.app.register_node('scope~', ScopeNode.factory)
    if plugin_hosting_available():
        Node.app.register_node('vst~', VstNode.factory)
        Node.app.register_node('plugin~', VstNode.factory)
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
        self._modulation_ports = []
        self._header_port = None
        self._header_depth = None
        # Set by add_switch_option. Nodes that are neither a source nor a
        # processor -- audio_out~, which already has its own mute -- leave it
        # alone and are never switched.
        self.switch_input = None
        self._is_processor = False
        self._labels_aligned = False
        self._align_attempts = 0
        self._registered = False

    # -- port construction --------------------------------------------------

    def add_signal_input(self, label, inlet):
        """An inlet that only accepts a patched signal (an audio input)."""
        port = self.add_input(label)
        port.synth_inlet = inlet
        self.signal_inputs.append(port)
        return port

    # Width of the knob and of the attenuverter beside it. The depth is the
    # narrower of the two: it is a trim, and giving it equal weight would say
    # otherwise.
    KNOB_WIDTH = 76
    DEPTH_WIDTH = 52
    # Captions over the two columns, on the first pair only. Repeating them on
    # every row would say the same thing five times.
    COLUMN_LABELS = ('value', 'depth')

    def add_modulation_input(self, label, inlet, widget_type='drag_float',
                             default_value=None, minimum=None, maximum=None,
                             speed=None, attenuverter=True, slider=None,
                             callback=None):
        """Knob + CV inlet + attenuverter for one parameter.

        A parameter with both ends fixed is drawn as a slider rather than a
        drag box: the position shows where in its range the value sits without
        having to read it, and the whole range is one gesture. Bounded drag
        boxes were already clamped, so this changes how a value is set, not
        what it can be.

        Pass slider=False where a range is wide enough that linear travel is
        the wrong reading of it -- a value that spends its useful life in the
        first tenth of its range wants a drag box, not a slider it can never
        land on.

        The three are one control, so they are drawn as one row: the name, the
        value, and -- where there is one -- the depth that scales whatever is
        patched to the inlet. The depth remains an option for saving and for
        messages; it is only drawn in the inlet's row rather than in the
        options block, where it sat a screen away from the thing it modified.
        """
        if default_value is None:
            default_value = inlet.base
        if widget_type == 'drag_float':
            bounded = minimum is not None and maximum is not None
            if slider is None:
                slider = bounded
            if slider and bounded:
                widget_type = 'slider_float'
        port = self.add_input(label, widget_type=widget_type,
                              default_value=default_value,
                              min=minimum, max=maximum,
                              widget_width=SynthNode.KNOB_WIDTH,
                              callback=callback or self.parameters_changed)
        if port.widget is not None:
            if speed is not None:
                port.widget.speed = speed
            # The name moves to the left of the value; '##' keeps dpg from
            # drawing it again on the right, and save/load strip it anyway.
            port.widget.prefix_label = label
            port.widget._label = '##' + label
        port.synth_inlet = inlet
        self.signal_inputs.append(port)
        self._parameter_bindings.append((port, inlet))
        self._modulation_ports.append(port)

        if attenuverter:
            option = self.add_option(label + ' depth', widget_type='drag_float',
                                     default_value=inlet.depth,
                                     width=SynthNode.DEPTH_WIDTH,
                                     callback=self.parameters_changed)
            if option.widget is not None:
                option.widget.speed = 0.01
                option.widget._label = '##' + label + ' depth'
                option.widget.set_tooltip('depth: scales whatever is patched '
                                          'to the ' + label + ' inlet')
                option.inline_with = port.widget
                # The first pair carries the captions for both columns, drawn
                # in its own attribute so the two lines share a left edge.
                if self._header_port is None and port.widget is not None:
                    port.widget.header_labels = SynthNode.COLUMN_LABELS
                    self._header_port = port
                    self._header_depth = option
            self._depth_bindings.append((option, inlet))
        return port

    def align_modulation_labels(self):
        """Square the name column off once text can be measured.

        Names are proportional, so the values only line up if each name is
        padded to the width of the longest. Nothing can be measured until a
        frame has been drawn -- and nodes are built during patch load, before
        that -- so this runs from the frame task until it succeeds once.
        """
        widths = []
        for port in self._modulation_ports:
            if port.widget is None:
                continue
            measured = port.widget.measure_prefix()
            if measured is None:
                return False
            widths.append(measured)
        if not widths:
            return True
        column = max(widths) + 8
        for port in self._modulation_ports:
            if port.widget is not None:
                port.widget.set_prefix_column(column)
        return self.align_column_headers()

    def align_column_headers(self):
        """Put the captions over the widgets they name.

        Their positions are corrected from where the widgets actually landed
        rather than worked out in advance, so the theme's spacing and the
        proportional text take care of themselves. Returns True once settled.
        """
        if self._header_port is None or self._header_port.widget is None:
            return True
        header = self._header_port.widget
        if not header.header_texts:
            return True
        targets = []
        for item in (self._header_port.widget,
                     self._header_depth.widget if self._header_depth else None):
            if item is None or not dpg.does_item_exist(item.uuid):
                targets.append(None)
                continue
            try:
                targets.append(dpg.get_item_rect_min(item.uuid)[0])
            except Exception:
                return False
        # Everything reads as 0 until the node has actually been drawn, and a
        # zero would look like a perfect fit and stop the correction with the
        # captions still where they started.
        if any(not target for target in targets):
            return False
        return header.align_header(targets)

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

    def add_switch(self):
        """The on/off checkbox, on the face of the node rather than hidden.

        A source is switched off to silence and a processor to leave its input
        alone, so the same unit-level flag wants opposite names and opposite
        senses: 'enable', ticked to run, against 'bypass', ticked to stand
        aside. The unit decides which it is by whether it declares a dry path,
        so the node does not have to be told.

        An inlet rather than an option, so it is visible without opening
        anything and can be driven from the patch -- which is how a voice gets
        switched in and out by something other than a mouse.
        """
        self._is_processor = bool(self.unit.bypass_pairs())
        label = 'bypass' if self._is_processor else 'enable'
        self.switch_input = self.add_input(
            label, widget_type='checkbox',
            default_value=False if self._is_processor else True,
            callback=self.parameters_changed)
        if self.switch_input.widget is not None:
            self.switch_input.widget.set_tooltip(
                'passes the input through untouched, and stops processing'
                if self._is_processor else
                'off fades out over a few ms and then stops rendering'
                ' altogether')
        return self.switch_input

    def add_signal_output(self, label, signal):
        port = self.add_output(label)
        port.synth_signal = signal
        port.synth_unit = self.unit
        self.signal_outputs.append(port)
        return port

    def finish_synth_node(self):
        """Call at the end of __init__, once ports and the unit exist."""
        # Elements are drawn in the order they were made, and the switch is
        # made last so that every node can add it with one line -- so move it
        # to the head of the inlets, where the first thing about a node should
        # be whether it is doing anything at all. Only the drawing order
        # moves: the inlet's place in self.inputs is what a saved link refers
        # to, and that stays where it was put, so patches keep their cords.
        elements = getattr(self, 'ordered_elements', None)
        if self.switch_input is not None and elements:
            first_inlet = None
            for index, element in enumerate(elements):
                if element in self.inputs and element is not self.switch_input:
                    first_inlet = index
                    break
            if first_inlet is not None:
                elements.remove(self.switch_input)
                elements.insert(first_inlet, self.switch_input)
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
        if self.switch_input is not None:
            ticked = any_to_bool(self.switch_input())
            # 'bypass' is the same flag read the other way round.
            self.unit.enabled = (not ticked) if self._is_processor else ticked
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
        if not self._labels_aligned:
            # Retried until it settles, since none of it can be measured
            # before something has been drawn -- and given up on eventually,
            # rather than measuring every frame for the life of the patch.
            self._align_attempts += 1
            self._labels_aligned = (self.align_modulation_labels()
                                    or self._align_attempts > 240)
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
        self.add_switch()
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
# phasor~
# ----------------------------------------------------------------------------

class PhasorNode(SynthNode):
    """A 0..1 ramp, built to drive sampler_osc~ position.

    Patch sampler_osc~'s 'length' outlet into 'period' and its 'phase' outlet
    into sampler_osc~'s 'position', with the sampler in scrub mode: one cycle
    then scans the whole file at natural speed, whatever file is loaded and
    without working the rate out by hand.

    From there the position is yours: freeze it and the sound holds where it
    stands, run the period negative and it plays backwards, narrow
    'start'/'end' to sweep a window.

    Which sampler mode you scan matters, because the two do different things:

      scrub      varispeed, like dragging a tape head. Slowing the period
                 drops the pitch with it -- measured 2.17x down for a 2x
                 slower scan, i.e. an octave. Reach for it when you want that
                 sound, not when you want stretch.
      granular   pitch stays put while the scan speed changes -- measured
                 within 6% across the same 2x -- because the grains keep
                 playing at their own rate wherever the playhead points. This
                 is the one for actual time-stretching, and for driving scan
                 speed from effort without the pitch sliding around.

    'wrap' emits a one-sample pulse each time the ramp turns over, for firing
    an adsr~ or anything else once per cycle.

    Arguments: phasor~ <frequency in Hz>.
    """

    @staticmethod
    def factory(name, data, args=None):
        return PhasorNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = PhasorUnit(synth_graph.sample_rate)

        frequency = 1.0
        if args is not None and len(args) > 0:
            value, arg_type = decode_arg(args, 0)
            if arg_type in [float, int]:
                frequency = float(value)
        self.unit.frequency_in.base = frequency

        self.add_modulation_input('frequency', self.unit.frequency_in,
                                  default_value=frequency, speed=0.01)
        self.add_modulation_input('period', self.unit.period_in,
                                  minimum=0.0, speed=0.01, attenuverter=False)
        self.add_modulation_input('phase', self.unit.phase_in, speed=0.01,
                                  attenuverter=False)
        self.add_modulation_input('start', self.unit.start_in, speed=0.01,
                                  attenuverter=False)
        self.add_modulation_input('end', self.unit.end_in, speed=0.01,
                                  attenuverter=False)
        self.add_signal_input('reset', self.unit.reset_in)

        self.phase_output = self.add_signal_output('phase', self.unit.out)
        self.wrap_output = self.add_signal_output('wrap', self.unit.wrap)

        self.start_phase_option = self.add_option(
            'start phase', widget_type='drag_float', default_value=0.0,
            min=0.0, max=1.0, callback=self.parameters_changed)
        self.reset_option = self.add_option('reset now', widget_type='button',
                                            callback=self.reset_phase)
        self.add_switch()
        self.finish_synth_node()

    def sync_options(self):
        self.unit.start_phase = any_to_float(self.start_phase_option())

    def reset_phase(self):
        self.unit.reset()


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

    Set 'time source' to 'measured' and the move takes as long as the gap
    between the values arriving, which the node times for itself. A stream at
    some frame rate -- effort data at 60, a sequencer, anything regular --
    becomes a continuous signal that reaches each value exactly as the next
    one lands, with no steps and nothing to set by hand.

    The estimate is smoothed and ignores intervals far from it, so one late
    frame does not stretch the following move to match; several in a row are
    taken as the rate having genuinely changed. Until two values have arrived
    there is nothing to measure, so the 'time' inlet is used, which is why it
    still matters in this mode.

    'stretch' scales the measured interval. Slightly over 1 is the useful
    setting: arriving a little late is inaudible, while arriving early means
    sitting still at the destination until the next value -- the steps you
    were trying to be rid of. Note that the output necessarily lags the input
    by one frame, since a ramp cannot start before it knows where it is going.

    Arguments: ramp~ <time in seconds> <starting value>, e.g. 'ramp~ 0.25'.
    """

    TIME_SOURCES = ('manual', 'measured')
    # How hard a new interval pulls the estimate: low enough that one ragged
    # frame barely moves it, high enough to follow a real change in a few.
    ESTIMATE_RATE = 0.25
    # Intervals outside this band of the estimate are a dropped frame or a
    # pause rather than the stream's rate.
    OUTLIER_LOW = 0.4
    OUTLIER_HIGH = 2.5
    # ... unless this many arrive in a row, which means the rate did change.
    OUTLIERS_BEFORE_RELOCK = 3

    @staticmethod
    def factory(name, data, args=None):
        return RampNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = RampUnit(synth_graph.sample_rate)
        self._last_arrive_count = 0

        # Timing of the incoming stream, kept on the main thread.
        self._last_arrival = None
        self._interval = None
        self._outliers = 0
        self._shown_rate = ''

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

        self.add_modulation_input('target', self.unit.target_in, speed=0.01,
                                  callback=self.target_arrived)
        self.add_modulation_input('time', self.unit.time_in,
                                  default_value=self.unit.time_in.base,
                                  minimum=0.0, speed=0.001, attenuverter=False)
        self.add_trigger_signal_input('trigger', self.unit.trigger_in,
                                      self.restart)

        self.rate_display = self.add_property('stream', widget_type='label',
                                              default_value='-')

        self.jump_option = self.add_option('jump to target',
                                           widget_type='button', width=110,
                                           callback=self.jump_now)
        self.time_source_option = self.add_option('time source',
                                                  widget_type='combo',
                                                  default_value='manual',
                                                  callback=self.parameters_changed)
        self.time_source_option.widget.combo_items = list(RampNode.TIME_SOURCES)
        self.stretch_option = self.add_option('stretch', widget_type='drag_float',
                                              default_value=1.1, min=0.1, max=4.0,
                                              callback=self.stretch_changed)
        if self.stretch_option.widget is not None:
            self.stretch_option.widget.speed = 0.01

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.done_output = self.add_output('done')
        self.add_switch()
        self.finish_synth_node()

    def restart(self):
        self.unit.restart()

    def jump_now(self):
        self.unit.jump()

    def sync_options(self):
        self.unit.auto_time = (any_to_string(self.time_source_option())
                               == 'measured')

    # -- timing the incoming stream -----------------------------------------

    def target_arrived(self, now=None):
        """A new target: time the gap since the last one and ramp over it.

        Called on the main thread whichever way the value came -- a cord, or
        the widget being dragged. `now` is a parameter so the estimator can be
        driven with known times rather than by waiting in real seconds.
        """
        self.parameters_changed()

        if now is None:
            now = time.perf_counter()
        previous = self._last_arrival
        self._last_arrival = now
        if previous is None:
            return
        interval = now - previous
        if interval <= 0.0:
            return

        if self._interval is None:
            self._interval = interval
            self._outliers = 0
        elif (RampNode.OUTLIER_LOW * self._interval <= interval
                <= RampNode.OUTLIER_HIGH * self._interval):
            self._interval += (interval - self._interval) * RampNode.ESTIMATE_RATE
            self._outliers = 0
        else:
            # One of these is a frame gone astray, and following it would
            # stretch the next move to match. Several in a row is the stream
            # having actually changed rate, which is worth following.
            self._outliers += 1
            if self._outliers < RampNode.OUTLIERS_BEFORE_RELOCK:
                return
            self._interval = interval
            self._outliers = 0

        self.push_measured_time()

    def push_measured_time(self):
        if self._interval is None:
            return
        stretch = max(0.1, any_to_float(self.stretch_option()))
        self.unit.measured_time = min(10.0, max(0.001, self._interval * stretch))

    def stretch_changed(self):
        self.push_measured_time()

    def synth_frame_task(self):
        # Several short ramps can land between GUI frames; report each arrival
        # rather than only the most recent state.
        count = self.unit.arrive_count
        if count != self._last_arrive_count:
            self._last_arrive_count = count
            self.done_output.send('bang')

        # What rate the node thinks it is being fed at -- the one thing you
        # cannot tell by looking at the sound.
        if self._interval and self.unit.auto_time:
            text = '{:.1f} Hz'.format(1.0 / self._interval)
        else:
            text = '-'
        if text != self._shown_rate:
            self._shown_rate = text
            self.rate_display.set(text)


# ----------------------------------------------------------------------------
# one_euro~
# ----------------------------------------------------------------------------

class OneEuroNode(SynthNode):
    """Smoothing that gets out of the way when you move. Also smooth~.

    Any fixed smoothing has to choose between passing jitter and lagging
    behind a gesture, because those are the same setting. The one euro filter
    chooses per sample: at rest the cutoff drops to 'min cutoff' and the
    signal settles hard; as it moves the cutoff opens in proportion to its
    speed, so the lag falls away exactly when it would have been noticed. It
    was made for interactive motion data (Casiello and Roussel, CHI 2012),
    which is what effort data is.

    The two controls do separate jobs, and it is worth setting them in order.
    Hold still and lower 'min cutoff' until the signal stops shivering; then
    move, and raise 'beta' until the movement stops feeling dragged. Setting
    beta first tends to end with both too high.

    Put it after ramp~ to round the corners between one frame's move and the
    next, or on its own between a jittery stream and whatever will make a
    sound of it. Patch 'right in' for stereo, as with the filters.

    Arguments: one_euro~ <min cutoff in Hz> <beta>.
    """

    @staticmethod
    def factory(name, data, args=None):
        return OneEuroNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = OneEuroUnit(synth_graph.sample_rate)

        numbers = []
        if args is not None:
            for arg in args:
                try:
                    numbers.append(float(arg))
                except (ValueError, TypeError):
                    continue
        if len(numbers) > 0:
            self.unit.min_cutoff_in.base = max(0.0001, numbers[0])
        if len(numbers) > 1:
            self.unit.beta_in.base = max(0.0, numbers[1])

        self.add_signal_input('in', self.unit.signal_in)
        self.add_signal_input('right in', self.unit.right_in)
        self.add_modulation_input('min cutoff', self.unit.min_cutoff_in,
                                  default_value=self.unit.min_cutoff_in.base,
                                  minimum=0.0001, speed=0.05,
                                  attenuverter=False)
        self.add_modulation_input('beta', self.unit.beta_in,
                                  default_value=self.unit.beta_in.base,
                                  minimum=0.0, speed=0.02, attenuverter=False)

        self.derivative_option = self.add_option('speed cutoff',
                                                 widget_type='drag_float',
                                                 default_value=1.0, min=0.01,
                                                 callback=self.parameters_changed)
        if self.derivative_option.widget is not None:
            self.derivative_option.widget.speed = 0.05
        self.reset_option = self.add_option('reset', widget_type='button',
                                            callback=self.reset_filter)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.right_output = self.add_signal_output('right', self.unit.right)
        self.add_switch()
        self.finish_synth_node()

    def sync_options(self):
        self.unit.derivative_cutoff = max(0.01,
                                          any_to_float(self.derivative_option()))

    def reset_filter(self):
        self.unit.reset()


# ----------------------------------------------------------------------------
# vco~
# ----------------------------------------------------------------------------

class VcoNode(SynthNode):
    """Band-limited oscillator, with detuned unison.

    Pitch is a base frequency in Hz scaled by the exponential 'pitch' inlet in
    octaves (patch an envelope there for sweeps, an LFO for vibrato), with a
    separate linear FM inlet in Hz for inharmonic tones.

    'voices' stacks up to eight oscillators on the note, detuned symmetrically
    about it by 'detune' cents and spread across the stereo field by 'spread'.
    One voice is the plain oscillator, unchanged; the cost of the rest is
    mostly their band limiting, so a stack here runs about a quarter cheaper
    than the same thing patched from separate oscillators through pans and a
    mixer -- and it is one object to play rather than seven to keep in step.

    Detune is an inlet rather than a setting because opening it as a note
    develops is worth having: an envelope or a slow LFO patched there turns a
    single tone into a swarm and back.

    'drift' gives each voice its own slow random wander in cents, which is
    what keeps a stack from sounding like a fixed chorus. It is off by default,
    since it is the one thing here that makes the output non-repeatable.

    'enable' is not a mute. It fades out over a few milliseconds -- cutting a
    running oscillator between two samples would be a step, and a click -- and
    once it has faded the unit stops rendering altogether and its outlets go
    constant, which takes everything downstream onto its scalar path too. A
    disabled voice therefore costs close to nothing, which is the point when
    there are two dozen of them. Phase stops where it is and carries on from
    there when it comes back.

    The 'right' outlet carries the other half of the stereo spread. At one
    voice, or with spread at 0, it carries the same signal as 'signal', so a
    mono patch can ignore it. For the spread to survive to the speakers the
    chain after it has to be stereo too -- two vcf~, two vca~ -- which costs
    about a tenth of what the unison saves.
    """

    @staticmethod
    def factory(name, data, args=None):
        return VcoNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = VcoUnit(synth_graph.sample_rate)

        frequency = 110.0
        shape = 'saw'
        voices = 1
        detune = self.unit.detune_in.base
        numbers = []
        if args is not None:
            for arg in args:
                if arg in VCO_SHAPES:
                    shape = arg
                else:
                    try:
                        numbers.append(float(arg))
                    except (ValueError, TypeError):
                        continue
        # vco~ <frequency> <voices> <detune in cents>, in the order you reach
        # for them: the note first, then how many of it, then how far apart.
        if len(numbers) > 0:
            frequency = numbers[0]
        if len(numbers) > 1:
            voices = max(1, min(VcoUnit.MAX_VOICES, int(numbers[1])))
        if len(numbers) > 2:
            detune = max(0.0, numbers[2])
        self.unit.shape = shape
        self.unit.frequency_in.base = frequency
        self.unit.detune_in.base = detune
        self.unit.voices = voices

        self.add_modulation_input('frequency', self.unit.frequency_in,
                                  default_value=frequency, minimum=0.0,
                                  speed=1.0)
        self.add_modulation_input('pitch', self.unit.pitch_in, speed=0.01)
        self.add_modulation_input('linear fm', self.unit.linear_fm_in, speed=1.0)
        self.add_modulation_input('width', self.unit.width_in,
                                  minimum=0.01, maximum=0.99, speed=0.01)
        self.add_modulation_input('phase mod', self.unit.phase_mod_in, speed=0.01)
        self.add_signal_input('sync', self.unit.sync_in)
        self.add_modulation_input('detune', self.unit.detune_in,
                                  default_value=detune, minimum=0.0, speed=0.5,
                                  attenuverter=False)

        self.shape_input = self.add_input('shape', widget_type='combo',
                                          default_value=shape,
                                          callback=self.parameters_changed)
        self.shape_input.widget.combo_items = list(VCO_SHAPES)

        self.voices_option = self.add_option('voices', widget_type='slider_int',
                                             default_value=voices, min=1,
                                             max=VcoUnit.MAX_VOICES,
                                             callback=self.parameters_changed)
        self.spread_option = self.add_option('spread', widget_type='slider_float',
                                             default_value=0.0, min=0.0, max=1.0,
                                             callback=self.parameters_changed)
        if self.spread_option.widget is not None:
            self.spread_option.widget.speed = 0.01
        self.drift_option = self.add_option('drift', widget_type='drag_float',
                                            default_value=0.0, min=0.0,
                                            callback=self.parameters_changed)
        if self.drift_option.widget is not None:
            self.drift_option.widget.speed = 0.05
        self.phase_option = self.add_option('start phase', widget_type='slider_float',
                                            default_value=0.0, min=0.0, max=1.0,
                                            callback=self.parameters_changed)
        self.reset_option = self.add_option('reset phase', widget_type='button',
                                            callback=self.reset_phase)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        # Appended, so links saved against the old single outlet keep their
        # index and existing patches load unchanged.
        self.right_output = self.add_signal_output('right', self.unit.right)
        self.add_switch()
        self.finish_synth_node()

    def sync_options(self):
        shape = any_to_string(self.shape_input())
        if shape in VCO_SHAPES:
            self.unit.shape = shape
        self.unit.start_phase = any_to_float(self.phase_option())
        self.unit.voices = any_to_int(self.voices_option())
        self.unit.spread = any_to_float(self.spread_option())
        self.unit.drift = max(0.0, any_to_float(self.drift_option()))

    def reset_phase(self):
        self.unit.reset()


# ----------------------------------------------------------------------------
# additive~
# ----------------------------------------------------------------------------

# A preset is a starting point, not a patch: it sets the drawn curve and the
# four controls that shape it, and leaves pitch and phase alone. The classic
# waveforms are here because they are worth knowing as spectra -- a saw is a
# flat curve at -6 dB an octave, a square is the same thing with the even
# partials taken out, a triangle is that again at twice the slope. Seeing them
# arrive as three settings of the same three controls is the point.
_FLAT = [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]

ADDITIVE_PRESETS = {
    'saw':      {'points': _FLAT, 'tilt': -6.02, 'balance': 0.5,
                 'stretch': 0.0, 'partials': 48},
    'square':   {'points': _FLAT, 'tilt': -6.02, 'balance': 0.0,
                 'stretch': 0.0, 'partials': 48},
    'triangle': {'points': _FLAT, 'tilt': -12.04, 'balance': 0.0,
                 'stretch': 0.0, 'partials': 48},
    'pulse':    {'points': _FLAT, 'tilt': 0.0, 'balance': 0.5,
                 'stretch': 0.0, 'partials': 24},
    'sine':     {'points': _FLAT, 'tilt': 0.0, 'balance': 0.5,
                 'stretch': 0.0, 'partials': 1},
    'organ':    {'points': [[0.0, 1.0, 0.0], [0.12, 0.85, 0.0],
                            [0.25, 0.2, 0.0], [0.5, 0.45, 0.0],
                            [0.75, 0.1, 0.0], [1.0, 0.05, 0.0]],
                 'tilt': -3.0, 'balance': 0.5, 'stretch': 0.0,
                 'partials': 16},
    'vocal':    {'points': [[0.0, 0.25, 0.0], [0.06, 0.95, 0.0],
                            [0.16, 0.3, 0.0], [0.3, 0.7, 0.0],
                            [0.4, 0.15, 0.0], [0.62, 0.35, 0.0],
                            [0.72, 0.08, 0.0], [1.0, 0.04, 0.0]],
                 'tilt': -4.0, 'balance': 0.5, 'stretch': 0.0,
                 'partials': 40},
    'bell':     {'points': _FLAT, 'tilt': -7.0, 'balance': 0.5,
                 'stretch': 0.35, 'partials': 12},
    'gong':     {'points': _FLAT, 'tilt': -4.0, 'balance': 0.5,
                 'stretch': -0.25, 'partials': 24},
}

ADDITIVE_SPANS = ('partials', 'all')
ADDITIVE_EDITORS = ('curve', 'bars')


class AdditiveNode(SynthNode):
    """An oscillator built from a drawn spectrum.

    Draw the amplitude of each partial against its index -- partial 1 is the
    fundamental, 2 the octave above it, 3 the twelfth -- and the node sounds
    their sum. The gestures are shaper~'s: drag a point, right-click to add or
    remove one, shift + left-drag a segment to bend it.

    The drawn curve is the character; the inlets around it are the ways it
    moves, and all of them are audio-rate:

      tilt      a slope in dB per octave over the whole spectrum, i.e. its
                brightness. Drawing a shape and then sweeping the tilt with an
                envelope is the workhorse gesture here -- it is a filter sweep
                that cannot ring, resonate or lose the fundamental, because
                nothing is being filtered.
      partials  how many partials sound, and so how much bandwidth the tone
                occupies. Fractional: the top partial fades in rather than
                arriving.
      odd/even  fades between the odd partials alone (0), all of them (0.5)
                and the even alone (1). The odd-only end is the hollow,
                stopped-pipe sound -- a square, a clarinet.
      stretch   bends the partials off the harmonic series, as the exponent in
                ratio = k ** (1 + stretch). Zero is harmonic. A few thousandths
                is the stiffness of a real string, which is why a piano's top
                octave is tuned sharp. Further out are bells; negative
                compresses the partials instead, towards gongs.

    Zero stretch is not just a value, it is a different engine: the partials
    are all multiples of the fundamental, so the sum repeats every cycle and is
    baked into a wavetable by one inverse FFT. Five hundred partials then cost
    what one does. Off zero there is nothing periodic to bake and it falls back
    to a real oscillator bank, which is why the partial count is capped much
    lower there. The two agree exactly at the boundary, so a stretch envelope
    crosses it without a seam.

    The table is rebuilt for exactly as many partials as fit below Nyquist at
    the moment, so band limiting follows a pitch sweep by the sample and there
    is no aliasing to trade against brightness.

    'spread' is the phase control, and it is continuous. At 0 every partial
    starts together: one narrow spike per cycle, a high crest factor -- so
    normalising it costs a lot of level -- and it reads as a buzz. Opening it
    disperses the partials across the cycle for about half the crest at the
    same spectrum, which sounds like the spike smearing into a swoosh. 'phase'
    only chooses what it disperses towards, so at spread 0 the three agree and
    changing between them is silent.

    Nothing here jumps. A new set of phases is reached by rotating each partial
    to it over 'phase glide' seconds rather than stepping, because every
    partial moving at once is a step in the waveform and therefore a click.
    A phase turning slowly is heard as a momentary detune of a fraction of a
    cent, if at all. Set the glide to 0 to get the jump back.

    'span' decides what the curve's x axis means. On 'partials' it stretches
    to whatever the partial count is, so the shape you drew survives being
    opened and closed. On 'all' it is pinned to the full range, so raising the
    count extends the spectrum into new territory instead.

    'edit' chooses the instrument. On 'curve' the spectrum is drawn as a shape,
    which is how you get a family of partials to fall off, bulge or notch
    together. On 'bars' it is one bar per partial and bar 9 is partial 9
    exactly -- for a timbre that is a specific handful of harmonics, 1, 2 and 9
    and nothing else, where a curve through the ninth would have to cross the
    eighth and tenth to reach it. Left-drag sets a bar and paints across the
    ones you sweep, right-drag clears them, 'clear bars' empties them all to
    build a spectrum up from silence.

    The bars follow the partial count, so what is on screen is what sounds:
    ask for nine partials and there are nine bars. They are always read as
    partial-per-bar, which is what 'span all' means, so 'span' applies to the
    curve only while they are up. Above 64 partials the count is past what a
    bar can usefully be, and the drawing stops there.

    The small button at the right of the tilt, odd/even and stretch rows puts
    that control exactly on its default -- -6, 0.5 and 0. All three are values
    a drag steps over rather than lands on, and stretch's is the one that
    matters most: zero there is not a setting but the line between the
    wavetable and the oscillator bank.

    The two are separate drawings on the same spectrum, not two views of one:
    switching to bars fills them from the curve as it currently sounds, so
    nothing jumps, and switching back leaves the curve as it was. Both are
    saved. Note that 'tilt' still shapes whatever the bars say -- at the
    default -6 dB/octave a row of equal bars is a saw, not a flat spectrum, so
    for bars to mean their own amplitudes set the tilt to 0.

    Patch a list of amplitudes into 'spectrum' to load a curve from elsewhere
    -- an analysis, a preset, another node's 'spectrum out'.

    Arguments: additive~ <frequency> <partials> <tilt> and/or a preset name.
    """

    TABLE_SAMPLES_PER_SEGMENT = 256
    CUSTOM = 'custom'
    # Most bars worth drawing. Past this they are a couple of pixels wide and
    # the mouse cannot pick one out, which is the only thing bars are for.
    MAX_BARS = 64

    @staticmethod
    def factory(name, data, args=None):
        return AdditiveNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = AdditiveUnit(synth_graph.sample_rate)

        # Read before an argument or a preset overwrites any of them, so the
        # snap buttons go to the unit's own defaults rather than to numbers
        # written out a second time here and able to drift from it.
        self._tilt_home = self.unit.tilt_in.base
        self._balance_home = self.unit.balance_in.base
        self._stretch_home = self.unit.stretch_in.base

        preset = AdditiveNode.CUSTOM
        numbers = []
        if args is not None:
            for arg in args:
                if arg in ADDITIVE_PRESETS:
                    preset = arg
                else:
                    try:
                        numbers.append(float(arg))
                    except (ValueError, TypeError):
                        continue
        frequency = self.unit.frequency_in.base
        partials = self.unit.partials_in.base
        tilt = self.unit.tilt_in.base
        if preset != AdditiveNode.CUSTOM:
            recipe = ADDITIVE_PRESETS[preset]
            partials = float(recipe['partials'])
            tilt = recipe['tilt']
        # additive~ <frequency> <partials> <tilt>, in the order you reach for
        # them: the note, how wide it is, how bright it is.
        if len(numbers) > 0:
            frequency = numbers[0]
        if len(numbers) > 1:
            partials = max(1.0, min(float(AdditiveUnit.MAX_PARTIALS),
                                    numbers[1]))
        if len(numbers) > 2:
            tilt = numbers[2]
        self.unit.frequency_in.base = frequency
        self.unit.partials_in.base = partials
        self.unit.tilt_in.base = tilt

        self.plot_width = 220
        self.plot_height = 96
        # Same editor as shaper~, meaning something else: x is the partial
        # index, y the amplitude of that partial.
        from dpg_system.interface_nodes import BreakpointEditor, BarEditor
        self.editor = BreakpointEditor(x_max=1.0, y_min=0.0, y_max=1.0,
                                       width=self.plot_width,
                                       height=self.plot_height,
                                       on_change=self.spectrum_edited,
                                       line_color=(240, 170, 80),
                                       name=label)
        # The same spectrum, editable a partial at a time. A drawn curve
        # cannot single out the ninth harmonic without passing over the
        # eighth; one bar per partial can.
        self.bars = BarEditor(count=self.bar_count_for(partials),
                              capacity=AdditiveUnit.MAX_PARTIALS,
                              y_min=0.0, y_max=1.0,
                              width=self.plot_width,
                              height=self.plot_height,
                              on_change=self.bars_edited,
                              bar_color=(240, 170, 80),
                              name=label)
        self._shown_bars = self.bars.count
        # Which editor is live. Set before any option exists, since a widget
        # callback during creation or load can reach sync_options first.
        self._edit_shown = ADDITIVE_EDITORS[0]
        for name in BreakpointEditor.MESSAGES:
            self.message_handlers[name] = self.spectrum_message

        self.add_modulation_input('frequency', self.unit.frequency_in,
                                  default_value=frequency, minimum=0.0,
                                  speed=1.0)
        self.add_modulation_input('pitch', self.unit.pitch_in, speed=0.01)
        self.add_modulation_input('linear fm', self.unit.linear_fm_in,
                                  speed=1.0)
        self.tilt_input = self.add_modulation_input(
            'tilt', self.unit.tilt_in, default_value=tilt, minimum=-36.0,
            maximum=12.0, speed=0.05)
        # 1..512 with everything interesting in the first tenth: a slider could
        # never be set to 8 partials, so this stays a drag box.
        self.partials_input = self.add_modulation_input(
            'partials', self.unit.partials_in, default_value=partials,
            minimum=1.0, maximum=float(AdditiveUnit.MAX_PARTIALS),
            speed=0.25, slider=False)
        self.balance_input = self.add_modulation_input(
            'odd/even', self.unit.balance_in, minimum=0.0, maximum=1.0,
            speed=0.01)
        self.stretch_input = self.add_modulation_input(
            'stretch', self.unit.stretch_in, minimum=-0.5, maximum=1.0,
            speed=0.002)
        self.spread_input = self.add_modulation_input(
            'spread', self.unit.spread_in, minimum=0.0, maximum=4.0,
            speed=0.01, slider=False)
        self.add_modulation_input('phase mod', self.unit.phase_mod_in,
                                  speed=0.01)
        self.add_signal_input('sync', self.unit.sync_in)
        self.spectrum_input = self.add_input('spectrum',
                                             callback=self.spectrum_received)

        self.preset_input = self.add_input('preset', widget_type='combo',
                                           default_value=preset,
                                           callback=self.preset_changed)
        self.preset_input.widget.combo_items = ([AdditiveNode.CUSTOM]
                                                + list(ADDITIVE_PRESETS))
        self.phase_input = self.add_input('phase', widget_type='combo',
                                          default_value='aligned',
                                          callback=self.parameters_changed)
        self.phase_input.widget.combo_items = list(AdditiveUnit.PHASE_MODES)

        self.spectrum_display = self.add_display('')
        self.spectrum_display.submit_callback = self.submit_display

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.spectrum_output = self.add_output('spectrum out')

        self.normalize_option = self.add_option('normalize',
                                                widget_type='combo',
                                                default_value='rms',
                                                callback=self.parameters_changed)
        self.normalize_option.widget.combo_items = list(
            AdditiveUnit.NORMALIZE_MODES)
        self.edit_option = self.add_option('edit', widget_type='combo',
                                           default_value=ADDITIVE_EDITORS[0],
                                           callback=self.edit_mode_changed)
        self.edit_option.widget.combo_items = list(ADDITIVE_EDITORS)
        self.edit_option.widget.set_tooltip(
            'curve: draw a spectral shape. bars: set partials one at a time, '
            'bar k is partial k exactly')
        self.span_option = self.add_option('span', widget_type='combo',
                                           default_value=ADDITIVE_SPANS[0],
                                           callback=self.parameters_changed)
        self.span_option.widget.combo_items = list(ADDITIVE_SPANS)
        self.span_option.widget.set_tooltip(
            'what the curve x axis means. Ignored while editing bars, where '
            'a bar is always its own partial')
        self.glide_option = self.add_option('phase glide',
                                            widget_type='drag_float',
                                            default_value=0.08, min=0.0,
                                            callback=self.parameters_changed)
        if self.glide_option.widget is not None:
            self.glide_option.widget.speed = 0.005
            self.glide_option.widget.set_tooltip(
                'seconds for a phase to turn half a circle: a speed limit on '
                'phase changes, not a lag on them. 0 jumps, which lurches')
        self.phase_option = self.add_option('start phase',
                                            widget_type='slider_float',
                                            default_value=0.0, min=0.0,
                                            max=1.0,
                                            callback=self.parameters_changed)
        self.reset_option = self.add_option('reset phase',
                                            widget_type='button',
                                            callback=self.reset_phase)
        self.flatten_option = self.add_option('flatten', widget_type='button',
                                              width=110,
                                              callback=self.flatten_spectrum)
        self.clear_option = self.add_option('clear bars', widget_type='button',
                                            width=110,
                                            callback=self.clear_bars)
        # Three values worth returning to exactly, and a drag step lands
        # either side of all of them: the saw slope, all partials equally,
        # harmonic. Each button is drawn in its own knob's row rather than
        # down here with the other buttons: the whole complaint is that the
        # value is awkward to reach, and an answer kept behind the options
        # toggle would not be much of one. The label is the value it goes to.
        self.tilt_home_option = self.snap_button(
            self.tilt_input, self._tilt_home, self.snap_tilt, 'tilt')
        self.balance_home_option = self.snap_button(
            self.balance_input, self._balance_home, self.snap_balance,
            'odd/even')
        self.stretch_home_option = self.snap_button(
            self.stretch_input, self._stretch_home, self.snap_stretch,
            'stretch')
        self.width_option = self.add_option('width', widget_type='drag_int',
                                            default_value=self.plot_width,
                                            callback=self.size_changed)
        self.height_option = self.add_option('height', widget_type='drag_int',
                                             default_value=self.plot_height,
                                             callback=self.size_changed)

        # What the preset combo last actually applied. A preset restored by the
        # loader must not be re-applied over the curve and controls the loader
        # is in the middle of restoring, and a hand edit afterwards means the
        # patch is no longer that preset.
        self._preset_shown = preset
        self._applying_preset = False
        self.editor.set_points(_FLAT, notify=False)
        self.add_switch()
        self.finish_synth_node()

    # -- display -------------------------------------------------------------

    def submit_display(self):
        # Both editors are built, and one is shown. Rebuilding a plot to
        # switch would mean tearing down a node attribute mid-patch; hiding
        # one costs a hidden item and keeps either drawing intact while the
        # other is in use.
        self.editor.submit(self.spectrum_display.uuid,
                           width_option=self.width_option,
                           height_option=self.height_option)
        self.bars.submit(self.spectrum_display.uuid,
                         width_option=self.width_option,
                         height_option=self.height_option)

    def custom_create(self, from_file):
        # Options only hold their real values once every element exists, so
        # anything that reads one waits until here.
        self.size_changed()
        self._edit_shown = any_to_string(self.edit_option())
        self.show_active_editor()
        if not from_file and self._preset_shown != AdditiveNode.CUSTOM:
            self.apply_preset(self._preset_shown)
        self.build_spectrum()

    def size_changed(self):
        width = any_to_int(self.width_option())
        height = any_to_int(self.height_option())
        self.editor.set_size(width, height)
        self.bars.set_size(width, height)

    def editing_bars(self):
        return self._edit_shown == 'bars'

    def show_active_editor(self):
        bars = self.editing_bars()
        self.editor.set_visible(not bars)
        self.bars.set_visible(bars)

    def edit_mode_changed(self):
        mode = any_to_string(self.edit_option())
        if mode not in ADDITIVE_EDITORS or mode == self._edit_shown:
            return
        # A load restores the bars themselves a moment later, and the curve
        # this would read from has not been restored yet, so converting now
        # would seed the bars from a shape that is about to be replaced.
        converting = (mode == 'bars' and not self.in_loading_process)
        self._edit_shown = mode
        if converting:
            # Start the bars from what is currently sounding, sampled at the
            # positions the partials are reading the curve at right now.
            self.bars.set_values(self.curve_at_partials(), notify=False)
        self.show_active_editor()
        self.parameters_changed()
        self.build_spectrum()

    def bar_count_for(self, partials):
        """How many bars to draw for a partial count.

        The bars follow the partial count, so what is on screen is what
        sounds -- ask for nine partials and you are given nine bars to set.
        Capped where a bar would get too narrow to hit; past that the count
        is the wrong instrument for the job anyway.
        """
        try:
            count = int(np.ceil(any_to_float(partials)))
        except (TypeError, ValueError):
            count = 1
        return max(1, min(AdditiveNode.MAX_BARS, count))

    def synth_frame_task(self):
        if self.editing_bars():
            wanted = self.bar_count_for(self.partials_input())
            if wanted != self._shown_bars:
                self._shown_bars = wanted
                self.bars.set_count(wanted)
            self.bars.poll()
        else:
            self.editor.poll()

    # -- settings ------------------------------------------------------------

    def sync_options(self):
        mode = any_to_string(self.phase_input())
        if mode in AdditiveUnit.PHASE_MODES:
            self.unit.phase_mode = AdditiveUnit.PHASE_MODES.index(mode)
        normalize = any_to_string(self.normalize_option())
        if normalize in AdditiveUnit.NORMALIZE_MODES:
            self.unit.normalize = AdditiveUnit.NORMALIZE_MODES.index(normalize)
        span = any_to_string(self.span_option())
        if span in ADDITIVE_SPANS:
            self.unit.spectrum_span = ADDITIVE_SPANS.index(span)
        if self.editing_bars():
            # A bar is a partial, which is only true on the fixed span: it is
            # the one where partial k reads entry k - 1 and nothing is
            # interpolated. The span option keeps its value for the curve.
            self.unit.spectrum_span = AdditiveUnit.FIXED_SPAN
        self.unit.start_phase = any_to_float(self.phase_option())
        self.unit.phase_glide = max(0.0, any_to_float(self.glide_option()))

    def reset_phase(self):
        self.unit.reset()

    def flatten_spectrum(self):
        if self.editing_bars():
            self.bars.set_values(np.ones(AdditiveUnit.MAX_PARTIALS))
            return
        self.editor.set_points(_FLAT)

    def clear_bars(self):
        """Silence every bar, for building a spectrum up from nothing."""
        self.bars.clear()

    def snap_button(self, port, value, callback, name):
        """A button on `value`, drawn at the right of that knob's row.

        The label is the number, which is all it needs to say sitting where it
        does; '##' keeps dpg from drawing the part that only makes the id
        unique. It stays an ordinary option, so it saves and takes messages
        like any other -- only where it is drawn changes.
        """
        option = self.add_option('{:g}##{} home'.format(value, name),
                                 widget_type='button', width=38,
                                 callback=callback)
        if option.widget is not None:
            option.widget.set_tooltip(
                'set ' + name + ' to ' + '{:g}'.format(value)
                + ', its default -- a drag steps over it')
            if port is not None:
                option.inline_with = port.widget
        return option

    def snap_input(self, port, value):
        """Put one knob exactly on a value and let the unit hear about it.

        The same path a drag takes -- set the widget, then push every
        parameter -- so nothing else has to know these buttons exist.
        """
        if port is None or port.widget is None:
            return
        port.widget.set(value)
        self.parameters_changed()

    def snap_tilt(self):
        self.snap_input(self.tilt_input, self._tilt_home)

    def snap_balance(self):
        self.snap_input(self.balance_input, self._balance_home)

    def snap_stretch(self):
        # Worth a button of its own more than the other two: zero is not a
        # value here but the boundary between the wavetable and the bank, and
        # a hair off it is the expensive engine sounding identical.
        self.snap_input(self.stretch_input, self._stretch_home)

    # -- the spectrum --------------------------------------------------------

    def spectrum_message(self, message='', message_data=[]):
        self.editor.handle_message(message, message_data)

    def spectrum_edited(self):
        """The editor moved: rebake the partials and pass the curve on."""
        self.build_spectrum()
        if not self._applying_preset:
            self.mark_custom()
        self.spectrum_output.send(self.editor.get_points())

    def bars_edited(self):
        """A bar moved: hand the partials over as they stand."""
        self.build_spectrum()
        if not self._applying_preset:
            self.mark_custom()
        self.spectrum_output.send(self.bars.get_visible().tolist())

    def build_spectrum(self):
        """Sample the drawn curve onto the unit's uniform partial table.

        breakpoint_line_data owns what a curved segment means, so sampling it
        rather than reimplementing the easing is what keeps this, shaper~ and
        the envelope nodes agreeing about the same curve.

        The bars need no sampling at all: the table is SPECTRUM_POINTS long,
        which is MAX_PARTIALS, so entry k - 1 is partial k and the values
        arrive exactly as they were set. That equality is also why bars run on
        the fixed span -- see sync_options.
        """
        if self.editing_bars():
            self.unit.set_spectrum(self.bars.get_values())
            return
        xs, ys = self.editor.line_data(AdditiveNode.TABLE_SAMPLES_PER_SEGMENT)
        if len(xs) < 2:
            return
        grid = np.linspace(0.0, self.editor.x_max,
                           AdditiveUnit.SPECTRUM_POINTS)
        self.unit.set_spectrum(np.interp(grid, xs, ys))

    def curve_at_partials(self):
        """The drawn curve read where the partials are reading it.

        Switching to bars should sound like nothing happened, so the bars are
        filled from the curve at each partial's own position rather than at
        even spacing -- which are different things whenever the span is
        'partials'. Bars above the count are left silent: they are not
        sounding, and carrying the curve's tail up there would mean raising
        the count later brought in partials nobody set.
        """
        values = np.zeros(AdditiveUnit.MAX_PARTIALS, dtype=np.float64)
        xs, ys = self.editor.line_data(AdditiveNode.TABLE_SAMPLES_PER_SEGMENT)
        if len(xs) < 2:
            return values
        count = self.bar_count_for(self.partials_input())
        if any_to_string(self.span_option()) == 'all':
            divisor = float(AdditiveUnit.MAX_PARTIALS - 1)
        else:
            divisor = max(1.0, float(count) - 1.0)
        positions = np.arange(count, dtype=np.float64) / divisor
        np.clip(positions, 0.0, self.editor.x_max, out=positions)
        values[:count] = np.clip(np.interp(positions, xs, ys), 0.0, 1.0)
        return values

    def spectrum_received(self):
        """A curve from elsewhere -- an analysis, another additive~.

        A list of plain numbers is taken as one amplitude per partial, which is
        what an analysis produces; breakpoints are taken as a drawn curve. Both
        are normalised to the editor's axes, so either arrives looking like
        itself.
        """
        data = self.spectrum_input()
        if self.editing_bars():
            # Bars take a bare list at face value: amplitude per partial, in
            # order, no normalising and no resampling. Sending [1, 0.5, 0, 0,
            # 0, 0, 0, 0, 0.3] is then a way of saying exactly that.
            amplitudes = self.bar_values_from(data)
            if amplitudes is None:
                return
            self._applying_preset = True
            try:
                self.bars.set_values(amplitudes)
            finally:
                self._applying_preset = False
            self.mark_custom()
            return
        points = self.spectrum_points_from(data)
        if not points:
            return
        self._applying_preset = True
        try:
            self.editor.set_points(points)
        finally:
            self._applying_preset = False
        self.mark_custom()

    @staticmethod
    def bar_values_from(data):
        """A list of amplitudes as bars: one per partial, in order.

        Breakpoints arriving while the bars are up are read for their y values
        alone -- the x positions are the curve's business, and this side is
        indexed by partial.
        """
        if isinstance(data, np.ndarray):
            data = data.tolist()
        if not isinstance(data, (list, tuple)) or len(data) == 0:
            return None
        values = []
        for entry in data:
            if isinstance(entry, np.ndarray):
                entry = entry.tolist()
            if isinstance(entry, dict):
                values.append(any_to_float(entry.get('y', 0.0)))
            elif isinstance(entry, (list, tuple)):
                if len(entry) < 2:
                    return None
                values.append(any_to_float(entry[1]))
            else:
                values.append(any_to_float(entry))
        if not values:
            return None
        return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)

    @staticmethod
    def spectrum_points_from(data):
        """[[x, y, curve], ...] as the editor wants it, on 0..1 in both axes."""
        if isinstance(data, np.ndarray):
            data = data.tolist()
        if not isinstance(data, (list, tuple)) or len(data) < 2:
            return []

        first = data[0]
        if isinstance(first, np.ndarray):
            first = first.tolist()
        if not isinstance(first, (list, tuple, dict)):
            # A bare list of amplitudes: one per partial, evenly spaced.
            values = [any_to_float(entry) for entry in data]
            peak = max(values)
            if peak <= 0.0:
                return []
            last = len(values) - 1
            return [[index / last, value / peak, 0.0]
                    for index, value in enumerate(values)]

        points = []
        for entry in data:
            if isinstance(entry, dict):
                entry = [entry.get('x', 0.0), entry.get('y', 0.0),
                         entry.get('curve', 0.0)]
            if isinstance(entry, np.ndarray):
                entry = entry.tolist()
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            points.append([any_to_float(entry[0]), any_to_float(entry[1]),
                           any_to_float(entry[2]) if len(entry) > 2 else 0.0])
        if len(points) < 2:
            return []

        low = min(point[0] for point in points)
        span = max(point[0] for point in points) - low
        if span <= 0.0:
            return []
        peak = max(max(point[1] for point in points), 1.0e-9)
        for point in points:
            point[0] = (point[0] - low) / span
            point[1] = max(0.0, point[1] / peak)
        return points

    # -- presets -------------------------------------------------------------

    def preset_changed(self):
        chosen = any_to_string(self.preset_input())
        if chosen == self._preset_shown:
            return
        self._preset_shown = chosen
        if chosen == AdditiveNode.CUSTOM:
            return
        # During a load the widgets are still being restored; applying a preset
        # now would overwrite the curve and the controls the loader is about to
        # put back. custom_create applies it for a node created by hand.
        if self.in_loading_process:
            return
        self.apply_preset(chosen)

    def apply_preset(self, name):
        recipe = ADDITIVE_PRESETS.get(name)
        if recipe is None:
            return
        self._applying_preset = True
        try:
            self.editor.set_points(recipe['points'])
            for port, value in ((self.tilt_input, recipe['tilt']),
                                (self.balance_input, recipe['balance']),
                                (self.stretch_input, recipe['stretch']),
                                (self.partials_input, recipe['partials'])):
                if port is not None and port.widget is not None:
                    port.widget.set(value)
            self.parameters_changed()
        finally:
            self._applying_preset = False

    def mark_custom(self):
        """A hand edit means the patch is no longer the preset it started as.

        Without this, reloading would re-apply the preset over the edit -- the
        combo would still say 'bell' and would be believed.
        """
        if self._preset_shown == AdditiveNode.CUSTOM:
            return
        self._preset_shown = AdditiveNode.CUSTOM
        if self.preset_input.widget is not None:
            self.preset_input.widget.set(AdditiveNode.CUSTOM)

    # -- persistence ---------------------------------------------------------

    def save_custom(self, container):
        container['additive_points'] = self.editor.get_points()
        # Both drawings are saved whichever one is showing, so a patch that
        # was switched to bars and back still has each of them.
        container['additive_bars'] = self.bars.get_values().tolist()

    def load_custom(self, container):
        if 'additive_points' in container:
            self.editor.set_points(container['additive_points'], notify=False)
        if 'additive_bars' in container:
            self.bars.set_values(container['additive_bars'], notify=False)
        if 'additive_points' in container or 'additive_bars' in container:
            self.build_spectrum()


# ----------------------------------------------------------------------------
# delay~
# ----------------------------------------------------------------------------

class DelayNode(SynthNode):
    """Delay line with damped feedback and an audio-rate delay time.

    Feedback belongs to the object rather than to the patch, and has to: a
    cord from the outlet back to the inlet is a cycle, the compiler runs a
    cycle one block late, so the shortest delay a patched loop can make is
    around 12 ms. Everything below that is only reachable from inside -- and
    that is where the good things live. A few milliseconds with feedback is a
    comb filter; the same with damping is a plucked string; under a
    millisecond, modulated, is a flanger.

    'damping' is a one pole inside the loop, so each repeat comes back darker
    than the one before. At 0 it is transparent. It is the difference between
    a feedback that decays like something in a room and one that builds to a
    shriek, and between a comb filter and a string.

    Feedback past 1 is allowed on purpose -- it is how a delay becomes an
    oscillator. The loop has a soft stop that is exactly linear below 1.5 and
    bends above it, so it settles at a level instead of running away, and
    ordinary levels are not coloured on the way past.

    'time' is audio-rate and the read is interpolated, so it can be modulated
    as hard as you like:

      slide  one read head, moved through the buffer. The pitch moves with it,
             which is what tape does. Effort driving the delay time becomes
             pitch, which is probably why you want this node.
      fade   the head is held still and crossfaded to the new time when the
             time changes, so a delay time can be *set* without gliding to it.
             Standing still the two modes are identical.

    'freeze' stops the input and loops what is already in the buffer.

    Stereo when something is patched to the right inlet: two lines, one set of
    times and gains, so the channels cannot drift apart.

    Arguments: delay~ <time in seconds> <feedback> and/or a mode.
    """

    @staticmethod
    def factory(name, data, args=None):
        return DelayNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = DelayUnit(synth_graph.sample_rate)

        mode = 'slide'
        numbers = []
        if args is not None:
            for arg in args:
                if arg in DelayUnit.MODES:
                    mode = arg
                else:
                    try:
                        numbers.append(float(arg))
                    except (ValueError, TypeError):
                        continue
        seconds = 0.25
        feedback = 0.0
        if len(numbers) > 0:
            seconds = max(0.0, numbers[0])
        if len(numbers) > 1:
            feedback = numbers[1]
        self.unit.time_in.base = seconds
        self.unit.feedback_in.base = feedback
        self.unit.mode = DelayUnit.MODES.index(mode)

        self.add_signal_input('in', self.unit.signal_in)
        self.add_signal_input('right in', self.unit.right_in)
        # A wide range whose useful part is all at the bottom: a slider could
        # never be set to the few milliseconds that make a comb.
        self.time_input = self.add_modulation_input(
            'time', self.unit.time_in, default_value=seconds, minimum=0.0,
            speed=0.002, slider=False)
        self.feedback_input = self.add_modulation_input(
            'feedback', self.unit.feedback_in, default_value=feedback,
            minimum=-1.2, maximum=1.2, speed=0.01)
        self.damping_input = self.add_modulation_input(
            'damping', self.unit.damping_in, minimum=0.0, maximum=0.999,
            speed=0.01)
        self.freeze_input = self.add_signal_input('freeze',
                                                  self.unit.freeze_in)

        self.mode_input = self.add_input('mode', widget_type='combo',
                                         default_value=mode,
                                         callback=self.parameters_changed)
        self.mode_input.widget.combo_items = list(DelayUnit.MODES)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.right_output = self.add_signal_output('right', self.unit.right)

        self.freeze_option = self.add_option('freeze', widget_type='checkbox',
                                             default_value=False,
                                             callback=self.parameters_changed)
        self.max_option = self.add_option('max delay', widget_type='drag_float',
                                          default_value=2.0, min=0.01,
                                          callback=self.max_changed)
        if self.max_option.widget is not None:
            self.max_option.widget.speed = 0.1
            self.max_option.widget.set_tooltip(
                'seconds of buffer; the time inlet is clamped to it')
        self.clear_option = self.add_option('clear', widget_type='button',
                                            callback=self.clear_line)
        self.add_switch()
        self.finish_synth_node()

    def custom_create(self, from_file):
        # An option holds its real value only once every element has been
        # created, so the unit is brought into line here. Without this a node
        # made by hand keeps whatever the unit was built with, and the panel
        # says one thing while the sound does another.
        self.max_changed()
        self.parameters_changed()

    def sync_options(self):
        mode = any_to_string(self.mode_input())
        if mode in DelayUnit.MODES:
            self.unit.mode = DelayUnit.MODES.index(mode)
        # The checkbox and the inlet are summed by the inlet itself, so either
        # can freeze it and the button does not fight a patched gate.
        self.unit.freeze_in.base = 1.0 if any_to_bool(self.freeze_option()) \
            else 0.0

    def max_changed(self):
        self.unit.set_max_delay(any_to_float(self.max_option()))

    def clear_line(self):
        self.unit.reset()


# ----------------------------------------------------------------------------
# fold~
# ----------------------------------------------------------------------------

class FoldNode(SynthNode):
    """Saturation and wavefolding, with the aliasing dealt with.

    Any nonlinearity makes harmonics above the ones it was handed, and the
    ones that land past Nyquist come back down as tones unrelated to the pitch
    that do not move when the pitch moves. That is the fizz around bright
    distorted sound. shaper~ will apply any curve you can draw but can do
    nothing about this, because band-limiting a curve needs its integral.
    These four shapes were chosen partly because they have one.

    'shape' runs along the four and, like formant~'s vowel, between them:

      0 tanh   soft saturation: odd harmonics, gently, and a ceiling.
      1 clip   hard clipping: the same ceiling arrived at abruptly, so far
               more harmonics -- the sound of too much gain.
      2 sine   sine folding: past the limit the signal turns back on itself
               rather than stopping, but smoothly, so the harmonics sweep and
               change places without the very high ones a corner would make.
      3 fold   triangle wavefolding: the same turning back, with corners. The
               brightest of the four by some way, and the one whose timbre
               moves most under a drive envelope -- which is why 'drive' is
               the interesting inlet on this node.

    The run only ever gets harsher, which is what a control swept upwards
    should do. Driven six times over, the four measure a spectral centroid of
    about 140 Hz, 155, 400 and 740, in that order. Each step is one decision:
    how sharp the knee (0 to 1), whether the signal stops at the limit or
    turns back through it (1 to 2), how sharp that turn is (2 to 3). 2.5 is a
    fold with its corners taken off. The position is an audio-rate inlet, so
    the shape can move with whatever drives it.

    'bias' pushes the signal off centre before shaping. A symmetrical curve
    makes only odd harmonics and can sound hollow; asymmetry brings in the
    even ones, which is most of what is meant by warmth. It makes DC too, so
    there is a blocker on the way out.

    'antialias' is the cheap repair -- ask what the curve did on average
    between the last sample and this one rather than what it does at this one.
    It costs almost nothing and is worth 6 to 8 dB.

    'oversample' is the real one, and folding needs it. Driven hard at 3 kHz,
    the aliasing measures about -25 dB for tanh and around the level of the
    signal itself for the folder; antialias alone takes 8 dB off that, 2x
    takes about 30, and 4x about 50. That is the difference between a folder
    that can be swept and one that can only be whispered to. It costs roughly
    3-4x, which at 4x measured 156 us a block against an 11.6 ms budget --
    cheap enough that 2x is the default and 1 is the saving, not the norm.

    Also registered as distort~.

    Arguments: fold~ <drive> and/or a shape name.
    """

    @staticmethod
    def factory(name, data, args=None):
        return FoldNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = FoldUnit(synth_graph.sample_rate)

        shape = 0.0
        drive = 1.0
        numbers = []
        if args is not None:
            for arg in args:
                if arg in FoldUnit.SHAPES:
                    shape = float(FoldUnit.SHAPES.index(arg))
                else:
                    try:
                        numbers.append(float(arg))
                    except (ValueError, TypeError):
                        continue
        # fold~ <drive> <shape position>, or a shape by name in either place.
        if len(numbers) > 0:
            drive = numbers[0]
        if len(numbers) > 1:
            shape = min(float(len(FoldUnit.SHAPES) - 1), max(0.0, numbers[1]))
        self.unit.shape_in.base = shape
        self.unit.drive_in.base = drive

        self.add_signal_input('in', self.unit.signal_in)
        self.add_signal_input('right in', self.unit.right_in)
        self.drive_input = self.add_modulation_input(
            'drive', self.unit.drive_in, default_value=drive, minimum=0.0,
            speed=0.02, slider=False)
        self.bias_input = self.add_modulation_input(
            'bias', self.unit.bias_in, minimum=-2.0, maximum=2.0, speed=0.01)
        self.level_input = self.add_modulation_input(
            'level', self.unit.level_in, minimum=0.0, maximum=2.0, speed=0.01)
        # A slider, because the whole range is useful and the position between
        # the named shapes is the point of it.
        self.shape_input = self.add_modulation_input(
            'shape', self.unit.shape_in, default_value=shape, minimum=0.0,
            maximum=float(len(FoldUnit.SHAPES) - 1), speed=0.01)
        if self.shape_input.widget is not None:
            self.shape_input.widget.set_tooltip(
                '0 tanh, 1 clip, 2 fold, 3 sine -- and every point between')

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.right_output = self.add_signal_output('right', self.unit.right)

        self.antialias_option = self.add_option(
            'antialias', widget_type='checkbox', default_value=True,
            callback=self.parameters_changed)
        self.oversample_option = self.add_option(
            'oversample', widget_type='combo', default_value='2',
            callback=self.parameters_changed)
        self.oversample_option.widget.combo_items = [
            str(factor) for factor in FoldUnit.FACTORS]
        if self.oversample_option.widget is not None:
            self.oversample_option.widget.set_tooltip(
                'run the shaper this many times faster; folding wants 2 or 4')
        self.dc_option = self.add_option('block dc', widget_type='checkbox',
                                         default_value=True,
                                         callback=self.parameters_changed)
        self.add_switch()
        self.finish_synth_node()

    def custom_create(self, from_file):
        # See DelayNode: the oversampling default lives in the option, so the
        # unit only learns about it once the options are real.
        self.parameters_changed()

    def sync_options(self):
        self.unit.antialias = any_to_bool(self.antialias_option())
        self.unit.block_dc = any_to_bool(self.dc_option())
        try:
            factor = int(any_to_string(self.oversample_option()))
        except (ValueError, TypeError):
            factor = 1
        self.unit.set_oversample(factor)


# ----------------------------------------------------------------------------
# crush~
# ----------------------------------------------------------------------------

class CrushNode(SynthNode):
    """Bit depth and sample rate reduction.

    Its own object rather than a shape in fold~, because neither of these is a
    curve: quantising is a staircase whose steps are fixed in amplitude,
    holding is a staircase in time, and they sound nothing alike.

    'bits' quantises the amplitude. The error is roughly noise high up and
    plainly harmonic low down, and unlike tape hiss it is loudest when the
    signal is loudest, which is what makes it sound like a machine rather than
    like dirt.

    'rate' is the more useful of the two, and the one to modulate. Holding
    each sample until the next is due is a sample and hold at audio rate, and
    the images it throws off around that rate are inharmonic and stay put when
    the pitch moves -- so sweeping it against a held note is an instrument in
    itself. At or above the sample rate it does nothing at all.

    Neither is anti-aliased, and neither should be: here the aliasing is the
    effect rather than a defect of it. Patch fold~ before this for dirt with a
    shape to it, or after it to break up what the folder made.

    Also registered as decimate~.

    Arguments: crush~ <bits> <rate in Hz>.
    """

    @staticmethod
    def factory(name, data, args=None):
        return CrushNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = CrushUnit(synth_graph.sample_rate)

        bits = 24.0
        rate = synth_graph.sample_rate
        numbers = []
        if args is not None:
            for arg in args:
                try:
                    numbers.append(float(arg))
                except (ValueError, TypeError):
                    continue
        if len(numbers) > 0:
            bits = min(24.0, max(1.0, numbers[0]))
        if len(numbers) > 1:
            rate = max(1.0, numbers[1])
        self.unit.bits_in.base = bits
        self.unit.rate_in.base = rate

        self.add_signal_input('in', self.unit.signal_in)
        self.add_signal_input('right in', self.unit.right_in)
        self.bits_input = self.add_modulation_input(
            'bits', self.unit.bits_in, default_value=bits, minimum=1.0,
            maximum=24.0, speed=0.05)
        # Everything worth hearing is in the bottom few percent of this range,
        # so it stays a drag box rather than becoming a slider.
        self.rate_input = self.add_modulation_input(
            'rate', self.unit.rate_in, default_value=rate, minimum=1.0,
            maximum=synth_graph.sample_rate, speed=20.0, slider=False)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.right_output = self.add_signal_output('right', self.unit.right)
        self.add_switch()
        self.finish_synth_node()


# ----------------------------------------------------------------------------
# vcf~
# ----------------------------------------------------------------------------

class VcfNode(SynthNode):
    """Resonant multimode filter with per-sample cutoff modulation.

    'tracking' is an exponential cutoff input in octaves, so patching the same
    signal that drives a vco~'s pitch inlet makes the filter track the
    oscillator. 'drive' saturates into the filter for a dirtier tone.

    Patch 'right in' and it filters in stereo -- one cutoff, one resonance,
    one envelope, both channels. That is the point of it being one node: the
    coefficients are worked out once and the channels cannot drift apart the
    way two vcf~ with separately patched cutoffs can. Leave it unpatched and
    nothing changes; 'right' then carries the same signal as 'signal', so a
    mono chain can ignore it.
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
        self.add_signal_input('right in', self.unit.right_in)
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
        # Appended, so links saved against the single outlet keep their index.
        self.right_output = self.add_signal_output('right', self.unit.right)
        self.add_switch()
        self.finish_synth_node()

    def sync_options(self):
        mode = any_to_string(self.mode_input())
        if mode in VcfUnit.MODES:
            self.unit.mode = VcfUnit.MODES.index(mode)


# ----------------------------------------------------------------------------
# formant~
# ----------------------------------------------------------------------------

class FormantNode(SynthNode):
    """A vowel, as five resonances in parallel. Also registered as vowel~.

    'vowel' runs 0..1 across a, e, i, o, u, and runs between them rather than
    switching -- the formants are interpolated as ratios, so a slow sweep is a
    mouth changing shape rather than a crossfade between two mouths. Patch an
    envelope, an lfo~, or effort data there and the sound speaks.

    'shift' multiplies every formant at once: the size of the head making the
    sound. Below 1 is larger, above 1 smaller. 'q' is how sharp the resonances
    are -- low is a vowel-ish colour, high rings, and past about 20 the bank
    sings on its own with whatever the input gives it. Each band is normalised
    for its own Q, so sharpening a vowel does not simply make it louder.

    Feed it something harmonically dense. A saw works, a detuned vco~ stack
    works better, noise gives you a whisper. A sine has nothing at the formant
    frequencies for the bank to find.

    Patch 'right in' for stereo, on one set of coefficients, as with vcf~. The
    five resonances cost about 8 us a block together -- less than the plumbing
    around a single vcf~ -- because they share one kernel.

    Arguments: formant~ <vowel name or 0..1 position> <q>.
    """

    @staticmethod
    def factory(name, data, args=None):
        return FormantNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = FormantUnit(synth_graph.sample_rate)

        vowel = 0.0
        numbers = []
        if args is not None:
            for arg in args:
                if arg in FORMANT_VOWELS:
                    vowel = (FORMANT_VOWELS.index(arg)
                             / (len(FORMANT_VOWELS) - 1))
                else:
                    try:
                        numbers.append(float(arg))
                    except (ValueError, TypeError):
                        continue
        if len(numbers) > 0:
            vowel = min(1.0, max(0.0, numbers[0]))
        if len(numbers) > 1:
            self.unit.q_in.base = max(0.5, numbers[1])
        self.unit.vowel_in.base = vowel

        self.add_signal_input('in', self.unit.signal_in)
        self.add_signal_input('right in', self.unit.right_in)
        self.add_modulation_input('vowel', self.unit.vowel_in,
                                  default_value=vowel, minimum=0.0,
                                  maximum=1.0, speed=0.005)
        self.add_modulation_input('shift', self.unit.shift_in,
                                  default_value=self.unit.shift_in.base,
                                  minimum=0.05, speed=0.005,
                                  attenuverter=False)
        self.add_modulation_input('q', self.unit.q_in,
                                  default_value=self.unit.q_in.base,
                                  minimum=0.5, speed=0.05,
                                  attenuverter=False)

        self.vowel_display = self.add_property('formants', widget_type='label',
                                               default_value='')

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.right_output = self.add_signal_output('right', self.unit.right)
        self.add_switch()
        self.finish_synth_node()
        self._shown = ''

    def synth_frame_task(self):
        # What the bank is actually resonating at, which is the one thing you
        # cannot infer from the vowel knob once shift is in play.
        position = any_to_float(self.unit.vowel_in.base)
        span = len(FORMANT_VOWELS) - 1
        index = int(round(min(1.0, max(0.0, position)) * span))
        text = (FORMANT_VOWELS[index] + '  '
                + ' '.join(str(int(f)) for f in self.unit.frequencies[:3]))
        if text != self._shown:
            self._shown = text
            self.vowel_display.set(text)


# ----------------------------------------------------------------------------
# vocoder~
# ----------------------------------------------------------------------------

class VocoderNode(SynthNode):
    """One signal's spectrum imposed on another.

    The modulator is split into bands, each band's level is followed, and the
    carrier is passed through the same bands with those levels as gains. The
    result has the carrier's pitch and the modulator's shape -- speech through
    an oscillator being the classic case, though nothing here needs the
    modulator to be a voice.

    Give the carrier plenty to filter: a detuned vco~ stack is close to ideal,
    a single sine has nothing in most bands to let through. 'sibilance' mixes
    noise into the carrier for the top third of the range only, which is what
    lets 's' and 't' through without the whole voice turning breathy.

    'attack' and 'release' are the follower's times in seconds. Fast attack and
    slower release is what makes consonants arrive intact while vowels hold;
    long releases smear the modulator into a wash. 'freeze' stops the followers
    where they are, turning the current spectrum into a fixed filter that keeps
    its vowel after the voice stops.

    The 'bands' outlet reports the band levels as a list every frame, and the
    'gains' inlet takes a list back, which with 'band source' set to 'list'
    replaces the analysis entirely. That is the interesting direction here: the
    bank stops being a speech effect and becomes a spectral surface for
    whatever else you have -- effort data, a sequencer, a hand.

    Arguments: vocoder~ <bands> <low Hz> <high Hz>.
    """

    BAND_SOURCES = ('modulator', 'list')

    @staticmethod
    def factory(name, data, args=None):
        return VocoderNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = VocoderUnit(synth_graph.sample_rate)
        self._sent = None

        numbers = []
        if args is not None:
            for arg in args:
                try:
                    numbers.append(float(arg))
                except (ValueError, TypeError):
                    continue
        if len(numbers) > 0:
            self.unit.bands = max(2, min(VocoderUnit.MAX_BANDS,
                                         int(numbers[0])))
        if len(numbers) > 1:
            self.unit.low = max(20.0, numbers[1])
        if len(numbers) > 2:
            self.unit.high = numbers[2]

        self.add_signal_input('modulator', self.unit.modulator_in)
        self.add_signal_input('carrier', self.unit.carrier_in)
        self.add_signal_input('right carrier', self.unit.right_carrier_in)
        self.add_modulation_input('attack', self.unit.attack_in,
                                  minimum=0.0, speed=0.001,
                                  attenuverter=False)
        self.add_modulation_input('release', self.unit.release_in,
                                  minimum=0.0, speed=0.005,
                                  attenuverter=False)
        self.add_modulation_input('sibilance', self.unit.sibilance_in,
                                  minimum=0.0, maximum=1.0, speed=0.01,
                                  attenuverter=False)
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, speed=0.05, attenuverter=False)
        self.gains_input = self.add_input('gains', callback=self.gains_received)

        self.freeze_input = self.add_input('freeze', widget_type='checkbox',
                                           default_value=False,
                                           callback=self.parameters_changed)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.right_output = self.add_signal_output('right', self.unit.right)
        self.bands_output = self.add_output('bands')

        self.bands_option = self.add_option('bands', widget_type='slider_int',
                                            default_value=self.unit.bands,
                                            min=2, max=VocoderUnit.MAX_BANDS,
                                            callback=self.parameters_changed)
        self.low_option = self.add_option('low', widget_type='drag_float',
                                          default_value=self.unit.low, min=20.0,
                                          callback=self.parameters_changed)
        self.high_option = self.add_option('high', widget_type='drag_float',
                                           default_value=self.unit.high,
                                           callback=self.parameters_changed)
        self.q_option = self.add_option('q', widget_type='drag_float',
                                        default_value=self.unit.q, min=0.5,
                                        callback=self.parameters_changed)
        self.source_option = self.add_option('band source', widget_type='combo',
                                             default_value='modulator',
                                             callback=self.parameters_changed)
        self.source_option.widget.combo_items = list(VocoderNode.BAND_SOURCES)
        self.report_option = self.add_option('report bands',
                                             widget_type='checkbox',
                                             default_value=True)
        self.add_switch()
        self.finish_synth_node()

    def sync_options(self):
        self.unit.bands = any_to_int(self.bands_option())
        self.unit.low = max(20.0, any_to_float(self.low_option()))
        self.unit.high = any_to_float(self.high_option())
        self.unit.q = max(0.5, any_to_float(self.q_option()))
        self.unit.freeze = any_to_bool(self.freeze_input())
        self.unit.external = (any_to_string(self.source_option()) == 'list')

    def gains_received(self):
        """A list of band gains from the patch, in place of the analysis."""
        data = self.gains_input()
        if isinstance(data, np.ndarray):
            values = data.reshape(-1)
        elif isinstance(data, (list, tuple)):
            values = np.asarray([any_to_float(v) for v in data])
        else:
            values = np.asarray([any_to_float(data)])
        count = min(len(values), VocoderUnit.MAX_BANDS)
        if count == 0:
            return
        # Assigned into the live array one band at a time: the audio thread
        # may read it mid-write, and a partly-updated set of gains is a
        # momentary timbre rather than a fault.
        for band in range(count):
            self.unit.supplied[band] = float(values[band])

    def synth_frame_task(self):
        if not any_to_bool(self.report_option()):
            return
        count = self.unit.band_count()
        levels = [round(float(v), 5) for v in self.unit.envelopes[:count]]
        if levels != self._sent:
            self._sent = levels
            self.bands_output.send(levels)


# ----------------------------------------------------------------------------
# vca~
# ----------------------------------------------------------------------------

class VcaNode(SynthNode):
    """Voltage controlled amplifier.

    Gain is the sum of the knob and any patched CV, so the usual patch is knob
    at 0 with an adsr~ into the gain inlet.

    Patch 'right in' and it amplifies in stereo, one gain curve driving both
    channels. Unpatched it is the mono vca~ it always was, and 'right' carries
    the same signal as 'signal'.
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
        self.add_signal_input('right in', self.unit.right_in)
        self.add_modulation_input('gain', self.unit.gain_in,
                                  default_value=gain, minimum=0.0, speed=0.01)

        self.response_input = self.add_input('response', widget_type='combo',
                                             default_value=response,
                                             callback=self.parameters_changed)
        self.response_input.widget.combo_items = list(VcaNode.RESPONSES)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        # Appended, so links saved against the single outlet keep their index.
        self.right_output = self.add_signal_output('right', self.unit.right)
        self.add_switch()
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
        self.add_switch()
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

    Not being band limited is also what makes its 'ramp' shape the thing to
    index a table or drive shaper~ with, where vco~'s saw is not: band
    limiting spreads the wrap over a couple of samples at intermediate values,
    and a lookup maps those through the middle of the curve, so every cycle
    ends in a spike however well the curve's endpoints match.
    """

    @staticmethod
    def factory(name, data, args=None):
        return LfoNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = LfoUnit(synth_graph.sample_rate)

        rate = 1.0
        shape = 'sine'
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

        self.bipolar_option = self.add_option('bipolar', widget_type='checkbox',
                                              default_value=True,
                                              callback=self.parameters_changed)
        self.phase_option = self.add_option('start phase', widget_type='slider_float',
                                            default_value=0.0, min=0.0, max=1.0,
                                            callback=self.parameters_changed)
        self.reset_option = self.add_option('reset now', widget_type='button',
                                            callback=self.reset_phase)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.add_switch()
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
    # that stall is simply not heard. The 'count' outlet still advances by the
    # whole backlog: what is skipped is the sound of those beats, not the
    # clock's place in the bar.
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
        self.add_switch()
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

        # A stall -- a patch load, a window resize, a heavy node -- leaves a
        # backlog behind it, and only the most recent of it is banged; the
        # rest no longer means anything musically.
        #
        # The count is not a tally of those bangs, though. It is which beat
        # this is, so it advances by the whole backlog whether or not each
        # tick was heard. Counting only what was delivered would leave a
        # sequencer permanently behind the clock after a single resize, and
        # silently: the bangs would keep coming, just from the wrong bar.
        delivered = min(pending, ClockNode.MAX_TICKS_PER_FRAME)
        self._tick_index += pending - delivered
        for _ in range(delivered):
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
        self.add_switch()
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
        # 1.0 is the neutral value and sits at a twentieth of this range, so
        # a linear slider could never be set to it. Stays a drag box.
        self.add_modulation_input('curve', self.unit.curve_in,
                                  minimum=0.01, maximum=16.0, speed=0.01,
                                  attenuverter=False, slider=False)

        self.mode_input = self.add_input('mode', widget_type='combo',
                                         default_value=mode,
                                         callback=self.parameters_changed)
        self.mode_input.widget.combo_items = list(ScalerNode.MODES)

        self.clip_option = self.add_option('clip', widget_type='checkbox',
                                           default_value=True,
                                           callback=self.parameters_changed)

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.add_switch()
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
        self.add_switch()
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
        self.add_switch()
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
        self.add_switch()
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
        self.precision_option = self.add_option('precision', widget_type='slider_int',
                                                default_value=3, min=0, max=8)
        self.add_switch()
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

    'enable' is not a mute. It fades out over a few milliseconds -- cutting
    playback between two samples would be a step, and a click -- and once it
    has faded the unit stops rendering altogether and its outlets go constant,
    which takes everything downstream onto its scalar path too. A disabled
    voice therefore costs close to nothing, which is the point when there are
    two dozen of them. The playhead stops where it is and carries on from
    there, so switching back in does not restart the material.

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
        self.add_switch()
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
    Also registered as array~. To look at the signal rather than compute on
    it, use scope~, which draws the same ring buffer with a trigger.
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
        self.add_switch()
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


# ----------------------------------------------------------------------------
# scope~
# ----------------------------------------------------------------------------

class ScopeNode(SynthNode):
    """The signal, drawn. An oscilloscope with a trigger.

    plot takes a stream of control values, one per frame, so an audio signal
    reaching it through snapshot~ is aliased beyond recognition: 60 samples a
    second of something oscillating at 440 Hz is noise. This keeps every
    sample in the same ring buffer capture~ uses and draws a window of it, so
    what you see is the waveform rather than a beat pattern between the audio
    rate and the frame rate.

    Untriggered ('free'), successive frames start at unrelated phases and a
    steady tone scrolls and tears. The sync modes fix that: the window starts
    at a crossing of the trigger 'level' -- zero by default, so a zero
    crossing -- going up ('rising') or down ('falling'), and a periodic signal
    then stands still. What moves on the screen is what is actually changing
    in the sound.

    'noise reject' is the trigger's hysteresis. A crossing only counts once
    the signal has been clear of the level by that much on the other side, so
    a waveform that hovers around the level, or one riding on noise, triggers
    once per cycle instead of on every wiggle. Raise it until the trace sits
    still; too high and slow or quiet material stops triggering at all.

    Nothing to trigger on -- silence, or a period longer than the window --
    holds the last trace for a moment and then free-runs, so the display goes
    back to showing the truth rather than freezing on a stale waveform.

    The readout gives the window's duration and, when synced, the frequency
    implied by the spacing of the triggers. 'array' sends the displayed window
    out, phase-aligned, for a patch that wants to measure what it is looking
    at; capture~ is the better source where the trigger is beside the point.

    Arguments: scope~ <samples> and/or 'free' | 'rising' | 'falling'.
    """

    SYNC_MODES = ('free', 'rising', 'falling')

    # How long a triggered trace survives without a new trigger before the
    # display gives up and free-runs. Half a second: long enough to ride out
    # a gap between notes, short enough that silence does not look like sound.
    HOLD_FRAMES = 30

    @staticmethod
    def factory(name, data, args=None):
        return ScopeNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = CaptureUnit(synth_graph.sample_rate)

        samples = 512
        sync = 'rising'
        if args is not None:
            for arg in args:
                if arg in ScopeNode.SYNC_MODES:
                    sync = arg
                else:
                    try:
                        samples = int(float(arg))
                    except (ValueError, TypeError):
                        continue
        self.samples = self._clamp_samples(samples)

        self.plot_width = 300
        self.plot_height = 128

        self.plot_tag = dpg.generate_uuid()
        self.x_axis_tag = dpg.generate_uuid()
        self.y_axis_tag = dpg.generate_uuid()
        self.trace_tag = dpg.generate_uuid()
        self.level_tag = dpg.generate_uuid()
        self.plot_ready = False

        self.x_data = np.arange(self.samples, dtype=np.float32)
        self._held_frames = 0
        self._shown_readout = ''

        self.add_signal_input('in', self.unit.signal_in)

        self.sync_input = self.add_input('sync', widget_type='combo',
                                         default_value=sync,
                                         callback=self.sync_changed)
        self.sync_input.widget.combo_items = list(ScopeNode.SYNC_MODES)
        self.level_input = self.add_input('level', widget_type='drag_float',
                                          default_value=0.0,
                                          callback=self.level_changed)
        if self.level_input.widget is not None:
            self.level_input.widget.speed = 0.01
            self.level_input.widget.set_tooltip(
                'the signal value the trace starts on -- 0 is a zero crossing')

        self.scope_display = self.add_display('')
        self.scope_display.submit_callback = self.submit_display
        self.readout_display = self.add_property('window', widget_type='label',
                                                 default_value='-')

        self.array_output = self.add_array_output('array')

        self.samples_option = self.add_option('samples', widget_type='drag_int',
                                              default_value=self.samples,
                                              min=16,
                                              max=self.unit.max_window // 2,
                                              callback=self.window_changed)
        if self.samples_option.widget is not None:
            self.samples_option.widget.set_tooltip(
                'width of the window in samples: how much time is on screen')
        self.hysteresis_option = self.add_option('noise reject',
                                                 widget_type='drag_float',
                                                 default_value=0.0, min=0.0,
                                                 callback=self.parameters_changed)
        if self.hysteresis_option.widget is not None:
            self.hysteresis_option.widget.speed = 0.005
            self.hysteresis_option.widget.set_tooltip(
                'trigger hysteresis: how far past the level the signal must '
                'go before the next crossing counts')
        self.min_y_option = self.add_option('min y', widget_type='drag_float',
                                            default_value=-1.0,
                                            callback=self.range_changed)
        self.max_y_option = self.add_option('max y', widget_type='drag_float',
                                            default_value=1.0,
                                            callback=self.range_changed)
        for option in (self.min_y_option, self.max_y_option):
            if option.widget is not None:
                option.widget.speed = 0.01
        self.width_option = self.add_option('width', widget_type='drag_int',
                                            default_value=self.plot_width,
                                            max=3840, callback=self.size_changed)
        self.height_option = self.add_option('height', widget_type='drag_int',
                                             default_value=self.plot_height,
                                             max=3840, callback=self.size_changed)
        self.add_switch()
        self.finish_synth_node()

    def _clamp_samples(self, samples):
        # Half the readable window, not all of it: the trigger needs a window's
        # worth of search region ahead of the one on screen, and a scope that
        # quietly stopped triggering at its widest setting would look broken.
        return max(16, min(self.unit.max_window // 2, int(samples)))

    # -- display ------------------------------------------------------------

    def submit_display(self):
        # Options do not hold their values yet -- they are created after the
        # displays -- so this draws at the instance defaults and custom_create
        # applies whatever was actually saved.
        with dpg.theme() as self.trace_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (120, 220, 150),
                                    category=dpg.mvThemeCat_Plots)
        with dpg.theme() as self.level_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (120, 120, 120, 110),
                                    category=dpg.mvThemeCat_Plots)

        with dpg.plot(label='', tag=self.plot_tag,
                      height=self.plot_height, width=self.plot_width,
                      no_title=True, no_menus=True, no_box_select=True,
                      no_mouse_pos=True):
            dpg.add_plot_axis(dpg.mvXAxis, label='', tag=self.x_axis_tag,
                              no_tick_labels=True)
            dpg.add_plot_axis(dpg.mvYAxis, label='', tag=self.y_axis_tag,
                              no_tick_labels=True)
            # The level line is drawn under the trace so the waveform stays
            # readable where the two coincide, which is exactly at the trigger.
            dpg.add_line_series([], [], parent=self.y_axis_tag,
                                tag=self.level_tag)
            dpg.add_line_series([], [], parent=self.y_axis_tag,
                                tag=self.trace_tag)
            dpg.bind_item_theme(self.level_tag, self.level_theme)
            dpg.bind_item_theme(self.trace_tag, self.trace_theme)
        self.plot_ready = True
        self.install_resize_handle()

    def install_resize_handle(self):
        from dpg_system.node import ResizeHandle, _get_resize_handle_theme
        btn_uuid = dpg.add_button(parent=self.scope_display.uuid, label='',
                                  width=self.plot_width, height=4)
        handle = ResizeHandle(
            btn_uuid, self.plot_tag, axis='xy',
            width_option=self.width_option, height_option=self.height_option,
            sync_width=True, sync_height=False,
            on_resize=self.handle_resized
        )
        dpg.set_item_user_data(btn_uuid, handle)
        dpg.bind_item_handler_registry(btn_uuid, "resize handle handler")
        dpg.bind_item_theme(btn_uuid, _get_resize_handle_theme())
        self.resize_handle = handle

    def handle_resized(self, new_w, new_h):
        self.plot_width = int(new_w)
        self.plot_height = int(new_h)

    def custom_create(self, from_file):
        self.window_changed()
        self.range_changed()
        self.size_changed()

    def size_changed(self):
        if not self.plot_ready:
            return
        self.plot_width = any_to_int(self.width_option())
        self.plot_height = any_to_int(self.height_option())
        dpg.set_item_width(self.plot_tag, self.plot_width)
        dpg.set_item_height(self.plot_tag, self.plot_height)
        handle = getattr(self, 'resize_handle', None)
        if handle is not None and dpg.does_item_exist(handle.uuid):
            dpg.set_item_width(handle.uuid, self.plot_width)

    def window_changed(self):
        samples = self._clamp_samples(any_to_int(self.samples_option()))
        if samples != any_to_int(self.samples_option()):
            self.samples_option.set(samples)
        if samples != self.samples or self.x_data.size != samples:
            self.samples = samples
            self.x_data = np.arange(samples, dtype=np.float32)
        if self.plot_ready:
            dpg.set_axis_limits(self.x_axis_tag, 0, max(1, self.samples - 1))
        self._shown_readout = ''

    def range_changed(self):
        low = any_to_float(self.min_y_option())
        high = any_to_float(self.max_y_option())
        if high <= low:
            high = low + 1.0
            self.max_y_option.set(high)
        if self.plot_ready:
            dpg.set_axis_limits(self.y_axis_tag, low, high)

    def sync_changed(self):
        # A mode change invalidates whatever is frozen on screen.
        self._held_frames = ScopeNode.HOLD_FRAMES
        self.parameters_changed()

    def level_changed(self):
        self._held_frames = ScopeNode.HOLD_FRAMES
        self.parameters_changed()

    # -- trigger ------------------------------------------------------------

    @staticmethod
    def _trigger_indices(data, level, hysteresis, rising):
        """Where the signal crosses `level` in the wanted direction, armed.

        A bare comparison retriggers on every sample of a signal that sits on
        the level, and on every noise excursion near it. This is the usual
        armed edge instead: a crossing counts only if the signal has been on
        the far side of level -/+ hysteresis since the previous crossing.
        Done by comparing, at each sample, how recently each of those two
        conditions last held -- which vectorises, where the scan does not.
        """
        if rising:
            above = data >= level
            below = data <= level - hysteresis
        else:
            above = data <= level
            below = data >= level + hysteresis
        index = np.arange(data.size)
        last_below = np.maximum.accumulate(np.where(below, index, -1))
        last_above = np.maximum.accumulate(np.where(above, index, -1))
        armed = np.empty(data.size, dtype=bool)
        armed[0] = False
        armed[1:] = last_below[:-1] > last_above[:-1]
        return np.flatnonzero(above & armed)

    @staticmethod
    def _aligned_window(data, start, count, level):
        """The window from the crossing, placed between samples.

        The crossing almost never falls on a sample: it lies somewhere in the
        step from data[start - 1] to data[start]. Starting at the sample
        rounds the trigger to the sample grid, which at audio frequencies is
        a sizeable fraction of a cycle -- a 1 kHz tone is 44 samples long, so
        the trace visibly shivers by a fortieth of a period from frame to
        frame. Reading the window at the crossing's real position instead
        holds it still, and since the samples wanted are a unit apart that is
        one linear blend of two slices rather than a resampling.
        """
        base = start - 1
        if base < 0 or base + count + 1 > data.size:
            return data[start:start + count]
        span = float(data[start]) - float(data[base])
        if span == 0.0:
            fraction = 0.0
        else:
            fraction = min(1.0, max(0.0, (level - float(data[base])) / span))
        earlier = data[base:base + count]
        later = data[base + 1:base + count + 1]
        return earlier * (1.0 - fraction) + later * fraction

    def _readout(self, period_samples):
        duration = 1000.0 * self.samples / max(1.0, self.unit.sample_rate)
        text = '{:.1f} ms'.format(duration)
        if period_samples:
            frequency = self.unit.sample_rate / period_samples
            if frequency >= 1000.0:
                text += '   {:.2f} kHz'.format(frequency / 1000.0)
            else:
                text += '   {:.1f} Hz'.format(frequency)
        if text != self._shown_readout:
            self._shown_readout = text
            self.readout_display.set(text)

    def synth_frame_task(self):
        if not self.plot_ready:
            return
        if any_to_int(self.samples_option()) != self.samples:
            self.window_changed()

        count = self.samples
        mode = any_to_string(self.sync_input())
        # Search a whole window's worth ahead of the displayed one, so any
        # period that fits on screen has a trigger somewhere to be found.
        data = self.unit.read_latest(min(count * 2, self.unit.max_window))
        if data is None or data.size == 0:
            return

        level = any_to_float(self.level_input())
        start = None
        period = 0
        if mode in ('rising', 'falling') and data.size > count:
            fires = self._trigger_indices(
                data, level, max(0.0, any_to_float(self.hysteresis_option())),
                mode == 'rising')
            if fires.size > 1:
                # Median rather than the last gap: one missed or extra trigger
                # would otherwise halve or double the reported frequency.
                period = float(np.median(np.diff(fires)))
            usable = fires[fires <= data.size - count]
            if usable.size:
                # The latest usable crossing, so the trace is as current as the
                # trigger allows rather than lagging by a whole search window.
                start = int(usable[-1])

        if start is None:
            if mode != 'free' and self._held_frames < ScopeNode.HOLD_FRAMES:
                # Nothing to lock to yet. Leave the last good trace up rather
                # than replacing it with an untriggered one that will tear.
                self._held_frames += 1
                return
            window = data[max(0, data.size - count):]
        else:
            self._held_frames = 0
            window = self._aligned_window(data, start, count, level)

        if window.size < count:
            return

        dpg.set_value(self.trace_tag, [self.x_data, window])
        level = any_to_float(self.level_input())
        if mode == 'free':
            dpg.set_value(self.level_tag, [[], []])
        else:
            dpg.set_value(self.level_tag,
                          [[0.0, float(count - 1)], [level, level]])
        self._readout(period)
        self.array_output.send(window)


# ----------------------------------------------------------------------------
# vst~
# ----------------------------------------------------------------------------

class VstNode(SynthNode):
    """A VST3 or AudioUnit effect, patched like any other unit.

    Arguments: vst~ <part of a plugin filename> -- 'valhalla', 'waveshell'.
    A file holding several plugins (a Waves shell, say) offers them in the
    'plugin' option; pick one and it reloads.

    'params=N' and 'choices=N' set how many slots the node has, since a plugin
    with fifteen useful controls does not fit through four. Slots are cheap
    enough to ask for freely: an empty one costs nothing at all, and a bound
    one costs about 0.33 us a block plus the plugin's own charge for a write,
    and only when the value moves. Sixteen of them running flat out is under
    half a percent of the block. What they cost is height on screen, so take
    what you need and no more. Defaults are 8 and 3, up to 24 and 12.
    Anything that is not a keyword is part of the plugin name, so
    'vst~ WaveShell1-VST3 17.1 params=12' reads correctly.

    A plugin's parameters come in two kinds and are offered as two kinds of
    control, because they are not the same thing.

    The ones with a numeric range are knobs. Four 'param' inlets drive them,
    chosen by name in the matching 'param n source' option. They run 0..1,
    which is what plugin automation is underneath whatever the plugin calls
    its own range, so a knob, an lfo~ or a joint's effort all reach them the
    same way.

    The rest are menus -- a reverb mode, a delay sync division. Three 'choice'
    slots hold them: pick the parameter in 'choice n', pick its setting in
    'choice n value'. They are deliberately not modulatable. The numbers
    behind a menu are not evenly spaced (Supermassive's 23 modes quantise
    unevenly and list 'Gemini' at both ends of the range), so stepping one by
    arithmetic lands on the wrong entry for half the list; the setting is
    resolved by name instead, which is exact.

    Press 'print parameters' to see what a loaded plugin offers, marked
    [knob] or [menu]. 'open editor' brings up the plugin's own window; it is
    modal, so the patch UI waits while the audio carries on. When it closes,
    every parameter the panel shows is read back from the plugin, so the
    knobs and menus follow what was done in the window -- and are saved with
    the patch. Edits to parameters the panel does not show stay in the
    plugin only.

    Three things worth knowing before patching one in:

    Parameters move once per block, about 86 times a second. That is what
    plugin automation is, not a shortcoming of the wrapper. Patch an audio-rate
    signal to a param inlet and it is read at the last sample of each block and
    nothing between -- fine for effort data or an LFO, wrong for anything you
    want to hear as modulation. The native units do that.

    Latency is not compensated. Plugins declare a delay, sometimes a large one,
    and the 'status' line reports it. In series that is just delay. The 'mix'
    control is a parallel dry path, so on a latent plugin it combs rather than
    blends -- leave mix at 1.0 there and blend outside with mix~.

    A plugin that throws, returns the wrong number of frames, or keeps missing
    the deadline is dropped, and the node passes its input through instead.
    The reason lands in 'status' and on the console. This is the same thing
    'bypass' does by hand, and it is the only failure a performance survives.
    """

    # Enough for the menus a plugin actually has -- Supermassive has three,
    # and most have none.
    CHOICE_SLOTS = 3
    MAX_PARAMETER_SLOTS = 24
    MAX_CHOICE_SLOTS = 12

    @staticmethod
    def factory(name, data, args=None):
        return VstNode(name, data, args)

    @staticmethod
    def read_arguments(args):
        """Split 'supermassive params=16' into a name and the slot counts.

        A keyword rather than a trailing number, because plugin names end in
        numbers all the time -- 'Reverb 2', 'WaveShell1-VST3 17.1' -- and a
        positional count would silently eat one and load the wrong plugin.
        Everything that is not a keyword joins back into the name, so a name
        with spaces in it needs no quoting.
        """
        counts = {'params': VstUnit.PARAMETER_SLOTS,
                  'choices': VstNode.CHOICE_SLOTS}
        limits = {'params': VstNode.MAX_PARAMETER_SLOTS,
                  'choices': VstNode.MAX_CHOICE_SLOTS}
        words = []
        for arg in (args or ()):
            text = any_to_string(arg)
            key, sep, number = text.partition('=')
            key = key.strip().lower()
            if sep and key in counts:
                try:
                    counts[key] = max(0, min(limits[key], int(number)))
                except (ValueError, TypeError):
                    print('vst~: ' + key + ' needs a whole number, not '
                          + repr(number))
                continue
            words.append(text)
        return ' '.join(words), counts['params'], counts['choices']

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        requested, parameter_slots, self.choice_slots = \
            VstNode.read_arguments(args)
        self.unit = VstUnit(synth_graph.sample_rate, parameter_slots)
        self.plugin = None
        self.parameter_names = []
        self.numeric_names = []
        self.choice_names = []
        self._parameters = {}
        self._applied_choices = {}
        self._reload_pending = True
        self._editor_pending = False
        self._reported_error = ''
        self._cost_countdown = 0

        self.add_signal_input('in', self.unit.signal_in)
        self.add_signal_input('right in', self.unit.right_in)
        self.add_modulation_input('mix', self.unit.mix_in, default_value=1.0,
                                  minimum=0.0, maximum=1.0, speed=0.01,
                                  attenuverter=False)
        self.slot_inputs = []
        for index, inlet in enumerate(self.unit.parameter_in):
            self.slot_inputs.append(self.add_modulation_input(
                'param ' + str(index + 1), inlet, default_value=0.0,
                minimum=0.0, maximum=1.0, speed=0.005))

        self.file_option = self.add_option('file', widget_type='text_input',
                                           width=240, default_value=requested,
                                           callback=self.request_reload)
        self.plugin_option = self.add_option('plugin', widget_type='combo',
                                             default_value='',
                                             callback=self.request_reload)
        if self.plugin_option.widget is not None:
            self.plugin_option.widget.combo_items = ['']

        self.slot_options = []
        for index in range(len(self.unit.parameter_in)):
            option = self.add_option('param ' + str(index + 1) + ' source',
                                     widget_type='combo', default_value='',
                                     callback=self.bind_parameters)
            if option.widget is not None:
                option.widget.combo_items = ['']
            self.slot_options.append(option)

        # Menus, not knobs. A reverb mode or a sync division is a choice from a
        # list, and the numbers behind those lists are not evenly spaced, so
        # they belong nowhere near a 0..1 modulation inlet -- picking
        # Supermassive's 23rd mode by dragging a slider is not a control.
        # Each pair names a parameter and then holds its setting.
        self.choice_options = []
        for index in range(self.choice_slots):
            chooser = self.add_option('choice ' + str(index + 1),
                                      widget_type='combo', default_value='',
                                      callback=self.choice_source_changed)
            value = self.add_option('choice ' + str(index + 1) + ' value',
                                    widget_type='combo', default_value='',
                                    callback=self.apply_choices)
            for option in (chooser, value):
                if option.widget is not None:
                    option.widget.combo_items = ['']
            self.choice_options.append((chooser, value))

        self.print_option = self.add_option('print parameters',
                                            widget_type='button',
                                            callback=self.print_parameters)
        self.editor_option = self.add_option('open editor',
                                             widget_type='button',
                                             callback=self.request_editor)
        if self.editor_option.widget is not None:
            self.editor_option.widget.set_tooltip(
                "the plugin's own window, modal: the patch UI pauses while it "
                'is open, the audio does not')
        self.status_property = self.add_property(
            'status', widget_type='label',
            default_value='pedalboard not installed'
            if not plugin_hosting_available() else 'no plugin')

        self.signal_output = self.add_signal_output('signal', self.unit.out)
        self.right_output = self.add_signal_output('right', self.unit.right)
        self.add_switch()
        self.finish_synth_node()

    # -- loading ------------------------------------------------------------
    #
    # Never from a widget callback. Loading a plugin takes hundreds of
    # milliseconds and some of them touch UI toolkits while initialising, and
    # widget callbacks arrive on whatever thread dpg happens to be on -- so
    # the callbacks only raise a flag and the frame task, which is the main
    # loop, does the work.

    def request_reload(self):
        self._reload_pending = True

    def request_editor(self):
        self._editor_pending = True

    def update_parameters_from_widgets(self):
        # The loader restores the file and plugin options without necessarily
        # firing their callbacks, so ask for the load here as well.
        self._reload_pending = True
        super().update_parameters_from_widgets()

    def open_editor(self):
        """The plugin's own window, as a modal interlude.

        show_editor blocks the calling thread until the window is closed, and
        on macOS it has to be the main thread -- which is exactly the thread
        the patch UI runs on. So opening the editor pauses dpg: no knobs, no
        repatching, no metro~ bangs until it closes. The audio thread is not
        this thread, and pedalboard releases the GIL for the duration, so
        everything already sounding keeps sounding, and patched modulation
        keeps modulating -- measured, not assumed: a rendering thread lost no
        blocks across an open editor.

        When the window closes, sync_from_plugin pulls what it changed back
        into the panel, so a parameter the node shows is saved with the patch
        exactly as the editor left it. Parameters the panel does not show
        stay in the plugin only, and do not survive a save and reload.
        """
        if self.plugin is None:
            self.set_status('no plugin to show')
            return
        self.set_status('editor open -- patch UI paused until it closes')
        try:
            self.plugin.show_editor()
        except Exception as error:
            self.set_status('editor failed: ' + str(error))
            print('vst~: editor failed (' + str(error) + ')')
            return
        self.sync_from_plugin()
        self.set_status(self.describe_plugin())

    def sync_from_plugin(self):
        """Pull the plugin's current values back into the panel.

        The editor writes into the plugin directly, so once it closes any
        knob or menu showing one of its parameters is out of date. This is
        the moment to catch up -- and the only one there is, since the frame
        loop is inside show_editor the whole time the window is open.

        A knob whose inlet has a cord patched into it is left alone. Its
        parameter is being driven, and the knob is an offset in that sum,
        not a readout; writing the modulation's momentary value into it
        would bake a passing instant into the patch.

        Parameters the panel does not show are simply not the panel's
        business -- the editor's edits to them stay in the plugin either way.
        """
        if self.plugin is None:
            return
        for index, option in enumerate(self.slot_options):
            name = any_to_string(option()).strip()
            parameter = self._parameters.get(name)
            if parameter is None:
                continue
            if self.unit.parameter_in[index].sources:
                continue
            port = self.slot_inputs[index]
            if port.widget is None:
                continue
            value = float(parameter.raw_value)
            if abs(any_to_float(port()) - value) > 1e-6:
                port.widget.set(value)
        for chooser, value_option in self.choice_options:
            name = any_to_string(chooser()).strip()
            if name not in self._parameters:
                continue
            current = any_to_string(getattr(self.plugin, name, ''))
            if current and any_to_string(value_option()).strip() != current:
                value_option.set(current)
                # The plugin already holds this setting; remembering it here
                # keeps apply_choices from sending it again.
                self._applied_choices[name] = current
        self.parameters_changed()

    def synth_frame_task(self):
        if self._reload_pending:
            self._reload_pending = False
            self.load_requested_plugin()
        if self._editor_pending:
            self._editor_pending = False
            self.open_editor()
        if self.unit.error:
            if self.unit.error != self._reported_error:
                self._reported_error = self.unit.error
                self.plugin = None
                self.set_status('dropped -- ' + self.unit.error)
                print('vst~: dropped ' + str(self.unit.plugin_name) + ' -- '
                      + self.unit.error)
            return
        # The cost figure is worth showing but not worth redrawing every
        # frame for; a couple of times a second reads as live.
        if self.plugin is not None:
            self._cost_countdown -= 1
            if self._cost_countdown <= 0:
                self._cost_countdown = 30
                self.set_status(self.describe_plugin())

    def load_requested_plugin(self):
        self.unit.attach(None, 1)
        self.plugin = None
        self.parameter_names = []
        self.numeric_names = []
        self.choice_names = []
        self._reported_error = ''

        fragment = any_to_string(self.file_option()).strip()
        if not fragment:
            self.set_status('no plugin')
            self.offer_parameters()
            return
        if not plugin_hosting_available():
            self.set_status('pedalboard not installed')
            return

        path = find_plugin_file(fragment)
        if path is None:
            self.set_status('nothing installed matching "' + fragment + '"')
            print('vst~: no plugin file matching "' + fragment + '". '
                  'Installed:')
            for installed in installed_plugin_files():
                print('   ' + os.path.basename(installed))
            return

        refusal = plugin_file_refusal(path)
        if refusal:
            self.set_status(refusal)
            print('vst~: ' + refusal)
            return

        names = plugin_names_in_file(path)
        self.set_combo_items(self.plugin_option, names or [''])
        wanted = any_to_string(self.plugin_option()).strip()
        if wanted not in names:
            wanted = names[0] if names else None
            if wanted:
                self.plugin_option.set(wanted)

        try:
            plugin, channels = open_plugin(path, wanted,
                                           synth_graph.sample_rate)
        except Exception as error:
            self.set_status(str(error))
            print('vst~: ' + str(error))
            return

        self.plugin = plugin
        # One pass over the parameter dictionary, here on the main thread.
        # Reading `plugin.parameters` rebuilds it every time and costs a
        # quarter of a block period, so nothing on the audio path may touch it.
        parameters = dict(plugin.parameters)
        self.parameter_names = sorted(parameters.keys())
        self._parameters = parameters
        self._applied_choices = {}
        # A parameter with a numeric range is a knob and can be modulated; one
        # without is a menu, and its values are only meaningful by name.
        self.numeric_names = [name for name in self.parameter_names
                              if parameters[name].range[0] is not None]
        self.choice_names = [name for name in self.parameter_names
                             if parameters[name].range[0] is None
                             and len(parameters[name].valid_values or ()) > 1]
        self.unit.attach(plugin, channels, name=str(plugin.name),
                         latency=int(plugin.reported_latency_samples))
        self.offer_parameters()
        self.bind_parameters()
        self.apply_choices()
        self.set_status(self.describe_plugin())

    # -- parameters ---------------------------------------------------------

    def offer_parameters(self):
        for option in self.slot_options:
            self.set_combo_items(option, [''] + list(self.numeric_names))
        for chooser, value in self.choice_options:
            self.set_combo_items(chooser, [''] + list(self.choice_names))
            self.set_combo_items(value, self.values_for(chooser) or [''])

    def set_combo_items(self, option, items):
        """Combos take their items at creation, so a live list needs both."""
        if option.widget is None:
            return
        option.widget.combo_items = list(items)
        if dpg.does_item_exist(option.widget.uuid):
            try:
                dpg.configure_item(option.widget.uuid, items=list(items))
            except Exception:
                pass

    def bind_parameters(self):
        """Hand the unit the parameter objects its slots drive."""
        self.relabel_slots()
        if self.plugin is None:
            self.unit.bind_parameters(())
            return
        pairs = []
        for option, inlet in zip(self.slot_options, self.unit.parameter_in):
            name = any_to_string(option()).strip()
            parameter = self._parameters.get(name)
            if parameter is not None:
                pairs.append((parameter, inlet))
        self.unit.bind_parameters(pairs)

    def relabel_slots(self):
        """Let each slot wear the name of the parameter it drives.

        Only the drawn text changes. Renaming a port for real would be the
        dangerous thing, but nothing here needs it: what a widget shows beside
        itself is a separate label from the one it was built with, and it is
        the built one that everything durable keys on. Links are restored by
        the input's position in the node (node_editor resolves
        dest_input_index against node.inputs), and saved values are matched
        against the widget's own label, which stays 'param n' for life. So a
        slot can read 'feedback' on screen while remaining, to every cord and
        every saved patch, the third input of this node.

        An empty slot goes back to its number rather than showing nothing --
        an unnamed inlet you can still patch into would be a worse lie than a
        dull name.
        """
        for index, port in enumerate(self.slot_inputs):
            widget = getattr(port, 'widget', None)
            if widget is None:
                continue
            chosen = any_to_string(self.slot_options[index]()).strip()
            text = chosen or ('param ' + str(index + 1))
            if widget.prefix_label == text:
                continue
            widget.prefix_label = text
            prefix = getattr(widget, 'prefix_uuid', None)
            if prefix is not None and dpg.does_item_exist(prefix):
                dpg.set_value(prefix, text)
            # The name column is sized to the longest name in it, and that has
            # just changed, so ask for it to be squared off again.
            self._labels_aligned = False
            self._align_attempts = 0

    # -- choices ------------------------------------------------------------

    def values_for(self, chooser):
        """The settings the parameter named in this chooser can take."""
        parameter = self._parameters.get(any_to_string(chooser()).strip())
        if parameter is None:
            return []
        return [any_to_string(item) for item in (parameter.valid_values or ())]

    def choice_source_changed(self):
        """A different parameter was picked, so its settings replace the old.

        The value combo is left showing whatever the plugin currently has,
        which is both the honest reading and a sensible starting point.
        """
        for chooser, value in self.choice_options:
            options = self.values_for(chooser)
            self.set_combo_items(value, options or [''])
            if options and any_to_string(value()).strip() not in options:
                name = any_to_string(chooser()).strip()
                current = any_to_string(getattr(self.plugin, name, options[0]))
                value.set(current if current in options else options[0])
        self.apply_choices()

    def apply_choices(self):
        """Push each chosen setting to the unit, which applies it next block.

        The raw value comes from the parameter's own get_raw_value_for rather
        than from the position of the name in the list. Discrete parameters do
        not quantise evenly -- stepping Supermassive's modes by index lands on
        the wrong one for half the list -- and this is exact.
        """
        if self.plugin is None:
            return
        for chooser, value in self.choice_options:
            name = any_to_string(chooser()).strip()
            parameter = self._parameters.get(name)
            if parameter is None:
                continue
            wanted = any_to_string(value()).strip()
            if not wanted or wanted not in (parameter.valid_values or ()):
                continue
            # Every knob move calls through here; only send what changed.
            if self._applied_choices.get(name) == wanted:
                continue
            self._applied_choices[name] = wanted
            try:
                self.unit.set_choice(parameter,
                                     parameter.get_raw_value_for(wanted))
            except Exception as error:
                print('vst~: could not set ' + name + ' to ' + wanted
                      + ' (' + str(error) + ')')

    def print_parameters(self):
        if self.plugin is None:
            print('vst~: no plugin loaded')
            return
        print('vst~: ' + str(self.plugin.name) + ', '
              + str(len(self.parameter_names)) + ' parameters')
        for name in self.numeric_names:
            print('   ' + name + ' = '
                  + str(getattr(self.plugin, name, '?')) + '   [knob]')
        for name in self.choice_names:
            parameter = self._parameters[name]
            print('   ' + name + ' = '
                  + str(getattr(self.plugin, name, '?')) + '   [menu: '
                  + ', '.join(any_to_string(v)
                              for v in parameter.valid_values) + ']')

    # -- display ------------------------------------------------------------

    def describe_plugin(self):
        latency = self.unit.latency
        text = (str(self.unit.plugin_name) + '  '
                + ('stereo' if self.unit.channels > 1 else 'mono')
                + '  ' + format(self.unit.cost_ms, '.2f') + ' ms/block')
        if latency > 0:
            text += ('  +' + format(latency / synth_graph.sample_rate * 1000.0,
                                    '.0f') + ' ms latency')
        return text

    def set_status(self, text):
        if self.status_property is not None:
            self.status_property.set(text)

    def sync_options(self):
        self.bind_parameters()
        self.apply_choices()
