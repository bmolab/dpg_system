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
import math
import time
import os
import json

from fuzzywuzzy import fuzz

from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.synth_core import (
    VesselUnit, RattleUnit, StrikeUnit, FaderOutUnit,
    synth_graph, start_filter_warm_up,
    SigUnit, VcoUnit, VcfUnit, VcaUnit, AdsrUnit, LfoUnit, ClockUnit, RampUnit,
    AdditiveUnit, DelayUnit, FoldUnit, CrushUnit,
    ShaperUnit, FormantUnit, VocoderUnit, OneEuroUnit, FORMANT_VOWELS,
    MixUnit, MultUnit, PanUnit, AudioOutUnit, SpaceUnit, CleanUnit, VuUnit,
    SnapshotUnit, ScalerUnit,
    CaptureUnit, StreamUnit, SamplerOscUnit, SamplerBuffer, PhasorUnit, VstUnit,
    StringUnit, ModalUnit, WindUnit, BowUnit, RubUnit, BlowUnit, FaderUnit,
    StrokeUnit, ShakerUnit, BrassUnit, StrainUnit, WhooshUnit,
    plugin_hosting_available, installed_plugin_files, find_plugin_file,
    plugin_names_in_file, open_plugin, plugin_file_refusal,
    LFO_SHAPES, VCO_SHAPES, SAMPLER_MODES, NoiseUnit, BounceUnit, DrumUnit, MotorUnit, BubblesUnit,
    SpinUnit)

import os

import numpy as np

AUDIO_FILE_EXTENSIONS = ('.wav', '.aif', '.aiff', '.mp3', '.flac', '.ogg', '.m4a')


def register_synth_nodes():
    Node.app.register_node('audio_out~', AudioOutNode.factory)
    Node.app.register_node('place~', PlaceNode.factory)
    Node.app.register_node('clean~', CleanNode.factory)
    Node.app.register_node('condition~', CleanNode.factory)
    Node.app.register_node('vu~', VuNode.factory)
    Node.app.register_node('meter~', VuNode.factory)
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
    Node.app.register_node('string~', StringNode.factory)
    Node.app.register_node('pluck~', StringNode.factory)
    Node.app.register_node('modal~', ModalNode.factory)
    Node.app.register_node('resonator~', ModalNode.factory)
    Node.app.register_node('shape_modes', ShapeModesNode.factory)
    Node.app.register_node('strike~', StrikeNode.factory)
    Node.app.register_node('fader_out~', FaderOutNode.factory)
    Node.app.register_node('vessel~', VesselNode.factory)
    Node.app.register_node('wind~', WindNode.factory)
    Node.app.register_node('reed~', WindNode.factory)
    Node.app.register_node('flute~', WindNode.factory)
    Node.app.register_node('bow~', BowNode.factory)
    Node.app.register_node('bowed~', BowNode.factory)
    Node.app.register_node('brass~', BrassNode.factory)
    Node.app.register_node('horn~', BrassNode.factory)
    Node.app.register_node('bubbles~', BubblesNode.factory)
    Node.app.register_node('gurgle~', BubblesNode.factory)
    Node.app.register_node('motor~', MotorNode.factory)
    Node.app.register_node('engine~', MotorNode.factory)
    Node.app.register_node('bounce~', BounceNode.factory)
    Node.app.register_node('drop~', BounceNode.factory)
    Node.app.register_node('spin~', SpinNode.factory)
    Node.app.register_node('coin~', SpinNode.factory)
    Node.app.register_node('drum~', DrumNode.factory)
    Node.app.register_node('skin~', DrumNode.factory)
    Node.app.register_node('strain~', StrainNode.factory)
    Node.app.register_node('creak~', StrainNode.factory)
    Node.app.register_node('noise~', NoiseNode.factory)
    Node.app.register_node('hiss~', NoiseNode.factory)
    Node.app.register_node('whoosh~', WhooshNode.factory)
    Node.app.register_node('swish~', WhooshNode.factory)
    Node.app.register_node('rub~', RubNode.factory)
    Node.app.register_node('glass~', RubNode.factory)
    Node.app.register_node('blow~', BlowNode.factory)
    Node.app.register_node('pipe~', BlowNode.factory)
    Node.app.register_node('fader~', FaderNode.factory)
    Node.app.register_node('stroke~', StrokeNode.factory)
    Node.app.register_node('bowing~', StrokeNode.factory)
    Node.app.register_node('shaker~', ShakerNode.factory)
    Node.app.register_node('rattle~', RattleNode.factory)
    Node.app.register_node('rain~', ShakerNode.factory)
    Node.app.register_node('capture~', CaptureNode.factory)
    Node.app.register_node('array~', CaptureNode.factory)
    Node.app.register_node('stream~', StreamNode.factory)
    Node.app.register_node('audio_in~', StreamNode.factory)
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
        self._switch_placed = True     # armed by add_switch
        self._proportional_ports = []
        self._labels_aligned = False
        self._align_attempts = 0
        self._registered = False

    # -- port construction --------------------------------------------------

    # What today's port names used to be called. The loader resolves a saved
    # cord by name first -- the stored index is only a hint, and a port whose
    # label has changed is searched for by name and then by this archive
    # before the link is given up on. So a rename without an archive entry
    # silently disconnects every old patch that used the port; these entries
    # are what let 'in' become 'left in' without costing anyone a cord.
    LEGACY_PORT_NAMES = {
        'left in': ('in',),
        'left carrier': ('carrier',),
        'left out': ('signal', 'left'),
        'right out': ('right',),
        # 'drive' was the wrong voice for it. A resonator is passive --
        # 'excite in' says so correctly -- and what this sets is how
        # keenly the body hears what arrives, not how hard something
        # pushes. It sits under the excite inlet now, where the thing it
        # scales is.
        'sensitivity': ('drive',),
        # spin~ once made a blow every time its rim unloaded. Testing
        # against real coins said otherwise -- a settling coin stays on
        # the table and its flop is the grinding gone sharp -- so the
        # only impact left is the one at the end, and the outlet is
        # named for it.
        'landing': ('strikes',),
    }

    def add_signal_input(self, label, inlet):
        """An inlet that only accepts a patched signal (an audio input)."""
        port = self.add_input(label)
        port.synth_inlet = inlet
        for old_name in SynthNode.LEGACY_PORT_NAMES.get(label, ()):
            port.name_archive.append(old_name)
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
        # A renamed knob has to answer to what it used to be called, or the
        # loader searches by name, fails, and drops the cord -- silently,
        # in every patch that ever used it. Signal ports have carried this
        # since the stereo rename; modulation ports were missed, which made
        # renaming one of them a quiet way to break saved work.
        for old_name in SynthNode.LEGACY_PORT_NAMES.get(label, ()):
            port.name_archive.append(old_name)
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
                # The depth is saved under its own name, so it needs the
                # same memory as the knob it belongs to.
                for old_name in SynthNode.LEGACY_PORT_NAMES.get(label, ()):
                    option.name_archive.append(old_name + ' depth')
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

    def make_drag_proportional(self, port, fraction=0.04, floor=0.002,
                               ceiling=1.0):
        """Drag speed that follows the value: exponential travel, honestly.

        A decay knob spans twenty milliseconds to a minute. A fixed drag
        step is too coarse at one end or takes all day at the other; a
        hidden exponential mapping would fix the feel but the number shown
        would stop being seconds. Proportional stepping keeps the number
        and the feel: each pixel moves the value by a fraction of itself,
        so short values adjust finely and long ones sweep.
        """
        self._proportional_ports.append((port, fraction, floor, ceiling))
        self._sync_proportional_speeds()

    def _sync_proportional_speeds(self):
        for port, fraction, floor, ceiling in self._proportional_ports:
            if port.widget is None:
                continue
            value = abs(any_to_float(port()))
            speed = min(ceiling, max(floor, value * fraction))
            port.widget.speed = speed
            if dpg.does_item_exist(port.widget.uuid):
                try:
                    dpg.configure_item(port.widget.uuid, speed=speed)
                except Exception:
                    pass

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
        # Created last, so every other port keeps the link index it has
        # always had -- and then *drawn* up under the audio inputs, where a
        # bypass reads as what it is: a valve on what comes in. Display and
        # link order are separate concerns; the move happens in the frame
        # task once the rows exist to be moved.
        self._switch_placed = False
        return self.switch_input

    def place_switch(self):
        """Draw the switch under the audio-in rows; sources get it on top.

        The row after the plain signal inputs -- the widgetless ports at the
        head of the node, 'left in' and 'right in' on a processor -- is where
        the switch is moved to. A source has no such prefix, so its 'enable'
        becomes the first row instead, which is where the master switch of a
        sound-maker belongs anyway.
        """
        port = self.switch_input
        if port is None:
            return True
        if not dpg.does_item_exist(port.uuid):
            return False
        target = None
        for candidate in self.inputs:
            if candidate is port:
                continue
            if candidate in self.signal_inputs and candidate.widget is None:
                target = None      # still inside the audio-in prefix
                continue
            target = candidate
            break
        if target is None or not dpg.does_item_exist(target.uuid):
            return True            # nothing after the prefix; last is fine
        try:
            dpg.move_item(port.uuid, parent=self.uuid, before=target.uuid)
        except Exception as error:
            print(self.label + ': could not place switch (' + str(error) + ')')
        return True

    def add_signal_output(self, label, signal):
        port = self.add_output(label)
        port.synth_signal = signal
        port.synth_unit = self.unit
        for old_name in SynthNode.LEGACY_PORT_NAMES.get(label, ()):
            port.name_archive.append(old_name)
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
        self._sync_proportional_speeds()
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
        if not self._switch_placed:
            self._switch_placed = self.place_switch()
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

        self.add_signal_input('left in', self.unit.signal_in)
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

        self.signal_output = self.add_signal_output('left out', self.unit.out)
        self.right_output = self.add_signal_output('right out', self.unit.right)
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

    The 'right out' outlet carries the other half of the stereo spread. At
    one voice, or with spread at 0, it carries the same signal as 'left out',
    so a mono patch can ignore it. For the spread to survive to the speakers the
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

        self.signal_output = self.add_signal_output('left out', self.unit.out)
        # Appended, so links saved against the old single outlet keep their
        # index and existing patches load unchanged.
        self.right_output = self.add_signal_output('right out', self.unit.right)
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

        self.add_signal_input('left in', self.unit.signal_in)
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

        self.signal_output = self.add_signal_output('left out', self.unit.out)
        self.right_output = self.add_signal_output('right out', self.unit.right)

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

        self.add_signal_input('left in', self.unit.signal_in)
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

        self.signal_output = self.add_signal_output('left out', self.unit.out)
        self.right_output = self.add_signal_output('right out', self.unit.right)

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

        self.add_signal_input('left in', self.unit.signal_in)
        self.add_signal_input('right in', self.unit.right_in)
        self.bits_input = self.add_modulation_input(
            'bits', self.unit.bits_in, default_value=bits, minimum=1.0,
            maximum=24.0, speed=0.05)
        # Everything worth hearing is in the bottom few percent of this range,
        # so it stays a drag box rather than becoming a slider.
        self.rate_input = self.add_modulation_input(
            'rate', self.unit.rate_in, default_value=rate, minimum=1.0,
            maximum=synth_graph.sample_rate, speed=20.0, slider=False)

        self.signal_output = self.add_signal_output('left out', self.unit.out)
        self.right_output = self.add_signal_output('right out', self.unit.right)
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
    nothing changes; 'right out' then carries the same signal as 'left out',
    so a mono chain can ignore it.
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

        self.add_signal_input('left in', self.unit.signal_in)
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

        self.signal_output = self.add_signal_output('left out', self.unit.out)
        # Appended, so links saved against the single outlet keep their index.
        self.right_output = self.add_signal_output('right out', self.unit.right)
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

        self.add_signal_input('left in', self.unit.signal_in)
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

        self.signal_output = self.add_signal_output('left out', self.unit.out)
        self.right_output = self.add_signal_output('right out', self.unit.right)
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
        self.add_signal_input('left carrier', self.unit.carrier_in)
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

        self.signal_output = self.add_signal_output('left out', self.unit.out)
        self.right_output = self.add_signal_output('right out', self.unit.right)
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
    channels. Unpatched it is the mono vca~ it always was, and 'right out'
    carries the same signal as 'left out'.
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

        self.add_signal_input('left in', self.unit.signal_in)
        self.add_signal_input('right in', self.unit.right_in)
        self.add_modulation_input('gain', self.unit.gain_in,
                                  default_value=gain, minimum=0.0, speed=0.01)

        self.response_input = self.add_input('response', widget_type='combo',
                                             default_value=response,
                                             callback=self.parameters_changed)
        self.response_input.widget.combo_items = list(VcaNode.RESPONSES)

        self.signal_output = self.add_signal_output('left out', self.unit.out)
        # Appended, so links saved against the single outlet keep their index.
        self.right_output = self.add_signal_output('right out', self.unit.right)
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

        self.left_output = self.add_signal_output('left out', self.unit.left)
        self.right_output = self.add_signal_output('right out', self.unit.right)
        self.add_switch()
        self.finish_synth_node()


# ----------------------------------------------------------------------------
# audio_out~ / snapshot~
# ----------------------------------------------------------------------------

class VuNode(SynthNode):
    """Level meter: branch a cord into it, and watch.

    A tap, not a link -- there are no audio outlets, so the chain it
    reads is untouched by construction. Two bars with meter ballistics
    (quick up, slow down) over a dimmed color scale, and a peak readout
    in dB, held long enough to see. Bypassing a gauge means nothing, so
    there is no switch either. The 'peak' outlet reports the held peak
    each frame for patch logic -- ducking, auto-gain, a warning light.
    """

    METER_FLOOR_DB = -60.0
    METER_CEIL_DB = 6.0
    METER_WIDTH = 150
    METER_HEIGHT = 13
    # (from dB, to dB, color): the customary reading -- comfortable,
    # hot, and about to be sorry.
    ZONES = ((-60.0, -12.0, (70, 190, 105, 255)),
             (-12.0, -3.0, (235, 180, 70, 255)),
             (-3.0, 6.0, (235, 85, 70, 255)))

    @staticmethod
    def factory(name, data, args=None):
        return VuNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = VuUnit(synth_graph.sample_rate)

        self.add_signal_input('left in', self.unit.signal_in)
        self.add_signal_input('right in', self.unit.right_in)

        self.meter_display = self.add_display('')
        self.meter_display.submit_callback = self.submit_display
        self._bar_tags = []
        self.db_property = self.add_property('dB', widget_type='label',
                                             default_value='-inf dB')

        self.peak_output = self.add_output('peak')
        self.finish_synth_node()
        self._shown = (-999.0, -999.0, -999.0, -999.0)

    def submit_display(self):
        width = VuNode.METER_WIDTH
        height = VuNode.METER_HEIGHT
        self._bar_tags = []
        for _channel in range(2):
            drawlist = dpg.add_drawlist(width=width, height=height)
            fills = []
            for low, high, color in VuNode.ZONES:
                x0 = self._bar_fraction(low) * width
                x1 = self._bar_fraction(high) * width
                # The scale itself, dimmed: the zones are visible before
                # anything reaches them, which is what makes them a scale.
                dpg.draw_rectangle(pmin=(x0, 0), pmax=(x1, height),
                                   fill=(color[0], color[1], color[2], 48),
                                   color=(0, 0, 0, 0), parent=drawlist)
            for low, high, color in VuNode.ZONES:
                x0 = self._bar_fraction(low) * width
                fills.append(dpg.draw_rectangle(
                    pmin=(x0, 1), pmax=(x0, height - 1), fill=color,
                    color=(0, 0, 0, 0), parent=drawlist))
            peak = dpg.draw_line((0, 0), (0, height),
                                 color=(230, 230, 230, 0), thickness=2,
                                 parent=drawlist)
            self._bar_tags.append({'fills': fills, 'peak': peak})

    @staticmethod
    def _to_db(value):
        if value <= 1.0e-6:
            return None
        return 20.0 * math.log10(value)

    def _bar_fraction(self, db):
        if db is None:
            return 0.0
        span = VuNode.METER_CEIL_DB - VuNode.METER_FLOOR_DB
        return min(1.0, max(0.0, (db - VuNode.METER_FLOOR_DB) / span))

    def synth_frame_task(self):
        state = tuple(self.unit.levels) + tuple(self.unit.peaks)
        if all(abs(now - was) < 0.001 for now, was in zip(state, self._shown)):
            return
        self._shown = state
        width = VuNode.METER_WIDTH
        height = VuNode.METER_HEIGHT
        for channel, meter in enumerate(self._bar_tags):
            level_frac = self._bar_fraction(
                self._to_db(self.unit.levels[channel]))
            for zone, (low, high, _color) in enumerate(VuNode.ZONES):
                tag = meter['fills'][zone]
                if not dpg.does_item_exist(tag):
                    continue
                x0 = self._bar_fraction(low) * width
                x1 = max(x0, min(level_frac,
                                 self._bar_fraction(high)) * width)
                dpg.configure_item(tag, pmin=(x0, 1), pmax=(x1, height - 1))
            peak_db = self._to_db(self.unit.peaks[channel])
            tag = meter['peak']
            if dpg.does_item_exist(tag):
                if peak_db is None:
                    dpg.configure_item(tag, color=(230, 230, 230, 0))
                else:
                    x = self._bar_fraction(peak_db) * width
                    hot = peak_db >= 0.0
                    dpg.configure_item(
                        tag, p1=(x, 0), p2=(x, height),
                        color=(235, 85, 70, 255) if hot
                        else (230, 230, 230, 180))
        peak_db = self._to_db(max(self.unit.peaks))
        if self.db_property.widget is not None:
            self.db_property.widget.set(
                '-inf dB' if peak_db is None else '%+.1f dB' % peak_db)
        self.peak_output.send(float(max(self.unit.peaks)))


class CleanNode(SynthNode):
    """Conditioner: subsonics off the bottom, fizz off the top.

    The channel strip's hygiene stage -- source -> fader~ -> clean~ ->
    place~ -> audio_out~ -- for when a bowed 5 Hz or a blooming low mode
    is eating headroom without being music. 24 dB per octave each way,
    flat and resonance-free between; bypassed, the signal passes
    untouched. Pull 'low cut' down when the subsonics are the point.

    clean~ <low> <high>, e.g. clean~ 30 12000.
    """

    @staticmethod
    def factory(name, data, args=None):
        return CleanNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = CleanUnit(synth_graph.sample_rate)

        numbers = []
        if args is not None:
            for arg in args:
                try:
                    numbers.append(float(arg))
                except (ValueError, TypeError):
                    continue
        if len(numbers) > 0:
            self.unit.low_in.base = max(5.0, min(300.0, numbers[0]))
        if len(numbers) > 1:
            self.unit.high_in.base = max(1000.0, min(20000.0, numbers[1]))

        self.add_signal_input('left in', self.unit.signal_in)
        self.add_signal_input('right in', self.unit.right_in)
        self.make_drag_proportional(
            self.add_modulation_input('low cut', self.unit.low_in,
                                      default_value=self.unit.low_in.base,
                                      minimum=5.0, maximum=300.0,
                                      slider=False))
        self.add_modulation_input('high cut', self.unit.high_in,
                                  default_value=self.unit.high_in.base,
                                  minimum=1000.0, maximum=20000.0,
                                  speed=50.0, slider=False)

        self.signal_output = self.add_signal_output('left out', self.unit.out)
        self.right_output = self.add_signal_output('right out',
                                                   self.unit.right)
        self.add_switch()
        self.finish_synth_node()


class PlaceNode(SynthNode):
    """Spatializer: put a source somewhere among the speakers.

    One outlet per speaker, patched onward to audio_out~'s inputs; several
    place~ into one output sum at its inlets, which is how each source gets
    its own position in the room. Stereo is a fact, not a switch: patch
    'right in' and the pair is held apart by 'width'; a mono source is a
    single point and width goes quiet.

    'ring': speakers around a circle in outlet order, 'pan' the angle -- 0
    front centre, +-0.5 the sides, +-1 the rear. 'corners': outlets are the
    corners of the room (bottom front-left, front-right, rear-left,
    rear-right, then the top four), position is three equal-power faders,
    and pan wears 'left/right' since that is what it means there. Corners
    wants 4 or 8 speakers; other counts fall back to the ring. All four
    position controls are inlets, so an lfo~ orbits a sound and effort
    data pushes it around the room.

    place~ <speakers> <space>, e.g. place~ 8 corners. Speaker count is
    fixed when the node is made.
    """

    @staticmethod
    def factory(name, data, args=None):
        return PlaceNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        count = 4
        space = 'ring'
        if args is not None:
            for arg in args:
                if arg in SpaceUnit.SPACES:
                    space = arg
                else:
                    try:
                        count = int(float(arg))
                    except (ValueError, TypeError):
                        continue
        self.unit = SpaceUnit(synth_graph.sample_rate, count)
        self.unit.space = space

        self.add_signal_input('left in', self.unit.signal_in)
        self.add_signal_input('right in', self.unit.right_in)
        self.pan_input = self.add_modulation_input(
            'pan', self.unit.position_in,
            minimum=-1.0, maximum=1.0, speed=0.01, attenuverter=False)
        if self.pan_input.widget is not None:
            self.pan_input.widget.set_tooltip(
                'pair: -1 left, +1 right · ring: 0 front centre, +-0.5 '
                'sides, +-1 rear · corners: -1 left, +1 right')
        self.width_input = self.add_modulation_input(
            'width', self.unit.width_in,
            default_value=self.unit.width_in.base,
            minimum=0.0, maximum=2.0, speed=0.01, attenuverter=False)
        if self.width_input.widget is not None:
            self.width_input.widget.set_tooltip(
                'stereo separation: 0 merges the pair to one point in '
                'space, wider pulls left and right apart')
        self.depth_input = self.add_modulation_input(
            'front/rear', self.unit.depth_in,
            minimum=-1.0, maximum=1.0, speed=0.01, attenuverter=False)
        self.height_input = self.add_modulation_input(
            'top/bottom', self.unit.height_in,
            minimum=-1.0, maximum=1.0, speed=0.01, attenuverter=False)

        self.space_option = self.add_option('space', widget_type='combo',
                                            default_value=space,
                                            callback=self.parameters_changed)
        if self.space_option.widget is not None:
            self.space_option.widget.combo_items = list(SpaceUnit.SPACES)

        for speaker in range(self.unit.count):
            self.add_signal_output('out %d' % (speaker + 1),
                                   self.unit.outs[speaker])
        self.add_switch()
        self.finish_synth_node()
        self._space_applied = False

    def sync_options(self):
        chosen = any_to_string(self.space_option()).strip()
        if chosen in SpaceUnit.SPACES:
            self.unit.space = chosen
        self._space_applied = False

    def _relabel(self, port, text):
        widget = port.widget
        if widget is None or widget.prefix_label == text:
            return
        widget.prefix_label = text
        prefix = getattr(widget, 'prefix_uuid', None)
        if prefix is not None and dpg.does_item_exist(prefix):
            dpg.set_value(prefix, text)
        self._labels_aligned = False
        self._align_attempts = 0

    @staticmethod
    def _show_port(port, visible):
        """Hide a control the current space ignores -- unless something is
        patched into it. A hidden cord would misreport the patch."""
        if port._parents:
            visible = True
        if dpg.does_item_exist(port.uuid):
            if visible:
                dpg.show_item(port.uuid)
            else:
                dpg.hide_item(port.uuid)

    def synth_frame_task(self):
        if not self._space_applied:
            if dpg.does_item_exist(self.height_input.uuid):
                corners = self.unit.active_space() == 'corners'
                self._relabel(self.pan_input,
                              'left/right' if corners else 'pan')
                self._show_port(self.depth_input, corners)
                self._show_port(self.height_input, corners)
                self._space_applied = True


class AudioOutNode(SynthNode):
    """Graph terminus: which inputs land on which device channels.

    Nothing is heard without one of these, and this is all it is now: a
    socket. One input per listed channel, a mute, the device. Level is
    fader~'s job and position is place~'s; patch them in front, several
    place~ summing into these inputs as everywhere else.

    'channels' lists the device outputs, counted from 1 the way the
    interface's front panel counts them; input k lands on the k-th listed
    channel. A channel the device does not have is silent rather than an
    error, so a wide patch still runs on a laptop.

    'device' is engine-wide: one stream, shared with the sampler, so
    changing it here changes it for everything.

    audio_out~ <channels...>, e.g. audio_out~ 3 4, audio_out~ 1 2 3 4 5 6.
    The input count is fixed when the node is made.
    """

    @staticmethod
    def factory(name, data, args=None):
        return AudioOutNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        channel_list = [1, 2]
        if args is not None and len(args) > 0:
            values = [decode_arg(args, index) for index in range(len(args))]
            whole = [int(value) for value, kind in values if kind == int]
            if len(whole) >= 2:
                channel_list = whole[:AudioOutUnit.MAX_CHANNELS]
            elif len(values) == 1 and values[0][1] in (float, int):
                print('audio_out~: a bare number used to be level; '
                      'level lives on fader~ now')
        self.unit = AudioOutUnit(synth_graph.sample_rate, len(channel_list))
        self.unit.channels = [max(0, channel - 1) for channel in channel_list]
        self._device_pending = False

        names = ['left in', 'right in'] if self.unit.count >= 2 else ['in']
        names += ['in %d' % (index + 1) for index in range(2, self.unit.count)]
        for name, inlet in zip(names, self.unit.ins):
            self.add_signal_input(name, inlet)
        self.mute_input = self.add_input('mute', widget_type='checkbox',
                                         default_value=False,
                                         callback=self.parameters_changed)

        self.channels_option = self.add_option(
            'channels', widget_type='text_input', width=140,
            default_value=' '.join(str(channel) for channel in channel_list),
            callback=self.parameters_changed)
        self._devices = self.list_output_devices()
        self.device_option = self.add_option('device', widget_type='combo',
                                             default_value='',
                                             callback=self.device_chosen)
        if self.device_option.widget is not None:
            self.device_option.widget.combo_items = \
                [''] + [name for name, _index, _count in self._devices]
            self.device_option.widget.set_tooltip(
                'engine-wide: one stream is shared with the sampler, so this '
                'changes the device for everything')

        # Two lines rather than one long one: the device is one fact, where
        # this node's channels land is another.
        self.device_property = self.add_property('out', widget_type='label',
                                                 default_value='')
        self.channels_property = self.add_property('ch', widget_type='label',
                                                   default_value='')

        self.level_output = self.add_output('peak')
        self.status_output = self.add_output('status')
        self._last_status = ''
        self._last_device_text = ''
        self._last_channel_text = ''
        self.finish_synth_node()

    @staticmethod
    def list_output_devices():
        """(display name, device index, channel count) for the combo."""
        try:
            from dpg_system.sampler import output_devices
        except ImportError:
            return []
        return [('%s (%d ch)' % (name, count), index, count)
                for index, name, count in output_devices()]

    def device_chosen(self):
        # Deferred: reopening the stream stalls for ~100 ms, which belongs in
        # the frame task, not in whatever thread the widget callback rides in.
        self._device_pending = True
        self.parameters_changed()

    def update_parameters_from_widgets(self):
        # A patch saved with a device choice reopens that device on load.
        if any_to_string(self.device_option()).strip():
            self._device_pending = True
        super().update_parameters_from_widgets()

    def apply_device_choice(self):
        chosen = any_to_string(self.device_option()).strip()
        if not chosen:
            return
        for display, index, _count in self._devices:
            if display == chosen:
                engine = ensure_engine()
                if engine is None:
                    self.status_output.send('no audio engine')
                    return
                ok, message = engine.set_device(index)
                self.status_output.send(message)
                if not ok:
                    print('audio_out~: ' + message)
                return
        self.status_output.send('unknown device ' + chosen)

    def sync_options(self):
        self.unit.muted = any_to_bool(self.mute_input())
        parsed = []
        for word in any_to_string(
                self.channels_option()).replace(',', ' ').split():
            try:
                parsed.append(max(1, min(32, int(word))))
            except (ValueError, TypeError):
                continue
        if parsed:
            if len(parsed) != self.unit.count:
                self.status_output.send(
                    'this audio_out~ has %d inputs; using the first %d '
                    'channels listed' % (self.unit.count, self.unit.count))
            listed = parsed[:self.unit.count]
            while len(listed) < self.unit.count:
                listed.append(listed[-1] + 1 if listed else 1)
            self.unit.channels = [channel - 1 for channel in listed]

    def synth_frame_task(self):
        # Re-attach if the sampler engine was restarted or replaced.
        engine = ensure_engine()
        if self._device_pending:
            self._device_pending = False
            self.apply_device_choice()
        self.level_output.send(self.unit.peak)
        status = synth_graph.last_error
        if status != self._last_status:
            self._last_status = status
            self.status_output.send(status if status else 'ok')
        # The face shows where the channels actually land, which is worth a
        # glance precisely when it is not what the panel promises.
        if engine is not None:
            device_text = engine.device_name or 'default'
            listed = '/'.join(str(channel + 1)
                              for channel in self.unit.channels)
            channel_text = 'ch %s of %d' % (listed, engine.channels)
            if device_text != self._last_device_text:
                self._last_device_text = device_text
                self.device_property.set(device_text)
            if channel_text != self._last_channel_text:
                self._last_channel_text = channel_text
                self.channels_property.set(channel_text)


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

        self.left_output = self.add_signal_output('left out', self.unit.left)
        self.right_output = self.add_signal_output('right out', self.unit.right)
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


# ----------------------------------------------------------------------------
# string~ / modal~
# ----------------------------------------------------------------------------

class StringNode(SynthNode):
    """Karplus-Strong string, playable three ways at once.

    Click 'pluck' or bang it and the string sounds; patch a trigger signal
    and it plays with sample accuracy, striking as hard as the trigger is
    tall; patch anything into 'excite in' and it drives the string
    continuously -- an enveloped noise bows it, and an effort stream played
    into a string is a string played by a body.

    string~ <frequency> <mode>, e.g. string~ 110 or string~ 220 tube.
    """

    @staticmethod
    def factory(name, data, args=None):
        return StringNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = StringUnit(synth_graph.sample_rate)

        frequency = 220.0
        mode = 'string'
        if args is not None:
            for arg in args:
                if arg in StringUnit.MODES:
                    mode = arg
                else:
                    try:
                        frequency = float(arg)
                    except (ValueError, TypeError):
                        continue
        self.unit.frequency_in.base = frequency
        self.unit.mode = StringUnit.MODES.index(mode)

        self.add_signal_input('excite in', self.unit.excite_in)
        sense_port = self.add_modulation_input(
            'sensitivity', self.unit.sensitivity_in,
            default_value=self.unit.sensitivity_in.base,
            minimum=0.0, maximum=8.0, speed=0.01, slider=False)
        self.make_drag_proportional(sense_port)
        if sense_port.widget is not None:
            sense_port.widget.set_tooltip(
                'how keenly the string hears what is patched to the excite '
                'inlet above. Unity leaves it as it always was; it reaches '
                'past that for excitations too sparse to speak')
        self.add_trigger_signal_input('pluck', self.unit.trigger_in,
                                      self.pluck)
        self.add_modulation_input('frequency', self.unit.frequency_in,
                                  default_value=frequency,
                                  minimum=StringUnit.MIN_FREQUENCY, speed=1.0)
        self.add_modulation_input('pitch', self.unit.pitch_in, speed=0.01)
        self.make_drag_proportional(
            self.add_modulation_input('decay', self.unit.decay_in,
                                      minimum=0.01, maximum=60.0, speed=0.05,
                                      slider=False))
        self.add_modulation_input('brightness', self.unit.brightness_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('position', self.unit.position_in,
                                  minimum=0.0, maximum=0.5, speed=0.01)
        self.add_modulation_input('stiffness', self.unit.stiffness_in,
                                  minimum=0.0, maximum=0.9, speed=0.01)
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self.mode_input = self.add_input('mode', widget_type='combo',
                                         default_value=mode,
                                         callback=self.parameters_changed)
        self.mode_input.widget.combo_items = list(StringUnit.MODES)

        self.color_option = self.add_option('pluck color',
                                            widget_type='slider_float',
                                            default_value=0.3, min=0.0,
                                            max=1.0,
                                            callback=self.parameters_changed)
        if self.color_option.widget is not None:
            self.color_option.widget.set_tooltip(
                'spectrum of the pluck burst: 0 is fresh white noise, 1 is '
                'darker and rounder')

        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.add_switch()
        self.finish_synth_node()

    def pluck(self):
        self.unit.fire()

    def sync_options(self):
        mode = any_to_string(self.mode_input())
        if mode in StringUnit.MODES:
            self.unit.mode = StringUnit.MODES.index(mode)
        self.unit.pluck_color = any_to_float(self.color_option())


# A material is rows of (frequency ratio, weight, decay multiple) -- where
# each mode sits against the fundamental, how loudly it speaks, and how long
# it lasts relative to the 'decay' knob. The ratio sets are the point: a free
# bar's modes spread as the squares of odd integers, a drumhead's follow
# Bessel zeros, a church bell was tuned by its founders to put a minor third
# where physics did not -- and hearing one table against another is hearing
# why those objects sound like themselves. Weights and decay multiples are
# voiced by ear against the references the ratios come from.
#
# Two kinds of table live here, used oppositely. The struck objects (bell,
# bar, membrane... and 'wood', which is honestly a block of wood) are
# pitch-relative: the table rides the frequency knob and you play it. The
# bodies (violin, guitar) are the other thing entirely -- an instrument box
# is engineered *against* singular resonance: its lowest mode is the air of
# the chamber, above it a deliberately irregular forest of plate modes of
# comparable weight, overlapping by mid-range so no note finds a wolf. A
# body table is referenced to its air mode, so it is used by FIXING the
# frequency there (violin ~275 Hz, guitar ~100 Hz), keeping decay short
# (0.05-0.15 s: body Q, where the modes widen into a landscape), and
# balancing drive against dry around whatever is patched in.
MODAL_MATERIALS = {
    'bell': [
        (0.5, 0.8, 1.6), (1.0, 1.0, 1.0), (1.2, 0.8, 0.8), (1.5, 0.6, 0.7),
        (2.0, 0.7, 0.6), (2.5, 0.5, 0.5), (2.61, 0.4, 0.45), (3.0, 0.45, 0.4),
        (3.37, 0.3, 0.3), (4.1, 0.25, 0.25), (4.5, 0.2, 0.2),
        (5.43, 0.15, 0.15),
    ],
    'bowl': [
        (1.0, 1.0, 1.0), (2.936, 0.6, 0.8), (5.505, 0.4, 0.6),
        (8.934, 0.25, 0.4), (12.827, 0.15, 0.3),
    ],
    'marimba': [
        (1.0, 1.0, 1.0), (3.984, 0.5, 0.4), (9.538, 0.25, 0.2),
        (16.688, 0.12, 0.1),
    ],
    'bar': [
        (1.0, 1.0, 1.0), (2.756, 0.55, 0.6), (5.404, 0.35, 0.35),
        (8.933, 0.2, 0.2), (13.345, 0.12, 0.12), (18.638, 0.07, 0.08),
    ],
    'membrane': [
        (1.0, 1.0, 1.0), (1.594, 0.7, 0.7), (2.136, 0.5, 0.5),
        (2.296, 0.45, 0.45), (2.653, 0.4, 0.4), (2.918, 0.3, 0.35),
        (3.156, 0.25, 0.3), (3.501, 0.2, 0.25), (3.6, 0.18, 0.22),
        (4.06, 0.12, 0.18),
    ],
    'tabla': [
        (1.0, 1.0, 1.0), (2.0, 0.8, 0.7), (3.0, 0.6, 0.5), (4.0, 0.4, 0.35),
        (5.0, 0.25, 0.25),
    ],
    'glass': [
        (1.0, 1.0, 1.0), (2.32, 0.5, 0.75), (4.25, 0.3, 0.5),
        (6.63, 0.2, 0.32), (9.38, 0.1, 0.2),
    ],
    'wood': [
        (1.0, 1.0, 1.0), (2.572, 0.6, 0.5), (4.644, 0.35, 0.3),
        (6.984, 0.2, 0.15),
    ],
    # A free circular plate: a coin, a cymbal, a dropped saucepan lid.
    # The flexural modes of a disc clamped nowhere, so ratio 1 is the
    # two-nodal-diameter mode rather than any kind of fundamental, and
    # nothing above it is an integer of anything. Ratios are the free
    # plate's tabulated series, and only the first few of them are
    # precise -- the upper rows are voiced by ear, like every other
    # table's weights. This is what spin~ wants; fix frequency high for
    # a coin (two to three kHz) and low for a plate.
    'plate': [
        (1.0, 1.0, 1.0), (1.73, 0.85, 0.9), (2.33, 0.7, 0.75),
        (3.91, 0.55, 0.55), (4.06, 0.5, 0.5), (5.94, 0.35, 0.35),
        (8.72, 0.22, 0.25), (11.75, 0.15, 0.18),
    ],
    'gong': [
        (1.0, 1.0, 1.0), (1.16, 0.9, 0.95), (1.42, 0.85, 0.9),
        (1.79, 0.7, 0.8), (2.06, 0.65, 0.7), (2.33, 0.55, 0.65),
        (2.76, 0.5, 0.55), (3.09, 0.4, 0.5), (3.4, 0.35, 0.4),
        (3.85, 0.3, 0.35), (4.31, 0.25, 0.3), (5.02, 0.2, 0.25),
    ],
    'metal': [
        (1.0, 1.0, 1.0), (1.35, 0.75, 0.85), (1.77, 0.65, 0.7),
        (2.23, 0.55, 0.6), (2.68, 0.45, 0.5), (3.24, 0.4, 0.45),
        (3.81, 0.3, 0.35), (4.5, 0.25, 0.3), (5.19, 0.2, 0.25),
        (6.02, 0.15, 0.2), (6.9, 0.1, 0.15), (7.84, 0.07, 0.1),
    ],
    # Lake ice: low, inharmonic, ringing -- the body under the pew.
    'ice': [
        (1.0, 1.0, 1.0), (1.83, 0.7, 0.8), (2.51, 0.6, 0.65),
        (3.42, 0.5, 0.5), (4.6, 0.45, 0.4), (6.1, 0.35, 0.3),
        (8.2, 0.25, 0.22), (10.9, 0.15, 0.15),
    ],
    # Paper: barely a resonator at all -- a few broad, instantly damped
    # modes. The crumple is the events; this is only their coloration.
    'paper': [
        (1.0, 1.0, 0.06), (1.7, 0.8, 0.05), (2.9, 0.65, 0.04),
        (4.3, 0.5, 0.03), (6.5, 0.35, 0.025),
    ],
    # Ratio 1 is the Helmholtz air mode (fix frequency ~275 Hz); then CBR,
    # B1-, B1+, and the plate forest rising into the bridge hill around
    # ratio 7.5-9. Spacings are irregular on purpose; that is what a good
    # box is.
    'violin': [
        (1.0, 0.9, 1.0), (1.47, 0.5, 0.8), (1.66, 1.0, 0.8),
        (1.96, 0.95, 0.7), (2.36, 0.55, 0.6), (2.65, 0.6, 0.55),
        (3.05, 0.5, 0.5), (3.45, 0.55, 0.45), (3.93, 0.45, 0.4),
        (4.44, 0.5, 0.38), (5.05, 0.45, 0.34), (5.75, 0.4, 0.3),
        (6.55, 0.45, 0.28), (7.45, 0.5, 0.25), (8.5, 0.55, 0.22),
        (9.7, 0.5, 0.2), (11.1, 0.35, 0.18), (12.7, 0.25, 0.15),
    ],
    # Ratio 1 is the soundhole's air mode (fix frequency ~100 Hz), then the
    # top plate, the back, and the forest.
    # --- added for tailoring, grouped by MODAL_MATERIAL_ORDER ---
    # A stopped pipe: odd harmonics only, which is the clarinet family
    # and the one shape none of the others has.
    'tube': [
        (1.0, 1.0, 1.0), (3.0, 0.5, 0.6), (5.0, 0.3, 0.35),
        (7.0, 0.2, 0.22), (9.0, 0.12, 0.14), (11.0, 0.08, 0.1),
    ],
    # A thin-walled vessel -- a tin, a dish, a hubcap. Close low-order
    # partials over a strong ring, which is what a coin spun in a dish
    # is actually sounding.
    'can': [
        (1.0, 1.0, 1.0), (1.28, 0.85, 0.9), (2.15, 0.6, 0.7),
        (2.94, 0.45, 0.5), (4.36, 0.3, 0.35), (5.72, 0.2, 0.25),
        (7.31, 0.12, 0.16),
    ],
    # Dense and barely tuned: modes crowded close and falling slowly, so
    # it stays bright for as long as it rings. Where 'gong' has pitch,
    # this has none.
    'cymbal': [
        (1.0, 1.0, 1.0), (1.11, 0.95, 0.95), (1.27, 0.9, 0.9),
        (1.48, 0.88, 0.85), (1.66, 0.85, 0.82), (1.93, 0.8, 0.8),
        (2.21, 0.78, 0.75), (2.55, 0.72, 0.7), (2.94, 0.7, 0.68),
        (3.41, 0.65, 0.62), (3.9, 0.6, 0.58), (4.6, 0.55, 0.52),
    ],
    # Solid metal struck: few modes, wide apart, high and hard.
    'anvil': [
        (1.0, 1.0, 1.0), (2.71, 0.7, 0.8), (4.93, 0.5, 0.6),
        (7.68, 0.35, 0.4), (11.2, 0.2, 0.25),
    ],
    # Dense and dark: almost everything above the first mode already
    # gone, which is why stone reads as a thud with a pitch in it.
    'stone': [
        (1.0, 1.0, 1.0), (1.84, 0.45, 0.5), (2.93, 0.22, 0.25),
        (4.31, 0.1, 0.12),
    ],
    # A membrane with the life damped out of it -- the same ratios as
    # 'membrane', the upper modes taken well down. A tom with a cloth on
    # it rather than a tuned drum.
    'skin': [
        (1.0, 1.0, 1.0), (1.594, 0.55, 0.6), (2.136, 0.3, 0.35),
        (2.296, 0.25, 0.3), (2.653, 0.15, 0.2), (2.918, 0.1, 0.12),
    ],
    'guitar': [
        (1.0, 1.0, 1.0), (1.93, 0.9, 0.8), (2.5, 0.6, 0.7),
        (2.9, 0.5, 0.6), (3.4, 0.55, 0.55), (4.05, 0.5, 0.5),
        (4.8, 0.45, 0.45), (5.7, 0.5, 0.4), (6.8, 0.45, 0.35),
        (8.1, 0.4, 0.3), (9.6, 0.35, 0.28), (11.4, 0.3, 0.25),
        (13.5, 0.25, 0.22), (16.0, 0.2, 0.2),
    ],
}


# Dearpygui's combo cannot nest, so the only grouping available is the
# order things appear in. Families are kept adjacent here rather than
# renamed, so every patch that already names a material keeps working.
# Anything not listed -- a material saved since, or one added to the
# table without being placed -- follows on the end rather than
# disappearing from the menu.
MODAL_MATERIAL_ORDER = (
    'bell', 'gong', 'cymbal', 'metal', 'anvil', 'bowl', 'can', 'plate',
    'tube',
    'marimba', 'bar', 'wood',
    'glass', 'ice', 'stone',
    'membrane', 'skin', 'tabla',
    'violin', 'guitar',
    'paper',
)

# Materials saved from the editor live beside the app's other state, in
# the same working directory and under the same naming as
# dpg_system_config.json. They are merged over the built-ins at import,
# so a saved material with a built-in's name shadows it -- which is the
# point: it is how a table gets voiced and kept.
MODAL_MATERIALS_FILE = 'dpg_system_materials.json'
CUSTOM_MATERIALS = {}


def _clean_mode_table(table):
    """A table is rows of (ratio, weight, decay), all positive."""
    rows = []
    for row in table or ():
        try:
            ratio, weight, decay = (float(row[0]), float(row[1]),
                                    float(row[2]))
        except (TypeError, ValueError, IndexError):
            continue
        if ratio > 0.0:
            rows.append((ratio, max(0.0, weight), max(0.0, decay)))
    rows.sort(key=lambda r: r[0])
    return rows


def load_custom_materials(path=MODAL_MATERIALS_FILE):
    """Read the saved library and merge it in. Missing is not an error --
    most installations will never have one -- but a file that IS there and
    will not parse is reported, because silently starting with a library
    the user has spent time on is worse than saying so."""
    CUSTOM_MATERIALS.clear()
    if not os.path.exists(path):
        return CUSTOM_MATERIALS
    try:
        with open(path, 'r') as handle:
            stored = json.load(handle)
    except (OSError, ValueError) as problem:
        print('could not read', path, '--', problem)
        return CUSTOM_MATERIALS
    if not isinstance(stored, dict):
        print(path, 'is not a table of materials')
        return CUSTOM_MATERIALS
    for name, table in stored.items():
        rows = _clean_mode_table(table)
        if rows:
            CUSTOM_MATERIALS[str(name)] = rows
    MODAL_MATERIALS.update(CUSTOM_MATERIALS)
    return CUSTOM_MATERIALS


def save_custom_material(name, table, path=MODAL_MATERIALS_FILE):
    """Add a material to the library and write it out. Returns the name
    saved, or None with a reason printed."""
    name = str(name).strip()
    if not name:
        print('save_material needs a name')
        return None
    rows = _clean_mode_table(table)
    if not rows:
        print('save_material: nothing to save for', name)
        return None
    CUSTOM_MATERIALS[name] = rows
    MODAL_MATERIALS[name] = rows
    try:
        with open(path, 'w') as handle:
            json.dump({key: [list(row) for row in value]
                       for key, value in CUSTOM_MATERIALS.items()},
                      handle, indent=1)
    except OSError as problem:
        print('could not write', path, '--', problem)
        return None
    return name


def forget_custom_material(name, path=MODAL_MATERIALS_FILE):
    """Drop a saved material. A built-in of the same name comes back."""
    name = str(name).strip()
    if name not in CUSTOM_MATERIALS:
        print('no saved material called', name)
        return None
    del CUSTOM_MATERIALS[name]
    if name in BUILTIN_MATERIALS:
        MODAL_MATERIALS[name] = BUILTIN_MATERIALS[name]
    else:
        MODAL_MATERIALS.pop(name, None)
    try:
        with open(path, 'w') as handle:
            json.dump({key: [list(row) for row in value]
                       for key, value in CUSTOM_MATERIALS.items()},
                      handle, indent=1)
    except OSError as problem:
        print('could not write', path, '--', problem)
    return name


def material_names():
    """Menu order: families adjacent, saved materials last."""
    ordered = [name for name in MODAL_MATERIAL_ORDER
               if name in MODAL_MATERIALS]
    ordered += [name for name in MODAL_MATERIALS if name not in ordered]
    return ordered


def rank_materials(text, limit=8):
    """The materials a few keystrokes could mean, best first.

    Exact, then prefix, then fuzzy -- so typing the start of a name
    always puts it at the top however the scorer feels about it, which
    is the difference between a finder and a guess.
    """
    text = str(text).strip().lower()
    if not text:
        return []
    names = material_names()
    exact = [name for name in names if name.lower() == text]
    starts = [name for name in names
              if name.lower().startswith(text) and name not in exact]
    rest = []
    for name in names:
        if name in exact or name in starts:
            continue
        score = (fuzz.partial_ratio(name.lower(), text)
                 + fuzz.ratio(name.lower(), text)) / 2.0
        if score > 45.0:
            rest.append((score, name))
    rest.sort(key=lambda pair: (-pair[0], pair[1]))
    return (exact + starts + [name for _, name in rest])[:limit]


def find_material(text):
    """The best material for a few keystrokes, the way the new-object box
    finds a node. Returns None if nothing is close."""
    text = str(text).strip().lower()
    if not text:
        return None
    names = material_names()
    for name in names:
        if name == text:
            return name
    ranked = rank_materials(text, limit=1)
    return ranked[0] if ranked else None


# Kept so 'forget' can put a shadowed built-in back.
BUILTIN_MATERIALS = {key: list(value) for key, value in
                     MODAL_MATERIALS.items()}
load_custom_materials()


class StrikeNode(SynthNode):
    """A deliberate hit, in a choice of characters.

    bounce~ is a mallet DROPPED, and everything after the first fall is
    gravity -- which is what a roll is made of. This is a mallet SWUNG:
    one hit when you ask for one. Patch its output at any resonator's
    excite inlet -- modal~, drum~, string~, resonator~.

    The trigger carries timing AND how hard, the way it does everywhere
    here: a taller edge hits harder, so an envelope or an effort value
    patched there plays dynamics with no second cord. What that buys is
    the thing neither bounce~ nor a unit's own strike does --

    a contact STIFFENS as it compresses. Press a ball against a plate
    and the harder you press the stiffer it gets, since more of it is
    touching. What follows is that a harder blow is a SHORTER one, so
    hitting harder does not merely make a thing louder, it makes it
    brighter. That is most of what dynamics on a struck instrument are,
    and with a fixed contact time none of it happens -- everything just
    gets louder, evenly, the way a sampler does.

    Every hit keeps its impulse and spends it over a contact the style,
    the hardness and the force decide, so hardness colours a blow rather
    than weighing it.

      tap     one hard short contact -- a fingernail, claves, a rim.
      mallet  one long soft one, and felt stiffens harder than a ball,
              so it brightens more steeply as you lean on it.
      stick   hard, held loosely, so it comes back once.
      flam    a grace note and then the hit.
      drag    three quick ones into the hit.
      brush   a spray of tiny contacts -- wire, or a bundle of straws.

    A brush is ONE stroke divided among its hairs, so its momentum is
    the same as a tap's. A flam is two strokes and a drag is four, and
    those really do put in more, which is what they are for.
    """

    @staticmethod
    def factory(name, data, args=None):
        return StrikeNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = StrikeUnit(synth_graph.sample_rate)
        self.add_trigger_signal_input('hit', self.unit.trigger_in,
                                      self.hit_once)
        self.style_input = self.add_input(
            'style', widget_type='combo', default_value=StrikeUnit.STYLES[0],
            callback=self.parameters_changed)
        self.style_input.widget.combo_items = list(StrikeUnit.STYLES)
        force_port = self.add_modulation_input(
            'force', self.unit.force_in, minimum=0.0, maximum=2.0,
            speed=0.01)
        hard_port = self.add_modulation_input(
            'hardness', self.unit.hardness_in, minimum=0.0, maximum=1.0,
            speed=0.01)
        spread_port = self.add_modulation_input(
            'spread', self.unit.spread_in, minimum=0.0, maximum=1.0,
            speed=0.01)
        scatter_port = self.add_modulation_input(
            'scatter', self.unit.scatter_in, minimum=0.0, maximum=1.0,
            speed=0.01)
        level_port = self.add_modulation_input(
            'level', self.unit.level_in, minimum=0.0, maximum=4.0,
            speed=0.02)
        if level_port.widget is not None:
            level_port.widget.set_tooltip(
                'reaches well past unity, and needs to. A unit blow is '
                'worth what a resonator\'s own trigger is worth at a '
                'height of one -- but that cannot be ONE number, because '
                'the excite inlet scales each mode by the root of one '
                'minus its pole radius, so that a SUSTAINED drive sounds '
                'the same at any decay. That makes an IMPULSE weaker the '
                'longer the decay: over a spread of pitches, decays and '
                'hardnesses the same blow wants anywhere from eight to '
                'two hundred times. The default is the middle of that '
                'and this is how you cover the rest')
        self.signal_output = self.add_signal_output('out', self.unit.out)
        for port, tip in (
                (self.style_input,
                 'what kind of hit. The styles differ in three things '
                 'and nothing else: how long the contact is, how much '
                 'it stiffens as it squashes, and how many contacts '
                 'there are'),
                (force_port,
                 'how hard, multiplying whatever the trigger\'s own '
                 'height says -- so an envelope on the cord plays the '
                 'dynamics and this sets the room it plays in. It moves '
                 'colour as well as loudness: the impulse goes up with '
                 'it and the contact time DOWN, as the fifth root, so a '
                 'harder blow is a brighter one'),
                (hard_port,
                 'how long the contact is, eight milliseconds of felt '
                 'down to a third of one of glass, before the style '
                 'scales it. Impulse is kept, so this colours a blow '
                 'rather than weighing it -- and a contact too soft to '
                 'reach a mode will not ring it, which is why a felt '
                 'beater does not make a bell speak'),
                (spread_port,
                 'how long the multiple contacts are laid over: the gap '
                 'of a flam, the rate of a drag, the width of a brush. '
                 'Nothing at all to the single-contact styles'),
                (scatter_port,
                 'how much one hit differs from the last, so a repeated '
                 'figure is not a copy of itself')):
            if port is not None and port.widget is not None:
                port.widget.set_tooltip(tip)
        self.add_switch()
        self.finish_synth_node()

    def hit_once(self):
        self.unit.fire()

    def sync_options(self):
        style = any_to_string(self.style_input())
        if style in StrikeUnit.STYLES:
            self.unit.style = StrikeUnit.STYLES.index(style)


class ShapeModesNode(Node):
    """A mode table worked out from a shape, instead of looked up.

    Patch its 'modes' outlet into modal~ or rub~ and their table stops
    being a preset and becomes whatever you described. Give it an
    outline -- a list of half-widths along the length -- say how that
    outline is swept into a volume, and say what it is made of, and it
    solves for the modes.

    The three sweeps:

      revolve   the outline is a RADIUS, spun about the long axis: a
                club, an egg, a cone, a turned bead.
      extrude   it is a HALF-WIDTH, and the section is that wide by
                'depth' deep: a bar with a shaped outline, which is what
                an undercut marimba bar is.

    'wall' hollows whichever of them you chose: a tube, a pipe, a thick
    bowl, a mortar. It reaches down to a wall a sixth of the radius and
    no further, which is where solid bricks stop being honest -- a bell
    or a wine glass has walls of one or two percent, and those want a
    shell element rather than more mesh.

    'mirror' is a checkbox beside them, not a third one of them: it says
    the outline is only half of one, running from an end to the middle,
    and reflects it. It happens to the OUTLINE, before either sweep, so
    it goes with both -- a symmetrically undercut bar, or a vase drawn as
    one half.

    WHAT WAITS FOR 'compute' AND WHAT DOES NOT. Anything you CLICK
    re-solves at once -- the sweep, the material, how it is carved -- and
    so does a profile arriving on the cord. Anything you DRAG waits: the
    length, the width, the depth and the detail, because a solve takes
    tens to hundreds of milliseconds and you do not want one on every
    frame of a drag. The mallet controls are free either way, since they
    only reweigh modes that are already solved.

    'strike' is where along it the mallet lands and 'direction' is which
    way, and both matter as much as the shape. A free-free bar has a
    node at its middle in the second mode, so struck there that mode is
    simply absent -- which is the model getting something right, not
    losing something. Neither needs the shape solving again, so they
    stay live; the outline, the size and the material do, so those wait
    for 'compute'.

    MATERIAL AND SIZE BARELY TOUCH THE TABLE, and that is not a fault:
    every mode's frequency scales as the root of stiffness over density,
    so a change of material moves them all together and cancels out of a
    RATIO. Only Poisson's ratio is left, which shifts things by under a
    percent -- though it can push a mode over or under the floor, so one
    may appear or vanish. What material and size really change is the
    PITCH, by a factor of twenty-four across the materials here, and
    modal~ overrides that with its own 'frequency'. Patch this node's
    'frequency' outlet into it and they are heard -- and note that is
    modal~'s 'frequency', not its 'pitch', which is a transposition on
    top of it and takes octaves.

    The three outlets are named for the inlets they feed, so a solved
    shape is three straight cords: frequency, decay, modes.

    The 'decay' outlet is how long the MATERIAL says it should ring, at
    the pitch it came out at -- from the loss factor, which is the one
    property of a material that survives into the sound, since
    everything else about it cancels out of a ratio. Patch it into
    modal~'s 'decay' and steel rings for twenty-seven seconds where wood
    rings for less than one. Note it is per CYCLE, so a low thing rings
    longer in seconds even when it is lossier: rubber at 36 Hz outlasts
    wood at 690.

    What it does NOT work out is the rest of decay. Frequencies follow from geometry
    and elasticity; decay follows from the material's own losses, from
    how the thing is held and from what it radiates, none of which is in
    here. 'damping' is an imposed power law, honestly labelled: 0 leaves
    every mode ringing as long as the last, 1 has a mode an octave up
    die twice as fast.

    Solids only for now. A thin shell -- a bell, a bowl, a can, a glass
    -- wants a shell element to do properly, since bricks lock in
    bending unless several sit across the wall.
    """

    @staticmethod
    def factory(name, data, args=None):
        return ShapeModesNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        from dpg_system import modal_shape
        self._shape = modal_shape
        self._solved = None
        self._key = None
        self._cavity = False
        self._openness = 'one end'
        # Flat, so what comes up first is a plain bar with straight
        # edges. A waist by default looked like the drawing was wrong.
        self._profile = [1.0, 1.0, 1.0, 1.0]

        self.profile_input = self.add_input('profile',
                                            callback=self.profile_received)
        self.solve_input = self.add_input(
            'solve', widget_type='combo', default_value='body',
            callback=self.solve_and_send)
        self.solve_input.widget.combo_items = ['body', 'cavity']
        if self.solve_input.widget is not None:
            self.solve_input.widget.set_tooltip(
                'which resonator: the THING, or the AIR inside it. They '
                'are two, and they are not alike -- a 40 cm brass tube '
                'rings at 1112 Hz as metal and 429 Hz as an air column, '
                'and nothing about one is in the other. A cavity needs '
                '"wall" below 1, since a solid has no inside. Use two of '
                'these nodes into two resonators if you want both, which '
                'is what a real vessel is')
        self.openness_input = self.add_input(
            'open', widget_type='combo', default_value='one end',
            callback=self.solve_and_send)
        self.openness_input.widget.combo_items = ['both ends', 'one end',
                                                 'sealed']
        if self.openness_input.widget is not None:
            self.openness_input.widget.set_tooltip(
                'which ends of the cavity are open, and it is worth an '
                'octave: a pipe stopped at one end sounds an octave BELOW '
                'the same pipe open at both, and gives only the odd '
                'harmonics -- 1, 3, 5, 7 -- where the open one gives them '
                'all. Sealed rings longest and radiates least. Nothing to '
                'the body')
        self.sweep_input = self.add_input(
            'sweep', widget_type='combo', default_value='extrude',
            callback=self.solve_and_send)
        self.sweep_input.widget.combo_items = list(modal_shape.SWEEPS)
        self.mirror_input = self.add_input(
            'mirror', widget_type='checkbox', default_value=False,
            callback=self.solve_and_send)
        if self.mirror_input.widget is not None:
            self.mirror_input.widget.set_tooltip(
                'the outline is only HALF of one, running from an end to '
                'the middle, and is reflected from there: n half-widths '
                'become 2n-1 stations and the last one you give sits at '
                'the centre. A checkbox and not a kind of sweep, because '
                'it happens to the OUTLINE before the sweep does '
                'anything -- so it goes with a revolve as well, which a '
                'third sweep could never have done. Draw half a vase')
        self.carve_input = self.add_input(
            'carve', widget_type='combo', default_value='depth',
            callback=self.solve_and_send)
        self.carve_input.widget.combo_items = ['depth', 'width']
        if self.carve_input.widget is not None:
            self.carve_input.widget.set_tooltip(
                'which way an extruded outline is taken, and it is not a '
                'detail: a marimba bar\'s arch is cut into its UNDERSIDE, '
                'not into its plan. Carving the depth takes the second '
                'mode 2.69 to 3.95 for the same cut that only reaches '
                '3.20 across the width, because bending stiffness goes '
                'as the depth CUBED and only as the width itself. '
                'Nothing to a revolve, which has one way across')
        self.material_input = self.add_input(
            'material', widget_type='combo', default_value='wood',
            callback=self.solve_and_send)
        self.material_input.widget.combo_items = sorted(
            modal_shape.MATERIALS)
        if self.material_input.widget is not None:
            self.material_input.widget.set_tooltip(
                'what it is made of. It hardly touches the TABLE, and '
                'cannot: every mode scales as the root of stiffness over '
                'density, so a change of material moves them all '
                'together and cancels out of a ratio -- only Poisson\'s '
                'ratio is left, worth under a percent, though it can '
                'push a mode over or under the floor so one may appear '
                'or vanish. What it changes is the PITCH, 36 Hz of '
                'rubber against 858 of glass for the same bar. Patch '
                'the "frequency" outlet into modal~\'s frequency inlet '
                'to hear it -- not into its "pitch", which is a '
                'transposition on top and takes octaves')
        # All three in METRES, over four orders of magnitude: a two
        # millimetre tine to a twenty metre beam. Nothing about the
        # solve minds -- a bar's ratios do not depend on its size at all
        # -- and a LONG one is the more accurate of the two, since it is
        # the more slender and so the closer to the ideal a bar formula
        # describes. Element shape does not seem to matter much either:
        # bricks a hundred and eighty times longer than they are thick
        # still gave the book ratios, because the bending happens along
        # the length where there are plenty of them.
        self.length_input = self.add_input(
            'length', widget_type='drag_float', default_value=0.4,
            min=0.002, max=20.0, callback=self.sizes_changed)
        self.width_input = self.add_input(
            'width', widget_type='drag_float', default_value=0.05,
            min=0.0005, max=5.0, callback=self.sizes_changed)
        self.depth_input = self.add_input(
            'depth', widget_type='drag_float', default_value=0.02,
            min=0.0005, max=5.0, callback=self.sizes_changed)
        # Waits for 'compute' like the sizes do, and more than any of
        # them: it is a DRAG, and it is the one control where dragging
        # costs more the further you drag. Wired to re-solve it fired
        # thirty-odd solves on the way from 6 to 40, each slower than
        # the last.
        self.wall_input = self.add_input(
            'wall', widget_type='drag_float', default_value=1.0,
            min=0.15, max=1.0, callback=self.solve_and_send)
        if self.wall_input.widget is not None:
            self.wall_input.widget.set_tooltip(
                'hollows it, as a fraction of the way in from the '
                'outside: 1 is solid, 0.4 leaves a wall four tenths of '
                'the radius. A tube, a pipe, a thick bowl, a mortar. It '
                'stops at 0.15 on purpose -- bricks hold to about a '
                'fifth, where bending is still within three percent of a '
                'bar\'s and the ovalling modes a tube has turn up where '
                'they should, but a tenth already puts a spurious mode '
                'just above the fundamental. A bell or a wine glass is '
                'one or two percent and wants a shell element, which is '
                'a different job and not done')
        self.detail_input = self.add_input(
            'detail', widget_type='drag_int', default_value=16,
            min=6, max=40)
        self.strike_input = self.add_input(
            'strike', widget_type='drag_float', default_value=1.0,
            min=0.0, max=1.0, callback=self.send_table)
        self.direction_input = self.add_input(
            'direction', widget_type='combo', default_value='face',
            callback=self.send_table)
        self.direction_input.widget.combo_items = ['face', 'edge', 'along']
        self.damping_input = self.add_input(
            'damping', widget_type='drag_float', default_value=1.0,
            min=0.0, max=2.0, callback=self.send_table)
        # Capped at what the far end will actually hold. modal~ keeps
        # the first MAX_MODES rows and drops the rest without a word, so
        # a knob that goes higher only promises modes that are quietly
        # thrown away.
        self.count_input = self.add_input(
            'count', widget_type='drag_int', default_value=14,
            min=2, max=ModalUnit.MAX_MODES, callback=self.send_table)
        self.show_input = self.add_input(
            'show', widget_type='drag_int', default_value=0, min=0, max=32,
            callback=self.send_mesh)
        self.swell_input = self.add_input(
            'swell', widget_type='drag_float', default_value=1.0,
            min=0.0, max=4.0, callback=self.send_mesh)
        self.compute_input = self.add_input('compute', widget_type='button',
                                            callback=self.solve_and_send)
        self.report_output = self.add_output('report')
        # Named for the inlet it feeds, like 'decay' and 'modes' beside
        # it, so the three cords read themselves. NOT 'pitch': modal~
        # has both, and they are different -- 'frequency' is where the
        # first mode sits, in hertz, and 'pitch' is a transposition on
        # top of it, in octaves. An outlet called pitch carrying hertz
        # would invite the one cord that does not work.
        self.frequency_output = self.add_output('frequency')
        self.decay_output = self.add_output('decay')
        self.modes_output = self.add_output('modes')
        self.mesh_output = self.add_output('mesh')
        for port, tip in (
                (self.profile_input,
                 'the outline: half-widths along the length, as a list. '
                 'A list of at least two numbers. They are taken as '
                 'PROPORTIONS and scaled to "width", so '
                 'a flat list is a plain bar and a dip in the middle is '
                 'an undercut one. On an extrude or a mirror it shapes '
                 'the WIDTH only and "depth" stays the depth all the way '
                 'along, which is what an undercut bar is; revolved, it '
                 'is a radius and there is only one way across. '
                 '[1, 2, 3] and [10, 20, 30] are the same outline. '
                 'Spaced evenly, and resampled to "detail" stations '
                 'however many you send. A nought is pulled up to a '
                 'hundredth of the widest rather than refused, so a cone '
                 'or a teardrop can come to a point'),
                (self.strike_input,
                 'where along it the mallet lands, nought at one end and '
                 'one at the other. It decides what you HEAR: a bar has '
                 'a node at its middle in the second mode, so struck '
                 'there that mode is missing altogether. Free -- it does '
                 'not need the shape solving again'),
                (self.direction_input,
                 'which way it is struck. A bar hit on its face wakes '
                 'the modes that bend it that way and hardly touches the '
                 'ones that bend it sideways or twist it, so this thins '
                 'the table down to what a mallet could actually reach'),
                (self.damping_input,
                 'how much faster the upper modes die. 1 is not an '
                 'arbitrary default: a material with a constant loss '
                 'factor loses the same FRACTION of a mode\'s energy '
                 'every cycle, and a mode an octave up gets through its '
                 'cycles twice as fast, so its ring is exactly half as '
                 'long. What is still imposed is everything that is not '
                 'a constant loss factor -- how it is held, what it '
                 'radiates -- so the knob is here. 0 rings them all '
                 'alike'),
                (self.detail_input,
                 'how finely it is chopped up. The ratios settle by '
                 'about 16 and cost more above that -- worth raising '
                 'once to see whether the answer moves, and leaving low '
                 'while you draw. Waits for "compute", like the sizes: '
                 'it is the one control where dragging costs more the '
                 'further you drag'),
                (self.count_input,
                 'how many modes to hand over. It is a cut, not a '
                 'search: all of them are solved either way, and this '
                 'trims the list that is SENT -- so it costs nothing to '
                 'move, and the resonator gets a shorter bank to run. '
                 'The list is in order of frequency, so this keeps the '
                 'LOWEST, which are usually also the loudest but need '
                 'not be. Capped at what modal~ holds; above that it '
                 'drops the extra rows without saying so'),
                (self.show_input,
                 'what the mesh outlet shows: 0 the shape itself, 1 and '
                 'up that numbered mode, pushed into its own shape. Free '
                 '-- the modes are already solved, this only moves them'),
                (self.swell_input,
                 'how far a mode pushes the shape, against the size of '
                 'the thing: 1 moves the surface about a tenth of it. '
                 'Only for looking at -- it is not part of the table'),
                (self.compute_input,
                 'solve it. Everything you DRAG waits for this -- the '
                 'length, width, depth and detail -- because a solve is '
                 'tens to hundreds of milliseconds and you do not want '
                 'one per frame of a drag. Everything you CLICK has '
                 'already re-solved itself, and the mallet controls '
                 'never needed to')):
            if port.widget is not None:
                port.widget.set_tooltip(tip)

    def sizes_changed(self):
        """Each pixel moves a size by a fraction of ITSELF.

        Four orders of magnitude on a linear drag is unusable at the
        bottom: a step fine enough to set two millimetres takes all day
        to reach twenty metres, and one coarse enough for twenty metres
        cannot find two millimetres at all. Proportional stepping keeps
        the number honest -- it stays metres -- and the feel even.

        These do not re-solve, and neither does 'detail'. The rule is
        not geometry versus the rest -- the material and the sweep are
        clicks and they re-solve at once -- it is DRAGS versus clicks: a
        solve is too slow to want one on every frame of a drag.
        """
        for port in (self.length_input, self.width_input,
                     self.depth_input):
            widget = port.widget
            if widget is None:
                continue
            speed = min(0.5, max(0.0002,
                                 abs(any_to_float(port())) * 0.04))
            widget.speed = speed
            if dpg.does_item_exist(widget.uuid):
                try:
                    dpg.configure_item(widget.uuid, speed=speed)
                except Exception:
                    pass

    def custom_create(self, from_file):
        self.sizes_changed()
        # NOT in __init__. The widgets do not exist yet there, so every
        # combo reads back as an empty string and the material lookup
        # raises before the node is ever drawn. This runs once they do.
        self.solve_and_send()

    # A half-width may not be nothing -- a station of no width at all is
    # an element of no volume, and the solve would hand back nonsense --
    # but a POINT is a shape anyone would want: a cone, a teardrop, a
    # club. So a thin one is pulled up to this fraction of the widest
    # rather than the whole outline being refused. A tip a hundredth of
    # the base solves perfectly well and looks pointed.
    PROFILE_FLOOR = 0.01

    def profile_received(self):
        """An outline off the cord: a list of at least two numbers.

        Taken as PROPORTIONS, not metres. They are divided by the largest
        of them and scaled to 'width', so [1, 2, 3] and [10, 20, 30] are
        the same outline, and how wide the thing really is stays 'width's
        business. Spaced evenly along the length, and resampled to
        'detail' stations however many you send.
        """
        data = self.profile_input()
        if isinstance(data, np.ndarray):
            data = data.tolist()
        if not isinstance(data, (list, tuple)):
            self.report_output.send(['error', 'an outline is a list of '
                                     'numbers'])
            return
        values = []
        for item in data:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                continue
        if len(values) < 2:
            self.report_output.send(['error', 'an outline needs at least '
                                     'two numbers'])
            return
        peak = max(values)
        if peak <= 0.0:
            self.report_output.send(['error', 'an outline needs something '
                                     'wider than nothing in it'])
            return
        floor = peak * ShapeModesNode.PROFILE_FLOOR
        pulled = sum(1 for value in values if value < floor)
        self._profile = [max(value, floor) for value in values]
        if pulled:
            self.report_output.send(['outline', len(values), 'pulled up',
                                     pulled, 'to a hundredth of the '
                                     'widest'])
        self.solve_and_send()

    DIRECTIONS = {'face': (0.0, 0.0, 1.0),
                  'edge': (0.0, 1.0, 0.0),
                  'along': (1.0, 0.0, 0.0)}

    def _direction(self):
        return ShapeModesNode.DIRECTIONS.get(
            self._chosen(self.direction_input, ShapeModesNode.DIRECTIONS,
                         'face'), (0.0, 0.0, 1.0))

    def _chosen(self, port, allowed, fallback):
        """A combo's value, or the fallback if it is not one we know.

        A widget that has not been made yet reads back as an empty
        string, and one typed into by hand can read back as anything at
        all. Neither should raise from inside a solve.
        """
        value = any_to_string(port())
        return value if value in allowed else fallback

    def solve_and_send(self):
        """Mesh it and solve it, unless nothing that matters has moved."""
        sweep_mode = self._chosen(self.sweep_input, self._shape.SWEEPS,
                                  'extrude')
        mirror = any_to_bool(self.mirror_input())
        # A patch written while mirroring was a third KIND of sweep still
        # means what it meant: take it as an extrude with the box ticked,
        # and move the controls to match so the face stops lying.
        if any_to_string(self.sweep_input()) == 'mirror':
            sweep_mode, mirror = 'extrude', True
            self.sweep_input.set('extrude')
            self.mirror_input.set(True)
        material = self._chosen(self.material_input,
                                self._shape.MATERIALS, 'wood')
        length = max(0.01, any_to_float(self.length_input()))
        width = max(0.002, any_to_float(self.width_input()))
        depth = max(0.002, any_to_float(self.depth_input()))
        detail = max(6, any_to_int(self.detail_input()))
        carve = self._chosen(self.carve_input, ('depth', 'width'), 'depth')
        wall = min(1.0, max(0.15, any_to_float(self.wall_input())))
        want_cavity = self._chosen(self.solve_input, ('body', 'cavity'),
                                   'body') == 'cavity'
        openness = self._chosen(self.openness_input,
                                ('both ends', 'one end', 'sealed'),
                                'one end')
        peak = max(self._profile)
        profile = [width * 0.5 * v / peak for v in self._profile]
        # Resampled to the asked-for detail, so how finely it is chopped
        # up is not decided by how many numbers were drawn.
        # Halved before doubling back, so a mirrored shape ends up as
        # finely chopped as an unmirrored one rather than twice.
        want = max(3, detail // 2) if mirror else detail
        if len(profile) != want + 1:
            src = np.linspace(0.0, 1.0, len(profile))
            profile = list(np.interp(np.linspace(0.0, 1.0, want + 1),
                                     src, profile))
        key = (tuple(profile), sweep_mode, material, length, depth, detail,
               carve, mirror, wall, want_cavity, openness)
        if want_cavity and wall >= 0.999:
            self.report_output.send(['error', 'a solid has no cavity -- '
                                     'bring "wall" below 1'])
            return
        if key != self._key:
            try:
                # No section handed over: sweep picks a solid one or a
                # hollow one from 'wall', and its defaults are the ones
                # this used to name.
                if want_cavity:
                    nodes, hexes = self._shape.cavity_mesh(
                        profile, length, sweep_mode, depth, carve,
                        mirror, wall)
                    freq, shape = self._shape.cavity_modes(
                        nodes, hexes,
                        open_start=openness == 'both ends',
                        open_end=openness != 'sealed', want=32)
                else:
                    nodes, hexes = self._shape.sweep(
                        profile, length, sweep_mode, depth, None, carve,
                        mirror, wall)
                    freq, shape = self._shape.solve_modes(
                        nodes, hexes, material, want=32)
            except (ValueError, RuntimeError) as err:
                self.report_output.send(['error', str(err)])
                return
            self._solved = (nodes, freq, shape, length, hexes)
            self._cavity = want_cavity
            self._openness = openness
            self._key = key
        self.send_table()
        self.send_mesh()

    def send_mesh(self):
        """The skin of it, for looking at: bare geometry, no drawing.

        A solver has no business owning a graphics context, and the
        chain's draw runs on its own thread -- so this only hands over
        vertices, triangles and normals and lets mgl_mesh do the rest,
        which keeps everything that touches the GPU on the thread that
        owns it.
        """
        if self._solved is None or getattr(self, 'in_loading_process',
                                           False):
            return
        nodes, freq, shape, length, hexes = self._solved
        which = any_to_int(self.show_input()) - 1
        if which >= 0:
            nodes = self._shape.displaced(
                nodes, shape, which, any_to_float(self.swell_input()))
        verts, tris, normals = self._shape.surface(nodes, hexes)
        self.mesh_output.send({'vertices': verts, 'faces': tris,
                               'normals': normals})

    def send_table(self):
        if self._solved is None or getattr(self, 'in_loading_process', False):
            return
        nodes, freq, shape, length = self._solved[:4]
        if self._cavity:
            table = self._shape.cavity_table(
                nodes, freq, shape, length,
                strike=min(1.0, max(0.0,
                                    any_to_float(self.strike_input()))),
                damping=any_to_float(self.damping_input()))
            table = table[:max(2, any_to_int(self.count_input()))]
            if not table or freq.size == 0:
                self.report_output.send(['error', 'nothing that drive '
                                         'could reach'])
                return
            opens = {'both ends': 2, 'one end': 1, 'sealed': 0}
            self.report_output.send(['cavity', len(table), 'lowest',
                                     round(float(freq[0]), 2), 'Hz'])
            self.frequency_output.send(float(freq[0]))
            self.decay_output.send(self._shape.cavity_ring(
                opens.get(self._openness, 1), float(freq[0])))
            self.modes_output.send(table)
            return
        table = self._shape.table_from(
            nodes, freq, shape, length,
            strike=min(1.0, max(0.0, any_to_float(self.strike_input()))),
            damping=any_to_float(self.damping_input()),
            direction=self._direction())
        table = table[:max(2, any_to_int(self.count_input()))]
        if not table:
            self.report_output.send(['error', 'nothing that mallet could '
                                     'reach'])
            return
        self.report_output.send(['modes', len(table), 'lowest',
                                 round(float(freq[0]), 2), 'Hz'])
        # The pitch it worked out, so the material and the size can be
        # HEARD. They barely touch the table: every mode scales as the
        # root of stiffness over density together, so that cancels out
        # of a ratio and only Poisson's ratio is left, which moves things
        # by under a percent. What they do move is the PITCH -- 36 Hz of
        # rubber against 858 of glass for the same bar -- and modal~
        # overrides that with its own 'frequency' unless this is wired
        # to it.
        self.frequency_output.send(float(freq[0]))
        # And how long the material says it should ring, at the pitch it
        # actually came out at. This is where a material's own character
        # survives into the sound: everything else about it cancels out
        # of a ratio, and this does not.
        self.decay_output.send(self._shape.ring_time(
            self._chosen(self.material_input, self._shape.MATERIALS,
                         'wood'), float(freq[0])))
        self.modes_output.send(table)


class ModeTableNode(SynthNode):
    """Base for nodes whose instrument is a mode table.

    modal~ strikes its table and rub~ bows it, but the table itself -- the
    stem editor, the material presets and their 'custom' discipline, the
    modes in/out ports, what the patch saves -- means the same thing on
    both, so it lives here once. A subclass builds its unit, calls
    _init_mode_editor before its ports, and places _add_mode_table_ports
    and _add_mode_table_options where its face wants them.
    """

    CUSTOM = 'custom'
    SAVE_KEY = 'modes_table'

    def _init_mode_editor(self, label, material):
        self.plot_width = 220
        self.plot_height = 96
        from dpg_system.interface_nodes import ModeEditor
        self.editor = ModeEditor(width=self.plot_width,
                                 height=self.plot_height,
                                 on_change=self.modes_edited,
                                 name=label)
        self.editor.set_modes(MODAL_MATERIALS[material], notify=False)
        # What the material combo last actually applied, and the guards that
        # keep a load from re-applying it over the table being restored --
        # the additive~ preset arrangement, for the same reasons.
        self._material_shown = material
        self._applying_material = False
        self._modes_loaded = False
        for name in ModeEditor.MESSAGES:
            self.message_handlers[name] = self.modes_message
        self.message_handlers['find'] = self.find_material_message
        self.message_handlers['save_material'] = self.save_material_message
        self.message_handlers['forget_material'] = self.forget_material_message

    def _add_mode_table_ports(self, material):
        self.modes_input = self.add_input('modes',
                                          callback=self.modes_received)
        # Twenty-odd materials is already more than a combo is pleasant
        # to hunt through, and dearpygui's combo cannot nest to group
        # them. Typing a few letters here jumps straight to one, the
        # way the new-object box finds a node -- the combo stays for
        # browsing.
        self.find_input = self.add_input('find', widget_type='text_input',
                                         widget_width=110,
                                         callback=self.find_material_typed)
        # What those keystrokes could mean, best first, hidden until
        # there are keystrokes. The top one is adopted as you type, so
        # the table follows the typing and the list says what else was
        # close; arrow down the list to take one of the others.
        self.match_list = self.add_property('##matches',
                                            widget_type='list_box',
                                            width=110)
        self._matches = []
        self.material_input = self.add_input('material', widget_type='combo',
                                             default_value=material,
                                             callback=self.material_changed)
        self.material_input.widget.combo_items = ([ModeTableNode.CUSTOM]
                                                  + material_names())
        self.modes_display = self.add_display('')
        self.modes_display.submit_callback = self.submit_display

    def _add_mode_table_options(self):
        self.modes_option = self.add_option('modes', widget_type='slider_int',
                                            default_value=self.unit.MAX_MODES,
                                            min=1, max=self.unit.MAX_MODES,
                                            callback=self.parameters_changed)
        if self.modes_option.widget is not None:
            self.modes_option.widget.set_tooltip(
                'how many of the table\'s modes ring, lowest ratio first; '
                'fewer is a simpler, cheaper object')
        self.width_option = self.add_option('width', widget_type='drag_int',
                                            default_value=self.plot_width,
                                            callback=self.size_changed)
        self.height_option = self.add_option('height', widget_type='drag_int',
                                             default_value=self.plot_height,
                                             callback=self.size_changed)

    # -- the editor ----------------------------------------------------------

    def submit_display(self):
        self.editor.submit(self.modes_display.uuid,
                           width_option=self.width_option,
                           height_option=self.height_option)

    def custom_create(self, from_file):
        self.size_changed()
        self.push_modes()
        self._show_matches(None)

    def size_changed(self):
        self.editor.set_size(any_to_int(self.width_option()),
                             any_to_int(self.height_option()))

    def synth_frame_task(self):
        self.editor.poll()

    def modes_edited(self):
        """The editor moved: retune the bank and report the table."""
        if not self._applying_material:
            self.mark_custom()
        self.push_modes()
        self.modes_output.send(self.editor.get_modes())

    def push_modes(self):
        """The editor's table, capped by the modes option, into the unit."""
        count = self.unit.MAX_MODES
        if getattr(self, 'modes_option', None) is not None:
            wanted = any_to_int(self.modes_option())
            if wanted > 0:
                count = min(wanted, self.unit.MAX_MODES)
        self.unit.set_modes(self.editor.get_modes()[:count])

    def sync_options(self):
        self.push_modes()

    # -- materials -----------------------------------------------------------

    def material_changed(self):
        chosen = any_to_string(self.material_input())
        if chosen == self._material_shown:
            return
        self._material_shown = chosen
        if chosen == ModeTableNode.CUSTOM:
            return
        # During a load the table is restored by load_custom; applying the
        # material now would overwrite what the loader is putting back.
        if self.in_loading_process:
            return
        self.apply_material(chosen)

    def apply_material(self, name):
        table = MODAL_MATERIALS.get(name)
        if table is None:
            return
        self._applying_material = True
        try:
            self.editor.set_modes(table)
        finally:
            self._applying_material = False

    def mark_custom(self):
        """A hand edit means the table is no longer the material it started as.

        Without this, reloading would re-apply the material over the edit --
        the combo would still say 'bell' and would be believed.
        """
        if self._material_shown == ModeTableNode.CUSTOM:
            return
        self._material_shown = ModeTableNode.CUSTOM
        if self.material_input.widget is not None:
            self.material_input.widget.set(ModeTableNode.CUSTOM)

    def update_parameters_from_widgets(self):
        # Patches saved before the editor existed carry only the combo, so a
        # load restores 'marimba' with no table behind it. Once the widgets
        # are back, a material that does not match what the editor holds --
        # and was not overridden by a saved table -- is applied the old way.
        chosen = any_to_string(self.material_input())
        if (chosen in MODAL_MATERIALS and chosen != self._material_shown
                and not self._modes_loaded):
            self._material_shown = chosen
            self.editor.set_modes(MODAL_MATERIALS[chosen], notify=False)
        elif chosen:
            self._material_shown = chosen
        super().update_parameters_from_widgets()

    # -- the table by patch --------------------------------------------------

    def modes_message(self, message='', message_data=[]):
        self.editor.handle_message(message, message_data)

    def _refresh_material_menu(self, select=None):
        """Put the menu back after the library has changed."""
        widget = self.material_input.widget
        if widget is None:
            return
        items = [ModeTableNode.CUSTOM] + material_names()
        widget.combo_items = items
        if dpg.does_item_exist(widget.uuid):
            dpg.configure_item(widget.uuid, items=items)
        if select is not None:
            self.material_input.set(select)

    def _adopt_material(self, name):
        """Adopt a material by name and move the combo to match, so the
        menu never says one thing while the table is another."""
        if name not in MODAL_MATERIALS:
            return False
        self.material_input.set(name)
        self._material_shown = name
        self.apply_material(name)
        return True

    def _show_matches(self, items):
        """Put the match list up, or take it away when there is nothing
        to say. Guarded on the widget existing, since ports are built
        before the window is."""
        box = getattr(self, 'match_list', None)
        if box is None or box.widget is None:
            return
        if not dpg.does_item_exist(box.widget.uuid):
            return
        if not items:
            dpg.configure_item(box.widget.uuid, show=False)
            return
        # Only 'items' and 'show' are configured after creation, which is
        # the path the new-object box already proves works.
        dpg.configure_item(box.widget.uuid, items=list(items), show=True)

    def on_edit(self, widget):
        """Per keystroke, not per return: the match is shown and taken as
        it is typed, which is the whole point of a finder."""
        finder = getattr(self, 'find_input', None)
        box = getattr(self, 'match_list', None)
        if finder is None or finder.widget is None:
            return
        if self._applying_material or self.in_loading_process:
            return
        if widget is finder.widget:
            ranked = rank_materials(dpg.get_value(finder.widget.uuid))
            self._matches = ranked
            self._show_matches(ranked)
            if ranked:
                if box is not None and box.widget is not None \
                        and dpg.does_item_exist(box.widget.uuid):
                    dpg.set_value(box.widget.uuid, ranked[0])
                self._adopt_material(ranked[0])
        elif box is not None and widget is box.widget:
            chosen = dpg.get_value(box.widget.uuid)
            if chosen:
                self._adopt_material(chosen)
                self._end_search()

    def _end_search(self):
        """The search is over: the list goes away and the field is
        emptied, so the next few keystrokes start clean rather than
        landing on the end of the last search."""
        self._matches = []
        self._show_matches(None)
        finder = getattr(self, 'find_input', None)
        if finder is None or finder.widget is None:
            return
        finder.set('')
        if dpg.does_item_exist(finder.widget.uuid):
            dpg.set_value(finder.widget.uuid, '')

    def _step_matches(self, widget, step):
        """Walk the list with the arrow keys. Returns whether the key was
        ours -- everything else has to reach the widget it was aimed at,
        or arrowing the node's other controls would stop working."""
        box = getattr(self, 'match_list', None)
        finder = getattr(self, 'find_input', None)
        if box is None or box.widget is None or not self._matches:
            return False
        if widget is not box.widget and (finder is None
                                         or widget is not finder.widget):
            return False
        if not dpg.does_item_exist(box.widget.uuid):
            return False
        current = dpg.get_value(box.widget.uuid)
        index = (self._matches.index(current)
                 if current in self._matches else 0)
        index += step
        if index < 0 or index >= len(self._matches):
            # Consumed at the ends, so arrowing past the edge of the
            # list does not fall through and start nudging a knob.
            return True
        chosen = self._matches[index]
        dpg.set_value(box.widget.uuid, chosen)
        self._adopt_material(chosen)
        return True

    def increment_widget(self, widget):
        if self._step_matches(widget, -1):
            return
        super().increment_widget(widget)

    def decrement_widget(self, widget):
        if self._step_matches(widget, 1):
            return
        super().decrement_widget(widget)

    def on_deactivate(self, widget):
        """Leaving the field puts the list away -- unless the pointer is
        on the list itself, which is someone about to pick from it."""
        finder = getattr(self, 'find_input', None)
        box = getattr(self, 'match_list', None)
        if finder is None or widget is not finder.widget:
            return
        if box is not None and box.widget is not None \
                and dpg.does_item_exist(box.widget.uuid):
            if dpg.is_item_hovered(box.widget.uuid) \
                    or dpg.is_item_clicked(box.widget.uuid):
                return
        self._end_search()

    def find_material_typed(self):
        # Widgets can fire while a patch is still loading, before the
        # table being restored has arrived -- adopting a material here
        # would overwrite it. The same guard the material combo uses.
        if self._applying_material or self.in_loading_process:
            return
        found = find_material(any_to_string(self.find_input()))
        if found is not None:
            self._adopt_material(found)
        # Return was pressed: the search is over either way.
        self._end_search()

    def find_material_message(self, message='', message_data=[]):
        found = find_material(' '.join(any_to_list(message_data)))
        if found is None:
            print('no material like that')
            return
        self._adopt_material(found)

    def save_material_message(self, message='', message_data=[]):
        """save_material <name> -- put the table being edited into the
        library, so it is in the menu next time as well as this one."""
        name = ' '.join(str(part) for part in any_to_list(message_data))
        saved = save_custom_material(name, self.editor.get_modes())
        if saved is not None:
            self._refresh_material_menu(select=saved)
            self._material_shown = saved

    def forget_material_message(self, message='', message_data=[]):
        name = ' '.join(str(part) for part in any_to_list(message_data))
        if forget_custom_material(name) is not None:
            self._refresh_material_menu()

    def modes_received(self):
        """A whole table sent to the 'modes' inlet replaces the drawing.

        Rows of [ratio, weight, decay] or a flat list of triples; a bare
        list of ratios gets weight 1 and decay 1, so 'modes 1 2.1 3.4 5.8'
        is a quick way to sketch a tuning.
        """
        data = self.modes_input()
        table = self.modes_table_from(data)
        if table:
            self.editor.set_modes(table)

    @staticmethod
    def modes_table_from(data):
        if isinstance(data, np.ndarray):
            data = data.tolist()
        if not isinstance(data, (list, tuple)) or len(data) == 0:
            return None
        if isinstance(data[0], (list, tuple, np.ndarray)):
            return [row for row in data]
        numbers = []
        for value in data:
            try:
                numbers.append(float(value))
            except (TypeError, ValueError):
                return None
        if len(numbers) >= 3 and len(numbers) % 3 == 0:
            return [numbers[i:i + 3] for i in range(0, len(numbers), 3)]
        return [[ratio, 1.0, 1.0] for ratio in numbers]

    # -- persistence ---------------------------------------------------------

    def save_custom(self, container):
        container[self.SAVE_KEY] = self.editor.get_modes()

    def load_custom(self, container):
        if self.SAVE_KEY in container:
            self._modes_loaded = True
            self.editor.set_modes(container[self.SAVE_KEY], notify=False)
            self.push_modes()


class ModalNode(ModeTableNode):
    """Struck resonator bank: bells, bars, bowls, membranes as mode tables.

    'material' picks the tuning table, 'frequency' places it, 'decay' and
    'brightness' stretch it, 'hardness' is the mallet and 'position' where it
    lands. Strike it from the button, from a trigger signal (its height is
    the velocity), or drive it continuously through 'excite in' -- noise
    through a slow envelope bows it, and a body's effort stream resonates
    through it.

    'dry' is the difference between being an instrument and being a body.
    At 0 only the modes speak: strike it, it is a bell. Raised, the excite
    input passes through alongside the ring -- and with the frequency held
    fixed (not tracking the player's pitch) and the decay short enough to
    widen the modes into formants, the bank becomes the resonant box around
    whatever is patched in: bow~ through a wood table at decay 0.1 and dry
    up is a violin.

    The mode table itself is drawn on the node: one stem per mode, standing
    at its ratio, as tall as its weight, colored by how long it rings. Drag
    a stem to tune and weight it, shift-drag vertically to set its ring
    time, right-click to add or remove one. A material is a starting point:
    edit it and the combo says 'custom', and the table you drew is what the
    patch saves. Edits land while the bank is ringing -- drag a sounding
    mode and it glisses. The same gestures arrive as messages (mode / add /
    remove), a table sent to 'modes' replaces the drawing, and 'modes out'
    reports every edit, so tables can be built, morphed and sequenced by
    patch.

    modal~ <frequency> <material>, e.g. modal~ 220 marimba.
    """

    SAVE_KEY = 'modal_modes'

    @staticmethod
    def factory(name, data, args=None):
        return ModalNode(name, data, args)

    # So a vessel can be the same node with water in it.
    UNIT = ModalUnit
    DEFAULT_MATERIAL = 'bell'
    DEFAULT_FREQUENCY = 220.0

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = self.UNIT(synth_graph.sample_rate)

        frequency = self.DEFAULT_FREQUENCY
        material = self.DEFAULT_MATERIAL
        if args is not None:
            for arg in args:
                if arg in MODAL_MATERIALS:
                    material = arg
                else:
                    try:
                        frequency = float(arg)
                    except (ValueError, TypeError):
                        continue
        self.unit.frequency_in.base = frequency
        self.unit.set_modes(MODAL_MATERIALS[material])
        self._init_mode_editor(label, material)

        self.add_signal_input('excite in', self.unit.excite_in)
        sense_port = self.add_modulation_input(
            'sensitivity', self.unit.sensitivity_in,
            default_value=self.unit.sensitivity_in.base,
            minimum=0.0, maximum=8.0, speed=0.01, slider=False)
        self.make_drag_proportional(sense_port)
        if sense_port.widget is not None:
            sense_port.widget.set_tooltip(
                'how keenly the modes hear what is patched to the excite '
                'inlet above -- it reaches well past unity, because a '
                'sparse excitation into a high bank needs real gain')
        self.add_trigger_signal_input('strike', self.unit.trigger_in,
                                      self.strike)
        self.add_modulation_input('frequency', self.unit.frequency_in,
                                  default_value=frequency, minimum=1.0,
                                  speed=1.0)
        self.add_modulation_input('pitch', self.unit.pitch_in, speed=0.01)
        self.make_drag_proportional(
            self.add_modulation_input('decay', self.unit.decay_in,
                                      minimum=0.01, maximum=60.0, speed=0.05,
                                      slider=False))
        self.add_modulation_input('brightness', self.unit.brightness_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        tilt_port = self.add_modulation_input('tilt', self.unit.tilt_in,
                                              minimum=-24.0, maximum=24.0,
                                              speed=0.1)
        if tilt_port.widget is not None:
            tilt_port.widget.set_tooltip(
                'the same slope brightness works on, in decibels per '
                'octave of each mode\'s own ratio -- as additive~ tilts '
                'its partials. Negative takes the upper modes down for a '
                'duller, heavier object, positive lifts them for a '
                'thinner brighter one. Brightness is this on a 0-1 knob '
                'spanning about plus or minus six; this reaches four '
                'times as far and is worth reading off a number. They '
                'multiply, so use either. It moves weight BETWEEN the '
                'modes rather than changing the level')
        self.add_modulation_input('hardness', self.unit.hardness_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('position', self.unit.position_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        dry_port = self.add_modulation_input('dry', self.unit.dry_in,
                                             minimum=0.0, maximum=1.0,
                                             speed=0.01)
        if dry_port.widget is not None:
            dry_port.widget.set_tooltip(
                'passes the excite input through alongside the ring; with a '
                'fixed frequency and a short decay this turns the bank into '
                'a body around whatever is patched in')
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self._add_mode_table_ports(material)
        self._add_mode_table_options()

        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.modes_output = self.add_output('modes out')
        self.add_switch()
        self.finish_synth_node()

    def strike(self):
        self.unit.fire()


class VesselNode(ModalNode):
    """A vessel with water in it: glass, bowl, can, and tipped.

    modal~ with the water added, so everything there still holds --
    material, frequency, decay, the mallet, the excite input. What is
    new is what the water does, and it does three separate things.

    'fill' takes the pitch DOWN, by about ten semitones from empty to
    full, because water touching a moving wall has to move with it and
    that is mass without stiffness. It is nowhere near linear: the wall
    hardly moves near the base and moves most at the rim, and the
    loading follows the square of that, so the fill counts to the fifth
    power. A third full is worth almost nothing; the last centimetre
    under the rim is worth two thirds of the whole range. That is the
    real behaviour and it is why a glass has to be nearly full before it
    sounds full.

    'tip' hardly changes the pitch at all -- under a semitone at thirty
    degrees. What it does is BEAT. Upright, the modes come in pairs at
    the same frequency; tipping loads one side more than the other, the
    pair comes apart, and two close frequencies beat against each other.
    It has a threshold: below about twenty degrees almost nothing
    happens, because a tilted surface is the wrong SHAPE to split a
    pair, and only starts to be the right shape once the water line runs
    into the base or the rim. Past that it comes on fast -- a slow
    warble around thirty degrees, a flutter by forty-five.

    And moving 'tip' sets the water sloshing, at around three hertz
    whatever the fill, which rides on top as a waver that settles. Tilt
    it quickly and it wavers and calms; hold it there and it is steady
    but beating.

    'turn' is where the water sits low against where it is struck, and
    it decides how much of that beat is heard at all: on a belly of the
    pattern only one of the pair answers and there is no beat, between
    them both answer and it is deepest. Round every ninety degrees.

    'swirl' is that low point going ROUND, in turns a second, and it is
    a different sound from a tilt held still. A static tilt gives two
    pitches beating slowly; a swirl gives sidebands either side of every
    mode spaced at four times the swirl rate -- a shimmer locked to the
    hand rather than a beat. It also pushes the water round and round,
    so swirling near the sloshing rate builds the slop up on itself, the
    way it does in a glass.

    'size' is the radius in metres. It sets the slosh rate only -- the
    ringing pitch is 'frequency', as everywhere else.
    """

    UNIT = VesselUnit
    DEFAULT_MATERIAL = 'glass'
    DEFAULT_FREQUENCY = 800.0

    @staticmethod
    def factory(name, data, args=None):
        return VesselNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        fill_port = self.add_modulation_input('fill', self.unit.fill_in,
                                              minimum=0.0, maximum=1.0,
                                              speed=0.01)
        if fill_port.widget is not None:
            fill_port.widget.set_tooltip(
                'how full, and it takes the pitch DOWN -- about ten '
                'semitones from empty to full. Weighted to the fifth '
                'power of the fill, because the wall barely moves at '
                'the base and moves most at the rim: a third full does '
                'almost nothing, and the last centimetre does most of '
                'it')
        tip_port = self.add_modulation_input('tip', self.unit.tip_in,
                                             minimum=0.0, maximum=60.0,
                                             speed=0.1)
        if tip_port.widget is not None:
            tip_port.widget.set_tooltip(
                'degrees off level. Hardly moves the pitch -- under a '
                'semitone at thirty -- it makes it BEAT, by loading one '
                'side more than the other and splitting the mode pairs. '
                'There is a threshold: nothing much under twenty '
                'degrees, a slow warble by thirty, a flutter by '
                'forty-five. MOVING it also sets the water sloshing, '
                'which wavers and settles')
        turn_port = self.add_modulation_input('turn', self.unit.turn_in,
                                              minimum=0.0, maximum=360.0,
                                              speed=0.5)
        if turn_port.widget is not None:
            turn_port.widget.set_tooltip(
                'where the water sits low, against where the vessel is '
                'struck. Tipping splits the ring into two; this is which '
                'of them a blow wakes. Strike where the pattern has a '
                'belly and only one answers -- one pitch, no beat at '
                'all. Strike between them and both answer and the beat '
                'is deepest. It comes round every ninety degrees, '
                'because the pattern has four bellies. It moves the '
                'sound between the two without changing how loud it is')
        swirl_port = self.add_modulation_input('swirl', self.unit.swirl_in,
                                               minimum=-8.0, maximum=8.0,
                                               speed=0.02)
        if swirl_port.widget is not None:
            swirl_port.widget.set_tooltip(
                'turns a second of the low point going round the rim -- '
                'a swirl rather than a tilt held still. It does not beat '
                'like a static tilt does; it puts sidebands either side '
                'of every mode, spaced at FOUR times the swirl, so the '
                'shimmer is locked to the hand. And it pushes the water '
                'round: swirl near the sloshing rate (a few hertz, and '
                '\'size\' sets it) and the slop builds on itself')
        size_port = self.add_modulation_input('size', self.unit.size_in,
                                              minimum=0.005, maximum=0.5,
                                              speed=0.001)
        if size_port.widget is not None:
            size_port.widget.set_tooltip(
                'the vessel\'s radius in metres. This sets how fast the '
                'water sloshes and nothing else -- the ringing pitch is '
                '\'frequency\'. A tumbler is about 0.035, a mixing bowl '
                '0.12')


class RubNode(ModeTableNode):
    """Bowed glass: modal~'s tables under bow~'s hands.

    The friction curve is fused with the mode bank inside the unit, which
    is what makes this bowing an object rather than filtering a bow sound:
    played gently it locks to one mode and sings nearly pure -- the wine
    glass -- and faster bowing breaks upward into mode-jump squeals, none
    of it programmed. Slow the bow to a stop and the object rings down at
    its own decay; strike it with a modal~ on the same table if it also
    needs a mallet.

    The mode table is the same drawn editor as modal~, with the same
    materials, messages, modes in/out and 'custom' discipline.

    rub~ <frequency> <material>, e.g. rub~ 440 glass.
    """

    SAVE_KEY = 'rub_modes'

    @staticmethod
    def factory(name, data, args=None):
        return RubNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = RubUnit(synth_graph.sample_rate)

        frequency = 440.0
        material = 'glass'
        if args is not None:
            for arg in args:
                if arg in MODAL_MATERIALS:
                    material = arg
                else:
                    try:
                        frequency = float(arg)
                    except (ValueError, TypeError):
                        continue
        self.unit.frequency_in.base = frequency
        self.unit.set_modes(MODAL_MATERIALS[material])
        self._init_mode_editor(label, material)

        self.add_modulation_input('velocity', self.unit.velocity_in,
                                  minimum=0.0, maximum=1.5, speed=0.01)
        self.add_modulation_input('force', self.unit.force_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('position', self.unit.position_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('frequency', self.unit.frequency_in,
                                  default_value=frequency,
                                  minimum=RubUnit.MIN_FREQUENCY, speed=1.0)
        self.add_modulation_input('pitch', self.unit.pitch_in, speed=0.01)
        self.make_drag_proportional(
            self.add_modulation_input('decay', self.unit.decay_in,
                                      minimum=0.01, maximum=60.0, speed=0.05,
                                      slider=False))
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self._add_mode_table_ports(material)
        self._add_mode_table_options()

        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.modes_output = self.add_output('modes out')
        self.add_switch()
        self.finish_synth_node()


class BlowNode(ModeTableNode):
    """A blown mode table: the third hand, after the mallet and the bow.

    modal~ strikes a table, rub~ bows it, this blows it -- and it is the
    driver shape_modes' cavity tables had been waiting for, an air column
    having had nothing to sound it but a mallet. Patch a shape_modes
    'modes' outlet in with the cavity solved and it plays that air.

    The reed is fused with the bank inside the unit, which is what makes
    this blowing an object rather than filtering a breath sound. Nothing
    is triggered: raise 'pressure' and above a threshold the loop finds
    its own oscillation, which is what starting a note is.

    'pressure' is the breath, measured against what it takes to shut the
    reed, so a third of the way up is a floor no reed gets under and real
    losses put the note nearer 0.6. Push past about 0.95 and the breath
    holds the reed shut and the sound stops, as over-blowing one does.
    Breath buys brightness more than level, which is how a wind
    instrument actually gets louder, and pulls the pitch sharp as it goes.

    'stiffness' is what it takes to shut the reed, so it divides the
    breath: a harder reed wants more air and gives more back. 'breath' is
    turbulence, and it is not decoration -- it is the disturbance the loop
    grows into speech, and with none at all the note starts too cleanly
    to believe. 'position' is where on the bore the mouthpiece sits, with
    0 the closed end where a reed belongs; away from it the upper modes
    are nulled one by one, which is a colour rather than a pitch.

    'register' is the register key: it spoils the lowest resonance until
    the reed gives up on it and takes the next one it can hold. That is
    how a wind player reaches the upper register, and blowing harder is
    not -- past the top of the breath range the air simply holds the reed
    shut, here as on the instrument.

    A bore with only odd modes -- the 'tube' table, a stopped pipe, which
    is to say a clarinet -- speaks on odd partials, and opening the
    register takes it up a TWELFTH rather than an octave, since an octave
    would need an even mode to jump to and there is none. Neither is
    arranged anywhere; both fall out of the table. Strike the same table
    with a modal~ if it also needs a tap.

    blow~ <frequency> <material>, e.g. blow~ 220 tube.
    """

    SAVE_KEY = 'blow_modes'

    @staticmethod
    def factory(name, data, args=None):
        return BlowNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = BlowUnit(synth_graph.sample_rate)

        frequency = 220.0
        material = 'tube'
        if args is not None:
            for arg in args:
                if arg in MODAL_MATERIALS:
                    material = arg
                else:
                    try:
                        frequency = float(arg)
                    except (ValueError, TypeError):
                        continue
        self.unit.frequency_in.base = frequency
        self.unit.set_modes(MODAL_MATERIALS[material])
        self._init_mode_editor(label, material)

        self.add_modulation_input('pressure', self.unit.pressure_in,
                                  minimum=0.0, maximum=1.5, speed=0.01)
        self.add_modulation_input('stiffness', self.unit.stiffness_in,
                                  minimum=0.1, maximum=3.0, speed=0.01)
        self.add_modulation_input('breath', self.unit.breath_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('position', self.unit.position_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('register', self.unit.register_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('frequency', self.unit.frequency_in,
                                  default_value=frequency,
                                  minimum=BlowUnit.MIN_FREQUENCY, speed=1.0)
        self.add_modulation_input('pitch', self.unit.pitch_in, speed=0.01)
        self.make_drag_proportional(
            self.add_modulation_input('decay', self.unit.decay_in,
                                      minimum=0.01, maximum=60.0, speed=0.05,
                                      slider=False))
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self._add_mode_table_ports(material)
        self._add_mode_table_options()
        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.modes_output = self.add_output('modes out')
        self.add_switch()
        self.finish_synth_node()


class BounceNode(SynthNode):
    """A dropped mallet: rolls and rebounds as gravity, not patterns.

    'drop' rising from zero drops the mallet from that height; the
    bounces accelerate and weaken geometrically until the buzz, which
    is just what gravity does. An LFO here is a stroke per cycle; a
    hand's height is a roll played by lowering it. 'gravity' is the
    first fall's time, 'bounce' the rebound, 'press' the lean into the
    roll -- faster, deader, sooner buzz -- 'hardness' each contact.

    The output is a train of half-cosine force pulses: patch it into
    drum~'s or modal~'s excite in. bounce~ <fall-time>, drop~ too.
    """

    @staticmethod
    def factory(name, data, args=None):
        return BounceNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = BounceUnit(synth_graph.sample_rate)

        if args is not None:
            for arg in args:
                try:
                    self.unit.gravity_in.base = max(0.02, min(2.0,
                                                              float(arg)))
                except (ValueError, TypeError):
                    continue

        self.add_modulation_input('drop', self.unit.drop_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        gravity_port = self.add_modulation_input(
            'gravity', self.unit.gravity_in,
            default_value=self.unit.gravity_in.base,
            minimum=0.02, maximum=2.0, speed=0.005, slider=False)
        self.make_drag_proportional(gravity_port)
        if gravity_port.widget is not None:
            gravity_port.widget.set_tooltip(
                'the first fall, in seconds: the timescale everything '
                'else follows')
        self.add_modulation_input('bounce', self.unit.bounce_in,
                                  minimum=0.0, maximum=0.99, speed=0.01)
        press_port = self.add_modulation_input('press', self.unit.press_in,
                                               minimum=0.0, maximum=1.0,
                                               speed=0.01)
        if press_port.widget is not None:
            press_port.widget.set_tooltip(
                'leaning into the roll: faster returns, deader rebound, '
                'sooner buzz. THE roll control -- map an effort here')
        self.add_modulation_input('hardness', self.unit.hardness_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.add_switch()
        self.finish_synth_node()


class SpinNode(SynthNode):
    """A spinning disc settling: the rattle that runs away.

    A dropped coin, a plate set down spinning, a hubcap in the road --
    and it is not a bounce. The contact point races around the rim at a
    rate that goes as one over the square root of the tilt, so as the
    lean bleeds away the rattle accelerates without limit and then
    stops dead. That runaway is the whole mechanism; the strikes weaken
    as the tilt's square root while their rate rises as its inverse
    square root, which is why a settling disc thins to a shimmer rather
    than building. And the disc's own face turns slower as the rattle
    gets faster, heard as a wobble that drags while the pitch runs
    away.

    'spin' can only add energy: raise it and the disc leans that far,
    and everything after is loss. Hold it and the clatter holds at the
    pitch that level sets; let go and it settles from wherever it had
    got to. That is the point of patching movement here -- the tail is
    what the disc does after the hand has stopped, and a big gesture
    buys a long low one.

    'size' is the radius in metres (a coin turns ten times a second at
    full lean, a dinner plate four), 'settle' the seconds to flat,
    'rush' where in the tail the acceleration lives -- 0 spreads it
    evenly, 0.7 is rolling friction, 1 is Moffatt's viscous-air law,
    which holds still and then spends the last per cent of the settle
    on the whole scream.

    A rolling disc keeps its contact, and a settling coin hardly ever
    leaves the table, so this is one continuous sound modulated by the
    rotation rather than a series of blows. The rim passes under the
    contact at the precession rate, which rises, and that ripple in the
    load is the pitch; the disc's own weight comes round at the face
    rate, which FALLS, and that is the slow waver in intensity.

    'twist' is how much spin was in the fall, and it is the main thing.
    At 1 the coin was set true on its edge: it rolls, the contact
    drifts round the rim regularly, nothing leaves the table. At 0 it
    was simply pushed over: it never rolls at all, the lean swings past
    level every cycle, the face slaps, and it rattles to a stop. That
    works by nutation -- the lean oscillating instead of falling
    smoothly -- so a bad cast warbles the PITCH as well as the load,
    and since spin is what holds a disc in steady precession, the swing
    dies away in proportion to the twist it was given.

    'wobble' is the coin's own trueness, which matters but matters
    less: the swing in how hard it presses, several times its weight at
    full. It buys the tone (a true disc has no ripple to hear) and the
    flop. That flop is not an impact; the contact stays down carrying
    far more weight for a moment, and since a loaded contact engages
    more surface and stiffens, the grinding grows faster than the load
    and brightens as it grows. 'hardness' is how sharply it answers
    that load -- coin on stone cuts, something soft merely leans.

    'scrape' is the roughness under the contact, 'polish' how near flat
    it gets before it lands -- how high the whir climbs, and whether
    the end is a clack or a vanishing.

    Five outlets, because the dynamics are worth more than the sound.
    'out' is everything, 'grind' the rolling, which is nearly all of
    it, and 'landing' the single impact at the end, alone, so it can
    have its own resonator and gain. 'rate' is the precession frequency
    in Hz, rising, and 'face' the disc's profile coming round, falling
    -- two counter-moving controls out of one gesture, for a pitch, a
    cutoff, or rub~'s velocity.

    For a coin, into modal~ or drum~ on the 'plate' table. A coin's
    modes are up at two or three kilohertz and a bank that high rings
    small, so bring modal~'s drive and level up -- a real coin is not
    loud. spin~ <radius>, coin~ too.
    """

    @staticmethod
    def factory(name, data, args=None):
        return SpinNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = SpinUnit(synth_graph.sample_rate)

        if args is not None:
            for arg in args:
                try:
                    self.unit.size_in.base = max(0.004, min(0.6, float(arg)))
                except (ValueError, TypeError):
                    continue

        spin_port = self.add_modulation_input('spin', self.unit.spin_in,
                                              minimum=0.0, maximum=1.0,
                                              speed=0.01)
        if spin_port.widget is not None:
            spin_port.widget.set_tooltip(
                'the lean, and it only ever adds: hold it for the sound, '
                'release it for the tail. THE control -- map a movement here')
        size_port = self.add_modulation_input(
            'size', self.unit.size_in,
            default_value=self.unit.size_in.base,
            minimum=0.004, maximum=0.6, speed=0.002, slider=False)
        self.make_drag_proportional(size_port)
        if size_port.widget is not None:
            size_port.widget.set_tooltip(
                'the disc\'s radius in metres: it sets every rate. 0.012 is '
                'a coin, 0.12 a dinner plate')
        settle_port = self.add_modulation_input(
            'settle', self.unit.settle_in,
            default_value=self.unit.settle_in.base,
            minimum=0.05, maximum=60.0, speed=0.01, slider=False)
        self.make_drag_proportional(settle_port)
        if settle_port.widget is not None:
            settle_port.widget.set_tooltip(
                'seconds from full lean to flat: the length of the tail')
        rush_port = self.add_modulation_input('rush', self.unit.rush_in,
                                              minimum=0.0, maximum=1.0,
                                              speed=0.01)
        if rush_port.widget is not None:
            rush_port.widget.set_tooltip(
                'which loss dominates, so where in the tail the acceleration '
                'lives: 0 spreads it evenly, 0.7 is rolling friction, 1 is '
                'the viscous-air law -- still, then all of it at once')
        twist_port = self.add_modulation_input('twist', self.unit.twist_in,
                                               minimum=0.0, maximum=1.0,
                                               speed=0.01)
        if twist_port.widget is not None:
            twist_port.widget.set_tooltip(
                'how much spin was in the fall. 1 is a coin set true on '
                'its edge -- it rolls, and the contact drifts round in a '
                'regular way. 0 is one simply pushed over: it never rolls '
                'at all, the lean swings past level every cycle, and it '
                'rattles to a stop. Everything between is a real throw')
        wobble_port = self.add_modulation_input('wobble', self.unit.wobble_in,
                                                minimum=0.0, maximum=1.0,
                                                speed=0.01)
        if wobble_port.widget is not None:
            wobble_port.widget.set_tooltip(
                'how far off centre the disc is -- the rim once round. '
                'It swells the load on the face\'s own slowing turn, up '
                'to several times the disc\'s weight, and buys the tone '
                'AND the flop. A true disc has no ripple to hear. This '
                'is a SHAPE, so it swells; it does not chatter')
        profile_port = self.add_modulation_input(
            'profile', self.unit.profile_in, minimum=0.0, maximum=1.0,
            speed=0.01)
        if profile_port.widget is not None:
            profile_port.widget.set_tooltip(
                'the state of the EDGE -- nicks, burrs, a milled rim worn '
                'unevenly. A different fault from being off centre and it '
                'sounds different: where wobble swells, this makes the '
                'contact JUMP, throwing the disc clear of the table and '
                'back once a turn. It rides the traverse rate SQUARED, so '
                'it grows as the contact races round the rim')
        scrape_port = self.add_modulation_input('scrape', self.unit.scrape_in,
                                                minimum=0.0, maximum=1.0,
                                                speed=0.01)
        if scrape_port.widget is not None:
            scrape_port.widget.set_tooltip(
                'the roughness under the contact, rising with the speed it '
                'travels: the body of the whir')
        hard_port = self.add_modulation_input('hardness',
                                              self.unit.hardness_in,
                                              minimum=0.0, maximum=1.0,
                                              speed=0.01)
        if hard_port.widget is not None:
            hard_port.widget.set_tooltip(
                'how sharply the grinding answers a load: high and each '
                'flop cuts, low and it merely swells. This is what makes '
                'a flop sound like a flop')
        polish_port = self.add_modulation_input('polish', self.unit.polish_in,
                                                minimum=0.0, maximum=1.0,
                                                speed=0.01)
        if polish_port.widget is not None:
            polish_port.widget.set_tooltip(
                'how near flat it gets before it lands -- how high the whir '
                'climbs, and whether the end is a clack or a vanishing')
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self.spin_mode_input = self.add_input(
            'spin mode', widget_type='combo',
            default_value=SpinUnit.SPIN_MODES[0],
            callback=self.parameters_changed)
        self.spin_mode_input.widget.combo_items = list(SpinUnit.SPIN_MODES)
        if self.spin_mode_input.widget is not None:
            self.spin_mode_input.widget.set_tooltip(
                "how 'spin' is read. 'throw' works by CHANGE -- rising "
                "injects energy, falling drains it, and holding still "
                "does nothing at all, so a gesture throws a coin. "
                "'hold' works by LEVEL -- the gesture is the lean it "
                "asks for, and sustained motion keeps the coin going")

        self.model_input = self.add_input('model', widget_type='combo',
                                          default_value=SpinUnit.MODELS[1],
                                          callback=self.parameters_changed)
        self.model_input.widget.combo_items = list(SpinUnit.MODELS)
        if self.model_input.widget is not None:
            self.model_input.widget.set_tooltip(
                "'derived' integrates the disc's own equations of motion "
                "and reads the sound off it; 'voiced' is the earlier "
                "model, assembled behaviour by behaviour and fitted by "
                "ear. Note that 'wobble' has no meaning under 'derived' "
                "-- those equations describe a perfectly uniform disc, "
                "so all of the roughness comes from the cast")

        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.grind_output = self.add_signal_output('grind', self.unit.grind)
        self.landing_output = self.add_signal_output('landing',
                                                     self.unit.landing)
        self.rate_output = self.add_signal_output('rate', self.unit.rate)
        self.face_output = self.add_signal_output('face', self.unit.face)
        self.add_switch()
        self.finish_synth_node()

    def sync_options(self):
        chosen = any_to_string(self.model_input())
        if chosen in SpinUnit.MODELS:
            self.unit.model = SpinUnit.MODELS.index(chosen)
        mode = any_to_string(self.spin_mode_input())
        if mode in SpinUnit.SPIN_MODES:
            self.unit.spin_mode = SpinUnit.SPIN_MODES.index(mode)


class BubblesNode(SynthNode):
    """Liquid: the Minnaert chorus, played by flow.

    Each bubble is a decaying sine at the pitch its size dictates,
    rising as it dies -- the inflection that makes water sound like
    water. 'flow' is the whole interface: rate rides it, stillness is
    silent. 'size' runs fizz to glug, 'spread' widens the population,
    'chirp' is the upward inflection (0 is submerged pings), 'gulp'
    the low mouth-cavity breath under each birth, 'regular' the
    timing from boil to dumped-bottle metronome, 'density' arrivals
    per second at full flow. Layer noise~ underneath for the splash.

    bubbles~ <size>, e.g. bubbles~ 0.8. gurgle~ is the same node.
    """

    @staticmethod
    def factory(name, data, args=None):
        return BubblesNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = BubblesUnit(synth_graph.sample_rate)

        if args is not None:
            for arg in args:
                try:
                    self.unit.size_in.base = max(0.0, min(1.0, float(arg)))
                except (ValueError, TypeError):
                    continue

        flow_port = self.add_modulation_input('flow', self.unit.flow_in,
                                              minimum=0.0, maximum=1.5,
                                              speed=0.01)
        if flow_port.widget is not None:
            flow_port.widget.set_tooltip(
                'the stream: bubble arrivals ride it, stillness is '
                'silent. An effort stream belongs here')
        self.add_modulation_input('size', self.unit.size_in,
                                  default_value=self.unit.size_in.base,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('spread', self.unit.spread_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        chirp_port = self.add_modulation_input('chirp', self.unit.chirp_in,
                                               minimum=0.0, maximum=1.0,
                                               speed=0.01)
        if chirp_port.widget is not None:
            chirp_port.widget.set_tooltip(
                'the upward inflection of each bubble as it dies: 0 is '
                'pings deep under the surface, up is the gurgle')
        gulp_port = self.add_modulation_input('gulp', self.unit.gulp_in,
                                              minimum=0.0, maximum=1.0,
                                              speed=0.01)
        if gulp_port.widget is not None:
            gulp_port.widget.set_tooltip(
                'the onset twin of chirp: a deeper, quickly-dying partial '
                'under each bubble\'s birth -- the mouth-cavity breath of '
                'a glug. Timing is untouched')
        bloom_port = self.add_modulation_input(
            'bloom', self.unit.bloom_in,
            minimum=0.0, maximum=1.0, speed=0.01)
        if bloom_port.widget is not None:
            bloom_port.widget.set_tooltip(
                'the glug\'s glide-and-swell clock: snappy blip at 0, '
                'lazy resonant cavity at 1')
        regular_port = self.add_modulation_input(
            'regular', self.unit.regular_in,
            minimum=0.0, maximum=1.0, speed=0.01)
        if regular_port.widget is not None:
            regular_port.widget.set_tooltip(
                'arrival timing from fully random (a boil) to metronomic '
                '(a dumped bottle), same rate throughout')
        decay_port = self.add_modulation_input(
            'decay', self.unit.decay_in,
            minimum=0.0, maximum=1.0, speed=0.01)
        if decay_port.widget is not None:
            decay_port.widget.set_tooltip(
                'each bubble\'s ring: a quarter of the physical length '
                'to four times it -- dry drip to droplet in a cave')
        self.make_drag_proportional(
            self.add_modulation_input('density', self.unit.density_in,
                                      default_value=80.0,
                                      minimum=5.0, maximum=400.0,
                                      speed=0.5, slider=False))
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.add_switch()
        self.finish_synth_node()


class MotorNode(SynthNode):
    """A machine: speed and load as two effort streams.

    The mapping is the physics: 'speed' is rotation -- pitch linear in
    it, loudness rising, stillness silent -- and 'load' is torque:
    each firing punchier and less regular, the bearing grind rising
    underneath. Velocity into one, torque into the other, and a joint
    is an engine.

    'rate' is full-speed rotation in Hz; 'parts' the firings per
    revolution (one thumps, four is an engine, eight whines); 'tone'
    widens the firing from knock to electric hum; 'throb' spreads the
    parts' fixed strengths into a once-per-revolution lope -- the
    idle shudder; 'grind' the bearings; 'housing' a fixed pair of
    body resonances. For a specific machine, patch through modal~ or
    formant~ instead.

    motor~ <rate>, e.g. motor~ 30. engine~ is the same node.
    """

    @staticmethod
    def factory(name, data, args=None):
        return MotorNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = MotorUnit(synth_graph.sample_rate)

        if args is not None:
            for arg in args:
                try:
                    self.unit.rate_in.base = max(2.0, min(200.0,
                                                          float(arg)))
                except (ValueError, TypeError):
                    continue

        speed_port = self.add_modulation_input('speed', self.unit.speed_in,
                                               minimum=0.0, maximum=1.5,
                                               speed=0.01)
        if speed_port.widget is not None:
            speed_port.widget.set_tooltip(
                'rotation: pitch linear in it, loudness rising with it, '
                'stillness silent. A velocity stream belongs here')
        load_port = self.add_modulation_input('load', self.unit.load_in,
                                              minimum=0.0, maximum=1.0,
                                              speed=0.01)
        if load_port.widget is not None:
            load_port.widget.set_tooltip(
                'torque: punchier, rougher firings and rising bearing '
                'grind. An effort or torque stream belongs here')
        rate_port = self.add_modulation_input(
            'rate', self.unit.rate_in,
            default_value=self.unit.rate_in.base,
            minimum=2.0, maximum=200.0, speed=0.2, slider=False)
        self.make_drag_proportional(rate_port)
        self.add_modulation_input('parts', self.unit.parts_in,
                                  minimum=1.0, maximum=12.0, speed=0.05)
        self.add_modulation_input('tone', self.unit.tone_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        throb_port = self.add_modulation_input('throb', self.unit.throb_in,
                                               minimum=0.0, maximum=1.0,
                                               speed=0.01)
        if throb_port.widget is not None:
            throb_port.widget.set_tooltip(
                'uneven firing: this motor\'s own cylinders, loping at '
                'once per revolution. The idle shudder')
        beat_port = self.add_modulation_input('beat', self.unit.beat_in,
                                              minimum=0.0, maximum=1.0,
                                              speed=0.01)
        if beat_port.widget is not None:
            beat_port.widget.set_tooltip(
                'depth of the slip beat: a second shaft slightly behind, '
                'beating at slip times rotation -- the slow breathing '
                'speeds up and slows down with the machine')
        slip_port = self.add_modulation_input('slip', self.unit.slip_in,
                                              minimum=0.0, maximum=1.0,
                                              speed=0.01)
        if slip_port.widget is not None:
            slip_port.widget.set_tooltip(
                'how far behind the second shaft runs: half a percent to '
                'eight percent of the rotation, exponentially -- the '
                'beat from near-still to seasick, always scaling with '
                'speed')
        self.add_modulation_input('grind', self.unit.grind_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('housing', self.unit.housing_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.add_switch()
        self.finish_synth_node()


# What the kinds carry is a drum, not just a tuning: head frequency and
# ring, how much the pitch rides the hit, whether wires are on the far
# side, and which table the head is.
DRUM_KINDS = {
    'kick':  {'frequency': 55.0, 'decay': 0.35, 'tension': 0.5,
              'snares': 0.0, 'hardness': 0.35, 'position': 0.5,
              'material': 'membrane'},
    'snare': {'frequency': 185.0, 'decay': 0.18, 'tension': 0.15,
              'snares': 0.85, 'hardness': 0.8, 'position': 0.35,
              'material': 'membrane'},
    'tom':   {'frequency': 110.0, 'decay': 0.5, 'tension': 0.45,
              'snares': 0.0, 'hardness': 0.55, 'position': 0.35,
              'material': 'membrane'},
    'tabla': {'frequency': 170.0, 'decay': 0.7, 'tension': 0.9,
              'snares': 0.0, 'hardness': 0.75, 'position': 0.2,
              'material': 'tabla'},
    'frame': {'frequency': 90.0, 'decay': 0.9, 'tension': 0.25,
              'snares': 0.0, 'hardness': 0.4, 'position': 0.45,
              'material': 'membrane'},
}


class DrumNode(ModeTableNode):
    """A drum: the membrane bank plus the physics modal~ leaves out.

    A hard hit lands pitched sharp and bends down through its ring --
    'tension', where tabla and toms live -- and 'snares' are wires
    shaken by the head itself, rattling while the drum speaks and
    dying with it. 'hit' is the mallet (trigger height is velocity);
    'excite in' hears audio, which is where bounce~ goes, and a roll
    is bounce~ pressed into the head. The head is a mode table in the
    same editor as modal~.

    drum~ <kind>, e.g. drum~ snare. Kinds load the knobs and let go.
    """

    SAVE_KEY = 'drum_modes'

    @staticmethod
    def factory(name, data, args=None):
        return DrumNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = DrumUnit(synth_graph.sample_rate)

        kind = 'tom'
        if args is not None:
            for arg in args:
                if arg in DRUM_KINDS:
                    kind = arg
        self._kind_shown = kind
        preset = DRUM_KINDS[kind]
        material = preset['material']
        self.unit.frequency_in.base = preset['frequency']
        self.unit.set_modes(MODAL_MATERIALS[material])
        self._init_mode_editor(label, material)

        self.add_signal_input('excite in', self.unit.excite_in)
        sense_port = self.add_modulation_input(
            'sensitivity', self.unit.sensitivity_in,
            default_value=self.unit.sensitivity_in.base,
            minimum=0.0, maximum=8.0, speed=0.01, slider=False)
        self.make_drag_proportional(sense_port)
        if sense_port.widget is not None:
            sense_port.widget.set_tooltip(
                'how keenly the head hears what is patched to the excite '
                'inlet above. Unity leaves it as it always was; it reaches '
                'past that for excitations too sparse to speak')
        self.add_trigger_signal_input('hit', self.unit.trigger_in,
                                      self.hit)
        self.add_modulation_input('frequency', self.unit.frequency_in,
                                  default_value=preset['frequency'],
                                  minimum=20.0, speed=1.0)
        self.add_modulation_input('pitch', self.unit.pitch_in, speed=0.01)
        self.make_drag_proportional(
            self.add_modulation_input('decay', self.unit.decay_in,
                                      default_value=preset['decay'],
                                      minimum=0.01, maximum=30.0,
                                      speed=0.05, slider=False))
        self.add_modulation_input('hardness', self.unit.hardness_in,
                                  default_value=preset['hardness'],
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('position', self.unit.position_in,
                                  default_value=preset['position'],
                                  minimum=0.0, maximum=1.0, speed=0.01)
        tension_port = self.add_modulation_input(
            'tension', self.unit.tension_in,
            default_value=preset['tension'],
            minimum=0.0, maximum=1.0, speed=0.01)
        if tension_port.widget is not None:
            tension_port.widget.set_tooltip(
                'the head stiffens as it moves: a hard hit lands sharp '
                'and bends down through its ring. Tabla at the top')
        snares_port = self.add_modulation_input(
            'snares', self.unit.snares_in,
            default_value=preset['snares'],
            minimum=0.0, maximum=1.0, speed=0.01)
        if snares_port.widget is not None:
            snares_port.widget.set_tooltip(
                'wires shaken by the head itself: they rattle while the '
                'drum speaks and die with it')
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self.kind_input = self.add_input('kind', widget_type='combo',
                                         default_value=kind,
                                         callback=self.kind_changed)
        self.kind_input.widget.combo_items = list(DRUM_KINDS)

        self._add_mode_table_ports(material)
        self._add_mode_table_options()

        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.modes_output = self.add_output('modes out')
        self.add_switch()
        self.finish_synth_node()

    def hit(self):
        self.unit.fire()

    def custom_create(self, from_file):
        super().custom_create(from_file)
        # A node made by hand starts as its kind; a loaded one keeps the
        # knobs the patch saved.
        if not from_file:
            self.apply_kind(self._kind_shown)

    def kind_changed(self):
        chosen = any_to_string(self.kind_input())
        if chosen == self._kind_shown:
            return
        self._kind_shown = chosen
        if chosen not in DRUM_KINDS or self.in_loading_process:
            return
        self.apply_kind(chosen)

    def apply_kind(self, name):
        recipe = DRUM_KINDS.get(name)
        if recipe is None:
            return
        for port in self.inputs:
            label = port.get_label()
            if label in recipe and port.widget is not None:
                port.widget.set(recipe[label])
        self.parameters_changed()
        material = recipe['material']
        if any_to_string(self.material_input()) != material:
            self.material_input.widget.set(material)
            self.material_changed()


# A regime is the statistics of release: how much motion builds a
# threshold's worth of stress, how the sizes are distributed, how sharply
# the material remembers where it has already been bent. The values that
# go to knobs (chirp, and a suggested body) are set through the widgets so
# they stay the user's after; the rest are the physics of the regime.
STRAIN_REGIMES = {
    'creak':   {'thresh': 0.004, 'spread': 0.5, 'alpha': 0.0, 'cap': 2.0,
                'habituate': 0.6, 'grain': 0.003, 'amp': 0.25,
                'chirp': 0.0, 'decay': 0.3, 'stretch': 0.3, 'squeal': 0.55, 'vary': 0.15, 'grind': 0.4, 'texture': 0.45},
    'crumple': {'thresh': 0.006, 'spread': 1.0, 'alpha': 0.6, 'cap': 15.0,
                'habituate': 0.25, 'grain': 0.0012, 'amp': 0.15,
                'chirp': 0.1, 'decay': 0.15, 'stretch': 0.1, 'squeal': 0.05, 'vary': 0.3, 'grind': 0.25, 'texture': 0.7},
    'crack':   {'thresh': 0.03, 'spread': 1.5, 'alpha': 0.8, 'cap': 40.0,
                'habituate': 0.1, 'grain': 0.0008, 'amp': 0.1,
                'chirp': 0.6, 'decay': 1.5, 'stretch': 0.2, 'squeal': 0.15, 'vary': 0.1, 'grind': 0.1, 'texture': 0.35},
}


class WhooshNode(SynthNode):
    """Motion through air: patch a speed, hear the swish.

    Pitch is speed over size -- the physics of vortex shedding, not a
    mapping anyone chose -- and loudness rises steeply the way aeolian
    sound does, so slow motion whispers, fast motion roars, and
    stillness is silent. 'size' runs thin edge (sings high) to thick
    limb (rumbles); 'edge' is how bladelike the shedding is; 'wake'
    mixes the broadband hiss of stirred air.

    whoosh~ <size>.
    """

    @staticmethod
    def factory(name, data, args=None):
        return WhooshNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = WhooshUnit(synth_graph.sample_rate)

        if args is not None:
            for arg in args:
                try:
                    self.unit.size_in.base = max(0.0, min(1.0, float(arg)))
                except (ValueError, TypeError):
                    continue

        self.add_modulation_input('speed', self.unit.speed_in,
                                  minimum=0.0, maximum=1.5, speed=0.01)
        self.add_modulation_input('size', self.unit.size_in,
                                  default_value=self.unit.size_in.base,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('edge', self.unit.edge_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('wake', self.unit.wake_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.add_switch()
        self.finish_synth_node()


class NoiseNode(SynthNode):
    """A leak: the rack's noise source, played by pressure.

    'pressure' is the whole interface -- loudness rises steeply with it
    as turbulence does, brightness gently, stillness is silence -- and
    a control-rate effort stream drives it without zippering. 'color'
    tilts dark rumble to full white at constant loudness. 'sputter'
    breaks the flow up physically: a blockage that builds pressure
    while closed and spits on reopening -- flutter at low values, hard
    dropouts at full -- with 'rate' its tempo, slow gulps to buzz.

    Through formant~ it is breath, through modal~ a rattling surface,
    through vcf~ classic subtractive hiss.

    noise~ <color>, e.g. noise~ 0.5. hiss~ is the same node.
    """

    @staticmethod
    def factory(name, data, args=None):
        return NoiseNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = NoiseUnit(synth_graph.sample_rate)

        if args is not None:
            for arg in args:
                try:
                    self.unit.color_in.base = max(0.0, min(1.0, float(arg)))
                except (ValueError, TypeError):
                    continue

        press_port = self.add_modulation_input('pressure',
                                               self.unit.pressure_in,
                                               default_value=1.0,
                                               minimum=0.0, maximum=2.0,
                                               speed=0.01)
        if press_port.widget is not None:
            press_port.widget.set_tooltip(
                'how hard the leak is driven: loudness rises steeply, '
                'brightness gently. 0 is silence -- patch an effort '
                'stream here and stillness stays still')
        self.add_modulation_input('color', self.unit.color_in,
                                  default_value=self.unit.color_in.base,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        sputter_port = self.add_modulation_input('sputter',
                                                 self.unit.sputter_in,
                                                 minimum=0.0, maximum=1.0,
                                                 speed=0.01)
        if sputter_port.widget is not None:
            sputter_port.widget.set_tooltip(
                'the flow breaking up: a blockage that builds pressure '
                'and spits on reopening. Partial flutter low, hard '
                'dropouts at full')
        aperture_port = self.add_modulation_input(
            'aperture', self.unit.aperture_in,
            minimum=0.0, maximum=1.0, speed=0.01)
        if aperture_port.widget is not None:
            aperture_port.widget.set_tooltip(
                'the hole the jet sings through: a pinhole whistles high '
                'and tight, a wide gap breathes low and broad. Rises a '
                'little with pressure')
        whistle_port = self.add_modulation_input(
            'whistle', self.unit.whistle_in,
            minimum=0.0, maximum=1.0, speed=0.01)
        if whistle_port.widget is not None:
            whistle_port.widget.set_tooltip(
                'tightness of the jet resonance: breathy at 0, piercing '
                'near-pure at 1')
        rate_port = self.add_modulation_input('rate', self.unit.rate_in,
                                              minimum=0.2, maximum=200.0,
                                              speed=0.1, slider=False)
        self.make_drag_proportional(rate_port)
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.add_switch()
        self.finish_synth_node()


class StrainNode(ModeTableNode):
    """Solids under stress: bending made audible.

    The first node whose input is effort itself: patch a joint angle, a
    stretch, a slow fader into 'strain' and the model runs on it --
    motion releases events, stillness is silent by construction, and the
    material remembers where it has been bent (repeat movements quiet
    down; rest restores them over tens of seconds). 'regime' is what the
    material does under stress -- a hinge creaks every time, paper is
    loud only in new territory, ice cracks rarely and hugely -- and the
    body it rings is a mode table, drawn in the same editor as modal~.
    'resist' runs tissue paper to oak door; 'chirp' disperses each event
    the way lake ice does.

    strain~ <frequency> <regime> <material>, e.g. strain~ 300 crack ice.
    """

    SAVE_KEY = 'strain_modes'

    @staticmethod
    def factory(name, data, args=None):
        return StrainNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = StrainUnit(synth_graph.sample_rate)

        frequency = 700.0
        regime = 'creak'
        material = 'wood'
        if args is not None:
            for arg in args:
                if arg in STRAIN_REGIMES:
                    regime = arg
                elif arg in MODAL_MATERIALS:
                    material = arg
                else:
                    try:
                        frequency = float(arg)
                    except (ValueError, TypeError):
                        continue
        self.unit.frequency_in.base = frequency
        self.unit.set_modes(MODAL_MATERIALS[material])
        self._init_mode_editor(label, material)
        self._regime_shown = regime
        self.apply_regime_constants(regime)

        self.add_modulation_input('strain', self.unit.strain_in,
                                  minimum=0.0, maximum=1.0, speed=0.002)
        self.add_modulation_input('resist', self.unit.resist_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        stretch_port = self.add_modulation_input(
            'stretch', self.unit.stretch_in,
            default_value=STRAIN_REGIMES[regime]['stretch'],
            minimum=-1.0, maximum=1.0, speed=0.005)
        if stretch_port.widget is not None:
            stretch_port.widget.set_tooltip(
                'how the body stiffens under load: its resonances climb '
                '(or fall, negative) with the strain, up to an octave -- '
                'the rings bend as the bending continues')
        squeal_port = self.add_modulation_input(
            'squeal', self.unit.squeal_in,
            default_value=STRAIN_REGIMES[regime]['squeal'],
            minimum=0.0, maximum=1.0, speed=0.01)
        if squeal_port.widget is not None:
            squeal_port.widget.set_tooltip(
                'how firmly the sliding interface engages the body: the '
                'top third sings, locking onto a mode and jumping like '
                'brass; full leans hard enough to groan. It only sounds '
                'while actually sliding, and only at hinge speeds -- too '
                'slow judders, too fast breaks up, like a real door')
        grind_port = self.add_modulation_input(
            'grind', self.unit.grind_in,
            default_value=STRAIN_REGIMES[regime]['grind'],
            minimum=0.0, maximum=1.0, speed=0.01)
        if grind_port.widget is not None:
            grind_port.widget.set_tooltip(
                'continuous frictional shear between the slips, riding '
                'speed and load: what breath is to the winds, the scrub '
                'of surfaces is to a bend')
        texture_port = self.add_modulation_input(
            'texture', self.unit.texture_in,
            default_value=STRAIN_REGIMES[regime]['texture'],
            minimum=0.0, maximum=1.0, speed=0.01)
        if texture_port.widget is not None:
            texture_port.widget.set_tooltip(
                'where the grind sits spectrally: fine surfaces rub dark, '
                'coarse ones bright -- character, not level')
        vary_port = self.add_modulation_input(
            'vary', self.unit.vary_in,
            default_value=STRAIN_REGIMES[regime]['vary'],
            minimum=0.0, maximum=1.0, speed=0.01)
        if vary_port.widget is not None:
            vary_port.widget.set_tooltip(
                'ensemble: each release retunes the body up to half an '
                'octave -- a walk crosses many boards, not one. Best with '
                'short decay; with long, the ringing tails gliss')
        self.chirp_input = self.add_modulation_input(
            'chirp', self.unit.chirp_in,
            default_value=STRAIN_REGIMES[regime]['chirp'],
            minimum=0.0, maximum=1.0, speed=0.01)
        if self.chirp_input.widget is not None:
            self.chirp_input.widget.set_tooltip(
                'dispersion: each event arrives high-first and sweeps '
                'down, longer with more chirp -- the fracture is farther '
                'away. Lake ice at 0.6 and up')
        pops_port = self.add_modulation_input(
            'pops', self.unit.pops_in, default_value=0.5,
            minimum=0.0, maximum=1.0, speed=0.01)
        if pops_port.widget is not None:
            pops_port.widget.set_tooltip(
                'how often the geometry lets go, exponentially: 0 never, '
                '0.5 occasional, 1 a rolling grumble of releases. Each '
                'pop also jolts the squeal')
        self.add_modulation_input('frequency', self.unit.frequency_in,
                                  default_value=frequency, minimum=20.0,
                                  speed=1.0)
        self.add_modulation_input('pitch', self.unit.pitch_in, speed=0.01)
        self.make_drag_proportional(
            self.add_modulation_input('decay', self.unit.decay_in,
                                      default_value=STRAIN_REGIMES[regime]['decay'],
                                      minimum=0.01, maximum=60.0, speed=0.05,
                                      slider=False))
        dry_port = self.add_modulation_input('dry', self.unit.dry_in,
                                             minimum=0.0, maximum=1.0,
                                             speed=0.01)
        if dry_port.widget is not None:
            dry_port.widget.set_tooltip(
                'the raw slips and scrapes alongside the body: what keeps '
                'a strain from being only its resonance')
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self.regime_input = self.add_input('regime', widget_type='combo',
                                           default_value=regime,
                                           callback=self.regime_changed)
        self.regime_input.widget.combo_items = list(STRAIN_REGIMES)

        self._add_mode_table_ports(material)
        self._add_mode_table_options()

        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.grains_output = self.add_signal_output('grains out',
                                                    self.unit.grains)
        self.modes_output = self.add_output('modes out')
        self.add_switch()
        self.finish_synth_node()

    def apply_regime_constants(self, name):
        p = STRAIN_REGIMES[name]
        self.unit.thresh = p['thresh']
        self.unit.spread = p['spread']
        self.unit.alpha = p['alpha']
        self.unit.size_cap = p['cap']
        self.unit.habituate = p['habituate']
        self.unit.grain_seconds = p['grain']
        self.unit.amp = p['amp']

    def regime_changed(self):
        chosen = any_to_string(self.regime_input())
        if chosen == self._regime_shown or chosen not in STRAIN_REGIMES:
            return
        self._regime_shown = chosen
        self.apply_regime_constants(chosen)
        # The knobs a regime suggests are set through their widgets, so
        # they stay the user's afterwards -- and stay put during a load.
        if not self.in_loading_process:
            p = STRAIN_REGIMES[chosen]
            if self.chirp_input.widget is not None:
                self.chirp_input.widget.set(p['chirp'])
            for port in self.inputs:
                name = port.get_label()
                if name in ('decay', 'stretch', 'squeal', 'vary',
                            'grind', 'texture') and port.widget is not None:
                    port.widget.set(p[name])
        self.parameters_changed()

    def update_parameters_from_widgets(self):
        # Restore the regime's physics before the knobs land on top.
        chosen = any_to_string(self.regime_input())
        if chosen in STRAIN_REGIMES:
            self._regime_shown = chosen
            self.apply_regime_constants(chosen)
        super().update_parameters_from_widgets()


class WindNode(SynthNode):
    """Blown instrument -- no trigger, only breath.

    Everything about playing it lives in the pressure inlet: lean on the
    slider, patch an adsr~ for tongued notes, an lfo~ for vibrato, or an
    effort stream so that moving hard is blowing hard. The reed speaks from
    about half pressure; the flute wants nearly a full breath and cracks when
    pushed past it, which is the model being honest about flutes.

    wind~ <frequency> <model>; reed~ and flute~ are the same node with its
    model preset.
    """

    @staticmethod
    def factory(name, data, args=None):
        return WindNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = WindUnit(synth_graph.sample_rate)

        frequency = 220.0
        model = 'flute' if label == 'flute~' else 'reed'
        if args is not None:
            for arg in args:
                if arg in WindUnit.MODES:
                    model = arg
                else:
                    try:
                        frequency = float(arg)
                    except (ValueError, TypeError):
                        continue
        self.unit.frequency_in.base = frequency
        self.unit.mode = WindUnit.MODES.index(model)

        self.add_modulation_input('pressure', self.unit.pressure_in,
                                  minimum=0.0, maximum=1.5, speed=0.01)
        self.add_modulation_input('frequency', self.unit.frequency_in,
                                  default_value=frequency,
                                  minimum=WindUnit.MIN_FREQUENCY, speed=1.0)
        self.add_modulation_input('pitch', self.unit.pitch_in, speed=0.01)
        self.add_modulation_input('embouchure', self.unit.embouchure_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('brightness', self.unit.brightness_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        # Breath lives logarithmically: the audible difference is between
        # 0.02 and 0.06, not 0.6 and 0.9, so the drag is proportional.
        self.make_drag_proportional(
            self.add_modulation_input('breath', self.unit.noise_in,
                                      minimum=0.0, maximum=1.0,
                                      slider=False))
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self.model_input = self.add_input('model', widget_type='combo',
                                          default_value=model,
                                          callback=self.parameters_changed)
        self.model_input.widget.combo_items = list(WindUnit.MODES)

        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.add_switch()
        self.finish_synth_node()

    def sync_options(self):
        model = any_to_string(self.model_input())
        if model in WindUnit.MODES:
            self.unit.mode = WindUnit.MODES.index(model)


class BrassNode(SynthNode):
    """Brass: one bore, the note chosen by the lip.

    'frequency' is the instrument's size -- its pedal fundamental -- and
    'lip' is the embouchure, mapped across the first sixteen harmonics:
    sweep it and the pitch climbs the series in steps, like a bugler,
    because the lip's own resonance locks to the nearest bore mode.
    'pressure' is the breath: a threshold to speak, dynamics above it,
    and pushed hard it cracks the lock and blats toward the pedal, as
    brass does. No trigger: tonguing is an adsr~ on pressure.

    brass~ <frequency>.
    """

    @staticmethod
    def factory(name, data, args=None):
        return BrassNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = BrassUnit(synth_graph.sample_rate)

        frequency = 110.0
        if args is not None:
            for arg in args:
                try:
                    frequency = float(arg)
                except (ValueError, TypeError):
                    continue
        self.unit.frequency_in.base = frequency

        self.add_modulation_input('pressure', self.unit.pressure_in,
                                  minimum=0.0, maximum=1.5, speed=0.01)
        lip_port = self.add_modulation_input('lip', self.unit.lip_in,
                                             minimum=0.0, maximum=1.0,
                                             speed=0.005)
        if lip_port.widget is not None:
            lip_port.widget.set_tooltip(
                'embouchure: tension across the first sixteen harmonics; '
                'the pitch climbs the series in steps -- the arpeggio '
                'below halfway, the near-scale of the high series above')
        self.add_modulation_input('frequency', self.unit.frequency_in,
                                  default_value=frequency,
                                  minimum=BrassUnit.MIN_FREQUENCY, speed=1.0)
        self.add_modulation_input('pitch', self.unit.pitch_in, speed=0.01)
        self.add_modulation_input('brightness', self.unit.brightness_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.make_drag_proportional(
            self.add_modulation_input('breath', self.unit.noise_in,
                                      minimum=0.0, maximum=1.0,
                                      slider=False))
        mute_port = self.add_modulation_input('mute', self.unit.mute_in,
                                              minimum=0.0, maximum=1.0,
                                              speed=0.01)
        if mute_port.widget is not None:
            mute_port.widget.set_tooltip(
                'the harmon: a cork-sealed cavity in the bell. More back '
                'to the lips, the body stripped from what escapes. 0 is '
                'the open horn, exactly')
        stem_port = self.add_modulation_input('stem', self.unit.stem_in,
                                              minimum=0.0, maximum=1.0,
                                              speed=0.01)
        if stem_port.widget is not None:
            stem_port.widget.set_tooltip(
                'the harmon stem: out (0) is the bare small hole, thin '
                'and buzzy; in (1) is the tube the hand plays -- wah '
                'works on this')
        wah_port = self.add_modulation_input('wah', self.unit.wah_in,
                                             minimum=0.0, maximum=1.0,
                                             speed=0.01)
        if wah_port.widget is not None:
            wah_port.widget.set_tooltip(
                'the hand over the harmon stem: the mute cavity sweeps '
                'dark to bright. Only audible with mute up -- patch a '
                'slow stream here and the horn talks')
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.add_switch()
        self.finish_synth_node()


class BowNode(SynthNode):
    """Bowed string: velocity and force are the whole bow arm.

    The two sliders are mapped so their middles bow cleanly at any pitch;
    the misbehavior at the edges -- octave whistle from a fast light bow,
    subharmonic scratch from a slow heavy one -- is the model's own, in the
    same directions as the instrument's. The output is the raw bridge wave:
    patch it into modal~ or formant~ to give it a body.

    bow~ <frequency>.
    """

    @staticmethod
    def factory(name, data, args=None):
        return BowNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = BowUnit(synth_graph.sample_rate)

        frequency = 220.0
        if args is not None:
            for arg in args:
                try:
                    frequency = float(arg)
                except (ValueError, TypeError):
                    continue
        self.unit.frequency_in.base = frequency

        self.add_modulation_input('velocity', self.unit.velocity_in,
                                  minimum=0.0, maximum=1.5, speed=0.01)
        self.add_modulation_input('force', self.unit.force_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('position', self.unit.position_in,
                                  minimum=0.05, maximum=0.4, speed=0.005)
        self.add_modulation_input('frequency', self.unit.frequency_in,
                                  default_value=frequency,
                                  minimum=BowUnit.MIN_FREQUENCY, speed=1.0)
        self.add_modulation_input('pitch', self.unit.pitch_in, speed=0.01)
        self.add_modulation_input('brightness', self.unit.brightness_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.add_switch()
        self.finish_synth_node()


# A kind is a starting point for the statistics: how many things collide,
# how long they keep moving, how sharp each tick is, and what they rattle
# inside. Values go to the knobs, so a kind is somewhere to start from,
# not a mode the node is in.
# resonance rides an exponential ring-time curve (3 ms thud at 0 to a
# tenth-of-a-second bell at 1); the jingled kinds live where the
# members truly ring past each other. Values here are David's, tuned
# by ear 2026-08 (cabasa is a sketch: chain beads scraping a ridged
# steel cylinder -- bright, dense, dead vessel).
SHAKER_KINDS = {
    'maraca':      {'density': 800.0, 'settle': 0.08, 'hardness': 0.294,
                    'vessel': 4270.0, 'resonance': 0.17, 'jingle': 0.118,
                    'vary': 0.56},
    'cabasa':      {'density': 1200.0, 'settle': 0.03, 'hardness': 0.85,
                    'vessel': 6000.0, 'resonance': 0.03, 'jingle': 0.4,
                    'vary': 0.3},
    'tambourine':  {'density': 144.0, 'settle': 0.03, 'hardness': 0.0,
                    'vessel': 5750.0, 'resonance': 0.7, 'jingle': 0.3,
                    'vary': 0.7},
    'sleighbells': {'density': 800.0, 'settle': 0.09, 'hardness': 0.6,
                    'vessel': 8000.0, 'resonance': 0.779, 'jingle': 1.0,
                    'vary': 0.838},
    'rain':        {'density': 2000.0, 'settle': 0.02, 'hardness': 1.0,
                    'vessel': 5000.0, 'resonance': 0.088, 'jingle': 1.0,
                    'vary': 1.0},
    'downpour':    {'density': 2000.0, 'settle': 0.02, 'hardness': 0.0,
                    'vessel': 5000.0, 'resonance': 0.088, 'jingle': 1.0,
                    'vary': 1.0},
    'gravel':      {'density': 90.0, 'settle': 0.08, 'hardness': 0.4,
                    'vessel': 900.0, 'resonance': 0.1, 'jingle': 0.1,
                    'vary': 0.6},
}


class RattleNode(SynthNode):
    """Loose things in a container, shaken and turned -- simulated.

    shaker~ is a collision RATE driven by an agitation, which is cheap
    and very good for rain and sleighbells where nobody wants to
    simulate a hundred thousand grains. This has particles instead:
    positions, velocities, and walls that hit them when the walls come
    to them.

    What that buys is that the gesture stops needing translating. Shake
    it along a line and swirl it in a circle and those are the same
    simulation given a line and a circle -- and the difference between
    them comes out on its own: more glancing contact, and an envelope
    that stops pulsing because a circle never stops the way a line does
    at each end. None of that is modelled here; it is what happens.

    'shake x/y/z' is where the container is being ACCELERATED, in
    gravities, and 'turn x/y/z' is how far it is TIPPED, in degrees --
    an angle, not a rate. A body's movement drives it
    directly. Turning matters three ways -- the centrifugal push
    outward, the Coriolis deflection of whatever is already moving, and
    the Euler shove when the turning itself changes -- and all three are
    what a swirl is made of.

    'shape' is sphere, box, egg or tube, and 'aspect' is how long it is
    against how wide. A tube is a cylinder -- curved round the barrel,
    flat at the ends -- so it glances one way and strikes the other,
    which no single-surface shape can do. Both are only the boundary test, so they are nearly free
    and change a great deal: flat walls take a bean head on where a
    curved one lets it glance, and a long container lets things travel
    its length where a flat one pins them between two close walls.

    'knock' and 'scrape' come out separately as well as mixed. Like
    bounce~, what comes out is a train of force pulses rather than a
    sound -- patch it into modal~, drum~ or resonator~.

    A thing resting against the wall is HELD while the slope under it is
    shallower than the friction can support, and slides or lets go when
    it is not -- so 'friction' and 'texture' are two different things.
    Friction is the coefficient, one number everywhere, and on a smooth
    shell a thing that starts sliding goes on sliding. Texture differs
    from place to place, so a thing catches, is carried, lets go, and
    catches again somewhere else. A slow turn on a smooth shell is a
    continuous slide; the same turn on a rough one is a rattle.

    Inter-particle collisions are left out on purpose: most of the cost,
    least of the sound. What is heard is the wall. The one case where
    they are missed is a handful driven along a single axis onto a flat
    wall -- with no bounce the grains are then EXACTLY identical, since
    they leave together carrying the wall's speed, fall in a field that
    is the same everywhere, and land dead. Raise 'variety', and give the
    wall some 'texture' for them to sit on unevenly: a rough wall is not
    a plane, so it throws each of them off at its own speed, and that is
    the only thing that parts them.
    """

    @staticmethod
    def factory(name, data, args=None):
        return RattleNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = RattleUnit(synth_graph.sample_rate)
        count = 48
        if args is not None:
            for arg in args:
                try:
                    count = int(arg)
                except (ValueError, TypeError):
                    continue
        self.unit.set_count(count)

        for axis in ('x', 'y', 'z'):
            port = self.add_modulation_input(
                f'shake {axis}', getattr(self.unit, f'shake_{axis}_in'),
                minimum=-8.0, maximum=8.0, speed=0.02)
            if port.widget is not None:
                port.widget.set_tooltip(
                    f'how hard the container is being accelerated along '
                    f'{axis}, in GRAVITIES -- 1.0 is its own weight, '
                    f'which is roughly where things start leaving the '
                    f'floor. Patch a body: this is what a body gives, '
                    f'and it needs no translating into how agitated '
                    f'anything is')
        for axis in ('x', 'y', 'z'):
            port = self.add_modulation_input(
                f'turn {axis}', getattr(self.unit, f'turn_{axis}_in'),
                minimum=-180.0, maximum=180.0, speed=0.5)
            if port.widget is not None:
                port.widget.set_tooltip(
                    f'how far it is TIPPED about {axis}, in degrees -- '
                    f'where it is pointing, not how fast it is going. '
                    f'Send 30 and it sits tipped at 30, and gravity '
                    f'sits tipped with it. Rock it with a sine and it '
                    f'rocks through that many degrees. For a continuous '
                    f'roll, send a ramp. How fast it is turning is '
                    f'taken from how fast the angle changes, and that '
                    f'is what throws the particles outward, deflects '
                    f'whatever is already moving, and shoves them when '
                    f'the turning changes -- which is what a swirl is, '
                    f'as against a shake')
        self.count_input = self.add_input(
            'count', widget_type='drag_int', default_value=count,
            callback=self.count_changed)
        if self.count_input.widget is not None:
            self.count_input.widget.set_tooltip(
                'how many things are in there. Changes the texture and '
                'not the level -- a handful is countable, hundreds is a '
                'wash')
        self.shape_input = self.add_input(
            'shape', widget_type='combo',
            default_value=RattleUnit.SHAPES[0],
            callback=self.parameters_changed)
        self.shape_input.widget.combo_items = list(RattleUnit.SHAPES)
        if self.shape_input.widget is not None:
            self.shape_input.widget.set_tooltip(
                'the container. Only a boundary test, so it costs almost '
                'nothing and changes a great deal: a box takes a bean '
                'head on where a sphere lets it glance, and an egg is '
                'the sphere with one axis stretched -- struck at its '
                'pointed end it answers at the angle that end really '
                'presents. A tube is a cylinder, curved round the '
                'barrel and flat at the two ends, so it does BOTH and '
                'which one you get depends on which way you shake it: '
                'along its length things are taken head on by the caps, '
                'across it they glance off the side. With a long '
                '"aspect" that is a rainstick or a tube shaker')
        aspect_port = self.add_modulation_input(
            'aspect', self.unit.aspect_in, minimum=0.2, maximum=5.0,
            speed=0.01)
        if aspect_port.widget is not None:
            aspect_port.widget.set_tooltip(
                'how long it is against how wide, along z. 1 is a ball '
                'or a cube; below that a slab, above it a tube. Like '
                'shape this is only the boundary test, so it costs '
                'nothing at all and changes a great deal: in a long one '
                'things travel the length and pile at whichever end is '
                'down, in a flat one they are pinned between two close '
                'walls and rattle far more often -- 112 contacts a '
                'second against 199 for the same handful. An egg keeps '
                'its own stretch on top of this. Lay it on its side '
                'with "turn" if you want the length horizontal')
        size_port = self.add_modulation_input(
            'size', self.unit.size_in, minimum=0.005, maximum=0.5,
            speed=0.001)
        if size_port.widget is not None:
            size_port.widget.set_tooltip(
                'how big the container is, as a half-width in METRES: '
                '0.04 is a maraca, 0.5 is a metre across. It sets the '
                'pitch of everything, because it sets how far things '
                'fall and how fast the wall sweeps past when it is '
                'turned')
        grain_port = self.add_modulation_input(
            'grain', self.unit.grain_in, minimum=0.0, maximum=0.6,
            speed=0.005)
        if grain_port.widget is not None:
            grain_port.widget.set_tooltip(
                'how big each thing in there is, as a FRACTION of the '
                'container -- so it stays a handful of beans whatever '
                'size the container is. 0.05 is a bean in a maraca; at '
                '0.16 a hundred and twenty-eight of them fill over half '
                'the shell, which is gravel in a bucket. It sets how '
                'far a thing rides between the bumps of the surface, so '
                'it sets how bright a rub is')
        # Small at one end and not at the other, and the useful part is
        # all down at the bottom -- so each pixel moves these by a
        # fraction of themselves. That is exponential travel while the
        # number shown stays a real size, which matters here because
        # both are modulation inlets: a hidden mapping would quietly
        # change what a patch cord into them means.
        self.make_drag_proportional(aspect_port, fraction=0.05,
                                    floor=0.005, ceiling=0.2)
        self.make_drag_proportional(size_port, fraction=0.05,
                                    floor=0.0002, ceiling=0.02)
        self.make_drag_proportional(grain_port, fraction=0.05,
                                    floor=0.0004, ceiling=0.02)
        self.add_modulation_input('bounce', self.unit.bounce_in,
                                  minimum=0.0, maximum=0.95, speed=0.01)
        grip_port = self.add_modulation_input(
            'friction', self.unit.friction_in, minimum=0.0, maximum=1.0,
            speed=0.01)
        if grip_port.widget is not None:
            grip_port.widget.set_tooltip(
                'how much purchase the shell has -- the friction '
                'coefficient itself, so more of it means more grip, not '
                'less. It sets the angle a resting thing is held at: '
                'below that slope it does not move at all, above it it '
                'slides. Turn a slippery container slowly and the '
                'contents stay put while the shell goes round under '
                'them, which is a continuous slide and no knocks. Grip '
                'it harder and they are carried further up before the '
                'slope stops holding them, and then they have further '
                'to fall')
        texture_port = self.add_modulation_input(
            'texture', self.unit.texture_in, minimum=0.0, maximum=1.0,
            speed=0.01)
        if texture_port.widget is not None:
            texture_port.widget.set_tooltip(
                'how rough the inside is. It resists like friction '
                'does -- riding over bumps costs something, so a rough '
                'shell holds and rasps even with the friction at zero '
                '-- but it differs from PLACE to place, where friction '
                'is one number everywhere. So a thing catches, is '
                'carried until that place no longer holds it, lets go, '
                'drops to the next and catches again, and each letting '
                'go is a tick. That is the whole difference between a '
                'hiss and a rattle, and it comes in by degrees rather '
                'than at a threshold. For pure slide and no knocks at '
                'all, leave this at zero and set friction around 0.15 '
                'to 0.3')
        self.add_modulation_input('hardness', self.unit.hardness_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        variety_port = self.add_modulation_input(
            'variety', self.unit.variety_in, minimum=0.0, maximum=1.0,
            speed=0.01)
        if variety_port.widget is not None:
            variety_port.widget.set_tooltip(
                'how unalike the things in there are. It spreads their '
                'SIZES, spreads how BOUNCY they are, varies how bouncy '
                'each single LANDING is -- an irregular grain presents '
                'a different face every time, and its contact sits off '
                'to one side of its middle -- gives them a bounce of '
                'their own from tipping on their corners even when the '
                'material has none, and lets each of them sit on the '
                'wall\'s roughness differently, so a rough wall throws '
                'them off unevenly. All of that matters because '
                'grains here do not collide with EACH OTHER: that is '
                'most of the cost and least of the sound, except in one '
                'case, which is a handful driven along one axis onto a '
                'flat wall. Identical grains there all leave and land '
                'together for ever, a hundred and twenty-eight of them '
                'inside a fifth of a millisecond -- one enormous click '
                'a cycle. Turn this up, with some "texture" on the wall '
                'for them to sit unevenly on, and they spread over 45 '
                'ms of it instead. It does not change how loud anything '
                'is')
        self.add_modulation_input('gravity', self.unit.gravity_in,
                                  minimum=0.0, maximum=40.0, speed=0.1)
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)
        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.knock_output = self.add_signal_output('knock',
                                                   self.unit.knock)
        self.scrape_output = self.add_signal_output('scrape',
                                                    self.unit.scrape)
        self.add_switch()
        self.finish_synth_node()

    def count_changed(self):
        self.unit.set_count(any_to_int(self.count_input()))

    def sync_options(self):
        shape = any_to_string(self.shape_input())
        if shape in RattleUnit.SHAPES:
            self.unit.shape = RattleUnit.SHAPES.index(shape)


class ShakerNode(SynthNode):
    """Shaken percussion: grains by the statistics of a gesture.

    'shake' is how hard the vessel is moving right now -- a flick is a
    burst that settles, a tremble is a wash, stillness is silence. Patch
    an effort stream and shaking a sensor is shaking the shaker; there is
    no trigger because a shaker has none.

    'kind' loads the statistics of a familiar object onto the knobs and
    lets go -- edit freely from there. 'grains out' is the raw collisions
    before the vessel resonance: into modal~ (drive up, dry 0) it puts
    the beans inside any object the table editor can draw.

    shaker~ <kind>, e.g. shaker~ tambourine. rain~ starts as rain.
    """

    @staticmethod
    def factory(name, data, args=None):
        return ShakerNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = ShakerUnit(synth_graph.sample_rate)

        kind = 'rain' if label == 'rain~' else 'maraca'
        if args is not None:
            for arg in args:
                if arg in SHAKER_KINDS:
                    kind = arg
        self._kind_shown = kind

        self.add_modulation_input('shake', self.unit.shake_in,
                                  minimum=0.0, maximum=2.0, speed=0.01,
                                  slider=False)
        self.shake_mode_input = self.add_input(
            'shake mode', widget_type='combo',
            default_value=ShakerUnit.SHAKE_MODES[0],
            callback=self.parameters_changed)
        self.shake_mode_input.widget.combo_items = list(
            ShakerUnit.SHAKE_MODES)
        if self.shake_mode_input.widget is not None:
            self.shake_mode_input.widget.set_tooltip(
                "how 'shake' is read. 'throw' is a STROKE: it pumps the "
                "beans and they carry on by themselves and settle, "
                "which is what a shaker does in a hand. 'hold' is how "
                "agitated they are right NOW -- the gesture is the "
                "agitation, a steady hand gives a steady wash, and "
                "letting go stops it at the settle rate. An effort "
                "stream already means the second thing")
        swirl_port = self.add_modulation_input(
            'swirl', self.unit.swirl_in, minimum=0.0, maximum=1.0,
            speed=0.01)
        if swirl_port.widget is not None:
            swirl_port.widget.set_tooltip(
                'the ANGLE the beans meet the shell at, not a second '
                'gesture. You cannot shake a maraca while you are '
                'rolling it, or roll it while you are shaking it -- '
                'there is one agitation and this is how it arrives. At '
                '0 they go head on and stop dead against the wall and '
                'ring it: the tick. At 1 they go tangential and keep '
                'their speed along it and drag: the graze. Between is '
                'both, and finer and more numerous on the way, because '
                'a bean that skips rather than stops makes more '
                'contacts and smaller ones. Nothing in here wobbles it '
                '-- a real roll surges as the heap comes round, but '
                'that is a shape a hand makes, so make it: patch the '
                'movement, or an LFO, into this or into shake')
        self.add_modulation_input('density', self.unit.density_in,
                                  minimum=1.0, maximum=2000.0, speed=2.0,
                                  slider=False)
        self.add_modulation_input('settle', self.unit.settle_in,
                                  minimum=0.02, maximum=1.0, speed=0.005,
                                  slider=False)
        self.add_modulation_input('hardness', self.unit.hardness_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('vessel', self.unit.vessel_in,
                                  minimum=100.0, maximum=12000.0, speed=10.0,
                                  slider=False)
        res_port = self.add_modulation_input('resonance',
                                             self.unit.resonance_in,
                                             minimum=0.0, maximum=1.0,
                                             speed=0.01)
        if res_port.widget is not None:
            res_port.widget.set_tooltip(
                'ring time, exponential: a 3 ms thud at 0 to a tenth of a '
                'second of bell at 1. Tambourine and sleighbells live in '
                'the top third')
        jingle_port = self.add_modulation_input('jingle', self.unit.jingle_in,
                                                minimum=0.0, maximum=1.0,
                                                speed=0.01)
        if jingle_port.widget is not None:
            jingle_port.widget.set_tooltip(
                'the vessel is eight bells: jingle spreads their fixed '
                'tunings around the vessel pitch, each collision striking '
                'one while the others ring on. 0 is a single voice')
        vary_port = self.add_modulation_input('vary', self.unit.vary_in,
                                              minimum=0.0, maximum=1.0,
                                              speed=0.01)
        if vary_port.widget is not None:
            vary_port.widget.set_tooltip(
                'bean size spread: each collision draws its own ring time, '
                'up to an octave either side of hardness at full')
        self.add_modulation_input('level', self.unit.level_in,
                                  minimum=0.0, maximum=2.0, speed=0.01)

        self.kind_input = self.add_input('kind', widget_type='combo',
                                         default_value=kind,
                                         callback=self.kind_changed)
        self.kind_input.widget.combo_items = list(SHAKER_KINDS)

        self.signal_output = self.add_signal_output('out', self.unit.out)
        self.grains_output = self.add_signal_output('grains out',
                                                    self.unit.grains)
        self.add_switch()
        self.finish_synth_node()

    KIND_PORTS = ('density', 'settle', 'hardness', 'vessel', 'resonance',
                  'jingle')

    def custom_create(self, from_file):
        # A node made by hand starts as its kind; a loaded one keeps the
        # knobs the patch saved.
        if not from_file:
            self.apply_kind(self._kind_shown)

    def sync_options(self):
        mode = any_to_string(self.shake_mode_input())
        if mode in ShakerUnit.SHAKE_MODES:
            self.unit.shake_mode = ShakerUnit.SHAKE_MODES.index(mode)

    def kind_changed(self):
        chosen = any_to_string(self.kind_input())
        if chosen == self._kind_shown:
            return
        self._kind_shown = chosen
        if chosen not in SHAKER_KINDS or self.in_loading_process:
            return
        self.apply_kind(chosen)

    def apply_kind(self, name):
        recipe = SHAKER_KINDS.get(name)
        if recipe is None:
            return
        for port in self.inputs:
            name = port.get_label()
            if name in recipe and port.widget is not None:
                port.widget.set(recipe[name])
        self.parameters_changed()


class StrokeNode(SynthNode):
    """A bow arm: coordinated velocity and force from one gesture.

    Patch 'velocity' and 'force' to the same inlets on bow~ or rub~ (with
    the destination's own knobs at zero -- the triad sums), and the two
    outputs move the way a player's arm does: velocity a cornered
    trapezoid that crosses the awkward low-speed region quickly, force
    leaning in exactly where velocity dips. 'tick' pulses at each
    turnaround, so anything else can happen in time with the bowing.

    'run' strokes continuously; 'gate' draws while the gate is high and
    lifts on release; the trigger plays one complete stroke either way.
    Also blows: velocity into wind~'s pressure phrases breathing.

    stroke~ <rate>.
    """

    @staticmethod
    def factory(name, data, args=None):
        return StrokeNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = StrokeUnit(synth_graph.sample_rate)

        rate = 1.0
        mode = 'run'
        if args is not None:
            for arg in args:
                if arg in StrokeUnit.MODES:
                    mode = arg
                else:
                    try:
                        rate = float(arg)
                    except (ValueError, TypeError):
                        continue
        self.unit.rate_in.base = rate
        self.unit.mode = StrokeUnit.MODES.index(mode)

        self.add_modulation_input('gate', self.unit.gate_in,
                                  widget_type='checkbox', default_value=False,
                                  attenuverter=False)
        self.add_trigger_signal_input('stroke', self.unit.trigger_in,
                                      self.stroke)
        self.add_modulation_input('rate', self.unit.rate_in,
                                  default_value=rate, minimum=0.05,
                                  maximum=8.0, speed=0.01, slider=False)
        self.add_modulation_input('speed', self.unit.speed_in,
                                  minimum=0.0, maximum=1.5, speed=0.01)
        self.add_modulation_input('dip', self.unit.dip_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('corner', self.unit.corner_in,
                                  minimum=0.005, maximum=0.3, speed=0.002,
                                  slider=False)
        self.add_modulation_input('force', self.unit.force_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('lean', self.unit.lean_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)
        self.add_modulation_input('swell', self.unit.swell_in,
                                  minimum=0.0, maximum=1.0, speed=0.01)

        self.mode_input = self.add_input('mode', widget_type='combo',
                                         default_value=mode,
                                         callback=self.parameters_changed)
        self.mode_input.widget.combo_items = list(StrokeUnit.MODES)

        self.velocity_output = self.add_signal_output('velocity',
                                                      self.unit.velocity_out)
        self.force_output = self.add_signal_output('force',
                                                   self.unit.force_out)
        self.tick_output = self.add_signal_output('tick', self.unit.tick_out)
        self.add_switch()
        self.finish_synth_node()

    def stroke(self):
        self.unit.fire()

    def sync_options(self):
        mode = any_to_string(self.mode_input())
        if mode in StrokeUnit.MODES:
            self.unit.mode = StrokeUnit.MODES.index(mode)


class FaderNode(SynthNode):
    """A channel fader: long-throw vertical handle, desk taper, dB readout.

    Unity sits at three quarters of the travel, +6 dB above it, 60 dB of
    dB-linear reach below, and the bottom twentieth fades to true silence.
    The handle is also an inlet, so automation rides the same taper as the
    hand. Stereo when something is patched to the right inlet. The face is
    kept to the throw: pins say only left and right (which side of the node
    they sit on says the rest), and there is no bypass -- a fader's own
    bottom is its off. There is a MUTE, though, which is a different
    thing: it silences the channel and leaves the handle where it was, so
    unmuting comes back to the balance you had rather than to wherever
    silence left the hand.
    """

    @staticmethod
    def factory(name, data, args=None):
        return FaderNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        # Through a hook, so a strip that ends at the socket can be this
        # same face with a different unit under it rather than a copy of
        # it -- the taper, the pan law and the meters then cannot drift
        # apart from fader~'s.
        self.unit = self._make_unit(args)

        # Bare 'left' and 'right': which side of the node a pin sits on
        # already says in or out, and a fader wants no more face than its
        # throw. No bypass either -- a fader's own bottom is its off.
        self.add_signal_input('left', self.unit.signal_in)
        self.add_signal_input('right', self.unit.right_in)
        # A plain widget input rather than a modulation row: the handle is
        # its own display, so the name/value/depth columns would only be
        # width. The inlet stays patchable at full depth.
        self.fader_input = self.add_input(
            'fader', widget_type='slider_float_vertical',
            default_value=FaderUnit.UNITY_POSITION,
            min=0.0, max=1.0, callback=self.parameters_changed)
        self.fader_input.synth_inlet = self.unit.position_in
        self.signal_inputs.append(self.fader_input)
        self._parameter_bindings.append((self.fader_input,
                                         self.unit.position_in))
        if self.fader_input.widget is not None:
            self.fader_input.widget.slider_height = 150
            self.fader_input.widget._label = '##fader'
            self.fader_input.widget.meter_count = 2
            self.fader_input.widget.set_tooltip(
                'desk taper: unity at 3/4 travel, +6 dB at the top, '
                'true silence at the bottom')
        self.pan_input = self.add_input('pan', widget_type='knob_float',
                                        default_value=0.0,
                                        min=-1.0, max=1.0,
                                        callback=self.parameters_changed)
        self.pan_input.synth_inlet = self.unit.pan_in
        self.signal_inputs.append(self.pan_input)
        self._parameter_bindings.append((self.pan_input, self.unit.pan_in))
        if self.pan_input.widget is not None:
            self.pan_input.widget.set_tooltip(
                'equal-power pan, unity at center: existing patches hear '
                'no change until the knob moves')

        self.mute_input = self.add_input('mute', widget_type='checkbox',
                                        default_value=False,
                                        callback=self.parameters_changed)
        if self.mute_input.widget is not None:
            self.mute_input.widget.set_tooltip(
                'silence without moving the handle -- which is the whole '
                'difference between muting a channel and pulling it '
                'down: unmuting comes back to the balance you had. '
                'Ramped over a few milliseconds, so it does not click')
        self.db_display = self.add_property('dB', widget_type='label',
                                            default_value='+0.0 dB')

        self._add_outputs()
        self._extra_face()
        self.finish_synth_node()
        self._shown_db = 0.0
        self._meter_tags = None
        self._meter_shown = (-1.0, -1.0, -1.0, -1.0)

    def sync_options(self):
        self.unit.muted = any_to_bool(self.mute_input())

    def _make_unit(self, args):
        return FaderUnit(synth_graph.sample_rate)

    def _add_outputs(self):
        self.signal_output = self.add_signal_output('left', self.unit.out)
        self.right_output = self.add_signal_output('right', self.unit.right)

    def _extra_face(self):
        """Anything a subclass wants after the strip and before the end."""

    def _fraction(self, value):
        # vu~'s scale, standing up: -60 dB at the foot, +6 at the crown.
        if value <= 1.0e-6:
            return 0.0
        db = 20.0 * math.log10(value)
        span = VuNode.METER_CEIL_DB - VuNode.METER_FLOOR_DB
        return min(1.0, max(0.0, (db - VuNode.METER_FLOOR_DB) / span))

    def _init_meters(self):
        widget = self.fader_input.widget
        drawlist = getattr(widget, 'meter_drawlist', None)
        if drawlist is None or not dpg.does_item_exist(drawlist):
            return False
        lane = widget.meter_lane_width
        height = widget.slider_height
        self._meter_tags = []
        for channel in range(2):
            x0 = channel * lane + 1
            x1 = x0 + lane - 2
            for low, high, color in VuNode.ZONES:
                ytop = height * (1.0 - self._db_frac(high))
                ybot = height * (1.0 - self._db_frac(low))
                dpg.draw_rectangle(pmin=(x0, ytop), pmax=(x1, ybot),
                                   fill=(color[0], color[1], color[2], 48),
                                   color=(0, 0, 0, 0), parent=drawlist)
            fills = []
            for low, high, color in VuNode.ZONES:
                ybot = height * (1.0 - self._db_frac(low))
                fills.append(dpg.draw_rectangle(
                    pmin=(x0, ybot), pmax=(x1, ybot), fill=color,
                    color=(0, 0, 0, 0), parent=drawlist))
            peak = dpg.draw_line((x0, 0), (x1, 0),
                                 color=(230, 230, 230, 0), thickness=2,
                                 parent=drawlist)
            self._meter_tags.append({'fills': fills, 'peak': peak,
                                     'x0': x0, 'x1': x1})
        return True

    @staticmethod
    def _db_frac(db):
        span = VuNode.METER_CEIL_DB - VuNode.METER_FLOOR_DB
        return min(1.0, max(0.0, (db - VuNode.METER_FLOOR_DB) / span))

    def _update_meters(self):
        if self._meter_tags is None:
            if not self._init_meters():
                return
        state = tuple(self.unit.levels) + tuple(self.unit.peaks)
        if all(abs(now - was) < 0.001
               for now, was in zip(state, self._meter_shown)):
            return
        self._meter_shown = state
        height = self.fader_input.widget.slider_height
        for channel, meter in enumerate(self._meter_tags):
            level_frac = self._fraction(self.unit.levels[channel])
            for zone, (low, high, _color) in enumerate(VuNode.ZONES):
                tag = meter['fills'][zone]
                if not dpg.does_item_exist(tag):
                    continue
                fbot = self._db_frac(low)
                ftop = min(level_frac, self._db_frac(high))
                ybot = height * (1.0 - fbot)
                ytop = height * (1.0 - max(fbot, ftop))
                dpg.configure_item(tag, pmin=(meter['x0'], ytop),
                                   pmax=(meter['x1'], ybot))
            peak = self.unit.peaks[channel]
            tag = meter['peak']
            if dpg.does_item_exist(tag):
                if peak <= 1.0e-6:
                    dpg.configure_item(tag, color=(230, 230, 230, 0))
                else:
                    y = height * (1.0 - self._fraction(peak))
                    hot = peak >= 1.0
                    dpg.configure_item(
                        tag, p1=(meter['x0'], y), p2=(meter['x1'], y),
                        color=(235, 85, 70, 255) if hot
                        else (230, 230, 230, 180))

    def synth_frame_task(self):
        self._update_meters()
        db = self.unit.current_db()
        if db is None:
            if self._shown_db is not None:
                self._shown_db = None
                if self.db_display.widget is not None:
                    self.db_display.widget.set('-inf dB')
            return
        if self._shown_db is None or abs(db - self._shown_db) > 0.05:
            self._shown_db = db
            if self.db_display.widget is not None:
                self.db_display.widget.set(f'{db:+.1f} dB')



class FaderOutNode(FaderNode):
    """A fader and the socket it lands on, in one.

    The split between fader~ and audio_out~ is right and it stays: level
    is one job, the wall is another, and a patch with several sources
    wants several faders into one socket. But almost every new patch
    begins by making both and joining them, so this is that pair
    ready-made -- a strip that ends at the device.

    Everything fader~ has, because it IS fader~ with a different unit
    under it: the long-throw handle on the desk taper, the pan, the pair
    of meters, the dB readout, the mute. What it adds is the only part of
    a socket that differs from one strip to the next --

    'channels' is which device outputs it lands on, counted from 1 the
    way an interface's front panel counts them, so the default 1 2 is the
    first pair. A channel the device does not have is silent rather than
    an error, so a patch written for a rig still runs on a laptop.

    There is deliberately no device chooser. The device is ENGINE-WIDE --
    one stream, shared with the sampler -- so a copy of that control on
    every strip would be several ways of saying one thing. It follows
    whatever the engine is on; set it on an audio_out~ if you need to
    move it.

    Left and right still come out as well as going to the device, so a
    meter, a recorder or a second socket can take the same signal.

    fader_out~ <channels...>, e.g. fader_out~ 3 4.
    """

    @staticmethod
    def factory(name, data, args=None):
        return FaderOutNode(name, data, args)

    def _make_unit(self, args):
        channels = [1, 2]
        if args is not None and len(args) > 0:
            values = [decode_arg(args, index) for index in range(len(args))]
            whole = [int(value) for value, kind in values if kind == int]
            if len(whole) >= 2:
                channels = whole[:2]
        self._channel_list = channels
        unit = FaderOutUnit(synth_graph.sample_rate)
        unit.channels = [max(0, channel - 1) for channel in channels]
        return unit

    def _extra_face(self):
        self.channels_option = self.add_option(
            'channels', widget_type='text_input', width=110,
            default_value=' '.join(str(c) for c in self._channel_list),
            callback=self.parameters_changed)
        if self.channels_option.widget is not None:
            self.channels_option.widget.set_tooltip(
                'which device outputs this lands on, counted from 1 the '
                'way the interface\'s front panel counts them. A channel '
                'the device does not have is silent rather than an error, '
                'so a patch written for a rig still runs on a laptop. The '
                'DEVICE itself is engine-wide and is not here -- set it '
                'on an audio_out~')

    def sync_options(self):
        super().sync_options()
        wanted = []
        for word in any_to_string(
                self.channels_option()).replace(',', ' ').split():
            try:
                wanted.append(max(1, min(32, int(word))))
            except (ValueError, TypeError):
                continue
        if len(wanted) >= 2:
            self.unit.channels = [max(0, c - 1) for c in wanted[:2]]

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
        self._rate_sent = False

        self.add_signal_input('in', self.unit.signal_in)
        self.bang_input = self.add_input('bang', widget_type='button',
                                         callback=self.send_now)

        self.array_output = self.add_array_output('array')
        self.dropped_output = self.add_output('dropped')
        # The engine rate, sent once, so a speech node or stream~ fed from
        # this array can be told what it is.
        self.rate_output = self.add_output('rate')

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
        if not self._rate_sent:
            self._rate_sent = True
            self.rate_output.send(int(self.unit.sample_rate))
        if any_to_string(self.send_option()) == 'on bang':
            # Keep the read cursor current so switching back to streaming does
            # not immediately dump a large backlog.
            self._last_read = self.unit.written
            return
        self._emit()


# ----------------------------------------------------------------------------
# stream~
# ----------------------------------------------------------------------------

class StreamNode(SynthNode):
    """Audio from the node world into the graph. The reverse of capture~.

    Patch a microphone (t.audio_source), a file streamer (t.audio.file_stream),
    a capture~ from another part of the graph, or any numpy / torch chain
    into 'audio in', and the audio comes out as a signal for vocoder~,
    string~, vst~ or anything else that takes one. Chunks may be 1-D,
    (channels, frames) or (frames, channels); a stereo chunk fills both
    outlets, a mono one both alike.

    'rate' is the rate the chunks were made at, in Hz. It is an inlet so a
    streamer's sample_rate outlet can drive it; there is no way to read it
    off the numbers themselves. 'latency' is how much audio, in ms, to hold
    before starting: too little and a bursty source runs dry (counted on
    'underruns'), too much and the sound lags. A backlog beyond a quarter
    second is skipped to keep the stream live, counted on 'dropped'.

    Arguments: stream~ <rate> <latency ms>. Also registered as audio_in~.
    """

    @staticmethod
    def factory(name, data, args=None):
        return StreamNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.unit = StreamUnit(synth_graph.sample_rate)

        rate = int(synth_graph.sample_rate)
        latency_ms = 50.0
        numbers = []
        if args is not None:
            for arg in args:
                value, arg_type = decode_arg([arg], 0)
                if arg_type in (float, int):
                    numbers.append(float(value))
        if len(numbers) > 0:
            rate = int(numbers[0])
        if len(numbers) > 1:
            latency_ms = numbers[1]

        self.audio_input = self.add_input('audio in', triggers_execution=True)
        self.rate_input = self.add_input('rate', widget_type='drag_int',
                                         default_value=rate, min=1000, max=384000,
                                         callback=self.settings_changed)
        self.latency_input = self.add_input('latency', widget_type='drag_float',
                                            default_value=latency_ms, min=0.0,
                                            max=2000.0, callback=self.settings_changed)
        self.add_modulation_input('level', self.unit.level_in, default_value=1.0,
                                  minimum=0.0, maximum=2.0)

        self.left_output = self.add_signal_output('left out', self.unit.out)
        self.right_output = self.add_signal_output('right out', self.unit.right)
        self.underruns_output = self.add_output('underruns')
        self.dropped_output = self.add_output('dropped')
        self._reported = (0, 0)

        self.add_switch()
        self.finish_synth_node()
        self.settings_changed()

    def settings_changed(self):
        self.unit.source_rate = float(max(1000, any_to_int(self.rate_input())))
        self.unit.latency = max(0.0, any_to_float(self.latency_input())) / 1000.0

    def update_parameters_from_widgets(self):
        super().update_parameters_from_widgets()
        self.settings_changed()

    def execute(self):
        data = self.audio_input()
        if data is None:
            return
        if hasattr(data, 'detach'):
            data = data.detach().cpu().numpy()
        elif not isinstance(data, np.ndarray):
            data = any_to_array(data)
        if data is None or data.size == 0:
            return
        self.unit.push(data)

    def synth_frame_task(self):
        counts = (self.unit.underruns, self.unit.dropped)
        if counts != self._reported:
            if counts[1] != self._reported[1]:
                self.dropped_output.send(counts[1])
            if counts[0] != self._reported[0]:
                self.underruns_output.send(counts[0])
            self._reported = counts


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
        self.time_input = self.add_input(
            'time', widget_type='drag_float',
            default_value=self.samples / synth_graph.sample_rate * 1000.0,
            callback=self.time_changed)
        if self.time_input.widget is not None:
            self.time_input.widget.speed = 0.2
            self.time_input.widget.set_tooltip(
                'how much time is on screen, in milliseconds -- the easy '
                'handle on the timescale. The samples option follows')
        self.make_drag_proportional(self.time_input)

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
        # The two handles agree: samples moved, milliseconds follow.
        if getattr(self, 'time_input', None) is not None \
                and self.time_input.widget is not None:
            ms = self.samples / synth_graph.sample_rate * 1000.0
            if abs(any_to_float(self.time_input()) - ms) > 0.05:
                self.time_input.widget.set(ms)

    def time_changed(self):
        ms = max(0.05, any_to_float(self.time_input()))
        samples = self._clamp_samples(ms * 0.001 * synth_graph.sample_rate)
        self.samples_option.set(samples)
        self.window_changed()

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

        self.add_signal_input('left in', self.unit.signal_in)
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

        self.signal_output = self.add_signal_output('left out', self.unit.out)
        self.right_output = self.add_signal_output('right out', self.unit.right)
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

    def custom_cleanup(self):
        # Unregister first: the recompile takes the unit out of the program,
        # so the audio thread has let go of the plugin before it is dropped.
        # The parameter objects each hold the plugin, so they go too. What is
        # left is destroyed here, on the main thread, with JUCE still whole,
        # which is the only order pedalboard tears down cleanly in.
        super().custom_cleanup()
        self.unit.attach(None, 1)
        self._parameters = {}
        self._applied_choices = {}
        self.plugin = None

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
