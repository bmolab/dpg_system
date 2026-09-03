"""
Joint torque gang nodes for dpg_system.

One node is one gang. Patch several and they cost what one costs: every live
node's declaration is folded into a single term matrix by gang_core, and the
first node to run in a frame evaluates the whole bank while the rest read
their row out of it. Adding gangs to a patch buys expressiveness without
buying per-frame work, which is the reason for the compiler underneath.

Arguments: torque_gang <preset> [side] [stream]

    torque_gang spine_flex
    torque_gang leg_push differential dynamic
    torque_gang arm_reach left

Outputs are net, total, coherence and surprise -- see gang_core for the first
three. The short version is that net is direction, total is how much work, and
coherence is whether the group is acting as one thing or against itself.

surprise is the companion: how UNUSUAL this gang's activation is, measured
against a prior over 12.5M frames of AMASS. A gang firing says the body did
the expected thing; a gang firing in a way the corpus rarely does is the more
informative event. See gang_prior for the measure and its cautions.
"""

import dearpygui.dearpygui as dpg

from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.gang_core import (
    gang_graph, GANG_PRESETS, STREAMS, SIDES, NO_SIDE,
    JOINT_COUNT, AXIS_COUNT,
    preset_names, preset_is_bilateral, sides_for, spec_from_preset,
    max_torque_array,
)
from dpg_system.gang_prior import get_prior, PRIOR_STREAM

import numpy as np


GENDERS = ('neutral', 'female')


def register_gang_nodes():
    Node.app.register_node('torque_gang', TorqueGangNode.factory)
    Node.app.register_node('gang', TorqueGangNode.factory)
    Node.app.register_node('torque_residual', TorqueResidualNode.factory)


class GangStreamNode(Node):
    """Base for nodes fed by the four smpl_torque streams.

    smpl_torque sends torque_vectors first and gravity, dynamic and passive
    after it, so a node that triggered on the wrong inlet would read this
    frame's total against last frame's dynamic. Rather than reassemble the
    streams by frame number, exactly one inlet triggers execution -- the one
    the node actually reads -- and the rest are stored as they arrive.
    triggers_execution is consulted when data is received, so retargeting it
    is just an assignment.
    """

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.stream_inputs = {}
        self._registered = False
        self._warned_unpatched = False

    def add_stream_inputs(self):
        for name in STREAMS:
            label = 'torque' if name == 'total' else name
            self.stream_inputs[name] = self.add_input(
                label, triggers_execution=False)

    def set_trigger_stream(self, stream):
        """Point execution at one inlet and make the others passive."""
        for name, port in self.stream_inputs.items():
            port.triggers_execution = (name == stream)

    def current_bundle(self):
        bundle = {}
        for name, port in self.stream_inputs.items():
            value = port()
            bundle[name] = value if isinstance(value, np.ndarray) else None
        return bundle

    def check_stream_patched(self, stream):
        """Warn once if the selected stream is the one inlet left unpatched.

        Silence is otherwise the only symptom, and it looks identical to a
        gang whose weights happen to cancel.
        """
        if self._warned_unpatched:
            return
        port = self.stream_inputs.get(stream)
        if port is None or len(port.get_parents()) > 0:
            return
        if any(len(other.get_parents()) > 0
               for other in self.stream_inputs.values()):
            self._warned_unpatched = True
            print(self.label + ": '" + stream + "' stream is not patched; "
                  'this node triggers on that inlet and will stay silent')

    def frame_task(self):
        # Any gang node drives the shared declaration check; it acts once per
        # frame no matter how many nodes call it.
        gang_graph.tick(Node.app.frame_number)

    def custom_cleanup(self):
        if self._registered:
            gang_graph.unregister(self)
            self._registered = False


class TorqueGangNode(GangStreamNode):
    """One named group of joint torques, as three scalars.

    The declaration -- which preset, which side, which stream -- is built in
    __init__ from the arguments rather than read back from widgets, because
    patch load skips a widget's callback when the saved value already matches
    the widget's current value. A node that waited to be told what it was
    would come back from a load holding nothing.
    """

    @staticmethod
    def factory(name, data, args=None):
        return TorqueGangNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        preset = 'spine_flex'
        side = None
        stream = 'total'
        unrecognised = []
        if args is not None:
            for arg in args:
                text = any_to_string(arg)
                if text in GANG_PRESETS:
                    preset = text
                elif text in SIDES or text == NO_SIDE:
                    side = text
                elif text in STREAMS:
                    stream = text
                else:
                    unrecognised.append(text)
        if unrecognised:
            # Falling back silently would leave a mistyped preset name looking
            # like a working gang that reports the wrong thing.
            print(label + ': ignored argument(s) ' + ', '.join(unrecognised)
                  + '; using preset "' + preset + '". Known presets: '
                  + ', '.join(preset_names()))

        if side is None:
            side = 'left' if preset_is_bilateral(preset) else NO_SIDE
        elif not preset_is_bilateral(preset):
            side = NO_SIDE

        self.add_stream_inputs()

        self.gang_input = self.add_input('gang', widget_type='combo',
                                         default_value=preset,
                                         callback=self.declaration_changed)
        self.gang_input.widget.combo_items = list(preset_names())

        self.side_input = self.add_input('side', widget_type='combo',
                                         default_value=side,
                                         callback=self.declaration_changed)
        self.side_input.widget.combo_items = list(sides_for(preset))

        self.stream_input = self.add_input('stream', widget_type='combo',
                                           default_value=stream,
                                           callback=self.declaration_changed)
        self.stream_input.widget.combo_items = list(STREAMS)

        self.normalize_option = self.add_option(
            'normalize', widget_type='checkbox', default_value=True,
            callback=self.declaration_changed)
        self.gender_option = self.add_option(
            'gender', widget_type='combo', default_value='neutral',
            callback=self.declaration_changed)
        self.gender_option.widget.combo_items = list(GENDERS)
        self.invert_option = self.add_option(
            'invert', widget_type='checkbox', default_value=False,
            callback=self.declaration_changed)

        self.net_output = self.add_output('net')
        self.total_output = self.add_output('total')
        self.coherence_output = self.add_output('coherence')
        # How unusual this gang's activation is, measured against the corpus
        # prior: the whitened deviation along this gang's own direction,
        # divided by the frame's torque magnitude so it reports strangeness of
        # shape rather than amount of effort. See gang_prior for the cautions.
        # 0 when there is no prior, or when the gang reads a stream the prior
        # was not built on.
        self.surprise_output = self.add_output('surprise')
        self._warned_stream = False

        # Built from the parsed arguments, not from the widgets: widget values
        # read None until the widget is created, and options are created after
        # inputs. This is the declaration until something actually changes it.
        self.gang_spec = None
        self._preset = preset
        self._side = side
        self._invert = False
        self._build_spec(preset, side, stream, normalize=True,
                         gender='neutral')

        gang_graph.register(self)
        self._registered = True
        self.add_frame_task()

    # -- declaration --------------------------------------------------------

    def _build_spec(self, preset, side, stream, normalize, gender):
        try:
            self.gang_spec = spec_from_preset(preset, side=side, stream=stream,
                                              normalize=normalize,
                                              gender=gender)
        except ValueError as error:
            print(self.label + ': ' + str(error))
            self.gang_spec = None
        self.set_trigger_stream(stream)
        self._warned_unpatched = False

    def declaration_changed(self):
        """Rebuild this node's spec; the graph recompiles on the next frame."""
        preset = any_to_string(self.gang_input())
        if preset not in GANG_PRESETS:
            return

        # The side list is preset-dependent, so a preset change can leave the
        # side widget holding a value the new preset cannot use.
        if preset != self._preset:
            self._preset = preset
            self._refresh_side_items(preset)

        side = any_to_string(self.side_input())
        valid = sides_for(preset)
        if side not in valid:
            side = valid[0]
            self._set_widget(self.side_input, side)
        self._side = side

        stream = any_to_string(self.stream_input())
        if stream not in STREAMS:
            stream = 'total'

        normalize = self._option_bool(self.normalize_option, True)
        self._invert = self._option_bool(self.invert_option, False)
        gender = any_to_string(self.gender_option())
        if gender not in GENDERS:
            gender = 'neutral'

        self._build_spec(preset, side, stream, normalize, gender)

    def _refresh_side_items(self, preset):
        items = list(sides_for(preset))
        widget = self.side_input.widget
        if widget is None:
            return
        try:
            if dpg.does_item_exist(widget.uuid):
                dpg.configure_item(widget.uuid, items=items)
        except SystemError:
            pass
        widget.combo_items = items

    @staticmethod
    def _set_widget(port, value):
        if port.widget is not None:
            port.widget.set(value)

    @staticmethod
    def _option_bool(option, default):
        # Options are created after inputs, so an option read during
        # construction answers None rather than its default value.
        value = option()
        if value is None:
            return default
        return any_to_bool(value)

    def update_parameters_from_widgets(self):
        # Called once the loader has finished restoring every widget, which is
        # the first moment all of them can be trusted to answer.
        self.declaration_changed()

    # -- evaluation ---------------------------------------------------------

    def execute(self):
        if self.in_loading_process:
            return
        spec = self.gang_spec
        if spec is None:
            return

        self.check_stream_patched(spec.stream)

        row = gang_graph.row_for(spec)
        if row is None:
            # Registered this frame, or the declaration changed after the
            # graph last compiled. tick() will pick it up.
            return

        net, total, coherence = gang_graph.evaluate(Node.app.frame_number,
                                                    self.current_bundle())
        if row >= len(net):
            return

        net_value = float(net[row])
        if self._invert:
            net_value = -net_value

        # Right to left, matching send_all.
        self.surprise_output.send(self._shape_surprise(spec))
        self.coherence_output.send(float(coherence[row]))
        self.total_output.send(float(total[row]))
        self.net_output.send(net_value)

    def _shape_surprise(self, spec):
        """How unusual this gang's activation is, per unit torque.

        z is whitened once per frame for the whole patch; this is the component
        of it along this gang's direction, over the frame's torque magnitude.
        Sign is dropped -- unusual in either direction is unusual -- and invert
        is deliberately not applied, since it is an aesthetic flip of net and
        says nothing about rarity.
        """
        if spec.stream != PRIOR_STREAM:
            if not self._warned_stream:
                self._warned_stream = True
                print(self.label + ": surprise needs the '" + PRIOR_STREAM
                      + "' stream; this gang reads '" + spec.stream
                      + "' and will report 0")
            return 0.0

        whitened = gang_graph.whitened(Node.app.frame_number,
                                       self.current_bundle())
        if whitened is None:
            return 0.0
        z, magnitude = whitened
        if magnitude <= 1e-9:
            return 0.0

        prior = get_prior()
        if prior is None:
            return 0.0
        direction = prior.direction(spec, self._weight_vector(spec))
        if direction is None:
            return 0.0
        return float(abs(z @ direction) / magnitude)

    @staticmethod
    def _weight_vector(spec):
        """This gang's terms as a 66-channel weight vector.

        spec.terms carries the raw weights: capacity normalization is applied
        by the compiler, not folded into the spec. It has to be applied here
        too, or surprise would be measured along a different direction than the
        net this node reports -- they differ whenever a gang spans joints of
        unequal capacity, which is most of the interesting ones (leg_push runs
        hip 300, knee 250, ankle 100).
        """
        capacity = max_torque_array(spec.gender) if spec.normalize else None
        weights = np.zeros(66)
        for joint, axis, weight in spec.terms:
            if joint < 22:
                if capacity is not None:
                    weight = weight / capacity[joint][axis]
                weights[joint * 3 + axis] += weight
        return weights


class TorqueResidualNode(GangStreamNode):
    """Per-joint torque left over after every live gang takes its share.

    Whatever the bank consumes, this is the rest of it -- the effort that does
    not fit any named gesture. It exists so that a gang bank cannot silently
    drop information: patch it and you can hear what the gangs are not
    describing. The projector is rebuilt whenever the bank recompiles, so this
    tracks the gangs currently in the patch rather than any fixed set.

    Outputs a (24,) magnitude per joint.
    """

    @staticmethod
    def factory(name, data, args=None):
        return TorqueResidualNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        stream = 'total'
        if args is not None:
            for arg in args:
                text = any_to_string(arg)
                if text in STREAMS:
                    stream = text

        self.add_stream_inputs()
        self.stream_input = self.add_input('stream', widget_type='combo',
                                           default_value=stream,
                                           callback=self.stream_changed)
        self.stream_input.widget.combo_items = list(STREAMS)

        self.residual_output = self.add_output('residual')
        self.magnitude_output = self.add_output('magnitude')

        self._stream = stream
        self.set_trigger_stream(stream)
        self.add_frame_task()

    def stream_changed(self):
        stream = any_to_string(self.stream_input())
        if stream not in STREAMS:
            return
        self._stream = stream
        self.set_trigger_stream(stream)
        self._warned_unpatched = False

    def update_parameters_from_widgets(self):
        self.stream_changed()

    def execute(self):
        if self.in_loading_process:
            return
        self.check_stream_patched(self._stream)

        torque = self.stream_inputs[self._stream]()
        if not isinstance(torque, np.ndarray):
            return
        if torque.shape[-2:] != (JOINT_COUNT, AXIS_COUNT):
            return

        residual = gang_graph.program.residual(torque)
        self.magnitude_output.send(float(np.linalg.norm(residual)))
        self.residual_output.send(residual)
