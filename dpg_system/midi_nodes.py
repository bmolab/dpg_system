import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import mido
import platform
import threading

if platform.system() == "Darwin":
    try:
        mido.set_backend('mido.backends.rtmidi')
    except Exception as e:
        print('midi_nodes: failed to set rtmidi backend:', e)


def _safe_message_from_bytes(midi_data):
    """Build a mido.Message from a bytes-like; return None on failure."""
    try:
        return mido.Message.from_bytes(midi_data)
    except Exception as e:
        print('midi_nodes: from_bytes failed for', midi_data, ':', e)
        return None


def _send_out(out_port, midi_data):
    """Send raw MIDI bytes if a usable port is available."""
    if out_port is None or getattr(out_port, 'port', None) is None:
        return
    msg = _safe_message_from_bytes(midi_data)
    if msg is None:
        return
    try:
        out_port.send(msg)
    except Exception as e:
        print('midi_nodes: send failed:', e)


def register_midi_nodes():
    Node.app.register_node('midi_in', MidiInNode.factory)
    Node.app.register_node('midi_control_in', MidiControllerInNode.factory)
    Node.app.register_node('midi_pitchbend_in', MidiPitchBendInNode.factory)
    Node.app.register_node('midi_aftertouch_in', MidiChannelAfterTouchNode.factory)
    Node.app.register_node('midi_program_in', MidiProgramInNode.factory)
    Node.app.register_node('midi_note_in', MidiNoteInNode.factory)
    Node.app.register_node('midi_poly_pressure_in', MidiPolyPressureInNode.factory)
    Node.app.register_node('midi_out', MidiOutNode.factory)
    Node.app.register_node('midi_control_out', MidiControlOutNode.factory)
    Node.app.register_node('midi_pitchbend_out', MidiPitchBendOutNode.factory)
    Node.app.register_node('midi_program_out', MidiProgramOutNode.factory)
    Node.app.register_node('midi_aftertouch_out', MidiAftertouchOutNode.factory)
    Node.app.register_node('midi_note_out', MidiNoteOutNode.factory)
    Node.app.register_node('midi_poly_pressure_out', MidiPolyPressureOutNode.factory)

    Node.app.register_node('midi_device', MidiDeviceNode.factory)
    Node.app.register_node('blue_board', BlueBoardNode.factory)
    Node.app.register_node('mpd218', MPD218Node.factory)
    Node.app.register_node('mpe_in', MPEInNode.factory)


note_off_code = 128
note_on_code = 144
poly_pressure_code = 160
controller_code = 176
program_code = 192
aftertouch_code = 208
pitch_bend_code = 224

def strip_Linux_midi_port_name(name):
    chunks = name.split(' ')
    subchunks = chunks[-1].split(':')
    if is_number(subchunks[0]) and is_number(subchunks[1]):
        name = ' '.join(subchunks[:-1])
    return name


class MidiInPort:
    ports = {}

    def __init__(self, name=None):
        self.port_name = name
        self.port = None

        self.clients = {}
        self.general_clients = []

        if self.port_name is not None:
            if self.port_name in MidiInPort.ports:
                self.port = MidiInPort.ports[self.port_name].port
            else:
                try:
                    self.port = mido.open_input(self.port_name, callback=self.receive)
                    MidiInPort.ports[self.port_name] = self
                    print('added in port', self.port_name, self.port)
                except Exception as e:
                    print('MidiInPort init: exception: could not find in port', self.port_name)
                    self.port = None
        else:
            keys = list(MidiInPort.ports.keys())
            if len(keys) > 0:
                self.port = MidiInPort.ports[keys[0]].port
            else:
                try:
                    self.port = mido.open_input(callback=self.receive)
                except Exception as e:
                    print('MidiInPort init: exception could not find MIDI in port')
                    self.port = None

        if self.port:
            self.port_name = self.port.name
            if self.port_name not in MidiInPort.ports:
                MidiInPort.ports[self.port_name] = self
        else:
            self.port_name = ''

    def add_client(self, client, code=None):
        if code is None:
            if client not in self.general_clients:
                self.general_clients.append(client)
        else:
            if code not in self.clients:
                self.clients[code] = client

    def remove_client(self, client, code=None):
        if code is None:
            if client in self.general_clients:
                self.general_clients.remove(client)
        else:
            if code in self.clients:
                self.clients.pop(code, None)

    def receive(self, msg):
        midi_bytes = msg.bytes()
        if msg.is_cc():
            code = midi_bytes[0] * 256 + midi_bytes[1]
        else:
            code = midi_bytes[0] * 256 + 128

        if code in self.clients:
            self.clients[code].receive_midi_bytes(midi_bytes)
        else:
            if len(self.general_clients) > 0:
                for client in self.general_clients:
                    client.receive_midi_bytes(midi_bytes)


class MidiIn:
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.in_port_name = None
        self.in_port = None
        self.codes = [None]
        self.input_list = []

        if args:
            val, t = decode_arg(args, 0)
            if t == str:
                self.in_port_name = args[0]

        if self.in_port_name is not None:
            if self.in_port_name in MidiInPort.ports:
                self.in_port = MidiInPort.ports[self.in_port_name]
            else:
                self.find_in_port_from_partial_name(self.in_port_name)

        if self.in_port is None:
            if len(self.input_list) == 0:
                self.input_list = mido.get_input_names()
            if len(self.input_list) > 0:
                self.in_port_name = self.input_list[0]
                self.in_port = MidiInPort(self.in_port_name)

    def find_in_port_from_partial_name(self, partial_name):
        if len(self.input_list) == 0:
            self.input_list = mido.get_input_names()
        if len(self.input_list) > 0:
            # An exact name wins over a longer one that merely starts the same way:
            # 'Erae 2 MIDI' must not resolve to 'Erae 2 MIDI (MPE)'.
            if partial_name in self.input_list:
                self.in_port_name = partial_name
                if self.in_port_name in MidiInPort.ports:
                    self.in_port = MidiInPort.ports[self.in_port_name]
                else:
                    self.in_port = MidiInPort(self.in_port_name)
                return self.in_port_name
            for input in self.input_list:
                if len(input) > len(partial_name):
                    length = len(partial_name)
                    if input[:length] == partial_name:
                        self.in_port_name = input
                        if self.in_port_name in MidiInPort.ports:
                            self.in_port = MidiInPort.ports[self.in_port_name]
                        else:
                            self.in_port = MidiInPort(self.in_port_name)
                        if self.in_port_name not in MidiInPort.ports:
                            print('created in port', self.in_port_name, 'ports:', MidiInPort.ports.keys())
                        return self.in_port_name

            for input in self.input_list:
                if self.in_port_name in input:
                    self.in_port_name = input
                    self.in_port = MidiInPort(self.in_port_name)
                    return self.in_port_name
        return None

    def receive_midi_bytes(self, midi_bytes):
        pass

    def port_changed(self):
        if self.in_port is not None:
            for code in self.codes:
                self.in_port.remove_client(self, code=code)
        if self.in_port_name in MidiInPort.ports:
            self.in_port = MidiInPort.ports[self.in_port_name]
            print('found in port', self.in_port_name, 'in', MidiInPort.ports.keys(), self.in_port)
        else:
            result = self.find_in_port_from_partial_name(self.in_port_name)
        if self.in_port is not None:
            for code in self.codes:
                self.in_port.add_client(self, code=code)
        else:
            print('could not find in port', self.in_port_name)


class MidiInNode(MidiIn, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiIn.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        name = ''
        if self.in_port is not None and self.in_port.port is not None:
            name = self.in_port.port.name

        self.in_port_name_property = self.add_string_input('port', widget_type='combo', widget_width=200, default_value=name, callback=self.port_changed)
        self.in_port_name_property.widget.combo_items = self.input_list

        self.output = self.add_output('midi out')
        if self.in_port is not None:
            self.in_port.add_client(self, code=None)

    def receive_midi_bytes(self, midi_bytes):
        self.output.send(midi_bytes)

    def port_changed(self):
        self.in_port_name = self.in_port_name_property()
        super().port_changed()
        self.in_port_name_property.set(self.in_port_name)

    def custom_cleanup(self):
        if self.in_port is not None:
            self.in_port.remove_client(self)


class MidiMessageInNode(MidiIn, Node):
    def __init__(self, label: str, data, args):
        MidiIn.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.codes = [None]
        self.channel = 1
        self.controller = 0     # only for controller in
        self.channel_property = None
        self.in_port_name_property = None
        self.output = None

    def finish_init(self, out_name):
        self.create_channel_and_port()

        self.output = self.add_output(out_name)
        self.codes = self.make_codes()
        self.attach_to_port()

    def create_channel_and_port(self):
        self.channel_property = self.add_option('channel', widget_type='input_int', default_value=self.channel, min=1,
                                                max=16,
                                                callback=self.params_changed)
        name = ''
        if self.in_port is not None and self.in_port.port is not None:
            name = self.in_port.port.name
        self.in_port_name_property = self.add_string_input('port', widget_type='combo', widget_width=200, default_value=name,
                                                    callback=self.port_changed)
        self.in_port_name_property.widget.combo_items = self.input_list

    def make_codes(self):
        return [None]

    def params_changed(self):
        self.channel = self.channel_property()
        new_codes = self.make_codes()
        if new_codes[0] != self.codes[0]:
            # Same guard: this runs on patch load whenever a saved channel
            # differs from the default, with or without a port present.
            if self.in_port is not None:
                for code in self.codes:
                    self.in_port.remove_client(self, code=code)
            self.codes = new_codes
            if self.in_port is not None:
                for code in self.codes:
                    self.in_port.add_client(self, code=code)

    def receive_midi_bytes(self, midi_bytes):
        pass

    def constrain_channel(self, channel):
        if channel > 16:
            channel = 16
        elif channel < 1:
            channel = 1
        return channel

    def port_changed(self):
        if self.in_port_name_property is not None:
            self.in_port_name = self.in_port_name_property()
            super().port_changed()
            self.in_port_name_property.set(self.in_port_name)

    def attach_to_port(self):
        if self.in_port:
            for code in self.codes:
                self.in_port.add_client(self, code=code)

    def custom_cleanup(self):
        if self.in_port:
            for code in self.codes:
                self.in_port.remove_client(self, code=code)


class MidiControllerInNode(MidiMessageInNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiControllerInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.controller = 64
        control_found = False

        for i in range(len(args or [])):
            val, t = decode_arg(args, i)
            if t == int:
                if control_found:
                    self.channel = self.constrain_channel(val)
                else:
                    controller = val
                    if controller > 127:
                        controller = 127
                    elif controller < 0:
                        controller = 0
                    self.controller = controller
                    control_found = True

        self.control_number = self.add_input('controller #', widget_type='input_int', default_value=self.controller, min=0,
                                             max=127, callback=self.controller_changed)
        self.finish_init('controller out')

    def controller_changed(self):
        self.controller = self.control_number()
        self.params_changed()

    def make_codes(self):
        return [(controller_code + self.channel - 1) * 256 + self.controller]

    def receive_midi_bytes(self, midi_bytes):
        self.output.send(midi_bytes[2])


class MidiSingleParamInNode(MidiMessageInNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiSingleParamInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        for i in range(len(args or [])):
            val, t = decode_arg(args, i)
            if t == int:
                self.channel = self.constrain_channel(val)


class MidiPitchBendInNode(MidiSingleParamInNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiPitchBendInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.finish_init('pitch bend out')

    def make_codes(self):
        return [(pitch_bend_code + self.channel - 1) * 256 + 128]

    def receive_midi_bytes(self, midi_bytes):
        self.output.send(midi_bytes[1] + midi_bytes[2] * 128)


class MidiProgramInNode(MidiSingleParamInNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiProgramInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.finish_init('program out')

    def make_codes(self):
        return [(program_code + self.channel - 1) * 256 + 128]

    def receive_midi_bytes(self, midi_bytes):
        self.output.send(midi_bytes[1])


class MidiChannelAfterTouchNode(MidiSingleParamInNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiChannelAfterTouchNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.finish_init('after touch out')

    def make_codes(self):
        return [(aftertouch_code + self.channel - 1) * 256 + 128]

    def receive_midi_bytes(self, midi_bytes):
        self.output.send(midi_bytes[1])


class MidiNoteInNode(MidiMessageInNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiNoteInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        for i in range(len(args or [])):
            val, t = decode_arg(args, i)
            if t == int:
                self.channel = self.constrain_channel(val)
                break

        self.create_channel_and_port()

        self.note_output = self.add_output('note out')
        self.velocity_output = self.add_output('velocity out')
        self.codes = self.make_codes()
        self.attach_to_port()

    def make_codes(self):
        return [(note_on_code + self.channel - 1) * 256 + 128, (note_off_code + self.channel - 1) * 256 + 128]

    def receive_midi_bytes(self, msg_bytes):
        self.velocity_output.send(int(msg_bytes[2]))
        self.note_output.send(int(msg_bytes[1]))


class MidiPolyPressureInNode(MidiMessageInNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiPolyPressureInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        for i in range(len(args or [])):
            val, t = decode_arg(args, i)
            if t == int:
                self.channel = self.constrain_channel(val)
                break

        self.create_channel_and_port()
        self.note_output = self.add_output('note out')
        self.pressure_output = self.add_output('pressure out')
        self.codes = self.make_codes()
        self.attach_to_port()

    def make_codes(self):
        return [(poly_pressure_code + self.channel - 1) * 256 + 128]

    def receive_midi_bytes(self, msg_bytes):
        self.pressure_output.send(int(msg_bytes[2]))
        self.note_output.send(int(msg_bytes[1]))


class MidiOutPort:
    ports = {}

    def __init__(self, name=None):
        self.port_name = name
        self.port = None

        if self.port_name is not None:
            if self.port_name in MidiOutPort.ports:
                self.port = MidiOutPort.ports[self.port_name].port
            else:
                try:
                    self.port = mido.open_output(self.port_name)
                    MidiOutPort.ports[self.port_name] = self
                    print('added out port', self.port_name)
                except Exception as e:
                    print('could not find', self.port_name, 'in', MidiOutPort.ports.keys())
                    self.port = None
        else:
            keys = list(MidiOutPort.ports.keys())
            if len(keys) > 0:
                self.port = MidiOutPort.ports[keys[0]].port
                print('MidiOutPort self-assigned', MidiOutPort.ports[keys[0]].name)
            else:
                try:
                    self.port = mido.open_output()
                except Exception as e:
                    print('MidiOutPort init exception could not find MIDI out port', self.port_name)
                    self.port = None

            if self.port:
                self.port_name = self.port.name
                if self.port_name not in MidiOutPort.ports:
                    MidiOutPort.ports[self.port_name] = self
            else:
                self.port_name = ''

    def send(self, msg):
        if self.port:
            self.port.send(msg)


class MidiOut:
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.out_port_name = None
        self.out_port = None
        self.output_list = []

        if args:
            val, t = decode_arg(args, 0)
            if t == str:
                self.out_port_name = args[0]

        if self.out_port_name is not None:
            if self.out_port_name in MidiOutPort.ports:
                self.out_port = MidiOutPort.ports[self.out_port_name]
            else:
                self.find_out_port_from_partial_name(self.out_port_name)

        if self.out_port is None:
            if len(self.output_list) == 0:
                self.output_list = mido.get_output_names()
            if len(self.output_list) > 0:
                self.out_port_name = self.output_list[0]
                self.out_port = MidiOutPort(self.out_port_name)

    def find_out_port_from_partial_name(self, partial_name):
        if len(self.output_list) == 0:
            self.output_list = mido.get_output_names()
        if len(self.output_list) > 0:
            if partial_name in self.output_list:
                self.out_port_name = partial_name
                if self.out_port_name in MidiOutPort.ports:
                    self.out_port = MidiOutPort.ports[self.out_port_name]
                else:
                    self.out_port = MidiOutPort(self.out_port_name)
                return self.out_port_name
            for output in self.output_list:
                if len(output) > len(partial_name):
                    length = len(partial_name)
                    if output[:length] == partial_name:
                        self.out_port_name = output
                        if self.out_port_name in MidiOutPort.ports:
                            self.out_port = MidiOutPort.ports[self.out_port_name]
                        else:
                            self.out_port = MidiOutPort(self.out_port_name)
                        if self.out_port_name not in MidiOutPort.ports:
                            MidiOutPort.ports[self.out_port_name] = self.out_port
                            print('created out port', self.out_port_name, 'ports:', MidiOutPort.ports.keys())
                        return self.out_port_name

            for output in self.output_list:
                if self.out_port_name in output:
                    self.out_port_name = output
                    self.out_port = MidiOutPort(self.out_port_name)
                    return self.out_port_name
        return None

    def port_changed(self):
        if self.out_port_name in MidiOutPort.ports:
            self.out_port = MidiOutPort.ports[self.out_port_name]
            print('found out port', self.out_port_name, 'in', MidiOutPort.ports.keys(), self.out_port)
            # print(MidiOutPort.ports[self.out_port_name].name)
        else:
            result = self.find_out_port_from_partial_name(self.out_port_name)
        if self.out_port is None:
            print('could not find out port', self.out_port_name)


class MidiOutNode(MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.midi_to_send = self.add_input('midi to send', triggers_execution=True)
        name = self.out_port.port_name if self.out_port is not None else ''
        self.out_port_name_property = self.add_string_input('port', widget_type='combo', widget_width=200, default_value=name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def execute(self):
        raw = self.midi_to_send()
        if raw is None:
            return
        midi_data = any_to_numerical_list(raw)
        if not midi_data:
            return
        _send_out(self.out_port, midi_data)

    def port_changed(self):
        self.out_port_name = self.out_port_name_property()
        super().port_changed()
        self.out_port_name_property.set(self.out_port_name)


class MidiControlOutNode(MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiControlOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        controller = 64
        channel = 1
        control_found = False

        for i in range(len(args or [])):
            val, t = decode_arg(args, i)
            if t == int:
                if control_found:
                    channel = val
                    if channel > 16:
                        channel = 16
                    elif channel < 1:
                        channel = 1
                else:
                    controller = val
                    if controller > 127:
                        controller = 127
                    elif controller < 0:
                        controller = 0
                    control_found = True

        self.midi_to_send = self.add_input('midi to send', triggers_execution=True)
        self.control_number = self.add_input('controller #', widget_type='input_int', default_value=controller, min=0, max=127)
        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16)
        name = self.out_port.port_name if self.out_port is not None else ''
        self.out_port_name_property = self.add_string_input('port', widget_type='combo', widget_width=200, default_value=name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def execute(self):
        raw = self.midi_to_send()
        if raw is None:
            return
        controller_value = any_to_int(raw)
        if controller_value > 127:
            controller_value = 127
        elif controller_value < 0:
            controller_value = 0
        midi_data = [controller_code + self.channel() - 1, self.control_number(), controller_value]
        _send_out(self.out_port, midi_data)

    def port_changed(self):
        self.out_port_name = self.out_port_name_property()
        super().port_changed()
        self.out_port_name_property.set(self.out_port_name)


class MidiPitchBendOutNode(MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiPitchBendOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        channel = 1

        for i in range(len(args or [])):
            val, t = decode_arg(args, i)
            if t == int:
                channel = val
                if channel > 16:
                    channel = 16
                elif channel < 1:
                    channel = 1

        self.midi_to_send = self.add_input('pitchbend to send', triggers_execution=True)
        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16)
        name = self.out_port.port_name if self.out_port is not None else ''
        self.out_port_name_property = self.add_string_input('port', widget_type='combo', widget_width=200, default_value=name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def execute(self):
        raw = self.midi_to_send()
        if raw is None:
            return
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return
        if value < 0:
            value = 0
        elif value > 16383:
            value = 16383
        control_val_msb = value // 128
        control_val_lsb = value % 128
        midi_data = [self.channel() - 1 + pitch_bend_code, control_val_lsb, control_val_msb]
        _send_out(self.out_port, midi_data)

    def port_changed(self):
        self.out_port_name = self.out_port_name_property()
        super().port_changed()
        self.out_port_name_property.set(self.out_port_name)

class MidiProgramOutNode(MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiProgramOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        channel = 1

        for i in range(len(args or [])):
            val, t = decode_arg(args, i)
            if t == int:
                channel = val
                if channel > 16:
                    channel = 16
                elif channel < 1:
                    channel = 1

        self.midi_to_send = self.add_input('program to send', triggers_execution=True)
        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16)
        name = self.out_port.port_name if self.out_port is not None else ''
        self.out_port_name_property = self.add_string_input('port', widget_type='combo', widget_width=200, default_value=name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def execute(self):
        raw = self.midi_to_send()
        if raw is None:
            return
        val = any_to_int(raw)
        if val > 127:
            val = 127
        elif val < 0:
            val = 0
        _send_out(self.out_port, [program_code + self.channel() - 1, val])

    def port_changed(self):
        self.out_port_name = self.out_port_name_property()
        super().port_changed()
        self.out_port_name_property.set(self.out_port_name)


class MidiAftertouchOutNode(MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiAftertouchOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        channel = 1

        for i in range(len(args or [])):
            val, t = decode_arg(args, i)
            if t == int:
                channel = val
                if channel > 16:
                    channel = 16
                elif channel < 1:
                    channel = 1

        self.midi_to_send = self.add_input('aftertouch to send', triggers_execution=True)
        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16)
        name = self.out_port.port_name if self.out_port is not None else ''
        self.out_port_name_property = self.add_string_input('port', widget_type='combo', widget_width=200, default_value=name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def execute(self):
        raw = self.midi_to_send()
        if raw is None:
            return
        val = any_to_int(raw)
        if val > 127:
            val = 127
        elif val < 0:
            val = 0
        _send_out(self.out_port, [aftertouch_code + self.channel() - 1, val])

    def port_changed(self):
        self.out_port_name = self.out_port_name_property()
        super().port_changed()
        self.out_port_name_property.set(self.out_port_name)


class MidiNoteOutNode(MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiNoteOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):

        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        args = args or []
        velocity = 64
        if len(args) > 0:
            velocity = any_to_int(args[0])
            if velocity > 127:
                velocity = 127
            elif velocity < 0:
                velocity = 0
        channel = 1
        if len(args) > 1:
            channel = any_to_int(args[1])
            if channel > 16:
                channel = 16
            elif channel < 1:
                channel = 1


        self.midi_to_send = self.add_input('midi to send', triggers_execution=True)
        self.velocity = self.add_input('velocity', widget_type='drag_int', default_value=velocity, min=0, max=127)
        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16)
        name = self.out_port.port_name if self.out_port is not None else ''
        self.out_port_name_property = self.add_string_input('port', widget_type='combo', widget_width=200, default_value=name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def execute(self):
        # Read the input WITHOUT coercing to int first; the list/tuple branch
        # below was unreachable because the int() call collapsed the value.
        incoming = self.midi_to_send()
        note = 64
        velocity = int(self.velocity())
        channel = int(self.channel())

        if isinstance(incoming, (list, tuple)):
            length = len(incoming)
            if length > 0:
                note = any_to_int(incoming[0])
                if note > 127:
                    note = 127
                if note < 0:
                    note = 0
            if length > 1:
                velocity = any_to_int(incoming[1])
                if velocity > 127:
                    velocity = 127
                if velocity < 0:
                    velocity = 0
            if length > 2:
                channel = any_to_int(incoming[2])
                if channel > 16:
                    channel = 16
                if channel < 1:
                    channel = 1
        else:
            note = any_to_int(incoming)
            if note > 127:
                note = 127
            if note < 0:
                note = 0
        _send_out(self.out_port, [note_on_code + channel - 1, note, velocity])

    def port_changed(self):
        self.out_port_name = self.out_port_name_property()
        super().port_changed()
        self.out_port_name_property.set(self.out_port_name)


class MidiPolyPressureOutNode(MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiPolyPressureOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):

        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        args = args or []
        pressure = 64
        if len(args) > 0:
            pressure = any_to_int(args[0])
            if pressure > 127:
                pressure = 127
            elif pressure < 0:
                pressure = 0
        channel = 1
        if len(args) > 1:
            channel = any_to_int(args[1])
            if channel > 16:
                channel = 16
            elif channel < 1:
                channel = 1


        self.midi_to_send = self.add_input('midi to send', triggers_execution=True)
        self.pressure = self.add_input('pressure', widget_type='drag_int', default_value=pressure, min=0, max=127)
        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16)
        name = self.out_port.port_name if self.out_port is not None else ''
        self.out_port_name_property = self.add_string_input('port', widget_type='combo', widget_width=200, default_value=name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def execute(self):
        # See note on MidiNoteOutNode.execute: do not coerce to int first.
        incoming = self.midi_to_send()
        note = 64
        pressure = int(self.pressure())
        channel = int(self.channel())

        if isinstance(incoming, (list, tuple)):
            length = len(incoming)
            if length > 0:
                note = any_to_int(incoming[0])
                if note > 127:
                    note = 127
                if note < 0:
                    note = 0
            if length > 1:
                pressure = any_to_int(incoming[1])
                if pressure > 127:
                    pressure = 127
                if pressure < 0:
                    pressure = 0
            if length > 2:
                channel = any_to_int(incoming[2])
                if channel > 16:
                    channel = 16
                if channel < 1:
                    channel = 1
        else:
            note = any_to_int(incoming)
            if note > 127:
                note = 127
            if note < 0:
                note = 0
        _send_out(self.out_port, [poly_pressure_code + channel - 1, note, pressure])

    def port_changed(self):
        self.out_port_name = self.out_port_name_property()
        super().port_changed()
        self.out_port_name_property.set(self.out_port_name)


class MidiDeviceNode(MidiIn, MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiDeviceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiIn.__init__(self, label, data, args)
        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        channel = 1
        for i in range(len(args or [])):
            val, t = decode_arg(args, i)
            if t == int:
                channel = val
                if channel > 16:
                    channel = 16
                elif channel < 1:
                    channel = 1

        self.create_properties_inputs_and_outputs()
        # in_port is None when no MIDI input exists -- which is the normal case
        # away from the hardware. The lines just below already guard this way;
        # unguarded here, midi_device, mpd218 and blue_board could not be
        # created at all with nothing plugged in.
        if self.in_port is not None:
            self.in_port.add_client(self, code=None)
        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16)
        port_name = ''
        if self.in_port:
            port_name = self.in_port.port_name
        self.in_port_name_property = self.add_string_input('in port', widget_type='combo', widget_width=200, default_value=port_name, callback=self.port_changed)
        self.in_port_name_property.widget.combo_items = self.input_list
        port_name = ''
        if self.out_port:
            port_name = self.out_port.port_name
        self.out_port_name_property = self.add_string_input('out port', widget_type='combo', widget_width=200, default_value=port_name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def create_properties_inputs_and_outputs(self):
        self.add_input('midi to send', triggers_execution=True)
        self.add_output('midi received')

    def receive_midi_bytes(self, midi_bytes):
        if len(self.outputs) > 0:
           self.outputs[0].send(midi_bytes)

    def execute(self):
        if len(self.inputs) > 0:
            raw = self.inputs[0]()
            if raw is None:
                return
            midi_data = any_to_numerical_list(raw)
            if not midi_data:
                return
            _send_out(self.out_port, midi_data)

    def port_changed(self):
        self.in_port_name = self.in_port_name_property()
        self.out_port_name = self.out_port_name_property()
        MidiOut.port_changed(self)
        MidiIn.port_changed(self)
        self.in_port_name_property.set(self.in_port_name)
        self.out_port_name_property.set(self.out_port_name)

    def custom_cleanup(self):
        # Guarded like the other two custom_cleanup methods in this file:
        # in_port is None with no MIDI input present, and an exception here
        # leaves the patch half-closed.
        if self.in_port is not None:
            self.in_port.remove_client(self)


class BlueBoardNode(MidiDeviceNode):
    @staticmethod
    def factory(name, data, args=None):
        node = BlueBoardNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        force_args = ['iRig BlueBoard Bluetooth']
        super().__init__(label, data, force_args)

        self.set_LED_inputs = []
        self.modes = []

        self.add_output('A')
        self.modes.append(self.add_property('###A_mode', widget_type='combo', default_value='momentary'))
        self.set_LED_inputs.append(self.add_input('LED', widget_type='checkbox', triggers_execution=True))
        self.add_spacer()
        self.add_output('B')
        self.modes.append(self.add_property('###B_mode', widget_type='combo', default_value='momentary'))
        self.set_LED_inputs.append(self.add_input('LED', widget_type='checkbox', triggers_execution=True))
        self.add_spacer()
        self.add_output('C')
        self.modes.append(self.add_property('###C_mode', widget_type='combo', default_value='momentary'))
        self.set_LED_inputs.append(self.add_input('LED', widget_type='checkbox', triggers_execution=True))
        self.add_spacer()
        self.add_output('D')
        self.modes.append(self.add_property('###D_mode', widget_type='combo', default_value='momentary'))
        self.set_LED_inputs.append(self.add_input('LED', widget_type='checkbox', triggers_execution=True))

        self.states = [0, 0, 0, 0]

        for mode in self.modes:
            mode.widget.combo_items = ['toggle', 'momentary', 'raw']

    def create_properties_inputs_and_outputs(self):
        pass

    def receive_midi_bytes(self, midi_bytes):
        if len(midi_bytes) < 3:
            return
        state = midi_bytes[2]
        if state == 127:
            state = 1
        which = midi_bytes[1] - 20
        if which < 0 or which >= len(self.modes):
            return
        mode = self.modes[which]()
        if mode == 'momentary':
            self.states[which] = state
            self.set_LED(which, state)
        elif mode == 'toggle':
            if state == 1:
                self.states[which] = 1 - self.states[which]
                self.set_LED(which, self.states[which])
            else:
                return
        else:
            self.states[which] = state
        self.outputs[which].send(self.states[which])

    def set_LED(self, which, state):
        _send_out(self.out_port, [controller_code, which + 20, state * 127])
        self.set_LED_inputs[which].set(state)

    def execute(self):
        out = 0
        controller = 20

        for which, set_LED in enumerate(self.set_LED_inputs):
            # Compare by identity, not equality. NodeInput's __eq__ semantics
            # don't behave like identity here, so == would never match and
            # this loop body never executed.
            if set_LED is self.active_input:
                if self.modes[which]() in ['toggle', 'momentary']:
                    if self.states[which] != self.active_input():
                        self.states[which] = self.active_input()
                        self.outputs[which].send(self.states[which])
                out = self.active_input() * 127
                controller = 20 + which
                break

        _send_out(self.out_port, [controller_code, controller, out])


class MPD218Node(MidiDeviceNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MPD218Node(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        force_args = ['MPD218']
        super().__init__(label, data, force_args)

        self.select_in = self.add_input('select', triggers_execution=True)
        self.pad_out = self.add_output('pad')
        self.controller_out = self.add_output('controller')

        self.last_pad = -1
        self.active_pad = -1
        self.disable_all()

    def create_properties_inputs_and_outputs(self):
        pass

    def receive_midi_bytes(self, midi_bytes):
        if len(midi_bytes) < 3:
            return
        sys_byte = midi_bytes[0]
        if sys_byte & 0xF0 == 0x90 or sys_byte & 0xF0 == 0x80:
            note_byte = midi_bytes[1]
            velocity_byte = midi_bytes[2]
            if velocity_byte > 0:
                self.disable_pressed()
                self.active_pad = note_byte
                self.enable(note_byte)
            else:
                if note_byte == self.active_pad:
                    self.enable(note_byte)
        elif sys_byte & 0xF0 == 0xB0:
            controller_code_byte = midi_bytes[1]
            controller_value = midi_bytes[2]
            self.controller_out.send([controller_code_byte, controller_value])

    def disable_all(self):
        for pad in range(16):
            _send_out(self.out_port, [0x80, pad, 0])

    def disable_pressed(self):
        if self.last_pad != -1:
            pad = self.last_pad
            self.last_pad = -1
            _send_out(self.out_port, [0x80, pad, 0])

    def enable(self, pad):
        self.last_pad = pad
        _send_out(self.out_port, [0x90, pad, 127])
        self.pad_out.send(pad)

    def execute(self):
        self.disable_all()
        selection = self.select_in()
        if selection is None:
            return
        try:
            self.enable(int(selection))
        except (TypeError, ValueError):
            return




class MPEInNode(MidiIn, Node):
    """mpe_in - a multi-touch controller's MIDI as fingers.

    MPE gives every finger its own MIDI channel: the note says where it landed,
    pitch bend says how far it has slid sideways since, controller 74 says how
    far up, and channel pressure says how hard. This node keeps one voice per
    channel and hands each finger out as a list, so an MPE pad (Erae, Linnstrument,
    Seaboard...) reads the way a touch surface should rather than as sixteen
    keyboards. A plain single-channel controller works the same way: its one
    channel is voice 0.

    Expressive controllers stream bend and pressure at ~100 Hz on every finger.
    With 'coalesce' on, the node collects those on the MIDI thread and sends one
    'move' per finger per frame from the main thread; note on/off are never
    dropped.
    """
    STATE_DOWN, STATE_MOVE, STATE_UP = 0, 1, 2

    @staticmethod
    def factory(name, data, args=None):
        node = MPEInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiIn.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        # per channel (0..15): note, velocity, bend (semitones), pressure, slide, active
        self.notes = np.zeros(16, dtype=np.int32)
        self.velocities = np.zeros(16, dtype=np.float32)
        self.bends = np.zeros(16, dtype=np.float32)
        self.pressures = np.zeros(16, dtype=np.float32)
        self.slides = np.zeros(16, dtype=np.float32)
        self.active = np.zeros(16, dtype=bool)
        self.master_bend = 0.0
        self.lock = threading.Lock()
        self.pending_events = []            # (state, channel) - note on/off, in order
        self.dirty = np.zeros(16, dtype=bool)
        self.any_change = False

        name = ''
        if self.in_port is not None and self.in_port.port is not None:
            name = self.in_port.port.name
        self.in_port_name_property = self.add_string_input('port', widget_type='combo', widget_width=200, default_value=name, callback=self.port_changed)
        self.in_port_name_property.widget.combo_items = self.input_list

        self.touch_out = self.add_output('touch')
        self.voices_out = self.add_output('voices')
        self.count_out = self.add_output('count')
        self.master_out = self.add_output('master')

        self.mode = self.add_option('mode', widget_type='combo', default_value='all channels', callback=self.mode_changed)
        self.mode.widget.combo_items = ['all channels', 'mpe lower', 'mpe upper']
        self.bend_range = self.add_option('bend range', widget_type='drag_float', default_value=48.0, min=0.0, max=96.0)
        self.master_bend_range = self.add_option('master bend range', widget_type='drag_float', default_value=2.0, min=0.0, max=96.0)
        self.slide_cc = self.add_option('slide cc', widget_type='input_int', default_value=74, min=0, max=127)
        self.normalize = self.add_option('normalize', widget_type='checkbox', default_value=True)
        self.coalesce = self.add_option('coalesce', widget_type='checkbox', default_value=True)

        self.master_channel = -1
        if self.in_port is not None:
            self.in_port.add_client(self, code=None)
        self.add_frame_task()

    def mode_changed(self):
        mode = self.mode()
        self.master_channel = {'mpe lower': 0, 'mpe upper': 15}.get(mode, -1)

    def port_changed(self):
        self.in_port_name = self.in_port_name_property()
        super().port_changed()
        self.in_port_name_property.set(self.in_port_name)

    def custom_cleanup(self):
        self.remove_frame_tasks()
        if self.in_port is not None:
            self.in_port.remove_client(self)

    # ------------------------------------------------------------ MIDI thread
    def receive_midi_bytes(self, midi_bytes):
        if len(midi_bytes) < 2:
            return
        status = midi_bytes[0]
        if status >= 0xF0:
            return
        kind = status & 0xF0
        ch = status & 0x0F
        if ch == self.master_channel:
            self.receive_master(kind, midi_bytes)
            return

        with self.lock:
            if kind == 0x90 and len(midi_bytes) >= 3 and midi_bytes[2] > 0:
                self.notes[ch] = midi_bytes[1]
                self.velocities[ch] = midi_bytes[2]
                self.active[ch] = True
                self.pending_events.append((self.STATE_DOWN, ch))
            elif kind == 0x80 or (kind == 0x90 and len(midi_bytes) >= 3):
                if self.active[ch] and self.notes[ch] == midi_bytes[1]:
                    self.active[ch] = False
                    self.velocities[ch] = midi_bytes[2] if len(midi_bytes) >= 3 else 0
                    self.pending_events.append((self.STATE_UP, ch))
            elif kind == 0xE0 and len(midi_bytes) >= 3:
                raw = (midi_bytes[2] << 7 | midi_bytes[1]) - 8192
                self.bends[ch] = raw / 8192.0 * float(self.bend_range())
                self.dirty[ch] = True
            elif kind == 0xD0:
                self.pressures[ch] = midi_bytes[1]
                self.dirty[ch] = True
            elif kind == 0xA0 and len(midi_bytes) >= 3:
                self.pressures[ch] = midi_bytes[2]
                self.dirty[ch] = True
            elif kind == 0xB0 and len(midi_bytes) >= 3 and midi_bytes[1] == self.slide_cc():
                self.slides[ch] = midi_bytes[2]
                self.dirty[ch] = True
            else:
                return
            self.any_change = True

        if not self.coalesce():
            self.flush()

    def receive_master(self, kind, midi_bytes):
        if kind == 0xE0 and len(midi_bytes) >= 3:
            raw = (midi_bytes[2] << 7 | midi_bytes[1]) - 8192
            self.master_bend = raw / 8192.0 * float(self.master_bend_range())
            with self.lock:
                self.dirty[:] = self.active
                self.any_change = True
            if not self.coalesce():
                self.flush()
        self.master_out.send(list(midi_bytes))

    # ------------------------------------------------------------ main thread
    def frame_task(self):
        if self.any_change and self.coalesce():
            self.flush()

    def flush(self):
        with self.lock:
            if not self.any_change:
                return
            events = self.pending_events
            self.pending_events = []
            dirty = self.dirty.copy()
            self.dirty[:] = False
            self.any_change = False
            active = self.active.copy()
            notes = self.notes.copy()
            vel = self.velocities.copy()
            pitch = notes + self.bends + self.master_bend
            pressure = self.pressures.copy()
            slide = self.slides.copy()

        if self.normalize():
            vel = vel / 127.0
            pressure = pressure / 127.0
            slide = slide / 127.0

        for state, ch in events:
            self.touch_out.send([int(ch), state, int(notes[ch]), float(pitch[ch]), float(pressure[ch]), float(slide[ch]), float(vel[ch])])
            dirty[ch] = False
        for ch in np.nonzero(dirty & active)[0]:
            self.touch_out.send([int(ch), self.STATE_MOVE, int(notes[ch]), float(pitch[ch]), float(pressure[ch]), float(slide[ch]), float(vel[ch])])

        voices = np.stack([active.astype(np.float32), notes.astype(np.float32), pitch.astype(np.float32),
                           pressure.astype(np.float32), slide.astype(np.float32)], axis=1)
        self.voices_out.send(voices)
        self.count_out.send(int(active.sum()))
