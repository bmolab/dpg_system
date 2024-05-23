import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import mido


def register_midi_nodes():
    Node.app.register_node('midi_in', MidiInNode.factory)
    Node.app.register_node('midi_control_in', MidiControllerInNode.factory)
    Node.app.register_node('midi_pitchbend_in', MidiPitchBendInNode)
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


note_off_code = 128
note_on_code = 144
poly_pressure_code = 160
controller_code = 176
program_code = 192
aftertouch_code = 208
pitch_bend_code = 224


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
                except Exception as e:
                    print('could not find port', self.port_name)
                    self.port = None
        else:
            keys = list(MidiInPort.ports.keys())
            if len(keys) > 0:
                self.port = MidiInPort.ports[keys[0]].port
            else:
                try:
                    self.port = mido.open_input(callback=self.receive)
                except Exception as e:
                    print('could not find MIDI in port')
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

        if len(args) > 0:
            val, t = decode_arg(args, 0)
            if t == str:
                self.in_port_name = args[0]

        if self.in_port_name is not None:
            if self.in_port_name in MidiInPort.ports:
                self.in_port = MidiInPort.ports[self.in_port_name]
            else:
                self.in_port = MidiInPort(self.in_port_name)
        else:
            keys = list(MidiInPort.ports.keys())
            if len(keys) > 0:
                self.in_port = MidiInPort.ports[keys[0]]
            else:
                self.in_port = MidiInPort(self.in_port_name)
            self.in_port_name = self.in_port.port_name
        self.input_list = mido.get_input_names()

    def receive_midi_bytes(self, midi_bytes):
        pass

    def port_changed(self):
        if self.in_port is not None:
            for code in self.codes:
                self.in_port.remove_client(self, code=code)
        if self.in_port_name in MidiInPort.ports:
            self.in_port = MidiInPort.ports[self.in_port_name]
            print('found in port')
        else:
            self.in_port = MidiInPort(self.in_port_name)
            print('created in port')
        if self.in_port is not None:
            for code in self.codes:
                self.in_port.add_client(self, code=code)
        else:
            print('no in port')


class MidiInNode(MidiIn, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiIn.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.in_port_name_property = self.add_input('port', widget_type='combo', default_value=self.in_port.port.name, callback=self.port_changed)
        self.in_port_name_property.widget.combo_items = self.input_list

        self.output = self.add_output('midi out')
        self.in_port.add_client(self, code=None)

    def receive_midi_bytes(self, midi_bytes):
        self.output.send(midi_bytes)

    def port_changed(self):
        self.in_port_name = self.in_port_name_property()
        super().port_changed()

    def custom_cleanup(self):
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

        self.in_port_name_property = self.add_input('port', widget_type='combo', default_value=self.in_port.port.name,
                                                    callback=self.port_changed)
        self.in_port_name_property.widget.combo_items = self.input_list

    def make_codes(self):
        return [None]

    def params_changed(self):
        self.channel = self.channel_property()
        new_codes = self.make_codes()
        if new_codes[0] != self.codes[0]:
            for code in self.codes:
                self.in_port.remove_client(self, code=code)
            self.codes = new_codes
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

        for i in range(len(args)):
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

        for i in range(len(args)):
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

        for i in range(len(args)):
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

        for i in range(len(args)):
            val, t = decode_arg(args, i)
            if t == int:
                self.channel = self.constrain_channel(val)
                break

        self.create_channel_and_port()
        self.note_output = self.add_output('note out')
        self.pressure_output = self.add_output('pressure out')
        self.code = self.make_codes()
        self.attach_to_port()

    def make_codes(self):
        return (poly_pressure_code + self.channel - 1) * 256 + 128

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
                except Exception as e:
                    print('could not find', self.port_name)
                    self.port = None
        else:
            keys = list(MidiOutPort.ports.keys())
            if len(keys) > 0:
                self.port = MidiOutPort.ports[keys[0]].port
            else:
                try:
                    self.port = mido.open_output()
                except Exception as e:
                    print('could not find MIDI out port')
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

        if len(args) > 0:
            val, t = decode_arg(args, 0)
            if t == str:
                self.out_port_name = args[0]

        if self.out_port_name is not None:
            if self.out_port_name in MidiOutPort.ports:
                self.out_port = MidiOutPort.ports[self.out_port_name]
            else:
                self.out_port = MidiOutPort(self.out_port_name)
        else:
            keys = list(MidiOutPort.ports.keys())
            if len(keys) > 0:
                self.out_port = MidiOutPort.ports[keys[0]]
            else:
                self.out_port = MidiOutPort(self.out_port_name)
            self.out_port_name = self.out_port.port_name
        self.output_list = mido.get_output_names()

    def port_changed(self):
        if self.out_port_name in MidiOutPort.ports:
            self.out_port = MidiOutPort.ports[self.out_port_name]
            print('found out port')
        else:
            self.out_port = MidiOutPort(self.out_port_name)
            print('created out port')


class MidiOutNode(MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.midi_to_send = self.add_input('midi to send', triggers_execution=True)
        self.out_port_name_property = self.add_input('port', widget_type='combo', default_value=self.out_port.port_name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def execute(self):
        midi_data = any_to_numerical_list(self.midi_to_send())
        msg = mido.Message.from_bytes(midi_data)
        self.out_port.send(msg)

    def port_changed(self):
        self.out_port_name = self.out_port_name_property()
        super().port_changed()


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

        for i in range(len(args)):
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
        self.out_port_name_property = self.add_option('port', widget_type='combo', default_value=self.out_port.port_name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def execute(self):
        controller_value = any_to_int(self.midi_to_send())
        if controller_value > 127:
            controller_value = 127
        elif controller_value < 0:
            controller_value = 0
        midi_data = [controller_code + self.channel() - 1, self.control_number(), controller_value]
        msg = mido.Message.from_bytes(midi_data)
        self.out_port.send(msg)

    def port_changed(self):
        self.out_port_name = self.out_port_name_property()
        super().port_changed()


class MidiPitchBendOutNode(MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiPitchBendOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        channel = 1

        for i in range(len(args)):
            val, t = decode_arg(args, i)
            if t == int:
                channel = val
                if channel > 16:
                    channel = 16
                elif channel < 1:
                    channel = 1

        self.midi_to_send = self.add_input('pitchbend to send', triggers_execution=True)
        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16)
        self.out_port_name_property = self.add_option('port', widget_type='combo', default_value=self.out_port.port_name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def execute(self):
        control_val_msb = int(self.midi_to_send() / 128)
        control_val_lsb = int(self.midi_to_send() % 128)
        midi_data = [self.channel() - 1 + pitch_bend_code, control_val_lsb, control_val_msb]
        msg = mido.Message.from_bytes(midi_data)
        self.out_port.send(msg)

    def port_changed(self):
        self.out_port_name = self.out_port_name_property()
        super().port_changed()


class MidiProgramOutNode(MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiProgramOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        channel = 1

        for i in range(len(args)):
            val, t = decode_arg(args, i)
            if t == int:
                channel = val
                if channel > 16:
                    channel = 16
                elif channel < 1:
                    channel = 1

        self.midi_to_send = self.add_input('program to send', triggers_execution=True)
        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16)
        self.out_port_name_property = self.add_option('port', widget_type='combo', default_value=self.out_port.port_name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def execute(self):
        val = self.midi_to_send()
        if val > 127:
            val = 127
        elif val < 0:
            val = 0
        midi_data = [program_code + self.channel() - 1, val]
        msg = mido.Message.from_bytes(midi_data)
        self.out_port.send(msg)

    def port_changed(self):
        self.out_port_name = self.out_port_name_property()
        super().port_changed()


class MidiAftertouchOutNode(MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiAftertouchOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        channel = 1

        for i in range(len(args)):
            val, t = decode_arg(args, i)
            if t == int:
                channel = val
                if channel > 16:
                    channel = 16
                elif channel < 1:
                    channel = 1

        self.midi_to_send = self.add_input('aftertouch to send', triggers_execution=True)
        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16)
        self.out_port_name_property = self.add_option('port', widget_type='combo',
                                                      default_value=self.out_port.port_name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def execute(self):
        val = self.midi_to_send()
        if val > 127:
            val = 127
        elif val < 0:
            val = 0
        midi_data = [aftertouch_code + self.channel() - 1, val]
        msg = mido.Message.from_bytes(midi_data)
        self.out_port.send(msg)

    def port_changed(self):
        self.out_port_name = self.out_port_name_property()
        super().port_changed()


class MidiNoteOutNode(MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiNoteOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):

        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

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
        self.out_port_name_property = self.add_option('port', widget_type='combo', default_value=self.out_port.port_name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def execute(self):
        input = int(self.midi_to_send())
        note = 64
        velocity = int(self.velocity())
        channel = int(self.channel())

        if type(input) in [list, tuple]:
            length = len(input)
            if length > 0:
                note = any_to_int(input[0])
                if note > 127:
                    note = 127
                if note < 0:
                    note = 0
            if length > 1:
                velocity = any_to_int(input[1])
                if velocity > 127:
                    velocity = 127
                if velocity < 0:
                    velocity = 0
            if length > 2:
                channel = any_to_int(input[2])
                if channel > 16:
                    channel = 16
                if channel < 1:
                    channel = 1
        else:
            note = any_to_int(input)
            if note > 127:
                note = 127
            if note < 0:
                note = 0
        midi_data = [note_on_code + channel - 1, note, velocity]
        msg = mido.Message.from_bytes(midi_data)
        self.out_port.send(msg)

    def port_changed(self):
        self.out_port_name = self.out_port_name_property()
        super().port_changed()


class MidiPolyPressureOutNode(MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiPolyPressureOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):

        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

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
        self.out_port_name_property = self.add_option('port', widget_type='combo', default_value=self.out_port.port_name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def execute(self):
        input = int(self.midi_to_send())
        note = 64
        pressure = int(self.pressure())
        channel = int(self.channel())

        if type(input) in [list, tuple]:
            length = len(input)
            if length > 0:
                note = any_to_int(input[0])
                if note > 127:
                    note = 127
                if note < 0:
                    note = 0
            if length > 1:
                pressure = any_to_int(input[1])
                if pressure > 127:
                    pressure = 127
                if pressure < 0:
                    pressure = 0
            if length > 2:
                channel = any_to_int(input[2])
                if channel > 16:
                    channel = 16
                if channel < 1:
                    channel = 1
        else:
            note = any_to_int(input)
            if note > 127:
                note = 127
            if note < 0:
                note = 0
        midi_data = [poly_pressure_code + channel - 1, note, pressure]
        msg = mido.Message.from_bytes(midi_data)
        self.out_port.send(msg)

    def port_changed(self):
        self.out_port_name = self.out_port_name_property()
        super().port_changed()


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
        for i in range(len(args)):
            val, t = decode_arg(args, i)
            if t == int:
                channel = val
                if channel > 16:
                    channel = 16
                elif channel < 1:
                    channel = 1

        self.create_properties_inputs_and_outputs()
        self.in_port.add_client(self, code=None)
        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16)
        port_name = ''
        if self.in_port:
            port_name = self.in_port.port_name
        self.in_port_name_property = self.add_option('in port', widget_type='combo', default_value=port_name, callback=self.port_changed)
        self.in_port_name_property.widget.combo_items = self.input_list
        port_name = ''
        if self.out_port:
            port_name = self.out_port.port_name
        self.out_port_name_property = self.add_option('out port', widget_type='combo', default_value=port_name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def create_properties_inputs_and_outputs(self):
        self.add_input('midi to send', triggers_execution=True)
        self.add_output('midi received')

    def receive_midi_bytes(self, midi_bytes):
        if len(self.outputs) > 0:
           self.outputs[0].send(midi_bytes)

    def execute(self):
        if len(self.inputs) > 0:
            midi_data = any_to_numerical_list(self.inputs[0]())
            msg = mido.Message.from_bytes(midi_data)
            self.out_port.send(msg)

    def port_changed(self):
        self.in_port_name = self.in_port_name_property()
        self.out_port_name = self.out_port_name_property()
        MidiOut.port_changed(self)
        MidiIn.port_changed(self)

    def custom_cleanup(self):
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
        state = midi_bytes[2]
        if state == 127:
            state = 1
        which = midi_bytes[1] - 20
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
        midi_data = [controller_code, which + 20, state * 127]
        msg = mido.Message.from_bytes(midi_data)
        self.out_port.send(msg)
        self.set_LED_inputs[which].set(state)

    def execute(self):
        out = 0
        controller = 20

        for which, set_LED in enumerate(self.set_LED_inputs):
            if set_LED == self.active_input:
                if self.modes[which]() in ['toggle', 'momentary']:
                    if self.states[which] != self.active_input():
                        self.states[which] = self.active_input()
                        self.outputs[which].send(self.states[which])
                out = self.active_input() * 127
                controller = 20 + which
                break

        midi_data = [controller_code, controller, out]
        msg = mido.Message.from_bytes(midi_data)
        self.out_port.send(msg)


class MPD218Node(MidiDeviceNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MPD218Node(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        force_args = ['MPD218 Port A']
        super().__init__(label, data, force_args)

        self.pad_out = self.add_output('pad')
        self.controller_out = self.add_output('controller')

        self.last_pad = -1
        self.active_pad = -1

    def create_properties_inputs_and_outputs(self):
        pass

    def receive_midi_bytes(self, midi_bytes):
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
            controller_code = midi_bytes[1]
            controller_value = midi_bytes[2]
            self.controller_out.send([controller_code, controller_value])


    def disable_pressed(self):
        if self.last_pad != -1:
            midi_data = [0x80, self.last_pad, 0]
            self.last_pad = -1
            msg = mido.Message.from_bytes(midi_data)
            self.out_port.send(msg)

    def enable(self, pad):
        midi_data = [0x90, pad, 127]
        self.last_pad = pad
        msg = mido.Message.from_bytes(midi_data)
        self.out_port.send(msg)
        self.pad_out.send(pad)


