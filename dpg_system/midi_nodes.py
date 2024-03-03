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

    Node.app.register_node('blue_board', BlueBoardNode.factory)


class MidiInPort:
    ports = {}

    def __init__(self, name=None):
        self.port_name = name
        self.port = None

        self.clients = {}
        self.general_clients = []

        if self.port_name is not None:
            if self.port_name in MidiInPort.ports:
                self.port = MidiInPort.ports[self.port_name]
            else:
                self.port = mido.open_input(self.port_name, callback=self.receive)
                MidiInPort.ports[self.port_name] = self.port
        else:
            keys = list(MidiInPort.ports.keys())
            if len(keys) > 0:
                self.port = MidiInPort.ports[keys[0]]
            else:
                self.port = mido.open_input(callback=self.receive)
            self.port_name = self.port.name
            self.ports[self.port_name] = self

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
        if self.in_port_name in MidiIn.ports:
            self.in_port = MidiInPort.ports[self.in_port_name]
        else:
            self.in_port = MidiInPort(self.in_port_name)


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


class MidiControllerInNode(MidiIn, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiControllerInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiIn.__init__(self, label, data, args)
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

        self.control_number = self.add_input('controller #', widget_type='input_int', default_value=controller, min=0, max=127, callback=self.params_changed)
        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16, callback=self.params_changed)

        self.in_port_name_property = self.add_input('port', widget_type='combo', default_value=self.in_port.port.name, callback=self.port_changed)
        self.in_port_name_property.widget.combo_items = self.input_list

        self.output = self.add_output('controller out')
        self.code = self.make_code(channel, controller)
        if self.in_port:
            self.in_port.add_client(self, code=self.code)

    def make_code(self, channel, controller):
        return (176 + channel - 1) * 256 + controller

    def params_changed(self):
        new_code = self.make_code(self.channel(), self.control_number())
        if new_code != self.code:
            self.in_port.remove_client(self, code=self.code)
            self.code = new_code
            self.in_port.add_client(self, self.code)

    def receive_midi_bytes(self, midi_bytes):
        self.output.send(midi_bytes[2])

    def port_changed(self):
        self.in_port_name = self.in_port_name_property()
        super().port_changed()

    def custom_cleanup(self):
        self.in_port.remove_client(self, code=self.code)


class MidiControllerInNode(MidiIn, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiControllerInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiIn.__init__(self, label, data, args)
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

        self.control_number = self.add_input('controller #', widget_type='input_int', default_value=controller, min=0, max=127, callback=self.params_changed)
        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16, callback=self.params_changed)

        self.in_port_name_property = self.add_input('port', widget_type='combo', default_value=self.in_port.port.name, callback=self.port_changed)
        self.in_port_name_property.widget.combo_items = self.input_list

        self.output = self.add_output('midi out')
        self.code = self.make_code(channel, controller)
        if self.in_port:
            self.in_port.add_client(self, code=self.code)

    def make_code(self, channel, controller):
        return (176 + channel - 1) * 256 + controller

    def params_changed(self):
        new_code = self.make_code(self.channel(), self.control_number())
        if new_code != self.code:
            self.in_port.remove_client(self, code=self.code)
            self.code = new_code
            self.in_port.add_client(self, self.code)

    def receive_midi_bytes(self, midi_bytes):
        self.output.send(midi_bytes[2])

    def port_changed(self):
        self.in_port_name = self.in_port_name_property()
        super().port_changed()

    def custom_cleanup(self):
        self.in_port.remove_client(self, code=self.code)


class MidiSingleParamInNode(MidiIn, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiSingleParamInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiIn.__init__(self, label, data, args)
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

        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16,
                                       callback=self.params_changed)

        self.in_port_name_property = self.add_input('port', widget_type='combo', default_value=self.in_port.port.name,
                                                    callback=self.port_changed)
        self.in_port_name_property.widget.combo_items = self.input_list

        self.output = self.add_output('out')
        self.code = self.make_code(channel)
        if self.in_port:
            self.in_port.add_client(self, code=self.code)

    def make_code(self, channel):
        return (224 + channel - 1) * 256 + 128

    def params_changed(self):
        new_code = self.make_code(self.channel())
        if new_code != self.code:
            self.in_port.remove_client(self, code=self.code)
            self.code = new_code
            self.in_port.add_client(self, self.code)

    def receive_midi_bytes(self, midi_bytes):
        pass

    def port_changed(self):
        self.in_port_name = self.in_port_name_property()
        super().port_changed()

    def custom_cleanup(self):
        self.in_port.remove_client(self, code=self.code)


class MidiPitchBendInNode(MidiSingleParamInNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiPitchBendInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def post_creation_callback(self):
        self.output.set_label('pitchbend out')

    def make_code(self, channel):
        return (224 + channel - 1) * 256 + 128

    def receive_midi_bytes(self, midi_bytes):
        self.output.send(midi_bytes[1] + midi_bytes[2] * 128)


class MidiProgramInNode(MidiSingleParamInNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiProgramInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def post_creation_callback(self):
        self.output.set_label('program change out')

    def make_code(self, channel):
        return (192 + channel - 1) * 256 + 128

    def receive_midi_bytes(self, midi_bytes):
        self.output.send(midi_bytes[1])


class MidiChannelAfterTouchNode(MidiSingleParamInNode):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiChannelAfterTouchNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def post_creation_callback(self):
        self.output.set_label('after touch out')

    def make_code(self, channel):
        return (208 + channel - 1) * 256 + 128

    def receive_midi_bytes(self, midi_bytes):
        self.output.send(midi_bytes[1])


class MidiNoteInNode(MidiIn, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiNoteInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiIn.__init__(self, label, data, args)
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
                break

        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16, callback=self.params_changed)
        self.in_port_name_property = self.add_input('port', widget_type='combo', default_value=self.in_port.port.name, callback=self.port_changed)
        self.in_port_name_property.widget.combo_items = self.input_list

        self.note_output = self.add_output('note out')
        self.velocity_output = self.add_output('velocity out')
        self.code_on, self.code_off = self.make_codes(channel)

        self.in_port.add_client(self, code=self.code_on)
        self.in_port.add_client(self, code=self.code_off)

    def make_codes(self, channel):
        return (144 + channel - 1) * 256 + 128, (128 + channel - 1) * 256 + 128

    def receive_midi_bytes(self, msg_bytes):
        self.velocity_output.send(int(msg_bytes[2]))
        self.note_output.send(int(msg_bytes[1]))

    def params_changed(self):
        new_code_on, new_code_off = self.make_codes(self.channel())
        if new_code_on != self.code_on:
            self.in_port.remove_client(self, code=self.code_on)
            self.in_port.remove_client(self, code=self.code_off)
            self.code_on = new_code_on
            self.code_off = new_code_off
            self.in_port.add_client(self, self.code_on)
            self.in_port.add_client(self, self.code_off)

    def port_changed(self):
        self.in_port_name = self.in_port_name_property()
        super().port_changed()

    def custom_cleanup(self):
        self.in_port.remove_client(self, code=self.code_on)
        self.in_port.remove_client(self, code=self.code_off)


class MidiPolyPressureInNode(MidiIn, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiPolyPressureInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiIn.__init__(self, label, data, args)
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
                break

        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16, callback=self.params_changed)
        self.in_port_name_property = self.add_input('port', widget_type='combo', default_value=self.in_port.port.name, callback=self.port_changed)
        self.in_port_name_property.widget.combo_items = self.input_list

        self.note_output = self.add_output('note out')
        self.pressure_output = self.add_output('pressure out')
        self.code = self.make_code(channel)

        self.in_port.add_client(self, code=self.code)

    def make_code(self, channel):
        return (160 + channel - 1) * 256 + 128

    def receive_midi_bytes(self, msg_bytes):
        self.pressure_output.send(int(msg_bytes[2]))
        self.note_output.send(int(msg_bytes[1]))

    def params_changed(self):
        new_code = self.make_code(self.channel())
        if new_code != self.code:
            self.in_port.remove_client(self, code=self.code)
            self.code = new_code
            self.in_port.add_client(self, self.code)

    def port_changed(self):
        self.in_port_name = self.in_port_name_property()
        super().port_changed()

    def custom_cleanup(self):
        self.in_port.remove_client(self, code=self.code)


class MidiOutPort:
    ports = {}

    def __init__(self, name=None):
        self.port_name = name
        self.port = None

        if self.port_name is not None:
            if self.port_name in MidiOutPort.ports:
                self.port = MidiOutPort.ports[self.port_name]
            else:
                self.port = mido.open_output(self.port_name)
                MidiOutPort.ports[self.port_name] = self.port
        else:
            keys = list(MidiOutPort.ports.keys())
            if len(keys) > 0:
                self.port = MidiOutPort.ports[keys[0]]
            else:
                self.port = mido.open_output()
            self.port_name = self.port.name
            self.ports[self.port_name] = self

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
                self.out_port = mido.open_output(self.out_port_name)
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
        else:
            self.out_port = MidiOutPort(self.out_port_name)


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
        midi_data = [self.channel() - 1 + 176, self.control_number(), controller_value]
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
        midi_data = [self.channel() - 1 + 224, control_val_lsb, control_val_msb]
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
        midi_data = [self.channel() - 1 + 192, val]
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
        midi_data = [self.channel() - 1 + 208, val]
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
        midi_code = 144
        midi_data = [channel - 1 + midi_code, note, velocity]
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
        midi_code = 160
        midi_data = [channel - 1 + midi_code, note, pressure]
        msg = mido.Message.from_bytes(midi_data)
        self.out_port.send(msg)

    def port_changed(self):
        self.out_port_name = self.out_port_name_property()
        super().port_changed()


class BlueBoardNode(MidiIn, MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = BlueBoardNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        force_args = ['iRig BlueBoard Bluetooth']
        MidiIn.__init__(self, label, data, force_args)
        MidiOut.__init__(self, label, data, force_args)
        Node.__init__(self, label, data, force_args)

        self.set_A_input = self.add_input('set LED A', triggers_execution=True)
        self.set_B_input = self.add_input('set LED B', triggers_execution=True)
        self.set_C_input = self.add_input('set LED C', triggers_execution=True)
        self.set_D_input = self.add_input('set LED D', triggers_execution=True)
        # self.out_port_name = self.add_input('port', widget_type='combo', default_value=self.out_port.name,
        #                                     callback=self.port_changed)
        # self.out_port_name.widget.combo_items = self.output_list
        #
        # self.in_port_name_property = self.add_input('port', widget_type='combo', default_value=self.in_port.name, callback=self.port_changed)
        # self.in_port_name_property.widget.combo_items = self.input_list

        self.output_A = self.add_output('button A out')
        self.output_B = self.add_output('button B out')
        self.output_C = self.add_output('button C out')
        self.output_D = self.add_output('button D out')

    def receive_midi_bytes(self, midi_bytes):
        state = midi_bytes[2]
        if state == 127:
            state = 1
        if midi_bytes[1] == 20:
            self.output_A.send(state)
        elif midi_bytes[1] == 21:
            self.output_B.send(state)
        elif midi_bytes[1] == 22:
            self.output_C.send(state)
        elif midi_bytes[1] == 23:
            self.output_D.send(state)

    def execute(self):
        out = 0
        which = 20
        if self.active_input == self.set_A_input:
            out = self.set_A_input() * 127
            which = 20
        elif self.active_input == self.set_B_input:
            out = self.set_B_input() * 127
            which = 21
        elif self.active_input == self.set_C_input:
            out = self.set_C_input() * 127
            which = 22
        elif self.active_input == self.set_D_input:
            out = self.set_D_input() * 127
            which = 23
        midi_data = [176, which, out]
        msg = mido.Message.from_bytes(midi_data)
        self.out_port.send(msg)

