import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import mido


def register_midi_nodes():
    Node.app.register_node('midi_in', MidiInNode.factory)
    Node.app.register_node('midi_out', MidiOutNode.factory)
    Node.app.register_node('midi_control_out', MidiControlOutNode.factory)
    Node.app.register_node('midi_note_out', MidiNoteOutNode.factory)
    Node.app.register_node('blue_board', BlueBoardNode.factory)


class MidiIn:
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.in_port_name = None
        if len(args) > 0:
            self.in_port_name = args[0]

        if self.in_port_name is not None:
            self.in_port = mido.open_input(self.in_port_name, callback=self.receive)
        else:
            self.in_port = mido.open_input(callback=self.receive)
        self.input_list = mido.get_input_names()
        self.in_port.callback = self.receive

    def receive(self, msg):
        pass

    def port_changed(self):
        if self.in_port:
            self.in_port.close()
        self.in_port = mido.open_input(self.in_port_name, callback=self.receive)
        pass


class MidiInNode(MidiIn, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        MidiIn.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.in_port_name_property = self.add_input('port', widget_type='combo', default_value=self.in_port.name, callback=self.port_changed)
        self.in_port_name_property.widget.combo_items = self.input_list

        self.output = self.add_output('midi out')

    def receive(self, msg):
        self.output.send(msg.bytes())

    def port_changed(self):
        self.in_port_name = self.in_port_name_property()
        super().port_changed()


class MidiOut:
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.out_port_name = None
        if len(args) > 0:
            self.out_port_name = args[0]

        if self.out_port_name is not None:
            self.out_port = mido.open_output(self.out_port_name)
        else:
            self.out_port = mido.open_output()
        self.output_list = mido.get_output_names()

    def port_changed(self):
        if self.out_port:
            self.out_port.close()
        self.out_port = mido.open_output(self.out_port_name)


class MidiOutNode(MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):

        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.midi_to_send = self.add_input('midi to send', triggers_execution=True)
        self.out_port_name_property = self.add_input('port', widget_type='combo', default_value=self.out_port.name, callback=self.port_changed)
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
        if len(args) > 0:
            controller = any_to_int(args[0])
            if controller > 127:
                controller = 127
            elif controller < 0:
                controller = 0

        channel = 1
        if len(args) > 1:
            channel = any_to_int(args[1])
            if channel > 16:
                channel = 16
            elif channel < 1:
                channel = 1

        self.midi_to_send = self.add_input('midi to send', triggers_execution=True)
        self.control_number = self.add_input('controller #', widget_type='input_int', default_value=controller, min=0, max=127)
        self.channel = self.add_option('channel', widget_type='input_int', default_value=channel, min=1, max=16)
        self.out_port_name_property = self.add_option('port', widget_type='combo', default_value=self.out_port.name, callback=self.port_changed)
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
        self.out_port_name_property = self.add_option('port', widget_type='combo', default_value=self.out_port.name, callback=self.port_changed)
        self.out_port_name_property.widget.combo_items = self.output_list

    def execute(self):
        input = self.midi_to_send()
        note = 64
        velocity = self.velocity()
        channel = self.channel()

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

    def receive(self, msg):
        bytes = msg.bytes()
        state = bytes[2]
        if state == 127:
            state = 1
        if bytes[1] == 20:
            self.output_A.send(state)
        elif bytes[1] == 21:
            self.output_B.send(state)
        elif bytes[1] == 22:
            self.output_C.send(state)
        elif bytes[1] == 23:
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

