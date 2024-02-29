import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import mido


def register_midi_nodes():
    Node.app.register_node('midi_in', MidiInNode.factory)
    Node.app.register_node('midi_out', MidiOutNode.factory)
    Node.app.register_node('blue_board', BlueBoardNode.factory)


class MidiIn:
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.in_port_name = None
        if len(args) > 0:
            self.in_port_name = args[0]

        if self.in_port_name is not None:
            self.in_port = mido.open_input(self.in_port_name, callback=self.receive)
            print(self.in_port)
        else:
            self.in_port = mido.open_input(callback=self.receive)
            print(self.in_port)
        self.input_list = mido.get_input_names()
        print(self.input_list)
        self.in_port.callback = self.receive

    def receive(self, msg):
        pass

    def port_changed(self):
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


class MidiOut:
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.out_port_name = None
        if len(args) > 0:
            self.out_port_name = args[0]

        if self.out_port_name is not None:
            self.out_port = mido.open_output(self.out_port_name)
            print(self.out_port_name)
        else:
            self.out_port = mido.open_output()
            print(self.out_port_name)
        self.output_list = mido.get_output_names()
        print(self.output_list)

    def port_changed(self):
        pass


class MidiOutNode(MidiOut, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):

        MidiOut.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.midi_to_send = self.add_input('midi to send', triggers_execution=True)
        self.out_port_name = self.add_input('port', widget_type='combo', default_value=self.out_port.name, callback=self.port_changed)
        self.out_port_name.widget.combo_items = self.output_list

        self.output = self.add_output('midi out')


    def execute(self):
        midi_data = any_to_list(self.midi_to_send())
        msg = mido.Message.from_bytes(midi_data)
        self.out_port.send(msg)

    def port_changed(self):
        pass


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

