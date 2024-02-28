import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import mido


def register_midi_nodes():
    Node.app.register_node('midi_in', MidiInNode.factory)
    Node.app.register_node('midi_out', MidiOutNode.factory)


class MidiInNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiInNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.port_name = None
        if len(args) > 0:
            self.port_name = args[0]

        if self.port_name is not None:
            self.port = mido.open_input(self.port, callback=self.receive)
            print(self.port)
        else:
            self.port = mido.open_input(callback=self.receive)
            print(self.port)
        self.input_list = mido.get_input_names()
        print(self.input_list)
        self.port.callback = self.receive

        self.port_name = self.add_input('port', widget_type='combo', default_value=self.port.name, callback=self.port_changed)
        self.port_name.widget.combo_items = self.input_list

        self.output = self.add_output('midi out')

    def receive(self, msg):
        self.output.send(msg.bytes())

    def port_changed(self):
        pass


class MidiOutNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MidiOutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.port_name = None
        if len(args) > 0:
            self.port_name = args[0]

        if self.port_name is not None:
            self.port = mido.open_output(self.port)
            print(self.port)
        else:
            self.port = mido.open_output()
            print(self.port)
        self.output_list = mido.get_output_names()
        print(self.output_list)

        self.midi_to_send = self.add_input('midi to send', triggers_execution=True)
        self.port_name = self.add_input('port', widget_type='combo', default_value=self.port.name, callback=self.port_changed)
        self.port_name.widget.combo_items = self.output_list

        self.output = self.add_output('midi out')

    def execute(self):
        input = any_to_list(self.midi_to_send())
        msg = mido.Message.from_bytes(input)
        self.port.send(msg)


    def port_changed(self):
        pass
