import dearpygui.dearpygui as dpg
from dpg_system.node import Node, NodeInput
from dpg_system.conversion_utils import *
import threading
from dpg_system.osc_nodes import *

def register_eos_nodes():
    Node.app.register_node('eos_send', OSCSendEOSNode.factory)

class OSCSendEOSNode(OSCSender, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCSendEOSNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.channel = 1

        if len(args) > 0:
            if is_number(args[0]):
                self.channel = any_to_int(args[0])
            else:
                self.address = args[0]

        if len(args) > 1:
            if is_number(args[1]):
                self.channel = any_to_int(args[1])
            else:
                self.address = args[1]
        self.name = 'eos'
        min = 0
        max = 100
        if self.address in ['pan', 'tilt']:
            min = -360
            max = 360

        self.input = self.add_input('osc to send', widget_type='drag_int', callback=self.change_in_value, min=min, max=max)
        self.target_address_property = self.add_input('parameter', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.target_channel_property = self.add_input('target channel', widget_type='input_int', default_value=self.channel, min=1)
        self.target_name_property = self.add_option('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)

        self.min_property = self.add_option('min', widget_type='drag_int', default_value=min, callback=self.min_max_changed)
        self.max_property = self.add_option('max', widget_type='drag_int', default_value=max, callback=self.min_max_changed)

    def min_max_changed(self):
        self.input.widget.set_limits(min_=self.min_property(), max_=self.max_property())

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)

    def find_target_node(self, name):
        if self.osc_manager is not None:
            self.target = self.osc_manager.find_target(name)
            if self.target is not None:
                self.osc_manager.connect_send_node_to_target(self, self.target)
                return True
            else:
                self.osc_manager.connect_send_node_to_target(self, None)
        return False

    def cleanup(self):
        super().cleanup()

    def change_in_value(self):
        data = self.input()
        t = type(data)
        if t not in [str, int, float, bool, np.int64, np.double]:
            data = list(data)
            data, homogenous, types = list_to_hybrid_list(data)
        if data is not None:
            if self.target and self.address != '':
                address = '/eos/user/99/chan/' + str(self.target_channel_property()) + '/param/' + self.address
                self.target.send_message(address, data)

    def execute(self):
        self.change_in_value()
