import dearpygui.dearpygui as dpg
from dpg_system.node import Node, NodeInput
from dpg_system.conversion_utils import *
import threading
from dpg_system.osc_nodes import *

def register_eos_nodes():
    Node.app.register_node('eos_console', EOSConsoleNode.factory)
    Node.app.register_node('color_source', ColorSourceNode.factory)
    Node.app.register_node('eos_send', OSCSendEOSNode.factory)

class EOSConsoleNode(OSCDeviceNode):
    @staticmethod
    def factory(name, data, args=None):
        node = EOSConsoleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        OSCDeviceNode.__init__(self, label, data, args)
        self.target_ip_property.set_default_value('10.1.3.11')
        self.target_port_property.set_default_value('1101')
        self.source_port_property.set_default_value('1102')
        self.name_property.set_default_value('eos')

    def custom_create(self, from_file):
        self.target_changed()
        self.source_changed()


class ColorSourceNode(OSCSender, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ColorSourceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        OSCSender.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.changed = False
        self.channel = 1
        self.intensity = 0
        self.red = 0
        self.green = 0
        self.blue = 0
        self.lime = 0
        self.indigo = 0

        if self.name == '':
            self.name = 'eos'
        if self.address == '':
            self.address = '/eos/user/99/chan'

        if len(args) > 0:
            for i in len(args):
                if is_number(args[0]):
                    self.channel = any_to_int(args[0])
                    break

        self.intensity_input = self.add_input('intensity', widget_type='slider_int', widget_width=120, min=0, max=100, default_value=self.intensity, callback=self.intensity_changed)
        self.red_input = self.add_input('red', widget_type='slider_int', widget_width=120, min=0, max=100,
                                              default_value=self.red, callback=self.red_changed)

        self.green_input = self.add_input('green', widget_type='slider_int', widget_width=120, min=0, max=100,
                                              default_value=self.green, callback=self.green_changed)

        self.blue_input = self.add_input('blue', widget_type='slider_int', widget_width=120, min=0, max=100,
                                              default_value=self.blue, callback=self.blue_changed)

        self.lime_input = self.add_input('lime', widget_type='slider_int', widget_width=120, min=0, max=100,
                                              default_value=self.lime, callback=self.lime_changed)

        self.target_name_property = self.add_input('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_input('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.target_channel_property = self.add_input('target channel', widget_type='input_int', default_value=self.channel, min=1)

        self.add_frame_task()

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)

    def intensity_changed(self):
        self.intensity = self.intensity_input()
        self.changed = True
        self.intensity_changed = True

    def red_changed(self):
        self.red = self.red_input()
        self.changed = True
        self.red_changed = True

    def green_changed(self):
        self.green = self.green_input()
        self.changed = True
        self.green_changed = True

    def blue_changed(self):
        self.blue = self.blue_input()
        self.changed = True
        self.blue_changed = True

    def lime_changed(self):
        self.lime = self.lime_input()
        self.changed = True
        self.lime_changed = True

    # def indigo_changed(self):
    #     self.indigo = self.indigo_input()
    #     self.changed = True
    #     self.indigo_changed = True

    def frame_task(self):
        if self.target and self.address != '':
            if self.changed:
                address = self.address + '/' + str(self.target_channel_property()) + '/param/'

                self.changed = False
                if self.intensity_changed:
                    self.intensity_changed = False
                    self.target.send_message(address + 'intens', self.intensity)

                if self.red_changed:
                    self.red_changed = False
                    self.target.send_message(address + 'red', self.red)

                if self.green_changed:
                    self.green_changed = False
                    self.target.send_message(address + 'green', self.green)

                if self.blue_changed:
                    self.blue_changed = False
                    self.target.send_message(address + 'blue', self.blue)

                if self.lime_changed:
                    self.lime_changed = False
                    self.target.send_message(address + 'lime', self.lime)

                # if self.indigo_changed:
                #     self.indigo_changed = False
                #     self.target.send_message(address + 'indigo', self.indigo)




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
