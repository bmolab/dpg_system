from dpg_system.conversion_utils import *
from dpg_system.node import Node
import threading
from dpg_system.osc_nodes import *


def register_lighting_nodes():
    Node.app.register_node('color_source', ColorSourceNode.factory)


class ColorSourceNode(OSCSender, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ColorSourceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        OSCSender.__init__(self, label, data, args)
        Node.__init__(self, label, data, args)

        self.channel = 1
        self.intensity = 0
        self.red = 0
        self.green = 0
        self.blue = 0
        self.lime = 0
        self.indigo = 0
        self.address = '/eos/user/0/chan/'

        if len(args) > 0:
            self.channel = any_to_int(args[0])

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
        self.target_channel_property = self.add_input('target channel', widget_type='input_int')

        self.add_frame_task()

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

    def indigo_changed(self):
        self.indigo = self.indigo_input()
        self.changed = True
        self.indigo_changed = True

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

                if self.indigo_changed:
                    self.indigo_changed = False
                    self.target.send_message(address + 'indigo', self.indigo)












