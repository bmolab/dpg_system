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
        self.address = '/eos/chan/'
        self.changed = False
        self.intensity_change = False
        self.red_change = False
        self.green_change = False
        self.blue_change = False
        self.lime_change = False
        self.indigo_change = False

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

        self.indigo_input = self.add_input('indigo', widget_type='slider_int', widget_width=120, min=0, max=100,
                                              default_value=self.indigo, callback=self.indigo_changed)

        self.target_name_property = self.add_input('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.target_address_property = self.add_input('address', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.target_channel_property = self.add_input('target channel', widget_type='input_int')

        self.add_frame_task()

    def name_changed(self, force=False):
        OSCSender.name_changed(self, force=True)

    def intensity_changed(self):
        self.intensity = self.intensity_input()
        self.change = True
        self.intensity_change = True

    def red_changed(self):
        self.red = self.red_input()
        self.change = True
        self.red_change = True

    def green_changed(self):
        self.green = self.green_input()
        self.change = True
        self.green_change = True

    def blue_changed(self):
        self.blue = self.blue_input()
        self.change = True
        self.blue_change = True

    def lime_changed(self):
        self.lime = self.lime_input()
        self.change = True
        self.lime_change = True

    def indigo_changed(self):
        self.indigo = self.indigo_input()
        self.change = True
        self.indigo_change = True

    def frame_task(self):
        if self.target and self.address != '':
            if self.change:
                address = self.address + '/' + str(self.target_channel_property()) + '/param/'

                self.change = False
                if self.intensity_change:
                    self.intensity_change = False
                    self.target.send_message(address + 'intens', self.intensity)

                if self.red_change:
                    self.red_change = False
                    self.target.send_message(address + 'red', self.red)

                if self.green_change:
                    self.green_change = False
                    self.target.send_message(address + 'green', self.green)

                if self.blue_change:
                    self.blue_change = False
                    self.target.send_message(address + 'blue', self.blue)

                if self.lime_change:
                    self.lime_change = False
                    self.target.send_message(address + 'lime', self.lime)

                if self.indigo_change:
                    self.indigo_change = False
                    self.target.send_message(address + 'indigo', self.indigo)












