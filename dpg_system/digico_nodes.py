import dearpygui.dearpygui as dpg
from dpg_system.node import Node, NodeInput
from dpg_system.conversion_utils import *
import threading
from dpg_system.osc_nodes import *

def register_digico_nodes():
    Node.app.register_node('digico.fader', DigicoFaders.factory)


class DigicoFaders(OSCReceiver, OSCSender, Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DigicoFaders(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.fader_count = 20
        if len(args) > 1:
            self.fader_count = int(args[1])
        self.faders = []

        for fader_num in range(1, self.fader_count+1):
            fader = self.add_input('fader ' + str(fader_num), widget_type='slider_float', min=-80, max=10, triggers_execution=True)
            self.faders.append(fader)

        self.target_name_property = self.add_option('target name', widget_type='text_input', default_value=self.name,
                                                    callback=self.name_changed)
        self.source_name_property = self.target_name_property

    def name_changed(self):
        OSCReceiver.name_changed(self)
        OSCSender.name_changed(self, force=True)

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)
            self.find_source_node(self.name)

    def cleanup(self):
        OSCSender.cleanup(self)
        OSCReceiver.cleanup(self)

    def execute(self):
        fader = self.active_input
        index = fader.input_index
        data = any_to_float(fader())
        address = '/channel/' + str(index + 1) + '/fader'
        if data is not None:
            if self.target and self.address != '':
                self.target.send_message(address, data)

    # def receive(self, data):
    #     t = type(data)
    #     data = any_to_list(data)
    #
    #     if self.label not in ['osc_string', 'osc_message']:
    #         if type(data[0]) == str:
    #             if data[0][0] == '/':
    #                 return
    #     if self.label == 'osc_string':
    #         data = any_to_string(data)
    #         if self.space_replacement():
    #             data = data.replace('_', ' ')
    #     self.inputs[0].receive_data(data)
    #     ValueNode.execute(self)