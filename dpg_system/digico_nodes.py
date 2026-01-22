import dearpygui.dearpygui as dpg
from dpg_system.node import Node, NodeInput
from dpg_system.conversion_utils import *
import threading
from dpg_system.osc_nodes import *

def register_digico_nodes():
    Node.app.register_node('digico.fader', DigicoFaders.factory)


class DigicoFaders(Node, OSCBase, OSCReceiver, OSCSender, OSCRegistrableMixin):
    @staticmethod
    def factory(name, data, args=None):
        node = DigicoFaders(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        Node.__init__(self, label, data, args)
        OSCSender.__init__(self, label, data, args)
        OSCReceiver.__init__(self, label, data, args)

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
        self.node = self
        self._registerable_init()

    def name_changed(self):
        OSCReceiver.name_changed(self)
        OSCSender.name_changed(self, force=True)

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)
            self.find_source_node(self.name)

        self._registerable_custom_create()  # This calls self.register()
        if self.path: # Only run the hook if registration was successful
            self._on_registration_complete()

    def _on_registration_complete(self):
        """
        Hook method for subclasses to implement. Called after the node has been
        successfully registered. This is the place to sync initial state like
        value, range, etc., to the registry.
        """
        pass # Default implementation does nothing.

    def get_addresses(self):
        addresses = []
        for index, fader in enumerate(self.faders):
            addresses.append('/channel/' + str(index) + '/fader')
        return addresses

    def cleanup(self):
        OSCSender.cleanup(self)
        OSCReceiver.cleanup(self)
        self._registerable_cleanup()

    def execute(self):
        fader = self.active_input
        index = fader.input_index
        if index == 0:
            fader_value = fader()
            if isinstance(fader_value, list):
                for index, value in enumerate(fader_value):
                    data = any_to_float(value)
                    address = '/channel/' + str(index + 1) + '/fader'
                    if data is not None:
                        self.inputs[index].set(data)
                        if self.target and address != '':
                            self.target.send_message(address, data)
                return

        data = any_to_float(fader())
        address = '/channel/' + str(index + 1) + '/fader'

        if data is not None:
            if self.target and address != '':
                self.target.send_message(address, data)

    def receive(self, data, address=None):
        data = any_to_list(data)
        if address is not None:
            split_address = address.split('/')
            if len(split_address) == 4:
                if split_address[1] == 'channel':
                    if split_address[3] == 'fader':
                        input = any_to_int(split_address[2])
                        if input < len(self.faders):
                            self.faders[input].receive_data(data)

    def _get_registry_path_components(self) -> list:
        # # Assumes `self.get_patcher_path` exists because self is a Node instance.
        return [self.get_patcher_path(), self.name, self.address]

    def _create_registry_entry(self, path_components: list) -> str:
        """
        This is the only piece of registration logic specific to this node class.
        """
        return self.osc_manager.registry.add_float_to_registry(path_components)

