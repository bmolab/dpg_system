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
        # Supply the console defaults as args rather than assigning them after
        # OSCDeviceNode.__init__. OSCAsyncIOSource.__init__ calls start_serving()
        # immediately, binding source_port at that moment — assigning 1102
        # afterwards came too late, so the node listened on an auto-picked port
        # and never heard the console. Feeding them through the normal arg
        # parsing sets name/ip/ports before either base class needs them.
        if args is None or len(args) == 0:
            args = ['eos', '10.1.3.11', '1101', '1102']
        OSCDeviceNode.__init__(self, label, data, args)

    def custom_create(self, from_file):
        OSCDeviceNode.custom_create(self, from_file)


class ColorSourceNode(Node, OSCBase, OSCSender):
    @staticmethod
    def factory(name, data, args=None):
        node = ColorSourceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        # OSCSender.__init__ reads self.osc_manager (when given a single
        # argument, to auto-pick the only local target). That lives on OSCBase,
        # which this class did not inherit -- its siblings all do. Without it
        # 'color_source 7' died with AttributeError and the node could not be
        # created at all, while a bare 'color_source' happened to work, because
        # that branch is only taken when there is exactly one argument.
        Node.__init__(self, label, data, args)
        OSCSender.__init__(self, label, data, args)

        self.changed = False
        self.channel = 1
        self.intensity = 0
        self.red = 0
        self.green = 0
        self.blue = 0
        self.lime = 0
        self.indigo = 0
        # Separate names from the callbacks above. These used to be
        # self.<param>_changed, the same name as the method, so before a
        # slider had ever moved the flag WAS the bound method -- truthy.
        # The first change to any one parameter therefore sent all five,
        # including intensity 0, which blacks out the channel.
        self.intensity_dirty = False
        self.red_dirty = False
        self.green_dirty = False
        self.blue_dirty = False
        self.lime_dirty = False
        self.indigo_dirty = False

        if self.name == '':
            self.name = 'eos'
        # A lone numeric argument means the CHANNEL here (see the loop below),
        # but OSCSender has already taken any single argument as the address --
        # so 'color_source 7' arrived with address '/7' and composed
        # '/7/7/param/red'. Treat a purely numeric address as "not given".
        if self.address == '' or self.address.lstrip('/').isdigit():
            self.address = '/eos/user/99/chan'

        if len(args) > 0:
            for i in range(len(args)):
                if is_number(args[i]):
                    self.channel = any_to_int(args[i])
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
        self.intensity_dirty = True

    def red_changed(self):
        self.red = self.red_input()
        self.changed = True
        self.red_dirty = True

    def green_changed(self):
        self.green = self.green_input()
        self.changed = True
        self.green_dirty = True

    def blue_changed(self):
        self.blue = self.blue_input()
        self.changed = True
        self.blue_dirty = True

    def lime_changed(self):
        self.lime = self.lime_input()
        self.changed = True
        self.lime_dirty = True

    # def indigo_changed(self):
    #     self.indigo = self.indigo_input()
    #     self.changed = True
    #     self.indigo_changed = True

    def address_changed(self):
        # OSCSender supplies name_changed but not this one, and the widget was
        # wired to it regardless -- so the node raised AttributeError while
        # being built. The address is only ever used to compose the outgoing
        # path in frame_task, so keeping self.address in step is all it needs.
        self.address = any_to_string(self.target_address_property())
        if self.address != '' and not self.address.startswith('/'):
            self.address = '/' + self.address

    def frame_task(self):
        if self.target and self.address != '':
            if self.changed:
                address = self.address + '/' + str(self.target_channel_property()) + '/param/'

                self.changed = False
                if self.intensity_dirty:
                    self.intensity_dirty = False
                    self.target.send_message(address + 'intens', self.intensity)

                if self.red_dirty:
                    self.red_dirty = False
                    self.target.send_message(address + 'red', self.red)

                if self.green_dirty:
                    self.green_dirty = False
                    self.target.send_message(address + 'green', self.green)

                if self.blue_dirty:
                    self.blue_dirty = False
                    self.target.send_message(address + 'blue', self.blue)

                if self.lime_dirty:
                    self.lime_dirty = False
                    self.target.send_message(address + 'lime', self.lime)

                # if self.indigo_dirty:
                #     self.indigo_dirty = False
                #     self.target.send_message(address + 'indigo', self.indigo)




class OSCSendEOSNode(Node, OSCBase, OSCSender, OSCRegistrableMixin):
    @staticmethod
    def factory(name, data, args=None):
        node = OSCSendEOSNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.channel = 1
        self.address = 'empty'

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

        self._registerable_init()

    def min_max_changed(self):
        self.input.widget.set_limits(min_=self.min_property(), max_=self.max_property())

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)
        self._registerable_custom_create()

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
        self._registerable_cleanup()

    def _get_registry_path_components(self) -> list:
        return [self.get_patcher_path(), self.name, self.address]

    def _create_registry_entry(self, path_components: list) -> str:
        return self.osc_manager.registry.add_generic_sender_to_registry(path_components)

    def change_in_value(self):
        data = self.input()
        if data is None:
            return
        t = type(data)
        if t not in [str, int, float, bool, np.int64, np.double]:
            try:
                data = list(data)
            except TypeError:
                return
            data, homogenous, types = list_to_hybrid_list(data)
        if data is not None and self.target and self.address != '':
            address = '/eos/user/99/chan/' + str(self.target_channel_property()) + '/param/' + self.address
            self.target.send_message(address, data)

    def execute(self):
        self.change_in_value()

    def address_changed(self):
        """
        Handles changes to the node's OSC address, ensuring the registry is
        updated correctly.
        """
        address_property = None
        if hasattr(self, 'target_address_property'):
            address_property = self.target_address_property

        if address_property is None:
            return

        new_address = any_to_string(address_property())

        if new_address != self.address:
            # 1. CAPTURE the old path components BEFORE changing the state.
            old_path_components = self._get_registry_path_components()

            # 2. CHANGE the internal state.
            # This is the logic that was in the base OSCReceiver/OSCSender.
            if self.target is not None:
                self.target.unregister_send_node(self)  # For senders

            self.address = new_address

            # Re-register with the source/target under the new address
            if self.target is not None:
                self.target.register_send_node(self)

            # 3. UPDATE the registry, passing in the captured old path.
            self._update_registration(old_path_components=old_path_components)

