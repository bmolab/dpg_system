import dearpygui.dearpygui as dpg
from dpg_system.node import Node, NodeInput
from dpg_system.conversion_utils import *
import threading
import re
from dpg_system.osc_nodes import *

def register_eos_nodes():
    Node.app.register_node('eos_console', EOSConsoleNode.factory)
    Node.app.register_node('color_source', ColorSourceNode.factory)
    Node.app.register_node('eos_send', OSCSendEOSNode.factory)
    Node.app.register_node('eos_int', OSCSendEOSNode.factory)
    Node.app.register_node('eos_float', OSCSendEOSNode.factory)
    Node.app.register_node('eos_slider', OSCSendEOSNode.factory)
    Node.app.register_node('eos_knob', OSCSendEOSNode.factory)
    Node.app.register_node('eos_toggle', OSCSendEOSNode.factory)

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
    """One value to one named parameter of one Eos channel.

    The registered name picks the input widget - the same idiom as the value
    nodes (int / float / slider / knob / toggle). Everything else is shared:
    the composed path is

        /eos/user/<user>/chan/<channel>/param/<parameter>

    and it is shown in full under the inputs so what goes out is never a guess.
    'target channel' is a channel SPEC, not one number - '1-10', '1 3 5 7 9',
    '1,3,5' or '1 thru 10' - and one message goes out per channel, because the
    parameter path on the desk takes exactly one channel.
    'eos_send' keeps its original drag_int, so patches saved before the family
    existed load unchanged.
    """
    # registered name suffix -> (input widget, value family, limit widget)
    _VARIANTS = {
        'send':   ('drag_int',     int,   'drag_int'),
        'int':    ('drag_int',     int,   'drag_int'),
        'float':  ('drag_float',   float, 'drag_float'),
        'slider': ('slider_float', float, 'drag_float'),
        'knob':   ('knob_float',   float, 'drag_float'),
        'toggle': ('checkbox',     float, 'drag_float'),
    }

    @staticmethod
    def factory(name, data, args=None):
        node = OSCSendEOSNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        variant = label.split('_')[-1]
        self.widget_type, self.value_family, limit_widget = self._VARIANTS.get(variant, self._VARIANTS['send'])

        self.channel_spec = '1'
        self.user = 99
        self.address = 'empty'

        # Anything that reads as channels (numbers, ranges) is channels, in the
        # order given, so 'eos_send intens 1 3 5' and 'eos_send intens 1-10'
        # both work; the one word that is not is the parameter name.
        channel_tokens = []
        for arg in args:
            arg = any_to_string(arg)
            if arg.lower() in ('thru', 'to') or re.fullmatch(r'[\d,\-]+', arg):
                channel_tokens.append(arg)
            else:
                self.address = arg
        if channel_tokens:
            self.channel_spec = ' '.join(channel_tokens)
        self.name = 'eos'
        min = 0
        max = 100
        if self.address in ['pan', 'tilt']:
            min = -360
            max = 360
        if self.value_family is float:
            min = float(min)
            max = float(max)

        if self.widget_type == 'checkbox':
            self.input = self.add_input('osc to send', widget_type='checkbox', callback=self.change_in_value)
        else:
            self.input = self.add_input('osc to send', widget_type=self.widget_type, callback=self.change_in_value, min=min, max=max)
        self.target_address_property = self.add_input('parameter', widget_type='text_input', default_value=self.address, callback=self.address_changed)
        self.target_channel_property = self.add_input('target channel', widget_type='text_input', default_value=self.channel_spec, widget_width=120, callback=self.refresh_address_display)
        # A label is display only: never saved, never restored, never an inlet.
        self.address_display = self.add_label(self.displayed_address())

        self.target_name_property = self.add_option('target name', widget_type='text_input', default_value=self.name, callback=self.name_changed)
        self.user_property = self.add_option('user', widget_type='input_int', default_value=self.user, min=1, callback=self.refresh_address_display)

        self.min_property = self.add_option('min', widget_type=limit_widget, default_value=min, callback=self.min_max_changed)
        self.max_property = self.add_option('max', widget_type=limit_widget, default_value=max, callback=self.min_max_changed)

        self._registerable_init()

    def min_max_changed(self):
        self.input.widget.set_limits(min_=self.min_property(), max_=self.max_property())

    _CHAN_TOKEN = re.compile(r'^\d+(-\d+)?$')

    @staticmethod
    def parse_channels(spec):
        """'1-10', '1 3 5', '1,3,5', '1 thru 10', '1 to 4 7' -> [ints], in order, no repeats."""
        text = any_to_string(spec).lower().replace(',', ' ')
        text = re.sub(r'\s*(thru|to|-)\s*', '-', text)
        channels = []
        for token in text.split():
            if not OSCSendEOSNode._CHAN_TOKEN.match(token):
                continue
            if '-' in token:
                a, b = (int(x) for x in token.split('-'))
                step = 1 if b >= a else -1
                run = range(a, b + step, step)
            else:
                run = [int(token)]
            for c in run:
                if c > 0 and c not in channels:
                    channels.append(c)
        return channels

    @staticmethod
    def format_channels(channels):
        """[1,2,3,5,7,8] -> '1-3,5,7-8'; consecutive runs collapse."""
        if not channels:
            return '?'
        parts = []
        start = prev = channels[0]
        for c in channels[1:] + [None]:
            if c is not None and c == prev + 1:
                prev = c
                continue
            parts.append(str(start) if start == prev else f'{start}-{prev}')
            if c is not None:
                start = prev = c
        return ','.join(parts)

    def current_user(self):
        # The widgets exist only after create; before that the parsed defaults
        # are what the display should show.
        if hasattr(self, 'user_property') and self.user_property() is not None:
            return any_to_int(self.user_property())
        return self.user

    def current_channels(self):
        spec = self.channel_spec
        if hasattr(self, 'target_channel_property') and self.target_channel_property() is not None:
            spec = self.target_channel_property()
        return self.parse_channels(spec)

    def composed_address(self, channel):
        return '/eos/user/' + str(self.current_user()) + '/chan/' + str(channel) + '/param/' + self.address

    def displayed_address(self):
        channels = self.current_channels()
        shown = self.composed_address(self.format_channels(channels))
        if len(channels) > 1:
            shown += f'  ({len(channels)} channels)'
        return shown

    def refresh_address_display(self):
        if hasattr(self, 'address_display'):
            self.address_display.set(self.displayed_address())

    def custom_create(self, from_file):
        if self.name != '':
            self.find_target_node(self.name)
        self._registerable_custom_create()
        self.refresh_address_display()

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
        if self.widget_type == 'checkbox':
            # A toggle is full or out: max when on, min when off.
            data = self.max_property() if any_to_bool(data) else self.min_property()
        t = type(data)
        if t not in [str, int, float, bool, np.int64, np.double]:
            try:
                data = list(data)
            except TypeError:
                return
            data, homogenous, types = list_to_hybrid_list(data)
        elif t is not str:
            data = any_to_int(data) if self.value_family is int else any_to_float(data)
        if data is not None and self.target and self.address != '':
            for channel in self.current_channels():
                self.target.send_message(self.composed_address(channel), data)

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

        self.refresh_address_display()

