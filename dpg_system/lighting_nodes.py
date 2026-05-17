from dpg_system.conversion_utils import *
from dpg_system.node import Node
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

        # Dirty flags. Named distinctly from the *_changed callback methods
        # to avoid shadowing the bound methods on the instance.
        self._any_dirty = False
        self._intensity_dirty = False
        self._red_dirty = False
        self._green_dirty = False
        self._blue_dirty = False
        self._lime_dirty = False
        # self._indigo_dirty = False

        if self.name == '':
            self.name = 'eos'
        if self.address == '':
            self.address = '/eos/user/99/chan'

        if args:
            if is_number(args[0]):
                self.channel = max(1, any_to_int(args[0]))

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

    def address_changed(self):
        # OSCSender's base class does not define address_changed; provide a
        # local implementation so edits to the address widget take effect.
        if self.target_address_property is None:
            return
        try:
            new_address = any_to_string(self.target_address_property())
        except Exception:
            return
        if new_address and not new_address.startswith('/'):
            new_address = '/' + new_address
        self.address = new_address

    def _read_slider_value(self, widget):
        try:
            value = any_to_int(widget())
        except Exception:
            return None
        if value < 0:
            return 0
        if value > 100:
            return 100
        return value

    def intensity_changed(self):
        v = self._read_slider_value(self.intensity_input)
        if v is None:
            return
        self.intensity = v
        self._intensity_dirty = True
        self._any_dirty = True

    def red_changed(self):
        v = self._read_slider_value(self.red_input)
        if v is None:
            return
        self.red = v
        self._red_dirty = True
        self._any_dirty = True

    def green_changed(self):
        v = self._read_slider_value(self.green_input)
        if v is None:
            return
        self.green = v
        self._green_dirty = True
        self._any_dirty = True

    def blue_changed(self):
        v = self._read_slider_value(self.blue_input)
        if v is None:
            return
        self.blue = v
        self._blue_dirty = True
        self._any_dirty = True

    def lime_changed(self):
        v = self._read_slider_value(self.lime_input)
        if v is None:
            return
        self.lime = v
        self._lime_dirty = True
        self._any_dirty = True

    # def indigo_changed(self):
    #     v = self._read_slider_value(self.indigo_input)
    #     if v is None:
    #         return
    #     self.indigo = v
    #     self._indigo_dirty = True
    #     self._any_dirty = True

    def _clear_dirty(self):
        self._any_dirty = False
        self._intensity_dirty = False
        self._red_dirty = False
        self._green_dirty = False
        self._blue_dirty = False
        self._lime_dirty = False
        # self._indigo_dirty = False

    def frame_task(self):
        if not self._any_dirty:
            return
        if self.target is None or not self.address:
            # Drop pending updates rather than spinning every frame waiting
            # for a target/address that may never resolve.
            self._clear_dirty()
            return

        try:
            channel = any_to_int(self.target_channel_property())
        except Exception:
            self._clear_dirty()
            return
        if channel < 1:
            channel = 1

        base = self.address.rstrip('/') + '/' + str(channel) + '/param/'
        self._any_dirty = False

        try:
            if self._intensity_dirty:
                self._intensity_dirty = False
                self.target.send_message(base + 'intens', self.intensity)

            if self._red_dirty:
                self._red_dirty = False
                self.target.send_message(base + 'red', self.red)

            if self._green_dirty:
                self._green_dirty = False
                self.target.send_message(base + 'green', self.green)

            if self._blue_dirty:
                self._blue_dirty = False
                self.target.send_message(base + 'blue', self.blue)

            if self._lime_dirty:
                self._lime_dirty = False
                self.target.send_message(base + 'lime', self.lime)

            # if self._indigo_dirty:
            #     self._indigo_dirty = False
            #     self.target.send_message(base + 'indigo', self.indigo)
        except Exception as e:
            print('color_source: OSC send failed:', e)
