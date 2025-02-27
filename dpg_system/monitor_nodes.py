import subprocess
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import platform

def register_display_nodes():
    Node.app.register_node('display_info', DisplayInfoNode.factory)

class DisplayData:
    def __init__(self):
        self.id = -1
        self.connection = ''
        self.offsets = [0, 0]
        self.resolution = [-1, -1]
        self.primary = False


if platform.system() == 'Darwin':
    print('Darwin')
    import Quartz.CoreGraphics

    # ids = Quartz.CGDisplayCopyAllDisplayIDs()
    # print(ids)

if platform.system() == "Linux":
    def get_displays():
        result = subprocess.run(['xrandr', '--listactivemonitors'], stdout=subprocess.PIPE, text=True)
        display_info = result.stdout
        display_info = display_info.split('\n')
        display_count = int(display_info[0].split(' ')[-1])
        displays = []
        for i in range(len(display_info) - 1):
            d = DisplayData()

            if display_info[i + 1] != '':
                display_data = display_info[i + 1].split(' ')
                display_data_stripped = []
                for dd in display_data:
                    if dd != '':
                        display_data_stripped.append(dd)
                display_number = int(display_data_stripped[0].split(':')[0])
                d.id = display_number
                display_connection = display_data_stripped[1]
                if '+' in display_connection:
                    display_connection = display_connection.split('+')[-1]
                if '*' in display_connection:
                    primary_display = i
                    d.primary = True
                    display_connection = display_connection.split('*')[-1]
                d.connection = display_connection
                display_res = display_data_stripped[2]
                res = display_res.split('+')
                x_offset = res[1]
                y_offset = res[2]
                d.offsets = [x_offset, y_offset]
                res_break = res[0].split('x')
                res_x = res_break[0].split('/')[0]
                res_y = res_break[1].split('/')[0]
                d.resolution = [res_x, res_y]
                displays.append(d)
        return displays


class DisplayInfoNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DisplayInfoNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('in', widget_type='input_int', default_value=0, min=0, triggers_execution=True)
        self.width_property = self.add_property('width', widget_type='drag_int', default_value=0)
        self.height_property = self.add_property('height', widget_type='drag_int', default_value=0)
        self.x_offset_property = self.add_property('x offset', widget_type='drag_int', default_value=0)
        self.y_offset_property = self.add_property('y offset', widget_type='drag_int', default_value=0)
        self.connection_property = self.add_property('connection', widget_type='text_input', default_value='')
        self.primary_property = self.add_property('primary display', widget_type='checkbox', default_value=False)

        self.output = self.add_output('data out')

    def execute(self):
        which = self.input()
        displays = get_displays()
        if 0 <= which < len(displays):
            self.primary_property.set(displays[which].primary)
            self.connection_property.set(displays[which].connection)
            self.width_property.set(displays[which].resolution[0])
            self.height_property.set(displays[which].resolution[1])
            self.x_offset_property.set(displays[which].offsets[0])
            self.y_offset_property.set(displays[which].offsets[1])
            out_data = [displays[which].resolution[0], displays[which].resolution[1], displays[which].offsets[0], displays[which].offsets[1], displays[which].connection, displays[which].primary]
            self.output.send(out_data)
        else:
            self.primary_property.set(False)
            self.connection_property.set('')
            self.width_property.set(0)
            self.height_property.set(0)
            self.x_offset_property.set(0)
            self.y_offset_property.set(0)
            out_data = [0, 0, 0, 0, '', False]
            self.output.send(out_data)

    def custom_create(self, from_file):
        self.execute()