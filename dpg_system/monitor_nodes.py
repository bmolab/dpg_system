import subprocess
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import platform


class DisplayData:
    def __init__(self):
        self.id = -1
        self.connection = ''
        self.offsets = [0, 0]
        self.resolution = [-1, -1]
        self.primary = False


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

print(get_displays())