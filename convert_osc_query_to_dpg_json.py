import dearpygui.dearpygui as dpg
import numpy as np
import string
import sys
import os
import string
from dpg_system.node import Node, SaveDialog, LoadDialog
from dpg_system.conversion_utils import *

'''
NOTE:
How to convert '''


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    length = len(hex_str)

    # Handle standard 6-digit hex (RRGGBB)
    if length == 6:
        return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))

    # Handle 3-digit shorthand (RGB -> RRGGBB)
    elif length == 3:
        return tuple(int(c * 2, 16) for c in hex_str)

    # Handle 8-digit hex with Alpha (RRGGBBAA)
    elif length == 8:
        return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4, 6))

    else:
        raise ValueError("Invalid hex color format")


def convert_osc_query_to_dpg_json(osc_spec):
    min = None
    max = None
    options = None
    access = None

    if isinstance(osc_spec, dict):
        types = osc_spec['TYPE']
        if 'ACCESS' in osc_spec:
            access = osc_spec['ACCESS']

        for type in types:
            # instantiate oscwidget
            # how do we specify widget?
            # how do we create nodes (location?)
            # do we include other properties in the oscquery file
            # ????
            if type == 'f': # float
                value = osc_spec['VALUE']
                if 'RANGE' in osc_spec:
                    if 'MAX' in osc_spec['RANGE']:
                        max = osc_spec['RANGE']['MAX']
                    if 'MIN' in osc_spec['RANGE']:
                        min = osc_spec['RANGE']['MIN']
                # create node
            elif type == 'i': # integer
                value = osc_spec['VALUE']
                if 'RANGE' in osc_spec:
                    if 'MAX' in osc_spec['RANGE']:
                        max = osc_spec['RANGE']['MAX']
                    if 'MIN' in osc_spec['RANGE']:
                        min = osc_spec['RANGE']['MIN']
                # create node
            elif type == 's': # string or string options
                options = None
                value = osc_spec['VALUE']
                if 'RANGE' in osc_spec:
                    # this would be a set of options (menu or radio buttons)
                    if 'VALS' in osc_spec['RANGE']:
                        options = osc_spec['RANGE']['VALS']
                # create node
            elif type == 'T': # toggle
                value = osc_spec['VALUE']
                if 'RANGE' in osc_spec:
                    if 'MAX' in osc_spec['RANGE']:
                        max = osc_spec['RANGE']['MAX']
                    if 'MIN' in osc_spec['RANGE']:
                        min = osc_spec['RANGE']['MIN']
                # create node
            elif type == 'r': # color
                value_string = osc_spec['VALUE']
                if isinstance(value_string, str):
                    if value_string[0] == '#':
                        value = hex_to_rgb(value_string)
                # create node



