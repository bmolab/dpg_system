import ctypes
import os
import multiprocessing
import numpy as np

from dpg_system.node import Node
from dpg_system.conversion_utils import *
import time

import threading
from queue import Queue

from dpg_system.Cairo_Text_Layout import *
from dpg_system.colormaps import _viridis_data
import random
from datetime import datetime


def _to_byte(v):
    return max(0, min(255, int(v)))


def register_layout_nodes():
    Node.app.register_node('llm_layout', TextLayoutNode.factory)
    Node.app.register_node('cairo_layout', CairoTextLayoutNode.factory)


class TextLayoutNode(Node):
    llama_nodes = []

    @staticmethod
    def factory(name, data, args=None):
        node = TextLayoutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.sleep = 0.00
        self.layout = None
        self.input = self.add_input('input', triggers_execution=True)
        self.clear_button = self.add_input('clear', widget_type='button', callback=self.clear_layout)
        self.active_line = self.add_input('active_line', widget_type='input_int', default_value=17, callback=self.active_line_changed)
        self.color_mode_input = self.add_input('colour_mode', widget_type='combo', default_value='temperature')
        self.color_mode_input.widget.combo_items = ['temperature', 'entropy', 'probability']
        self.include_prompt = self.add_input('include prompt', widget_type='checkbox', default_value=True)
        self.image_output = self.add_output('layout')
        self.previous_layout_length = 0
        self.cmap = _viridis_data
        self.cmap2 = make_heatmap()
        self.cmap3 = make_coldmap()
        self.layout = LLMLayout([0, 0, 1920, 1080])
        self.layout.set_active_line(17)  # 17
        # self.active_line.set(17)
        self.layout.show_list = False  # False
        self.show_choices = False
        self.chosen_index = 0
        self.speech_lines = 0
        self.max_speech_lines = 100
        self.temp = 1.0
        self.poss_dict = []
        self.new_poss_dict = False
        self.showing_poss_dict = False
        self.streaming_prompt_active = False
        self.streaming_prompt_pos = 0
        self.entropy = 0.1
        self.probability = 0.5

    def execute(self):
        if self.layout is not None:
            data = self.input()
            t = type(data)
            if t == str:
                data = string_to_list(data)
                t = list
            if t == list and len(data) > 0 and type(data[0]) == str:
                command = data[0]
                if command == 'save':
                    dir = os.getcwd()
                    date = time.strftime("%d-%m-%Y-%H_%M_%S")
                    self.layout.save_layout_as_text(dir + os.sep + 'llama_output_' + date + '.txt')
                elif command == 'add':
                    if len(data) > 1:
                        self.previous_layout_length = len(self.layout.layout)
                        self.add_text([data[1:]])
                elif command == 'step_back':
                    if len(data) > 1:
                        count = any_to_int(data[1])
                        self.step_back(count, redraw=False)
                        self.streaming_prompt_active = False
                elif command == 'prompt':
                    if len(data) > 1 and self.include_prompt():
                        temp_temp = self.temp
                        self.temp = 0.8
                        self.add_text([data[1:]])
                        self.temp = temp_temp
                elif command == 'streaming_prompt':
                    if not self.streaming_prompt_active:
                        self.streaming_prompt_active = True
                        if self.layout.show_list:
                            self.streaming_prompt_pos = self.previous_layout_length
                        else:
                            self.streaming_prompt_pos = len(self.layout.layout)
                    delete_element_count = len(self.layout.layout) - self.streaming_prompt_pos
                    if delete_element_count > 0:
                        self.layout.step_back(delete_element_count)
                    if len(data) > 1:
                        pairs = data[1]
                        self.add_text(pairs)
                elif command == 'accept_streamed_prompt':
                    if self.streaming_prompt_active:
                        self.streaming_prompt_active = False
                elif command == 'backspace_streaming_prompt':
                    if self.streaming_prompt_active and len(self.layout.layout) > self.streaming_prompt_pos:
                        if len(self.layout.layout[-1][1]) > 0:
                            self.layout.layout[-1][1] = self.layout.layout[-1][1][:-1]
                        self.display_layout()
                elif command == 'choice_list':
                    if len(data) > 2:
                        self.add_text_with_choices(data[1], data[2])
                elif command == 'choose':
                    if len(data) > 1:
                        self.chosen_index = any_to_int(data[1])
                        self.layout.choose_from_next_word_list(self.chosen_index)
                        self.display_layout()
                elif command == 'show_probs':
                    if len(data) > 1:
                        self.showing_poss_dict = any_to_bool(data[1])
                        old_active_line = self.layout.active_line
                        if self.showing_poss_dict:
                            self.layout.set_active_line(5)
                            self.active_line.set(5)
                        else:
                            self.layout.set_active_line(17)
                            self.active_line.set(17)
                        self.layout.adjust_active_line(old_active_line)
                        self.display_layout()
                elif command == 'clear' or command == 'reset':
                    self.clear_layout()
                elif command == 'scroll_up':
                    self.scroll_up()
                elif command == 'scroll_down':
                    self.scroll_down()

    def scroll_up(self):
        old_active_line = self.layout.active_line
        old_active_line -= 1
        self.active_line.set(old_active_line)
        self.active_line_changed()

    def scroll_down(self):
        old_active_line = self.layout.active_line
        old_active_line += 1
        self.active_line.set(old_active_line)
        self.active_line_changed()

    def active_line_changed(self):
        old_active_line = self.layout.active_line
        self.layout.active_line = self.active_line()
        self.layout.adjust_active_line(old_active_line)
        self.display_layout()

    def clear_layout(self):
        self.layout.clear_layout()
        self.display_layout()

    def add_text_with_choices(self, poss_dict, selected ):
        self.poss_dict = poss_dict
        self.chosen_index = selected
        self.new_poss_dict = True

    def add_text(self, new_data):
        # if type(new_data) == list:
        #     if type(new_data[0]) == list:
        #         new_data = new_data[0]
        out_string = ''
        for t in new_data:
            colour_index = None
            if len(t) == 1:
                t = [0, t[0]]
            if len(t) > 2:
                colour_index = t[2]
            out_string += t[1]

            spread_color = [255, 255, 255]

            if colour_index is not None:
                mode = self.color_mode_input()
                if mode == 'temperature':
                    if colour_index > 1:
                        temper = _to_byte((1.0 - ((colour_index - 1.0) / 8)) * 255)
                        spread_color = (self.cmap2[temper][0], self.cmap2[temper][1], self.cmap2[temper][2])
                    else:
                        temper = _to_byte(((colour_index - 0.5) * 2) * 255)
                        spread_color = (self.cmap3[temper][0], self.cmap3[temper][1], self.cmap3[temper][2])
                elif mode == 'entropy':
                    entropy = 255 - _to_byte(colour_index * 64)
                    spread_color = (self.cmap[entropy][0], self.cmap[entropy][1], self.cmap[entropy][2])
                else:
                    probability = _to_byte(colour_index * 255)
                    spread_color = (self.cmap[probability][0], self.cmap[probability][1], self.cmap[probability][2])
            if '\\' in t[1]:
                t[1] = t[1].replace('\\n', '\n')
                if t[1].startswith('\\c'):
                    self.clear_layout()


            self.layout.add_word(t[1], spread_color, t[0])
            if self.new_poss_dict:
                self.layout.show_list = True
                self.layout.clear_list()
                spread_color = self.layout.display_next_word_list_all(self.poss_dict, self.chosen_index)
                self.new_poss_dict = False
            else:
                self.layout.show_list = False

            self.display_layout()

    def step_back(self, count=2, redraw=True):
        if len(self.layout.layout) > count:
            self.layout.step_back(step_size=count)
            if redraw:
                self.display_layout()

    def display_layout(self):
        self.layout.draw_layout()
        self.idle_routine()

    def idle_routine(self, dt=0.016):
        if self.layout:
            layout_image = self.layout.dest[..., 0:3].astype(np.float32) / 255
            if self.layout.do_animate_scroll:
                self.layout.animate_scroll(self.layout.scroll_delta)

            if layout_image is not None:
                self.image_output.send(layout_image)


class CairoTextLayoutNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CairoTextLayoutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        width = 1920
        height = 1080
        if args is not None and len(args) > 1:
            width = any_to_int(args[0])
            height = any_to_int(args[1])
        self.sleep = 0.00
        self.layout = None
        self.input = self.add_input('input', triggers_execution=True)
        self.command_input = self.add_input('command', triggers_execution=True)
        self.clear_button = self.add_input('clear', widget_type='button', callback=self.clear_layout)
        self.font_input = self.add_input('font path', widget_type='text_input', default_value='', callback=self.font_changed)
        self.font_size_input = self.add_input('font size', widget_type='drag_float', default_value=40, callback=self.font_size_changed)
        self.text_brightness = self.add_input('brightness', widget_type='drag_float', default_value=1.0)
        self.alpha_power = self.add_input('alpha power', widget_type='drag_float', default_value=0.1)
        self.leading_input = self.add_input('leading', widget_type='drag_float', default_value=60, callback=self.leading_changed)
        self.active_line = self.add_input('active_line', widget_type='input_int', default_value=17, callback=self.active_line_changed)
        self.wrap_input = self.add_input('wrap text', widget_type='checkbox', default_value=True, callback=self.wrap_changed)
        self.image_output = self.add_output('layout')
        self.language = None

        self.layout = CairoTextLayout([0, 0, width, height])
        self.layout.set_active_line(1)  # 17

    def leading_changed(self):
        self.layout.leading = self.leading_input()

    def wrap_changed(self):
        self.layout.wrap_text = self.wrap_input()

    def font_changed(self):
        self.layout.get_font(self.font_input())

    def font_size_changed(self):
        self.layout.set_font_size(self.font_size_input())

    def execute(self):
        if self.layout is not None:
            if self.active_input is self.input:
                data = self.input()
                self.clear_layout()
                self.add_text(data)
            elif self.active_input is self.command_input:
                data = self.command_input()
                t = type(data)
                if t == str:
                    data = string_to_list(data)
                    t = list
                if t == list and len(data) > 0 and type(data[0]) == str:
                    command = data[0]
                    if command == 'save':
                        dir = os.getcwd()
                        date = time.strftime("%d-%m-%Y-%H_%M_%S")
                        self.layout.save_layout_as_text(dir + os.sep + 'llama_output_' + date + '.txt')
                    elif command == 'add':
                        if len(data) > 1:
                            self.add_text(data[1:])
                    elif command == 'add_char':
                        if len(data) > 1:
                            added_char = data[1]
                            if added_char == '':
                                added_char = ' '
                        else:
                            added_char = ' '
                        self.add_text(added_char, add_space=False)
                    elif command == 'clear' or command == 'reset':
                        self.clear_layout()
                    elif command == 'scroll_up':
                        self.scroll_up()
                    elif command == 'scroll_down':
                        self.scroll_down()
                    elif command == 'delete_line':
                        self.delete_line()

    def scroll_up(self):
        old_active_line = self.layout.active_line
        old_active_line -= 1
        self.active_line.set(old_active_line)
        self.active_line_changed()

    def scroll_down(self):
        old_active_line = self.layout.active_line
        old_active_line += 1
        self.active_line.set(old_active_line)
        self.active_line_changed()

    def active_line_changed(self):
        old_active_line = self.layout.active_line
        self.layout.active_line = self.active_line()
        self.layout.adjust_active_line(old_active_line)
        self.display_layout()

    def clear_layout(self):
        self.layout.clear_layout()
        self.display_layout()

    def delete_line(self):
        self.layout.step_back_to_last_return()
        self.display_layout()

# ISSUE HERE OF OVERNESTED LIST
    def add_text(self, new_data, add_space=True):
        tp = type(new_data)
        if new_data == ' ':
            new_data = [' ']
            tp = type(new_data)
        if tp is str:
            new_data = string_to_list(new_data)
        else:
            new_data = any_to_list(new_data)
        for t in new_data:
            if type(t) == str:
                t = [t, 1.0]
            elif type(t) is not list:
                t = [any_to_string(t), 1.0]
            if len(t) > 1:
                tt = t.copy()
                tt[1] = pow(tt[1], self.alpha_power()) * self.text_brightness()
                if tt[1] != 0.0:
                    if '\\' in tt[0]:
                        tt[0] = tt[0].replace('\\n', '\n')
                        if tt[0] == '\\c':
                            self.clear_layout()
                            tt[0] = ''
                        elif tt[0] == '\\d':
                            self.delete_line()
                            tt[0] = ''
                    self.layout.add_string([tt], add_space)

        self.display_layout()

    def step_back(self):
        if len(self.layout.layout) > 2:
            self.layout.step_back(step_size=2)
            self.display_layout()

    def display_layout(self):
        self.layout.draw_layout()
        self.idle_routine()

    def idle_routine(self, dt=0.016):
        if self.layout:
            layout_image = self.layout.dest[..., 0:3].astype(np.float32) / 255
            # if self.layout.do_animate_scroll:
            #     self.layout.animate_scroll(self.layout.scroll_delta)

            if layout_image is not None:
                self.image_output.send(layout_image)
