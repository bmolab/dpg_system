import ctypes
import os
import multiprocessing
import numpy as np

from dpg_system.node import Node
from dpg_system.conversion_utils import *
import time
import os

import threading
from queue import Queue

from dpg_system.Cairo_Text_Layout import *

import random
from datetime import datetime


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
        self.include_prompt = self.add_input('include prompt', widget_type='checkbox', default_value=True)
        self.image_output = self.add_output('layout')
        self.previous_layout_length = 0
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

    def execute(self):
        if self.layout is not None:
            data = self.input()
            t = type(data)
            if t == str:
                data = string_to_list(data)
                t = list
            if t == list:
                if type(data[0]) == str:
                    if data[0] == 'save':
                        dir = os.getcwd()
                        date = time.strftime("%d-%m-%Y-%H_%M_%S")
                        self.layout.save_layout_as_text(dir + os.sep + 'llama_output_' + date + '.txt')
                    if data[0] == 'add':
                        self.previous_layout_length = len(self.layout.layout)
                        self.add_text([data[1:]])
                    elif data[0] == 'step_back':
                        count = any_to_int(data[1])
                        self.step_back(count, redraw=False)
                        self.streaming_prompt_active = False
                    elif data[0] == 'temperature':
                        self.temp = any_to_float(data[1])
                        self.layout.temp = self.temp
                    elif data[0] == 'prompt':
                        if self.include_prompt():
                            temp_temp = self.temp
                            self.temp = 0.8
                            self.add_text([data[1:]])
                            self.temp = temp_temp
                    elif data[0] == 'streaming_prompt':
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
                    elif data[0] == 'accept_streamed_prompt':
                        if self.streaming_prompt_active:
                            self.streaming_prompt_active = False
                    elif data[0] == 'backspace_streaming_prompt':
                        if self.streaming_prompt_active and len(self.layout.layout) > self.streaming_prompt_pos:
                            if len(self.layout.layout[-1][1]) > 0:
                                self.layout.layout[-1][1] = self.layout.layout[-1][1][:-1]
                            self.display_layout()
                    elif data[0] == 'choice_list':
                        self.add_text_with_choices(data[1], data[2])
                    elif data[0] == 'choose':
                        self.chosen_index = data[1]
                        self.layout.choose_from_next_word_list(self.chosen_index)
                        self.display_layout()
                    elif data[0] == 'show_probs':
                        self.showing_poss_dict = data[1]
                        old_active_line = self.layout.active_line
                        if self.showing_poss_dict:
                            self.layout.set_active_line(5)
                            self.active_line.set(5)
                        else:
                            self.layout.set_active_line(17)
                            self.active_line.set(17)
                        self.layout.adjust_active_line(old_active_line)
                        self.display_layout()
                    elif data[0] == 'clear' or data[0] == 'reset':
                        self.clear_layout()
                    elif data[0] == 'scroll_up':
                        self.scroll_up()
                    elif data[0] == 'scroll_down':
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
            if len(t) == 1:
                t = [0, t[0]]
            out_string += t[1]
            temper = self.temp
            spread_color = [1.0, 1.0, 1.0]
            if temper > 1:
                temper = int((1.0 - ((temper - 1.0) / 8)) * 255)
                if temper < 0:
                    temper = 0
                elif temper > 255:
                    temper = 255
                spread_color = (self.cmap2[temper][0], self.cmap2[temper][1], self.cmap2[temper][2])
            else:
                temper = int(((temper - 0.5) * 2) * 255)
                if temper < 0:
                    temper = 0
                elif temper > 255:
                    temper = 255
                spread_color = (self.cmap3[temper][0], self.cmap3[temper][1], self.cmap3[temper][2])

            if '\\' in t[1]:
                t[1].replace('\\n', '\n')

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

    def reset_layout(self):
        self.layout.clear_layout()

    def idle_routine(self, dt=0.016):
        if self.layout:
            layout_image = self.layout.dest[..., 0:3].astype(np.float32) / 255
            if self.layout.do_animate_scroll:
                self.layout.animate_scroll(self.layout.scroll_delta)

            if layout_image is not None:
                self.image_output.send(layout_image)

    def make_heatmap(self):
        red = [1.0, 0.0, 0.0]
        orange = [1.0, 0.5, 0.0]
        yellow = [1.0, 1.0, 0.0]
        white = [1.0, 1.0, 1.0]
        cmap = []

        colors = [red, orange, yellow, white]
        positions = [0, 85, 170, 256]

        for idx, color in enumerate(colors):
            start_position = positions[idx]
            if idx < len(colors) - 1:
                position = start_position
                next_position = positions[idx + 1]
                next_color = colors[idx + 1]
                while position < next_position:
                    progress = float(position - start_position) / float(next_position - start_position)
                    this_color = [0.0, 0.0, 0.0]
                    this_color[0] = color[0] * (1.0 - progress) + next_color[0] * progress
                    this_color[1] = color[1] * (1.0 - progress) + next_color[1] * progress
                    this_color[2] = color[2] * (1.0 - progress) + next_color[2] * progress
                    cmap.append(this_color)
                    position += 1
        return cmap

    def make_coldmap(self):
        blue = [0.5, 0.0, 0.8]
        green = [0.5, 0.3, 1.0]
        yellow = [0.5, 0.8, 1.0]
        white = [1.0, 1.0, 1.0]
        cmap = []

        colors = [blue, green, yellow, white]
        positions = [0, 85, 170, 256]

        for idx, color in enumerate(colors):
            start_position = positions[idx]
            if idx < len(colors) - 1:
                position = start_position
                next_position = positions[idx + 1]
                next_color = colors[idx + 1]
                while position < next_position:
                    progress = float(position - start_position) / float(next_position - start_position)
                    this_color = [0.0, 0.0, 0.0]
                    this_color[0] = color[0] * (1.0 - progress) + next_color[0] * progress
                    this_color[1] = color[1] * (1.0 - progress) + next_color[1] * progress
                    this_color[2] = color[2] * (1.0 - progress) + next_color[2] * progress
                    cmap.append(this_color)
                    position += 1
        return cmap


class CairoTextLayoutNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CairoTextLayoutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        width = 1920
        height = 1080
        if len(args) > 1:
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
                if t == list:
                    if type(data[0]) == str:
                        if data[0] == 'save':
                            dir = os.getcwd()
                            date = time.strftime("%d-%m-%Y-%H_%M_%S")
                            self.layout.save_layout_as_text(dir + os.sep + 'llama_output_' + date + '.txt')
                        if data[0] == 'add':
                            self.add_text(data[1:])
                        elif data[0] == 'add_char':
                            self.add_text(data[1:], add_space=False)
                        elif data[0] == 'clear' or data[0] == 'reset':
                            self.clear_layout()
                        elif data[0] == 'scroll_up':
                            self.scroll_up()
                        elif data[0] == 'scroll_down':
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

# ISSUE HERE OF OVERNESTED LIST
    def add_text(self, new_data, add_space=True):
        tp = type(new_data)

        if tp is str:
            new_data = string_to_list(new_data)
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
                        tt[0].replace('\\n', '\n')
                    self.layout.add_string([tt], add_space)

        self.display_layout()

    def step_back(self):
        if len(self.layout.layout) > 2:
            self.layout.step_back(step_size=2)
            self.display_layout()

    def display_layout(self):
        self.layout.draw_layout()
        self.idle_routine()

    def reset_layout(self):
        self.layout.clear_layout()

    def idle_routine(self, dt=0.016):
        if self.layout:
            layout_image = self.layout.dest[..., 0:3].astype(np.float32) / 255
            # if self.layout.do_animate_scroll:
            #     self.layout.animate_scroll(self.layout.scroll_delta)

            if layout_image is not None:
                self.image_output.send(layout_image)
