import dearpygui.dearpygui as dpg
import math
import numpy as np
import random
import time

import torch

from dpg_system.node import Node
import threading
from dpg_system.conversion_utils import *
import json
from fuzzywuzzy import fuzz


def register_prompt_nodes():
    Node.app.register_node('ambient_prompt', AmbientPromptNode.factory)
    Node.app.register_node('weighted_prompt', WeightedPromptNode.factory)


class AmbientPromptNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = AmbientPromptNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.count = 6
        self.subprompts = []
        self.subprompt_weights = []
        self.prompt_inputs = []
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t == int:
                self.count = v

        for i in range(self.count):
            self.subprompts.append('')
            self.subprompt_weights.append(0.0)
            self.prompt_inputs.append(self.add_input('in_' + str(i), widget_type='text_input', default_value='', triggers_execution=True))

        # self.clear_input = self.add_input('clear', callback=self.clear_fifo)
        self.output = self.add_output("weighted prompt out")

    def execute(self):
        index = self.active_input.input_index
        prompt = self.prompt_inputs[index]()
        relative_weight = 0.0
        if type(prompt) == str:
            prompt_split = prompt.split('@')
            self.subprompts[index] = prompt_split[0]
            if len(prompt_split) > 1:
                relative_weight = any_to_float(prompt_split[1])
        elif type(prompt) == list:
            if len(prompt) == 2:
                if type(prompt[0]) == str:
                    self.subprompts[index] = prompt[0]
                if type(prompt[1]) in [float, int]:
                    relative_weight = prompt[1]
                elif type(prompt[1]) == str:
                    relative_weight = any_to_float(prompt[1])
        if self.subprompts[index][-1] == ' ':
            self.subprompts[index] = self.subprompts[index][:-1]

        self.subprompt_weights[index] = relative_weight
        ambient_prompt_string = ''
        for i in range(len(self.subprompts)):
            square_bracket_count = 0
            parentheses_count = 0
            if len(self.subprompts[i]) > 0:
                if self.subprompt_weights[i] < 0:
                    square_bracket_count = int(-self.subprompt_weights[i])
                    for j in range(square_bracket_count):
                        ambient_prompt_string += '['
                    ambient_prompt_string += self.subprompts[i]
                    for j in range(square_bracket_count):
                        ambient_prompt_string += ']'
                    ambient_prompt_string += ', '
                else:
                    parentheses_count = int(self.subprompt_weights[i])
                    for j in range(parentheses_count):
                        ambient_prompt_string += '('
                    ambient_prompt_string += self.subprompts[i]
                    for j in range(parentheses_count):
                        ambient_prompt_string += ')'
                    ambient_prompt_string += ', '
        self.output.send(ambient_prompt_string)


class WeightedPromptNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = WeightedPromptNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.count = 6
        self.subprompts = []
        self.subprompt_weights = []
        self.prompt_inputs = []
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t == int:
                self.count = v

        self.clear_button = self.add_property('clear', widget_type='button', callback=self.clear)
        for i in range(self.count):
            self.subprompts.append('')
            self.subprompt_weights.append(0.0)
            self.prompt_inputs.append(self.add_input('##' + str(i), widget_type='text_input', widget_width=200, default_value='', triggers_execution=True))

        # self.clear_input = self.add_input('clear', callback=self.clear_fifo)
        self.strength = self.add_input('strength', widget_type='drag_float', default_value=1.0, triggers_execution=True)
        self.output = self.add_output("weighted prompt out")
        self.width_option = self.add_option("width", widget_type='drag_int', default_value=200, callback=self.set_size)

    def set_size(self):
        for i in range(self.count):
            dpg.set_item_width(self.prompt_inputs[i].widget.uuid, self.width_option())

    def clear(self):
        for i in range(self.count):
            self.prompt_inputs[i].set('')
            self.subprompts = []
            self.subprompt_weights = []
            self.subprompts.append('')
            self.subprompt_weights.append(0.0)
        self.output.send([])

    def process_prompt(self, index):
        prompt = self.prompt_inputs[index]()
        relative_weight = self.subprompt_weights[index]
        if is_number(prompt):
            prompt = any_to_float(prompt)
        if type(prompt) == str:
            prompt_split = prompt.split('@')
            self.subprompts[index] = prompt_split[0]
            if len(prompt_split) > 1:
                relative_weight = any_to_float(prompt_split[1])
            if len(self.subprompts[index]) > 0:
                self.prompt_inputs[index].set(self.subprompts[index] + '@{:.3f}'.format(relative_weight))
        elif type(prompt) == list:
            sub = ''
            for i in range(len(prompt)):
                if type(prompt[i]) == str:
                    if len(sub) > 0:
                        sub += ' '
                    sub += prompt[i]
                elif type(prompt[i]) == float:
                    relative_weight = prompt[i]
            if sub != '':
                self.subprompts[index] = sub
            if len(self.subprompts[index]) > 0:
                self.prompt_inputs[index].set(self.subprompts[index] + '@{:.3f}'.format(relative_weight))
        elif type(prompt) in [float, int]:
            relative_weight = any_to_float(prompt)
            if len(self.subprompts[index]) > 0:
                self.prompt_inputs[index].set(self.subprompts[index] + '@{:.3f}'.format(relative_weight))

        if len(self.subprompts[index]) > 0:
            if self.subprompts[index][-1] == ' ':
                self.subprompts[index] = self.subprompts[index][:-1]

        self.subprompt_weights[index] = relative_weight

    def load_custom(self, container):
        for i in range(self.count):
            self.process_prompt(i)

    def execute(self):
        index = self.active_input.input_index
        if index < self.count:
            self.process_prompt(index)
        strength = self.strength()
        ambient_prompt_list = []
        for i in range(len(self.subprompts)):
            if self.subprompts[i] != '':
                entry = [self.subprompts[i], self.subprompt_weights[i] * strength]
                ambient_prompt_list.append(entry)
        self.output.send(ambient_prompt_list)

