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

