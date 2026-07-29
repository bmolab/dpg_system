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
    Node.app.register_node('prompt_composer', PromptComposerNode.factory)


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
        self.output = self.add_output('weighted prompt out')

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
                if isinstance(prompt[1], (float, int, np.floating, np.integer)):
                    relative_weight = float(prompt[1])
                elif type(prompt[1]) == str:
                    relative_weight = any_to_float(prompt[1])
        if self.subprompts[index] and self.subprompts[index][-1] == ' ':
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


class PromptComposerNode(Node):
    """Merge live STT phrases (fifo_string 'weighted out') with enduring
    context (context_tracker 'context out') into one weighted prompt list
    for the image generator, which does the blending internally.

    The phrase block passes through in EXACTLY the order the fifo emitted
    it (run the fifo newest_at_start to lead with the in-progress text).
    'newest phrase at' does not reorder anything — it tells the budget
    logic which end of the incoming list is newest, so drops take the
    oldest end first and the newest entry always survives.

    Policies: context items already present in the live phrases are
    soft-deduped (weight scaled down, not removed); total prompt length
    is held to a char and chunk budget by dropping oldest phrases first,
    then lowest-weight context — never the newest phrase, prefix or suffix."""
    @staticmethod
    def factory(name, data, args=None):
        node = PromptComposerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.phrases = []
        self.context = []
        self.prefix = ''
        self.suffix = ''
        self.lock = threading.Lock()

        self.phrases_input = self.add_input('phrases', triggers_execution=True)
        self.context_input = self.add_input('context', triggers_execution=True)
        self.prefix_input = self.add_input('prefix', callback=self.receive_prefix)
        self.suffix_input = self.add_input('suffix', callback=self.receive_suffix)
        self.clear_input = self.add_input('clear', widget_type='button', callback=self.clear)

        self.order = self.add_property('order', widget_type='combo', width=180, default_value='context_first')
        self.order.widget.combo_items = ['context_first', 'phrases_first']
        self.newest_at = self.add_property('newest phrase at', widget_type='combo', width=100, default_value='end')
        self.newest_at.widget.combo_items = ['end', 'start']
        self.char_budget = self.add_property('char budget', widget_type='drag_int', default_value=300)
        self.max_chunks = self.add_property('max chunks', widget_type='drag_int', default_value=8)
        self.dedupe_scale = self.add_property('dedupe scale', widget_type='drag_float', default_value=0.25)
        self.prefix_weight = self.add_property('prefix weight', widget_type='drag_float', default_value=1.0)
        self.suffix_weight = self.add_property('suffix weight', widget_type='drag_float', default_value=1.0)
        self.strength = self.add_property('strength', widget_type='drag_float', default_value=1.0)

        self.output = self.add_output('weighted prompt out')
        self.string_output = self.add_output('string out')

    @staticmethod
    def _as_weighted(data):
        # normalize incoming data to [[text, weight], ...], dropping
        # empty texts and expired (weight <= 0) entries
        entries = []
        if type(data) == str:
            if data != '':
                entries.append([data, 1.0])
        elif type(data) == list:
            for item in data:
                if type(item) in [list, tuple] and len(item) >= 2:
                    text = any_to_string(item[0]).strip()
                    weight = any_to_float(item[1])
                    if text != '' and weight > 0:
                        entries.append([text, weight])
                elif type(item) == str:
                    if item.strip() != '':
                        entries.append([item.strip(), 1.0])
        return entries

    def receive_prefix(self, value=None):
        self.prefix = any_to_string(self.prefix_input()).strip()
        self.compose_and_send()

    def receive_suffix(self, value=None):
        self.suffix = any_to_string(self.suffix_input()).strip()
        self.compose_and_send()

    def clear(self, value=None):
        self.phrases = []
        self.context = []
        self.compose_and_send()

    def execute(self):
        if self.phrases_input.fresh_input:
            self.phrases = self._as_weighted(self.phrases_input())
        if self.context_input.fresh_input:
            self.context = self._as_weighted(self.context_input())
        self.compose_and_send()

    def compose_and_send(self):
        self.lock.acquire(blocking=True)
        phrases = [list(p) for p in self.phrases]
        context = [list(c) for c in self.context]
        self.lock.release()

        # soft dedupe: context already voiced in the live phrase window
        # fades rather than doubling up (it returns to full weight once
        # the phrase scrolls out of the fifo)
        dedupe_scale = self.dedupe_scale()
        phrase_text = ' '.join(p[0] for p in phrases).lower()
        for c in context:
            if c[0].lower() in phrase_text:
                c[1] *= dedupe_scale
        context.sort(key=lambda c: c[1], reverse=True)
        context_entries = context                       # sorted weight desc

        # phrases pass through in EXACTLY the fifo's order; 'newest at'
        # only identifies which end is newest for budget protection
        if self.newest_at() == 'start':
            droppable_phrases = phrases[:0:-1]  # oldest first
        else:
            droppable_phrases = phrases[:-1]    # oldest first

        entries = []
        if self.prefix != '':
            entries.append([self.prefix, self.prefix_weight()])
        if self.order() == 'phrases_first':
            entries += phrases
            entries += context_entries
        else:
            entries += context_entries
            entries += phrases
        if self.suffix != '':
            entries.append([self.suffix, self.suffix_weight()])

        # enforce budgets: drop oldest phrases first, then lowest-weight
        # context; prefix, suffix and the newest phrase always survive
        candidates = droppable_phrases + sorted(context_entries, key=lambda c: c[1])

        char_budget = self.char_budget()
        max_chunks = self.max_chunks()

        def over_budget(current):
            chars = sum(len(e[0]) for e in current) + max(0, len(current) - 1)
            return len(current) > max_chunks or chars > char_budget

        while len(candidates) > 0 and over_budget(entries):
            victim = candidates.pop(0)
            entries = [e for e in entries if e is not victim]

        strength = self.strength()
        prompt_list = [[e[0], e[1] * strength] for e in entries]
        self.output.send(prompt_list)
        self.string_output.send(' '.join(e[0] for e in entries))


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
        self.output = self.add_output('weighted prompt out')
        self.width_option = self.add_option('width', widget_type='drag_int', default_value=200, callback=self.set_size)

    def set_size(self):
        for i in range(self.count):
            dpg.set_item_width(self.prompt_inputs[i].widget.uuid, self.width_option())

    def clear(self):
        # Reset per-input state once, not once per iteration. Previously
        # the inside-loop reset left self.subprompts with length 1 (just
        # the last iteration's append) instead of self.count, so the next
        # process_prompt(index) for index >= 1 raised IndexError.
        for i in range(self.count):
            self.prompt_inputs[i].set('')
        self.subprompts = [''] * self.count
        self.subprompt_weights = [0.0] * self.count
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

