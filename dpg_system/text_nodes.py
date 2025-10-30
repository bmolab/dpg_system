import dearpygui.dearpygui as dpg
import math
import numpy as np
import random
import time
import string
import sys
import os
import torch
import copy
import queue
import html
import string

from dpg_system.node import Node, SaveDialog, LoadDialog

import threading
from dpg_system.conversion_utils import *
import json
from fuzzywuzzy import fuzz

NOPRINT_TRANS_TABLE = {
    i: None for i in range(0, sys.maxunicode + 1) if not chr(i).isprintable()
}


def register_text_nodes():
    Node.app.register_node('text_change', TextChangeNode.factory)
    Node.app.register_node("combine", CombineNode.factory)
    Node.app.register_node("kombine", CombineNode.factory)
    Node.app.register_node('word_replace', WordReplaceNode.factory)
    Node.app.register_node('string_replace', StringReplaceNode.factory)
    Node.app.register_node('word_trigger', WordTriggerNode.factory)
    Node.app.register_node('first_letter_trigger', WordFirstLetterNode.factory)
    Node.app.register_node('gather_sentence', GatherSentences.factory)
    Node.app.register_node('string_builder', StringBuilder.factory)
    Node.app.register_node('character', CharConverterNode.factory)
    Node.app.register_node('ascii', ASCIIConverterNode.factory)
    Node.app.register_node('char', CharConverterNode.factory)
    Node.app.register_node('ord', ASCIIConverterNode.factory)
    Node.app.register_node('printable', PrintableNode.factory)
    Node.app.register_node('text_file', TextFileNode.factory)
    Node.app.register_node('text_editor', TextFileNode.factory)
    Node.app.register_node('unescape_text', UnescapeHTMLNode.factory)
    Node.app.register_node('word_gate', WordGateNode.factory)
    Node.app.register_node('fuzzy_match', FuzzyMatchNode.factory)
    Node.app.register_node('split', SplitNode.factory)
    Node.app.register_node('join', JoinNode.factory)
    Node.app.register_node('fifo_string', CombineFIFONode.factory)


class TextChangeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TextChangeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.text_input = self.add_input('text input', triggers_execution=True)
        self.persistence = self.add_input('persistence', widget_type='input_int', default_value=3)
        self.reset_period = self.add_input('reset_period', widget_type='input_int', default_value=10)
        self.clear_input = self.add_input('clear', widget_type='button', callback=self.clear)
        self.output = self.add_output('new words out')
        self.remove_punctuation = str.maketrans('', '', string.punctuation)
        self.current_list = {}

    def clear(self):
        self.current_list = {}

    def execute(self):
        input_text = self.text_input()
        if type(input_text) is list:
            input_text = list(flatten_list(input_text))
            input_text = ''.join(input_text)
        if type(input_text) == str:
            word_list = input_text.split(' ')
        else:
            word_list = input_text
        new_words = []
        dead_words = []
        out_words = []
        lower_case_word_list = [x.lower() for x in word_list]
        word_list = lower_case_word_list
        for index, word in enumerate(word_list):
            word = word.translate(self.remove_punctuation)
            word_list[index] = word
            if word not in self.current_list:
                if word not in new_words:
                    new_words.append(word)
                if word not in dead_words:
                    out_words.append(word)
        for word in self.current_list:
            self.current_list[word] += 1
            if self.current_list[word] <= self.persistence():
                if word not in out_words:
                    out_words.append(word)
            else:
                if word not in word_list:
                    if self.current_list[word] > self.reset_period():
                        if word not in dead_words:
                            dead_words.append(word)

        for word in dead_words:
            self.current_list.pop(word, None)
        for word in new_words:
            if word not in self.current_list:
                self.current_list[word] = 0
        out_list = []
        for word in self.current_list:
            if word in out_words:
                out_list.append(word)
        self.output.send(' '.join(out_list))


class CombineNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CombineNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.combine_list = []

        self.num_ins = self.arg_as_int(default_value=2)

        for i in range(self.num_ins):
            if i == 0:
                input_ = self.add_input("in " + str(i + 1), triggers_execution=True)
            else:
                if label == 'kombine':
                    input_ = self.add_input("in " + str(i + 1), triggers_execution=True)
                else:
                    input_ = self.add_input("in " + str(i + 1))
            input_._data = ''

        self.output = self.add_output("out")

    def execute(self):
        output_string = ''
        for i in range(self.num_ins):
            output_string += any_to_string(self.inputs[i]._data)
        self.output.send(output_string)


class WordReplaceNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = WordReplaceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        find = ''
        replace = ''
        if len(args) > 1:
            find = any_to_string(args[0])
            replace = any_to_string(args[1])
        self.input = self.add_string_input('string in', triggers_execution=True)
        self.find_input = self.add_string_input('find', widget_type='text_input', default_value=find)
        self.replace_input = self.add_string_input('replace', widget_type='text_input', default_value=replace)
        self.output = self.add_output('string out')

    def execute(self):
        data = self.input()
        find = self.find_input()
        replace = self.replace_input()
        if find != '':
            if type(data) == list:
                for i, w in enumerate(data):
                    if type(w) == str:
                        if w == find:
                            data[i] = replace
                self.output.send(data)
            elif type(data) == str:
                if type(data) == str:
                    data = re.sub(r"\b{}\b".format(find), replace, data)
                    self.output.send(data)
                else:
                    self.output.send(data)
            else:
                self.output.send(data)
        else:
            self.output.send(data)


class StringReplaceNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = StringReplaceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        find = ''
        replace = ''
        if len(args) > 1:
            find = any_to_string(args[0])
            replace = any_to_string(args[1])
        self.input = self.add_input('string in', triggers_execution=True)
        self.find_input = self.add_input('find', widget_type='text_input', default_value=find)
        self.replace_input = self.add_input('replace', widget_type='text_input', default_value=replace)
        self.output = self.add_output('string out')

    def execute(self):
        data = self.input()
        find = self.find_input()
        replace = self.replace_input()
        if find != '':
            if type(data) == list:
                for i, w in enumerate(data):
                    if type(w) == str:
                        w = re.sub(r"{}".format(find), replace, w)
                        data[i] = w
                self.output.send(data)
            elif type(data) == str:
                data = re.sub(r"{}".format(find), replace, data)
                self.output.send(data)
            else:
                self.output.send(data)
        else:
            self.output.send(data)


class WordTriggerNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = WordTriggerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.find_list = []
        if len(args) > 0:
            for i in range(len(args)):
                self.find_list.append(args[i])
        self.input = self.add_input('string in', triggers_execution=True)
        self.trigger_outputs = []
        for i in range(len(args)):
            self.trigger_outputs.append(self.add_output(self.find_list[i]))

    def execute(self):
        data = any_to_string(self.input()).lower()
        if len(self.find_list) > 0:
            for index, word_trigger in enumerate(self.find_list):
                if re.search(r'\b{}\b'.format(word_trigger), data) is not None:
                    self.trigger_outputs[index].send('bang')


class WordFirstLetterNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = WordFirstLetterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.find_list = []
        if len(args) > 0:
            for i in range(len(args)):
                self.find_list.append(args[i])
        self.input = self.add_input('string in', triggers_execution=True)
        self.trigger_outputs = []
        for i in range(len(args)):
            self.trigger_outputs.append(self.add_output(self.find_list[i]))

    def execute(self):
        data = any_to_string(self.input()).lower()
        if len(self.find_list) > 0:

            for index, first_letter_trigger in enumerate(self.find_list):
                if data[0] == first_letter_trigger:
                    self.trigger_outputs[index].send('bang')
                elif data.find(' ' + first_letter_trigger) != -1:
                    self.trigger_outputs[index].send('bang')


class GatherSentences(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = GatherSentences(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.received_sentence = ''
        self.received_tokens = []
        self.input = self.add_input('string in', triggers_execution=True)
        self.end_input = self.add_input('force string end', triggers_execution=True)
        self.enforce_spaces = self.add_property('enforce spaces', widget_type='checkbox')
        self.auto_sentence_end = self.add_property('auto sentence end', widget_type='checkbox', default_value=True)
        self.end_on_return = self.add_property('end on return', widget_type='checkbox')
        self.skip_framed_by = self.add_property('skip framed by', widget_type='text_input')
        self.sentence_output = self.add_output('sentences out')

    def convert_list_of_tokens_to_string(self):
        if len(self.received_tokens) > 0:
            self.received_sentence = ''.join(self.received_tokens)
        else:
            self.received_sentence = ''

    def execute(self):
        self.skipper = self.skip_framed_by()

        if self.active_input == self.input:
            data = any_to_string(self.input())
            if len(data) > 0:
                if data == '<backspace>':
                    if len(self.received_tokens) > 0:
                        self.received_tokens = self.received_tokens[:-1]
                        return
                if self.auto_sentence_end():
                    if data[-1] == '\n' and len(data) > 1:
                        if data[-2] == '\n':
                            self.received_tokens.append(data)
                            self.send_sentence()
                            return
                    if data[-1] == '-' and len(data) > 1:
                        if data[-2] == '-':
                            self.received_tokens.append(data)
                            self.send_sentence()
                            return
                    elipsis = False
                    if data[-1] == '.' and len(data) > 2:
                        if data[-2] == '.' and data[-3] == '.':
                            elipsis = True
                    if not elipsis and data[-1] in ['.', '?', '!', ';', ':']:
                        self.received_tokens.append(data)
                        self.send_sentence()
                        return

                    if data[-1] == ')' and len(self.received_tokens) > 0:
                        if self.received_tokens[0] == '' and len(self.received_tokens) > 1:
                            if self.received_tokens[1][-1] == '(':
                                self.received_tokens.append(data)
                                self.send_sentence()
                                return
                            elif self.received_tokens[1][0] == '(':
                                self.received_tokens.append(data)
                                self.send_sentence()
                                return
                        elif self.received_tokens[0][-1] == '(':
                            self.received_tokens.append(data)
                            self.send_sentence()
                            return
                        elif self.received_tokens[0][0] == '(':
                            self.received_tokens.append(data)
                            self.send_sentence()
                            return
                elif self.end_on_return():
                    if data[0] == '\n':
                        self.send_sentence()

            if self.enforce_spaces() and len(self.received_sentence) > 0 and len(data) > 0:
                if self.received_tokens[-1][-1] != ' ' and data[0] != ' ':
                    self.received_tokens.append(' ')
                    self.received_sentence += ' '
            if len(data) > 0:
                self.received_tokens.append(data)
        else:
            self.send_sentence()

    def send_sentence(self):
        skipping = False
        self.convert_list_of_tokens_to_string()
        self.received_tokens = []
        self.skipper = self.skip_framed_by()
        if len(self.skipper) == 1:
            stripped_sentence = ''
            for i in range(len(self.received_sentence)):
                if self.received_sentence[i] == self.skipper:
                    if not skipping:
                        skipping = True
                    else:
                        skipping = False
                elif not skipping:
                    stripped_sentence += self.received_sentence[i]
            self.received_sentence = stripped_sentence
        if self.received_sentence.isprintable() == False:
            self.received_sentence = self.received_sentence.translate(NOPRINT_TRANS_TABLE)

        self.sentence_output.send(self.received_sentence)
        self.received_sentence = ''


class StringBuilder(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = StringBuilder(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.received_sentence = ''
        self.trigger_input = self.add_input('issue text', triggers_execution=True, trigger_button=True)
        self.input = self.add_input('string in', triggers_execution=True)
        self.sentence_output = self.add_output('text out')

    def execute(self):
        if self.active_input == self.input:
            data = any_to_string(self.input())
            if len(self.received_sentence) > 0 and len(data) > 0:
                if self.received_sentence[-1] != ' ' and data[0] != ' ':
                    self.received_sentence += ' '
            self.received_sentence += data
        elif self.active_input == self.trigger_input:
            if self.trigger_input() == 'clear':
                self.received_sentence = ''
            else:
                self.sentence_output.send(self.received_sentence)
                self.received_sentence = ''


class CharConverterNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CharConverterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('char code in', triggers_execution=True)
        self.output = self.add_output('character out')

    def execute(self):
        code = any_to_int(self.input())
        char = chr(code)
        self.output.send(char)


class ASCIIConverterNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ASCIIConverterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('character in', triggers_execution=True)
        self.output = self.add_output('ascii out')

    def execute(self):
        codes = []
        received = self.input()
        if received == '\n':
            codes.append(10)
        else:
            string = any_to_string(self.input())
            for char in string:
                code = ord(char)
                codes.append(code)
        if len(codes) == 1:
            self.output.send(codes[0])
        else:
            self.output.send(codes)


class PrintableNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = PrintableNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('characters in', triggers_execution=True)
        self.output = self.add_output('printable characters out')

    def execute(self):
        a_string = any_to_string(self.input())
        filtered_string = ''.join(filter(lambda x: x in string.printable, a_string))
        if len(filtered_string) == 1:
            self.output.send(filtered_string[0])
        elif len(filtered_string) > 0:
            self.output.send(filtered_string)


class TextFileNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TextFileNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.text_contents = ''
        self.file_name = ''
        self.save_pointer = -1
        self.read_pointer = -1
        self.char_pointer = -1

        self.file_name = self.arg_as_string(default_value='')
        self.dump_button = self.add_input('send', widget_type='button', triggers_execution=True)

        self.text_editor = self.add_string_input('##text in', widget_type='text_editor', widget_width=500, callback=self.new_text)

        # self.text_input = self.add_string_input('text in', triggers_execution=True)
        self.text_editor.set_strip_returns(False)
        self.append_text_input = self.add_string_input('append text in', triggers_execution=True)
        self.append_text_input.set_strip_returns(False)

        self.clear_button = self.add_input('clear', widget_type='button', callback=self.clear_text)
        self.load_button = self.add_input('load', widget_type='button', callback=self.load_message)
        self.save_button = self.add_input('save', widget_type='button', callback=self.save_message)
        self.output = self.add_output("out")
        self.message_out = self.add_output("messages")

        self.file_name_property = self.add_option('name', widget_type='text_input', width=500, default_value=self.file_name)
        self.editor_width = self.add_option('editor width', widget_type='drag_int', default_value=500, callback=self.adjust_editor)
        self.editor_height = self.add_option('editor height', widget_type='drag_int', default_value=200, callback=self.adjust_editor)
        self.message_handlers['output_char'] = self.output_character_message
        self.message_handlers['next'] = self.next_character
        self.message_handlers['reset_pointer'] = self.reset_char_pointer

    def new_text(self):
        data = any_to_string(self.text_editor(), strip_returns=False)
        self.text_contents = data

    def reset_char_pointer(self, message='', data=[]):
        self.char_pointer = -1

    def next_character(self, message='', data=[]):
        self.char_pointer += 1
        if self.char_pointer >= len(self.text_contents):
            self.message_out.send('done')
            self.char_pointer = -1
        else:
            self.output_character_message('output_char', [self.char_pointer])

    def output_character_message(self, message='', data=[]):
        if len(data) > 0:
            char_pos = any_to_int(data[0])
            if char_pos < len(self.text_contents) and char_pos >= 0:
                out_char = self.text_contents[char_pos]
                if out_char == '\\':
                    if char_pos + 1 < len(self.text_contents):
                        out_char_2 = self.text_contents[char_pos + 1]
                        if out_char_2 == 'n':
                            out_char = '\n'
                            self.char_pointer += 1
                        elif out_char_2 == 't':
                            out_char = '\t'
                            self.char_pointer += 1
                        elif out_char_2 == 'c':
                            out_char = '\c'
                            self.char_pointer += 1
                        elif out_char_2 == 'd':
                            out_char = '\d'
                            self.char_pointer += 1
                self.output.send(out_char)
        self.input_handled = True

    def clear_text(self):
        self.text_contents = ''
        self.text_editor.set(self.text_contents)

    def adjust_editor(self):
        dpg.set_item_width(self.text_editor.widget.uuid, self.editor_width())
        dpg.set_item_height(self.text_editor.widget.uuid, self.editor_height())

    def post_load_callback(self):
        if self.file_name_property() != '':
            self.load_data(self.file_name_property())

    def save_dialog(self):
        SaveDialog(self, callback=self.save_text_file_callback, extensions=['.txt'])
        # with dpg.file_dialog(directory_selector=False, show=True, height=400, width=800, user_data=self, callback=save_text_file_callback, cancel_callback=cancel_textfile_callback,
        #                      tag="text_dialog_id"):
        #     dpg.add_file_extension(".txt")

    def save_text_file_callback(self, save_path):
        if save_path != '':
            self.save_data(save_path)
        else:
            print('no file chosen')

    def save_data(self, path):
        self.text_contents = self.text_editor()
        with open(path, 'w+') as f:
            f.write(self.text_contents)
        self.file_name_property.set(path)

    def save_message(self):
        data = self.save_button()
        if data is not None:
            path = any_to_string(data)
            self.save_data(path)
        else:
            self.save_dialog()

    def load_dialog(self):
        LoadDialog(self, callback=self.load_text_file_callback, extensions=['.txt'])

    def load_text_file_callback(self, load_path):
        if load_path != '':
            self.load_data(load_path)
        else:
            print('no file chosen')

    def load_message(self):
        data = self.load_button()
        if data is not None:
            path = any_to_string(data)
            self.load_data(path)
        else:
            self.load_dialog()

    def load_data_from_file_name(self):
        self.load_data(self.file_name_property())

    def load_data(self, path):
        try:
            with open(path, 'r') as f:
                self.text_contents = f.read()
            self.file_name_property.set(path)
            self.text_editor.set(self.text_contents)
        except FileNotFoundError:
            print('TextFile node error:', path, 'not found')

    def execute(self):
        if self.active_input == self.dump_button:
            self.text_contents = self.text_editor()
            self.output.send(self.text_contents)
        elif self.active_input == self.append_text_input:
            self.text_contents = self.text_editor()
            data = any_to_string(self.append_text_input(), strip_returns=False)
            # if len(self.text_contents) > 0:
            #     if self.text_contents[-1] not in [' ', '\n']:
            #         self.text_contents += ' '
            self.text_contents += data
            self.text_editor.set(self.text_contents)
        else:
            data = any_to_string(self.text_editor(), strip_returns=False)
            self.text_contents = data

def cancel_textfile_callback(sender, app_data):
    if sender is not None:
        dpg.delete_item(sender)
    Node.app.active_widget = -1


class UnescapeHTMLNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = UnescapeHTMLNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('escaped string in', triggers_execution=True)
        self.output = self.add_output('unescaped string out')

    def execute(self):
        escaped = any_to_string(self.input())
        unescaped = html.unescape(escaped)
        self.output.send(unescaped)


class WordGateNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = WordGateNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('string in', triggers_execution=True)
        self.dict_in = self.add_input('dictionary in', callback=self.receive_dict)
        self.dict_path_in = self.add_input('load dictionary', widget_type='text_input', callback=self.load_dict)
        self.add_word_input = self.add_input('include word', callback=self.include_word)
        self.output = self.add_output('gated string out')
        self.gate_dict = {}

    def load_dict(self):
        path = self.dict_path_in()
        if type(path) is str:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.gate_dict = json.load(f)

    def include_word(self):
        word = self.add_word_input()
        if type(word) is str:
            self.gate_dict[word] = word
        elif type(word) is list:
            for w in word:
                if type(w) is str:
                    self.gate_dict[w] = w

    def receive_dict(self):
        d = self.dict_in()
        if type(d) is dict:
            self.gate_dict = d

    def execute(self):
        incoming = any_to_string(self.input())
        gated_list = []
        previous_word = ''
        incoming_list = re.split(r"[^a-zA-Z']+", incoming)
        incoming_list = [word for word in incoming_list if word]
        for word in incoming_list:
            if word in self.gate_dict:
                gated_list.append(self.gate_dict[word])
                previous_word = word.lower()
            elif word.lower() in self.gate_dict:
                gated_list.append(self.gate_dict[word.lower()])
                previous_word = word.lower()
            elif previous_word != '' and previous_word + ' ' + word in self.gate_dict:
                gated_list.append(self.gate_dict[previous_word + ' ' + word])
                previous_word = ''
            elif previous_word != '' and previous_word + ' ' + word.lower() in self.gate_dict:
                gated_list.append(self.gate_dict[previous_word + ' ' + word.lower()])
                previous_word = ''
            else:
                previous_word = word.lower()
        if len(gated_list) > 0:
            gated_string = ' '.join(gated_list)
            self.output.send(gated_string)


class FuzzyMatchNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = FuzzyMatchNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('string in', triggers_execution=True)
        self.load_button = self.add_property('load match file', width=120, widget_type='button', callback=self.load_match_file)
        self.file_name = self.add_label('')
        self.threshold = self.add_property('threshold', widget_type='drag_float', default_value=60)
        self.output = self.add_output('string out')
        self.score_output = self.add_output('score out')
        self.replacement_output = self.add_output('replacement out')
        self.load_path = self.add_option('path', widget_type='text_input', default_value='', callback=self.load_from_load_path)

        self.filtered_list = []
        self.option_list = []
        self.best_score = 0
        self.list_path = ''
        if len(args) > 0:
            self.list_path = args[0]
            path = self.list_path.split('/')[-1]
            self.file_name.set(path)
            f = open(args[0])
            data = json.load(f)
            for artist in data:
                self.option_list.append(artist)

    def load_match_file(self):
        LoadDialog(self, callback=self.load_match_file_callback, extensions=['.json'])
        # with dpg.file_dialog(modal=True, directory_selector=False, show=True, height=400, width=800,
        #                      user_data=self, callback=self.load_match_file_callback, tag="match_file_dialog_id"):
        #     dpg.add_file_extension(".json")

    def load_match_file_callback(self, path):
        if path != '':
            self.load_path.set(path)
            self.load_from_load_path()
        else:
            print('no file chosen')

    def load_from_load_path(self):
        path = self.load_path()
        print('load_from_load_path', path)
        if path != '':
            self.load_match_file_from_json(self.load_path())

    def load_match_file_from_json(self, path):
        print('load from json')
        self.list_path = path
        if os.path.exists(path):
            with open(path, 'r') as f:
                path = self.list_path.split('/')[-1]
                self.file_name.set(path)
                data = json.load(f)
                self.option_list = []
                for artist in data:
                    self.option_list.append(artist)

    def execute(self):
        if self.input.fresh_input:
            start = time.time()
            prestring = ''
            substring = ''
            post_string = ''
            pass_data = False
            found = False
            indicator = ''
            data = self.input()
            if type(data) == list:
                data = ' '.join(data)
            if type(data) == str:
                index = data.find('by ')
                if index != -1:
                    if index == 0 or data[index - 1] in [' ', ',', ':', ';']:
                        index += 3
                        if data[index:index + 2] != 'a ' and data[index:index + 4] != 'the ':
                            indicator = 'by '
                            substring = data[index:]
                            prestring = data[:index]
                            index = substring.find(' of ')
                            index2 = substring.find(' from ')
                            if index != -1:
                                if index2 != -1:
                                    if index < index2:
                                        post_string = substring[index:]
                                        substring = substring[:index]
                                        found = True
                                    else:
                                        post_string = substring[index2:]
                                        substring = substring[:index2]
                                        found = True
                                else:
                                    post_string = substring[index:]
                                    substring = substring[:index]
                                    found = True
                            elif index2 != -1:
                                post_string = substring[index2:]
                                substring = substring[:index2]
                                found = True
                if not found:
                    index = data.find('style of ')
                    if index != -1:
                        indicator = 'style of '
                        index += len('style of ')
                        substring = data[index:]
                        prestring = data[:index]
                        found = True
                    # if not found:
                    #     print('else')
                    #     if len(data) <= 32 and data.count(' ') <= 5:
                    #         print('test upper')
                    #         uppers = 0
                    #         names = data.split(' ')
                    #         for name in names:
                    #             if name[0].isupper():
                    #                 uppers += 1
                    #         if uppers > 1:
                    #             substring = data
                elif len(substring) > 32 or substring.count(' ') > 5:
                    substring = data
                    pass_data = True

                if not pass_data:
                    self.fuzzy_score(substring)

                    self.score_output.send(self.best_score)
                    if len(self.filtered_list) > 0 and self.best_score > self.threshold():
                        self.output.send(prestring + self.filtered_list[0] + post_string)
                        self.replacement_output.send(indicator + self.filtered_list[0])
                    else:
                        self.output.send(data)
                else:
                    self.output.send(data)

    def fuzzy_score(self, test):
        scores = {}
        for index, node_name in enumerate(self.option_list):
            ratio = fuzz.partial_ratio(node_name.lower(), test.lower())
            full_ratio = fuzz.ratio(node_name.lower(), test.lower())        # partial matchi should be less important if size diff is big
            len_ratio = len(test) / len(node_name)
            if len_ratio > 1:
                len_ratio = 1 / len_ratio
            len_ratio = len_ratio * .5 + 0.5  # 0.25 - 0.75
            ratio = (ratio * (1 - len_ratio) + full_ratio * len_ratio)
            scores[node_name] = ratio

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self.filtered_list = []
        self.best_score = 20
        for index, item in enumerate(sorted_scores):
            if item[1] == 100:
                self.filtered_list.append(item[0])
                self.best_score = item[1]
            elif item[1] > self.best_score and len(self.filtered_list) < 10:
                self.filtered_list.append(item[0])
                self.best_score = item[1]


class SplitNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SplitNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('in', triggers_execution=True)
        self.split_token = ''
        if len(args) > 0:
            self.split_token = any_to_string(args[0])
            if self.split_token == '\\n':
                self.split_token = '<return>'
        self.split_token_in = self.add_input('split at', widget_type='text_input', default_value=self.split_token)
        self.output = self.add_output("substrings out")

    def execute(self):
        in_string = self.input()
        t = type(in_string)
        if t == list:
            in_string = ' '.join(in_string)
        if self.split_token == '':
            splits = in_string.split()
        elif self.split_token == '<return>':
            splits = in_string.split('\n')
        else:
            splits = in_string.split(self.split_token_in())
        # splits = in_string.split(self.split_token())
        self.output.send(splits)


class JoinNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = JoinNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('in', triggers_execution=True)
        join_token = ' '
        if len(args) > 0:
            join_token = any_to_string(args[0])

        self.join_token = self.add_input('join with', widget_type='text_input', default_value=join_token)
        self.output = self.add_output("string out")

    def execute(self):
        in_list = self.input()
        t = type(in_list)
        if t == str:
            in_list = in_list.split(' ')
            t = list
        elif t != list:
            in_list = any_to_list(in_list)
        if t is list:
            for index, el in enumerate(in_list):
                in_list[index] = any_to_string(el)

        joined = self.join_token().join(in_list)
        self.output.send(joined)


class CombineFIFONode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CombineFIFONode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.count = 4
        self.pointer = 0
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t == int:
                self.count = v

        self.combine_list = [''] * self.count
        self.age = [1.0] * self.count
        self.last_time = time.time()
        self.decay_rate = .01
        self.length_threshold = 100

        self.input = self.add_input("in", triggers_execution=True)
        self.progress_input = self.add_input("progress", triggers_execution=True)

        self.clear_input = self.add_input('clear', callback=self.clear_fifo)
        self.drop_oldest_input = self.add_input('dump_oldest', callback=self.dump_oldest)
        self.order = self.add_property('order', widget_type='combo', width=150, default_value='newest_at_end')
        self.order.widget.combo_items = ['newest_at_end', 'newest_at_start']
        self.decay_rate_property = self.add_property('decay_rate', widget_type='drag_float',
                                                     default_value=self.decay_rate)
        self.length_threshold_property = self.add_property('length_threshold', widget_type='drag_int',
                                                           default_value=self.length_threshold)
        self.output = self.add_output("weighted out")
        self.string_output = self.add_output("string out")
        self.last_was_progress = False
        self.lock = threading.Lock()

    def dump_oldest(self, value):
        for i in range(self.count):
            j = (self.pointer - i) % self.count
            if self.combine_list[j] != '':
                self.combine_list[j] = ''
                break
        self.execute()

    def clear_fifo(self, value=0):
        self.combine_list = [''] * self.count
        output_string = ''
        self.output.send(self.combine_list)
        self.string_output.send(output_string)

    def advance_age(self):
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now
        decay = self.decay_rate_property() * elapsed
        for i in range(len(self.age)):
            self.age[i] -= decay
            if self.age[i] < 0:
                self.age[i] = 0

    def execute(self):
        empty_phrase = False
        self.lock.acquire(blocking=True)

        # note: TODO: if progress chunk is getting too long, split...
        if self.progress_input.fresh_input:
            progress = any_to_string(self.progress_input())
            if progress != '' and progress != ' ':
                self.advance_age()

                p = self.pointer
                self.combine_list[p] = progress
                self.age[p] = 1.0
                self.last_was_progress = True
            else:
                if not self.input.fresh_input:
                    self.lock.release()
                    return

        if self.input.fresh_input:
            # TODO: add way of dumping older phrases if the leading phrases are too long.
            # TODO: keep track of overall output length. If too large, drop out older phrases until we are at a good length
            phrase = any_to_string(self.input())
            self.last_was_progress = False
            self.advance_age()

            # split super long phrases into sub-phrases
            length = len(phrase)
            if length > self.length_threshold_property():
                sub_phrases = re.split(r'[\!\?\.\:\;\,]', phrase)
            else:
                sub_phrases = phrase.split('.')

            # flag situations where a period potentially splitting a phrase is just part of Mr. Dr. Mrs. Ms. St. etc
            joiners = []
            for index, sp in enumerate(sub_phrases):
                if len(sp) == 1 and sp[0] == ' ':
                    sp = ''
                if len(sp) > 0:
                    if len(sp) > 1 and sp[-1] == 'r':  # Do not split at the period after Dr or Mr
                        if sp[-2] in ['D', 'M']:
                            joiners.append(index)
                    elif len(sp) > 2 and sp[-1] == 's':  # Do not split at the period after Mrs
                        if sp[-2] == 'r' and sp[-3] == 'M':
                            joiners.append(index)
                    if len(sp) > 1 and sp[-1] == 't':  # Do not split at the period after St
                        if sp[-2] == 'S':
                            joiners.append(index)

            join_next = False
            adjusted_phrases = []

            for index, p in enumerate(sub_phrases):
                if len(p) == 1 and p[0] == ' ':
                    p = ''
                if len(p) > 0:
                    if join_next and len(
                            adjusted_phrases) > 0:  # join sub phrases split by periods identified above
                        adjusted_phrases[-1] = adjusted_phrases[-1] + p + '.'
                        join_next = False
                    else:
                        if p[-1] not in ['.', '?',
                                         '!']:  # make sure there is at least a terminating period at the end of a sub phrase
                            adjusted_phrases.append(p + '.')
                        else:
                            adjusted_phrases.append(p)

                    if index in joiners:
                        join_next = True

            # add sub-phrases to the fifo and shift pointer
            for p in adjusted_phrases:
                self.combine_list[self.pointer] = p
                self.age[self.pointer] = 1.0
                self.pointer = (self.pointer - 1) % self.count
            if len(adjusted_phrases) == 0:
                empty_phrase = True

        output_string_list = []
        output_string = ''
        pointer = self.pointer

        # if the last thing received was 'progress', we will replace the progress string with the phrase(s)
        if self.last_was_progress or empty_phrase:
            pointer = (self.pointer - 1) % self.count

        # added string out
        if self.order() == 'newest_at_end':
            for i in range(self.count):
                j = (pointer - i) % self.count
                if self.combine_list[j] != '':
                    output_string_list.append([any_to_string(self.combine_list[j]), self.age[j]])
                    if self.age[j] > 0:
                        output_string += (self.combine_list[j] + ' ')
        else:
            for i in range(self.count):
                j = (pointer + i + 1) % self.count
                if self.combine_list[j] != '':
                    output_string_list.append([any_to_string(self.combine_list[j]), self.age[j]])
                    if self.age[j] > 0:
                        output_string += (self.combine_list[j] + ' ')
        self.output.send(output_string_list)
        self.string_output.send(output_string)
        self.lock.release()



