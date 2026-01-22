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
import itertools
from typing import List, Any, Callable, Union, Tuple, Optional, Dict, Set, Type, TypeVar, cast

from dpg_system.node import Node, SaveDialog, LoadDialog

import threading
from dpg_system.conversion_utils import *
import json
from fuzzywuzzy import fuzz


def register_basic_nodes():
    Node.app.register_node('prepend', PrependNode.factory)
    Node.app.register_node('append', AppendNode.factory)
    Node.app.register_node("type", TypeNode.factory)
    Node.app.register_node("info", TypeNode.factory)
    Node.app.register_node('array', ArrayNode.factory)
    Node.app.register_node("counter", CounterNode.factory)
    Node.app.register_node("range_counter", RangeCounterNode.factory)
    Node.app.register_node('dict', DictNode.factory)
    Node.app.register_node('construct_dict', ConstructDictNode.factory)
    Node.app.register_node('gather_to_dict', ConstructDictNode.factory)
    Node.app.register_node('dict_extract', DictExtractNode.factory)
    Node.app.register_node('unpack_dict', DictExtractNode.factory)
    Node.app.register_node('dict_stream', DictStreamNode.factory)

    Node.app.register_node("concat", ConcatenateNode.factory)

    Node.app.register_node("delay", DelayNode.factory)
    Node.app.register_node("select", SelectNode.factory)
    Node.app.register_node("route", RouteNode.factory)
    Node.app.register_node("gate", GateNode.factory)
    Node.app.register_node("switch", SwitchNode.factory)
    Node.app.register_node("metro", MetroNode.factory)
    Node.app.register_node("unpack", UnpackNode.factory)
    Node.app.register_node("pack", PackNode.factory)
    Node.app.register_node("pak", PackNode.factory)
    Node.app.register_node('repeat', RepeatNode.factory)
    Node.app.register_node('repeat_in_order', RepeatNode.factory)
    Node.app.register_node("timer", TimerNode.factory)
    Node.app.register_node("elapsed", TimerNode.factory)
    Node.app.register_node("date_time", DateTimeNode.factory)

    Node.app.register_node("decode", SelectNode.factory)
    Node.app.register_node("decode_message", DecodeToNode.factory)
    Node.app.register_node("t", TriggerNode.factory)
    Node.app.register_node('var', VariableNode.factory)
    Node.app.register_node('send', ConduitSendNode.factory)
    Node.app.register_node('receive', ConduitReceiveNode.factory)
    Node.app.register_node('s', ConduitSendNode.factory)
    Node.app.register_node('r', ConduitReceiveNode.factory)
    Node.app.register_node('ramp', RampNode.factory)
    Node.app.register_node('bucket_brigade', BucketBrigadeNode.factory)
    Node.app.register_node('tick', TickNode.factory)
    Node.app.register_node('comment', CommentNode.factory)
    Node.app.register_node('text_block', TextBlockNode.factory)
    Node.app.register_node('length', LengthNode.factory)
    Node.app.register_node('time_between', TimeBetweenNode.factory)
    Node.app.register_node('int_replace', IntReplaceNode.factory)
    Node.app.register_node('replace', ReplaceNode.factory)
    Node.app.register_node('defer', DeferNode.factory)
    Node.app.register_node('present', PresentationModeNode.factory)
    Node.app.register_node('clamp', ClampNode.factory)
    Node.app.register_node('save', SaveNode.factory)
    Node.app.register_node('close', ClosePatchNode.factory)
    Node.app.register_node('active_widget', ActiveWidgetNode.factory)
    Node.app.register_node('pass_with_triggers', TriggerBeforeAndAfterNode.factory)
    Node.app.register_node('micro_metro', MicrosecondTimerNode.factory)
    Node.app.register_node('stream_list', StreamListNode.factory)
    Node.app.register_node('directory_iterator', NPZDirectoryIteratorNode.factory)
    Node.app.register_node('patch_window_position', PositionPatchesNode.factory)
    Node.app.register_node('slice_list', SliceNode.factory)

    Node.app.register_node('start_trace', StartTraceNode.factory)
    Node.app.register_node('end_trace', EndTraceNode.factory)

    Node.app.register_node('dict_replace', DictReplaceNode.factory)
    Node.app.register_node('sublist', SublistNode.factory)
    Node.app.register_node('dict_search', DictNavigatorNode.factory)
    Node.app.register_node('dict_keys', DictKeysNode.factory)
    Node.app.register_node('list_box', ListBoxNode.factory)
    Node.app.register_node('dict_retrieve', DictRetrieveNode.factory)

class SliceNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SliceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.list_input = self.add_input('list input', triggers_execution=True)
        self.slice_after = self.add_input('slice after', widget_type='input_int', default_value=0)
        self.output_1 = self.add_output('slice 1 out')
        self.output_2 = self.add_output('slice 2 out')
        self.output_only_if_both = self.add_option('output only if slice 2', widget_type='checkbox',
                                                   default_value=False)

    def execute(self):
        incoming = self.list_input()
        if type(incoming) is str:
            incoming = incoming.split(' ')
        first_slice = []
        second_slice = []
        if type(incoming) is list:
            if len(incoming) > self.slice_after() + 1:
                first_slice = incoming[:self.slice_after() + 1]
                second_slice = incoming[self.slice_after() + 1:]
            else:
                first_slice = incoming
            if self.output_only_if_both():
                if len(second_slice) > 0:
                    self.output_2.send(second_slice)
                    self.output_1.send(first_slice)
            else:
                self.output_2.send(second_slice)
                self.output_1.send(first_slice)


class ActiveWidgetNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ActiveWidgetNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.active_widget_property = self.add_property('active_widget', widget_type='drag_int')
        self.add_frame_task()

    def frame_task(self):
        self.active_widget_property.set(self.app.active_widget)


class TriggerBeforeAndAfterNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TriggerBeforeAndAfterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('input', triggers_execution=True)
        self.post_trigger_output = self.add_output('post_trigger')
        self.output = self.add_output('pass input')
        self.pre_trigger_output = self.add_output('pre_trigger')

    def execute(self):
        data = self.input()
        self.pre_trigger_output.send('bang')
        self.output.send(data)
        self.post_trigger_output.send('bang')


# DeferNode -- delays received input until next frame
class DeferNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DeferNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('input', triggers_execution=True)
        self.output = self.add_output('deferred output')
        self.queue = queue.Queue()
        self.output_only_last = self.add_option('output only last', widget_type='checkbox', default_value=False)
        self.add_frame_task()

    def execute(self):
        self.queue.put(self.input())

    def frame_task(self):
        received_data = None
        while not self.queue.empty():
            try:
                received_data = self.queue.get_nowait()
                if not self.output_only_last():
                    self.output.send(received_data)
            except queue.Empty:
                break
        if self.output_only_last() and received_data is not None:
            self.output.send(received_data)

    def custom_cleanup(self):
        self.remove_frame_tasks()


class CommentNode(Node):
    comment_theme = None
    comment_text_theme = None
    inited = False

    @staticmethod
    def factory(name, data, args=None):
        node = CommentNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.comment_text = 'comment'
        if args is not None and len(args) > 0:
            self.comment_text = ' '.join(args)
        self.setup_themes()
        self.comment_label = self.add_label(self.comment_text)
        self.comment_text_option = self.add_option('text', widget_type='text_input', width=200, default_value=self.comment_text, callback=self.comment_changed)
        self.font_size_option = self.add_option('font size', widget_type='combo', default_value='24', callback=self.large_font_changed)
        self.font_size_option.widget.combo_items = ['24', '30', '36', '48']

    def setup_themes(self):
        if not CommentNode.inited:
            with dpg.theme() as CommentNode.comment_theme:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, [0, 0, 0, 0],
                                        category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, [0, 0, 0, 0],
                                        category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBar, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
            CommentNode.inited = True

    def large_font_changed(self):
        font_size = self.font_size_option()
        if font_size == '24':
            self.comment_label.widget.set_font(self.app.font_24)
        elif font_size == '30':
            self.comment_label.widget.set_font(self.app.font_30)
        elif font_size == '36':
            self.comment_label.widget.set_font(self.app.font_36)
        elif font_size == '48':
            self.comment_label.widget.set_font(self.app.font_48)
        self.comment_text_option.widget.adjust_to_text_width()


    def comment_changed(self):
        self.comment_text = self.comment_text_option()
        dpg.set_value(self.comment_label.widget.uuid, self.comment_text)
        self.comment_text_option.widget.adjust_to_text_width()

    def custom_create(self, from_file):
        dpg.bind_item_theme(self.uuid, CommentNode.comment_theme)
        dpg.configure_item(self.uuid, label='')

    def save_custom(self, container):
        container['name'] = 'comment'
        container['comment'] = self.comment_text

    def load_custom(self, container):
        self.comment_text = container['comment']
        dpg.bind_item_theme(self.uuid, CommentNode.comment_theme)
        dpg.configure_item(self.uuid, label='')

    def set_custom_visibility(self):
        dpg.configure_item(self.uuid, label='')
        dpg.bind_item_theme(self.uuid, CommentNode.comment_theme)


class TextBlockNode(Node):
    theme = None
    text_block_text_theme = None
    inited = False

    @staticmethod
    def factory(name, data, args=None):
        node = TextBlockNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.comment_text = 'text_block'
        if args is not None and len(args) > 0:
            self.comment_text = ' '.join(args)
        self.setup_themes()
        self.text_block = self.add_property('###block', widget_type='text_editor', width=400, default_value='')
        self.lock_option = self.add_option('lock', widget_type='checkbox', default_value=False, callback=self.lock)
        self.width_option = self.add_option('width', widget_type='drag_int',
                                                      default_value=400, callback=self.width_changed)
        self.height_option = self.add_option('height', widget_type='drag_int',
                                                      default_value=400, callback=self.height_changed)
        self.text_size_option = self.add_option('text_size', widget_type='combo', default_value='24', callback=self.font_size_changed)
        self.text_size_option.widget.combo_items = ['24', '30', '36', '48']

    def font_size_changed(self):
        font_size = self.text_size_option()
        if font_size == '24':
            self.text_block.widget.set_font(self.app.font_24)
        elif font_size == '30':
            self.text_block.widget.set_font(self.app.font_30)
        elif font_size == '36':
            self.text_block.widget.set_font(self.app.font_36)
        elif font_size == '48':
            self.text_block.widget.set_font(self.app.font_48)

    def lock(self):
        if self.lock_option():
            dpg.configure_item(self.text_block.widget.uuid, readonly=True)
        else:
            dpg.configure_item(self.text_block.widget.uuid, readonly=False)

    def width_changed(self):
        dpg.configure_item(self.text_block.widget.uuid, width=self.width_option())

    def height_changed(self):
        dpg.configure_item(self.text_block.widget.uuid, height=self.height_option())

    def setup_themes(self):
        if not TextBlockNode.inited:
            with dpg.theme() as TextBlockNode.theme:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, [0, 0, 0, 0],
                                        category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, [0, 0, 0, 0],
                                        category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBar, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBg, [0, 0, 0, 0], category=dpg.mvThemeCat_Core)
            TextBlockNode.inited = True

    def custom_create(self, from_file):
        dpg.bind_item_theme(self.uuid, TextBlockNode.theme)
        dpg.configure_item(self.uuid, label='')

    def save_custom(self, container):
        container['name'] = 'text_block'
        container['text'] = self.text_block()

    def load_custom(self, container):
        text = container['text']
        self.text_block.set(text)
        dpg.bind_item_theme(self.uuid, TextBlockNode.theme)
        dpg.configure_item(self.uuid, label='')

    def set_custom_visibility(self):
        dpg.configure_item(self.uuid, label='')
        dpg.bind_item_theme(self.uuid, TextBlockNode.theme)


class ClosePatchNode(Node):
    theme = None
    inited = False

    @staticmethod
    def factory(name, data, args=None):
        node = ClosePatchNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('close patch', widget_type='button', callback=self.close_call)
        self.setup_themes()

    def setup_themes(self):
        if not ClosePatchNode.inited:
            with dpg.theme() as ClosePatchNode.theme:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, [0, 0, 0, 0],
                                        category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, [0, 0, 0, 0],
                                        category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBar, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
            ClosePatchNode.inited = True

    def custom_create(self, from_file):
        dpg.bind_item_theme(self.uuid, ClosePatchNode.theme)
        dpg.configure_item(self.uuid, label='')
        self.input.widget.set_active_theme(Node.active_theme_yellow)
        dpg.set_item_height(self.input.widget.uuid, 28)

    def close_call(self):
        Node.app.close_current_node_editor()

    def save_custom(self, container):
        container['name'] = 'close'

    def load_custom(self, container):
        dpg.bind_item_theme(self.uuid, ClosePatchNode.theme)
        dpg.configure_item(self.uuid, label='')

    def set_custom_visibility(self):
        dpg.configure_item(self.uuid, label='')
        dpg.bind_item_theme(self.uuid, ClosePatchNode.theme)


class SaveNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SaveNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('save', widget_type='button', callback=self.save_call)

    def custom_create(self, from_file):
        self.input.widget.set_active_theme(Node.active_theme_green)
        dpg.set_item_height(self.input.widget.uuid, 28)

    def save_call(self):
        Node.app.save_nodes()


class TickNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TickNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('', widget_type='checkbox', default_value=True)
        self.output = self.add_output('')
        self.add_frame_task()

    def frame_task(self):
        if self.input():
            self.output.send('bang')


class MetroNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MetroNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # NOTE: __init__ does not create the node, but it defines the components of the node.
        # the actual dearpygui widgets and nodes do not exist, therefore cannot be modified, etc.
        # until they are submitted for creation which happens after __init__ is complete
        # custom_create() is called after that creation routine, allowing you to do any
        # special initialization that might be required that requires the UI elements to actually exist

        # set internal variables
        self.last_tick = 0
        self.on = False
        self.units = 1000
        self.units_dict = {'seconds': 1, 'milliseconds': 1000, 'minutes': 1.0/60.0, 'hours': 1.0/60.0/60.0}
        self.period = self.arg_as_float(30.0)
        self.streaming = False

        # set inputs / properties / outputs / options
        self.on_off_input = self.add_bool_input('on', widget_type='checkbox', callback=self.start_stop)
        self.period_input = self.add_float_input('period', widget_type='drag_float', default_value=self.period, callback=self.change_period)
        self.units_property = self.add_property('units', widget_type='combo', default_value='milliseconds', callback=self.set_units)
        self.units_property.widget.combo_items = ['seconds', 'milliseconds', 'minutes', 'hours']
        self.output = self.add_output('')

    def change_period(self, input=None):
        self.period = self.period_input()
        if self.period <= 0:
            self.period = .001

    def start_stop(self, input=None):
        self.on = self.on_off_input()
        if self.on:
            if not self.streaming:
                self.add_frame_task()
                self.streaming = True
                self.execute()
            self.last_tick = time.time()
        else:
            if self.streaming:
                self.remove_frame_tasks()
                self.streaming = False

    def set_units(self, input=None):
        units_string = self.units_property()
        if units_string in self.units_dict:
            self.units = self.units_dict[units_string]

    # this routine is called every update frame (usually 60 fps). It is optional... for those nodes that need constant updating

    def frame_task(self):
        if self.on:
            current = time.time()
            period_in_seconds = self.period / self.units
            if current - self.last_tick >= period_in_seconds:
                self.execute()
                self.last_tick = self.last_tick + period_in_seconds
                if self.last_tick + self.period < current:
                    self.last_tick = current

    # the execute function is what causes output. It is called whenever something is received in an input that declares a trigger_node
    # it can also be called from other functions like frame_task() above

    def execute(self):
        self.output.send('bang')


class MicrosecondTimerNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MicrosecondTimerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # set internal variables
        self.last_tick = 0
        self.on = False
        self.units = 1000
        self.units_dict = {'seconds': 1, 'milliseconds': 1000, 'minutes': 1.0/60.0, 'hours': 1.0/60.0/60.0}
        self.period = self.arg_as_float(30.0)
        self.streaming = False
        self.last_tick = 0
        # set inputs / properties / outputs / options
        self.on_off_input = self.add_bool_input('on', widget_type='checkbox', callback=self.start_stop)
        self.period_input = self.add_float_input('period', widget_type='drag_float', default_value=self.period, callback=self.change_period)
        self.units_property = self.add_property('units', widget_type='combo', default_value='milliseconds', callback=self.set_units)
        self.units_property.widget.combo_items = ['seconds', 'milliseconds', 'minutes', 'hours']
        self.output = self.add_output('')
        self.stop_event = threading.Event()
        self.worker_thread = None


    def change_period(self, input=None):
        self.period = self.period_input()
        if self.period <= 0:
            self.period = .001

    def start_stop(self, input=None):
        self.on = self.on_off_input()
        if self.on:
            if self.worker_thread is None or not self.worker_thread.is_alive():
                self.stop_event.clear()
                self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
                self.worker_thread.start()

                # Reset the timer when starting
            self.last_tick = time.perf_counter()
            # No need for an 'else' block; the worker loop handles the `self.on = False` state.

    def set_units(self, input=None):
        units_string = self.units_property()
        if units_string in self.units_dict:
            self.units = self.units_dict[units_string]

    def _worker_loop(self):
        """This function runs in a separate thread to generate bangs with high precision."""
        while not self.stop_event.is_set():
            if self.on:
                # Calculate the period in seconds, ensuring it's not zero
                period_in_seconds = self.period / self.units if self.units != 0 else 0.001
                if period_in_seconds <= 0:
                    period_in_seconds = 0.001  # Prevent zero/negative sleep time

                # Use a high-precision monotonic clock
                current_time = time.perf_counter()

                if current_time - self.last_tick >= period_in_seconds:
                    # It's time for a bang. Put it in the queue for the main thread to process.
                    # self.bang_queue.put('bang')
                    self.output.send('bang')
                    # Advance the last_tick time. This prevents drift over time.
                    self.last_tick += period_in_seconds

                    # Catch-up mechanism: If we've fallen behind by more than one period,
                    # reset the tick to the current time to avoid a burst of bangs.
                    if self.last_tick < current_time - period_in_seconds:
                        self.last_tick = current_time

                # Smart sleep: sleep for a short duration to yield CPU
                # This prevents a 100% CPU usage busy-wait loop.
                # A sleep time of 0.0005s (0.5 ms) is a good balance.
                time.sleep(0.0005)

            else:
                # If the metronome is off, sleep for a bit longer to reduce idle CPU usage
                time.sleep(0.01)

    # this routine is called every update frame (usually 60 fps). It is optional... for those nodes that need constant updating

    # def frame_task(self):
    #     if self.on:
    #         current = time.time()
    #         period_in_seconds = self.period / self.units
    #         if current - self.last_tick >= period_in_seconds:
    #             self.execute()
    #             self.last_tick = self.last_tick + period_in_seconds
    #             if self.last_tick + self.period < current:
    #                 self.last_tick = current

    # the execute function is what causes output. It is called whenever something is received in an input that declares a trigger_node
    # it can also be called from other functions like frame_task() above

    def execute(self):
        self.output.send('bang')

    def cleanup(self):
        """
        A custom cleanup method to be called when the node is deleted.
        Ensures the worker thread is stopped properly.
        """
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)  # Wait up to 1 sec for thread to finish
        super().cleanup()  # Call parent cleanup if it exists

    # In Python, __del__ can be unreliable. It's better to have an explicit
    # cleanup method that your framework calls when a node is destroyed.
    # If your framework supports it, use that. Otherwise, __del__ is a fallback.
    def __del__(self):
        self.cleanup()


class ClampNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ClampNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('in', triggers_execution=True)
        self.min_input = self.add_input('min', widget_type='drag_float', default_value=0.0)
        self.max_input = self.add_input('max', widget_type='drag_float', default_value=1.0)
        self.output = self.add_output('clamped output')

    def execute(self):
        data = self.input()
        min_v = self.min_input()
        max_v = self.max_input()

        # 1. Handle Lists (The specific fix you requested)
        if isinstance(data, list):
            # Convert to numpy for fast clipping, then convert back to list
            arr = any_to_array(data, validate=True)
            if arr is not None:
                # .tolist() converts the numpy array back to a standard Python list
                self.output.send(np.clip(arr, min_v, max_v).tolist())

        # 2. Handle Torch Tensors
        elif torch_available and isinstance(data, torch.Tensor):
            self.output.send(torch.clamp(data, min_v, max_v))

        # 3. Handle Numpy Arrays
        elif isinstance(data, np.ndarray):
            self.output.send(np.clip(data, min_v, max_v))

        # 4. Handle Numpy Scalars (np.int64, np.float32, etc.)
        elif isinstance(data, np.number):
            self.output.send(np.clip(data, min_v, max_v))

        # 5. Handle Standard Python Scalars (int, float)
        elif isinstance(data, (int, float)):
            # Use standard python math to strictly preserve python types
            # (avoids wrapping them in numpy objects)
            self.output.send(max(min_v, min(data, max_v)))


class RampNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RampNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.base_time = time.time()
        default_units = 'seconds'
        self.units = 1
        self.units_dict = {'seconds': 1, 'milliseconds': 1000, 'minutes': 1.0/60.0, 'hours': 1.0/60.0/60.0}
        self.update_time_base()

        self.start_value = 0
        self.current_value = 0
        self.target = 0
        self.duration = 0.0
        self.elapsed = 0.0
        self.new_target = False

        if len(self.ordered_args) > 0:
            if self.ordered_args[0] in self.units_dict:
                self.units = self.units_dict[self.ordered_args[0]]
                default_units = self.ordered_args[0]

        self.input = self.add_input('in', triggers_execution=True)
        self.units_property = self.add_option('units', widget_type='combo', default_value=default_units, callback=self.set_units)
        self.units_property.widget.combo_items = ['seconds', 'milliseconds', 'minutes', 'hours']
        self.output_always_option = self.add_option('output_always', widget_type='checkbox', default_value=False)
        self.output = self.add_output("out")
        self.ramp_done_out = self.add_output("done")

        self.lock = threading.Lock()
        self.add_frame_task()

    def go_to_value(self, value):
        self.start_value = value
        self.current_value = value
        self.target = value
        self.duration = 0

    def set_units(self):
        units_string = self.units_property()
        if units_string in self.units_dict:
            self.units = self.units_dict[units_string]

    def update_time_base(self):
        self.base_time = time.time()
        self.calc_time()

    def calc_time(self):
        current_time = time.time()
        self.elapsed = (current_time - self.base_time) * self.units

    def frame_task(self):
        if self.lock.acquire(blocking=False):
            if self.current_value != self.target:
                self.calc_time()
                if self.duration != 0.0:
                    fraction = self.elapsed / self.duration
                else:
                    fraction = 1.0
                if fraction > 1.0:
                    fraction = 1.0
                self.current_value = self.target * fraction + self.start_value * (1.0 - fraction)

                if self.current_value == self.target:
                    self.ramp_done_out.send('bang')
                    self.start_value = self.current_value
                self.output.send(self.current_value)
            else:
                if self.new_target:
                    self.ramp_done_out.send('bang')
                    self.output.send(self.current_value)
                elif self.output_always_option():
                    self.output.send(self.current_value)
            self.new_target = False
            self.lock.release()

    def execute(self):
        if not self.input.fresh_input:
            return

        raw_data = self.input.get_received_data()
        print(raw_data)
        # 1. Normalize input to a list of floats
        # We do this outside the lock to minimize blocking time
        try:
            if isinstance(raw_data, str):
                args = [any_to_float(x) for x in any_to_list(raw_data)]
            elif isinstance(raw_data, (int, float)):
                args = [raw_data]
            elif isinstance(raw_data, list):
                args = [any_to_float(x) for x in raw_data]
            else:
                return  # Unknown type, ignore
        except (ValueError, TypeError):
            return  # Handle parsing errors gracefully

        # 2. Update State safely
        with self.lock:
            count = len(args)

            if count == 1:
                self.go_to_value(args[0])

            elif count == 2:
                self.start_value = self.current_value
                self.target, self.duration = args

            elif count == 3:
                self.start_value, self.target, self.duration = args
                self.current_value = self.start_value

            else:
                return  # Handle invalid list lengths (0 or >3)

            # Common actions for all valid updates
            self.new_target = True
            self.update_time_base()


class TimeBetweenNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TimeBetweenNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.start_time = time.time()
        self.end_time = time.time()
        default_units = 'milliseconds'
        self.units = 1000
        self.units_dict = {'seconds': 1, 'milliseconds': 1000, 'minutes': 1.0/60.0, 'hours': 1.0/60.0/60.0}

        if len(self.ordered_args) > 0:
            if self.ordered_args[0] in self.units_dict:
                self.units = self.units_dict[self.ordered_args[0]]
                default_units = self.ordered_args[0]

        self.start_input = self.add_input('start', triggers_execution=True)
        self.end_input = self.add_input('end', triggers_execution=True)

        self.units_property = self.add_property('units', widget_type='combo', default_value=default_units, callback=self.set_units)
        self.units_property.widget.combo_items = ['seconds', 'milliseconds', 'minutes', 'hours']
        self.output = self.add_output("")

    def set_units(self):
        units_string = self.units_property()
        if units_string in self.units_dict:
            self.units = self.units_dict[units_string]

    def execute(self):
        if self.active_input == self.start_input:
            self.start_time = time.time()
        elif self.active_input == self.end_input:
            self.end_time = time.time()
            elapsed = (self.end_time - self.start_time) * self.units
            self.output.send(elapsed)


class DateTimeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DateTimeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('get time', trigger_button=True, triggers_execution=True)
        self.hours_output = self.add_output('hours')
        self.minutes_output = self.add_output('minutes')
        self.seconds_output = self.add_output('seconds')
        self.day_output = self.add_output('day')
        self.month_output = self.add_output('month')
        self.year_output = self.add_output('year')
        self.date_string_output = self.add_output('date string')
        self.time_string_output = self.add_output('time string')

    def execute(self):
        date_time = time.localtime()
        date_string = str(date_time[0]) + '-' + str(date_time[1]) + '-' + str(date_time[2])
        time_string = str(date_time[3]) + ':' + str(date_time[4]) + ':' + str(date_time[5])

        self.time_string_output.send(time_string)
        self.date_string_output.send(date_string)

        self.year_output.send(date_time[0])
        self.month_output.send(date_time[1])
        self.day_output.send(date_time[2])
        self.hours_output.send(date_time[3])
        self.minutes_output.send(date_time[4])
        self.seconds_output.send(date_time[5])


class TimerNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TimerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.mode = 0
        self.base_time = time.time()
        default_units = 'milliseconds'
        self.units = 1000
        self.units_dict = {'seconds': 1, 'milliseconds': 1000, 'minutes': 1.0/60.0, 'hours': 1.0/60.0/60.0}
        self.elapsed = 0
        self.previous_elapsed = 0
        self.output_elapsed = 0
        self.update_time_base()

        if len(self.ordered_args) > 0:
            if self.ordered_args[0] in self.units_dict:
                self.units = self.units_dict[self.ordered_args[0]]
                default_units = self.ordered_args[0]

        if label == 'elapsed':
            self.mode = 1
            self.input = self.add_input('', widget_type='drag_float', triggers_execution=True)
        else:
            self.input = self.add_bool_input('on', widget_type='checkbox', triggers_execution=True, callback=self.start_stop)
        self.units_property = self.add_property('units', widget_type='combo', default_value=default_units, width=100, callback=self.set_units)
        self.units_property.widget.combo_items = ['seconds', 'milliseconds', 'minutes', 'hours']
        self.output = self.add_output("")

        self.output_integers_option = self.add_option('output integers', widget_type='checkbox', default_value=False)

        if self.mode == 0:
            self.add_frame_task()

    def update_time_base(self):
        self.base_time = time.time()

    def start_stop(self, input=None):
        on = self.input()
        if on:
            self.update_time_base()

    def set_units(self):
        units_string = self.units_property()
        if units_string in self.units_dict:
            self.units = self.units_dict[units_string]

    def calc_time(self):
        do_execute = False
        current_time = time.time()
        self.elapsed = (current_time - self.base_time) * self.units
        output_ints = self.output_integers_option()
        if not output_ints or (int(self.elapsed) != int(self.previous_elapsed)):
            if output_ints:
                self.output_elapsed = int(self.elapsed)
            else:
                self.output_elapsed = self.elapsed
            do_execute = True
        self.previous_elapsed = self.elapsed
        return do_execute

    def frame_task(self):
        on = self.input()
        if on:
            if self.calc_time():
                self.execute()

    def execute(self):
        if self.mode == 1:
            self.calc_time()
            self.input.set(self.output_elapsed)
        self.output.send(self.output_elapsed)
        if self.mode == 1:
            self.update_time_base()


'''counter : CounterNode
    description:
        counts input events, wrapping at maximum and signalling end of count

    inputs:
        input: (triggers) <anything> increments count on each input event

        count: <int> sets the count maximum

        step: <int> sets the increment step per input event

    messages:
        count: 'count <int>' sets the count maximum

        step: 'step <int>' sets the increment step per input event
        
        set: 'set <int>' sets the counter value 
        
        reset: 'reset' resets the counter value to 0
        
    outputs:
        count out: <int> the current count is output for every input event

        carry out: <int> outputs 1 when the count maximum - 1 is achieved
            outputs 0 when counter wraps back to 0
'''


class CounterNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CounterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.current_value = 0
        self.carry_state = 0
        self.max_count = self.arg_as_int(default_value=255, index=0)
        self.step = self.arg_as_int(default_value=1, index=1)

        self.input = self.add_input("input", triggers_execution=True, trigger_button=True)
        self.input.bang_repeats_previous = False
        self.max_input = self.add_int_input('count', widget_type='drag_int', default_value=self.max_count, callback=self.update_max_count_from_widget)
        self.step_input = self.add_int_input('step', widget_type='drag_int', default_value=self.step, callback=self.update_step_from_widget)
        self.output = self.add_output("count out")
        self.carry_output = self.add_output("carry out")
        self.carry_output.output_always = False

        self.message_handlers['reset'] = self.reset_message
        self.message_handlers['set'] = self.set_message
        # self.message_handlers['step'] = self.step_message

    # widget callbacks
    def update_max_count_from_widget(self, input=None):
        self.max_count = self.max_input()

    def update_step_from_widget(self, input=None):
        self.step = self.step_input()

    # messages
    def reset_message(self, message='', message_data=[]):
        self.current_value = -1

    def set_message(self, message='', message_data=[]):
        self.current_value = any_to_int(message_data[0])

    def execute(self):
        self.input()  # Clear input

        next_val = self.current_value + self.step
        max_val = self.max_count

        # 1. Determine Carry State (-1, 0, or 1)
        if next_val < 0:
            carry = -1
        elif next_val >= max_val:
            carry = 1
        else:
            carry = 0

        # 2. Handle Carry Output
        # Send if we are currently carrying/wrapping OR if we need to reset previous state
        if carry != 0 or self.carry_state != 0:
            self.carry_state = carry
            self.carry_output.send(carry)

        # 3. Wrap Value and Send
        # Python's % operator handles both overflow and negative underflow correctly
        self.current_value = next_val % max_val
        self.output.send(self.current_value)


class RangeCounterNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RangeCounterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.current_value = 0
        self.carry_state = 0

        self.start_count = self.arg_as_int(default_value=0, index=0)

        self.end_count = self.arg_as_int(default_value=255, index=1)
        self.step = self.arg_as_int(default_value=1, index=2)

        self.input = self.add_input("input", triggers_execution=True, trigger_button=True)
        self.input.bang_repeats_previous = False
        self.start_count_input = self.add_int_input('start', widget_type='drag_int', default_value=self.start_count,
                                            callback=self.update_start_count_from_widget)

        self.end_count_input = self.add_int_input('end', widget_type='drag_int', default_value=self.end_count, callback=self.update_end_count_from_widget)
        self.step_input = self.add_int_input('step', widget_type='drag_int', default_value=self.step, callback=self.update_step_from_widget)
        self.output = self.add_output("count out")
        self.carry_output = self.add_output("carry out")
        self.carry_output.output_always = False

        self.message_handlers['reset'] = self.reset_message
        self.message_handlers['set'] = self.set_message
        # self.message_handlers['step'] = self.step_message

    # widget callbacks
    def update_start_count_from_widget(self, input=None):
        self.start_count = self.start_count_input()

    def update_end_count_from_widget(self, input=None):
        self.end_count = self.end_count_input()

    def update_step_from_widget(self, input=None):
        self.step = self.step_input()

    # messages
    def reset_message(self, message='', message_data=[]):
        self.current_value = 0

    def set_message(self, message='', message_data=[]):
        self.current_value = any_to_int(message_data[0])

    # def step_message(self, message='', message_data=[]):
    #     self.step = any_to_int(message_data[0])

    def execute(self):
        self.input()  # Clear input

        start = self.start_count
        end = self.end_count
        gap = end - start + 1

        # Calculate potential next value
        next_val = self.current_value + self.step

        # 1. Determine Carry (-1, 0, or 1)
        new_carry = 0
        if next_val < start:
            new_carry = -1
        elif next_val > end:
            new_carry = 1

        # 2. Calculate Wrapped Value
        # Using modulo (%) handles both underflow and overflow in one line.
        # We normalize to zero (next_val - start), wrap (%), then add start back.
        self.current_value = start + ((next_val - start) % gap)

        # 3. Handle Carry Output
        # Update if we are currently carrying, or if we need to reset the state to 0
        if new_carry != 0 or self.carry_state != 0:
            self.carry_state = new_carry
            self.carry_output.send(new_carry)

        self.output.send(self.current_value)


class GateNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = GateNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.num_gates = self.arg_as_int(default_value=1)
        self.state = 0
        self.bool_state = False

        if self.num_gates > 1:
            self.choice_input = self.add_int_input('', widget_type='drag_int', triggers_execution=True, default_value=self.state, callback=self.change_state, max=self.num_gates, min=0)
        else:
            self.choice_input = self.add_bool_input('', widget_type='checkbox', triggers_execution=True, default_value=self.bool_state, widget_width=40, callback=self.change_state)
        self.gated_input = self.add_input("input", triggers_execution=True)

        for i in range(self.num_gates):
            self.add_output("out " + str(i + 1))

    def change_state(self, input=None):
        if self.num_gates == 1:
            self.bool_state = any_to_bool(self.choice_input())
        else:
            self.state = any_to_int(self.choice_input())

    def execute(self):
        if self.num_gates == 1:
            if self.bool_state:
                if self.gated_input.fresh_input:
                    self.outputs[0].send(self.gated_input())
        else:
            if self.num_gates >= self.state > 0:
                if self.gated_input.fresh_input:
                    value = self.gated_input()
                    self.outputs[self.state - 1].send(self.gated_input())


class SwitchNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SwitchNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.num_switches = self.arg_as_int(default_value=1)
        self.state = 0
        self.bool_state = False

        self.choice_input = self.add_int_input('which input', widget_type='input_int', callback=self.change_state)
        self.switch_inputs = []
        for i in range(self.num_switches):
            self.switch_inputs.append(self.add_input('in ' + str(i + 1)))

        self.out = self.add_output('out')

    def change_state(self, input=None):
        self.state = self.choice_input()
        if self.state < 0:
            self.state = 0
            self.choice_input.set(self.state)
        elif self.state > self.num_switches:
            self.state = self.num_switches
            self.choice_input.set(self.state)
        if self.state != 0:
            self.switch_inputs[self.state - 1].triggers_execution = True
        for i in range(self.num_switches):
            if i + 1 != self.state:
                self.switch_inputs[i].triggers_execution = False

    def execute(self):
        received = self.switch_inputs[self.state - 1]()
        self.out.send(received)


class UnpackNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = UnpackNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.num_outs = 2
        self.out_types = []
        # self.out_functions = []
        out_names = []
        self.types = {'s': str, 'i': int, 'f': float, 'l': list, 'b': bool, 'a': np.ndarray}
        self.kinds = [str, int, float, list, bool, np.ndarray]

        if torch_available:
            self.types['t'] = torch.Tensor
            self.kinds.append(torch.Tensor)

        if len(args) > 0:
            if is_number(args[0]):
                self.out_types = []
                self.out_functions = []
                self.num_outs = self.arg_as_int(default_value=1)

                if len(args) > 1:
                    if args[1] in self.types:
                        for i in range(self.num_outs):
                            self.out_types.append(self.types[args[1]])
                            out_names.append(self.types[args[1]].__name__ + ' ' + str(i + 1))
                else:
                    for i in range(self.num_outs):
                        self.out_types.append(None)
                        out_names.append('out ' + str(i + 1))

            else:
                self.num_outs = len(args)
                for arg in args:
                    if arg in self.types:
                        self.out_types.append(self.types[arg])
                        out_names.append(self.types[arg].__name__)
                    else:
                        if is_number(arg):
                            self.out_types.append(any_to_numerical(arg))
                            out_names.append(any_to_numerical(arg))
                        else:
                            self.out_types.append(any_to_string(arg))
                            out_names.append(any_to_string(arg))
        else:
            self.out_types = [None, None]

        self.input = self.add_input("", triggers_execution=True)

        for i in range(self.num_outs):
            if self.out_types[i] in self.kinds:
                if self.out_types[i] == str:
                    self.add_string_output(out_names[i])
                elif self.out_types[i] == int:
                    self.add_int_output(out_names[i])
                elif self.out_types[i] == float:
                    self.add_float_output(out_names[i])
                elif self.out_types[i] == list:
                    self.add_list_output(out_names[i])
                elif self.out_types[i] == bool:
                    self.add_bool_output(out_names[i])
                elif self.out_types[i] == np.ndarray:
                    self.add_array_output(out_names[i])
                elif self.out_types[i] == torch.Tensor and torch_available:
                    self.add_tensor_output(out_names[i])
            else:
                self.add_output(out_names[i])

    def execute(self):
        if self.input.fresh_input:
            value = self.input()
            t = type(value)
            if t in [float, int, bool, np.float32, np.int64]:
                self.outputs[0].send(value)
            elif t == 'str':
                listing, _, _ = string_to_hybrid_list(value)
                out_count = len(listing)
                if out_count > self.num_outs:
                    out_count = self.num_outs
                for j in reversed(range(out_count)):
                    self.outputs[j].send(listing[j])
            elif t == list:
                listing, _, _ = list_to_hybrid_list(value)
                out_count = len(listing)
                if out_count > self.num_outs:
                    out_count = self.num_outs
                for j in reversed(range(out_count)):
                    self.outputs[j].send(listing[j])
            elif t == np.ndarray:
                out_count = value.shape[0]
                if out_count > self.num_outs:
                    out_count = self.num_outs
                for j in reversed(range(out_count)):
                    self.outputs[j].send(value[j])
            elif torch_available:
                if t == torch.Tensor:
                    out_count = value.shape[0]
                    if out_count > self.num_outs:
                        out_count = self.num_outs
                    for j in reversed(range(out_count)):
                        self.outputs[j].send(value[j])


class PackNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = PackNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.num_ins = 2
        self.in_types = []
        in_names = []
        self.types = {'s': str, 'i': int, 'f': float, 'l': list, 'b': bool, 'a': np.ndarray}
        self.kinds = [str, int, float, list, bool, np.ndarray]

        if torch_available:
            self.types['t'] = torch.Tensor
            self.kinds.append(torch.Tensor)

        if len(args) > 0:
            if is_number(args[0]) and len(args) == 1:
                self.num_ins = self.arg_as_int(default_value=2)
                for i in range(self.num_ins):
                    self.in_types.append(None)
                    in_names.append('in ' + str(i + 1))
            else:
                self.num_ins = len(args)
                for arg in args:
                    if arg in self.types:
                        self.in_types.append(self.types[arg])
                        in_names.append(self.types[arg].__name__)
                    else:
                        if is_number(arg):
                            self.in_types.append(any_to_numerical(arg))
                            in_names.append(any_to_numerical(arg))
                        else:
                            self.in_types.append(any_to_string(arg))
                            in_names.append(any_to_string(arg))
        else:
            for i in range(self.num_ins):
                self.in_types.append(None)
                in_names.append('in ' + str(i + 1))

        for i in range(self.num_ins):
            triggers = False
            if label == 'pak' or i == 0:
                triggers = True
            if self.in_types[i] in self.kinds:
                if self.in_types[i] == str:
                    self.add_string_input(in_names[i], triggers_execution=triggers, default_value='')
                elif self.in_types[i] == int:
                    self.add_int_input(in_names[i], triggers_execution=triggers, default_value=0)
                elif self.in_types[i] == float:
                    self.add_float_input(in_names[i], triggers_execution=triggers, default_value=0.0)
                elif self.in_types[i] == list:
                    self.add_list_input(in_names[i], triggers_execution=triggers, default_value=[])
                elif self.in_types[i] == bool:
                    self.add_bool_input(in_names[i], triggers_execution=triggers, default_value=False)
                elif self.in_types[i] == np.ndarray:
                    self.add_array_input(in_names[i], triggers_execution=triggers)
                elif torch_available and self.in_types[i] == torch.Tensor:
                    self.add_tensor_input(in_names[i], triggers_execution=triggers)
            else:
                if type(self.in_types[i]) is str:
                    inp = self.add_string_input(in_names[i], triggers_execution=triggers, default_value=self.in_types[i])
                elif type(self.in_types[i]) in [int, np.int64]:
                    inp = self.add_int_input(in_names[i], triggers_execution=triggers,
                                                default_value=self.in_types[i])
                elif type(self.in_types[i]) in [float, np.float32, np.float64]:
                    inp = self.add_float_input(in_names[i], triggers_execution=triggers,
                                                default_value=self.in_types[i])
                else:
                    if self.in_types[i] == None:
                        inp = self.add_input(in_names[i], triggers_execution=triggers,
                                             default_value=0)
                    else:
                        inp = self.add_input(in_names[i], triggers_execution=triggers,
                                               default_value=self.in_types[i])


        self.output = self.add_output("out")
        self.output_preference_option = self.add_option('output pref', widget_type='combo', default_value='list')
        self.output_preference_option.widget.combo_items = ['list', 'array', 'tensor']

    def execute(self):
        trigger = False
        if self.label == 'pak':
            trigger = True
        elif self.inputs[0].fresh_input:
            trigger = True
        if trigger:
            output_option = self.output_preference_option()
            if output_option == 'list':
                out_list = []
                for i in range(self.num_ins):
                    value = self.inputs[i]()
                    t = type(value)
                    if t in [list, tuple]:
                        out_list += [value]
                    elif t == np.ndarray:
                        array_list = any_to_list(value)
                        out_list += [array_list]
                    else:
                        out_list.append(value)
                self.output.send(out_list)
            elif output_option == 'array':
                out_list = []
                all_array = False
                first_data = self.inputs[0]()
                if type(first_data) is np.ndarray:
                    all_array = True
                    for i in range(self.num_ins):
                        if self.in_types[i] != np.ndarray:
                            all_array = False
                            break

                if all_array:
                    for i in range(self.num_ins):
                        out_list.append(self.inputs[i]())
                    try:
                        out_array = np.stack(out_list)
                        self.output.send(out_array)
                    except:
                        self.output.send(out_list)
                else:
                    for i in range(self.num_ins):
                        value = self.inputs[i]()
                        t = type(value)
                        if t in [list, tuple]:
                            out_list += [value]
                        elif t == np.ndarray:
                            array_list = any_to_list(value)
                            out_list += [array_list]
                        else:
                            out_list.append(value)
                    out_list, _ = list_to_array_or_list_if_hetero(out_list)
                    self.output.send(out_list)
            elif output_option == 'tensor':
                out_list = []
                all_tensors = True
                for i in range(self.num_ins):
                    if self.in_types[i] != torch.Tensor:
                        all_tensors = False
                        break
                for i in range(self.num_ins):
                    out_list.append(self.inputs[i]())

                if all_tensors:
                    try:
                        out_tensor = torch.stack(out_list)
                        self.output.send(out_tensor)
                    except:
                        self.output.send(out_list)
                else:
                    for i in range(self.num_ins):
                        value = self.inputs[i]()
                        t = type(value)
                        if t in [list, tuple]:
                            out_list += [value]
                        elif t == np.ndarray:
                            array_list = any_to_list(value)
                            out_list += [array_list]
                        else:
                            out_list.append(value)
                    out_list, _ = list_to_array_or_list_if_hetero(out_list)
                    if type(out_list) is np.ndarray:
                        out_list = torch.from_numpy(out_list)
                    self.output.send(out_list)


class BucketBrigadeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = BucketBrigadeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.bucket_count = self.arg_as_int(default_value=8)
        self.buckets = [0] * self.bucket_count
        self.head = 0

        self.input = self.add_input("in", triggers_execution=True)
        self.outs = []
        for i in range(self.bucket_count):
            self.outs.append(self.add_output("out " + str(i)))

    def execute(self):
        if self.input.fresh_input:
            self.head = (self.head + 1) % self.bucket_count
            data = self.input()
            self.buckets[self.head] = data
            for rev_i in reversed(range(self.bucket_count)):
                source = (self.head - rev_i) % self.bucket_count
                self.outs[rev_i].send(self.buckets[source])


class DelayNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DelayNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.delay = self.arg_as_int(default_value=8)
        self.buffer = [None] * self.delay
        self.buffer_position = 0

        self.input = self.add_input("in")
        self.input.bang_repeats_previous = False
        self.delay_input = self.add_int_input('delay', widget_type='drag_int', default_value=self.delay, min=0, max=4000, callback=self.delay_changed)
        self.cancel_input = self.add_input('cancel', callback=self.delay_cancelled)
        self.output = self.add_output("out")

        self.add_frame_task()
        self.new_delay = self.delay

    def delay_cancelled(self):
        for i in range(self.delay):
            self.buffer[i] = None

    def delay_changed(self):
        self.new_delay = self.delay_input()

    def frame_task(self):
        self.execute()

    def execute(self):
        out_data = self.buffer[self.buffer_position]
        if out_data is not None:
            self.output.send(out_data)

        if self.input.fresh_input:
            self.buffer[self.buffer_position] = self.input()
        else:
            self.buffer[self.buffer_position] = None
        self.buffer_position += 1
        if self.buffer_position >= self.delay:
            self.buffer_position = 0

        if self.new_delay != self.delay:
            if self.new_delay > self.delay:
                self.increase_delay()
            else:
                self.decrease_delay()

    def increase_delay(self):
        hold_buffer = self.buffer[self.buffer_position:]
        additional = self.new_delay - self.delay
        if self.buffer_position > 0:
            self.buffer = self.buffer[:self.buffer_position]
        self.buffer += ([None] * additional + hold_buffer)
        self.buffer_position += additional
        self.delay = self.new_delay

    def decrease_delay(self):
        hold_buffer = self.buffer[self.buffer_position:]
        remaining = self.delay - self.buffer_position
        if remaining > self.new_delay:
            self.buffer = hold_buffer[:self.new_delay]
        else:
            left_to_copy = self.new_delay - remaining
            additional_buffer = self.buffer[:left_to_copy]
            self.buffer = hold_buffer + additional_buffer
        self.buffer_position = 0
        self.delay = self.new_delay


class SelectNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SelectNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        if label == 'decode':
            self.mode = 1
        else:
            self.mode = 0
        self.selector_count = 0

        if len(args) > 0:
            if self.mode == 0:
                self.selector_count = len(args)
            else:
                self.selector_count = any_to_int(args[0])

        self.out_mode = 0
        self.selectors = []
        self.selector_options = []
        self.last_states = []
        self.current_states = []

        self.input = self.add_input("in", triggers_execution=True)
        self.input.bang_repeats_previous = False

        if self.mode == 0:
            for i in range(self.selector_count):
                self.add_output(any_to_string(args[i]))
            for i in range(self.selector_count):
                val, t = decode_arg(args, i)
                self.selectors.append(val)
                self.last_states.append(0)
                self.current_states.append(0)
            for i in range(self.selector_count):
                an_option = self.add_option('selector ' + str(i), widget_type='text_input', default_value=args[i], callback=self.selectors_changed)
                self.selector_options.append(an_option)
        else:
            for i in range(self.selector_count):
                self.add_output(str(i))
                self.selectors.append(i)
                self.last_states.append(0)
                self.current_states.append(0)

        self.output_mode_option = self.add_option('output_mode', widget_type='combo', default_value='bang',  callback=self.output_mode_changed)
        self.output_mode_option.widget.combo_items = ['bang', 'flag']

        self.new_selectors = False

    def output_mode_changed(self):
        if self.output_mode_option() == 'bang':
            self.out_mode = 0
        else:
            self.out_mode = 1

    def selectors_changed(self):
        self.new_selectors = True

    def update_selectors(self):
        new_selectors = []
        for i in range(self.selector_count):
            new_selectors.append(self.selector_options[i]())
        for i in range(self.selector_count):
            # this does not update the label
            dpg.set_item_label(self.outputs[i].uuid, label=new_selectors[i])
            sel, t = decode_arg(new_selectors, i)
            self.selectors[i] = sel

    def execute(self):
        if self.new_selectors:
            self.update_selectors()
            self.new_selectors = False
        value = self.input()

        if self.out_mode == 0:
            if type(value) == list:
                for item in value:
                    for i in range(self.selector_count):
                        if item == self.selectors[i]:
                            self.outputs[i].send('bang')
            else:
                for i in range(self.selector_count):
                    if value == self.selectors[i]:
                        self.outputs[i].send('bang')
                        return
        else:
            self.current_states = [0] * self.selector_count

            if type(value) == list:
                for item in value:
                    for i in range(self.selector_count):
                        if item == self.selectors[i]:
                            self.current_states[i] = 1
            else:
                for i in range(self.selector_count):
                    if value == self.selectors[i]:
                        self.current_states[i] = 1
                        break

            if self.current_states != self.last_states:
                for i in range(self.selector_count):
                    if self.current_states[i] != self.last_states[i]:
                        self.outputs[i].send(self.current_states[i])

            self.last_states = self.current_states


class RouteNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RouteNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.router_count = 0

        if len(args) > 0:
            self.router_count = len(args)

        self.out_mode = 0
        self.routers = []
        self.router_options = []
        self.last_states = []
        self.current_states = []

        self.input = self.add_input("in", triggers_execution=True)

        for i in range(self.router_count):
            self.add_output(any_to_string(args[i]))
        self.miss_out = self.add_output('unmatched')
        for i in range(self.router_count):
            val, t = decode_arg(args, i)
            self.routers.append(val)
        for i in range(self.router_count):
            an_option = self.add_option('route address ' + str(i), widget_type='text_input', default_value=args[i], callback=self.routers_changed)
            self.router_options.append(an_option)

        self.new_routers = False

    def routers_changed(self):
        self.new_routers = True

    def update_routers(self):
        new_routers = []
        for i in range(self.router_count):
            new_routers.append(self.router_options[i]())
        for i in range(self.router_count):
            # this does not update the label
            dpg.set_item_label(self.outputs[i].uuid, label=new_routers[i])
            sel, t = decode_arg(new_routers, i)
            self.routers[i] = sel

    def execute(self):
        if self.new_routers:
            self.update_routers()
            self.new_routers = False
        data = self.input()
        t = type(data)
        if t == str:
            data = data.split(' ')
            t = list
        if t == list:
            if len(data) > 1:
                router = any_to_string(data[0])
                if router in self.routers:
                    index = self.routers.index(router)
                    message = data[1:]
                    if index < len(self.outputs):
                        self.outputs[index].send(message)
                elif is_number(data[0]):
                    num_router = any_to_int(data[0])
                    if num_router in self.routers:
                        index = self.routers.index(num_router)
                        message = data[1:]
                        if index < len(self.outputs):
                            self.outputs[index].send(message)
                else:
                    self.miss_out.send(self.input())
            else:
                self.miss_out.send(self.input())
        else:
            self.miss_out.send(self.input())


class TriggerNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TriggerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.trigger_count = 0
        if args is not None:
            self.trigger_count = len(args)
        self.triggers = []
        self.trigger_options = []
        self.trigger_pass = []
        self.force_trigger = False
        self.target_time = 0
        self.flash_duration = .100

        self.input = self.add_input("", widget_type='button', widget_width=14, triggers_execution=True, callback=self.call_execution)

        for i in range(self.trigger_count):
            self.add_output(any_to_string(args[i]))
        for i in range(self.trigger_count):
            val, t = decode_arg(args, i)
            self.triggers.append(val)
        for i in range(self.trigger_count):
            an_option = self.add_option('trigger ' + str(i), widget_type='text_input', default_value=args[i], callback=self.triggers_changed)
            self.trigger_options.append(an_option)
            self.trigger_pass.append(0)

        self.new_triggers = True

    def triggers_changed(self):
        self.new_triggers = True
        self.update_triggers()

    def update_triggers(self):
        new_triggers = []
        for i in range(self.trigger_count):
            new_triggers.append(self.trigger_options[i]())
        for i in range(self.trigger_count):
            # this does not update the label
            if new_triggers[i] == 'int':
                self.trigger_pass[i] = 1
            elif new_triggers[i] == 'float':
                self.trigger_pass[i] = 2
            elif new_triggers[i] == 'string':
                self.trigger_pass[i] = 3
            elif new_triggers[i] == 'list':
                self.trigger_pass[i] = 4
            elif new_triggers[i] == 'array':
                self.trigger_pass[i] = 5
            elif new_triggers[i] == 'bang':
                self.trigger_pass[i] = 6
            else:
                self.trigger_pass[i] = 0

            self.outputs[i].label = new_triggers[i]
            dpg.set_value(self.outputs[i].label_uuid, new_triggers[i])
            # dpg.set_item_label(self.outputs[i].uuid, label=new_triggers[i])
            sel, t = decode_arg(new_triggers, i)
            self.triggers[i] = sel

    def call_execution(self, value=0):
        self.force_trigger = True
        self.target_time = time.time() + self.flash_duration
        dpg.bind_item_theme(self.input.widget.uuid, Node.active_theme)
        self.add_frame_task()
        self.execute()

    def frame_task(self):
        now = time.time()
        if now >= self.target_time:
            dpg.bind_item_theme(self.input.widget.uuid, Node.inactive_theme)
            self.remove_frame_tasks()

    def execute(self):
        if self.new_triggers:
            self.update_triggers()
            self.new_triggers = False

        if self.input.fresh_input or self.force_trigger:
            self.force_trigger = False
            in_data = self.input()
            for j in reversed(range(self.trigger_count)):
                trig_mode = self.trigger_pass[j]
                if trig_mode == 0:
                    self.outputs[j].send(self.triggers[j])
                elif trig_mode == 1:
                    self.outputs[j].send(any_to_int(in_data))
                elif trig_mode == 2:
                    self.outputs[j].send(any_to_float(in_data))
                elif trig_mode == 3:
                    self.outputs[j].send(any_to_string(in_data))
                elif trig_mode == 4:
                    self.outputs[j].send(any_to_list(in_data))
                elif trig_mode == 5:
                    self.outputs[j].send(any_to_array(in_data))
                elif trig_mode == 6:
                    self.outputs[j].send('bang')


class DecodeToNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DecodeToNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.trigger_count = 0
        if args is not None:
            self.trigger_count = len(args)
        self.triggers = []
        self.trigger_options = []
        self.trigger_pass = []
        self.force_trigger = False
        self.target_time = 0
        self.flash_duration = .100

        self.input = self.add_input("decode", widget_type='drag_int', triggers_execution=True, callback=self.call_execution)

        for i in range(self.trigger_count):
            self.add_output(any_to_string(args[i]))
        self.combined_output = self.add_output('combined')
        for i in range(self.trigger_count):
            val, t = decode_arg(args, i)
            self.triggers.append(val)
        for i in range(self.trigger_count):
            an_option = self.add_option('trigger ' + str(i), widget_type='text_input', default_value=args[i], callback=self.triggers_changed)
            self.trigger_options.append(an_option)
            self.trigger_pass.append(0)

        self.new_triggers = True

    def triggers_changed(self):
        self.new_triggers = True
        self.update_triggers()

    def update_triggers(self):
        new_triggers = []
        for i in range(self.trigger_count):
            new_triggers.append(self.trigger_options[i]())
        for i in range(self.trigger_count):
            self.outputs[i].label = new_triggers[i]
            dpg.set_value(self.outputs[i].label_uuid, new_triggers[i])
            # dpg.set_item_label(self.outputs[i].uuid, label=new_triggers[i])
            sel, t = decode_arg(new_triggers, i)
            self.triggers[i] = sel

    def call_execution(self, value=0):
        self.force_trigger = True
        self.target_time = time.time() + self.flash_duration
        dpg.bind_item_theme(self.input.widget.uuid, Node.active_theme)
        self.add_frame_task()
        self.execute()

    def frame_task(self):
        now = time.time()
        if now >= self.target_time:
            dpg.bind_item_theme(self.input.widget.uuid, Node.inactive_theme)
            self.remove_frame_tasks()

    def execute(self):
        if self.new_triggers:
            self.update_triggers()
            self.new_triggers = False

        if self.input.fresh_input or self.force_trigger:
            self.force_trigger = False
            in_data = any_to_int(self.input())
            if 0 <= in_data < self.trigger_count:
                self.outputs[in_data].send(self.triggers[in_data])
                self.combined_output.send(self.triggers[in_data])



class ConcatenateNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ConcatenateNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.count = 2
        if len(args) > 0:
            val, t = decode_arg(args, 0)
            if t == int:
                self.count = val
        self.input_list = []
        for i in range(self.count):
            in_ = self.add_input('list in ' + str(i + 1))
            self.input_list.append(in_)
        self.input_list[0].triggers_execution = True
        self.output = self.add_output("concatenated list out")
        self.all_inputs_trigger = self.add_option('all inputs trigger', widget_type='checkbox', callback=self.all_trigger)

    def all_trigger(self):
        if self.all_inputs_trigger():
            for i in range(self.count - 1):
                self.input_list[i + 1].triggers_execution = True
        else:
            for i in range(self.count - 1):
                self.input_list[i + 1].triggers_execution = False

    def execute(self):
        out_list = []
        out_value = any_to_list(self.input_list[0]())
        if out_value:
            out_list.extend(out_value)

            for i in range(self.count - 1):
                l = any_to_list(self.input_list[i + 1]())
                if l:
                    out_list.extend(l)
            if len(out_list) > 0:
                self.output.send(out_list)


'''info : TypeNode
    description:
        reports type and additional info of received input

    inputs:
        in: <anything>

    properties:
        info : <str> : shows type of the input
            float, int, bang, numpy.double, numpy.float32, numpy.int64, numpy.bool_: type name
            list input: list[length]
            string: str
            array: array[shape] dtype
            tensor: tensor[shape] dtype device requires_grad
'''


def print_info(input_):
    # Default values to ensure variables are always defined
    type_string = 'unknown'
    value_string = ''

    # 1. Handle Basic Primitives (int, float, bool)
    if isinstance(input_, (int, bool)):
        type_string = type(input_).__name__
        value_string = str(input_)

    elif isinstance(input_, float):
        type_string = type(input_).__name__
        value_string = f'{input_:.3f}'

    # 2. Handle Strings
    elif isinstance(input_, str):
        if input_ == 'bang':
            type_string = 'bang'
        else:
            type_string = 'string'
            value_string = input_

    # 3. Handle Lists
    elif isinstance(input_, list):
        type_string = f"list[{len(input_)}]"

    # 4. Handle Dicts
    elif isinstance(input_, dict):
        type_string = f"dict{list(input_.keys())}"

    # 5. Handle Numpy Arrays
    elif isinstance(input_, np.ndarray):
        # Join shape dimensions dynamically: "2, 3, 4"
        shape_str = ", ".join(map(str, input_.shape))
        # Get dtype name automatically (e.g., 'float32', 'int64')
        dtype_str = input_.dtype.name
        type_string = f"array[{shape_str}] {dtype_str}"

    # 6. Handle Numpy Scalars (e.g. np.float32(1.0))
    elif isinstance(input_, (np.number, np.bool_)):
        type_string = f"numpy.{input_.dtype.name}"
        value_string = str(input_)

    # 7. Handle Torch Tensors
    # We check if torch is defined and matches type
    elif 'torch' in globals() and isinstance(input_, torch.Tensor):
        shape_str = ", ".join(map(str, input_.shape))

        # Clean up dtype string (e.g., 'torch.float32' -> 'float32')
        dtype_str = str(input_.dtype).replace('torch.', '')

        # Device handling
        device_str = input_.device.type  # 'cpu', 'cuda', 'mps'

        # Grad handling
        grad_str = ' requires_grad' if input_.requires_grad else ''

        type_string = f"tensor[{shape_str}] {dtype_str} {device_str}{grad_str}"

    return type_string, value_string


class TypeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TypeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input("in", triggers_execution=True)
        self.input.bang_repeats_previous = False

        # 'info' usually requires more space for the detailed string
        width = 192 if label == 'info' else 128
        self.type_property = self.add_property(self.label, widget_type='text_input', width=width)

    def execute(self):
        input_ = self.input()

        # If label is 'type', we want brief info. If 'info', we want detailed info.
        is_detailed = (self.label != 'type')

        type_string = print_info(input_)
        self.type_property.set(type_string)


class LengthNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = LengthNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('input', triggers_execution=True)
        self.output = self.add_output('length')

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            t = type(data)
            if t is str:
                self.output.send(len(data))
            elif t in [list, tuple]:
                self.output.send(len(data))
            elif t == np.ndarray:
                self.output.send(data.size)
            elif self.app.torch_available and t == torch.Tensor:
                self.output.send(data.numel())
            else:
                self.output.send(1)


'''array : ArrayNode
    description:
        convert input into an array

    inputs:
        in: anything (triggers)

    properties: (optional)
        shape: a list of numbers separated by spaces or commas
            if empty, then the input is not reshaped

    outputs:
        array out:
            any input is converted into a numpy array. If a shape is supplied the array is reshaped before output
'''


class ArrayNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ArrayNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        shape_text = ''
        self.shape = None
        if args is not None and len(args) > 0:
            shape_text = ' '.join(args)
            shape_split = re.findall(r'\d+', shape_text)
            shape_list, _, _ = list_to_hybrid_list(shape_split)
            self.shape = shape_list

        self.input = self.add_array_input("in", triggers_execution=True)
        self.output = self.add_output('array out')

        self.shape_property = self.add_option('shape', widget_type='text_input', default_value=shape_text, callback=self.shape_changed)

    def shape_changed(self):
        shape_text = self.shape_property()
        if shape_text != '':
            shape_split = re.findall(r'\d+', shape_text)
            shape_list, _, _ = list_to_hybrid_list(shape_split)
            self.shape = shape_list
        else:
            self.shape = None

    def execute(self):
        out_array = self.input()
        # out_array = any_to_array(in_data)
        if type(out_array) is np.ndarray:
            if self.shape is not None:
                out_array = np.reshape(out_array, tuple(self.shape))
            self.output.send(out_array)

'''string : StringNode
    description:
        convert input into a string

    inputs:
        in: anything (triggers)

    outputs:
        string out:
            any input is converted into a string and output
'''


class StringNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = StringNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_string_input("in", triggers_execution=True)
        self.output = self.add_output('string out')

    def execute(self):
        self.output.send(self.input())


'''prepend : PrependNode
    description:
        prepend a prefix element to this list or string

    inputs:
        in: list, str, scalar, array

    arguments:
        <list, str, bool, number> : the value to prepend to the input

    properties:
        prefix : str : str to prepend to the input

    options:
        always output list <bool> : if True, output list with prefix as first element

    output:
        list input: output a list [prefix input_list]
        str input:
            'always output list' is False : output a str of 'prefix input'
            'always output list' is True : output a list [prefix input_str]
        scalar input: output a list [prefix input_scalar]
        array input: convert the array to a list and output a list [prefix input_array_values]
'''


class PrependNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = PrependNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.num_ins = 2

        self.as_list = False

        self.prepender = ''

        if len(args) > 0:
            in_value, t = decode_arg(args, 0)
            self.prepender = in_value

        self.input = self.add_input("in", triggers_execution=True)
        self.prepender_property = self.add_input("prefix", widget_type='text_input', default_value=self.prepender)
        self.output = self.add_output("out")
        self.always_as_list_option = self.add_option('always output list', widget_type='checkbox', default_value=False, callback=self.option_changed)

    def option_changed(self):
        self.as_list = self.always_as_list_option()

    def execute(self):
        prepender = self.prepender_property()
        out_list = [prepender]
        data = self.input()
        t = type(data)

        if t == str:
            if self.as_list:
                out_list.append(data)
            else:
                out_list = prepender + ' ' + data
        elif t in [int, float, bool, np.int64, np.double, np.bool_]:
            out_list.append(data)
        elif t == list:
            out_list += data
        elif t == np.ndarray:
            if type(prepender) is str:
                out_list.append(data)
            else:
                out_list += any_to_list(data)
        elif self.app.torch_available and t == torch.Tensor:
            if type(prepender) is str:
                out_list.append(data)
            else:
                out_list += any_to_list(data)
        self.output.send(out_list)


'''append : AppendNode
    description:
        append a suffix element to this list or string

    inputs:
        in: <list, str, scalar, array>

    arguments:
        <list, str, bool, number> : the value to append to the input

    properties:
        suffix : <str> : string to append to the input

    options:
        always output list <bool> : if True, output list with suffix as last element

    output:
        list input: output a list [input_list suffix]
        str input:
            'always output list' is False : output a str of 'input suffix'
            'always output list' is True : output a list [input_str suffix]
        scalar input: output a list [input_scalar suffix]
        array input: convert the array to a list and output a list [input_array_values suffix]
'''


class AppendNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = AppendNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.as_list = False

        self.appender = ''

        if len(args) > 0:
            in_value, t = decode_arg(args, 0)
            self.appender = in_value

        self.input = self.add_input("in", triggers_execution=True)
        self.appender = self.add_input("suffix", widget_type='text_input', default_value=self.appender)
        self.output = self.add_output("out")
        self.always_as_list = self.add_option('always output list', widget_type='checkbox', default_value=False)

    def execute(self):
        out_list = []
        data = self.input()
        t = type(data)

        if t == str:
            if self.always_as_list():
                out_list = [data]
                out_list.append(self.appender())
            else:
                out_list = data + ' ' + self.appender()
        elif t in [int, float, bool, np.int64, np.double, np.bool_]:
            out_list = [data]
            out_list.append(self.appender())
        elif t == list:
            out_list = data
            out_list.append(self.appender())
        elif t == np.ndarray:
            appender = self.appender()
            if type(appender) is str:
                out_list = [data]
                out_list.append(appender)
            else:
                out_list = any_to_list(data)
                out_list.append(self.appender())
        elif self.app.torch_available and t == torch.Tensor:
            appender = self.appender()
            if type(appender) is str:
                out_list = [data]
                out_list.append(appender)
            else:
                out_list = any_to_list(data)
                out_list.append(self.appender())
        self.output.send(out_list)


class DictExtractNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DictExtractNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('dict in', triggers_execution=True)
        for i in range(32):
            self.add_output('')
        self.extract_keys = []
        for arg in args:
            if type(arg) is str:
                self.extract_keys.append(arg)
        self.output_count = len(self.extract_keys)
        self.output_count_option = self.add_option('key count', widget_type='input_int', default_value=self.output_count, callback=self.output_count_changed)
        self.include_key_in_output_option = self.add_option('include key in output', widget_type='checkbox', default_value=False)
        self.extract_opt = []
        for i in range(32):
            if i < self.output_count:
                self.extract_opt.append(self.add_option('key ' + str(i), widget_type='text_input', default_value=self.extract_keys[i], callback=self.key_changed))
            else:
                self.extract_opt.append(self.add_option('key ' + str(i), widget_type='text_input', default_value='', callback=self.key_changed))

    def output_count_changed(self):
        output_count = self.output_count_option()
        self.output_count = output_count
        self.key_changed()

    def custom_create(self, from_file):
        for i in range(self.output_count):
            if i < len(self.extract_keys):
                key = self.extract_keys[i]
            else:
                key = ''
            self.outputs[i].set_label(key)
        for i in range(self.output_count, 32):
            dpg.hide_item(self.outputs[i].uuid)
            dpg.hide_item(self.extract_opt[i].uuid)

    def toggle_show_hide_options(self) -> None:
        if len(self.options) > 0:
            self.options_visible = not self.options_visible
            if self.options_visible:
                dpg.show_item(self.output_count_option.uuid)
                dpg.show_item(self.output_count_option.widget.uuid)
                dpg.show_item(self.include_key_in_output_option.uuid)
                dpg.show_item(self.include_key_in_output_option.widget.uuid)

                for index, option_att in enumerate(self.extract_opt):
                    if index < self.output_count:
                        dpg.show_item(option_att.uuid)
                        dpg.show_item(option_att.widget.uuid)
            else:
                dpg.hide_item(self.output_count_option.uuid)
                dpg.hide_item(self.output_count_option.widget.uuid)
                dpg.hide_item(self.include_key_in_output_option.uuid)
                dpg.hide_item(self.include_key_in_output_option.widget.uuid)

                for index, option_att in enumerate(self.extract_opt):
                    dpg.hide_item(option_att.uuid)
                    dpg.hide_item(option_att.widget.uuid)

    def key_changed(self):
        for i in range(32):
            key = self.extract_opt[i]()
            if i < self.output_count:
                if key != '':
                    if len(self.extract_keys) > i:
                        self.extract_keys[i] = key
                    else:
                        self.extract_keys.append(key)
                else:
                    if len(self.extract_keys) > i:
                        self.extract_keys[i] = key
                    else:
                        self.extract_keys.append(key)
                dpg.show_item(self.outputs[i].uuid)
                dpg.show_item(self.extract_opt[i].uuid)
                dpg.show_item(self.extract_opt[i].widget.uuid)
        for index, key in enumerate(self.extract_keys):
            self.outputs[index].set_label(key)
        for i in range(self.output_count, 32):
            dpg.hide_item(self.outputs[i].uuid)
            dpg.hide_item(self.extract_opt[i].uuid)
            dpg.hide_item(self.extract_opt[i].widget.uuid)

    def adjust_keys(self, received_dict=None):
        if received_dict is not None:
            self.extract_keys = list(received_dict.keys())
            self.output_count = len(received_dict.keys())
            self.output_count_option.set(self.output_count)

        for i in range(self.output_count):
            if i < len(self.extract_keys):
                key = self.extract_keys[i]
                self.outputs[i].set_label(key)
                self.outputs[i].send(received_dict[key])
                self.extract_opt[i].set(key)
                dpg.show_item(self.outputs[i].uuid)
            else:
                self.extract_keys.append('')
                self.extract_opt[i].set('')
                dpg.show_item(self.outputs[i].uuid)
        for i in range(self.output_count, 32):
            dpg.hide_item(self.outputs[i].uuid)
            self.extract_opt[i].set('')
        self.unparsed_args = self.extract_keys

    def execute(self):
        received_dict = self.input()
        if len(self.extract_keys) == 0:
            self.adjust_keys(received_dict)
            # self.extract_keys = list(received_dict.keys())
            # for i in range(self.output_count):
            #     if i < len(self.extract_keys):
            #         key = self.extract_keys[i]
            #         self.outputs[i].set_label(key)
            #         self.outputs[i].send(received_dict[key])
            #         self.extract_opt[i].set(key)
            #         dpg.show_item(self.outputs[i].uuid)
            #     else:
            #         self.extract_opt[i].set('')
            #         dpg.show_item(self.outputs[i].uuid)
            # for i in range(self.output_count, 32):
            #     dpg.hide_item(self.outputs[i].uuid)
            #     self.extract_opt[i].set('')
            # self.unparsed_args = self.extract_keys
        include_key = self.include_key_in_output_option()
        for index, key in enumerate(self.extract_keys):
            if key in received_dict:
                if include_key:
                    self.outputs[index].send([key, received_dict[key]])
                else:
                    self.outputs[index].send(received_dict[key])


class DictRetrieveNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DictRetrieveNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.dict = {}
        self.key = ''
        self.input = self.add_input('dict in', triggers_execution=True)
        self.key_input = self.add_input('key', callback=self.key_changed)
        self.output = self.add_output('value out')

    def key_changed(self):
        key = self.key_input()
        self.key = key
        if self.key in self.dict:
            self.output.send(self.dict[self.key])

    def execute(self):
        received_dict = self.input()
        self.dict = copy.deepcopy(received_dict)
        if self.key in self.dict:
            self.output.send(self.dict[self.key])


class DictStreamNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DictStreamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('dict in', triggers_execution=True)
        self.output = self.add_output('key value lists out')

    def execute(self):
        received_dict = self.input()
        keys = list(received_dict.keys())
        for key in keys:
            self.output.send([key, received_dict[key]])


class ConstructDictNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ConstructDictNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('send dict', widget_type='button', triggers_execution=True)
        self.arg_inputs = {}
        for arg in args:
            temp_input = self.add_input(arg)
            self.arg_inputs[arg] = temp_input
        self.data_input = self.add_input('labelled data in', callback=self.received_data)
        self.dict_output = self.add_output('dict out')
        self.input_keys = []
        self.dict = {}

    def received_data(self):
        incoming = self.data_input()
        if type(incoming) is str:
            incoming = string_to_list(incoming)
        if type(incoming) is list:
            key = incoming[0]
            value = incoming[1:]
            self.dict[key] = value

    def execute(self):
        for data_input in self.arg_inputs:
            temp_input = self.arg_inputs[data_input]
            data = temp_input()
            self.dict[data_input] = data
        self.dict_output.send(self.dict)
        self.dict = {}

# deprecated - use gather_to_dict

# class PackDictNode(Node):
#     @staticmethod
#     def factory(name, data, args=None):
#         node = PackDictNode(name, data, args)
#         return node
#
#     def __init__(self, label: str, data, args):
#         super().__init__(label, data, args)
#         self.input = self.add_input('send dict', widget_type='button', triggers_execution=True)
#         self.labels = []
#         if len(args) > 0:
#             self.labels = args
#
#         for label in self.labels:
#             self.data_input = self.add_input(label, callback=self.received_data)
#         self.dict_output = self.add_output('dict out')
#         self.input_keys = []
#         self.dict = {}
#
#     def received_data(self):
#         this_input = self.active_input
#         key = this_input.get_label()
#         incoming = this_input()
#         self.dict[key] = incoming
#
#     def execute(self):
#         self.dict_output.send(self.dict)
#         self.dict = {}

class DictKeysNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DictKeysNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('dict in', triggers_execution=True)
        self.output = self.add_output('keys out')

    def execute(self):
        in_dict = self.input()
        if type(in_dict) is dict:
            keys = list(in_dict.keys())
            self.output.send(keys)


class DictReplaceNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DictReplaceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('in', triggers_execution=True)
        self.replace_pairs_in = self.add_input('replace pairs', callback=self.add_replace_pair)
        self.clear_input = self.add_input('clear', callback=self.clear)
        self.output = self.add_output('out')
        self.dict_output = self.add_output('dict out')
        self.dict = {}
        self.update_dict()

    def add_replace_pair(self):
        pair = self.replace_pairs_in()
        if isinstance(pair, str):
            pair = string_to_list(pair)
        if isinstance(pair, list):
            if len(pair) == 2:
                if isinstance(pair[0], str) and isinstance(pair[1], str):
                    self.dict[pair[0]] = pair[1]
            elif len(pair) == 1:
                self.dict.pop(pair[0])
        self.update_dict()

    def update_dict(self):
        # 1. Create a normalized lookup dictionary (keys to lowercase)
        # This allows us to find the key regardless of how the user typed it in the dict
        self.lookup = {k.lower(): v for k, v in self.dict.items()}
        # 2. Sort keys by length (longest first) to avoid partial match errors
        keys = sorted(self.lookup.keys(), key=len, reverse=True)
        # 3. Create the regex pattern
        # \b ensures whole words only. re.IGNORECASE handles the search.
        self.pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in keys) + r')\b', re.IGNORECASE)

    def clear(self):
        self.dict = {}
        self.update_dict()

    def replace_with_case(self, text):
        def match_case_replace(match):
            original_word = match.group()
            lower_key = original_word.lower()
            replacement = self.lookup[lower_key]
            return replacement

        # 5. Perform the substitution
        return self.pattern.sub(match_case_replace, text)

    def execute(self):
        incoming = self.input()
        if isinstance(incoming, dict):
            self.dict = copy.deepcopy(incoming)
            return
        incoming = any_to_string(incoming)
        outgoing = self.replace_with_case(incoming)
        self.output.send(outgoing)

    def save_custom(self, container):
        container['dict'] = self.dict

    def load_custom(self, container):
        if 'dict' in container:
            self.dict = container['dict']
            self.update_dict()

    def dump_dict(self):
        self.dict_out.send(self.dict)


class DictNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DictNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.collection = {}
        self.collection_name = 'untitled'
        self.save_pointer = -1
        self.read_pointer = -1

        self.collection_name = self.arg_as_string(default_value='untitled')

        self.input = self.add_input('retrieve by key', triggers_execution=True)
        self.store_input = self.add_input('store', triggers_execution=True)

        self.load_dict_input = self.add_input('load', widget_type='button', callback=self.load_coll)
        self.collection_name_input = self.add_property('name', widget_type='label', default_value=self.collection_name)
        self.save_dict_input = self.add_property('save', widget_type='button', callback=self.save_coll)
        self.clear_input = self.add_input('clear', widget_type='button', callback=self.clear)
        self.send_input = self.add_input('send dict', widget_type='button', callback=self.send_dict)
        self.output = self.add_output("out")
        self.unmatched_output = self.add_output('unmatched')
        self.dict_out = self.add_output('dict out')

        self.message_handlers['clear'] = self.clear_message
        self.message_handlers['dump'] = self.dump
        self.message_handlers['save'] = self.save_message
        self.message_handlers['load'] = self.load_message
        self.message_handlers['search'] = self.search
        self.message_handlers['next'] = self.next
        self.message_handlers['reset'] = self.reset_counter
        self.message_handlers['keys'] = self.get_keys
        self.message_handlers['random'] = self.random_retrieve
        self.path = ''
        self.file_name = ''
        self.key_pointer = -1

    def random_retrieve(self, message='', data=[]):
        key_list = list(self.collection.keys())
        count = len(key_list)
        random_index = random.randint(0, count - 1)
        key = key_list[random_index]
        value = self.collection[key]
        out_dict = {key: value}
        self.output.send(out_dict)

    def get_keys(self, message='', data=[]):
        keys = list(self.collection.keys())
        self.output.send(keys)

    def reset_counter(self, message='', data=[]):
        self.key_pointer = -1

    def next(self, message='', data=[]):
        self.key_pointer += 1
        if self.key_pointer < len(self.collection):
            keys = list(self.collection.keys())
            value = self.collection[keys[self.key_pointer]]
            out_dict = {keys[self.key_pointer]: value}
            self.output.send(out_dict)

    def save_coll(self):
        path = self.save_dict_input()
        if path is not None and path != '':
            if self.save_data(path):
                # self.path = path
                # self.file_name = path.split('/')[-1]
                # self.file_name = self.file_name.split('.')[0]
                # self.collection_name_input.set(self.file_name)
                return
        self.save_dialog()

    def load_coll(self):
        path = self.load_dict_input()
        if path is not None and path != '':
            if self.load_data(path):
                # self.path = path
                # self.file_name = path.split('/')[-1]
                # self.file_name = self.file_name.split('.')[0]
                # self.collection_name_input.set(self.file_name)
                return
        self.load_dialog()

    def load_coll_by_name(self):
        try:
            self.load_data(self.collection_name_input())
        except Exception as e:
            print(e)

    def dump(self, message='', data=[]):
        for key in self.collection:
            out_list = [key]
            out_list += self.collection[key]
            self.output.send(out_list)

    def send_dict(self):
        self.dict_out.send(self.collection)

    def save_dialog(self):
        SaveDialog(self, callback=self.save_coll_callback, extensions=['.json'])

    def save_coll_callback(self, save_path):
        if save_path != '':
            self.save_data(save_path)
        else:
            print('no file chosen')

    def save_data(self, path):
        with open(path, 'w') as f:
            json.dump(self.collection, f, indent=4, cls=NumpyTorchEncoder)
        self.path = path
        self.file_name = path.split('/')[-1]
        self.file_name = self.file_name.split('.')[0]
        self.collection_name_input.set(self.file_name)

    def save_message(self, message='', data=[]):
        if len(data) > 0:
            path = data[0]
            self.save_data(path)
        else:
            self.save_dialog()

    def save_custom(self, container):
        container['collection'] = self.collection

    def load_custom(self, container):
        if 'collection' in container:
            self.collection = container['collection']

    def load_dialog(self):
        LoadDialog(self, callback=self.load_coll_callback, extensions=['.json'])

    def load_coll_callback(self, load_path):
        if load_path != '':
            self.load_data(load_path)
        else:
            print('no file chosen')

    def load_message(self, message='', data=[]):
        if len(data) > 0:
            path = data[0]
            self.load_data(path)
        else:
            self.load_dialog()

    def load_data(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.collection = json.load(f)
                self.path = path
                self.file_name = path.split('/')[-1]
                self.file_name = self.file_name.split('.')[0]
                self.collection_name_input.set(self.file_name)
                return True
        return False

    def clear(self):
        self.collection = {}
        self.save_pointer = -1

    def clear_message(self, message='', data=[]):
        self.clear()

    # def drill_down(self, dict_level, address_heirachy):
    #     local_address = any_to_string(address_heirachy[0])
    #     if local_address in dict_level:
    #         if len(address_heirachy) > 1:
    #             return self.drill_down(dict_level[local_address], address_heirachy[1:])
    #         else:
    #             return dict_level[local_address]
    #     else:
    #         return None

    def search(self, message='', data=[]):
        data = any_to_string(data)
        results = find_keys_by_prefix(self.collection, data)
        final_results = []
        for result in results:
            address = result[0]
            final_results.append([address, result[1]])

        self.output.send(final_results)

        # def _recurse(current_data, current_path_stack):
        #     # Handle Dictionaries
        #     if isinstance(current_data, dict):
        #         for key, value in current_data.items():
        #             str_key = str(key)
        #             # Build the current path
        #             new_path_stack = current_path_stack + [str_key]
        #
        #             # Check for partial match
        #             if str_key.startswith(prefix):
        #                 path_str = "/".join(new_path_stack)
        #                 results.append((path_str, value))
        #
        #             # Continue recursing if the value is a nested container
        #             if isinstance(value, (dict, list)):
        #                 _recurse(value, new_path_stack)
        #
        #     # Handle Lists (to maintain path integrity through arrays)
        #     elif isinstance(current_data, list):
        #         for index, item in enumerate(current_data):
        #             new_path_stack = current_path_stack + [str(index)]
        #
        #             if isinstance(item, (dict, list)):
        #                 _recurse(item, new_path_stack)
        #
        # _recurse(data, [])
        # return results

    def execute(self):
        if self.active_input == self.input:
            data = self.input()
            # handled, do_output = self.check_for_messages(data)
            # if not handled:
            t = type(data)
            address = any_to_string(data)

            if t in [int, float, np.int64, np.double]:
                if data in self.collection:
                    self.output.send(self.collection[data])
                elif address in self.collection:
                    self.output.send(self.collection[address])
                else:
                    self.unmatched_output.send(data)
            elif t == str:
                address_hierarchy = address.split('/')
                result = drill_down(self.collection, address_hierarchy)
                if result is not None:
                    self.output.send(result)
                else:
                    self.unmatched_output.send(data)

            elif t in [list]:
                index = any_to_string(data[0])
                if index == 'delete':
                    if len(data) > 1:
                        index = any_to_string(data[1])
                        if index in self.collection:
                            self.collection.__delitem__(index)
                    return
                elif index == 'append':
                    self.save_pointer += 1
                    while self.save_pointer in self.collection:
                        self.save_pointer += 1
                    index = self.save_pointer
                if len(data) == 1:
                    if address in self.collection:
                        self.output.send(self.collection[address])
                    else:
                        self.unmatched_output.send(data)
                else:
                    data = data[1:]
                    if len(data) == 1:
                        t = type(data[0])
                        if t == list:
                            self.collection[index] = data[0]
                        elif t == tuple:
                            self.collection[index] = list(data[0])
                        else:
                            self.collection[index] = data
                    else:
                        self.collection[index] = data
        elif self.active_input == self.store_input:
            data = self.store_input()
            # handled, do_output = self.check_for_messages(data)
            # if not handled:
            t = type(data)

            if t is str:
                if data[0] == '{':
                    data = json.loads(data)
                    t = dict
                else:
                    data = string_to_list(data)
                    t = list
            if t is dict:
                self.collection = copy.deepcopy(data)
            elif t == list:
                index = data[0]
                if type(index) not in [str, int]:
                    index = any_to_string(data[0])
                data = data[1:]
                if len(data) == 1:
                    t = type(data[0])
                    if t == list:
                        self.collection[index] = data[0]
                    elif t == tuple:
                        self.collection[index] = list(data[0])
                    else:
                        self.collection[index] = data[0]
                else:
                    self.collection[index] = data


def drill_down(dict_level, address_heirarchy):
    local_address = any_to_string(address_heirarchy[0])
    if local_address in dict_level:
        if len(address_heirarchy) > 1:
            return drill_down(dict_level[local_address], address_heirarchy[1:])
        else:
            return dict_level[local_address]
    else:
        return None

def find_keys_by_prefix(data, search_term):
    """
    Recursively searches a dictionary for keys starting with a specific prefix.

    Args:
        data (dict | list): The data structure to search.
        prefix (str): The substring the key must start with.

    Returns:
        tuple: (list of values, list of paths)
    """
    # data = self.collection

    results = []

    search_parts = search_term.split('/')

    def is_sublist_with_prefix(main_list, sub_list):
        """
        Checks if sub_list is inside main_list where:
        1. All items except the last must be exact matches.
        2. The last item of sub_list must be a prefix of the corresponding main_list item.
        """
        n = len(main_list)
        m = len(sub_list)

        # Edge cases
        if m == 0: return True
        if m > n: return False

        # Iterate through the main list
        for i in range(n - m + 1):
            # 1. Check the "Head" (all items except the last one)
            # We compare the slice of main_list vs sub_list excluding the last item
            # If sub_list is length 1, this compares [] == [], which is True.
            head_matches = (main_list[i: i + m - 1] == sub_list[:-1])

            if head_matches:
                # 2. Check the "Tail" (the last item)
                # We convert to string to safely use .startswith()
                main_tail_val = str(main_list[i + m - 1])
                sub_tail_val = str(sub_list[-1])

                if main_tail_val.startswith(sub_tail_val):
                    return True

        return False

    def _recurse(current_data, current_path_stack):
        # --- Handle Dictionary ---
        if isinstance(current_data, dict):
            if is_sublist_with_prefix(current_path_stack, search_parts):
                if current_path_stack[-1] != 'CONTENTS':
                    results.append(("/".join(current_path_stack), current_data))

            for key, value in current_data.items():
                str_key = str(key)
                new_stack = current_path_stack + [str_key]
                if isinstance(value, dict):
                    _recurse(value, new_stack)

    _recurse(data, [])
    return results


class DictNavigatorNode(Node):
    @staticmethod
    def factory(name: str, data: Any, args: Optional[List[str]] = None) -> 'DictNavigatorNode':
        node = DictNavigatorNode(name, data, args)
        return node

    def __init__(self, label: str, data: Any, args: Optional[List[str]]) -> None:
        super().__init__(label, data, args)
        self.filtered_list: List[str] = []
        self.input = self.add_input('dict in', triggers_execution=True)
        self.search_property = self.add_input(label='##search_term', widget_type='text_input', widget_width=300, callback=self.select)
        self.node_list_box = self.add_property('###options', widget_type='list_box', width=300)
        self.select_button = self.add_property('select', widget_type='button', callback=self.select)
        self.dict = {}
        self.output = self.add_output('results out')

    # in process -
    def search_received(self):
        search = dpg.get_value(self.search_property.widget.uuid)
        if isinstance(search, dict):
            key_list = list(search.keys())
            for key in key_list:
                value = search[key]
                expanded_key = key.replace('/', '/CONTENTS/')
                search_results = find_keys_by_prefix(self.dict, expanded_key)
                for result in search_results:
                    if result[0] == key:
                        pass


    def select(self):
        selected_path = dpg.get_value(self.node_list_box.widget.uuid)
        selected_path = any_to_string(selected_path)
        selected_path = 'CONTENTS/' + selected_path.replace('/', '/CONTENTS/')
        results = find_keys_by_prefix(self.dict, any_to_string(selected_path))
        final_results = []
        for result in results:
            address = result[0]
            stripped_address = address.replace('CONTENTS/', '')
            final_results.append([stripped_address, address, result[1]])

        self.output.send(final_results)

    def execute(self):
        incoming = self.input()
        if isinstance(incoming, dict):
            self.dict = copy.deepcopy(incoming)

    def on_edit(self, widget) -> None:
        if widget == self.search_property.widget and len(self.dict) > 0:
            self.filtered_list = []
            search_name = dpg.get_value(self.search_property.widget.uuid)
            search_name = any_to_string(search_name)
            search_name = search_name.replace('/', '/CONTENTS/')
            search_results = find_keys_by_prefix(self.dict, search_name)
            for result in search_results:
                data = any_to_string(result[0])
                data = data.replace('CONTENTS/', '')
                self.filtered_list.append(data)
            dpg.configure_item(self.node_list_box.widget.uuid, items=self.filtered_list)
            if len(self.filtered_list) > 0:
                self.node_list_box.set(self.filtered_list[0])
                # dpg.set_value(self.node_list_box.widget.uuid, self.filtered_list[0])
                self.current_name = self.filtered_list[0][1]

    def on_deactivate(self, widget):
        if widget == self.search_property.widget:
            if dpg.is_item_hovered(self.node_list_box.widget.uuid) or dpg.is_item_clicked(self.node_list_box.widget.uuid):
                pass
            else:
                self.select()
        elif widget == self.node_list_box.widget:
            self.select()


class RepeatNode(Node):
    output_dict = {'1': 'first',
                   '2': 'second',
                   '3': 'third',
                   '4': 'fourth',
                   '5': 'fifth',
                   '6': 'sixth',
                   '7': 'seventh',
                   '8': 'eighth',
                   '9': 'ninth',
                   '10': 'tenth',
                   '11': 'eleventh',
                   '12': 'twelth',
                   '13': 'thirteenth',
                   '14': 'fourteenth',
                   '15': 'fifteenth',
                   '16': 'sixteenth',
                   '17': 'seventeenth',
                   '18': 'eighteenth',
                   '19': 'nineteenth',
                   '20': 'twentyth',
                   '21': 'twentyfirst',
                   '22': 'twentysecond',
                   '23': 'twentythird',
                   '24': 'twentyfourth',
                   '25': 'twentyfifth',
                   '26': 'twentysixth',
                   '27': 'twentyseventh',
                   '28': 'twentyeighth',
                   '29': 'twentyninth',
                   '30': 'thirtieth',
                   '31': 'thirtyfirst'
                   }

    @staticmethod
    def factory(name, data, args=None):
        node = RepeatNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.repeat_count = self.arg_as_int(default_value=2)

        self.input = self.add_input("", triggers_execution=True)
        self.input.bang_repeats_previous = False
        if self.label == 'repeat_in_order':
            for i in range(self.repeat_count):
                j = self.repeat_count - i
                if str(j) in RepeatNode.output_dict:
                    self.add_output(RepeatNode.output_dict[str(j)])
        else:
            for i in range(self.repeat_count):
                self.add_output('out ' + str(i))

    def execute(self):
        data = self.input()
        for j in reversed(range(self.repeat_count)):
            self.outputs[j].send(data)


class ConduitSendNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ConduitSendNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.conduit_name = ''
        self.conduit = None
        if args is not None and len(args) > 0:
            self.conduit_name = ' '.join(args)
            self.conduit = self.app.find_conduit(self.conduit_name)
            if self.conduit is None:
                self.conduit = Node.app.add_conduit(self.conduit_name)

        self.input = self.add_input(self.conduit_name, triggers_execution=True)
        self.conduit_name_option = self.add_option('name', widget_type='text_input', default_value=self.conduit_name, callback=self.conduit_name_changed)

    def conduit_name_changed(self):
        conduit_name = self.conduit_name_option()
        self.conduit = None
        self.conduit_name = conduit_name
        self.conduit = self.app.find_conduit(self.conduit_name)
        if self.conduit is None:
            self.conduit = Node.app.add_conduit(self.conduit_name)
        self.input.set_label(self.conduit_name)

    def execute(self):
        if self.input.fresh_input and self.conduit is not None:
            data = self.input()
            self.conduit.transmit(data)


class ConduitReceiveNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ConduitReceiveNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.conduit_name = ''
        self.conduit = None
        if args is not None and len(args) > 0:
            self.conduit_name = ' '.join(args)
            self.conduit = self.app.find_conduit(self.conduit_name)
            if self.conduit is None:
                self.conduit = Node.app.add_conduit(self.conduit_name)
            if self.conduit is not None:
                self.conduit.attach_client(self)

        self.output = self.add_output(self.conduit_name)
        self.conduit_name_option = self.add_option('name', widget_type='text_input',
                                                       default_value=self.conduit_name,
                                                       callback=self.conduit_name_changed)

    def conduit_name_changed(self):
        conduit_name = self.conduit_name_option()
        self.conduit.detach_client(self)
        self.conduit = None
        self.conduit_name = conduit_name
        self.conduit = self.app.find_conduit(self.conduit_name)
        if self.conduit is None:
            self.conduit = Node.app.add_conduit(self.conduit_name)
        if self.conduit is not None:
            self.conduit.attach_client(self)
        self.output.set_label(self.conduit_name)

    def custom_cleanup(self):
        if self.conduit is not None:
            self.conduit.detach_client(self)

    def receive(self, name, data):
        self.output.send(data)

    def execute(self):
        pass


class VariableNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = VariableNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.variable_name = ''
        self.variable = None
        if args is not None and len(args) > 0:
            self.variable_name = ' '.join(args)
            self.variable = self.app.find_variable(self.variable_name)
            if self.variable is None:
                default = 0.0
                self.variable = Node.app.add_variable(self.variable_name, default_value=default)
            if self.variable is not None:
                self.variable.attach_client(self)

        self.input = self.add_input("in", triggers_execution=True)
        self.variable_name_property = self.add_property('name', widget_type='text_input', default_value=self.variable_name, callback=self.variable_name_changed)
        self.output = self.add_output("out")

    def variable_name_changed(self):
        variable_name = self.variable_name_property()
        self.variable.detach_client(self)
        self.variable = None
        self.variable_name = variable_name
        self.variable = self.app.find_variable(self.variable_name)
        if self.variable is None:
            default = 0.0
            self.variable = Node.app.add_variable(self.variable_name, default_value=default)
        if self.variable is not None:
            self.variable.attach_client(self)
            self.execute()

    def custom_cleanup(self):
        if self.variable is not None:
            self.variable.detach_client(self)

    def set_variable_value(self, data):
        if self.variable:
            self.variable.set(data, from_client=self)

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            if type(data) == list and len(data) == 1 and type(data[0]) == str and data[0] == 'bang':
                data = self.variable()
            elif type(data) == str and data == 'bang':
                data = self.variable()
            else:
                self.set_variable_value(data)
            self.output.send(data)
        else:
            data = self.variable()
            self.output.send(data)


class ReplaceNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ReplaceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        find = ''
        replace = ''
        if len(args) > 1:
            find = args[0]
            replace = args[1]
        self.input = self.add_input('int in', triggers_execution=True)
        self.find_input = self.add_input('find', widget_type='text_input', default_value=find)
        self.replace_input = self.add_input('replace', widget_type='text_input', default_value=replace)
        self.output = self.add_output('out')

    def execute(self):
        data = self.input()
        find = self.find_input()
        replace = self.replace_input()
        enlisted = False

        t = type(data)
        if t in [str, int, float]:
            data = [data]
            enlisted = True
        out_data = data.copy()
        for index, d in enumerate(data):
            out_data[index] = self.replace_el(d, find, replace)
        if enlisted:
            out_data = data[0]
        self.output.send(out_data)

    def replace_el(self, d, find, replace):
        t = type(d)
        out_el = d
        if t == str:
            out_el = re.sub(r"{}".format(find), replace, d)
        elif t == int:
            find = any_to_int(find)
            if d == find:
                out_el = any_to_int(replace)
        elif t == float:
            find = any_to_float(find)
            if d == find:
                out_el = any_to_float(replace)
        elif t == list:
            out_el = d.copy()
            for index_index, e in enumerate(d):
                out_el[index_index] = self.replace_el(e, find, replace)
        return out_el


class IntReplaceNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = IntReplaceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        find = ''
        replace = ''
        if len(args) > 1:
            find = any_to_int(args[0])
            replace = any_to_int(args[1])
        self.input = self.add_input('int in', triggers_execution=True)
        self.find_input = self.add_input('find', widget_type='drag_int', default_value=find)
        self.replace_input = self.add_input('replace', widget_type='drag_int', default_value=replace)
        self.output = self.add_output('string out')

    def execute(self):
        data = self.input()
        find = self.find_input()
        replace = self.replace_input()
        if type(data) == list:
            for i, w in enumerate(data):
                if type(w) == int:
                    if w == find:
                        data[i] = replace
            self.output.send(data)
        elif type(data) == int:
            if data == find:
                self.output.send(replace)
            else:
                self.output.send(data)
        else:
            self.output.send(data)

class PresentationModeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = PresentationModeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.open_as_presentation = self.add_property('open as presentation', widget_type='checkbox', callback=self.present)

    def present(self):
        if self.open_as_presentation():
            self.app.get_current_editor().enter_presentation_state()
        else:
            self.app.get_current_editor().enter_edit_state()


class StreamListNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = StreamListNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.list_in = self.add_list_input('list in', triggers_execution=True)
        self.stream_out = self.add_output('stream out')

    def execute(self):
        incoming = self.list_in()
        for item in incoming:
            self.stream_out.send(item)


class NpzFileIterator:
    """
    A class to recursively iterate through a directory and yield .npz files.

    This class implements the Python iterator protocol, making it memory-efficient
    for traversing large directory structures. It finds the next .npz file only
    when requested.

    Usage:
        npz_finder = NpzFileIterator('/path/to/your/directory')
        for file_path in npz_finder:
            print(f"Found .npz file: {file_path}")

        # Or using next() manually
        # first_file = next(npz_finder)
        # second_file = next(npz_finder)
    """

    def __init__(self, root_dir: str):
        """
        Initializes the iterator.

        Args:
            root_dir (str): The absolute or relative path to the directory
                            you want to search.

        Raises:
            ValueError: If the provided path is not a valid directory.
        """
        if not os.path.isdir(root_dir):
            raise ValueError(f"Error: The provided path '{root_dir}' is not a valid directory.")

        self.root_dir = root_dir
        # os.walk returns a generator, which is perfect for our lazy iteration
        self._walker = os.walk(self.root_dir)
        self._current_files = []
        self._current_path = ''

    def reset(self):
        """
        Resets the iterator to the initial state.
        """
        self._walker = os.walk(self.root_dir)
        self._current_files = []
        self._current_path = ''

    def __iter__(self):
        """Returns the iterator object itself."""
        return self

    def next_file(self):
        return self.__next__()

    def __next__(self):
        """
        Returns the full path to the next .npz file found.

        Raises:
            StopIteration: When there are no more .npz files to be found.
        """
        while True:
            # If we have files in the current directory buffer, process them
            if self._current_files:
                filename = self._current_files.pop(0)
                if filename.lower().endswith('.npz'):
                    return os.path.join(self._current_path, filename)
                # If not a .npz file, the loop continues to the next file

            # If the buffer is empty, get the next directory from the walker
            else:
                try:
                    # This advances the os.walk generator to the next directory
                    self._current_path, _, self._current_files = next(self._walker)
                except StopIteration:
                    # If os.walk is exhausted, there are no more directories or files
                    # So, we are done iterating.
                    raise StopIteration


class NPZDirectoryIteratorNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NPZDirectoryIteratorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.iterator = None
        self.current_file = None
        self._did_resume = False
        self.next_file_input = self.add_input('next file', triggers_execution=True, trigger_button=True)
        self.directory_input = self.add_input('directory in', widget_type='text_input', callback=self.new_directory)
        self.saving_path = self.add_input('saving path', widget_type='text_input',default_value="/home/bmolab/Projects/AMASS/iterator_state.json")
        self.reset_input = self.add_input('reset', trigger_button=True, callback=self.reset_iterator, trigger_callback=self.reset_iterator)
        self.resume = self.add_input('resume from last run', widget_type="checkbox", default_value=False)
        self.output = self.add_output('next path out')
        self.done_output = self.add_output('done')

    def new_directory(self):
        dir = self.directory_input()
        if dir != '' and os.path.exists(dir):
            self.iterator = NpzFileIterator(dir)
            self.current_file = None
            self._did_resume = False

    def reset_iterator(self):
        self.iterator.reset()
        self.current_file = None
        self._did_resume = False

    def execute(self):
        if self.resume() and not self._did_resume:
            if os.path.exists(self.saving_path()):
                with open(self.saving_path(), "r") as f:
                    target_file = os.path.abspath(json.load(f)["file_path"])
                try:
                    while True:
                        path = self.iterator.next_file()
                        if os.path.abspath(path) == target_file:
                            # path = self.iterator.next_file()
                            self.output.send(path)
                            break
                except StopIteration:
                    self.done_output.send('bang')
            self._did_resume = True
            return

        try:
            if self.iterator is not None:
                file_name = self.iterator.next_file()
                self.current_file = os.path.abspath(file_name)
                self.output.send(file_name)

                with open(self.saving_path(), "w") as f:
                    json.dump(
                        {"file_path": self.current_file}, f)

        except StopIteration:
            self.done_output.send('bang')


class PositionPatchesNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = PositionPatchesNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        top, left = dpg.get_viewport_pos()
        width = dpg.get_viewport_width()
        height = dpg.get_viewport_height()
        self.top_input = self.add_int_input('top', widget_type='drag_int', default_value=top, callback=self.reposition)
        self.left_input = self.add_int_input('left', widget_type='drag_int', default_value=left, callback=self.reposition)
        self.width_input = self.add_int_input('width', widget_type='drag_int', default_value=width, callback=self.reposition)
        self.height_input = self.add_int_input('height', widget_type='drag_int', default_value=height, callback=self.reposition)

    def reposition(self):
        self.execute()

    def execute(self):
        Node.app.position_viewport(self.top_input(), self.left_input())
        Node.app.resize_viewport(self.width_input(), self.height_input())

    def post_load_callback(self):
        self.reposition()


class StartTraceNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = StartTraceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.add_input('start trace', triggers_execution=True)
        self.enable = self.add_input('enable', widget_type='checkbox')
        self.add_output('pass input')

    def execute(self):
        if self.enable():
            Node.app.trace = True
            print()
            print('trace start: frame', Node.app.frame_number)
        self.outputs[0].send(self.inputs[0]())


class EndTraceNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = EndTraceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.add_input('end trace', triggers_execution=True)
        self.add_output('pass input')

    def execute(self):
        if Node.app.trace:
            Node.app.trace = False
            print('trace end')
        self.outputs[0].send(self.inputs[0]())


class SublistNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SublistNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.operations = []
        index_string = ''.join(args) if args else ':'

        self.input = self.add_input('list in', triggers_execution=True)
        self.indices_input = self.add_input(
            'Indices',
            widget_type='text_input',
            widget_width=200,
            default_value=index_string,
            callback=self.indices_changed
        )
        self.output = self.add_output('output')

        # Initialize
        self.indices_changed()

    def indices_changed(self):
        indices_text = any_to_string(self.indices_input())
        indices_text = indices_text.strip().strip("[]")

        if not indices_text:
            self.operations = [slice(None)]
            self.execute()
            return

        parts = indices_text.split(',')
        new_ops = []

        for part in parts:
            part = part.strip()
            if ':' in part:
                slice_params = part.split(':')
                args = []
                for p in slice_params:
                    if p.strip() == '':
                        args.append(None)
                    else:
                        try:
                            args.append(int(p))
                        except ValueError:
                            args.append(None)
                new_ops.append(slice(*args))
            else:
                try:
                    new_ops.append(int(part))
                except ValueError:
                    new_ops.append(slice(None))

        self.operations = new_ops
        self.execute()

    def execute(self):
        raw_input = self.input()

        # 1. No Connection / No Data -> Silent
        if raw_input is None:
            return

        data = any_to_list(raw_input)

        # 2. Conversion failed or returned None -> Silent
        if data is None:
            return

        try:
            current_data = data

            for i, op in enumerate(self.operations):

                # 3. Check if current_data is indexable (list, tuple, etc)
                if not hasattr(current_data, '__getitem__') or isinstance(current_data, str):
                    # We hit a "dead end" (e.g. an integer) but have more ops.
                    # Error implies data mismatch.
                    print(f"SublistNode Error: Cannot index item at depth {i} (Not a list).")
                    self.output.send(None)
                    return

                # 4. Handle Integer Indexing Bounds
                if isinstance(op, int):
                    size = len(current_data)

                    # --- CRITICAL FIX ---
                    # If the list is empty, we assume data hasn't arrived/populated yet.
                    # Return Silently.
                    if size == 0:
                        return

                    # Now we know size > 0, so if index is bad, it's a real error.
                    if op >= size or (op < 0 and abs(op) > size):
                        print(f"SublistNode Error: Index {op} out of bounds for list of size {size}.")
                        self.output.send(None)
                        return

                # Apply
                current_data = current_data[op]

            self.output.send(current_data)

        except Exception as e:
            print(f"SublistNode Error: {e}")
            self.output.send(None)

# class SublistNode(Node):
#     @staticmethod
#     def factory(name, data, args=None):
#         node = SublistNode(name, data, args)
#         return node
#
#     def __init__(self, label: str, data, args):
#         super().__init__(label, data, args)
#         self.indices_list = None
#         index_string = ''
#         for i in range(len(args)):
#             index_string += args[i]
#         if index_string == '':
#             index_string = ':'
#         self.input = self.add_input('tensor in', triggers_execution=True)
#         self.indices_input = self.add_input('', widget_type='text_input', widget_width=200, default_value=index_string, callback=self.indices_changed)
#         self.output = self.add_output('output')
#
#     def indices_changed(self):
#         indices_text = any_to_string(self.indices_input())
#
#         index_split = indices_text.split(',')
#         indices = []
#
#         for i in range(len(index_split)):
#             dimmer = index_split[i]
#             dimmer = dimmer.split(':')
#             dim_nums = []
#             if len(dimmer) == 1:
#                 dim_num = re.findall(r'[-+]?\d+', dimmer[0])
#                 if len(dim_num) > 0:
#                     dim_num = string_to_int(dim_num[0])
#                     dim_nums.append([dim_num])
#                     # dim_nums.append([dim_num + 1])
#             else:
#                 for j in range(len(dimmer)):
#                     dim_num = re.findall(r'[-+]?\d+', dimmer[j])
#                     if len(dim_num) == 0:
#                         if j == 0:
#                             dim_nums.append([0])
#                         else:
#                             dim_nums.append([1000000])
#                     else:
#                         dim_num = string_to_int(dim_num[0])
#                         dim_nums.append([dim_num])
#             indices.append(dim_nums)
#
#         self.indices_list = indices
#         self.execute()
#
#     def execute(self):
#         input_list = any_to_list(self.input())
#         if input_list is not None:
#             if self.indices_list is None:
#                 self.indices_changed()
#
#             if len(self.indices_list) == 0:
#                 self.output.send(input_list)
#             else:
#                 index_list_now = []
#                 for i in range(len(self.indices_list)):
#                     dim_dim = self.indices_list[i]
#                     index_list_now.append(any_to_int(dim_dim[0][0]))
#                     if len(dim_dim) > 1:
#                         if dim_dim[1][0] == 1000000:
#                             index_list_now.append(len(input_list))
#                         else:
#                             index_list_now.append(any_to_int(dim_dim[1][0]))
#                 # sub_list = input_list
#                 if len(index_list_now) > 1:
#                     sub_list = input_list[index_list_now[0]:index_list_now[1]]
#                 else:
#                     sub_list = input_list[index_list_now[0]]
#                 # elif len(index_list_now) == 4:
#                 #
#                 #     sub_list = input_list[index_list_now[0]:index_list_now[1], index_list_now[2]:index_list_now[3]]
#                 # elif len(index_list_now) == 6:
#                 #     sub_list = input_list[index_list_now[0]:index_list_now[1], index_list_now[2]:index_list_now[3], index_list_now[4]:index_list_now[5]]
#                 # elif len(index_list_now) == 8:
#                 #     sub_list = input_list[index_list_now[0]:index_list_now[1], index_list_now[2]:index_list_now[3],
#                 #              index_list_now[4]:index_list_now[5], index_list_now[6]:index_list_now[7]]
#                 if isinstance(sub_list, list):
#                     if len(sub_list) > 1:
#                         self.output.send(sub_list)
#                     else:
#                         self.output.send(sub_list[0])
#                 else:
#                     self.output.send(sub_list)


class ListBoxNode(Node):
    @staticmethod
    def factory(name: str, data: Any, args: Optional[List[str]] = None) -> 'ListBoxNode':
        node = ListBoxNode(name, data, args)
        return node

    def __init__(self, label: str, data: Any, args: Optional[List[str]]) -> None:
        super().__init__(label, data, args)
        self.filtered_list: List[str] = []
        self.input = self.add_input('list in', callback=self.list_received)
        self.search_property = self.add_input(label='##search_term', widget_type='text_input', widget_width=300, callback=self.select)
        self.node_list_box = self.add_property('###options', widget_type='list_box', width=300)
        self.list = []
        self.output = self.add_output('results out')

    def list_received(self):
        new_list = self.input()
        self.list = any_to_list(new_list)
        search_name = dpg.get_value(self.search_property.widget.uuid)
        self.build_box_list(search_name)

    def on_edit(self, widget) -> None:
        if widget == self.search_property.widget and len(self.list) > 0:
            search_name = dpg.get_value(self.search_property.widget.uuid)
            search_name = any_to_string(search_name)
            self.build_box_list(search_name)

    def build_box_list(self, search_name):
        self.filtered_list = []
        for item in self.list:
            if search_name in item or search_name == '':
                self.filtered_list.append(item)
        dpg.configure_item(self.node_list_box.widget.uuid, items=self.filtered_list)
        if self.node_list_box() in self.filtered_list:
            self.node_list_box.set(self.node_list_box())
        elif len(self.filtered_list) > 0:
            self.node_list_box.set(self.filtered_list[0])

    def on_deactivate(self, widget):
        if widget == self.search_property.widget:
            if dpg.is_item_hovered(self.node_list_box.widget.uuid) or dpg.is_item_clicked(self.node_list_box.widget.uuid):
                pass
            else:
                self.select()
        elif widget == self.node_list_box.widget:
            self.select()

    def select(self):
        selected_path = dpg.get_value(self.node_list_box.widget.uuid)
        selected_path = any_to_string(selected_path)
        self.node_list_box.set(selected_path)
        self.output.send(selected_path)
