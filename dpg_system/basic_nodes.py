import dearpygui.dearpygui as dpg
import math
import numpy as np
import random
import time
import string

import torch

from dpg_system.node import Node
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
    Node.app.register_node("string", StringNode.factory)
    Node.app.register_node("list", ListNode.factory)
    Node.app.register_node("counter", CounterNode.factory)
    Node.app.register_node('coll', CollectionNode.factory)
    Node.app.register_node('dict', CollectionNode.factory)
    Node.app.register_node("combine", CombineNode.factory)
    Node.app.register_node("kombine", CombineNode.factory)
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
    Node.app.register_node("timer", TimerNode.factory)
    Node.app.register_node("elapsed", TimerNode.factory)
    Node.app.register_node("decode", SelectNode.factory)
    Node.app.register_node("t", TriggerNode.factory)
    Node.app.register_node('var', VariableNode.factory)
    Node.app.register_node('send', ConduitSendNode.factory)
    Node.app.register_node('receive', ConduitReceiveNode.factory)
    Node.app.register_node('s', ConduitSendNode.factory)
    Node.app.register_node('r', ConduitReceiveNode.factory)
    Node.app.register_node('ramp', RampNode.factory)
    Node.app.register_node('fifo_string', CombineFIFONode.factory)
    Node.app.register_node('bucket_brigade', BucketBrigadeNode.factory)
    Node.app.register_node('tick', TickNode.factory)
    Node.app.register_node('comment', CommentNode.factory)
    Node.app.register_node('fuzzy_match', FuzzyMatchNode.factory)
    Node.app.register_node('length', LengthNode.factory)
    Node.app.register_node('time_between', TimeBetweenNode.factory)
    Node.app.register_node('word_replace', WordReplaceNode.factory)
    Node.app.register_node('string_replace', StringReplaceNode.factory)
    Node.app.register_node('word_trigger', WordTriggerNode.factory)
    Node.app.register_node('split', SplitNode.factory)
    Node.app.register_node('join', JoinNode.factory)
    Node.app.register_node('defer', DeferNode.factory)
    Node.app.register_node('gather_sentence', GatherSentences.factory)
    Node.app.register_node('string_builder', StringBuilder.factory)
    Node.app.register_node('character', CharConverterNode.factory)
    Node.app.register_node('ascii', ASCIIConverterNode.factory)
    Node.app.register_node('char', CharConverterNode.factory)
    Node.app.register_node('ord', ASCIIConverterNode.factory)
    Node.app.register_node('printable', PrintableNode.factory)


# DeferNode -- delays received input until next frame
class DeferNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DeferNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.received = False
        self.received_data = None
        self.input = self.add_input('input', triggers_execution=True)
        self.output = self.add_output('deferred output')
        self.add_frame_task()

    def execute(self):
        self.received_data = self.input()

    def frame_task(self):
        if self.received_data is not None:
            self.output.send(self.received_data)
            self.received_data = None

    def custom_cleanup(self):
        self.remove_frame_tasks()


class CommentNode(Node):
    comment_theme = None
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
        self.setup_theme()
        self.comment_text_option = self.add_option('text', widget_type='text_input', width=200, default_value=self.comment_text, callback=self.comment_changed)
        self.large_text_option = self.add_option('large', widget_type='checkbox', default_value=False, callback=self.large_font_changed)

    def setup_theme(self):
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
        use_large = self.large_text_option()
        if use_large:
            self.set_font(self.app.large_font)
            self.comment_text_option.widget.set_font(self.app.default_font)
            self.large_text_option.widget.set_font(self.app.default_font)
            self.comment_text_option.widget.adjust_to_text_width()
        else:
            self.set_font(self.app.default_font)

    def comment_changed(self):
        self.comment_text = self.comment_text_option()
        self.set_title(self.comment_text)
        self.comment_text_option.widget.adjust_to_text_width()

    def custom_create(self, from_file):
        self.setup_theme()
        dpg.bind_item_theme(self.uuid, CommentNode.comment_theme)
        dpg.configure_item(self.uuid, label=self.comment_text)

    def save_custom(self, container):
        container['name'] = 'comment'
        container['comment'] = self.comment_text

    def load_custom(self, container):
        self.comment_text = container['comment']
        self.setup_theme()
        dpg.bind_item_theme(self.uuid, CommentNode.comment_theme)


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
        self.on_off_input = self.add_input('on', widget_type='checkbox', callback=self.start_stop)
        self.period_input = self.add_input('period', widget_type='drag_float', default_value=self.period, callback=self.change_period)
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
        # if we receive a list... target time... esle fo
        if self.input.fresh_input:
            data = self.input.get_received_data()
            t = type(data)
            self.lock.acquire(blocking=True)
            if t == str:
                t = list
                data = any_to_list(data)
            if t == list:
                if len(data) == 2:
                    self.new_target = True
                    self.target = any_to_float(data[0])
                    self.duration = any_to_float(data[1])
                    self.start_value = self.current_value
                    self.update_time_base()
                elif len(data) == 3:
                    self.new_target = True
                    self.start_value = any_to_float(data[0])
                    self.current_value = self.start_value
                    self.target = any_to_float(data[1])
                    self.duration = any_to_float(data[2])
                    self.update_time_base()
                elif len(data) == 1:
                    self.new_target = True
                    self.go_to_value(any_to_float(data[0]))
                    self.update_time_base()
            elif t in [int, float]:
                self.new_target = True
                self.go_to_value(data)
                self.update_time_base()
            self.lock.release()


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
            self.input = self.add_input('on', widget_type='checkbox', triggers_execution=True, callback=self.start_stop)
        self.units_property = self.add_property('units', widget_type='combo', default_value=default_units, width=100, callback=self.set_units)
        self.units_property.widget.combo_items = ['seconds', 'milliseconds', 'minutes', 'hours']
        self.output = self.add_output("")

        self.output_integers_option = self.add_option('output integers', widget_type='checkbox', default_value=True)

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

        self.max_count = self.arg_as_int(default_value=255, index=0)
        self.step = self.arg_as_int(default_value=1, index=1)

        self.input = self.add_input("input", triggers_execution=True)
        self.input.bang_repeats_previous = False
        self.max_input = self.add_input('count', widget_type='drag_int', default_value=self.max_count, callback=self.update_max_count_from_widget)
        self.step_input = self.add_input('step', widget_type='drag_int', default_value=self.step, callback=self.update_step_from_widget)
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
        self.current_value = 0

    def set_message(self, message='', message_data=[]):
        self.current_value = any_to_int(message_data[0])

    # def step_message(self, message='', message_data=[]):
    #     self.step = any_to_int(message_data[0])

    def execute(self):
        in_data = self.input()
        # handled, do_output = self.check_for_messages(in_data)
        #
        # if not handled:
        self.current_value += self.step
        if self.current_value < 0:
            self.carry_output.set_value(-1)
            self.current_value += self.max_count
            self.current_value &= self.max_count
        elif self.current_value >= self.max_count:
            self.carry_output.set_value(0)
            self.current_value %= self.max_count
        elif self.current_value >= self.max_count - self.step:
            self.carry_output.set_value(1)

        self.output.set_value(self.current_value)
        self.send_all()


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
            self.choice_input = self.add_input('', widget_type='drag_int', triggers_execution=True, default_value=self.state, callback=self.change_state, max=self.num_gates, min=0)
        else:
            self.choice_input = self.add_input('', widget_type='checkbox', triggers_execution=True, default_value=self.bool_state, widget_width=40, callback=self.change_state)
        self.gated_input = self.add_input("input", triggers_execution=True)

        for i in range(self.num_gates):
            self.add_output("out " + str(i))

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

        self.choice_input = self.add_input('which input', widget_type='input_int', callback=self.change_state)
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

        self.num_outs = self.arg_as_int(default_value=1)
        self.input = self.add_input("", triggers_execution=True)

        for i in range(self.num_outs):
            self.add_output("out " + str(i))

    def execute(self):
        if self.input.fresh_input:
            value = self.input()
            t = type(value)
            if t in [float, int, bool]:
                self.outputs[0].set_value(value)
            elif t == 'str':
                listing, _, _ = string_to_hybrid_list(value)
                out_count = len(listing)
                if out_count > self.num_outs:
                    out_count = self.num_outs
                for i in range(out_count):
                    self.outputs[i].set_value(listing[i])
            elif t == list:
                listing, _, _ = list_to_hybrid_list(value)
                # print(listing)
                out_count = len(listing)
                if out_count > self.num_outs:
                    out_count = self.num_outs
                for i in range(out_count):
                    self.outputs[i].set_value(listing[i])
            elif t == np.ndarray:
                out_count = value.size
                if out_count > self.num_outs:
                    out_count = self.num_outs
                if value.dtype in [np.double, float, np.float32]:
                    for i in range(out_count):
                        self.outputs[i].set_value(float(value[i]))
                elif value.dtype in [np.int64, int, np.bool_]:
                    for i in range(out_count):
                        self.outputs[i].set_value(int(value[i]))
            self.send_all()


class PackNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = PackNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.num_ins = self.arg_as_int(default_value=2)

        for i in range(self.num_ins):
            if i == 0:
                self.add_input("in " + str(i + 1), triggers_execution=True)
            else:
                if label == 'pak':
                    self.add_input("in " + str(i + 1), triggers_execution=True)
                else:
                    self.add_input("in " + str(i + 1))

        self.output = self.add_output("out")
        self.output_preference_option = self.add_option('output pref', widget_type='combo')
        self.output_preference_option.widget.combo_items = ['list', 'array']

    def custom_create(self, from_file):
        for i in range(self.num_ins):
            self.inputs[i].receive_data(0)

    def execute(self):
        trigger = False
        if self.label == 'pak':
            trigger = True
        elif self.inputs[0].fresh_input:
            trigger = True
        if trigger:
            out_list = []
            for i in range(self.num_ins):
                value = self.inputs[i].get_data()
                t = type(value)
                # print(t)
                if t in [list, tuple]:
                    out_list += [value]
                elif t == np.ndarray:
                    array_list = any_to_list(value)
                    out_list += [array_list]
                else:
                    out_list.append(value)
            out_list, _ = list_to_array_or_list_if_hetero(out_list)
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
            for i in range(self.bucket_count):
                rev_i = self.bucket_count - i - 1
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
        self.delay_input = self.add_input('delay', widget_type='drag_int', default_value=self.delay, min=0, max=4000, callback=self.delay_changed)
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
            self.buffer[self.buffer_position] = self.input.get_data()
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
            self.routers[i] = any_to_string(sel)

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

        with dpg.theme() as self.active_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 255, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 255, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)
        with dpg.theme() as self.inactive_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)

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
        dpg.bind_item_theme(self.input.widget.uuid, self.active_theme)
        self.add_frame_task()
        self.execute()

    def frame_task(self):
        now = time.time()
        if now >= self.target_time:
            dpg.bind_item_theme(self.input.widget.uuid, self.inactive_theme)
            self.remove_frame_tasks()

    def execute(self):
        if self.new_triggers:
            self.update_triggers()
            self.new_triggers = False

        if self.input.fresh_input or self.force_trigger:
            self.force_trigger = False
            in_data = self.input()
            for i in range(self.trigger_count):
                j = self.trigger_count - i - 1
                trig_mode = self.trigger_pass[j]
                if trig_mode == 0:
                    self.outputs[j].set_value(self.triggers[j])
                elif trig_mode == 1:
                    self.outputs[j].set_value(any_to_int(in_data))
                elif trig_mode == 2:
                    self.outputs[j].set_value(any_to_float(in_data))
                elif trig_mode == 3:
                    self.outputs[j].set_value(any_to_string(in_data))
                elif trig_mode == 4:
                    self.outputs[j].set_value(any_to_list(in_data))
                elif trig_mode == 5:
                    self.outputs[j].set_value(any_to_array(in_data))
                elif trig_mode == 6:
                    self.outputs[j].set_value('bang')
            self.send_all()


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


class SplitNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SplitNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('in', triggers_execution=True)
        self.split_token = None
        if len(args) > 0:
            split_token = any_to_string(args[0])
        self.split_token = self.add_input('split at', widget_type='text_input', default_value=split_token)
        self.output = self.add_output("substrings out")

    def execute(self):
        in_string = self.input()
        t = type(in_string)
        if t == list:
            in_string = ' '.join(in_string)
        if self.split_token == None:
            splits = in_string.split()
        else:
            splits = in_string.split(self.split_token())
        splits = in_string.split(self.split_token())
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
        elif t != list:
            in_list = any_to_list(in_list)
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
            self.decay_rate_property = self.add_property('decay_rate', widget_type='drag_float', default_value=self.decay_rate)
            self.length_threshold_property = self.add_property('length_threshold', widget_type='drag_int', default_value=self.length_threshold)
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
                phrase = any_to_string(self.input())
                self.last_was_progress = False
                self.advance_age()

                length = len(phrase)
                if length > self.length_threshold_property():
                    sub_phrases = re.split(r'[\!\?\.\:\;\,]', phrase)
                else:
                    sub_phrases = phrase.split('.')

                joiners = []
                for index, sp in enumerate(sub_phrases):
                    if len(sp) == 1 and sp[0] == ' ':
                        sp = ''
                    if len(sp) > 0:
                        if len(sp) > 1 and sp[-1] == 'r':
                            if sp[-2] in ['D', 'M']:
                                joiners.append(index)
                        elif len(sp) > 2 and sp[-1] == 's':
                            if sp[-2] == 'r' and sp[-3] == 'M':
                                joiners.append(index)
                        if len(sp) > 1 and sp[-1] == 't':
                            if sp[-2] == 'S':
                                joiners.append(index)

                join_next = False
                adjusted_phrases = []

                for index, p in enumerate(sub_phrases):
                    if len(p) == 1 and p[0] == ' ':
                        p = ''
                    if len(p) > 0:
                        if join_next and len(adjusted_phrases) > 0:
                            adjusted_phrases[-1] = adjusted_phrases[-1] + p + '.'
                            join_next = False
                        else:
                            if p[-1] not in ['.', '?', '!']:
                                adjusted_phrases.append(p + '.')
                            else:
                                adjusted_phrases.append(p)

                        if index in joiners:
                            join_next = True

                for p in adjusted_phrases:
                    self.combine_list[self.pointer] = p
                    self.age[self.pointer] = 1.0
                    self.pointer = (self.pointer - 1) % self.count
                if len(adjusted_phrases) == 0:
                    empty_phrase = True

            output_string_list = []
            output_string = ''
            pointer = self.pointer

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

'''type : TypeNode
    description:
        reports type of received input

    inputs:
        in: <anything>

    properties:
        type : <str> : shows type of the input
            float, int, bang, string, list[length], bool, array[shape], tensor[shape], numpy.double, numpy.float32, numpy.int64, numpy.bool_
'''

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


class TypeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TypeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input("in", triggers_execution=True)
        self.input.bang_repeats_previous = False
        width = 128
        if label == 'info':
            width = 192
        self.type_property = self.add_property(self.label, widget_type='text_input', width=width)

    def execute(self):
        input = self.input()
        if self.label == 'type':
            # print('type', type(input))
            t = type(input)
            if t == float:
                self.type_property.set('float')
            elif t == int:
                self.type_property.set('int')
            elif t == str:
                if input == 'bang':
                    self.type_property.set('bang')
                else:
                    self.type_property.set('string')
            elif t == list:
                self.type_property.set('list[' + str(len(input)) + ']')
            elif t == bool:
                self.type_property.set('bool')
            elif t == np.ndarray:
                shape = input.shape
                if len(shape) == 1:
                    self.type_property.set('array[' + str(shape[0]) + ']')
                elif len(shape) == 2:
                    self.type_property.set('array[' + str(shape[0]) + ', ' + str(shape[1]) + ']')
                elif len(shape) == 3:
                    self.type_property.set('array[' + str(shape[0]) + ', ' + str(shape[1]) + ', ' + str(shape[2]) + ']')
                elif len(shape) == 4:
                    self.type_property.set('array[' + str(shape[0]) + ', ' + str(shape[1]) + ', ' + str(shape[2]) + ', ' + str(shape[3]) + ']')
            elif self.app.torch_available and t == torch.Tensor:
                shape = input.shape
                if len(shape) == 0:
                    self.type_property.set('tensor[]')
                elif len(shape) == 1:
                    self.type_property.set('tensor[' + str(shape[0]) + ']')
                elif len(shape) == 2:
                    self.type_property.set('tensor[' + str(shape[0]) + ', ' + str(shape[1]) + ']')
                elif len(shape) == 3:
                    self.type_property.set('tensor[' + str(shape[0]) + ', ' + str(shape[1]) + ', ' + str(shape[2]) + ']')
                elif len(shape) == 4:
                    self.type_property.set(
                        'tensor[' + str(shape[0]) + ', ' + str(shape[1]) + ', ' + str(shape[2]) + ', ' + str(
                            shape[3]) + ']')
            elif t == np.double:
                self.type_property.set('numpy.double')
            elif t == float:
                self.type_property.set('float')
            elif t == np.float32:
                self.type_property.set('numpy.float32')
            elif t == np.int64:
                self.type_property.set('numpy.int64')
            elif t == np.bool_:
                self.type_property.set('numpy.bool_')
        else:
            t = type(input)
            if t == float:
                self.type_property.set('float')
            elif t == int:
                self.type_property.set('int')
            elif t == str:
                if input == 'bang':
                    self.type_property.set('bang')
                else:
                    self.type_property.set('string')
            elif t == list:
                self.type_property.set('list[' + str(len(input)) + ']')
            elif t == bool:
                self.type_property.set('bool')
            elif t == np.ndarray:
                shape = input.shape
                if input.dtype == float:
                    comp = 'float'
                elif input.dtype == np.double:
                    comp = 'double'
                elif input.dtype == np.float32:
                    comp = 'float32'
                elif input.dtype == np.int64:
                    comp = 'int64'
                elif input.dtype == np.bool_:
                    comp = 'bool'
                elif input.dtype == np.uint8:
                    comp = 'uint8'

                if len(shape) == 1:
                    self.type_property.set('array[' + str(shape[0]) + '] ' + comp)
                elif len(shape) == 2:
                    self.type_property.set('array[' + str(shape[0]) + ', ' + str(shape[1]) + '] ' + comp)
                elif len(shape) == 3:
                    self.type_property.set('array[' + str(shape[0]) + ', ' + str(shape[1]) + ', ' + str(shape[2]) + '] ' + comp)
                elif len(shape) == 4:
                    self.type_property.set(
                        'array[' + str(shape[0]) + ', ' + str(shape[1]) + ', ' + str(shape[2]) + ', ' + str(
                            shape[3]) + '] ' + comp)
            elif self.app.torch_available and t == torch.Tensor:
                shape = input.shape
                if input.dtype == torch.float:
                    comp = 'float'
                elif input.dtype == torch.double:
                    comp = 'double'
                elif input.dtype == torch.float32:
                    comp = 'float32'
                elif input.dtype == torch.int64:
                    comp = 'int64'
                elif input.dtype == torch.bool:
                    comp = 'bool'
                elif input.dtype == torch.uint8:
                    comp = 'uint8'
                elif input.dtype == torch.float16:
                    comp = 'float16'
                elif input.dtype == torch.bfloat16:
                    comp = 'bfloat16'
                elif input.dtype == torch.complex128:
                    comp = 'complex128'
                elif input.dtype == torch.complex64:
                    comp = 'complex64'
                elif input.dtype == torch.complex32:
                    comp = 'complex32'

                device = 'cpu'
                if input.is_cuda:
                    device = 'cuda'
                elif input.is_mps:
                    device = 'mps'

                if input.requires_grad:
                    grad = 'requires_grad'
                else:
                    grad = ''

                if len(shape) == 0:
                    self.type_property.set('tensor[] ' + comp + ' ' + device + ' ' + grad)
                elif len(shape) == 1:
                    self.type_property.set('tensor[' + str(shape[0]) + '] ' + comp + ' ' + device + ' ' + grad)
                elif len(shape) == 2:
                    self.type_property.set('tensor[' + str(shape[0]) + ', ' + str(shape[1]) + '] ' + comp + ' ' + device + ' ' + grad)
                elif len(shape) == 3:
                    self.type_property.set(
                        'tensor[' + str(shape[0]) + ', ' + str(shape[1]) + ', ' + str(shape[2]) + '] ' + comp + ' ' + device + ' ' + grad)
                elif len(shape) == 4:
                    self.type_property.set(
                        'tensor[' + str(shape[0]) + ', ' + str(shape[1]) + ', ' + str(shape[2]) + ', ' + str(
                            shape[3]) + '] ' + comp + ' ' + device + ' ' + grad)
            elif t == float:
                self.type_property.set('float')
            elif t == np.float32:
                self.type_property.set('numpy.float32')
            elif t == np.double:
                self.type_property.set('numpy.double')
            elif t == np.int64:
                self.type_property.set('numpy.int64')
            elif t == np.bool_:
                self.type_property.set('numpy.bool_')


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
            if t in [list, tuple]:
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

        self.input = self.add_input("in", triggers_execution=True)
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
        in_data = self.input()
        out_array = any_to_array(in_data)
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

        self.input = self.add_input("in", triggers_execution=True)
        self.output = self.add_output('string out')

    def execute(self):
        in_data = self.input()
        out_string = any_to_string(in_data)
        self.output.send(out_string)


'''list : ListNode
    description:
        convert input into a list

    inputs:
        in: anything (triggers)

    outputs:
        string out:
            any input is converted into a list and output
            for scalar inputs, a single element list is output
'''


class ListNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ListNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input("in", triggers_execution=True)
        self.output = self.add_output('list out')

    def execute(self):
        in_data = self.input()
        out_list = any_to_list(in_data)
        self.output.send(out_list)


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
                out_list = data + ' ' + self.appender
        elif t in [int, float, bool, np.int64, np.double, np.bool_]:
            out_list = [data]
            out_list.append(self.appender())
        elif t == list:
            out_list = data
            out_list.append(self.appender())
        elif t == np.ndarray:
            out_list = any_to_list(data)
            out_list.append(self.appender())
        self.output.send(out_list)


def save_coll_callback(sender, app_data):
    global save_path
    if 'file_path_name' in app_data:
        save_path = app_data['file_path_name']
        coll_node = dpg.get_item_user_data(sender)
        if save_path != '':
            coll_node.save_data(save_path)
    else:
        print('no file chosen')
    dpg.delete_item(sender)
    Node.app.active_widget = -1


def load_coll_callback(sender, app_data):
    global load_path
    if 'file_path_name' in app_data:
        load_path = app_data['file_path_name']
        coll_node = dpg.get_item_user_data(sender)
        if load_path != '':
            coll_node.load_data(load_path)
    else:
        print('no file chosen')
    dpg.delete_item(sender)
    Node.app.active_widget = -1


class CollectionNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CollectionNode(name, data, args)
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
        self.collection_name_property = self.add_property('name', widget_type='text_input', default_value=self.collection_name)
        self.output = self.add_output("out")
        self.message_handlers['clear'] = self.clear_message
        self.message_handlers['dump'] = self.dump
        self.message_handlers['save'] = self.save_message
        self.message_handlers['load'] = self.load_message

    def dump(self, message='', data=[]):
        for key in self.collection:
            out_list = [key]
            out_list += self.collection[key]
            self.output.send(out_list)

    def save_dialog(self):
        with dpg.file_dialog(directory_selector=False, show=True, height=400, width=800, user_data=self, callback=save_coll_callback,
                             tag="coll_dialog_id"):
            dpg.add_file_extension(".json")

    def save_data(self, path):
        with open(path, 'w') as f:
            json.dump(self.collection, f, indent=4)

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
        with dpg.file_dialog(directory_selector=False, show=True, height=400, width=800, user_data=self, callback=load_coll_callback,
                             tag="coll_dialog_id"):
            dpg.add_file_extension(".json")

    def load_message(self, message='', data=[]):
        if len(data) > 0:
            path = data[0]
            self.load_data(path)
        else:
            self.load_dialog()

    def load_data(self, path):
        with open(path, 'r') as f:
            self.collection = json.load(f)

    def clear_message(self, message='', data=[]):
        self.collection = {}
        self.save_pointer = -1

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
            elif t == str:
                if address in self.collection:
                    self.output.send(self.collection[address])
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

            if t == list:
                index = any_to_string(data[0])
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


class RepeatNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RepeatNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.trigger_count = self.arg_as_int(default_value=2)

        self.input = self.add_input("", triggers_execution=True)
        self.input.bang_repeats_previous = False
        for i in range(self.trigger_count):
            self.add_output('out ' + str(i))

    def execute(self):
        data = self.input()
        for i in range(self.trigger_count):
            j = self.trigger_count - i - 1
            self.outputs[j].set_value(data)
        self.send_all()


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
            if type(data) == str and data == 'bang':
                data = self.variable()
            else:
                self.set_variable_value(data)
            self.output.send(data)
        else:
            data = self.variable()
            self.output.send(data)


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

    def load_from_load_path(self):
        path = self.load_path()
        print('load_from_load_path', path)
        if path != '':
            self.load_match_file_from_json(self.load_path())

    def load_match_file(self):
        with dpg.file_dialog(modal=True, directory_selector=False, show=True, height=400, width=800,
                             user_data=self, callback=self.load_match_file_callback, tag="match_file_dialog_id"):
            dpg.add_file_extension(".json")

    def load_match_file_callback(self, sender, app_data):
        if 'file_path_name' in app_data:
            path = app_data['file_path_name']
            if path != '':
                self.load_path.set(path)
                self.load_from_load_path()
        else:
            print('no file chosen')
        dpg.delete_item(sender)

    def load_match_file_from_json(self, path):
        print('load from json')
        self.list_path = path
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
                elapsed = time.time() - start
                # print(elapsed)

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
        # print(sorted_scores)
        self.filtered_list = []
        self.best_score = 20
        for index, item in enumerate(sorted_scores):
            if item[1] == 100:
                self.filtered_list.append(item[0])
                self.best_score = item[1]
            elif item[1] > self.best_score and len(self.filtered_list) < 10:
                self.filtered_list.append(item[0])
                self.best_score = item[1]


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
                if type(data) == str:
                    data = re.sub(r"{}".format(find), replace, data)
                    self.output.send(data)
                else:
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
                if data.find(word_trigger) != -1:
                    self.trigger_outputs[index].send('bang')


class GatherSentences(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = GatherSentences(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.received_sentence = ''
        self.input = self.add_input('string in', triggers_execution=True)
        self.end_input = self.add_input('force string end', triggers_execution=True)
        self.enforce_spaces = self.add_property('enforce spaces', widget_type='checkbox')
        self.sentence_output = self.add_output('sentences out')

    def execute(self):
        if self.active_input == self.input:
            data = any_to_string(self.input())
            if len(data) > 0:
                if data[-1] == '\n' and len(data) > 1:
                    if data[-2] == '\n':
                        self.received_sentence += data
                        # self.received_sentence = self.received_sentence.replace('\n', ' ')

                        self.sentence_output.send(self.received_sentence)
                        self.received_sentence = ''
                        return
                if data[-1] == '-' and len(data) > 1:
                    if data[-2] == '-':
                        self.received_sentence += data
                        # self.received_sentence = self.received_sentence.replace('\n', ' ')
                        self.sentence_output.send(self.received_sentence)
                        self.received_sentence = ''
                        return
                if data[-1] in ['.', '?', '!', ';', ':']:
                    self.received_sentence += data
                    # self.received_sentence = self.received_sentence.replace('\n', ' ')
                    self.sentence_output.send(self.received_sentence)
                    self.received_sentence = ''
                    return
            if self.enforce_spaces() and len(self.received_sentence) > 0 and len(data) > 0:
                if self.received_sentence[-1] != ' ' and data[0] != ' ':
                    self.received_sentence += ' '
            self.received_sentence += data
        else:
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
        self.trigger_input = self.add_input('issue text', triggers_execution=True)
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
