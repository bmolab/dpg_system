import dearpygui.dearpygui as dpg
import math
import numpy as np
import random
import time
from dpg_system.node import Node
import threading
from dpg_system.conversion_utils import *
import json
from re import match, I as insensitive
from fuzzywuzzy import fuzz


def register_basic_nodes():
    Node.app.register_node('prepend', PrependNode.factory)
    Node.app.register_node('append', AppendNode.factory)
    Node.app.register_node("type", TypeNode.factory)
    Node.app.register_node("array", ArrayNode.factory)
    Node.app.register_node("string", StringNode.factory)
    Node.app.register_node("list", ListNode.factory)
    Node.app.register_node("counter", CounterNode.factory)
    Node.app.register_node('coll', CollectionNode.factory)
    Node.app.register_node("combine", CombineNode.factory)
    Node.app.register_node("kombine", CombineNode.factory)
    Node.app.register_node("delay", DelayNode.factory)
    Node.app.register_node("select", SelectNode.factory)
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


class PlaceholderNode(Node):
    node_list = []

    @staticmethod
    def factory(name, data, args=None):
        node = PlaceholderNode('New Node', data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.filtered_list = []
        self.name_property = self.add_property(label='##node_name', widget_type='text_input', width=180)
        self.static_name = self.add_property(label='##static_name', widget_type='text_input', width=180)
        self.args_property = self.add_property(label='args', widget_type='text_input', width=180)
        self.args_property.add_callback(self.execute())
        if len(self.node_list) == 0:
            self.node_list = self.app.node_factory_container.get_node_list()
        self.variable_list = self.app.get_variable_list()

        self.node_list_box = self.add_property('###options', widget_type='list_box', width=180)
        self.node_list_box.add_callback(self.selection_callback)

    def custom(self):
        dpg.configure_item(self.args_property.widget.uuid, show=False, on_enter=True)
        dpg.configure_item(self.static_name.widget.uuid, show=False)
        dpg.configure_item(self.node_list_box.widget.uuid, show=False)

    def selection_callback(self):
        selection = dpg.get_value(self.node_list_box.widget.uuid)

    def fuzzy_score(self, test):
        scores = {}
        for index, node_name in enumerate(self.node_list):
            ratio = fuzz.partial_ratio(node_name.lower(), test.lower())
            if ratio == 100:
                len_diff = abs(len(test.lower()) - len(node_name.lower()))
                full_ratio = fuzz.ratio(node_name.lower(), test.lower())
                ratio = (ratio * len_diff + full_ratio) / (1 + len_diff)
                if test.lower() == node_name.lower()[:len(test)]:
                    ratio += 10
            scores[node_name] = ratio
        for index, variable_name in enumerate(self.variable_list):
            ratio = fuzz.partial_ratio(variable_name.lower(), test.lower())
            if ratio == 100:
                len_diff = abs(len(test.lower()) - len(variable_name.lower()))
                full_ratio = fuzz.ratio(variable_name.lower(), test.lower())
                ratio = (ratio * len_diff + full_ratio) / (1 + len_diff)
                if test.lower() == variable_name.lower()[:len(test)]:
                    ratio += 10
            scores[variable_name] = ratio
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self.filtered_list = []
        for index, item in enumerate(sorted_scores):
            if item[1] == 100:
                self.filtered_list.append(item[0])
            elif item[1] > 20 and len(self.filtered_list) < 10:
                self.filtered_list.append(item[0])

    def increment_widget(self, widget):
        filter_name = dpg.get_value(self.node_list_box.widget.uuid)
        print(filter_name)
        if filter_name in self.filtered_list:
            print('name in list')
            index = self.filtered_list.index(filter_name)
            print(index)
            index -= 1
            print(index)
            if index >= 0:
                print('ok index')
                filter_name = self.filtered_list[index]
                print(filter_name)
                self.node_list_box.set(filter_name)

    def decrement_widget(self, widget):
        filter_name = dpg.get_value(self.node_list_box.widget.uuid)
        print(filter_name)
        if filter_name in self.filtered_list:
            print('name in list')
            index = self.filtered_list.index(filter_name)
            print(index)
            index += 1
            print(index)
            if index < len(self.filtered_list):
                print('ok index')
                filter_name = self.filtered_list[index]
                print(filter_name)
                self.node_list_box.set(filter_name)

    def on_edit(self, widget):
        if widget == self.static_name:
            return
        if widget == self.name_property.widget and len(self.node_list) > 0:
            self.filtered_list = []
            filter_name = dpg.get_value(self.name_property.widget.uuid)
            if len(filter_name) > 0:
                dpg.configure_item(self.node_list_box.widget.uuid, show=True)
            if len(filter_name) > 0 and filter_name[-1] == ' ':
                selection = dpg.get_value(self.node_list_box.widget.uuid)
                dpg.focus_item(self.node_list_box.widget.uuid)
                dpg.configure_item(self.name_property.widget.uuid, enabled=False)
                dpg.configure_item(self.node_list_box.widget.uuid, items=[], show=False)
                dpg.configure_item(self.name_property.widget.uuid, show=False)
                dpg.configure_item(self.static_name.widget.uuid, show=True)
                dpg.configure_item(self.args_property.widget.uuid, show=True, on_enter=True)
                self.static_name.set(selection)
                dpg.focus_item(self.args_property.widget.uuid)
            else:
                f = filter_name.lower()
                self.fuzzy_score(f)
                dpg.configure_item(self.node_list_box.widget.uuid, items=self.filtered_list)
                if len(self.filtered_list) > 0:
                    dpg.set_value(self.node_list_box.widget.uuid, self.filtered_list[0])

        elif widget == self.node_list_box.widget:
            selection = dpg.get_value(self.node_list_box.widget.uuid)
            dpg.focus_item(self.node_list_box.widget.uuid)
            dpg.configure_item(self.name_property.widget.uuid, enabled=False)
            dpg.configure_item(self.node_list_box.widget.uuid, items=[], show=False)
            dpg.configure_item(self.name_property.widget.uuid, show=False)
            dpg.configure_item(self.static_name.widget.uuid, show=True)
            dpg.configure_item(self.args_property.widget.uuid, show=True, on_enter=True)
            self.static_name.set(selection)
            dpg.focus_item(self.args_property.widget.uuid)

    def on_deactivate(self, widget):
        if widget == self.args_property.widget:
            self.execute()

    def execute(self):
        if dpg.is_item_active(self.name_property.widget.uuid):
            print(self.name_property.get_widget_value())
        else:
            selection_name = dpg.get_value(self.node_list_box.widget.uuid)
            new_node_name = dpg.get_value(self.name_property.widget.uuid)
            arg_string = dpg.get_value(self.args_property.widget.uuid)
            new_node_args = []
            if len(arg_string) > 0:
                args = arg_string.split(' ')
                new_node_args = [new_node_name] + args
            else:
                new_node_args = [new_node_name]
            node_model = None
            found = False
            if new_node_args[0] in self.node_list:
                found = True
            elif selection_name in self.node_list:
                new_node_args[0] = selection_name
                found = True
            if found:
                if len(new_node_args) > 1:
                    Node.app.create_node_by_name(new_node_args[0], self, new_node_args[1:])
                else:
                    Node.app.create_node_by_name(new_node_args[0], self, )
                return
            elif new_node_args[0] in self.variable_list:
                v = self.app.find_variable(new_node_args[0])
                if v is not None:
                    found = True
            elif selection_name in self.variable_list:
                new_node_args[0] = selection_name
                v = self.app.find_variable(new_node_args[0])
                if v is not None:
                    found = True
            if found:
                additional = []
                if len(new_node_args) > 1:
                    additional = new_node_args[1:]
                t = type(v.value)
                found = False
                if t == int:
                    new_node_args = ['int', new_node_args[0]]
                    found = True
                elif t == float:
                    new_node_args = ['float', new_node_args[0]]
                    found = True
                elif t == str:
                    new_node_args = ['message', new_node_args[0]]
                    found = True
                elif t == bool:
                    new_node_args = ['toggle', new_node_args[0]]
                    found = True
                if found:
                    if len(additional) > 0:
                        new_node_args += additional
                    if len(new_node_args) > 1:
                        Node.app.create_node_by_name(new_node_args[0], self, new_node_args[1:])
                    else:
                        Node.app.create_node_by_name(new_node_args[0], self, )


class MetroNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MetroNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.last_tick = 0
        self.period = 30
        self.on = False
        self.units = 1000

        self.on_off_input = self.add_input('on', widget_type='checkbox', trigger_node=self)
        self.on_off_input.add_callback(self.start_stop)

        self.period_input = self.add_input('period', widget_type='drag_float', default_value=self.period)
        self.period_input.add_callback(self.change_period)

        self.units_property = self.add_property('units', widget_type='combo', default_value='milliseconds')
        self.units_property.widget.combo_items = ['seconds', 'milliseconds', 'minutes', 'hours']
        self.units_property.add_callback(self.set_units)

        self.output = self.add_output("")

        self.units_dict = {'seconds': 1, 'milliseconds': 1000, 'minutes': 1.0/60.0, 'hours': 1.0/60.0/60.0}

    def change_period(self):
        self.period = self.period_input.get_widget_value()
        if self.period <= 0:
            self.period = .001

    def start_stop(self):
        # NEED SOMETHING THAT SETS VALUE FIRST THEN CALLS
        self.on = self.on_off_input.get_widget_value()
        if self.on:
            self.last_tick = time.time()

    def set_units(self):
        units_string = self.units_property.get_widget_value() #dpg.get_value(self.shape_input._widget.uuid)
        if units_string in self.units_dict:
            self.units = self.units_dict[units_string]

    def custom(self):
        in_value, t = decode_arg(self.args, 0)
        if t in [int, float]:
            self.period = in_value
            self.period_input.widget.set(self.period)
        self.add_frame_task()

    def frame_task(self):
        if self.on:
            current = time.time()
            period_in_seconds = self.period / self.units
            if current - self.last_tick >= period_in_seconds:
                self.execute()
                self.last_tick = self.last_tick + period_in_seconds
                if self.last_tick + self.period < current:
                    self.last_tick = current

    def execute(self):
        self.output.set(1)
        self.send_outputs()


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

        if args is not None and len(args) > 0:
            if args[0] in self.units_dict:
                self.units = self.units_dict[args[0]]
                default_units = args[0]

        if label == 'elapsed':
            self.mode = 1
            self.input = self.add_input('', trigger_node=self)
        else:
            self.input = self.add_input('on', widget_type='checkbox', trigger_node=self)
            self.input.add_callback(self.start_stop)

        self.units_property = self.add_property('units', widget_type='combo', default_value=default_units)
        self.units_property.widget.combo_items = ['seconds', 'milliseconds', 'minutes', 'hours']
        self.units_property.add_callback(self.set_units)

        self.output = self.add_output("")

        self.output_integers_option = self.add_option('output integers', widget_type='checkbox', default_value=True)

    def update_time_base(self):
        self.base_time = time.time()

    def start_stop(self):
        on = self.input.get_widget_value()
        if on:
            self.update_time_base()

    def set_units(self):
        units_string = self.units_property.get_widget_value()
        if units_string in self.units_dict:
            self.units = self.units_dict[units_string]

    def custom(self):
        if self.mode == 0:
            self.add_frame_task()

    def calc_time(self):
        do_execute = False
        current_time = time.time()
        self.elapsed = (current_time - self.base_time) * self.units
        output_ints = self.output_integers_option.get_widget_value()
        if not output_ints or (int(self.elapsed) != int(self.previous_elapsed)):
            if output_ints:
                self.output_elapsed = int(self.elapsed)
            else:
                self.output_elapsed = self.elapsed
            do_execute = True
        self.previous_elapsed = self.elapsed
        return do_execute

    def frame_task(self):
        on = self.input.get_widget_value()
        if on:
            if self.calc_time():
                self.execute()

    def execute(self):
        if self.mode == 1:
            self.calc_time()
        self.output.set(self.output_elapsed)
        if self.mode == 1:
            self.update_time_base()
        self.send_outputs()


class CounterNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CounterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.max = 255
        self.current_value = 0
        self.step = 1

        self.input = self.add_input("input", trigger_node=self)

        self.max_input = self.add_input('count', widget_type='drag_int', default_value=self.max)
        self.max_input.add_callback(self.change_max)

        self.step_input = self.add_input('step', widget_type='drag_int', default_value=self.step)
        self.step_input.add_callback(self.change_step)

        self.output = self.add_output("count out")

        self.carry_output = self.add_output("carry out")
        self.carry_output.output_always = False

    def change_max(self):
        self.max = self.max_input.get_widget_value()

    def change_step(self):
        self.step = self.step_input.get_widget_value()

    def custom(self):
        in_value, t = decode_arg(self.args, 0)
        if t in [int, float]:
            self.max = int(in_value)
            self.max_input.widget.set(self.max)

        in_value, t = decode_arg(self.args, 1)
        if t in [int, float]:
            self.step = int(in_value)

    def execute(self):
        handled = False
        in_data = self.input.get_data()
        if type(in_data) is str:
            if in_data == 'reset':
                self.current_value = 0
                handled = True
        elif type(in_data) == list:
            list_len = len(in_data)
            if list_len > 0:
                if type(in_data[0]) == str:
                    if in_data[0] == 'set':
                        if list_len > 1:
                            val, t = decode_arg(in_data, 1)
                            if t == int:
                                self.current_value = val
                            elif t == float:
                                self.current_value = int(val)
                            handled = True
                    elif in_data[1] == 'step':
                        if list_len > 1:
                            val, t = decode_arg(in_data, 1)
                            if t == int:
                                self.step = val
                            elif t == float:
                                self.step = int(val)
                            handled = True
        if not handled:
            self.current_value += self.step
            if self.current_value < 0:
                self.carry_output.set(-1)
                self.current_value += self.max
                self.current_value &= self.max
            elif self.current_value >= self.max:
                self.carry_output.set(0)
                self.current_value %= self.max
            elif self.current_value > self.max - self.step:
                self.carry_output.set(1)
        self.output.set(self.current_value)
        self.send_outputs()


class GateNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = GateNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.num_gates = 1
        self.state = 0
        self.bool_state = False

        in_value, t = decode_arg(self.args, 0)
        if t in [int, float]:
            self.num_gates = int(in_value)

        self.open = dpg.generate_uuid()
        self.outputs = []
        if self.num_gates > 1:
            self.choice_input = self.add_input('', widget_type='drag_int', default_value=self.state)
        else:
            self.choice_input = self.add_input('', widget_type='checkbox', widget_uuid=self.bool_state, widget_width=40)
        self.choice_input.add_callback(self.change_state)

        self.gated_input = self.add_input("input", trigger_node=self)

        for i in range(self.num_gates):
            self.outputs.append(self.add_output("out " + str(i)))

    def change_state(self):
        if self.num_gates == 1:
            self.bool_state = self.choice_input.get_widget_value()
        else:
            self.state = self.choice_input.get_widget_value()

    def execute(self):
        if self.num_gates == 1:
            if self.bool_state:
                if self.gated_input.fresh_input:
                    value = self.gated_input.get_data()
                    self.outputs[0].set(value)
        else:
            if self.num_gates >= self.state > 0:
                if self.gated_input.fresh_input:
                    value = self.gated_input.get_data()
                    self.outputs[self.state - 1].set(value)
        self.send_outputs()


class SwitchNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SwitchNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.num_switches = 1
        self.state = 0
        self.bool_state = False

        in_value, t = decode_arg(self.args, 0)
        if t in [int, float]:
            self.num_switches = int(in_value)

        self.choice_input = self.add_input('which input', widget_type='input_int')
        self.choice_input.add_callback(self.change_state)

        self.switch_inputs = []
        for i in range(self.num_switches):
            self.switch_inputs.append(self.add_input('in ' + str(i + 1)))

        self.output = self.add_output('out')

    def change_state(self):
        self.state = self.choice_input.get_widget_value()
        if self.state < 0:
            self.state = 0
            self.choice_input.set(self.state)
        elif self.state > self.num_switches:
            self.state = self.num_switches
            self.choice_input.set(self.state)
        if self.state != 0:
            self.switch_inputs[self.state - 1].trigger_node = self
        for i in range(self.num_switches):
            if i + 1 != self.state:
                self.switch_inputs[i].trigger_node = None

    def execute(self):
        received = self.switch_inputs[self.state - 1].get_received_data()
        self.output.set(received)
        self.send_outputs()


class UnpackNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = UnpackNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.num_outs = 1
        self.outputs = []
        in_value, t = decode_arg(self.args, 0)
        if t in [int, float]:
            self.num_outs = int(in_value)

        self.input = self.add_input("", trigger_node=self)

        for i in range(self.num_outs):
            self.outputs.append(self.add_output("out " + str(i)))

    def execute(self):
        if self.input.fresh_input:
            value = self.input.get_received_data()
            t = type(value)
            if t in [float, int, bool]:
                self.outputs[0].set(value)
            elif t == 'str':
                listing, _, _ = string_to_hybrid_list(value)
                out_count = len(listing)
                if out_count > self.num_outs:
                    out_count = self.num_outs
                for i in range(out_count):
                    self.outputs[i].set(listing[i])
            elif t == list:
                listing, _, _ = list_to_hybrid_list(value)
                out_count = len(listing)
                if out_count > self.num_outs:
                    out_count = self.num_outs
                for i in range(out_count):
                    self.outputs[i].set(listing[i])
            elif t == np.ndarray:
                out_count = value.size
                if out_count > self.num_outs:
                    out_count = self.num_outs
                for i in range(out_count):
                    self.outputs[i].set(value[i])
            self.send_outputs()


class PackNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = PackNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.num_ins = 2

        in_value, t = decode_arg(self.args, 0)
        if t in [int, float]:
            self.num_ins = int(in_value)

        for i in range(self.num_ins):
            if i == 0:
                self.add_input("in " + str(i + 1), trigger_node=self)
            else:
                if label == 'pak':
                    self.add_input("in " + str(i + 1), trigger_node=self)
                else:
                    self.add_input("in " + str(i + 1))

        self.output = self.add_output("out")

        self.output_preference_option = self.add_option('output pref', widget_type='combo')
        self.output_preference_option.widget.combo_items = ['list', 'array']

    def custom(self):
        for i in range(self.num_ins):
            self._input_attributes[i].set_data(0)

    def execute(self):
        trigger = False
        if self.label == 'pak':
            trigger = True
        elif self._input_attributes[0].fresh_input:
            trigger = True
        if trigger:
            out_list = []
            for i in range(self.num_ins):
                value = self._input_attributes[i].get_received_data()
                t = type(value)
                if t in [list, tuple]:
                    out_list += value
                elif t == np.ndarray:
                    array_list = any_to_list(value)
                    out_list += array_list
                else:
                    out_list.append(value)
            out_list, _ = list_to_array_or_list_if_hetero(out_list)
            self.output.set(out_list)
            self.send_outputs()


class DelayNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DelayNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        if len(args) > 0:
            self.delay = any_to_int(args[0])
        else:
            self.delay = 8
        self.buffer = [None] * self.delay
        self.buffer_position = 0

        self.input = self.add_input("in")

        self.delay_input = self.add_input('delay', widget_type='drag_int', default_value=self.delay, min=0, max=4000)
        self.delay_input.add_callback(self.delay_changed)

        self.output = self.add_output("out")

        self.add_frame_task()
        self.new_delay = self.delay

    def delay_changed(self):
        self.new_delay = self.delay_input.get_widget_value()

    def frame_task(self):
        self.execute()

    def execute(self):
        out_data = self.buffer[self.buffer_position]
        if out_data is not None:
            self.output.send(out_data)

        if self.input.fresh_input:
            self.buffer[self.buffer_position] = self.input.get_received_data()
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

        if args is not None:
            if self.mode == 0:
                self.selector_count = len(args)
            else:
                self.selector_count = any_to_int(args[0])

        self.out_mode = 0
        self.selectors = []
        self.selector_options = []
        self.last_states = []
        self.current_states = []
        self.outputs = []

        self.input = self.add_input("in", trigger_node=self)

        if self.mode == 0:
            for i in range(self.selector_count):
                self.outputs.append(self.add_output(any_to_string(args[i])))
            for i in range(self.selector_count):
                val, t = decode_arg(args, i)
                self.selectors.append(val)
                self.last_states.append(0)
                self.current_states.append(0)
            for i in range(self.selector_count):
                an_option = self.add_option('selector ' + str(i), widget_type='text_input', default_value=args[i])
                self.selector_options.append(an_option)
                an_option.add_callback(self.selectors_changed)

        else:
            for i in range(self.selector_count):
                self.outputs.append(self.add_output(str(i)))
                self.selectors.append(i)
                self.last_states.append(0)
                self.current_states.append(0)

        self.output_mode_option = self.add_option('output_mode', widget_type='combo', default_value='bang')
        self.output_mode_option.widget.combo_items = ['bang', 'flag']
        self.output_mode_option.add_callback(self.output_mode_changed)

        self.new_selectors = False

    def output_mode_changed(self):
        if self.output_mode_option.get_widget_value() == 'bang':
            self.out_mode = 0
        else:
            self.out_mode = 1

    def selectors_changed(self):
        self.new_selectors = True

    def update_selectors(self):
        new_selectors = []
        for i in range(self.selector_count):
            new_selectors.append(self.selector_options[i].get_widget_value())
        for i in range(self.selector_count):
            # this does not update the label
            dpg.set_item_label(self.outputs[i].uuid, label=new_selectors[i])
            sel, t = decode_arg(new_selectors, i)
            self.selectors[i] = sel

    def execute(self):
        if self.new_selectors:
            self.update_selectors()
            self.new_selectors = False
        value = self.input.get_data()

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
        self.outputs = []

        self.input = self.add_input("", trigger_node=self)

        for i in range(self.trigger_count):
            self.outputs.append(self.add_output(any_to_string(args[i])))
        for i in range(self.trigger_count):
            val, t = decode_arg(args, i)
            self.triggers.append(val)
        for i in range(self.trigger_count):
            an_option = self.add_option('trigger ' + str(i), widget_type='text_input', default_value=args[i])
            self.trigger_options.append(an_option)
            an_option.add_callback(self.triggers_changed)

        self.new_triggers = False

    def triggers_changed(self):
        self.new_triggers = True

    def update_triggers(self):
        new_triggers = []
        for i in range(self.trigger_count):
            new_triggers.append(self.trigger_options[i].get_widget_value())
        for i in range(self.trigger_count):
            # this does not update the label
            dpg.set_item_label(self.outputs[i].uuid, label=new_triggers[i])
            sel, t = decode_arg(new_triggers, i)
            self.triggers[i] = sel

    def execute(self):
        if self.new_triggers:
            self.update_triggers()
            self.new_triggers = False

        for i in range(self.trigger_count):
            j = self.trigger_count - i - 1
            self.outputs[j].send(self.triggers[j])


class CombineNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CombineNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.num_ins = 2

        self.combine_list = []

        in_value, t = decode_arg(self.args, 0)
        if t in [int, float]:
            self.num_ins = int(in_value)

        for i in range(self.num_ins):
            if i == 0:
                input_ = self.add_input("in " + str(i + 1), trigger_node=self)
            else:
                if label == 'kombine':
                    input_ = self.add_input("in " + str(i + 1), trigger_node=self)
                else:
                    input_ = self.add_input("in " + str(i + 1))
            input_._data = ''

        self.output = self.add_output("out")

    def execute(self):
        output_string = ''
        for i in range(self.num_ins):
            output_string += any_to_string(self._input_attributes[i]._data)
        self.output.set(output_string)
        self.send_outputs()


class TypeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TypeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.num_ins = 2

        self.combine_list = []

        in_value, t = decode_arg(self.args, 0)
        if t in [int, float]:
            self.num_ins = int(in_value)

        self.input = self.add_input("in", trigger_node=self)
        self.type_property = self.add_property('type', widget_type='text_input', width=120)

    def execute(self):
        input = self.input.get_received_data()
        t = type(input)
        if t == float:
            self.type_property.set('float')
        elif t == int:
            self.type_property.set('int')
        elif t == str:
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
        elif t == np.double:
            self.type_property.set('numpy.double')
        elif t == np.int64:
            self.type_property.set('numpy.int64')
        elif t == np.bool_:
            self.type_property.set('numpy.bool_')


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

        self.input = self.add_input("in", trigger_node=self)
        self.output = self.add_output('array out')

        self.shape_property = self.add_option('shape', widget_type='text_input', default_value = shape_text)
        self.shape_property.add_callback(self.shape_changed)

    def shape_changed(self):
        shape_text = self.shape_property.get_widget_value()
        shape_split = re.findall(r'\d+', shape_text)
        shape_list, _, _ = list_to_hybrid_list(shape_split)
        self.shape = shape_list

    def execute(self):
        in_data = self.input.get_received_data()
        out_array = any_to_array(in_data)
        if self.shape is not None:
            out_array = np.reshape(out_array, tuple(self.shape))
        self.output.send(out_array)


class StringNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = StringNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input("in", trigger_node=self)
        self.output = self.add_output('string out')

    def execute(self):
        in_data = self.input.get_received_data()
        out_string = any_to_string(in_data)
        self.output.send(out_string)


class ListNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ListNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input("in", trigger_node=self)
        self.output = self.add_output('list out')

    def execute(self):
        in_data = self.input.get_received_data()
        out_list = any_to_list(in_data)
        self.output.send(out_list)


class PrependNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = PrependNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.num_ins = 2

        self.as_list = False

        in_value, t = decode_arg(self.args, 0)
        self.prepender = in_value

        self.input = self.add_input("in", trigger_node=self)

        self.prepender_property = self.add_property("prefix", widget_type='text_input', default_value=self.prepender)
        self.prepender_property.add_callback(self.prepender_changed)

        self.output = self.add_output("out")

        self.always_as_list_option = self.add_option('always output list', widget_type='checkbox', default_value=False)
        self.always_as_list_option.add_callback(self.option_changed)

    def option_changed(self):
        self.as_list = self.always_as_list_option.get_widget_value()

    def prepender_changed(self):
        self.prepender = self.prepender_property.get_widget_value()

    def execute(self):
        out_list = [self.prepender]
        data = self.input.get_received_data()
        t = type(data)

        if t == str:
            if self.as_list:
                out_list.append(data)
            else:
                out_list = self.prepender + ' ' + data
        elif t in [int, float, bool, np.int64, np.double, np.bool_]:
            out_list.append(data)
        elif t == list:
            out_list += data
        elif t == np.ndarray:
            out_list += any_to_list(data)
        self.output.send(out_list)


class AppendNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = AppendNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.num_ins = 2

        self.as_list = False

        in_value, t = decode_arg(self.args, 0)
        self.appender = in_value

        self.input = self.add_input("in", trigger_node=self)

        self.apender_property = self.add_property("prefix", widget_type='text_input', default_value=self.appender)
        self.apender_property.add_callback(self.appender_changed)

        self.output = self.add_output("out")

        self.always_as_list_option = self.add_option('always output list', widget_type='checkbox', default_value=False)
        self.always_as_list_option.add_callback(self.option_changed)

    def option_changed(self):
        self.as_list = self.always_as_list_option.get_widget_value()

    def appender_changed(self):
        self.appender = self.appender_property.get_widget_value()

    def execute(self):
        out_list = []
        data = self.input.get_received_data()
        t = type(data)

        if t == str:
            if self.as_list:
                out_list = [data]
                out_list.append(self.appender)
            else:
                out_list = data + ' ' + self.appender
        elif t in [int, float, bool, np.int64, np.double, np.bool_]:
            out_list = [data]
            out_list.append(self.appender)
        elif t == list:
            out_list = data
            out_list.append(self.appender)
        elif t == np.ndarray:
            out_list = any_to_list(data)
            out_list.append(self.appender)
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

        in_value, t = decode_arg(self.args, 0)
        if t in [str]:
            self.collection_name = in_value

        self.input = self.add_input('in', trigger_node=self)
        print(self.input.uuid)

        self.collection_name_property = self.add_property('name', widget_type='text_input', default_value=self.collection_name)
        print(self.collection_name_property.uuid)
        print(self.collection_name_property.widget.uuid)

        self.output = self.add_output("out")
        print(self.output.uuid)

    def dump(self):
        for key in self.collection:
            out_list = [key]
            out_list += self.collection[key]
            self.output.set(out_list)
            self.send_outputs()

    def save_dialog(self):
        with dpg.file_dialog(directory_selector=False, show=True, height=400, user_data=self, callback=save_coll_callback,
                             tag="coll_dialog_id"):
            dpg.add_file_extension(".json")

    def save_data(self, path):
        with open(path, 'w') as f:
            json.dump(self.collection, f, indent=4)

    def save_custom(self, container):
        container['collection'] = self.collection

    def load_custom(self, container):
        if 'collection' in container:
            self.collection = container['collection']

    def load_dialog(self):
        with dpg.file_dialog(directory_selector=False, show=True, height=400, user_data=self, callback=load_coll_callback,
                             tag="coll_dialog_id"):
            dpg.add_file_extension(".json")

    def load_data(self, path):
        with open(path, 'r') as f:
            self.collection = json.load(f)

    def execute(self):
        data = self.input.get_received_data()
        t = type(data)
        address = any_to_string(data)
        if t == str:
            if address == 'clear':
                self.collection = {}
                self.save_pointer = -1
                return
            elif address == 'dump':
                self.dump()
                return
            elif address == 'save':
                self.save_dialog()
                return
            elif address == 'load':
                self.load_dialog()
                return
        if t in [int, float, np.int64, np.double]:
            if data in self.collection:
                self.output.set(self.collection[data])
                self.send_outputs()
            elif address in self.collection:
                self.output.set(self.collection[address])
                self.send_outputs()
        elif t == str:
            if address in self.collection:
                self.output.set(self.collection[address])
                self.send_outputs()
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

class RepeatNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RepeatNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.outputs = []
        self.trigger_count = 2
        if args is not None:
            a, t = decode_arg(args, 0)
            if t in [int, float]:
                self.trigger_count = int(a)

        self.input = self.add_input("", trigger_node=self)

        for i in range(self.trigger_count):
            self.outputs.append(self.add_output('out ' + str(i)))

    def execute(self):
        data = self.input.get_received_data()
        for i in range(self.trigger_count):
            j = self.trigger_count - i - 1
            self.outputs[j].send(data)


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
            if self.variable is not None:
                self.variable.attach_client(self)

        self.input = self.add_input("in", trigger_node=self)
        self.variable_name_property = self.add_property('name', widget_type='text_input', default_value=self.variable_name)
        self.output = self.add_output("out")

    def custom_cleanup(self):
        if self.variable is not None:
            self.variable.detach_client(self)

    def set_value(self, data):
        if self.variable:
            self.variable.set(data, from_client=self)

    def get_value(self):
        if self.variable:
            return self.variable.get()

    def execute(self):
        if self.input.fresh_input:
            data = self.input.get_received_data()
            if type(data) == str and data == 'bang':
                data = self.get_value()
            else:
                self.set_value(data)
            self.output.send(data)
        else:
            data = self.get_value()
            self.output.send(data)

#  get variable

#   must be an easy way to declare a value which:
#       when value changes, outputs from nodes
#       or streams continuously
#   can be attached easily to a value in the code / cleanly instantiated







