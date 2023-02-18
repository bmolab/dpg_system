import dearpygui.dearpygui as dpg
import math
import numpy as np
import random
import time
from dpg_system.node import Node
import threading
from dpg_system.conversion_utils import *
import json


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
        # custom_setup() is called after that creation routine, allowing you to do any
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
        self.output = self.add_output("")

    def change_period(self):
        self.period = self.period_input.get_widget_value()
        if self.period <= 0:
            self.period = .001

    def start_stop(self):
        self.on = self.on_off_input.get_widget_value()
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

    def set_units(self):
        units_string = self.units_property.get_widget_value()
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
            self.input = self.add_input('', triggers_execution=True)
        else:
            self.input = self.add_input('on', widget_type='checkbox', triggers_execution=True, callback=lambda: self.start_stop())

        self.units_property = self.add_property('units', widget_type='combo', default_value=default_units, callback=lambda: self.set_units())
        self.units_property.widget.combo_items = ['seconds', 'milliseconds', 'minutes', 'hours']
        self.output = self.add_output("")

        self.output_integers_option = self.add_option('output integers', widget_type='checkbox', default_value=True)

        if self.mode == 0:
            self.add_frame_task()

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
        self.output.send(self.output_elapsed)
        if self.mode == 1:
            self.update_time_base()


class CounterNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CounterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.current_value = 0

        self.max_count = self.arg_as_int(default_value=255)
        self.step = self.arg_as_int(default_value=1)

        self.input = self.add_input("input", triggers_execution=True)
        self.max_input = self.add_input('count', widget_type='drag_int', default_value=self.max_count, callback=lambda: self.update_max_count_from_widget())
        self.step_input = self.add_input('step', widget_type='drag_int', default_value=self.step, callback=lambda: self.update_step_from_widget())
        self.output = self.add_output("count out")
        self.carry_output = self.add_output("carry out")
        self.carry_output.output_always = False

        self.message_handlers['reset'] = self.reset_message
        self.message_handlers['set'] = self.set_message
        # self.message_handlers['step'] = self.step_message

    # widget callbacks
    def update_max_count_from_widget(self):
        self.max_count = self.max_input.get_widget_value()

    def update_step_from_widget(self):
        self.step = self.step_input.get_widget_value()

    # messages
    def reset_message(self, message='', message_data=[]):
        self.current_value = 0

    def set_message(self, message='', message_data=[]):
        self.current_value = any_to_int(message_data[0])

    # def step_message(self, message='', message_data=[]):
    #     self.step = any_to_int(message_data[0])

    def execute(self):
        in_data = self.input.get_received_data()
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
        elif self.current_value > self.max_count - self.step:
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

    def change_state(self):
        if self.num_gates == 1:
            self.bool_state = self.choice_input.get_widget_value()
        else:
            self.state = self.choice_input.get_widget_value()

    def execute(self):
        if self.num_gates == 1:
            if self.bool_state:
                if self.gated_input.fresh_input:
                    self.outputs[0].send(self.gated_input.get_received_data())
        else:
            if self.num_gates >= self.state > 0:
                if self.gated_input.fresh_input:
                    value = self.gated_input.get_received_data()
                    self.outputs[self.state - 1].send(self.gated_input.get_received_data())


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

    def change_state(self):
        self.state = self.choice_input.get_widget_value()
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
        received = self.switch_inputs[self.state - 1].get_received_data()
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
            value = self.input.get_received_data()
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
                out_count = len(listing)
                if out_count > self.num_outs:
                    out_count = self.num_outs
                for i in range(out_count):
                    self.outputs[i].set_value(listing[i])
            elif t == np.ndarray:
                out_count = value.size
                if out_count > self.num_outs:
                    out_count = self.num_outs
                if value.dtype in [np.double, np.float]:
                    for i in range(out_count):
                        self.outputs[i].set_value(float(value[i]))
                elif value.dtype in [np.int64, np.int, np.bool_]:
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

    def custom_setup(self):
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
                if t in [list, tuple]:
                    out_list += value
                elif t == np.ndarray:
                    array_list = any_to_list(value)
                    out_list += array_list
                else:
                    out_list.append(value)
            out_list, _ = list_to_array_or_list_if_hetero(out_list)
            self.output.send(out_list)


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
        value = self.input.get_received_data()

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
        self.trigger_pass = []

        self.input = self.add_input("", triggers_execution=True)

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
#        print('trigger changed')
        self.new_triggers = True
        self.update_triggers()

    def update_triggers(self):
 #       print('trigger updates')
        new_triggers = []
        for i in range(self.trigger_count):
            new_triggers.append(self.trigger_options[i].get_widget_value())
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

    def execute(self):
        if self.new_triggers:
            self.update_triggers()
            self.new_triggers = False

        if self.input.fresh_input:
            in_data = self.input.get_received_data()
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


class TypeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TypeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input("in", triggers_execution=True)
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

        self.input = self.add_input("in", triggers_execution=True)
        self.output = self.add_output('array out')

        self.shape_property = self.add_option('shape', widget_type='text_input', default_value=shape_text, callback=self.shape_changed)

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

        self.input = self.add_input("in", triggers_execution=True)
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

        self.input = self.add_input("in", triggers_execution=True)
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

        self.prepender = ''

        if len(args) > 0:
            in_value, t = decode_arg(args, 0)
            self.prepender = in_value

        self.input = self.add_input("in", triggers_execution=True)
        self.prepender_property = self.add_property("prefix", widget_type='text_input', default_value=self.prepender, callback=self.prepender_changed)
        self.output = self.add_output("out")
        self.always_as_list_option = self.add_option('always output list', widget_type='checkbox', default_value=False, callback=self.option_changed)

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

        self.as_list = False

        self.appender = ''

        if len(args) > 0:
            in_value, t = decode_arg(args, 0)
            self.appender = in_value

        self.input = self.add_input("in", triggers_execution=True)
        self.appender_property = self.add_property("prefix", widget_type='text_input', default_value=self.appender, callback=self.appender_changed)
        self.output = self.add_output("out")
        self.always_as_list_option = self.add_option('always output list', widget_type='checkbox', default_value=False, callback=self.option_changed)

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

        self.collection_name = self.arg_as_string(default_value='untitled')

        self.input = self.add_input('in', triggers_execution=True)
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
        with dpg.file_dialog(directory_selector=False, show=True, height=400, user_data=self, callback=save_coll_callback,
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

    def save_custom_setup(self, container):
        container['collection'] = self.collection

    def load_custom_setup(self, container):
        if 'collection' in container:
            self.collection = container['collection']

    def load_dialog(self):
        with dpg.file_dialog(directory_selector=False, show=True, height=400, user_data=self, callback=load_coll_callback,
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
        data = self.input.get_received_data()
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


class RepeatNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RepeatNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.trigger_count = self.arg_as_int(default_value=2)

        self.input = self.add_input("", triggers_execution=True)
        for i in range(self.trigger_count):
            self.add_output('out ' + str(i))

    def execute(self):
        data = self.input.get_received_data()
        for i in range(self.trigger_count):
            j = self.trigger_count - i - 1
            self.outputs[j].set_value(data)
        self.send_all()


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
        variable_name = self.variable_name_property.get_widget_value()
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

    def get_variable_value(self):
        if self.variable:
            return self.variable.get()

    def execute(self):
        if self.input.fresh_input:
            data = self.input.get_received_data()
            if type(data) == str and data == 'bang':
                data = self.get_variable_value()
            else:
                self.set_variable_value(data)
            self.output.send(data)
        else:
            data = self.get_variable_value()
            self.output.send(data)

#  get variable

#   must be an easy way to declare a value which:
#       when value changes, outputs from nodes
#       or streams continuously
#   can be attached easily to a value in the code / cleanly instantiated







