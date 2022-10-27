import dearpygui.dearpygui as dpg
import math
import numpy as np
import random
import time
from dpg_system.node import Node
from dpg_system.conversion_utils import *


def register_signal_nodes():
    Node.app.register_node("filter", FilterNode.factory)
    Node.app.register_node("smooth", FilterNode.factory)
    Node.app.register_node("diff_filter_bank", MultiDiffFilterNode.factory)
    Node.app.register_node("diff_filter", MultiDiffFilterNode.factory)
    Node.app.register_node("random", RandomNode.factory)
    Node.app.register_node("signal", SignalNode.factory)
    Node.app.register_node("togedge", TogEdgeNode.factory)
    Node.app.register_node("subsample", SubSampleNode.factory)
    Node.app.register_node("diff", DifferentiateNode.factory)
    Node.app.register_node('noise_gate', NoiseGateNode.factory)
    Node.app.register_node('trigger', ThresholdTriggerNode.factory)
    Node.app.register_node('hysteresis', ThresholdTriggerNode.factory)
    Node.app.register_node('sample_hold', SampleHoldNode.factory)


class DifferentiateNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DifferentiateNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.previous_value = None
        self.previousType = None

        self.input = self.add_input("", trigger_node=self)
        self.output = self.add_output("")

    def float_diff(self, received):
        if float == self.previousType:
            output = received - self.previous_value
        else:
            prev = any_to_float(self.previous_value)
            output = received - prev
        return output

    def array_diff(self, received):
        if np.ndarray == self.previousType:
            output = received - self.previous_value
        else:
            prev = any_to_array(self.previous_value)
            output = received - prev
        return output

    def list_diff(self, received):
        received = any_to_array(received)
        if np.ndarray == self.previousType:
            output = received - self.previous_value
        else:
            prev = any_to_array(self.previous_value)
            output = received - prev
        return output, received

    def int_diff(self, received):
        if int == self.previousType:
            output = received - self.previous_value
        else:
            prev = any_to_int(self.previous_value)
            output = received - prev
        return output

    def bool_diff(self, received):
        if bool == self.previousType:
            output = received != self.previous_value
        else:
            prev = any_to_bool(self.previous_value)
            output = received != prev
        return output

    def execute(self):
        received = self.input.get_received_data()
        t = type(received)
        output = None
        if self.previous_value is not None:
            if t == float:
                output = self.float_diff(received)
            elif t == int:
                output = self.int_diff(received)
            elif t == bool:
                output = self.bool_diff(received)
            elif t == list:
                output, received = self.list_diff(received)
                t = np.ndarray
            if t == np.ndarray:
                output = self.array_diff(received)
            self.output.set(output)
            self.send_outputs()

        self.previous_value = received
        self.previousType = t


class RandomNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RandomNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.range = 1.0
        self.bipolar = False

        self.trigger_input = self.add_input('trigger', trigger_node=self)
        self.range_input = self.add_input('range', widget_type='drag_float', default_value=self.range)
        self.range_input.widget.speed = 0.01
        self.bipolar_property = self.add_option('bipolar', widget_type='checkbox', default_value=self.bipolar)
        self.output = self.add_output('out')

    def custom(self):
        in_value, t = decode_arg(self.args, 0)
        if t in [int, float]:
            self.range_input.widget.set(self.range)

    def execute(self):
        if self.bipolar_property.get_widget_value():
            output_value = random.random() * self.range_input.get_widget_value() * 2 - self.range_input.get_widget_value()
        else:
            output_value = random.random() * self.range_input.get_widget_value()
        self.output.set(output_value)
        self.send_outputs()


class SignalNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SignalNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.period = 1.0
        self.on = False
        self.shape = 'sin'
        self.signal_value = 0
        self.first_tick = 0
        self.last_tick = 0
        self.time = 0
        self.range = 1
        self.bipolar = True
        self.vector_size = 1
        self.vector = None

        self.on_off_input = self.add_input('on', widget_type='checkbox', trigger_node=self)
        self.on_off_input.add_callback(self.start_stop)

        self.period_input = self.add_input('period', widget_type='drag_float', default_value=self.period)
        self.period_input.add_callback(self.change_period)

        self.shape_input = self.add_input('shape', widget_type='combo', default_value=self.shape)
        self.shape_input.widget.combo_items = ['sin', 'cos', 'saw', 'square', 'triangle', 'random']
        self.shape_input.add_callback(self.set_shape)

        self.range_property = self.add_option('range', widget_type='drag_float', default_value=self.range)
        self.range_property.add_callback(self.change_range)
        self.range_property.widget.speed = 0.01

        self.bipolar_property = self.add_option('bipolar', widget_type='checkbox', default_value=self.bipolar)
        self.bipolar_property.add_callback(self.change_bipolar)

        self.size_property = self.add_option('vector size', widget_type='drag_int', default_value=self.vector_size)
        self.size_property.add_callback(self.change_size)

        self.output = self.add_output("")

    def custom(self):
        if self.args is not None and len(self.args) > 0:
            for arg in self.args:
                if arg in ['sin', 'cos', 'saw', 'square', 'triangle', 'random']:
                    self.shape = arg
                    self.shape_input.widget.set(self.shape)
                elif float(arg) != 0:
                    self.period = float(arg)
                    self.period_input.widget.set(self.period)

        self.add_frame_task()

    def update_parameters_from_widgets(self):
        self.set_shape()
        self.change_period()
        self.change_range()
        self.change_bipolar()
        self.start_stop()

    def set_shape(self):
        self.shape = self.shape_input.get_widget_value()

    def change_size(self):
        self.vector_size = self.size_property.get_widget_value()
        if self.vector_size != 1:
            self.vector = np.ndarray((self.vector_size))

    def change_period(self):
        self.period = self.period_input.get_widget_value()
        if self.period <= 0:
            self.period = .001

    def change_range(self):
        self.range = self.range_property.get_widget_value()

    def change_bipolar(self):
        self.bipolar = self.bipolar_property.get_widget_value()

    def start_stop(self):
        self.on = self.on_off_input.get_widget_value()
        if self.on:
            self.first_tick = time.time()

    def frame_task(self):
        if self.on:
            current = time.time()
            elapsed = current - self.last_tick
            if self.vector_size == 1:
                self.time += (elapsed / self.period)
                delta = self.time % 1
                if self.shape == 'sin':
                    self.signal_value = math.sin(delta * math.pi * 2)
                elif self.shape == 'cos':
                    self.signal_value = math.cos(delta * math.pi * 2)
                elif self.shape == 'saw':
                    self.signal_value = delta * 2 - 1
                elif self.shape == 'square':
                    self.signal_value = (delta * 2 // 1) * 2 - 1
                elif self.shape == 'triangle':
                    self.signal_value = abs(0.5 - delta) * 4 - 1
                elif self.shape == 'random':
                    self.signal_value = random.random() * 2 - 1
                if not self.bipolar:
                    self.signal_value += 1
                    self.signal_value /= 2
                self.signal_value *= self.range
                self.last_tick = current
            else:
                sub_period = elapsed / self.period / self.vector_size
                for i in range(self.vector_size):
                    self.time += sub_period
                    delta = self.time % 1
                    if self.shape == 'sin':
                        self.vector[i] = math.sin(delta * math.pi * 2)
                    elif self.shape == 'cos':
                        self.vector[i] = math.cos(delta * math.pi * 2)
                    elif self.shape == 'saw':
                        self.vector[i] = delta * 2 - 1
                    elif self.shape == 'square':
                        self.vector[i] = (delta * 2 // 1) * 2 - 1
                    elif self.shape == 'triangle':
                        self.vector[i] = abs(0.5 - delta) * 4 - 1
                    elif self.shape == 'random':
                        self.vector[i] = random.random() * 2 - 1
                if not self.bipolar:
                    self.vector += 1
                    self.vector /= 2
                self.vector *= self.range
                self.signal_value = self.vector
                self.last_tick = current
            self.execute()

    def execute(self):
        self.output.set(self.signal_value)
        self.send_outputs()


class SubSampleNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SubSampleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.subsampler = 2
        self.sample_count = 0

        if args is not None and len(args) > 0:
            self.subsampler = any_to_int((args[0]))

        self.input = self.add_input("input", trigger_node=self)
        self.input.add_callback(self.execute)

        self.rate_property = self.add_property('rate', widget_type='drag_int', default_value=self.subsampler, min=0, max=math.inf)
        self.rate_property.add_callback(self.rate_changed)

        self.output = self.add_output("out")

    def rate_changed(self):
        self.subsampler = self.rate_property.get_widget_value()

    def execute(self):
        if self.input.fresh_input:
            self.sample_count += 1
            if self.sample_count + 1 >= self.subsampler:
                self.sample_count = 0
                self.output.send(self.input._data)


class NoiseGateNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NoiseGateNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.threshold = 0.1
        self.bipolar = False
        self.squeeze = False

        if args is not None and len(args) > 0:
            self.threshold = any_to_float((args[0]))

        self.input = self.add_input("input", trigger_node=self)

        self.threshold_property = self.add_property('threshold', widget_type='drag_float', default_value=self.threshold)
        self.threshold_property.add_callback(self.option_changed)

        self.output = self.add_output("out")

        self.bipolar_option = self.add_option('bipolar', widget_type='checkbox', default_value=self.bipolar)
        self.bipolar_option.add_callback(self.option_changed)

        self.squeeze_option = self.add_option('squeeze', widget_type='checkbox', default_value=self.bipolar)
        self.squeeze_option.add_callback(self.option_changed)

    def option_changed(self):
        self.threshold = self.threshold_property.get_widget_value()
        self.squeeze = self.squeeze_option.get_widget_value()
        self.bipolar = self.bipolar_option.get_widget_value()

    def execute(self):
        data = self.input.get_received_data()
        t = type(data)
        output_data = data
        if t in [float, np.double]:
            if self.bipolar:
                if self.squeeze:
                    if output_data < 0:
                        output_data += self.threshold
                        if output_data > 0:
                            output_data = 0.0
                    else:
                        output_data -= self.threshold
                        if output_data < 0:
                            output_data = 0.0
                else:
                    if -self.threshold < data < self.threshold:
                        output_data = 0.0

            else:
                if self.squeeze:
                    output_data -= self.threshold
                    if output_data < 0:
                        output_data = 0.0
                else:
                    if data < self.threshold:
                        output_data = 0.0
        elif t in [int, np.int64]:
            if self.bipolar:
                if self.squeeze:
                    if output_data < 0:
                        output_data += self.threshold
                        if output_data > 0:
                            output_data = 0
                    else:
                        output_data -= self.threshold
                        if output_data < 0:
                            output_data = 0
                else:
                    if -self.threshold < data < self.threshold:
                        output_data = 0
            else:
                if self.squeeze:
                    output_data -= self.threshold
                    if output_data < 0:
                        output_data = 0
                else:
                    if data < self.threshold:
                        output_data = 0
        elif t == np.ndarray:
            if self.bipolar:
                sign_ = np.sign(output_data)
                output_data = np.clip(np.abs(output_data) - self.threshold, 0, None)
                mask = output_data != 0
                if not self.squeeze:
                    output_data += self.threshold * mask
                output_data *= sign_
            else:
                output_data = np.clip(output_data - self.threshold, 0, None)
                mask = output_data > 0
                if not self.squeeze:
                    output_data = output_data + self.threshold * mask

        if output_data is not None:
            self.output.send(output_data)


class ThresholdTriggerNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ThresholdTriggerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.threshold = 0.1
        self.bipolar = False
        self.release_threshold = 0.1
        if label == 'hysteresis':
            self.threshold = 0.2
        self.retrigger_delay = 0
        self.last_trigger_time = time.time()
        self.state = False
        self.output_mode = 0
        self.previous_on = None
        self.previous_off = None

        if args is not None and len(args) > 0:
            self.threshold = any_to_float((args[0]))
            if len(args) > 1:
                self.release_threshold = any_to_float((args[1]))

        self.input = self.add_input("input", trigger_node=self)

        self.threshold_property = self.add_property('threshold', widget_type='drag_float', default_value=self.threshold)
        self.threshold_property.add_callback(self.option_changed)

        self.release_threshold_property = self.add_property('threshold', widget_type='drag_float', default_value=self.release_threshold)
        self.release_threshold_property.add_callback(self.option_changed)

        self.output = self.add_output("out")

        self.output_mode_option = self.add_option('trigger mode', widget_type='combo', default_value='output toggle', width=100)
        self.output_mode_option.add_callback(self.option_changed)
        self.output_mode_option.widget.combo_items = ['output toggle', 'output bang']

        self.retrigger_delay_option = self.add_option('retrig delay', widget_type='drag_float', default_value=self.retrigger_delay)
        self.retrigger_delay_option.add_callback(self.option_changed)

    def option_changed(self):
        self.threshold = self.threshold_property.get_widget_value()
        self.release_threshold = self.release_threshold_property.get_widget_value()
        self.retrigger_delay = self.retrigger_delay_option.get_widget_value()
        mode = self.output_mode_option.get_widget_value()
        if mode == 'output toggle':
            self.output_mode = 0
        else:
            self.output_mode = 1

    def execute(self):
        data = self.input.get_received_data()
        t = type(data)

        if t in [float, np.double, int, np.int64]:
            if self.state:
                if data < self.release_threshold:
                    self.state = False
                    if self.output_mode == 0:
                        self.output.send(0)
            else:
                if data > self.threshold:
                    now = time.time()
                    if now - self.last_trigger_time > self.retrigger_delay:
                        self.state = True
                        if self.output_mode == 0:
                            self.output.send(1)
                        else:
                            self.output.send('bang')
                        self.last_trigger_time = now
        elif t == np.ndarray:
            prev_state = self.state
            on = data > self.threshold
            not_off = data >= self.release_threshold
            if type(self.state) is not np.ndarray:
                self.state = on
            else:
                self.state = np.logical_or(self.state, on)
                self.state = np.logical_and(self.state, not_off)
            if np.any(self.state != prev_state):
                self.output.send(self.state)


class MultiDiffFilterNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MultiDiffFilterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.filter_count = 2
        self.degrees = np.array([0.7, 0.9])
        self.accums = np.array([0.0, 0.0])
        self.ones = np.array([1.0, 1.0])
        self.out_values = None

        if args is not None and len(args) > 0:
            self.filter_count = len(args)
            self.degrees.resize([self.filter_count])
            self.accums.resize([self.filter_count])
            self.ones.resize([self.filter_count])
            for index, degree_str in enumerate(args):
                degree = any_to_float(degree_str)
                if degree > 1.0:
                    degree = 1.0
                elif degree < 0:
                    degree = 0
                self.degrees[index] = degree

        self.ones.fill(1.0)
        self.accums.fill(0.0)
        self.minus_degrees = self.ones - self.degrees

        self.input = self.add_input('in', trigger_node=self)

        self.filter_degree_inputs = []
        for i in range(self.filter_count):
            input_ = self.add_input('filter ' + str(i), widget_type='drag_float', min=0.0, max=1.0, default_value=float(self.degrees[i]))
            input_.add_callback(self.degree_changed)
            self.filter_degree_inputs.append(input_)
        self.output = self.add_output('out')

    def degree_changed(self):
        for i in range(self.filter_count):
            self.degrees[i] = self.filter_degree_inputs[i].get_widget_value()
        self.minus_degrees = self.ones - self.degrees

    def execute(self):
        input_value = self.input.get_data()
        if type(input_value) == list:
            if type(input_value[0]) == str:
                if input_value[0] == 'set':
                    set_count = len(input_value) - 1
                    if set_count > self.filter_count:
                        set_count = self.filter_count
                    for i in range(set_count):
                        self.accums[i] = float(input_value[i + 1])
                elif input_value[0] == 'clear':
                    self.accums.fill(0)
        elif type(input_value) == str:
            if input_value == 'clear':
                self.accums.fill(0)
        else:
            self.accums = self.accums * self.degrees + input_value * self.minus_degrees
        out_values = self.accums[:-1] - self.accums[1:]
        self.output.set(out_values)
        self.send_outputs()


class FilterNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = FilterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.degree = 0.9
        self.accum = 0

        self.input = self.add_input('in', trigger_node=self)

        self.degree_input = self.add_input('degree', widget_type='drag_float', min=0.0, max=1.0)
        self.degree_input.add_callback(self.change_degree)
        self.degree_input.widget.speed = .01

        self.output = self.add_output("out")

    def change_degree(self):
        self.degree = self.degree_input.get_widget_value()
        if self.degree < 0:
            self.degree = 0
        elif self.degree > 1:
            self.degree = 1

    def custom(self):
        in_degree, t = decode_arg(self.args, 0)
        if t in [int, float]:
            self.degree = in_degree
            self.degree_input.widget.set(self.degree)

    def execute(self):
        input_value = self.input.get_data()
        self.accum = self.accum * self.degree + input_value * (1.0 - self.degree)
        self.output.set(self.accum)
        self.send_outputs()


class SampleHoldNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SampleHoldNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.sample_hold = True
        self.sample = 0
        self.sample_hold_input = self.add_input("sample/hold", widget_type='checkbox')
        self.input = self.add_input("input", trigger_node=self)
        self.output = self.add_output("out")

    def execute(self):
        self.sample_hold = self.sample_hold_input.get_widget_value()
        if self.sample_hold:
            self.sample = self.input.get_received_data()
        self.output.send(self.sample)


class TogEdgeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TogEdgeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.state = False
        self.input = self.add_input("", trigger_node=self)
        self.on_output = self.add_output("on")
        self.off_output = self.add_output("off")

    def execute(self):
        new_state = self.input.get_data() > 0
        if self.state:
            if not new_state:
                self.off_output.send('bang')
        elif new_state:
            if not self.state:
                self.on_output.send('bang')
        self.state = new_state

