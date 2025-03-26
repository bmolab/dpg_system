import dearpygui.dearpygui as dpg
import math
import numpy as np
import random
import time
from scipy import signal
from dpg_system.node import Node
from dpg_system.conversion_utils import *


def register_signal_nodes():
    Node.app.register_node("filter", FilterNode.factory)
    Node.app.register_node("adaptive_filter", AdaptiveFilterNode.factory)
    Node.app.register_node("smooth", FilterNode.factory)
    Node.app.register_node("diff_filter_bank", MultiDiffFilterNode.factory)
    Node.app.register_node("diff_filter", MultiDiffFilterNode.factory)
    Node.app.register_node("multi_filter", MultiFilterNode.factory)
    Node.app.register_node("random", RandomNode.factory)
    Node.app.register_node("random.gauss", RandomGaussNode.factory)
    Node.app.register_node("random.normalvariate", RandomGaussNode.factory)
    Node.app.register_node("random.lognormvariate", RandomGaussNode.factory)
    Node.app.register_node("random.vonmisesvariate", RandomGaussNode.factory)
    Node.app.register_node("random.gammavariate", RandomGammaNode.factory)
    Node.app.register_node("random.betavariate", RandomGammaNode.factory)
    Node.app.register_node("random.weibullvariate", RandomGammaNode.factory)
    Node.app.register_node("random.triangular", RandomTriangularNode.factory)
    Node.app.register_node("random.paretovariate", RandomParetoNode.factory)
    Node.app.register_node("random.expovariate", RandomParetoNode.factory)
    Node.app.register_node("signal", SignalNode.factory)
    Node.app.register_node("togedge", TogEdgeNode.factory)
    Node.app.register_node("subsample", SubSampleNode.factory)
    Node.app.register_node("diff", DifferentiateNode.factory)
    Node.app.register_node('noise_gate', NoiseGateNode.factory)
    Node.app.register_node('trigger', ThresholdTriggerNode.factory)
    Node.app.register_node('hysteresis', ThresholdTriggerNode.factory)
    Node.app.register_node('sample_hold', SampleHoldNode.factory)
    Node.app.register_node('register', SampleAndTriggerNode.factory)
    Node.app.register_node('band_pass', BandPassFilterNode.factory)
    Node.app.register_node('filter_bank', FilterBankNode.factory)
    Node.app.register_node('spectrum', SpectrumNode.factory)
    Node.app.register_node('ranger', RangerNode.factory)


class DifferentiateNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DifferentiateNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.previous_value = None
        self.previousType = None

        self.input = self.add_input('', triggers_execution=True)
        self.absolute = self.add_input('absolute', widget_type='checkbox', default_value=False)
        self.output = self.add_output('')

    def float_diff(self, received, absolute):
        if float == self.previousType:
            output = received - self.previous_value
        else:
            prev = any_to_float(self.previous_value)
            output = received - prev
        if absolute:
            return abs(output)
        return output

    def array_diff(self, received, absolute):
        if np.ndarray == self.previousType:
            output = received - self.previous_value
        else:
            prev = any_to_array(self.previous_value)
            output = received - prev
        if absolute:
            return np.abs(output)
        return output

    def tensor_diff(self, received, absolute):
        if torch.Tensor == self.previousType:
            output = received - self.previous_value
        else:
            prev = any_to_tensor(self.previous_value)
            output = received - prev
        if absolute:
            return torch.abs(output)
        return output

    def list_diff(self, received, absolute):
        received = any_to_array(received)
        if np.ndarray == self.previousType:
            output = received - self.previous_value
        else:
            prev = any_to_array(self.previous_value)
            output = received - prev
        if absolute:
            return abs(output), received
        return output, received

    def int_diff(self, received, absolute):
        if int == self.previousType:
            output = received - self.previous_value
        else:
            prev = any_to_int(self.previous_value)
            output = received - prev
        if absolute:
            return abs(output)
        return output

    def bool_diff(self, received, absolute):
        if bool == self.previousType:
            output = received != self.previous_value
        else:
            prev = any_to_bool(self.previous_value)
            output = received != prev
        return output

    def execute(self):
        received = self.input()
        t = type(received)
        output = None
        if self.previous_value is not None:
            if t == float:
                output = self.float_diff(received, self.absolute())
            elif t == int:
                output = self.int_diff(received, self.absolute())
            elif t == bool:
                output = self.bool_diff(received, self.absolute())
            elif t == list:
                output, received = self.list_diff(received, self.absolute())
                t = np.ndarray
            if t == np.ndarray:
                output = self.array_diff(received, self.absolute())
            elif self.app.torch_available and t == torch.Tensor:
                output = self.tensor_diff(received, self.absolute())

            self.output.send(output)

        self.previous_value = received
        self.previousType = t


class RandomGaussNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RandomGaussNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        label_1 = 'mean'
        label_2 = 'dev'
        self.op = random.gauss
        if self.label == 'random.normalvariate':
            self.op = random.normalvariate
        elif self.label == 'random.lognormvariate':
            self.op = random.lognormvariate
        elif self.label == 'random.vonmisesvariate':
            self.op = random.vonmisesvariate
            label_1 = 'mu'
            label_2 = 'kappa'

        mean = 0.0
        dev = 1.0
        if len(args) > 0:
            mean = self.arg_as_number(default_value=0.0)
        if len(args) > 1:
            dev = self.arg_as_number(default_value=1.0)

        self.trigger_input = self.add_input('trigger', triggers_execution=True)
        self.mean = self.add_input(label_1, widget_type='drag_float', default_value=mean)
        self.dev = self.add_input(label_2, widget_type='drag_float', default_value=dev)
        self.mean.widget.speed = 0.01
        self.dev.widget.speed = 0.01
        self.output = self.add_output('out')

    def execute(self):
        output_value = self.op(any_to_float(self.mean()), any_to_float(self.dev()))
        self.output.send(output_value)


class RandomGammaNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RandomGammaNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        if self.label == 'random.gammavariate':
            self.op = random.gammavariate
        elif self.label == 'random.betavariate':
            self.op = random.betavariate
        elif self.label == 'random.weibullvariate':
            self.op = random.weibullvariate
        alpha = 1.0
        beta = 0.5
        if len(args) > 0:
            alpha = self.arg_as_number(default_value=alpha)
        if len(args) > 1:
            beta = self.arg_as_number(default_value=beta)

        self.trigger_input = self.add_input('trigger', triggers_execution=True)
        self.alpha = self.add_input('alpha', widget_type='drag_float', default_value=alpha)
        self.beta = self.add_input('beta', widget_type='drag_float', default_value=beta)
        self.alpha.widget.speed = 0.01
        self.beta.widget.speed = 0.01
        self.output = self.add_output('out')

    def execute(self):
        output_value = self.op(any_to_float(self.alpha()), any_to_float(self.beta()))
        self.output.send(output_value)


class RandomTriangularNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RandomTriangularNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        low = -1.0
        high = 1.0
        mode = 0.0
        if len(args) > 0:
            low = self.arg_as_number(default_value=-1.0)
        if len(args) > 1:
            high = self.arg_as_number(default_value=1.0)
        if len(args) > 2:
            mode = self.arg_as_number(default_value=0.0)

        self.trigger_input = self.add_input('trigger', triggers_execution=True)
        self.low = self.add_input('low', widget_type='drag_float', default_value=low)
        self.high = self.add_input('high', widget_type='drag_float', default_value=high)
        self.mode = self.add_input('mode', widget_type='drag_float', default_value=mode)
        self.low.widget.speed = 0.01
        self.high.widget.speed = 0.01
        self.mode.widget.speed = 0.01
        self.output = self.add_output('out')

    def execute(self):
        output_value = random.triangular(any_to_float(self.low()), any_to_float(self.high()), any_to_float(self.mode()))
        self.output.send(output_value)


class RandomParetoNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RandomParetoNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        alpha = 1.0
        if len(args) > 0:
            alpha = self.arg_as_number(default_value=1.0)
        self.op = random.paretovariate
        param_1_name = 'alpha'

        if self.label == 'random.expovariate':
            self.op = random.expovariate
            param_1_name = 'lambda'
        self.trigger_input = self.add_input('trigger', triggers_execution=True)
        self.alpha = self.add_input(param_1_name, widget_type='drag_float', default_value=alpha)
        self.alpha.widget.speed = 0.01
        self.output = self.add_output('out')

    def execute(self):
        output_value = self.op(any_to_float(self.alpha()))
        self.output.send(output_value)


class RandomNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RandomNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        range = self.arg_as_number(default_value=1.0)
        bipolar = False

        self.trigger_input = self.add_input('trigger', triggers_execution=True)
        self.range = self.add_input('range', widget_type='drag_float', default_value=range)
        self.range.widget.speed = 0.01
        self.bipolar = self.add_option('bipolar', widget_type='checkbox', default_value=bipolar)
        self.output = self.add_output('out')

    def execute(self):
        if self.bipolar():
            output_value = random.random() * self.range() * 2 - self.range()
        else:
            output_value = random.random() * self.range()
        self.output.send(output_value)


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

        if self.ordered_args is not None and len(self.ordered_args) > 0:
            for arg in self.ordered_args:
                if arg in ['sin', 'cos', 'saw', 'square', 'triangle', 'random']:
                    self.shape = arg
                elif float(arg) != 0:
                    self.period = float(arg)

        self.on_off_input = self.add_input('on', widget_type='checkbox', triggers_execution=True, callback=self.start_stop)
        self.period_input = self.add_input('period', widget_type='drag_float', default_value=self.period, callback=self.change_period)
        self.shape_input = self.add_input('shape', widget_type='combo', default_value=self.shape, callback=self.set_shape)
        self.shape_input.widget.combo_items = ['sin', 'cos', 'saw', 'square', 'triangle', 'random']
        self.range_property = self.add_option('range', widget_type='drag_float', default_value=self.range, callback=self.change_range)
        self.range_property.widget.speed = 0.01
        self.bipolar_property = self.add_option('bipolar', widget_type='checkbox', default_value=self.bipolar, callback=self.change_bipolar)
        self.size_property = self.add_option('vector size', widget_type='drag_int', min=1, default_value=self.vector_size, callback=self.change_size)
        self.output = self.add_output('')
        self.add_frame_task()

    def update_parameters_from_widgets(self):
        self.set_shape()
        self.change_period()
        self.change_range()
        self.change_bipolar()
        self.start_stop()

    def set_shape(self, input=None):
        self.shape = self.shape_input()

    def change_size(self):
        self.vector_size = self.size_property()
        if self.vector_size != 1:
            self.vector = np.ndarray((self.vector_size))

    def change_period(self, input=None):
        self.period = self.period_input()
        if self.period <= 0:
            self.period = .001

    def change_range(self):
        self.range = self.range_property()

    def change_bipolar(self):
        self.bipolar = self.bipolar_property()

    def start_stop(self, input=None):
        self.on = self.on_off_input()
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
        self.output.send(self.signal_value)


class SubSampleNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SubSampleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        subsampler = self.arg_as_int(default_value=2)
        self.sample_count = 0
        self.input = self.add_input('input', triggers_execution=True)
        self.rate = self.add_input('rate', widget_type='drag_int', default_value=subsampler, min=0, max=math.inf)
        self.output = self.add_output('out')
        self.output_period = 0
        self.last_time = time.time()
        self.add_frame_task()
        self.forced = False
        self.active = False

    def execute(self):
         if self.input.fresh_input:
            self.forced = False
            self.active = True
            now = time.time()
            elapsed = now - self.last_time
            self.last_time = now

            self.input.fresh_input = False
            self.sample_count += 1
            if self.sample_count >= self.rate():
                self.sample_count = 0
                self.output.send(self.input())

            self.output_period = self.output_period * 0.9 + elapsed * 0.1

    def frame_task(self):
        if self.active and not self.forced:
            now = time.time()
            elapsed = now - self.last_time
            if elapsed > self.output_period * self.rate():
                self.last_time = now
                self.output.send(self.input())
                self.forced = True
                self.active = False
                self.sample_count = 0


class NoiseGateNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NoiseGateNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.threshold = self.arg_as_float(default_value=0.1)
        self.bipolar = False
        self.squeeze = False

        self.input = self.add_input('input', triggers_execution=True)

        self.threshold_property = self.add_property('threshold', widget_type='drag_float', default_value=self.threshold, callback=self.option_changed)
        self.output = self.add_output('out')
        self.bipolar_option = self.add_option('bipolar', widget_type='checkbox', default_value=self.bipolar, callback=self.option_changed)
        self.squeeze_option = self.add_option('squeeze', widget_type='checkbox', default_value=self.bipolar, callback=self.option_changed)

    def option_changed(self):
        self.threshold = self.threshold_property()
        self.squeeze = self.squeeze_option()
        self.bipolar = self.bipolar_option()

    def execute(self):
        data = self.input()
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

        elif self.app.torch_available and t == torch.Tensor:
            if self.bipolar:
                sign_ = torch.sign(output_data)
                output_data = torch.clamp(torch.abs(output_data) - self.threshold, 0, None)
                mask = output_data != 0
                if not self.squeeze:
                    output_data += self.threshold * mask
                output_data *= sign_
            else:
                output_data = torch.clip(output_data - self.threshold, 0, None)
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

        self.threshold = self.arg_as_float(default_value=0.1)
        self.release_threshold = self.arg_as_float(index=1, default_value=0.1)
        if label == 'hysteresis':
            self.threshold = 0.2
        self.retrigger_delay = 0
        self.last_trigger_time = time.time()
        self.state = False
        self.output_mode = 0
        self.previous_on = None
        self.previous_off = None

        self.input = self.add_input('input', triggers_execution=True)
        self.threshold_property = self.add_property('threshold', widget_type='drag_float', default_value=self.threshold, callback=self.option_changed)
        self.release_threshold_property = self.add_property('threshold', widget_type='drag_float', default_value=self.release_threshold, callback=self.option_changed)
        self.output = self.add_output('out')
        self.release_output = self.add_output('release')
        self.output_mode_option = self.add_option('trigger mode', widget_type='combo', default_value='output toggle', width=100, callback=self.option_changed)
        self.output_mode_option.widget.combo_items = ['output toggle', 'output bang']
        self.retrigger_delay_option = self.add_option('retrig delay', widget_type='drag_float', default_value=self.retrigger_delay, callback=self.option_changed)

    def option_changed(self):
        self.threshold = self.threshold_property()
        self.release_threshold = self.release_threshold_property()
        self.retrigger_delay = self.retrigger_delay_option()
        mode = self.output_mode_option()
        if mode == 'output toggle':
            self.output_mode = 0
        else:
            self.output_mode = 1

    def execute(self):
        data = self.input()
        t = type(data)

        if t in [float, np.double, int, np.int64]:
            if self.state:
                if data < self.release_threshold:
                    self.state = False
                    if self.output_mode == 0:
                        self.output.send(0)
                        self.release_output.send(1)
                    else:
                        self.release_output.send('bang')
            else:
                if data > self.threshold:
                    now = time.time()
                    if now - self.last_trigger_time > self.retrigger_delay:
                        self.state = True
                        if self.output_mode == 0:
                            self.output.send(1)
                            self.release_output.send(0)
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
                if self.output_mode == 0:
                    self.output.send(self.state)
                    self.release_output.send(not self.state)
                else:
                    if self.state:
                        self.output.send('bang')
                    else:
                        self.release_output.send('bang')

        elif self.app.torch_available and t == torch.Tensor:
            prev_state = self.state
            on = data > self.threshold
            not_off = data >= self.release_threshold
            if type(self.state) is not torch.Tensor:
                self.state = on
            else:
                self.state = torch.logical_or(self.state, on)
                self.state = torch.logical_and(self.state, not_off)
            if torch.any(self.state != prev_state):
                if self.output_mode == 0:
                    self.output.send(self.state)
                    self.release_output.send(torch.logical_not(self.state))
                else:
                    if self.state:
                        self.output.send('bang')
                    else:
                        self.release_output.send('bang')


class RangerNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = RangerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.inMin = 0.0
        self.inMax = 1.0
        self.outMin = 0.0
        self.outMax = 1.0
        self.clamp = True
        self.calibrating_min = math.inf
        self.calibrating_max = -math.inf
        self.calibrating = False
        self.was_calibrating = False
        if len(args) > 0:
            self.inMin = any_to_float(args[0])
        if len(args) > 1:
            self.inMax = any_to_float(args[1])
        if len(args) > 2:
            self.outMin = any_to_float(args[2])
        if len(args) > 3:
            self.outMax = any_to_float(args[3])

        self.input = self.add_input('in', triggers_execution=True)
        self.in_min_input = self.add_input('input_min', widget_type='drag_float', default_value=self.inMin)
        self.in_max_input = self.add_input('input_max', widget_type='drag_float', default_value=self.inMax)
        self.out_min_input = self.add_input('output_min', widget_type='drag_float', default_value=self.outMin)
        self.out_max_input = self.add_input('output_max', widget_type='drag_float', default_value=self.outMax)
        self.clamp_input = self.add_input('clamp', widget_type='checkbox', default_value=True)

        self.calibrate_input = self.add_input('calibrate', widget_type='checkbox', default_value=self.calibrating)
        self.output = self.add_output('rescaled')

    def execute(self):
        self.calibrating = self.calibrate_input()
        if self.calibrating != self.was_calibrating:
            if self.calibrating:
                self.calibrating_min = math.inf
                self.calibrating_max = -math.inf
                self.was_calibrating = True
            else:
                self.inMin = self.calibrating_min
                self.in_min_input.set(self.inMin)
                self.inMax = self.calibrating_max
                self.in_max_input.set(self.inMax)
                self.was_calibrating = False
        out = 0.0
        if self.input.fresh_input:
            inData = self.input()

            t = type(inData)
            if t in [int, float]:
                in_value = any_to_float(inData)
                if self.calibrating:
                    if in_value > self.calibrating_max:
                        self.calibrating_max = in_value
                    elif in_value < self.calibrating_min:
                        self.calibrating_min = in_value
                self.inMin = self.in_min_input()
                self.inMax = self.in_max_input()
                self.outMin = self.out_min_input()
                self.outMax = self.out_max_input()

                range = self.inMax - self.inMin
                if range == 0:
                    range = 1e-5
                out = (in_value - self.inMin) / range
                out = (self.outMax - self.outMin) * out + self.outMin
                if self.clamp_input():
                    if out > self.outMax:
                        out = self.outMax
                    elif out < self.outMin:
                        out = self.outMin
                self.output.send(out)
            elif t == list:
                inData = list_to_array(inData, validate=True)
                if inData is None:
                    return
                t = np.ndarray
            if t == np.ndarray:
                if self.calibrating:
                    min = inData.min()
                    max = inData.max()
                    if max > self.calibrating_max:
                        self.calibrating_max = max
                    if min < self.calibrating_min:
                        self.calibrating_min = min
                range = self.inMax - self.inMin
                if range == 0:
                    range = 1e-5
                out = (in_value - self.inMin) / range
                out = (self.outMax - self.outMin) * out + self.outMin
                if self.clamp_input():
                    out = out.clip(self.outMin, self.outMax)
                self.output.send(out)
            if self.app.torch_available and t == torch.Tensor:
                if self.calibrating:
                    min = inData.min()
                    max = inData.max()
                    if max > self.calibrating_max:
                        self.calibrating_max = max
                    if min < self.calibrating_min:
                        self.calibrating_min = min
                range = self.inMax - self.inMin
                if range == 0:
                    range = 1e-5
                out = (in_value - self.inMin) / range
                out = (self.outMax - self.outMin) * out + self.outMin
                if self.clamp_input():
                    out = out.clamp(self.outMin, self.outMax)
                self.output.send(out)


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
            self.accums = np.zeros([self.filter_count])
            self.ones = np.ones([self.filter_count])
            for index, degree_str in enumerate(args):
                degree = any_to_float(degree_str)
                if degree > 1.0:
                    degree = 1.0
                elif degree < 0.0:
                    degree = 0.0
                self.degrees[index] = degree
        self.minus_degrees = self.ones - self.degrees
        self.input = self.add_input('in', triggers_execution=True)
        self.filter_degree_inputs = []
        for i in range(self.filter_count):
            input_ = self.add_input('filter ' + str(i), widget_type='drag_float', min=0.0, max=1.0, default_value=float(self.degrees[i]), callback=self.degree_changed)
            self.filter_degree_inputs.append(input_)
        self.output = self.add_output('out')
        self.message_handlers['set'] = self.set
        self.message_handlers['clear'] = self.clear

    def degree_changed(self):
        for i in range(self.filter_count):
            self.degrees[i] = self.filter_degree_inputs[i]()
        self.minus_degrees = self.ones - self.degrees

    def execute(self):
        input_value = self.input.get_data()
        # handled, do_output = self.check_for_messages(input_value)
        # if not handled:
        self.accums = self.accums * self.degrees + input_value * self.minus_degrees
        out_values = self.accums[:-1] - self.accums[1:]
        self.output.send(out_values)

    def set(self, message, args):
        set_count = len(args)
        if set_count > self.filter_count:
            set_count = self.filter_count
        for i in range(set_count):
            self.accums[i] = float(args[i])

    def clear(self, message, args):
        self.accums.fill(0)


class MultiFilterNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MultiFilterNode(name, data, args)
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
            self.accums = np.zeros([self.filter_count])
            self.ones = np.ones([self.filter_count])
            for index, degree_str in enumerate(args):
                degree = any_to_float(degree_str)
                if degree > 1.0:
                    degree = 1.0
                elif degree < 0.0:
                    degree = 0.0
                self.degrees[index] = degree
        self.minus_degrees = self.ones - self.degrees
        self.input = self.add_input('in', triggers_execution=True)
        self.filter_degree_inputs = []
        for i in range(self.filter_count):
            input_ = self.add_input('filter ' + str(i), widget_type='drag_float', min=0.0, max=1.0, default_value=float(self.degrees[i]), callback=self.degree_changed)
            self.filter_degree_inputs.append(input_)
        self.output = self.add_output('out')
        self.message_handlers['set'] = self.set
        self.message_handlers['clear'] = self.clear

    def degree_changed(self):
        for i in range(self.filter_count):
            self.degrees[i] = self.filter_degree_inputs[i]()
        self.minus_degrees = self.ones - self.degrees

    def execute(self):
        input_value = self.input.get_data()
        # handled, do_output = self.check_for_messages(input_value)
        # if not handled:
        self.accums = self.accums * self.degrees + input_value * self.minus_degrees
        self.output.send(self.accums)

    def set(self, message, args):
        set_count = len(args)
        if set_count > self.filter_count:
            set_count = self.filter_count
        for i in range(set_count):
            self.accums[i] = float(args[i])

    def clear(self, message, args):
        self.accums.fill(0)


class FilterNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = FilterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.degree = self.arg_as_float(default_value=0.9)
        if self.degree < 0.0:
            self.degree = 0.0
        elif self.degree > 1.0:
            self.degree = 1.0

        self.accum = 0.0

        self.input = self.add_input('in', triggers_execution=True)
        self.degree_input = self.add_input('degree', widget_type='drag_float', min=0.0, max=1.0, default_value=self.degree, callback=self.change_degree)
        self.degree_input.widget.speed = .01
        self.output = self.add_output('out')

    def change_degree(self, input=None):
        self.degree = self.degree_input()
        if self.degree < 0:
            self.degree = 0
        elif self.degree > 1:
            self.degree = 1

    def execute(self):
        input_value = self.input.get_data()
        if type(self.accum) != type(input_value):
            self.accum = any_to_match(self.accum, input_value)
        elif self.app.torch_available and type(input_value) == torch.Tensor:
            if input_value.device != self.accum.device:
                self.accum = any_to_match(self.accum, input_value)

        self.accum = self.accum * self.degree + input_value * (1.0 - self.degree)
        self.output.send(self.accum)


class AdaptiveFilterNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = AdaptiveFilterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.power = self.arg_as_float(default_value=2.0)
        if self.power < 0.0:
            self.power = 0.0
        self.degree = 0.9
        self.accum = 0.0
        self.offset_accum = 0.0

        self.input = self.add_input('in', triggers_execution=True)
        self.power_input = self.add_input('power', widget_type='drag_float', min=0.0, default_value=self.power, callback=self.change_power)
        self.power_input.widget.speed = .01
        self.base_degree = self.add_input('responsiveness', widget_type='drag_float', min=0.0, max=1.0, default_value=1.0)
        self.base_degree.widget.speed = .01
        self.base_degree.widget.speed = .01
        self.range = self.add_input('signal range', widget_type='drag_float', min=0.0001, default_value=1.0)
        self.adaption_smoothing = self.add_input('smooth response', widget_type='drag_float', min=0.0, default_value=0.0)
        self.offset_smoothing = self.add_input('offset response', widget_type='drag_float', min=0.0, default_value=0.9)
        self.output = self.add_output('out')

    def change_power(self):
        self.power = self.power_input()
        if self.power < 0:
            self.power = 0

    def execute(self):
        input_value = self.input.get_data()
        t = type(input_value)

        if type(self.accum) != t:
            self.accum = any_to_match(self.accum, input_value)
        elif self.app.torch_available and t == torch.Tensor:
            if input_value.device != self.accum.device:
                self.accum = any_to_match(self.accum, input_value)

        if self.app.torch_available and t == torch.Tensor:
            offset = (input_value - self.accum) / self.range()
            offset = offset.sum().item()
        elif t == np.ndarray:
            offset = (input_value - self.accum) / self.range()
            offset = offset.sum().item()
        else:
            offset = (input_value - self.accum) / self.range()

        offset_smooth = self.offset_smoothing()
        self.offset_accum = self.offset_accum * offset_smooth + offset * (1.0 - offset_smooth)

        degree = pow(abs(offset), self.power)
        if degree < 0:
            degree = 0
        elif degree > self.base_degree():
            degree = self.base_degree()
        adapt_smooth = self.adaption_smoothing()
        self.degree = self.degree * adapt_smooth + degree * (1.0 - adapt_smooth)

        self.accum = self.accum * (1.0 - self.degree) + input_value * self.degree + self.offset_accum

        self.output.send(self.accum)


class SampleHoldNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SampleHoldNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.sample_hold = True
        self.sample = 0
        self.sample_hold_input = self.add_input('sample/hold', widget_type='checkbox')
        self.input = self.add_input('input', triggers_execution=True)
        self.output = self.add_output('out')

    def execute(self):
        self.sample_hold = self.sample_hold_input()
        if self.sample_hold:
            self.sample = self.input()
        self.output.send(self.sample)


class SampleAndTriggerNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SampleAndTriggerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.register_value = 0
        self.sample_input = self.add_input('sample', callback=self.sample)
        self.trigger_input = self.add_input('trigger', callback=self.trigger)
        self.input = self.add_input('input')
        self.output = self.add_output('out')

    def trigger(self):
        sample = self.input()
        if sample is not None:
            self.register_value = sample
            self.execute()

    def sample(self):
        sample = self.input()
        if sample is not None:
            self.register_value = sample

    def execute(self):
        self.output.send(self.register_value)

class TogEdgeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TogEdgeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.state = False
        self.input = self.add_input('', triggers_execution=True)
        self.on_output = self.add_output('on')
        self.off_output = self.add_output('off')

    def execute(self):
        new_state = self.input.get_data() > 0
        if self.state:
            if not new_state:
                self.off_output.send('bang')
        elif new_state:
            if not self.state:
                self.on_output.send('bang')
        self.state = new_state


# class BandpassNode(Node):
#     @staticmethod
#     def factory(name, data, args=None):
#         node = BandpassNode(name, data, args)
#         return node
#
#     def __init__(self, label: str, data, args):
#         super().__init__(label, data, args)
#
#         self.low_cut = 10
#         self.high_cut = 20
#         self.sample_frequency = 60
#         self.nyquist = self.sample_frequency * 0.5
#         self.low = self.low_cut / self.nyquist
#         self.high = self.high_cut / self.nyquist
#         self.order = 5
#
#         self.input = self.add_input("", triggers_execution=True)
#         self.order_property = self.add_property('order', widget_type='input_int', default_value=self.order, min=1, max=8, callback=self.params_changed)
#         self.low_cut_property = self.add_property('low', widget_type='drag_float', default_value=self.low_cut, callback=self.params_changed)
#         self.high_cut_property = self.add_property('high', widget_type='drag_float', default_value=self.high_cut, callback=self.params_changed)
#
#         self.on_output = self.add_output("on")
#         self.off_output = self.add_output("off")
#         self.sos = signal.butter(self.order, [self.low, self.high], btype='band', output='sos')
#
#     def params_changed(self):
#         self.low_cut = self.low_cut_property()
#         self.high_cut = self.high_cut_property()
#         self.low = self.low_cut / self.nyquist
#         self.high = self.high_cut / self.nyquist
#         self.order = self.order_property()
#         self.sos = signal.butter(self.order, [self.low, self.high], btype='band', output='sos')
#
#     def execute(self):
#         signal = self.input()

# envelope:
# square of the signal + square of (diff of the signal * period * 9.5)

# filter bank
# calc high and low cut for n bands across frequency range in octaves

class FilterBankNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = FilterBankNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.order = 5
        self.low_bound = 1
        self.high_bound = 20
        self.sample_frequency = 60
        self.filter_type = 'bandpass'
        self.filter_design = 'butter'
        self.nyquist = self.sample_frequency * 0.5
        self.number_of_bands = 8
        self.ready = False
        self.bands = np.logspace(np.log10(self.low_bound), np.log10(self.high_bound), self.number_of_bands + 1)
        self.centers = []
        for i in range(self.number_of_bands):
            self.centers.append((self.bands[i] + self.bands[i + 1]) / 2)

        self.input = self.add_input('signal', triggers_execution=True)
        self.number_of_bands_property = self.add_property('band count', widget_type='input_int', default_value=self.number_of_bands, callback=self.params_changed)
        # self.filter_type_property = self.add_property('filter type', widget_type='combo', default_value=self.filter_type, callback=self.params_changed)
        # self.filter_type_property.widget.combo_items = ['bandpass', 'lowpass', 'highpass', 'bandstop']
        self.filter_design_property = self.add_property('filter design', widget_type='combo', default_value=self.filter_design, callback=self.params_changed)
        self.filter_design_property.widget.combo_items = ['butter', 'cheby1', 'cheby2']
        self.order_property = self.add_property('order', widget_type='input_int', default_value=self.order, min=1, max=8, callback=self.params_changed)
        self.low_cut_property = self.add_property('low', widget_type='drag_float', default_value=self.low_bound, callback=self.params_changed)
        self.high_cut_property = self.add_property('high', widget_type='drag_float', default_value=self.high_bound, callback=self.params_changed)
        self.sample_frequency_property = self.add_property('sample freq', widget_type='drag_float', default_value=self.sample_frequency, callback=self.params_changed)

        self.output = self.add_output('filtered')

        self.filters = []
        for i in range(self.number_of_bands):
            filter = IIR2Filter(self.order, [self.bands[i], self.bands[i + 1]], filter_type=self.filter_type, design=self.filter_design, fs=self.sample_frequency)
            self.filters.append(filter)
        self.signal_out = np.zeros((self.number_of_bands))
        self.ready = True

    def params_changed(self):
        self.ready = False
        self.low_bound = self.low_cut_property()
        self.high_bound = self.high_cut_property()
        self.order = self.order_property()
        self.sample_frequency = self.sample_frequency_property()
        self.nyquist = self.sample_frequency * 0.5
        # self.filter_type = self.filter_type_property()
        self.filter_design = self.filter_design_property()

        if self.high_bound > self.nyquist:
            self.high_bound = self.nyquist - 1
        if self.low_bound > self.high_bound:
            self.low_bound = self.high_bound * .5

        self.bands = np.logspace(np.log10(self.low_bound), np.log10(self.high_bound), self.number_of_bands + 1)
        self.centers = []
        for i in range(self.number_of_bands):
            self.centers.append((self.bands[i] + self.bands[i + 1]) / 2)
        # print(self.bands)
        # print(self.centers)
        self.filters = []
        for i in range(self.number_of_bands):
            filter = IIR2Filter(self.order, [self.bands[i], self.bands[i + 1]], filter_type=self.filter_type, design=self.filter_design, fs=self.sample_frequency)
            self.filters.append(filter)
        self.signal_out = np.zeros((self.number_of_bands))
        self.ready = True

    def execute(self):
        signal = self.input()

        if self.ready:
            for i, filter in enumerate(self.filters):
                self.signal_out[i] = filter.filter(signal)
        self.output.send(self.signal_out)


class SpectrumNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = SpectrumNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.order = 5
        self.low_bound = 1
        self.high_bound = 20
        self.sample_frequency = 60
        self.filter_type = 'bandpass'
        self.filter_design = 'butter'
        self.nyquist = self.sample_frequency * 0.5
        self.number_of_bands = 8
        self.ready = False
        self.bands = np.logspace(np.log10(self.low_bound), np.log10(self.high_bound), self.number_of_bands + 1)
        self.centers = []
        self.gain = []
        for i in range(self.number_of_bands):
            self.centers.append((self.bands[i] + self.bands[i + 1]) / 2)
            self.gain.append(9.6 / self.centers[i])
        self.gain = np.array(self.gain)

        self.input = self.add_input('signal', triggers_execution=True)
        self.number_of_bands_property = self.add_property('band count', widget_type='input_int', default_value=self.number_of_bands, callback=self.params_changed)
        # self.filter_type_property = self.add_property('filter type', widget_type='combo', default_value=self.filter_type, callback=self.params_changed)
        # self.filter_type_property.widget.combo_items = ['bandpass', 'lowpass', 'highpass', 'bandstop']
        self.filter_design_property = self.add_property('filter design', widget_type='combo', default_value=self.filter_design, callback=self.params_changed)
        self.filter_design_property.widget.combo_items = ['butter', 'cheby1', 'cheby2']
        self.order_property = self.add_property('order', widget_type='input_int', default_value=self.order, min=1, max=8, callback=self.params_changed)
        self.low_cut_property = self.add_property('low', widget_type='drag_float', default_value=self.low_bound, callback=self.params_changed)
        self.high_cut_property = self.add_property('high', widget_type='drag_float', default_value=self.high_bound, callback=self.params_changed)
        self.sample_frequency_property = self.add_property('sample freq', widget_type='drag_float', default_value=self.sample_frequency, callback=self.params_changed)

        self.output = self.add_output('spectrum')

        self.filters = []
        for i in range(self.number_of_bands):
            filter = IIR2Filter(self.order, [self.bands[i], self.bands[i + 1]], filter_type=self.filter_type, design=self.filter_design, fs=self.sample_frequency)
            self.filters.append(filter)
        self.signal_out = np.zeros((self.number_of_bands))
        self.previous_signal = np.zeros((self.number_of_bands))
        self.ready = True

    def params_changed(self):
        self.ready = False
        self.low_bound = self.low_cut_property()
        self.high_bound = self.high_cut_property()
        self.order = self.order_property()
        self.sample_frequency = self.sample_frequency_property()
        self.nyquist = self.sample_frequency * 0.5
        # self.filter_type = self.filter_type_property()
        self.filter_design = self.filter_design_property()
        self.number_of_bands = self.number_of_bands_property()

        if self.high_bound > self.nyquist:
            self.high_bound = self.nyquist - 1
        if self.low_bound > self.high_bound:
            self.low_bound = self.high_bound * .5

        self.bands = np.logspace(np.log10(self.low_bound), np.log10(self.high_bound), self.number_of_bands + 1)
        self.centers = []
        self.gain = []
        for i in range(self.number_of_bands):
            self.centers.append((self.bands[i] + self.bands[i + 1]) / 2)
            self.gain.append(9.6 / self.centers[i])
        self.gain = np.array(self.gain)
        # print(self.bands)
        # print(self.centers)
        self.filters = []
        for i in range(self.number_of_bands):
            filter = IIR2Filter(self.order, [self.bands[i], self.bands[i + 1]], filter_type=self.filter_type, design=self.filter_design, fs=self.sample_frequency)
            self.filters.append(filter)
        self.signal_out = np.zeros((self.number_of_bands))
        self.previous_signal = np.zeros((self.number_of_bands))
        self.ready = True

    def execute(self):
        signal = self.input()

        if self.ready:
            for i, filter in enumerate(self.filters):
                self.signal_out[i] = filter.filter(signal)
            slur = (self.signal_out + self.previous_signal) / 2
            diff = (self.signal_out - self.previous_signal) * self.gain
            output_signal = slur * slur + diff * diff
            self.previous_signal = self.signal_out.copy()
            self.output.send(output_signal)


class BandPassFilterNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = BandPassFilterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.order = 5
        self.low_cut = 10
        self.high_cut = 20
        self.sample_frequency = 60
        self.filter_type = 'bandpass'
        self.filter_design = 'butter'
        self.nyquist = self.sample_frequency * 0.5

        self.input = self.add_input("signal", triggers_execution=True)
        self.filter_type_property = self.add_property('filter type', widget_type='combo', default_value=self.filter_type, callback=self.params_changed)
        self.filter_type_property.widget.combo_items = ['bandpass', 'lowpass', 'highpass', 'bandstop']
        self.filter_design_property = self.add_property('filter design', widget_type='combo', default_value=self.filter_design, callback=self.params_changed)
        self.filter_design_property.widget.combo_items = ['butter', 'cheby1', 'cheby2']
        self.order_property = self.add_property('order', widget_type='input_int', default_value=self.order, min=1, max=8, callback=self.params_changed)
        self.low_cut_property = self.add_property('low', widget_type='drag_float', default_value=self.low_cut, callback=self.params_changed)
        self.high_cut_property = self.add_property('high', widget_type='drag_float', default_value=self.high_cut, callback=self.params_changed)
        self.sample_frequency_property = self.add_property('sample freq', widget_type='drag_float', default_value=self.sample_frequency, callback=self.params_changed)

        self.output = self.add_output('filtered')
        self.filter = IIR2Filter(self.order, [self.low_cut, self.high_cut], filter_type=self.filter_type, design=self.filter_design, fs=self.sample_frequency)

    def params_changed(self):
        self.filter = None
        self.low_cut = self.low_cut_property()
        self.high_cut = self.high_cut_property()
        self.order = self.order_property()
        self.sample_frequency = self.sample_frequency_property()
        self.nyquist = self.sample_frequency * 0.5
        self.filter_type = self.filter_type_property()
        self.filter_design = self.filter_design_property()
        if self.high_cut > self.nyquist:
            self.high_cut = self.nyquist - 1
        if self.low_cut > self.high_cut:
            self.low_cut = self.high_cut * .5
        if self.filter_type in ['bandpass', 'bandstop']:
            self.filter = IIR2Filter(self.order, [self.low_cut, self.high_cut], filter_type=self.filter_type, design=self.filter_design, fs=self.sample_frequency)
        elif self.filter_type == 'lowpass':
            self.filter = IIR2Filter(self.order, [self.high_cut], filter_type=self.filter_type, design=self.filter_design, fs=self.sample_frequency)
        elif self.filter_type == 'highpass':
            self.filter = IIR2Filter(self.order, [self.low_cut], filter_type=self.filter_type, design=self.filter_design, fs=self.sample_frequency)

    def execute(self):
        signal = self.input()
        if self.filter is not None:
            signal_out = self.filter.filter(signal)
            self.output.send(signal_out)


class IIR2Filter():
    def __init__(self, order, cutoff, filter_type, design='butter', rp=1, rs=1, fs=0):
        self.designs = ['butter', 'cheby1', 'cheby2']
        self.filter_types_1 = ['lowpass', 'highpass', 'Lowpass', 'Highpass', 'low', 'high']
        self.filter_types_2 = ['bandstop', 'bandpass', 'Bandstop', 'Bandpass']
        self.error_flag = 0
        self.fir_coefficients = None
        self.coefficients = None
        self.coefficients = self.create_coefficients(order, cutoff, filter_type, design, rp, rs, fs)
        self.acc_input = np.zeros(len(self.coefficients))
        self.acc_output = np.zeros(len(self.coefficients))
        self.buffer1 = np.zeros(len(self.coefficients))
        self.buffer2 = np.zeros(len(self.coefficients))
        self.input = 0
        self.output = 0

    def filter(self, input):
        # len(coefficients[0,:] == 1 means that there was an error in the generation of the coefficients
        # and the filtering should not be used

        if len(self.coefficients[0, :]) > 1:
            self.input = input
            self.output = 0

            # The for loop creates a chain of second order filters according to the order desired.
            # If a 10th order filter is to be created the loop will iterate 5 times to create a chain of
            # 5 second order filters.
            
            for i in range(len(self.coefficients)):
                self.fir_coefficients = self.coefficients[i][0:3]
                self.iir_coefficients = self.coefficients[i][3:6]

                # Calculating the accumulated input consisting of the input and the values coming from
                # the feedback loops (delay buffers weighed by the IIR coefficients).
                
                self.acc_input[i] = (self.input + self.buffer1[i] * -self.iir_coefficients[1] + self.buffer2[i] * -self.iir_coefficients[2])

                # Calculating the accumulated output provided by the accumulated input and the values from the delay
                # buffers weighed by the FIR coefficients.
                
                self.acc_output[i] = (self.acc_input[i] * self.fir_coefficients[0] + self.buffer1[i] * self.fir_coefficients[1] + self.buffer2[i] * self.fir_coefficients[2])

                # Shifting the values on the delay line: acc_input -> buffer1 -> buffer2
                
                self.buffer2[i] = self.buffer1[i]
                self.buffer1[i] = self.acc_input[i]

                self.input = self.acc_output[i]

            self.output = self.acc_output[-1]  # was i
        return self.output

    def create_coefficients(self, order, cutoff, filter_type, design='butter', rp=1, rs=1, fs=0):
        # Error handling: other errors can arise too, but those are dealt with in the signal package.

        self.error_flag = 1  # if there was no error then it will be set to 0
        self.coefficients = [0]  # with no error this will hold the coefficients

        if design not in self.designs:
            print('Gave wrong filter design! Remember: butter, cheby1, cheby2.')
        elif filter_type not in self.filter_types_1 and filter_type not in self.filter_types_2:
            print('Gave wrong filter type! Remember: lowpass, highpass, bandpass, bandstop.')
        elif fs < 0:
            print('The sampling frequency has to be positive!')
        else:
            self.error_flag = 0

        # if fs was given then the given cutoffs need to be normalised to Nyquist
        if fs and self.error_flag == 0:
            for i in range(len(cutoff)):
                cutoff[i] = cutoff[i] / fs * 2

        if design == 'butter' and self.error_flag == 0:
            self.coefficients = signal.butter(order, cutoff, filter_type, output='sos')
        elif design == 'cheby1' and self.error_flag == 0:
            self.coefficients = signal.cheby1(order, rp, cutoff, filter_type, output='sos')
        elif design == 'cheby2' and self.error_flag == 0:
            self.coefficients = signal.cheby2(order, rs, cutoff, filter_type, output='sos')

        return self.coefficients

