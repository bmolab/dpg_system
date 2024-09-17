from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import math
import numpy as np
import torch.fft

from dpg_system.node import Node
from dpg_system.conversion_utils import *

import torch


from dpg_system.torch_base_nodes import *

# torch.scatter / torch.gather
# torch.cov
# torch.renorm
# torch.convxd (functional)


# thinking through model construction

from dpg_system.torch_activation_nodes import *
from dpg_system.torch_analysis_nodes import *
from dpg_system.torch_calculation_nodes import *
from dpg_system.torch_manipulation_nodes import *
from dpg_system.torch_generator_nodes import *
from dpg_system.torch_signal_processing_nodes import *
from dpg_system.torch_butterworth_nodes import *
from dpg_system.torchvision_nodes import *
from dpg_system.torch_kornia_nodes import *
from dpg_system.torch_loss_nodes import *

torchaudio_avail = True
try:
    import pyaudio
    from dpg_system.torchaudio_nodes import *
except Exception as e:
    print('pyaudio not found - torchaudio nodes not available')
    torchaudio_avail = False

from dpg_system.wavelet_nodes import *
# import wavelets_pytorch.transform

def register_torch_nodes():
    Node.app.torch_available = True
    Node.app.register_node('tensor', TensorNode.factory)
    Node.app.register_node('t.to', TensorNode.factory)
    Node.app.register_node('t.info', TorchInfoNode.factory)
    Node.app.register_node('t.numel', TorchNumElNode.factory)
    Node.app.register_node('t.contiguous', TorchContiguousNode.factory)
    Node.app.register_node('t.is_contiguous', TorchIsContiguousNode.factory)
    Node.app.register_node('t.detach', TorchDetachNode.factory)
    Node.app.register_node('t.buffer', TorchBufferNode.factory)
    Node.app.register_node('t.rolling_buffer', TorchRollingBufferNode.factory)

    register_torch_activation_nodes()
    register_torch_analyze_nodes()
    register_torch_calculation_nodes()
    register_torch_manipulation_nodes()
    register_torch_generator_nodes()
    register_torch_signal_processing_nodes()
    register_torch_butterworth_nodes()
    register_torchvision_nodes()
    register_kornia_nodes()
    register_torch_loss_nodes()
    if torchaudio_avail:
        register_torchaudio_nodes()
    register_wavelet_nodes()


class TensorNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TensorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('in', triggers_execution=True)
        self.setup_dtype_device_grad(args)
        self.output = self.add_output('tensor out')
        self.create_dtype_device_grad_properties()


    def execute(self):
        in_data = self.input_to_tensor()
        if in_data is not None:
            out_array = any_to_tensor(in_data, self.device, self.dtype, self.requires_grad)
            self.output.send(out_array)


class TorchNumElNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchNumElNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in',  triggers_execution=True)
        self.output = self.add_output('numel out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(torch.numel(input_tensor))


class TorchContiguousNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchContiguousNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('max index')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            output_tensor = input_tensor.contiguous()
            self.output.send(output_tensor)


class TorchIsContiguousNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchIsContiguousNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('result')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            contiguous = input_tensor.is_contiguous()
            self.output.send(contiguous)


class TorchDetachNode(TorchNode):

    @staticmethod
    def factory(name, data, args=None):
        node = TorchDetachNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            out_tensor = torch.detach(input_tensor)
            self.output.send(out_tensor)


t_TorchBufferFill = 0
t_TorchCircularBufferSerialInput = 1
t_TorchCircularBufferParallelInput = 2

t_TorchBufferOutputContents = 0
t_TorchBufferOutputValueAtIndex = 1


class TorchBufferNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchBufferNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.sample_count = self.arg_as_int(256)
        self.setup_dtype_device_grad(args)

        self.update_style = t_TorchCircularBufferSerialInput
        self.output_style = t_TorchBufferOutputValueAtIndex
        self.input = self.add_input('input', triggers_execution=True)
        self.index_input = self.add_input('sample to output', triggers_execution=True)
        self.output = self.add_output('output')
        self.sample_count_option = self.add_option('sample count', widget_type='drag_int', default_value=self.sample_count)
        self.update_style_option = self.add_option('update style', widget_type='combo', default_value='input is stream of samples', width=250, callback=self.update_style_changed)
        self.update_style_option.widget.combo_items = ['buffer holds one sample of input', 'input is stream of samples', 'input is multi-channel sample']
        self.output_style_option = self.add_option('output style', widget_type='combo', default_value='output samples on demand by index', width=250, callback=self.output_style_changed)
        self.output_style_option.widget.combo_items = ['output buffer on every input', 'output samples on demand by index']
        self.create_dtype_device_grad_properties()

        self.buffer = torch.zeros((self.sample_count), dtype=self.dtype, device=self.device)
        self.write_pos = 0

    def output_style_changed(self):
        output_style = self.output_style_option()
        if output_style == 'output buffer on every input':
            self.output_style = t_TorchBufferOutputContents
        elif output_style == 'output samples on demand by index':
            self.output_style = t_TorchBufferOutputValueAtIndex

    def update_style_changed(self):
        update_style = self.update_style_option()
        if update_style == 'buffer holds one sample of input':
            self.update_style = t_TorchBufferFill
        elif update_style == 'input is stream of samples':
            self.update_style = t_TorchCircularBufferSerialInput
        elif update_style == 'input is multi-channel sample':
            self.update_style = t_TorchCircularBufferParallelInput

    def device_changed(self):
        super().device_changed()
        self.buffer = self.buffer.to(self.device)

    def dtype_changed(self):
        super().dtype_changed()
        self.buffer = self.buffer.to(self.dtype)

    def requires_grad_changed(self):
        super().requires_grad_changed()
        self.buffer.requires_grad = self.requires_grad

    def execute(self):
        self.sample_count = self.sample_count_option()

        if self.input.fresh_input:
            data = self.input()
            t = type(data)
            if t != torch.tensor:
                data = any_to_tensor(data)
            if self.update_style == t_TorchBufferFill:
                self.buffer = data.detach().clone().to_device(self.device)
            elif self.update_style == t_TorchCircularBufferSerialInput:
                if self.sample_count != self.buffer.shape[0] or len(self.buffer.shape) > 1:
                    self.buffer.resize_(self.sample_count)
                    self.write_pos = 0
                if self.write_pos > self.sample_count:
                    self.write_pos = 0
                front_size = data.shape[0]
                back_size = 0
                if front_size + self.write_pos >= self.sample_count:
                    front_size = self.sample_count - self.write_pos
                    back_size = data.shape[0] - front_size
                if back_size > self.sample_count:
                    self.buffer[:] = data[-back_size:]
                    self.write_pos = 0
                else:
                    self.buffer[self.write_pos:self.write_pos + front_size] = data[:front_size]
                    if back_size > 0:
                        self.buffer[:back_size] = data[front_size:front_size + back_size]
                        self.write_pos = back_size
                    else:
                        self.write_pos += front_size
                if self.write_pos >= self.sample_count:
                    self.write_pos = 0
            elif self.update_style == t_TorchCircularBufferParallelInput:
                if len(self.buffer.shape) == 1 or self.buffer.shape[1] != data.shape[0]:
                    self.buffer.resize_((self.sample_count, data.shape[0]))
                    self.write_pos = 0
                self.buffer[self.write_pos, :] = data
                self.write_pos += 1
                if self.write_pos >= self.sample_count:
                    self.write_pos = 0
            if self.output_style == t_TorchBufferOutputContents:
                if self.write_pos != 0:
                    output_buffer = torch.cat((self.buffer[self.write_pos:], self.buffer[:self.write_pos]), 0)
                    self.output.send(output_buffer)
                    del output_buffer
                else:
                    self.output.send(self.buffer)

        if self.index_input.fresh_input:
            index = any_to_int(self.index_input())
            if 0 <= index < self.sample_count:
                output_sample = self.buffer[index].item()
                self.output.send(output_sample)


class TorchRollingBuffer:
    def __init__(self, shape, dtype, device):
        self.breadth = 1
        self.buffer_changed_callback = None
        self.owner = None
        self.dtype = dtype
        self.device = device
        if type(shape) == tuple:
            shape = list(shape)
        if type(shape) == list:
            self.sample_count = shape[0]
            self.breadth = shape[1]
            if len(shape) > 1 and shape[1] > 1:
                self.update_style = t_TorchCircularBufferParallelInput
            else:
                self.update_style = t_TorchCircularBufferSerialInput
        else:
            length = any_to_int(shape)
            self.sample_count = length
            self.update_style = t_TorchCircularBufferSerialInput
            self.breadth = 1

        self.in_get_buffer = False
        self.lock = threading.Lock()
        self.buffer = None
        self.allocate((self.sample_count, self.breadth))
        self.elapsed = 0

    def set_update_style(self, update_style):
        if update_style == 'buffer holds one sample of input':
            self.update_style = t_TorchBufferFill
        elif update_style == 'input is stream of samples':
            self.update_style = t_TorchCircularBufferSerialInput
        elif update_style == 'input is multi-channel sample':
            self.update_style = t_TorchCircularBufferParallelInput
        self.allocate((self.sample_count, self.breadth))

    def set_update_style_code(self, style_code):
        self.update_style = style_code
        self.allocate((self.sample_count, self.breadth))

    def get_value(self, x):
        if self.buffer.shape[1] > x >= 0:
            return self.buffer[x, 0].item()

    def set_value(self, x, value):
        if not self.lock.locked():
            if self.lock.acquire(blocking=False):
                if x < self.buffer.shape[1] and x >= 0:
                    self.buffer[x, 0] = value
                self.lock.release()

    def set_write_pos(self, pos):
        if not self.lock.locked():
            if self.lock.acquire(blocking=False):
                self.write_pos = pos
            self.lock.release()

    def update(self, incoming):
        if not self.lock.locked():
            if self.update_style == t_TorchBufferFill:
                if self.buffer.shape != incoming.shape:
                    if len(incoming.shape) == 1:
                        if self.buffer.shape != (incoming.shape[0], 1):
                            self.allocate((incoming.shape[0], 1))
                    else:
                        self.allocate(incoming.shape)
                if self.lock.acquire(blocking=False):
                    if len(incoming.shape) == 1:
                        self.buffer[:, 0] = incoming[:]
                    else:
                        self.buffer[:, :] = incoming[:, :]
                    self.lock.release()
                self.write_pos = 0

            elif self.update_style == t_TorchCircularBufferSerialInput:
                if self.sample_count * 2 != self.buffer.shape[0] or self.buffer.shape[1] != 1:
                    shape = (self.sample_count, 1)
                    self.allocate(shape)

                front_size = incoming.shape[0]
                back_size = 0

                if self.lock.acquire(blocking=False):
                    if self.write_pos > self.sample_count:
                        self.write_pos = 0

                    if front_size + self.write_pos >= self.sample_count:
                        front_size = self.sample_count - self.write_pos
                        back_size = incoming.shape[0] - front_size
                    if back_size > self.sample_count:
                        self.buffer[:self.sample_count, 0] = self.buffer[self.sample_count:, 0] = incoming[-back_size:]
                        self.write_pos = 0
                        self.lock.release()
                    else:
                        start = self.write_pos
                        end = self.write_pos + front_size
                        self.buffer[start:end, 0] = incoming[:front_size]
                        self.buffer[start + self.sample_count:end + self.sample_count, 0] = incoming[:front_size]
                        if back_size > 0:
                            self.buffer[:back_size, 0] = self.buffer[self.sample_count:back_size + self.sample_count, 0] = incoming[front_size:front_size + back_size]
                            self.write_pos = back_size
                        else:
                            self.write_pos += front_size

                        if self.write_pos >= self.sample_count:
                            self.write_pos = 0
                    self.lock.release()

            elif self.update_style == t_TorchCircularBufferParallelInput:
                if len(self.buffer.shape) == 1 or self.buffer.shape[1] != incoming.shape[0]:
                    shape = (self.sample_count, incoming.shape[0])
                    self.allocate(shape)
                if self.lock.acquire(blocking=False):
                    if len(incoming.shape) == 1:
                        self.buffer[self.write_pos, :] = self.buffer[self.write_pos + self.sample_count, :] = incoming[:]
                    else:
                        self.buffer[self.write_pos, :] = self.buffer[self.write_pos + self.sample_count, :] = incoming[0, :]

                    self.write_pos += 1
                    if self.write_pos >= self.sample_count:
                        self.write_pos = 0
                    self.lock.release()
            return True
        return False

    def get_buffer(self, block=False):
        if self.lock.acquire(blocking=block):
            self.in_get_buffer = True
            b = self.buffer[self.write_pos:self.write_pos + self.sample_count]
            return b
        return None

    def release_buffer(self):
        if self.lock.locked and self.in_get_buffer:
            self.in_get_buffer = False
            self.lock.release()

    def allocate(self, shape):
        self.lock.acquire(blocking=True)
        if self.update_style == t_TorchBufferFill:
            self.breadth = shape[1]
            self.sample_count = shape[0]
            self.buffer = torch.zeros(shape, dtype=self.dtype, device=self.device)
        elif self.update_style == t_TorchCircularBufferSerialInput:
            self.sample_count = shape[0]
            self.breadth = 1
            if len(shape) > 1:
                if shape[0] == 1:
                    self.sample_count = shape[1]
            self.buffer = torch.zeros((self.sample_count * 2, 1), dtype=self.dtype, device=self.device)
        elif self.update_style == t_TorchCircularBufferParallelInput:
            self.sample_count = shape[0]
            self.breadth = shape[1]
            self.buffer = torch.zeros((self.sample_count * 2, self.breadth), dtype=self.dtype, device=self.device)
        self.write_pos = 0
        if self.buffer_changed_callback is not None:
            self.buffer_changed_callback(self)
        self.lock.release()


class TorchRollingBufferNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchRollingBufferNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.sample_count = 256
        if self.ordered_args is not None and len(self.ordered_args) > 0:
            count, t = decode_arg(self.ordered_args, 0)
            if t == int:
                self.sample_count = count

        self.setup_dtype_device_grad(args)

        self.rolling_buffer = TorchRollingBuffer(self.sample_count, dtype=self.dtype, device=self.device)
        self.input = self.add_input('input', triggers_execution=True)
        self.output = self.add_output('output')
        self.sample_count_option = self.add_option('sample count', widget_type='drag_int', default_value=self.sample_count)
        self.update_style_option = self.add_option('update style', widget_type='combo', default_value='input is stream of samples', width=250, callback=self.update_style_changed)
        self.update_style_option.widget.combo_items = ['buffer holds one sample of input', 'input is stream of samples', 'input is multi-channel sample']
        self.create_dtype_device_grad_properties()

    def update_style_changed(self):
        update_style = self.update_style_option()
        self.rolling_buffer.set_update_style(update_style)

    def device_changed(self):
        super().device_changed()
        self.rolling_buffer.buffer = self.rolling_buffer.buffer.to(self.device)

    def dtype_changed(self):
        super().dtype_changed()
        self.rolling_buffer.buffer = self.rolling_buffer.buffer.to(self.dtype)

    def requires_grad_changed(self):
        super().requires_grad_changed()
        self.rolling_buffer.buffer.requires_grad = self.requires_grad

    def execute(self):
        self.rolling_buffer.sample_count = self.sample_count_option()

        if self.input.fresh_input:
            data = self.input()
            t = type(data)
            if t != torch.tensor:
                data = any_to_tensor(data)

            self.rolling_buffer.update(data)
            output_buffer = self.rolling_buffer.get_buffer()
            if output_buffer is not None:
                self.output.send(output_buffer)
                del output_buffer
                self.rolling_buffer.release_buffer()


