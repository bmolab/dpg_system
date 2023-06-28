from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from dpg_system.torchvision_nodes import *
from dpg_system.torch_kornia_nodes import *
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

    register_torch_activation_nodes()
    register_torch_analyze_nodes()
    register_torch_calculation_nodes()
    register_torch_manipulation_nodes()
    register_torch_generator_nodes()
    register_torch_signal_processing_nodes()
    register_torchvision_nodes()
    register_kornia_nodes()
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
        self.input = self.add_input("in", triggers_execution=True)
        self.setup_dtype_device_grad(args)
        self.output = self.add_output('tensor out')

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
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.output = self.add_output("max index")

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
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.output = self.add_output("result")

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





