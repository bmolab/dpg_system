import dearpygui.dearpygui as dpg
import math
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *

import torch
import torchvision

# torch.fft
# t.count_non_zero

def register_torch_nodes():
    Node.app.torch_available = True
    Node.app.register_node('tensor', TensorNode.factory)
    Node.app.register_node('t.to', TensorNode.factory)
    Node.app.register_node('torch[]', TorchSubtensorNode.factory)

    Node.app.register_node('t.cdist', TorchCDistanceNode.factory)
    Node.app.register_node('t.dist', TorchDistanceNode.factory)
    Node.app.register_node('t.length', TorchDistanceNode.factory)
    Node.app.register_node('t.cosine_similarity', CosineSimilarityNode.factory)

    Node.app.register_node('t.rand', TorchGeneratorNode.factory)
    Node.app.register_node('t.ones', TorchGeneratorNode.factory)
    Node.app.register_node('t.zeros', TorchGeneratorNode.factory)
    Node.app.register_node('t.full', TorchFullNode.factory)
    Node.app.register_node('t.linspace', TorchLinSpaceNode.factory)
    Node.app.register_node('t.logspace', TorchLinSpaceNode.factory)
    Node.app.register_node('t.eye', TorchEyeNode.factory)

    Node.app.register_node('t.rand_like', TorchGeneratorLikeNode.factory)
    Node.app.register_node('t.ones_like', TorchGeneratorLikeNode.factory)
    Node.app.register_node('t.zeros_like', TorchGeneratorLikeNode.factory)

    Node.app.register_node('t.bernoulli', TorchDistributionNode.factory)
    Node.app.register_node('t.poisson', TorchDistributionNode.factory)
    Node.app.register_node('t.exponential', TorchDistributionOneParamNode.factory)
    Node.app.register_node('t.geometric', TorchDistributionOneParamNode.factory)
    Node.app.register_node('t.cauchy', TorchDistributionTwoParamNode.factory)
    Node.app.register_node('t.log_normal', TorchDistributionTwoParamNode.factory)
    Node.app.register_node('t.normal', TorchDistributionTwoParamNode.factory)
    Node.app.register_node('t.uniform', TorchDistributionTwoParamNode.factory)

    Node.app.register_node('t.permute', TorchPermuteNode.factory)
    Node.app.register_node('t.transpose', TorchTransposeNode.factory)
    Node.app.register_node('t.flip', TorchFlipNode.factory)
    Node.app.register_node('t.select', TorchSelectNode.factory)
    Node.app.register_node('t.squeeze', TorchSqueezeNode.factory)
    Node.app.register_node('t.unsqueeze', TorchUnsqueezeNode.factory)
    Node.app.register_node('t.cat', TorchCatNode.factory)
    Node.app.register_node('t.stack', TorchStackNode.factory)
    Node.app.register_node('t.hstack', TorchHStackNode.factory)
    Node.app.register_node('t.row_stack', TorchHStackNode.factory)
    Node.app.register_node('t.vstack', TorchHStackNode.factory)
    Node.app.register_node('t.column_stack', TorchHStackNode.factory)
    Node.app.register_node('t.dstack', TorchHStackNode.factory)
    Node.app.register_node('t.repeat', TorchRepeatNode.factory)
    Node.app.register_node('t.tile', TorchTileNode.factory)
    Node.app.register_node('t.chunk', TorchChunkNode.factory)
    Node.app.register_node('t.tensor_split', TorchChunkNode.factory)

    Node.app.register_node('t.adjoint', TorchViewNode.factory)
    Node.app.register_node('t.detach', TorchViewNode.factory)
    Node.app.register_node('t.t', TorchViewNode.factory)

    Node.app.register_node('t.real', TorchRealImaginaryNode.factory)
    Node.app.register_node('t.imag', TorchRealImaginaryNode.factory)
    Node.app.register_node('t.complex', TorchComplexNode.factory)

    Node.app.register_node('t.nn.Threshold', TorchNNThresholdNode.factory)
    Node.app.register_node('t.nn.relu', TorchActivationNode.factory)
    Node.app.register_node('t.nn.hardswish', TorchActivationNode.factory)
    Node.app.register_node('t.nn.relu6', TorchActivationNode.factory)
    Node.app.register_node('t.nn.selu', TorchActivationNode.factory)
    Node.app.register_node('t.nn.glu', TorchActivationNode.factory)
    Node.app.register_node('t.nn.gelu', TorchActivationNode.factory)
    Node.app.register_node('t.nn.logsigmoid', TorchActivationNode.factory)
    Node.app.register_node('t.nn.tanhshrink', TorchActivationNode.factory)
    Node.app.register_node('t.nn.softsign', TorchActivationNode.factory)
    Node.app.register_node('t.nn.tanh', TorchActivationNode.factory)
    Node.app.register_node('t.nn.sigmoid', TorchActivationNode.factory)
    Node.app.register_node('t.nn.hardsigmoid', TorchActivationNode.factory)
    Node.app.register_node('t.nn.silu', TorchActivationNode.factory)
    Node.app.register_node('t.nn.mish', TorchActivationNode.factory)
    Node.app.register_node('t.nn.elu', TorchActivationOneParamNode.factory)
    Node.app.register_node('t.nn.celu', TorchActivationOneParamNode.factory)
    Node.app.register_node('t.nn.leaky_relu', TorchActivationOneParamNode.factory)
    # Node.app.register_node('t.nn.prelu', TorchActivationOneParamNode.factory)
    Node.app.register_node('t.nn.hardshrink', TorchActivationOneParamNode.factory)
    Node.app.register_node('t.nn.softshrink', TorchActivationOneParamNode.factory)
    Node.app.register_node('t.nn.hardtanh', TorchActivationTwoParamNode.factory)
    Node.app.register_node('t.nn.rrelu', TorchActivationTwoParamNode.factory)
    Node.app.register_node('t.nn.softplus', TorchActivationTwoParamNode.factory)
    Node.app.register_node('t.nn.softmax', TorchSoftmaxNode.factory)
    Node.app.register_node('t.nn.softmin', TorchSoftmaxNode.factory)
    Node.app.register_node('t.nn.log_softmax', TorchSoftmaxNode.factory)
    Node.app.register_node('t.nn.gumbel_softmax', TorchActivationThreeParamNode.factory)

    Node.app.register_node('t.any', TorchAnyAllNode.factory)
    Node.app.register_node('t.all', TorchAnyAllNode.factory)


    Node.app.register_node('t.argmax', TorchArgMaxNode.factory)
    Node.app.register_node('t.argwhere', TorchArgWhereNode.factory)
    Node.app.register_node('t.non_zero', TorchArgWhereNode.factory)
    Node.app.register_node('t.cumsum', TorchCumSumNode.factory)
    Node.app.register_node('t.masked_select', TorchMaskedSelectNode.factory)
    Node.app.register_node('t.index_select', TorchIndexSelectNode.factory)

    Node.app.register_node('t.copysign', TorchCopySignNode.factory)

    Node.app.register_node('t.linalg.qr', TorchLinalgRQNode.factory)
    Node.app.register_node('t.linalg.svd', TorchLinalgSVDNode.factory)
    Node.app.register_node('t.linalg.pca_low_rank', TorchPCALowRankNode.factory)
    Node.app.register_node('t.linalg.eig', TorchLinalgEigenNode.factory)

    Node.app.register_node('t.window.blackman', TorchWindowNode.factory)
    Node.app.register_node('t.window.bartlett', TorchWindowNode.factory)
    Node.app.register_node('t.window.cosine', TorchWindowNode.factory)
    Node.app.register_node('t.window.hamming', TorchWindowNode.factory)
    Node.app.register_node('t.window.hann', TorchWindowNode.factory)
    Node.app.register_node('t.window.nuttall', TorchWindowNode.factory)
    Node.app.register_node('t.window.gaussian', TorchWindowOneParamNode.factory)
    Node.app.register_node('t.window.general_hamming', TorchWindowOneParamNode.factory)
    Node.app.register_node('t.window.kaiser', TorchWindowOneParamNode.factory)
    Node.app.register_node('t.window.exponential', TorchWindowTwoParamNode.factory)

    Node.app.register_node('t.special.airy_ai', TorchSpecialNode.factory)
    Node.app.register_node('t.special.bessel_j0', TorchSpecialNode.factory)
    Node.app.register_node('t.special.bessel_j1', TorchSpecialNode.factory)
    Node.app.register_node('t.special.digamma', TorchSpecialNode.factory)
    Node.app.register_node('t.special.entr', TorchSpecialNode.factory)
    Node.app.register_node('t.special.erf', TorchSpecialNode.factory)
    Node.app.register_node('t.special.erfc', TorchSpecialNode.factory)
    Node.app.register_node('t.special.erfcx', TorchSpecialNode.factory)
    Node.app.register_node('t.special.erfinv', TorchSpecialNode.factory)
    Node.app.register_node('t.special.exp2', TorchSpecialNode.factory)
    Node.app.register_node('t.special.expit', TorchSpecialNode.factory)
    Node.app.register_node('t.special.expm1', TorchSpecialNode.factory)
    Node.app.register_node('t.special.gammaln', TorchSpecialNode.factory)
    Node.app.register_node('t.special.i0', TorchSpecialNode.factory)
    Node.app.register_node('t.special.i0e', TorchSpecialNode.factory)
    Node.app.register_node('t.special.i1', TorchSpecialNode.factory)
    Node.app.register_node('t.special.i1e', TorchSpecialNode.factory)
    Node.app.register_node('t.special.log1p', TorchSpecialNode.factory)
    Node.app.register_node('t.special.logndtr', TorchSpecialNode.factory)
    Node.app.register_node('t.special.ndtr', TorchSpecialNode.factory)
    Node.app.register_node('t.special.ndtri', TorchSpecialNode.factory)
    Node.app.register_node('t.special.scaled_modified_bessel_k0', TorchSpecialNode.factory)
    Node.app.register_node('t.special.sinc', TorchSpecialNode.factory)
    Node.app.register_node('t.special.spherical_bessel_j0', TorchSpecialNode.factory)
    Node.app.register_node('t.special.softmax', TorchSpecialDimNode.factory)
    Node.app.register_node('t.special.log_softmax', TorchSpecialDimNode.factory)
    Node.app.register_node('t.special.polygamma', TorchSpecialPolygammaNode.factory)
    Node.app.register_node('t.special.logits', TorchSpecialLogitNode.factory)
    Node.app.register_node('t.special.multigammaln', TorchSpecialMultiGammaLnNode.factory)
    Node.app.register_node('t.special.zeta', TorchSpecialTwoTensorOrNumberNode.factory)
    Node.app.register_node('t.special.xlogy', TorchSpecialTwoTensorOrNumberNode.factory)
    Node.app.register_node('t.special.xlog1py', TorchSpecialTwoTensorOrNumberNode.factory)
    Node.app.register_node('t.special.xlog1py', TorchSpecialTwoTensorOrNumberNode.factory)
    Node.app.register_node('t.special.gammainc', TorchSpecialTwoTensorNode.factory)
    Node.app.register_node('t.special.gammaincc', TorchSpecialTwoTensorNode.factory)

    Node.app.register_node('tv.Grayscale', TorchvisionGrayscaleNode.factory)
    Node.app.register_node('tv.gaussian_blur', TorchvisionGaussianBlurNode.factory)
    Node.app.register_node('tv.adjust_hue', TorchvisionAdjustOneParamNode.factory)
    Node.app.register_node('tv.adjust_saturation', TorchvisionAdjustOneParamNode.factory)
    Node.app.register_node('tv.adjust_contrast', TorchvisionAdjustOneParamNode.factory)
    Node.app.register_node('tv.adjust_sharpness', TorchvisionAdjustOneParamNode.factory)
    Node.app.register_node('tv.adjust_brightness', TorchvisionAdjustOneParamNode.factory)

class TorchNode(Node):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = None
        self.output = None

    def input_to_tensor(self):
        if self.input is not None:
            input_tensor = self.input.get_received_data()
            if input_tensor is None:
                return input_tensor
            if type(input_tensor) != torch.Tensor:
                input_tensor = any_to_tensor(input_tensor)
            return input_tensor
        return None

    def data_to_tensor(self, input_tensor):
        if input_tensor is None:
            return input_tensor
        if type(input_tensor) != torch.Tensor:
            input_tensor = any_to_tensor(input_tensor)
        return input_tensor

    def data_to_torchvision_tensor(self, input_tensor):
        if input_tensor is None:
            return input_tensor
        if type(input_tensor) != torch.Tensor:
            input_tensor = any_to_tensor(input_tensor)
        if len(input_tensor.shape) > 2:
            if input_tensor.shape[-3] > 5:
                if input_tensor.shape[-1] <= 5:
                    input_tensor = input_tensor.transpose(-1, -3).transpose(-1, -2)
        return input_tensor

    def input_to_torchvision_tensor(self):
        if self.input is not None:
            input_tensor = self.input.get_received_data()
            if input_tensor is None:
                return input_tensor
            if type(input_tensor) != torch.Tensor:
                input_tensor = any_to_tensor(input_tensor)
            if len(input_tensor.shape) > 2:
                if input_tensor.shape[-3] > 5:
                    if input_tensor.shape[-1] <= 5:
                        input_tensor = input_tensor.transpose(-1, -3).transpose(-1, -2)
            return input_tensor
        return None

class TorchDeviceDtypeNode(TorchNode):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.device_string = 'cpu'
        self.dtype_string = 'float32'
        self.device_property = None
        self.dtype_property = None
        self.dtype_dict = self.create_dtype_dict()
        self.device_list = self.create_device_list()
        self.device = torch.device(self.device_string)
        self.dtype = self.dtype_dict[self.dtype_string]
        self.requires_grad = False

    def setup_dtype_device_grad(self, args):
        self.parse_args_for_dtype_and_device(args)
        self.device = torch.device(self.device_string)
        self.dtype = self.dtype_dict[self.dtype_string]
        self.create_device_property()
        self.create_dtype_property()
        self.create_requires_grad_property()

    def parse_args_for_dtype_and_device(self, args):
        for i in range(len(args)):
            if args[i] in self.device_list:
                self.device_string = args[i]
            elif args[i] in list(self.dtype_dict.keys()):
                self.dtype_string = args[i]

    def create_device_property(self):
        self.device_property = self.add_property('device', widget_type='combo', default_value=self.device_string,
                                                 callback=self.device_changed)
        self.device_property.widget.combo_items = self.device_list

    def create_dtype_property(self):
        self.dtype_property = self.add_property('dtype', widget_type='combo', default_value=self.dtype_string,
                                              callback=self.dtype_changed)
        self.dtype_property.widget.combo_items = list(self.dtype_dict.keys())

    def create_requires_grad_property(self):
        self.grad_property = self.add_property('requires_grad', widget_type='checkbox', default_value=self.requires_grad,
                                              callback=self.requires_grad_changed)

    def device_changed(self):
        device_name = self.device_property.get_widget_value()
        self.device = torch.device(device_name)

    def requires_grad_changed(self):
        self.requires_grad = self.grad_property.get_widget_value()

    def dtype_changed(self):
        dtype = self.dtype_property.get_widget_value()
        if dtype in self.dtype_dict:
            self.dtype = self.dtype_dict[dtype]

    def create_dtype_dict(self):
        dtype_dict = {}
        dtype_dict['float32'] = torch.float32
        dtype_dict['float'] = torch.float
        dtype_dict['float16'] = torch.float16
        # dtype_dict['bfloat16'] = torch.bfloat16
        # dtype_dict['double'] = torch.double
        dtype_dict['int64'] = torch.int64
        dtype_dict['uint8'] = torch.uint8
        dtype_dict['bool'] = torch.bool
        # dtype_dict['complex32'] = torch.complex32
        dtype_dict['complex64'] = torch.complex64
        dtype_dict['complex128'] = torch.complex128
        return dtype_dict

    def create_device_list(self):
        device_list = ['cpu']
        if torch.backends.mps.is_available():
            device_string = 'mps'
            device_list.append(device_string)
        if torch.cuda.is_available():
            device_string = 'cuda'
            device_list.append(device_string)
        return device_list


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


class TorchGeneratorNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchGeneratorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.shape = []
        for i in range(len(args)):
            d, t = decode_arg(args, i)
            if t == int:
                self.shape += (d,)
        if len(self.shape) == 0:
            self.shape = [1]

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)

        self.shape_properties = []
        for i in range(len(self.shape)):
            self.shape_properties.append(self.add_property('dim ' + str(i), widget_type='input_int', default_value=self.shape[i]))

        self.setup_dtype_device_grad(args)

        if self.label == 't.rand':
            self.min = 0
            self.max = 1
            self.min_input = self.add_input('min', widget_type='drag_float', default_value=self.min,
                                            callback=self.range_changed)
            self.max_input = self.add_input('max', widget_type='drag_float', default_value=self.max,
                                            callback=self.range_changed)

        out_label = 'random tensor'
        if self.label == 't.ones':
            out_label = 'tensor of ones'
        elif self.label == 't.zeros':
            out_label = 'tensor of zeros'
        self.output = self.add_output(out_label)

    def range_changed(self, val=None):
        self.min = self.min_input.get_widget_value()
        self.max = self.max_input.get_widget_value()

    def dtype_changed(self):
        super().dtype_changed()
        if self.label == 't.rand':
            if self.dtype == torch.uint8:
                if self.min < 0:
                    self.min_input.set(0.0)
                if self.max == 1.0 or self.max < 255:
                    self.max_input.set(255.0)
            elif self.dtype == torch.int64:
                if self.min < -32768:
                    self.min_input.set(-32768)
                if self.max == 1.0:
                    self.max_input.set(32767)
            elif self.dtype in [torch.float, torch.double, torch.float32, torch.float16, torch.bfloat16]:
                if self.min == -32768:
                    self.min_input.set(0.0)
                if self.max == 255:
                    self.max_input.set(1.0)
                elif self.max == 32767:
                    self.max_input.set(1.0)
            self.range_changed()

    def execute(self):
        for i in range(len(self.shape)):
            self.shape[i] = self.shape_properties[i].get_widget_value()
        size = tuple(self.shape)
        if self.label == 't.rand':
            if self.dtype in [torch.float, torch.float32, torch.double, torch.float16, torch.bfloat16, torch.complex32, torch.complex64, torch.complex128]:
                range_ = self.max - self.min
                out_array = torch.rand(size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad) * range_ + self.min
            elif self.dtype in [torch.int64, torch.uint8]:
                out_array = torch.randint(low=int(self.min), high=int(self.max), size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
            elif self.dtype == torch.bool:
                out_array = torch.randint(low=0, high=1, size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        elif self.label == 't.ones':
            out_array = torch.ones(size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        elif self.label == 't.zeros':
            out_array = torch.zeros(size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        self.output.send(out_array)

class TorchFullNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchFullNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.shape = []
        for i in range(len(args)):
            d, t = decode_arg(args, i)
            if t == int:
                self.shape += (d,)
        if len(self.shape) == 0:
            self.shape = [1]

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.value = 1.0
        self.value_input = self.add_input('value', widget_type='drag_float', default_value=self.value, callback=self.val_changed)
        self.shape_properties = []
        for i in range(len(self.shape)):
            self.shape_properties.append(self.add_property('dim ' + str(i), widget_type='input_int', default_value=self.shape[i]))

        self.setup_dtype_device_grad(args)

        out_label = 'filled tensor'
        self.output = self.add_output(out_label)

    def val_changed(self, val=None):
        self.value = self.value_input.get_widget_value()

    def execute(self):
        for i in range(len(self.shape)):
            self.shape[i] = self.shape_properties[i].get_widget_value()
        size = tuple(self.shape)
        out_array = torch.full(size=size, fill_value=self.value, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        self.output.send(out_array)


class TorchGeneratorLikeNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchGeneratorLikeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('tensor in', triggers_execution=True)

        self.setup_dtype_device_grad(args)

        if self.label == 't.rand_like':
            self.min = 0
            self.max = 1
            self.min_input = self.add_input('min', widget_type='drag_float', default_value=self.min,
                                            callback=self.range_changed)
            self.max_input = self.add_input('max', widget_type='drag_float', default_value=self.max,
                                            callback=self.range_changed)

        out_label = 'random tensor'
        if self.label == 't.ones_like':
            out_label = 'tensor of ones'
        elif self.label == 't.zeros_like':
            out_label = 'tensor of zeros'
        self.output = self.add_output(out_label)

    def range_changed(self, val=None):
        self.min = self.min_input.get_widget_value()
        self.max = self.max_input.get_widget_value()

    def dtype_changed(self):
        super().dtype_changed()
        if self.label == 't.rand':
            if self.dtype == torch.uint8:
                if self.min < 0:
                    self.min_input.set(0.0)
                if self.max == 1.0 or self.max < 255:
                    self.max_input.set(255.0)
            elif self.dtype == torch.int64:
                if self.min < -32768:
                    self.min_input.set(-32768)
                if self.max == 1.0:
                    self.max_input.set(32767)
            elif self.dtype in [torch.float, torch.double, torch.float32, torch.float16, torch.bfloat16]:
                if self.min == -32768:
                    self.min_input.set(0.0)
                if self.max == 255:
                    self.max_input.set(1.0)
                elif self.max == 32767:
                    self.max_input.set(1.0)
            self.range_changed()

    def execute(self):
        if self.input.fresh_input:
            data = self.input.get_received_data()
            if type(data) == torch.Tensor:
                shape = data.shape
                size = tuple(shape)
                if self.label == 't.rand_like':
                    if self.dtype in [torch.float, torch.float32, torch.double, torch.float16, torch.bfloat16, torch.complex32, torch.complex64, torch.complex128]:
                        range_ = self.max - self.min
                        out_array = torch.rand(size=size, device=self.device, dtype=self.dtype) * range_ + self.min
                    elif self.dtype in [torch.int64, torch.uint8]:
                        out_array = torch.randint(low=int(self.min), high=int(self.max), size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
                    elif self.dtype == torch.bool:
                        out_array = torch.randint(low=0, high=1, size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
                elif self.label == 't.ones_like':
                    out_array = torch.ones(size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
                elif self.label == 't.zeros_like':
                    out_array = torch.zeros(size=size, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
                self.output.send(out_array)


class TorchLinSpaceNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchLinSpaceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.shape = []
        self.start = 0.0
        self.stop = 1.0
        self.steps = 50

        self.op = torch.linspace
        out_label = 'linspace out'
        if self.label == 't.logspace':
            self.op = torch.logspace
            out_label = 'logspace out'

        if len(args) > 0:
            d, t = decode_arg(args, 0)
            if t in [float, int]:
                self.start = any_to_float(d)
        if len(args) > 1:
            d, t = decode_arg(args, 1)
            if t in [float, int]:
                self.stop = any_to_float(d)
        if len(args) > 2:
            d, t = decode_arg(args, 2)
            if t in [float, int]:
                self.steps = any_to_int(d)

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.start_property = self.add_property('start', widget_type='drag_float', default_value=self.start)
        self.stop_property = self.add_property('stop', widget_type='drag_float', default_value=self.stop)
        self.steps_property = self.add_property('steps', widget_type='drag_int', default_value=self.steps)

        self.setup_dtype_device_grad(args)

        self.output = self.add_output(out_label)

    def execute(self):
        self.start = self.start_property.get_widget_value()
        self.stop = self.stop_property.get_widget_value()
        self.steps = self.steps_property.get_widget_value()
        out_array = self.op(self.start, self.stop, self.steps, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        self.output.send(out_array)


class TorchEyeNode(TorchDeviceDtypeNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchEyeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.shape = []
        if len(args) > 0:
            self.n = any_to_int(args[0])
        else:
            self.n = 4
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.n_input = self.add_input('n', widget_type='input_int', default_value=self.n, callback=self.n_changed)

        self.setup_dtype_device_grad(args)

        out_label = 'eye tensor'
        self.output = self.add_output(out_label)

    def n_changed(self, val=None):
        self.n = self.n_input.get_widget_value()

    def execute(self):
        out_array = torch.eye(n=self.n, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        self.output.send(out_array)

class TorchViewNode(TorchNode):
    op_dict = {
        't.adjoint': torch.adjoint,
        't.detach': torch.detach,
        't.t': torch.t
    }
    @staticmethod
    def factory(name, data, args=None):
        node = TorchViewNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = torch.t
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            out_tensor = self.op(input_tensor)
            self.output.send(out_tensor)

class TorchAnyAllNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchAnyAllNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = torch.all
        output_name = 'all'
        if self.label == 't.any':
            self.op = torch.any
            output_name == 'any'

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            result = self.op(input_tensor)
            self.output.send(result)


class TorchDistributionNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if self.label == 't.bernoulli':
            self.op = torch.bernoulli
        elif self.label == 't.poisson':
            self.op = torch.poisson

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.output = self.add_output('tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            out_tensor = self.op(input_tensor)
            self.output.send(out_tensor)

class TorchDistributionOneParamNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionOneParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        param_1_name = ''
        self.param_1 = 1
        if self.label == 't.exponential':
            param_1_name = 'lambda'
        elif self.label == 't.geometric':
            param_1_name = 'p'

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.param_1_property = self.add_input(param_1_name, widget_type='drag_float', default_value=self.param_1, callback=self.params_changed)
        self.output = self.add_output('tensor out')

    def params_changed(self, val=0):
        self.param_1 = self.param_1_property.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            out_tensor = input_tensor.clone()
            if self.label == 't.exponential':
                out_tensor.exponential_(self.param_1)
            elif self.label == 't.geometric':
                out_tensor.log_normal_(self.param_1)
            self.output.send(out_tensor)

class TorchDistributionTwoParamNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchDistributionTwoParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        param_1_name = ''
        param_2_name = ''
        self.param_1 = 0
        self.param_2 = 1
        if self.label == 't.cauchy':
            param_1_name = 'median'
            param_2_name = 'sigma'
        elif self.label in ['t.log_normal', 't_normal']:
            param_1_name = 'mean'
            param_2_name = 'std'
        elif self.label == 't.uniform':
            param_1_name = 'from'
            param_2_name = 'to'

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.param_1_property = self.add_input(param_1_name, widget_type='drag_float', default_value=self.param_1, callback=self.params_changed)
        self.param_2_property = self.add_input(param_2_name, widget_type='drag_float', default_value=self.param_2, callback=self.params_changed)
        self.output = self.add_output('tensor out')

    def params_changed(self, val=0):
        self.param_1 = self.param_1_property.get_widget_value()
        self.param_2 = self.param_2_property.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            out_tensor = input_tensor.clone()
            if self.label == 't.cauchy':
                out_tensor.cauchy_(self.param_1, self.param_2)
            elif self.label == 't.log_normal':
                out_tensor.log_normal_(self.param_1, self.param_2)
            elif self.label == 't.normal':
                out_tensor.normal_(self.param_1, self.param_2)
            elif self.label == 't.uniform':
                out_tensor.uniform_(self.param_1, self.param_2)
            self.output.send(out_tensor)

class TorchCDistanceNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCDistanceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input("input", triggers_execution=True)
        self.output = self.add_output("output")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            input_tensor = input_tensor.unsqueeze(dim=0)
            euclidean_length = torch.cdist(input_tensor, torch.zeros_like(input_tensor))

            self.output.send(euclidean_length.item())

class TorchDistanceNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCDistanceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        out_label = 'length'
        self.input2 = None
        self.input = self.add_input("tensor in", triggers_execution=True)
        if self.label == 't.dist':
            self.input2 = self.add_input("tensor 2 in")
            out_label = 'distance'

        self.output = self.add_output(out_label)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if self.input2 is not None:
                input2 = self.data_to_tensor(self.input2.get_received_data())
                if input2 is not None:
                    euclidean_length = torch.dist(input_tensor, torch.zeros_like(input_tensor))
                else:
                    euclidean_length = torch.dist(input_tensor, input_tensor)
            else:
                euclidean_length = torch.dist(input_tensor, torch.zeros_like(input_tensor))

            self.output.send(euclidean_length.item())


class TorchWithDimNode(TorchNode):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.dim_specified = False
        self.dim_input = None
        self.dim = 0
        if len(args) > 0:
            self.dim = any_to_int(args[0])
            self.dim_specified = True

    def add_dim_input(self):
        if self.dim_specified:
            self.dim_input = self.add_input('dim', widget_type='input_int', default_value=self.dim, callback=self.dim_changed)

    def dim_changed(self, val=None):
        if self.dim_input is not None:
            self.dim = self.dim_input.get_widget_value()


class TorchArgMaxNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchArgMaxNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input("tensor in", triggers_execution=True)
        if self.dim_specified:
            self.add_dim_input()
        self.output = self.add_output("max index")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if self.dim_specified:
                output_tensor = torch.argmax(input_tensor, dim=self.dim)
            else:
                output_tensor = torch.argmax(input_tensor)
            self.output.send(output_tensor)


class TorchFlipNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchFlipNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.flip_list = [0]
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.flip_property = self.add_property('flip dims', widget_type='text_input', default_value='0',
                                                  callback=self.flip_changed)
        self.output = self.add_output("output")

    def flip_changed(self):
        flip_text = self.flip_property.get_widget_value()
        flip_split = re.findall(r'\d+', flip_text)
        flip_list, _, _ = list_to_hybrid_list(flip_split)
        self.flip_list = flip_list

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if len(input_tensor.shape) < len(self.flip_list) or len(self.flip_list) == 0:
                self.output.send(input_tensor)
            else:
                permuted = torch.flip(input_tensor, self.flip_list)
                self.output.send(permuted)


# class TorchCropNode(Node):
#     @staticmethod
#     def factory(name, data, args=None):
#         node = TorchCropNode(name, data, args)
#         return node
#
#     def __init__(self, label: str, data, args):
#         super().__init__(label, data, args)
#         self.dim_count = 2
#         if len(args) > 0:
#             self.dim_count = string_to_int(args[0])
#
#         self.input = self.add_input("tensor in", triggers_execution=True)
#         self.croppers = []
#         for i in range(self.dims_count):
#             crop_min = self.add_input('dim ' + str(i) + ' min', widget_type='dragint', )
#             crop_max = self.add_input('dim ' + str(i) + ' max', widget_type='dragint', )
#         self.indices_property = self.add_property('', widget_type='text_input', width=200, default_value=index_string,
#                                                   callback=self.dim_changed)
#         self.output = self.add_output("output")

class TorchSqueezeNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSqueezeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input("tensor in", triggers_execution=True)
        if self.dim_specified:
            self.add_dim_input()
        self.output = self.add_output("output")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if -1 - len(input_tensor.shape) < self.dim < len(input_tensor.shape):
                if self.dim_specified:
                    self.output.send(torch.squeeze(input_tensor, self.dim))
                else:
                    self.output.send(torch.squeeze(input_tensor))
                return

class TorchIndexSelectNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchIndexSelectNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.index_input = self.add_input('indices in')
        if self.dim_specified:
            self.add_dim_input()

        self.output = self.add_output("output")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.index_input.get_received_data()
            if data is not None:
                index_tensor = self.data_to_tensor(data)
                if index_tensor is not None:
                    if -1 - len(input_tensor.shape) < self.dim < len(input_tensor.shape):
                        self.output.send(torch.index_select(input_tensor, self.dim, index_tensor))
                    else:
                        if self.app.verbose:
                            print('t.index_select dim is invalid', self.dim)
                else:
                    if self.app.verbose:
                        print('t.index_select no index tensor')
            else:
                if self.app.verbose:
                    print('t.index_select invalid input tensor')

class TorchUnsqueezeNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchUnsqueezeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input("tensor in", triggers_execution=True)
        if self.dim_specified:
            self.dim_input = self.add_input('dim', widget_type='input_int', default_value=self.dim, callback=self.dim_changed)
        self.output = self.add_output("output")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if -1 - len(input_tensor.shape) < self.dim <= len(input_tensor.shape):
                self.output.send(torch.unsqueeze(input_tensor, self.dim))
                return


class TorchMaskedSelectNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchMaskedSelectNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('source tensor', triggers_execution=True)
        self.mask_input = self.add_input('mask')
        self.out = self.add_output('selection tensor')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.mask_input.get_received_data()
            if data is not None:
                mask_tensor = self.data_to_tensor(data)
                if mask_tensor is not None:
                    if mask_tensor.dtype is not torch.bool:
                        mask_tensor.to(dtype=torch.bool)
                    if mask_tensor.shape[0] == input_tensor.shape[0]:
                        out_tensor = torch.masked_select(input_tensor, mask_tensor)
                        self.out.send(out_tensor)



class TorchCumSumNode(TorchWithDimNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCumSumNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input("tensor in", triggers_execution=True)
        if self.dim_specified:
            self.add_dim_input()
        self.output = self.add_output("output")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if -1 - len(input_tensor.shape) < self.dim <= len(input_tensor.shape):
                self.output.send(torch.cumsum(input_tensor, self.dim))
                return


class TorchArgWhereNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchArgWhereNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.output = self.add_output("index tensor where non-zero")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(torch.argwhere(input_tensor))


class TorchSpecialNode(TorchNode):
    op_dict = {
        't.special.airy_ai': torch.special.airy_ai,
        't.special.bessel_j0': torch.special.bessel_j0,
        't.special.bessel_j1': torch.special.bessel_j1,
        't.special.digamma': torch.special.digamma,
        't.special.entr': torch.special.entr,
        't.special.erf': torch.special.erf,
        't.special.erfc': torch.special.erfc,
        't.special.erfcx': torch.special.erfcx,
        't.special.erfinv': torch.special.erfinv,
        't.special.exp2': torch.special.exp2,
        't.special.expit': torch.special.expit,
        't.special.expm1': torch.special.expm1,
        't.special.gammaln': torch.special.gammaln,
        't.special.i0': torch.special.i0,
        't.special.i0e': torch.special.i0e,
        't.special.i1': torch.special.i1,
        't.special.i1e': torch.special.i1e,
        't.special.log1p': torch.special.log1p,
        't.special.logndtr': torch.special.log_ndtr,
        't.special.ndtr': torch.special.ndtr,
        't.special.ndtri': torch.special.ndtri,
        't.special.scaled_modified_bessel_k0': torch.special.scaled_modified_bessel_k0,
        't.special.sinc': torch.special.sinc,
        't.special.spherical_bessel_j0': torch.special.spherical_bessel_j0
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = torch.special.exp2
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.output = self.add_output("tensor out")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor))


class TorchSpecialDimNode(TorchWithDimNode):
    op_dict = {
        't.special.log_softmax': torch.special.log_softmax,
        't.special.softmax': torch.special.softmax
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialDimNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = torch.special.log_softmax
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        self.input = self.add_input("tensor in", triggers_execution=True)
        if self.dim_specified:
            self.add_dim_input()
        self.output = self.add_output("tensor out")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor, dim=self.dim))


class TorchSpecialPolygammaNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialPolygammaNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.n = 0
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.n_input = self.add_input('n', widget_type='input_int', default_value=self.n, min=0, callback=self.n_changed)
        self.output = self.add_output("tensor out")

    def n_changed(self, val=0):
        self.n = self.n_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(torch.special.polygamma(self.n, input_tensor))



class TorchSpecialLogitNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialLogitNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.eps = 1e-8
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.eps_input = self.add_input('eps', widget_type='drag_float', default_value=self.eps, callback=self.eps_changed)

        self.output = self.add_output("tensor out")

    def custom_setup(self, from_file):
        self.eps_input.widget.set_format('%.8f')

    def eps_changed(self, val=0):
        self.eps = self.eps_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(torch.special.logit(input_tensor, self.eps))


class TorchSpecialTwoTensorNode(TorchNode):
    op_dict = {
        't.special.gammainc': torch.special.gammainc,
        't.special.gammaincc': torch.special.gammaincc
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialTwoTensorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.op = torch.special.gammainc
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        self.input = self.add_input("tensor 1 in", triggers_execution=True)
        self.second_input = self.add_input('tensor 2 in')
        self.output = self.add_output("tensor out")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.second_input.get_received_data()
            if data is not None:
                second_tensor = self.data_to_tensor(data)
                if second_tensor is not None:
                    self.output.send(self.op(input_tensor, second_tensor))


class TorchSpecialTwoTensorOrNumberNode(TorchNode):
    op_dict = {
        't.special.zeta': torch.special.zeta,
        't.special.xlogy': torch.special.xlogy,
        't.special.xlog1py': torch.special.xlog1py
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialTwoTensorOrNumberNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.op = torch.special.gammainc
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        self.input = self.add_input("tensor 1 or number in", triggers_execution=True)
        self.second_input = self.add_input('tensor 2 or number in')
        self.output = self.add_output("tensor out")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.second_input.get_received_data()
            if data is not None:
                second_tensor = self.data_to_tensor(data)
                if second_tensor is not None:
                    self.output.send(self.op(input_tensor, second_tensor))

class TorchSpecialMultiGammaLnNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSpecialMultiGammaLnNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.p = 1e-8
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.p_input = self.add_input('p', widget_type='input_int', default_value=self.p, callback=self.p_changed)
        self.output = self.add_output("tensor out")

    def p_changed(self, val=0):
        self.p = self.p_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(torch.special.multigammaln(input_tensor, self.p))


class TorchRealImaginaryNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchRealImaginaryNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.real = True
        output_name = 'real tensor'
        if self.label == 't.imag':
            self.real = False
            output_name = 'imaginary tensor'

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.output = self.add_output(output_name)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if input_tensor.dtype in [torch.complex32, torch.complex64, torch.complex128]:
                if self.real:
                    self.output.send(torch.real(input_tensor))
                else:
                    self.output.send(torch.imag(input_tensor))


class TorchComplexNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchComplexNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.real = True
        self.input = self.add_input("real tensor in", triggers_execution=True)
        self.imag_input = self.add_input("imag tensor in", triggers_execution=True)
        self.output = self.add_output('complex tensor out')

    def execute(self):
        real_tensor = self.input_to_tensor()
        if real_tensor is not None:
            data = self.imag_input.get_received_data()
            if data is not None:
                imag_tensor = self.data_to_tensor(data)
                if imag_tensor is not None:
                    if real_tensor.shape == imag_tensor.shape:
                        if real_tensor.dtype in [torch.float16, torch.float32, torch.float64]:
                            if real_tensor.dtype == imag_tensor.dtype:
                                complex_tensor = torch.complex(real_tensor, imag_tensor)
                                self.output.send(complex_tensor)
                            else:
                                if self.app.verbose:
                                    print(self.label, 'real and imaginary tensor dtypes don\'t match', real_tensor.dtype, imag_tensor.dtype)
                        else:
                            if self.app.verbose:
                                print(self.label, 'real tensor wrong dtype', real_tensor.dtype)
                    else:
                        if self.app.verbose:
                            print(self.label, 'imaginary tensor is None')
                else:
                    if self.app.verbose:
                        print(self.label, 'no input for imaginary tensor')
        else:
            if self.app.verbose:
                print(self.label, 'real tensor is None')

class TorchCopySignNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCopySignNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.sign_input = self.add_input("sign tensor")
        self.output = self.add_output("tensor with copied sign")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            data = self.sign_input.get_received_data()
            if data is not None:
                sign_tensor = self.data_to_tensor(data)
                if sign_tensor is not None:
                    if sign_tensor.device == input_tensor.device:
                        try:
                            self.output.send(torch.copysign(input_tensor, sign_tensor))
                        except Exception as error:
                            print('t.copysign:', error)


class TorchStackCatNode(TorchNode):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input_count = 2
        self.dim = 0
        if len(args) > 0:
            self.input_count = string_to_int(args[0])
        self.other_inputs = []
        self.input = self.add_input("tensor 1", triggers_execution=True)
        for i in range(self.input_count - 1):
            self.other_inputs.append(self.add_input('tensor ' + str(i + 2)))
        self.dim_input = self.add_input('dim', widget_type='input_int', default_value=self.dim, callback=self.dim_changed)
        self.output = self.add_output("output")

    def dim_changed(self, val=None):
        self.dim = self.dim_input.get_widget_value()


class TorchStackNode(TorchStackCatNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchStackNode(name, data, args)
        return node
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            stack_list = [input_tensor]
            for i in range(self.input_count - 1):
                an_in = self.other_inputs[i]
                a_tensor = self.data_to_tensor(an_in.get_received_data())
                if a_tensor is not None:
                    if len(a_tensor.shape) == len(input_tensor.shape):
                        ok_shape = True
                        for j in range(len(a_tensor.shape)):
                            if a_tensor.shape[j] != input_tensor.shape[j]:
                                ok_shape = False
                                if self.app.verbose:
                                    print('t.stack input tensor ' + str(i + 1) + ' has wrong shape')
                                break
                        if ok_shape:
                            stack_list.append(a_tensor)
                    else:
                        if self.app.verbose:
                            print('t.stack input tensor ' + str(i + 1) + ' has wrong number of dimensions')
            if -len(input_tensor.shape) <= self.dim <= len(input_tensor.shape):
                output_tensor = torch.stack(stack_list, self.dim)
                self.output.send(output_tensor)
            else:
                if self.app.verbose:
                    print('t.stack dim is out of range', self.dim)


class TorchCatNode(TorchStackCatNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCatNode(name, data, args)
        return node
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            cat_list = [input_tensor]
            for i in range(self.input_count - 1):
                an_in = self.other_inputs[i]
                a_tensor = self.data_to_tensor(an_in.get_received_data())
                if a_tensor is not None:
                    if len(a_tensor.shape) == len(input_tensor.shape):
                        ok_shape = True
                        for j in range(len(a_tensor.shape)):
                            if j != self.dim:
                                if a_tensor.shape[j] != input_tensor.shape[j]:
                                    ok_shape = False
                                    if self.app.verbose:
                                        print('t.cat input tensor ' + str(i) + ' has wrong shape')
                                    break
                        if ok_shape:
                            cat_list.append(a_tensor)
                    else:
                        if self.app.verbose:
                            print('t.cat input tensor ' + str(i) + ' has wrong number of dimensions')
            if self.dim < len(input_tensor.shape):
                output_tensor = torch.cat(cat_list, self.dim)
                self.output.send(output_tensor)
            else:
                if self.app.verbose:
                    print('t.cat dim is out of range', self.dim)


class TorchHStackNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchHStackNode(name, data, args)
        return node
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input_count = 2
        if len(args) > 0:
            self.input_count = string_to_int(args[0])
        self.op = torch.hstack
        if self.label in ['t.vstack', 't.row_stack']:
            self.op = torch.vstack
        elif self.label == 't.dstack':
            self.op = torch.dstack

        self.other_inputs = []
        self.input = self.add_input("tensor 1", triggers_execution=True)
        for i in range(self.input_count - 1):
            self.other_inputs.append(self.add_input('tensor ' + str(i + 2)))
        self.output = self.add_output("stacked tensors")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            stack_list = [input_tensor]
            for i in range(self.input_count - 1):
                an_in = self.other_inputs[i]
                a_tensor = self.data_to_tensor(an_in.get_received_data())
                if a_tensor is not None:
                    if a_tensor.shape != input_tensor.shape:
                        if self.app.verbose:
                            print(self.label + ' input tensors must have the same shape')
                        return
                    else:
                        stack_list.append(a_tensor)
            print(stack_list)
            output_tensor = self.op(tuple(stack_list))
            self.output.send(output_tensor)


class TorchSelectNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSelectNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.dim = 0
        self.index = 0
        if len(args) > 0:
            self.dim = any_to_int(args[0])
        if len(args) > 1:
            self.index = any_to_int(args[1])

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.dim_input = self.add_input('dim', widget_type='input_int', default_value=self.dim, callback=self.dim_changed)
        self.index_input = self.add_input('index', widget_type='input_int', default_value=self.index, callback=self.index_changed)
        self.output = self.add_output("output")

    def dim_changed(self, val=None):
        self.dim = self.dim_input.get_widget_value()

    def index_changed(self, val=None):
        self.index = self.index_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if -1 - len(input_tensor.shape) < self.dim < len(input_tensor.shape):
                if -1 - input_tensor.shape[self.dim] < self.index < input_tensor.shape[self.dim]:
                    self.output.send(torch.select(input_tensor, self.dim, self.index))
                    return


class TorchChunkNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchChunkNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.dim = 0
        self.splits = 2
        self.op = torch.tensor_split
        if self.label == 't.chunk':
            self.op = torch.chunk
        if len(args) > 0:
            self.splits = any_to_int(args[0])
        if len(args) > 1:
            self.dim = any_to_int(args[1])

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.dim_input = self.add_input('dim', widget_type='input_int', default_value=self.dim, callback=self.dim_changed)
        self.split_count_option = self.add_option('split', widget_type='input_int', default_value=self.splits)
        self.tensor_outputs = []

        for i in range(self.splits):
            self.tensor_outputs.append(self.add_output("tensor " + str(i)))

    def dim_changed(self, val=None):
        self.dim = self.dim_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if -1 - len(input_tensor.shape) < self.dim < len(input_tensor.shape):
                if self.splits < input_tensor.shape[self.dim]:
                    tensors = self.op(input_tensor, self.splits, self.dim)
                    for idx, tensor_ in enumerate(tensors):
                        if idx < len(self.outputs):
                            self.tensor_outputs[idx].send(tensor_)


class TorchActivationNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchActivationNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = None
        if self.label == 't.nn.relu':
            self.op = torch.nn.functional.relu
        elif self.label == 't.nn.hardswish':
            self.op = torch.nn.functional.hardswish
        elif self.label == 't.nn.hardtanh':
            self.op = torch.nn.functional.hardtanh
        elif self.label == 't.nn.relu6':
            self.op = torch.nn.functional.relu6
        elif self.label == 't.nn.selu':
            self.op = torch.nn.functional.selu
        elif self.label == 't.nn.glu':
            self.op = torch.nn.functional.glu
        elif self.label == 't.nn.gelu':
            self.op = torch.nn.functional.gelu
        elif self.label == 't.nn.logsigmoid':
            self.op = torch.nn.functional.logsigmoid
        elif self.label == 't.nn.tanhshrink':
            self.op = torch.nn.functional.tanhshrink
        elif self.label == 't.nn.softsign':
            self.op = torch.nn.functional.softsign
        elif self.label == 't.nn.tanh':
            self.op = torch.nn.functional.tanh
        elif self.label == 't.nn.sigmoid':
            self.op = torch.nn.functional.sigmoid
        elif self.label == 't.nn.hardsigmoid':
            self.op = torch.nn.functional.hardsigmoid
        elif self.label == 't.nn.silu':
            self.op = torch.nn.functional.silu
        elif self.label == 't.nn.mish':
            self.op = torch.nn.functional.mish

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.output = self.add_output("output")

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor))


class TorchActivationTwoParamNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchActivationTwoParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = None
        param_1_name = 'minimum'
        param_2_name = 'maximum'
        self.parameter_1 = -1
        self.parameter_2 = 1

        if self.label == 't.nn.hardtanh':
            self.op = torch.nn.functional.hardtanh
        if self.label == 't.nn.rrelu':
            self.op = torch.nn.functional.rrelu
            param_1_name = 'lower'
            param_2_name = 'upper'
            self.parameter_1 = 0.125
            self.parameter_2 = 0.3333333333333
        if self.label == 't.nn.softplus':
            self.op = torch.nn.functional.softplus
            param_1_name = 'beta'
            param_2_name = 'threshold'
            self.parameter_1 = 1.0
            self.parameter_2 = 20

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.parameter_1_input = self.add_input(param_1_name, widget_type='drag_float', default_value=self.parameter_1, callback=self.parameter_changed)
        self.parameter_2_input = self.add_input(param_2_name, widget_type='drag_float', default_value=self.parameter_2, callback=self.parameter_changed)
        self.output = self.add_output("output")

    def parameter_changed(self, val=None):
        self.parameter_1 = self.parameter_1_input.get_widget_value()
        self.parameter_2 = self.parameter_2_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor, self.parameter_1, self.parameter_2))


class TorchActivationThreeParamNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchActivationThreeParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = None
        param_1_name = 'tau'
        param_2_name = 'hard'
        param_3_name = 'dim'
        self.parameter_1 = 1
        self.parameter_2 = False
        self.parameter_3 = -1

        if self.label == 't.nn.gumbel_softmax':
            self.op = torch.nn.functional.gumbel_softmax

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.parameter_1_input = self.add_input(param_1_name, widget_type='drag_float', default_value=self.parameter_1, callback=self.parameter_changed)
        self.parameter_2_input = self.add_input(param_2_name, widget_type='checkbox', default_value=self.parameter_2, callback=self.parameter_changed)
        self.parameter_3_input = self.add_input(param_3_name, widget_type='input_int', default_value=self.parameter_3, callback=self.parameter_changed)
        self.output = self.add_output("output")

    def parameter_changed(self, val=None):
        self.parameter_1 = self.parameter_1_input.get_widget_value()
        self.parameter_2 = self.parameter_2_input.get_widget_value()
        self.parameter_3 = self.parameter_3_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor, self.parameter_1, self.parameter_2, self.parameter_3))


class TorchSoftmaxNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSoftmaxNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = None
        self.dim = 0

        if self.label == 't.nn.softmax':
            self.op = torch.nn.functional.softmax
        if self.label == 't.nn.softmin':
            self.op = torch.nn.functional.softmin
        if self.label == 't.nn.log_softmax':
            self.op = torch.nn.functional.log_softmax

        if len(args) > 0:
            self.dim = any_to_int(args[0])

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.dim_input = self.add_input('dim', widget_type='input_int', default_value=self.dim, callback=self.dim_changed)
        self.output = self.add_output("output")

    def dim_changed(self, val=None):
        self.dim = self.dim_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor, self.dim))


class TorchActivationOneParamNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchActivationOneParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = None
        param_name = ''
        self.parameter = 1
        if self.label == 't.nn.elu':
            self.op = torch.nn.functional.elu
            param_name = 'alpha'
            self.parameter = 1.0
        elif self.label == 't.nn.celu':
            self.op = torch.nn.functional.celu
            param_name = 'alpha'
            self.parameter = 1.0
        elif self.label == 't.nn.leaky_relu':
            self.op = torch.nn.functional.leaky_relu
            param_name = 'negative slope'
            self.parameter = 0.01
        # elif self.label == 't.nn.prelu':
        #     self.op = torch.nn.functional.prelu
        #     param_name = 'weight'
        #     self.parameter = 1.0
        elif self.label == 't.nn.hardshrink':
            self.op = torch.nn.functional.hardshrink
            param_name = 'lambda'
            self.parameter = 0.5
        elif self.label == 't.nn.softshrink':
            self.op = torch.nn.functional.softshrink
            param_name = 'lambda'
            self.parameter = 0.5

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.parameter_input = self.add_input(param_name, widget_type='drag_float', default_value=self.parameter, callback=self.parameter_changed)
        self.output = self.add_output("output")

    def parameter_changed(self, val=None):
        self.parameter = self.parameter_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor, self.parameter))


class TorchNNThresholdNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchNNThresholdNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.threshold = 0
        self.replace = 0
        if len(args) > 0:
            self.threshold = any_to_int(args[0])
        if len(args) > 1:
            self.replace = any_to_int(args[1])

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.threshold_input = self.add_input('threshold', widget_type='drag_float', default_value=self.threshold, callback=self.threshold_changed)
        self.replace_input = self.add_input('replacenent', widget_type='drag_float', default_value=self.replace, callback=self.replacement_changed)
        self.op = torch.nn.Threshold(self.threshold, self.replace)
        self.output = self.add_output("output")

    def threshold_changed(self, val=None):
        self.threshold = self.threshold_input.get_widget_value()
        self.op = torch.nn.Threshold(self.threshold, self.replace)

    def replacement_changed(self, val=None):
        self.replace = self.replace_input.get_widget_value()
        self.op = torch.nn.Threshold(self.threshold, self.replace)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            self.output.send(self.op(input_tensor))


class TorchvisionGrayscaleNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchvisionGrayscaleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.output = self.add_output("output")
        self.op = torchvision.transforms.Grayscale()

    def execute(self):
        input_tensor = self.input_to_torchvision_tensor()
        if input_tensor is not None:
            output_tensor = self.op(input_tensor)
            self.output.send(output_tensor)

class TorchvisionGaussianBlurNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchvisionGaussianBlurNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.kernel_size = 9
        self.sigma = 0.5

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.kernel_size_property = self.add_property('kernel size', widget_type='combo', default_value=self.kernel_size, callback=self.params_changed)
        self.kernel_size_property.widget.combo_items = [3, 5, 7, 9, 11, 13, 15, 17, 19]
        self.sigma_property = self.add_property('sigma', widget_type='drag_float', default_value=self.sigma, callback=self.params_changed)
        self.output = self.add_output("output")
        self.op = torchvision.transforms.functional.gaussian_blur

    def params_changed(self):
        self.sigma = self.sigma_property.get_widget_value()
        if self.sigma <= 0:
            self.sigma = 0.1
        self.kernel_size = int(self.kernel_size_property.get_widget_value())
        self.op = torchvision.transforms.functional.gaussian_blur
    def execute(self):
        input_tensor = self.input_to_torchvision_tensor()
        if input_tensor is not None:
            output_tensor = self.op(input_tensor, self.kernel_size, self.sigma)
            self.output.send(output_tensor)


class TorchvisionAdjustOneParamNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchvisionAdjustOneParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.param = 0.0
        param_name = 'hue offset'
        min = -.5
        max = .5

        if self.label == 'torchvision.adjust_hue':
            self.param = 0.0
            param_name = 'hue offset'
            min = -.5
            max = .5
            self.op = torchvision.transforms.functional.adjust_hue
        elif self.label == 'torchvision.adjust_saturation':
            self.param = 1.0
            param_name = 'saturation'
            min = 0
            max = 10
            self.op = torchvision.transforms.functional.adjust_saturation
        elif self.label == 'torchvision.adjust_sharpness':
            self.param = 1.0
            param_name = 'sharpness'
            min = 0
            max = 10
            self.op = torchvision.transforms.functional.adjust_sharpness
        elif self.label == 'torchvision.adjust_contrast':
            self.param = 1.0
            param_name = 'contrast'
            min = 0
            max = 10
            self.op = torchvision.transforms.functional.adjust_contrast
        elif self.label == 'torchvision.adjust_brightness':
            self.param = 1.0
            param_name = 'brightness'
            min = 0
            max = 10
            self.op = torchvision.transforms.functional.adjust_brightness

        self.input = self.add_input("tensor in", triggers_execution=True)
        self.param_input = self.add_input(param_name, widget_type='drag_float', default_value=self.param, min=min, max=max, callback=self.params_changed)
        self.output = self.add_output("output")
        print(self.op)

    def params_changed(self):
        self.param = self.param_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_torchvision_tensor()
        if input_tensor is not None:
            output_tensor = self.op(input_tensor, self.param)
            self.output.send(output_tensor)


class TorchSubtensorNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSubtensorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.dim_list = None
        index_string = ''
        for i in range(len(args)):
            index_string += args[i]
        if index_string == '':
            index_string = ':'
        self.input = self.add_input("tensor in", triggers_execution=True)
        self.indices_input = self.add_input('', widget_type='text_input', widget_width=200, default_value=index_string,
                                                  callback=self.dim_changed)
        self.output = self.add_output("output")

    def dim_changed(self, val=None):
        dim_text = self.indices_input.get_widget_value()
        dim_split = dim_text.split(',')
        dimmers = []

        for i in range(len(dim_split)):
            dimmer = dim_split[i]
            dimmer = dimmer.split(':')
            dim_nums = []
            if len(dimmer) == 1:
                dim_num = re.findall(r'\d+', dimmer[0])
                if len(dim_num) > 0:
                    dim_num = string_to_int(dim_num[0])
                    dim_nums.append([dim_num])
                    dim_nums.append([dim_num + 1])
            else:
                for j in range(len(dimmer)):
                    dim_num = re.findall(r'\d+', dimmer[j])
                    if len(dim_num) == 0:
                        if j == 0:
                            dim_nums.append([0])
                        else:
                            dim_nums.append([1000000])
                    else:
                        dim_num = string_to_int(dim_num[0])
                        dim_nums.append([dim_num])
            dimmers.append(dim_nums)

        self.dim_list = dimmers

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if self.dim_list is None:
                self.dim_changed()
            if len(input_tensor.shape) < len(self.dim_list) or len(self.dim_list) == 0:
                self.output.send(input_tensor)
            else:
                dim_list_now = []
                for i in range(len(self.dim_list)):
                    dim_dim = self.dim_list[i]
                    dim_list_now.append(dim_dim[0][0])
                    if dim_dim[1][0] == 1000000:
                        dim_list_now.append(input_tensor.shape[i])
                    else:
                        dim_list_now.append(dim_dim[1][0])
                sub_tensor = input_tensor
                if len(dim_list_now) == 2:
                    sub_tensor = input_tensor[dim_list_now[0]:dim_list_now[1]]
                elif len(dim_list_now) == 4:
                    sub_tensor = input_tensor[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3]]
                elif len(dim_list_now) == 6:
                    sub_tensor = input_tensor[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3], dim_list_now[4]:dim_list_now[5]]
                elif len(dim_list_now) == 8:
                    sub_tensor = input_tensor[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3],
                             dim_list_now[4]:dim_list_now[5], dim_list_now[6]:dim_list_now[7]]
                self.output.send(sub_tensor)


class TorchPermuteNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchPermuteNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.permute = []
        if len(args) > 0:
            for i in range(len(args)):
                self.permute.append(any_to_int(args[i]))
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.permute_property = self.add_input('permute', widget_type='text_input', default_value=self.permute, callback=self.permute_changed)
        self.output = self.add_output('permuted tensor out')

    def permute_changed(self, val=None):
        permute_text = self.permute_property.get_widget_value()
        permute_split = re.findall(r'\d+', permute_text)
        permute_list, _, _ = list_to_hybrid_list(permute_split)
        self.permute = permute_list

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if len(input_tensor.shape) != len(self.permute):
                self.output.send(input_tensor)
                if self.app.verbose:
                    print('WARNING: torch.permute - permute list and channel count mismatch')
            else:
                permuted = torch.permute(input_tensor, self.permute)
                self.output.send(permuted)


class TorchRepeatNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchRepeatNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.repeat = []
        if len(args) > 0:
            for i in range(len(args)):
                self.repeat.append(any_to_int(args[i]))
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.repeat_property = self.add_input('repeats', widget_type='text_input', default_value=self.repeat, callback=self.repeat_changed)
        self.output = self.add_output('repeated tensor out')

    def repeat_changed(self, val=None):
        repeat_text = self.repeat_property.get_widget_value()
        repeat_split = re.findall(r'\d+', repeat_text)
        repeat_list, _, _ = list_to_hybrid_list(repeat_split)
        self.repeat = repeat_list

    def execute(self):
        input_tensor = self.input_to_tensor()
        repeat = self.repeat
        if input_tensor is not None:
            repeated = input_tensor.repeat(repeat)
            self.output.send(repeated)

class TorchTileNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchTileNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.tile = []
        if len(args) > 0:
            for i in range(len(args)):
                self.tile.append(any_to_int(args[i]))
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.tile_property = self.add_input('tiling', widget_type='text_input', default_value=self.tile, callback=self.tile_changed)
        self.output = self.add_output('repeated tensor out')

    def tile_changed(self, val=None):
        tile_text = self.tile_property.get_widget_value()
        tile_split = re.findall(r'\d+', tile_text)
        tile_list, _, _ = list_to_hybrid_list(tile_split)
        self.tile = tile_list

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            tiled = input_tensor.repeat(self.tile)
            self.output.send(tiled)

class TorchTransposeNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchTransposeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.transpose1 = 0
        self.transpose2 = 1
        if len(args) > 0:
            self.transpose1 = any_to_int(args[0])
        if len(args) > 1:
            self.transpose2 = any_to_int(args[1])
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.transpose1_property = self.add_input('dim 1', widget_type='input_int', default_value=self.transpose1, callback=self.transpose_changed)
        self.transpose2_property = self.add_input('dim 2', widget_type='input_int', default_value=self.transpose2, callback=self.transpose_changed)
        self.output = self.add_output('permuted tensor out')

    def transpose_changed(self, val=None):
        self.transpose1 = self.transpose1_property.get_widget_value()
        self.transpose2 = self.transpose2_property.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if len(input_tensor.shape) <= 1:
                self.output.send(input_tensor)
                if self.app.verbose:
                    print('WARNING: torch.transpose - too few dims to transpose')
            else:
                transposed = torch.transpose(input_tensor, self.transpose1, self.transpose2)
                self.output.send(transposed)

class CosineSimilarityNode(TorchNode):
    cos = None
    inited = False

    @staticmethod
    def factory(name, data, args=None):
        node = CosineSimilarityNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.vector_2 = None
        if not self.inited:
            self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            self.inited = True
        self.input = self.add_input("input 1", triggers_execution=True)
        self.input2 = self.add_input("input 2")
        self.output = self.add_output("output")

    def execute(self):
        if self.input2.fresh_input:
            self.vector_2 = self.data_to_tensor(self.input2.get_received_data())
        vector_1 = self.input_to_tensor()
        if self.vector_2 is not None and vector_1 is not None:
            similarity = self.cos(vector_1, self.vector_2)
            self.output.send(similarity.item())

class TorchLinalgRQNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchLinalgRQNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.mode = 'reduced'
        self.mode_property = self.add_property('mode', widget_type='combo', default_value=self.mode, callback=self.mode_changed)
        self.mode_property.widget.combo_items = ['reduced', 'complete', 'r']
        self.q_output = self.add_output('Q tensor out')
        self.r_output = self.add_output('R tensor out')

    def mode_changed(self, val='reduced'):
        self.mode = self.mode_property.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            q, r = torch.linalg.qr(input_tensor, self.mode)
            self.r_output.send(r)
            self.q_output.send(q)

class TorchLinalgSVDNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchLinalgSVDNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.full = True
        self.full_property = self.add_property('full', widget_type='checkbox', default_value=self.full, callback=self.full_changed)
        self.s_output = self.add_output('S tensor out')
        self.v_output = self.add_output('V tensor out')
        self.d_output = self.add_output('D tensor out')

    def full_changed(self, val='reduced'):
        self.full = self.full_property.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            s, v, d = torch.linalg.svd(input_tensor, self.full)
            self.d_output.send(d)
            self.v_output.send(v)
            self.s_output.send(s)

class TorchPCALowRankNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchPCALowRankNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.center = False
        self.center_property = self.add_property('center', widget_type='checkbox', default_value=self.center, callback=self.params_changed)
        self.niter = 2
        self.niter_property = self.add_property('full', widget_type='input_int', default_value=self.niter, callback=self.params_changed)
        self.u_output = self.add_output('U tensor out')
        self.s_output = self.add_output('S tensor out')
        self.v_output = self.add_output('V tensor out')

    def params_changed(self, val=2):
        self.niter = self.niter_property.get_widget_value()
        self.center = self.center_property.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            u, s, v = torch.pca.low_rank(input_tensor)
            self.v_output.send(v)
            self.s_output.send(s)
            self.u_output.send(u)

class TorchLinalgEigenNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchLinalgEigenNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.l_output = self.add_output('L tensor out')
        self.v_output = self.add_output('V tensor out')

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if len(input_tensor.shape) > 1:
                if input_tensor.shape[-1] == input_tensor.shape[-1]:
                    l, v = torch.linalg.eig(input_tensor)
                    self.v_output.send(v)
                    self.l_output.send(l)
                else:
                    if self.app.verbose:
                        print(self.label, 'tensor is not square')
            else:
                if self.app.verbose:
                    print(self.label, 'tensor has less than 2 dimensions')


class TorchWindowNode(TorchDeviceDtypeNode):
    op_dict = {'t.window.blackman': torch.signal.windows.blackman,
               't.window.bartlett': torch.signal.windows.bartlett,
               't.window.cosine': torch.signal.windows.cosine,
               't.window.hamming': torch.signal.windows.hamming,
               't.window.hann': torch.signal.windows.hann,
               't.window.nuttall': torch.signal.windows.nuttall
               }
    @staticmethod
    def factory(name, data, args=None):
        node = TorchWindowNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        self.m = 256
        self.sym = True
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.m_input = self.add_input('m', widget_type='drag_int', default_value=self.m, callback=self.m_changed)
        self.sym_input = self.add_input('sym', widget_type='checkbox', default_value=self.sym, callback=self.sym_changed)

        self.setup_dtype_device_grad(args)

        self.output = self.add_output('window tensor out')

    def m_changed(self, val=64):
        self.m = self.m_input.get_widget_value()

    def sym_changed(self, val=True):
        self.sym = self.sym_input.get_widget_value()

    def execute(self):
        window_tensor = self.op(self.m, sym=self.sym, dtype=self.dtype, device=self.device)
        self.output.send(window_tensor)


class TorchWindowOneParamNode(TorchDeviceDtypeNode):
    op_dict = {'t.window.gaussian': torch.signal.windows.gaussian,
               't.window.general_hamming': torch.signal.windows.general_hamming,
               't.window.kaiser': torch.signal.windows.kaiser
               }
    @staticmethod
    def factory(name, data, args=None):
        node = TorchWindowOneParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        param_1_name = 'std'
        self.param_1 = 1.0
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
            if self.label == 't.window.gaussian':
                param_1_name = 'std'
                self.param_1 = 1.0
            elif self.label == 't.window.general_hamming':
                param_1_name = 'alpha'
                self.param_1 = 0.54
            elif self.label == 't.window.kaiser':
                param_1_name = 'beta'
                self.param_1 = 12.0

        self.m = 256
        self.sym = True
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.m_input = self.add_input('m', widget_type='drag_int', default_value=self.m, callback=self.m_changed)
        self.param_input = self.add_input(param_1_name, widget_type='drag_float', default_value=self.param_1, callback=self.param_changed)
        self.sym_input = self.add_input('sym', widget_type='checkbox', default_value=self.sym, callback=self.sym_changed)

        self.setup_dtype_device_grad(args)

        self.output = self.add_output('window tensor out')

    def m_changed(self, val=64):
        self.m = self.m_input.get_widget_value()

    def sym_changed(self, val=True):
        self.sym = self.sym_input.get_widget_value()


    def param_changed(self, val=64):
        self.param_1 = self.param_input.get_widget_value()

    def execute(self):
        if self.label == 't.window.gaussian':
            window_tensor = self.op(self.m, std=self.param_1, sym=self.sym, dtype=self.dtype, device=self.device)
            self.output.send(window_tensor)
        elif self.label == 't.window.general_hamming':
            window_tensor = self.op(self.m, alpha=self.param_1, sym=self.sym, dtype=self.dtype, device=self.device)
            self.output.send(window_tensor)
        elif self.label == 't.window.kaiser':
            window_tensor = self.op(self.m, beta=self.param_1, sym=self.sym, dtype=self.dtype, device=self.device)
            self.output.send(window_tensor)

class TorchWindowTwoParamNode(TorchDeviceDtypeNode):
    op_dict = {'t.window.exponential': torch.signal.windows.exponential
               }
    @staticmethod
    def factory(name, data, args=None):
        node = TorchWindowTwoParamNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
            if self.label == 't.window.exponential':
                param_1_name = 'center'
                param_2_name = 'tau'
                self.param_1 = 1.0
                self.param_2 = 1.0

        self.m = 256
        self.sym = True
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.m_input = self.add_input('m', widget_type='drag_int', default_value=self.m, callback=self.m_changed)
        self.param_input = self.add_input(param_1_name, widget_type='drag_float', default_value=self.param_1, callback=self.param_changed)
        self.param_2_input = self.add_input(param_2_name, widget_type='drag_float', default_value=self.param_2, callback=self.param_changed)
        self.sym_input = self.add_input('sym', widget_type='checkbox', default_value=self.sym, callback=self.sym_changed)

        self.setup_dtype_device_grad(args)

        self.output = self.add_output('window tensor out')

    def m_changed(self, val=64):
        self.m = self.m_input.get_widget_value()

    def sym_changed(self, val=True):
        self.sym = self.sym_input.get_widget_value()

    def param_changed(self, val=64):
        self.param_1 = self.param_input.get_widget_value()
        self.param_2 = self.param_2_input.get_widget_value()
        if self.param_2 <= 0:
            self.param_2 = 0

    def execute(self):
        if self.sym:
            window_tensor = self.op(self.m, tau=self.param_2, sym=self.sym, dtype=self.dtype,
                                    device=self.device)
        else:
            window_tensor = self.op(self.m, center=self.param_1, tau=self.param_2, sym=self.sym, dtype=self.dtype, device=self.device)

        self.output.send(window_tensor)