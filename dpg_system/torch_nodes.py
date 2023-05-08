import dearpygui.dearpygui as dpg
import math
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *

import torch
import torchvision

def register_torch_nodes():
    Node.app.torch_available = True
    Node.app.register_node('torch.cdist', TorchCDistanceNode.factory)
    Node.app.register_node('torch.dist', TorchDistanceNode.factory)
    Node.app.register_node('torch.length', TorchDistanceNode.factory)
    Node.app.register_node('cosine_similarity', CosineSimilarityNode.factory)
    Node.app.register_node('torch.rand', TorchGeneratorNode.factory)
    Node.app.register_node('torch.ones', TorchGeneratorNode.factory)
    Node.app.register_node('torch.zeros', TorchGeneratorNode.factory)
    Node.app.register_node('tensor', TensorNode.factory)
    Node.app.register_node('torch.permute', TorchPermuteNode.factory)
    Node.app.register_node('torch.flip', TorchFlipNode.factory)
    Node.app.register_node('torch[]', TorchSubtensorNode.factory)
    Node.app.register_node('torch.select', TorchSelectNode.factory)
    Node.app.register_node('torch.squeeze', TorchSqueezeNode.factory)
    Node.app.register_node('torch.unsqueeze', TorchUnsqueezeNode.factory)
    Node.app.register_node('torch.nn.Threshold', TorchNNThresholdNode.factory)
    Node.app.register_node('torch.nn.relu', TorchActivationNode.factory)
    Node.app.register_node('torch.nn.hardswish', TorchActivationNode.factory)
    Node.app.register_node('torch.nn.relu6', TorchActivationNode.factory)
    Node.app.register_node('torch.nn.selu', TorchActivationNode.factory)
    Node.app.register_node('torch.nn.glu', TorchActivationNode.factory)
    Node.app.register_node('torch.nn.gelu', TorchActivationNode.factory)
    Node.app.register_node('torch.nn.logsigmoid', TorchActivationNode.factory)
    Node.app.register_node('torch.nn.tanhshrink', TorchActivationNode.factory)
    Node.app.register_node('torch.nn.softsign', TorchActivationNode.factory)
    Node.app.register_node('torch.nn.tanh', TorchActivationNode.factory)
    Node.app.register_node('torch.nn.sigmoid', TorchActivationNode.factory)
    Node.app.register_node('torch.nn.hardsigmoid', TorchActivationNode.factory)
    Node.app.register_node('torch.nn.silu', TorchActivationNode.factory)
    Node.app.register_node('torch.nn.mish', TorchActivationNode.factory)
    Node.app.register_node('torch.nn.elu', TorchActivationOneParamNode.factory)
    Node.app.register_node('torch.nn.celu', TorchActivationOneParamNode.factory)
    Node.app.register_node('torch.nn.leaky_relu', TorchActivationOneParamNode.factory)
    # Node.app.register_node('torch.nn.prelu', TorchActivationOneParamNode.factory)
    Node.app.register_node('torch.nn.hardshrink', TorchActivationOneParamNode.factory)
    Node.app.register_node('torch.nn.softshrink', TorchActivationOneParamNode.factory)
    Node.app.register_node('torch.nn.hardtanh', TorchActivationTwoParamNode.factory)
    Node.app.register_node('torch.nn.rrelu', TorchActivationTwoParamNode.factory)
    Node.app.register_node('torch.nn.softplus', TorchActivationTwoParamNode.factory)

    Node.app.register_node('torch.nn.softmax', TorchSoftmaxNode.factory)
    Node.app.register_node('torch.nn.softmin', TorchSoftmaxNode.factory)
    Node.app.register_node('torch.nn.log_softmax', TorchSoftmaxNode.factory)

    Node.app.register_node('torch.nn.gumbel_softmax', TorchActivationThreeParamNode.factory)
    Node.app.register_node('torch.linspace', TorchLinSpaceNode.factory)

    Node.app.register_node('torch.argmax', TorchArgMaxNode.factory)

    Node.app.register_node('torchvision.Grayscale', TorchvisionGrayscaleNode.factory)
    Node.app.register_node('torchvision.gaussian_blur', TorchvisionGaussianBlurNode.factory)
    Node.app.register_node('torchvision.adjust_hue', TorchvisionAdjustOneParamNode.factory)
    Node.app.register_node('torchvision.adjust_saturation', TorchvisionAdjustOneParamNode.factory)
    Node.app.register_node('torchvision.adjust_contrast', TorchvisionAdjustOneParamNode.factory)
    Node.app.register_node('torchvision.adjust_sharpness', TorchvisionAdjustOneParamNode.factory)
    Node.app.register_node('torchvision.adjust_brightness', TorchvisionAdjustOneParamNode.factory)




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


class TensorNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TensorNode(name, data, args)
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
        self.device = torch.device('cpu')
        self.device_property = self.add_property('device', widget_type='combo', default_value='cpu', callback=self.device_changed)
        self.device_property.widget.combo_items = ['cpu']
        if torch.backends.mps.is_available():
            self.device_property.widget.combo_items.append('mps')
        if torch.cuda.is_available():
            self.device_property.widget.combo_items.append('cuda')
        self.output = self.add_output('tensor out')

        self.shape_property = self.add_option('shape', widget_type='text_input', default_value=shape_text, callback=self.shape_changed)

    def shape_changed(self):
        shape_text = self.shape_property.get_widget_value()
        shape_split = re.findall(r'\d+', shape_text)
        shape_list, _, _ = list_to_hybrid_list(shape_split)
        self.shape = shape_list

    def device_changed(self):
        device_name = self.device_property.get_widget_value()
        self.device = torch.device(device_name)

    def execute(self):
        in_data = self.input_to_tensor()
        if in_data is not None:
            out_array = any_to_tensor(in_data, self.device)
            if self.shape is not None:
                out_array = torch.reshape(out_array, tuple(self.shape))
            self.output.send(out_array)

class TorchGeneratorNode(TorchNode):
    # operations = {'np.rand': np.random.Generator.random, 'np.ones': np.ones, 'np.zeros': np.zeros}

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
        self.device = torch.device('cpu')
        self.device_property = self.add_property('device', widget_type='combo', default_value='cpu', callback=self.device_changed)
        self.device_property.widget.combo_items = ['cpu']
        if torch.backends.mps.is_available():
            self.device_property.widget.combo_items.append('mps')
        if torch.cuda.is_available():
            self.device_property.widget.combo_items.append('cuda')

        if self.label == 'torch.rand':
            self.min = 0
            self.max = 1
            self.min_input = self.add_input('min', widget_type='drag_float', default_value=self.min,
                                            callback=self.range_changed)
            self.max_input = self.add_input('max', widget_type='drag_float', default_value=self.max,
                                            callback=self.range_changed)

        self.dtype_dict = {}
        self.dtype_dict['float32'] = torch.float32
        self.dtype_dict['float'] = torch.float
        self.dtype_dict['float16'] = torch.float16
        self.dtype_dict['bfloat16'] = torch.bfloat16
        self.dtype_dict['double'] = torch.double
        self.dtype_dict['int64'] = torch.int64
        self.dtype_dict['uint8'] = torch.uint8
        self.dtype_dict['bool'] = torch.bool
        self.dtype_option = self.add_option('dtype', widget_type='combo', default_value='float32', callback=self.dtype_changed)
        self.dtype_option.widget.combo_items = list(self.dtype_dict.keys())
        self.dtype = torch.float32

        out_label = 'random tensor'
        if self.label == 'torch.ones':
            out_label = 'tensor of ones'
        elif self.label == 'torch.zeros':
            out_label = 'tensor of zeros'
        self.output = self.add_output(out_label)

    def device_changed(self):
        device_name = self.device_property.get_widget_value()
        self.device = torch.device(device_name)

    def range_changed(self, val=None):
        self.min = self.min_input.get_widget_value()
        self.max = self.max_input.get_widget_value()

    def dtype_changed(self):
        dtype = self.dtype_option.get_widget_value()
        if dtype in self.dtype_dict:
            self.dtype = self.dtype_dict[dtype]
            if self.label == 'torch.rand':
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
        if self.label == 'torch.rand':
            if self.dtype in [torch.float, torch.float32, torch.double, torch.float16, torch.bfloat16]:
                range_ = self.max - self.min
                out_array = torch.rand(size=size, device=self.device, dtype=self.dtype) * range_ + self.min
            elif self.dtype in [torch.int64, torch.uint8]:
                out_array = torch.randint(low=int(self.min), high=int(self.max), size=size, device=self.device, dtype=self.dtype)
            elif self.dtype == torch.bool:
                out_array = torch.randint(low=0, high=1, size=size, device=self.device, dtype=self.dtype)
        elif self.label == 'torch.ones':
            out_array = torch.ones(size=size, device=self.device, dtype=self.dtype)
        elif self.label == 'torch.zeros':
            out_array = torch.zeros(size=size, device=self.device, dtype=self.dtype)
        self.output.send(out_array)


class TorchLinSpaceNode(TorchNode):
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
        if len(args) > 0:
            d, t = decode_arg(args, 0)
            self.start = any_to_float(d)
        if len(args) > 1:
            d, t = decode_arg(args, 1)
            self.stop = any_to_float(d)
        if len(args) > 2:
            d, t = decode_arg(args, 2)
            self.steps = any_to_int(d)

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.start_property = self.add_property('start', widget_type='drag_float', default_value=self.start)
        self.stop_property = self.add_property('stop', widget_type='drag_float', default_value=self.stop)
        self.steps_property = self.add_property('steps', widget_type='drag_int', default_value=self.steps)
        self.dtype_dict = {}
        self.dtype_dict['float32'] = torch.float32
        self.dtype_dict['float'] = torch.float
        self.dtype_dict['float16'] = torch.float16
        self.dtype_dict['bfloat16'] = torch.bfloat16
        self.dtype_dict['double'] = torch.double
        self.dtype_dict['int64'] = torch.int64
        self.dtype_dict['uint8'] = torch.uint8
        self.dtype_dict['bool'] = torch.bool
        self.dtype_option = self.add_property('dtype', widget_type='combo', default_value='float32', callback=self.dtype_changed)
        self.dtype_option.widget.combo_items = list(self.dtype_dict.keys())
        self.dtype = torch.float32
        self.device = torch.device('cpu')
        self.device_property = self.add_property('device', widget_type='combo', default_value='cpu', callback=self.device_changed)
        self.device_property.widget.combo_items = ['cpu']
        if torch.backends.mps.is_available():
            self.device_property.widget.combo_items.append('mps')
        if torch.cuda.is_available():
            self.device_property.widget.combo_items.append('cuda')
        self.output = self.add_output('linspace out')

    def device_changed(self):
        device_name = self.device_property.get_widget_value()
        self.device = torch.device(device_name)

    def dtype_changed(self):
        dtype = self.dtype_option.get_widget_value()
        if dtype in self.dtype_dict:
            self.dtype = self.dtype_dict[dtype]

    def execute(self):
        self.start = self.start_property.get_widget_value()
        self.stop = self.stop_property.get_widget_value()
        self.steps = self.steps_property.get_widget_value()
        out_array = torch.linspace(self.start, self.stop, self.steps, dtype=self.dtype, device=self.device)
        self.output.send(out_array)

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
        if self.label == 'torch.dist':
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

class TorchArgMaxNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchArgMaxNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.dim_specified = False
        self.dim = 0
        if len(args) > 0:
            self.dim = any_to_int(args[0])
            self.dim_specified = True
        self.input = self.add_input("tensor in", triggers_execution=True)
        if self.dim_specified:
            self.dim_input = self.add_input('dim', widget_type='input_int', default_value=self.dim, callback=self.dim_changed)
        self.output = self.add_output("max index")

    def dim_changed(self, val=None):
        self.dim = self.dim_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if self.dim_specified:
                output_tensor = torch.argmax(input_tensor, dim=self.dim)
            else:
                output_tensor = torch.argmax(input_tensor)
            print(output_tensor)
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

class TorchSqueezeNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchSqueezeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.dim_specified = False
        self.dim = 0
        if len(args) > 0:
            self.dim = any_to_int(args[0])
            self.dim_specified = True
        self.input = self.add_input("tensor in", triggers_execution=True)
        if self.dim_specified:
            self.dim_input = self.add_input('dim', widget_type='input_int', default_value=self.dim, callback=self.dim_changed)
        self.output = self.add_output("output")

    def dim_changed(self, val=None):
        self.dim = self.dim_input.get_widget_value()

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if -1 - len(input_tensor.shape) < self.dim < len(input_tensor.shape):
                if self.dim_specified:
                    self.output.send(torch.squeeze(input_tensor, self.dim))
                else:
                    self.output.send(torch.squeeze(input_tensor))
                return

class TorchUnsqueezeNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchUnsqueezeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.dim = 0
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
            if -1 - len(input_tensor.shape) < self.dim < len(input_tensor.shape):
                self.output.send(torch.unsqueeze(input_tensor, self.dim))
                return

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


class TorchActivationNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchActivationNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = None
        if self.label == 'torch.nn.relu':
            self.op = torch.nn.functional.relu
        elif self.label == 'torch.nn.hardswish':
            self.op = torch.nn.functional.hardswish
        elif self.label == 'torch.nn.hardtanh':
            self.op = torch.nn.functional.hardtanh
        elif self.label == 'torch.nn.relu6':
            self.op = torch.nn.functional.relu6
        elif self.label == 'torch.nn.selu':
            self.op = torch.nn.functional.selu
        elif self.label == 'torch.nn.glu':
            self.op = torch.nn.functional.glu
        elif self.label == 'torch.nn.gelu':
            self.op = torch.nn.functional.gelu
        elif self.label == 'torch.nn.logsigmoid':
            self.op = torch.nn.functional.logsigmoid
        elif self.label == 'torch.nn.tanhshrink':
            self.op = torch.nn.functional.tanhshrink
        elif self.label == 'torch.nn.softsign':
            self.op = torch.nn.functional.softsign
        elif self.label == 'torch.nn.tanh':
            self.op = torch.nn.functional.tanh
        elif self.label == 'torch.nn.sigmoid':
            self.op = torch.nn.functional.sigmoid
        elif self.label == 'torch.nn.hardsigmoid':
            self.op = torch.nn.functional.hardsigmoid
        elif self.label == 'torch.nn.silu':
            self.op = torch.nn.functional.silu
        elif self.label == 'torch.nn.mish':
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

        if self.label == 'torch.nn.hardtanh':
            self.op = torch.nn.functional.hardtanh
        if self.label == 'torch.nn.rrelu':
            self.op = torch.nn.functional.rrelu
            param_1_name = 'lower'
            param_2_name = 'upper'
            self.parameter_1 = 0.125
            self.parameter_2 = 0.3333333333333
        if self.label == 'torch.nn.softplus':
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

        if self.label == 'torch.nn.gumbel_softmax':
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

        if self.label == 'torch.nn.softmax':
            self.op = torch.nn.functional.softmax
        if self.label == 'torch.nn.softmin':
            self.op = torch.nn.functional.softmin
        if self.label == 'torch.nn.log_softmax':
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
        if self.label == 'torch.nn.elu':
            self.op = torch.nn.functional.elu
            param_name = 'alpha'
            self.parameter = 1.0
        elif self.label == 'torch.nn.celu':
            self.op = torch.nn.functional.celu
            param_name = 'alpha'
            self.parameter = 1.0
        elif self.label == 'torch.nn.leaky_relu':
            self.op = torch.nn.functional.leaky_relu
            param_name = 'negative slope'
            self.parameter = 0.01
        # elif self.label == 'torch.nn.prelu':
        #     self.op = torch.nn.functional.prelu
        #     param_name = 'weight'
        #     self.parameter = 1.0
        elif self.label == 'torch.nn.hardshrink':
            self.op = torch.nn.functional.hardshrink
            param_name = 'lambda'
            self.parameter = 0.5
        elif self.label == 'torch.nn.softshrink':
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
        # input_tensor = self.input.get_received_data()
        # if type(input_tensor) != torch.Tensor:
        #     input_tensor = any_to_tensor(self.input.get_received_data())
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
        # input_tensor = self.input.get_received_data()
        # if type(input_tensor) != torch.Tensor:
        #     input_tensor = any_to_tensor(self.input.get_received_data())
        # if len(input_tensor.shape) > 2:
        #     if input_tensor.shape[-3] > 5:
        #         if input_tensor.shape[-1] <= 5:
        #             input_tensor = input_tensor.transpose(-1, -3).transpose(-1, -2)
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
        # input_tensor = self.input.get_received_data()
        # if type(input_tensor) != torch.Tensor:
        #     input_tensor = any_to_tensor(self.input.get_received_data())
        # if len(input_tensor.shape) > 2:
        #     if input_tensor.shape[-3] > 5:
        #         if input_tensor.shape[-1] <= 5:
        #             input_tensor = input_tensor.transpose(-1, -3).transpose(-1, -2)
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
        #
        # if type(input_tensor) != torch.Tensor:
        #     input_tensor = any_to_tensor(self.input.get_received_data())
        # if len(input_tensor.shape) > 2:
        #     if input_tensor.shape[-3] > 5:
        #         if input_tensor.shape[-1] <= 5:
        #             input_tensor = input_tensor.transpose(-1, -3).transpose(-1, -2)
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
        self.permute = [1, 0]
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.permute_property = self.add_property('permute', widget_type='text_input', default_value='1, 0', callback=self.permute_changed)
        self.output = self.add_output('permuted tensor out')

    def permute_changed(self):
        permute_text = self.permute_property.get_widget_value()
        permute_split = re.findall(r'\d+', permute_text)
        permute_list, _, _ = list_to_hybrid_list(permute_split)
        self.permute = permute_list

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is not None:
            if len(input_tensor.shape) < len(self.permute):
                self.output.send(input_tensor)
            else:
                permuted = torch.permute(input_tensor, self.permute)
                self.output.send(permuted)

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
