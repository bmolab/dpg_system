import dearpygui.dearpygui as dpg
from dpg_system.node import Node
from dpg_system.conversion_utils import *

import torch

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
        self.device_input = None
        self.dtype_input = None
        self.dtype_dict = self.create_dtype_dict()
        self.device_list = self.create_device_list()
        self.device = torch.device(self.device_string)
        self.dtype = self.dtype_dict[self.dtype_string]
        self.grad_input = None
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
        self.device_input = self.add_input('device', widget_type='combo', default_value=self.device_string,
                                                 callback=self.device_changed)
        self.device_input.widget.combo_items = self.device_list

    def create_dtype_property(self):
        self.dtype_input = self.add_input('dtype', widget_type='combo', widget_width=120, default_value=self.dtype_string,
                                              callback=self.dtype_changed)
        self.dtype_input.widget.combo_items = list(self.dtype_dict.keys())

    def create_requires_grad_property(self):
        self.grad_input = self.add_input('requires_grad', widget_type='checkbox', default_value=self.requires_grad,
                                              callback=self.requires_grad_changed)

    def device_changed(self):
        device_name = self.device_input.get_widget_value()
        self.device = torch.device(device_name)

    def requires_grad_changed(self):
        self.requires_grad = self.grad_input.get_widget_value()

    def dtype_changed(self):
        dtype = self.dtype_input.get_widget_value()
        if dtype in self.dtype_dict:
            self.dtype = self.dtype_dict[dtype]

    def create_dtype_dict(self):
        dtype_dict = {}
        dtype_dict['float32'] = torch.float32
        dtype_dict['torch.float32'] = torch.float32
        dtype_dict['float'] = torch.float
        dtype_dict['torch.float'] = torch.float
        dtype_dict['float16'] = torch.float16
        dtype_dict['torch.float16'] = torch.float16
        # dtype_dict['bfloat16'] = torch.bfloat16
        # dtype_dict['double'] = torch.double
        dtype_dict['int64'] = torch.int64
        dtype_dict['torch.int64'] = torch.int64
        dtype_dict['uint8'] = torch.uint8
        dtype_dict['torch.uint8'] = torch.uint8
        dtype_dict['bool'] = torch.bool
        dtype_dict['torch.bool'] = torch.bool
        # dtype_dict['complex32'] = torch.complex32
        dtype_dict['complex64'] = torch.complex64
        dtype_dict['torch.complex64'] = torch.complex64
        dtype_dict['complex128'] = torch.complex128
        dtype_dict['torch.complex128'] = torch.complex128
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



class TorchInfoNode(TorchNode):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchInfoNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input("in", triggers_execution=True)
        self.shape_output = self.add_output('shape')
        self.dtype_output = self.add_output('dtype')
        self.device_output = self.add_output('device')
        self.requires_grad_output = self.add_output('grad')

    def execute(self):
        in_data = self.input_to_tensor()
        if in_data is not None:
            rg = 'inference'
            if in_data.requires_grad:
                rg = 'grad'
            self.requires_grad_output.set_label(rg)
            self.requires_grad_output.send(in_data.requires_grad)

            self.device_output.set_label(str(in_data.device))
            self.device_output.send(str(in_data.device))

            self.dtype_output.set_label(str(in_data.dtype))
            self.dtype_output.send(str(in_data.dtype))

            self.shape_output.set_label(str(list(in_data.shape)))
            self.shape_output.send(list(in_data.shape))



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

    def dim_changed(self):
        if self.dim_input is not None:
            self.dim = self.dim_input.get_widget_value()

