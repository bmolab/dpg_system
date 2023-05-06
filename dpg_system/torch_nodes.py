import dearpygui.dearpygui as dpg
import math
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *

import torch


def register_torch_nodes():
    Node.app.register_node('torch.cdist', TorchCDistanceNode.factory)
    Node.app.register_node('torch.dist', TorchDistanceNode.factory)
    Node.app.register_node('torch.length', TorchDistanceNode.factory)
    Node.app.register_node('cosine_similarity', CosineSimilarityNode.factory)
    Node.app.register_node('torch.rand', TorchGeneratorNode.factory)
    Node.app.register_node('torch.ones', TorchGeneratorNode.factory)
    Node.app.register_node('torch.zeros', TorchGeneratorNode.factory)
    Node.app.register_node('tensor', TensorNode.factory)
    Node.app.register_node('torch.permute', TorchPermuteNode.factory)
    Node.app.register_node('torch.flio', TorchFlipNode.factory)

class TensorNode(Node):
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
        in_data = self.input.get_received_data()
        out_array = any_to_tensor(in_data, self.device)
        if self.shape is not None:
            out_array = torch.reshape(out_array, tuple(self.shape))
        self.output.send(out_array)

class TorchGeneratorNode(Node):
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

        out_label = 'random tensor'
        if self.label == 'torch.ones':
            out_label = 'tensor of ones'
        elif self.label == 'torch.zeros':
            out_label = 'tensor of zeros'
        self.output = self.add_output(out_label)

    def device_changed(self):
        device_name = self.device_property.get_widget_value()
        self.device = torch.device(device_name)
    def execute(self):
        for i in range(len(self.shape)):
            self.shape[i] = self.shape_properties[i].get_widget_value()
        size = tuple(self.shape)
        if self.label == 'torch.rand':
            out_array = torch.rand(size=size, dtype=torch.float32, device=self.device)
        elif self.label == 'torch.ones':
            out_array = torch.ones(size=size, dtype=torch.float32, device=self.device)
        elif self.label == 'torch.zeros':
            out_array = torch.zeros(size=size, dtype=torch.float32, device=self.device)
        self.output.send(out_array)

class TorchCDistanceNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = TorchCDistanceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input("input", triggers_execution=True)
        self.output = self.add_output("output")

    def execute(self):
        input_ = self.input.get_received_data()
        input_tensor = torch.tensor(input_)
        input_tensor = input_tensor.unsqueeze(dim=0)
        euclidean_length = torch.cdist(input_tensor, torch.zeros_like(input_tensor))

        self.output.send(euclidean_length.item())

class TorchDistanceNode(Node):
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

        self.output = self.add_output("output")

    def execute(self):
        input_tensor = self.input.get_received_data()
        if self.input2 is not None:
            input2 = self.input2.get_received_data()
            if input2 is not None:
                euclidean_length = torch.dist(input_tensor, torch.zeros_like(input_tensor))
            else:
                euclidean_length = torch.dist(input_tensor, input_tensor)
        else:
            euclidean_length = torch.dist(input_tensor, torch.zeros_like(input_tensor))

        self.output.send(euclidean_length.item())

class TorchFlipNode(Node):
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
        input_tensor = self.input.get_received_data()
        if len(input_tensor.shape) < len(self.flip_list) or len(self.flip_list) == 0:
            self.output.send(input_tensor)
        else:
            permuted = torch.flip(input_tensor, self.flip_list)
            self.output.send(permuted)


class TorchPermuteNode(Node):
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
        input_tensor = self.input.get_received_data()
        if len(input_tensor.shape) < len(self.permute):
            self.output.send(input_tensor)
        else:
            permuted = torch.permute(input_tensor, self.permute)
            self.output.send(permuted)

class CosineSimilarityNode(Node):
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
        self.input1 = self.add_input("input 1", triggers_execution=True)
        self.input2 = self.add_input("input 2")
        self.output = self.add_output("output")

    def execute(self):
        if self.input2.fresh_input:
            self.vector_2 = torch.tensor(any_to_array(self.input2.get_received_data()))
        vector_1 = torch.tensor(any_to_array(self.input1.get_received_data()))
        if self.vector_2 is not None:
            similarity = self.cos(vector_1, self.vector_2)

            self.output.send(similarity.item())
