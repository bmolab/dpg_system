import dearpygui.dearpygui as dpg
import math
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *

import torch


def register_torch_nodes():
    Node.app.register_node('torch.cdist', TorchCDistanceNode.factory)
    Node.app.register_node('cosine_similarity', CosineSimilarityNode.factory)


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
