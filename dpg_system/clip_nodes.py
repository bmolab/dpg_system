import dearpygui.dearpygui as dpg
import math
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *

import torch
from transformers import CLIPTokenizer, CLIPTextModel


# REQUIRES hugging face transformers and pytorch


def register_clip_nodes():
    Node.app.register_node('clip_embedding', ClipEmbeddingNode.factory)
    Node.app.register_node('clip_embedding_length', ClipEmbeddingDistanceNode.factory)
    Node.app.register_node('cosine_similarity', CosineSimilarityNode.factory)


class ClipNode(Node):
    model = None
    tokenizer = None
    inited = False

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if not self.__class__.inited:
            self.__class__.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.__class__.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.__class__.inited = True


class ClipEmbeddingNode(ClipNode):
    @staticmethod
    def factory(name, data, args=None):
        node = ClipEmbeddingNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input("input", triggers_execution=True)
        self.output = self.add_output("output")

    def execute(self):
        input = any_to_string(self.input.get_received_data())
        with torch.no_grad():
            tokens = self.__class__.tokenizer([input], padding=True, return_tensors="pt")
            outputs = self.__class__.model(**tokens)
            pooled_output = outputs.pooler_output
            embedding = pooled_output[0].numpy()
        self.output.send(embedding)


class ClipEmbeddingDistanceNode(ClipNode):
    @staticmethod
    def factory(name, data, args=None):
        node = ClipEmbeddingDistanceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input("input", triggers_execution=True)
        self.output = self.add_output("output")

    def execute(self):
        input = any_to_string(self.input.get_received_data())
        tokens = self.__class__.tokenizer([input], padding=True, return_tensors="pt")
        outputs = self.__class__.model(**tokens)
        pooled_output = outputs.pooler_output
        euclidean_length = torch.pow(pooled_output[0], 2).sum()

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
