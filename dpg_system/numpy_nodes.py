import dearpygui.dearpygui as dpg
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import time
import numpy as np


def register_numpy_nodes():
    Node.app.register_node('np.linalg.norm', NumpyUnaryLinearAlgebraNode.factory)
    Node.app.register_node('euclidean_distance', NumpyUnaryLinearAlgebraNode.factory)
    Node.app.register_node('np.linalg.det', NumpyUnaryLinearAlgebraNode.factory)
    Node.app.register_node('np.linalg.matrix_rank', NumpyUnaryLinearAlgebraNode.factory)
    Node.app.register_node('flatten', FlattenMatrixNode.factory)
    Node.app.register_node('np.ravel', FlattenMatrixNode.factory)
    Node.app.register_node('np.dot', NumpyDotProductNode.factory)
    Node.app.register_node('np.cross', NumpyCrossProductNode.factory)
    Node.app.register_node('np.sum', NumpyUnaryNode.factory)
    Node.app.register_node('np.mean', NumpyUnaryNode.factory)
    Node.app.register_node('np.std', NumpyUnaryNode.factory)
    Node.app.register_node('np.var', NumpyUnaryNode.factory)
    Node.app.register_node('np.median', NumpyUnaryNode.factory)
    Node.app.register_node('np.linalg.det', NumpyUnaryNode.factory)
    Node.app.register_node('np.shape', NumpyShapeNode.factory)
    Node.app.register_node('np.matmul', NumpyMatMulNode.factory)
    Node.app.register_node('np.stack', NumpyBinaryNode.factory)
    Node.app.register_node('np.concatenate', NumpyBinaryNode.factory)


class NumpyUnaryNode(Node):
    operations = {'np.sum': np.sum, 'np.mean': np.mean, 'np.std': np.std, 'np.var': np.var, 'np.median': np.median}
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyUnaryNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.axis = 0
        self.op = np.sum
        if label in self.operations:
            self.op = self.operations[label]
            print(self.op)
        if len(args) > 0:
            d, t = decode_arg(args, 0)
            if t == int:
                self.axis = d
        output_name = label.split('.')[0]
        self.input = self.add_input('in', triggers_execution=True)
        self.output = self.add_output(output_name)

    def execute(self):
        input_value = any_to_array(self.input.get_received_data())
        if len(input_value.shape) > self.axis:
            output_value = self.op(input_value, axis=self.axis)
            self.output.send(output_value)
        else:
            print(self.label, 'dim =', self.axis, 'out of range', 'for shape', input_value.shape)


class NumpyBinaryNode(Node):
    operations = {'np.stack': np.stack, 'np.concatenate': np.concatenate}
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyBinaryNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.op = np.stack
        self.input_2_vector = None
        self.axis = self.arg_as_int(default_value=0)
        if label in self.operations:
            self.op = self.operations[label]
        output_name = label.split('.')[0]
        self.input = self.add_input('in', triggers_execution=True)
        self.input_2 = self.add_input('in 2')
        self.output = self.add_output(output_name)

    def execute(self):
        if self.input_2.fresh_input:
            self.input_2_vector = any_to_array(self.input_2.get_received_data())

        input_value = any_to_array(self.input.get_received_data())
        if self.input_2_vector is not None:
            self.output.send(self.op((input_value, self.input_2_vector), axis=self.axis))


class NumpyDotProductNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyDotProductNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.operand = 0
        self.input = self.add_input('in 1', triggers_execution=True)
        self.operand_input = self.add_input('in 2')
        self.output = self.add_output("dot_product")

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = any_to_array(self.operand_input.get_received_data())
        input_value = any_to_array(self.input.get_received_data())

        output_value = np.dot(input_value, self.operand)
        self.output.send(output_value)


class NumpyMatMulNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyMatMulNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.operand = 0
        self.input = self.add_input('in 1', triggers_execution=True)
        self.operand_input = self.add_input('in 2')
        self.output = self.add_output("dot_product")

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = any_to_array(self.operand_input.get_received_data())
        input_value = any_to_array(self.input.get_received_data())

        output_value = np.matmul(input_value, self.operand)
        self.output.send(output_value)


class NumpyCrossProductNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyCrossProductNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.operand = 0
        self.input = self.add_input('in 1', triggers_execution=True)
        self.operand_input = self.add_input('in 2')
        self.output = self.add_output("dot_product")

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = any_to_array(self.operand_input.get_received_data())
        input_value = any_to_array(self.input.get_received_data())

        output_value = np.cross(input_value, self.operand)
        self.output.send(output_value)


class FlattenMatrixNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = FlattenMatrixNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.order = 'C'
        if len(args) > 0:
            order, t = decode_arg(args, 0)
            if t == str and order in ['C', 'F', 'A', 'K']:
                self.order = order
        self.input = self.add_input("input", triggers_execution=True)
        self.output = self.add_output("output")

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input.get_received_data())
            data = np.ravel(data, order=self.order)
            self.output.send(data)


class NumpyUnaryLinearAlgebraNode(Node):
    operations = {'np.linalg.norm': np.linalg.norm, 'euclidean_distance': np.linalg.norm, 'np.linalg.det': np.linalg.det, 'np.linalg.matrix_rank': np.linalg.matrix_rank}

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyUnaryLinearAlgebraNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = np.linalg.norm
        if label in self.operations:
            self.op = self.operations[label]
        self.input = self.add_input("input", triggers_execution=True)
        self.output = self.add_output("output")

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input.get_received_data())
            self.output.send(self.op(data))


class NumpyShapeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyShapeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input("np in", triggers_execution=True)
        self.output = self.add_output("shape")

    def execute(self):
        if self.input.fresh_input:
            data = self.input.get_received_data()
            data = any_to_array(data)
            shape = list(data.shape)
            self.output.send(shape)
