import dearpygui.dearpygui as dpg
import math
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *


def register_math_nodes():
    Node.app.register_node("+", ArithmeticNode.factory)
    Node.app.register_node("-", ArithmeticNode.factory)
    Node.app.register_node("!-", ArithmeticNode.factory)
    Node.app.register_node("*", ArithmeticNode.factory)
    Node.app.register_node("/", ArithmeticNode.factory)
    Node.app.register_node("!/", ArithmeticNode.factory)
    Node.app.register_node("min", ArithmeticNode.factory)
    Node.app.register_node("max", ArithmeticNode.factory)
    Node.app.register_node("mod", ArithmeticNode.factory)
    Node.app.register_node("%", ArithmeticNode.factory)
    Node.app.register_node("^", ArithmeticNode.factory)
    Node.app.register_node("pow", ArithmeticNode.factory)
    Node.app.register_node("sin", OpSingleTrigNode.factory)
    Node.app.register_node("cos", OpSingleTrigNode.factory)
    Node.app.register_node("asin", OpSingleTrigNode.factory)
    Node.app.register_node("acos", OpSingleTrigNode.factory)
    Node.app.register_node("tan", OpSingleTrigNode.factory)
    Node.app.register_node("atan", OpSingleTrigNode.factory)
    Node.app.register_node("log10", OpSingleNode.factory)
    Node.app.register_node("log2", OpSingleNode.factory)
    Node.app.register_node("exp", OpSingleNode.factory)
    Node.app.register_node("inverse", OpSingleNode.factory)
    Node.app.register_node("abs", OpSingleNode.factory)
    Node.app.register_node("sqrt", OpSingleNode.factory)
    Node.app.register_node("norm", OpSingleNode.factory)
    Node.app.register_node(">", ComparisonNode.factory)
    Node.app.register_node(">=", ComparisonNode.factory)
    Node.app.register_node("==", ComparisonNode.factory)
    Node.app.register_node("!=", ComparisonNode.factory)
    Node.app.register_node("<", ComparisonNode.factory)
    Node.app.register_node("<=", ComparisonNode.factory)


class ArithmeticNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ArithmeticNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.operand = 0

        self.input = self.add_input('in', trigger_node=self)
        self.operand_input = self.add_input('', widget_type='drag_float')
        self.operand_input.add_callback(self.change_operand)
        self.output = self.add_output("")

        if label == '+':
            self.operation = self.add
        elif label == '-':
            self.operation = self.subtract
        elif label == '!-':
            self.operation = self.inverse_subtract
        elif label == '*':
            self.operation = self.multiply
        elif label == '/':
            self.operation = self.divide
        elif label == '//':
            self.operation = self.int_divide
        elif label == '!/':
            self.operation = self.inverse_divide
        elif label == 'pow':
            self.operation = self.power
        elif label == '^':
            self.operation = self.power
        elif label == 'min':
            self.operation = self.min
        elif label == 'max':
            self.operation = self.max
        elif label == 'mod':
            self.operation = self.mod
        elif label == '%':
            self.operation = self.mod

    def mod(self, a, b):
        if b == 0:
            return 0
        return a % b

    def min(self, a, b):
        if a > b:
            return b
        return a

    def max(self, a, b):
        if a < b:
            return b
        return a

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def inverse_subtract(self, a, b):
        return b - a

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            return a / 1e-8
        return a / b

    def int_divide(self, a, b):
        if b == 0:
            return a // 1e-8
        return a // b

    def inverse_divide(self, a, b):
        if a == 0:
            return b / 1e-8
        return b / a

    def power(self, a, b):
        return math.pow(a, b)

    def custom(self):
        in_value, t = decode_arg(self.args, 0)
        if t in [int, float]:
            self.operand_input.widget.set(in_value)

    def change_operand(self):
        self.operand = self.operand_input.get_widget_value()

    def execute(self):
        if self.operand_input.fresh_input:
            if type(self.operand_input.data) == np.ndarray:
                self.operand = self.operand_input.get_received_data()
            else:
                self.operand = self.operand_input.get_widget_value()

        input_value = self.input.get_received_data()
        t = type(input_value)
        if t == list:
            input_value = list_to_array(input_value)

        output_value = self.operation(input_value, self.operand)
        self.output.set(output_value)
        self.send_outputs()


class ComparisonNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ComparisonNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.operand = 0

        self.input = self.add_input('in', trigger_node=self)
        self.operand_input = self.add_input('', widget_type='drag_float')
        self.operand_input.add_callback(self.change_operand)
        self.output = self.add_output("")

        if label == '>':
            self.operation = self.greater
        elif label == '>=':
            self.operation = self.greater_equal
        elif label == '==':
            self.operation = self.equal
        elif label == '<':
            self.operation = self.less
        elif label == '<=':
            self.operation = self.less_equal
        elif label == '!=':
            self.operation = self.not_equal

    def greater(self, a, b):
        return a > b

    def greater_equal(self, a, b):
        return a >= b

    def less(self, a, b):
        return a < b

    def less_equal(self, a, b):
        return a <= b

    def equal(self, a, b):
        return a == b

    def not_equal(self, a, b):
        return a != b

    def change_operand(self):
        self.operand = self.operand_input.get_widget_value()

    def custom(self):
        in_value, t = decode_arg(self.args, 0)
        if t in [int, float]:
            self.operand_input.widget.set(in_value)
            self.operand = in_value

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = self.operand_input.get_received_data()
        input_value = self.input.get_received_data()

        t = type(input_value)
        if t == np.ndarray:
            output_value = self.np_operation(input_value, self.operand)
        elif t == list:
            a = list_to_array(input_value)
            output_value = self.np_operation(a, self.operand)
        else:
            output_value = self.operation(input_value, self.operand)

        self.output.send(output_value)


class OpSingleNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OpSingleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('in', trigger_node=self)
        self.output = self.add_output("")

        if label == 'log10':
            self.operation = self.log10
            self.np_operation = self.np_log10
        elif label == 'log2':
            self.operation = self.log2
            self.np_operation = self.np_log2
        elif label == 'exp':
            self.operation = self.exp
            self.np_operation = self.np_exp
        elif label == 'inverse':
            self.operation = self.inverse
            self.np_operation = self.np_inverse
        elif label == 'abs':
            self.operation = self.abs
            self.np_operation = self.np_abs
        elif label == 'sqrt':
            self.operation = self.square_root
            self.np_operation = self.np_square_root
        elif label == 'norm':
            self.operation = self.normalize
            self.np_operation = self.np_normalize

    def np_normalize(self, a):
        with np.errstate(divide='ignore'):
            result = np.divide(a, np.linalg.norm(a))
        return result

    def normalize(self, a):
        return 1.0

    def np_log10(self, a):
        with np.errstate(divide='ignore'):
            result = np.log10(np.abs(a))
        return result

    def log10(self, a):
        if a > 0.0:
            return math.log10(a)
        return -math.inf

    def np_log2(self, a):
        with np.errstate(divide='ignore'):
            result = np.log2(np.abs(a))
        return result

    def log2(self, a):
        if a > 0.0:
            return math.log2(a)
        return -math.inf

    def np_inverse(self, a):
        with np.errstate(divide='ignore'):
            result = np.divide(1, a)
        return result

    def inverse(self, a):
        if a == 0:
            return math.inf
        return 1 / a

    def np_exp(self, a):
        return np.exp(a)

    def exp(self, a):
        return math.exp(a)

    def np_abs(self, a):
        return np.abs(a)

    def abs(self, a):
        if a >= 0:
            return a
        return -a

    def np_square_root(self, a):
        signs = np.sign(a)
        result = np.sqrt(np.abs(a))
        return result * signs

    def square_root(self, a):
        if a > 0:
            return math.sqrt(a)
        return - math.sqrt(-a)

    def custom(self):
        pass

    def execute(self):
        # get values from static attributes
        input_value = self.input.get_received_data()
        t = type(input_value)
        if t == np.ndarray:
            output_value = self.np_operation(input_value)
        elif t == list:
            a = list_to_array(input_value)
            output_value = self.np_operation(a)
        else:
            output_value = self.operation(input_value)
        self.output.send(output_value)


class OpSingleTrigNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OpSingleTrigNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.use_degrees = True

        self.input = self.add_input('in', trigger_node=self)
        self.use_degrees_property = self.add_property('degrees', widget_type='checkbox', default_value=self.use_degrees)
        self.output = self.add_output("out")

        if label == 'sin':
            self.operation = self.sin
        elif label == 'cos':
            self.operation = self.cos
        elif label == 'asin':
            self.operation = self.asin
        elif label == 'acos':
            self.operation = self.acos
        elif label == 'tan':
            self.operation = self.tan
        elif label == 'atan':
            self.operation = self.atan

        self.degrees_to_radians = math.pi / 180

    def sin(self, a):
        t = type(a)
        if t == np.ndarray:
            if self.use_degrees:
                return np.sin(a * self.degrees_to_radians)
            else:
                return np.sin(a)
        else:
            if self.use_degrees:
                return math.sin(a * self.degrees_to_radians)
            else:
                return math.sin(a)

    def cos(self, a):
        t = type(a)
        if t == np.ndarray:
            if self.use_degrees:
                return np.cos(a * self.degrees_to_radians)
            else:
                return np.cos(a)
        else:
            if self.use_degrees:
                return math.cos(a * self.degrees_to_radians)
            else:
                return math.cos(a)

    def tan(self, a):
        t = type(a)
        if t == np.ndarray:
            if self.use_degrees:
                return np.tan(a * self.degrees_to_radians)
            else:
                return np.tan(a)
        else:
            if self.use_degrees:
                return math.tan(a * self.degrees_to_radians)
            else:
                return math.tan(a)

    def asin(self, a):
        t = type(a)
        if t == np.ndarray:
            a = np.clip(a, -1.0, 1.0)
            if self.use_degrees:
                return np.arcsin(a) / self.degrees_to_radians
            else:
                return np.arcsin(a)
        else:
            if a < -1:
                a = -1
            elif a > 1:
                a = 1
            if self.use_degrees:
                return math.asin(a) / self.degrees_to_radians
            else:
                return math.asin(a)

    def acos(self, a):
        t = type(a)
        if t == np.ndarray:
            a = np.clip(a, -1.0, 1.0)
            if self.use_degrees:
                return np.arccos(a) / self.degrees_to_radians
            else:
                return np.arccos(a)
        else:
            if a < -1:
                a = -1.0
            elif a > 1:
                a = 1.0
            if self.use_degrees:
                return math.acos(a) / self.degrees_to_radians
            else:
                return math.acos(a)

    def atan(self, a):
        t = type(a)
        if t == np.ndarray:
            if self.use_degrees:
                return np.arctan(a) / self.degrees_to_radians
            else:
                return np.arctan(a)
        else:
            if self.use_degrees:
                return math.atan(a) / self.degrees_to_radians
            else:
                return math.atan(a)

    def custom(self):
        pass

    def execute(self):
        # get values from static attributes
        self.use_degrees = self.use_degrees_property.get_widget_value()
        input_value = self.input.get_received_data()
        t = type(input_value)
        if t == list:
            input_value = list_to_array(input_value)
        elif t in [int, bool, np.int64, np.bool_]:
            input_value = float(input_value)
        output_value = self.operation(input_value)
        self.output.send(output_value)

