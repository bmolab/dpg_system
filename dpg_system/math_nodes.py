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

        self.operand = self.arg_as_number(default_value=0.0)
        self.input = self.add_input('in', triggers_execution=True)
        self.operand_input = self.add_input('', widget_type='drag_float', default_value=self.operand, callback=self.operand_changed)
        self.output = self.add_output("")
        self.operations = {'+': self.add, '-': self.subtract, '!-': self.inverse_subtract,
                           '*': self.multiply, '/': self.divide, '//': self.int_divide,
                           '!/': self.inverse_divide, 'pow': self.power, '^': self.power,
                           'min': self.min, 'max': self.max, 'mod': self.mod, '%': self.mod}
        if label in self.operations:
            self.operation = self.operations[label]
        else:
            self.operation = self.operations['+']

    def operand_changed(self):
        self.operand = self.operand_input.get_widget_value()

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = self.operand_input.get_received_data()
            if type(self.operand) == list:
                self.operand = list_to_array(self.operand)

        input_value = self.input.get_received_data()
        t = type(input_value)
        if t == list:
            input_value = list_to_array(input_value)

        output_value = self.operation(input_value, self.operand)
        self.output.send(output_value)

    def mod(self, a, b):
        if type(a) == np.ndarray:
            if b == 0:
                return np.zeros_like(a)
            return np.mod(a, b)
        if b == 0:
            return 0
        return a % b

    def min(self, a, b):
        if type(a) == np.ndarray:
            return np.minimum(a, b)
        if a > b:
            return b
        return a

    def max(self, a, b):
        if type(a) == np.ndarray:
            return np.maximum(a, b)
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
        if type(a) == np.ndarray:
            old_errs = np.seterr(divide='ignore')
            out = np.divide(a, b)
            np.seterr(**old_errs)
            return out
        if b == 0:
            return a / 1e-8
        return a / b

    def int_divide(self, a, b):
        if b == 0:
            return a // 1e-8
        return a // b

    def inverse_divide(self, a, b):
        if type(a) == np.ndarray:
            old_errs = np.seterr(divide='ignore')
            out = np.divide(b, a)
            np.seterr(**old_errs)
            return out
        if a == 0:
            return b / 1e-8
        return b / a

    def power(self, a, b):
        if type(a) == np.ndarray:
            return np.power(a, b)
        else:
            return math.pow(a, b)


class ComparisonNode(Node):
    output_op = bool
    @staticmethod
    def factory(name, data, args=None):
        node = ComparisonNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.operand = self.arg_as_number(default_value=0.0)
        self.input = self.add_input('in', triggers_execution=True)
        self.operand_input = self.add_input('', widget_type='drag_float', default_value=self.operand, callback=self.operand_changed)
        self.output = self.add_output("")
        self.operations = {'>': self.greater, '>=': self.greater_equal, '==': self.equal,
                           '<': self.less, '<=': self.less_equal, '!=': self.not_equal}
        if label in self.operations:
            self.operation = self.operations[label]
        else:
            self.operation = self.operations['>']
        self.output_type_option = self.add_option('output_type', widget_type='combo', default_value='bool', callback=self.output_type_changed)
        self.output_type_option.widget.combo_items = ['bool', 'int', 'float']

    def operand_changed(self):
        self.operand = self.operand_input.get_widget_value()

    def output_type_changed(self):
        output_type = self.output_type_option.get_widget_value()
        print('got output type ' + output_type)
        self.output_op = bool
        if output_type == 'bool':
            self.output_op = bool
        elif output_type == 'int':
            self.output_op = int
        elif output_type == 'float':
            self.output_op = float

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = self.operand_input.get_received_data()
            if type(self.operand) == list:
                self.operand = list_to_array(self.operand)

        input_value = self.input.get_data()
        t = type(input_value)
        if t == list:
            input_value = list_to_array(input_value)

        if type(input_value) == np.ndarray:
            output_value = self.operation(input_value, self.operand).astype(self.output_op)
        else:
            output_value = self.output_op(self.operation(input_value, self.operand))

        self.output.send(output_value)

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


class OpSingleNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OpSingleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('in', triggers_execution=True)
        self.output = self.add_output("")

        self.operations = {'log10': self.log10, 'log2': self.log2, 'exp': self.exp,
                           'inverse': self.inverse, 'abs': self.abs,
                           'sqrt': self.square_root, 'norm': self.normalize}
        if label in self.operations:
            self.operation = self.operations[label]
        else:
            self.operation = self.operations['log10']

    def execute(self):
        # get values from static attributes
        input_value = self.input.get_received_data()
        t = type(input_value)
        if t == list:
            input_value = list_to_array(input_value)
        elif t in [int, bool, np.int64, np.bool_]:
            input_value = float(input_value)
        output_value = self.operation(input_value)
        self.output.send(output_value)

    def normalize(self, a):
        if type(a) == np.ndarray:
            with np.errstate(divide='ignore'):
                result = np.divide(a, np.linalg.norm(a))
            return result
        return 1.0

    def log10(self, a):
        if type(a) == np.ndarray:
            with np.errstate(divide='ignore'):
                result = np.log10(np.abs(a))
            return result
        if a > 0.0:
            return math.log10(a)
        return -math.inf

    def log2(self, a):
        if type(a) == np.ndarray:
            with np.errstate(divide='ignore'):
                result = np.log2(np.abs(a))
            return result
        if a > 0.0:
            return math.log2(a)
        return -math.inf

    def inverse(self, a):
        if type(a) == np.ndarray:
            with np.errstate(divide='ignore'):
                result = np.divide(1, a)
            return result
        if a == 0:
            return math.inf
        return 1 / a

    def exp(self, a):
        if type(a) == np.ndarray:
            return np.exp(a)
        return math.exp(a)

    def abs(self, a):
        if type(a) == np.ndarray:
            return np.abs(a)
        if a >= 0:
            return a
        return -a

    def square_root(self, a):
        if type(a) == np.ndarray:
            signs = np.sign(a)
            result = np.sqrt(np.abs(a))
            return result * signs
        if a > 0:
            return math.sqrt(a)
        return - math.sqrt(-a)


class OpSingleTrigNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OpSingleTrigNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.use_degrees = True
        self.degrees_to_radians = math.pi / 180

        self.input = self.add_input('in', triggers_execution=True)
        self.use_degrees_property = self.add_property('degrees', widget_type='checkbox', default_value=self.use_degrees)
        self.output = self.add_output("out")
        self.operations = {'sin': self.sin, 'cos': self.cos, 'asin': self.asin,
                           'acos': self.acos, 'tan': self.tan, 'atan': self.atan}
        if label in self.operations:
            self.operation = self.operations[label]
        else:
            self.operation = self.operations['sin']

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

