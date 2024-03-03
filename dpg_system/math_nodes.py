import dearpygui.dearpygui as dpg
import math
import numpy as np
import torch

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
    Node.app.register_node('pass', ComparisonAndPassNode.factory)
    Node.app.register_node('change', ComparisonAndPassNode.factory)
    Node.app.register_node('increasing', ComparisonAndPassNode.factory)
    Node.app.register_node('decreasing', ComparisonAndPassNode.factory)
    Node.app.register_node('perm', ArithmeticNode.factory)
    Node.app.register_node('combination', ArithmeticNode.factory)


class ArithmeticNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ArithmeticNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        widget_type = 'drag_float'
        self.operand = 0
        supplied_operand = False
        if len(args) > 0:
            supplied_operand = True
            self.operand = any_to_float_or_int(args[0])
            t = type(self.operand)
            if t == float:
                widget_type = 'drag_float'
            elif t == int:
                widget_type = 'drag_int'

        self.op_dict = {
            'perm': self.permutation,
            'combination': self.combination,
            '+': self.add,
            '-': self.subtract,
            '!-': self.inverse_subtract,
            '*': self.multiply,
            '/': self.divide,
            '//': self.int_divide,
            '!/': self.inverse_divide,
            'pow': self.power,
            '^': self.power,
            'min': self.min,
            'max': self.max,
            'mod': self.mod,
            '%': self.mod
        }
        if label in self.op_dict:
            self.op = self.op_dict[label]
            if label in ['pow', '^']:
                widget_type = 'drag_float'
        else:
            self.op = self.op_dict['+']

        self.input = self.add_input('in', triggers_execution=True)

        if supplied_operand:
            self.operand_input = self.add_input('', widget_type=widget_type, default_value=self.operand, callback=self.operand_changed)
        else:
            self.operand_input = self.add_input('operand')

        self.output = self.add_output('result')

    def operand_changed(self):
        self.operand = any_to_numerical(self.operand_input())

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = any_to_numerical(self.operand_input())
        input_value = any_to_numerical(self.input())
        output_value = self.op(input_value, self.operand)
        self.output.send(output_value)

    def matrix_mult(self, a, b):
        if type(a) == np.array and type(b) == np.array:
            a_row = len(a[0])
            b_col = len(b)
            if a_row == b_col:
                return np.multiply(a, b)

    def permutation(self, a, b):
        return math.factorial(a) // math.factorial(abs(a - b))

    def combination(self, a, b):
        return math.factorial(a) // (math.factorial(abs(a - b)) * math.factorial(b))

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
        elif self.app.torch_available and type(a) == torch.Tensor:
            return torch.minimum(a, b)
        if a > b:
            return b
        return a

    def max(self, a, b):
        if type(a) == np.ndarray:
            return np.maximum(a, b)
        elif self.app.torch_available and type(a) == torch.Tensor:
            return torch.maximum(a, b)
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
        elif self.app.torch_available and type(a) == torch.Tensor:
            return a / b
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
        elif self.app.torch_available and type(a) == torch.Tensor:
            return b / a
        if a == 0:
            return b / 1e-8
        return b / a

    def power(self, a, b):
        if type(a) == np.ndarray:
            return np.power(a, b)
        elif self.app.torch_available and type(a) == torch.Tensor:
            return torch.pow(a, b)
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

        self.output_op = bool
        self.torch_output_op = torch.bool
        self.numpy_output_op = np.bool_

        widget_type = 'drag_float'
        self.operand = 0
        supplied_operand = False
        if len(args) > 0:
            supplied_operand = True
            self.operand = any_to_float_or_int(args[0])
            t = type(self.operand)
            if t == float:
                widget_type = 'drag_float'
            elif t == int:
                widget_type = 'drag_int'

        self.operations = {
            '>': self.greater,
            '>=': self.greater_equal,
            '==': self.equal,
            '<': self.less,
            '<=': self.less_equal,
            '!=': self.not_equal
        }
        if label in self.operations:
            self.op = self.operations[label]
        else:
            self.op = self.operations['>']

        self.input = self.add_input('in', triggers_execution=True)

        if supplied_operand:
            self.operand_input = self.add_input('', widget_type=widget_type, default_value=self.operand, callback=self.operand_changed)
        else:
            self.operand_input = self.add_input('operand')

        self.output = self.add_output('result')
        self.output_type_option = self.add_option('output_type', widget_type='combo', default_value='bool', callback=self.output_type_changed)
        self.output_type_option.widget.combo_items = ['bool', 'int', 'float']

    def operand_changed(self):
        self.operand = any_to_numerical(self.operand_input())

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = any_to_numerical(self.operand_input())
        input_value = any_to_numerical(self.input())

        t = type(input_value)
        if t == np.ndarray:
            output_value = self.op(input_value, self.operand).astype(self.numpy_output_op)
        elif t == torch.Tensor:
            output_value = self.op(input_value, self.operand).to(self.torch_output_op)
        else:
            output_value = self.output_op(self.op(input_value, self.operand))

        self.output.send(output_value)

    def output_type_changed(self):
        output_type = self.output_type_option()
        self.output_op = bool
        self.torch_output_op = torch.bool
        self.numpy_output_op = np.bool_
        if output_type == 'bool':
            self.output_op = bool
        elif output_type == 'int':
            self.output_op = int
            self.torch_output_op = torch.int
            self.numpy_output_op = int
        elif output_type == 'float':
            self.output_op = float
            self.torch_output_op = torch.float
            self.numpy_output_op = float

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


class ComparisonAndPassNode(Node):
    output_op = bool
    @staticmethod
    def factory(name, data, args=None):
        node = ComparisonAndPassNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.simple = True
        force_int = False

        self.operation = 'always'
        self.operations = {'!=': self.not_equal, '==': self.equal, '>': self.greater, '>=': self.greater_equal,
                           '<': self.less, '<=': self.less_equal, 'always': self.no_op}
        if label == 'change':
            self.operation = '!='
            # force_int = True
        elif label == 'increasing':
            self.operation = '>'
        elif label == 'decreasing':
            self.operation = '<'

        if len(args) > 0:
            self.simple = False
            if args[0] in self.operations:
                self.operation = args[0]

        self.operand = None
        self_compare = False

        if self.simple:
            if len(args) > 1:
                t, self.operand = decode_arg(args, 1)
            #     # self.operand = self.arg_as_number(default_value=0.0, index=1)
            else:
                self_compare = True
        else:
            if len(args) > 2:
                t, self.operand = decode_arg(args, 2)
                # self.operand = self.arg_as_number(default_value=0.0, index=2)
            else:
                self_compare = True

        self.input = self.add_input('in', triggers_execution=True)

        if self.simple:
            self.comparison_property = self.add_option('', widget_type='combo', default_value=self.operation, callback=self.comparison_changed)
            self.comparison_property.widget.combo_items = list(self.operations)
            self.operand_property = self.add_option('', widget_type='drag_float', default_value=self.operand,
                                                callback=self.operand_changed)
            self.self_compare_property = self.add_option('self_compare', widget_type='checkbox', default_value=self_compare)
            self.force_int_property = self.add_option('force_int', widget_type='checkbox', default_value=force_int)
        else:
            self.comparison_property = self.add_property('', widget_type='combo', default_value=self.operation, callback=self.comparison_changed)
            self.comparison_property.widget.combo_items = list(self.operations)
            self.operand_property = self.add_input('', widget_type='drag_float', default_value=self.operand,
                                                callback=self.operand_changed)
            self.self_compare_property = self.add_property('self_compare', widget_type='checkbox')
            self.force_int_property = self.add_property('force_int', widget_type='checkbox')

        self.output = self.add_output('result')

    def comparison_changed(self):
        self.operation = self.comparison_property()

    def operand_changed(self):
        self.operand = self.operand_property()

    def execute(self):
        if not self.simple and self.operand_property.fresh_input:
            self.operand = self.operand_property()

        input_value = self.input()
        if self.operand is not None:
            input_value = conform_type(input_value, self.operand)
        else:
            t = type(input_value)
            if t not in [float, int, np.int64, np.float32, np.double]:
                self.force_int_property.set(False)
        # input_value = any_to_numerical(self.input())

        if type(input_value) == np.ndarray:
            if type(self.operand) != np.ndarray:
                self.operand = np.zeros_like(input_value)
                self.operand_property.set(input_value)
            if self.force_int_property():
                input_value = input_value.round()
                op = self.operand.round()
            else:
                op = self.operand
            output_value = self.operations[self.operation](input_value, op)
            if output_value.any():
                self.output.send(input_value)
        elif self.app.torch_available and type(input_value) == torch.Tensor:
            if type(self.operand) != torch.Tensor:
                self.operand = torch.zeros_like(input_value)
                self.operand_property.set(input_value)
            if self.force_int_property():
                input_value = input_value.round()
                op = self.operand.round()
            else:
                op = self.operand
            output_value = self.operations[self.operation](input_value, op)
            if output_value.any():
                self.output.send(input_value)
        else:
            if type(self.operand) == np.ndarray:
                self.operand = 0
            if self.force_int_property():
                input_value = any_to_int(input_value)
                op = any_to_int(self.operand)
            else:
                op = self.operand
            output_value = self.operations[self.operation](input_value, op)
            if output_value:
                self.output.send(input_value)
        if self.self_compare_property():
            self.operand = input_value
            self.operand_property.set(input_value)

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

    def no_op(self, a, b):
        return True


class OpSingleNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = OpSingleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.operations = {
            'log10': self.log10,
            'log2': self.log2,
            'exp': self.exp,
            'inverse': self.inverse,
            'abs': self.abs,
            'sqrt': self.square_root,
            'norm': self.normalize
        }
        if label in self.operations:
            self.op = self.operations[label]
        else:
            self.op = self.operations['log10']

        self.input = self.add_input('in', triggers_execution=True)
        self.output = self.add_output('result')

    def execute(self):
        # get values from static attributes
        input_value = self.input()
        t = type(input_value)
        if t in [int, bool, np.int64, np.bool_]:
            input_value = float(input_value)
        output_value = self.op(input_value)
        self.output.send(output_value)

    def normalize(self, a):
        if type(a) == np.ndarray:
            with np.errstate(divide='ignore'):
                result = np.divide(a, np.linalg.norm(a))
            return result
        elif self.app.torch_available and type(a) == torch.Tensor:
            result = torch.divide(a, torch.linalg.norm(a))
            return result
        return 1.0

    def log10(self, a):
        if type(a) == np.ndarray:
            with np.errstate(divide='ignore'):
                result = np.log10(np.abs(a))
            return result
        elif self.app.torch_available and type(a) == torch.Tensor:
            result = torch.log10(torch.abs(a))
            return result
        if a > 0.0:
            return math.log10(a)
        return -math.inf

    def log2(self, a):
        if type(a) == np.ndarray:
            with np.errstate(divide='ignore'):
                result = np.log2(np.abs(a))
            return result
        elif self.app.torch_available and type(a) == torch.Tensor:
            result = torch.log2(torch.abs(a))
            return result
        if a > 0.0:
            return math.log2(a)
        return -math.inf

    def inverse(self, a):
        if type(a) == np.ndarray:
            with np.errstate(divide='ignore'):
                result = np.divide(1, a)
            return result
        elif self.app.torch_available and type(a) == torch.Tensor:
            result = torch.divide(1, a)
            return result
        if a == 0:
            return math.inf
        return 1 / a

    def exp(self, a):
        if type(a) == np.ndarray:
            return np.exp(a)
        elif self.app.torch_available and type(a) == torch.Tensor:
            return torch.exp(a)
        return math.exp(a)

    def abs(self, a):
        if type(a) == np.ndarray:
            return np.abs(a)
        elif self.app.torch_available and type(a) == torch.Tensor:
            return torch.abs(a)
        if a >= 0:
            return a
        return -a

    def square_root(self, a):
        if type(a) == np.ndarray:
            signs = np.sign(a)
            result = np.sqrt(np.abs(a))
            return result * signs
        elif self.app.torch_available and type(a) == torch.Tensor:
            signs = torch.sign(a)
            result = torch.sqrt(torch.abs(a))
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

        self.operations = {
            'sin': self.sin,
            'cos': self.cos,
            'asin': self.asin,
            'acos': self.acos,
            'tan': self.tan,
            'atan': self.atan
        }
        if label in self.operations:
            self.op = self.operations[label]
        else:
            self.op = self.operations['sin']

        self.use_degrees = True
        self.degrees_to_radians = math.pi / 180

        self.input = self.add_input('in', triggers_execution=True)
        self.use_degrees_property = self.add_property('degrees', widget_type='checkbox', default_value=self.use_degrees)
        self.output = self.add_output('out')

    def execute(self):
        # get values from static attributes
        self.use_degrees = self.use_degrees_property()

        input_value = any_to_numerical(self.input(), validate=True)
        if input is not None:
            t = type(input_value)
            if t in [int, bool, np.int64, np.bool_]:
                input_value = float(input_value)
            output_value = self.op(input_value)
            self.output.send(output_value)

    def sin(self, a):
        t = type(a)
        if t == np.ndarray:
            if self.use_degrees:
                return np.sin(a * self.degrees_to_radians)
            else:
                return np.sin(a)
        elif self.app.torch_available and t == torch.Tensor:
            if self.use_degrees:
                return torch.sin(torch.rad2deg(a))
            else:
                return torch.sin(a)
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
        elif self.app.torch_available and t == torch.Tensor:
            if self.use_degrees:
                return torch.cos(torch.rad2deg(a))
            else:
                return torch.cos(a)
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
        elif self.app.torch_available and t == torch.Tensor:
            if self.use_degrees:
                return torch.tan(torch.rad2deg(a))
            else:
                return torch.tan(a)
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
        elif self.app.torch_available and t == torch.Tensor:
            a = torch.clamp(a, -1.0, 1.0)
            if self.use_degrees:
                return torch.deg2rad(torch.arcsin(a))
            else:
                return torch.arcsin(a)
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
        elif self.app.torch_available and t == torch.Tensor:
            a = torch.clamp(a, -1.0, 1.0)
            if self.use_degrees:
                return torch.deg2rad(torch.arccos(a))
            else:
                return torch.arccos(a)
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
        elif self.app.torch_available and t == torch.Tensor:
            if self.use_degrees:
                return torch.deg2rad(torch.arctan(a))
            else:
                return torch.arctan(a)
        else:
            if self.use_degrees:
                return math.atan(a) / self.degrees_to_radians
            else:
                return math.atan(a)

