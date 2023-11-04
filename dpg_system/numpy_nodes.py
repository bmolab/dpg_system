import dearpygui.dearpygui as dpg
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import time
import numpy as np


# add random, linspace, ones, zeros
def register_numpy_nodes():
    Node.app.register_node('np.linalg.norm', NumpyUnaryLinearAlgebraNode.factory)
    Node.app.register_node('euclidean_distance', NumpyUnaryLinearAlgebraNode.factory)
    Node.app.register_node('np.linalg.det', NumpyUnaryLinearAlgebraNode.factory)
    Node.app.register_node('np.linalg.matrix_rank', NumpyUnaryLinearAlgebraNode.factory)
    Node.app.register_node('flatten', FlattenMatrixNode.factory)
    Node.app.register_node('np.ravel', FlattenMatrixNode.factory)
    Node.app.register_node('np.dot', NumpyDotProductNode.factory)
    Node.app.register_node('np.cross', NumpyCrossProductNode.factory)
    Node.app.register_node('np.outer', NumpyInnerOuterProductNode.factory)
    Node.app.register_node('np.inner', NumpyInnerOuterProductNode.factory)
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
    Node.app.register_node('np.rand', NumpyGeneratorNode.factory)
    Node.app.register_node('np.ones', NumpyGeneratorNode.factory)
    Node.app.register_node('np.zeros', NumpyGeneratorNode.factory)
    Node.app.register_node('np.linspace', NumpyLinSpaceNode.factory)
    Node.app.register_node('np.squeeze', NumpySqueezeNode.factory)
    Node.app.register_node('np.expand_dims', NumpyExpandDimsNode.factory)
    Node.app.register_node('np.unsqueeze', NumpyExpandDimsNode.factory)
    Node.app.register_node('np.repeat', NumpyRepeatNode.factory)
    Node.app.register_node('np.transpose', NumpyTransposeNode.factory)
    Node.app.register_node('np.flip', NumpyFlipNode.factory)
    Node.app.register_node('np.roll', NumpyRollNode.factory)
    Node.app.register_node('np.rotate', NumpyRotateNode.factory)
    Node.app.register_node('np.rot90', NumpyRotateNode.factory)
    Node.app.register_node('np.astype', NumpyAsTypeNode.factory)
    Node.app.register_node('np.crop', NumpyCropNode.factory)
    Node.app.register_node('np.clip', NumpyClipNode.factory)
    Node.app.register_node('np.min', NumpyClipNode.factory)
    Node.app.register_node('np.max', NumpyClipNode.factory)


class NumpyGeneratorNode(Node):
    operations = {'np.rand': np.random.Generator.random, 'np.ones': np.ones, 'np.zeros': np.zeros}
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyGeneratorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.op = np.random.Generator.random
        if label in self.operations:
            self.op = self.operations[label]

        self.shape = []
        for i in range(len(args)):
            d, t = decode_arg(args, i)
            if t == int:
                self.shape += (d,)
        if len(self.shape) == 0:
            self.shape = [1]

        self.rng = None
        if self.label == 'np.rand':
            self.rng = np.random.default_rng(seed=None)

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)

        self.shape_properties = []
        for i in range(len(self.shape)):
            self.shape_properties.append(self.add_property('dim ' + str(i), widget_type='input_int', default_value=self.shape[i]))
        if self.label == 'np.rand':
            self.min = 0
            self.max = 1
            self.min = self.add_input('min', widget_type='drag_float', default_value=self.min)
            self.max = self.add_input('max', widget_type='drag_float', default_value=self.max)

        self.dtype_dict = {}
        self.dtype_dict['float32'] = np.float32
        self.dtype_dict['float'] = float
        self.dtype_dict['double'] = np.double
        self.dtype_dict['int64'] = np.int64
        self.dtype_dict['uint8'] = np.uint8
        self.dtype_dict['bool'] = np.bool_
        self.dtype_option = self.add_option('dtype', widget_type='combo', default_value='float32', callback=self.dtype_changed)
        self.dtype_option.widget.combo_items = list(self.dtype_dict.keys())
        self.dtype = np.float32

        out_label = 'random array'
        if self.label == 'np.ones':
            out_label = 'array of ones'
        elif self.label == 'np.zeros':
            out_label = 'array of zeros'

        self.output = self.add_output(out_label)

    def dtype_changed(self):
        dtype = self.dtype_option()
        if dtype in self.dtype_dict:
            self.dtype = self.dtype_dict[dtype]
            if self.label == 'np.rand':
                if self.dtype == np.uint8:
                    if self.min < 0:
                        self.min.set(0.0)
                    if self.max == 1.0 or self.max < 255:
                        self.max.set(255.0)
                elif self.dtype == np.int64:
                    if self.min < -32768:
                        self.min.set(-32768)
                    if self.max == 1.0:
                        self.max.set(32767)
                elif self.dtype in [float, np.double, np.float32]:
                    if self.min == -32768:
                        self.min.set(0.0)
                    if self.max == 255:
                        self.max.set(1.0)
                    elif self.max == 32767:
                        self.max.set(1.0)

    def execute(self):
        out_array = None
        for i in range(len(self.shape)):
            self.shape[i] = self.shape_properties[i]()
        if self.label == 'np.rand':
            size = tuple(self.shape)
            if self.dtype in [float, np.float32, np.double]:
                range_ = self.max() - self.min()
                out_array = self.rng.random(size=size, dtype=self.dtype) * range_ + self.min()
            elif self.dtype == np.int64:
                out_array = self.rng.integers(low=self.min(), high=self.max(), size=size, dtype=self.dtype, endpoint=True)
            elif self.dtype == np.uint8:
                out_array = self.rng.integers(low=self.min(), high=self.max(), size=size, dtype=self.dtype, endpoint=True)
            elif self.dtype == np.bool_:
                out_array = self.rng.integers(low=0, high=1, size=size, dtype=self.dtype, endpoint=True)
        else:
            out_array = self.op(tuple(self.shape), dtype=self.dtype)
        self.output.send(out_array)


class NumpyLinSpaceNode(Node):
    operations = {'np.linspace': np.linspace}

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyLinSpaceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = np.linspace
        if label in self.operations:
            self.op = self.operations[label]
        self.shape = []
        start = 0.0
        stop = 1.0
        steps = 50
        if len(args) > 0:
            d, t = decode_arg(args, 0)
            start = any_to_float(d)
        if len(args) > 1:
            d, t = decode_arg(args, 1)
            stop = any_to_float(d)
        if len(args) > 2:
            d, t = decode_arg(args, 2)
            steps = any_to_int(d)

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.start = self.add_property('start', widget_type='drag_float', default_value=start)
        self.stop = self.add_property('stop', widget_type='drag_float', default_value=stop)
        self.steps = self.add_property('steps', widget_type='drag_int', default_value=steps)

        self.dtype_dict = {}
        self.dtype_dict['float32'] = np.float32
        self.dtype_dict['float'] = float
        self.dtype_dict['double'] = np.double
        self.dtype_dict['int64'] = np.int64
        self.dtype_dict['uint8'] = np.uint8
        self.dtype_option = self.add_property('dtype', widget_type='combo', default_value='float32', callback=self.dtype_changed)
        self.dtype_option.widget.combo_items = list(self.dtype_dict.keys())
        self.dtype = np.float32

        self.output = self.add_output('linspace out')

    def dtype_changed(self):
        dtype = self.dtype_option()
        if dtype in self.dtype_dict:
            self.dtype = self.dtype_dict[dtype]

    def execute(self):
        out_array = self.op(self.start(), self.stop(), self.steps(), dtype=self.dtype)
        self.output.send(out_array)


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
        if len(args) > 0:
            d, t = decode_arg(args, 0)
            if t == int:
                self.axis = d
        output_name = label.split('.')[0]
        self.input = self.add_input('in', triggers_execution=True)
        self.output = self.add_output(output_name)

    def execute(self):
        input_value = any_to_array(self.input())
        if len(input_value.shape) > self.axis:
            output_value = self.op(input_value, axis=self.axis)
            self.output.send(output_value)
        else:
            if self.app.verbose:
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
            self.input_2_vector = any_to_array(self.input_2())

        input_value = any_to_array(self.input())
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
        self.output = self.add_output('dot product')

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = any_to_array(self.operand_input())
        input_value = any_to_array(self.input())

        output_value = np.dot(input_value, self.operand)
        self.output.send(output_value)


class NumpyInnerOuterProductNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyInnerOuterProductNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.operand = 0
        self.input = self.add_input('in 1', triggers_execution=True)
        self.operand_input = self.add_input('in 2')
        if self.label == 'np.outer':
            self.output = self.add_output('outer_product')
        elif self.label == 'np.inner':
            self.output = self.add_output('inner_product')

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = any_to_array(self.operand_input())
        input_value = any_to_array(self.input())

        if self.label == 'np.outer':
            output_value = np.outer(input_value, self.operand)
            self.output.send(output_value)
        else:
            output_value = np.inner(input_value, self.operand)
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
        self.output = self.add_output('mat mul result')

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = any_to_array(self.operand_input())
        input_value = any_to_array(self.input())

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
        self.output = self.add_output('cross product')

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = any_to_array(self.operand_input())
        input_value = any_to_array(self.input())

        output_value = np.cross(input_value, self.operand)
        self.output.send(output_value)


class NumpySqueezeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpySqueezeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        axis = 0
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                axis = int(v)
        self.input = self.add_input('input', triggers_execution=True)
        self.axis = self.add_property('axis', widget_type='input_int', default_value=axis)
        self.output = self.add_output('squeezed array')

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input())
            axis = self.axis()
            if 0 <= axis < len(data.shape):
                if data.shape[axis] == 1:
                    data = np.squeeze(data, axis=axis)
                    self.output.send(data)
                else:
                    print('axis to squeeze has a size not equal to 1')
            else:
                print('axis to squeeze is out of range')


class NumpyExpandDimsNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyExpandDimsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        axis = 0
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                axis = int(v)
        self.input = self.add_input('input', triggers_execution=True)
        self.axis = self.add_property('axis', widget_type='input_int', default_value=axis)
        self.output = self.add_output('array out')

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input())
            axis = self.axis()
            if axis < 0:
                axis = 0
            elif axis > len(data.shape):
                axis = len(data.shape)
            data = np.expand_dims(data, axis=axis)
            self.output.send(data)


class NumpyRepeatNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyRepeatNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        axis = 0
        repeats = 2

        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                repeats = int(v)

        if len(args) > 1:
            v, t = decode_arg(args, 1)
            if t in [int, float]:
                axis = int(v)

        self.input = self.add_input('input', triggers_execution=True)
        self.repeats = self.add_input('repeats', widget_type='input_int', default_value=repeats)
        self.axis = self.add_input('axis', widget_type='input_int', default_value=axis)

        self.output = self.add_output('array out')

    def execute(self):
        repeated_data = None
        data = any_to_array(self.input())
        axis = self.axis()
        if axis < 0:
            axis = 0
        elif axis >= len(data.shape):
            if axis == len(data.shape):
                repeated_data = np.repeat(np.expand_dims(data, axis), repeats=self.repeats(), axis=axis)
            else:
                axis = len(data.shape) - 1
                repeated_data = np.repeat(data, repeats=self.repeats(), axis=axis)
        else:
            repeated_data = np.repeat(data, repeats=self.repeats(), axis=axis)
        self.output.send(repeated_data)


class NumpyCropNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyCropNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('image in', triggers_execution=True)
        self.left = 0
        self.right = 256
        self.top = 0
        self.bottom = 256
        self.input_shape = None
        if len(args) > 0:
            self.left = any_to_int(args[0])
        if len(args) > 1:
            self.top = any_to_int(args[1])
        if len(args) > 2:
            self.right = any_to_int(args[2])
        if len(args) > 3:
            self.bottom = any_to_int(args[3])
        if self.right < self.left:
            self.right = self.left
        if self.bottom < self.top:
            self.bottom = self.top
        self.left_property = self.add_input('left', widget_type='drag_int', default_value=self.left, callback=self.crop_changed)
        self.top_property = self.add_input('top', widget_type='drag_int', default_value=self.top, callback=self.crop_changed)
        self.right_property = self.add_input('right', widget_type='drag_int', default_value=self.right, callback=self.crop_changed)
        self.bottom_property = self.add_input('bottom', widget_type='drag_int', default_value=self.bottom, callback=self.crop_changed)
        self.uncrop_button = self.add_property('uncrop', widget_type='button', callback=self.uncrop)
        self.output = self.add_output('out array')

    def uncrop(self):
        if self.input_shape is not None:
            self.left = 0
            self.top = 0
            self.right = self.input_shape[1]
            self.bottom = self.input_shape[0]
            self.left_property.set(self.left)
            self.top_property.set(self.top)
            self.right_property.set(self.right)
            self.bottom_property.set(self.bottom)

    def crop_changed(self):
        self.left = self.left_property()
        self.top = self.top_property()
        self.right = self.right_property()
        self.bottom = self.bottom_property()

        if self.left < 0:
            self.left = 0
        if self.top < 0:
            self.top = 0
        if self.right < 1:
            self.right = 1
        if self.bottom < 1:
            self.bottom = 1

        if self.input_shape is not None:
            if self.left >= self.input_shape[1]:
                self.left = self.input_shape[1] - 1
            if self.top >= self.input_shape[0]:
                self.top = self.input_shape[0] - 1
            if self.right > self.input_shape[1]:
                self.right = self.input_shape[1]
            if self.bottom > self.input_shape[0]:
                self.bottom = self.input_shape[0]

        if self.left >= self.right:
            self.left = self.right - 1
        if self.top >= self.bottom:
            self.top = self.bottom - 1


    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input())
            if type(data) == np.ndarray:
                if len(data.shape) > 1:
                    if self.input_shape is None or data.shape != self.input_shape:
                        self.input_shape = data.shape
                        self.crop_changed()
                    data = data[self.top:self.bottom, self.left:self.right]
                    self.output.send(np.ascontiguousarray(data))


class NumpyClipNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyClipNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('input', triggers_execution=True)
        self.min = 0.0
        self.max = 1.0
        if len(args) > 0:
            self.min = any_to_float(args[0])
        if len(args) > 1:
            self.max = any_to_float(args[1])
        print(self.min, self.max)
        self.min_property = None
        self.max_property = None
        if self.label in ['np.min', 'np.clip']:
            print('add min prop')
            self.min_property = self.add_property('min', widget_type='drag_float', default_value=self.min, callback=self.min_changed)
        if self.label in ['np.max', 'np.clip']:
            print('add max prop')
            self.max_property = self.add_property('max', widget_type='drag_float', default_value=self.max, callback=self.max_changed)
        self.output = self.add_output('out array')
        self.mode = 0
        if self.label == 'np.min':
            self.mode = 1
        elif self.label == 'np.max':
            self.mode = 2

    def min_changed(self):
        self.min = self.min_property()

    def max_changed(self):
        self.max = self.max_property()

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input())
            if type(data) == np.ndarray:
                if self.mode == 0:
                    self.output.send(np.clip(data, self.min, self.max))
                elif self.mode == 1:
                    self.output.send(np.min(data, self.min))
                elif self.mode == 2:
                    self.output.send(np.max(data, self.max))


class NumpyRollNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyRollNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        axis = 0
        shifts = 2
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                shifts = int(v)
        if len(args) > 1:
            v, t = decode_arg(args, 1)
            if t in [int, float]:
                axis = int(v)
        self.input = self.add_input('input', triggers_execution=True)
        self.shifts = self.add_input('shifts', widget_type='input_int', default_value=shifts)
        self.axis = self.add_input('axis', widget_type='input_int', default_value=axis)

        self.output = self.add_output('rolled array')

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input())
            axis = self.axis()
            if axis < 0:
                axis = 0
            elif axis >= len(data.shape):
                axis = len(data.shape) - 1
            repeated_data = np.roll(data, shift=self.shifts(), axis=axis)
            self.output.send(repeated_data)


class NumpyFlipNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyFlipNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        axis = 0
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                axis = int(v)
        self.input = self.add_input('input', triggers_execution=True)
        self.axis = self.add_input('axis', widget_type='input_int', default_value=axis)

        self.output = self.add_output('flipped array')

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input())
            axis = self.axis()
            if axis < 0:
                axis = 0
            elif axis >= len(data.shape):
                axis = len(data.shape) - 1
            flipped_data = np.flip(data, axis=axis)
            self.output.send(flipped_data)


class NumpyTransposeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyTransposeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        axis = 0
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                axis = int(v)
        self.input = self.add_input('input', triggers_execution=True)
        # self.axis_input = self.add_input('axis', widget_type='input_int', default_value=axis)

        self.output = self.add_output('transposed array')

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input())
            # axis = self.axis_input()
            # if axis < 0:
            #     axis = 0
            # elif axis >= len(data.shape):
            #     axis = len(data.shape) - 1
            transposed_data = np.transpose(data)
            self.output.send(transposed_data)


class NumpyAsTypeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyAsTypeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('input array', triggers_execution=True)
        self.type = self.add_property('type', widget_type='combo', default_value='float')
        self.type.widget.combo_items = ['bool', 'uint8', 'int8', 'int64', 'float', 'float32', 'double']

        self.output = self.add_output('converted array')

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input())
            type = self.type()
            if type == 'float':
                self.output.send(data.astype(float))
            elif type == 'float32':
                self.output.send(data.astype(np.float32))
            elif type == 'uint8':
                self.output.send(data.astype(np.uint8))
            elif type == 'int8':
                self.output.send(data.astype(np.int8))
            elif type == 'int64':
                self.output.send(data.astype(np.int64))
            elif type == 'bool':
                self.output.send(data.astype(np.bool_))
            elif type == 'double':
                self.output.send(data.astype(np.double))


class FlattenMatrixNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = FlattenMatrixNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        order = 'C'
        if len(args) > 0:
            order_, t = decode_arg(args, 0)
            if t == str and order_ in ['C', 'F', 'A', 'K']:
                order = order_
        self.input = self.add_input('input', triggers_execution=True)
        self.order = self.add_property('order', widget_type='combo', default_value=order)
        self.order.widget.combo_items = ['C', 'F', 'A', 'K']
        self.output = self.add_output('flattened array')

    def execute(self):
        if self.input.fresh_input:
            order = self.order()
            data = any_to_array(self.input())
            data = np.ravel(data, order=order)
            self.output.send(data)


class NumpyUnaryLinearAlgebraNode(Node):
    operations = {
        'np.linalg.norm': np.linalg.norm,
        'euclidean_distance': np.linalg.norm,
        'np.linalg.det': np.linalg.det,
        'np.linalg.matrix_rank': np.linalg.matrix_rank
    }

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyUnaryLinearAlgebraNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = np.linalg.norm
        if label in self.operations:
            self.op = self.operations[label]
        self.input = self.add_input('input', triggers_execution=True)
        out_name = 'norm'
        if self.label == 'euclidean_distance':
            out_name = 'euclidean_distance'
        elif self.label == 'np.linalg.det':
            out_name = 'determiner'
        elif self.label == 'np.linalg.matrix_rank':
            out_name = 'matrix rank'
        self.output = self.add_output(out_name)

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input())
            self.output.send(self.op(data))


class NumpyShapeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyShapeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('np in', triggers_execution=True)
        self.output = self.add_output('shape')

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            data = any_to_array(data)
            shape = list(data.shape)
            self.output.send(shape)


class NumpyRotateNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyRotateNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        k = 1
        axis1 = 0
        axis2 = 1
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                k = int(v)
        if len(args) > 1:
            v, t = decode_arg(args, 1)
            if t in [int, float]:
                axis1 = int(v)
        if len(args) > 2:
            v, t = decode_arg(args, 2)
            if t in [int, float]:
                axis2 = int(v)

        self.input = self.add_input('input', triggers_execution=True)
        self.k = self.add_input('k', widget_type='input_int', default_value=k)
        self.axis1 = self.add_input('axis 1', widget_type='input_int', default_value=axis1)
        self.axis2 = self.add_input('axis 2', widget_type='input_int', default_value=axis2)

        self.output = self.add_output('rotated array')

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input())
            if len(data.shape) == 1:
                data = np.expand_dims(data, axis=0)

            axis1 = self.axis1()
            if axis1 < 0:
                axis1 = 0
            elif axis1 >= len(data.shape):
                axis1 = len(data.shape) - 1

            axis2 = self.axis2()
            if axis2 < 0:
                axis2 = 0
            elif axis2 >= len(data.shape):
                axis2 = len(data.shape) - 1

            if axis1 != axis2:
                rotated_data = np.rot90(data, k=self.k(), axes=(axis1, axis2))
                self.output.send(rotated_data)

