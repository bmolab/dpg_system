from copy import deepcopy
import dearpygui.dearpygui as dpg
from dpg_system.node import Node, NodeInput, LoadDialog, SaveDialog
from dpg_system.conversion_utils import *
import os
import re
import time
import traceback
import warnings
import numpy as np
from scipy.stats import median_abs_deviation


def clamp_axis(axis, ndim):
    # negative counts from the end, as in numpy; anything still out of range
    # is clamped to the first or last axis
    if axis < 0:
        axis += ndim
    return min(max(axis, 0), max(ndim - 1, 0))

# add random, linspace, ones, zeros
def register_numpy_nodes():
    Node.app.register_node('np.linalg.norm', NumpyLinalgNormNode.factory)
    Node.app.register_node('euclidean_distance', NumpyLinalgNormNode.factory)
    Node.app.register_node('np.distance_from_target', NumpyDistanceFromTargetNode.factory)
    Node.app.register_node('np.proximity_to_target', NumpyProximityToTargetNode.factory)
    Node.app.register_node('proximity_trigger', NumpyProximityTriggerNode.factory)

    Node.app.register_node('np.linalg.det', NumpyUnaryLinearAlgebraNode.factory)
    Node.app.register_node('np.linalg.matrix_rank', NumpyUnaryLinearAlgebraNode.factory)
    Node.app.register_node('np.linalg.inv', NumpyUnaryLinearAlgebraNode.factory)
    Node.app.register_node('np.linalg.solve', NumpyLinalgSolveNode.factory)
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
    Node.app.register_node('np.amax', NumpyUnaryNode.factory)
    Node.app.register_node('np.amin', NumpyUnaryNode.factory)
    Node.app.register_node('np.argmax', NumpyUnaryNode.factory)
    Node.app.register_node('np.argmin', NumpyUnaryNode.factory)
    Node.app.register_node('np.nansum', NumpyUnaryNode.factory)
    Node.app.register_node('np.nanmean', NumpyUnaryNode.factory)
    Node.app.register_node('np.nanmedian', NumpyUnaryNode.factory)
    Node.app.register_node('np.shape', NumpyShapeNode.factory)
    Node.app.register_node('np.matmul', NumpyMatMulNode.factory)
    Node.app.register_node('np.stack', NumpyBinaryNode.factory)
    Node.app.register_node('np.concatenate', NumpyBinaryNode.factory)
    Node.app.register_node('np.rand', NumpyGeneratorNode.factory)
    Node.app.register_node('np.ones', NumpyGeneratorNode.factory)
    Node.app.register_node('np.zeros', NumpyGeneratorNode.factory)
    Node.app.register_node('np.linspace', NumpyLinSpaceNode.factory)
    Node.app.register_node('np.grid', NumpyMGridNode.factory)
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
    Node.app.register_node('np.line_intersection', NumpyLineIntersectionNode.factory)
    Node.app.register_node('np.add_alpha', NumpyAddAlphaNode.factory)
    Node.app.register_node('np.rolling_buffer', NumpyRollingBufferNode.factory)
    Node.app.register_node('np.[]', NumpySubtensorNode.factory)
    Node.app.register_node('np.edit', NumpyEditNode.factory)
    Node.app.register_node('np.sequence', NumpySequenceNode.factory)
    Node.app.register_node('np.load', NumpyLoadNode.factory)
    Node.app.register_node('np.reshape', NumpyReshapeNode.factory)
    Node.app.register_node('np.any', NumpyAnyNode.factory)
    Node.app.register_node('np.all', NumpyAllNode.factory)
    Node.app.register_node('rotate_position', RotatePositionNode.factory)
    Node.app.register_node('np.robust_mean', NumpyRobustMeanNode.factory)
    Node.app.register_node('np.where', NumpyWhereNode.factory)
    Node.app.register_node('np.argwhere', NumpyWhereNode.factory)
    Node.app.register_node('np.sort', NumpySortNode.factory)
    Node.app.register_node('np.argsort', NumpySortNode.factory)
    Node.app.register_node('np.diff', NumpyDiffNode.factory)
    Node.app.register_node('np.cumsum', NumpyCumulativeNode.factory)
    Node.app.register_node('np.cumprod', NumpyCumulativeNode.factory)
    Node.app.register_node('np.split', NumpySplitNode.factory)
    Node.app.register_node('np.logical_and', NumpyLogicalNode.factory)
    Node.app.register_node('np.logical_or', NumpyLogicalNode.factory)
    Node.app.register_node('np.logical_xor', NumpyLogicalNode.factory)
    Node.app.register_node('np.logical_not', NumpyLogicalNode.factory)
    Node.app.register_node('np.isnan', NumpyFiniteTestNode.factory)
    Node.app.register_node('np.isfinite', NumpyFiniteTestNode.factory)
    Node.app.register_node('np.nan_to_num', NumpyNanToNumNode.factory)
    Node.app.register_node('np.percentile', NumpyPercentileNode.factory)
    Node.app.register_node('np.histogram', NumpyHistogramNode.factory)
    Node.app.register_node('np.interp', NumpyInterpNode.factory)
    Node.app.register_node('np.convolve', NumpyConvolveNode.factory)
    Node.app.register_node('np.correlate', NumpyConvolveNode.factory)
    Node.app.register_node('np.linalg.pinv', NumpyUnaryLinearAlgebraNode.factory)
    Node.app.register_node('np.linalg.lstsq', NumpyLinalgLstsqNode.factory)
    Node.app.register_node('np.einsum', NumpyEinsumNode.factory)
    Node.app.register_node('np.ptp', NumpyUnaryNode.factory)
    Node.app.register_node('np.unique', NumpyUniqueNode.factory)
    Node.app.register_node('np.cov', NumpyCovarianceNode.factory)
    Node.app.register_node('np.corrcoef', NumpyCovarianceNode.factory)
    Node.app.register_node('np.gradient', NumpyGradientNode.factory)
    Node.app.register_node('np.fft.rfft', NumpyRfftNode.factory)
    Node.app.register_node('np.fft.irfft', NumpyIrfftNode.factory)
    Node.app.register_node('np.take', NumpyTakeNode.factory)
    Node.app.register_node('np.pad', NumpyPadNode.factory)
    Node.app.register_node('np.tile', NumpyTileNode.factory)
    Node.app.register_node('np.digitize', NumpyDigitizeNode.factory)
    Node.app.register_node('np.arange', NumpyArangeNode.factory)
    Node.app.register_node('np.eye', NumpyEyeNode.factory)
    Node.app.register_node('np.diag', NumpyDiagNode.factory)
    Node.app.register_node('np.polyfit', NumpyPolyfitNode.factory)
    Node.app.register_node('np.polyval', NumpyPolyvalNode.factory)
    Node.app.register_node('np.linalg.svd', NumpySvdNode.factory)
    Node.app.register_node('np.linalg.eigh', NumpyEighNode.factory)
    Node.app.register_node('np.full', NumpyFullNode.factory)
    Node.app.register_node('np.meshgrid', NumpyMeshgridNode.factory)
    Node.app.register_node('np.trace', NumpyTraceNode.factory)
    Node.app.register_node('np.tril', NumpyTriangleNode.factory)
    Node.app.register_node('np.triu', NumpyTriangleNode.factory)
    Node.app.register_node('np.take_along_axis', NumpyTakeAlongAxisNode.factory)
    Node.app.register_node('np.tensordot', NumpyTensordotNode.factory)
    Node.app.register_node('np.linalg.qr', NumpyQrNode.factory)
    Node.app.register_node('np.linalg.eig', NumpyEigNode.factory)
    Node.app.register_node('np.real', NumpyComplexPartNode.factory)
    Node.app.register_node('np.imag', NumpyComplexPartNode.factory)
    Node.app.register_node('np.angle', NumpyComplexPartNode.factory)
    Node.app.register_node('np.unwrap', NumpyUnwrapNode.factory)
    Node.app.register_node('np.nanmin', NumpyUnaryNode.factory)
    Node.app.register_node('np.nanmax', NumpyUnaryNode.factory)
    Node.app.register_node('np.nanstd', NumpyUnaryNode.factory)
    Node.app.register_node('np.nanvar', NumpyUnaryNode.factory)
    Node.app.register_node('np.average', NumpyAverageNode.factory)
    Node.app.register_node('np.histogram2d', NumpyHistogram2dNode.factory)
    Node.app.register_node('np.top_k', NumpyTopKNode.factory)
    Node.app.register_node('np.isclose', NumpyIsCloseNode.factory)
    Node.app.register_node('np.allclose', NumpyIsCloseNode.factory)
    Node.app.register_node('np.trapezoid', NumpyTrapezoidNode.factory)
    Node.app.register_node('np.cumulative_trapezoid', NumpyTrapezoidNode.factory)
    Node.app.register_node('np.fft.fft', NumpyFftNode.factory)
    Node.app.register_node('np.fft.ifft', NumpyIfftNode.factory)
    Node.app.register_node('np.save', NumpySaveNode.factory)
    Node.app.register_node('np.savetxt', NumpySaveNode.factory)
    Node.app.register_node('np.loadtxt', NumpyLoadTextNode.factory)
    Node.app.register_node('np.random.normal', NumpyRandomArrayNode.factory)
    Node.app.register_node('np.random.integers', NumpyRandomArrayNode.factory)
    Node.app.register_node('np.random.choice', NumpyRandomChoiceNode.factory)
    Node.app.register_node('np.random.permutation', NumpyRandomPermutationNode.factory)
    Node.app.register_node('np.sliding_window_view', NumpySlidingWindowNode.factory)
    Node.app.register_node('np.maximum.accumulate', NumpyRunningExtremeNode.factory)
    Node.app.register_node('np.minimum.accumulate', NumpyRunningExtremeNode.factory)
    Node.app.register_node('np.insert', NumpyInsertNode.factory)
    Node.app.register_node('np.delete', NumpyDeleteNode.factory)
    Node.app.register_node('np.append', NumpyAppendNode.factory)
    Node.app.register_node('np.nanargmax', NumpyUnaryNode.factory)
    Node.app.register_node('np.nanargmin', NumpyUnaryNode.factory)
    Node.app.register_node('np.bincount', NumpyBincountNode.factory)

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
                # self.min and self.max are NodeInputs — call them to read values.
                cur_min = self.min()
                cur_max = self.max()
                if self.dtype == np.uint8:
                    if cur_min < 0:
                        self.min.set(0.0)
                    if cur_max == 1.0 or cur_max < 255:
                        self.max.set(255.0)
                elif self.dtype == np.int64:
                    if cur_min < -32768:
                        self.min.set(-32768)
                    if cur_max == 1.0:
                        self.max.set(32767)
                elif self.dtype in [float, np.double, np.float32]:
                    if cur_min == -32768:
                        self.min.set(0.0)
                    if cur_max == 255:
                        self.max.set(1.0)
                    elif cur_max == 32767:
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
        steps = any_to_int(self.steps())
        if steps < 0:
            print(f'{self.label}: steps must be 0 or more, got {steps}')
            return
        out_array = self.op(any_to_float(self.start()), any_to_float(self.stop()), steps, dtype=self.dtype)
        self.output.send(out_array)


class NumpyMGridNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyMGridNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        min_x = 0.0
        max_x = 1.0
        min_y = 0.0
        max_y = 1.0
        divisions_x = 10
        divisions_y = 10

        if len(args) > 0:
            min_x = any_to_float(args[0])
        if len(args) > 1:
            max_x = any_to_float(args[1])
        if len(args) > 2:
            min_y = any_to_float(args[2])
        if len(args) > 3:
            max_y = any_to_float(args[3])
        if len(args) > 4:
            divisions_x = any_to_int(args[4])
        if len(args) > 5:
            divisions_y = any_to_int(args[5])

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.min_x = self.add_property('min x', widget_type='drag_float', default_value=min_x)
        self.max_x = self.add_property('max x', widget_type='drag_float', default_value=max_x)
        self.min_y = self.add_property('min y', widget_type='drag_float', default_value=min_y)
        self.max_y = self.add_property('max y', widget_type='drag_float', default_value=max_y)
        self.divisions_x = self.add_property('divisions x', widget_type='drag_int', default_value=divisions_x)
        self.divisions_y = self.add_property('divisions y', widget_type='drag_int', default_value=divisions_y)
        self.z_value = self.add_property('z', widget_type='drag_float', default_value=0.0)

        self.output = self.add_output('grid points')

    def execute(self):
        min_x = any_to_float(self.min_x())
        max_x = any_to_float(self.max_x())
        min_y = any_to_float(self.min_y())
        max_y = any_to_float(self.max_y())
        div_x = any_to_int(self.divisions_x())
        div_y = any_to_int(self.divisions_y())
        z = any_to_float(self.z_value())

        if div_x < 2:
            div_x = 2
        if div_y < 2:
            div_y = 2

        yy, xx = np.mgrid[min_y:max_y:complex(0, div_y), min_x:max_x:complex(0, div_x)]
        zz = np.full_like(xx, z)
        points = np.stack([xx, yy, zz], axis=-1).astype(np.float32)
        self.output.send(points)


class NumpyNodeWithAxisNode(Node):
    full_axis_items = ['None', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    axis_dict = {'None': None, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.axis = None
        self.current_axis_count = 2
        self.dim_option = None
        self.first_dim_option = 0
        # np.stack can put its new axis after the last one: axis == ndim
        self.extra_axis = 0
        # the axis this call uses: self.axis clamped to the array in hand.
        # self.axis itself is the user's choice and is never lowered.
        self.call_axis = None

    def add_dim_option(self, none_allowed=True, extra_axis=0):
        self.dim_option = self.add_input('axis', widget_type='combo', default_value=str(self.axis),
                                         callback=self.axis_changed)
        self.extra_axis = extra_axis
        self.first_dim_option = 0 if none_allowed else 1
        # the menu for current_axis_count dims, until an array shows the real count
        self.dim_option.widget.combo_items = NumpyNodeWithAxisNode.full_axis_items[
            self.first_dim_option:self.current_axis_count + 1 + extra_axis]

    def axis_changed(self):
        if self.dim_option is not None:
            axis = self.dim_option()
            # a message ('None', 1) arrives as a list
            if isinstance(axis, (list, tuple)):
                axis = axis[0] if len(axis) > 0 else None
            if axis is None or (isinstance(axis, str) and axis.strip() == 'None'):
                if self.first_dim_option == 0:
                    self.axis = None
            else:
                try:
                    self.axis = int(float(axis))
                except (TypeError, ValueError):
                    return
            self.dim_option.set(str(self.axis))

    def adjust_dim_option(self, data):
        dims = len(data.shape)
        if dims != self.current_axis_count and self.dim_option is not None:
            dpg.configure_item(self.dim_option.widget.uuid, items=NumpyNodeWithAxisNode.full_axis_items[self.first_dim_option:dims + 1 + self.extra_axis])
            self.current_axis_count = dims
        # clamp for this call only, so a 1-D array through 'axis 1' does not
        # leave the axis at 0 for the 2-D arrays that follow
        self.call_axis = self.axis
        if self.axis is not None:
            last = dims - 1 + self.extra_axis
            if self.axis > last:
                self.call_axis = max(last, 0)
            elif self.axis < -dims - self.extra_axis:
                return False
        if self.call_axis is None or dims + self.extra_axis > self.call_axis:
            return True
        return False


class NumpyUnaryNode(NumpyNodeWithAxisNode):
    operations = {'np.sum': np.sum, 'np.mean': np.mean, 'np.std': np.std, 'np.var': np.var, 'np.median': np.median, 'np.amax': np.amax, 'np.amin': np.amin, 'np.argmax': np.argmax, 'np.argmin': np.argmin, 'np.nansum': np.nansum, 'np.nanmean': np.nanmean, 'np.nanmedian': np.nanmedian, 'np.ptp': np.ptp, 'np.nanmin': np.nanmin, 'np.nanmax': np.nanmax, 'np.nanstd': np.nanstd, 'np.nanvar': np.nanvar, 'np.nanargmax': np.nanargmax, 'np.nanargmin': np.nanargmin}

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyUnaryNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = np.sum
        if label in self.operations:
            self.op = self.operations[label]
        if len(args) > 0:
            d, t = decode_arg(args, 0)
            if t == int:
                self.axis = d
        output_name = label.split('.')[-1]
        self.input = self.add_input('in', triggers_execution=True)
        # None reduces the whole array to one value
        self.add_dim_option()
        self.output = self.add_output(output_name)
        # patches saved before the fix carry the old mistaken outlet name
        self.output.name_archive.append('np')

    def execute(self):
        input_value = any_to_array(self.input())
        if self.adjust_dim_option(input_value):
            with warnings.catch_warnings():
                # the nan- reductions warn on an all-nan slice; its nan is the answer
                warnings.simplefilter('ignore', RuntimeWarning)
                try:
                    output_value = self.op(input_value, axis=self.call_axis)
                except ValueError as e:
                    # nanargmax / nanargmin of an all-nan slice: no position to give
                    if self.app.verbose:
                        print(self.label, e)
                    return
            self.output.send(output_value)
        else:
            if self.app.verbose:
                print(self.label, 'dim =', self.axis, 'out of range', 'for shape', input_value.shape)


class NumpyBinaryNode(NumpyNodeWithAxisNode):
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
        output_name = label.split('.')[-1]
        self.input = self.add_input('in', triggers_execution=True)
        self.input_2 = self.add_input('in 2')
        if label == 'np.stack':
            # np.stack has no flattened (None) form, and its new axis may go
            # after the last one (axis == ndim)
            self.add_dim_option(none_allowed=False, extra_axis=1)
        else:
            self.add_dim_option()
        self.output = self.add_output(output_name)
        # patches saved before the fix carry the old mistaken outlet name
        self.output.name_archive.append('np')

    def execute(self):
        if self.input_2.fresh_input:
            self.input_2_vector = any_to_array(self.input_2())

        input_value = any_to_array(self.input())
        if self.adjust_dim_option(input_value):
            if self.input_2_vector is not None:
                try:
                    joined = self.op((input_value, self.input_2_vector), axis=self.call_axis)
                except ValueError as e:
                    print(f'{self.label}: cannot join shapes {input_value.shape} and '
                          f'{self.input_2_vector.shape} on axis {self.call_axis}: {e}')
                    return
                self.output.send(joined)


class NumpyDotProductNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyDotProductNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.operand = None
        self.input = self.add_input('in 1', triggers_execution=True)
        self.operand_input = self.add_input('in 2')
        self.output = self.add_output('dot product')

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = any_to_array(self.operand_input())
        if self.operand is None:
            return
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

        self.operand = None
        self.input = self.add_input('in 1', triggers_execution=True)
        self.operand_input = self.add_input('in 2')
        if self.label == 'np.outer':
            self.output = self.add_output('outer_product')
        elif self.label == 'np.inner':
            self.output = self.add_output('inner_product')

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = any_to_array(self.operand_input())
        if self.operand is None:
            return
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

        self.operand = None
        self.input = self.add_input('in 1', triggers_execution=True)
        self.operand_input = self.add_input('in 2')
        self.output = self.add_output('mat mul result')

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = any_to_array(self.operand_input())
        if self.operand is None:
            return
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

        self.operand = None
        self.input = self.add_input('in 1', triggers_execution=True)
        self.operand_input = self.add_input('in 2')
        self.output = self.add_output('cross product')

    def execute(self):
        if self.operand_input.fresh_input:
            self.operand = any_to_array(self.operand_input())
        if self.operand is None:
            return
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
            # negative counts from the end, as in numpy
            if -len(data.shape) <= axis < len(data.shape):
                if data.shape[axis] == 1:
                    data = np.squeeze(data, axis=axis)
                    self.output.send(data)
                else:
                    if self.app.verbose:
                        print(f'{self.label}: axis {axis} has size {data.shape[axis]} '
                              f'(not 1), cannot squeeze')
            else:
                if self.app.verbose:
                    print(f'{self.label}: axis {axis} out of range for shape {data.shape}')


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
            # the result has one more axis: -1 appends a new last axis, as in numpy
            out_dims = len(data.shape) + 1
            if axis < -out_dims:
                axis = 0
            elif axis >= out_dims:
                axis = out_dims - 1
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
        data = any_to_array(self.input())
        repeats = self.repeats()
        if repeats < 0:
            print(f'{self.label}: repeats must be 0 or more, got {repeats}')
            return
        axis = self.axis()
        if axis < 0:
            # negative counts from the end, as in numpy
            axis = max(axis + len(data.shape), 0)
        if axis == len(data.shape):
            # one past the last axis: repeat along a new last axis
            data = np.expand_dims(data, axis)
        elif axis > len(data.shape):
            axis = max(len(data.shape) - 1, 0)
        repeated_data = np.repeat(data, repeats=repeats, axis=axis)
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
            if isinstance(data, np.ndarray):
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
        if self.label == 'np.max':
            # np.max has only its 'max' limit (the floor): the first argument sets it
            if len(args) > 0:
                self.max = any_to_float(args[0])
        else:
            # np.clip: min max.  np.min has only its 'min' limit (the ceiling)
            if len(args) > 0:
                self.min = any_to_float(args[0])
            if len(args) > 1:
                self.max = any_to_float(args[1])
        # print(self.min, self.max)
        self.min_property = None
        self.max_property = None
        if self.label in ['np.min', 'np.clip']:
            # print('add min prop')
            self.min_property = self.add_property('min', widget_type='drag_float', default_value=self.min, callback=self.min_changed)
        if self.label in ['np.max', 'np.clip']:
            # print('add max prop')
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
            if isinstance(data, np.ndarray):
                if self.mode == 0:
                    self.output.send(np.clip(data, self.min, self.max))
                elif self.mode == 1:
                    # np.min is a reduction; np.minimum is the element-wise clamp.
                    self.output.send(np.minimum(data, self.min))
                elif self.mode == 2:
                    self.output.send(np.maximum(data, self.max))


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
            axis = clamp_axis(self.axis(), len(data.shape))
            repeated_data = np.roll(data, shift=self.shifts(), axis=axis)
            self.output.send(repeated_data)


class NumpyAddAlphaNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyAddAlphaNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('input', triggers_execution=True)
        self.output = self.add_output('image with alpha')

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input())
            if len(data.shape) == 3:
                if data.shape[2] == 4:
                    # already has an alpha channel: pass it through unchanged
                    self.output.send(data)
                    return
                # alpha = mean of the channels, in the input's dtype (uint8 stays uint8)
                mean = np.mean(data, axis=2, keepdims=True).astype(data.dtype)
                data = np.append(data, mean, axis=2)
                self.output.send(data)
            else:
                print(f'{self.label}: expected 3-D input (H, W, C), got shape {data.shape}')



class NumpyLineIntersectionNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyLineIntersectionNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.point1_input = self.add_input('line 1 start', triggers_execution=True)
        self.point2_input = self.add_input('line 1 end')
        self.point3_input = self.add_input('line 2 start')
        self.point4_input = self.add_input('line 2 end')
        self.directed = self.add_property('direction matters', widget_type='checkbox', default_value=True)
        self.must_hit_line_2_segment = self.add_property('intersect line 2 segment', widget_type='checkbox', default_value=True)

        self.valid_intersection = self.add_output('intersection is valid')
        self.output = self.add_output('point of intersection')
        self.relative_point = self.add_output('fraction of segment 2')
        self._warned_shape = False

    def execute(self):
        def _ok(p):
            # Require a 1-D vector with at least 2 components (x, y).
            # Without ndim check, scalar / 0-d input raises IndexError on shape[0].
            return isinstance(p, np.ndarray) and p.ndim >= 1 and p.shape[0] >= 2

        a1 = any_to_array(self.point1_input())
        if not _ok(a1):
            return
        a2 = any_to_array(self.point2_input())
        if not _ok(a2):
            return
        b1 = any_to_array(self.point3_input())
        if not _ok(b1):
            return
        b2 = any_to_array(self.point4_input())
        if not _ok(b2):
            return
        # two dimensions only: np.cross on 3-number points (4 homogeneous) raises
        if any(p.ndim != 1 or p.shape[0] != 2 for p in (a1, a2, b1, b2)):
            if not self._warned_shape:
                print(f'{self.label}: points must be x y pairs (2-D only), got shapes '
                      f'{[p.shape for p in (a1, a2, b1, b2)]}')
                self._warned_shape = True
            return

        intersection = self.get_intersect(a1, a2, b1, b2)
        line_2_length = np.linalg.norm(b1 - b2)
        if not np.all(np.isfinite(intersection)) or line_2_length == 0:
            # parallel lines, or a line whose two points coincide: no crossing
            self.relative_point.send(float('inf'))
            self.output.send(intersection)
            self.valid_intersection.send(False)
            return
        direction_ok = True
        if self.directed():
            a1_intersect_length = np.linalg.norm(a1 - intersection)
            a2_intersect_length = np.linalg.norm(a2 - intersection)
            if a1_intersect_length <= a2_intersect_length:
                direction_ok = False
        segment_intersect_ok = True
        intersection_seg_length_1 = np.linalg.norm(b1 - intersection)
        intersection_seg_length_2 = np.linalg.norm(b2 - intersection)
        fraction = float(intersection_seg_length_1 / line_2_length)
        if self.must_hit_line_2_segment():
            if intersection_seg_length_1 > line_2_length or intersection_seg_length_2 > line_2_length:
                segment_intersect_ok = False
        self.relative_point.send(fraction)
        self.output.send(intersection)
        if direction_ok and segment_intersect_ok:
            self.valid_intersection.send(True)
        else:
            self.valid_intersection.send(False)

    def get_intersect(self, a1, a2, b1, b2):
        """
        Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
        a1: [x, y] a point on the first line
        a2: [x, y] another point on the first line
        b1: [x, y] a point on the second line
        b2: [x, y] another point on the second line
        """
        s = np.vstack([a1, a2, b1, b2])        # s for stacked
        h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        if z == 0:                          # lines are parallel
            return np.array([float('inf'), float('inf')])
        return np.array([x/z, y/z])

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
            axis = clamp_axis(self.axis(), len(data.shape))
            flipped_data = np.flip(data, axis=axis)
            self.output.send(flipped_data)


class NumpyTransposeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyTransposeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        # optional new order of the axes, e.g. np.transpose 1 0 2;
        # empty reverses them all (numpy's default)
        order = ' '.join(args) if args else ''
        self.input = self.add_input('input', triggers_execution=True)
        self.axes_input = self.add_input('axes', widget_type='text_input', widget_width=80, default_value=order)

        self.output = self.add_output('transposed array')

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input())
            order = self.axes_input()
            if not isinstance(order, str):
                order = ' '.join(str(v) for v in np.ravel(np.asarray(order, dtype=object)))
            axes = [int(v) for v in re.findall(r'[-+]?\d+', order)]
            if len(axes) == 0:
                transposed_data = np.transpose(data)
            else:
                try:
                    transposed_data = np.transpose(data, axes)
                except ValueError as e:
                    print(f'{self.label}: axes {axes} do not fit shape {data.shape} '
                          f'(give each axis once): {e}')
                    return
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

class NumpyReshapeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyReshapeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.shape = [-1]
        if len(args) > 0:
            self.shape = []
            for arg in args:
                self.shape.append(any_to_int(arg))
        shape_string = str(self.shape)
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.shape_input = self.add_input('', widget_type='text_input', widget_width=200, default_value=shape_string,
                                                  callback=self.shape_changed)
        self.output = self.add_output('output')

    def shape_changed(self):
        shape_text = self.shape_input()
        if not isinstance(shape_text, str):
            # a list, array or single number of dims arriving at the inlet
            dims = [any_to_int(v) for v in np.ravel(np.asarray(shape_text, dtype=object))]
            shape_text = str(dims)
            self.shape_input.set(shape_text)
        shape_list = re.findall(r'[-+]?\d+', shape_text)
        shape = []
        for dim_text in shape_list:
            shape.append(any_to_int(dim_text))
        self.shape = shape

    def execute(self):
        input_array = any_to_array(self.input())
        if input_array is not None:
            try:
                reshaped_array = np.reshape(input_array, self.shape)
                self.output.send(reshaped_array)
            except Exception as e:
                print(f'{self.label}: reshape to {self.shape} failed for '
                      f'shape {getattr(input_array, "shape", "?")}: '
                      f'{type(e).__name__}: {e}')

class NumpyUnaryLinearAlgebraNode(Node):
    operations = {
        'np.linalg.det': np.linalg.det,
        'np.linalg.matrix_rank': np.linalg.matrix_rank,
        'np.linalg.inv': np.linalg.inv,
        'np.linalg.pinv': np.linalg.pinv
    }
    needs_square = ['np.linalg.det', 'np.linalg.inv']

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
        out_name = 'determinant'

        if self.label == 'np.linalg.det':
            out_name = 'determinant'
        elif self.label == 'np.linalg.matrix_rank':
            out_name = 'matrix rank'
        elif self.label == 'np.linalg.inv':
            out_name = 'inverse matrix'
        elif self.label == 'np.linalg.pinv':
            out_name = 'pseudo-inverse'
        self.output = self.add_output(out_name)
        # patches saved before the rename carry the old outlet name
        self.output.name_archive.append('determiner')

    def execute(self):
        if self.input.fresh_input:
            data = any_to_array(self.input())
            if self.label == 'np.linalg.pinv' and len(data.shape) < 2:
                if self.app.verbose:
                    print(self.label, 'needs a matrix, got shape', data.shape)
                return
            if self.label in self.needs_square:
                if len(data.shape) < 2 or data.shape[-1] != data.shape[-2]:
                    if self.app.verbose:
                        print(self.label, 'needs a square matrix, got shape', data.shape)
                    return
            try:
                self.output.send(self.op(data))
            except np.linalg.LinAlgError as e:
                if self.app.verbose:
                    print(self.label, e)


class NumpyLinalgSolveNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyLinalgSolveNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.b_value = None
        self.input = self.add_input('A matrix', triggers_execution=True)
        self.b_input = self.add_input('b')
        self.output = self.add_output('x')

    def execute(self):
        if self.b_input.fresh_input:
            self.b_value = any_to_array(self.b_input())
        a = any_to_array(self.input())
        if self.b_value is None:
            return
        if len(a.shape) < 2 or a.shape[-1] != a.shape[-2]:
            if self.app.verbose:
                print(self.label, 'A must be a square matrix, got shape', a.shape)
            return
        try:
            self.output.send(np.linalg.solve(a, self.b_value))
        except np.linalg.LinAlgError as e:
            if self.app.verbose:
                print(self.label, e)
        except ValueError:
            if self.app.verbose:
                print(self.label, 'shapes incompatible:', a.shape, 'vs', self.b_value.shape)


class NumpyDistanceFromTargetNode(NumpyNodeWithAxisNode):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyDistanceFromTargetNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('input', triggers_execution=True)
        self.set_target = self.add_input('set target', widget_type='button', callback=self.set_the_target)
        self.target = None
        self.add_dim_option()
        self.output = self.add_output('norm')

    def set_the_target(self):
        target = self.input()
        if target is not None:
            self.target = any_to_array(target)
            self.execute()

    def save_custom(self, container):
        if self.target is not None:
            container['target'] = np.asarray(self.target, dtype=float).tolist()

    def load_custom(self, container):
        # target is plain node state, not a widget, so restore order doesn't matter
        if 'target' in container:
            self.target = np.array(container['target'], dtype=float)

    def execute(self):
        input_value = any_to_array(self.input())
        if self.target is not None:
            try:
                diff = input_value - self.target
            except ValueError as e:
                if self.app.verbose:
                    print(f'{self.label}: input shape {input_value.shape} incompatible '
                          f'with target shape {self.target.shape}: {e}')
                return
            if self.adjust_dim_option(input_value):
                self.output.send(np.linalg.norm(diff, axis=self.call_axis))


class NumpyProximityToTargetNode(Node):
    # Proximity of a tracker to a set of targets: 1 at a target, falling to 0 at
    # 'radius', shaped by 'exponent' (>1 tightens the zone, <1 broadens it).
    # Targets are captured from the tracker itself: pick a slot with 'target index',
    # press 'capture'. Unset slots (index jumped past the end) report 0.
    # Output feeds polyphonic_sampler~ 'fade': [[sound_id, fade], ...], target i
    # driving sound 'first sound' + i. Sparse like effort_fader: a sound is sent
    # while nonzero, gets one 0.0 the frame it falls silent, then is omitted.
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyProximityToTargetNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.targets = None     # (N, D) array, NaN rows = unset slots
        self.sounding = set()   # sound ids last sent a nonzero fade
        self.input = self.add_input('input', triggers_execution=True)
        self.index_input = self.add_input('target index', widget_type='input_int', default_value=0, min=0)
        self.capture_input = self.add_input('capture', widget_type='button', callback=self.capture)
        self.capture_input.name_archive.append('set target')
        self.targets_input = self.add_input('targets', callback=self.receive_targets)
        self.clear_input = self.add_input('clear', widget_type='button', callback=self.clear)
        self.radius = self.add_input('radius', widget_type='drag_float', min=0.001, default_value=.5)
        self.radius.name_archive.append('threshold')
        self.exponent = self.add_input('exponent', widget_type='drag_float', min=0.01, default_value=1.0)
        self.first_sound = self.add_input('first sound', widget_type='input_int', default_value=0, min=0,
                                          callback=self.silence)
        self.output = self.add_output('fade')
        self.nearest_out = self.add_output('nearest')
        self.targets_out = self.add_output('targets')

    def capture(self):
        point = self.input()
        if point is None:
            return
        point = any_to_array(point).astype(float).reshape(-1)
        index = max(int(self.index_input()), 0)
        if self.targets is None or self.targets.shape[1] != point.shape[0]:
            self.targets = np.full((index + 1, point.shape[0]), np.nan)
        elif index >= self.targets.shape[0]:
            pad = np.full((index + 1 - self.targets.shape[0], point.shape[0]), np.nan)
            self.targets = np.concatenate([self.targets, pad])
        self.targets[index] = point
        self.send_targets()
        self.execute()

    def receive_targets(self):
        data = self.targets_input()
        if data is None:
            return
        targets = any_to_array(data).astype(float)
        if targets.ndim == 1:
            targets = targets.reshape(1, -1)
        self.targets = targets
        self.send_targets()
        self.execute()

    def clear(self):
        self.silence()
        self.targets = None
        self.targets_out.send([])

    def silence(self):
        if self.sounding:
            self.output.send([[sid, 0.0] for sid in sorted(self.sounding)])
            self.sounding = set()

    def send_targets(self):
        if self.targets is not None:
            self.targets_out.send(self.targets.copy())

    def save_custom(self, container):
        if self.targets is not None:
            container['targets'] = [[None if np.isnan(v) else float(v) for v in row] for row in self.targets]

    def load_custom(self, container):
        if 'targets' in container:
            rows = container['targets']
            if len(rows) > 0:
                self.targets = np.array([[np.nan if v is None else v for v in row] for row in rows], dtype=float)

    def execute(self):
        if self.targets is None:
            return
        tracker = self.input()
        if tracker is None:
            return
        tracker = any_to_array(tracker).astype(float)
        if tracker.shape[-1] != self.targets.shape[1]:
            if self.app.verbose:
                print(f'{self.label}: input shape {tracker.shape} incompatible '
                      f'with targets of size {self.targets.shape[1]}')
            return
        # tracker (..., D) against targets (N, D) -> distances (..., N)
        distance = np.linalg.norm(tracker[..., None, :] - self.targets, axis=-1)
        radius = self.radius()
        proximity = np.clip((radius - distance) / radius, 0.0, 1.0)
        proximity = np.nan_to_num(proximity, nan=0.0) ** self.exponent()
        nearest = np.where(proximity.max(axis=-1) > 0, proximity.argmax(axis=-1), -1)
        self.nearest_out.send(int(nearest) if nearest.ndim == 0 else nearest)

        # one sound per target: with several trackers the closest one sets the fade
        levels = np.round(proximity.reshape(-1, proximity.shape[-1]).max(axis=0), 4)
        base = int(self.first_sound())
        fades = {base + i: float(level) for i, level in enumerate(levels) if level > 0.0}
        for sid in self.sounding - fades.keys():
            fades[sid] = 0.0
        self.sounding = {sid for sid, level in fades.items() if level > 0.0}
        if fades:
            self.output.send([[sid, fades[sid]] for sid in sorted(fades)])


class NumpyProximityTriggerNode(NumpyNodeWithAxisNode):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyProximityTriggerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('input', triggers_execution=True)
        self.arm_input = self.add_input('arm', widget_type='checkbox', default_value=False, callback=self.arm)
        self.reset_count_input = self.add_input('reset_count', widget_type='button', callback=self.reset_counter)
        self.set_target = self.add_input('set target', widget_type='button', callback=self.set_the_target)
        self.threshold = self.add_property('threshold', widget_type='drag_float', min=0.001, default_value=.3)
        self.release_threshold = self.add_property('release_threshold', widget_type='drag_float', min=0.001, default_value=.5)

        self.target = None
        self.active = False
        self.counter = 0
        self.add_dim_option()
        self.output = self.add_output('state')
        self.count_out = self.add_output('count')
        self.prox_out = self.add_output('proximity')

    def reset_counter(self):
        self.counter = 0
        self.count_out.send(self.counter)

    def arm(self):
        self.counter = 0
        if not self.in_loading_process:
            self.count_out.send(self.counter)

    def set_the_target(self):
        target = self.input()
        if target is not None:
            self.target = any_to_array(target)
            self.execute()

    def save_custom(self, container):
        # the whole target, any size; older patches saved only target_x/y/z
        if self.target is not None:
            container['target'] = np.asarray(self.target, dtype=float).tolist()

    def load_custom(self, container):
        if 'target' in container:
            self.target = np.array(container['target'], dtype=float)
        elif 'target_x' in container:
            x = float(container['target_x'])
            y = float(container['target_y'])
            z = float(container['target_z'])
            self.target = np.array([x, y, z])


    def execute(self):
        if self.arm_input():
            input_value = any_to_array(self.input())
            if self.target is not None:
                try:
                    diff = input_value - self.target
                except ValueError as e:
                    if self.app.verbose:
                        print(f'{self.label}: input shape {input_value.shape} incompatible '
                              f'with target shape {self.target.shape}: {e}')
                    return
                if self.adjust_dim_option(input_value):
                    distance = np.linalg.norm(diff, axis=self.call_axis)
                    self.prox_out.send(distance)
                    # with an axis the distance is per row: the nearest one triggers
                    # (any row inside threshold) and releases (all rows beyond release)
                    distance = float(np.min(distance)) if np.size(distance) > 0 else np.inf
                    if distance < self.threshold() and not self.active:
                        self.active = True
                        self.counter += 1
                        self.count_out.send(self.counter)
                        self.output.send(self.active)
                    elif distance >= self.release_threshold() and self.active:
                        self.active = False
                        self.output.send(self.active)



class NumpyLinalgNormNode(NumpyNodeWithAxisNode):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyLinalgNormNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = np.linalg.norm
        self.input = self.add_input('input', triggers_execution=True)

        self.add_dim_option()
        self.output = self.add_output('norm')

    def execute(self):
        input_value = any_to_array(self.input())

        if self.adjust_dim_option(input_value):
            self.output.send(np.linalg.norm(input_value, axis=self.call_axis))


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
            if data is None:
                return
            data = any_to_array(data)
            if not isinstance(data, np.ndarray):
                return
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

            axis1 = clamp_axis(self.axis1(), len(data.shape))
            axis2 = clamp_axis(self.axis2(), len(data.shape))

            if axis1 != axis2:
                rotated_data = np.rot90(data, k=self.k(), axes=(axis1, axis2))
                self.output.send(rotated_data)
            else:
                print(f'{self.label}: axis 1 and axis 2 are both axis {axis1} of a '
                      f'{len(data.shape)}-D array - a turn needs two different axes')


class NumpyRobustMeanNode(Node):
    """Accumulates incoming numpy arrays and computes a MAD-based
    outlier-rejected mean.  Useful for resolving a single best estimate
    (e.g. SMPL betas) from noisy per-frame predictions.

    Inputs:
        in          – numpy array (one sample per execution)
        threshold   – number of MADs from the median to accept (default 3.0)
    Outputs:
        robust mean – the outlier-rejected mean of accumulated samples
        inlier count – number of inlier samples used
    """

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyRobustMeanNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        threshold = 3.0
        if len(args) > 0:
            threshold = any_to_float(args[0])

        self.samples = []  # list of 1-D or N-D arrays

        self.input = self.add_input('in', triggers_execution=True)
        self.threshold_input = self.add_input('threshold', widget_type='drag_float',
                                              default_value=threshold)
        self.reset_button = self.add_property('reset', widget_type='button',
                                              callback=self.reset)

        # shows how many samples are held; editing it just snaps back to the count
        self.count_display = self.add_property('samples', widget_type='drag_int',
                                               default_value=0, callback=self.show_count)

        self.output = self.add_output('robust mean')
        self.inlier_output = self.add_output('inlier count')
        # the oldest samples are dropped beyond this, so the estimate follows
        # the recent past and the store cannot grow without limit (0 keeps all)
        self.max_samples_option = self.add_option('max samples', widget_type='drag_int',
                                                  default_value=1000, min=0)

    def show_count(self):
        self.count_display.set(len(self.samples))

    def reset(self):
        self.samples = []
        self.count_display.set(0)

    def execute(self):
        data = self.input()
        if data is None:
            return

        # Handle 'bang' – recompute from accumulated samples without adding
        if isinstance(data, str) and data == 'bang':
            pass
        else:
            arr = any_to_array(data).astype(np.float64)

            if arr.ndim >= 2:
                # Batch mode: treat as (N_samples, D) — replace accumulated buffer
                self.samples = [arr[i] for i in range(arr.shape[0])]
            else:
                # Single sample: accumulate
                sample = arr.flatten()
                if len(self.samples) > 0 and self.samples[-1].shape != sample.shape:
                    # a different kind of estimate: mixing lengths means nothing, start over
                    print(f'{self.label}: sample length changed from {self.samples[-1].shape[0]} '
                          f'to {sample.shape[0]} - starting a new set of samples')
                    self.samples = []
                self.samples.append(sample)

            # keep at most 'max samples', dropping the oldest (0 keeps all)
            max_samples = any_to_int(self.max_samples_option())
            if max_samples > 0 and len(self.samples) > max_samples:
                self.samples = self.samples[-max_samples:]
            self.count_display.set(len(self.samples))

        if len(self.samples) < 2:
            if len(self.samples) == 1:
                self.output.send(self.samples[0])
                self.inlier_output.send(1)
            return

        # Stack all accumulated samples: (N, D)
        stacked = np.stack(self.samples, axis=0)
        threshold = any_to_float(self.threshold_input())

        # Per-component median and MAD
        med = np.median(stacked, axis=0)
        mad = median_abs_deviation(stacked, axis=0)

        # Where MAD is zero (constant component), all samples are inliers
        mad_safe = np.where(mad > 0, mad, 1.0)

        # Inlier mask: a sample is an inlier if ALL its components are within threshold
        deviations = np.abs(stacked - med) / mad_safe
        # For components with zero MAD, deviation should be zero
        deviations = np.where(mad > 0, deviations, 0.0)
        mask = np.all(deviations < threshold, axis=1)

        inlier_count = int(np.sum(mask))
        if inlier_count > 0:
            robust = np.mean(stacked[mask], axis=0)
        else:
            # Fallback to median if no inliers (extreme threshold)
            robust = med
            inlier_count = 0

        self.output.send(robust.astype(np.float32))
        self.inlier_output.send(inlier_count)


class NumpyRollingBufferNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyRollingBufferNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.shape = []
        for arg in args:
            dimension = any_to_int(arg)
            self.shape.append(dimension)

        # Args may be empty or contain zero/negative dims; the pointer logic in
        # execute() assumes shape[0] >= 1 and all dims are positive. Fall back
        # to a small default rather than constructing a degenerate buffer.
        if len(self.shape) == 0 or any(d <= 0 for d in self.shape):
            print(f'{self.label}: invalid shape from args {args!r}, falling back to [16]')
            self.shape = [16]

        self.buffer_shape = self.shape.copy()
        self.buffer_shape[0] = self.buffer_shape[0] * 2
        self.buffer = np.zeros(self.buffer_shape)
        self.pointer = 0
        # Samples received since the last reset. The output window is only this
        # long, so unwritten slots are never emitted as spurious zero entries.
        self.count = 0

        self.input = self.add_input('input', triggers_execution=True)
        self.reset_input = self.add_input('reset', widget_type='button', callback=self.reset)

        self.output = self.add_output('out')

    def reset(self):
        self.buffer[:] = 0
        self.pointer = 0
        self.count = 0
        # Send the now-empty window so downstream displays clear immediately
        # rather than holding the old contents until the next sample arrives.
        self.output.send(self.buffer[0:0].copy())

    def execute(self):
        input = any_to_array(self.input())
        if len(input.shape) == len(self.shape):
            input = input[0]
        if len(input.shape) + 1 == len(self.shape):
            if list(input.shape) == self.shape[1:]:
                self.pointer -= 1
                if self.pointer < 0:
                    self.pointer = self.shape[0] - 1
                self.buffer[self.pointer] = input
                self.buffer[self.pointer + self.shape[0]] = input
                if self.count < self.shape[0]:
                    self.count += 1
                # A copy: a slice is a view of the ring, which the next sample
                # would rewrite under anything downstream that kept it.
                self.output.send(self.buffer[self.pointer:self.pointer + self.count].copy())


class NumpySubtensorNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpySubtensorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # Initialize storage
        self.slice_obj = (slice(None),)

        # Default string
        # joined with spaces: [0 3 7] arrives as '[0', '3', '7]' and must stay a list
        index_string = ' '.join(args) if args else ':'

        self.input = self.add_input('tensor in', triggers_execution=True)
        self.indices_input = self.add_input(
            'Indices',
            widget_type='text_input',
            widget_width=120,
            default_value=index_string,
            callback=self.dim_changed
        )
        self.output = self.add_output('output')
        self.width_option = self.add_option('indices field width', widget_type='drag_int', default_value=120, callback=self.resize_indices_field)

    def resize_indices_field(self):
        dpg.set_item_width(self.indices_input.widget.uuid, self.width_option())

    def auto_resize_indices_field(self):
        width = self.indices_input.widget.get_text_width(pad=16, minimum_width=40)
        if width is not None:
            dpg.configure_item(self.indices_input.widget.uuid, width=int(width))
            self.width_option.set(int(width))

    def custom_create(self, from_file=False):
        self.dim_changed()
        self.auto_resize_indices_field()

    def dim_changed(self):
        # 1. Update the Plan
        indices = self.indices_input()
        if isinstance(indices, np.ndarray) and indices.dtype.kind in 'iu':
            # an index array (np.argsort's output, say) picks those
            # positions along the first axis - no round trip through text
            self.slice_obj = (indices.astype(int),)
        else:
            self.slice_obj = self.parse_slice_string(any_to_string(indices))

        # 2. Auto-resize widget to fit text
        if self.created:
            self.auto_resize_indices_field()

        # 3. Attempt Execute
        # If no data exists, execute() will simply return, producing no output.
        self.execute()

    def parse_slice_string(self, dim_text):
        dim_text = dim_text.strip()

        # Strip outer [] only if they wrap the entire string and don't
        # contain inner brackets (i.e. they are just decorative, not a
        # fancy-index list for a single dimension).
        if dim_text.startswith('[') and dim_text.endswith(']'):
            inner = dim_text[1:-1]
            if '[' not in inner and ']' not in inner:
                # Could be either "[:, 3]" style wrapper or "[0 3 7]" fancy list.
                # If it contains a colon or comma-separated colons, treat as wrapper.
                # Otherwise treat as a single fancy-index list.
                if ':' in inner or '...' in inner:
                    dim_text = inner
                else:
                    # Check whether it looks like a space/semicolon-separated
                    # list of ints (fancy index) vs. comma-separated dims.
                    tokens = inner.replace(';', ' ').split()
                    all_int = all(self._is_int_token(t) for t in tokens)
                    if all_int and len(tokens) > 1:
                        # Single fancy-index list, e.g. [0 3 7]
                        return (np.array([int(t) for t in tokens], dtype=int),)
                    else:
                        dim_text = inner

        if not dim_text:
            return (slice(None),)

        parts = self._split_respecting_brackets(dim_text)
        slices = []

        for part in parts:
            part = part.strip()
            if part == '...':
                slices.append(Ellipsis)
            elif part.startswith('[') and part.endswith(']'):
                # Fancy index list: [0 3 7] or [0;3;7] or [0,3,7]
                indices = self._parse_index_list(part[1:-1].strip())
                slices.append(np.array(indices, dtype=int))
            elif ':' in part:
                sub_parts = part.split(':')
                slice_args = []
                for sp in sub_parts:
                    if sp.strip() == '':
                        slice_args.append(None)
                    else:
                        try:
                            slice_args.append(int(sp))
                        except ValueError:
                            slice_args.append(None)
                slices.append(slice(*slice_args))
            else:
                try:
                    slices.append(int(part))
                except ValueError:
                    slices.append(slice(None))

        return tuple(slices)

    # ------------------------------------------------------------------
    # Helpers for fancy-index parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _is_int_token(text):
        """Return True if *text* looks like an integer (with optional sign)."""
        try:
            int(text)
            return True
        except ValueError:
            return False

    @staticmethod
    def _split_respecting_brackets(text):
        """Split *text* on commas that are NOT inside square brackets."""
        parts = []
        depth = 0
        current = []
        for ch in text:
            if ch == '[':
                depth += 1
                current.append(ch)
            elif ch == ']':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        parts.append(''.join(current))
        return parts

    @staticmethod
    def _parse_index_list(inner_text):
        """Parse '0 3 7' or '0;3;7' or '0,3,7' into [0, 3, 7]."""
        normalized = inner_text.replace(';', ' ').replace(',', ' ')
        tokens = normalized.split()
        indices = []
        for t in tokens:
            try:
                indices.append(int(t))
            except ValueError:
                pass  # skip unparseable tokens
        return indices

    def execute(self):
        input_data = self.input()

        # STOP if no input
        if input_data is None:
            return

        input_array = any_to_array(input_data)
        if input_array is None:
            return

        # Check Dimensions
        required_dims = sum(1 for s in self.slice_obj if s is not Ellipsis)

        # STOP if dimensions don't match (e.g., slicing 3D on a 1D array)
        if input_array.ndim < required_dims:
            return

        try:
            sub_tensor = input_array[self.slice_obj]
            self.output.send(sub_tensor)
        except Exception as e:
            print(f'{self.label}: slice {self.slice_obj} failed on shape '
                  f'{input_array.shape}: {type(e).__name__}: {e}')
            traceback.print_exc()


class NumpyEditNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyEditNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.dim_list = None
        self.value = None
        index_string = ''
        self.edited_array = None
        for i in range(len(args)):
            index_string += args[i]
        if index_string == '':
            index_string = ':'

        self.input = self.add_input('array in', triggers_execution=True)
        self.input.name_archive.append('tensor in')
        self.indices_input = self.add_input('indices', widget_type='text_input', widget_width=70, default_value=index_string,
                                                  callback=self.edit_section_changed)
        self.indices_input.widget.wants_resize_handle = True
        # values start empty: a fresh node passes arrays through until a value is set
        self.edit_values_input = self.add_input('values', widget_type='text_input', widget_width=70, default_value='',
                                                  callback=self.edit_value_changed)
        self.output = self.add_output('output')
        self.indices_width_option = self.add_option('widget width', widget_type='drag_int', default_value=70,
                                                    callback=self.indices_width_changed)

    def indices_width_changed(self):
        w = self.indices_width_option()
        dpg.set_item_width(self.indices_input.widget.uuid, w)
        dpg.set_item_width(self.edit_values_input.widget.uuid, w)

    def custom_create(self, from_file):
        self.add_resize_handle(self.indices_input.widget, axis='x',
                               width_option=self.indices_width_option,
                               extra_targets=[self.edit_values_input.widget])
        w = self.indices_width_option()
        dpg.set_item_width(self.indices_input.widget.uuid, w)
        dpg.set_item_width(self.edit_values_input.widget.uuid, w)

    def edit_section_changed(self):
        self.parse_indices()
        self.edit_array_and_output()

    def parse_indices(self):
        dim_text = any_to_string(self.indices_input())

        dim_split = dim_text.split(',')
        dimmers = []

        for i in range(len(dim_split)):
            dimmer = dim_split[i]
            dimmer = dimmer.split(':')
            dim_nums = []
            if len(dimmer) == 1:
                dim_num = re.findall(r'[-+]?\d+', dimmer[0])
                if len(dim_num) > 0:
                    dim_num = string_to_int(dim_num[0])
                    dim_nums.append([dim_num])
                    if dim_num == -1:
                        dim_nums.append([1000000])  # -1:0 would be empty; -1 means the last element
                    else:
                        dim_nums.append([dim_num + 1])
            else:
                for j in range(len(dimmer)):
                    dim_num = re.findall(r'[-+]?\d+', dimmer[j])
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

    def edit_value_changed(self):
        self.value = self.edit_values_input()
        self.edit_array_and_output()

    def execute(self):
        input_array = any_to_array(self.input())
        if input_array is not None:
            if self.dim_list is None:
                self.parse_indices()
            if len(self.dim_list) == 0 or self.no_edit_value():
                self.output.send(input_array)
            else:
                self.edit_array_and_output()

    def no_edit_value(self):
        value = self.edit_values_input() if self.value is None else self.value
        return value is None or (isinstance(value, str) and value.strip() == '')

    def edit_array_and_output(self):
        if self.in_loading_process:
            return
        if self.dim_list is None:
            self.parse_indices()
        # each edit starts from the most recent input, so widget edits don't pile up
        input_value = self.input()
        if input_value is None:
            return
        input_array = any_to_array(input_value)
        if self.no_edit_value():
            self.output.send(input_array)
            return
        self.edited_array = input_array.copy()
        if self.value is None:
            self.value = self.edit_values_input()
        dim_list_now = []
        if len(self.dim_list) <= len(self.edited_array.shape):
            for i in range(len(self.dim_list)):
                dim_dim = self.dim_list[i]
                dim_list_now.append(dim_dim[0][0])
                if dim_dim[1][0] == 1000000:
                    dim_list_now.append(self.edited_array.shape[i])
                else:
                    dim_list_now.append(dim_dim[1][0])
            if len(dim_list_now) < len(self.edited_array.shape) * 2:
                base = len(dim_list_now) // 2
                diff = (len(self.edited_array.shape) * 2 - len(dim_list_now)) // 2
                for i in range(diff):
                    dim_list_now.append(0)
                    dim_list_now.append(self.edited_array.shape[base + i])

            value_array = None
            if isinstance(self.value, str):
                self.value = string_to_array(self.value)
            if isinstance(self.value, (int, float)):
                if len(dim_list_now) == 2:
                    self.edited_array[dim_list_now[0]:dim_list_now[1]] = self.value
                elif len(dim_list_now) == 4:
                    self.edited_array[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3]] = self.value
                elif len(dim_list_now) == 6:
                    self.edited_array[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3], dim_list_now[4]:dim_list_now[5]] = self.value
                elif len(dim_list_now) == 8:
                    self.edited_array[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3], dim_list_now[4]:dim_list_now[5], dim_list_now[6]:dim_list_now[7]] = self.value
            elif isinstance(self.value, list):
                value_array = any_to_array(self.value).copy()
            elif isinstance(self.value, np.ndarray):
                value_array = self.value
            if value_array is not None:
                if len(dim_list_now) == 2:
                    target_size = dim_list_now[1] - dim_list_now[0]
                    source_size = value_array.shape[0]
                    if source_size == 1 and len(value_array.shape) > 1:
                        source_size = value_array.shape[1]
                        value_array = value_array[0]
                    if target_size == source_size:
                        self.edited_array[dim_list_now[0]:dim_list_now[1]] = value_array
                    elif source_size == 1:
                        self.edited_array[dim_list_now[0]:dim_list_now[1]] = value_array[0]
                    elif target_size < source_size:
                        self.edited_array[dim_list_now[0]:dim_list_now[1]] = value_array[:target_size]
                    else:
                        self.edited_array[dim_list_now[0]:dim_list_now[0] + source_size] = value_array
                elif len(dim_list_now) == 4:
                    target_shape = [dim_list_now[1] - dim_list_now[0], dim_list_now[3] - dim_list_now[2]]
                    source_shape = list(value_array.shape)
                    if target_shape == source_shape:
                        self.edited_array[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3]] = value_array
                    elif len(source_shape) < len(target_shape):
                        test_shape = target_shape[-len(source_shape):]
                        if test_shape == source_shape:
                            self.edited_array[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3]] = value_array
                        elif len(value_array.shape) > 0 and value_array.shape[0] == 1:
                            self.edited_array[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3]] = value_array[0]
                        else:
                            self.report_bad_size(source_shape, target_shape)
                    else:
                        self.report_bad_size(source_shape, target_shape)
                elif len(dim_list_now) == 6:
                    target_shape = [dim_list_now[1] - dim_list_now[0], dim_list_now[3] - dim_list_now[2], dim_list_now[5] - dim_list_now[4]]
                    source_shape = list(value_array.shape)
                    if target_shape == source_shape:
                        self.edited_array[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3], dim_list_now[4]:dim_list_now[5]] = value_array
                    elif len(source_shape) < len(target_shape):
                        test_shape = target_shape[-len(source_shape):]
                        if test_shape == source_shape:
                            self.edited_array[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3], dim_list_now[4]:dim_list_now[5]] = value_array
                        elif len(value_array.shape) > 0 and value_array.shape[0] == 1:
                            self.edited_array[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3], dim_list_now[4]:dim_list_now[5]] = value_array[0]
                        else:
                            self.report_bad_size(source_shape, target_shape)
                    else:
                        self.report_bad_size(source_shape, target_shape)
                elif len(dim_list_now) == 8:
                    target_shape = [dim_list_now[1] - dim_list_now[0], dim_list_now[3] - dim_list_now[2], dim_list_now[5] - dim_list_now[4], dim_list_now[7] - dim_list_now[6]]
                    source_shape = list(value_array.shape)
                    if target_shape == source_shape:
                        self.edited_array[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3], dim_list_now[4]:dim_list_now[5], dim_list_now[6]:dim_list_now[7]] = value_array
                    elif len(source_shape) < len(target_shape):
                        test_shape = target_shape[-len(source_shape):]
                        if test_shape == source_shape:
                            self.edited_array[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3], dim_list_now[4]:dim_list_now[5], dim_list_now[6]:dim_list_now[7]] = value_array
                        elif len(value_array.shape) > 0 and value_array.shape[-1] == 1:
                            self.edited_array[dim_list_now[0]:dim_list_now[1], dim_list_now[2]:dim_list_now[3], dim_list_now[4]:dim_list_now[5], dim_list_now[6]:dim_list_now[7]] = value_array[0]
                        else:
                            self.report_bad_size(source_shape, target_shape)
                    else:
                        self.report_bad_size(source_shape, target_shape)
            self.output.send(self.edited_array)

    def report_bad_size(self, source_shape, target_shape):
        # the array still goes out, unedited
        print(f'{self.label}: values of shape {tuple(source_shape)} do not fit the '
              f'selected part {tuple(target_shape)} - array sent unedited')


class NumpySequenceNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpySequenceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.sequence = []
        self.num_frames = 0
        self.data_input = self.add_input('data to record', triggers_execution=True)
        self.frame_input = self.add_input('current frame', widget_type='drag_int', default_value=-1, callback=self.output_frame)
        # last inlet, so links saved by index before it existed still land right
        self.clear_input = self.add_input('clear', widget_type='button', callback=self.clear_sequence)
        self.save_button = self.add_property('save', widget_type='button', callback=self.save_sequence)

        self.load_button = self.add_property('load', widget_type='button', callback=self.load_sequence)
        self.load_path = self.add_option('path', widget_type='text_input', default_value='',
                                         callback=self.load_from_load_path)
        self.data_output = self.add_output('data out')

    def execute(self):
        if self.active_input == self.data_input:
            self.sequence.append(any_to_array(self.data_input()).copy())
            self.num_frames = len(self.sequence)
            self.frame_input.set(self.num_frames - 1)

    def output_frame(self):
            frame_num = self.frame_input()
            if -1 < frame_num < self.num_frames:
                self.data_output.send(self.sequence[frame_num])

    def clear_sequence(self, input=None):
        # empty the recording so the next frames start a fresh take at frame 0
        self.sequence = []
        self.num_frames = 0
        self.frame_input.set(-1)

    def save_sequence(self):
        SaveDialog(self, callback=self.save_file_callback, extensions=['.npy'])
        # Node.app.active_widget = 1
        # with dpg.file_dialog(modal=True, directory_selector=False, show=True, height=400, width=800,
        #                      user_data=self, callback=self.save_file_callback, tag="file_dialog_id"):
        #     dpg.add_file_extension(".npy")

    def save_file_callback(self, save_path):
        if save_path != '':
            self.save_array(save_path)
        else:
            print('no file chosen')

    def save_array(self, path):
        numpy_array = np.array(self.sequence)
        np.save(path, numpy_array)

    def load_sequence(self):
        LoadDialog(self, callback=self.load_npz_callback, extensions=['.npy'])
        # Node.app.active_widget = 1
        # with dpg.file_dialog(modal=True, directory_selector=False, show=True, height=400, width=800,
        #                          user_data=self, callback=self.load_npz_callback, tag="file_dialog_id"):
        #     dpg.add_file_extension(".npy")

    def load_from_load_path(self):
        path = self.load_path()
        if path != '':
            try:
                self.load_array(path)
            except Exception as e:
                print(f'np.sequence: failed to load {path}: {type(e).__name__}: {e}')
                traceback.print_exc()

    def load_npz_callback(self, load_path):
        if load_path != '':
            self.load_path.set(load_path)
            self.load_array(load_path)
        else:
            print('no file chosen')

    def load_array(self, path):
        numpy_array = np.load(path)
        self.sequence = []
        for frame_num in range(numpy_array.shape[0]):
            self.sequence.append(numpy_array[frame_num].copy())
        self.num_frames = len(self.sequence)
        self.frame_input.set(-1)


class NumpyLoadNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyLoadNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        default_path = ''
        if len(args) > 0:
            default_path = args[0]

        self.dict = None
        self.load_button = self.add_input('load', widget_type='button', callback=self.open_load_dialog)
        self.add_input('send', widget_type='button', callback=self.send_dict)
        self.path_option = self.add_option('path', widget_type='text_input', default_value=default_path,
                                           callback=self.path_changed)
        self.data_output = self.add_output('dict out')

    def path_changed(self):
        # a typed path, or one restored when the patch opens (custom_create runs
        # before the saved options are restored, so this is what reads it then)
        path = self.path_option()
        if path is not None and path != '':
            self.load_file(path)

    def send_dict(self):
        if self.dict is not None:
            if self.app.verbose:
                print(f'{self.label}: sending dict with keys {list(self.dict.keys())}')
            self.data_output.send(self.dict)

    def custom_create(self, from_file=False):
        # load the path given as an argument whether typed in or opened from a patch
        path = self.path_option()
        if path is not None and path != '':
            self.load_file(path)

    def open_load_dialog(self):
        LoadDialog(self, callback=self.load_dialog_callback, extensions=['.npz', '.npy'])

    def load_dialog_callback(self, load_path):
        if load_path != '':
            self.path_option.set(load_path)
            self.load_file(load_path)
        else:
            print('no file chosen')

    def load_file(self, path):
        try:
            npz_data = np.load(path, allow_pickle=True)
            if isinstance(npz_data, np.ndarray):
                # a .npy file holds a single array: send it as one, under 'data'
                npz_data = {'data': npz_data}
                self.dict = npz_data
                if self.app.verbose:
                    print(f'{self.label}: loaded {path}, one array of shape {npz_data["data"].shape}')
                self.data_output.send(self.dict)
                return
            temp_dict = {key: npz_data[key] for key in npz_data.files}
            self.dict = deepcopy(temp_dict)
            npz_data.close()
            if self.app.verbose:
                print(f'{self.label}: loaded {path} with keys {list(self.dict.keys())}')
            self.data_output.send(self.dict)
        except Exception as e:
            print(f'np.load error loading {path}: {type(e).__name__}: {e}')
            traceback.print_exc()


class NumpyAnyNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyAnyNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('in', triggers_execution=True)
        self.result_out = self.add_output('result')

    def execute(self):
        a = self.input()
        if a is None:
            return
        a_arr = np.asarray(a)
        if a_arr.dtype != bool:
            a_arr = a_arr.astype(bool)
        result = bool(np.any(a_arr))
        self.result_out.send(result)


class NumpyAllNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyAllNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('in', triggers_execution=True)
        self.result_out = self.add_output('result')

    def execute(self):
        a = self.input()
        if a is None:
            return
        a_arr = np.asarray(a)
        if a_arr.dtype != bool:
            a_arr = a_arr.astype(bool)
        result = bool(np.all(a_arr))
        self.result_out.send(result)


class NumpyWhereNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyWhereNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.if_true_value = None
        self.if_false_value = None
        self.input = self.add_input('condition', triggers_execution=True)
        if label == 'np.where':
            self.if_true_input = self.add_input('if true')
            self.if_false_input = self.add_input('if false')
            self.output = self.add_output('selected')
        else:
            self.output = self.add_output('indices')

    def execute(self):
        condition = any_to_array(self.input())
        if self.label == 'np.argwhere':
            self.output.send(np.argwhere(condition))
            return
        if self.if_true_input.fresh_input:
            self.if_true_value = any_to_array(self.if_true_input())
        if self.if_false_input.fresh_input:
            self.if_false_value = any_to_array(self.if_false_input())
        if self.if_true_value is None or self.if_false_value is None:
            return
        try:
            self.output.send(np.where(condition, self.if_true_value, self.if_false_value))
        except ValueError:
            if self.app.verbose:
                print(self.label, 'shapes do not broadcast:', condition.shape,
                      self.if_true_value.shape, self.if_false_value.shape)


class NumpySortNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpySortNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        axis = -1
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                axis = int(v)
        self.op = np.argsort if label == 'np.argsort' else np.sort
        self.input = self.add_input('in', triggers_execution=True)
        self.axis = self.add_input('axis', widget_type='input_int', default_value=axis)
        self.descending = self.add_option('descending', widget_type='checkbox', default_value=False)
        output_name = 'indices' if label == 'np.argsort' else 'sorted'
        self.output = self.add_output(output_name)

    def execute(self):
        data = any_to_array(self.input())
        axis = self.axis()
        if not -len(data.shape) <= axis < len(data.shape):
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', data.shape)
            return
        result = self.op(data, axis=axis)
        if self.descending():
            result = np.flip(result, axis=axis)
        self.output.send(result)


class NumpyDiffNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyDiffNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        order = 1
        axis = -1
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                order = int(v)
        if len(args) > 1:
            v, t = decode_arg(args, 1)
            if t in [int, float]:
                axis = int(v)
        self.input = self.add_input('in', triggers_execution=True)
        self.order = self.add_input('order', widget_type='input_int', default_value=order)
        self.axis = self.add_input('axis', widget_type='input_int', default_value=axis)
        self.output = self.add_output('differences')

    def execute(self):
        data = any_to_array(self.input())
        axis = self.axis()
        order = self.order()
        if order < 1:
            order = 1
        if not -len(data.shape) <= axis < len(data.shape):
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', data.shape)
            return
        if data.shape[axis] <= order:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'has size', data.shape[axis],
                      '- too short for order', order)
            return
        self.output.send(np.diff(data, n=order, axis=axis))


class NumpyCumulativeNode(Node):
    operations = {'np.cumsum': np.cumsum, 'np.cumprod': np.cumprod}

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyCumulativeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        axis = -1
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                axis = int(v)
        self.op = np.cumsum
        if label in self.operations:
            self.op = self.operations[label]
        self.input = self.add_input('in', triggers_execution=True)
        self.axis = self.add_input('axis', widget_type='input_int', default_value=axis)
        output_name = label.split('.')[-1]
        self.output = self.add_output(output_name)

    def execute(self):
        data = any_to_array(self.input())
        axis = self.axis()
        if not -len(data.shape) <= axis < len(data.shape):
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', data.shape)
            return
        self.output.send(self.op(data, axis=axis))


class NumpySplitNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpySplitNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.splits = 2
        axis = 0
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                self.splits = max(int(v), 2)
        if len(args) > 1:
            v, t = decode_arg(args, 1)
            if t in [int, float]:
                axis = int(v)
        self.input = self.add_input('in', triggers_execution=True)
        self.axis = self.add_input('axis', widget_type='input_int', default_value=axis)
        self.array_outputs = []
        for i in range(self.splits):
            self.array_outputs.append(self.add_output('array ' + str(i)))

    def execute(self):
        data = any_to_array(self.input())
        axis = self.axis()
        if not -len(data.shape) <= axis < len(data.shape):
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', data.shape)
            return
        if data.shape[axis] < self.splits:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'has size', data.shape[axis],
                      '- fewer elements than', self.splits, 'splits')
            return
        pieces = np.array_split(data, self.splits, axis=axis)
        for idx, piece in enumerate(pieces):
            self.array_outputs[idx].send(piece)


class RotatePositionNode(Node):
    """Rotates an xyz position around X, Y, or Z axis by a given angle (degrees)."""

    @staticmethod
    def factory(name, data, args=None):
        return RotatePositionNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('position', triggers_execution=True)
        self.angle_input = self.add_input('angle', widget_type='drag_float', default_value=0.0)
        self.axis_prop = self.add_property('axis', widget_type='combo', default_value='Y', callback=self.execute)
        self.axis_prop.widget.combo_items = ['X', 'Y', 'Z']
        self._warned_shape = False

        self.output = self.add_output('rotated')

    def execute(self):
        raw = self.input()
        if raw is None:
            return
        pos = any_to_array(raw).astype(np.float64)
        angle_deg = float(self.angle_input())
        angle = np.radians(angle_deg)
        c, s = np.cos(angle), np.sin(angle)

        axis = self.axis_prop()
        if axis == 'X':
            mat = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 'Y':
            mat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:  # Z
            mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        if pos.ndim == 1 and pos.size == 3:
            self.output.send((mat @ pos).astype(np.float32))
        elif pos.ndim == 2 and pos.shape[-1] == 3:
            self.output.send((pos @ mat.T).astype(np.float32))
        else:
            if not self._warned_shape:
                print(f'{self.label}: expected xyz positions (3 or (N, 3)), got shape {pos.shape}; passing through unrotated')
                self._warned_shape = True
            self.output.send(pos.astype(np.float32))

class NumpyLogicalNode(Node):
    operations = {'np.logical_and': np.logical_and, 'np.logical_or': np.logical_or,
                  'np.logical_xor': np.logical_xor, 'np.logical_not': np.logical_not}
    output_types = {'bool': np.bool_, 'int': np.int64, 'float': np.float32}

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyLogicalNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = np.logical_and
        if label in self.operations:
            self.op = self.operations[label]
        self.unary = label == 'np.logical_not'
        self.input_2_value = None
        self.input = self.add_input('in', triggers_execution=True)
        if not self.unary:
            self.input_2 = self.add_input('in 2')
        self.output = self.add_output('result')
        self.output_type_option = self.add_option('output_type', widget_type='combo', default_value='bool')
        self.output_type_option.widget.combo_items = list(self.output_types.keys())

    def execute(self):
        a = any_to_array(self.input())
        if self.unary:
            result = self.op(a)
        else:
            if self.input_2.fresh_input:
                self.input_2_value = any_to_array(self.input_2())
            if self.input_2_value is None:
                return
            try:
                result = self.op(a, self.input_2_value)
            except ValueError:
                if self.app.verbose:
                    print(self.label, 'shapes do not broadcast:', a.shape, self.input_2_value.shape)
                return
        out_type = self.output_types.get(self.output_type_option(), np.bool_)
        self.output.send(result.astype(out_type))


class NumpyFiniteTestNode(Node):
    operations = {'np.isnan': np.isnan, 'np.isfinite': np.isfinite}

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyFiniteTestNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = np.isnan
        if label in self.operations:
            self.op = self.operations[label]
        self.input = self.add_input('in', triggers_execution=True)
        self.output = self.add_output('mask')
        self.count_output = self.add_output('count')

    def execute(self):
        data = any_to_array(self.input())
        if not np.issubdtype(data.dtype, np.number) and data.dtype != np.bool_:
            if self.app.verbose:
                print(self.label, 'needs numbers, got dtype', data.dtype)
            return
        mask = self.op(data)
        self.count_output.send(int(np.count_nonzero(mask)))
        self.output.send(mask)


class NumpyNanToNumNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyNanToNumNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        nan_value = 0.0
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                nan_value = float(v)
        self.input = self.add_input('in', triggers_execution=True)
        self.nan_input = self.add_input('nan', widget_type='drag_float', default_value=nan_value)
        self.posinf_input = self.add_input('posinf', widget_type='drag_float', default_value=0.0)
        self.neginf_input = self.add_input('neginf', widget_type='drag_float', default_value=0.0)
        self.output = self.add_output('out')

    def execute(self):
        data = any_to_array(self.input())
        if not np.issubdtype(data.dtype, np.inexact):
            # whole numbers cannot hold nan or inf: nothing to replace
            self.output.send(data)
            return
        self.output.send(np.nan_to_num(data, nan=self.nan_input(), posinf=self.posinf_input(),
                                       neginf=self.neginf_input()))


class NumpyPercentileNode(NumpyNodeWithAxisNode):
    methods = ['linear', 'lower', 'higher', 'nearest', 'midpoint']

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyPercentileNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.percentile = 50.0
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                self.percentile = float(v)
        if len(args) > 1:
            v, t = decode_arg(args, 1)
            if t == int:
                self.axis = v
        self.input = self.add_input('in', triggers_execution=True)
        self.percentile_input = self.add_input('percentile', widget_type='drag_float',
                                               default_value=self.percentile, callback=self.percentile_changed)
        # None reduces the whole array to one value
        self.add_dim_option()
        self.output = self.add_output('result')
        self.method_option = self.add_option('method', widget_type='combo', default_value='linear')
        self.method_option.widget.combo_items = self.methods
        self.ignore_nan_option = self.add_option('ignore nan', widget_type='checkbox', default_value=False)

    def percentile_changed(self):
        # a list asks for several percentiles at once
        value = any_to_numerical(self.percentile_input(), validate=True)
        if value is not None:
            self.percentile = value

    def execute(self):
        if self.percentile_input.fresh_input:
            self.percentile_changed()
        data = any_to_array(self.input())
        if data.size == 0:
            return
        if not self.adjust_dim_option(data):
            if self.app.verbose:
                print(self.label, 'axis =', self.axis, 'out of range for shape', data.shape)
            return
        q = np.clip(np.asarray(self.percentile, dtype=float), 0.0, 100.0)
        op = np.nanpercentile if self.ignore_nan_option() else np.percentile
        method = self.method_option()
        if method not in self.methods:
            method = 'linear'
        with warnings.catch_warnings():
            # an all-nan slice gives nan, which is the answer; no need to say so
            warnings.simplefilter('ignore', RuntimeWarning)
            self.output.send(op(data, q, axis=self.call_axis, method=method))


class NumpyHistogramNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyHistogramNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        bins = 10
        range_min = 0.0
        range_max = 1.0
        auto_range = True
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                bins = int(v)
        if len(args) > 2:
            v, t = decode_arg(args, 1)
            w, u = decode_arg(args, 2)
            if t in [int, float] and u in [int, float]:
                range_min = float(v)
                range_max = float(w)
                auto_range = False
        self.input = self.add_input('in', triggers_execution=True)
        self.bins_input = self.add_input('bins', widget_type='input_int', default_value=bins)
        self.min_input = self.add_input('min', widget_type='drag_float', default_value=range_min)
        self.max_input = self.add_input('max', widget_type='drag_float', default_value=range_max)
        self.counts_output = self.add_output('counts')
        self.edges_output = self.add_output('bin edges')
        self.auto_range_option = self.add_option('auto range', widget_type='checkbox', default_value=auto_range)
        self.density_option = self.add_option('density', widget_type='checkbox', default_value=False)

    def execute(self):
        data = any_to_array(self.input())
        if not np.issubdtype(data.dtype, np.number) and data.dtype != np.bool_:
            if self.app.verbose:
                print(self.label, 'needs numbers, got dtype', data.dtype)
            return
        data = data.ravel().astype(float)
        # nan and inf belong in no bin, and with them np.histogram cannot find its range
        data = data[np.isfinite(data)]
        bins = max(self.bins_input(), 1)
        if self.auto_range_option():
            if data.size == 0:
                return
            value_range = (data.min(), data.max())
        else:
            value_range = (self.min_input(), self.max_input())
            if value_range[1] <= value_range[0]:
                if self.app.verbose:
                    print(self.label, 'max must be above min, got', value_range)
                return
        counts, edges = np.histogram(data, bins=bins, range=value_range,
                                     density=self.density_option() and data.size > 0)
        self.edges_output.send(edges)
        self.counts_output.send(counts)


class NumpyInterpNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyInterpNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.xp_value = None
        self.fp_value = None
        self.input = self.add_input('x', triggers_execution=True)
        self.xp_input = self.add_input('xp')
        self.fp_input = self.add_input('fp')
        self.output = self.add_output('values')

    def execute(self):
        if self.xp_input.fresh_input:
            self.xp_value = any_to_array(self.xp_input()).astype(float).ravel()
        if self.fp_input.fresh_input:
            self.fp_value = any_to_array(self.fp_input()).astype(float)
        if self.fp_value is None or len(self.fp_value) == 0:
            return
        fp = self.fp_value
        # without xp the table is indexed by position: 0, 1, 2 ...
        xp = self.xp_value if self.xp_value is not None else np.arange(fp.shape[0], dtype=float)
        if xp.shape[0] != fp.shape[0]:
            if self.app.verbose:
                print(self.label, 'xp has', xp.shape[0], 'positions but fp has', fp.shape[0], 'values')
            return
        if np.any(np.diff(xp) < 0):
            # np.interp silently gives nonsense for unsorted positions
            order = np.argsort(xp, kind='stable')
            xp = xp[order]
            fp = fp[order]
        x = any_to_array(self.input()).astype(float)
        if fp.ndim == 1:
            self.output.send(np.interp(x, xp, fp))
            return
        # several columns of values: interpolate each, keeping the column shape
        columns = fp.reshape(fp.shape[0], -1)
        result = np.stack([np.interp(x, xp, columns[:, c]) for c in range(columns.shape[1])], axis=-1)
        self.output.send(result.reshape(x.shape + fp.shape[1:]))


class NumpyConvolveNode(Node):
    operations = {'np.convolve': np.convolve, 'np.correlate': np.correlate}
    modes = ['same', 'full', 'valid']

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyConvolveNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = np.convolve
        if label in self.operations:
            self.op = self.operations[label]
        mode = 'same'
        kernel = []
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [int, float]:
                kernel.append(float(v))
            elif t == str and v in self.modes:
                mode = v
        self.kernel = np.array(kernel) if len(kernel) > 0 else None
        self.input = self.add_input('in', triggers_execution=True)
        self.kernel_input = self.add_input('kernel' if label == 'np.convolve' else 'pattern')
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=-1)
        self.output = self.add_output('convolved' if label == 'np.convolve' else 'correlated')
        self.mode_option = self.add_option('mode', widget_type='combo', default_value=mode)
        self.mode_option.widget.combo_items = self.modes

    def execute(self):
        if self.kernel_input.fresh_input:
            self.kernel = any_to_array(self.kernel_input()).astype(float).ravel()
        if self.kernel is None or self.kernel.size == 0:
            return
        data = any_to_array(self.input()).astype(float)
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', data.shape)
            return
        mode = self.mode_option()
        if mode not in self.modes:
            mode = 'same'
        if mode == 'valid' and data.shape[axis] < self.kernel.size:
            if self.app.verbose:
                print(self.label, 'valid mode needs at least', self.kernel.size, 'values along the axis')
            return
        kernel = self.kernel
        if data.ndim == 1:
            self.output.send(self.op(data, kernel, mode=mode))
            return
        self.output.send(np.apply_along_axis(lambda row: self.op(row, kernel, mode=mode), axis, data))


class NumpyLinalgLstsqNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyLinalgLstsqNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.b_value = None
        self.input = self.add_input('A matrix', triggers_execution=True)
        self.b_input = self.add_input('b')
        self.output = self.add_output('x')
        self.residuals_output = self.add_output('residuals')
        self.rank_output = self.add_output('rank')

    def execute(self):
        if self.b_input.fresh_input:
            self.b_value = any_to_array(self.b_input()).astype(float)
        if self.b_value is None:
            return
        a = any_to_array(self.input()).astype(float)
        if a.ndim != 2:
            if self.app.verbose:
                print(self.label, 'A must be a 2-D matrix, got shape', a.shape)
            return
        if self.b_value.shape[0] != a.shape[0]:
            if self.app.verbose:
                print(self.label, 'b needs', a.shape[0], 'rows to match A, got shape', self.b_value.shape)
            return
        try:
            x, residuals, rank, singular_values = np.linalg.lstsq(a, self.b_value, rcond=None)
        except np.linalg.LinAlgError as e:
            if self.app.verbose:
                print(self.label, e)
            return
        self.rank_output.send(int(rank))
        self.residuals_output.send(residuals)
        self.output.send(x)


class NumpyEinsumNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyEinsumNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        # 'ij, jk -> ik' arrives as several args; spaces mean nothing to einsum
        subscripts = ''.join(str(a) for a in args) if len(args) > 0 else 'ij,jk->ik'
        self.operand_count = subscripts.split('->')[0].count(',') + 1
        self.held = [None] * self.operand_count
        self.operand_inputs = [self.add_input('in', triggers_execution=True)]
        for i in range(1, self.operand_count):
            self.operand_inputs.append(self.add_input('in ' + str(i + 1)))
        self.output = self.add_output('result')
        self.subscripts_option = self.add_option('subscripts', widget_type='text_input', default_value=subscripts)

    def execute(self):
        for i in range(1, self.operand_count):
            if self.operand_inputs[i].fresh_input:
                self.held[i] = any_to_array(self.operand_inputs[i]())
        self.held[0] = any_to_array(self.operand_inputs[0]())
        if any(h is None for h in self.held):
            return
        subscripts = self.subscripts_option().replace(' ', '')
        if subscripts.split('->')[0].count(',') + 1 != self.operand_count:
            if self.app.verbose:
                print(self.label, subscripts, 'names a different number of operands than the',
                      self.operand_count, 'inlets this node was made with')
            return
        try:
            self.output.send(np.einsum(subscripts, *self.held))
        except ValueError as e:
            if self.app.verbose:
                print(self.label, e)


def parse_ints(value):
    """Whole numbers from a text box, a list or a single number: '2 3', [2, 3], 2."""
    if isinstance(value, str):
        return [int(v) for v in re.findall(r'[-+]?\d+', value)]
    return [any_to_int(v) for v in np.ravel(np.asarray(value, dtype=object))]


class NumpyUniqueNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyUniqueNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('in', triggers_execution=True)
        self.values_output = self.add_output('values')
        self.counts_output = self.add_output('counts')
        self.inverse_output = self.add_output('inverse')

    def execute(self):
        data = any_to_array(self.input())
        values, inverse, counts = np.unique(data, return_inverse=True, return_counts=True)
        self.inverse_output.send(inverse.reshape(data.shape))
        self.counts_output.send(counts)
        self.values_output.send(values)


class NumpyCovarianceNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyCovarianceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = np.corrcoef if label == 'np.corrcoef' else np.cov
        variables = 'columns'
        if len(args) > 0 and args[0] in ['rows', 'columns']:
            variables = args[0]
        self.input = self.add_input('in', triggers_execution=True)
        self.output = self.add_output('correlation' if label == 'np.corrcoef' else 'covariance')
        self.variables_option = self.add_option('variables', widget_type='combo', default_value=variables)
        self.variables_option.widget.combo_items = ['columns', 'rows']

    def execute(self):
        data = any_to_array(self.input()).astype(float)
        if data.ndim != 2:
            if self.app.verbose:
                print(self.label, 'needs a 2-D array of observations, got shape', data.shape)
            return
        rowvar = self.variables_option() == 'rows'
        if data.shape[1 if rowvar else 0] < 2:
            if self.app.verbose:
                print(self.label, 'needs at least 2 observations of each variable')
            return
        with np.errstate(divide='ignore', invalid='ignore'):
            # a variable that never changes has no correlation: nan, not a warning
            self.output.send(self.op(data, rowvar=rowvar))


class NumpyGradientNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyGradientNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        spacing = 1.0
        axis = -1
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                spacing = float(v)
        if len(args) > 1:
            v, t = decode_arg(args, 1)
            if t in [int, float]:
                axis = int(v)
        self.input = self.add_input('in', triggers_execution=True)
        self.spacing_input = self.add_input('spacing', widget_type='drag_float', default_value=spacing)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=axis)
        self.output = self.add_output('gradient')
        self.edge_order_option = self.add_option('edge order', widget_type='combo', default_value='1')
        self.edge_order_option.widget.combo_items = ['1', '2']

    def execute(self):
        data = any_to_array(self.input()).astype(float)
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', data.shape)
            return
        edge_order = 2 if self.edge_order_option() == '2' else 1
        if data.shape[axis] < edge_order + 1:
            if self.app.verbose:
                print(self.label, 'needs at least', edge_order + 1, 'values along the axis')
            return
        spacing = self.spacing_input()
        if spacing == 0:
            return
        self.output.send(np.gradient(data, spacing, axis=axis, edge_order=edge_order))


class NumpyRfftNode(Node):
    norms = ['backward', 'ortho', 'forward']

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyRfftNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        rate = 1.0
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                rate = float(v)
        self.input = self.add_input('in', triggers_execution=True)
        self.rate_input = self.add_input('sample rate', widget_type='drag_float', default_value=rate)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=-1)
        self.spectrum_output = self.add_output('spectrum')
        self.magnitude_output = self.add_output('magnitude')
        self.phase_output = self.add_output('phase')
        self.frequencies_output = self.add_output('frequencies')
        self.norm_option = self.add_option('norm', widget_type='combo', default_value='backward')
        self.norm_option.widget.combo_items = self.norms

    def execute(self):
        data = any_to_array(self.input())
        if np.iscomplexobj(data):
            if self.app.verbose:
                print(self.label, 'takes real values; use the real part or np.fft.fft')
            return
        data = data.astype(float)
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim or data.shape[axis] < 2:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range or too short for shape', data.shape)
            return
        norm = self.norm_option()
        if norm not in self.norms:
            norm = 'backward'
        spectrum = np.fft.rfft(data, axis=axis, norm=norm)
        rate = self.rate_input()
        if rate > 0:
            self.frequencies_output.send(np.fft.rfftfreq(data.shape[axis], d=1.0 / rate))
        self.phase_output.send(np.angle(spectrum))
        self.magnitude_output.send(np.abs(spectrum))
        self.spectrum_output.send(spectrum)


class NumpyIrfftNode(Node):
    norms = ['backward', 'ortho', 'forward']

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyIrfftNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        length = 0
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                length = int(v)
        self.phase_value = None
        self.input = self.add_input('spectrum', triggers_execution=True)
        self.phase_input = self.add_input('phase')
        self.length_input = self.add_input('length', widget_type='input_int', default_value=length)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=-1)
        self.output = self.add_output('signal')
        self.norm_option = self.add_option('norm', widget_type='combo', default_value='backward')
        self.norm_option.widget.combo_items = self.norms

    def execute(self):
        if self.phase_input.fresh_input:
            self.phase_value = any_to_array(self.phase_input()).astype(float)
        spectrum = any_to_array(self.input())
        if not np.iscomplexobj(spectrum):
            # magnitudes: put the held phases back with them, or none at all
            magnitude = spectrum.astype(float)
            if self.phase_value is not None and self.phase_value.shape == magnitude.shape:
                spectrum = magnitude * np.exp(1j * self.phase_value)
            else:
                spectrum = magnitude.astype(complex)
        axis = self.axis_input()
        if not -spectrum.ndim <= axis < spectrum.ndim:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', spectrum.shape)
            return
        length = self.length_input()
        norm = self.norm_option()
        if norm not in self.norms:
            norm = 'backward'
        self.output.send(np.fft.irfft(spectrum, n=length if length > 0 else None, axis=axis, norm=norm))


class NumpyTakeNode(Node):
    modes = ['clip', 'wrap']

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyTakeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        indices = []
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [int, float]:
                indices.append(int(v))
        self.indices = np.array(indices, dtype=int) if len(indices) > 0 else None
        self.input = self.add_input('in', triggers_execution=True)
        self.indices_input = self.add_input('indices')
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=0)
        self.output = self.add_output('taken')
        self.mode_option = self.add_option('mode', widget_type='combo', default_value='clip')
        self.mode_option.widget.combo_items = self.modes

    def execute(self):
        if self.indices_input.fresh_input:
            self.indices = np.round(any_to_array(self.indices_input()).astype(float)).astype(int)
        if self.indices is None:
            return
        data = any_to_array(self.input())
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim or data.shape[axis] == 0:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range or empty for shape', data.shape)
            return
        mode = self.mode_option()
        if mode not in self.modes:
            mode = 'clip'
        indices = self.indices
        if mode == 'clip':
            # numpy's clip mode turns -1 into 0; here it counts from the end, as an index does
            n = data.shape[axis]
            indices = np.clip(np.where(indices < 0, indices + n, indices), 0, n - 1)
        self.output.send(np.take(data, indices, axis=axis, mode=mode))


class NumpyPadNode(Node):
    modes = ['constant', 'edge', 'reflect', 'symmetric', 'wrap']

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyPadNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        before = 1
        after = 1
        mode = 'constant'
        numbers = []
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [int, float]:
                numbers.append(int(v))
            elif t == str and v in self.modes:
                mode = v
        if len(numbers) > 0:
            before = after = numbers[0]
        if len(numbers) > 1:
            after = numbers[1]
        self.input = self.add_input('in', triggers_execution=True)
        self.before_input = self.add_input('before', widget_type='input_int', default_value=before)
        self.after_input = self.add_input('after', widget_type='input_int', default_value=after)
        self.value_input = self.add_input('value', widget_type='drag_float', default_value=0.0)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=-1)
        self.output = self.add_output('padded')
        self.mode_option = self.add_option('mode', widget_type='combo', default_value=mode)
        self.mode_option.widget.combo_items = self.modes

    def execute(self):
        data = any_to_array(self.input())
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', data.shape)
            return
        before = max(self.before_input(), 0)
        after = max(self.after_input(), 0)
        widths = [(0, 0)] * data.ndim
        widths[axis] = (before, after)
        mode = self.mode_option()
        if mode not in self.modes:
            mode = 'constant'
        if mode != 'constant' and data.shape[axis] == 0:
            return
        if mode == 'constant':
            self.output.send(np.pad(data, widths, mode='constant', constant_values=self.value_input()))
        else:
            self.output.send(np.pad(data, widths, mode=mode))


class NumpyTileNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyTileNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.reps = [2]
        if len(args) > 0:
            self.reps = [max(any_to_int(a), 0) for a in args]
        self.input = self.add_input('in', triggers_execution=True)
        self.reps_input = self.add_input('reps', widget_type='text_input', widget_width=120,
                                         default_value=' '.join(str(r) for r in self.reps),
                                         callback=self.reps_changed)
        self.output = self.add_output('tiled')

    def reps_changed(self):
        reps = parse_ints(self.reps_input())
        if len(reps) > 0:
            self.reps = [max(r, 0) for r in reps]

    def execute(self):
        if self.reps_input.fresh_input:
            self.reps_changed()
        self.output.send(np.tile(any_to_array(self.input()), self.reps))


class NumpyDigitizeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyDigitizeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        edges = []
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [int, float]:
                edges.append(float(v))
        self.edges = np.array(edges) if len(edges) > 0 else None
        self.input = self.add_input('in', triggers_execution=True)
        self.edges_input = self.add_input('edges')
        self.output = self.add_output('bin indices')
        self.right_option = self.add_option('right', widget_type='checkbox', default_value=False)

    def execute(self):
        if self.edges_input.fresh_input:
            self.edges = any_to_array(self.edges_input()).astype(float).ravel()
        if self.edges is None or self.edges.size == 0:
            return
        edges = self.edges
        diffs = np.diff(edges)
        if np.any(diffs < 0) and np.any(diffs > 0):
            # np.digitize raises for edges that are neither rising nor falling
            edges = np.sort(edges)
        data = any_to_array(self.input()).astype(float)
        self.output.send(np.digitize(data, edges, right=self.right_option()))


class NumpyArangeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyArangeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        numbers = []
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [int, float]:
                numbers.append(v)
        # like np.arange: one number is the stop, two are start and stop
        start, stop, step = 0, 10, 1
        if len(numbers) == 1:
            stop = numbers[0]
        elif len(numbers) >= 2:
            start, stop = numbers[0], numbers[1]
            if len(numbers) > 2:
                step = numbers[2]
        widget = 'drag_float' if any(isinstance(n, float) for n in (start, stop, step)) else 'drag_int'
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.start_input = self.add_input('start', widget_type=widget, default_value=start)
        self.stop_input = self.add_input('stop', widget_type=widget, default_value=stop)
        self.step_input = self.add_input('step', widget_type=widget, default_value=step)
        self.output = self.add_output('arange out')

    def execute(self):
        start, stop, step = self.start_input(), self.stop_input(), self.step_input()
        if step == 0:
            return
        if (stop - start) / step > 10000000:
            if self.app.verbose:
                print(self.label, 'would make more than 10 million values')
            return
        self.output.send(np.arange(start, stop, step))


class NumpyEyeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyEyeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        n, m, k = 3, 0, 0
        if len(args) > 0:
            n = any_to_int(args[0])
        if len(args) > 1:
            m = any_to_int(args[1])
        if len(args) > 2:
            k = any_to_int(args[2])
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.rows_input = self.add_input('rows', widget_type='input_int', default_value=n)
        self.columns_input = self.add_input('columns', widget_type='input_int', default_value=m)
        self.k_input = self.add_input('diagonal', widget_type='input_int', default_value=k)
        self.output = self.add_output('eye out')

    def execute(self):
        n = self.rows_input()
        if n < 1:
            return
        m = self.columns_input()
        # 0 columns means square
        self.output.send(np.eye(n, m if m > 0 else n, k=self.k_input(), dtype=np.float32))


class NumpyDiagNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyDiagNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        k = 0
        if len(args) > 0:
            k = any_to_int(args[0])
        self.input = self.add_input('in', triggers_execution=True)
        self.k_input = self.add_input('diagonal', widget_type='input_int', default_value=k)
        self.output = self.add_output('diag out')

    def execute(self):
        data = any_to_array(self.input())
        if data.ndim > 2:
            if self.app.verbose:
                print(self.label, 'takes a list or a 2-D matrix, got shape', data.shape)
            return
        if data.ndim == 0:
            data = data.reshape(1)
        self.output.send(np.diag(data, k=self.k_input()))


class NumpyPolyfitNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyPolyfitNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        degree = 1
        if len(args) > 0:
            degree = max(any_to_int(args[0]), 0)
        self.x_value = None
        self.input = self.add_input('y', triggers_execution=True)
        self.x_input = self.add_input('x')
        self.degree_input = self.add_input('degree', widget_type='input_int', default_value=degree)
        self.output = self.add_output('coefficients')

    def execute(self):
        if self.x_input.fresh_input:
            self.x_value = any_to_array(self.x_input()).astype(float).ravel()
        y = any_to_array(self.input()).astype(float)
        if y.ndim > 2:
            if self.app.verbose:
                print(self.label, 'takes a list, or a 2-D array with one curve per column')
            return
        # without x the points are at 0, 1, 2 ...
        x = self.x_value if self.x_value is not None else np.arange(y.shape[0], dtype=float)
        if x.shape[0] != y.shape[0]:
            if self.app.verbose:
                print(self.label, 'x has', x.shape[0], 'values but y has', y.shape[0])
            return
        degree = max(self.degree_input(), 0)
        if y.shape[0] < degree + 1:
            if self.app.verbose:
                print(self.label, 'degree', degree, 'needs at least', degree + 1, 'points')
            return
        with warnings.catch_warnings():
            # a fit with more freedom than the points can pin down is still sent
            warnings.simplefilter('ignore', np.exceptions.RankWarning)
            self.output.send(np.polyfit(x, y, degree))


class NumpyPolyvalNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyPolyvalNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        coefficients = []
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [int, float]:
                coefficients.append(float(v))
        self.coefficients = np.array(coefficients) if len(coefficients) > 0 else None
        self.input = self.add_input('x', triggers_execution=True)
        self.coefficients_input = self.add_input('coefficients')
        self.output = self.add_output('values')

    def execute(self):
        if self.coefficients_input.fresh_input:
            self.coefficients = any_to_array(self.coefficients_input()).astype(float)
        if self.coefficients is None or self.coefficients.size == 0:
            return
        x = any_to_array(self.input()).astype(float)
        c = self.coefficients
        if c.ndim == 1:
            self.output.send(np.polyval(c, x))
            return
        # one column of coefficients per curve, as np.polyfit gives for several curves
        self.output.send(np.stack([np.polyval(c[:, i], x) for i in range(c.shape[1])], axis=-1))


class NumpySvdNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpySvdNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('matrix', triggers_execution=True)
        self.u_output = self.add_output('U')
        self.s_output = self.add_output('singular values')
        self.vh_output = self.add_output('Vh')
        self.full_option = self.add_option('full matrices', widget_type='checkbox', default_value=False)

    def execute(self):
        data = any_to_array(self.input()).astype(float)
        if data.ndim < 2:
            if self.app.verbose:
                print(self.label, 'needs a matrix, got shape', data.shape)
            return
        try:
            u, s, vh = np.linalg.svd(data, full_matrices=self.full_option())
        except np.linalg.LinAlgError as e:
            if self.app.verbose:
                print(self.label, e)
            return
        self.vh_output.send(vh)
        self.s_output.send(s)
        self.u_output.send(u)


class NumpyEighNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyEighNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('matrix', triggers_execution=True)
        self.values_output = self.add_output('eigenvalues')
        self.vectors_output = self.add_output('eigenvectors')
        self.order_option = self.add_option('order', widget_type='combo', default_value='largest first')
        self.order_option.widget.combo_items = ['largest first', 'smallest first']

    def execute(self):
        data = any_to_array(self.input()).astype(float)
        if data.ndim < 2 or data.shape[-1] != data.shape[-2]:
            if self.app.verbose:
                print(self.label, 'needs a square matrix, got shape', data.shape)
            return
        try:
            values, vectors = np.linalg.eigh(data)
        except np.linalg.LinAlgError as e:
            if self.app.verbose:
                print(self.label, e)
            return
        if self.order_option() == 'largest first':
            values = values[..., ::-1]
            vectors = vectors[..., ::-1]
        self.vectors_output.send(vectors)
        self.values_output.send(values)


class NumpyFullNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyFullNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        # np.full <dims ...> <value>: the last number is the value when there are two or more
        numbers = []
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [int, float]:
                numbers.append(v)
        self.shape = [3]
        value = 0.0
        if len(numbers) == 1:
            self.shape = [max(int(numbers[0]), 0)]
        elif len(numbers) > 1:
            self.shape = [max(int(n), 0) for n in numbers[:-1]]
            value = float(numbers[-1])
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.shape_input = self.add_input('shape', widget_type='text_input', widget_width=120,
                                          default_value=' '.join(str(d) for d in self.shape),
                                          callback=self.shape_changed)
        self.value_input = self.add_input('value', widget_type='drag_float', default_value=value)
        self.output = self.add_output('full out')

    def shape_changed(self):
        shape = parse_ints(self.shape_input())
        if len(shape) > 0:
            self.shape = [max(d, 0) for d in shape]

    def execute(self):
        if self.shape_input.fresh_input:
            self.shape_changed()
        self.output.send(np.full(self.shape, self.value_input(), dtype=np.float32))


class NumpyMeshgridNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyMeshgridNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.y_value = None
        self.input = self.add_input('x', triggers_execution=True)
        self.y_input = self.add_input('y')
        self.x_output = self.add_output('X')
        self.y_output = self.add_output('Y')
        self.indexing_option = self.add_option('indexing', widget_type='combo', default_value='xy')
        self.indexing_option.widget.combo_items = ['xy', 'ij']

    def execute(self):
        if self.y_input.fresh_input:
            self.y_value = any_to_array(self.y_input()).ravel()
        x = any_to_array(self.input()).ravel()
        # without y the grid is square, x by x
        y = self.y_value if self.y_value is not None else x
        indexing = 'ij' if self.indexing_option() == 'ij' else 'xy'
        grid_x, grid_y = np.meshgrid(x, y, indexing=indexing)
        self.y_output.send(grid_y)
        self.x_output.send(grid_x)


class NumpyTraceNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyTraceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        k = 0
        if len(args) > 0:
            k = any_to_int(args[0])
        self.input = self.add_input('in', triggers_execution=True)
        self.k_input = self.add_input('diagonal', widget_type='input_int', default_value=k)
        self.output = self.add_output('trace')

    def execute(self):
        data = any_to_array(self.input())
        if data.ndim < 2:
            if self.app.verbose:
                print(self.label, 'needs a matrix, got shape', data.shape)
            return
        # the last two axes, so a stack of matrices gives one trace each
        self.output.send(np.trace(data, offset=self.k_input(), axis1=-2, axis2=-1))


class NumpyTriangleNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyTriangleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = np.triu if label == 'np.triu' else np.tril
        k = 0
        if len(args) > 0:
            k = any_to_int(args[0])
        self.input = self.add_input('in', triggers_execution=True)
        self.k_input = self.add_input('diagonal', widget_type='input_int', default_value=k)
        self.output = self.add_output('triangle')

    def execute(self):
        data = any_to_array(self.input())
        if data.ndim < 2:
            if self.app.verbose:
                print(self.label, 'needs a matrix, got shape', data.shape)
            return
        self.output.send(self.op(data, k=self.k_input()))


class NumpyTakeAlongAxisNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyTakeAlongAxisNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        axis = -1
        if len(args) > 0:
            axis = any_to_int(args[0])
        self.indices = None
        self.input = self.add_input('in', triggers_execution=True)
        self.indices_input = self.add_input('indices')
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=axis)
        self.output = self.add_output('taken')

    def execute(self):
        if self.indices_input.fresh_input:
            self.indices = np.round(any_to_array(self.indices_input()).astype(float)).astype(int)
        if self.indices is None:
            return
        data = any_to_array(self.input())
        axis = self.axis_input()
        if self.indices.ndim != data.ndim or not -data.ndim <= axis < data.ndim:
            if self.app.verbose:
                print(self.label, 'indices must have as many axes as the array, and axis must be one of them:',
                      'shapes', data.shape, self.indices.shape, 'axis', axis)
            return
        n = data.shape[axis]
        if n == 0:
            return
        # out-of-range positions are clipped, and -1 counts from the end, as for np.take
        indices = np.clip(np.where(self.indices < 0, self.indices + n, self.indices), 0, n - 1)
        try:
            self.output.send(np.take_along_axis(data, indices, axis=axis))
        except (ValueError, IndexError):
            if self.app.verbose:
                print(self.label, 'shapes do not match along the other axes:', data.shape, indices.shape)


class NumpyTensordotNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyTensordotNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        axes = 2
        if len(args) > 0:
            axes = max(any_to_int(args[0]), 0)
        self.input_2_value = None
        self.input = self.add_input('in', triggers_execution=True)
        self.input_2 = self.add_input('in 2')
        self.axes_input = self.add_input('axes', widget_type='input_int', default_value=axes)
        self.output = self.add_output('result')

    def execute(self):
        if self.input_2.fresh_input:
            self.input_2_value = any_to_array(self.input_2())
        if self.input_2_value is None:
            return
        a = any_to_array(self.input())
        try:
            self.output.send(np.tensordot(a, self.input_2_value, axes=max(self.axes_input(), 0)))
        except ValueError:
            if self.app.verbose:
                print(self.label, 'the last', self.axes_input(), 'axes of', a.shape,
                      'do not match the first of', self.input_2_value.shape)


class NumpyQrNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyQrNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('matrix', triggers_execution=True)
        self.q_output = self.add_output('Q')
        self.r_output = self.add_output('R')
        self.mode_option = self.add_option('mode', widget_type='combo', default_value='reduced')
        self.mode_option.widget.combo_items = ['reduced', 'complete']

    def execute(self):
        data = any_to_array(self.input()).astype(float)
        if data.ndim < 2:
            if self.app.verbose:
                print(self.label, 'needs a matrix, got shape', data.shape)
            return
        mode = 'complete' if self.mode_option() == 'complete' else 'reduced'
        try:
            q, r = np.linalg.qr(data, mode=mode)
        except np.linalg.LinAlgError as e:
            if self.app.verbose:
                print(self.label, e)
            return
        self.r_output.send(r)
        self.q_output.send(q)


class NumpyEigNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyEigNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('matrix', triggers_execution=True)
        self.values_output = self.add_output('eigenvalues')
        self.vectors_output = self.add_output('eigenvectors')

    def execute(self):
        data = any_to_array(self.input()).astype(float)
        if data.ndim < 2 or data.shape[-1] != data.shape[-2]:
            if self.app.verbose:
                print(self.label, 'needs a square matrix, got shape', data.shape)
            return
        try:
            values, vectors = np.linalg.eig(data)
        except np.linalg.LinAlgError as e:
            if self.app.verbose:
                print(self.label, e)
            return
        # most matrices met in practice have real eigenvalues; send plain
        # numbers then, and complex ones only when a rotation makes them so
        if np.allclose(values.imag, 0) and np.allclose(vectors.imag, 0):
            values = values.real
            vectors = vectors.real
        self.vectors_output.send(vectors)
        self.values_output.send(values)


class NumpyComplexPartNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyComplexPartNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.part = label.split('.')[-1]
        self.input = self.add_input('in', triggers_execution=True)
        self.output = self.add_output(self.part)
        self.degrees_option = None
        if self.part == 'angle':
            self.degrees_option = self.add_option('degrees', widget_type='checkbox', default_value=False)

    def execute(self):
        data = any_to_array(self.input())
        if self.part == 'real':
            self.output.send(np.real(data))
        elif self.part == 'imag':
            self.output.send(np.imag(data))
        else:
            self.output.send(np.angle(data, deg=self.degrees_option()))


class NumpyUnwrapNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyUnwrapNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        units = 'degrees'
        if len(args) > 0 and args[0] in ['degrees', 'radians']:
            units = args[0]
        self.input = self.add_input('in', triggers_execution=True)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=-1)
        self.output = self.add_output('unwrapped')
        self.units_option = self.add_option('units', widget_type='combo', default_value=units)
        self.units_option.widget.combo_items = ['degrees', 'radians']

    def execute(self):
        data = any_to_array(self.input()).astype(float)
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', data.shape)
            return
        period = 360.0 if self.units_option() == 'degrees' else 2 * np.pi
        self.output.send(np.unwrap(data, period=period, axis=axis))


class NumpyAverageNode(NumpyNodeWithAxisNode):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyAverageNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t == int:
                self.axis = v
        self.weights = None
        self.input = self.add_input('in', triggers_execution=True)
        self.weights_input = self.add_input('weights')
        # None averages the whole array to one value
        self.add_dim_option()
        self.output = self.add_output('average')

    def execute(self):
        if self.weights_input.fresh_input:
            self.weights = any_to_array(self.weights_input()).astype(float)
        data = any_to_array(self.input()).astype(float)
        if not self.adjust_dim_option(data):
            if self.app.verbose:
                print(self.label, 'axis =', self.axis, 'out of range for shape', data.shape)
            return
        weights = self.weights
        if weights is not None:
            # one weight per element along the axis, or one per element of the whole array
            fits = weights.shape == data.shape or \
                (self.call_axis is not None and weights.ndim == 1 and weights.shape[0] == data.shape[self.call_axis])
            if not fits:
                if self.app.verbose:
                    print(self.label, 'weights of shape', weights.shape, 'do not fit', data.shape,
                          'along axis', self.call_axis)
                return
            if np.sum(weights) == 0:
                return
        self.output.send(np.average(data, axis=self.call_axis, weights=weights))


class NumpyHistogram2dNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyHistogram2dNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        numbers = []
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [int, float]:
                numbers.append(v)
        bins = int(numbers[0]) if len(numbers) > 0 else 10
        ranges = [float(n) for n in numbers[1:5]] if len(numbers) >= 5 else [0.0, 1.0, 0.0, 1.0]
        auto_range = len(numbers) < 5
        self.input = self.add_input('points', triggers_execution=True)
        self.bins_input = self.add_input('bins', widget_type='input_int', default_value=bins)
        self.x_min_input = self.add_input('x min', widget_type='drag_float', default_value=ranges[0])
        self.x_max_input = self.add_input('x max', widget_type='drag_float', default_value=ranges[1])
        self.y_min_input = self.add_input('y min', widget_type='drag_float', default_value=ranges[2])
        self.y_max_input = self.add_input('y max', widget_type='drag_float', default_value=ranges[3])
        self.counts_output = self.add_output('counts')
        self.x_edges_output = self.add_output('x edges')
        self.y_edges_output = self.add_output('y edges')
        self.auto_range_option = self.add_option('auto range', widget_type='checkbox', default_value=auto_range)
        self.density_option = self.add_option('density', widget_type='checkbox', default_value=False)

    def execute(self):
        points = any_to_array(self.input()).astype(float)
        if points.ndim != 2 or points.shape[1] < 2:
            if self.app.verbose:
                print(self.label, 'needs points as rows of x y, got shape', points.shape)
            return
        x, y = points[:, 0], points[:, 1]
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        bins = max(self.bins_input(), 1)
        if self.auto_range_option():
            if x.size == 0:
                return
            value_range = [[x.min(), x.max()], [y.min(), y.max()]]
        else:
            value_range = [[self.x_min_input(), self.x_max_input()], [self.y_min_input(), self.y_max_input()]]
            if value_range[0][1] <= value_range[0][0] or value_range[1][1] <= value_range[1][0]:
                if self.app.verbose:
                    print(self.label, 'each max must be above its min, got', value_range)
                return
        counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=value_range,
                                                  density=self.density_option() and x.size > 0)
        self.y_edges_output.send(y_edges)
        self.x_edges_output.send(x_edges)
        # as a picture: one row per y bin, one column per x bin
        self.counts_output.send(counts.T)


class NumpyTopKNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyTopKNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        k = 3
        which = 'largest'
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [int, float]:
                k = max(int(v), 1)
            elif t == str and v in ['largest', 'smallest']:
                which = v
        self.input = self.add_input('in', triggers_execution=True)
        self.k_input = self.add_input('k', widget_type='input_int', default_value=k)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=-1)
        self.indices_output = self.add_output('indices')
        self.values_output = self.add_output('values')
        self.which_option = self.add_option('which', widget_type='combo', default_value=which)
        self.which_option.widget.combo_items = ['largest', 'smallest']

    def execute(self):
        data = any_to_array(self.input())
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim or data.shape[axis] == 0:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range or empty for shape', data.shape)
            return
        n = data.shape[axis]
        k = min(max(self.k_input(), 1), n)
        largest = self.which_option() != 'smallest'
        keys = -data if largest else data
        # argpartition finds the k without sorting the rest; then only those k are sorted
        part = np.argpartition(keys, k - 1, axis=axis) if k < n else np.argsort(keys, axis=axis)
        part = np.take(part, np.arange(k), axis=axis)
        order = np.argsort(np.take_along_axis(keys, part, axis=axis), axis=axis, kind='stable')
        indices = np.take_along_axis(part, order, axis=axis)
        self.values_output.send(np.take_along_axis(data, indices, axis=axis))
        self.indices_output.send(indices)


class NumpyIsCloseNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyIsCloseNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.all = label == 'np.allclose'
        tolerance = 0.001
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                tolerance = float(v)
        self.input_2_value = None
        self.input = self.add_input('in', triggers_execution=True)
        self.input_2 = self.add_input('in 2')
        self.atol_input = self.add_input('tolerance', widget_type='drag_float', default_value=tolerance)
        self.rtol_input = self.add_input('relative', widget_type='drag_float', default_value=0.0)
        self.output = self.add_output('all close' if self.all else 'close')

    def execute(self):
        if self.input_2.fresh_input:
            self.input_2_value = any_to_array(self.input_2())
        if self.input_2_value is None:
            return
        a = any_to_array(self.input())
        atol = max(self.atol_input(), 0.0)
        rtol = max(self.rtol_input(), 0.0)
        try:
            if self.all:
                self.output.send(bool(np.allclose(a, self.input_2_value, rtol=rtol, atol=atol)))
            else:
                self.output.send(np.isclose(a, self.input_2_value, rtol=rtol, atol=atol))
        except ValueError:
            if self.app.verbose:
                print(self.label, 'shapes do not broadcast:', a.shape, self.input_2_value.shape)


class NumpyTrapezoidNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyTrapezoidNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.cumulative = label == 'np.cumulative_trapezoid'
        spacing = 1.0
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                spacing = float(v)
        self.input = self.add_input('in', triggers_execution=True)
        self.spacing_input = self.add_input('spacing', widget_type='drag_float', default_value=spacing)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=-1)
        self.output = self.add_output('running total' if self.cumulative else 'area')

    def execute(self):
        data = any_to_array(self.input()).astype(float)
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', data.shape)
            return
        spacing = self.spacing_input()
        if not self.cumulative:
            self.output.send(np.trapezoid(data, dx=spacing, axis=axis))
            return
        # the area so far at every sample, starting from 0: the same length as the input
        moved = np.moveaxis(data, axis, -1)
        steps = (moved[..., 1:] + moved[..., :-1]) * 0.5 * spacing
        totals = np.concatenate([np.zeros(moved.shape[:-1] + (1,)), np.cumsum(steps, axis=-1)], axis=-1)
        self.output.send(np.moveaxis(totals, -1, axis))


class NumpyFftNode(Node):
    norms = ['backward', 'ortho', 'forward']

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyFftNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        rate = 1.0
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                rate = float(v)
        self.input = self.add_input('in', triggers_execution=True)
        self.rate_input = self.add_input('sample rate', widget_type='drag_float', default_value=rate)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=-1)
        self.spectrum_output = self.add_output('spectrum')
        self.magnitude_output = self.add_output('magnitude')
        self.phase_output = self.add_output('phase')
        self.frequencies_output = self.add_output('frequencies')
        self.norm_option = self.add_option('norm', widget_type='combo', default_value='backward')
        self.norm_option.widget.combo_items = self.norms

    def execute(self):
        data = any_to_array(self.input())
        if not np.iscomplexobj(data):
            data = data.astype(float)
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim or data.shape[axis] < 1:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range or empty for shape', data.shape)
            return
        norm = self.norm_option()
        if norm not in self.norms:
            norm = 'backward'
        spectrum = np.fft.fft(data, axis=axis, norm=norm)
        rate = self.rate_input()
        if rate > 0:
            self.frequencies_output.send(np.fft.fftfreq(data.shape[axis], d=1.0 / rate))
        self.phase_output.send(np.angle(spectrum))
        self.magnitude_output.send(np.abs(spectrum))
        self.spectrum_output.send(spectrum)


class NumpyIfftNode(Node):
    norms = ['backward', 'ortho', 'forward']

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyIfftNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('spectrum', triggers_execution=True)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=-1)
        self.output = self.add_output('signal')
        self.norm_option = self.add_option('norm', widget_type='combo', default_value='backward')
        self.norm_option.widget.combo_items = self.norms

    def execute(self):
        spectrum = any_to_array(self.input())
        axis = self.axis_input()
        if not -spectrum.ndim <= axis < spectrum.ndim or spectrum.shape[axis] < 1:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range or empty for shape', spectrum.shape)
            return
        norm = self.norm_option()
        if norm not in self.norms:
            norm = 'backward'
        signal = np.fft.ifft(spectrum, axis=axis, norm=norm)
        # a spectrum that came from a real signal gives it back with only rounding
        # dust in the imaginary part: send plain numbers then
        if np.allclose(signal.imag, 0, atol=1e-9 * max(np.abs(signal).max(initial=0.0), 1.0)):
            signal = signal.real
        self.output.send(signal)


class NumpySaveNode(Node):
    # np.save writes .npy (or .npz, under the key 'data'); np.savetxt writes a
    # text table (.csv). Arriving arrays are only held: the file is written when
    # 'write' is clicked or banged, never on every input.
    delimiters = {'comma': ',', 'tab': '\t', 'space': ' '}

    @staticmethod
    def factory(name, data, args=None):
        node = NumpySaveNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.text = label == 'np.savetxt'
        self.extension = '.csv' if self.text else '.npy'
        default_path = args[0] if len(args) > 0 else ''
        self.array = None
        self.input = self.add_input('in', triggers_execution=True)
        self.write_button = self.add_input('write', widget_type='button', callback=self.write)
        self.save_as_button = self.add_input('save as', widget_type='button', callback=self.open_save_dialog)
        self.path_option = self.add_option('path', widget_type='text_input', default_value=default_path)
        self.delimiter_option = None
        if self.text:
            self.delimiter_option = self.add_option('delimiter', widget_type='combo', default_value='comma')
            self.delimiter_option.widget.combo_items = list(self.delimiters.keys())
        self.output = self.add_output('saved path')

    def execute(self):
        self.array = any_to_array(self.input())

    def write(self, input=None):
        path = self.path_option()
        if path is None or path == '':
            self.open_save_dialog()
            return
        self.write_file(path)

    def open_save_dialog(self, input=None):
        SaveDialog(self, callback=self.save_dialog_callback, extensions=['.csv', '.txt'] if self.text else ['.npy', '.npz'],
                   default_filename='untitled' + self.extension)

    def save_dialog_callback(self, save_path):
        if save_path != '':
            self.path_option.set(save_path)
            self.write_file(save_path)

    def write_file(self, path):
        if self.array is None:
            if self.app.verbose:
                print(self.label, 'has no array to write yet')
            return
        if os.path.splitext(path)[1] == '':
            path += self.extension
        try:
            if self.text:
                if self.array.ndim > 2:
                    print(f'{self.label}: a text table holds 1-D or 2-D arrays, not shape {self.array.shape}')
                    return
                delimiter = self.delimiters.get(self.delimiter_option(), ',')
                np.savetxt(path, self.array, delimiter=delimiter, fmt='%.9g')
            elif path.endswith('.npz'):
                np.savez(path, data=self.array)
            else:
                np.save(path, self.array)
        except Exception as e:
            print(f'{self.label} error writing {path}: {type(e).__name__}: {e}')
            return
        self.output.send(path)


class NumpyLoadTextNode(Node):
    delimiters = {'any space': None, 'comma': ',', 'tab': '\t'}

    @staticmethod
    def factory(name, data, args=None):
        node = NumpyLoadTextNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        default_path = args[0] if len(args) > 0 else ''
        self.array = None
        self.load_button = self.add_input('load', widget_type='button', callback=self.open_load_dialog)
        self.add_input('send', widget_type='button', callback=self.send_array)
        self.path_option = self.add_option('path', widget_type='text_input', default_value=default_path,
                                           callback=self.path_changed)
        self.delimiter_option = self.add_option('delimiter', widget_type='combo', default_value='comma',
                                                callback=self.path_changed)
        self.delimiter_option.widget.combo_items = list(self.delimiters.keys())
        self.skip_option = self.add_option('skip rows', widget_type='input_int', default_value=0,
                                           callback=self.path_changed)
        self.output = self.add_output('array')

    def custom_create(self, from_file=False):
        self.path_changed()

    def path_changed(self):
        path = self.path_option()
        if path is not None and path != '':
            self.load_file(path)

    def send_array(self, input=None):
        if self.array is not None:
            self.output.send(self.array)

    def open_load_dialog(self, input=None):
        LoadDialog(self, callback=self.load_dialog_callback, extensions=['.csv', '.txt', '.tsv'])

    def load_dialog_callback(self, load_path):
        if load_path != '':
            self.path_option.set(load_path)
            self.load_file(load_path)

    def load_file(self, path):
        delimiter = self.delimiters.get(self.delimiter_option(), ',')
        skip = max(self.skip_option() or 0, 0)
        try:
            # genfromtxt rather than loadtxt: an empty or unreadable cell becomes nan
            # instead of stopping the whole load
            array = np.genfromtxt(path, delimiter=delimiter, skip_header=skip, dtype=float)
        except Exception as e:
            print(f'{self.label} error loading {path}: {type(e).__name__}: {e}')
            return
        self.array = array
        self.output.send(array)


def items_to_array(value):
    """An array of things to choose from or shuffle. A list holding words stays a
    list of words; any_to_array would read them as number words and give nothing."""
    if isinstance(value, str):
        value = value.split()
    if isinstance(value, (list, tuple)) and any(isinstance(v, str) for v in value):
        return np.array(value, dtype=object)
    return any_to_array(value)


class NumpyRandomBase(Node):
    # a seed of -1 draws different numbers every time the patch opens; any other
    # seed repeats the same sequence from the start
    def add_seed_option(self):
        self.rng = np.random.default_rng()
        self.seed_option = self.add_option('seed', widget_type='input_int', default_value=-1,
                                           callback=self.seed_changed)

    def seed_changed(self):
        seed = self.seed_option()
        self.rng = np.random.default_rng(None if seed is None or seed < 0 else seed)


class NumpyRandomArrayNode(NumpyRandomBase):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyRandomArrayNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.integers = label == 'np.random.integers'
        self.shape = [max(any_to_int(a), 0) for a in args] if len(args) > 0 else [10]
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.shape_input = self.add_input('shape', widget_type='text_input', widget_width=120,
                                          default_value=' '.join(str(d) for d in self.shape),
                                          callback=self.shape_changed)
        if self.integers:
            self.low_input = self.add_input('low', widget_type='input_int', default_value=0)
            self.high_input = self.add_input('high', widget_type='input_int', default_value=10)
        else:
            self.low_input = self.add_input('mean', widget_type='drag_float', default_value=0.0)
            self.high_input = self.add_input('spread', widget_type='drag_float', default_value=1.0)
        self.output = self.add_output('random array')
        self.add_seed_option()

    def shape_changed(self):
        shape = parse_ints(self.shape_input())
        if len(shape) > 0:
            self.shape = [max(d, 0) for d in shape]

    def execute(self):
        if self.shape_input.fresh_input:
            self.shape_changed()
        if self.integers:
            low, high = self.low_input(), self.high_input()
            if high <= low:
                if self.app.verbose:
                    print(self.label, 'high must be above low')
                return
            self.output.send(self.rng.integers(low, high, size=self.shape))
        else:
            self.output.send(self.rng.normal(self.low_input(), max(self.high_input(), 0.0),
                                             size=self.shape).astype(np.float32))


class NumpyRandomChoiceNode(NumpyRandomBase):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyRandomChoiceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        # np.random.choice <n>: choose from 0 to n-1 until a list arrives at 'from'
        self.choices = None
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                self.choices = np.arange(max(int(v), 1))
        self.weights = None
        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.from_input = self.add_input('from')
        self.count_input = self.add_input('count', widget_type='input_int', default_value=1)
        self.weights_input = self.add_input('weights')
        self.output = self.add_output('chosen')
        self.repeats_option = self.add_option('repeats', widget_type='checkbox', default_value=True)
        self.add_seed_option()

    def execute(self):
        if self.from_input.fresh_input:
            self.choices = items_to_array(self.from_input())
        if self.weights_input.fresh_input:
            self.weights = any_to_array(self.weights_input()).astype(float).ravel()
        if self.choices is None or len(self.choices) == 0:
            return
        n = len(self.choices)
        p = None
        if self.weights is not None:
            if self.weights.shape[0] != n or np.any(self.weights < 0) or self.weights.sum() == 0:
                if self.app.verbose:
                    print(self.label, 'needs one non-negative weight per choice, not all zero')
                return
            p = self.weights / self.weights.sum()
        count = max(self.count_input(), 1)
        replace = self.repeats_option()
        if not replace and count > n:
            count = n
        chosen = self.rng.choice(self.choices, size=count, replace=replace, p=p, axis=0)
        # a single pick is sent as the item itself, not a list of one
        self.output.send(chosen[0] if count == 1 else chosen)


class NumpyRandomPermutationNode(NumpyRandomBase):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyRandomPermutationNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('in', triggers_execution=True)
        self.output = self.add_output('shuffled')
        self.add_seed_option()

    def execute(self):
        value = self.input()
        # a message box sends 8 as the list [8]; shuffling one item means nothing, so
        # a single whole number in a list is taken as n too
        if isinstance(value, (list, tuple, np.ndarray)) and np.size(value) == 1:
            single = np.ravel(np.asarray(value, dtype=object))[0]
            if isinstance(single, (int, np.integer, float, np.floating)) and float(single).is_integer():
                value = int(single)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            # a whole number n: the numbers 0 to n-1 in random order
            self.output.send(self.rng.permutation(max(int(value), 0)))
            return
        data = items_to_array(value)
        if data.ndim == 0 or data.size == 0:
            return
        # shuffles the first axis: the elements of a list, the rows of a matrix
        self.output.send(self.rng.permutation(data, axis=0))


class NumpySlidingWindowNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpySlidingWindowNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        window, step = 4, 1
        if len(args) > 0:
            window = max(any_to_int(args[0]), 1)
        if len(args) > 1:
            step = max(any_to_int(args[1]), 1)
        self.input = self.add_input('in', triggers_execution=True)
        self.window_input = self.add_input('window', widget_type='input_int', default_value=window)
        self.step_input = self.add_input('step', widget_type='input_int', default_value=step)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=-1)
        self.output = self.add_output('windows')

    def execute(self):
        data = any_to_array(self.input())
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', data.shape)
            return
        axis = axis % data.ndim
        window = max(self.window_input(), 1)
        if data.shape[axis] < window:
            if self.app.verbose:
                print(self.label, 'a window of', window, 'is longer than the', data.shape[axis], 'values')
            return
        windows = np.lib.stride_tricks.sliding_window_view(data, window, axis=axis)
        step = max(self.step_input(), 1)
        if step > 1:
            windows = np.take(windows, np.arange(0, windows.shape[axis], step), axis=axis)
        # a copy: a view would share memory with the input, which a later node may change
        self.output.send(np.array(windows))


class NumpyRunningExtremeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyRunningExtremeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.largest = label == 'np.maximum.accumulate'
        axis = -1
        if len(args) > 0:
            axis = any_to_int(args[0])
        self.input = self.add_input('in', triggers_execution=True)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=axis)
        self.output = self.add_output('running max' if self.largest else 'running min')

    def execute(self):
        data = any_to_array(self.input())
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', data.shape)
            return
        # fmax / fmin skip nan, so one gap does not wipe out the rest of the run
        op = np.fmax if self.largest else np.fmin
        self.output.send(op.accumulate(data, axis=axis))


class NumpyInsertNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyInsertNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        index = 0
        if len(args) > 0:
            index = any_to_int(args[0])
        self.values = None
        self.input = self.add_input('in', triggers_execution=True)
        self.values_input = self.add_input('values')
        self.index_input = self.add_input('index', widget_type='input_int', default_value=index)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=0)
        self.output = self.add_output('inserted')

    def execute(self):
        if self.values_input.fresh_input:
            self.values = any_to_array(self.values_input())
        if self.values is None:
            return
        data = any_to_array(self.input())
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', data.shape)
            return
        n = data.shape[axis]
        index = self.index_input()
        # a negative index counts from the end; out of range goes to the nearest end
        if index < 0:
            index += n + 1
        index = min(max(index, 0), n)
        try:
            self.output.send(np.insert(data, index, self.values, axis=axis))
        except ValueError:
            if self.app.verbose:
                print(self.label, 'values of shape', self.values.shape, 'do not fit', data.shape)


class NumpyDeleteNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyDeleteNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        indices = []
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [int, float]:
                indices.append(int(v))
        self.indices = np.array(indices, dtype=int)
        self.input = self.add_input('in', triggers_execution=True)
        self.indices_input = self.add_input('indices')
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=0)
        self.output = self.add_output('remaining')

    def execute(self):
        if self.indices_input.fresh_input:
            self.indices = np.round(any_to_array(self.indices_input()).astype(float)).astype(int).ravel()
        data = any_to_array(self.input())
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', data.shape)
            return
        n = data.shape[axis]
        # -1 counts from the end; positions that do not exist are ignored
        indices = np.where(self.indices < 0, self.indices + n, self.indices)
        indices = indices[(indices >= 0) & (indices < n)]
        self.output.send(np.delete(data, indices, axis=axis))


class NumpyAppendNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyAppendNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        values = []
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [int, float]:
                values.append(v)
        self.values = np.array(values) if len(values) > 0 else None
        self.input = self.add_input('in', triggers_execution=True)
        self.values_input = self.add_input('values')
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=0)
        self.output = self.add_output('appended')

    def execute(self):
        if self.values_input.fresh_input:
            self.values = any_to_array(self.values_input())
        if self.values is None:
            return
        data = any_to_array(self.input())
        values = self.values
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', data.shape)
            return
        # a single row onto a matrix: give it the missing axis
        if values.ndim == data.ndim - 1:
            values = np.expand_dims(values, axis)
        try:
            self.output.send(np.append(data, values, axis=axis))
        except ValueError:
            if self.app.verbose:
                print(self.label, 'values of shape', self.values.shape, 'do not fit', data.shape,
                      'along axis', axis)


class NumpyBincountNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NumpyBincountNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        length = 0
        if len(args) > 0:
            length = max(any_to_int(args[0]), 0)
        self.weights = None
        self.input = self.add_input('labels', triggers_execution=True)
        self.weights_input = self.add_input('weights')
        self.length_input = self.add_input('length', widget_type='input_int', default_value=length)
        self.output = self.add_output('counts')

    def execute(self):
        if self.weights_input.fresh_input:
            self.weights = any_to_array(self.weights_input()).astype(float).ravel()
        labels = np.round(any_to_array(self.input()).astype(float)).astype(int).ravel()
        if np.any(labels < 0):
            if self.app.verbose:
                print(self.label, 'labels must be 0 or more')
            return
        weights = self.weights
        if weights is not None and weights.shape[0] != labels.shape[0]:
            if self.app.verbose:
                print(self.label, 'has', weights.shape[0], 'weights for', labels.shape[0], 'labels; counting instead')
            weights = None
        self.output.send(np.bincount(labels, weights=weights, minlength=max(self.length_input(), 0)))
