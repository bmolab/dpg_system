import numpy as np
import warnings
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from scipy import ndimage
from fractions import Fraction
import re

from dpg_system.node import Node
from dpg_system.conversion_utils import *


def register_scipy_nodes():
    Node.app.register_node('scipy.find_peaks', ScipyFindPeaksNode.factory)
    Node.app.register_node('scipy.filtfilt', ScipyFiltFiltNode.factory)
    Node.app.register_node('scipy.kdtree', ScipyKDTreeNode.factory)
    Node.app.register_node('scipy.cubic_spline', ScipyCubicSplineNode.factory)
    Node.app.register_node('scipy.resample_poly', ScipyResampleNode.factory)
    Node.app.register_node('scipy.spectrogram', ScipySpectrumNode.factory)
    Node.app.register_node('scipy.welch', ScipySpectrumNode.factory)
    Node.app.register_node('scipy.label', ScipyLabelNode.factory)
    Node.app.register_node('scipy.gaussian_filter', ScipyImageFilterNode.factory)
    Node.app.register_node('scipy.median_filter', ScipyImageFilterNode.factory)


class ScipyFindPeaksNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ScipyFindPeaksNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        prominence = 0.0
        spacing = 1
        if len(args) > 0:
            v, t = decode_arg(args, 0)
            if t in [int, float]:
                prominence = float(v)
        if len(args) > 1:
            v, t = decode_arg(args, 1)
            if t in [int, float]:
                spacing = max(int(v), 1)
        self.input = self.add_input('in', triggers_execution=True)
        self.prominence_input = self.add_input('prominence', widget_type='drag_float', default_value=prominence)
        self.spacing_input = self.add_input('spacing', widget_type='input_int', default_value=spacing)
        self.height_input = self.add_input('min height', widget_type='drag_float', default_value=0.0)
        self.indices_output = self.add_output('indices')
        self.heights_output = self.add_output('heights')
        self.count_output = self.add_output('count')
        self.use_height_option = self.add_option('use min height', widget_type='checkbox', default_value=False)
        self.find_option = self.add_option('find', widget_type='combo', default_value='peaks')
        self.find_option.widget.combo_items = ['peaks', 'valleys']

    def execute(self):
        data = any_to_array(self.input()).astype(float)
        if data.ndim > 1:
            if sum(d > 1 for d in data.shape) > 1:
                if self.app.verbose:
                    print(self.label, 'looks along a single line of values, got shape', data.shape)
                return
            data = data.ravel()
        valleys = self.find_option() == 'valleys'
        # valleys are the peaks of the curve turned upside down
        curve = -data if valleys else data
        # a gap would hide every peak near it; treat it as a deep dip instead
        curve = np.where(np.isfinite(curve), curve, -np.inf)
        kwargs = {'distance': max(self.spacing_input(), 1)}
        prominence = self.prominence_input()
        if prominence > 0:
            kwargs['prominence'] = prominence
        if self.use_height_option():
            # for valleys the limit is a ceiling: only dips below it count
            height = self.height_input()
            kwargs['height'] = -height if valleys else height
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            indices, _ = signal.find_peaks(curve, **kwargs)
        self.count_output.send(int(len(indices)))
        self.heights_output.send(data[indices])
        self.indices_output.send(indices)


class ScipyFiltFiltNode(Node):
    kinds = ['lowpass', 'highpass', 'bandpass', 'bandstop']

    @staticmethod
    def factory(name, data, args=None):
        node = ScipyFiltFiltNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        kind = 'lowpass'
        numbers = []
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t == str and v in self.kinds:
                kind = v
            elif t in [int, float]:
                numbers.append(float(v))
        cutoff = numbers[0] if len(numbers) > 0 else 5.0
        cutoff_2 = numbers[1] if len(numbers) > 1 else cutoff * 2
        self.input = self.add_input('in', triggers_execution=True)
        self.cutoff_input = self.add_input('cutoff', widget_type='drag_float', default_value=cutoff)
        self.cutoff_2_input = self.add_input('cutoff 2', widget_type='drag_float', default_value=cutoff_2)
        self.rate_input = self.add_input('sample rate', widget_type='drag_float', default_value=60.0)
        self.order_input = self.add_input('order', widget_type='input_int', default_value=4)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=-1)
        self.output = self.add_output('filtered')
        self.kind_option = self.add_option('kind', widget_type='combo', default_value=kind)
        self.kind_option.widget.combo_items = self.kinds

    def execute(self):
        data = any_to_array(self.input()).astype(float)
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', data.shape)
            return
        rate = self.rate_input()
        nyquist = rate / 2
        kind = self.kind_option()
        if kind not in self.kinds:
            kind = 'lowpass'
        if kind in ['bandpass', 'bandstop']:
            edges = sorted([self.cutoff_input(), self.cutoff_2_input()])
            if edges[0] <= 0 or edges[1] >= nyquist or edges[0] == edges[1]:
                if self.app.verbose:
                    print(self.label, 'the two cutoffs must differ and lie between 0 and half the sample rate', nyquist)
                return
            frequencies = edges
        else:
            frequencies = self.cutoff_input()
            if not 0 < frequencies < nyquist:
                if self.app.verbose:
                    print(self.label, 'the cutoff must lie between 0 and half the sample rate', nyquist)
                return
        order = min(max(self.order_input(), 1), 10)
        sos = signal.butter(order, frequencies, btype=kind, fs=rate, output='sos')
        n = data.shape[axis]
        # a short recording cannot take the default padding; use what it has
        default_pad = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
        if n < 2:
            return
        self.output.send(signal.sosfiltfilt(sos, data, axis=axis, padlen=min(default_pad, n - 1)))


class ScipyKDTreeNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ScipyKDTreeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        k = 1
        if len(args) > 0:
            k = max(any_to_int(args[0]), 1)
        self.tree = None
        self.point_count = 0
        self.input = self.add_input('query', triggers_execution=True)
        self.points_input = self.add_input('points')
        self.k_input = self.add_input('k', widget_type='input_int', default_value=k)
        self.max_distance_input = self.add_input('max distance', widget_type='drag_float', default_value=0.0)
        self.indices_output = self.add_output('indices')
        self.distances_output = self.add_output('distances')

    def points_changed(self):
        points = any_to_array(self.points_input()).astype(float)
        if points.ndim == 1:
            # a list of single numbers: points on a line
            points = points.reshape(-1, 1)
        if points.ndim != 2 or points.shape[0] == 0:
            if self.app.verbose:
                print(self.label, 'points must be rows of coordinates, got shape', points.shape)
            return
        # building the tree is the slow part; it is done once per set of points
        self.tree = cKDTree(points)
        self.point_count = points.shape[0]

    def execute(self):
        if self.points_input.fresh_input:
            self.points_changed()
        if self.tree is None:
            return
        query = any_to_array(self.input()).astype(float)
        dims = self.tree.m
        if query.shape[-1] != dims:
            if dims == 1 and query.ndim <= 1:
                query = query.reshape(-1, 1)
            else:
                if self.app.verbose:
                    print(self.label, 'query points need', dims, 'coordinates, got shape', query.shape)
                return
        k = min(max(self.k_input(), 1), self.point_count)
        max_distance = self.max_distance_input()
        bound = max_distance if max_distance > 0 else np.inf
        distances, indices = self.tree.query(query, k=k, distance_upper_bound=bound)
        # nothing within max distance: scipy gives the point count as the index; send -1
        indices = np.where(np.isfinite(distances), indices, -1)
        self.distances_output.send(distances)
        self.indices_output.send(indices)


class ScipyCubicSplineNode(Node):
    boundaries = ['not-a-knot', 'natural', 'clamped', 'periodic']

    @staticmethod
    def factory(name, data, args=None):
        node = ScipyCubicSplineNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        boundary = 'not-a-knot'
        if len(args) > 0 and args[0] in self.boundaries:
            boundary = args[0]
        self.xp_value = None
        self.fp_value = None
        self.spline = None
        self.input = self.add_input('x', triggers_execution=True)
        self.xp_input = self.add_input('xp')
        self.fp_input = self.add_input('fp')
        self.output = self.add_output('values')
        self.boundary_option = self.add_option('boundary', widget_type='combo', default_value=boundary,
                                               callback=self.invalidate)
        self.boundary_option.widget.combo_items = self.boundaries
        self.extrapolate_option = self.add_option('extrapolate', widget_type='checkbox', default_value=False)

    def invalidate(self, input=None):
        self.spline = None

    def build(self):
        fp = self.fp_value
        xp = self.xp_value if self.xp_value is not None else np.arange(fp.shape[0], dtype=float)
        if xp.shape[0] != fp.shape[0]:
            if self.app.verbose:
                print(self.label, 'xp has', xp.shape[0], 'positions but fp has', fp.shape[0], 'values')
            return None
        # positions in increasing order, each once
        order = np.argsort(xp, kind='stable')
        xp, fp = xp[order], fp[order]
        keep = np.concatenate([[True], np.diff(xp) > 0])
        xp, fp = xp[keep], fp[keep]
        if xp.shape[0] < 2:
            if self.app.verbose:
                print(self.label, 'needs at least 2 distinct points')
            return None
        boundary = self.boundary_option()
        if boundary not in self.boundaries:
            boundary = 'not-a-knot'
        if boundary == 'periodic' and not np.allclose(fp[0], fp[-1]):
            # a loop must end where it starts: close it
            fp = fp.copy()
            fp[-1] = fp[0]
        try:
            return CubicSpline(xp, fp, axis=0, bc_type=boundary)
        except ValueError as e:
            if self.app.verbose:
                print(self.label, e)
            return None

    def execute(self):
        if self.xp_input.fresh_input:
            self.xp_value = any_to_array(self.xp_input()).astype(float).ravel()
            self.spline = None
        if self.fp_input.fresh_input:
            self.fp_value = any_to_array(self.fp_input()).astype(float)
            self.spline = None
        if self.fp_value is None or self.fp_value.shape[0] == 0:
            return
        if self.spline is None:
            self.spline = self.build()
            if self.spline is None:
                return
        x = any_to_array(self.input()).astype(float)
        if self.boundary_option() == 'periodic':
            self.output.send(self.spline(x, extrapolate='periodic'))
            return
        if not self.extrapolate_option():
            # held flat beyond the ends, as np.interp does
            x = np.clip(x, self.spline.x[0], self.spline.x[-1])
        self.output.send(self.spline(x))


class ScipyResampleNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ScipyResampleNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        from_rate, to_rate = 60.0, 100.0
        if len(args) > 0:
            from_rate = any_to_float(args[0])
        if len(args) > 1:
            to_rate = any_to_float(args[1])
        self.input = self.add_input('in', triggers_execution=True)
        self.from_input = self.add_input('from rate', widget_type='drag_float', default_value=from_rate)
        self.to_input = self.add_input('to rate', widget_type='drag_float', default_value=to_rate)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=-1)
        self.output = self.add_output('resampled')

    def execute(self):
        data = any_to_array(self.input()).astype(float)
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim or data.shape[axis] < 2:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range or too short for shape', data.shape)
            return
        from_rate, to_rate = self.from_input(), self.to_input()
        if from_rate <= 0 or to_rate <= 0:
            return
        # the rates as a ratio of small whole numbers: 60 to 100 is up 5, down 3
        ratio = Fraction(to_rate / from_rate).limit_denominator(1000)
        self.output.send(signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=axis))


class ScipySpectrumNode(Node):
    # scipy.spectrogram: power at every frequency for each moment; scipy.welch: the
    # average over the whole signal, steadier than a single spectrum
    @staticmethod
    def factory(name, data, args=None):
        node = ScipySpectrumNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.over_time = label == 'scipy.spectrogram'
        window = 64
        rate = 60.0
        if len(args) > 0:
            window = max(any_to_int(args[0]), 2)
        if len(args) > 1:
            rate = any_to_float(args[1])
        self.input = self.add_input('in', triggers_execution=True)
        self.rate_input = self.add_input('sample rate', widget_type='drag_float', default_value=rate)
        self.window_input = self.add_input('window', widget_type='input_int', default_value=window)
        self.overlap_input = self.add_input('overlap', widget_type='drag_float', default_value=0.5)
        self.power_output = self.add_output('power')
        self.frequencies_output = self.add_output('frequencies')
        self.times_output = None
        if self.over_time:
            self.times_output = self.add_output('times')
        self.decibels_option = self.add_option('decibels', widget_type='checkbox', default_value=self.over_time)

    def execute(self):
        data = any_to_array(self.input()).astype(float)
        if data.ndim > 1:
            if sum(d > 1 for d in data.shape) > 1:
                if self.app.verbose:
                    print(self.label, 'takes a single line of samples, got shape', data.shape)
                return
            data = data.ravel()
        rate = self.rate_input()
        if rate <= 0 or data.size < 2:
            return
        window = min(max(self.window_input(), 2), data.size)
        # overlap as a fraction of the window: 0.5 starts each window halfway into the last
        overlap = int(window * min(max(self.overlap_input(), 0.0), 0.95))
        data = np.nan_to_num(data)
        if self.over_time:
            frequencies, times, power = signal.spectrogram(data, fs=rate, nperseg=window, noverlap=overlap)
        else:
            frequencies, power = signal.welch(data, fs=rate, nperseg=window, noverlap=overlap)
        if self.decibels_option():
            # 10 log10 of power; a floor keeps silence from becoming -infinity
            power = 10.0 * np.log10(np.maximum(power, 1e-12))
        if self.times_output is not None:
            self.times_output.send(times)
        self.frequencies_output.send(frequencies)
        self.power_output.send(power)


class ScipyLabelNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ScipyLabelNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        threshold = 0.5
        if len(args) > 0:
            threshold = any_to_float(args[0])
        self.input = self.add_input('in', triggers_execution=True)
        self.threshold_input = self.add_input('threshold', widget_type='drag_float', default_value=threshold)
        self.labels_output = self.add_output('labels')
        self.count_output = self.add_output('count')
        self.sizes_output = self.add_output('sizes')
        self.centres_output = self.add_output('centres')
        self.diagonal_option = self.add_option('diagonals connect', widget_type='checkbox', default_value=False)

    def execute(self):
        data = any_to_array(self.input())
        if data.ndim < 1:
            return
        if data.dtype == np.bool_:
            mask = data
        else:
            mask = np.nan_to_num(data.astype(float), nan=-np.inf) > self.threshold_input()
        # neighbours: side by side only, or corners too
        structure = ndimage.generate_binary_structure(mask.ndim, mask.ndim if self.diagonal_option() else 1)
        labels, count = ndimage.label(mask, structure=structure)
        index = np.arange(1, count + 1)
        if count > 0:
            centres = np.array(ndimage.center_of_mass(mask, labels, index), dtype=float).reshape(count, mask.ndim)
            sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
        else:
            centres = np.zeros((0, mask.ndim))
            sizes = np.zeros(0, dtype=int)
        self.centres_output.send(centres)
        self.sizes_output.send(sizes)
        self.count_output.send(int(count))
        self.labels_output.send(labels.astype(np.int64))


class ScipyImageFilterNode(Node):
    # gaussian_filter: a smooth blur, sigma wide; median_filter: each value replaced
    # by the median of its neighbourhood, which removes spikes and keeps edges
    modes = ['reflect', 'nearest', 'wrap', 'constant']

    @staticmethod
    def factory(name, data, args=None):
        node = ScipyImageFilterNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.median = label == 'scipy.median_filter'
        default = '3' if self.median else '1'
        if len(args) > 0:
            default = ' '.join(str(a) for a in args)
        self.size = self.parse(default)
        self.input = self.add_input('in', triggers_execution=True)
        self.size_input = self.add_input('size' if self.median else 'sigma', widget_type='text_input',
                                         widget_width=100, default_value=default, callback=self.size_changed)
        self.output = self.add_output('filtered')
        self.mode_option = self.add_option('edges', widget_type='combo', default_value='reflect')
        self.mode_option.widget.combo_items = self.modes

    def parse(self, value):
        if isinstance(value, str):
            numbers = [float(v) for v in re.findall(r'[-+]?\d*\.?\d+', value)]
        else:
            numbers = [float(v) for v in np.ravel(np.asarray(value, dtype=float))]
        return [max(n, 0.0) for n in numbers] or [1.0]

    def size_changed(self):
        self.size = self.parse(self.size_input())

    def execute(self):
        if self.size_input.fresh_input:
            self.size_changed()
        data = any_to_array(self.input()).astype(float)
        if data.ndim == 0:
            return
        size = self.size
        # one number for every axis, or one per axis (0 leaves that axis alone)
        if len(size) == 1:
            size = size * data.ndim
        elif len(size) != data.ndim:
            if self.app.verbose:
                print(self.label, 'needs one size, or one per axis:', data.ndim, 'for shape', data.shape)
            return
        mode = self.mode_option()
        if mode not in self.modes:
            mode = 'reflect'
        if self.median:
            self.output.send(ndimage.median_filter(data, size=[max(int(round(s)), 1) for s in size], mode=mode))
        else:
            self.output.send(ndimage.gaussian_filter(data, sigma=size, mode=mode))
