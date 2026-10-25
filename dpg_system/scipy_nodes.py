import numpy as np
import warnings
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from scipy import ndimage
from fractions import Fraction
import re
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import ConvexHull, procrustes
from scipy.spatial import QhullError
from scipy import stats
from scipy.cluster.vq import kmeans2
from scipy.optimize import curve_fit

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
    Node.app.register_node('scipy.slerp', ScipySlerpNode.factory)
    Node.app.register_node('scipy.rotation_mean', ScipyRotationMeanNode.factory)
    Node.app.register_node('scipy.align_vectors', ScipyAlignVectorsNode.factory)
    Node.app.register_node('scipy.rotation_apply', ScipyRotationApplyNode.factory)
    Node.app.register_node('scipy.circmean', ScipyCircularNode.factory)
    Node.app.register_node('scipy.circstd', ScipyCircularNode.factory)
    Node.app.register_node('scipy.hilbert', ScipyHilbertNode.factory)
    Node.app.register_node('scipy.detrend', ScipyDetrendNode.factory)
    Node.app.register_node('scipy.convex_hull', ScipyConvexHullNode.factory)
    Node.app.register_node('scipy.kmeans', ScipyKMeansNode.factory)
    Node.app.register_node('scipy.procrustes', ScipyProcrustesNode.factory)
    Node.app.register_node('scipy.curve_fit', ScipyCurveFitNode.factory)


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


# Quaternions here are scalar first - w x y z - as in the quaternion nodes.

def quaternions_from(value):
    """Rows of w x y z (a single quaternion becomes one row), or None."""
    q = any_to_array(value).astype(float)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if q.ndim != 2 or q.shape[1] != 4:
        return None
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms == 0) or not np.all(np.isfinite(q)):
        return None
    return q


class ScipySlerpNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ScipySlerpNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.keys = None
        self.key_times = None
        self.slerp = None
        self.input = self.add_input('t', triggers_execution=True)
        self.keys_input = self.add_input('quaternions')
        self.times_input = self.add_input('times')
        self.output = self.add_output('quaternions')

    def execute(self):
        if self.keys_input.fresh_input:
            self.keys = quaternions_from(self.keys_input())
            self.slerp = None
            if self.keys is None and self.app.verbose:
                print(self.label, 'quaternions must be rows of w x y z')
        if self.times_input.fresh_input:
            self.key_times = any_to_array(self.times_input()).astype(float).ravel()
            self.slerp = None
        if self.keys is None or self.keys.shape[0] < 2:
            return
        if self.slerp is None:
            n = self.keys.shape[0]
            # without times the keys are at 0, 1, 2 ...: two keys make t a fraction
            times = self.key_times if self.key_times is not None and self.key_times.size == n else np.arange(n, dtype=float)
            order = np.argsort(times, kind='stable')
            times, keys = times[order], self.keys[order]
            if np.any(np.diff(times) <= 0):
                if self.app.verbose:
                    print(self.label, 'times must all be different')
                return
            self.slerp = Slerp(times, Rotation.from_quat(keys, scalar_first=True))
        t = any_to_array(self.input()).astype(float)
        single = t.ndim == 0 or t.size == 1
        t = np.clip(t.ravel(), self.slerp.times[0], self.slerp.times[-1])
        q = self.slerp(t).as_quat(scalar_first=True)
        self.output.send(q[0] if single else q)


class ScipyRotationMeanNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ScipyRotationMeanNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.weights = None
        self.input = self.add_input('quaternions', triggers_execution=True)
        self.weights_input = self.add_input('weights')
        self.output = self.add_output('mean')

    def execute(self):
        if self.weights_input.fresh_input:
            self.weights = any_to_array(self.weights_input()).astype(float).ravel()
        q = quaternions_from(self.input())
        if q is None:
            if self.app.verbose:
                print(self.label, 'needs rows of w x y z')
            return
        weights = self.weights
        if weights is not None and (weights.size != q.shape[0] or np.any(weights < 0) or weights.sum() == 0):
            weights = None
        # the proper average of orientations: q and -q are the same turn, and this
        # does not mind which one each row happens to be
        self.output.send(Rotation.from_quat(q, scalar_first=True).mean(weights=weights).as_quat(scalar_first=True))


class ScipyAlignVectorsNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ScipyAlignVectorsNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.target = None
        self.input = self.add_input('from', triggers_execution=True)
        self.target_input = self.add_input('onto')
        self.rotation_output = self.add_output('rotation')
        self.aligned_output = self.add_output('aligned')
        self.error_output = self.add_output('error')
        self.as_option = self.add_option('treat as', widget_type='combo', default_value='points')
        self.as_option.widget.combo_items = ['points', 'directions']

    def execute(self):
        if self.target_input.fresh_input:
            self.target = any_to_array(self.target_input()).astype(float)
        if self.target is None:
            return
        source = any_to_array(self.input()).astype(float)
        target = self.target
        if source.ndim == 1:
            source = source.reshape(1, -1)
        if target.ndim == 1:
            target = target.reshape(1, -1)
        if source.shape != target.shape or source.shape[-1] != 3:
            if self.app.verbose:
                print(self.label, 'needs the same number of xyz rows in both, got', source.shape, target.shape)
            return
        points = self.as_option() == 'points'
        # points: match the shapes about their centres (the Kabsch method); directions
        # are already about the origin
        source_centre = source.mean(axis=0) if points else np.zeros(3)
        target_centre = target.mean(axis=0) if points else np.zeros(3)
        with warnings.catch_warnings():
            # one pair of vectors leaves a turn about them free; that is still an answer
            warnings.simplefilter('ignore', UserWarning)
            rotation, error = Rotation.align_vectors(target - target_centre, source - source_centre)
        aligned = rotation.apply(source - source_centre) + target_centre
        self.error_output.send(float(error))
        self.aligned_output.send(aligned)
        self.rotation_output.send(rotation.as_quat(scalar_first=True))


class ScipyRotationApplyNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ScipyRotationApplyNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.rotation = None
        self.input = self.add_input('vectors', triggers_execution=True)
        self.quaternion_input = self.add_input('quaternion')
        self.output = self.add_output('rotated')
        self.inverse_option = self.add_option('inverse', widget_type='checkbox', default_value=False)

    def execute(self):
        if self.quaternion_input.fresh_input:
            q = quaternions_from(self.quaternion_input())
            self.rotation = None if q is None else Rotation.from_quat(q if q.shape[0] > 1 else q[0], scalar_first=True)
        if self.rotation is None:
            return
        vectors = any_to_array(self.input()).astype(float)
        if vectors.shape[-1] != 3:
            if self.app.verbose:
                print(self.label, 'needs xyz vectors, got shape', vectors.shape)
            return
        rotation = self.rotation.inv() if self.inverse_option() else self.rotation
        try:
            self.output.send(rotation.apply(vectors))
        except ValueError:
            if self.app.verbose:
                print(self.label, 'one quaternion, or one per vector:', len(rotation) if not rotation.single else 1,
                      'for', vectors.shape)


class ScipyCircularNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ScipyCircularNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.spread = label == 'scipy.circstd'
        units = 'degrees'
        axis = None
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t == str and v in ['degrees', 'radians']:
                units = v
            elif t == int:
                axis = v
        self.input = self.add_input('angles', triggers_execution=True)
        self.axis_input = self.add_input('axis', widget_type='text_input', widget_width=60,
                                         default_value='None' if axis is None else str(axis))
        self.output = self.add_output('spread' if self.spread else 'mean')
        self.units_option = self.add_option('units', widget_type='combo', default_value=units)
        self.units_option.widget.combo_items = ['degrees', 'radians']

    def execute(self):
        angles = any_to_array(self.input()).astype(float)
        axis_text = str(self.axis_input()).strip()
        axis = None
        if axis_text not in ['', 'None', 'none']:
            try:
                axis = int(float(axis_text))
            except ValueError:
                axis = None
        if axis is not None and not -angles.ndim <= axis < angles.ndim:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range for shape', angles.shape)
            return
        degrees = self.units_option() == 'degrees'
        high, low = (180.0, -180.0) if degrees else (np.pi, -np.pi)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            if self.spread:
                self.output.send(stats.circstd(angles, high=high, low=low, axis=axis, nan_policy='omit'))
            else:
                self.output.send(stats.circmean(angles, high=high, low=low, axis=axis, nan_policy='omit'))


class ScipyHilbertNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ScipyHilbertNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        rate = 60.0
        if len(args) > 0:
            rate = any_to_float(args[0])
        self.input = self.add_input('in', triggers_execution=True)
        self.rate_input = self.add_input('sample rate', widget_type='drag_float', default_value=rate)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=-1)
        self.envelope_output = self.add_output('envelope')
        self.phase_output = self.add_output('phase')
        self.frequency_output = self.add_output('frequency')
        self.centre_option = self.add_option('remove mean', widget_type='checkbox', default_value=True)

    def execute(self):
        data = any_to_array(self.input()).astype(float)
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim or data.shape[axis] < 2:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range or too short for shape', data.shape)
            return
        data = np.nan_to_num(data)
        # phase means swinging about a centre: take the average level away first
        if self.centre_option():
            data = data - data.mean(axis=axis, keepdims=True)
        analytic = signal.hilbert(data, axis=axis)
        phase = np.angle(analytic)
        rate = self.rate_input()
        if rate > 0:
            # how fast the phase turns, in cycles per second; the first sample repeats
            # the second so the result keeps the input's length
            turning = np.diff(np.unwrap(phase, axis=axis), axis=axis) * rate / (2 * np.pi)
            first = np.take(turning, [0], axis=axis)
            self.frequency_output.send(np.concatenate([first, turning], axis=axis))
        self.phase_output.send(np.degrees(phase))
        self.envelope_output.send(np.abs(analytic))


class ScipyDetrendNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ScipyDetrendNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        kind = 'linear'
        if len(args) > 0 and args[0] in ['linear', 'constant']:
            kind = args[0]
        self.input = self.add_input('in', triggers_execution=True)
        self.axis_input = self.add_input('axis', widget_type='input_int', default_value=-1)
        self.output = self.add_output('detrended')
        self.kind_option = self.add_option('remove', widget_type='combo', default_value=kind)
        self.kind_option.widget.combo_items = ['linear', 'constant']

    def execute(self):
        data = any_to_array(self.input()).astype(float)
        axis = self.axis_input()
        if not -data.ndim <= axis < data.ndim or data.shape[axis] < 2:
            if self.app.verbose:
                print(self.label, 'axis', axis, 'out of range or too short for shape', data.shape)
            return
        kind = 'constant' if self.kind_option() == 'constant' else 'linear'
        self.output.send(signal.detrend(data, axis=axis, type=kind))


class ScipyConvexHullNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ScipyConvexHullNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input('points', triggers_execution=True)
        self.size_output = self.add_output('size')
        self.boundary_output = self.add_output('boundary')
        self.vertices_output = self.add_output('vertices')

    def execute(self):
        points = any_to_array(self.input()).astype(float)
        if points.ndim != 2 or points.shape[1] not in [2, 3]:
            if self.app.verbose:
                print(self.label, 'needs rows of x y or x y z, got shape', points.shape)
            return
        points = points[np.all(np.isfinite(points), axis=1)]
        if points.shape[0] < points.shape[1] + 1:
            return
        try:
            hull = ConvexHull(points)
        except QhullError:
            # all the points in a line (or a plane, in 3-D): nothing is enclosed
            self.vertices_output.send(np.zeros(0, dtype=int))
            self.boundary_output.send(0.0)
            self.size_output.send(0.0)
            return
        # scipy names them for 3-D: volume is the area in 2-D, area the perimeter
        self.vertices_output.send(hull.vertices)
        self.boundary_output.send(float(hull.area))
        self.size_output.send(float(hull.volume))


class ScipyKMeansNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ScipyKMeansNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        k = 3
        if len(args) > 0:
            k = max(any_to_int(args[0]), 1)
        self.centres = None
        self.input = self.add_input('points', triggers_execution=True)
        self.k_input = self.add_input('k', widget_type='input_int', default_value=k)
        self.labels_output = self.add_output('labels')
        self.centres_output = self.add_output('centres')
        self.follow_option = self.add_option('follow', widget_type='checkbox', default_value=True)
        self.seed_option = self.add_option('seed', widget_type='input_int', default_value=-1,
                                           callback=self.seed_changed)
        self.rng = np.random.default_rng()

    def seed_changed(self):
        seed = self.seed_option()
        self.rng = np.random.default_rng(None if seed is None or seed < 0 else seed)
        self.centres = None

    def execute(self):
        points = any_to_array(self.input()).astype(float)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        if points.ndim != 2:
            if self.app.verbose:
                print(self.label, 'needs rows of coordinates, got shape', points.shape)
            return
        points = points[np.all(np.isfinite(points), axis=1)]
        k = min(max(self.k_input(), 1), points.shape[0])
        if k < 1:
            return
        with warnings.catch_warnings():
            # an empty group is reported by scipy as a warning; it is still an answer
            warnings.simplefilter('ignore')
            if self.follow_option() and self.centres is not None and self.centres.shape == (k, points.shape[1]):
                # start from the last centres, so group numbers stay with the same groups
                centres, labels = kmeans2(points, self.centres, minit='matrix', seed=self.rng)
            else:
                centres, labels = kmeans2(points, k, minit='++', seed=self.rng)
        self.centres = centres
        self.centres_output.send(centres)
        self.labels_output.send(labels)


class ScipyProcrustesNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ScipyProcrustesNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.reference = None
        self.input = self.add_input('shape', triggers_execution=True)
        self.reference_input = self.add_input('reference')
        self.difference_output = self.add_output('difference')
        self.aligned_output = self.add_output('aligned')
        self.standard_output = self.add_output('reference standardized')

    def execute(self):
        if self.reference_input.fresh_input:
            self.reference = any_to_array(self.reference_input()).astype(float)
        if self.reference is None:
            return
        shape = any_to_array(self.input()).astype(float)
        if shape.shape != self.reference.shape or shape.ndim != 2 or shape.shape[0] < 2:
            if self.app.verbose:
                print(self.label, 'needs two arrays of the same rows of coordinates, got',
                      shape.shape, self.reference.shape)
            return
        try:
            standard, aligned, difference = procrustes(self.reference, shape)
        except ValueError as e:
            # a shape with every point in the same place has no size to scale
            if self.app.verbose:
                print(self.label, e)
            return
        self.standard_output.send(standard)
        self.aligned_output.send(aligned)
        self.difference_output.send(float(difference))


class ScipyCurveFitNode(Node):
    # the formula is typed by the patch's author, and only numpy's functions are
    # available to it - no Python builtins
    functions = {name: getattr(np, name) for name in
                 ['exp', 'log', 'log10', 'log2', 'sqrt', 'sin', 'cos', 'tan', 'arctan', 'arcsin', 'arccos',
                  'tanh', 'sinh', 'cosh', 'abs', 'sign', 'pi', 'e', 'power', 'maximum', 'minimum']}

    @staticmethod
    def factory(name, data, args=None):
        node = ScipyCurveFitNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        formula = ' '.join(str(a) for a in args) if len(args) > 0 else 'a * exp(-b * x) + c'
        self.x_value = None
        self.guess = None
        self.compiled = None
        self.names = []
        self.input = self.add_input('y', triggers_execution=True)
        self.x_input = self.add_input('x')
        self.guess_input = self.add_input('guess')
        self.formula_input = self.add_input('formula', widget_type='text_input', widget_width=200,
                                            default_value=formula, callback=self.formula_changed)
        self.parameters_output = self.add_output('parameters')
        self.names_output = self.add_output('names')
        self.fitted_output = self.add_output('fitted')
        self.error_output = self.add_output('error')
        # a widget's value is not there yet while the node is being made: read the
        # starting formula from the arguments
        self.formula_changed(formula)

    def formula_changed(self, formula=None):
        if formula is None:
            formula = self.formula_input()
        formula = str(formula).replace('^', '**')
        try:
            code = compile(formula, '<formula>', 'eval')
        except SyntaxError:
            self.compiled = None
            if self.app.verbose:
                print(self.label, 'cannot read the formula', formula)
            return
        # every name that is not x and not a function is a parameter, in order of appearance
        names = []
        for name in re.findall(r'[A-Za-z_]\w*', formula):
            if name != 'x' and name not in self.functions and name not in names:
                names.append(name)
        self.compiled = code
        self.names = names

    def model(self, x, *params):
        scope = dict(self.functions)
        scope['x'] = x
        scope.update(zip(self.names, params))
        return eval(self.compiled, {'__builtins__': {}}, scope) * np.ones_like(x)

    def execute(self):
        if self.formula_input.fresh_input:
            self.formula_changed()
        if self.x_input.fresh_input:
            self.x_value = any_to_array(self.x_input()).astype(float).ravel()
        if self.guess_input.fresh_input:
            self.guess = any_to_array(self.guess_input()).astype(float).ravel()
        if self.compiled is None or len(self.names) == 0:
            return
        y = any_to_array(self.input()).astype(float).ravel()
        x = self.x_value if self.x_value is not None else np.arange(y.size, dtype=float)
        if x.size != y.size or y.size < len(self.names):
            if self.app.verbose:
                print(self.label, 'needs one x per y, and at least as many points as parameters', self.names)
            return
        keep = np.isfinite(x) & np.isfinite(y)
        guess = self.guess if self.guess is not None and self.guess.size == len(self.names) else np.ones(len(self.names))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                params, _ = curve_fit(self.model, x[keep], y[keep], p0=guess, maxfev=5000)
        except Exception as e:
            # no fit found (often a poor starting guess); say why, send nothing
            if self.app.verbose:
                print(self.label, 'could not fit:', e)
            return
        fitted = self.model(x, *params)
        self.error_output.send(float(np.sqrt(np.nanmean((fitted - y) ** 2))))
        self.fitted_output.send(fitted)
        self.names_output.send(list(self.names))
        self.parameters_output.send(params)
